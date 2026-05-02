#!/usr/bin/env python3
"""Theozyme active-site refinement v2 — physics-based, design-intent-aware.

Design philosophy:
  The DESIGN model embeds a DFT-quality transition-state geometry. We treat the
  ligand (HETATM complex) as a RIGID BODY (its bond lengths/angles ARE the
  catalytic intent), and we drive AF3's protein toward design's catres-ligand
  contact pattern using xTB-evaluated harmonic restraints.

Pipeline:
  1. Alignment is performed externally (woodbuse's TMalign + sidechain-opt
     script via the universal.sif container).  We just consume the
     <af3>_aligned.pdb file.

  2. Build a "design contact map":
       For each catalytic residue (REMARK 666), enumerate sidechain heavy atoms
       that come within ~4.5 Å of any ligand heavy atom in design.pdb.
       Each (catres_atom → lig_atom, distance, classification) tuple becomes a
       distance restraint target during the xTB relax.  Classification chooses
       the harmonic force constant:
            covalent     (1.0–1.6 Å)  → tight (1.0 Eh/Bohr²)
            metal-coord  (1.7–2.5 Å, M=Zn)  → tight-medium (0.3)
            strong H-bond (2.5–3.3 Å, polar pair) → medium (0.1)
            close pose (3.3–4.5 Å) → soft (0.03)

  3. PTM handling — currently KCX only:
       --ptm A/LYS/3:KCX → strip 2 of 3 NH3+ protons from LYS NZ to give NH⁻.
       Total charge is adjusted by -2 per KCX.  A KCX residue's NZ gets a
       *covalent* restraint to the carbamate C from the design map, ensuring
       the C–N bond is preserved during relax.

  4. xTB GFN-FF (default) constrained opt:
        $fix     — every HETATM atom (rigid body)
        $constrain — multiple blocks at different force constants for the
                     restraints derived from the design contact map.
        Backbone CA / N / C / O fixed by default; --unfreeze-shell N relaxes
        backbone for residues within ±N of any catalytic residue.

  5. Report deviations table (design distance vs initial AF3 vs final relaxed)
     and stitch refined cluster back into the full structure.

Usage:
  python enzyme_design_applications/active_site_refine/refine.py \\
      design.pdb af3_aligned.pdb -o refined.pdb \\
      --ptm A/LYS/3:KCX \\
      --radius 6.0 --gfn 0 --unfreeze-shell 1
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("refine_v2")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
XTB_BIN = str(_REPO_ROOT / "deps" / "xtb" / "install" / "bin" / "xtb")
XTB_LIB_DIRS = [
    str(_REPO_ROOT / "deps" / "xtb" / "install" / "lib" / "x86_64-linux-gnu"),
    "/home/woodbuse/conda/envs/qcb-xtb/lib",
]
GXTB_BIN = str(_REPO_ROOT / "deps" / "g-xtb" / "install" / "xtb-6.7.1" / "bin" / "xtb")

# ── Formal charges (pH 7, no PROPKA — simple, deterministic) ────────────
# Residues not listed are treated as neutral (charge 0).
RESIDUE_CHARGE: dict[str, int] = {
    "LYS": +1, "ARG": +1,
    "ASP": -1, "GLU": -1,
    "HIS":  0,  # default HID/HIE; HIP (+1) needs explicit override
}

# Final formal charge of common PTMs (NOT a delta — the absolute charge of the
# modified residue). Used when --ptm specifies a non-canonical residue.
PTM_CHARGE: dict[str, int] = {
    "KCX": -1,  # carbamylated lysine: -CH₂-NH-COO⁻
}

# Default formal charges of common HETATM residues. Per the user's spec,
# Zn is always Zn(II); for the ambiguous transition metals listed in
# AMBIGUOUS_METALS the user is expected to pass --ligand-charge or we'll
# just use the most common oxidation state and warn loudly.
DEFAULT_HETATM_CHARGE: dict[str, int] = {
    # unambiguous (use defaults silently)
    "ZN":  +2, "MG": +2, "CA": +2,
    "K":   +1, "NA": +1,
    "CL":  -1,
    "HOH":  0, "WAT": 0,
    # ambiguous — defaults to most common state, warn user
    "FE":  +3,  # could be +2 (ferrous) or +3 (ferric)
    "MN":  +2,  # could be +2/+3/+4/+7
    "CU":  +2,  # could be +1 (cuprous) or +2 (cupric)
    "NI":  +2,  # +2 most common, also +1/+3 in some enzymes
    "CO":  +2,  # +2 or +3
    "MO":  +6,  # +4/+5/+6
    "W":   +6,  # +4/+5/+6
}

# Metals where the +charge depends on the chemistry — caller must specify.
# We emit a warning if any of these are present without an explicit override.
AMBIGUOUS_METALS = {"FE", "MN", "CU", "NI", "CO", "MO", "W"}

PROTEIN_RES = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
               "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
BACKBONE_ATOMS = {"N", "CA", "C", "O"}


# ──────────────────────────────────────────────────────────────────
# PDB I/O
# ──────────────────────────────────────────────────────────────────

@dataclass
class Atom:
    line: str
    record: str
    serial: int
    aname: str
    rname: str
    chain: str
    rnum: int
    x: float
    y: float
    z: float
    element: str

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def is_hetatm(self) -> bool:
        return self.record == "HETATM"

    @property
    def is_heavy(self) -> bool:
        return self.element != "H"

    def with_pos(self, p: np.ndarray) -> "Atom":
        line = self.line[:30] + f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}" + self.line[54:]
        return Atom(line, self.record, self.serial, self.aname, self.rname,
                    self.chain, self.rnum, *p, self.element)


def parse_pdb(path: Path) -> tuple[list[Atom], list[str]]:
    atoms = []
    headers = []
    with open(path) as f:
        before = True
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                before = False
                try:
                    rec = line[:6].strip()
                    serial = int(line[6:11])
                    aname = line[12:16].strip()
                    rname = line[17:20].strip()
                    chain = line[21:22].strip() or "A"
                    rnum = int(line[22:26])
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    el = line[76:78].strip() if len(line) >= 78 else aname[0]
                    atoms.append(Atom(line.rstrip("\n"), rec, serial, aname, rname,
                                       chain, rnum, x, y, z, el))
                except (ValueError, IndexError):
                    continue
            elif before:
                headers.append(line.rstrip("\n"))
    return atoms, headers


def write_pdb(atoms: list[Atom], path: Path, headers: list[str] | None = None,
              extra_remarks: list[str] | None = None):
    with open(path, "w") as f:
        if headers:
            for h in headers:
                f.write(h + "\n")
        for r in (extra_remarks or []):
            f.write(r + "\n")
        for a in atoms:
            f.write(a.line + "\n")
        f.write("END\n")


def _atom_line(serial: int, aname: str, chain: str, resnum: int,
               resname: str, pos: np.ndarray, element: str) -> str:
    x, y, z = pos
    return (f"ATOM  {serial:>5} {aname:<4} {resname:<3} {chain}{resnum:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2}")


# ──────────────────────────────────────────────────────────────────
# REMARK 666 parsing — keep catres index (1-6 …) for PTM specs
# ──────────────────────────────────────────────────────────────────

@dataclass
class CatRes:
    chain: str
    rnum: int
    rname: str
    cat_idx: int   # 1-based slot from REMARK 666


def parse_remark666(pdb_path: Path) -> tuple[tuple[str,int,str] | None, list[CatRes]]:
    lig = None
    cats = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("REMARK 666"):
                continue
            m = re.search(
                r"TEMPLATE\s+(\S)\s+(\S+)\s+(\d+).+MOTIF\s+(\S)\s+(\S+)\s+(\d+)\s+(\d+)",
                line)
            if m:
                if lig is None:
                    lig = (m.group(1), int(m.group(3)), m.group(2))
                cats.append(CatRes(m.group(4), int(m.group(6)), m.group(5),
                                   int(m.group(7))))
    return lig, cats


# ──────────────────────────────────────────────────────────────────
# Design contact map
# ──────────────────────────────────────────────────────────────────

@dataclass
class DesignContact:
    """A heavy-atom pair (catres → ligand) extracted from design.pdb."""
    res_chain: str
    res_num: int
    res_name: str
    res_atom: str        # e.g. NE2, NZ, OE1
    lig_atom: str        # e.g. ZN1, C1, O5
    distance: float
    kind: str            # 'covalent', 'metal_coord', 'hbond', 'close'
    force: float         # xTB force constant (Eh/Bohr²)


_METALS = {"ZN", "FE", "MG", "MN", "CA", "CU", "NI", "CO"}
_POLAR = {"N", "O", "S"}


def _classify(d: float, atom_a: Atom, atom_b: Atom) -> tuple[str, float]:
    """Return (kind, force_constant) for a contact at distance d Å."""
    el_a, el_b = atom_a.element.upper(), atom_b.element.upper()
    is_metal = el_a in _METALS or el_b in _METALS
    is_polar_pair = el_a in _POLAR and el_b in _POLAR

    # Covalent (KCX-style): typical N–C, S–S, O–C distances
    if 1.0 <= d <= 1.7 and is_polar_pair is False and not is_metal:
        # not actually used — covalent flagging is done in apply_ptms
        return "close", 0.03
    if 1.0 <= d <= 1.7 and (el_a == "N" or el_a == "S" or el_a == "O") and el_b == "C":
        return "covalent", 1.0
    if 1.0 <= d <= 1.7 and (el_b == "N" or el_b == "S" or el_b == "O") and el_a == "C":
        return "covalent", 1.0

    # Metal-coordination: M–N or M–O around 1.9–2.5 Å
    if is_metal and 1.7 <= d <= 2.7:
        return "metal_coord", 0.3

    # H-bond: polar–polar 2.5–3.4 Å (no metal)
    if is_polar_pair and 2.4 <= d <= 3.5 and not is_metal:
        return "hbond", 0.1

    # Other close contact
    if d <= 4.5:
        return "close", 0.03

    return "none", 0.0


_HIS_RING_ATOMS = {"CG", "ND1", "CE1", "NE2", "CD2"}


def build_design_contact_map(
    design_atoms: list[Atom],
    catalytic: list[CatRes],
    radius: float = 4.5,
) -> list[DesignContact]:
    """Walk every catres sidechain heavy atom; record distances to ligand
    heavy atoms within `radius`, classified for restraint strength.

    HIS ring-orientation lock: for any HIS catres whose NE2 or ND1 directly
    coordinates a metal in design (≤2.5 Å), upgrade the entire ring's other
    atoms (CG/ND1/CE1/NE2/CD2) to a stronger restraint (force ≥ 0.15),
    so MLFF/xtb backends can't rotate the imidazole around the metal-N axis
    while still satisfying the metal-N distance.
    """
    lig = [a for a in design_atoms if a.is_hetatm and a.is_heavy]
    if not lig:
        log.warning("No ligand HETATM atoms found in design")
        return []
    lig_pos = np.array([a.pos for a in lig])

    # 1st pass: detect which HIS catres coordinate a metal, and to which
    metal_for_his: dict[tuple[str, int], Atom] = {}
    for c in catalytic:
        if c.rname != "HIS":
            continue
        for an in ("NE2", "ND1"):
            ar = next((a for a in design_atoms
                       if a.chain == c.chain and a.rnum == c.rnum
                       and a.aname == an), None)
            if ar is None:
                continue
            for L in lig:
                if L.element.upper() not in _METALS:
                    continue
                if np.linalg.norm(L.pos - ar.pos) <= 2.5:
                    metal_for_his[(c.chain, c.rnum)] = L
                    break
            if (c.chain, c.rnum) in metal_for_his:
                break

    contacts: list[DesignContact] = []
    for c in catalytic:
        sc_atoms = [a for a in design_atoms
                    if a.chain == c.chain and a.rnum == c.rnum
                    and a.is_heavy and a.aname not in BACKBONE_ATOMS]
        metal_anchor = metal_for_his.get((c.chain, c.rnum))
        for a in sc_atoms:
            d = np.linalg.norm(lig_pos - a.pos, axis=1)
            for j, dj in enumerate(d):
                if dj > radius:
                    continue
                kind, force = _classify(dj, a, lig[j])
                # Promote ring-atom→coord-metal contacts so they're strong
                # enough to lock the imidazole orientation. Without this, MLFF
                # backends can rotate the ring 180° about the N-metal axis
                # and still satisfy the metal-N distance.
                if (metal_anchor is not None
                        and a.aname in _HIS_RING_ATOMS
                        and lig[j].aname == metal_anchor.aname
                        and kind in ("close", "hbond")):
                    kind, force = "metal_ring_lock", 0.15
                if kind == "none":
                    continue
                contacts.append(DesignContact(
                    res_chain=c.chain, res_num=c.rnum, res_name=c.rname,
                    res_atom=a.aname, lig_atom=lig[j].aname,
                    distance=float(dj), kind=kind, force=force,
                ))
    return contacts


def filter_top_contacts(
    contacts: list[DesignContact],
    max_per_residue: int = 6,
) -> list[DesignContact]:
    """Keep at most N restraints per residue, prioritizing tight (covalent >
    metal_coord > hbond > close) and shortest first."""
    rank = {"covalent": 0, "metal_coord": 1, "metal_ring_lock": 2,
            "hbond": 3, "close": 4}
    by_res: dict[tuple[str, int], list[DesignContact]] = {}
    for c in contacts:
        by_res.setdefault((c.res_chain, c.res_num), []).append(c)
    out = []
    for cs in by_res.values():
        cs.sort(key=lambda c: (rank[c.kind], c.distance))
        out.extend(cs[:max_per_residue])
    return out


# ──────────────────────────────────────────────────────────────────
# PTM application — KCX (carbamylated lysine)
# ──────────────────────────────────────────────────────────────────

@dataclass
class PtmSpec:
    chain: str
    res_name: str       # original (LYS, CYS, …)
    cat_idx: int        # 1-based REMARK 666 slot
    new_name: str       # ncAA name (KCX, …)


def parse_ptm_spec(spec: str) -> PtmSpec:
    """A/LYS/3:KCX → PtmSpec(chain='A', res_name='LYS', cat_idx=3, new_name='KCX')"""
    m = re.match(r"([A-Za-z])/([A-Z]{3})/(\d+):([A-Z]{3})$", spec)
    if not m:
        raise ValueError(f"--ptm must be 'CHAIN/RES/CATIDX:NCAA', got {spec!r}")
    return PtmSpec(m.group(1), m.group(2), int(m.group(3)), m.group(4))


def apply_kcx_protonation(
    atoms: list[Atom],
    catalytic: list[CatRes],
    ptms: list[PtmSpec],
) -> tuple[list[Atom], int, list[tuple[str, int]]]:
    """For each KCX ptm spec: strip extra H atoms from NZ.

    Returns (new_atoms, charge_delta, kcx_residues).
      - charge_delta: -2 per KCX (LYS+ → KCX-NH⁻)
      - kcx_residues: list of (chain, rnum) where KCX was applied
    """
    drop: set[int] = set()
    charge_delta = 0
    kcx_residues: list[tuple[str, int]] = []

    for ptm in ptms:
        if ptm.new_name != "KCX":
            log.warning(f"  PTM {ptm.new_name} not yet implemented — skipping")
            continue
        # find the catres at slot cat_idx
        matches = [c for c in catalytic if c.cat_idx == ptm.cat_idx]
        if not matches:
            log.warning(f"  PTM cat_idx {ptm.cat_idx} not found in REMARK 666")
            continue
        c = matches[0]
        if c.rname != ptm.res_name:
            log.warning(f"  PTM cat_idx {ptm.cat_idx} expected {ptm.res_name} but "
                        f"REMARK 666 says {c.rname}")
            continue
        # find NZ + its H atoms (for LYS)
        nz = next((a for a in atoms if a.chain == c.chain and a.rnum == c.rnum
                   and a.aname == "NZ"), None)
        if nz is None:
            log.warning(f"  KCX: no NZ atom on {c.chain}{c.rnum}")
            continue
        h_atoms = [(i, a) for i, a in enumerate(atoms)
                   if a.chain == c.chain and a.rnum == c.rnum
                   and a.element == "H"
                   and np.linalg.norm(a.pos - nz.pos) < 1.3]
        # keep 1 H, drop the rest
        if len(h_atoms) >= 2:
            for i, _ in h_atoms[:-1]:
                drop.add(i)
            log.info(f"  KCX {c.chain}{c.rnum}: stripped {len(h_atoms)-1} H from NZ "
                     f"(NH3+ → NH⁻, kept '{h_atoms[-1][1].aname}')")
            charge_delta -= 2
        else:
            log.info(f"  KCX {c.chain}{c.rnum}: NZ already has only {len(h_atoms)} H atoms")
        kcx_residues.append((c.chain, c.rnum))

    if not drop:
        return atoms, charge_delta, kcx_residues
    new_atoms = [a for i, a in enumerate(atoms) if i not in drop]
    return new_atoms, charge_delta, kcx_residues


# ──────────────────────────────────────────────────────────────────
# Cluster building
# ──────────────────────────────────────────────────────────────────

def build_cluster(
    atoms: list[Atom],
    catalytic: list[CatRes],
    radius: float = 6.0,
) -> tuple[list[Atom], set[tuple[str, int]]]:
    """Ligand HETATM + residues within `radius` Å of any HETATM + catres."""
    het_pos = np.array([a.pos for a in atoms if a.is_hetatm and a.is_heavy])
    if len(het_pos) == 0:
        raise RuntimeError("No HETATM heavy atoms — nothing to cluster around")

    by_res: dict[tuple[str, int], list[Atom]] = {}
    for a in atoms:
        by_res.setdefault((a.chain, a.rnum), []).append(a)

    keep: set[tuple[str, int]] = set()
    for (ch, rn), res_atoms in by_res.items():
        if any(a.is_hetatm for a in res_atoms):
            keep.add((ch, rn)); continue
        coords = np.array([a.pos for a in res_atoms if a.is_heavy])
        if len(coords) == 0:
            continue
        d = np.min(np.linalg.norm(coords[:, None, :] - het_pos[None, :, :], axis=2), axis=1)
        if np.any(d <= radius):
            keep.add((ch, rn))

    for c in catalytic:
        keep.add((c.chain, c.rnum))

    cluster = [a for a in atoms if (a.chain, a.rnum) in keep]
    return cluster, keep


# ──────────────────────────────────────────────────────────────────
# Cluster net-charge calculation
# ──────────────────────────────────────────────────────────────────

# Atomic numbers for electron-parity sanity check.
_ATOMIC_NUMBER: dict[str, int] = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
    "CL": 17, "BR": 35, "I": 53, "ZN": 30, "MG": 12, "MN": 25, "FE": 26,
    "CA": 20, "K": 19, "NA": 11, "B": 5, "SE": 34,
}


def parse_ligand_charge_specs(specs: list[str]) -> dict[str, int]:
    """Turn ['YYE:0', 'ZN:2'] → {'YYE': 0, 'ZN': 2}. Format: 'RESNAME:CHARGE'."""
    out: dict[str, int] = {}
    for spec in specs:
        m = re.match(r"([A-Za-z0-9]{1,3}):([+-]?\d+)$", spec)
        if not m:
            raise ValueError(f"--ligand-charge must be 'RESNAME:CHARGE', got {spec!r}")
        out[m.group(1).upper()] = int(m.group(2))
    return out


def compute_cluster_charge(
    cluster: list[Atom],
    ptms: list[PtmSpec],
    catalytic: list[CatRes],
    user_ligand_charges: dict[str, int],
    pH: float = 7.0,
) -> tuple[int, list[str]]:
    """Walk every unique residue in the cluster, sum its formal charge, and
    return (total_charge, audit_lines).

    Per the user's stated assumptions:
      • spin = 1 always (no radicals)
      • Zn always Zn(II) → +2
      • HIS defaults to neutral (HID/HIE) — override via PTM if HIP is wanted
      • PTM residues use the PTM's absolute charge from PTM_CHARGE,
        replacing the canonical residue's charge
      • HETATM residues use --ligand-charge if given, else
        DEFAULT_HETATM_CHARGE, else 0 (with a warning)
    """
    # Resolve PTM-affected residues: (chain, rnum) → new resname
    ptm_by_residue: dict[tuple[str, int], str] = {}
    cat_by_idx = {c.cat_idx: c for c in catalytic}
    for p in ptms:
        c = cat_by_idx.get(p.cat_idx)
        if c is None:
            continue
        ptm_by_residue[(c.chain, c.rnum)] = p.new_name

    # Group atoms per residue
    by_res: dict[tuple[str, int], list[Atom]] = {}
    for a in cluster:
        by_res.setdefault((a.chain, a.rnum), []).append(a)

    total = 0
    audit: list[str] = []
    seen_hetatm_resnames: set[str] = set()

    for (ch, rn), atoms in sorted(by_res.items()):
        rname = atoms[0].rname
        is_het = any(a.is_hetatm for a in atoms)

        if (ch, rn) in ptm_by_residue:
            ncaa = ptm_by_residue[(ch, rn)]
            if ncaa not in PTM_CHARGE:
                audit.append(f"  WARN  PTM {ncaa} on {ch}{rn} has no known charge "
                             f"— treating as 0")
                q = 0
            else:
                q = PTM_CHARGE[ncaa]
                audit.append(f"  PTM   {ch}{rn} {rname}→{ncaa} = {q:+d}")
        elif is_het:
            seen_hetatm_resnames.add(rname)
            if rname in user_ligand_charges:
                q = user_ligand_charges[rname]
                audit.append(f"  HET   {ch}{rn} {rname} = {q:+d} (user)")
            elif rname in DEFAULT_HETATM_CHARGE:
                q = DEFAULT_HETATM_CHARGE[rname]
                tag = "default"
                if rname in AMBIGUOUS_METALS:
                    audit.append(f"  WARN  metal {rname} has variable oxidation "
                                 f"states; using +{q} but pass --ligand-charge "
                                 f"{rname}:N to be explicit (Fe: +2/+3, "
                                 f"Mn: +2/+3/+4/+7, Cu: +1/+2, etc.)")
                    tag = "ambiguous default"
                audit.append(f"  HET   {ch}{rn} {rname} = {q:+d} ({tag})")
            else:
                q = 0
                audit.append(f"  WARN  HETATM {rname} has no charge entry "
                             f"— treating as 0; pass --ligand-charge {rname}:N "
                             f"to override")
        elif rname in PROTEIN_RES:
            q = RESIDUE_CHARGE.get(rname, 0)
            if q != 0:
                audit.append(f"  RES   {ch}{rn} {rname} = {q:+d}")
        else:
            q = 0
            audit.append(f"  WARN  unknown residue {ch}{rn} {rname} = 0")

        total += q

    return total, audit


def cluster_electron_parity(cluster: list[Atom], total_charge: int) -> tuple[int, bool]:
    """Sum atomic numbers, subtract total_charge, return (n_electrons, even).
    Used to flag open-shell systems before xtb does."""
    n = 0
    for a in cluster:
        z = _ATOMIC_NUMBER.get(a.element.upper())
        if z is None:
            continue
        n += z
    n -= total_charge
    return n, (n % 2 == 0)


def cap_backbone(cluster: list[Atom], all_atoms: list[Atom],
                 keep: set[tuple[str, int]]) -> list[Atom]:
    """Add H caps at dangling backbone N or C boundaries."""
    by_res: dict[tuple[str, int], list[Atom]] = {}
    for a in all_atoms:
        by_res.setdefault((a.chain, a.rnum), []).append(a)
    new_caps = []
    serial = max((a.serial for a in cluster), default=0) + 1
    for (ch, rn) in keep:
        rs = by_res.get((ch, rn), [])
        if not rs or rs[0].rname not in PROTEIN_RES:
            continue
        for offset, target_aname in [(-1, "N"), (1, "C")]:
            if (ch, rn + offset) in keep:
                continue
            atom_target = next((a for a in cluster if a.chain == ch and a.rnum == rn
                                and a.aname == target_aname), None)
            atom_ca = next((a for a in cluster if a.chain == ch and a.rnum == rn
                            and a.aname == "CA"), None)
            if atom_target is None or atom_ca is None:
                continue
            v = atom_target.pos - atom_ca.pos
            v = v / (np.linalg.norm(v) + 1e-9)
            h_pos = atom_target.pos + v * 1.0
            new_caps.append(Atom(
                line=_atom_line(serial, "HCAP", ch, rn, rs[0].rname, h_pos, "H"),
                record="ATOM", serial=serial, aname="HCAP",
                rname=rs[0].rname, chain=ch, rnum=rn,
                x=h_pos[0], y=h_pos[1], z=h_pos[2], element="H",
            ))
            serial += 1
    return cluster + new_caps


# ──────────────────────────────────────────────────────────────────
# xTB constraint construction
# ──────────────────────────────────────────────────────────────────

def _index_in_cluster(cluster: list[Atom], chain: str, rnum: int, aname: str) -> int | None:
    for i, a in enumerate(cluster, 1):
        if a.chain == chain and a.rnum == rnum and a.aname == aname:
            return i
    return None


def write_xtb_input(
    cluster: list[Atom],
    inp_path: Path,
    catalytic: list[CatRes],
    contacts: list[DesignContact],
    unfreeze_shell: int = 0,
    rigidity: str = "backbone",
    angle_restraints: list | None = None,  # tuples (i, j, k, theta_rad, k_ev, label)
    k_scale: float = 1.0,
):
    """Write the xTB control file containing $fix (rigid ligand + frozen
    backbone) and one $constrain block per restraint tier."""
    backbone_atoms = set(BACKBONE_ATOMS)
    if rigidity == "backbone-cb":
        backbone_atoms |= {"CB"}

    unfrozen: set[tuple[str, int]] = set()
    if unfreeze_shell > 0:
        for c in catalytic:
            for delta in range(-unfreeze_shell, unfreeze_shell + 1):
                unfrozen.add((c.chain, c.rnum + delta))

    fix_indices = []
    n_lig = n_bb = n_cap = n_unfrozen = 0
    for i, a in enumerate(cluster, 1):
        if a.is_hetatm:
            fix_indices.append(i); n_lig += 1
        elif a.aname == "HCAP":
            fix_indices.append(i); n_cap += 1
        elif a.rname in PROTEIN_RES and a.aname in backbone_atoms:
            if (a.chain, a.rnum) in unfrozen:
                n_unfrozen += 1
            else:
                fix_indices.append(i); n_bb += 1

    # Group contacts by (scaled) force constant for separate $constrain blocks
    by_force: dict[float, list[tuple[int, int, float, str]]] = {}
    n_skipped = 0
    for ct in contacts:
        i = _index_in_cluster(cluster, ct.res_chain, ct.res_num, ct.res_atom)
        # ligand atoms might be HETATM with chain B
        j = next((k + 1 for k, a in enumerate(cluster)
                  if a.is_hetatm and a.aname == ct.lig_atom), None)
        if i is None or j is None:
            n_skipped += 1
            continue
        scaled = ct.force * k_scale
        by_force.setdefault(scaled, []).append(
            (i, j, ct.distance, f"{ct.res_name}{ct.res_num}.{ct.res_atom}-{ct.lig_atom}({ct.kind})"))

    # Internal-angle restraints (sidechain valence) — convert (i_0based, j_0based,
    # k_0based, theta_rad, ...) → 1-based + degrees, then group by k_ang/100 since
    # xtb uses Eh/Bohr² but our angle k is eV/rad²; small unit dance.
    # xtb angle force constant: same $constrain "force constant=X" applies in
    # Eh/(rad²) for angle constraints. Use 0.5 by default which is moderate.
    ang_lines: list[tuple[int, int, int, float, str]] = []
    for r in (angle_restraints or []):
        i0, j0, k0, theta_rad, k_ev, label = r
        # adjust to 1-based
        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
        # only emit if all three indices are non-fixed (i.e., not in fix_indices);
        # xtb may complain if any constrained atom is fixed
        # Actually xtb does allow constraints involving fixed atoms — leave as is
        ang_lines.append((i1, j1, k1, np.degrees(theta_rad), label))

    with open(inp_path, "w") as f:
        for force, entries in sorted(by_force.items(), reverse=True):
            f.write(f"$constrain\n   force constant={force}\n")
            for i, j, d, label in entries:
                f.write(f"   distance: {i}, {j}, {d:.4f}    # {label}\n")
            f.write("$end\n")
        if ang_lines:
            f.write(f"$constrain\n   force constant={0.5 * k_scale}\n")
            for i, j, k, deg, label in ang_lines:
                f.write(f"   angle: {i}, {j}, {k}, {deg:.3f}    # {label}\n")
            f.write("$end\n")
        if fix_indices:
            f.write("$fix\n")
            f.write(f"   atoms: {','.join(str(i) for i in fix_indices)}\n")
            f.write("$end\n")
        f.write("$opt\n   maxcycle=200\n$end\n")

    log.info(f"  Constraints: {len(fix_indices)}/{len(cluster)} fixed "
             f"(ligand={n_lig}, bb={n_bb}, caps={n_cap}, unfrozen-bb={n_unfrozen})")
    log.info(f"  Restraints: {sum(len(v) for v in by_force.values())} distance, "
             f"{len(ang_lines)} angle (k_scale={k_scale}), skipped={n_skipped}")


def run_xtb_opt(
    cluster: list[Atom],
    workdir: Path,
    charge: int,
    catalytic: list[CatRes],
    contacts: list[DesignContact],
    gfn: int = 0,
    rigidity: str = "backbone",
    unfreeze_shell: int = 0,
    timeout_s: int = 600,
    angle_restraints: list | None = None,
    k_scale: float = 1.0,
    binary: str | None = None,            # path to xtb (default vendored XTB_BIN)
    extra_method_args: list | None = None,  # e.g. ["--gxtb"] for g-xTB
    polish_steps: int = 0,                 # if >0, run an unrestrained 2nd pass
) -> tuple[list[Atom] | None, str]:
    bin_path = binary or XTB_BIN
    if not os.path.isfile(bin_path):
        return None, f"xtb binary not found at {bin_path}"

    workdir.mkdir(parents=True, exist_ok=True)
    xyz_path = workdir / "input.xyz"
    inp_path = workdir / "xcontrol.inp"

    with open(xyz_path, "w") as f:
        f.write(f"{len(cluster)}\n\n")
        for a in cluster:
            f.write(f"{a.element:<2} {a.x:.6f} {a.y:.6f} {a.z:.6f}\n")

    write_xtb_input(cluster, inp_path, catalytic, contacts,
                    unfreeze_shell=unfreeze_shell, rigidity=rigidity,
                    angle_restraints=angle_restraints, k_scale=k_scale)

    if extra_method_args:
        method_args = list(extra_method_args)
    else:
        method_args = ["--gfnff"] if gfn == 0 else ["--gfn", str(gfn)]
    cmd = [bin_path, xyz_path.name] + method_args + [
        "--chrg", str(charge), "--opt", "normal", "--input", inp_path.name,
    ]
    # Vendored xtb needs libxtb.so + libquadmath on LD_LIBRARY_PATH (g-xtb is
    # statically linked so it doesn't strictly need it, but harmless).
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        XTB_LIB_DIRS + [env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    env.setdefault("OMP_NUM_THREADS", "1")
    log.info(f"  {Path(bin_path).name} {' '.join(method_args)} on {len(cluster)} atoms (charge={charge:+d})")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s, cwd=str(workdir), env=env)
    except subprocess.TimeoutExpired:
        return None, f"xTB timeout after {timeout_s}s"
    opt_xyz = workdir / "xtbopt.xyz"
    if not opt_xyz.exists():
        return None, (proc.stderr or proc.stdout or "")[-2000:]

    lines = opt_xyz.read_text().strip().split("\n")
    if len(lines) < len(cluster) + 2:
        return None, "xtbopt.xyz too short"
    relaxed = []
    for i, a in enumerate(cluster):
        parts = lines[2 + i].split()
        relaxed.append(a.with_pos(np.array([float(parts[1]), float(parts[2]), float(parts[3])])))

    # Polish stage: rewrite xcontrol.inp with $fix only (no $constrain blocks),
    # then re-optimise from the restrained geometry. xtb-side polish.
    if polish_steps > 0:
        polish_inp = workdir / "polish.inp"
        # Write only $fix block
        with open(polish_inp, "w") as f:
            f.write("$opt\n   maxcycle=" + str(polish_steps) + "\n$end\n")
            # Rebuild fix block by re-parsing original xcontrol.inp $fix line
            for line in open(inp_path):
                if line.strip().startswith("atoms:") and "$fix" in inp_path.read_text():
                    f.write("$fix\n   " + line.strip() + "\n$end\n")
                    break
        # Restart from the relaxed coords
        polish_xyz = workdir / "polish.xyz"
        with open(polish_xyz, "w") as f:
            f.write(f"{len(cluster)}\n\n")
            for a in relaxed:
                f.write(f"{a.element:<2} {a.x:.6f} {a.y:.6f} {a.z:.6f}\n")
        polish_cmd = [bin_path, polish_xyz.name] + method_args + [
            "--chrg", str(charge), "--opt", "normal", "--input", polish_inp.name,
        ]
        log.info(f"  xtb polish: {polish_steps} unrestrained steps")
        polish_proc = subprocess.run(polish_cmd, capture_output=True, text=True,
                                     timeout=timeout_s, cwd=str(workdir), env=env)
        polish_out = workdir / "xtbopt.xyz"
        if polish_out.exists() and polish_proc.returncode == 0:
            ll = polish_out.read_text().strip().split("\n")
            if len(ll) >= len(cluster) + 2:
                relaxed = [a.with_pos(np.array([float(p[1]), float(p[2]), float(p[3])]))
                           for a, p in zip(cluster,
                                           [ll[2 + i].split() for i in range(len(cluster))])]
                log.info(f"  xtb polish: OK")
        else:
            log.warning(f"  xtb polish failed; keeping restrained result")

    return relaxed, "OK"


# ──────────────────────────────────────────────────────────────────
# MACE backend (ASE + custom harmonic-distance restraint)
# ──────────────────────────────────────────────────────────────────

# Single source of truth for model paths: quantum_engine.config.MACE_MODELS,
# consumed by quantum_engine.calc.make_calc(). We import the dict directly
# here so the refine MACE backend's --backend choices stay in sync with the
# rest of the engine.
from quantum_engine.site import MACE_MODELS  # noqa: E402


def _build_ase_atoms_and_constraints(
    cluster: list[Atom],
    catalytic: list[CatRes],
    contacts: list[DesignContact],
    unfreeze_shell: int,
    rigidity: str,
):
    """Build an ASE Atoms object + FixAtoms + harmonic distance restraints."""
    from ase import Atoms as AseAtoms

    # PDB stores element symbols upper-cased ("ZN", "MG"); ASE wants "Zn", "Mg".
    symbols = [a.element.capitalize() for a in cluster]
    positions = np.array([a.pos for a in cluster])
    atoms = AseAtoms(symbols=symbols, positions=positions)

    backbone_atoms = set(BACKBONE_ATOMS)
    if rigidity == "backbone-cb":
        backbone_atoms |= {"CB"}

    unfrozen: set[tuple[str, int]] = set()
    if unfreeze_shell > 0:
        for c in catalytic:
            for delta in range(-unfreeze_shell, unfreeze_shell + 1):
                unfrozen.add((c.chain, c.rnum + delta))

    fix_idx_0based: list[int] = []
    n_lig = n_bb = n_cap = n_unfrozen = 0
    for i, a in enumerate(cluster):  # 0-indexed for ASE
        if a.is_hetatm:
            fix_idx_0based.append(i); n_lig += 1
        elif a.aname == "HCAP":
            fix_idx_0based.append(i); n_cap += 1
        elif a.rname in PROTEIN_RES and a.aname in backbone_atoms:
            if (a.chain, a.rnum) in unfrozen:
                n_unfrozen += 1
            else:
                fix_idx_0based.append(i); n_bb += 1

    return atoms, fix_idx_0based, dict(
        n_lig=n_lig, n_bb=n_bb, n_cap=n_cap, n_unfrozen=n_unfrozen)


def _build_distance_restraints(
    cluster: list[Atom],
    contacts: list[DesignContact],
    k_scale: float = 1.0,
):
    """Translate DesignContact list → list of (i, j, target, k) where k is a
    spring constant in eV/Å² (suitable for ASE).

    xTB Eh/Bohr² convention: 1 Eh/Bohr² ≈ 97.18 eV/Å². To stay reasonable,
    we cap k_eff at force * 30 eV/Å² × k_scale.
    """
    pairs = []
    n_skipped = 0
    for ct in contacts:
        i_1based = _index_in_cluster(cluster, ct.res_chain, ct.res_num, ct.res_atom)
        j_1based = next((k + 1 for k, a in enumerate(cluster)
                         if a.is_hetatm and a.aname == ct.lig_atom), None)
        if i_1based is None or j_1based is None:
            n_skipped += 1
            continue
        # convert to 0-based
        i, j = i_1based - 1, j_1based - 1
        # k in eV/Å² (heuristic): tier × 30  → covalent ~30, metal_coord ~9, hbond ~3, close ~0.9
        k_ev = ct.force * 30.0 * k_scale
        pairs.append((i, j, ct.distance, k_ev,
                      f"{ct.res_name}{ct.res_num}.{ct.res_atom}-{ct.lig_atom}({ct.kind})"))
    return pairs, n_skipped


def build_internal_angle_restraints(
    cluster: list[Atom],
    design_atoms: list[Atom],
    catalytic: list[CatRes],
    k_ev_per_rad2: float = 8.0,
) -> list[tuple[int, int, int, float, float, str]]:
    """For each catalytic-residue sidechain, extract the *design's*
    internal angles (CA-CB-CG, CB-CG-CD, etc.) and return them as
    cluster-indexed (i, j, k, target_rad, k_eV/rad², label) tuples
    suitable for ASE HarmonicAngle restraints.

    This is what fixes the "CA-CB-CG drifts to 130°" failure mode: the
    contact-map distance restraints alone don't penalise sidechain
    bending; an explicit angle restraint anchored to design's value
    does.
    """
    # Only restrain SIDECHAIN PIVOT angles (CA-CB-CG and onward through chi
    # rotations) — never aromatic-ring or sp2-carboxylate angles, since those
    # are fixed by chemistry and MLFF handles them well. Restraining a ring
    # angle to design's value can fight aromaticity and force unphysical
    # geometry.
    SC_ANG = {
        "ALA": [], "GLY": [], "PRO": [], "SER": [], "CYS": [], "THR": [],
        "ARG": [("CA","CB","CG"),("CB","CG","CD"),("CG","CD","NE")],
        "ASN": [("CA","CB","CG")],
        "ASP": [("CA","CB","CG")],
        "GLN": [("CA","CB","CG"),("CB","CG","CD")],
        "GLU": [("CA","CB","CG"),("CB","CG","CD")],
        "HIS": [("CA","CB","CG")],
        "ILE": [("CA","CB","CG1"),("CB","CG1","CD1")],
        "LEU": [("CA","CB","CG")],
        "LYS": [("CA","CB","CG"),("CB","CG","CD"),("CG","CD","CE"),("CD","CE","NZ")],
        "MET": [("CA","CB","CG"),("CB","CG","SD"),("CG","SD","CE")],
        "PHE": [("CA","CB","CG")],
        "TRP": [("CA","CB","CG")],
        "TYR": [("CA","CB","CG")],
        "VAL": [("CA","CB","CG1")],
    }

    # Build (chain,rnum,aname) → cluster-0-based-index lookup
    cluster_idx: dict[tuple[str, int, str], int] = {}
    for k, a in enumerate(cluster):
        cluster_idx[(a.chain, a.rnum, a.aname)] = k

    # Build (chain,rnum,aname) → position lookup for design
    design_pos: dict[tuple[str, int, str], np.ndarray] = {}
    for a in design_atoms:
        if not a.is_hetatm:
            design_pos[(a.chain, a.rnum, a.aname)] = a.pos

    rests = []
    for c in catalytic:
        for trip in SC_ANG.get(c.rname, []):
            a0, a1, a2 = trip
            i = cluster_idx.get((c.chain, c.rnum, a0))
            j = cluster_idx.get((c.chain, c.rnum, a1))
            k = cluster_idx.get((c.chain, c.rnum, a2))
            d0 = design_pos.get((c.chain, c.rnum, a0))
            d1 = design_pos.get((c.chain, c.rnum, a1))
            d2 = design_pos.get((c.chain, c.rnum, a2))
            if i is None or j is None or k is None \
                    or d0 is None or d1 is None or d2 is None:
                continue
            v1, v2 = d0 - d1, d2 - d1
            cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            theta = float(np.arccos(np.clip(cos, -1, 1)))
            rests.append((i, j, k, theta, k_ev_per_rad2,
                          f"{c.rname}{c.rnum}.{a0}-{a1}-{a2}"))
    return rests


def _make_harmonic_distance_class():
    """Build the HarmonicDistance ASE constraint class lazily (so that ASE is
    only imported when MACE backend is used)."""
    import numpy as _np

    class HarmonicDistance:
        """Two-sided harmonic distance restraint between atoms i and j.

        E   = 0.5 * k * (d_ij - d0)²
        F_i = -∂E/∂r_i = +k * (d - d0) * (r_j - r_i)/d
        F_j = -F_i
        """
        def __init__(self, i: int, j: int, target: float, k: float):
            self.i = int(i); self.j = int(j)
            self.target = float(target); self.k = float(k)

        def adjust_positions(self, atoms, new):
            return  # soft restraint — never modify positions

        def adjust_forces(self, atoms, forces):
            r = atoms.positions[self.j] - atoms.positions[self.i]
            d = float(_np.linalg.norm(r))
            if d < 1e-9:
                return
            unit = r / d
            f_mag = self.k * (d - self.target)
            forces[self.i] += f_mag * unit
            forces[self.j] -= f_mag * unit

        def get_indices(self):
            return [self.i, self.j]

        def index_shuffle(self, atoms, ind):
            new_i = ind.index(self.i) if self.i in ind else None
            new_j = ind.index(self.j) if self.j in ind else None
            if new_i is None or new_j is None:
                raise IndexError("HarmonicDistance index removed by shuffle")
            self.i, self.j = new_i, new_j

        def todict(self):
            return {"name": "HarmonicDistance",
                    "kwargs": {"i": self.i, "j": self.j,
                               "target": self.target, "k": self.k}}

        def get_removed_dof(self, atoms):
            return 0  # soft restraint, no DOFs removed

    return HarmonicDistance


def _make_harmonic_angle_class():
    """Two-sided harmonic *angle* restraint between atoms i, j, k (j is the
    vertex). Energy E = 0.5 * k_ang * (theta - theta_0)^2.

    This is what fixes the CA-CB-CG → 130° distortion: distance restraints
    on far-away contacts can pull the chain in ways that break sidechain
    valence; an angle restraint anchored to design's value resists that.
    """
    import numpy as _np

    class HarmonicAngle:
        def __init__(self, i: int, j: int, k: int, theta0: float, k_ang: float):
            self.i = int(i); self.j = int(j); self.k = int(k)
            self.theta0 = float(theta0); self.k_ang = float(k_ang)

        def adjust_positions(self, atoms, new):
            return

        def adjust_forces(self, atoms, forces):
            r_i = atoms.positions[self.i]
            r_j = atoms.positions[self.j]
            r_k = atoms.positions[self.k]
            v1 = r_i - r_j; v2 = r_k - r_j
            n1 = _np.linalg.norm(v1); n2 = _np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9: return
            cos_t = _np.dot(v1, v2) / (n1 * n2)
            cos_t = max(-1.0 + 1e-9, min(1.0 - 1e-9, cos_t))
            theta = _np.arccos(cos_t)
            sin_t = _np.sqrt(1 - cos_t * cos_t)
            # dtheta/dr_i = -1/sin(theta) * d(cos_t)/dr_i
            # d(cos_t)/dr_i = (v2/(n1*n2) - cos_t * v1 / n1**2)
            dcos_dri = (v2 - cos_t * v1 * (n2 / n1)) / (n1 * n2)
            dcos_drk = (v1 - cos_t * v2 * (n1 / n2)) / (n1 * n2)
            dcos_drj = -(dcos_dri + dcos_drk)
            inv_sin = -1.0 / sin_t
            f_factor = self.k_ang * (theta - self.theta0)
            # F_x = -dE/dx = -k * (theta-theta0) * dtheta/dx
            #     = -k * (theta-theta0) * (-1/sin) * dcos/dx
            #     = k * (theta-theta0) / sin * dcos/dx
            forces[self.i] += -f_factor * inv_sin * dcos_dri
            forces[self.j] += -f_factor * inv_sin * dcos_drj
            forces[self.k] += -f_factor * inv_sin * dcos_drk

        def get_indices(self):
            return [self.i, self.j, self.k]

        def index_shuffle(self, atoms, ind):
            for nm in ("i", "j", "k"):
                v = getattr(self, nm)
                if v not in ind:
                    raise IndexError("HarmonicAngle index removed by shuffle")
                setattr(self, nm, ind.index(v))

        def todict(self):
            return {"name": "HarmonicAngle",
                    "kwargs": {"i": self.i, "j": self.j, "k": self.k,
                               "theta0": self.theta0, "k_ang": self.k_ang}}

        def get_removed_dof(self, atoms):
            return 0

    return HarmonicAngle


def run_mace_opt(
    cluster: list[Atom],
    workdir: Path,
    charge: int,
    catalytic: list[CatRes],
    contacts: list[DesignContact],
    model: str = "mace-mp",
    device: str = "cpu",
    dtype: str = "float32",
    unfreeze_shell: int = 1,
    rigidity: str = "backbone",
    fmax: float = 0.05,
    max_steps: int = 200,
    k_scale: float = 1.0,
    angle_restraints: list | None = None,  # tuples from build_internal_angle_restraints
    polish_steps: int = 0,                  # if >0, run an unrestrained pass after
) -> tuple[list[Atom] | None, str]:
    """ASE BFGS optimisation with MACE forcefield + FixAtoms + harmonic
    distance + (optional) harmonic angle restraints + (optional) polish pass."""
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        from ase.constraints import FixAtoms
        from ase.optimize import BFGS
        from mace.calculators import MACECalculator
    except ImportError as e:
        return None, f"ASE/MACE import failed: {e}. Use a python env with ase + mace."

    model_path = MACE_MODELS.get(model, model)
    if not os.path.isfile(model_path):
        return None, f"MACE model not found: {model_path}"

    atoms, fix_idx, counts = _build_ase_atoms_and_constraints(
        cluster, catalytic, contacts, unfreeze_shell, rigidity)
    pairs, n_skipped = _build_distance_restraints(cluster, contacts, k_scale=k_scale)

    HarmonicDistance = _make_harmonic_distance_class()
    HarmonicAngle = _make_harmonic_angle_class()
    constraints: list = [FixAtoms(indices=fix_idx)] if fix_idx else []
    constraints += [HarmonicDistance(i, j, d, k) for i, j, d, k, _ in pairs]
    if angle_restraints:
        constraints += [HarmonicAngle(i, j, k, t, ka)
                        for i, j, k, t, ka, _ in angle_restraints]
    atoms.set_constraint(constraints)

    log.info(f"  MACE constraints: {len(fix_idx)}/{len(cluster)} fixed "
             f"(ligand={counts['n_lig']}, bb={counts['n_bb']}, caps={counts['n_cap']}, "
             f"unfrozen-bb={counts['n_unfrozen']}); "
             f"{len(pairs)} distance restraints, "
             f"{len(angle_restraints) if angle_restraints else 0} angle restraints, "
             f"skipped={n_skipped}")
    log.info(f"  Loading MACE model '{model}' on device='{device}' …")

    # mace-omol is charge-aware → pass charge via atoms.info
    if "omol" in model:
        atoms.info["charge"] = charge

    calc = MACECalculator(model_paths=model_path, device=device,
                          default_dtype=dtype)
    atoms.calc = calc

    log.info(f"  Running BFGS opt: fmax={fmax} eV/Å, max_steps={max_steps}")
    log_path = workdir / "opt.log"
    # No trajectory file: newer ASE versions try to re-read it on init and
    # fail to deserialise our custom HarmonicDistance/Angle constraints.
    opt = BFGS(atoms, logfile=str(log_path))
    try:
        converged = opt.run(fmax=fmax, steps=max_steps)
    except Exception as e:
        import traceback as _tb
        return None, f"BFGS failed: {type(e).__name__}: {e}\n{_tb.format_exc()[-2000:]}"

    log.info(f"  BFGS converged={converged} after {opt.nsteps} steps "
             f"(final fmax={float(np.linalg.norm(atoms.get_forces(), axis=1).max()):.3f} eV/Å)")

    # Polish stage: drop the harmonic restraints (keep FixAtoms only) and let
    # MLFF's intrinsic bond/angle terms relax internal valence. This is the
    # standard QM/MM 2-stage workflow and recovers sp3 geometry that the
    # restrained-pull pass may have distorted.
    if polish_steps > 0:
        atoms.set_constraint([FixAtoms(indices=fix_idx)] if fix_idx else [])
        polish_log = workdir / "polish.log"
        log.info(f"  Polish pass: {polish_steps} unrestrained BFGS steps "
                 f"(restraints removed, {len(fix_idx)} atoms still frozen)")
        opt2 = BFGS(atoms, logfile=str(polish_log))
        try:
            polish_converged = opt2.run(fmax=fmax, steps=polish_steps)
        except Exception as e:
            log.warning(f"  Polish stage failed: {e}; keeping restrained result")
            polish_converged = False
        else:
            log.info(f"  Polish converged={polish_converged} after {opt2.nsteps} steps")

    relaxed = []
    for i, a in enumerate(cluster):
        relaxed.append(a.with_pos(atoms.positions[i]))
    return relaxed, "OK" if converged else "did not fully converge"


# ──────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────

def report_contact_deviations(
    contacts: list[DesignContact],
    af3_atoms: list[Atom],
    final_atoms: list[Atom],
    lig_atoms_design: list[Atom],
) -> str:
    """Return a multi-line table comparing design distance vs initial AF3 vs final."""
    def find(atoms, chain, rnum, aname):
        return next((a for a in atoms if a.chain == chain and a.rnum == rnum
                     and a.aname == aname), None)

    lines = []
    lines.append(f"{'res':<10} {'cat':<5} → {'lig':<4}  {'kind':<11}  "
                 f"{'design':>7}  {'AF3':>7}  {'final':>7}  {'Δ_AF3':>7}  {'Δ_final':>7}")
    lines.append("-" * 92)
    for c in contacts:
        a_des = find(lig_atoms_design, "any", -1, c.lig_atom)  # placeholder, design-frame
        a_af = find(af3_atoms, c.res_chain, c.res_num, c.res_atom)
        a_fin = find(final_atoms, c.res_chain, c.res_num, c.res_atom)
        # Use the AF3-aligned ligand position as the reference for both AF3 and final
        a_lig_af3 = next((a for a in af3_atoms if a.is_hetatm and a.aname == c.lig_atom), None)
        a_lig_fin = next((a for a in final_atoms if a.is_hetatm and a.aname == c.lig_atom), None)
        if not (a_af and a_fin and a_lig_af3 and a_lig_fin):
            continue
        d_af = np.linalg.norm(a_af.pos - a_lig_af3.pos)
        d_fin = np.linalg.norm(a_fin.pos - a_lig_fin.pos)
        delta_af = d_af - c.distance
        delta_fin = d_fin - c.distance
        flag = ""
        if abs(delta_fin) < abs(delta_af) - 0.05:
            flag = " ✓"
        elif abs(delta_fin) > abs(delta_af) + 0.05:
            flag = " ⚠"
        lines.append(f"{c.res_name}{c.res_num:<6} {c.res_atom:<5} → {c.lig_atom:<4}  "
                     f"{c.kind:<11}  {c.distance:>7.3f}  {d_af:>7.3f}  {d_fin:>7.3f}  "
                     f"{delta_af:>+7.3f}  {delta_fin:>+7.3f}{flag}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# Stitch refined cluster back
# ──────────────────────────────────────────────────────────────────

def stitch_back(full: list[Atom], cluster_refined: list[Atom]) -> list[Atom]:
    refined: dict[tuple[str, int, str], np.ndarray] = {}
    for a in cluster_refined:
        if a.aname == "HCAP":
            continue
        refined[(a.chain, a.rnum, a.aname)] = a.pos
    return [a.with_pos(refined[(a.chain, a.rnum, a.aname)])
            if (a.chain, a.rnum, a.aname) in refined else a
            for a in full]


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("design_pdb", help="Design PDB (with REMARK 666 + ligand)")
    p.add_argument("aligned_pdb",
                   help="AF3 prediction already aligned to design (output of "
                        "align_prediction_to_ref_pdb_and_copy_lig.py)")
    p.add_argument("-o", "--output", required=True, help="Refined output PDB")
    p.add_argument("--ptm", action="append", default=[],
                   help="PTM spec: 'CHAIN/RES/CAT_IDX:NCAA' "
                        "(e.g., 'A/LYS/3:KCX'). Repeatable.")
    p.add_argument("--radius", type=float, default=6.0,
                   help="Cluster cutoff radius around ligand (Å, default 6.0)")
    p.add_argument("--contact-radius", type=float, default=4.5,
                   help="Catres-ligand pair cutoff for design contact map (default 4.5)")
    p.add_argument("--max-contacts-per-res", type=int, default=6)
    p.add_argument("--backend", default="mace-mp",
                   choices=["mace-mp", "mace-omol", "mace-off",
                            "mace-polar-m", "mace-polar-s", "mace-polar-l",
                            "xtb", "g-xtb"],
                   help="Relaxation engine. Default mace-mp (r2SCAN, all elements). "
                        "mace-omol is charge-aware + TS-trained. "
                        "mace-polar-m is the polarisable gold-standard for ionic systems. "
                        "g-xtb uses the Grimme g-xTB binary (--gxtb method, modified xtb 6.7.1).")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="ASE device for MACE (default cpu)")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float64"],
                   help="MACE default_dtype — float32 is ~2× cheaper (default)")
    p.add_argument("--gfn", type=int, default=0, choices=[0, 1, 2],
                   help="(xtb backend only) 0=GFN-FF, 1/2=GFN-xTB")
    p.add_argument("--fmax", type=float, default=0.05,
                   help="Force convergence (eV/Å, MACE backend, default 0.05)")
    p.add_argument("--max-steps", type=int, default=200,
                   help="Maximum optimizer steps (MACE backend)")
    p.add_argument("--k-scale", type=float, default=1.0,
                   help="Multiplier on all restraint force constants")
    p.add_argument("--angle-restraints", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Add harmonic restraints on catres-sidechain internal angles "
                        "(CA-CB-CG, etc.) using design's values as targets. "
                        "Fixes the 'sidechain valence drifts to 130°' failure. "
                        "Default ON; --no-angle-restraints to disable.")
    p.add_argument("--polish-steps", type=int, default=0,
                   help="After the restrained pass, run N more optimisation steps "
                        "with all distance/angle restraints removed (FixAtoms only). "
                        "Lets MLFF/xtb's intrinsic bond/angle terms relax internal "
                        "valence. Default 0 (no polish). 30-100 is reasonable.")
    p.add_argument("--rigidity", default="backbone",
                   choices=["backbone", "backbone-cb"])
    p.add_argument("--unfreeze-shell", type=int, default=1,
                   help="Unfreeze backbone of residues within ±N of catalytic "
                        "(default 1 — catres backbone free)")
    p.add_argument("--ligand-charge", action="append", default=[],
                   metavar="RESNAME:CHARGE",
                   help="Net charge of a HETATM residue (e.g., 'YYE:0', 'ZN:2'). "
                        "Repeatable. Defaults: ZN +2, MG +2, MN +2, NA +1, CL -1, "
                        "HOH 0; protein residues use canonical pH-7 formal charges; "
                        "PTMs use their absolute charge (KCX = -1).")
    p.add_argument("--total-charge", type=int, default=None,
                   help="Override the auto-computed total cluster charge (use only "
                        "if you know what you're doing).")
    p.add_argument("--extra-charge", type=int, default=0,
                   help="(Legacy) Add this to the auto-computed charge. Prefer "
                        "--ligand-charge / --total-charge for new code.")
    p.add_argument("--workdir", default=None)
    p.add_argument("--keep-workdir", action="store_true")
    args = p.parse_args()

    design_pdb = Path(args.design_pdb).resolve()
    aligned_pdb = Path(args.aligned_pdb).resolve()
    output_pdb = Path(args.output).resolve()
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir).resolve() if args.workdir else \
              Path(tempfile.mkdtemp(prefix="qcb_refine_v2_"))
    workdir.mkdir(parents=True, exist_ok=True)
    log.info(f"Workdir: {workdir}")

    # ─── Parse design + catres ────────────────────────────────────
    log.info("=" * 70)
    log.info("Step 1: Parse design + REMARK 666")
    log.info("=" * 70)
    design_atoms, _ = parse_pdb(design_pdb)
    _, catalytic = parse_remark666(design_pdb)
    log.info(f"  Catalytic residues: " + ", ".join(
        f"{c.rname}{c.rnum}({c.cat_idx})" for c in catalytic))

    ptms = [parse_ptm_spec(s) for s in args.ptm]
    if ptms:
        cat_by_idx = {c.cat_idx: c for c in catalytic}
        ptm_descrs = []
        for p in ptms:
            c = cat_by_idx.get(p.cat_idx)
            label = f"{c.rname}{c.rnum}" if c else f"slot{p.cat_idx}"
            ptm_descrs.append(f"{label}→{p.new_name}")
        log.info(f"  PTM specs: " + ", ".join(ptm_descrs))

    # ─── Build design contact map ─────────────────────────────────
    log.info("=" * 70)
    log.info(f"Step 2: Build design contact map (radius={args.contact_radius} Å)")
    log.info("=" * 70)
    contacts = build_design_contact_map(
        design_atoms, catalytic, radius=args.contact_radius)
    contacts = filter_top_contacts(contacts, max_per_residue=args.max_contacts_per_res)
    log.info(f"  {len(contacts)} contacts retained")
    by_kind = {}
    for c in contacts:
        by_kind.setdefault(c.kind, 0)
        by_kind[c.kind] += 1
    for k, n in sorted(by_kind.items()):
        log.info(f"    {k:12s}: {n}")
    for c in contacts:
        log.info(f"    {c.res_name}{c.res_num:<3} {c.res_atom:<4} → {c.lig_atom:<4} "
                 f"= {c.distance:.3f} Å  [{c.kind}, k={c.force}]")

    # ─── Load aligned AF3 + apply PTMs ────────────────────────────
    log.info("=" * 70)
    log.info("Step 3: Load aligned AF3, apply PTM protonation")
    log.info("=" * 70)
    af3_atoms, headers = parse_pdb(aligned_pdb)
    af3_atoms_fixed, _charge_delta_unused, kcx_residues = apply_kcx_protonation(
        af3_atoms, catalytic, ptms)
    log.info(f"  KCX residues: {kcx_residues or 'none'}")

    # ─── Cluster + cap ────────────────────────────────────────────
    log.info("=" * 70)
    log.info(f"Step 4: Build cluster (radius={args.radius} Å)")
    log.info("=" * 70)
    cluster, keep = build_cluster(af3_atoms_fixed, catalytic, radius=args.radius)
    cluster_capped = cap_backbone(cluster, af3_atoms_fixed, keep)
    log.info(f"  Cluster: {len(keep)} residues, {len(cluster_capped)} atoms (with caps)")

    # ─── Cluster net charge ───────────────────────────────────────
    log.info("=" * 70)
    log.info("Step 4b: Compute cluster net charge")
    log.info("=" * 70)
    user_lig_charges = parse_ligand_charge_specs(args.ligand_charge)
    auto_charge, charge_audit = compute_cluster_charge(
        cluster_capped, ptms, catalytic, user_lig_charges)
    for line in charge_audit:
        log.info(line)
    n_e, even_e = cluster_electron_parity(cluster_capped, auto_charge)
    log.info(f"  Auto-computed charge: {auto_charge:+d}  ({n_e} electrons, "
             f"{'closed-shell' if even_e else 'OPEN-SHELL — odd e⁻'})")

    if args.total_charge is not None:
        total_charge = args.total_charge
        log.info(f"  --total-charge override: using {total_charge:+d}")
    else:
        total_charge = auto_charge + args.extra_charge

    if args.extra_charge:
        log.info(f"  --extra-charge: {args.extra_charge:+d} → total {total_charge:+d}")

    n_e_final, even_e_final = cluster_electron_parity(cluster_capped, total_charge)
    if not even_e_final:
        # User specified: no radicals, always closed-shell. If our auto sum is
        # odd-electron, nudge by ±1 toward zero charge (the charge most likely
        # right for a typical enzyme active site) and warn loudly.
        nudged = total_charge + (1 if total_charge < 0 else -1)
        n_e_nudged, even_nudged = cluster_electron_parity(cluster_capped, nudged)
        if even_nudged and args.total_charge is None:
            log.warning(f"  ⚠️ Auto-charge {total_charge:+d} gives ODD electrons "
                        f"({n_e_final}); nudging to {nudged:+d} ({n_e_nudged} e⁻) "
                        f"to satisfy closed-shell. Pass --total-charge to override.")
            total_charge = nudged
        else:
            log.warning(f"  ⚠️ Final charge {total_charge:+d} gives ODD electrons "
                        f"({n_e_final}). Even ±1 didn't help — pass --ligand-charge "
                        f"or --total-charge explicitly.")

    cluster_dir = output_pdb.parent / f"{output_pdb.stem}_cluster"
    cluster_dir.mkdir(exist_ok=True)
    write_pdb(cluster_capped, cluster_dir / "input.pdb")
    log.info(f"  Cluster input → {cluster_dir / 'input.pdb'}")

    # ─── Constrained opt (MACE+ASE or xTB) ────────────────────────
    log.info("=" * 70)
    log.info(f"Step 5: {args.backend} constrained opt (rigidity={args.rigidity}, "
             f"unfreeze=±{args.unfreeze_shell})")
    log.info("=" * 70)
    # Build sidechain internal angle restraints from design (if enabled).
    # For MACE: pass HarmonicAngle ASE constraints. For xTB: emit "angle: i,j,k"
    # lines into xcontrol.inp.
    angle_rests = None
    if args.angle_restraints:
        angle_rests = build_internal_angle_restraints(
            cluster_capped, design_atoms, catalytic,
            k_ev_per_rad2=8.0 * args.k_scale)
        log.info(f"  Sidechain angle restraints: {len(angle_rests)} "
                 f"(k_eV/rad² = {8.0 * args.k_scale:.2f})")

    if args.backend == "xtb":
        relaxed, msg = run_xtb_opt(
            cluster_capped, workdir / "xtb",
            charge=total_charge, catalytic=catalytic, contacts=contacts,
            gfn=args.gfn, rigidity=args.rigidity,
            unfreeze_shell=args.unfreeze_shell,
            angle_restraints=angle_rests,
            k_scale=args.k_scale,
            polish_steps=args.polish_steps,
        )
    elif args.backend == "g-xtb":
        relaxed, msg = run_xtb_opt(
            cluster_capped, workdir / "gxtb",
            charge=total_charge, catalytic=catalytic, contacts=contacts,
            gfn=args.gfn, rigidity=args.rigidity,
            unfreeze_shell=args.unfreeze_shell,
            angle_restraints=angle_rests,
            k_scale=args.k_scale,
            binary=GXTB_BIN, extra_method_args=["--gxtb"],
            polish_steps=args.polish_steps,
        )
    else:
        relaxed, msg = run_mace_opt(
            cluster_capped, workdir / "mace",
            charge=total_charge, catalytic=catalytic, contacts=contacts,
            model=args.backend, device=args.device, dtype=args.dtype,
            unfreeze_shell=args.unfreeze_shell, rigidity=args.rigidity,
            fmax=args.fmax, max_steps=args.max_steps, k_scale=args.k_scale,
            angle_restraints=angle_rests,
            polish_steps=args.polish_steps,
        )
    if relaxed is None:
        log.error(f"  Optimisation failed: {msg}")
        sys.exit(1)
    log.info(f"  Opt status: {msg}")

    # Diagnostics
    diff = np.array([a.pos for a in cluster_capped]) - np.array([a.pos for a in relaxed])
    rmsd_all = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    lig_idx = [i for i, a in enumerate(cluster_capped) if a.is_hetatm]
    lig_rmsd = float(np.sqrt(np.mean(np.sum(diff[lig_idx] * diff[lig_idx], axis=1)))) \
               if lig_idx else 0.0
    log.info(f"  All-atom cluster RMSD: {rmsd_all:.4f} Å")
    log.info(f"  Ligand RMSD (should be ≈0): {lig_rmsd:.6f} Å")

    write_pdb(relaxed, cluster_dir / "refined.pdb")
    log.info(f"  Cluster refined → {cluster_dir / 'refined.pdb'}")

    # ─── Stitch back ──────────────────────────────────────────────
    log.info("=" * 70)
    log.info("Step 6: Stitch refined cluster back into full structure")
    log.info("=" * 70)
    final = stitch_back(af3_atoms_fixed, relaxed)
    extra_remarks = [
        f"REMARK QCB REFINE_V2 design={design_pdb.name} aligned={aligned_pdb.name}",
        f"REMARK QCB METHOD backend={args.backend} rigidity={args.rigidity} "
        f"unfreeze=±{args.unfreeze_shell} radius={args.radius}",
        f"REMARK QCB CHARGE total={total_charge:+d} (auto={auto_charge:+d}, "
        f"extra={args.extra_charge}, override={args.total_charge})",
        f"REMARK QCB PTMs " + (",".join(args.ptm) or "none"),
        f"REMARK QCB RMSD all={rmsd_all:.3f} ligand={lig_rmsd:.6f}",
    ]
    write_pdb(final, output_pdb, headers=headers, extra_remarks=extra_remarks)
    log.info(f"  Final → {output_pdb}")

    # ─── Deviation table ──────────────────────────────────────────
    log.info("=" * 70)
    log.info("Contact deviations (design → AF3 → final)")
    log.info("=" * 70)
    lig_design = [a for a in design_atoms if a.is_hetatm]
    table = report_contact_deviations(contacts, af3_atoms_fixed, final, lig_design)
    for line in table.split("\n"):
        log.info(line)

    if not args.keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)

    log.info("=" * 70)
    log.info(f"DONE: refined → {output_pdb}")
    log.info(f"  PyMOL: load {design_pdb}; load {aligned_pdb}; load {output_pdb}; "
             f"load {cluster_dir/'input.pdb'}; load {cluster_dir/'refined.pdb'}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
