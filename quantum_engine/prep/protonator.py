#!/usr/bin/env python
"""protonator.py — deterministic, staged protonation of theozyme PDB/CIF structures.

A "theozyme" is a collection of isolated catalytic amino-acid residues (sometimes
small peptide-bonded fragments) together with a ligand and metals. This tool
protonates the *protein* deterministically in explicit, ordered stages:

    Stage 1  cap open backbone N/C termini (geometry-detected)
    Stage 2  PTM / covalent safety checks (warnings only)
    Stage 3  pH-aware canonical protonation
    Stage 4  histidine tautomer (metal / clash / H-bond geometry)
    Stage 5  propka refinement of ambiguous states
    Stage 6  optional multi-protomer output
    +        optional cheap CPU MLFF relax of ONLY the new hydrogens

HETATM (ligand / metal / water) is assumed already protonated and is left alone.

The file is organised into clearly delimited sections and is built increment by
increment; each stage is a pure function over one shared ``ProtonationState``.

This module is runnable standalone (``python quantum_engine/prep/protonator.py
-i in.pdb``) and is also wired as the ``qcb protonate`` subcommand.
"""
from __future__ import annotations

import sys
from pathlib import Path

# --- make the quantum_engine package importable when run directly ------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import copy as _copy
import heapq
import logging
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field, replace

import numpy as np

# ============================================================================
# Logging
# ============================================================================
log = logging.getLogger("protonator")


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
        force=True,
    )


def banner(msg: str) -> None:
    """Emit a verbose stage banner, preceded by a blank separator line."""
    if log.isEnabledFor(logging.INFO):       # blank line between stages
        print(file=sys.stderr)
    log.info("=" * 72)
    log.info(msg)
    log.info("=" * 72)


# ============================================================================
# Constants
# ============================================================================
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

# protonation-state / naming variants that still map to a standard residue
RESNAME_ALIASES = {
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
    "CYX": "CYS", "CYM": "CYS",
    "ASH": "ASP", "GLH": "GLU", "LYN": "LYS", "TYM": "TYR",
    "SED": "SER", "THD": "THR", "ARN": "ARG",
    "MSE": "MET",
}
PROTEIN_RESNAMES = STANDARD_AA | set(RESNAME_ALIASES)

METALS = {
    "LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA", "MN", "FE", "CO",
    "NI", "CU", "ZN", "CD", "HG", "MO", "W", "V", "CR", "PT", "PD", "AU", "AG",
}
TWO_LETTER_ELEMENTS = METALS | {"CL", "BR", "SE", "SI"}

WATER_RESNAMES = {"HOH", "WAT", "H2O", "DOD", "TIP", "TIP3", "SOL"}


def canonical_resname(resname: str) -> str:
    """Map a residue-name variant to its canonical standard-AA name (or itself)."""
    rn = resname.strip().upper()
    return RESNAME_ALIASES.get(rn, rn)


# ============================================================================
# Data model
# ============================================================================
@dataclass
class Atom:
    """A single atom record."""

    record: str          # "ATOM" or "HETATM"
    name: str            # atom name, e.g. "CA", "HB2"
    resname: str
    chain: str
    resid: int
    icode: str
    xyz: np.ndarray      # shape (3,), float
    element: str
    altloc: str = ""
    occ: float = 1.0
    bfactor: float = 0.0
    charge: str = ""     # raw 2-char PDB charge field, usually ""
    serial: int = 0      # (re)assigned on write
    tag: str = ""        # "" (input) | "cap" | "h" — script-added-atom marker

    @property
    def is_hydrogen(self) -> bool:
        return self.element.upper() in ("H", "D")

    @property
    def reskey(self) -> tuple:
        """(chain, resid, icode) — unique residue identifier."""
        return (self.chain, self.resid, self.icode)

    def copy(self) -> "Atom":
        """Deep-ish copy: keyword-safe, with a fresh coordinate array."""
        return replace(self, xyz=np.array(self.xyz, float))


@dataclass
class ProtonationState:
    """The single object that flows through every stage."""

    atoms: list[Atom] = field(default_factory=list)
    header_lines: list[str] = field(default_factory=list)
    source_path: str = ""
    source_format: str = "pdb"
    warnings: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    residue_records: dict = field(default_factory=dict)   # reskey -> dict
    settings: dict = field(default_factory=dict)
    # NOTE: residue_records / settings are plain dicts. When Stage 6 branches
    # protomers it must deep-copy them — a dedicated copy() will be added then.

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        log.warning(msg)

    def note(self, msg: str) -> None:
        self.decisions.append(msg)
        log.info(msg)

    def copy(self) -> "ProtonationState":
        """Independent copy — fresh atoms and deep-copied residue records
        (used by Stage 6 to branch protomers)."""
        return ProtonationState(
            atoms=[a.copy() for a in self.atoms],
            header_lines=list(self.header_lines),
            source_path=self.source_path, source_format=self.source_format,
            warnings=list(self.warnings), decisions=list(self.decisions),
            residue_records=_copy.deepcopy(self.residue_records),
            settings=_copy.deepcopy(self.settings))


def group_residues(atoms: list[Atom]) -> dict:
    """Return reskey -> list[Atom], preserving first-seen residue order."""
    groups: dict = {}
    for a in atoms:
        groups.setdefault(a.reskey, []).append(a)
    return groups


def reskey_str(reskey: tuple) -> str:
    """Human-readable residue key, e.g. 'A:212' or 'A:212B'."""
    chain, resid, icode = reskey
    return f"{chain.strip() or '_'}:{resid}{icode.strip()}"


# ============================================================================
# I/O — reading
# ============================================================================
# Record types that are NOT structural-header lines (excluded from header_lines).
_NON_HEADER_RECORDS = {
    "ATOM", "HETATM", "ANISOU", "SIGATM", "SIGUIJ",
    "TER", "CONECT", "MODEL", "ENDMDL", "END", "MASTER", "NUMMDL", "SPLIT",
}


def _infer_element(name: str, record: str) -> str:
    """Infer an element symbol from an atom name when the element field is blank.

    Record-aware: ATOM-record atoms are organic only (never a two-letter metal),
    so e.g. ``HG11`` / ``CD1`` / ``CA`` correctly resolve to H / C / C, not
    mercury / cadmium / calcium. HETATM atoms may be bare metal ions.
    """
    s = name.strip().lstrip("0123456789")
    if not s:
        return ""
    if record == "ATOM":
        if len(s) == 2 and s.upper() == "SE":      # selenomethionine
            return "SE"
        return s[0].upper()
    # HETATM — a bare two-letter name may be a metal / halide ion
    if len(s) == 2 and s.upper() in TWO_LETTER_ELEMENTS:
        return s.upper()
    return s[0].upper()


def _parse_pdb_atom_line(line: str) -> Atom:
    line = line.rstrip("\n").ljust(80)
    record = line[0:6].strip()
    name = line[12:16].strip()
    altloc = line[16].strip()
    resname = line[17:20].strip()
    chain = line[21].strip() or " "
    resid = int(line[22:26])
    icode = line[26].strip()
    x = float(line[30:38])
    y = float(line[38:46])
    z = float(line[46:54])
    occ = float(line[54:60].strip() or "1.0")
    bfac = float(line[60:66].strip() or "0.0")
    element = line[76:78].strip().upper()
    charge = line[78:80].strip()
    if not element:
        element = _infer_element(name, record)
    return Atom(record=record, name=name, resname=resname, chain=chain,
                resid=resid, icode=icode, xyz=np.array([x, y, z], float),
                element=element, altloc=altloc, occ=occ, bfactor=bfac,
                charge=charge)


def read_pdb(path: Path) -> tuple[list[Atom], list[str]]:
    """Parse a PDB file → (atoms, header_lines). First model only; CONECT dropped.

    Only non-coordinate lines seen *before* the first ATOM/HETATM are kept as
    headers; anything unrecognised after coordinates is warned about and dropped.
    A malformed ATOM/HETATM line is a hard error (no silent atom loss).
    """
    atoms: list[Atom] = []
    header_lines: list[str] = []
    n_models = 0
    seen_coords = False
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        rec = line[:6].strip()
        if rec == "MODEL":
            n_models += 1
            continue
        if rec == "ENDMDL":
            if n_models >= 1:
                break                       # keep only the first model
            continue
        if rec in ("ATOM", "HETATM"):
            seen_coords = True
            if n_models > 1:
                continue
            try:
                atoms.append(_parse_pdb_atom_line(line))
            except Exception as exc:        # noqa: BLE001 — surface, don't drop
                raise ValueError(
                    f"malformed {rec} line in {path.name}: {line!r} ({exc})"
                ) from exc
        elif rec in _NON_HEADER_RECORDS:
            continue                        # ANISOU / TER / CONECT / ... dropped
        elif line.strip():
            if seen_coords:
                log.warning(f"dropping non-coordinate line after atoms: {line!r}")
            else:
                header_lines.append(line.rstrip("\n"))
    return atoms, header_lines


def read_cif(path: Path) -> tuple[list[Atom], list[str]]:
    """Parse a CIF/mmCIF file via biotite → (atoms, []). No REMARK headers in CIF.

    Altlocs are resolved to the highest-occupancy conformer by biotite.
    """
    try:
        from biotite.structure.io.pdbx import CIFFile, get_structure
    except ImportError as exc:
        raise ImportError(
            "reading CIF input requires the 'biotite' package — "
            "run inside the quantum_chem container or `pip install biotite`"
        ) from exc

    cif = CIFFile.read(str(path))
    arr = None
    last_exc: Exception | None = None
    for fields in (["b_factor", "occupancy"], []):
        try:
            arr = get_structure(cif, model=1, altloc="occupancy",
                                extra_fields=fields)
            break
        except Exception as exc:            # noqa: BLE001 — progressive fallback
            last_exc = exc
    if arr is None:
        raise ValueError(f"biotite could not parse CIF {path.name}: {last_exc}")

    cats = set(arr.get_annotation_categories())
    atoms: list[Atom] = []
    for i in range(arr.array_length()):
        name = str(arr.atom_name[i]).strip()
        record = "HETATM" if bool(arr.hetero[i]) else "ATOM"
        element = str(arr.element[i]).strip().upper()
        if not element:
            element = _infer_element(name, record)
        icode = str(arr.ins_code[i]).strip() if "ins_code" in cats else ""
        occ = float(arr.occupancy[i]) if "occupancy" in cats else 1.0
        bfac = float(arr.b_factor[i]) if "b_factor" in cats else 0.0
        if not np.isfinite(occ):            # biotite emits NaN for missing fields
            occ = 1.0
        if not np.isfinite(bfac):
            bfac = 0.0
        atoms.append(Atom(
            record=record, name=name,
            resname=str(arr.res_name[i]).strip(),
            chain=str(arr.chain_id[i]).strip() or " ",
            resid=int(arr.res_id[i]), icode=icode,
            xyz=np.array(arr.coord[i], float), element=element,
            occ=occ, bfactor=bfac,
        ))
    return atoms, []


def _filter_altlocs(st: ProtonationState) -> None:
    """Collapse alternate conformations: per atom keep highest occupancy; clear
    the altloc field on every surviving atom. Warns once per affected residue."""
    by_atom: dict = defaultdict(list)
    for a in st.atoms:
        by_atom[(a.reskey, a.name)].append(a)
    drop: set[int] = set()
    affected: set[tuple] = set()
    for (reskey, _name), grp in by_atom.items():
        if len(grp) <= 1:
            continue
        affected.add(reskey)
        best = sorted(grp, key=lambda a: (-a.occ, a.altloc or "Z"))[0]
        for a in grp:
            if a is not best:
                drop.add(id(a))
    if drop:
        st.atoms = [a for a in st.atoms if id(a) not in drop]
    for a in st.atoms:
        a.altloc = ""
    for reskey in sorted(affected):
        st.warn(f"residue {reskey_str(reskey)} had alternate conformations — "
                f"kept the highest-occupancy altloc")


def read_structure(path) -> ProtonationState:
    """Read a .pdb or .cif file into a fresh ProtonationState."""
    path = Path(path)
    if path.suffix.lower() in (".cif", ".mmcif", ".pdbx"):
        atoms, header = read_cif(path)
        fmt = "cif"
        log.info(f"read {len(atoms)} atoms from CIF {path.name} "
                 f"(REMARK headers are not available for CIF input)")
    else:
        atoms, header = read_pdb(path)
        fmt = "pdb"
        log.info(f"read {len(atoms)} atoms and {len(header)} header line(s) "
                 f"from PDB {path.name}")
    st = ProtonationState(atoms=atoms, header_lines=header,
                          source_path=str(path), source_format=fmt)
    _filter_altlocs(st)
    n_protein = sum(1 for a in st.atoms if a.record == "ATOM")
    if not st.atoms:
        st.warn("input structure contains 0 atoms")
    elif n_protein == 0:
        st.warn("input structure has no ATOM-record (protein) atoms — "
                "there is nothing to protonate")
    return st


# ============================================================================
# I/O — writing
# ============================================================================
def _fmt_atom_name(name: str, element: str) -> str:
    """Format an atom name into the 4-char PDB name field (cols 13-16)."""
    if len(name) >= 4:
        return name[:4]
    if len(element) == 2:            # two-letter element → start at col 13
        return f"{name:<4}"
    return f" {name:<3}"             # one-letter element → name in cols 14-16


def _atom_line(a: Atom, serial: int) -> str:
    namefield = _fmt_atom_name(a.name, a.element)
    x, y, z = (float(c) for c in a.xyz)
    resname = a.resname[:3]
    chain = (a.chain or " ")[:1]
    icode = (a.icode or " ")[:1] or " "
    altloc = (a.altloc or " ")[:1] or " "
    serial_str = f"{serial:>5}" if 0 <= serial <= 99999 else "*****"
    resid_str = (f"{a.resid:>4}" if -999 <= a.resid <= 9999
                 else str(a.resid)[-4:])
    occ = a.occ if np.isfinite(a.occ) else 1.0          # never let NaN corrupt
    bfac = a.bfactor if np.isfinite(a.bfactor) else 0.0  # the fixed-width columns
    return (
        f"{a.record:<6}{serial_str} {namefield}{altloc}"
        f"{resname:>3} {chain}{resid_str}{icode}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}"
        f"          {a.element:>2}{a.charge[:2]:<2}"
    )


def write_pdb(st: ProtonationState, path, remarks=None) -> None:
    """Write a PDB: preserved headers + QCB REMARK block + atoms. No CONECT."""
    lines: list[str] = list(st.header_lines)
    for r in (remarks or []):
        lines.append(f"REMARK 999 QCB-PROTONATOR {r}")

    n = len(st.atoms)
    if n > 99999:
        log.warning(f"{n} atoms exceeds 99999 — serials written as '*****'")
    prev: Atom | None = None
    for i, a in enumerate(st.atoms, start=1):
        a.serial = i
        if not (-999 <= a.resid <= 9999):
            log.warning(f"resid {a.resid} for {reskey_str(a.reskey)} is outside "
                        f"the PDB column range — output may misalign")
        if max(abs(float(c)) for c in a.xyz) >= 10000.0:
            log.warning(f"coordinate magnitude >=10000 for atom {a.name} "
                        f"{reskey_str(a.reskey)} — PDB columns may misalign")
        if prev is not None and prev.record == "ATOM" and (
            a.record != "ATOM" or a.chain != prev.chain
        ):
            lines.append("TER")
        lines.append(_atom_line(a, i))
        prev = a
    if prev is not None and prev.record == "ATOM":
        lines.append("TER")
    lines.append("END")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"wrote {n} atoms to {path}")


# ============================================================================
# Hydrogen stripping
# ============================================================================
def strip_protein_hydrogens(st: ProtonationState) -> ProtonationState:
    """Remove existing protein (ATOM-record) hydrogens; keep HETATM hydrogens."""
    before = len(st.atoms)
    kept = [a for a in st.atoms if not (a.record == "ATOM" and a.is_hydrogen)]
    n_stripped = before - len(kept)
    n_het_h = sum(1 for a in kept if a.is_hydrogen)
    st.atoms = kept
    st.note(f"stripped {n_stripped} existing protein hydrogen(s); "
            f"kept {n_het_h} HETATM hydrogen(s)")
    return st


# ============================================================================
# Hydrogen-placement engine — geometry primitives
# ============================================================================
# Idealised X-H bond lengths (Angstrom).
H_BOND_LENGTH = {"C": 1.09, "N": 1.01, "O": 0.96, "S": 1.34, "SE": 1.47}
TETRAHEDRAL_DEG = 109.4712
_TET_HALF_DEG = 54.7356           # half the tetrahedral angle


class PlacementError(ValueError):
    """Raised when hydrogens cannot be placed for an atom — insufficient heavy
    neighbours, or a topology/count mismatch. Callers are expected to catch
    this, log a warning, and skip the atom rather than abort the whole run
    (theozyme inputs are cropped, so missing heavy atoms are normal)."""


def h_bond_length(parent_element: str) -> float:
    return H_BOND_LENGTH.get(parent_element.upper(), 1.09)


def _normalize(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("cannot normalize a zero-length vector")
    return v / n


def _perp(v) -> np.ndarray:
    """Return an arbitrary unit vector perpendicular to v. The 0.9 cutoff keeps
    the cross product well-conditioned when v lies near a basis axis."""
    v = _normalize(v)
    ref = (np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9
           else np.array([0.0, 1.0, 0.0]))
    return _normalize(np.cross(v, ref))


def _place_lone_pair(parent, neighbors, L: float) -> np.ndarray:
    """H opposite the mean bond direction. 1 nbr → 180°; 2 nbr → in-plane
    bisector pointing outward; 3 nbr → the sp3 4th tetrahedral vertex.

    If the neighbours are (near-)symmetric about the parent — common in a
    strained transition-state geometry — the outward direction is undefined;
    fall back to a perpendicular of the first bond so the H is still usable
    (the geometry-autocorrection stage / MLFF relax tidies it up)."""
    direction = np.zeros(3)
    for nb in neighbors:
        direction += _normalize(np.asarray(nb, float) - parent)
    if np.linalg.norm(direction) < 1e-3:
        direction = _perp(np.asarray(neighbors[0], float) - parent)
    return parent - L * _normalize(direction)


def _place_sp3_two(parent, n1, n2, L: float) -> list:
    """Two H completing an sp3 atom that already has 2 heavy neighbours."""
    u1 = _normalize(np.asarray(n1, float) - parent)
    u2 = _normalize(np.asarray(n2, float) - parent)
    # Generous 1e-3 thresholds: a strained TS frame can push the two heavy
    # neighbours near-collinear, where a tight 1e-6 guard lets noise through.
    bis = u1 + u2
    bis = _perp(u1) if np.linalg.norm(bis) < 1e-3 else _normalize(bis)
    perp = np.cross(u1, u2)
    perp = _perp(bis) if np.linalg.norm(perp) < 1e-3 else _normalize(perp)
    c = np.cos(np.radians(_TET_HALF_DEG))
    s = np.sin(np.radians(_TET_HALF_DEG))
    return [parent + L * _normalize(-bis * c + perp * s),
            parent + L * _normalize(-bis * c - perp * s)]


def _place_around_axis(parent, axis_nbr, n_h: int, L: float,
                       slots: int = 3) -> list:
    """Place n_h H around the parent–axis_nbr bond at the tetrahedral angle,
    evenly spread over `slots` positions. Methyl: n_h=3,slots=3. Amine: n_h=2,
    slots=3 (the 3rd slot is the lone pair). The azimuth about the bond axis is
    arbitrary — correct for a freely-rotating methyl / primary amine."""
    u = _normalize(np.asarray(axis_nbr, float) - parent)   # toward neighbour
    r = _perp(u)
    s = _normalize(np.cross(u, r))
    axial = -u / 3.0                       # cos(109.47°) component
    radial = np.sqrt(8.0) / 3.0            # sin(109.47°) component
    out = []
    for k in range(n_h):
        ang = 2.0 * np.pi * k / slots
        d = axial + radial * (np.cos(ang) * r + np.sin(ang) * s)
        out.append(parent + L * _normalize(d))
    return out


def _sp2_substituent_dirs(parent, anchor, plane_atom):
    """Unit vectors for the two non-anchor substituent directions of an sp2
    atom: in the parent–anchor–plane_atom plane, 120° either side of -anchor."""
    parent = np.asarray(parent, float)
    pa = np.asarray(anchor, float) - parent
    u = _normalize(pa)
    normal = np.cross(pa, np.asarray(plane_atom, float) - parent)
    normal = _perp(u) if np.linalg.norm(normal) < 1e-3 else _normalize(normal)
    ipp = _normalize(np.cross(normal, u))
    c, s = np.cos(np.radians(60.0)), np.sin(np.radians(60.0))
    return _normalize(-u * c + ipp * s), _normalize(-u * c - ipp * s)


def _place_sp2_planar(parent, anchor, plane_atom, n_h: int, L: float) -> list:
    """Place n_h H (1 or 2) on an sp2 parent that has ONE heavy neighbour
    (`anchor`), in the plane parent–anchor–plane_atom, at 120°."""
    parent = np.asarray(parent, float)
    d1, d2 = _sp2_substituent_dirs(parent, anchor, plane_atom)
    return [parent + L * d for d in (d1, d2)][:n_h]


def _place_hydroxyl(parent, anchor, L: float, plane_atom=None) -> np.ndarray:
    """O-H / S-H: one H at ~108° from the anchor bond (free rotation)."""
    u = _normalize(np.asarray(anchor, float) - parent)
    if plane_atom is not None:
        normal = np.cross(np.asarray(anchor, float) - parent,
                          np.asarray(plane_atom, float) - parent)
        ipp = (_normalize(np.cross(_normalize(normal), u))
               if np.linalg.norm(normal) > 1e-6 else _perp(u))
    else:
        ipp = _perp(u)
    ang = np.radians(108.0)
    return parent + L * _normalize(u * np.cos(ang) + ipp * np.sin(ang))


def _place_carboxyl_h(o2, cx, o1, L: float, angle_deg: float = 109.5) -> np.ndarray:
    """Carboxylic-acid H on o2, in the carboxylate plane, ~109.5° C-O-H, on the
    side away from the C=O partner o1. (Ported from ops/protonation.py.)"""
    o2 = np.asarray(o2, float)
    oc = _normalize(np.asarray(cx, float) - o2)
    normal = np.cross(np.asarray(cx, float) - o2, np.asarray(o1, float) - o2)
    if np.linalg.norm(normal) < 1e-6:
        # C/O1/O2 collinear — plane undefined; keep the ~109.5° C-O-H angle
        # with an arbitrary perpendicular instead of collapsing to a linear H.
        side = _perp(oc)
        a = np.radians(angle_deg)
        return o2 + L * _normalize(np.cos(a) * oc + np.sin(a) * side)
    normal = _normalize(normal)
    side = _normalize(np.cross(normal, oc))
    if np.dot(side, np.asarray(o1, float) - o2) > 0.0:
        side = -side
    a = np.radians(angle_deg)
    return o2 + L * _normalize(np.cos(a) * oc + np.sin(a) * side)


# Placement methods accepted by ``place_hydrogens``.
#   sp3          generic sp3, dispatched by (#heavy neighbours, #H)
#   sp2_ring_h   1 H, 2 heavy neighbours, in-plane (ring C-H / N-H, Arg NE)
#   sp2_term_h2  2 H, 1 heavy neighbour, in-plane (Asn/Gln NH2, Arg NH1/NH2)
#   hydroxyl     O-H ;  thiol  S-H ;  carboxyl  carboxylic-acid O-H
def place_hydrogens(parent, neighbors, n_h: int, method: str,
                    parent_element: str, plane_ref=None) -> list:
    """Return a list of exactly `n_h` hydrogen xyz positions for one heavy atom.

    parent      : np.ndarray xyz of the H-bearing heavy atom
    neighbors   : list of np.ndarray xyz of its bonded heavy neighbours
    plane_ref   : np.ndarray xyz of a 3rd atom defining the conjugated plane —
                  REQUIRED by sp2_term_h2 (NH2 orientation is otherwise wrong).

    Raises PlacementError on insufficient heavy neighbours, a missing plane_ref,
    or an H-count mismatch with the topology. The caller is expected to catch
    PlacementError, warn, and skip the atom (theozyme inputs are cropped, so
    missing heavy neighbours are normal).
    """
    parent = np.asarray(parent, float)
    neighbors = [np.asarray(n, float) for n in neighbors]
    L = h_bond_length(parent_element)
    nn = len(neighbors)

    if method == "sp3":
        if nn >= 3:
            result = [_place_lone_pair(parent, neighbors[:3], L)]
        elif nn == 2:
            result = _place_sp3_two(parent, neighbors[0], neighbors[1], L)[:n_h]
        elif nn == 1:
            result = _place_around_axis(parent, neighbors[0], n_h, L, slots=3)
        else:
            raise PlacementError("sp3 placement needs >=1 heavy neighbour")
    elif method == "sp2_ring_h":
        if nn < 2:
            raise PlacementError("sp2_ring_h needs 2 heavy neighbours")
        result = [_place_lone_pair(parent, neighbors[:2], L)]
    elif method == "sp2_term_h2":
        if nn != 1:
            raise PlacementError("sp2_term_h2 needs exactly 1 heavy neighbour")
        if plane_ref is None:
            raise PlacementError("sp2_term_h2 requires plane_ref — without the "
                                 "conjugated-plane atom the NH2 orientation is "
                                 "chemically wrong")
        result = _place_sp2_planar(parent, neighbors[0],
                                   np.asarray(plane_ref, float), n_h, L)
    elif method in ("hydroxyl", "thiol"):
        if nn < 1:
            raise PlacementError(f"{method} needs 1 heavy neighbour")
        ref = np.asarray(plane_ref, float) if plane_ref is not None else None
        result = [_place_hydroxyl(parent, neighbors[0], L, plane_atom=ref)]
    elif method == "carboxyl":
        if nn < 2:
            raise PlacementError("carboxyl needs the C and the partner O")
        result = [_place_carboxyl_h(parent, neighbors[0], neighbors[1], L)]
    else:
        raise PlacementError(f"unknown hydrogen-placement method: {method!r}")

    if len(result) != n_h:
        raise PlacementError(
            f"method {method!r} produced {len(result)} H but {n_h} were "
            f"expected — check the TOPOLOGY entry")
    return result


# ============================================================================
# Hydrogen-placement engine — per-residue topology table
# ============================================================================
# For each standard residue: sidechain heavy-atom bonds + the H-bearing atoms.
# Universal backbone bonds (N-CA, CA-C, C-O) are added by residue_heavy_graph().
# An h-entry is (parent_atom, method, [H names]); n_h = len(H names).
# This table places the ALWAYS-PRESENT hydrogens (side-chain C-H, amide NH2,
# and the neutral O-H / S-H of Ser/Thr/Tyr/Cys). CONDITIONAL hydrogens are
# placed by later stages: the backbone amide N-H (Stage 3 — cross-residue, and
# skipped for proline whose backbone N is tertiary), the His ND1/NE2 tautomer
# H (Stage 4), and the pH-dependent Asp/Glu carboxyl H (Stage 3/5) — which is
# why ASP/GLU list no carboxyl-O h-entry while ASN/GLN/ARG amide NH2 (always
# present) are placed here.
_M3 = "sp3"
TOPOLOGY: dict = {
    "ALA": dict(bonds=[("CA", "CB")],
                h=[("CA", _M3, ["HA"]),
                   ("CB", _M3, ["HB1", "HB2", "HB3"])]),
    "GLY": dict(bonds=[],
                h=[("CA", _M3, ["HA2", "HA3"])]),
    "VAL": dict(bonds=[("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")],
                h=[("CA", _M3, ["HA"]), ("CB", _M3, ["HB"]),
                   ("CG1", _M3, ["HG11", "HG12", "HG13"]),
                   ("CG2", _M3, ["HG21", "HG22", "HG23"])]),
    "LEU": dict(bonds=[("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")],
                h=[("CA", _M3, ["HA"]), ("CB", _M3, ["HB2", "HB3"]),
                   ("CG", _M3, ["HG"]),
                   ("CD1", _M3, ["HD11", "HD12", "HD13"]),
                   ("CD2", _M3, ["HD21", "HD22", "HD23"])]),
    "ILE": dict(bonds=[("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"),
                       ("CG1", "CD1")],
                h=[("CA", _M3, ["HA"]), ("CB", _M3, ["HB"]),
                   ("CG1", _M3, ["HG12", "HG13"]),
                   ("CG2", _M3, ["HG21", "HG22", "HG23"]),
                   ("CD1", _M3, ["HD11", "HD12", "HD13"])]),
    "PRO": dict(bonds=[("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "N")],
                h=[("CA", _M3, ["HA"]), ("CB", _M3, ["HB2", "HB3"]),
                   ("CG", _M3, ["HG2", "HG3"]),
                   ("CD", _M3, ["HD2", "HD3"])]),
    "MET": dict(bonds=[("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE")],
                h=[("CA", _M3, ["HA"]), ("CB", _M3, ["HB2", "HB3"]),
                   ("CG", _M3, ["HG2", "HG3"]),
                   ("CE", _M3, ["HE1", "HE2", "HE3"])]),
    "PHE": dict(bonds=[("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
                       ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"),
                       ("CE2", "CZ")],
                h=[("CA", _M3, ["HA"]), ("CB", _M3, ["HB2", "HB3"]),
                   ("CD1", "sp2_ring_h", ["HD1"]),
                   ("CD2", "sp2_ring_h", ["HD2"]),
                   ("CE1", "sp2_ring_h", ["HE1"]),
                   ("CE2", "sp2_ring_h", ["HE2"]),
                   ("CZ", "sp2_ring_h", ["HZ"])]),
    "TRP": dict(bonds=[("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
                       ("CD1", "NE1"), ("NE1", "CE2"), ("CD2", "CE2"),
                       ("CD2", "CE3"), ("CE3", "CZ3"), ("CZ3", "CH2"),
                       ("CH2", "CZ2"), ("CZ2", "CE2")],
                h=[("CA", _M3, ["HA"]), ("CB", _M3, ["HB2", "HB3"]),
                   ("CD1", "sp2_ring_h", ["HD1"]),
                   ("NE1", "sp2_ring_h", ["HE1"]),
                   ("CE3", "sp2_ring_h", ["HE3"]),
                   ("CZ3", "sp2_ring_h", ["HZ3"]),
                   ("CZ2", "sp2_ring_h", ["HZ2"]),
                   ("CH2", "sp2_ring_h", ["HH2"])]),
    "SER": dict(bonds=[("CA", "CB"), ("CB", "OG")],
                h=[("CA", _M3, ["HA"]), ("CB", _M3, ["HB2", "HB3"]),
                   ("OG", "hydroxyl", ["HG"])]),
    "THR": dict(bonds=[("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")],
                h=[("CA", _M3, ["HA"]), ("CB", _M3, ["HB"]),
                   ("OG1", "hydroxyl", ["HG1"]),
                   ("CG2", _M3, ["HG21", "HG22", "HG23"])]),
    "CYS": dict(bonds=[("CA", "CB"), ("CB", "SG")],
                h=[("CA", _M3, ["HA"]), ("CB", _M3, ["HB2", "HB3"]),
                   ("SG", "thiol", ["HG"])]),
    "TYR": dict(bonds=[("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
                       ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"),
                       ("CE2", "CZ"), ("CZ", "OH")],
                h=[("CA", _M3, ["HA"]), ("CB", _M3, ["HB2", "HB3"]),
                   ("CD1", "sp2_ring_h", ["HD1"]),
                   ("CD2", "sp2_ring_h", ["HD2"]),
                   ("CE1", "sp2_ring_h", ["HE1"]),
                   ("CE2", "sp2_ring_h", ["HE2"]),
                   ("OH", "hydroxyl", ["HH"])]),
    "ASN": dict(bonds=[("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")],
                h=[("CA", _M3, ["HA"]), ("CB", _M3, ["HB2", "HB3"]),
                   ("ND2", "sp2_term_h2", ["HD21", "HD22"])]),
    "GLN": dict(bonds=[("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"),
                       ("CD", "NE2")],
                h=[("CA", _M3, ["HA"]), ("CB", _M3, ["HB2", "HB3"]),
                   ("CG", _M3, ["HG2", "HG3"]),
                   ("NE2", "sp2_term_h2", ["HE21", "HE22"])]),
    "ASP": dict(bonds=[("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")],
                h=[("CA", _M3, ["HA"]), ("CB", _M3, ["HB2", "HB3"])]),
    "GLU": dict(bonds=[("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"),
                       ("CD", "OE2")],
                h=[("CA", _M3, ["HA"]), ("CB", _M3, ["HB2", "HB3"]),
                   ("CG", _M3, ["HG2", "HG3"])]),
    "LYS": dict(bonds=[("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"),
                       ("CE", "NZ")],
                h=[("CA", _M3, ["HA"]), ("CB", _M3, ["HB2", "HB3"]),
                   ("CG", _M3, ["HG2", "HG3"]), ("CD", _M3, ["HD2", "HD3"]),
                   ("CE", _M3, ["HE2", "HE3"])]),
    "ARG": dict(bonds=[("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE"),
                       ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2")],
                h=[("CA", _M3, ["HA"]), ("CB", _M3, ["HB2", "HB3"]),
                   ("CG", _M3, ["HG2", "HG3"]), ("CD", _M3, ["HD2", "HD3"]),
                   ("NE", "sp2_ring_h", ["HE"]),
                   ("NH1", "sp2_term_h2", ["HH11", "HH12"]),
                   ("NH2", "sp2_term_h2", ["HH21", "HH22"])]),
    "HIS": dict(bonds=[("CA", "CB"), ("CB", "CG"), ("CG", "ND1"), ("ND1", "CE1"),
                       ("CE1", "NE2"), ("NE2", "CD2"), ("CD2", "CG")],
                h=[("CA", _M3, ["HA"]), ("CB", _M3, ["HB2", "HB3"]),
                   ("CE1", "sp2_ring_h", ["HE1"]),
                   ("CD2", "sp2_ring_h", ["HD2"])]),
}

BACKBONE_BONDS = [("N", "CA"), ("CA", "C"), ("C", "O")]


def residue_heavy_graph(resname: str, present_names) -> dict:
    """Return atom_name -> set(bonded heavy-atom names) for one residue,
    from the universal backbone bonds + the per-residue topology table.
    Restricted to atoms actually present in the residue."""
    rn = canonical_resname(resname)
    bonds = list(BACKBONE_BONDS)
    spec = TOPOLOGY.get(rn)
    if spec:
        bonds += spec["bonds"]
    present = set(present_names)
    if "OXT" in present:                       # C-terminal carboxylate oxygen
        bonds.append(("C", "OXT"))
    graph: dict = defaultdict(set)
    for a, b in bonds:
        if a in present and b in present:
            graph[a].add(b)
            graph[b].add(a)
    # Return sorted neighbour lists — deterministic order (set iteration over
    # atom-name strings is hash-seed dependent and would make placement and
    # plane-reference selection non-reproducible run to run).
    return {k: sorted(v) for k, v in graph.items()}


# ============================================================================
# Stage 1 — terminus detection & capping
# ============================================================================
# Cap formal-charge contributions.
N_CAP_CHARGE = {"nh2": 0, "nh3+": 1, "nme": 0, "nfo": 0, "none": 0}
C_CAP_CHARGE = {"cho": 0, "coo-": -1, "cooh": 0, "conh2": 0, "conhme": 0,
                "none": 0}
N_CAP_TYPES = set(N_CAP_CHARGE)
C_CAP_TYPES = set(C_CAP_CHARGE)

# Cap bond lengths (Angstrom).
_L_NC = 1.47        # N-C single
_L_CN_AMIDE = 1.33  # C-N amide
_L_CO_CARBONYL = 1.23
_L_CO_HYDROXYL = 1.34
_L_CO_CARBOXYLATE = 1.25
_L_CC = 1.52


def _cap_atom(name: str, element: str, xyz, template: Atom) -> Atom:
    """Build a script-added cap atom, inheriting residue identity from a
    template atom (the residue's N or C)."""
    return Atom(record="ATOM", name=name, resname=template.resname,
                chain=template.chain, resid=template.resid,
                icode=template.icode, xyz=np.asarray(xyz, float),
                element=element, tag="cap")


def _place_one_at_angle(parent, anchor, angle_deg: float, L: float) -> np.ndarray:
    """Place one substituent at `angle_deg` from the parent→anchor bond, in an
    arbitrary plane — used to build an sp2 substituent when only one heavy
    neighbour is fixed (e.g. the formyl C of an N-formyl cap)."""
    parent = np.asarray(parent, float)
    u = _normalize(np.asarray(anchor, float) - parent)
    a = np.radians(angle_deg)
    return parent + L * _normalize(u * np.cos(a) + _perp(u) * np.sin(a))


def _c_cap_direction(C: Atom, CA: Atom, O: Atom) -> np.ndarray:
    """Unit direction for the 3rd (cap) substituent of a free carbonyl C — an
    idealised sp2 position 120° from C–CA in the CA-C-O plane, on the side away
    from the carbonyl O. Robust to a distorted input CA-C-O angle (TS frames)."""
    d1, d2 = _sp2_substituent_dirs(C.xyz, CA.xyz, O.xyz)
    ov = O.xyz - C.xyz
    return d1 if float(np.dot(d1, ov)) < float(np.dot(d2, ov)) else d2


# ---- N-terminus cap recipes (free backbone N, anchored on N + CA) ----------
def _cap_n_nh2(N: Atom, CA: Atom) -> list:
    hs = place_hydrogens(N.xyz, [CA.xyz], 2, "sp3", "N")
    return [_cap_atom(nm, "H", h, N) for nm, h in zip(["H1", "H2"], hs)]


def _cap_n_nh3(N: Atom, CA: Atom) -> list:
    hs = place_hydrogens(N.xyz, [CA.xyz], 3, "sp3", "N")
    return [_cap_atom(nm, "H", h, N) for nm, h in zip(["H1", "H2", "H3"], hs)]


def _cap_n_nme(N: Atom, CA: Atom) -> list:
    """N-methyl: secondary amine — N bonded to CA, methyl C (CM), one H."""
    cm = _place_around_axis(N.xyz, CA.xyz, 1, _L_NC, slots=3)[0]
    hN = place_hydrogens(N.xyz, [CA.xyz, cm], 1, "sp3", "N")[0]
    hms = place_hydrogens(cm, [N.xyz], 3, "sp3", "C")
    out = [_cap_atom("CM", "C", cm, N), _cap_atom("H1", "H", hN, N)]
    out += [_cap_atom(nm, "H", h, N)
            for nm, h in zip(["HM1", "HM2", "HM3"], hms)]
    return out


def _cap_n_nfo(N: Atom, CA: Atom) -> list:
    """N-formyl: amide — N bonded to CA, formyl C (CF), one H; CF=O with one H.
    CF is placed at an sp2 ~122° from N-CA (the amide N is planar, not sp3)."""
    cf = _place_one_at_angle(N.xyz, CA.xyz, 122.0, _L_CN_AMIDE)
    hN = place_hydrogens(N.xyz, [CA.xyz, cf], 1, "sp2_ring_h", "N")[0]
    d_o, d_h = _sp2_substituent_dirs(cf, N.xyz, CA.xyz)
    of = cf + _L_CO_CARBONYL * d_o
    hf = cf + h_bond_length("C") * d_h
    return [_cap_atom("CF", "C", cf, N), _cap_atom("H1", "H", hN, N),
            _cap_atom("OF", "O", of, N), _cap_atom("HF", "H", hf, N)]


# ---- C-terminus cap recipes (free carbonyl C, anchored on C + CA + O) ------
# All use _c_cap_direction — an idealised 120° sp2 position robust to a
# distorted input CA-C-O angle (the placed O/N heavy atoms are NOT relaxed).
def _cap_c_cho(C: Atom, CA: Atom, O: Atom) -> list:
    h = C.xyz + h_bond_length("C") * _c_cap_direction(C, CA, O)
    return [_cap_atom("HC", "H", h, C)]


def _cap_c_coo(C: Atom, CA: Atom, O: Atom) -> list:
    oxt = C.xyz + _L_CO_CARBOXYLATE * _c_cap_direction(C, CA, O)
    return [_cap_atom("OXT", "O", oxt, C)]


def _cap_c_cooh(C: Atom, CA: Atom, O: Atom) -> list:
    oxt = C.xyz + _L_CO_HYDROXYL * _c_cap_direction(C, CA, O)
    hxt = _place_carboxyl_h(oxt, C.xyz, O.xyz, h_bond_length("O"))
    return [_cap_atom("OXT", "O", oxt, C), _cap_atom("HXT", "H", hxt, C)]


def _cap_c_conh2(C: Atom, CA: Atom, O: Atom) -> list:
    nt = C.xyz + _L_CN_AMIDE * _c_cap_direction(C, CA, O)
    hs = place_hydrogens(nt, [C.xyz], 2, "sp2_term_h2", "N", plane_ref=O.xyz)
    return [_cap_atom("NT", "N", nt, C)] + [
        _cap_atom(nm, "H", h, C) for nm, h in zip(["HT1", "HT2"], hs)]


def _cap_c_conhme(C: Atom, CA: Atom, O: Atom) -> list:
    nt = C.xyz + _L_CN_AMIDE * _c_cap_direction(C, CA, O)
    d1, d2 = _sp2_substituent_dirs(nt, C.xyz, O.xyz)
    ov = O.xyz - nt                              # put the methyl trans to C=O
    d_c, d_h = ((d1, d2) if float(np.dot(d1, ov)) < float(np.dot(d2, ov))
                else (d2, d1))
    cm = nt + _L_NC * d_c
    ht1 = nt + h_bond_length("N") * d_h
    hms = place_hydrogens(cm, [nt], 3, "sp3", "C")
    out = [_cap_atom("NT", "N", nt, C), _cap_atom("CMT", "C", cm, C),
           _cap_atom("HT1", "H", ht1, C)]
    out += [_cap_atom(nm, "H", h, C)
            for nm, h in zip(["HMT1", "HMT2", "HMT3"], hms)]
    return out


_N_CAP_FN = {"nh2": _cap_n_nh2, "nh3+": _cap_n_nh3,
             "nme": _cap_n_nme, "nfo": _cap_n_nfo}
_C_CAP_FN = {"cho": _cap_c_cho, "coo-": _cap_c_coo, "cooh": _cap_c_cooh,
             "conh2": _cap_c_conh2, "conhme": _cap_c_conhme}


def _near_any(atom: Atom, others: list, cutoff: float) -> bool:
    return any(np.linalg.norm(atom.xyz - o.xyz) <= cutoff for o in others)


def stage1_caps(st: ProtonationState, *, n_cap: str = "nh2", c_cap: str = "cho",
                n_cap_overrides=None, c_cap_overrides=None,
                bond_cutoff: float = 1.8) -> ProtonationState:
    """Detect open backbone N/C termini by geometry and cap them.

    Records per-residue cap decisions in st.residue_records[rk]['n_cap'/'c_cap'].
    """
    banner("Stage 1 — terminus detection & capping")
    n_cap_overrides = {k: v for k, v in (n_cap_overrides or {}).items()}
    c_cap_overrides = {k: v for k, v in (c_cap_overrides or {}).items()}

    protein = [a for a in st.atoms if a.record == "ATOM"]
    hetero_heavy = [a for a in st.atoms
                    if a.record == "HETATM" and not a.is_hydrogen
                    and a.element.upper() not in METALS
                    and a.resname.upper() not in WATER_RESNAMES]
    residues = group_residues(protein)
    bb = {rk: {a.name: a for a in ats if a.name in ("N", "CA", "C", "O", "OXT")}
          for rk, ats in residues.items()}

    # --- peptide-bond detection by geometry --------------------------------
    n_bonded: set = set()
    c_bonded: set = set()
    bonds_found: list = []
    keys = list(bb)
    for rk_i in keys:
        Ci = bb[rk_i].get("C")
        if Ci is None:
            continue
        for rk_j in keys:
            if rk_j == rk_i:
                continue
            Nj = bb[rk_j].get("N")
            if Nj is None:
                continue
            d = float(np.linalg.norm(Ci.xyz - Nj.xyz))
            if d <= bond_cutoff:
                c_bonded.add(rk_i)
                n_bonded.add(rk_j)
                bonds_found.append(f"{reskey_str(rk_i)}-{reskey_str(rk_j)}")
                msg = (f"  peptide bond  {reskey_str(rk_i)}.C – "
                       f"{reskey_str(rk_j)}.N  ({d:.2f} A)")
                if rk_i[0] == rk_j[0] and rk_j[1] - rk_i[1] == 1:
                    st.note(msg)
                else:                       # geometry vs sequence disagree
                    st.warn(msg + "  [non-consecutive residues — verify]")
    if bonds_found:
        st.note(f"  peptide bonds found ({len(bonds_found)}): "
                + ", ".join(bonds_found))
    else:
        st.note("  no peptide bonds found — every residue is isolated")
    st.note(f"  {len(residues)} protein residue(s); "
            f"{len(n_bonded)} interior N, {len(c_bonded)} interior C")

    # --- apply caps --------------------------------------------------------
    caps_by_res: dict = defaultdict(list)
    n_done = c_done = 0
    for rk, ats in residues.items():
        d = bb[rk]
        N, CA, C, O, OXT = (d.get("N"), d.get("CA"), d.get("C"),
                            d.get("O"), d.get("OXT"))
        rec = st.residue_records.setdefault(rk, {})
        rec["resname"] = ats[0].resname
        rks = reskey_str(rk)

        # ---- N terminus ----
        if N is None:
            rec["n_cap"] = "none"
        elif rk in n_bonded:
            rec["n_cap"] = "peptide"
        else:
            cap = n_cap_overrides.get(rks, n_cap)
            if cap not in N_CAP_TYPES:
                st.warn(f"  unknown N-cap {cap!r} for {rks} — using nh2")
                cap = "nh2"
            if _near_any(N, hetero_heavy, bond_cutoff):
                st.warn(f"  {rks}.N is within bonding distance of a HETATM — "
                        f"possible covalent link; leaving uncapped")
                rec["n_cap"] = "covalent?"
            elif cap == "none":
                rec["n_cap"] = "none"
                st.note(f"  {rks}.N left uncapped (none)")
            elif CA is None:
                st.warn(f"  {rks}.N has no CA anchor — cannot cap")
                rec["n_cap"] = "none"
            else:
                try:
                    new = _N_CAP_FN[cap](N, CA)
                except PlacementError as exc:
                    st.warn(f"  {rks}.N cap {cap} failed ({exc}) — uncapped")
                    rec["n_cap"] = "none"
                else:
                    caps_by_res[rk] += new
                    rec["n_cap"] = cap
                    n_done += 1
                    st.note(f"  {rks}.N capped {cap}  (+{len(new)} atoms)")

        # ---- C terminus ----
        if C is None:
            rec["c_cap"] = "none"
        elif rk in c_bonded:
            rec["c_cap"] = "peptide"
        else:
            cap = c_cap_overrides.get(rks, c_cap)
            overridden = rks in c_cap_overrides
            if cap not in C_CAP_TYPES:
                st.warn(f"  unknown C-cap {cap!r} for {rks} — using cho")
                cap = "cho"
            oxt_present = (OXT is not None and float(
                np.linalg.norm(OXT.xyz - C.xyz)) <= bond_cutoff)
            if oxt_present:
                # the input already has a carboxylate oxygen on this C
                if cap == "cooh":
                    ref_o = O if O is not None else CA
                    hxt = _place_carboxyl_h(OXT.xyz, C.xyz, ref_o.xyz,
                                            h_bond_length("O"))
                    caps_by_res[rk].append(_cap_atom("HXT", "H", hxt, C))
                    rec["c_cap"] = "cooh"
                    c_done += 1
                    st.note(f"  {rks}.C already has OXT — added HXT (cooh)")
                else:
                    if overridden and cap != "coo-":
                        st.warn(f"  {rks}.C already has OXT — requested cap "
                                f"{cap!r} is incompatible; treating as coo-")
                    else:
                        st.note(f"  {rks}.C already has OXT — treated as coo-")
                    rec["c_cap"] = "coo-"
            elif _near_any(C, hetero_heavy, bond_cutoff):
                st.warn(f"  {rks}.C is within bonding distance of a HETATM — "
                        f"possible covalent link; leaving uncapped")
                rec["c_cap"] = "covalent?"
            elif cap == "none":
                rec["c_cap"] = "none"
                st.note(f"  {rks}.C left uncapped (none)")
            elif CA is None or O is None:
                st.warn(f"  {rks}.C missing CA/O anchor — cannot cap")
                rec["c_cap"] = "none"
            else:
                try:
                    new = _C_CAP_FN[cap](C, CA, O)
                except PlacementError as exc:
                    st.warn(f"  {rks}.C cap {cap} failed ({exc}) — uncapped")
                    rec["c_cap"] = "none"
                else:
                    caps_by_res[rk] += new
                    rec["c_cap"] = cap
                    c_done += 1
                    st.note(f"  {rks}.C capped {cap}  (+{len(new)} atoms)")

    # --- warn about override keys that matched no residue ------------------
    seen_keys = {reskey_str(rk) for rk in residues}
    for k in n_cap_overrides:
        if k not in seen_keys:
            st.warn(f"  --n-cap-override key {k!r} matched no residue")
    for k in c_cap_overrides:
        if k not in seen_keys:
            st.warn(f"  --c-cap-override key {k!r} matched no residue")

    # --- splice cap atoms in after each residue's last atom ----------------
    if caps_by_res:
        out: list = []
        for i, a in enumerate(st.atoms):
            out.append(a)
            nxt = st.atoms[i + 1] if i + 1 < len(st.atoms) else None
            if a.reskey in caps_by_res and (nxt is None
                                            or nxt.reskey != a.reskey):
                out.extend(caps_by_res.pop(a.reskey))
        for leftover in caps_by_res.values():     # non-contiguous residues
            out.extend(leftover)
        st.atoms = out

    st.note(f"Stage 1 done — capped {n_done} N-terminus(es), "
            f"{c_done} C-terminus(es)")
    return st


# ============================================================================
# Stage 2 — PTM / covalent safety checks (warnings only)
# ============================================================================
# Per-residue formal charge for declared post-translational modifications.
PTM_CHARGE = {
    "KCX": -1,                          # carboxylated (carbamylated) lysine
    "SEP": -2, "TPO": -2, "PTR": -2,    # phospho-Ser / Thr / Tyr (dianionic)
    "CSO": 0, "CSD": -1, "CSX": 0,      # oxidised cysteines
    "SEC": -1,                          # selenocysteine (selenolate at pH 7)
    "MLY": 1, "M3L": 1, "MLZ": 1,       # methyl-lysines
    "ALY": 0, "HYP": 0, "CGU": -2,      # acetyl-Lys, hydroxy-Pro, Gla
}


def stage2_checks(st: ProtonationState, *, ptm=None, ptm_charge=None,
                  bond_cutoff: float = 1.8,
                  disulfide_cutoff: float = 2.5) -> ProtonationState:
    """Diagnostic-only: warn about suspected disulfides, lysine carbamylation
    and protein-ligand covalent bonds. Register user-declared PTMs — a declared
    PTM freezes that residue's side-chain protonation for the later stages."""
    banner("Stage 2 — PTM / covalent safety checks (warnings only)")
    ptm = ptm or {}
    ptm_charge = ptm_charge or {}

    # Detection ignores Stage-1 cap atoms (script artefacts). The O(N^2) scans
    # are fine for theozyme-sized inputs (hundreds of atoms); a cKD-tree would
    # be needed only for full-protein-scale structures.
    protein_heavy = [a for a in st.atoms if a.record == "ATOM"
                     and not a.is_hydrogen and a.tag != "cap"]
    hetero_heavy = [a for a in st.atoms if a.record == "HETATM"
                    and not a.is_hydrogen]
    heavy_nocap = [a for a in st.atoms if not a.is_hydrogen and a.tag != "cap"]
    lig_heavy = [a for a in hetero_heavy
                 if a.element.upper() not in METALS
                 and a.resname.upper() not in WATER_RESNAMES]
    n_warn0 = len(st.warnings)

    # --- disulfides (S-S ~2.05 A; the 2.5 A detection cutoff tolerates a
    #     distorted / low-resolution geometry without false positives) -------
    sg = [a for a in protein_heavy if a.name == "SG"]
    for i in range(len(sg)):
        for j in range(i + 1, len(sg)):
            d = float(np.linalg.norm(sg[i].xyz - sg[j].xyz))
            if d <= disulfide_cutoff:
                st.warn(f"  possible disulfide  {reskey_str(sg[i].reskey)}.SG – "
                        f"{reskey_str(sg[j].reskey)}.SG  ({d:.2f} A) — Stage 3 "
                        f"would give each Cys an HG; treat both as CYX if this "
                        f"is a real disulfide")

    # --- lysine carbamylation: NZ bonded to a genuine carbamate carbon —
    #     a carbon coordinated to {NZ, O, O} and to NO other carbon. This
    #     {N,O,O,no-C} signature cleanly excludes carboxylate/carbonyl carbons
    #     (which always carry a C neighbour) and the Lys's own CE. ------------
    carbons = [a for a in heavy_nocap if a.element == "C"]
    oxygens = [a for a in heavy_nocap if a.element == "O"]
    for nz in [a for a in protein_heavy if a.name == "NZ"]:
        for c in carbons:
            if float(np.linalg.norm(nz.xyz - c.xyz)) > 1.6:   # true N-C bond
                continue
            n_o = sum(1 for o in oxygens
                      if float(np.linalg.norm(o.xyz - c.xyz)) <= 1.6)
            n_c = sum(1 for x in carbons if x is not c
                      and float(np.linalg.norm(x.xyz - c.xyz)) <= 1.7)
            if n_o >= 2 and n_c == 0:
                st.warn(f"  likely carbamylated lysine at "
                        f"{reskey_str(nz.reskey)} — NZ bonded to a carbamate "
                        f"carbon; declare --ptm CHAIN:RESID=KCX if real")

    # --- protein-ligand covalent bonds ------------------------------------
    for p in protein_heavy:
        for lg in lig_heavy:
            d = float(np.linalg.norm(p.xyz - lg.xyz))
            if d <= bond_cutoff:
                st.warn(f"  possible protein-ligand covalent bond  "
                        f"{reskey_str(p.reskey)}.{p.name} – "
                        f"{lg.resname}.{lg.name}  ({d:.2f} A) — declare --ptm "
                        f"if this residue is covalently modified")

    if len(st.warnings) == n_warn0:
        st.note("  no suspected PTMs or covalent anomalies")

    # --- register user-declared PTMs --------------------------------------
    # Match against ALL non-water residues — KCX/SEP/PTR/... are usually
    # written as HETATM records, so an ATOM-only lookup would silently miss
    # them and the freeze would never engage.
    by_res: dict = {}
    for rk, ats in group_residues(st.atoms).items():
        if ats[0].resname.upper() in WATER_RESNAMES:
            continue
        by_res[reskey_str(rk)] = (rk, ats[0].resname)
    for spec, code in ptm.items():
        hit = by_res.get(spec)
        if hit is None:
            st.warn(f"  --ptm {spec} matched no residue "
                    f"(spec is CHAIN:RESID, e.g. A:169; use '_' for a blank "
                    f"chain and append the insertion code if any)")
            continue
        rk, input_resname = hit
        rec = st.residue_records.setdefault(rk, {})
        rec["ptm"] = code.upper()
        rec["ptm_frozen"] = True
        rec["ptm_input_resname"] = input_resname
        if spec in ptm_charge:
            try:
                q = int(ptm_charge[spec])
            except ValueError:
                raise SystemExit(f"--ptm-charge {spec}={ptm_charge[spec]!r} "
                                 f"is not an integer")
        elif code.upper() in PTM_CHARGE:
            q = PTM_CHARGE[code.upper()]
        else:
            st.warn(f"  PTM code {code!r} at {spec} is not in the charge table "
                    f"— assuming 0; pass --ptm-charge {spec}=Q to set it")
            q = 0
        rec["ptm_charge"] = q
        msg = (f"  PTM declared: {spec} = {code.upper()} (charge {q:+d}); "
               f"side-chain protonation frozen")
        if "n_cap" in rec or "c_cap" in rec:
            msg += "  [residue was also capped in Stage 1]"
        st.note(msg)
    for spec in ptm_charge:
        if spec not in ptm:
            st.warn(f"  --ptm-charge {spec} given without a matching --ptm")

    return st


# ============================================================================
# Stage 3 — canonical, pH-aware protonation
# ============================================================================
# Model side-chain pKa values (used only for the canonical baseline; propka
# refines them in Stage 5).
MODEL_PKA = {"ASP": 3.9, "GLU": 4.1, "HIS": 6.0, "CYS": 8.3, "TYR": 10.1,
             "LYS": 10.5, "ARG": 12.5, "SER": 13.5, "THR": 13.5}
_ACIDS = {"ASP", "GLU", "CYS", "TYR", "SER", "THR"}
_BASES = {"LYS", "ARG"}

# Valid hard-override states per residue (for --set).
VALID_SET_STATES = {
    "ASP": {"ASP", "ASH"}, "GLU": {"GLU", "GLH"},
    "HIS": {"HID", "HIE", "HIP"},
    "LYS": {"LYS", "LYN"}, "ARG": {"ARG", "ARN"},
    "CYS": {"CYS", "CYM", "CYX"}, "TYR": {"TYR", "TYM"},
    "SER": {"SER", "SED"}, "THR": {"THR", "THD"},
}
# States whose titratable side-chain H is absent (deprotonated / disulfide).
# NB: CYX (disulfide) and CYM (thiolate) both lack HG but differ in charge
# (CYX 0, CYM -1) — the distinction is carried by rec['state'] for Stage 12.
_NO_SIDECHAIN_H = {"CYM", "SED", "THD", "TYM", "ARN", "CYX"}

# Input resnames that ARE an explicit protonation state — honoured over the
# pH-derived canonical state (a user who wrote CYX/HIP/ASH meant it).
INPUT_STATE_RESNAMES = {
    "HID": "HID", "HIE": "HIE", "HIP": "HIP",
    "HSD": "HID", "HSE": "HIE", "HSP": "HIP",
    "CYX": "CYX", "CYM": "CYM", "ASH": "ASH", "GLH": "GLH",
    "LYN": "LYN", "TYM": "TYM", "SED": "SED", "THD": "THD", "ARN": "ARN",
}
# The topology h-entry parent whose H is conditional on the (de)protonation.
_TITRATABLE_PARENT = {"SER": "OG", "THR": "OG1", "CYS": "SG",
                      "TYR": "OH", "ARG": "NE"}


def canonical_state(resname: str, pH: float) -> str | None:
    """Return the canonical protonation-state label for a residue at `pH`,
    or None for a non-titratable residue."""
    rn = canonical_resname(resname)
    pka = MODEL_PKA.get(rn)
    if pka is None:
        return None
    prot = pH < pka
    if rn == "HIS":
        return "HIP" if prot else "HIS"        # "HIS" = neutral (tautomer: St.4)
    table = {
        "ASP": ("ASH", "ASP"), "GLU": ("GLH", "GLU"),
        "CYS": ("CYS", "CYM"), "TYR": ("TYR", "TYM"),
        "SER": ("SER", "SED"), "THR": ("THR", "THD"),
        "LYS": ("LYS", "LYN"), "ARG": ("ARG", "ARN"),
    }
    return table[rn][0 if prot else 1]


def _h_atom(name: str, element: str, xyz, template: Atom) -> Atom:
    """Build a Stage-3 hydrogen, inheriting residue identity from its parent."""
    return Atom(record="ATOM", name=name, resname=template.resname,
                chain=template.chain, resid=template.resid,
                icode=template.icode, xyz=np.asarray(xyz, float),
                element=element, tag="h")


def _splice_after_residue(st: ProtonationState, atoms_by_res: dict) -> None:
    """Insert each residue's new atoms right after that residue's LAST atom —
    robust to a residue whose atoms are split into non-contiguous blocks."""
    pending = {k: list(v) for k, v in atoms_by_res.items() if v}
    if not pending:
        return
    last_idx: dict = {}
    for i, a in enumerate(st.atoms):
        if a.reskey in pending:
            last_idx[a.reskey] = i
    out: list = []
    for i, a in enumerate(st.atoms):
        out.append(a)
        for rk, idx in last_idx.items():
            if idx == i:
                out.extend(pending.pop(rk))
    for leftover in pending.values():       # reskey never seen in st.atoms
        out.extend(leftover)
    st.atoms = out


def _plane_ref_for(parent_name: str, graph: dict, atom_by_name: dict):
    """Find a 3rd atom defining the conjugated plane for an sp2_term_h2 group:
    a heavy neighbour of the parent's single anchor (preferring an oxygen)."""
    nbrs = [n for n in graph.get(parent_name, ()) if n in atom_by_name]
    if len(nbrs) != 1:
        return None
    others = [n for n in graph.get(nbrs[0], ())
              if n != parent_name and n in atom_by_name]
    if not others:
        return None
    # prefer an oxygen; deterministic name tie-break (reproducible NH2 plane)
    others.sort(key=lambda n: (0 if atom_by_name[n].element == "O" else 1, n))
    return atom_by_name[others[0]].xyz


def stage3_canonical(st: ProtonationState, *, pH: float = 7.5,
                     set_overrides=None, bond_cutoff: float = 1.8
                     ) -> ProtonationState:
    """Place every canonical protein hydrogen at the given pH: backbone C-H and
    amide N-H, all side-chain C-H, Asn/Gln amide NH2, and the pH-dependent
    functional-group H. Histidine ring N-H is deferred to Stage 4. PTM-frozen
    residues are skipped. Records each residue's protonation state."""
    banner(f"Stage 3 — canonical protonation (pH {pH})")
    set_overrides = set_overrides or {}

    protein = [a for a in st.atoms if a.record == "ATOM" and a.tag != "cap"]
    residues = group_residues(protein)
    bb_carbons = [a for a in protein if a.name == "C"]
    new_by_res: dict = defaultdict(list)
    n_h_total = 0

    for rk, atoms in residues.items():
        rec = st.residue_records.setdefault(rk, {})
        rks = reskey_str(rk)
        resname = atoms[0].resname
        rn = canonical_resname(resname)
        rec["resname"] = resname

        if rec.get("ptm_frozen"):
            if rks in set_overrides:
                st.warn(f"  --set {rks} ignored — residue is PTM-frozen")
            st.note(f"  {rks} {rec.get('ptm')} — PTM-frozen, side chain skipped")
            rec["state"] = rec.get("ptm", rn)
            continue
        if rn not in TOPOLOGY:
            if rks in set_overrides:
                st.warn(f"  --set {rks} ignored — {resname} is non-standard")
            st.warn(f"  {rks} {resname}: non-standard residue — not protonated "
                    f"(declare --ptm if it is a known modification)")
            rec["state"] = resname
            continue

        atom_by_name = {a.name: a for a in atoms}
        graph = residue_heavy_graph(resname, set(atom_by_name))

        # --- determine protonation state -----------------------------------
        # priority:  --set  >  explicit input-resname state  >  pH-canonical
        state = canonical_state(resname, pH)
        input_state = INPUT_STATE_RESNAMES.get(resname.upper())
        if input_state is not None:
            if state is not None and input_state != state:
                st.note(f"  {rks}: input resname {resname} is an explicit "
                        f"state — honoring {input_state} (over pH-canonical "
                        f"{state})")
            state = input_state
        if rks in set_overrides:
            want = set_overrides[rks].upper()
            valid = VALID_SET_STATES.get(rn)
            if valid is None:
                st.warn(f"  --set {rks}={want}: {rn} is not titratable — ignored")
            elif want not in valid:
                st.warn(f"  --set {rks}={want}: invalid {rn} state "
                        f"(choose from {sorted(valid)}) — ignored")
            else:
                state = want
                st.note(f"  --set {rks} = {want}")
        rec["state"] = state if state is not None else rn
        if rec["state"] == "HIS":          # neutral His — tautomer set in St.4
            rec["his_pending"] = True

        # --- place the topology hydrogens ----------------------------------
        skip_parent = (_TITRATABLE_PARENT.get(rn)
                       if state in _NO_SIDECHAIN_H else None)
        placed = 0
        for parent_name, method, h_names in TOPOLOGY[rn]["h"]:
            if parent_name == skip_parent:
                continue
            parent = atom_by_name.get(parent_name)
            if parent is None:
                continue                       # cropped residue — skip silently
            nbrs = [atom_by_name[n].xyz for n in graph.get(parent_name, ())
                    if n in atom_by_name]
            plane_ref = (_plane_ref_for(parent_name, graph, atom_by_name)
                         if method == "sp2_term_h2" else None)
            try:
                hs = place_hydrogens(parent.xyz, nbrs, len(h_names), method,
                                     parent.element, plane_ref=plane_ref)
            except PlacementError as exc:
                st.warn(f"  {rks}.{parent_name}: {exc} — H not placed")
                continue
            for nm, xyz in zip(h_names, hs):
                new_by_res[rk].append(_h_atom(nm, "H", xyz, parent))
            placed += len(hs)

        # --- backbone amide N-H (interior peptide N, not proline) ----------
        N = atom_by_name.get("N")
        CA = atom_by_name.get("CA")
        n_cap = rec.get("n_cap")
        if N is not None and CA is not None and rn != "PRO":
            if n_cap in ("peptide", None):
                # the partner is the CLOSEST backbone C of another residue
                cands = [(float(np.linalg.norm(c.xyz - N.xyz)), c)
                         for c in bb_carbons if c.reskey != rk]
                cands = [t for t in cands if t[0] <= bond_cutoff]
                prev_c = min(cands, key=lambda t: t[0])[1] if cands else None
                if prev_c is not None:
                    try:
                        h = place_hydrogens(N.xyz, [prev_c.xyz, CA.xyz], 1,
                                            "sp2_ring_h", "N")[0]
                        new_by_res[rk].append(_h_atom("H", "H", h, N))
                        placed += 1
                    except PlacementError as exc:
                        st.warn(f"  {rks} backbone amide H: {exc}")
                elif n_cap == "peptide":
                    st.warn(f"  {rks} flagged interior but no partner C within "
                            f"{bond_cutoff} A — backbone amide H not placed")
            elif n_cap == "covalent?":
                st.note(f"  {rks} backbone amide H skipped — N flagged covalent")

        # --- Lys NZ ammonium / amine ---------------------------------------
        if rn == "LYS":
            NZ, CE = atom_by_name.get("NZ"), atom_by_name.get("CE")
            if NZ is not None and CE is not None:
                n_hz = 3 if state == "LYS" else 2
                try:
                    hs = place_hydrogens(NZ.xyz, [CE.xyz], n_hz, "sp3", "N")
                    for nm, xyz in zip(["HZ1", "HZ2", "HZ3"], hs):
                        new_by_res[rk].append(_h_atom(nm, "H", xyz, NZ))
                    placed += n_hz
                except PlacementError as exc:
                    st.warn(f"  {rks}.NZ: {exc}")

        # --- Asp / Glu carboxyl H (protonated states only) -----------------
        if rn == "ASP" and state == "ASH":
            placed += _add_carboxyl_h(st, rk, atom_by_name, "OD2", "CG",
                                      "OD1", "HD2", new_by_res)
        if rn == "GLU" and state == "GLH":
            placed += _add_carboxyl_h(st, rk, atom_by_name, "OE2", "CD",
                                      "OE1", "HE2", new_by_res)

        n_h_total += placed
        st.note(f"  {rks:<10s} {resname:<4s} state={rec['state']:<4s} "
                f"+{placed} H")

    seen = {reskey_str(rk) for rk in residues}
    for k in set_overrides:
        if k not in seen:
            st.warn(f"  --set key {k!r} matched no residue")

    _splice_after_residue(st, new_by_res)
    st.note(f"Stage 3 done — placed {n_h_total} canonical hydrogen(s)")
    return st


def _add_carboxyl_h(st: ProtonationState, rk: tuple, atom_by_name: dict,
                    o_prot: str, c_name: str, o_partner: str, h_name: str,
                    new_by_res: dict) -> int:
    """Place a carboxylic-acid H on the protonated oxygen of an Asp/Glu."""
    if h_name in atom_by_name:                  # idempotent — already placed
        return 0
    o2 = atom_by_name.get(o_prot)
    cx = atom_by_name.get(c_name)
    o1 = atom_by_name.get(o_partner)
    if o2 is None or cx is None or o1 is None:
        st.warn(f"  {reskey_str(rk)}: cannot place {h_name} — carboxylate "
                f"atoms incomplete")
        return 0
    try:
        h = place_hydrogens(o2.xyz, [cx.xyz, o1.xyz], 1, "carboxyl", "O")[0]
    except PlacementError as exc:
        st.warn(f"  {reskey_str(rk)}.{h_name}: {exc}")
        return 0
    new_by_res[rk].append(_h_atom(h_name, "H", h, o2))
    return 1


# ============================================================================
# Stage 4 — histidine tautomer
# ============================================================================
def _his_ring_partners(d: dict, which: str):
    """Return (N_atom, [ring neighbour atoms], H-name) for ND1 or NE2, or None
    if the residue is missing a required ring atom."""
    if which == "ND1":
        n, a, b, hn = d.get("ND1"), d.get("CG"), d.get("CE1"), "HD1"
    else:
        n, a, b, hn = d.get("NE2"), d.get("CE1"), d.get("CD2"), "HE2"
    if n is None or a is None or b is None:
        return None
    return (n, [a, b], hn)


# H-bond geometry windows for the tautomer reward.
_HBOND_NA_MIN = 2.4        # N..acceptor minimum (A)
_HBOND_HA_MIN = 1.4        # H..acceptor minimum (A)
_HBOND_HA_MAX = 2.8        # H..acceptor maximum (A)
_HBOND_COS = 0.60          # cos of the max N-H..acceptor deviation (~53 deg)


def _metal_contacts(st: ProtonationState, cutoff: float = 2.8) -> dict:
    """Return {reskey: [(atom_name, metal_atom, distance), ...]} — every
    protein side-chain O/N/S/Se atom within `cutoff` of a metal HETATM."""
    metals = [a for a in st.atoms if a.record == "HETATM"
              and a.element.upper() in METALS and a.tag != "cap"]
    out: dict = defaultdict(list)
    if not metals:
        return out
    for a in st.atoms:
        if (a.record != "ATOM" or a.tag == "cap"
                or a.element.upper() not in ("O", "N", "S", "SE")):
            continue
        for m in metals:
            d = float(np.linalg.norm(a.xyz - m.xyz))
            if d <= cutoff:
                out[a.reskey].append((a.name, m, d))
    return out


def _hbond_partners(N: Atom, ring_nbrs: list, acceptors: list,
                    hbond_cut: float) -> list:
    """Trial-place a ring N-H and return [(acceptor_atom, H..acceptor distance)]
    for the polar acceptors the N-H would H-bond to (same windows + cone test
    as _his_n_score). Used for verbose Stage-4 reporting."""
    try:
        h = place_hydrogens(N.xyz, [n.xyz for n in ring_nbrs], 1,
                            "sp2_ring_h", "N")[0]
    except PlacementError:
        return []
    nh = h - N.xyz
    nh_n = float(np.linalg.norm(nh))
    out = []
    for acc in acceptors:
        if acc.reskey == N.reskey:
            continue
        d_na = float(np.linalg.norm(N.xyz - acc.xyz))
        if not (_HBOND_NA_MIN <= d_na <= hbond_cut):
            continue
        d_ha = float(np.linalg.norm(h - acc.xyz))
        if not (_HBOND_HA_MIN <= d_ha <= _HBOND_HA_MAX):
            continue
        na = acc.xyz - N.xyz
        cosang = float(np.dot(nh, na) / (nh_n * float(np.linalg.norm(na))))
        if cosang >= _HBOND_COS:
            out.append((acc, d_ha))
    return out


def _his_n_score(target, ligand: list, acceptors: list, clash_cut: float,
                 hbond_cut: float) -> tuple:
    """Score a candidate ring N: +2 per genuine H-bond a trial N-H would make
    (capped at 2), -3 per clashing ligand atom. Returns (score, reason).

    An H-bond requires: N..acceptor in [2.4, hbond_cut], H..acceptor in
    [1.4, 2.8], and the N-H pointing at the acceptor within a ~53 deg cone."""
    N, ring_nbrs, _ = target
    clash = sum(1 for a in ligand
                if float(np.linalg.norm(N.xyz - a.xyz)) <= clash_cut)
    try:
        h = place_hydrogens(N.xyz, [r.xyz for r in ring_nbrs], 1,
                            "sp2_ring_h", "N")[0]
    except PlacementError:
        return -3 * clash, f"clash={clash}"
    nh = h - N.xyz
    nh_norm = float(np.linalg.norm(nh))
    hb = 0
    for acc in acceptors:
        if acc.reskey == N.reskey:
            continue
        d_na = float(np.linalg.norm(N.xyz - acc.xyz))
        if not (_HBOND_NA_MIN <= d_na <= hbond_cut):
            continue
        d_ha = float(np.linalg.norm(h - acc.xyz))
        if not (_HBOND_HA_MIN <= d_ha <= _HBOND_HA_MAX):
            continue
        na = acc.xyz - N.xyz
        cosang = float(np.dot(nh, na) / (nh_norm * float(np.linalg.norm(na))))
        if cosang >= _HBOND_COS:               # N-H points at the acceptor
            hb += 1
    hb = min(hb, 2)
    return 2 * hb - 3 * clash, f"hbond={hb} clash={clash}"


def _decide_his_tautomer(d: dict, metals: list, ligand: list, acceptors: list,
                         st: ProtonationState, rk: tuple, metal_cut: float,
                         clash_cut: float, hbond_cut: float) -> tuple:
    """Choose ND1 vs NE2 for a neutral histidine. Returns (choice, reason)."""
    rks = reskey_str(rk)
    nd1, ne2 = d.get("ND1"), d.get("NE2")
    t_nd1 = _his_ring_partners(d, "ND1")
    t_ne2 = _his_ring_partners(d, "NE2")
    if t_nd1 is None and t_ne2 is None:
        return "NE2", "ring atoms missing — default HIE"
    if t_nd1 is None:
        return "NE2", "ND1 ring incomplete — default NE2"
    if t_ne2 is None:
        return "ND1", "NE2 ring incomplete — default ND1"

    # Rule 1 (absolute): never protonate a metal-coordinating N.
    nd1_metal = any(float(np.linalg.norm(nd1.xyz - m.xyz)) <= metal_cut
                    for m in metals)
    ne2_metal = any(float(np.linalg.norm(ne2.xyz - m.xyz)) <= metal_cut
                    for m in metals)
    if nd1_metal and not ne2_metal:
        return "NE2", "ND1 coordinates a metal (rule 1)"
    if ne2_metal and not nd1_metal:
        return "ND1", "NE2 coordinates a metal (rule 1)"
    if nd1_metal and ne2_metal:
        st.warn(f"  {rks}: both ring N coordinate a metal — leaving the "
                f"imidazole unprotonated (metal-bridging imidazolate)")
        return "NONE", "both ring N metal-coordinated — no N-H placed"

    # Rules 2 + 3: clash penalty + H-bond reward (additive).
    s_nd1, r_nd1 = _his_n_score(t_nd1, ligand, acceptors, clash_cut, hbond_cut)
    s_ne2, r_ne2 = _his_n_score(t_ne2, ligand, acceptors, clash_cut, hbond_cut)
    if s_nd1 > s_ne2:
        return "ND1", f"ND1 score {s_nd1:+d} ({r_nd1}) > NE2 {s_ne2:+d}"
    if s_ne2 > s_nd1:
        return "NE2", f"NE2 score {s_ne2:+d} ({r_ne2}) > ND1 {s_nd1:+d}"
    return "NE2", "no geometric signal — default HIE"


def stage4_his_tautomer(st: ProtonationState, *, metal_coord_cutoff: float = 2.8,
                        clash_cutoff: float = 2.0, hbond_cutoff: float = 3.5
                        ) -> ProtonationState:
    """Place histidine ring N-H. HID/HIE/HIP states (from --set or an explicit
    input resname) are honoured directly; a neutral His chooses its tautomer
    from metal coordination, ligand clashes and H-bond geometry."""
    banner("Stage 4 — histidine tautomer")
    protein = [a for a in st.atoms if a.record == "ATOM" and a.tag != "cap"]
    residues = group_residues(protein)
    metals = [a for a in st.atoms if a.record == "HETATM"
              and a.element.upper() in METALS and a.tag != "cap"]
    ligand = [a for a in st.atoms if a.record == "HETATM"
              and a.element.upper() not in METALS
              and a.resname.upper() not in WATER_RESNAMES and a.tag != "cap"]
    # H-bond acceptors: oxygen lone-pair atoms (carbonyl / carboxylate /
    # hydroxyl / water — water IS kept here, unlike the `ligand` clash set)
    # plus histidine ring N (the only common N acceptor; amide / ammonium /
    # guanidinium N are donors). Atoms inside a metal coordination sphere are
    # dropped — Rule 1 already covers the metal site.
    raw_acc = [a for a in st.atoms if not a.is_hydrogen and a.tag != "cap"
               and (a.element.upper() == "O"
                    or (a.element.upper() == "N" and a.name in ("ND1", "NE2")))]
    acceptors = [a for a in raw_acc
                 if not any(float(np.linalg.norm(a.xyz - m.xyz))
                            <= metal_coord_cutoff for m in metals)]
    mcontacts = _metal_contacts(st, metal_coord_cutoff)

    def _metal_near(N: Atom) -> bool:
        return any(float(np.linalg.norm(N.xyz - m.xyz)) <= metal_coord_cutoff
                   for m in metals)

    new_by_res: dict = defaultdict(list)
    n_his = 0
    for rk, atoms in residues.items():
        if canonical_resname(atoms[0].resname) != "HIS":
            continue
        rec = st.residue_records.setdefault(rk, {})
        rks = reskey_str(rk)
        if rec.get("ptm_frozen"):
            st.note(f"  {rks} PTM-frozen — imidazole ring left unprotonated")
            continue
        n_his += 1
        d = {a.name: a for a in atoms}
        state = rec.get("state", "HIS")
        if state not in ("HID", "HIE", "HIP", "HIS"):
            st.warn(f"  {rks}: unexpected His state {state!r} — auto-deciding")
            state = "HIS"
        rec.pop("his_pending", None)            # handled here, whatever path

        # --- verbose environment summary (always log) ----------------------
        ring_contacts = [(an, m, dist) for (an, m, dist)
                         in mcontacts.get(rk, [])
                         if an in ("ND1", "NE2")]
        if ring_contacts:
            st.note(f"  {rks} HIS  metal on ring: " + ", ".join(
                f"{an} ↔ {m.resname}({reskey_str(m.reskey)}) {dist:.2f} Å"
                for an, m, dist in ring_contacts))
        else:
            st.note(f"  {rks} HIS  no metal contacts on ring N")
        t_nd1 = _his_ring_partners(d, "ND1")
        t_ne2 = _his_ring_partners(d, "NE2")
        hb_nd1 = (_hbond_partners(t_nd1[0], t_nd1[1], acceptors, hbond_cutoff)
                  if t_nd1 else [])
        hb_ne2 = (_hbond_partners(t_ne2[0], t_ne2[1], acceptors, hbond_cutoff)
                  if t_ne2 else [])

        def _hb_desc(hs):
            return ("none" if not hs else ", ".join(
                f"{a.resname}({reskey_str(a.reskey)}).{a.name} {di:.2f} Å"
                for a, di in hs))
        st.note(f"  {rks} HIS  H-bond candidates: "
                f"ND1-H → {_hb_desc(hb_nd1)}; NE2-H → {_hb_desc(hb_ne2)}")
        # acceptors masked by the metal-proximity filter — surface them too
        # so a 'none' candidate set is auditable (in a metalloprotein the most
        # interesting H-bond partner is often a metal-bound O)
        if t_nd1 or t_ne2:
            _acc_ids = {id(a) for a in acceptors}
            masked = [a for a in raw_acc if id(a) not in _acc_ids]
            if masked:
                hb_nd1_m = (_hbond_partners(t_nd1[0], t_nd1[1], masked,
                                            hbond_cutoff) if t_nd1 else [])
                hb_ne2_m = (_hbond_partners(t_ne2[0], t_ne2[1], masked,
                                            hbond_cutoff) if t_ne2 else [])
                if hb_nd1_m or hb_ne2_m:
                    st.note(f"  {rks} HIS  metal-masked acceptors "
                            f"(not scored): ND1-H → {_hb_desc(hb_nd1_m)}; "
                            f"NE2-H → {_hb_desc(hb_ne2_m)}")

        targets: list = []
        if state in ("HID", "HIE", "HIP"):
            if state in ("HID", "HIP"):
                t = _his_ring_partners(d, "ND1")
                if t:
                    targets.append(t)
            if state in ("HIE", "HIP"):
                t = _his_ring_partners(d, "NE2")
                if t:
                    targets.append(t)
            st.note(f"  {rks} {state} (explicit state)")
            for N, _, _ in targets:
                if _metal_near(N):
                    st.warn(f"  {rks}: explicit {state} protonates {N.name}, "
                            f"which coordinates a metal — honoring it, verify")
        else:                                   # neutral — decide tautomer
            choice, reason = _decide_his_tautomer(
                d, metals, ligand, acceptors, st, rk,
                metal_coord_cutoff, clash_cutoff, hbond_cutoff)
            if choice == "NONE":
                rec["state"] = "HIS"
                rec["his_metal_bridge"] = True
                st.note(f"  {rks} -> imidazolate ({reason})")
            else:
                t = _his_ring_partners(d, choice)
                if t:
                    targets.append(t)
                rec["state"] = "HID" if choice == "ND1" else "HIE"
                st.note(f"  {rks} -> {rec['state']}  ({reason})")

        for N, ring_nbrs, hname in targets:
            if hname in d:                       # idempotent — already placed
                st.note(f"  {rks}.{hname} already present — skipped")
                continue
            try:
                h = place_hydrogens(N.xyz, [r.xyz for r in ring_nbrs], 1,
                                    "sp2_ring_h", "N")[0]
                new_by_res[rk].append(_h_atom(hname, "H", h, N))
            except PlacementError as exc:
                st.warn(f"  {rks}.{hname}: {exc} — ring N-H not placed")

    _splice_after_residue(st, new_by_res)
    st.note(f"Stage 4 done — {n_his} histidine(s)")
    return st


# ============================================================================
# Geometry autocorrection — always-on clash repair of placed hydrogens
# ============================================================================
_H_BONDLEN_TOL = 0.15        # placed-H bond-length deviation warning threshold
# sp2 amine groups — only the 0/180 deg flip keeps the NH2 in its conjugated
# plane, so the torsion sweep is restricted to {0, pi} for these parents.
_SP2_NH2_PARENTS = {"ND2", "NE2", "NH1", "NH2", "NT"}


def _rotate_points(points, origin, axis, angle: float) -> list:
    """Rotate `points` about the line (origin, axis) by `angle` (Rodrigues)."""
    axis = _normalize(axis)
    c, s = np.cos(angle), np.sin(angle)
    out = []
    for p in points:
        v = np.asarray(p, float) - origin
        out.append(origin + v * c + np.cross(axis, v) * s
                   + axis * float(np.dot(axis, v)) * (1.0 - c))
    return out


def stage_autocorrect(st: ProtonationState, *, h_clash_cutoff: float = 1.5,
                      sweep_steps: int = 36, max_passes: int = 3
                      ) -> ProtonationState:
    """Validate every script-placed hydrogen and repair clashes. A rotatable
    group (a heavy atom with exactly one heavy neighbour: methyl, O-H, S-H,
    NH2/NH3, amide NH2) is torsion-swept to its least-clashing rotamer; clashes
    on non-rotatable H (backbone amide, aromatic ring) are reported only."""
    banner("Geometry autocorrection")
    heavy = [a for a in st.atoms if not a.is_hydrogen]
    placed = [a for a in st.atoms if a.is_hydrogen and a.tag in ("h", "cap")]
    if not placed:
        st.note("  no script-placed hydrogens to check")
        st.settings["autocorrect"] = {"clashes_before": 0, "groups_fixed": 0,
                                      "clashes_after": 0}
        return st

    by_res_heavy: dict = defaultdict(list)
    for a in heavy:
        by_res_heavy[a.reskey].append(a)

    def heavy_neighbours(x):
        return [a for a in heavy if a is not x
                and float(np.linalg.norm(a.xyz - x.xyz)) <= 1.95]

    # Parent heavy atom of each placed H = nearest heavy atom in its residue,
    # required to be within bonding distance (a placed H sits ~1.0 A from its
    # true parent; nothing non-bonded is that close — so this is unambiguous).
    group_h: dict = defaultdict(list)
    parent_by_id: dict = {}
    orphans = 0
    for h in placed:
        cands = by_res_heavy.get(h.reskey)
        p = (min(cands, key=lambda a: float(np.linalg.norm(a.xyz - h.xyz)))
             if cands else None)
        if p is None or float(np.linalg.norm(p.xyz - h.xyz)) > 1.4:
            orphans += 1
            st.warn(f"  placed H {reskey_str(h.reskey)}.{h.name}: no heavy "
                    f"parent within bonding distance — skipped")
            continue
        group_h[id(p)].append(h)
        parent_by_id[id(p)] = p

    # Per group: parent, its H, heavy neighbours, the EXACT bonded-exclusion id
    # set (parent + heavy neighbours + the group's own H), rotatability, and
    # whether it is an sp2 NH2 (swept only 0/180 to stay planar).
    groups: list = []
    for pid, hs in group_h.items():
        parent = parent_by_id[pid]
        nbrs = heavy_neighbours(parent)
        exclude = ({id(parent)} | {id(n) for n in nbrs}
                   | {id(h) for h in hs})
        groups.append(dict(parent=parent, hs=hs, nbrs=nbrs, exclude=exclude,
                           rotatable=len(nbrs) == 1,
                           sp2=parent.name in _SP2_NH2_PARENTS))

    # O(placed_H * atoms) per evaluation — fine at theozyme scale.
    def n_clash(xyz, exclude) -> int:
        return sum(1 for a in st.atoms if id(a) not in exclude
                   and float(np.linalg.norm(xyz - a.xyz)) < h_clash_cutoff)

    def soft_clash(xyz, exclude) -> float:
        s = 0.0
        for a in st.atoms:
            if id(a) in exclude:
                continue
            d = float(np.linalg.norm(xyz - a.xyz))
            if d < h_clash_cutoff:
                s += h_clash_cutoff - d
        return s

    bad_bondlen = 0
    clashes_before = 0
    for g in groups:
        ideal = h_bond_length(g["parent"].element)
        for h in g["hs"]:
            if abs(float(np.linalg.norm(h.xyz - g["parent"].xyz))
                   - ideal) > _H_BONDLEN_TOL:
                bad_bondlen += 1
        clashes_before += sum(n_clash(h.xyz, g["exclude"]) for h in g["hs"])

    # Torsion-sweep rotatable groups; iterate to a fixed point because
    # neighbouring groups can clash into each other.
    groups_fixed = 0
    for _pass in range(max_passes):
        moved = False
        for g in groups:
            if not g["rotatable"]:
                continue
            parent, hs, exclude = g["parent"], g["hs"], g["exclude"]
            cur = sum(n_clash(h.xyz, exclude) for h in hs)
            if cur == 0:
                continue
            axis = g["nbrs"][0].xyz - parent.xyz
            if float(np.linalg.norm(axis)) < 1e-6:
                continue                        # degenerate axis — skip
            angles = ([np.pi] if g["sp2"]
                      else [2.0 * np.pi * k / sweep_steps
                            for k in range(1, sweep_steps)])
            best_soft = sum(soft_clash(h.xyz, exclude) for h in hs)
            best_ang = None
            for ang in angles:
                rot = _rotate_points([h.xyz for h in hs], parent.xyz,
                                     axis, ang)
                soft = sum(soft_clash(p, exclude) for p in rot)
                if soft < best_soft - 1e-6:
                    best_soft, best_ang = soft, ang
            if best_ang is None:
                continue
            rot = _rotate_points([h.xyz for h in hs], parent.xyz, axis,
                                 best_ang)
            after = sum(n_clash(p, exclude) for p in rot)
            if after <= cur:                    # never keep a worsening move
                for h, p in zip(hs, rot):
                    h.xyz = p
                moved = True
                if after < cur and _pass == 0:
                    groups_fixed += 1
                    st.note(f"  {reskey_str(parent.reskey)}.{parent.name}: "
                            f"rotated {np.degrees(best_ang):.0f} deg — "
                            f"clashes {cur} -> {after}")
        if not moved:
            break

    clashes_after = 0
    for g in groups:
        after = sum(n_clash(h.xyz, g["exclude"]) for h in g["hs"])
        clashes_after += after
        if after and not g["rotatable"]:
            st.warn(f"  {reskey_str(g['parent'].reskey)}.{g['parent'].name}: "
                    f"{after} H clash(es) on a non-rotatable group — left for "
                    f"the optional MLFF relax")

    if bad_bondlen:
        st.warn(f"  {bad_bondlen} placed H deviate > {_H_BONDLEN_TOL} A from "
                f"the ideal bond length")
    if orphans:
        st.warn(f"  {orphans} placed H had no identifiable parent atom")
    st.settings["autocorrect"] = {"clashes_before": clashes_before,
                                  "groups_fixed": groups_fixed,
                                  "clashes_after": clashes_after}
    st.note(f"Autocorrection done — {clashes_before} initial H clash(es), "
            f"{groups_fixed} group(s) rotated, {clashes_after} remaining")
    return st


# ============================================================================
# Stage 5 — propka refinement
# ============================================================================
# Side-chain heteroatom(s) used by the metal guard, and the H name(s) removed
# to deprotonate, per residue.
_SIDECHAIN_HET = {"ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"], "CYS": ["SG"],
                  "SER": ["OG"], "THR": ["OG1"], "TYR": ["OH"]}
_DEPROT_H = {"ASP": ["HD2"], "GLU": ["HE2"], "CYS": ["HG"], "SER": ["HG"],
             "THR": ["HG1"], "TYR": ["HH"]}
# All titratable side-chain atoms (incl. HIS/LYS/ARG) used by atom-level
# verbose reporting — "which atoms are metal-coordinating vs free".
_SIDECHAIN_TITRATABLE_ATOMS = {
    **_SIDECHAIN_HET,
    "HIS": ["ND1", "NE2"], "LYS": ["NZ"], "ARG": ["NE", "NH1", "NH2"],
}
# Cation pair labels for LYS/ARG (protonated, deprotonated) — for the propka
# pKa summary section. HIS is handled in its own branch because the neutral
# form is one of two tautomers (HID/HIE), not a fixed label.
_BASE_PAIR = {"LYS": ("LYS", "LYN"), "ARG": ("ARG", "ARN")}
# (protonated, deprotonated) state label per acid-type residue
_ACID_STATES = {"ASP": ("ASH", "ASP"), "GLU": ("GLH", "GLU"),
                "CYS": ("CYS", "CYM"), "TYR": ("TYR", "TYM"),
                "SER": ("SER", "SED"), "THR": ("THR", "THD")}


# propka group types that are titratable amino-acid SIDE CHAINS — the N+/C-
# terminus groups and ligand groups share a residue's (chain, resnum) and must
# be excluded so they do not overwrite the side-chain group.
_PROPKA_SC_TYPES = {"ASP", "GLU", "HIS", "CYS", "TYR", "LYS", "ARG"}
_PROPKA_SENTINEL_PKA = 20.0          # |pKa| above this = propka "untitratable"


def _propka_pka(st: ProtonationState, pH: float):
    """Run propka on `st` and return {(chain, resnum, icode): {'pKa','resn'}}
    for titratable SIDE-CHAIN groups only, or None on any failure (propka can
    choke on isolated theozyme residues). All propka output and temp files are
    contained in a private temp directory and cleaned up."""
    try:
        import propka.input
        import propka.lib
        import propka.molecular_container
        import propka.parameters
    except ImportError:
        return None
    import contextlib
    import io
    import shutil

    cwd = os.getcwd()
    workdir = tempfile.mkdtemp(prefix="protonator_propka_")
    prop_log = logging.getLogger("propka")
    prop_level = prop_log.level
    prop_log.setLevel(logging.ERROR)         # silence propka's own warnings
    try:
        write_pdb(st, Path(workdir) / "input.pdb",
                  remarks=["stage 5 propka input"])
        os.chdir(workdir)
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            opts = propka.lib.loadOptions(["input.pdb"])
            params = propka.input.read_parameter_file(
                opts.parameters, propka.parameters.Parameters())
            molc = propka.molecular_container.MolecularContainer(params, opts)
            propka.input.read_molecule_file(opts.input_pdb, molc)
            molc.calculate_pka()
        out: dict = {}
        for grp in molc.conformations["AVR"].groups:
            pk = grp.pka_value
            if pk is None:
                continue
            gtype = str(getattr(grp, "type", "")).strip().upper()
            if gtype not in _PROPKA_SC_TYPES:    # skip N+/C-/ligand groups
                continue
            chain = str(grp.atom.chain_id).strip()
            resnum = int(grp.atom.res_num)
            icode = str(getattr(grp.atom, "icode", "")).strip()
            out[(chain, resnum, icode)] = {"pKa": float(pk), "resn": gtype}
        return out
    except Exception as exc:               # noqa: BLE001 — degrade gracefully
        log.debug(f"propka failed: {exc}")
        return None
    finally:
        os.chdir(cwd)
        prop_log.setLevel(prop_level)
        shutil.rmtree(workdir, ignore_errors=True)


def _remove_residue_atoms(st: ProtonationState, rk: tuple, names) -> int:
    """Delete named atoms of a residue; return how many were removed."""
    names = set(names)
    before = len(st.atoms)
    st.atoms = [a for a in st.atoms
                if not (a.reskey == rk and a.name in names)]
    return before - len(st.atoms)


def _add_titratable_h(st: ProtonationState, rk: tuple, abn: dict, rn: str,
                      new_by_res: dict) -> bool:
    """Re-place the neutral side-chain O-H / S-H of a CYS / SER / THR / TYR."""
    parent_name = _TITRATABLE_PARENT.get(rn)
    spec = TOPOLOGY.get(rn)
    if not parent_name or parent_name not in abn or spec is None:
        return False
    entry = next((e for e in spec["h"] if e[0] == parent_name), None)
    if entry is None:
        return False
    _, method, hnames = entry
    if any(nm in abn for nm in hnames):           # already present
        return False
    parent = abn[parent_name]
    graph = residue_heavy_graph(rn, set(abn))
    nbrs = [abn[n].xyz for n in graph.get(parent_name, ()) if n in abn]
    try:
        hs = place_hydrogens(parent.xyz, nbrs, len(hnames), method,
                             parent.element)
    except PlacementError:
        return False
    for nm, xyz in zip(hnames, hs):
        new_by_res[rk].append(_h_atom(nm, "H", xyz, parent))
    return True


def stage5_propka(st: ProtonationState, *, pH: float = 7.5,
                  skip_propka: bool = False, set_overrides=None,
                  metal_coord_cutoff: float = 2.8) -> ProtonationState:
    """Refine titratable states. A metal guard (always on) force-deprotonates
    any side-chain O/S that coordinates a metal. propka then swaps a residue's
    state when its predicted pKa disagrees with the canonical assignment —
    a swap that would protonate a metal-guarded atom is vetoed."""
    banner(f"Stage 5 — propka refinement  (user pH = {pH})")
    set_overrides = {k for k in (set_overrides or {})}
    protein = [a for a in st.atoms if a.record == "ATOM" and a.tag != "cap"]
    residues = group_residues(protein)
    metals = [a for a in st.atoms if a.record == "HETATM"
              and a.element.upper() in METALS and a.tag != "cap"]

    def metal_near(atom):
        return any(float(np.linalg.norm(atom.xyz - m.xyz))
                   <= metal_coord_cutoff for m in metals)

    # ---- atom-level metal coordination map (verbose) ---------------------
    mcontacts = _metal_contacts(st, metal_coord_cutoff)
    if mcontacts:
        st.note("  metal coordination (atom-level):")
        for rk in sorted(mcontacts):
            atoms_here = residues.get(rk)
            if atoms_here is None:
                continue
            resname = atoms_here[0].resname
            rn = canonical_resname(resname)
            entries = "; ".join(
                f"{an} ↔ {m.resname}({reskey_str(m.reskey)}) {d:.2f} Å"
                for an, m, d in mcontacts[rk])
            titratable = _SIDECHAIN_TITRATABLE_ATOMS.get(rn, [])
            contacted = {an for an, _, _ in mcontacts[rk]}
            free = [a for a in titratable if a not in contacted]
            if free:
                annotation = f"  (free: {', '.join(free)})"
            elif titratable:
                annotation = "  (fully chelated — no free titratable site)"
            else:
                annotation = ""
            st.note(f"    {reskey_str(rk)} {resname:<3s}  {entries}{annotation}")
    else:
        st.note("  no metal contacts detected in this structure")

    # ---- metal guard (always runs) ---------------------------------------
    n_guard = 0
    for rk, atoms in residues.items():
        rec = st.residue_records.setdefault(rk, {})
        if rec.get("ptm_frozen"):
            continue
        rn = canonical_resname(atoms[0].resname)
        if rn not in _SIDECHAIN_HET:
            continue
        abn = {a.name: a for a in atoms}
        het = [abn[n] for n in _SIDECHAIN_HET[rn] if n in abn]
        if not het or not any(metal_near(h) for h in het):
            continue
        if reskey_str(rk) in set_overrides:
            if rec.get("state") in ("ASH", "GLH", "CYS", "TYR", "SER", "THR"):
                st.warn(f"  --set {reskey_str(rk)} keeps a metal-coordinating "
                        f"{rn} protonated — the metal guard was NOT applied; "
                        f"verify this is intended")
            continue
        rec["metal_guarded"] = True
        removed = _remove_residue_atoms(st, rk, _DEPROT_H.get(rn, []))
        rec["state"] = _ACID_STATES[rn][1]
        n_guard += 1
        contact_str = "; ".join(
            f"{an}↔{m.resname}({reskey_str(m.reskey)}) {d:.2f} Å"
            for an, m, d in mcontacts.get(rk, [])
            if an in _SIDECHAIN_HET[rn])
        removed_names = ", ".join(_DEPROT_H.get(rn, [])) if removed else ""
        st.note(f"  metal guard: {reskey_str(rk)} {rn} via {contact_str} — "
                + (f"forced deprotonated (removed: {removed_names})" if removed
                   else "already deprotonated; flagged"))
    if n_guard:
        st.note(f"  metal guard applied to {n_guard} residue(s)")

    if skip_propka:
        st.note("  --skip-propka — propka refinement skipped")
        return st

    pka = _propka_pka(st, pH)
    if pka is None:
        st.warn("  propka unavailable or failed on this structure — keeping "
                "the canonical / metal-guarded states")
        return st
    st.settings["propka_pka"] = pka          # reused by Stage 6 (protomers)

    # ---- propka pKa estimates per titratable residue (verbose) ------------
    st.note(f"  propka pKa estimates @ pH {pH}:")
    for rk, atoms in residues.items():
        rec = st.residue_records.setdefault(rk, {})
        if rec.get("ptm_frozen"):
            continue
        rn = canonical_resname(atoms[0].resname)
        if rn not in MODEL_PKA:               # non-titratable
            continue
        info = pka.get((rk[0].strip(), rk[1], rk[2].strip()))
        cur = rec.get("state", rn)
        rks = reskey_str(rk)
        if info is None or info["resn"] != rn:
            extra = "  [metal-locked]" if rec.get("metal_guarded") else ""
            msg = f"propka has no estimate{extra} — current {cur} kept"
        elif abs(info["pKa"]) > _PROPKA_SENTINEL_PKA:
            extra = "  [metal-locked]" if rec.get("metal_guarded") else ""
            msg = (f"pKa={info['pKa']:+.2f} (sentinel — propka could not "
                   f"titrate){extra}; current {cur} kept")
        else:
            pp = info["pKa"] >= pH
            if rn == "HIS":
                if rec.get("his_metal_bridge"):
                    cur_note = "  [imidazolate — both ring N metal-bound]"
                else:
                    cur_note = ""
                pred = ("HIP (doubly protonated, +1)" if pp
                        else "neutral (HID or HIE)")
            elif rn in _BASE_PAIR:
                on, off = _BASE_PAIR[rn]
                cur_note = ""
                pred = (f"{on} (+1, protonated)" if pp
                        else f"{off} (neutral)")
            else:                             # ASP/GLU/CYS/TYR/SER/THR
                prot, deprot = _ACID_STATES[rn]
                cur_note = ""
                pred = (f"{prot} (protonated)" if pp
                        else f"{deprot} (deprotonated)")
            msg = (f"pKa={info['pKa']:+6.2f} → propka: {pred}; "
                   f"current {cur}{cur_note}")
            if rec.get("metal_guarded"):
                msg += "  [metal-guarded → no swap]"
            elif rks in set_overrides:
                msg += "  [--set override → no swap]"
        st.note(f"    {rks} {rn:3s}  {msg}")

    # ---- apply swaps where propka disagrees with the canonical state -----
    new_by_res: dict = defaultdict(list)
    n_swap = 0
    for rk, atoms in residues.items():
        rec = st.residue_records.setdefault(rk, {})
        if rec.get("ptm_frozen") or reskey_str(rk) in set_overrides:
            continue
        rn = canonical_resname(atoms[0].resname)
        info = pka.get((rk[0].strip(), rk[1], rk[2].strip()))
        if info is None or info["resn"] != rn:   # require a matching SC group
            continue
        rec["pKa"] = info["pKa"]                  # recorded for the info-JSON
        if abs(info["pKa"]) > _PROPKA_SENTINEL_PKA:
            st.note(f"  {reskey_str(rk)} {rn}: propka pKa {info['pKa']:.1f} is "
                    f"an untitratable sentinel — no swap")
            continue
        propka_prot = info["pKa"] >= pH          # protonated form when pH<=pKa
        cur = rec.get("state", rn)
        abn = {a.name: a for a in atoms}
        rks = reskey_str(rk)

        if rn in _ACID_STATES:                   # ASP/GLU/CYS/TYR/SER/THR
            prot_state, deprot_state = _ACID_STATES[rn]
            if propka_prot and cur == deprot_state:
                if rec.get("metal_guarded"):
                    st.note(f"  {rks} {rn}: propka pKa {info['pKa']:.1f} wants "
                            f"protonation — vetoed by the metal guard")
                    continue
                if rn in ("ASP", "GLU"):
                    o_p, c_n, o_o, h_n = (
                        ("OD2", "CG", "OD1", "HD2") if rn == "ASP"
                        else ("OE2", "CD", "OE1", "HE2"))
                    ok = bool(_add_carboxyl_h(st, rk, abn, o_p, c_n, o_o, h_n,
                                              new_by_res))
                else:                            # CYS/TYR/SER/THR re-protonate
                    ok = _add_titratable_h(st, rk, abn, rn, new_by_res)
                if ok:
                    rec["state"] = prot_state
                    n_swap += 1
                    st.note(f"  {rks} {rn} -> {prot_state} "
                            f"(propka pKa {info['pKa']:.1f})")
                else:
                    st.warn(f"  {rks} {rn}: propka wants protonation but the "
                            f"side-chain H could not be placed")
            elif (not propka_prot) and cur == prot_state:
                if _remove_residue_atoms(st, rk, _DEPROT_H.get(rn, [])):
                    rec["state"] = deprot_state
                    n_swap += 1
                    st.note(f"  {rks} {rn} -> {deprot_state} "
                            f"(propka pKa {info['pKa']:.1f})")

        elif rn == "HIS":
            if rec.get("his_metal_bridge"):       # metal-bridging imidazolate
                continue
            has_hd1, has_he2 = "HD1" in abn, "HE2" in abn
            is_hip = has_hd1 and has_he2
            if propka_prot and not is_hip:
                want = "ND1" if not has_hd1 else "NE2"
                npos = abn.get(want)
                if npos is not None and metal_near(npos):
                    st.note(f"  {rks} HIS: propka wants HIP but {want} "
                            f"coordinates a metal — vetoed")
                else:
                    t = _his_ring_partners(abn, want)
                    if t is not None:
                        N, nbrs, hname = t
                        try:
                            h = place_hydrogens(N.xyz, [n.xyz for n in nbrs],
                                                1, "sp2_ring_h", "N")[0]
                            new_by_res[rk].append(_h_atom(hname, "H", h, N))
                            rec["state"] = "HIP"
                            n_swap += 1
                            st.note(f"  {rks} HIS -> HIP "
                                    f"(propka pKa {info['pKa']:.1f})")
                        except PlacementError as exc:
                            st.warn(f"  {rks} HIP {hname}: {exc}")
            elif (not propka_prot) and is_hip:    # metal-aware HIP -> neutral
                nd1, ne2 = abn.get("ND1"), abn.get("NE2")
                nd1_m = nd1 is not None and metal_near(nd1)
                ne2_m = ne2 is not None and metal_near(ne2)
                if ne2_m and not nd1_m:
                    drop, keep = "HE2", "HID"
                else:
                    drop, keep = "HD1", "HIE"
                if _remove_residue_atoms(st, rk, [drop]):
                    rec["state"] = keep
                    n_swap += 1
                    st.note(f"  {rks} HIP -> {keep} "
                            f"(propka pKa {info['pKa']:.1f})")

        elif rn == "LYS":
            if (not propka_prot) and cur == "LYS":
                if _remove_residue_atoms(st, rk, ["HZ3"]):
                    rec["state"] = "LYN"
                    n_swap += 1
                    st.note(f"  {rks} LYS -> LYN "
                            f"(propka pKa {info['pKa']:.1f})")

    _splice_after_residue(st, new_by_res)
    st.note(f"Stage 5 done — {n_guard} metal-guarded, {n_swap} propka swap(s)")
    return st


# ============================================================================
# Stage 6 — protomer enumeration
# ============================================================================
# (protonated, deprotonated) state label per titratable residue (HIS special).
_PROT_PAIR = {"ASP": ("ASH", "ASP"), "GLU": ("GLH", "GLU"),
              "CYS": ("CYS", "CYM"), "TYR": ("TYR", "TYM"),
              "LYS": ("LYS", "LYN")}


def _frac_protonated(pKa: float, pH: float) -> float:
    """Henderson-Hasselbalch fraction of a group in its protonated form."""
    try:
        return 1.0 / (1.0 + 10.0 ** (pH - pKa))
    except OverflowError:
        return 0.0 if pH > pKa else 1.0


def _add_his_ring_h(st, rk, abn, which, new_by_res) -> bool:
    """Place one histidine ring N-H (which = 'ND1' or 'NE2')."""
    t = _his_ring_partners(abn, which)
    if t is None:
        return False
    N, nbrs, hname = t
    if hname in abn:
        return False
    try:
        h = place_hydrogens(N.xyz, [n.xyz for n in nbrs], 1,
                            "sp2_ring_h", "N")[0]
    except PlacementError:
        return False
    new_by_res[rk].append(_h_atom(hname, "H", h, N))
    return True


def _force_residue_state(st, rk, rn, target, abn, new_by_res) -> bool:
    """Edit residue rk so its side-chain matches `target`. Idempotent (a no-op
    when already in the target state). Returns True on success — the caller
    must NOT trust rec['state'] if this returns False."""
    if rn in ("ASP", "GLU"):
        if target in ("ASH", "GLH"):
            o_p, c_n, o_o, h_n = (("OD2", "CG", "OD1", "HD2") if rn == "ASP"
                                  else ("OE2", "CD", "OE1", "HE2"))
            if h_n in abn:
                return True
            return bool(_add_carboxyl_h(st, rk, abn, o_p, c_n, o_o, h_n,
                                        new_by_res))
        _remove_residue_atoms(st, rk, _DEPROT_H[rn])
        return True
    if rn in ("CYS", "TYR"):
        if target in ("CYS", "TYR"):
            if any(nm in abn for nm in _DEPROT_H[rn]):
                return True
            return _add_titratable_h(st, rk, abn, rn, new_by_res)
        _remove_residue_atoms(st, rk, _DEPROT_H[rn])
        return True
    if rn == "LYS":
        want = 3 if target == "LYS" else 2
        if len([nm for nm in ("HZ1", "HZ2", "HZ3") if nm in abn]) == want:
            return True
        nz, ce = abn.get("NZ"), abn.get("CE")
        if nz is None or ce is None:
            return False
        try:                               # place FIRST, then commit
            hs = place_hydrogens(nz.xyz, [ce.xyz], want, "sp3", "N")
        except PlacementError:
            return False
        _remove_residue_atoms(st, rk, ["HZ1", "HZ2", "HZ3"])
        for nm, xyz in zip(["HZ1", "HZ2", "HZ3"], hs):
            new_by_res[rk].append(_h_atom(nm, "H", xyz, nz))
        return True
    if rn == "HIS":
        want_hd1, want_he2 = target in ("HID", "HIP"), target in ("HIE", "HIP")
        ok = True
        if not want_hd1:
            _remove_residue_atoms(st, rk, ["HD1"])
        if not want_he2:
            _remove_residue_atoms(st, rk, ["HE2"])
        if want_hd1 and "HD1" not in abn:
            ok = _add_his_ring_h(st, rk, abn, "ND1", new_by_res) and ok
        if want_he2 and "HE2" not in abn:
            ok = _add_his_ring_h(st, rk, abn, "NE2", new_by_res) and ok
        return ok
    return False


def stage6_protomers(st: ProtonationState, *, n_protomers: int = 1,
                     pH: float = 7.5, min_prob: float = 0.15,
                     max_variable: int = 12, couples=None,
                     set_overrides=None) -> list:
    """Return a list of protomer dicts {rank, prob, state, assignment}.
    n_protomers <= 1 → the single base structure. Otherwise the N most likely
    joint protonation states of the propka-ambiguous residues are enumerated;
    caps, defaults, His tautomer choice, --set and --ptm residues never vary."""
    couples = couples or []
    set_overrides = {k for k in (set_overrides or {})}
    if n_protomers <= 1:
        return [{"rank": 1, "prob": 1.0, "state": st, "assignment": {}}]
    banner(f"Stage 6 — protomer enumeration (top {n_protomers})")

    pka = st.settings.get("propka_pka")
    if not pka:
        st.warn("  no propka pKa available — emitting only the canonical state")
        return [{"rank": 1, "prob": 1.0, "state": st, "assignment": {}}]

    residues = group_residues([a for a in st.atoms
                               if a.record == "ATOM" and a.tag != "cap"])

    # ---- variable titratable residues ------------------------------------
    variable: dict = {}
    for rk, atoms in residues.items():
        rec = st.residue_records.get(rk, {})
        if (rec.get("ptm_frozen") or rec.get("his_metal_bridge")
                or rec.get("metal_guarded")
                or reskey_str(rk) in set_overrides):
            continue
        rn = canonical_resname(atoms[0].resname)
        info = pka.get((rk[0].strip(), rk[1], rk[2].strip()))
        if info is None or info["resn"] != rn:
            continue
        if abs(info["pKa"]) > _PROPKA_SENTINEL_PKA:
            continue
        p_prot = _frac_protonated(info["pKa"], pH)
        if min(p_prot, 1.0 - p_prot) < min_prob:
            continue
        if rn == "HIS":
            # Stage 6 varies the CHARGE state (HIP vs neutral); the neutral
            # tautomer is whatever Stage 4/5 fixed and is never re-decided here.
            cur = rec.get("state", "HIE")
            if cur not in ("HID", "HIE"):
                st.note(f"  {reskey_str(rk)}: variable His is currently {cur} "
                        f"— neutral protomer end defaults to HIE")
                cur = "HIE"
            variable[rk] = (rn, "HIP", cur, p_prot)
        elif rn in _PROT_PAIR:
            variable[rk] = (rn, *_PROT_PAIR[rn], p_prot)
    if not variable:
        st.note("  no propka-ambiguous residues — only one protomer")
        return [{"rank": 1, "prob": 1.0, "state": st, "assignment": {}}]
    st.note(f"  {len(variable)} variable residue(s): "
            + ", ".join(reskey_str(rk) for rk in variable))

    # ---- units: coupled pairs vary in lockstep ---------------------------
    coupled: set = set()
    units: list = []          # each unit: [(prob, {rk: state}), ...]
    for a_spec, b_spec in couples:
        rk_a = next((rk for rk in variable if reskey_str(rk) == a_spec), None)
        rk_b = next((rk for rk in variable if reskey_str(rk) == b_spec), None)
        if rk_a is None or rk_b is None:
            st.warn(f"  --couple {a_spec},{b_spec}: both must be variable "
                    f"residues — ignored")
            continue
        if rk_a == rk_b:
            st.warn(f"  --couple {a_spec},{b_spec}: a residue cannot be "
                    f"coupled to itself — ignored")
            continue
        if rk_a in coupled or rk_b in coupled:
            st.warn(f"  --couple {a_spec},{b_spec}: a residue is already in "
                    f"another couple — ignored")
            continue
        coupled.update((rk_a, rk_b))
        _, pa, da, ppa = variable[rk_a]
        _, pb, db, ppb = variable[rk_b]
        # independent-model joint weights of the two concordant corners — used
        # directly (no renormalisation) so they compose with the other units.
        w_pp, w_dd = ppa * ppb, (1 - ppa) * (1 - ppb)
        units.append([(w_pp, {rk_a: pa, rk_b: pb}),
                      (w_dd, {rk_a: da, rk_b: db})])
        st.note(f"  coupled: {a_spec} + {b_spec} vary together")
    for rk, (rn, prot, deprot, p_prot) in variable.items():
        if rk not in coupled:
            units.append([(p_prot, {rk: prot}), (1 - p_prot, {rk: deprot})])

    if len(units) > max_variable:
        units.sort(key=lambda u: -min(p for p, _ in u))
        dropped = sorted({reskey_str(rk) for u in units[max_variable:]
                          for _, ch in u for rk in ch})
        st.warn(f"  {len(units)} variable units exceed --protomer-max-variable "
                f"{max_variable} — not varying: {', '.join(dropped)}")
        units = units[:max_variable]
    for u in units:
        u.sort(key=lambda x: -x[0])           # best state first

    # ---- best-first top-N over the product ------------------------------
    def joint(idx):
        p = 1.0
        for u, i in zip(units, idx):
            p *= u[i][0]
        return p

    start = tuple(0 for _ in units)
    heap = [(-joint(start), start)]
    seen = {start}
    ranked: list = []
    while heap and len(ranked) < n_protomers:
        negp, idx = heapq.heappop(heap)
        ranked.append((-negp, idx))
        for u in range(len(units)):
            if idx[u] + 1 < len(units[u]):
                nxt = idx[:u] + (idx[u] + 1,) + idx[u + 1:]
                if nxt not in seen:
                    seen.add(nxt)
                    heapq.heappush(heap, (-joint(nxt), nxt))

    # ---- build a structure per protomer ---------------------------------
    protomers: list = []
    for rank, (prob, idx) in enumerate(ranked, start=1):
        assignment: dict = {}
        for u, i in zip(units, idx):
            assignment.update(u[i][1])
        variant = st.copy()
        new_by_res: dict = defaultdict(list)
        var_res = group_residues([a for a in variant.atoms
                                  if a.record == "ATOM"])
        for rk, target in assignment.items():
            atoms = var_res.get(rk, [])
            if not atoms:
                continue
            rn = canonical_resname(atoms[0].resname)
            abn = {a.name: a for a in atoms}
            if _force_residue_state(variant, rk, rn, target, abn, new_by_res):
                variant.residue_records.setdefault(rk, {})["state"] = target
            else:
                st.warn(f"  protomer {rank}: could not set {reskey_str(rk)} "
                        f"-> {target} — residue left unchanged")
        _splice_after_residue(variant, new_by_res)
        protomers.append({"rank": rank, "prob": prob, "state": variant,
                          "assignment": assignment})
        st.note(f"  protomer {rank}: joint p={prob:.3f}  "
                + ", ".join(f"{reskey_str(rk)}={s}" for rk, s
                            in sorted(assignment.items(),
                                      key=lambda x: reskey_str(x[0]))))
    return protomers


# ============================================================================
# Optional MLFF relax of the newly placed hydrogens
# ============================================================================
# elements MACE-OFF covers — anything outside this forces the universal MACE-MP
_ORGANIC_ELEMENTS = {"H", "D", "C", "N", "O", "F", "P", "S", "CL", "BR", "I"}
_RELAX_CALC_CACHE: dict = {}          # (model, device) -> calculator, reused
_RELAX_REPORT_EVERY = 25              # log a progress line every N opt steps
_EV_TO_KCAL_PER_MOL = 23.060548       # ASE energies are in eV
# third-party loggers the MACE/torch import pulls in — silenced during relax
_RELAX_NOISY_LOGGERS = ("matplotlib", "numexpr", "numexpr.utils", "e3nn",
                        "ase", "torch", "PIL")


def _ase_symbol(element: str) -> str:
    """Internal element string -> ASE chemical symbol (D collapses to H)."""
    el = element.strip().capitalize()
    return "H" if el == "D" else el


def stage_relax_h(st: ProtonationState, *, model: str = "mace-off-small",
                  fmax: float = 0.05, steps: int = 200, device: str = "cpu"
                  ) -> ProtonationState:
    """Relax ONLY the script-placed hydrogens with a cheap MLFF (everything
    else frozen). Charge-free; CPU-capable. A non-organic element forces the
    universal MACE-MP. Failures degrade gracefully (the H are left as placed)."""
    banner("MLFF relax of the new hydrogens")
    new_h = [i for i, a in enumerate(st.atoms)
             if a.is_hydrogen and a.tag in ("h", "cap")]
    if not new_h:
        st.note("  no script-placed hydrogens to relax")
        return st
    new_h_set = set(new_h)
    if any(not a.element.strip() for a in st.atoms):
        st.warn("  some atoms have a blank element symbol — MLFF relax skipped")
        return st

    elements = {a.element.upper() for a in st.atoms}
    chosen = model
    if "off" in model.lower() and not elements.issubset(_ORGANIC_ELEMENTS):
        chosen = "mace-mp"
        st.note(f"  non-organic element present — switching {model} → mace-mp "
                f"(charge-free PES)")
    else:
        st.note(f"  using {chosen} (charge-free PES)")

    import time
    import warnings as _warnings
    # silence the noisy third-party loggers/warnings the MACE import drags in
    for _noisy in _RELAX_NOISY_LOGGERS:
        logging.getLogger(_noisy).setLevel(logging.WARNING)
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            from ase import Atoms as _ASEAtoms
            from ase.constraints import FixAtoms
            from ase.optimize import LBFGS
            from quantum_engine.calc.factory import make_calc
    except ImportError as exc:
        st.warn(f"  MLFF relax unavailable ({exc}) — hydrogens left as placed")
        return st

    pre = np.array([a.xyz for a in st.atoms], float)     # pre-relax snapshot
    try:
        ase_atoms = _ASEAtoms(
            symbols=[_ase_symbol(a.element) for a in st.atoms],
            positions=pre.copy())
        cache_key = (chosen, device)
        calc = _RELAX_CALC_CACHE.get(cache_key)
        if calc is None:                  # reuse across protomers
            st.note(f"  loading the {chosen} model (first call only — slow) ...")
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                calc = make_calc(model=chosen, device=device,
                                 default_dtype="float64", charge=None)
            _RELAX_CALC_CACHE[cache_key] = calc
        ase_atoms.calc = calc
        ase_atoms.set_constraint(FixAtoms(
            indices=[i for i in range(len(st.atoms)) if i not in new_h_set]))
        st.note(f"  relaxing {len(new_h)} H ({len(st.atoms)} atoms total, "
                f"device={device}, fmax={fmax}, max {steps} steps) — progress "
                f"every {_RELAX_REPORT_EVERY} steps")

        opt = LBFGS(ase_atoms, logfile=None)
        prog = {"e_last": None, "t0": time.time()}

        def _report():                     # ASE observer — every N opt steps
            step = opt.nsteps
            try:
                energy = float(ase_atoms.get_potential_energy())
            except Exception:              # noqa: BLE001
                return
            now = time.time()
            per_step = (now - prog["t0"]) / max(step, 1)
            eta = per_step * max(steps - step, 0)
            if prog["e_last"] is None:
                d_e = ""
            else:
                de = energy - prog["e_last"]
                d_e = (f"  dE={de:+.4f} eV "
                       f"({de * _EV_TO_KCAL_PER_MOL:+.2f} kcal/mol)")
            log.info(f"  relax step {step:4d}/{steps}  E={energy:.3f} eV{d_e}  "
                     f"{per_step:.2f} s/step  ETA <= {eta:.0f}s")
            prog["e_last"] = energy

        opt.attach(_report, interval=_RELAX_REPORT_EVERY)
        t0 = time.time()
        opt.run(fmax=fmax, steps=steps)
        wall = time.time() - t0
        n_taken = opt.nsteps
    except Exception as exc:               # noqa: BLE001 — degrade gracefully
        st.warn(f"  MLFF relax failed ({type(exc).__name__}: {exc}) — "
                f"hydrogens left as placed")
        return st

    relaxed = ase_atoms.get_positions()
    max_move = max((float(np.linalg.norm(relaxed[i] - pre[i]))
                    for i in new_h), default=0.0)
    heavy_idx = [i for i in range(len(st.atoms)) if i not in new_h_set]
    if heavy_idx:
        hd = np.array([relaxed[i] - pre[i] for i in heavy_idx])
        per_atom = np.sqrt((hd ** 2).sum(axis=1))
        heavy_rmsd = float(np.sqrt((per_atom ** 2).mean()))
        heavy_max = float(per_atom.max())
    else:
        heavy_rmsd = heavy_max = 0.0
    for i in new_h:
        st.atoms[i].xyz = np.array(relaxed[i], float)

    st.settings["relax"] = {"model": chosen, "fmax": fmax, "steps": steps,
                            "steps_taken": n_taken, "n_hydrogens": len(new_h),
                            "max_h_move": round(max_move, 4),
                            "non_h_rmsd": round(heavy_rmsd, 4),
                            "non_h_max_move": round(heavy_max, 4),
                            "wall_seconds": round(wall, 1)}
    st.note(f"  relaxed {len(new_h)} H with {chosen} — {n_taken} steps, "
            f"{wall:.1f}s, max H displacement {max_move:.3f} A")
    if heavy_max < 1e-3:
        st.note(f"  non-H atom RMSD = {heavy_rmsd:.4f} A "
                f"(heavy atoms frozen — verified)")
    else:
        st.warn(f"  non-H atom RMSD = {heavy_rmsd:.4f} A (max {heavy_max:.3f}) "
                f"— heavy atoms moved; FixAtoms is not holding!")
    return st


# ============================================================================
# Charge accounting & output annotations
# ============================================================================
# Formal side-chain charge per protonation-state label.
STATE_CHARGE = {
    "ARG": 1, "ARN": 0, "LYS": 1, "LYN": 0,
    "HIP": 1, "HID": 0, "HIE": 0, "HIS": 0,
    "ASP": -1, "ASH": 0, "GLU": -1, "GLH": 0,
    "CYS": 0, "CYM": -1, "CYX": 0,
    "TYR": 0, "TYM": -1, "SER": 0, "SED": -1, "THR": 0, "THD": -1,
}


def compute_net_charge(st: ProtonationState) -> tuple:
    """Return (net_protein_charge, [per-residue dict]). Net charge = sum of
    side-chain formal charges + cap charges + declared-PTM charges."""
    residues = group_residues([a for a in st.atoms if a.record == "ATOM"])
    net = 0
    per_res: list = []
    for rk, atoms in residues.items():
        rec = st.residue_records.get(rk, {})
        resname = atoms[0].resname
        if rec.get("ptm_frozen"):
            side_q = int(rec.get("ptm_charge", 0))
            state = rec.get("ptm", resname)
        else:
            state = rec.get("state", canonical_resname(resname))
            side_q = STATE_CHARGE.get(state, 0)
            if rec.get("his_metal_bridge"):
                side_q = -1                       # metal-bridging imidazolate
        n_cap, c_cap = rec.get("n_cap"), rec.get("c_cap")
        cap_q = N_CAP_CHARGE.get(n_cap, 0) + C_CAP_CHARGE.get(c_cap, 0)
        net += side_q + cap_q
        per_res.append({
            "residue": reskey_str(rk), "resname": resname, "state": state,
            "side_charge": side_q, "n_cap": n_cap, "c_cap": c_cap,
            "cap_charge": cap_q, "propka_pKa": rec.get("pKa")})
    return net, per_res


def ligand_charge_total(st: ProtonationState, ligand_charges: dict) -> tuple:
    """Sum declared ligand formal charges over HETATM residue INSTANCES.

    ``ligand_charges`` maps RESNAME -> int (case-insensitive). Each non-water
    HETATM residue *instance* whose resname is declared contributes its charge
    ONCE — so two ``ZN`` residues at +2 each give +4, and a multi-atom ligand
    counts once. Waters are skipped. Returns ``(total, per_instance, warnings)``;
    warns on (a) a declared resname not present, and (b) a non-water HETATM
    resname present but not declared (counted as 0 so the user notices).
    Reporting-only — HETATM geometry is never modified.
    """
    lc = {str(k).strip().upper(): int(v) for k, v in (ligand_charges or {}).items()}
    residues = group_residues([a for a in st.atoms if a.record == "HETATM"])
    total = 0
    per_instance: list = []
    seen: set = set()
    undeclared: set = set()
    for rk, atoms in residues.items():
        resname = atoms[0].resname.strip().upper()
        if resname in WATER_RESNAMES:
            continue
        seen.add(resname)
        q = lc.get(resname, 0)
        if resname not in lc:
            undeclared.add(resname)
        total += q
        per_instance.append({"residue": reskey_str(rk), "resname": resname, "charge": q})
    warnings: list = []
    for rn in sorted(lc):
        if rn not in seen:
            warnings.append(f"--ligand-charge {rn}={lc[rn]:+d} given but no HETATM "
                            f"residue {rn!r} is present")
    for rn in sorted(undeclared):
        warnings.append(f"HETATM ligand {rn!r} present with no --ligand-charge; "
                        f"counted as 0 in TOTAL_SYSTEM_CHARGE")
    return total, per_instance, warnings


def _qcb_remarks(version: str, pH: float, net: int, per_res: list,
                 protomer=None, n_protomers: int = 1,
                 total_system: int | None = None) -> list:
    """Minimal REMARK 999 QCB-PROTONATOR annotation block (kept short)."""
    out = [f"v2  pH={pH}  {version}", f"NET_PROTEIN_CHARGE {net:+d}"]
    if total_system is not None:
        out.append(f"TOTAL_SYSTEM_CHARGE {total_system:+d}  (protein + ligands)")
    real = {"peptide", "none", "covalent?", None}
    n_n = sum(1 for r in per_res if r["n_cap"] not in real)
    n_c = sum(1 for r in per_res if r["c_cap"] not in real)
    out.append(f"CAPS  {n_n} N-terminus, {n_c} C-terminus")
    his = [r for r in per_res if r["state"] in ("HID", "HIE", "HIP")]
    if his:
        out.append("HIS  " + "  ".join(f"{r['residue']}={r['state']}"
                                       for r in his))
    if protomer is not None and n_protomers > 1:
        out.append(f"PROTOMER {protomer['rank']}/{n_protomers}  "
                   f"joint_p={protomer['prob']:.3f}")
        if protomer.get("assignment"):
            out.append("VARIES  " + "  ".join(
                f"{reskey_str(rk)}={s}" for rk, s in
                sorted(protomer["assignment"].items(),
                       key=lambda x: reskey_str(x[0]))))
    return out


def _write_info_json(path, inp_path, pH: float, args,
                     protomer_results: list) -> None:
    """Write the optional structured run report (--output-info-file)."""
    import json
    payload = {
        "tool": "protonator v2",
        "input": str(inp_path),
        "pH": pH,
        "n_protomers": len(protomer_results),
        "settings": {
            "n_cap": args.n_cap, "c_cap": args.c_cap,
            "skip_propka": args.skip_propka, "relax_h": args.relax_h,
            "bond_cutoff": args.bond_cutoff,
            "metal_coord_cutoff": args.metal_coord_cutoff,
            "hbond_cutoff": args.hbond_cutoff,
            "clash_cutoff": args.clash_cutoff,
            "h_clash_cutoff": args.h_clash_cutoff,
            "disulfide_cutoff": args.disulfide_cutoff,
        },
        "protomers": [],
    }
    for pr in protomer_results:
        pst = pr["state"]
        payload["protomers"].append({
            "rank": pr["rank"],
            "joint_prob": round(pr["prob"], 6),
            "output_pdb": pr["output"],
            "net_protein_charge": pr["net"],
            "ligand_charge": pr.get("ligand_charge", 0),
            "total_system_charge": pr.get("total_system_charge", pr["net"]),
            "ligand_per_instance": pr.get("ligand_per_instance", []),
            "autocorrect": pst.settings.get("autocorrect"),
            "relax": pst.settings.get("relax"),
            "warnings": list(pst.warnings),
            "decisions": list(pst.decisions),
            "residues": pr["per_res"],
        })
    Path(path).write_text(json.dumps(payload, indent=2, default=str),
                          encoding="utf-8")


# ============================================================================
# CLI / main  (grows with each increment; currently: read → strip H → write)
# ============================================================================
def _parse_kv(items, label: str) -> dict:
    """Parse repeatable ``SPEC=VALUE`` CLI options into a dict."""
    out: dict = {}
    for it in (items or []):
        if "=" not in it:
            raise SystemExit(f"bad --{label} {it!r}: expected SPEC=VALUE")
        key, val = it.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"bad --{label} {it!r}: empty key")
        out[key] = val.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="protonator",
        description="Deterministic, staged protonation of theozyme PDB/CIF.",
    )
    p.add_argument("-i", "--input-pdb", required=True,
                   help="input structure (.pdb or .cif)")
    p.add_argument("-o", "--output-pdb", default=None,
                   help="output PDB (default: <input>_protonated.pdb)")
    p.add_argument("--pH", type=float, default=7.5,
                   help="target pH for canonical protonation (default: 7.5)")
    p.add_argument("--output-info-file", default=None, metavar="PATH",
                   help="optional structured JSON run report (default: off)")
    p.add_argument("--keep-input-hydrogens", action="store_true",
                   help="keep input protein H instead of strip + rebuild")
    p.add_argument("--set", action="append", metavar="CHAIN:RESID=STATE",
                   dest="set_state",
                   help="hard protonation-state override, e.g. A:226=HIP "
                        "(states: HID HIE HIP ASP ASH GLU GLH LYS LYN "
                        "CYS CYM CYX TYR TYM SER SED THR THD ARG ARN; repeatable)")
    # Stage 1 — terminus capping
    p.add_argument("--n-cap", default="nh2", choices=sorted(N_CAP_TYPES),
                   help="default N-terminus cap (default: nh2)")
    p.add_argument("--c-cap", default="cho", choices=sorted(C_CAP_TYPES),
                   help="default C-terminus cap (default: cho)")
    p.add_argument("--n-cap-override", action="append", metavar="CHAIN:RESID=TYPE",
                   help="per-residue N-cap override (repeatable)")
    p.add_argument("--c-cap-override", action="append", metavar="CHAIN:RESID=TYPE",
                   help="per-residue C-cap override (repeatable)")
    p.add_argument("--bond-cutoff", type=float, default=1.8,
                   help="peptide-bond detection distance in A (default: 1.8)")
    # Stage 2 — PTM / covalent checks
    p.add_argument("--ptm", action="append", metavar="CHAIN:RESID=CODE",
                   help="declare a post-translational modification — freezes "
                        "that residue's side-chain protonation (repeatable). "
                        "Spec is CHAIN:RESID, e.g. A:169; use '_' for a blank "
                        "chain and append any insertion code (A:169B)")
    p.add_argument("--ptm-charge", action="append", metavar="CHAIN:RESID=Q",
                   help="formal charge for a declared PTM residue (repeatable)")
    p.add_argument("--ligand-charge", action="append", metavar="RESNAME=Q",
                   help="formal charge of a ligand/metal HETATM residue, for "
                        "TOTAL system-charge reporting (repeatable; e.g. ZN=2 "
                        "PXN=-1). Counted once per residue instance; HETATM "
                        "geometry is left untouched.")
    p.add_argument("--disulfide-cutoff", type=float, default=2.5,
                   help="Cys SG-SG disulfide detection distance (default: 2.5)")
    # Stage 4 — histidine tautomer
    p.add_argument("--metal-coord-cutoff", type=float, default=2.8,
                   help="metal-coordination detection distance (default: 2.8)")
    p.add_argument("--clash-cutoff", type=float, default=2.0,
                   help="His N .. ligand clash distance (default: 2.0)")
    p.add_argument("--hbond-cutoff", type=float, default=3.5,
                   help="H-bond donor..acceptor distance (default: 3.5)")
    p.add_argument("--h-clash-cutoff", type=float, default=1.5,
                   help="placed-H non-bonded clash distance for the geometry "
                        "autocorrection pass (default: 1.5)")
    # Stage 5 — propka refinement
    p.add_argument("--skip-propka", action="store_true",
                   help="skip the propka refinement stage (canonical states "
                        "only; the metal guard still runs)")
    # Stage 6 — protomers
    p.add_argument("--protomers", type=int, default=1,
                   help="number of protomers to output (default: 1)")
    p.add_argument("--protomer-min-prob", type=float, default=0.15,
                   help="a residue varies only if its 2nd-best state's "
                        "marginal probability is >= this (default: 0.15)")
    p.add_argument("--protomer-max-variable", type=int, default=12,
                   help="cap on the number of varying residues (default: 12)")
    p.add_argument("--couple", action="append", metavar="CHAIN:RESID,CHAIN:RESID",
                   help="force a residue pair to vary in lockstep (repeatable)")
    # optional MLFF relax of the new hydrogens
    p.add_argument("--relax-h", action="store_true",
                   help="MLFF-relax only the newly placed hydrogens (CPU)")
    p.add_argument("--relax-h-model", default="mace-off-small",
                   help="MLFF model for --relax-h (default: mace-off-small; "
                        "auto-switches to mace-mp if a metal is present)")
    p.add_argument("--relax-h-fmax", type=float, default=0.05,
                   help="force convergence for --relax-h (default: 0.05)")
    p.add_argument("--relax-h-steps", type=int, default=200,
                   help="max optimiser steps for --relax-h (default: 200)")
    p.add_argument("--relax-h-device", default="cpu",
                   help="device for --relax-h (default: cpu)")
    p.add_argument("--log-level", default="INFO",
                   help="logging level (default: INFO)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.log_level)

    inp = Path(args.input_pdb)
    if not inp.exists():
        log.error(f"input file not found: {inp}")
        return 1
    out = (Path(args.output_pdb) if args.output_pdb
           else inp.with_name(inp.stem + "_protonated.pdb"))

    banner(f"protonator — {inp.name}")
    try:
        st = read_structure(inp)
    except Exception as exc:                # noqa: BLE001 — clean user-facing exit
        log.error(f"could not read input: {exc}")
        return 1

    if args.keep_input_hydrogens:
        log.info("--keep-input-hydrogens set: input hydrogens retained")
    else:
        strip_protein_hydrogens(st)

    stage1_caps(st,
                n_cap=args.n_cap, c_cap=args.c_cap,
                n_cap_overrides=_parse_kv(args.n_cap_override, "n-cap-override"),
                c_cap_overrides=_parse_kv(args.c_cap_override, "c-cap-override"),
                bond_cutoff=args.bond_cutoff)

    stage2_checks(st,
                  ptm=_parse_kv(args.ptm, "ptm"),
                  ptm_charge=_parse_kv(args.ptm_charge, "ptm-charge"),
                  bond_cutoff=args.bond_cutoff,
                  disulfide_cutoff=args.disulfide_cutoff)

    stage3_canonical(st, pH=args.pH,
                     set_overrides=_parse_kv(args.set_state, "set"),
                     bond_cutoff=args.bond_cutoff)

    stage4_his_tautomer(st,
                        metal_coord_cutoff=args.metal_coord_cutoff,
                        clash_cutoff=args.clash_cutoff,
                        hbond_cutoff=args.hbond_cutoff)

    stage5_propka(st, pH=args.pH, skip_propka=args.skip_propka,
                  set_overrides=_parse_kv(args.set_state, "set"),
                  metal_coord_cutoff=args.metal_coord_cutoff)

    # Stage 6 — enumerate protomers (or just the single canonical state)
    couples: list = []
    for spec in (args.couple or []):
        parts = [s.strip() for s in spec.split(",")]
        if len(parts) == 2 and all(parts):
            couples.append((parts[0], parts[1]))
        else:
            log.warning(f"bad --couple {spec!r}: expected "
                        f"CHAIN:RESID,CHAIN:RESID")
    protomers = stage6_protomers(
        st, n_protomers=args.protomers, pH=args.pH,
        min_prob=args.protomer_min_prob,
        max_variable=args.protomer_max_variable, couples=couples,
        set_overrides=_parse_kv(args.set_state, "set"))

    # per protomer: autocorrect (LAST geometry step, also clash-checks Stage
    # 5/6 H), optional MLFF relax, charge accounting, write
    ligand_charges = (_parse_kv(args.ligand_charge, "ligand-charge")
                      if getattr(args, "ligand_charge", None) else {})
    n = len(protomers)
    protomer_results: list = []
    _ligand_warned = False
    for pm in protomers:
        pst = pm["state"]
        stage_autocorrect(pst, h_clash_cutoff=args.h_clash_cutoff)
        if args.relax_h:
            stage_relax_h(pst, model=args.relax_h_model, fmax=args.relax_h_fmax,
                          steps=args.relax_h_steps, device=args.relax_h_device)
        net, per_res = compute_net_charge(pst)
        lig_total, lig_per, lig_warn = ligand_charge_total(pst, ligand_charges)
        if not _ligand_warned:
            for w in lig_warn:
                log.warning(w)
            _ligand_warned = True
        total_system = net + lig_total
        out_path = (out if n == 1 else out.with_name(
            f"{out.stem}_protomer{pm['rank']}{out.suffix or '.pdb'}"))
        remarks = _qcb_remarks(f"input={inp.name}", args.pH, net, per_res,
                               protomer=pm, n_protomers=n,
                               total_system=total_system)
        write_pdb(pst, out_path, remarks=remarks)
        log.info(f"  net protein charge = {net:+d}; ligands = {lig_total:+d}; "
                 f"TOTAL system = {total_system:+d}   ->  {out_path}")
        protomer_results.append({
            "rank": pm["rank"], "prob": pm["prob"], "output": str(out_path),
            "net": net, "ligand_charge": lig_total,
            "total_system_charge": total_system, "ligand_per_instance": lig_per,
            "per_res": per_res, "state": pst})

    if args.output_info_file:
        _write_info_json(args.output_info_file, inp, args.pH, args,
                         protomer_results)
        log.info(f"wrote run report to {args.output_info_file}")

    banner("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
