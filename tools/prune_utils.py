#!/usr/bin/env python
"""prune_utils.py — sidechain/backbone pruning with H-cap placement.

When pruning atoms from a structure (e.g. keep only the LYS169 amine warhead
``CD/CE/NZ`` or strip TRP131 backbone), the cut-bond stubs need to be capped
with hydrogens at chemically sensible positions. This module:

  1. Identifies cut bonds: for each KEPT heavy atom B with a DROPPED neighbor
     N (covalent distance), a hydrogen ("cap H") is placed at
     ``B + normalize(N - B) * d_CH`` where ``d_CH`` defaults to 1.09 Å.
  2. Cap H atoms get a special name ``HPN`` (where N is a per-residue counter,
     e.g. ``HP1``, ``HP2``) so they can never collide with existing PDB atom
     names. They inherit their parent residue's resname/chain/resseq so the
     downstream PDB/CIF writers continue to label them as part of the parent
     residue.
  3. Optionally relaxes the cap H positions with a fast QM (xTB-GFN2 by
     default) holding everything else fixed. Useful for chemically realistic
     starting structures before a full polish.

Public API:
    apply_prune_with_caps(lineage, prune_keep_specs, prune_backbone_residues,
                          *, cap_h_bond=1.09, do_xtb_relax=True,
                          xtb_max_steps=100)
        → (new_lineage, new_coords, list_of_HP_indices_0based)
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from structure_io import PdbAtom, StructureLineage, _guess_element  # noqa: E402

log = logging.getLogger("prune_utils")

BACKBONE_NAMES = {"N", "CA", "C", "O", "OXT", "H", "HA", "H2", "H3", "HXT", "HN"}

# Covalent-bond distance threshold (Å). Pure CHON range; metallic coordination
# bonds (Zn-N/O at 2.0-2.5 Å) are EXCLUDED — we don't cap dropped Zn-ligands
# with an H since that would create a chemically nonsensical fragment.
COVALENT_BOND_CUTOFF = 1.8

# Atom kinds for which capping with H is physical
CAPPABLE_ELEMENTS = {"C", "N", "O", "S", "P"}


# ----- bond detection ------------------------------------------------------
def _coords_array(atoms: list[PdbAtom]) -> np.ndarray:
    return np.array([[a.x, a.y, a.z] for a in atoms])


def _detect_neighbors(coords: np.ndarray,
                      cutoff: float = COVALENT_BOND_CUTOFF) -> dict[int, list[int]]:
    """Return {i: [j, k, ...]} for each atom i, listing j within `cutoff`."""
    n = len(coords)
    neighbors: dict[int, list[int]] = {i: [] for i in range(n)}
    # naive O(N^2) — fine up to ~1000 atoms; switch to KD-tree if it gets slow
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(coords[i] - coords[j]) <= cutoff:
                neighbors[i].append(j)
                neighbors[j].append(i)
    return neighbors


# ----- residue-keep prune w/ caps ------------------------------------------
def apply_prune_with_caps(
    lineage: StructureLineage,
    prune_keep_specs: dict[int, list[str]] | None = None,
    prune_backbone_residues: list[int] | None = None,
    *,
    cap_h_bond: float = 1.09,
    cap_atom_name_prefix: str = "HP",
    do_xtb_relax: bool = False,
    xtb_max_steps: int = 100,
    xtb_charge: int = 0,
    xtb_solvent: str | None = None,
) -> tuple[StructureLineage, np.ndarray, list[int]]:
    """Prune atoms; place cap H's at cut bonds; optionally xTB-relax the caps.

    Args:
        lineage: input parsed PDB.
        prune_keep_specs: {resid: [atom_names_to_keep]}. Heavy atoms NOT in the
            list are dropped; their attached Hs are dropped too. Atoms in
            non-listed residues are unaffected.
        prune_backbone_residues: list of resids — drop backbone N/C/O (CA may
            also be dropped depending on its bond pattern).
        cap_h_bond: bond length of new H-cap (Å). 1.09 is typical C-H.
        cap_atom_name_prefix: 2-3 char prefix; counters appended (HP1, HP2, …).
        do_xtb_relax: if True, run a quick GFN2-xTB opt with all non-cap atoms
            FixAtoms'd to relax cap H positions to chemically sensible values.
        xtb_charge: net molecular charge (used in xTB opt).
        xtb_solvent: ALPB solvent name or None for gas-phase.

    Returns:
        (new_lineage, new_coords, hp_indices_0based) where hp_indices is the
        list of indices in `new_lineage.atoms` corresponding to newly-added
        cap H atoms (so callers can FixAtoms everything else for downstream
        relaxations).
    """
    prune_keep_specs = dict(prune_keep_specs or {})
    prune_backbone_residues = list(prune_backbone_residues or [])
    if not prune_keep_specs and not prune_backbone_residues:
        return lineage, _coords_array(lineage.atoms), []

    coords_full = _coords_array(lineage.atoms)
    neighbors = _detect_neighbors(coords_full, COVALENT_BOND_CUTOFF)

    # ---- Step 1: decide which atoms to KEEP ----
    keep_mask = np.ones(len(lineage.atoms), dtype=bool)
    for i, a in enumerate(lineage.atoms):
        # backbone-prune
        if a.resseq in prune_backbone_residues and a.name in BACKBONE_NAMES:
            keep_mask[i] = False
            continue
        # residue-keep — only applies if this atom's residue is in the spec
        if a.resseq in prune_keep_specs:
            allowed = set(prune_keep_specs[a.resseq])
            # Heavy atom: keep iff name in allowed
            if a.element != "H" and not a.name.startswith("H"):
                if a.name not in allowed:
                    keep_mask[i] = False
                    continue
            else:
                # H atom: keep iff its parent heavy atom is kept. Parent =
                # nearest non-H neighbor.
                heavy_parents = [j for j in neighbors[i]
                                  if lineage.atoms[j].element != "H"
                                  and not lineage.atoms[j].name.startswith("H")]
                if not heavy_parents:
                    keep_mask[i] = False
                    continue
                parent = heavy_parents[0]
                parent_name = lineage.atoms[parent].name
                # If parent is also in this residue and is allowed, keep H
                if (lineage.atoms[parent].resseq == a.resseq
                        and parent_name in allowed):
                    keep_mask[i] = True
                else:
                    keep_mask[i] = False

    keep_indices = list(np.where(keep_mask)[0])
    drop_indices = list(np.where(~keep_mask)[0])
    log.info(f"  prune_with_caps: kept {len(keep_indices)}/{len(lineage.atoms)}, "
             f"dropped {len(drop_indices)}")

    # ---- Step 2: identify cut bonds & place cap H atoms ----
    keep_set = set(keep_indices)
    drop_set = set(drop_indices)
    cap_records: list[tuple[int, int]] = []   # (kept_idx, dropped_neighbor_idx)
    for i in keep_indices:
        a = lineage.atoms[i]
        if a.element not in CAPPABLE_ELEMENTS:
            continue
        for j in neighbors[i]:
            if j in drop_set:
                # only cap if the dropped atom was a heavy atom (skip H neighbors)
                if lineage.atoms[j].element != "H":
                    cap_records.append((i, j))

    log.info(f"  cap H records: {len(cap_records)} (one per cut bond)")

    # ---- Step 3: build new atom list (kept + caps) ----
    new_atoms: list[PdbAtom] = []
    new_coords: list[np.ndarray] = []
    old_to_new: dict[int, int] = {}

    for i in keep_indices:
        old_to_new[i] = len(new_atoms)
        new_atoms.append(lineage.atoms[i])
        new_coords.append(coords_full[i])

    # Per-residue cap counter for atom-name uniqueness
    per_resseq_counter: dict[tuple[str, int], int] = {}
    hp_indices: list[int] = []

    for kept_old_i, dropped_old_j in cap_records:
        kept_a = lineage.atoms[kept_old_i]
        dropped_xyz = coords_full[dropped_old_j]
        kept_xyz = coords_full[kept_old_i]
        # Place H at kept + (dropped - kept).unit * cap_h_bond
        v = dropped_xyz - kept_xyz
        n = np.linalg.norm(v)
        if n < 1e-6:
            continue
        cap_xyz = kept_xyz + (v / n) * cap_h_bond

        key = (kept_a.chain, kept_a.resseq)
        per_resseq_counter[key] = per_resseq_counter.get(key, 0) + 1
        cap_no = per_resseq_counter[key]
        cap_name = f"{cap_atom_name_prefix}{cap_no}"
        # craft a synthetic PDB record for the cap H — use same residue
        # annotation as the parent
        # PDB columns: 1-6 record  7-11 serial  13-16 atom name (with leading
        # space if 1-2 char element, or filled to 4 chars), 17 altloc,
        # 18-20 resname, 22 chain, 23-26 resseq, 27 icode, 31-38 x, 39-46 y,
        # 47-54 z, 55-60 occ, 61-66 bfac, 77-78 element, 79-80 charge
        new_serial = len(new_atoms) + 1   # provisional; renumbered on write
        # Atom name field formatting: H elements left-aligned in cols 13-16
        # but starting at col 14 (so cols 14-16 = name, col 13 = blank)
        an_padded = f"{cap_name:<4s}"
        if len(cap_name) <= 3:
            an_padded = f" {cap_name:<3s}"   # leading space, name in 14-16
        else:
            an_padded = cap_name[:4]
        raw = (
            f"ATOM  {new_serial:>5d} {an_padded[:4]} {kept_a.resname:>3s} "
            f"{kept_a.chain}{kept_a.resseq:>4d}{kept_a.icode:1s}   "
            f"{cap_xyz[0]:>8.3f}{cap_xyz[1]:>8.3f}{cap_xyz[2]:>8.3f}"
            f"  1.00  0.00           H  "
        )
        # Patch column slicing — ensure exact alignment
        line = list(" " * 80)
        line[0:6] = list(f"{'ATOM':<6s}")
        line[6:11] = list(f"{new_serial:>5d}")
        line[11] = " "
        # atom name 12-15 (0-indexed), we want " HP1" in cols 12-15
        if len(cap_name) <= 3:
            line[12] = " "; line[13:16] = list(f"{cap_name:<3s}")
        else:
            line[12:16] = list(f"{cap_name[:4]:<4s}")
        line[16] = " "  # altloc
        line[17:20] = list(f"{kept_a.resname[:3]:>3s}")
        line[20] = " "
        line[21] = kept_a.chain[:1] or " "
        line[22:26] = list(f"{kept_a.resseq:>4d}")
        line[26] = kept_a.icode[:1] or " "
        line[27:30] = list("   ")
        for k, v in enumerate(cap_xyz):
            s = f"{v:>8.3f}"
            line[30 + k * 8 : 38 + k * 8] = list(s)
        line[54:60] = list(f"{1.00:>6.2f}")
        line[60:66] = list(f"{0.00:>6.2f}")
        line[66:76] = list(" " * 10)
        line[76:78] = list(" H")
        line[78:80] = list("  ")
        raw = "".join(line)

        cap_atom = PdbAtom(
            raw=raw,
            serial=new_serial,
            name=cap_name,
            altloc=" ",
            resname=kept_a.resname,
            chain=kept_a.chain,
            resseq=kept_a.resseq,
            icode=kept_a.icode,
            x=float(cap_xyz[0]), y=float(cap_xyz[1]), z=float(cap_xyz[2]),
            occ=1.0, bfac=0.0, element="H", charge_field="",
        )
        hp_indices.append(len(new_atoms))
        new_atoms.append(cap_atom)
        new_coords.append(cap_xyz)

    new_coords_arr = np.array(new_coords)
    log.info(f"  added {len(hp_indices)} cap H atoms (named {cap_atom_name_prefix}1, "
             f"{cap_atom_name_prefix}2, …)")

    new_lineage = StructureLineage(
        atoms=new_atoms,
        raw_remarks=lineage.raw_remarks,
        matcher_remarks=lineage.matcher_remarks,
        qcb_lines=lineage.qcb_lines,
        other_remarks=lineage.other_remarks,
    )

    # ---- Step 4: xTB relax of cap H atoms (others FixAtoms'd) ----
    if do_xtb_relax and len(hp_indices) > 0:
        try:
            new_coords_arr = _xtb_relax_caps(
                new_lineage, new_coords_arr, hp_indices,
                xtb_charge=xtb_charge,
                xtb_solvent=xtb_solvent,
                xtb_max_steps=xtb_max_steps,
            )
            log.info("  cap H xTB relaxation: complete")
        except Exception as e:
            log.warning(f"  cap H xTB relaxation failed: {e}; leaving geometric guesses")

    return new_lineage, new_coords_arr, hp_indices


def _xtb_relax_caps(lineage: StructureLineage,
                    coords: np.ndarray,
                    hp_indices: list[int],
                    xtb_charge: int = 0,
                    xtb_solvent: str | None = None,
                    xtb_max_steps: int = 100,
                    ) -> np.ndarray:
    """Run xtb opt with everything except hp_indices fixed.

    Uses xtb's $fix block. Writes a temporary XYZ + constraints.inp, calls
    xtb, parses the resulting xtbopt.xyz.
    """
    XTB_BIN = REPO / "deps/xtb/install/bin/xtb"
    if not XTB_BIN.exists():
        raise FileNotFoundError(f"xtb binary not found at {XTB_BIN}")

    n = len(coords)
    fix_atoms_1based = [i + 1 for i in range(n) if i not in hp_indices]
    if not fix_atoms_1based:
        log.info("  nothing to fix (all atoms are caps?); skipping xtb relax")
        return coords

    elements = [a.element if a.element else _guess_element(a.name) for a in lineage.atoms]
    work = Path(tempfile.mkdtemp(prefix="prune_xtb_"))
    try:
        # Write input XYZ
        xyz_path = work / "input.xyz"
        with xyz_path.open("w") as fh:
            fh.write(f"{n}\nprune_caps charge={xtb_charge}\n")
            for el, (x, y, z) in zip(elements, coords):
                fh.write(f"{el:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}\n")
        # Constraint file
        with (work / "constraints.inp").open("w") as fh:
            fh.write("$fix\n  atoms: ")
            # compact-range printing
            ranges = []
            start = prev = fix_atoms_1based[0]
            for x in fix_atoms_1based[1:]:
                if x == prev + 1:
                    prev = x; continue
                ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
                start = prev = x
            ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
            fh.write(",".join(ranges) + "\n")
            fh.write("$end\n")
        # Run xtb
        cmd = [str(XTB_BIN), "input.xyz",
               "--gfn", "2", "--opt", "loose",
               "--chrg", str(xtb_charge),
               "--input", "constraints.inp"]
        if xtb_solvent:
            cmd += ["--alpb", xtb_solvent]
        log.info(f"  running xtb cap-relax: {len(hp_indices)} free atoms, "
                 f"{len(fix_atoms_1based)} fixed; max-steps={xtb_max_steps}")
        proc = subprocess.run(cmd, cwd=str(work), capture_output=True, text=True,
                              timeout=600)
        if proc.returncode != 0:
            log.warning(f"  xtb exited {proc.returncode}: {proc.stderr[-400:]}")
            return coords
        opt_xyz = work / "xtbopt.xyz"
        if not opt_xyz.exists():
            return coords
        # Parse xtbopt.xyz
        lines = opt_xyz.read_text().splitlines()
        new_coords = []
        for ln in lines[2 : 2 + n]:
            toks = ln.split()
            new_coords.append([float(toks[1]), float(toks[2]), float(toks[3])])
        return np.array(new_coords)
    finally:
        shutil.rmtree(work, ignore_errors=True)
