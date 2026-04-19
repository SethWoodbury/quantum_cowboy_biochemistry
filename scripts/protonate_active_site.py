#!/usr/bin/env python3
"""
Protonate a protein active site cluster for QM/MLFF calculations.

Strategy:
  1. reduce (Richardson lab) — adds hydrogens with clash-aware placement,
     handles HIS flip optimization, respects metal coordination
  2. propka — predicts pKa of ionizable residues for the given pH
  3. Custom post-processing — neutral termini, user overrides, charge calculation

Only ATOM records are protonated; HETATM (ligand) is left untouched.
Net charge = sum of protein residue charges + user-specified ligand charge.

Usage:
  python protonate_active_site.py input.pdb -o output.pdb --ligand-charge 3 --pH 7.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("protonate")

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

REDUCE_BIN = "/net/software/utils/reduce"

# Standard amino acid formal charges at neutral pH
# Positive: LYS (+1), ARG (+1)
# Negative: ASP (-1), GLU (-1)
# Neutral: HIS (0) by default — protonation depends on pKa & environment
# CYS: 0 by default (can be -1 if deprotonated, but rare in active sites)
STANDARD_CHARGES = {
    "ALA": 0, "ARG": 1, "ASN": 0, "ASP": -1, "CYS": 0,
    "GLN": 0, "GLU": -1, "GLY": 0, "HIS": 0, "ILE": 0,
    "LEU": 0, "LYS": 1, "MET": 0, "PHE": 0, "PRO": 0,
    "SER": 0, "THR": 0, "TRP": 0, "TYR": 0, "VAL": 0,
    # Modified residues
    "KCX": 0,   # Carboxylated lysine: NH charge (+1) + COO (-1) = 0
    "HIP": 1,   # Doubly-protonated histidine
    "HID": 0,   # ND1-protonated histidine
    "HIE": 0,   # NE2-protonated histidine
}


# ══════════════════════════════════════════════════════════════
#  STEP 1: REDUCE — add hydrogens to protein ATOM records
# ══════════════════════════════════════════════════════════════


def _rebuild_missing_backbone(input_pdb: Path) -> Path:
    """Detect and rebuild missing backbone atoms (N, C, O) on partial residues.

    Some theozyme residues only have sidechain + CA (no N, C, O).
    This builds the missing backbone atoms using standard geometry from CA/CB,
    with correct L-amino acid chirality.

    Returns the (possibly modified) PDB path.
    """
    import numpy as np

    lines = Path(input_pdb).read_text().splitlines()

    # Collect atoms per residue
    residues = {}  # {(chain, resnum): {atomname: (line_idx, x, y, z)}}
    resnames = {}
    for i, line in enumerate(lines):
        if not line.startswith("ATOM") or len(line) < 54:
            continue
        aname = line[12:16].strip()
        resname = line[17:20].strip()
        chain = line[21]
        try:
            resnum = int(line[22:26])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        key = (chain, resnum)
        residues.setdefault(key, {})[aname] = (i, x, y, z)
        resnames[key] = resname

    # Check for missing backbone on each residue
    required_backbone = {"N", "CA", "C", "O"}
    new_lines = []
    modified = False

    for key, atoms in residues.items():
        chain, resnum = key
        resname = resnames[key]
        present = set(atoms.keys())
        missing = required_backbone - present

        if not missing:
            continue
        if "CA" not in present:
            # Can't rebuild without CA
            log.warning(f"  Residue {resname} {chain}:{resnum} missing backbone ({missing}) "
                        f"AND CA — cannot rebuild")
            continue

        log.warning(f"  *** PARTIAL RESIDUE: {resname} {chain}:{resnum} "
                    f"missing {sorted(missing)} — REBUILDING ***")
        modified = True

        ca_pos = np.array(atoms["CA"][1:4])

        # Get CB position if available (for chirality)
        cb_pos = None
        if "CB" in atoms:
            cb_pos = np.array(atoms["CB"][1:4])

        # Build missing atoms using standard geometry
        # Standard L-amino acid: N-CA-C angle ~111°, CA-N ~1.47A, CA-C ~1.52A
        # N is roughly opposite to CB from CA; C is roughly between N and CB

        if cb_pos is not None:
            ca_cb = cb_pos - ca_pos
            ca_cb_norm = ca_cb / np.linalg.norm(ca_cb)

            # Build a proper tetrahedral frame from CA with CB as one arm.
            # For L-amino acid: N, C, H are the other 3 tetrahedral arms.
            # Tetrahedral angle = 109.47°, cos(109.47°) = -1/3
            #
            # Strategy: place N and C at proper tetrahedral angles from CB,
            # with L-chirality (right-handed cross product N×C · CB > 0).
            arb = np.array([1, 0, 0]) if abs(ca_cb_norm[0]) < 0.9 else np.array([0, 1, 0])
            perp1 = np.cross(ca_cb_norm, arb)
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(ca_cb_norm, perp1)

            # Tetrahedral directions (relative to CB direction):
            # The 3 other arms of a tetrahedron at angle 109.47° from CB
            # cos(109.47°) ≈ -0.333, sin(109.47°) ≈ 0.943
            tet_n = -0.333 * ca_cb_norm + 0.943 * (np.cos(0) * perp1 + np.sin(0) * perp2)
            tet_c = -0.333 * ca_cb_norm + 0.943 * (np.cos(2.094) * perp1 + np.sin(2.094) * perp2)
            # (2.094 rad = 120° rotation around CB axis)

            if "N" in missing:
                n_dir = tet_n / np.linalg.norm(tet_n)
                n_pos = ca_pos + n_dir * 1.47
                new_lines.append(_make_atom_line(resname, chain, resnum, "N", n_pos, "N"))

            if "C" in missing:
                c_dir = tet_c / np.linalg.norm(tet_c)
                c_pos = ca_pos + c_dir * 1.52
                new_lines.append(_make_atom_line(resname, chain, resnum, "C", c_pos, "C"))

                if "O" in missing:
                    # O on C: sp2 carbonyl, ~120° from CA-C bond.
                    # Place O in the N-CA-C plane, on the opposite side from N.
                    ca_c_dir = (c_pos - ca_pos) / np.linalg.norm(c_pos - ca_pos)
                    if "N" not in missing:
                        # Use N position for plane definition
                        n_pos_ref = n_pos if "N" in locals() else ca_pos + tet_n * 1.47
                    else:
                        n_pos_ref = ca_pos + tet_n * 1.47
                    ca_n_dir = (n_pos_ref - ca_pos) / np.linalg.norm(n_pos_ref - ca_pos)
                    # Normal to the N-CA-C plane
                    plane_normal = np.cross(ca_n_dir, ca_c_dir)
                    plane_normal = plane_normal / np.linalg.norm(plane_normal)
                    # O direction: in the N-CA-C plane, 120° from CA-C toward N side
                    in_plane_perp = np.cross(ca_c_dir, plane_normal)
                    in_plane_perp = in_plane_perp / np.linalg.norm(in_plane_perp)
                    # 120° from CA means: -cos(120°)*along_CA + sin(120°)*perp
                    o_dir = 0.5 * ca_c_dir + 0.866 * in_plane_perp
                    o_dir = o_dir / np.linalg.norm(o_dir)
                    o_pos = c_pos + o_dir * 1.23
                    new_lines.append(_make_atom_line(resname, chain, resnum, "O", o_pos, "O"))
        else:
            # No CB — place backbone in arbitrary but reasonable geometry
            if "N" in missing:
                n_pos = ca_pos + np.array([1.47, 0, 0])
                new_lines.append(_make_atom_line(resname, chain, resnum, "N", n_pos, "N"))
            if "C" in missing:
                c_pos = ca_pos + np.array([-0.76, -1.32, 0])
                new_lines.append(_make_atom_line(resname, chain, resnum, "C", c_pos, "C"))
                if "O" in missing:
                    o_pos = c_pos + np.array([-1.07, -0.62, 0])
                    new_lines.append(_make_atom_line(resname, chain, resnum, "O", o_pos, "O"))

        log.warning(f"    Added {len([m for m in missing])} backbone atom(s): {sorted(missing)}")
        log.warning(f"    *** These will be relaxed by xTB — verify geometry! ***")

    if not modified:
        return input_pdb

    # Insert new backbone atoms into the PDB (after the last ATOM of each residue)
    # Find the last ATOM line index for modified residues
    out_lines = list(lines)
    # Sort new_lines by the residue they belong to
    for new_line in reversed(new_lines):
        # Find the residue in the PDB and insert after the last atom of that residue
        resname = new_line[17:20].strip()
        chain = new_line[21]
        resnum = int(new_line[22:26])
        key = (chain, resnum)
        if key in residues:
            last_idx = max(atoms[0] for atoms in residues[key].values())
            out_lines.insert(last_idx + 1, new_line)

    rebuilt_pdb = Path(str(input_pdb) + ".rebuilt.pdb")
    rebuilt_pdb.write_text("\n".join(out_lines) + "\n")
    log.info(f"  Rebuilt backbone written to {rebuilt_pdb}")

    # Store rebuilt atom info globally so xTB knows not to freeze them
    global _REBUILT_ATOMS
    _REBUILT_ATOMS = set()
    for line in new_lines:
        aname = line[12:16].strip()
        chain = line[21]
        resnum = int(line[22:26])
        _REBUILT_ATOMS.add((chain, resnum, aname))
    log.info(f"  {len(_REBUILT_ATOMS)} rebuilt atoms will be FREE during xTB relaxation")

    return rebuilt_pdb

_REBUILT_ATOMS = set()  # module-level storage


def _make_atom_line(resname, chain, resnum, atomname, pos, element):
    """Create a PDB ATOM line."""
    import numpy as np
    return (
        f"ATOM  {0:>5d} {atomname:>4s} {resname:>3s} {chain}{resnum:>4d}    "
        f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
        f"  1.00  0.00           {element:>2s}  "
    )


def run_reduce_for_diagnostics(input_pdb: Path) -> tuple[str, Path]:
    """Run reduce on the input PDB to get metal-coordination diagnostics.

    reduce is NOT used for the actual hydrogen addition (pdbfixer does that),
    but it's the best tool for detecting which atoms coordinate metals
    (via its "NoAdj-H" clash detection).

    Returns:
        (reduce_output_pdb_path, diagnostics_string)
    """
    if not os.path.isfile(REDUCE_BIN):
        log.warning(f"reduce not found at {REDUCE_BIN} — skipping metal coordination detection")
        # Write a copy so parse_reduce_metal_coordination has something to read
        tmp = Path(str(input_pdb) + ".reduce_tmp.pdb")
        tmp.write_text(Path(input_pdb).read_text())
        return "", tmp

    args = [REDUCE_BIN, "-Quiet", "-FLIP", str(input_pdb)]
    result = subprocess.run(args, capture_output=True, text=True, timeout=60)

    # Write reduce output to temp file (for parsing USER MOD lines)
    tmp = Path(str(input_pdb) + ".reduce_tmp.pdb")
    tmp.write_text(result.stdout)

    return result.stderr, tmp


def run_pdbfixer(input_pdb: Path, output_pdb: Path, pH: float = 7.0):
    """Use PDBFixer (OpenMM) to add ALL missing hydrogens with proper PDB names.

    pdbfixer knows full amino acid templates, so it adds every hydrogen:
    HA on CA, HB2/HB3 on CB, aromatic HD2 on HIS, etc.

    Only adds H to ATOM records; HETATM lines are preserved from the original.
    """
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    fixer = PDBFixer(filename=str(input_pdb))
    fixer.addMissingHydrogens(pH)

    # Write pdbfixer output to temp file
    tmp_pdbfixer = Path(str(output_pdb) + ".pdbfixer_tmp.pdb")
    with open(tmp_pdbfixer, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    # Merge: take ATOM lines from pdbfixer, HETATM + headers from original
    original_lines = Path(input_pdb).read_text().splitlines()
    header_lines = [l for l in original_lines
                    if not l.startswith(("ATOM", "HETATM", "TER", "END", "CONECT"))]
    hetatm_lines = [l for l in original_lines if l.startswith("HETATM")]

    pdbfixer_lines = Path(tmp_pdbfixer).read_text().splitlines()
    atom_lines = [l for l in pdbfixer_lines if l.startswith("ATOM")]

    out = []
    out.extend(header_lines)
    out.extend(atom_lines)
    out.append("TER")
    out.extend(hetatm_lines)
    if hetatm_lines:
        out.append("TER")
    out.append("END")

    Path(output_pdb).write_text("\n".join(out) + "\n")

    # Cleanup
    tmp_pdbfixer.unlink(missing_ok=True)

    n_orig = sum(1 for l in original_lines if l.startswith("ATOM"))
    n_new = len(atom_lines)
    log.info(f"  pdbfixer: {n_orig} → {n_new} ATOM lines ({n_new - n_orig} H added)")
    log.info(f"  HETATM preserved: {len(hetatm_lines)} lines (untouched)")


# ══════════════════════════════════════════════════════════════
#  STEP 1d: POST-PROCESSING — fix terminals, cross-residue bonds
# ══════════════════════════════════════════════════════════════


def postprocess_hydrogens(pdb_path: Path, neutral_termini: bool = True) -> dict:
    """Fix common issues with pdbfixer's hydrogen placement.

    1. N-terminal: remove extra H to make NH2 (neutral) instead of NH3+ (charged)
    2. C-terminal: add aldehyde H to bare C=O (becomes C(=O)H)
    3. Cross-residue covalent bonds: detect ATOM NZ within 1.6A of HETATM C
       and remove excess H on NZ (e.g., carbamylated lysine: NH not NH3+)

    Returns dict with cross-residue bond info for charge correction:
      {'covalent_bonds': {(chain, resnum): {'atom': str, 'ligand_atom': str}}}
    """
    lines = Path(pdb_path).read_text().splitlines()
    covalent_bonds = {}  # {(chain, resnum): {'atom': str, ...}}

    # Parse atom positions for distance checks
    atom_records = []  # (line_idx, record_type, chain, resnum, resname, atomname, x, y, z)
    for i, line in enumerate(lines):
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
            rec = line[:6].strip()
            atomname = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21]
            try:
                resnum = int(line[22:26])
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            atom_records.append((i, rec, chain, resnum, resname, atomname, x, y, z))

    # Identify fragment termini — NOT just first/last residue per chain,
    # but every residue that has a gap in numbering (non-consecutive).
    # E.g., residues 55,56,57,131,132,169 → fragments [55-57], [131-132], [169]
    # Each fragment start is an N-terminus, each fragment end is a C-terminus.
    chain_residues = {}
    for _, rec, chain, resnum, resname, aname, *_ in atom_records:
        if rec == "ATOM":
            chain_residues.setdefault(chain, set()).add(resnum)

    terminal_n = set()
    terminal_c = set()
    for chain, resnums in chain_residues.items():
        srt = sorted(resnums)
        if not srt:
            continue
        # Walk through sorted residue numbers and find gaps
        terminal_n.add((chain, srt[0]))  # first residue is always N-terminal
        for i in range(1, len(srt)):
            if srt[i] - srt[i - 1] > 1:
                # Gap found: previous residue is C-terminal, current is N-terminal
                terminal_c.add((chain, srt[i - 1]))
                terminal_n.add((chain, srt[i]))
        terminal_c.add((chain, srt[-1]))  # last residue is always C-terminal

    log.info(f"  Fragment termini: {len(terminal_n)} N-term, {len(terminal_c)} C-term")

    lines_to_remove = set()
    lines_to_add = []  # (insert_after_idx, new_line)

    # ── Fix 1: Neutral N-termini → ensure exactly 2 H on N (NH2) ──
    if neutral_termini:
        for chain_resnum in terminal_n:
            chain, resnum = chain_resnum
            # Count existing H atoms on N for this residue
            n_idx = None
            h_on_n = []
            for i, rec, ch, rn, resname, aname, x, y, z in atom_records:
                if rec == "ATOM" and ch == chain and rn == resnum:
                    if aname == "N":
                        n_idx = i
                    elif aname in ("H", "H1", "H2", "H3", "HN"):
                        h_on_n.append((i, aname, x, y, z))

            if len(h_on_n) == 3:
                # NH3+ → remove H3 to get NH2
                for i, aname, *_ in h_on_n:
                    if aname == "H3":
                        lines_to_remove.add(i)
                        resname_str = next((rn for _, _, ch, rn, rname, _, *_ in atom_records if ch == chain and rn == resnum), "?")
                        log.info(f"  Terminal fix: removed H3 on {chain}:{resnum} (NH3+ → NH2)")
            elif len(h_on_n) == 1:
                # Only 1 H (peptide-like) → add a second H to make NH2
                # Place H2 using N, CA, and existing H geometry
                import numpy as np
                n_pos = None
                ca_pos = None
                h1_pos = None
                for _, rec, ch, rn, _, aname, x, y, z in atom_records:
                    if rec == "ATOM" and ch == chain and rn == resnum:
                        if aname == "N":
                            n_pos = np.array([x, y, z])
                        elif aname == "CA":
                            ca_pos = np.array([x, y, z])
                if n_pos is not None and ca_pos is not None and h_on_n:
                    h1_pos = np.array(h_on_n[0][2:5])
                    # Place H2 to make a tetrahedral-like NH2:
                    # H2 direction = mirror H1 across the N-CA axis
                    ca_dir = (ca_pos - n_pos) / np.linalg.norm(ca_pos - n_pos)
                    h1_dir = h1_pos - n_pos
                    # Project h1 onto ca_dir and reflect
                    proj = np.dot(h1_dir, ca_dir) * ca_dir
                    h2_dir = 2 * proj - h1_dir
                    h2_dir = h2_dir / np.linalg.norm(h2_dir) * 1.01
                    h2_pos = n_pos + h2_dir
                    resname_str = next((rname for _, _, ch, rn, rname, _, *_ in atom_records if ch == chain and rn == resnum), "UNK")
                    h_line = (
                        f"ATOM  {0:>5d}  H2  {resname_str:>3s} {chain}{resnum:>4d}    "
                        f"{h2_pos[0]:>8.3f}{h2_pos[1]:>8.3f}{h2_pos[2]:>8.3f}"
                        f"  1.00  0.00           H  "
                    )
                    lines_to_add.append((n_idx, h_line))
                    log.info(f"  Terminal fix: added H2 on {chain}:{resnum} (NH → NH2)")
            elif len(h_on_n) == 0 and n_idx is not None:
                log.warning(f"  Terminal: {chain}:{resnum} N has no H — skipping")

    # ── Fix 2: C-terminal aldehyde H (C=O → C(=O)H) ──
    for i, rec, chain, resnum, resname, aname, x, y, z in atom_records:
        if rec == "ATOM" and (chain, resnum) in terminal_c and aname == "C":
            # Find the O on the same residue
            o_pos = None
            for j, rec2, ch2, rn2, _, an2, ox, oy, oz in atom_records:
                if rec2 == "ATOM" and ch2 == chain and rn2 == resnum and an2 == "O":
                    o_pos = (ox, oy, oz)
                    break

            if o_pos is not None:
                import numpy as np
                c_pos = np.array([x, y, z])
                o_vec = np.array(o_pos) - c_pos
                o_dist = np.linalg.norm(o_vec)
                # Place aldehyde H opposite to O, ~1.1A from C, in the C-CA-O plane
                # Find CA
                ca_pos = None
                for j, rec2, ch2, rn2, _, an2, cx, cy, cz in atom_records:
                    if rec2 == "ATOM" and ch2 == chain and rn2 == resnum and an2 == "CA":
                        ca_pos = np.array([cx, cy, cz])
                        break

                if ca_pos is not None:
                    # sp2 aldehyde: H is the third arm at ~120° from both CA and O
                    # Direction = -(CA_unit + O_unit) gives the vector opposite
                    # to the bisector of CA-C-O, which is the sp2 third arm
                    ca_unit = (ca_pos - c_pos) / np.linalg.norm(ca_pos - c_pos)
                    o_unit = o_vec / o_dist
                    h_dir = -(ca_unit + o_unit)
                    h_dir = h_dir / np.linalg.norm(h_dir)
                    h_pos = c_pos + h_dir * 1.10  # C-H bond ~1.1 A

                    # Format PDB ATOM line
                    h_line = (
                        f"ATOM  {0:>5d}  HXT {resname:>3s} {chain}{resnum:>4d}    "
                        f"{h_pos[0]:>8.3f}{h_pos[1]:>8.3f}{h_pos[2]:>8.3f}"
                        f"  1.00  0.00           H  "
                    )
                    lines_to_add.append((i, h_line))
                    log.info(f"  Terminal fix: added aldehyde H (HXT) on C-terminus {resname} {chain}:{resnum}")

    # ── Fix 3: Cross-residue covalent bonds (carbamylated LYS etc.) ──
    # Find ATOM nitrogen atoms within 1.6A of any HETATM carbon
    import math
    atom_n_records = [(i, chain, resnum, resname, aname, x, y, z)
                      for i, rec, chain, resnum, resname, aname, x, y, z in atom_records
                      if rec == "ATOM" and aname in ("NZ", "NE2", "ND1", "SG")]
    hetatm_c_records = [(i, chain, resnum, resname, aname, x, y, z)
                        for i, rec, chain, resnum, resname, aname, x, y, z in atom_records
                        if rec == "HETATM" and aname.startswith("C")]

    for ai, ach, arn, arsn, aan, ax, ay, az in atom_n_records:
        for hi, hch, hrn, hrsn, han, hx, hy, hz in hetatm_c_records:
            d = math.sqrt((ax-hx)**2 + (ay-hy)**2 + (az-hz)**2)
            if d < 1.65:  # covalent bond distance
                log.info(
                    f"  Cross-residue bond: {arsn} {ach}:{arn} {aan} → "
                    f"{hrsn} {hch}:{hrn} {han} ({d:.2f} A)"
                )
                # This atom is covalently bonded to the ligand
                # Track for charge correction
                covalent_bonds[(ach, arn)] = {
                    "atom": aan, "resn": arsn, "ligand_atom": han,
                }
                # Remove excess H: NZ should have at most 1 H (NH), not 3 (NH3+)
                if aan == "NZ":
                    # Find all HZ* atoms on this residue and keep only HZ1
                    hz_indices = [
                        idx for idx, rec2, ch2, rn2, _, an2, *_ in atom_records
                        if rec2 == "ATOM" and ch2 == ach and rn2 == arn
                        and an2.startswith("HZ") and an2 != "HZ1"
                    ]
                    for hz_idx in hz_indices:
                        lines_to_remove.add(hz_idx)
                    if hz_indices:
                        log.info(
                            f"    Removed {len(hz_indices)} excess H on {aan} "
                            f"(covalent bond to ligand → NH, not NH3+)"
                        )

    # Apply removals and additions
    lines_to_add.sort(key=lambda x: x[0], reverse=True)

    new_lines = []
    for i, line in enumerate(lines):
        if i not in lines_to_remove:
            new_lines.append(line)
        for insert_after, new_line in lines_to_add:
            if i == insert_after:
                new_lines.append(new_line)

    # Renumber all ATOM/HETATM serial numbers sequentially
    renumbered = []
    serial = 1
    for line in new_lines:
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
            line = line[:6] + f"{serial:>5d}" + line[11:]
            serial += 1
        renumbered.append(line)

    Path(pdb_path).write_text("\n".join(renumbered) + "\n")

    return {"covalent_bonds": covalent_bonds}


def relax_hydrogens_xtb(pdb_path: Path, charge: int = 0):
    """Optionally relax hydrogen positions using xTB (GFN2, ~seconds on CPU).

    Fixes sp2/sp3 geometry issues by optimizing H positions while freezing
    all heavy atoms. Extremely cheap — seconds for 100-atom systems.
    """
    XTB_BIN = "/home/dme5188/bin/xtb/xtb-6.6.1/bin/xtb"
    if not os.path.isfile(XTB_BIN):
        log.warning(f"xTB not found at {XTB_BIN} — skipping H relaxation")
        return

    import numpy as np

    # Read PDB and extract ALL atoms (ATOM + HETATM) for xTB
    all_lines = Path(pdb_path).read_text().splitlines()
    coord_lines = [l for l in all_lines if l.startswith(("ATOM", "HETATM")) and len(l) >= 54]

    elements = []
    positions = []
    heavy_indices = []  # 1-indexed for xTB constraint file

    global _REBUILT_ATOMS
    n_freed = 0

    for idx, line in enumerate(coord_lines):
        el = line[76:78].strip()
        aname = line[12:16].strip()
        if not el:
            el = aname[0] if aname[0].isalpha() else aname[1]
        elements.append(el)
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        positions.append((x, y, z))

        if el != "H":
            # Check if this is a rebuilt atom — if so, let it be FREE in xTB
            chain = line[21] if len(line) > 21 else ""
            try:
                resnum = int(line[22:26])
            except (ValueError, IndexError):
                resnum = 0
            if (chain, resnum, aname) in _REBUILT_ATOMS:
                n_freed += 1
                # Don't add to heavy_indices → will be free to move
            else:
                heavy_indices.append(idx + 1)  # 1-indexed, frozen
        # H atoms are always free (not in heavy_indices)

    if n_freed > 0:
        log.info(f"  {n_freed} rebuilt backbone atoms will be FREE during xTB")

    if not heavy_indices:
        return

    # Write XYZ for xTB
    tmpdir = Path(pdb_path).parent / ".xtb_tmp"
    tmpdir.mkdir(exist_ok=True)

    xyz_file = tmpdir / "input.xyz"
    with open(xyz_file, "w") as f:
        f.write(f"{len(elements)}\n")
        f.write(f"charge={charge}\n")
        for el, (x, y, z) in zip(elements, positions):
            f.write(f"{el:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")

    # Write constraint file (fix all heavy atoms)
    fix_file = tmpdir / "fix.inp"
    # Format: $fix atoms: 1,2,3,...  $end
    idx_str = ",".join(str(i) for i in heavy_indices)
    with open(fix_file, "w") as f:
        f.write(f"$fix\n  atoms: {idx_str}\n$end\n")

    # Run xTB optimization (use relative paths from tmpdir)
    log.info(f"  xTB H relaxation: {len(elements)} atoms, {len(heavy_indices)} frozen ...")
    result = subprocess.run(
        [XTB_BIN, "input.xyz", "--opt", "tight", "--input", "fix.inp",
         "--gfn", "2", "--chrg", str(charge)],
        capture_output=True, text=True, timeout=300,
        cwd=str(tmpdir),
    )

    opt_xyz = tmpdir / "xtbopt.xyz"
    if not opt_xyz.exists():
        log.warning(f"  xTB optimization failed.")
        # Show last few lines of stderr for debugging
        for line in result.stderr.splitlines()[-5:]:
            log.warning(f"    xTB: {line}")
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        return

    # Read optimized positions
    opt_lines = opt_xyz.read_text().splitlines()
    new_positions = []
    for line in opt_lines[2:]:  # skip header
        parts = line.split()
        if len(parts) >= 4:
            new_positions.append((float(parts[1]), float(parts[2]), float(parts[3])))

    if len(new_positions) != len(positions):
        log.warning("  xTB output atom count mismatch — skipping")
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        return

    # Update ONLY hydrogen positions in the PDB
    coord_idx = 0
    new_pdb_lines = []
    for line in all_lines:
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
            el = line[76:78].strip()
            if not el:
                aname = line[12:16].strip()
                el = aname[0] if aname[0].isalpha() else aname[1]

            if el == "H" and coord_idx < len(new_positions):
                x, y, z = new_positions[coord_idx]
                line = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]

            coord_idx += 1
            new_pdb_lines.append(line)
        else:
            new_pdb_lines.append(line)

    Path(pdb_path).write_text("\n".join(new_pdb_lines) + "\n")
    log.info("  xTB H relaxation done (sp2/sp3 geometry fixed)")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════
#  STEP 1b: PARSE REDUCE DIAGNOSTICS — metal coordination
# ══════════════════════════════════════════════════════════════


def parse_reduce_metal_coordination(protonated_pdb: Path) -> dict:
    """Parse reduce's USER MOD lines to identify metal-coordinating residues.

    reduce marks residues whose H atoms would clash with metals as "NoAdj-H".
    For example:
      USER  MOD NoAdj-H: A   1 HIS HE2 : A   1 HIS NE2 : Z   9 YYE ZN1 :(H bumps)

    This means HIS A:1 NE2 coordinates to ZN1 and should NOT be protonated there.
    A HIS coordinating a metal via NE2 cannot be doubly-protonated (HIP), so it
    should be forced to neutral (0) regardless of what propka says.

    Returns:
        Dict of {(chain, resnum): {'resn': str, 'atom': str, 'metal': str, 'reason': str}}
    """
    metal_coord = {}

    with open(protonated_pdb) as f:
        for line in f:
            if "NoAdj-H" not in line:
                continue

            # Parse: "USER  MOD NoAdj-H: A   1 HIS HE2 : A   1 HIS NE2 : Z   9 YYE ZN1 :(H bumps)"
            parts = line.split(":")
            if len(parts) < 4:
                continue

            # The residue info is in the second segment (the coordinating atom)
            coord_part = parts[2].strip()  # "A   1 HIS NE2"
            metal_part = parts[3].strip()  # "Z   9 YYE ZN1"

            # Parse residue
            tokens = coord_part.split()
            if len(tokens) >= 4:
                chain = tokens[0]
                try:
                    resnum = int(tokens[1])
                except ValueError:
                    continue
                resn = tokens[2]
                coord_atom = tokens[3]

                # Parse metal
                metal_tokens = metal_part.split()
                metal_name = metal_tokens[3] if len(metal_tokens) >= 4 else "metal"

                key = (chain, resnum)
                metal_coord[key] = {
                    "resn": resn,
                    "coord_atom": coord_atom,
                    "metal": metal_name,
                    "reason": f"NE2/ND1 coordinates {metal_name} (reduce NoAdj-H)",
                }
                log.info(
                    f"  Metal coordination: {resn} {chain}:{resnum} "
                    f"{coord_atom} → {metal_name} (cannot be doubly-protonated)"
                )

    return metal_coord


# ══════════════════════════════════════════════════════════════
#  STEP 2: PROPKA — predict pKa values
# ══════════════════════════════════════════════════════════════


def run_propka(pdb_path: Path, pH: float = 7.0) -> dict:
    """Run propka to predict pKa of ionizable residues.

    Returns dict of {(chain_id, res_num): {'pKa': float, 'resn': str, 'protonated': bool}}
    """
    try:
        import propka.input
        import propka.lib
        import propka.molecular_container
        import propka.parameters
    except ImportError:
        log.warning("propka not available — using default protonation states")
        return {}

    pdb_path = Path(pdb_path).resolve()

    try:
        propka_opts = propka.lib.loadOptions([str(pdb_path)])
        propka_params = propka.input.read_parameter_file(
            propka_opts.parameters, propka.parameters.Parameters()
        )
        propka_molc = propka.molecular_container.MolecularContainer(propka_params, propka_opts)
        propka.input.read_molecule_file(propka_opts.input_pdb, propka_molc)
        propka_molc.calculate_pka()
    except Exception as e:
        log.warning(f"propka failed: {e} — using default protonation states")
        return {}

    resis_dict = {}
    for group in propka_molc.conformations["AVR"].groups:
        pka_value = group.pka_value
        resis_dict[(group.atom.chain_id, group.atom.res_num)] = {
            "pKa": pka_value,
            "resn": group.atom.res_name,
            "protonated": (pka_value >= pH) if pka_value is not None else None,
        }

    return resis_dict


# ══════════════════════════════════════════════════════════════
#  STEP 3: POST-PROCESSING — termini, overrides, charge calc
# ══════════════════════════════════════════════════════════════


def determine_residue_charges(
    pdb_path: Path,
    pH: float = 7.0,
    neutral_termini: bool = True,
    user_overrides: dict | None = None,
    pka_dict: dict | None = None,
    metal_coord: dict | None = None,
    covalent_bonds: dict | None = None,
) -> dict:
    """Determine the formal charge of each protein residue.

    Args:
        pdb_path: PDB file to analyze
        pH: pH for protonation state decisions
        neutral_termini: If True, terminal amines/carboxyls are neutral (capped model)
        user_overrides: Dict of {(chain, resnum): charge} for manual overrides
        pka_dict: Pre-computed propka results (or None to run propka)
        metal_coord: Dict from parse_reduce_metal_coordination() — residues
                     coordinating metals are forced to neutral charge
        covalent_bonds: Dict from postprocess_hydrogens() — residues with
                        atoms covalently bonded to ligand (e.g. carbamylated LYS)

    Returns:
        Dict of {(chain, resnum): {'resn': str, 'charge': int, 'reason': str}}
    """
    if pka_dict is None:
        pka_dict = run_propka(pdb_path, pH=pH)

    # Parse residues from PDB
    residues = {}
    terminal_n = set()  # (chain, resnum) of N-terminal residues
    terminal_c = set()  # (chain, resnum) of C-terminal residues

    residue_atoms = {}  # {(chain, resnum): set of atom names}

    with open(pdb_path) as f:
        chain_residues = {}  # {chain: [resnum, ...]}
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            resnum = int(line[22:26])
            resn = line[17:20].strip()
            aname = line[12:16].strip()
            key = (chain, resnum)
            if key not in residues:
                residues[key] = resn
            residue_atoms.setdefault(key, set()).add(aname)
            if chain not in chain_residues:
                chain_residues[chain] = []
            if resnum not in chain_residues[chain]:
                chain_residues[chain].append(resnum)

    # Identify fragment termini (same logic as postprocess_hydrogens)
    for chain, resnums in chain_residues.items():
        resnums_sorted = sorted(set(resnums))
        if not resnums_sorted:
            continue
        terminal_n.add((chain, resnums_sorted[0]))
        for i in range(1, len(resnums_sorted)):
            if resnums_sorted[i] - resnums_sorted[i - 1] > 1:
                terminal_c.add((chain, resnums_sorted[i - 1]))
                terminal_n.add((chain, resnums_sorted[i]))
            terminal_c.add((chain, resnums_sorted[-1]))

    # Assign charges
    charges = {}
    for key, resn in residues.items():
        chain, resnum = key

        # Check user override first
        if user_overrides and key in user_overrides:
            charges[key] = {
                "resn": resn,
                "charge": user_overrides[key],
                "reason": "user override",
            }
            continue

        # Start with standard charge
        base_charge = STANDARD_CHARGES.get(resn, 0)
        reason = "standard"

        # Check for pre-existing H on carboxylate oxygens (ASP/GLU).
        # If HD2/HE2 is already present in the PDB, the residue was intentionally
        # protonated — override propka and treat as neutral.
        atoms_in_res = residue_atoms.get(key, set())
        if resn == "ASP" and any(a in atoms_in_res for a in ("HD2", "HD1")):
            base_charge = 0
            reason = "PRE-EXISTING H on carboxylate → protonated (neutral, not -1)"
        elif resn == "GLU" and any(a in atoms_in_res for a in ("HE2", "HE1")):
            base_charge = 0
            reason = "PRE-EXISTING H on carboxylate → protonated (neutral, not -1)"

        # Check propka for pKa-shifted protonation (only if not already overridden)
        elif key in pka_dict:
            pka_info = pka_dict[key]
            pka_val = pka_info["pKa"]

            if resn in ("ASP", "GLU"):
                # Normally deprotonated (charge -1). Protonated if pKa > pH
                if pka_val is not None and pka_val > pH:
                    base_charge = 0  # protonated = neutral
                    reason = f"propka pKa={pka_val:.1f} > pH={pH} → protonated (neutral)"
                else:
                    reason = f"propka pKa={pka_val:.1f} → deprotonated (-1)"

            elif resn == "HIS":
                # Normally neutral. Doubly-protonated (+1) if pKa > pH
                if pka_val is not None and pka_val > pH:
                    # Check if both HD1 and HE2 are actually present.
                    # If pdbfixer couldn't place HD1 (due to clash), it's singly protonated.
                    has_hd1 = "HD1" in atoms_in_res
                    has_he2 = "HE2" in atoms_in_res
                    if has_hd1 and has_he2:
                        base_charge = 1  # HIP = +1
                        reason = f"propka pKa={pka_val:.1f} > pH → doubly protonated (+1, HD1+HE2 present)"
                    elif has_hd1 or has_he2:
                        base_charge = 0  # only one H on ring N → singly protonated
                        missing = "HD1" if not has_hd1 else "HE2"
                        reason = (
                            f"propka pKa={pka_val:.1f} suggests +1, but {missing} missing "
                            f"(likely clash) → singly protonated (0)"
                        )
                    else:
                        base_charge = 0
                        reason = f"propka pKa={pka_val:.1f} suggests +1, but no ring H found → neutral (0)"
                else:
                    reason = f"propka pKa={pka_val:.1f} → singly protonated (0)"

            elif resn == "LYS":
                # Normally protonated (+1). Deprotonated (0) if pKa < pH
                if pka_val is not None and pka_val < pH:
                    base_charge = 0
                    reason = f"propka pKa={pka_val:.1f} < pH={pH} → deprotonated (0)"
                else:
                    reason = f"propka pKa={pka_val:.1f} → protonated (+1)"

            elif resn == "CYS":
                # Normally protonated (0). Deprotonated (-1) if pKa < pH
                if pka_val is not None and pka_val < pH:
                    base_charge = -1
                    reason = f"propka pKa={pka_val:.1f} < pH={pH} → deprotonated (-1)"

        # Metal-coordination override: if reduce says this residue coordinates
        # a metal, it CANNOT be doubly-protonated. Force to neutral.
        # This overrides propka's pKa prediction for metal-coordinating HIS.
        if metal_coord and key in metal_coord:
            mc = metal_coord[key]
            if resn == "HIS" and base_charge == 1:
                base_charge = 0
                reason = (
                    f"METAL OVERRIDE: {mc['coord_atom']} coordinates {mc['metal']} "
                    f"→ cannot be doubly-protonated (propka pKa ignored)"
                )
            elif resn in ("CYS",) and base_charge == 0:
                # CYS coordinating metal is typically deprotonated (thiolate)
                base_charge = -1
                reason = (
                    f"METAL OVERRIDE: coordinates {mc['metal']} "
                    f"→ deprotonated thiolate (-1)"
                )
            elif resn in ("ASP", "GLU") and base_charge == 0:
                # ASP/GLU coordinating metal: keep as deprotonated (carboxylate)
                base_charge = -1
                reason = (
                    f"METAL OVERRIDE: coordinates {mc['metal']} "
                    f"→ deprotonated carboxylate (-1)"
                )

        # Covalent bond to ligand override: if a key atom (NZ, SG, etc.) is
        # covalently bonded to the ligand, it cannot carry the standard charge.
        # E.g., LYS NZ bonded to carbamate C → amide NH (neutral, not NH3+)
        if covalent_bonds and key in covalent_bonds:
            cb = covalent_bonds[key]
            if resn == "LYS" and cb.get("atom") == "NZ":
                base_charge = -1  # NZ-H is NH⁻ (fills open-shell C on ligand carbamate)
                reason = (
                    f"COVALENT BOND: NZ bonded to ligand {cb.get('ligand_atom', '?')} "
                    f"→ NH⁻ (-1, not NH3+ +1)"
                )
            elif resn == "CYS" and cb.get("atom") == "SG":
                base_charge = 0  # SG bonded to ligand = thioether, neutral
                reason = f"COVALENT BOND: SG bonded to ligand → thioether (0)"

        # Handle KCX (carboxylated lysine)
        if resn == "KCX":
            base_charge = 0  # NH group (+1) + carboxylate (-1) = 0
            reason = "KCX: carbamylated lysine (NH+ + COO- = 0)"

        # Handle termini — make neutral for capped/cropped models
        if neutral_termini:
            if key in terminal_n:
                # N-terminus: don't add +1 for NH3+ (treat as neutral NH2)
                if resn in ("LYS", "ARG"):
                    pass  # sidechain charge already counted
                reason += " | N-terminus (neutral)"
            if key in terminal_c:
                # C-terminus: don't add -1 for COO- (treat as neutral COOH)
                reason += " | C-terminus (neutral)"

        charges[key] = {
            "resn": resn,
            "charge": base_charge,
            "reason": reason,
        }

    return charges


def calculate_total_charge(
    residue_charges: dict,
    ligand_charge: int = 0,
) -> tuple[int, dict]:
    """Calculate total system charge.

    Returns (total_charge, breakdown_dict)
    """
    protein_charge = sum(v["charge"] for v in residue_charges.values())
    total = protein_charge + ligand_charge

    breakdown = {
        "protein_charge": protein_charge,
        "ligand_charge": ligand_charge,
        "total_charge": total,
        "residue_charges": {
            f"{k[0]}:{k[1]}:{v['resn']}": v["charge"]
            for k, v in residue_charges.items()
            if v["charge"] != 0
        },
    }

    return total, breakdown


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════


def protonate_active_site(
    input_pdb: str | Path,
    output_pdb: str | Path,
    pH: float = 7.0,
    ligand_charge: int = 0,
    neutral_termini: bool = True,
    user_overrides: dict | None = None,
    flip: bool = True,
    relax_h: bool = False,
    strip_h: bool = False,
) -> dict:
    """Protonate a protein active site cluster.

    Args:
        input_pdb: Input PDB file
        output_pdb: Output protonated PDB file
        pH: pH for protonation state decisions
        ligand_charge: Net charge of HETATM atoms (user-specified)
        neutral_termini: Treat termini as neutral (for cropped models)
        user_overrides: {(chain, resnum): charge} for manual protonation control
        flip: Optimize HIS/ASN/GLN sidechain orientations

    Returns:
        Summary dict with charges, pKa values, diagnostics
    """
    input_pdb = Path(input_pdb)
    output_pdb = Path(output_pdb)

    log.info(f"Input: {input_pdb}")
    log.info(f"Output: {output_pdb}")
    log.info(f"pH: {pH}, ligand charge: {ligand_charge:+d}")

    # Step 0: Optionally strip existing protein H before re-protonating
    if strip_h:
        log.info("Step 0: Stripping existing protein hydrogens ...")
        stripped_pdb = Path(str(input_pdb) + ".stripped.pdb")
        lines = Path(input_pdb).read_text().splitlines()
        kept = []
        n_stripped = 0
        for line in lines:
            if line.startswith("ATOM") and len(line) >= 78:
                el = line[76:78].strip()
                if el == "H":
                    n_stripped += 1
                    continue
            elif line.startswith("ATOM") and len(line) >= 16:
                aname = line[12:16].strip()
                if aname.startswith("H") and aname not in ("HG", "HG1"):
                    n_stripped += 1
                    continue
            kept.append(line)
        stripped_pdb.write_text("\n".join(kept) + "\n")
        log.info(f"  Stripped {n_stripped} H atoms from protein")
        input_pdb = stripped_pdb

    # Step 0b: Rebuild missing backbone atoms on partial residues
    input_pdb = _rebuild_missing_backbone(input_pdb)

    # Step 1a: Run reduce for metal-coordination detection (diagnostic only)
    log.info("Step 1a: Running reduce (metal coordination detection) ...")
    reduce_diag, reduce_tmp = run_reduce_for_diagnostics(input_pdb)

    # Step 1b: Parse reduce's metal coordination diagnostics
    log.info("Step 1b: Parsing metal coordination from reduce ...")
    metal_coord = parse_reduce_metal_coordination(reduce_tmp)
    reduce_tmp.unlink(missing_ok=True)  # cleanup temp file

    # Step 1c: Add ALL hydrogens with pdbfixer (proper PDB atom names)
    log.info("Step 1c: Running pdbfixer (full hydrogen addition) ...")
    run_pdbfixer(input_pdb, output_pdb, pH=pH)

    # Step 1d: Post-process — fix terminals, cross-residue covalent bonds
    log.info("Step 1d: Post-processing hydrogens ...")
    pp_info = postprocess_hydrogens(output_pdb, neutral_termini=neutral_termini)
    covalent_bonds = pp_info.get("covalent_bonds", {})

    if metal_coord:
        log.info(f"  Found {len(metal_coord)} metal-coordinating residue(s)")
    else:
        log.info("  No metal coordination detected")

    # Step 2: propka pKa prediction (needed before xTB to get correct total charge)
    log.info("Step 2: Running propka (pKa prediction) ...")
    pka_dict = run_propka(output_pdb, pH=pH)
    if pka_dict:
        log.info(f"  Predicted pKa for {len(pka_dict)} groups")
        for key, info in sorted(pka_dict.items()):
            if info["resn"] in ("HIS", "ASP", "GLU", "LYS", "CYS"):
                state = "protonated" if info["protonated"] else "deprotonated"
                coord_tag = " [METAL-COORD]" if key in metal_coord else ""
                log.info(f"    {info['resn']} {key[0]}:{key[1]} pKa={info['pKa']:.1f} → {state}{coord_tag}")

    # Step 3: Determine charges (with metal-coordination override)
    log.info("Step 3: Determining residue charges ...")
    residue_charges = determine_residue_charges(
        output_pdb, pH=pH, neutral_termini=neutral_termini,
        user_overrides=user_overrides, pka_dict=pka_dict,
        metal_coord=metal_coord, covalent_bonds=covalent_bonds,
    )

    for key, info in sorted(residue_charges.items()):
        if info["charge"] != 0:
            log.info(f"  {info['resn']} {key[0]}:{key[1]}: charge={info['charge']:+d} ({info['reason']})")

    # Step 4: Total charge
    total_charge, breakdown = calculate_total_charge(residue_charges, ligand_charge)

    log.info("")
    log.info("=" * 50)
    log.info(f"  Protein charge:  {breakdown['protein_charge']:+d}")
    log.info(f"  Ligand charge:   {breakdown['ligand_charge']:+d}")
    log.info(f"  TOTAL CHARGE:    {total_charge:+d}")
    log.info("=" * 50)

    # Step 5: Optional xTB H relaxation (now we know the correct total charge)
    if relax_h:
        log.info(f"Step 5: xTB hydrogen relaxation (charge={total_charge:+d}) ...")
        relax_hydrogens_xtb(output_pdb, charge=total_charge)

    # Write charge info as REMARK in the output PDB
    _add_charge_remarks(output_pdb, total_charge, breakdown)

    # Step 6: PDB quality checks
    log.info("Step 6: PDB validation ...")
    try:
        from qcb.prep.validate_pdb import validate_pdb as _validate
        issues = _validate(output_pdb)
        errors = [i for i in issues if i.severity == "ERROR"]
        warnings = [i for i in issues if i.severity == "WARNING"]
        if errors:
            log.error(f"  PDB VALIDATION: {len(errors)} ERRORS found!")
            for issue in errors:
                log.error(f"    {issue}")
        elif warnings:
            log.warning(f"  PDB validation: {len(warnings)} warnings")
            for issue in warnings:
                log.warning(f"    {issue}")
        else:
            log.info("  PDB validation PASSED (no errors or warnings)")
    except ImportError:
        log.info("  PDB validation skipped (qcb.prep.validate_pdb not importable)")

    summary = {
        "input_pdb": str(input_pdb),
        "output_pdb": str(output_pdb),
        "pH": pH,
        "total_charge": total_charge,
        "protein_charge": breakdown["protein_charge"],
        "ligand_charge": ligand_charge,
        "pka_predictions": {
            f"{v['resn']}_{k[0]}:{k[1]}": v["pKa"]
            for k, v in pka_dict.items()
            if v["resn"] in ("HIS", "ASP", "GLU", "LYS", "CYS")
        },
        "charged_residues": breakdown["residue_charges"],
        "h_added": sum(1 for l in Path(output_pdb).read_text().splitlines()
                       if l.startswith("ATOM") and l[76:78].strip() == "H"),
    }

    return summary


def _add_charge_remarks(pdb_path: Path, total_charge: int, breakdown: dict):
    """Add REMARK lines with charge information to the PDB."""
    lines = Path(pdb_path).read_text().splitlines()

    # Find where to insert (after existing REMARK lines)
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith(("REMARK", "HEADER", "USER")):
            insert_idx = i + 1
        else:
            break

    charge_remarks = [
        f"REMARK QCB TOTAL_CHARGE {total_charge:+d}",
        f"REMARK QCB PROTEIN_CHARGE {breakdown['protein_charge']:+d}",
        f"REMARK QCB LIGAND_CHARGE {breakdown['ligand_charge']:+d}",
    ]
    for key, charge in breakdown["residue_charges"].items():
        charge_remarks.append(f"REMARK QCB CHARGED_RES {key} {charge:+d}")

    for i, remark in enumerate(charge_remarks):
        lines.insert(insert_idx + i, remark)

    Path(pdb_path).write_text("\n".join(lines) + "\n")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════


def main():
    p = argparse.ArgumentParser(
        description="Protonate a protein active site cluster for QM/MLFF calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python protonate_active_site.py input.pdb -o protonated.pdb --ligand-charge 3
  python protonate_active_site.py input.pdb -o protonated.pdb --ligand-charge 3 --pH 8.0
  python protonate_active_site.py input.pdb -o protonated.pdb --ligand-charge 3 --override A:1:+1 C:3:0
        """,
    )
    p.add_argument("input_pdb", help="Input PDB file (protein active site cluster)")
    p.add_argument("-o", "--output", required=True, help="Output protonated PDB file")
    p.add_argument("--ligand-charge", type=int, default=0,
                   help="Net formal charge of HETATM atoms (default: 0)")
    p.add_argument("--pH", type=float, default=7.0, help="pH for protonation (default: 7.0)")
    p.add_argument("--no-flip", action="store_true",
                   help="Don't optimize HIS/ASN/GLN orientations (faster)")
    p.add_argument("--charged-termini", action="store_true",
                   help="Treat termini as charged (NH3+/COO-) instead of neutral")
    p.add_argument("--strip-h", action="store_true",
                   help="Strip ALL existing protein hydrogens before adding new ones. "
                        "Use when input has incorrect/partial H that confuse pdbfixer.")
    p.add_argument("--override", nargs="*", default=[],
                   help="Manual charge overrides: CHAIN:RESNUM:CHARGE (e.g. A:1:+1 C:3:0)")
    p.add_argument("--relax-h", action="store_true",
                   help="Relax hydrogen positions with xTB (fixes sp2/sp3 geometry, ~seconds)")
    p.add_argument("--summary-json", default=None,
                   help="Write summary JSON to this path")

    args = p.parse_args()

    # Parse user overrides
    user_overrides = {}
    for ov in args.override:
        parts = ov.split(":")
        if len(parts) != 3:
            print(f"ERROR: Override format is CHAIN:RESNUM:CHARGE, got: {ov}")
            sys.exit(1)
        chain, resnum, charge = parts[0], int(parts[1]), int(parts[2])
        user_overrides[(chain, resnum)] = charge

    summary = protonate_active_site(
        args.input_pdb,
        args.output,
        pH=args.pH,
        ligand_charge=args.ligand_charge,
        neutral_termini=not args.charged_termini,
        user_overrides=user_overrides if user_overrides else None,
        flip=not args.no_flip,
        relax_h=args.relax_h,
        strip_h=args.strip_h,
    )

    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"Summary written to {args.summary_json}")


if __name__ == "__main__":
    main()
