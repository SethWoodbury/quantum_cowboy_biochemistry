#!/usr/bin/env python
"""funnel_to_pdb.py — convert crest_funnel XYZ outputs back to PDBs.

Pulls atom names / residue records from the source PDB and substitutes
coordinates from the optimized XYZ files. CREST runs in a CMA-translated
frame, so we Kabsch-align Stage C and Stage E outputs back to the source
frame using the (always-frozen) CA atoms. Result: all PDBs overlay in
PyMOL/ChimeraX.

Outputs written:
  10_xtb_preopt/preopt.pdb        (no waters)
  30_gxtb_minimize/conf_NN/gxtb_opt.pdb  (no waters, Kabsch-aligned)
  40_water_relax/conf_NN/relaxed.pdb     (full system, Kabsch-aligned)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ----- minimal PDB reader/writer ---------------------------------------------
@dataclass
class PdbAtom:
    raw: str       # original record (we keep formatting)
    serial: int
    name: str
    resname: str
    chain: str
    resseq: int
    x: float
    y: float
    z: float


def parse_pdb(path: Path) -> list[PdbAtom]:
    out = []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            out.append(PdbAtom(
                raw=line,
                serial=int(line[6:11]),
                name=line[12:16].strip(),
                resname=line[17:20].strip(),
                chain=line[21:22].strip(),
                resseq=int(line[22:26]),
                x=float(line[30:38]),
                y=float(line[38:46]),
                z=float(line[46:54]),
            ))
    return out


def read_xyz_coords(path: Path) -> np.ndarray:
    t = path.read_text().splitlines()
    n = int(t[0].strip())
    return np.array([list(map(float, ln.split()[1:4])) for ln in t[2:2+n]])


def write_pdb_with_new_coords(template_atoms: list[PdbAtom], coords: np.ndarray,
                              out_path: Path, header_lines: list[str] | None = None) -> None:
    if len(template_atoms) != len(coords):
        raise ValueError(
            f"atom count mismatch: template has {len(template_atoms)}, xyz has {len(coords)}"
        )
    lines = []
    if header_lines:
        lines.extend(header_lines)
    for a, (x, y, z) in zip(template_atoms, coords):
        # rewrite cols 31-54 (1-indexed) i.e. 30:54 (0-indexed) with new coords
        # PDB fixed format: x[31-38] y[39-46] z[47-54], each F8.3
        new = f"{x:8.3f}{y:8.3f}{z:8.3f}"
        # left part up to col 30 (0-idx); right part from col 54 onward
        line = a.raw[:30] + new + a.raw[54:]
        lines.append(line)
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n")


# ----- Kabsch alignment ------------------------------------------------------
def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (R, cP, cQ) such that  P_aligned = (P - cP) @ R.T + cQ  matches Q."""
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cP, cQ


def align(coords: np.ndarray, source_coords: np.ndarray, anchor_idx_0: list[int]) -> np.ndarray:
    R, cP, cQ = kabsch(coords[anchor_idx_0], source_coords[anchor_idx_0])
    return (coords - cP) @ R.T + cQ


# ----- main ------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True,
                    help="crest_funnel output dir (containing 00_input/, 10_xtb_preopt/, ...)")
    args = ap.parse_args()

    base = args.out
    src_pdb = base / "00_input/source.pdb"
    partition = json.loads((base / "00_input/partition.json").read_text())
    fix_1 = partition["fix_indices_no_waters_1based"]
    fix_0 = [i - 1 for i in fix_1]
    n_no_waters = partition["n_atoms_no_waters"]

    # Source PDB ordering (with waters at end)
    src_atoms = parse_pdb(src_pdb)
    if len(src_atoms) != partition["n_atoms_total"]:
        raise RuntimeError(
            f"source PDB atom count {len(src_atoms)} ≠ partition total "
            f"{partition['n_atoms_total']}"
        )
    src_no_waters_atoms = src_atoms[:n_no_waters]   # waters live at the end
    src_no_waters_xyz = np.array([[a.x, a.y, a.z] for a in src_no_waters_atoms])

    written: list[Path] = []

    # Stage A: no waters, no realignment needed (xtb preserves frame)
    a_xyz = base / "10_xtb_preopt/preopt.xyz"
    if a_xyz.exists():
        coords = read_xyz_coords(a_xyz)
        out = base / "10_xtb_preopt/preopt.pdb"
        write_pdb_with_new_coords(
            src_no_waters_atoms, coords, out,
            header_lines=["REMARK   stage=A_xtb_preopt  n_atoms=%d" % len(coords)],
        )
        written.append(out)

    # Stage C: no waters, Kabsch-align via fixed CA(+Zn) anchors
    c_root = base / "30_gxtb_minimize"
    if c_root.exists():
        for conf_dir in sorted(c_root.glob("conf_*")):
            xyz = conf_dir / "xtbopt.xyz"
            if not xyz.exists():
                continue
            coords = read_xyz_coords(xyz)
            aligned = align(coords, src_no_waters_xyz, fix_0)
            out = conf_dir / "gxtb_opt.pdb"
            write_pdb_with_new_coords(
                src_no_waters_atoms, aligned, out,
                header_lines=[f"REMARK   stage=C_gxtb_opt  source_dir={conf_dir.name}",
                              f"REMARK   aligned_to=source.pdb  anchors=CA+Zn n={len(fix_0)}"],
            )
            written.append(out)

    # Stage E: full system, Kabsch-align via fixed CA(+Zn) anchors (within the no-waters slice)
    e_root = base / "40_water_relax"
    n_water_atoms = partition["n_water_atoms"]
    if e_root.exists():
        for conf_dir in sorted(e_root.glob("conf_*")):
            xyz = conf_dir / "xtbopt.xyz"
            if not xyz.exists():
                continue
            coords = read_xyz_coords(xyz)
            assert len(coords) == n_no_waters + n_water_atoms, (
                f"{xyz} has {len(coords)} atoms; expected {n_no_waters + n_water_atoms} "
                f"({n_no_waters} non-water + {n_water_atoms} water). Atom ordering is "
                "no-waters-then-waters; an XYZ in any other order would silently misalign.")
            # anchors live in the first n_no_waters atoms
            aligned = align(coords, np.vstack([src_no_waters_xyz,
                                               np.array([[a.x, a.y, a.z] for a in src_atoms[n_no_waters:]])]),
                            fix_0)
            out = conf_dir / "relaxed.pdb"
            write_pdb_with_new_coords(
                src_atoms, aligned, out,
                header_lines=[f"REMARK   stage=E_water_relax  source_dir={conf_dir.name}",
                              f"REMARK   aligned_to=source.pdb  anchors=CA+Zn n={len(fix_0)}",
                              f"REMARK   waters relaxed; non-water atoms held frozen via $fix 1-{n_no_waters}"],
            )
            written.append(out)

    print(f"wrote {len(written)} PDBs:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
