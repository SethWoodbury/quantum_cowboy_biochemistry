#!/usr/bin/env python3
"""Score how well a refined active-site PDB recapitulates the design.

Given the design.pdb and one or more refined PDBs, compute:

  Catalytic-residue contact recovery (lower-better, vs design distances)
    * mean absolute distance error on close (<4.5 Å) catres-ligand pairs
    * mean abs error on metal-coord (M-N/M-O <2.7 Å) pairs only

  Sidechain valence preservation (lower-better, vs design):
    * mean abs deviation of CA-CB-CG angle (degrees)
    * mean abs deviation of CB-CG bond length (Å)
    * mean abs deviation of CA-CB bond length (Å)
    * worst-case deviation of any of the above

  Ligand rigidity (should be ~0):
    * RMSD of HETATM atoms vs design (after global alignment is implicit:
      the inputs are already aligned)

Usage:
    python geom_score.py design.pdb refined1.pdb [refined2.pdb ...] \\
        [--catres 41,64,148,184,187,188]
    # or read REMARK 666 from design.pdb (default)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def parse_pdb(p: Path) -> dict[tuple[str, int, str], np.ndarray]:
    atoms: dict[tuple[str, int, str], np.ndarray] = {}
    for line in open(p):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        chain = line[21]
        rnum = int(line[22:26])
        aname = line[12:16].strip()
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        atoms[(chain, rnum, aname)] = np.array([x, y, z])
    return atoms


def parse_remark666(p: Path) -> list[tuple[str, str, int]]:
    out = []
    for line in open(p):
        if not line.startswith("REMARK 666"):
            continue
        m = re.search(r"MOTIF\s+(\S)\s+(\S+)\s+(\d+)", line)
        if m:
            out.append((m.group(1), m.group(2), int(m.group(3))))
    return out


def hetatms(p: Path) -> list[tuple[str, int, str, np.ndarray, str]]:
    out = []
    for line in open(p):
        if not line.startswith("HETATM"):
            continue
        chain = line[21]
        rnum = int(line[22:26])
        aname = line[12:16].strip()
        rname = line[17:20].strip()
        el = line[76:78].strip() if len(line) >= 78 else aname[0]
        if el == "H":
            continue
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        out.append((chain, rnum, aname, np.array([x, y, z]), el))
    return out


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def bond(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# Heavy-atom sidechain definitions (for valence checks)
SC_DEFS: dict[str, list[tuple[str, str, str]]] = {
    "ALA": [],
    "ARG": [("CA","CB","CG"), ("CB","CG","CD"), ("CG","CD","NE"), ("CD","NE","CZ")],
    "ASN": [("CA","CB","CG")],
    "ASP": [("CA","CB","CG")],
    "CYS": [],
    "GLN": [("CA","CB","CG"), ("CB","CG","CD")],
    "GLU": [("CA","CB","CG"), ("CB","CG","CD")],
    "GLY": [],
    "HIS": [("CA","CB","CG"), ("CB","CG","ND1"), ("CB","CG","CD2")],
    "ILE": [("CA","CB","CG1"), ("CB","CG1","CD1")],
    "LEU": [("CA","CB","CG"), ("CB","CG","CD1")],
    "LYS": [("CA","CB","CG"), ("CB","CG","CD"), ("CG","CD","CE"), ("CD","CE","NZ")],
    "MET": [("CA","CB","CG"), ("CB","CG","SD"), ("CG","SD","CE")],
    "PHE": [("CA","CB","CG")],
    "PRO": [("CA","CB","CG"), ("CB","CG","CD")],
    "SER": [],
    "THR": [],
    "TRP": [("CA","CB","CG")],
    "TYR": [("CA","CB","CG")],
    "VAL": [("CA","CB","CG1")],
}

SC_BONDS: dict[str, list[tuple[str, str]]] = {
    "ARG": [("CA","CB"),("CB","CG"),("CG","CD"),("CD","NE")],
    "ASN": [("CA","CB"),("CB","CG")],
    "ASP": [("CA","CB"),("CB","CG"),("CG","OD1"),("CG","OD2")],
    "GLN": [("CA","CB"),("CB","CG"),("CG","CD")],
    "GLU": [("CA","CB"),("CB","CG"),("CG","CD"),("CD","OE1"),("CD","OE2")],
    "HIS": [("CA","CB"),("CB","CG"),("CG","ND1"),("CG","CD2"),("ND1","CE1"),("CE1","NE2"),("NE2","CD2")],
    "LYS": [("CA","CB"),("CB","CG"),("CG","CD"),("CD","CE"),("CE","NZ")],
    "PHE": [("CA","CB"),("CB","CG")],
    "TYR": [("CA","CB"),("CB","CG")],
    "TRP": [("CA","CB"),("CB","CG")],
}


def score_one(design: dict, refined: dict,
              catres: list[tuple[str, str, int]],
              lig_atoms_design: list,
              lig_atoms_refined: list,
              contact_radius: float = 4.5) -> dict:
    # Build design + refined contact lists for catalytic residues
    contact_errors: list[float] = []
    metal_errors: list[float] = []
    angle_errors: list[float] = []
    bond_errors: list[float] = []
    angle_details: list[dict] = []

    for chain, rname, rnum in catres:
        # Sidechain heavy-atom contacts to ligand atoms (design defines targets)
        sc_atoms = []  # (aname, design_pos, refined_pos)
        for an in ("CB","CG","CD","CE","NZ","ND1","NE2","CD1","CD2",
                   "CE1","CZ","OE1","OE2","OD1","OD2","SG","NE","CG1","CG2"):
            d_pos = design.get((chain, rnum, an))
            r_pos = refined.get((chain, rnum, an))
            if d_pos is not None and r_pos is not None:
                sc_atoms.append((an, d_pos, r_pos))

        # Match each catres heavy atom to nearest ligand heavy atom in design
        # (NOT element-restricted so we capture all close geometry)
        for an, d_pos, r_pos in sc_atoms:
            for L_chain, L_num, L_an, L_d_pos, L_el in lig_atoms_design:
                d_des = bond(d_pos, L_d_pos)
                if d_des > contact_radius:
                    continue
                # find same ligand atom in refined
                lr = next(((Lc, Ln, La, Lp, Le)
                           for Lc, Ln, La, Lp, Le in lig_atoms_refined
                           if La == L_an and Ln == L_num), None)
                if lr is None:
                    continue
                d_ref = bond(r_pos, lr[3])
                err = abs(d_ref - d_des)
                contact_errors.append(err)
                if d_des < 2.7 and (
                    L_el.upper() in ("ZN","MG","MN","FE","CU","CA")
                ):
                    metal_errors.append(err)

        # Sidechain valence: angles + bonds defined per residue
        for trip in SC_DEFS.get(rname, []):
            a, b, c = trip
            d = (design.get((chain, rnum, a)), design.get((chain, rnum, b)),
                 design.get((chain, rnum, c)))
            r = (refined.get((chain, rnum, a)), refined.get((chain, rnum, b)),
                 refined.get((chain, rnum, c)))
            if all(p is not None for p in d) and all(p is not None for p in r):
                ad = angle(*d); ar = angle(*r)
                err = abs(ar - ad)
                angle_errors.append(err)
                angle_details.append({
                    "res": f"{rname}{rnum}", "atoms": "-".join(trip),
                    "design": ad, "refined": ar, "err": err,
                })
        for ab in SC_BONDS.get(rname, []):
            a, b = ab
            d = (design.get((chain, rnum, a)), design.get((chain, rnum, b)))
            r = (refined.get((chain, rnum, a)), refined.get((chain, rnum, b)))
            if all(p is not None for p in d) and all(p is not None for p in r):
                bd = bond(*d); br = bond(*r)
                bond_errors.append(abs(br - bd))

    # Ligand RMSD
    lig_pairs = []
    for L_chain, L_num, L_an, L_d_pos, _ in lig_atoms_design:
        lr = next(((Lp) for Lc, Ln, La, Lp, Le in lig_atoms_refined
                   if La == L_an and Ln == L_num), None)
        if lr is not None:
            lig_pairs.append(((L_d_pos, lr)))
    if lig_pairs:
        sq = np.array([np.sum((d - r) ** 2) for d, r in lig_pairs])
        lig_rmsd = float(np.sqrt(np.mean(sq)))
    else:
        lig_rmsd = float("nan")

    return {
        "n_catres": len(catres),
        "contacts_n": len(contact_errors),
        "contact_mae_A": float(np.mean(contact_errors)) if contact_errors else None,
        "contact_max_A": float(np.max(contact_errors)) if contact_errors else None,
        "metal_n": len(metal_errors),
        "metal_mae_A": float(np.mean(metal_errors)) if metal_errors else None,
        "metal_max_A": float(np.max(metal_errors)) if metal_errors else None,
        "angle_n": len(angle_errors),
        "angle_mae_deg": float(np.mean(angle_errors)) if angle_errors else None,
        "angle_max_deg": float(np.max(angle_errors)) if angle_errors else None,
        "bond_n": len(bond_errors),
        "bond_mae_A": float(np.mean(bond_errors)) if bond_errors else None,
        "bond_max_A": float(np.max(bond_errors)) if bond_errors else None,
        "ligand_rmsd_A": lig_rmsd,
        "worst_angle_deviation_deg": float(np.max(angle_errors)) if angle_errors else None,
        "worst_angle_detail": (
            sorted(angle_details, key=lambda d: d["err"])[-1]
            if angle_details else None
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("design", type=Path)
    p.add_argument("refined", type=Path, nargs="+")
    p.add_argument("--catres", type=str, default=None,
                   help="Comma-separated residue numbers (chain A) to score, "
                        "default: parse REMARK 666 from design")
    p.add_argument("--json", action="store_true",
                   help="Emit JSON (one record per refined input)")
    p.add_argument("--contact-radius", type=float, default=4.5)
    args = p.parse_args()

    design = parse_pdb(args.design)
    if args.catres:
        catres = [("A", "?", int(r)) for r in args.catres.split(",") if r.strip()]
        # we don't know rname without extra parsing; pull from design
        catres = []
        for r in args.catres.split(","):
            r = int(r.strip())
            for (ch, rn, an), _ in design.items():
                if ch == "A" and rn == r and an == "CA":
                    # try to find resname
                    catres.append(("A", "ALA", r))  # rname filled below
                    break
        # better: re-parse design once for residue names
        rmap = {}
        for line in open(args.design):
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
                rmap[(line[21], int(line[22:26]))] = line[17:20].strip()
        catres = [(ch, rmap.get((ch, rn), "ALA"), rn) for (ch, _, rn) in catres]
    else:
        catres = parse_remark666(args.design)

    lig_design = hetatms(args.design)

    rows = []
    for rp in args.refined:
        refined = parse_pdb(rp)
        lig_ref = hetatms(rp)
        s = score_one(design, refined, catres, lig_design, lig_ref, args.contact_radius)
        s["file"] = str(rp)
        rows.append(s)

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        # Pretty table
        cols = [("file", 30), ("contact_mae_A", 8), ("contact_max_A", 8),
                ("metal_mae_A", 8), ("metal_max_A", 8),
                ("angle_mae_deg", 8), ("angle_max_deg", 8),
                ("bond_mae_A", 8), ("bond_max_A", 8),
                ("ligand_rmsd_A", 9)]
        hdr = "  ".join(f"{c:<{w}}" for c, w in cols)
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            f = Path(r["file"]).name
            cells = [f[:30]] + [
                (f"{r[c]:.3f}" if r[c] is not None else "  -  ")
                for c, _ in cols[1:]]
            print("  ".join(f"{cell:<{w}}" for cell, (_, w) in zip(cells, cols)))
        # Highlight worst angle for each
        print()
        for r in rows:
            wd = r.get("worst_angle_detail")
            if wd:
                print(f"  {Path(r['file']).name:<30}  worst angle: "
                      f"{wd['res']} {wd['atoms']} "
                      f"design={wd['design']:.1f}° refined={wd['refined']:.1f}° "
                      f"err={wd['err']:.1f}°")


if __name__ == "__main__":
    main()
