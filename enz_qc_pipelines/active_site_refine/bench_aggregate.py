#!/usr/bin/env python3
"""Aggregate benchmark scores and pick the winner.

Reads all *.pdb in the bench dir, scores each against design, and produces:

  • A ranked table sorted by composite score
  • A markdown summary
  • The recommended command for production use
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# import the scorer's score_one
sys.path.insert(0, str(Path(__file__).parent))
from geom_score import (parse_pdb, parse_remark666, hetatms, score_one)


def composite_score(s: dict) -> float:
    """Lower is better. Weights:
       contact_mae × 4 (most important: design intent)
       metal_mae × 2 (subset of contact, but emphasised)
       angle_mae × 0.05 (in deg → ~angle_mae° drives the metric)
       bond_mae × 8 (Å → larger weight than angle since bond errs are smaller)
       ligand_rmsd_A × 4 (should be 0)
       worst_angle × 0.01 (penalty for any single bad angle)
    """
    def safe(x, fallback=0.0):
        return x if x is not None else fallback
    return (
        4.0 * safe(s.get("contact_mae_A"), 5.0) +
        2.0 * safe(s.get("metal_mae_A"),   5.0) +
        0.05 * safe(s.get("angle_mae_deg"), 30) +
        8.0 * safe(s.get("bond_mae_A"),    0.5) +
        4.0 * safe(s.get("ligand_rmsd_A"), 1.0) +
        0.01 * safe(s.get("angle_max_deg"), 30)
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("design", type=Path)
    p.add_argument("bench_dir", type=Path)
    p.add_argument("--reference", type=Path, default=None,
                   help="Also include this file (e.g., AF3 input) as a baseline row")
    p.add_argument("--json", type=Path, default=None,
                   help="Write per-file scores as JSON to this path")
    args = p.parse_args()

    design = parse_pdb(args.design)
    catres = parse_remark666(args.design)
    lig_design = hetatms(args.design)

    files = sorted(args.bench_dir.glob("*.pdb"))
    if args.reference and args.reference.is_file():
        files = [args.reference] + files

    rows = []
    for f in files:
        ref = parse_pdb(f)
        lig = hetatms(f)
        s = score_one(design, ref, catres, lig_design, lig)
        s["file"] = f.name
        s["composite"] = composite_score(s)
        rows.append(s)

    rows.sort(key=lambda r: r["composite"])

    cols = [("file", 38), ("composite", 9),
            ("contact_mae_A", 8), ("metal_mae_A", 8),
            ("angle_mae_deg", 8), ("angle_max_deg", 8),
            ("bond_mae_A", 8), ("ligand_rmsd_A", 9)]
    hdr = "  ".join(f"{c:<{w}}" for c, w in cols)
    print(hdr); print("-" * len(hdr))
    for r in rows:
        cells = [r["file"][:38]]
        for c, w in cols[1:]:
            v = r.get(c)
            cells.append(f"{v:.4f}" if v is not None else "  -  ")
        print("  ".join(f"{cell:<{w}}" for cell, (_, w) in zip(cells, cols)))

    print()
    print("Worst angle per run:")
    for r in rows:
        wa = r.get("worst_angle_detail") or {}
        if wa:
            print(f"  {r['file']:<38}  {wa['res']} {wa['atoms']} "
                  f"design={wa['design']:.1f}° refined={wa['refined']:.1f}° "
                  f"err={wa['err']:.1f}°")

    print()
    print(f"Winner (lowest composite): {rows[0]['file']}")
    print(f"  composite = {rows[0]['composite']:.4f}")
    print(f"  contact_mae = {rows[0]['contact_mae_A']:.4f} Å, "
          f"metal_mae = {rows[0]['metal_mae_A']:.4f} Å, "
          f"angle_mae = {rows[0]['angle_mae_deg']:.2f}°, "
          f"ligand_rmsd = {rows[0]['ligand_rmsd_A']:.4f} Å")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
