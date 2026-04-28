#!/usr/bin/env python3
"""Batch theozyme refinement: process thousands of design+AF3 pairs in parallel.

Two input modes:
  1. Pair file: TSV with columns design_pdb<TAB>af3_pdb<TAB>output_pdb<TAB>charge
  2. Directory pair: --design-dir A/ --af3-dir B/  → matches by stem

Run on CPU pool. Defaults to GFN-FF (~5-15 s per system).

Usage:
  python scripts/refine_batch.py pairs.tsv --jobs 32
  python scripts/refine_batch.py --design-dir designs/ --af3-dir af3/ \
      --output-dir refined/ --jobs 16 --gfn 0 --ptm-default A:64=KCX
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Make qcb importable regardless of cwd
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from qcb.pipelines.refine_active_site import refine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("refine_batch")


def _refine_one(args: dict) -> dict:
    """Worker process: run a single refinement, never raise."""
    try:
        result = refine(
            design_pdb=args["design"], af3_pdb=args["af3"],
            output_pdb=args["output"], radius=args["radius"],
            gfn=args["gfn"], rigidity=args["rigidity"],
            charge=args.get("charge", 0),
            ptm_map=args.get("ptm_map") or None,
            keep_workdir=False,
            write_cluster_files=args.get("write_cluster", False),
        )
        result["design"] = str(args["design"])
        result["af3"] = str(args["af3"])
        return result
    except Exception as e:
        return {
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "design": str(args["design"]),
            "af3": str(args["af3"]),
        }


def _collect_pairs_from_tsv(tsv: Path) -> list[dict]:
    pairs = []
    with open(tsv) as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 3:
                log.warning(f"  TSV row {i} has only {len(row)} columns; skipping")
                continue
            charge = int(row[3]) if len(row) > 3 else 0
            pairs.append({
                "design": row[0], "af3": row[1], "output": row[2], "charge": charge,
            })
    return pairs


def _collect_pairs_from_dirs(design_dir: Path, af3_dir: Path,
                              output_dir: Path, charge: int) -> list[dict]:
    """Match design and AF3 PDBs by filename stem (or substring)."""
    pairs = []
    af3_index: dict[str, Path] = {}
    for f in af3_dir.glob("*.pdb"):
        # Use the substring after a known marker if needed; here just use stem
        af3_index[f.stem] = f

    for design in sorted(design_dir.glob("*.pdb")):
        # Match by stem prefix (designs typically have an extra suffix)
        match = None
        for stem, path in af3_index.items():
            if stem.startswith(design.stem) or design.stem.startswith(stem):
                match = path; break
        if match is None:
            log.warning(f"  no AF3 match for {design.name}; skipping")
            continue
        out = output_dir / f"{design.stem}_refined.pdb"
        pairs.append({
            "design": str(design), "af3": str(match),
            "output": str(out), "charge": charge,
        })
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("pairs_tsv", nargs="?",
                     help="TSV file: design<TAB>af3<TAB>output[<TAB>charge]")
    src.add_argument("--design-dir", type=Path, help="Directory of design PDBs")
    p.add_argument("--af3-dir", type=Path, help="Directory of AF3 PDBs (with --design-dir)")
    p.add_argument("--output-dir", type=Path, help="Output dir (with --design-dir)")
    p.add_argument("--jobs", "-j", type=int, default=4,
                   help="Number of parallel CPU workers (default: 4)")
    p.add_argument("--radius", type=float, default=6.0)
    p.add_argument("--gfn", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--rigidity", default="backbone", choices=["backbone", "backbone-cb"])
    p.add_argument("--charge", type=int, default=0,
                   help="Default cluster charge if not in TSV (default: 0)")
    p.add_argument("--ptm-default", action="append", default=[],
                   help="PTM relabel applied to ALL pairs (e.g., A:64=KCX). Repeatable.")
    p.add_argument("--summary-json", type=Path, default=None,
                   help="Write summary JSON of all results")
    p.add_argument("--write-cluster", action="store_true",
                   help="Also write per-system cluster_input.pdb / cluster_refined.pdb")
    args = p.parse_args()

    # Collect pairs
    if args.pairs_tsv:
        pairs = _collect_pairs_from_tsv(Path(args.pairs_tsv))
    else:
        if not (args.af3_dir and args.output_dir):
            p.error("--design-dir requires --af3-dir and --output-dir")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pairs = _collect_pairs_from_dirs(args.design_dir, args.af3_dir,
                                          args.output_dir, args.charge)

    if not pairs:
        log.error("No design/AF3 pairs found; aborting")
        sys.exit(1)
    log.info(f"Refining {len(pairs)} pairs with {args.jobs} workers (GFN-{args.gfn})")

    # Default PTM map applied to every pair
    ptm_default = {}
    for p_spec in args.ptm_default:
        if "=" not in p_spec:
            log.error(f"--ptm-default must be 'CHAIN:RESNUM=NEWNAME', got {p_spec!r}")
            sys.exit(1)
        k, v = p_spec.split("=", 1)
        ptm_default[k.strip()] = v.strip()

    # Augment each pair with shared params
    work_items = []
    for pair in pairs:
        item = dict(pair)
        item.update(dict(
            radius=args.radius, gfn=args.gfn, rigidity=args.rigidity,
            ptm_map=ptm_default, write_cluster=args.write_cluster,
        ))
        work_items.append(item)

    # Run in parallel
    results = []
    n_done = n_failed = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(_refine_one, w): w for w in work_items}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            if r["status"] == "completed":
                n_done += 1
                if n_done % 25 == 0 or n_done == len(work_items):
                    log.info(f"  progress: {n_done} done, {n_failed} failed")
            else:
                n_failed += 1
                log.warning(f"  FAILED {Path(r['design']).name}: {r.get('error', 'unknown')}")

    log.info(f"DONE: {n_done}/{len(work_items)} succeeded, {n_failed} failed")

    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump({
                "total": len(work_items),
                "succeeded": n_done,
                "failed": n_failed,
                "results": results,
            }, f, indent=2, default=str)
        log.info(f"Summary → {args.summary_json}")


if __name__ == "__main__":
    main()
