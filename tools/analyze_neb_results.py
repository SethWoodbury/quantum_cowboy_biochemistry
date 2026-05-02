from quantum_engine.units import EV_TO_KCAL
#!/usr/bin/env python
"""
Analyze NEB-TS results: extract barriers, validate endpoints, summarize.

Usage:
  python scripts/analyze_neb_results.py /path/to/output_dir
  python scripts/analyze_neb_results.py /path/to/parent_dir --all  # all subdirs
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np



def analyze_single_run(outdir):
    """Analyze a single NEB-TS run directory.

    Returns dict with barrier, endpoint info, validation flags.
    """
    outdir = Path(outdir)
    result = {
        "name": outdir.name,
        "path": str(outdir),
        "status": "unknown",
    }

    # Check for log file
    log_files = list(outdir.glob("*.log")) + list(outdir.parent.glob(f"*{outdir.name}*.stdout"))
    if not log_files:
        # Check parent logs dir
        log_dir = outdir.parent.parent / "logs" if "mlff_outputs" in str(outdir) else None
        if log_dir and log_dir.exists():
            log_files = list(log_dir.glob(f"*{outdir.name}*.stdout"))

    # Try to find barriers from log
    fwd_barrier = None
    rev_barrier = None
    for lf in log_files:
        try:
            text = lf.read_text()
            for line in text.splitlines():
                if "Barrier (fwd)" in line:
                    m = re.search(r"Barrier \(fwd\):\s+([-\d.]+)", line)
                    if m:
                        fwd_barrier = float(m.group(1))
                if "Barrier (rev)" in line:
                    m = re.search(r"Barrier \(rev\):\s+([-\d.]+)", line)
                    if m:
                        rev_barrier = float(m.group(1))
        except Exception:
            pass

    result["barrier_fwd_kcal"] = fwd_barrier
    result["barrier_rev_kcal"] = rev_barrier

    # Check output files
    has_reactant = (outdir / "reactant.pdb").exists() or any(outdir.glob("*reactant.pdb"))
    has_product = (outdir / "product.pdb").exists() or any(outdir.glob("*product.pdb"))
    has_ts = (outdir / "transition_state.pdb").exists() or any(outdir.glob("*transition_state.pdb"))
    has_neb = (outdir / "neb_path.pdb").exists() or any(outdir.glob("*neb_path.pdb"))
    has_cif = any(outdir.glob("*.cif"))

    result["has_reactant"] = has_reactant
    result["has_product"] = has_product
    result["has_ts"] = has_ts
    result["has_neb_path"] = has_neb
    result["has_cif"] = has_cif

    if has_ts and has_reactant and has_product and fwd_barrier is not None:
        result["status"] = "complete"
    elif has_reactant or has_product:
        result["status"] = "partial"
    else:
        result["status"] = "not_started"

    # Validate endpoint bond distances if ASE available
    try:
        from ase.io import read as ase_read

        # Find the actual PDB files (may have system_name prefix)
        reactant_files = list(outdir.glob("*reactant.pdb"))
        product_files = list(outdir.glob("*product.pdb"))
        ts_files = list(outdir.glob("*transition_state.pdb"))

        if reactant_files and product_files:
            reactant = ase_read(str(reactant_files[0]))
            product = ase_read(str(product_files[0]))

            # Extract charge from PDB REMARK
            charge = 0
            with open(reactant_files[0]) as f:
                for line in f:
                    if "TOTAL_CHARGE" in line:
                        m = re.search(r"TOTAL_CHARGE\s+([-\d]+)", line)
                        if m:
                            charge = int(m.group(1))
                        break
            result["charge"] = charge
            result["n_atoms"] = len(reactant)

            # Check key bond distances in reactant and product
            # Need to know which atoms are nucleophile-P and leaving-group-P
            # Parse from filename: O3nuc_P1_O7lg or O1nuc_P1_O5lg
            name = outdir.name
            nuc_m = re.search(r"(\w+\d+)nuc", name)
            lg_m = re.search(r"(\w+\d+)lg", name)
            p_m = re.search(r"_P(\d+)_", name)

            if nuc_m and lg_m and p_m:
                nuc_name = nuc_m.group(1)
                lg_name = lg_m.group(1)
                p_idx = int(p_m.group(1)) - 1  # 0-indexed

                # Try to find atom indices by name from PDB
                nuc_idx = None
                lg_idx = None
                with open(reactant_files[0]) as f:
                    for line in f:
                        if line.startswith("ATOM") or line.startswith("HETATM"):
                            aname = line[12:16].strip()
                            serial = int(line[6:11]) - 1
                            if aname == nuc_name.upper():
                                nuc_idx = serial
                            elif aname == lg_name.upper():
                                lg_idx = serial

                if nuc_idx is not None and lg_idx is not None:
                    r_nuc_p = reactant.get_distance(nuc_idx, p_idx)
                    r_lg_p = reactant.get_distance(lg_idx, p_idx)
                    p_nuc_p = product.get_distance(nuc_idx, p_idx)
                    p_lg_p = product.get_distance(lg_idx, p_idx)

                    result["reactant_nuc_P"] = round(r_nuc_p, 2)
                    result["reactant_lg_P"] = round(r_lg_p, 2)
                    result["product_nuc_P"] = round(p_nuc_p, 2)
                    result["product_lg_P"] = round(p_lg_p, 2)

                    # Validation flags
                    result["nuc_formed"] = p_nuc_p < 1.8  # P-O bond formed
                    result["lg_broken"] = p_lg_p > 2.5    # leaving group departed
                    result["pentacoordinate"] = p_lg_p < 2.5 and p_nuc_p < 1.8

                    if ts_files:
                        ts = ase_read(str(ts_files[0]))
                        ts_nuc_p = ts.get_distance(nuc_idx, p_idx)
                        ts_lg_p = ts.get_distance(lg_idx, p_idx)
                        result["ts_nuc_P"] = round(ts_nuc_p, 2)
                        result["ts_lg_P"] = round(ts_lg_p, 2)

    except ImportError:
        pass
    except Exception as e:
        result["validation_error"] = str(e)

    return result


def print_summary(results):
    """Print a formatted summary table."""
    print("\n" + "=" * 100)
    print("NEB-TS Results Summary")
    print("=" * 100)

    # Header
    print(f"{'System':<45} {'Status':<10} {'Fwd':<8} {'Rev':<8} "
          f"{'R:Nuc-P':<8} {'P:Nuc-P':<8} {'P:LG-P':<8} {'Flags'}")
    print("-" * 100)

    for r in sorted(results, key=lambda x: x["name"]):
        name = r["name"][:44]
        status = r["status"]

        fwd = f"{r['barrier_fwd_kcal']:.1f}" if r.get("barrier_fwd_kcal") is not None else "---"
        rev = f"{r['barrier_rev_kcal']:.1f}" if r.get("barrier_rev_kcal") is not None else "---"

        r_nuc = f"{r['reactant_nuc_P']}" if r.get("reactant_nuc_P") else "---"
        p_nuc = f"{r['product_nuc_P']}" if r.get("product_nuc_P") else "---"
        p_lg = f"{r['product_lg_P']}" if r.get("product_lg_P") else "---"

        flags = []
        if r.get("barrier_fwd_kcal") is not None and r["barrier_fwd_kcal"] < 0:
            flags.append("NEG_BARRIER")
        if r.get("pentacoordinate"):
            flags.append("PENTA")
        if r.get("barrier_fwd_kcal") is not None and r["barrier_fwd_kcal"] > 40:
            flags.append("HIGH")
        if not r.get("has_cif"):
            flags.append("NO_CIF")
        flag_str = ",".join(flags)

        print(f"{name:<45} {status:<10} {fwd:<8} {rev:<8} "
              f"{r_nuc:<8} {p_nuc:<8} {p_lg:<8} {flag_str}")

    print("=" * 100)

    # Summary stats
    complete = [r for r in results if r["status"] == "complete"]
    penta = [r for r in results if r.get("pentacoordinate")]
    neg = [r for r in results if r.get("barrier_fwd_kcal") is not None and r["barrier_fwd_kcal"] < 0]

    print(f"\nTotal: {len(results)} systems | Complete: {len(complete)} | "
          f"Pentacoordinate: {len(penta)} | Negative barrier: {len(neg)}")

    if penta:
        print("\nSystems needing second-step NEB (pentacoordinate intermediate):")
        for r in penta:
            print(f"  - {r['name']} (product LG-P = {r.get('product_lg_P', '?')} A)")


def main():
    parser = argparse.ArgumentParser(description="Analyze NEB-TS results")
    parser.add_argument("path", help="Output directory or parent directory")
    parser.add_argument("--all", action="store_true",
                        help="Analyze all subdirectories")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args()

    path = Path(args.path)

    if args.all:
        dirs = sorted([d for d in path.iterdir() if d.is_dir()])
    else:
        dirs = [path]

    results = []
    for d in dirs:
        if d.name in ("technical", "logs", ".xtb_charges", ".xtb_formal", ".xtb_cif"):
            continue
        r = analyze_single_run(d)
        results.append(r)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_summary(results)


if __name__ == "__main__":
    main()
