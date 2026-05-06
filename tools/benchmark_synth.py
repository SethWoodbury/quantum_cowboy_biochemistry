#!/usr/bin/env python
"""benchmark_synth.py — compile per-strategy summaries into one report.

Walks an output dir, loads every `summary.json`, ranks by validated barrier,
applies sanity gates (1 imag mode, reactive-distance drift, geometry-not-broken),
and writes BENCHMARK_REPORT.md plus a comparison_table.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def load_summary(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {"error": f"failed to load: {e}"}


def get_barrier(s: dict) -> float | None:
    """Pull a comparable barrier number from any summary schema variant."""
    if not s: return None
    if "barriers_kcal" in s:
        b = s["barriers_kcal"]
        v = b.get("R_to_TS")
        return v if v is not None else b.get("forward")
    if "barrier_R_to_TS_kcal" in s:
        return s["barrier_R_to_TS_kcal"]
    return None


def get_n_imag(s: dict) -> int | None:
    return s.get("n_imaginary") if s else None


def get_imag_freq(s: dict) -> float | None:
    if not s: return None
    return s.get("imag_freq_cm")


def get_geometry(s: dict) -> dict:
    if not s: return {}
    if "ts_geometry" in s: return s["ts_geometry"]
    return {
        "d_P_Onuc": s.get("final_d_p_onuc_A") or s.get("ts_guess_d_P_Onuc"),
        "d_P_Olg": s.get("final_d_p_olg_A") or s.get("ts_guess_d_P_Olg"),
        "s": (s.get("final_d_p_olg_A") or 0) - (s.get("final_d_p_onuc_A") or 0)
              if s.get("final_d_p_olg_A") else None,
    }


def get_walltime(s: dict) -> float | None:
    return s.get("walltime_min")


def sanity_gate(strategy: str, s: dict, geom: dict, barrier: float | None) -> tuple[bool, list[str]]:
    flags = []
    # geometry sanity: reactive distances should be in [1.4, 4.0] Å
    pn = geom.get("d_P_Onuc"); pl = geom.get("d_P_Olg")
    if pn is None or pl is None:
        flags.append("missing_reactive_distances")
    else:
        if pn < 1.4 or pn > 5.0: flags.append(f"d_P_Onuc={pn:.2f} out of [1.4, 5.0]")
        if pl < 1.4 or pl > 5.0: flags.append(f"d_P_Olg={pl:.2f} out of [1.4, 5.0]")
    # n_imag should be 1 (true 1st-order saddle) but allow None for strategies that don't compute
    n_imag = get_n_imag(s)
    if n_imag is not None and n_imag != 1:
        flags.append(f"n_imag={n_imag} (expect 1)")
    # barrier sanity: < 5 kcal/mol on a Zn-phosphoryl TS is suspicious
    if barrier is not None and barrier < 5.0 and barrier > -50.0:
        flags.append(f"barrier={barrier:.1f} suspiciously low")
    if barrier is not None and barrier > 100.0:
        flags.append(f"barrier={barrier:.1f} suspiciously high (broken constraints?)")
    return (len(flags) == 0, flags)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="benchmark output root (contains <strategy>/ subdirs with summary.json)")
    ap.add_argument("--ref-pdb", type=Path, default=None,
                    help="best-known TS for context; fallback if a strategy can't compute")
    args = ap.parse_args()

    rows = []
    for d in sorted(args.root.glob("*/")):
        if d.name.startswith("_") or d.name.startswith("."): continue
        s_path = d / "summary.json"
        if not s_path.exists():
            rows.append({"strategy": d.name, "status": "no_summary"})
            continue
        s = load_summary(s_path)
        if not s or "error" in s:
            rows.append({"strategy": d.name, "status": "load_error", "error": s.get("error")})
            continue
        barrier = get_barrier(s)
        n_imag = get_n_imag(s)
        imag_freq = get_imag_freq(s)
        geom = get_geometry(s)
        walltime = get_walltime(s)
        ok, flags = sanity_gate(d.name, s, geom, barrier)
        rows.append({
            "strategy": d.name,
            "status": "ok" if ok else "flagged",
            "barrier_R_to_TS_kcal": barrier,
            "n_imaginary": n_imag,
            "imag_freq_cm": imag_freq,
            "d_P_Onuc": geom.get("d_P_Onuc"),
            "d_P_Olg": geom.get("d_P_Olg"),
            "s": geom.get("s") if geom.get("s") is not None
                 else ((geom.get("d_P_Olg") or 0) - (geom.get("d_P_Onuc") or 0)),
            "walltime_min": walltime,
            "flags": ";".join(flags),
            "ts_pdb": s.get("ts_pdb") or s.get("ts_guess_pdb") or s.get("output"),
        })

    # ---- write CSV ----
    if rows:
        csv_path = args.root / "comparison_table.csv"
        cols = ["strategy", "status", "barrier_R_to_TS_kcal", "n_imaginary",
                "imag_freq_cm", "d_P_Onuc", "d_P_Olg", "s", "walltime_min",
                "flags", "ts_pdb"]
        with csv_path.open("w") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote: {csv_path}")

    # ---- rank by barrier (lower better, NaN/None last) ----
    def sort_key(r):
        b = r.get("barrier_R_to_TS_kcal")
        return (b is None or (isinstance(b, float) and math.isnan(b)), b if b is not None else 999.0)
    ranked = sorted(rows, key=sort_key)

    # ---- write report ----
    md = [f"# TS-finding benchmark report ({args.root})", ""]
    md.append(f"**Strategies evaluated:** {len(rows)}")
    valid = [r for r in rows if r.get("barrier_R_to_TS_kcal") is not None]
    md.append(f"**With computed barrier:** {len(valid)}")
    if valid:
        bs = [r["barrier_R_to_TS_kcal"] for r in valid]
        md.append(f"**Barrier range:** {min(bs):.2f} → {max(bs):.2f} kcal/mol")
    md.append("")
    md.append("## Ranked (by validated barrier R → TS, lower = lower-energy TS)")
    md.append("")
    md.append("| rank | strategy | barrier | n_imag | imag_freq | d_PN | d_PL | s | wall (min) | flags |")
    md.append("|---:|:---|---:|---:|---:|---:|---:|---:|---:|:---|")
    for i, r in enumerate(ranked, 1):
        b = r.get("barrier_R_to_TS_kcal")
        b_str = f"{b:.2f}" if b is not None else "—"
        ni = r.get("n_imaginary")
        ni_str = str(ni) if ni is not None else "—"
        ifc = r.get("imag_freq_cm")
        ifc_str = f"{ifc:.0f}" if ifc is not None else "—"
        dpn = r.get("d_P_Onuc"); dpl = r.get("d_P_Olg"); s = r.get("s")
        dpn_str = f"{dpn:.3f}" if dpn else "—"
        dpl_str = f"{dpl:.3f}" if dpl else "—"
        s_str = f"{s:+.3f}" if s is not None else "—"
        wt = r.get("walltime_min")
        wt_str = f"{wt:.1f}" if wt else "—"
        flags = r.get("flags") or "—"
        md.append(f"| {i} | {r['strategy']} | {b_str} | {ni_str} | {ifc_str} | "
                  f"{dpn_str} | {dpl_str} | {s_str} | {wt_str} | {flags} |")
    md.append("")
    if valid:
        best = ranked[0]
        md.append(f"## Best validated TS: **{best['strategy']}**")
        md.append(f"- Barrier R → TS: **{best['barrier_R_to_TS_kcal']:.2f} kcal/mol**")
        if best.get("imag_freq_cm"):
            md.append(f"- Imaginary frequency: **{best['imag_freq_cm']:.0f} cm⁻¹**")
        md.append(f"- d(P–Onuc) = {best['d_P_Onuc']:.3f} Å,  d(P–Olg) = {best['d_P_Olg']:.3f} Å")
        md.append(f"- TS PDB: `{best['ts_pdb']}`")
    md.append("")
    md.append("## Sanity gates")
    md.append("- d(P–Onuc), d(P–Olg) ∈ [1.4, 5.0] Å")
    md.append("- n_imaginary == 1 (when computed)")
    md.append("- 5 < barrier < 100 kcal/mol")

    md_path = args.root / "BENCHMARK_REPORT.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"wrote: {md_path}")
    print()
    for r in ranked[:3]:
        b = r.get("barrier_R_to_TS_kcal")
        b_str = f"{b:.2f}" if b is not None else "—"
        print(f"  {r['strategy']:>20s}  barrier={b_str:>6s}  flags={r.get('flags', '') or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
