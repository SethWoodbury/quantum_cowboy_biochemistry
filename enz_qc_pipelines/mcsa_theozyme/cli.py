"""CLI entry: ``python -m enz_qc_pipelines.mcsa_theozyme``.

Launches the M-CSA-driven theozyme pipeline. M-CSA entry ID is the
key input; substrate SMILES resolves R-groups in the mechanism
diagrams.

Examples (M-CSA test cases the user flagged):
    # PTE — Zn/Zn + KCX, paraoxon hydrolysis
    python -m enz_qc_pipelines.mcsa_theozyme \\
        --mcsa-id 159 \\
        --substrate 'CCOP(=O)(OCC)Oc1ccc([N+](=O)[O-])cc1' \\
        --product   'CCOP(=O)(OCC)O.Oc1ccc([N+](=O)[O-])cc1' \\
        --outdir runs/pte_paraoxon_159

    # Anthrax LF — HExxH motif test
    python -m enz_qc_pipelines.mcsa_theozyme \\
        --mcsa-id 641 --tier2-mode motif --outdir runs/anthrax_lf_641

    # AChE — Ser-His-Glu triad
    python -m enz_qc_pipelines.mcsa_theozyme \\
        --mcsa-id 922 --outdir runs/ache_922
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="enz_qc_pipelines.mcsa_theozyme",
        description="M-CSA-driven theozyme generation (AME-benchmark feeder).",
    )
    p.add_argument("--mcsa-id", required=True, type=int,
                   help="M-CSA entry ID (e.g. 159 for PTE)")
    p.add_argument("--substrate", default=None,
                   help="Concrete substrate SMILES (resolves R-groups). "
                        "Optional — falls back to M-CSA's ChEBI lookup.")
    p.add_argument("--product", default=None,
                   help="Concrete product SMILES (optional)")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--refresh-mcsa", action="store_true",
                   help="Force re-fetch from M-CSA API (ignore cache)")
    p.add_argument("--tier2-mode", choices=["distance", "motif", "both", "skip"],
                   default="both")
    p.add_argument("--tier2-radius", type=float, default=6.0)
    p.add_argument("--vacuum-ts-tool",
                   choices=["scine", "autode", "molecularGSM"], default="scine")
    p.add_argument("--path-refind-tool",
                   choices=["pygsm-se", "molecularGSM-ssm", "scine-nt"],
                   default="pygsm-se")
    p.add_argument("--polish-tool",
                   choices=["sella", "pysisyphus-rsirfo", "scine-tsopt"],
                   default="sella")
    p.add_argument("--mace-model", default="mace-polar",
                   help="Key from quantum_engine.site.MACE_MODELS")
    p.add_argument("--constraint-mode",
                   choices=["ca-only", "backbone", "ca-restrained", "none"],
                   default="ca-only")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("enz_qc_pipelines.mcsa_theozyme")

    from quantum_engine.pipelines import Context
    from enz_qc_pipelines.mcsa_theozyme.orchestrator import (
        build_mcsa_theozyme_pipeline,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    ctx = Context(
        atoms=None,                       # Stage 0/2 fetch + crop
        calc=None,
        outdir=args.outdir,
        metadata={
            "mcsa_id": args.mcsa_id,
            "user_substrate": args.substrate,
            "user_product": args.product,
            "tier2_mode": args.tier2_mode,
            "tier2_radius": args.tier2_radius,
            "constraint_mode": args.constraint_mode,
        },
    )
    pipeline = build_mcsa_theozyme_pipeline(
        mcsa_id=args.mcsa_id,
        user_substrate=args.substrate,
        user_product=args.product,
        tier2_mode=args.tier2_mode,
        tier2_radius_A=args.tier2_radius,
        vacuum_ts_tool=args.vacuum_ts_tool,
        path_refind_tool=args.path_refind_tool,
        polish_tool=args.polish_tool,
        mace_model=args.mace_model,
        constraint_mode=args.constraint_mode,
    )
    log.info(f"Running mcsa_theozyme pipeline — {len(pipeline.steps)} stages, "
             f"M-CSA {args.mcsa_id}")
    pipeline.run(ctx)
    log.info(f"Done. Outputs under {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
