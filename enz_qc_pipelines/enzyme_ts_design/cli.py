"""Command-line entry: ``python -m enz_qc_pipelines.enzyme_ts_design``.

Wires the generic 8-stage TS-design pipeline. Reads SMILES + active
site PDB from CLI args; writes per-TS .cif files to ``--outdir``.

Example:
    python -m enz_qc_pipelines.enzyme_ts_design \\
        --reactant 'O=P(O)(OC(=O)C)c1ccccc1[N+](=O)[O-]' \\
        --product  'OP(=O)(O)O.OC(=O)Cc1ccccc1[N+](=O)[O-]' \\
        --active-site cropped_pte_1hzy.pdb \\
        --constraint-mode ca-only \\
        --vacuum-ts-tool autode \\
        --path-refind-tool pygsm-se \\
        --polish-tool sella \\
        --outdir runs/pte_paraoxon
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="enz_qc_pipelines.enzyme_ts_design",
        description="Generic enzyme TS-design pipeline (SMILES + cropped active site → docked TS .cif).",
    )
    p.add_argument("--reactant", required=True, help="SMILES of reactant(s)")
    p.add_argument("--product", required=True, help="SMILES of product(s)")
    p.add_argument("--active-site", required=True, type=Path,
                   help="Path to cropped active-site PDB")
    p.add_argument("--outdir", required=True, type=Path,
                   help="Output directory for per-stage results + final .cif")
    p.add_argument("--constraint-mode",
                   choices=["ca-only", "backbone", "backbone-water", "ca-restrained", "none"],
                   default="ca-only",
                   help="Atoms to fix during MD/opt — see docs/plans/enzyme_ts_design.md")
    p.add_argument("--vacuum-ts-tool",
                   choices=["autode", "scine", "molecularGSM", "pygsm", "auto"],
                   default="auto")
    p.add_argument("--path-refind-tool",
                   choices=["pygsm-se", "pysisyphus-neb", "scine-bspline"],
                   default="pygsm-se")
    p.add_argument("--polish-tool",
                   choices=["sella", "pysisyphus-rsirfo", "scine-tsopt"],
                   default="sella")
    p.add_argument("--mace-model", default="mace-polar",
                   help="Key from quantum_engine.site.MACE_MODELS")
    p.add_argument("--tier", choices=["1", "2"], default="1",
                   help="1=catalytic only; 2=+distance/motif residues")
    p.add_argument("--tier2-radius", type=float, default=6.0,
                   help="(tier=2) heavy-atom shell radius in Å")
    p.add_argument("--tier2-mode", choices=["distance", "motif", "both"], default="both")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("enz_qc_pipelines.enzyme_ts_design")

    from ase.io import read as ase_read
    from quantum_engine.pipelines import Context
    from enz_qc_pipelines.enzyme_ts_design.orchestrator import (
        build_enzyme_ts_design_pipeline,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    atoms = ase_read(str(args.active_site))
    ctx = Context(
        atoms=atoms,
        calc=None,
        outdir=args.outdir,
        metadata={
            "input_pdb": str(args.active_site),
            "reactant_smiles": args.reactant,
            "product_smiles": args.product,
            "constraint_mode": args.constraint_mode,
            "tier": int(args.tier),
            "tier2_radius": args.tier2_radius,
            "tier2_mode": args.tier2_mode,
        },
    )
    pipeline = build_enzyme_ts_design_pipeline(
        reactant_smiles=args.reactant,
        product_smiles=args.product,
        constraint_mode=args.constraint_mode,
        vacuum_ts_tool=args.vacuum_ts_tool,
        path_refind_tool=args.path_refind_tool,
        polish_tool=args.polish_tool,
        mace_model=args.mace_model,
    )
    log.info(f"Running enzyme_ts_design — {len(pipeline.steps)} stages")
    pipeline.run(ctx)
    log.info(f"Done. Outputs under {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
