"""End-to-end validation runner for M-CSA 159 (phosphotriesterase, PTE).

Pulls 159 from M-CSA, fetches 1hzy from RCSB, crops the active site,
expands tier-2 (distance + motif), and emits review-ready paths so a
human can open them in PyMOL and judge chemical plausibility.

If --vacuum-ts is passed, ALSO runs enzyme_ts_design Stages 1-2 on
paraoxon hydrolysis (paraoxon + H2O → diethyl phosphate +
4-nitrophenol) and prints the vacuum TS xyz path + barrier.

Usage:
    python tools/run_pte_159_validation.py
    python tools/run_pte_159_validation.py --vacuum-ts
    python tools/run_pte_159_validation.py --outdir runs/pte_review_$(date +%Y%m%d)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PARAOXON_SMI = "CCOP(=O)(OCC)Oc1ccc([N+](=O)[O-])cc1.O"
PRODUCTS_SMI = "CCOP(=O)(OCC)O.Oc1ccc([N+](=O)[O-])cc1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("runs/pte_159_validation"))
    p.add_argument("--vacuum-ts", action="store_true",
                   help="Also run enzyme_ts_design Stages 1-2 (autodE, slow ~60s)")
    p.add_argument("--mcsa-id", type=int, default=159)
    p.add_argument("--tier2-radius", type=float, default=6.0)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def run_mcsa_stages_0_3(args) -> dict:
    """Run mcsa_theozyme Stages 0-3 with concrete paraoxon substrate."""
    from quantum_engine.pipelines import Context, Pipeline
    from enz_qc_pipelines.mcsa_theozyme.orchestrator import (
        FetchMCSAEntry, ResolveSubstrateSMILES,
        CropActiveSiteFromPDB, Tier2ResidueExpansion,
    )
    outdir = args.outdir / "mcsa_theozyme"
    outdir.mkdir(parents=True, exist_ok=True)
    ctx = Context(atoms=None, calc=None, outdir=outdir,
                  metadata={"mcsa_id": args.mcsa_id})
    Pipeline([
        FetchMCSAEntry(mcsa_id=args.mcsa_id),
        ResolveSubstrateSMILES(
            user_substrate=PARAOXON_SMI,
            user_product=PRODUCTS_SMI,
        ),
        CropActiveSiteFromPDB(),
        Tier2ResidueExpansion(mode="both", radius_A=args.tier2_radius),
    ], write_summary=True).run(ctx)
    return ctx.history


def run_vacuum_ts(args) -> dict:
    """Run enzyme_ts_design Stages 1-2 on paraoxon hydrolysis."""
    from ase import Atoms
    from quantum_engine.pipelines import Context, Pipeline
    from enz_qc_pipelines.enzyme_ts_design.orchestrator import (
        ParseReaction, VacuumTSSearch,
    )
    outdir = args.outdir / "vacuum_ts"
    outdir.mkdir(parents=True, exist_ok=True)
    ctx = Context(atoms=Atoms(), calc=None, outdir=outdir, metadata={})
    Pipeline([
        ParseReaction(reactant_smiles=PARAOXON_SMI,
                      product_smiles=PRODUCTS_SMI),
        VacuumTSSearch(tool="autode", qm_method="g-xtb"),
    ], write_summary=True).run(ctx)
    return ctx.history


def print_review_summary(mcsa_hist: dict, vac_hist: dict | None, outdir: Path):
    print()
    print("=" * 78)
    print(f"  PTE (M-CSA 159) validation — review summary")
    print("=" * 78)

    s0 = mcsa_hist["fetch_mcsa"]
    s1 = mcsa_hist["resolve_smiles"]
    s2 = mcsa_hist["crop_active_site"]
    s3 = mcsa_hist["tier2_expansion"]
    print(f"\nM-CSA fetch:")
    print(f"  enzyme:           {s0.outputs['enzyme_name']}")
    print(f"  EC:               {s0.outputs['ec']}")
    print(f"  reference PDB:    {s0.outputs['reference_pdb']}")
    print(f"  catalytic res #:  {s0.outputs['n_catalytic_residues']}")
    print(f"  PTM residues:     {s0.outputs['ptm_residues']}")
    print(f"  cofactors flagged:{s0.outputs['cofactors']}")

    print(f"\nSMILES resolved:")
    print(f"  reactant:  {s1.outputs['reactant_smiles']}")
    print(f"  product:   {s1.outputs['product_smiles']}")
    print(f"  user-overrides applied: {s1.outputs['n_user_overrides']}")
    if s1.outputs["unresolved_chebi_ids"]:
        print(f"  unresolved (R-group SMILES): {s1.outputs['unresolved_chebi_ids']}")

    print(f"\nTier-1 active-site crop  (catalytic residues only):")
    print(f"  PATH ▶ {s2.outputs['cropped_pdb']}")
    print(f"  atoms:           {s2.outputs['n_atoms']}")
    print(f"  residues:        {s2.outputs['n_residues']}")
    print(f"  cofactors found: {s2.outputs['cofactors']} (PTE expects ZN/ZN)")

    print(f"\nTier-2 active-site crop  (mode='both', radius={s3.outputs.get('radius_A', 6.0)} Å):")
    print(f"  PATH ▶ {s3.outputs['cropped_pdb']}")
    print(f"  atoms:           {s3.outputs['n_atoms']}")
    print(f"  residues total:  {s3.outputs['n_residues_total']}")
    print(f"  added (distance):{s3.outputs['n_added_distance']}")
    print(f"  added (motif):   {s3.outputs['n_added_motif']}")
    added = s3.outputs.get("added_residues", [])
    if added:
        print(f"  added residues:  {added[:15]}{' …' if len(added) > 15 else ''}")

    if vac_hist:
        sp = vac_hist["parse_reaction"]
        sv = vac_hist["vacuum_ts"]
        print(f"\nVacuum TS (paraoxon + H2O → diethyl phosphate + 4-nitrophenol):")
        print(f"  PATH ▶ {sv.outputs['vacuum_ts_xyz']}")
        if "vacuum_barrier_kcal" in sv.outputs and sv.outputs["vacuum_barrier_kcal"] is not None:
            print(f"  barrier:         {sv.outputs['vacuum_barrier_kcal']:.2f} kcal/mol")
        else:
            print(f"  barrier:         (not reported)")
        # Bond changes from Stage 1
        bf = sp.outputs.get("bonds_formed", []) if hasattr(sp, "outputs") else []
        bb = sp.outputs.get("bonds_broken", []) if hasattr(sp, "outputs") else []
        net_charge = sp.outputs.get("net_charge", "?") if hasattr(sp, "outputs") else "?"
        print(f"  bonds formed:    {bf}")
        print(f"  bonds broken:    {bb}")
        print(f"  net charge:      {net_charge}")
        # extra dict on the StepResult may carry barrier + ts_source
        extra = getattr(sv, "extra", {}) or {}
        if extra:
            print(f"  extra:           {extra}")

    # Overall paths block — the user's "open in PyMOL" cheat sheet
    print()
    print("=" * 78)
    print("  PyMOL review:")
    print("=" * 78)
    print(f"  pymol {s2.outputs['cropped_pdb']}    # tier-1 active site")
    print(f"  pymol {s3.outputs['cropped_pdb']}    # tier-2 active site")
    if vac_hist:
        sv = vac_hist["vacuum_ts"]
        print(f"  pymol {sv.outputs['vacuum_ts_xyz']}    # vacuum TS")
    print(f"\nAll outputs under: {outdir}/")
    print("=" * 78)
    print()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"PTE 159 validation → {args.outdir}")
    mcsa_hist = run_mcsa_stages_0_3(args)
    vac_hist = run_vacuum_ts(args) if args.vacuum_ts else None
    print_review_summary(mcsa_hist, vac_hist, args.outdir)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main())
