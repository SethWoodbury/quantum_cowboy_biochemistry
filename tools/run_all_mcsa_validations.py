"""End-to-end M-CSA validation harness — runs the mcsa_theozyme pipeline
on all 5 priority test cases (159, 376, 641, 900, 922) and emits a
single review-ready report at <outdir>/REVIEW.md.

Tolerates per-stage NotImplementedError (so Stages 5-8 stubs don't
block the overall run) and per-entry exceptions (so a network hiccup
on one entry doesn't kill the rest of the harness).

Substrate / product SMILES are concrete, neutral-water mechanism
versions chosen for autodE compatibility (M-CSA's native compound
entries are R-group schematics that need user override anyway —
documented in docs/plans/mcsa_theozyme.md).

Usage:
    python tools/run_all_mcsa_validations.py
    python tools/run_all_mcsa_validations.py --mcsa-ids 159,376
    python tools/run_all_mcsa_validations.py --max-stage 4 --no-vacuum-ts
    python tools/run_all_mcsa_validations.py --container quantum_chem
    python tools/run_all_mcsa_validations.py --outdir runs/full_validation
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────
# Test-case table — concrete substrates for each M-CSA entry
# ─────────────────────────────────────────────────────────────────────
# All neutral-water mechanism (autodE-friendly, no charges to relay).
# When the autodE charge-relay bug (task #68) is fixed, the harness
# will optionally also run hydroxide-attack variants (net charge -1)
# for the metallohydrolases.

@dataclass
class TestCase:
    mcsa_id: int
    label: str
    substrate_smiles: str
    product_smiles: str
    notes: str = ""


TEST_CASES: list[TestCase] = [
    TestCase(
        mcsa_id=159,
        label="Phosphotriesterase (Zn/Zn + KCX)",
        substrate_smiles="CCOP(=O)(OCC)Oc1ccc([N+](=O)[O-])cc1.O",
        product_smiles="CCOP(=O)(OCC)O.Oc1ccc([N+](=O)[O-])cc1",
        notes="paraoxon + H2O → diethyl phosphate + 4-nitrophenol (neutral form)",
    ),
    TestCase(
        mcsa_id=376,
        label="Adenosine deaminase (Zn)",
        substrate_smiles="Nc1ncnc2c1ncn2[C@@H]3O[C@H](CO)[C@@H](O)[C@H]3O.O",
        product_smiles="O=c1[nH]cnc2c1ncn2[C@@H]3O[C@H](CO)[C@@H](O)[C@H]3O.N",
        notes="adenosine + H2O → inosine + NH3",
    ),
    TestCase(
        mcsa_id=641,
        label="Anthrax lethal factor (Zn, HExxH)",
        substrate_smiles="CC(=O)NC(C)C(=O)NCC(=O)O.O",
        product_smiles="CC(=O)NC(C)C(=O)O.NCC(=O)O",
        notes="Ac-Ala-Gly-OH + H2O → Ac-Ala-OH + Gly-OH (peptide bond hydrolysis model)",
    ),
    TestCase(
        mcsa_id=900,
        label="PNB esterase (Ser-His-Glu)",
        substrate_smiles="CC(=O)Oc1ccc([N+](=O)[O-])cc1.O",
        product_smiles="CC(=O)O.Oc1ccc([N+](=O)[O-])cc1",
        notes="p-nitrophenyl acetate + H2O → acetate + 4-nitrophenol",
    ),
    TestCase(
        mcsa_id=922,
        label="Acetylcholinesterase (Ser-His-Glu)",
        substrate_smiles="CC(=O)OCC[N+](C)(C)C.O",
        product_smiles="CC(=O)O.OCC[N+](C)(C)C",
        notes="acetylcholine + H2O → acetate + choline",
    ),
]


# ─────────────────────────────────────────────────────────────────────
# Per-entry runner
# ─────────────────────────────────────────────────────────────────────

@dataclass
class EntryResult:
    """Outcome of one M-CSA entry's pipeline run."""
    case: TestCase
    outdir: Path
    stages_completed: list[str] = field(default_factory=list)
    stages_skipped: list[str] = field(default_factory=list)
    stages_failed: dict[str, str] = field(default_factory=dict)  # stage_name → reason
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)  # stage_name → outputs
    overall_error: str | None = None
    wall_seconds: float = 0.0


def run_one_entry(case: TestCase, root: Path, *,
                  max_stage: int = 99,
                  include_vacuum_ts: bool = True,
                  log: logging.Logger) -> EntryResult:
    """Run the mcsa_theozyme pipeline on one entry; tolerate per-stage
    NIE and per-entry exceptions."""
    from quantum_engine.pipelines import Context, Pipeline
    from enz_qc_pipelines.mcsa_theozyme.orchestrator import (
        FetchMCSAEntry, ResolveSubstrateSMILES,
        CropActiveSiteFromPDB, Tier2ResidueExpansion,
        PerStepVacuumTS,
        IterativeRefineWithPTMs,
        InProteinPathRefindFromArrows,
        HighResTSPolish,
        WriteTheozyme,
    )

    outdir = root / f"{case.mcsa_id:03d}_{case.label.split()[0].lower()}"
    outdir.mkdir(parents=True, exist_ok=True)
    result = EntryResult(case=case, outdir=outdir)
    t0 = time.time()

    log.info(f"━━━ M-CSA {case.mcsa_id} — {case.label} ━━━")
    log.info(f"  workdir: {outdir}")
    log.info(f"  reactant: {case.substrate_smiles}")
    log.info(f"  product:  {case.product_smiles}")

    # Stage list — we instantiate only the stages up to max_stage; later
    # stages are recorded as "skipped" in the result.
    all_stages = [
        ("fetch_mcsa", FetchMCSAEntry(mcsa_id=case.mcsa_id)),
        ("resolve_smiles", ResolveSubstrateSMILES(
            user_substrate=case.substrate_smiles,
            user_product=case.product_smiles,
        )),
        ("crop_active_site", CropActiveSiteFromPDB()),
        ("tier2_expansion", Tier2ResidueExpansion(mode="both", radius_A=6.0)),
        ("per_step_vacuum_ts", PerStepVacuumTS(tool="autode", qm_method="g-xtb",
                                               overall=True)),
        ("iterative_refine", IterativeRefineWithPTMs()),
        ("path_refind_from_arrows", InProteinPathRefindFromArrows()),
        ("polish_ts", HighResTSPolish()),
        ("write_theozyme", WriteTheozyme()),
    ]

    if not include_vacuum_ts:
        all_stages = [s for s in all_stages if s[0] != "per_step_vacuum_ts"]

    # Filter by max_stage (1-indexed: stage 1 = fetch, stage 9 = write)
    selected = all_stages[:max_stage]
    for name, _ in all_stages[max_stage:]:
        result.stages_skipped.append(f"{name} (--max-stage cap)")

    ctx = Context(atoms=None, calc=None, outdir=outdir, metadata={})

    for stage_name, step in selected:
        log.info(f"  ▶ {stage_name}")
        try:
            sr = step.run(ctx)
            ctx.history[stage_name] = sr
            result.stages_completed.append(stage_name)
            if sr.outputs:
                result.outputs[stage_name] = dict(sr.outputs)
            log.info(f"    ✓ {stage_name} — outputs: {list(sr.outputs.keys()) if sr.outputs else 'none'}")
        except NotImplementedError as e:
            short = str(e).splitlines()[0] if str(e) else "stub"
            result.stages_skipped.append(f"{stage_name} (not yet implemented: {short[:80]})")
            log.warning(f"    ⊘ {stage_name} skipped — NotImplementedError: {short[:100]}")
        except Exception as e:
            tb = traceback.format_exc(limit=3)
            result.stages_failed[stage_name] = f"{type(e).__name__}: {e}"
            log.error(f"    ✗ {stage_name} FAILED — {type(e).__name__}: {e}")
            log.debug(tb)
            # Decide whether to abort the entry or continue:
            # - For Stages 0-3 a failure usually means we can't do later stages
            #   (no PDB, no SMILES, no metadata).
            # - For Stages 4+ we can still report what we got.
            stage_idx = next(i for i, (n, _) in enumerate(all_stages) if n == stage_name)
            if stage_idx <= 3:
                log.error(f"    Stage 0-3 failure aborts entry {case.mcsa_id}.")
                break

    result.wall_seconds = time.time() - t0
    return result


# ─────────────────────────────────────────────────────────────────────
# Markdown report generator
# ─────────────────────────────────────────────────────────────────────

def write_review_md(results: list[EntryResult], outdir: Path) -> Path:
    """Emit a single markdown review report at <outdir>/REVIEW.md."""
    out = outdir / "REVIEW.md"
    lines: list[str] = []
    lines.append(f"# M-CSA validation report — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"Run dir: `{outdir}`\n")

    # Summary table
    lines.append("\n## Summary\n")
    lines.append("| ID | Enzyme | Stages OK | Stages skipped | Stages failed | Wall (s) |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        ok = len(r.stages_completed)
        sk = len(r.stages_skipped)
        fl = len(r.stages_failed)
        lines.append(
            f"| {r.case.mcsa_id} | {r.case.label} | {ok} | {sk} | {fl} | {r.wall_seconds:.1f} |"
        )

    # Per-entry detail
    for r in results:
        lines.append(f"\n---\n\n## M-CSA {r.case.mcsa_id} — {r.case.label}\n")
        lines.append(f"**Reaction notes:** {r.case.notes}\n")
        lines.append(f"**Reactant SMILES:** `{r.case.substrate_smiles}`")
        lines.append(f"**Product SMILES:** `{r.case.product_smiles}`")
        lines.append(f"\n**Workdir:** `{r.outdir}`\n")

        if r.stages_completed:
            lines.append("\n### Stages completed\n")
            for s in r.stages_completed:
                outs = r.outputs.get(s, {})
                paths = [v for v in outs.values() if isinstance(v, str) and ("/" in v or "\\" in v)]
                if paths:
                    lines.append(f"- **{s}** — outputs: " + ", ".join(f"`{p}`" for p in paths[:4]))
                else:
                    keys = list(outs.keys())[:6]
                    lines.append(f"- **{s}** — keys: {keys}")

        if r.stages_skipped:
            lines.append("\n### Stages skipped\n")
            for s in r.stages_skipped:
                lines.append(f"- {s}")

        if r.stages_failed:
            lines.append("\n### Stages failed\n")
            for s, reason in r.stages_failed.items():
                lines.append(f"- **{s}**: `{reason}`")

        # PyMOL cheat sheet
        pymol_paths = []
        for s in ("crop_active_site", "tier2_expansion"):
            if s in r.outputs and "cropped_pdb" in r.outputs[s]:
                pymol_paths.append(r.outputs[s]["cropped_pdb"])
        for s in ("per_step_vacuum_ts",):
            if s in r.outputs:
                for k in ("vacuum_ts_pdb", "vacuum_ts_xyz"):
                    if k in r.outputs[s]:
                        pymol_paths.append(r.outputs[s][k])
        for s in ("iterative_refine",):
            if s in r.outputs and "refined_pdb" in r.outputs[s]:
                pymol_paths.append(r.outputs[s]["refined_pdb"])
        for s in ("write_theozyme",):
            if s in r.outputs and "theozyme_cif" in r.outputs[s]:
                pymol_paths.append(r.outputs[s]["theozyme_cif"])

        if pymol_paths:
            lines.append("\n### Open in PyMOL\n```bash")
            for p in pymol_paths:
                lines.append(f"pymol {p}")
            lines.append("```")

        if r.overall_error:
            lines.append(f"\n**Entry-level error:** `{r.overall_error}`\n")

    out.write_text("\n".join(lines) + "\n")
    return out


def write_summary_json(results: list[EntryResult], outdir: Path) -> Path:
    """Machine-readable counterpart to REVIEW.md."""
    out = outdir / "summary.json"
    payload = []
    for r in results:
        payload.append({
            "mcsa_id": r.case.mcsa_id,
            "label": r.case.label,
            "substrate_smiles": r.case.substrate_smiles,
            "product_smiles": r.case.product_smiles,
            "notes": r.case.notes,
            "outdir": str(r.outdir),
            "stages_completed": r.stages_completed,
            "stages_skipped": r.stages_skipped,
            "stages_failed": r.stages_failed,
            "outputs": {k: {kk: str(vv) for kk, vv in v.items()}
                        for k, v in r.outputs.items()},
            "wall_seconds": r.wall_seconds,
            "overall_error": r.overall_error,
        })
    out.write_text(json.dumps(payload, indent=2))
    return out


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    default_out = (
        f"runs/all_mcsa_validations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    p.add_argument("--outdir", type=Path, default=Path(default_out),
                   help="Run dir (default: runs/all_mcsa_validations_<timestamp>)")
    p.add_argument("--mcsa-ids", default=None,
                   help="Comma-separated subset (e.g. '159,376'). Default: all 5.")
    p.add_argument("--max-stage", type=int, default=99,
                   help="Stop after stage N (1=fetch, 5=vacuum_ts, 9=write_theozyme)")
    p.add_argument("--no-vacuum-ts", action="store_true",
                   help="Skip Stage 4 (vacuum TS); useful when autodE is broken")
    p.add_argument("--container", default=None,
                   help="Apptainer container key (e.g. 'quantum_chem'). "
                        "If set, this script must be invoked from a wrapper "
                        "that already wraps in apptainer exec — we don't "
                        "self-recurse here.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("run_all_mcsa")

    cases = list(TEST_CASES)
    if args.mcsa_ids:
        wanted = {int(x.strip()) for x in args.mcsa_ids.split(",") if x.strip()}
        cases = [c for c in cases if c.mcsa_id in wanted]
        if not cases:
            log.error(f"No matching M-CSA IDs in {args.mcsa_ids!r}; "
                      f"available: {[c.mcsa_id for c in TEST_CASES]}")
            return 2

    args.outdir.mkdir(parents=True, exist_ok=True)
    log.info(f"M-CSA validation harness → {args.outdir}")
    log.info(f"  entries: {[c.mcsa_id for c in cases]}")
    log.info(f"  max_stage: {args.max_stage}")
    log.info(f"  vacuum_ts: {'disabled' if args.no_vacuum_ts else 'enabled'}")

    results: list[EntryResult] = []
    for case in cases:
        try:
            r = run_one_entry(
                case, args.outdir,
                max_stage=args.max_stage,
                include_vacuum_ts=not args.no_vacuum_ts,
                log=log,
            )
            results.append(r)
        except KeyboardInterrupt:
            log.warning("interrupted; writing partial report")
            break
        except Exception as e:
            log.error(f"FATAL on entry {case.mcsa_id}: {type(e).__name__}: {e}")
            r = EntryResult(case=case, outdir=args.outdir / f"{case.mcsa_id:03d}_failed")
            r.overall_error = f"{type(e).__name__}: {e}"
            results.append(r)

    review_md = write_review_md(results, args.outdir)
    summary_json = write_summary_json(results, args.outdir)
    log.info(f"REVIEW.md → {review_md}")
    log.info(f"summary.json → {summary_json}")

    # Final terse summary on stdout
    print()
    print(f"=== Validation harness complete — {len(results)} entries ===")
    for r in results:
        ok = len(r.stages_completed)
        fl = len(r.stages_failed)
        sk = len(r.stages_skipped)
        flag = "✓" if fl == 0 else "✗"
        print(f"  {flag} M-CSA {r.case.mcsa_id:>3} {r.case.label:<46s} "
              f"OK={ok} fail={fl} skip={sk} ({r.wall_seconds:.1f}s)")
    print(f"\nReview: {review_md}")
    return 0 if all(len(r.stages_failed) == 0 for r in results) else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main())
