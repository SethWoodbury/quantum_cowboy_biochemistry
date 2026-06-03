# Repo Cleanup — Design Spec

**Date:** 2026-06-02
**Author:** Seth M. Woodbury (with Claude)
**Branch:** `protonator-rewrite-and-mlff-wiring`
**Status:** Approved design → ready for implementation plan

## Goal

Make `quantum_cowboy_biochemistry` feel **organized, modular, and centralized** by removing
duplication, dead code, and run-artifact clutter — WITHOUT disturbing the parts that are already
well-architected. This is the **cleanup-first** pass; it is the foundation for the subsequent
"OPAA theozyme TS notebook" track (deferred items, below).

This spec is the **exact checklist** for implementation. Every deletion/archival is enumerated so an
independent reviewer (and the user) can confirm nothing important is lost.

## Context: what's already good (do NOT touch the architecture)

The installable package `quantum_engine/` has the modular bones the user wants:
- `calc/factory.py` — `make_calc(model=...)` dispatches MACE / ORB / AIMNet2 / UMA (the plug-and-play energy-function layer).
- `opt/factory.py` — `make_optimizer(backend=...)` behind one `Optimizer` interface.
- `ops/` — reaction-agnostic Gaussian-style verbs (`sp opt md freq scan saddle irc neb mtd ts`).
- `select.py` — constraint mini-grammar + fix-presets.
- `analysis/`, `data/`, `pipelines/contract.py`, container strategy (main + UMA sidecar) — clean.

These stay. The cleanup removes cruft *around* and *duplicated within* this structure.

## Decisions (locked in with the user)

1. **Sequencing:** cleanup-first, then the notebook track.
2. **Deletion policy:** archive to a git tag/branch first, then delete from `main` (reversible).
3. **Protonation:** `protonator.py` is canonical → wire as `qcb protonate`, add missing `--ligand-charge`.
   Archive+delete `protonate_consensus.py` + `protonate_chimera.py`. Keep `protonate.py` (thin PROPKA
   helper used by `charge.py` and `protonator` stage5). `ops/protonation*.py` stay (separate
   microstate-sampler concern).
4. **Centralization model:** tagged releases → synced to `/net/software/lab/quantum_cowboy_biochemistry`
   + container rebuilt together, so dev tree / shared install / container never drift. Notebooks call
   the released `qcb`.
5. **Run artifacts (`logs/ runs/ outputs/`):** verified untracked + no code read-dependencies → delete
   on disk. Exception: **preserve `logs/container_builds/`** (container build provenance) by moving it
   to `deps/container_build_logs/`.
6. **Verification:** independent review agents check the work at each phase (user's explicit request).

## Verified facts (double-checked 2026-06-02)

- `logs/` (2.2M), `runs/` (361K), `outputs/` (2.1G): **0 git-tracked files** each. No tracked code reads
  from them (only runtime write-target defaults reference `runs/…`). Safe `rm -rf`.
- `logs/container_builds/` = 4 build logs for `quantum_chem-20260506.sif` → preserve (move to `deps/`).
- `runs/PTE_159/1hzy.pdb` is the overridable default `--input` for `tools/pte_159_theozyme.py`
  (fetchable standard PDB; deletion acceptable).
- `protonator.py`: all six stages (`stage1_caps`…`stage6_protomers`) + `main()` present; every notebook
  CLI flag exists **except `--ligand-charge`**.
- `qm/` stubs (`yarp`, `pygsm`, `molecular_gsm`, `scine`, `chemshell`): **VERIFIED safe** (independent
  review 2026-06-02) — only `tests/test_smoke.py` imports them; `chemshell` not even there. `qm/__init__.py`
  does NOT eagerly import them, so `import quantum_engine.qm` survives deletion.
- `mlff/dimer.py`: **VERIFIED** — only importer is `experiments/ts_dimer_KCX_set1_midpoint.sh` (deleted in
  Phase 1). Safe to delete **after Phase 1**. `mlff/__init__.py` does not import it.
- `mlff/endpoint_generation.py`: **NOT orphaned** — imported by `tools/run_neb_ts.py:2430,2432,2436,2450`
  (4 sites) and has 698 LOC of real strategy code. **REMOVED from deletion list this pass** (see deferrals).
- Protonation deletions (`protonate_consensus.py`, `protonate_chimera.py`): **VERIFIED importers** to update —
  `prep/__init__.py` (lines 6–15 imports, 36–42 `__all__`), `enz_qc_pipelines/active_site_ts/orchestrator.py:59`,
  `tools/pte_159_theozyme.py:495`. (`tools/legacy/protonate_active_site.py:1287` is deleted in Phase 2.)
  `protonate.py` is KEPT and still needed (`charge.py:19`, and `protonator` stage5 use `get_pka_dict`).
- 4 committed `slurm-*.out` at repo root (tracked) → `git rm` + gitignore.

## Implementation phases

Each phase ends with an **independent verification agent** that confirms (a) only the intended files
changed, (b) the package still imports, (c) no green→red in the test suite. Commit per phase.

### Phase 0 — Safety net
- Create archive tag `pre-cleanup-2026-06-02` (and branch `archive/pre-cleanup`) from current HEAD.
- Snapshot the test-suite baseline (run pytest inside `quantum_chem-20260506.sif`; record pass/fail set).
- Confirm `.gitignore` covers `outputs/ runs/ logs/ slurm-*.out`.

### Phase 1 — Artifact & hygiene purge
- `git rm --cached` the 4 root `slurm-*.out`; add `slurm-*.out` to `.gitignore`.
- Move `logs/container_builds/` → `deps/container_build_logs/` (preserve provenance).
- `rm -rf outputs/ runs/ logs/` (on-disk; untracked). Recommend future runs write to scratch.
- Archive-tag then delete `experiments/*.sh` (50+ hardcoded one-offs) + `notebooks/assemble_neb_ts_jobs.ipynb`.
- Keep `experiments/_ts_strategy_template.sh` → move to `docs/examples/`.

### Phase 2 — Dead-code deletion (package)
Runs AFTER Phase 1 (so `experiments/`, the sole `mlff/dimer.py` importer, is already gone).
Re-confirm zero importers (grep) for each, then delete:
- `qm/yarp.py`, `qm/pygsm.py`, `qm/molecular_gsm.py`, `qm/scine.py`, `qm/chemshell.py` (verified safe)
- `mlff/dimer.py` (verified: only importer was in `experiments/`, removed in Phase 1)
- 4 path-stub fns in `qm/pysisyphus.py` (keep the working `rsprfo_ts`/`dimer_ts`)
- `tools/scan_along_s_v1_legacy.py`, `tools/polish_ts_v2.py`
- `tools/legacy/protonate_active_site.py` (superseded); audit `tools/legacy/refine_ts_guess.py` for
  any logic missing from `quantum_engine.ops` before deleting.
- Update `tests/test_smoke.py` to drop deleted imports.
- NOT deleted: `mlff/endpoint_generation.py` (has a real importer + real logic; see deferrals).

### Phase 3 — Duplication collapse
- Protonation: wire `protonator.py` as `qcb protonate` (CLI subcommand in `quantum_engine/cli.py`);
  add `--ligand-charge`. Archive+delete `protonate_consensus.py` + `protonate_chimera.py`. Required edits
  to importers (verified list):
  - `prep/__init__.py` — remove imports (lines 6–15) + `__all__` entries (lines 36–42) for
    `add_hydrogens_chimera`, `add_hydrogens_with_charges`, `parse_pqr_charges`, `consensus_protonate`,
    `ConsensusResult`, `MethodResult`; add `run_protonation` export.
  - `enz_qc_pipelines/active_site_ts/orchestrator.py:59` — switch to `protonator.run_protonation`.
  - `tools/pte_159_theozyme.py:495` — switch to `protonator.run_protonation` (or `qcb protonate`).
  (`tools/legacy/protonate_active_site.py` is deleted in Phase 2, so its import needs no fix.)
- xTB: single `get_xtb_binary()` + one `_run_xtb_opt` helper; refactor `mlff/ligand_xtb`,
  `mlff/auto_spring_k`, `qm/xtb_refine` to use it.
- Move duplicated `FREQ_CONV` constant into `quantum_engine/units.py`.

### Phase 4 — Portability fixes (targeted)
- Replace foreign-user fallback paths (`gbg222`, `dme5188`, `ikalvet`) in `site.py` / `qm/submit.py`
  with env-overridable lookups (`shutil.which` / `QCB_*` env vars) + a warning on fallback.
- Fix `sys.path` hardcodes: `tools/ts_constrained_relax.py:155`, `tools/substrate_rotamer_sample.py:136`
  → `Path(__file__).resolve().parent.parent`.

### Phase 5 — Docs/naming reconciliation
- README + `docs/architecture.md`: `qcb/` → `quantum_engine/`; remove phantom `scripts/` references.
- Add "snapshot as of <date>" headers to dated result docs (`EXPERIMENTAL_RESULTS_*`,
  `AUTONOMOUS_VALIDATION_REPORT`). Cross-check `final_review.md` items vs current code.

### Phase 6 — Deploy convention
- Add `docs/deploy.md` + `deploy.sh`: tag release → rsync to `/net/software/lab/quantum_cowboy_biochemistry`
  → note container rebuild. Formalizes decision #4.

## Explicitly deferred (NOT this pass — belongs to notebook track or later)

- Notebook-enabling package fixes: UMA dispatch in `calc/factory.py`; per-bond constraint flags on
  `qcb opt`/`qcb scan`; resolving which charge-aware model (`mace-polar-m` vs `mace-mh-1 --head omol`)
  loads in the container.
- Collapsing the 3 orchestration stories (`pipelines.Pipeline` / `run_config` YAML / `ts_pipeline_v2`).
- Full `site.py` → cluster-config-file extraction.
- **`tools/run_neb_ts.py` + `mlff/endpoint_generation.py` "wire-or-retire" decision:** `run_neb_ts.py` is a
  legacy pre-CLI orchestrator (superseded by `quantum_engine.cli`) but is the sole consumer of
  `endpoint_generation.py`'s strategies. Decide later: (a) wire endpoint-generation strategies into
  `ops/neb`/`ops/ts` and delete `run_neb_ts.py`, or (b) keep both. Out of scope for this cleanup.

## Verification strategy (per user request: "independent agents check your work")

- After **each phase**: dispatch a read-only review agent that diffs the working tree vs the phase's
  intended file list, runs `python -c "import quantum_engine"` (+ key submodule imports), and confirms
  the pytest baseline didn't regress. It reports PASS/ISSUES; issues are fixed before the next phase.
- Final: confirm every `qcb <verb>` invoked by the OPAA notebook still resolves (argparse-level).

## Success criteria

- Repo root holds only code/docs/config (no run artifacts, no stray logs).
- `git grep -l "NotImplementedError" quantum_engine/qm` → only intentionally-unsupported paths remain.
- One protonation entry point; `qcb protonate --help` works.
- README/docs use `quantum_engine`/`qcb` consistently.
- Pytest baseline (inside container) has no new failures.
- Archive tag exists; every deletion recoverable.

## Outcome (2026-06-02)

Completed in 5 commits (`18b4824`→`289fa6b`); net **−4277 lines** (121 files).
Independent agent sign-off after the deletions and at the end. Recovery point:
tag `pre-cleanup-2026-06-02` / branch `archive/pre-cleanup`.

- Phase 1–6 executed as specced. Repo root is now code/docs/config only.
- Deviations from the plan (all flagged to the user):
  - `mlff/endpoint_generation.py` **kept** — it has a real importer
    (`tools/run_neb_ts.py`) and 698 LOC of real strategy; its fate ties to
    retiring `run_neb_ts.py` (deferred).
  - **xTB-wrapper consolidation deferred** — only Phase-3 item touching live
    numerical paths, unverifiable without the xtb binary + real fragments.
  - **`--ligand-charge` deferred** to the notebook track — it's a reporting-only
    feature best verified against the real OPAA di-Zn structure.
- Pre-existing failures (NOT caused by cleanup): `import quantum_engine.config`
  (pydantic v1 in container vs v2 schema) and `test_make_calc_unknown_raises`
  (stale assertion). Both pre-date this work; recorded in
  `2026-06-02-test-baseline.md`.
- Next: the **notebook track** (see the deferral list + the OPAA notebook).
