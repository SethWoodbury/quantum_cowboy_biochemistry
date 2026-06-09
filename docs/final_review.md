# QCB Final Review

> **Historical snapshot.** Review of commit `38f3069` (pre-cleanup). The package
> is now `quantum_engine/` (not `qcb/`) and there is no `scripts/` dir; several
> findings here may already be resolved — cross-check against current `git log`.

Reviewer: independent senior computational chemistry software reviewer
Scope: full `qcb/` package, `scripts/`, and `docs/` at commit `38f3069`
Date: 2026-04-21

---

## Summary verdict

**Grade: B-**

The refactor is substantial, coherent, and mostly well-executed: consistent `run()` signatures, a clean constraint grammar, sound MTD math, principled spring-k logic, decent docs. However, two never-exercised CLI paths have import/signature bugs that will hard-fail the first time a user runs them (`cowboy-qc neb` and the pysisyphus FSM/GSM wrappers). These are small in lines of code but high in blast radius because they sit directly in the happy path advertised in `docs/strategies.md`.

---

## Blockers

1. **`cowboy-qc neb` is broken at import time.**
   `qcb/cli.py:191` does `from qcb.calc import make_calc_fn`, but `qcb/calc/__init__.py` only exports `{make_calc, CalcSpec, list_models}`. Every invocation of `cowboy-qc neb reactant.pdb product.pdb ...` will `ImportError` before any science happens.
   Fix: add `make_calc_fn` to `qcb/calc/__init__.py` (it already exists in `factory.py`).

2. **`qcb/ops/gsm.py` passes wrong kwargs to pysisyphus.**
   - `gsm.py:125-129` (FSM) and `gsm.py:209-213` (GSM) construct `FreezingString(... calculator=calc ...)` and `GrowingString(... calculator=calc ...)`.
   - Pysisyphus signatures are `FreezingString(images, calc_getter, ...)` and `GrowingString(images, calc_getter, ...)` — the kwarg is `calc_getter`, not `calculator`, **and it must be a zero-arg callable returning a fresh Calculator**, not a bound Calculator instance. gsm.py currently passes an instantiated `_AsePysisCalc`. This fails with `TypeError` at construction.
   - Fix: pass `calc_getter=lambda: _make_ase_pysis_calc(calculator_fn, charge)`. Additional wart: the `_make_ase_pysis_calc(...).get_forces(self, atoms, coords)` signature looks like it was written against an older pysisyphus Calculator API; current pysis uses `calculate(coords)` / `get_energy(coords)` on a Geometry-bound path. Verify once a real run is attempted — I would be surprised if it passes end-to-end even after the kwarg fix.

---

## Concerns

3. **`cowboy-qc gsm` / `cowboy-qc fsm` is never wired into the CLI.** `qcb/ops/gsm.py` defines `run_fsm`, `run_gsm`, and a unified `run()`, but there is no `cowboy-qc gsm` subcommand, and `qcb/ops/__init__.py:18` does not import gsm. The strategies guide advertises FSM as the SOTA recommendation (96.6% per Wan 2026), but the user has no way to actually invoke it via the CLI. Either add `cowboy-qc gsm` as a first-class subcommand (parallel to `neb`) or remove the advertised capability from docs/strategies.md.

4. **OPES is implemented but unreachable from the CLI.** `qcb/mlff/metadynamics.py:579` exposes `run_opes_rescue` with correct Invernizzi-2020 reweighting math (centers/sigmas/log-weights stored, `reweight()` recomputes `log_weight = β·V(s_i,t)` after every deposition — this is correct). But `qcb/ops/mtd.py:28` hard-imports `run_metadynamics_rescue` only; `qcb/cli.py:349-355` has no `--variant wt|opes` flag. Users get classical WT-MTD, full stop.

5. **PDB writer silently drops incoming REMARK lines.** `_write_pdb_with_template` calls `biotite.io.pdb.PDBFile().set_structure(updated); pdb_file.write(...)`. biotite writes standard CRYST/ATOM/HETATM/TER/END only; any incoming `REMARK  QCB ...` or other application REMARKs on the input PDB are not carried through. Only the two REMARKs this writer adds (charge, energy) survive. If the input had, say, a protonation-pipeline REMARK that a downstream step reads, it is lost after `cowboy-qc opt --output-pdb`. Document this, or explicitly copy `REMARK 2 QCB ...` lines from the source PDB into `header`.

6. **`cowboy-qc ts --fix-preset` may silently collide with legacy script's own default.** `qcb/cli.py:250-251` forwards `--fix-preset X` as `--constraint-mode X` to `scripts/run_neb_ts.py`. The legacy script defaults `--constraint-mode ca-only` (run_neb_ts.py:2837). If the user passes `--passthrough --constraint-mode backbone` and `--fix-preset backbone`, argparse on the subprocess side gets two `--constraint-mode` flags and the later wins. Works fine, but the user has no warning that both exist. Either forbid `--constraint-mode` inside `--passthrough` (scan and reject) or document the precedence.

7. **`scripts/run_neb_ts.py` is 2912 lines.** It remains the sole implementation of the full TS pipeline; `qcb/ops/ts.py` is a thin subprocess wrapper. That's fine as a transition strategy, but `docs/architecture.md:138-144` calls it "deprecated but kept" while `cowboy-qc ts` literally cannot function without it. It is not deprecated; it is load-bearing. Update the doc, or split the monolith into real qcb.ops composition.

8. **R2 charge-hint logic is inconsistent between `cowboy-qc opt` and `cowboy-qc neb`.**
   - `cli.py:77-81` (for single-input ops): warns if CLI `--charge` disagrees with PDB REMARK, then uses CLI value.
   - `cli.py:199-203` (for NEB): warns if CLI disagrees with **reactant's** REMARK, but never checks product's REMARK. A user with mismatched R/P charges gets no warning.

9. **Geodesic interpolation silently falls back to linear.** `qcb/mlff/interpolation.py:203-207`: if geodesic raises, it logs a warning and falls back to IDPP, then to linear. The strategies guide explicitly says "Never use linear on enzyme-scale systems", but if the geodesic-interpolate pip package is missing, the user gets linear without a prominent error. At minimum, raise an error on linear fallback for systems >200 atoms, or require an opt-in `--allow-linear-fallback`.

10. **Freq mode-vector conversion is half-correct.** `qcb/ops/freq.py:122-131` stores mass-weighted Hessian eigenvectors directly as "modes" in Cartesian atom indices without unweighting. The result is a mass-weighted displacement, not a Cartesian one. For IRC initial displacement (Fukui's definition) mass-weighted is actually the right choice, but the docstring calls it "normal modes" which typically implies Cartesian. Either multiply by `1/sqrt(m)` before storing (Cartesian) and write MW separately, or document precisely which convention is used.

---

## Nits

- `auto_spring_k.py:712` — toy test uses a PO4 at d(P-O)=1.5 Å, which is *shorter* than the typical 1.55-1.62 Å single-bond P-O, so the Pauling BO comes out >2 (unphysical for a clean single bond). Still-useful smoke test, just rename the demo geometry to reflect that this is a compressed geometry or move to a 1.60 Å baseline.
- `qcb/ops/opt.py:93` uses a check-mark unicode in logging. Fine on most terminals but breaks ASCII-only SLURM log parsers. Minor.
- `qcb/cli.py:398` lists which keys to suppress in the pretty-print by name; add `images_final` when GSM is wired in.
- `qcb/io/constraints.py:46-50` — the `STANDARD_EXCLUDED_RES` set includes single-letter metal residue names (`ZN`, `MG`, `CA`, ...) but `CA` is ambiguous with the atom name "CA" (alpha carbon). It is never used as a **residue name** in `parse_constraints` for atom-name checks (that uses `bt_struct.atom_name`), so no bug — just worth a comment.
- `qcb/ops/md.py:138` mutates frame.info after-the-fact; fine but the in-loop snapshot could do it directly.
- `qcb/mlff/metadynamics.py:436` relies on `units.kJ / units.mol` as an eV-per-kJ/mol conversion. Correct by construction since ASE stores energies in eV, but a named constant would read better.
- `docs/strategies.md:634-653` — citation list includes arXiv:2604.00405 with a 2026 date; verify the arXiv ID format (usually YYMM.NNNNN so 2604 would mean April 2026 — plausible given today is 2026-04-21, but double-check before publication).
- `README.md:100-104` — "GPUs: A4000... A6000/H200" is stale per your own `feedback_gpu_queues.md` memory (L40 is the DIGS queue winner and you are now testing B4000). Update.

---

## Action items (ranked)

1. **Blocker.** Add `make_calc_fn` to `qcb/calc/__init__.py` `__all__` + import line. (`qcb/calc/__init__.py:2`)
2. **Blocker.** Fix pysisyphus kwargs in `qcb/ops/gsm.py:125-129, 209-213`. Rename `calculator=` → `calc_getter=` and wrap in a `lambda`. Then actually attempt one smoke run on the PO4 toy system to confirm the ASE-to-pysis calculator wrapper is API-compatible.
3. **Concern.** Expose `run_opes_rescue` in `qcb/ops/mtd.py` and add `--variant {wt,opes}` flag to `qcb/cli.py:349-355`. (`qcb/ops/mtd.py:28`, `qcb/cli.py:349`)
4. **Concern.** Add a `cowboy-qc gsm` subcommand (modeled on `cowboy-qc neb`) to `qcb/cli.py` after fix #2 above; export gsm from `qcb/ops/__init__.py:18`.
5. **Concern.** Propagate REMARK lines from input PDB through `write_pdb`. (`qcb/io/structure.py:91-131`)
6. **Concern.** Raise a loud error (not a warning) when geodesic falls back to linear, and tighten docs. (`qcb/mlff/interpolation.py:203-217`)
7. **Concern.** Document that `scripts/run_neb_ts.py` is still load-bearing, not deprecated. (`docs/architecture.md:138-148`)
8. **Concern.** Check product-side charge hint in `_cmd_neb`. (`qcb/cli.py:199-209`)
9. **Nit.** Update README GPU section. (`README.md:104`)
10. **Nit.** Add a minimal smoke test: `pytest tests/test_smoke.py` that imports every module and runs `sp` on a 3-atom molecule. Would have caught blocker #1.

---

## Questions for author

1. **Was any R2 or R3 job actually executed via `cowboy-qc neb`?** If yes, please paste the stack trace — my static reading says it would have `ImportError`'d. If no (they all went through `cowboy-qc ts` → `run_neb_ts.py`), then this codepath has never been exercised and the blocker has just been latent.
2. **`_make_ase_pysis_calc` in `qcb/ops/gsm.py`** — was this ever exercised end-to-end against current pysisyphus HEAD? The pysis Calculator base class's expected methods evolved between releases, and the `get_forces(self, atoms, coords)` signature here doesn't quite match any pysis version I could find. An actual smoke test of FSM on a toy H2+H → H + H2 system would settle it.
3. **R3 benchmark** uses `--gres=gpu:b4000:1` (scripts/R3_benchmark/R3_GLU_set0_irc.sh:4). Your `feedback_gpu_queues.md` memory says L40 is the DIGS queue to use — is B4000 now in steady state for this workflow, or is this a per-experiment override? If it's the new default, update the memory + README to match.
4. **Why does `cowboy-qc ts`'s `--strategy` choices (`legacy, irc, cv-spring, mtd`) not include `gsm`/`fsm`** when the strategies doc holds FSM up as the benchmark-winning default?
5. **OPES epsilon floor**: `metadynamics.py:164` defaults `epsilon_weight=1e-6`. PLUMED's default is 1/γ, i.e., 0.1 for γ=10. The much smaller ε will over-peak the bias early. Was this intentional for short rescue runs, or should it track PLUMED's convention?

---

## Production-readiness verdict

- `cowboy-qc opt input.pdb` today: **yes**, will work.
- `cowboy-qc ts input.pdb --strategy irc` today: **yes**, falls through to run_neb_ts.py which is the tested path.
- `cowboy-qc neb reactant.pdb product.pdb` today: **no**, ImportError (blocker #1).
- `cowboy-qc gsm` / FSM from pysisyphus today: **no**, unreachable (blocker #2 + not-wired).
- `cowboy-qc mtd --variant opes`: **no flag exists** (concern #4).
- R3 SLURM scripts: structurally correct; charge inference, strategy, paths, partition all look right. Only soft concern is B4000 vs L40 (question #3).

Fix blockers #1 and #2, wire GSM/OPES into the CLI, and the codebase is legitimately A-/A territory.
