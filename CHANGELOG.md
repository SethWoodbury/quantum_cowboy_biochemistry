# Changelog

All notable changes to this project. The project adopts [Semantic
Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`. The importable package is
`quantum_engine`; the CLI is `cowboy-qc`.

## [1.0.0] — 2026-06-04

Rebranded to **Cowboy Quantum Chemistry**: a reaction-agnostic, plug-and-play
transition-state pipeline. The OPAA di-Zn phosphotriesterase theozyme is the test
case only — nothing is hardcoded to it.

### Plug-and-play architecture (registries)
- `quantum_engine/registry.py` — generic `Registry` + `PredicateRegistry`, the
  extensibility backbone.
- Every pluggable axis is now a registry with a one-line `register_*` hook:
  - energy functions — `register_energy` (`calc/factory.py`, `ENERGY_FAMILIES`)
  - minimizers — `register_optimizer` (`opt/factory.py`) + precon-LBFGS / FIRE2
  - saddle optimizers — `register_saddle` (`ops/saddle.py`)
  - path methods — `register_path` (`ops/path_search.py`, neb/fsm/gsm-de)
  - IRC — `register_irc` (`ops/irc.py`)
  - QM-native engines — `register_engine` (`qm/engine.py`)
- `calc/qc_calc.py` — xTB / g-xTB as ASE calculators (the `qc` energy family);
  GFN2-not-charge-aware-on-metals guard.

### Reaction-agnostic orchestrator
- `reaction_spec.py` — `ReactionSpec` (forming/breaking bonds, CV, reactive atoms,
  atom_map) + `RunContext` (charge/spin/model/head/engine/device).
- `ops/ts_entry.py` — the canonical core: three entry points (ts-guess /
  reactant-product / reactant-only) → clean basins → path search → saddle refine →
  active-region partial Hessian (one imaginary mode below cutoff with reactive-atom
  overlap) → IRC-like validation. RigorPreset {draft, standard, publication}.
- `ops/gates.py` (`Gate`/`GateReport`) + `logging_utils.py` (`Step`: human log +
  machine `<stage>.json`) — structured, gated logging at every stage.
- `ops/scan_modes.py` (single / two-sided / bond-difference / auto) and
  `ops/bond_monitor.py` (non-constraining bond + metal report).

### QM-native engine + DFT
- `qm/engine.py` ORCA gateway: native NEB-TS (R+P) / OptTS+Freq (guess),
  `--no-execute` prepares a job to `sbatch`; `qm/orca.py` parser + NEB-TS input-gen
  (validated against the ORCA 4.1.1 binary); `qm/calc.py` ORCA-as-ASE-calculator.

### Correctness fixes
- Collapsed two divergent ASE↔pysisyphus adapters into one unit-correct adapter
  (Å↔Bohr, hartree, lowercase symbols, `mult`, threshold preset, robust base-class
  import); verified by a finite-difference self-consistency test.
- `n_imag` counts only significant imaginary modes (`RefineTSCriteria.n_imag_cutoff`),
  excluding near-zero trans/rot FD-Hessian contaminants.
- De-hardcoded: `suggest_cv_targets` requires explicit targets or a `reaction_type`;
  atom-map endpoint-consistency safeguard for double-ended path methods; ligand
  auto-detect is convenience-only with explicit errors; `data/ligand_bonds.py` is
  optional reference data.

### MLFF / containers
- MACE-POLAR baked into the main container (the `PolarMACE` fork + `graph_longrange`);
  factory guard fixed. UMA in a sidecar (numpy2/torch2.8). `site.py` auto-picks the
  newest image. Two containers total.

### CLI, notebook, docs
- New subcommands: `cowboy-qc ts-entry`, `cowboy-qc monitor`, `cowboy-qc reaction-spec`.
- Vendored notebook helpers (`notebooks/lib/`) + the OPAA command/`sbatch`-generator
  notebook.
- Docs: `ts_workflow.md`, `optimizers_and_engines.md`, `extending.md`,
  `dependencies_and_paths.md` + an updated `architecture.md`.

### Benchmarks
- Known-outcome regressions: HCN↔HNC and charged Cl⁻+CH₃Cl SN2 (xTB, in-container,
  `pytest -m slow`); the Diels-Alder / Pt(PH3)2+H2 / di-Zn-model set is wired as
  GPU-`sbatch` templates (`benchmarks/`).

## [0.x] — pre-1.0

Iterative development: the `quantum_engine` package + `cowboy-qc` CLI, the ops/calc/opt
factories, the MLFF calculators (MACE / ORB / AIMNet2 / UMA), the v2 extended TS
pipeline, the protonator, and the SLURM / notebook tooling. Summarized here; see the
git history for detail.
