# Notebooks

Command/`sbatch`-generator notebooks: each driver cell **builds** the
`python -m quantum_engine.cli ...` command(s) and submits a SLURM array — no
heavy compute runs in a cell.

- `lib/` — vendored helpers (`notebook_core` + `slurm_submission`); see
  `lib/README.md`. Put `sys.path.insert(0, "<repo>/notebooks/lib")` in the INIT
  cell, then `import notebook_core as nb`.
- `opaa_theozyme/opaa_ts_pipeline.ipynb` — the OPAA di-Zn phosphotriesterase TS
  pipeline (the *test case*): INIT cell + per-step driver cells (protonate →
  reaction-spec → `ts-entry` reactant-product MLFF → optional ORCA DFT NEB-TS).
  Nothing is hardcoded to OPAA — the reaction lives in `SPECS/opaa.spec.yaml`;
  point it at any system by editing the INIT inputs + the spec.

See `docs/ts_workflow.md` and `docs/optimizers_and_engines.md` for the knobs.
