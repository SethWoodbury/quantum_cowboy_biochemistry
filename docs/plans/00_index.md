# Plan Docs Index

Planning documents for the enzyme TS-search pipelines layered on top of
`quantum_engine` and `enz_qc_pipelines`.

## Documents

- [`enzyme_ts_design.md`](enzyme_ts_design.md) — Generic, tool-swappable
  enzyme TS pipeline. SMILES (R + P) + cropped active-site PDB → docked TS
  complex → high-res TS-in-protein → multi-TS `.cif`. 8 stages, each stage
  has Step-adapter alternates picked via `--stageN-tool=...`.
- [`mcsa_theozyme.md`](mcsa_theozyme.md) — M-CSA-driven theozyme generator
  for the AME benchmark (https://pmc.ncbi.nlm.nih.gov/articles/PMC12791007/).
  M-CSA entry ID + concrete substrate SMILES → automated theozyme. 9 stages,
  uses M-CSA mechanism XML, ChEBI lookup, and per-step TS chaining.

## Reading order

1. `enzyme_ts_design.md` — defines the stage contract and tool-adapter
   interface used by both pipelines.
2. `mcsa_theozyme.md` — specialisation: how M-CSA annotation drives
   stage-specific choices, plus PTM and R-group handling.

## Scope boundaries

- These plans cover orchestration only. Stage-internal algorithms
  (NEB scheduler, MACE calculator factory, etc.) are owned by
  `quantum_engine.ops` / `quantum_engine.mlff`.
- Default loop is DFT-free: MACE-POLAR-1M for high-res, g-xTB for fast
  evaluation. Gaussian / ORCA appear only as offline validators.
- All GPU stages target L40 on DIGS via SLURM (not A4000 — see
  `feedback_gpu_queues.md`).
