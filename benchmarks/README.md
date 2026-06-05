# Known-outcome benchmarks

End-to-end regression on small reactions with known outcomes — they exercise the
whole pipeline (energy function → saddle → partial Hessian → the
n_imag/imag-freq/overlap gate → `ts_entry`). The committed, container-runnable
ones live in `tests/test_benchmarks.py` (`pytest -m slow`); the heavier
charge-aware-MLFF / DFT ones are for a GPU `sbatch`.

## Tier 1 — xTB, in-container (committed, `pytest -m slow`)

| Reaction | Atoms | Charge | What it checks | Status |
|----------|-------|--------|----------------|--------|
| **HCN ↔ HNC** | 3 | 0 | clean 3-atom saddle; strong imag mode (< −1000 cm⁻¹); n_imag=1; gate PASS | ✅ validated (imag ≈ −1426 cm⁻¹) |
| **Cl⁻ + CH₃Cl SN2** | 6 | −1 | CHARGED pipeline mechanics: a first-order saddle with the mode on the reactive atoms; −1 propagates to xTB | ✅ validated |

GFN2-xTB is **qualitative** — its frequencies/barriers are method-soft (the SN2
imaginary frequency comes out softer than the textbook ~−460 cm⁻¹). HCN's strong,
well-separated mode is asserted tightly; SN2 asserts the charged mechanics, not a
tight frequency. The quantitative layer is the charge-aware-MLFF / DFT runs below.

## Tier 2 — charge-aware MLFF / DFT, GPU `sbatch` (templates)

Run these with `mace-polar-m` / `mace-mh-1 --head omol` / `uma-*` (GPU) or via
the ORCA engine (DFT, host-side). They need a verified TS guess or R/P endpoints
(build from literature geometries). Use `qcb ts-entry` (see `docs/ts_workflow.md`):

| Reaction | Atoms | Charge/spin | Literature outcome | Energy function |
|----------|-------|-------------|--------------------|-----------------|
| **Diels-Alder** (butadiene + ethylene) | 10 | 0 / 1 | concerted 2-bond TS, barrier ≈ 27.5 kcal/mol | MLFF or ORCA |
| **Pt(PH₃)₂ + H₂** oxidative addition | ~11 | 0 / 1 | well-documented organometallic TS (charge-agnostic OK) | MLFF (metal) |
| **di-Zn hydrolysis model** (e.g. `[Zn(OH)(H₂O)₂]⁺ + (MeO)₂PO₂⁻`) | ~20 | fix stoichiometry/net charge before running | the enzyme-relevant metal+charge test | **charge-aware MLFF only — NOT GFN2** |

The Zn model is the enzyme-relevant analogue of the OPAA theozyme: **use a
charge-aware MLFF** (GFN2-xTB is not charge-aware on metals — `make_qc_calc`
warns/forbids). Fix the exact stoichiometry + net charge first.

## Running

```bash
# Tier 1 (container):
apptainer exec --nv --bind /home --bind /net <quantum_chem.sif> \
  python -m pytest tests/test_benchmarks.py -m slow

# Tier 2 (GPU sbatch): build a ReactionSpec + endpoints, then
qcb ts-entry --entry reactant-product --reaction-spec rxn.yaml \
    --reactant R.xyz --product P.xyz --model mace-polar-m --charge <q> --rigor publication
```

The real OPAA theozyme runs are the user's to `sbatch`; we co-review.
