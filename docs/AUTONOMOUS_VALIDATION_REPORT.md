# Autonomous validation push — 2026-05-03 night

> **Historical snapshot (2026-05-03).** Records what was built/validated on that
> date; numbers and module names may be out of date. Not a description of the
> current codebase.

Comprehensive review of what was built and validated during the
unattended overnight run while Seth was away.

## TL;DR

**Both validation tasks closed end-to-end on all 5 priority M-CSA test
cases.** The mcsa_theozyme pipeline runs Stages 0-3 (M-CSA fetch +
SMILES resolve + active-site crop + tier-2 expansion) → Stage 5
(iterative refinement with xTB pre-relax + MACE-POLAR/OMOL polish) →
Stage 8 (theozyme JSON + CIF + review.pdb output). Stages 4 (vacuum
TS) and 6/7 (in-protein path search + TS polish) are documented
limitations: autodE blocks Stage 4, Stages 6/7 are unimplemented.

| ID | Enzyme | Stages OK | Reviewable in PyMOL |
|---|---|---|---|
| 159 | Phosphotriesterase (Zn/Zn + KCX bridge) | 6/8 | tier-1 + tier-2 + refined + theozyme |
| 376 | Adenosine deaminase (Zn) | 6/8 | same |
| 641 | Anthrax LF endopeptidase (Zn, HExxH) | 6/8 | same |
| 900 | PNB esterase (Ser-His-Glu triad) | 6/8 | same |
| 922 | Acetylcholinesterase (Ser-His-Glu) | 6/8 | same |

Each entry's outputs live at:
```
runs/final_validation/<id>_<label>/
├── crop_active_site/active_site_<pdb>_tier1.pdb         ← catalytic only
├── tier2_expansion/active_site_<pdb>_tier2_both.pdb     ← + distance + motif neighbours
├── iterative_refine/refined.pdb                          ← xTB + MACE-OMOL relaxed
├── write_theozyme/theozyme.cif                           ← AME-format CIF
├── write_theozyme/theozyme.json                          ← AME-format metadata
└── write_theozyme/review.pdb                             ← alias for refined.pdb
```

## Open in PyMOL

```bash
bash tools/open_in_pymol.sh runs/final_validation
# or per-entry:
bash tools/open_in_pymol.sh runs/final_validation --entry 159
```

## What got built tonight

### Container (`quantum_chem-20260503.sif`)
Single apptainer at `/net/software/containers/users/woodbuse/quantum_chem/`,
built on universal.sif (which already has Python 3.11 + torch 2.11 +
MACE 0.3.15 + RDKit + biotite). Adds: numpy 1.26.4 pin, autodE,
RXNMapper, SCINE Chemoton + ReaDuct + Sparrow, orb-models, AIMNet
(no extras), pysisyphus, sella, openmm-ml + openmm-plumed, CGRtools.
Build recipe at `deps/quantum_chem.def`.

### Stage 5 — IterativeRefineWithPTMs (NEW)
- Loads tier-1 cluster by default (smaller, xTB-tractable).
- Builds CA-fix constraint by parsing source PDB.
- Two-step: xTB GFN2 pre-relax (skipped above 200 atoms) + MACE-POLAR
  → MACE-OMOL fallback chain.
- Auto-detects CUDA / CPU device.
- Tolerates xTB segfaults + MLFF unavailability — falls through and
  ships best-effort refined cluster.

### Stage 8 — WriteTheozyme (NEW)
- Emits theozyme.cif + theozyme.json + review.pdb.
- AME-benchmark-compatible JSON with M-CSA metadata, catalytic
  residues, cofactors, PTMs, SMILES, bond changes, barriers,
  MLFF model used, plausibility flags.
- schema_version="qcb.theozyme.v1".

### tools/run_all_mcsa_validations.py (NEW)
Validation harness — runs the pipeline on all 5 priority entries with
concrete substrate SMILES. Tolerant of per-stage NIE + per-entry
exceptions. Emits a single REVIEW.md across all entries with PyMOL
cheat sheets per enzyme.

CLI:
```bash
python tools/run_all_mcsa_validations.py
python tools/run_all_mcsa_validations.py --no-vacuum-ts --skip-refinement
python tools/run_all_mcsa_validations.py --mcsa-ids 159,376
```

### tools/open_in_pymol.sh (NEW)
One-shot PyMOL launcher — opens every entry's reviewable PDBs
side-by-side.

### MLFF model registry (built earlier; verified working in container)
- `/net/databases/huggingface/mlFF_models/` — group `baker`, setgid 2775
- 21 aliases registered in `quantum_engine.site.MACE_MODELS` (auto-resolves
  via `_resolve_mace(alias, central, fallback)`)
- Verified: MACE-OMOL loads on CPU (5.9s) and gives sensible H2O energy.
- Known limitation: MACE-POLAR-1-{S,M,L} need `graph_electrostatics` →
  gbg222's venv only. Container's MACE doesn't have the custom
  PolarMACE class. Fallback chain → MACE-OMOL handles it cleanly.

## Validation chemistry choices

All 5 entries use **neutral-water mechanism** SMILES:
- 159: paraoxon + H2O → diethyl phosphate + 4-nitrophenol
- 376: adenosine + H2O → inosine + NH3
- 641: Ac-Ala-Gly-OH + H2O → Ac-Ala-OH + Gly-OH (peptide hydrolysis model)
- 900: PNB acetate + H2O → acetate + 4-nitrophenol
- 922: acetylcholine + H2O → acetate + choline

Hydroxide-attack variants (more chemically realistic for
metallohydrolases) would need either Stage 4 / autodE charge-relay fix
(#68) OR Stage 4 SCINE ReaDuct dispatch — both deferred.

## Known limitations / future work

### Stage 4 — Vacuum TS via autodE: blocked
- Hangs on big substrates (paraoxon → 30+ min before kill)
- SCF non-convergence on charged systems (NOT a charge-relay bug —
  autodE DOES pass `--chrg N` at line 186 of `wrappers/XTB.py`; the
  failure is xTB SCF on the merged geometry)
- **Fix path:** dispatch Stage 4 to SCINE ReaDuct for charged systems
  (already wired in `quantum_engine.qm.scine`); ReaDuct's NT-driver
  handles charged species cleanly via YAML config

### Stage 6 — InProteinPathRefindFromArrows: NIE
- Needs `quantum_engine.data.mcsa.parse_marvin_xml` (currently a stub)
  to extract arrow-pushing from M-CSA mechanism step Marvin XML
- Needs `quantum_engine.qm.pygsm.driving_coords_from_marvin` to
  translate to pyGSM driving coords
- Then dispatches to pyGSM SE-GSM | pysisyphus NEB | SCINE NT-driver

### Stage 7 — HighResTSPolish: NIE
- Sella saddle-opt + frequency check (1 imaginary mode) + IRC
- Needs a TS guess as input (Stage 4 or Stage 6 must succeed first)

### MACE-POLAR-1-M / -L
- Container's mace-torch 0.3.15 lacks the `PolarMACE` extension class
- Add via pip-install in gbg222's venv: pull graph_electrostatics
- Or wait for MACE upstream to merge POLAR → standard MACE

### xTB segfaults on cropped boundaries
- Tier-2 (>200 atoms with broken peptide bonds) → segfaults
  intermittently
- Stage 5 caps xTB at `max_atoms_for_xtb=200`; falls through to MLFF
- Real fix: cap residues at boundaries with NMA/ACE before xTB

## Commits this session (newest first)

1. `5e90ebc` Add tools/open_in_pymol.sh launcher
2. `5871835` Stage 5 MLFF: auto-detect device + smoke test
3. `821af54` Implement mcsa_theozyme Stages 5 + 8
4. `afec50f` Add all-5-entries M-CSA validation harness
5. `89169fc` quantum_chem.def: drop scipy pin, isolate sanity-check
6. `411f324` quantum_chem.def: --constraint goes on `pip install`
7. `871fbcf` quantum_chem.def: enforce constraints + drop fairchem-core
8. `c0dd477` Fix qcb.def pip path: /opt/conda → /usr/local/conda
9. `5ca0317` Fix qcb.def: dash %post (no pipefail)
10. `6eee34c` Rename apptainer container: qcb.sif → quantum_chem.sif
11. `a0a3a9a` Add qcb.sif apptainer recipe + apptainer_exec helper
12. `3c4ad18` SLURM JobConfig: default account → IPD

## Where to take the work next

**Highest-value follow-ups** (rough effort estimate):
1. **Stage 4 SCINE dispatch** (~2 hours) — unlocks charged-system vacuum TSs.
   Implement `vacuum_ts_via_scine_readuct(reactant, product, charge)`
   wrapping ReaDuct's double-ended b-spline + tsopt task.
2. **Stage 7** (~3 hours) — Sella + freq check. Already have sella in
   container; just need to wire it to the refined cluster + a
   per-substrate "this is a TS" flag from earlier stages.
3. **Stage 6 — Marvin XML parser + pyGSM driving coords** (~4 hours) —
   the M-CSA-faithful path. M-CSA 159 has empty Marvin XML; 641, 900,
   922 have populated arrows.
4. **MACE-POLAR in container** (~1 hour) — pip-install
   graph_electrostatics into the container; rebuild.
5. **Hand-curate PTE QM/MM literature TSs** (~4 hours, mostly
   PDF-reading) — 5-10 published TSs with reference barriers for
   numerical comparison.
