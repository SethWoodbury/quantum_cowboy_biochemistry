# Architecture

**Package:** `quantum_engine`  ·  **CLI binary:** `qcb`  (installed by `pip install -e .`)

## Package layout

```
quantum_engine/
  cli.py               # `qcb <op> ...` entry point (dispatches to ops/)
  site.py              # cluster paths + MLFF model registry (env-overridable)
  units.py             # unit-conversion constants (EV_TO_KCAL, FREQ_CONV, ...)
  select.py            # constraint grammar + fix-presets (residue/atoms/chain/...)
  calc/
    factory.py         # make_calc(model=...) — dispatches MACE / ORB / AIMNet2 / UMA
  opt/
    factory.py         # make_optimizer(backend=...) — ASE (LBFGS/FIRE/BFGS) + torch-sim
    base.py            # Optimizer interface + OptResult
  io/
    structure.py       # load_structure(), write_pdb() — PDB/XYZ/CIF via biotite
    cif.py             # CIF writer with bond orders + charges (atomworks)
    smiles_pdb.py      # SMILES -> 3D PDB / XYZ -> PDB (RDKit)
    legacy_enzts.py    # translate old enz-ts YAML -> qcb config schema
  ops/                 # Gaussian-style operations (each returns a result dict)
    sp / opt / md / freq / scan / saddle / irc / neb / mtd     # primitives
    ts.py              # native TS pipeline (composes saddle/irc/neb/mtd)
    ts_pipeline_v2.py  # YAML-driven, resumable, multi-stage TS orchestrator
    refine_ts.py       # post-NEB saddle + partial-Hessian validation
    expanded_hessian.py / imag_mode_displace.py   # tiered TS validation
    protonation.py / protonation_grid.py          # microstate-sampler helpers
    charge_ledger.py / run_config.py / chemoton_explore.py
  mlff/                # low-level ML-FF primitives (calculator-agnostic)
    irc.py             # Sella saddle + IRC primitives
    interpolation.py   # geodesic / IDPP / linear NEB interpolation
    cv_spring.py / auto_spring_k.py               # bond-difference CV spring
    metadynamics.py / plumed_runner.py            # WT-MTD / OPES (pure + PLUMED)
    endpoint_generation.py / ligand_xtb.py
  mm/
    openmm.py          # OpenMM MD scaffold (experimental)
  prep/                # structure preparation
    protonator.py      # CANONICAL protonation engine (CLI: `qcb protonate`)
    protonate.py       # PROPKA pKa prediction helper (get_pka_dict)
    charge / cap / extract / convert / validate_pdb
  qm/                  # external QM-engine + path-search backends
    gaussian / orca / xtb / xtb_refine / crest / autode       # working
    sella / dimer / pysisyphus                                # saddle/TS engines
    submit.py          # SLURM script generation
  analysis/            # barriers (Eyring), fes, geometry, kde
  data/                # chebi / mcsa / ligand_bonds loaders
  pipelines/           # contract.py (Step/Pipeline) + steps.py
  slurm/               # job_runner.py + submit_walkers.py
  config/
    schema.py          # Pydantic YAML config schema for `qcb run config.yaml`
```

There is **no** `scripts/` directory — every operation is a `qcb` subcommand
(`qcb sp/opt/md/freq/scan/saddle/irc/neb/mtd/ts/protonate/...`). Loose helper
scripts live in `tools/`; opinionated end-to-end applications live in
`enz_qc_pipelines/`.

## Design principles

1. **One operation per module.** Each `quantum_engine/ops/*.py` does one thing
   and returns a standardized result dict.

2. **Shared infrastructure via factories.** Energy-function creation
   (`quantum_engine.calc.make_calc`), optimizer creation
   (`quantum_engine.opt.make_optimizer`), structure I/O (`quantum_engine.io`),
   and constraint parsing (`quantum_engine.select`) are used by every op — no
   duplicated code. Adding a new energy function or optimizer is a localized
   change to the relevant factory.

3. **Consistent interface.** Every op's `run()` accepts `atoms`, a `calculator`,
   `outdir`, a `constraint`, and operation-specific `**kwargs`.

4. **Standardized output.** Every op returns a dict with at minimum `status`,
   `atoms`, `energy_eV`, and `outputs` (paths to trajectories/logs/plots), plus
   operation-specific keys.

5. **Composable.** Higher-level ops (`ts`) compose lower ones
   (`saddle` + `irc` + `neb`). Users can compose their own pipelines.

## Usage model

```python
# Python (from a notebook or script)
from quantum_engine.calc import make_calc
from quantum_engine.io import load_structure, parse_constraints, build_fix_atoms
from quantum_engine.ops import opt, md, freq

atoms, template, charge = load_structure("enzyme.pdb")
atoms.calc = make_calc("mace-omol", charge=charge)
atoms.info["charge"] = charge

mask = parse_constraints(atoms, template, ["atoms CA"])   # freeze alpha carbons
c = build_fix_atoms(mask)

r_opt = opt.run(atoms, outdir="out/opt", constraint=c, fmax=0.01)
r_md  = md.run(r_opt["atoms"], outdir="out/md", constraint=c,
               total_time_ps=5.0, temperature_K=300)
r_freq = freq.run(r_opt["atoms"], outdir="out/freq", constraint=c)
```

```bash
# CLI
qcb protonate enzyme.pdb -o enzyme_h.pdb --pH 7.0
qcb opt  enzyme_h.pdb --fmax 0.01 --fix-preset ca-only
qcb scan enzyme_h.pdb --coord bond --indices 133 120 --start 1.8 --end 3.2 --n-steps 20
qcb ts   enzyme_h.pdb --strategy irc
```

## Constraint grammar

The `--fix` and `--free` flags accept spec strings. Multiple specs union into
a mask:

| Spec | Meaning |
|------|---------|
| `residue HIS ASP` | atoms in HIS or ASP residues |
| `resid 100 101 102` | atoms in residue IDs 100-102 |
| `chain A B` | atoms in chain A or B |
| `atoms CA CB` | atoms named CA or CB (anywhere) |
| `element C H` | atoms of carbon or hydrogen |
| `range 0 50` | atom indices 0 through 50 |
| `all` | every atom |
| `none` | no atoms |

Presets (expand to specs + exclusions):
- `ca-only` → fix CA atoms of protein residues (ligand/water/metals free)
- `backbone` → fix CA, C, N, O of protein residues
- `backbone-water` → backbone + water oxygens
- `none` → no constraints

```bash
qcb md input.pdb --fix "chain B" "atoms CA" --free "residue YYE"
# Fix all chain-B atoms AND any CA anywhere, but unfix YYE ligand atoms.
```

## TS-search entry points

- `qcb ts --strategy {irc|cv-spring|mtd}` — native, in-process composition; best
  when you have a TS guess (`irc`) or want to explore endpoints (`cv-spring`).
- `qcb ts-pipeline-v2 config.yaml` — YAML-driven, resumable, multi-stage
  orchestrator for production runs (microstates, 2D scans, tiered validation).
- `qcb refine-ts --from-neb <dir>` — post-NEB saddle refinement + partial-Hessian
  validation (the standard polish step).
