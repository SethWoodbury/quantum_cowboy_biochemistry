# Quantum Cowboy Biochemistry

One-stop shop for enzyme computational chemistry: from raw PDB to reaction barrier.

## Top-level CLI: `qcb`

```bash
qcb sp       input.pdb                             # single-point energy
qcb opt      input.pdb --fmax 0.01                 # energy minimization
qcb md       input.pdb --time 10 --temp 300        # molecular dynamics
qcb freq     input.pdb                             # vibrational frequencies
qcb scan     input.pdb --coord bond --indices 5 12 --start 1.5 --end 3.5 --n-steps 20
qcb saddle   ts_guess.pdb                          # Sella saddle search
qcb irc      ts.pdb --step 0.1                     # IRC from a TS
qcb neb      reactant.pdb product.pdb              # NEB + CI-NEB
qcb mtd      input.pdb --p-idx 133 --nuc-idx 120 --lg-idx 135 --time 100
qcb ts       input.pdb --strategy irc              # full TS pipeline
```

All ops share `--model`, `--charge`, `--fix`/`--free`, `--fix-preset`, `--outdir` flags.
See [`docs/architecture.md`](docs/architecture.md) for the module layout and Python API,
and [`docs/strategies.md`](docs/strategies.md) for TS search strategies.

## Module structure

The installable package is **`quantum_engine`**; the CLI binary stays **`qcb`**.

| Module | Purpose |
|--------|---------|
| **`quantum_engine.calc`** | MLFF calculator factory: `make_calc()` dispatches MACE / ORB / AIMNet2 / UMA |
| **`quantum_engine.opt`** | Optimizer factory: ASE (LBFGS/FIRE/BFGS) + torch-sim behind one interface |
| **`quantum_engine.io`** + **`.select`** | Structure I/O + constraint spec grammar |
| **`quantum_engine.ops`** | Gaussian-style ops: sp, opt, md, freq, scan, saddle, irc, neb, mtd, ts |
| **`quantum_engine.mlff`** | Low-level ML-FF primitives: Sella/IRC, geodesic interp, CV spring, metadynamics |
| **`quantum_engine.prep`** | Active-site extraction, protonation (`protonator` = `qcb protonate`), charge calculation |
| **`quantum_engine.qm`** | Gaussian/ORCA/xTB input generation, SLURM submission |
| **`quantum_engine.analysis`** | Barrier comparison, Eyring equation, geometry/bond analysis |

## Quick Start

All commands run the `qcb` CLI inside the container. Define a prefix once:

```bash
SIF=/net/software/containers/users/woodbuse/quantum_chem/quantum_chem-20260506.sif
QCB="apptainer exec --nv --bind /home --bind /net $SIF qcb"

# Protonate an active site (deterministic staged protonator)
$QCB protonate input.pdb -o protonated.pdb --pH 7.0 --relax-h

# Minimize with MACE-OMOL, freeze CAs, write PDB
$QCB opt protonated.pdb --model mace-omol --fix-preset ca-only --fmax 0.01 \
    --output-pdb relaxed.pdb

# Full TS pipeline with the IRC strategy
$QCB ts relaxed.pdb --strategy irc --n-images 15 --fix-preset ca-only
```

> For a development checkout, bind-mount it over the installed package:
> `--bind <repo>:/opt/quantum_engine_src` and prepend it to `PYTHONPATH`.

## Documentation

Comprehensive guides are in [`docs/`](docs/README.md):

- **[NEB-TS Guide](docs/neb_ts_guide.md)** -- Endpoint generation, spring modes, MD strategies, constraint modes, model selection, validated results, pitfalls
- **[Protonation Guide](docs/protonation_guide.md)** -- Pipeline steps, metal coordination, fragment termini, charge calculation, edge cases
- **[Models Guide](docs/models_guide.md)** -- All MACE models on DIGS, GPU requirements, DFT levels, dual-model strategy
- **[Protocols](docs/protocols.md)** -- Quick screening, production, mechanistic investigation, chimeric structures, full workflows

## Validated Results

| System | MLFF Barrier | Experimental | Activity |
|--------|-------------|-------------|----------|
| Native PTE | ~15-16 kcal/mol | 15.5 | kcat = 2100 s-1 |
| ZAPP P1D1 (de novo) | 19.5 | 21.5 | kcat = 0.008 s-1 |
| Frankenstein chimera | 27.9 | ~30-35? | ~uncatalyzed |

Pipeline correctly ranks all systems across 4 orders of magnitude in activity; barriers within ~2 kcal/mol of experiment.

## Project Structure

```
quantum_cowboy_biochemistry/
    quantum_engine/             # the installable Python package (CLI: qcb)
        cli.py                  # `qcb <op>` entry point
        site.py                 # cluster paths + MLFF model registry (env-overridable)
        calc/ opt/              # energy-function + optimizer factories
        io/ select.py           # structure I/O + constraint grammar
        ops/                    # sp opt md freq scan saddle irc neb mtd ts ...
        mlff/ mm/               # ML-FF primitives + OpenMM scaffold
        prep/                   # protonator (qcb protonate), extraction, charge
        qm/                     # Gaussian/ORCA/xTB I/O, SLURM submit, saddle backends
        analysis/ data/ pipelines/ slurm/ config/
    tools/                      # loose helper scripts (wrappers around quantum_engine)
    enz_qc_pipelines/           # opinionated end-to-end applications
    data/examples/              # PTE test systems (inputs + outputs)
    docs/                       # full documentation
    deps/                       # vendored submodules + container build defs
```

See [`docs/architecture.md`](docs/architecture.md) for the full module map.

## Environment

- **Container**: `/net/software/containers/users/woodbuse/quantum_chem/quantum_chem-20260506.sif`
  (Python 3.11, torch 2.11, mace 0.3.15, SCINE, PLUMED). UMA (FairChem) lives in a
  separate sidecar `uma-*.sif` (numpy 2.0 / torch 2.8) — see `deps/uma_sidecar.def`.
- **Cluster**: DIGS (Baker Lab, University of Washington)
- **GPUs**: A4000 (16 GB) for mace-mp; A6000/H200 for mace-omol
