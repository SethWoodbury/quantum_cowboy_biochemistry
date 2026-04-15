# Quantum Cowboy Biochemistry

One-stop shop for enzyme computational chemistry: from raw PDB to reaction barrier.

## What this does

| Module | Purpose | Status |
|--------|---------|--------|
| **`qcb.prep`** | Extract active sites, protonate, calculate charges, cap backbones | In development |
| **`qcb.mlff`** | MACE ML force field NEB transition state searches | Working (validated against experiment) |
| **`qcb.qm`** | Gaussian/ORCA/xTB input generation and SLURM submission | In development |
| **`qcb.analysis`** | Barrier comparison, geometry analysis, energy profiles | In development |

## Quick start: NEB-TS search

```bash
# From a GPU interactive session (gpu_16g):
cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry

apptainer exec --nv \
    --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/run_neb_ts.py input.pdb \
        --model mace-mp \
        --mode standard \
        --constraint-mode ca-only
```

Or use the notebook: `notebooks/assemble_neb_ts_jobs.ipynb`

## Outputs

Each NEB-TS run produces:
```
outputs/<system>-<model>/
    reactant.pdb              # Relaxed reactant (input PDB format)
    product.pdb               # Relaxed product
    transition_state.pdb      # TS structure
    neb_path.pdb              # Multi-MODEL PDB (scrub in PyMOL)
    md_reactant.pdb           # MD equilibration trajectory
    md_product.pdb            # MD equilibration trajectory
    energy_profile.png        # Barrier plot (kcal/mol)
    summary.json              # Barriers, timings, metadata
    technical/                # Raw logs, trajectories, xyz files
```

## Available MACE models

| Model | DFT Level | Metals? | Charge-aware? | GPU needed |
|-------|-----------|---------|---------------|------------|
| `mace-mp` | r2SCAN | Yes | No | A4000 (16GB) |
| `mace-omol` | wB97M-V | Yes | Yes | A6000 (48GB) |
| `mace-mh --head omol` | wB97M-V | Yes | Yes* | A4000 |
| `mace-mh --head rgd1_b3lyp` | B3LYP | Yes | Yes* | A4000 |
| `mace-off[-small/-medium]` | wB97M-D3BJ | No | No | A4000 |
| `mace-polar[-s/-m/-l]` | Polarizable | Yes | Yes | A4000+ |

`*` via multi-head model (39 MB, fits on A4000)

### Dual-model strategy
Use a cheaper model for relaxation/MD, expensive model for NEB/TS:
```bash
--model mace-omol --model-relax mace-mp
```

## Constraint modes

| Mode | What's fixed | When to use |
|------|-------------|-------------|
| `ca-only` (default) | Only CA atoms | Most cases. Sidechains, waters, backbone free |
| `backbone` | CA/C/N/O during opt, CA during MD | When backbone flexibility causes problems |
| `ca-restrained` | CA + soft restraints on termini | Fragmented/cropped active sites |
| `none` | Nothing | Small systems, debugging |

## Pipeline modes

| Mode | Steps | Speed | Accuracy |
|------|-------|-------|----------|
| `quick` | NEB only | ~45 min | Rough (screening) |
| `standard` | NEB + CI-NEB | ~2 hrs | Good (production) |
| `full` | NEB + CI-NEB + Sella + freq | ~3+ hrs | Best (publication) |

## Validated results

| System | MLFF Barrier | Experimental | Activity |
|--------|-------------|-------------|----------|
| Native PTE | ~15-16 kcal/mol | 15.5 | kcat = 2100 s-1 |
| ZAPP P1D1 (de novo) | 19.5 | 21.5 | kcat = 0.008 s-1 |
| Frankenstein chimera | 27.9 | ~30-35? | ~uncatalyzed |

## Project structure

```
quantum_cowboy_biochemistry/
    qcb/                    # Python package
        config.py           # Cluster paths (edit for portability)
        prep/               # Active site extraction, protonation
        mlff/               # ML force field calculations
        qm/                 # Gaussian/ORCA/xTB job generation
        analysis/           # Post-processing
    scripts/                # CLI entry points
        run_neb_ts.py       # Main NEB-TS pipeline
    notebooks/              # Jupyter workflow notebooks
    data/                   # Bond definitions, example data
    deps/                   # Local package dependencies
    tests/                  # Test suite
```

## Environment

- **Container**: `/net/software/containers/universal.sif` (MACE, ASE, biotite)
- **Extra deps**: `deps/.local_pkgs/` (sella, jax — not in container)
- **POLAR models**: Needs gbg222's venv (has graph_electrostatics)

## Dependencies

Core: numpy, matplotlib, biotite, ASE, MACE, sella

All available via `universal.sif` + `deps/.local_pkgs/` — no pip install needed.
