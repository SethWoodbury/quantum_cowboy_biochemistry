# Quantum Cowboy Biochemistry

One-stop shop for enzyme computational chemistry: from raw PDB to reaction barrier.

## What this does

| Module | Purpose |
|--------|---------|
| **`qcb.prep`** | Active site extraction, protonation (pdbfixer + reduce + propka), charge calculation |
| **`qcb.mlff`** | MACE ML force field NEB/TS searches, Sella refinement, dimer method |
| **`qcb.qm`** | Gaussian/ORCA/xTB input generation, SLURM submission |
| **`qcb.analysis`** | Barrier comparison, Eyring equation, geometry/bond analysis |

## Quick Start

```bash
# Protonate an active site
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/protonate_active_site.py input.pdb \
        -o protonated.pdb --ligand-charge 0 --pH 7.0 --relax-h

# NEB transition state search
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/run_neb_ts.py protonated.pdb \
        --model mace-mp --mode standard --constraint-mode ca-only \
        --n-images 15 --md-steps 200 --spring-k 3.0 --spring-fmax 3.0
```

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
    qcb/                        # Python package
        config.py               # Cluster paths (edit for portability)
        prep/                   # Protonation, extraction, charge, validation
        mlff/                   # NEB, Sella, dimer, constraints
        qm/                     # Gaussian, ORCA, xTB, SLURM
        analysis/               # Barriers, geometry, bond tracking
    scripts/                    # CLI entry points
        run_neb_ts.py           # NEB-TS pipeline
        refine_ts_guess.py      # Sella + IRC refinement
        protonate_active_site.py  # Protonation pipeline
    notebooks/                  # Command assembly (JupyterHub)
    data/examples/              # 3 PTE systems with inputs + outputs
    docs/                       # Full documentation
    deps/.local_pkgs/           # Sella, JAX, pdbfixer (not in git)
```

## Environment

- **Container**: `/net/software/containers/universal.sif` (mace 0.3.15, e3nn 0.4.4, torch 2.11)
- **Cluster**: DIGS (Baker Lab, University of Washington)
- **GPUs**: A4000 (16 GB) for mace-mp; A6000/H200 for mace-omol
