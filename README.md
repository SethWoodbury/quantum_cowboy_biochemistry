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

| Module | Purpose |
|--------|---------|
| **`qcb.calc`** | MACE calculator factory |
| **`qcb.io`** | Structure I/O + constraint spec grammar |
| **`qcb.ops`** | Gaussian-style ops: sp, opt, md, freq, scan, saddle, irc, neb, mtd, ts |
| **`qcb.mlff`** | Low-level ML-FF primitives: Sella/IRC, geodesic interp, xTB refine, CV spring, metadynamics |
| **`qcb.prep`** | Active site extraction, protonation (pdbfixer + reduce + propka), charge calculation |
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

# Minimize with MACE-OMOL, freeze CAs, write PDB
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs:/path/to/quantum_cowboy_biochemistry" \
    /net/software/containers/universal.sif \
    python scripts/qcb opt protonated.pdb --fix-preset ca-only --fmax 0.01 \
        --output-pdb relaxed.pdb

# Full NEB-TS pipeline with IRC strategy
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs:/path/to/quantum_cowboy_biochemistry" \
    /net/software/containers/universal.sif \
    python scripts/qcb ts protonated.pdb --strategy irc \
        --passthrough --n-images 15 --fix-preset ca-only
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
