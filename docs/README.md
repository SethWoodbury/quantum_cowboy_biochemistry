# Quantum Cowboy Biochemistry -- Documentation

Complete documentation for the QCB pipeline: from raw PDB to validated reaction barrier.

## Guides

| Document | What it covers |
|----------|---------------|
| [NEB-TS Guide](neb_ts_guide.md) | Full NEB transition-state search pipeline: endpoint generation, spring modes, MD strategies, constraint modes, model selection, validated results |
| [Protonation Guide](protonation_guide.md) | Active-site protonation pipeline: pdbfixer, reduce, propka, xTB, edge cases, net charge calculation |
| [Models Guide](models_guide.md) | MACE model comparison: paths on DIGS, GPU requirements, DFT levels, when to use which, dual-model strategy |
| [Protocols](protocols.md) | Standard operating protocols for screening, production, mechanistic investigation, chimeric structures |

## Quick Reference

```bash
# Standard NEB-TS (designed theozyme, clean input)
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/run_neb_ts.py input.pdb \
        --model mace-mp --mode standard --constraint-mode ca-only \
        --n-images 15 --md-steps 200 --spring-k 3.0 --spring-fmax 3.0

# Protonation
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/protonate_active_site.py input.pdb \
        -o protonated.pdb --ligand-charge 0 --pH 7.0 --relax-h
```

## Project Layout

```
quantum_cowboy_biochemistry/
    qcb/                        # Python package
        config.py               # Cluster paths (edit for portability)
        prep/                   # Protonation, extraction, charge, validation
        mlff/                   # NEB, Sella, dimer, constraints, models
        qm/                     # Gaussian, ORCA, xTB, SLURM
        analysis/               # Barriers, geometry, bond tracking
    scripts/                    # CLI entry points
        run_neb_ts.py           # NEB-TS pipeline
        refine_ts_guess.py      # Sella + IRC refinement
        protonate_active_site.py  # Protonation pipeline
    notebooks/                  # Command assembly (JupyterHub)
    data/examples/              # 3 PTE systems with inputs + outputs
    docs/                       # This documentation
    deps/.local_pkgs/           # Sella, JAX, pdbfixer (not in git)
```

## Environment

- **Cluster**: DIGS (Baker Lab, University of Washington)
- **Container**: `/net/software/containers/universal.sif` (mace 0.3.15, e3nn 0.4.4, torch 2.11, ASE, biotite)
- **Extra deps**: `deps/.local_pkgs/` overlaid via PYTHONPATH (sella, jax, pdbfixer)
- **QM software**: Gaussian 16, ORCA, xTB (all installed on DIGS)
- **GPUs**: A4000 (16 GB), B4000 (32 GB), H200 (80 GB)
