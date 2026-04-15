# Quantum Cowboy Biochemistry

One-stop shop for enzyme computational chemistry: from raw PDB to reaction barrier.

## What this does

| Module | Purpose | Status |
|--------|---------|--------|
| **`qcb.prep`** | Active site extraction, protonation (pdbfixer + reduce + propka), charge calculation, PDB validation | Working |
| **`qcb.mlff`** | MACE ML force field NEB/TS searches, Sella refinement, dimer method, IRC | Working (validated against experiment) |
| **`qcb.qm`** | Gaussian/ORCA/xTB input generation, SLURM submission | Working |
| **`qcb.analysis`** | Barrier comparison, Eyring equation, geometry/bond analysis | Working |

## Scripts

| Script | What it does |
|--------|-------------|
| `scripts/run_neb_ts.py` | Full NEB-TS pipeline: endpoints → NEB → CI-NEB → (optional Sella) |
| `scripts/refine_ts_guess.py` | Refine a TS guess: Sella → IRC → frequency validation |
| `scripts/protonate_active_site.py` | Protonate protein active site: pdbfixer + reduce + propka + xTB |

## Quick start

### Protonate an active site
```bash
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/protonate_active_site.py input.pdb \
        -o protonated.pdb --ligand-charge 3 --pH 7.0 --relax-h
```

### NEB transition state search
```bash
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/run_neb_ts.py input.pdb \
        --model mace-mp --mode standard --constraint-mode ca-only
```

### Refine a TS guess (from DFT or NEB)
```bash
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/refine_ts_guess.py ts_guess.pdb \
        --charge 1 --model mace-omol --skip-prerelax
```

## Available MACE models

| Model | DFT Level | Metals? | Charge? | GPU | Best for |
|-------|-----------|---------|---------|-----|----------|
| `mace-mp` | r2SCAN | Yes | No | A4000 | General purpose, default |
| `mace-omol` | wB97M-V | Yes | Yes | A6000 | Reaction barriers (best accuracy) |
| `mace-mh --head omol` | wB97M-V | Yes | Yes | A4000 | Compact version of OMOL |
| `mace-mh --head rgd1_b3lyp` | B3LYP | Yes | Yes | A4000 | TS-trained head |
| `mace-polar[-s/-m/-l]` | Polarizable | Yes | Yes | A4000+ | Long-range electrostatics |

### Dual-model strategy
```bash
--model mace-omol --model-relax mace-mp    # cheap relax, accurate NEB
```

## TS search methods

| Method | Script | When to use |
|--------|--------|-------------|
| **NEB + CI-NEB** | `run_neb_ts.py` | You have (or can generate) reactant + product endpoints |
| **Sella** | `refine_ts_guess.py` | You have a good TS guess (<300 atoms) |
| **Dimer** | `qcb.mlff.dimer` | TS guess, large systems (>300 atoms), no Hessian needed |
| **IRC** | `refine_ts_guess.py` | Validate TS connects reactant ↔ product |

## Constraint modes

| Mode | Fixed atoms | Use case |
|------|------------|----------|
| `ca-only` (default) | CA only | Most cases |
| `backbone` | CA/C/N/O (opt), CA (MD) | When backbone needs rigidity |
| `ca-only --fix-chains B` | CA on chain B only | Chimeric structures (fix one chain) |
| `none` | Nothing | Small systems, theozymes |

## Protonation pipeline

Tools used (in order):
1. **reduce** (Richardson lab) — detect metal-coordinating residues
2. **pdbfixer** (OpenMM) — add all H with proper PDB names (HA, HB2, HB3, etc.)
3. **Post-processing** — neutral termini, aldehyde C-terminal H, cross-residue bonds (carbamylated LYS)
4. **propka** — pKa prediction with metal/covalent bond overrides
5. **xTB** (optional) — relax H positions (fixes sp2/sp3 geometry)
6. **PDB validation** — check serials, clashes, H coverage, backbone completeness

## Validated results

| System | MLFF Barrier | Experimental | Activity |
|--------|-------------|-------------|----------|
| Native PTE | ~15-16 kcal/mol | 15.5 | kcat = 2100 s⁻¹ |
| ZAPP P1D1 (de novo) | 19.5 | 21.5 | kcat = 0.008 s⁻¹ |
| Frankenstein chimera | 27.9-36.6 | ~30-35? | ~uncatalyzed |

## Project structure

```
quantum_cowboy_biochemistry/
    qcb/                        # Python package
        config.py               # Cluster paths (edit for portability)
        prep/                   # Protonation, extraction, charge, validation
        mlff/                   # NEB, Sella, dimer, constraints
        qm/                    # Gaussian, ORCA, xTB, SLURM
        analysis/              # Barriers, geometry, bond tracking
    scripts/                   # CLI entry points
        run_neb_ts.py          # NEB-TS pipeline
        refine_ts_guess.py     # Sella + IRC refinement
        protonate_active_site.py  # Protonation pipeline
    notebooks/                 # Command assembly (JupyterHub)
    data/examples/             # 3 PTE systems with inputs + outputs
    deps/.local_pkgs/          # Sella, JAX, pdbfixer (not in git)
    tests/test_data/           # Protonation test case
```

## Environment

- **Container**: `/net/software/containers/universal.sif`
- **Extra deps**: `deps/.local_pkgs/` (sella, jax, pdbfixer)
- **QM software**: Gaussian 16, ORCA, xTB (all on DIGS)
- **Cluster**: DIGS (Baker Lab, University of Washington)
