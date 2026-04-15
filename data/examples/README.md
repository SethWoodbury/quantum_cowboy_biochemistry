# Example NEB-TS Runs

Three phosphotriesterase (PTE) systems tested with MACE-MP (r2SCAN) on A4000 GPUs.

## Systems

| System | Atoms | Ligand | Charge | kcat | Expected barrier |
|--------|-------|--------|--------|------|-----------------|
| ZAPP_i1_P1D1 bestHIT | 890 | YYE | 0 | 0.0082 s⁻¹ | 21.5 kcal/mol |
| Enhanced PTE set3 | 475 | YYL | -1 | ~2100 s⁻¹ (native-like) | ~15.5 kcal/mol |
| Frankenstein ACHE_PTE | 1060 | YYL | -2 | unknown (chimera) | ~30-35 (uncatalyzed) |

## Best run commands

### ZAPP_i1_P1D1 bestHIT (barrier: 19.5 kcal/mol)

```bash
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/run_neb_ts.py \
        data/examples/zapp_p1d1_bestHIT/input.pdb \
        --model mace-mp \
        --mode standard \
        --constraint-mode ca-only \
        --n-images 15 \
        --md-steps 250 \
        --spring-k 4.0 --spring-fmax 4.0 \
        --skip-freq
```

**Parameters that produced 19.5 kcal/mol (best_run/):**
- model: mace-mp (r2SCAN)
- mode: standard (NEB + CI-NEB, no Sella)
- constraints: old backbone (CA/CB/CG/C/N/O fixed) — before ca-only was implemented
- n-images: 15
- md-steps: 250
- spring-k: 4.0
- GPU: A4000 (16 GB), ~129 min

**ca-only constraint run (best_run_caonly/, 19.6 kcal/mol):**
- Same as above but constraint-mode: ca-only
- md-steps: 200, spring-k: 3.0
- Result nearly identical — ca-only is the correct default

### Enhanced PTE set3 (barrier: 6.4 kcal/mol, ca-only)

```bash
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/run_neb_ts.py \
        data/examples/enhanced_PTE_set3/input.pdb \
        --model mace-mp \
        --mode standard \
        --constraint-mode ca-only \
        --n-images 15 \
        --md-steps 200 \
        --spring-k 3.0 --spring-fmax 3.0 \
        --skip-freq
```

**Parameters (best_run_caonly/):**
- GPU: A4000, ~105 min
- Lower barrier than ZAPP because it's modeled on the highly active native enzyme

### Frankenstein ACHE_PTE (barrier: 27.9 kcal/mol with pre-relax)

```bash
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/run_neb_ts.py \
        data/examples/frankenstein_ACHE_PTE/input.pdb \
        --model mace-mp \
        --mode standard \
        --constraint-mode ca-only \
        --n-images 15 \
        --md-steps 300 \
        --spring-k 3.0 --spring-fmax 3.0 \
        --pre-relax \
        --skip-freq
```

**Parameters (best_run/, 27.9 with old constraints + pre-relax):**
- --pre-relax is CRITICAL for chimeric structures with docking clashes
- Without pre-relax: 53 kcal/mol (artificially inflated by clashes)
- ca-only + pre-relax: 36.6 kcal/mol (more freedom, worse pathway)
- GPU: A4000, ~187-205 min

## Recommended defaults (from 16+ experiments)

```
--model mace-mp          # or mace-omol on A6000+
--mode standard          # CI-NEB sufficient, Sella crashes on >500 atoms
--constraint-mode ca-only
--n-images 15
--md-steps 200           # critical: 100 too few, 500 too many
--spring-k 3.0           # sweet spot: k=2 too soft, k=6+ too aggressive
--spring-fmax 3.0
```

Add `--pre-relax` for docked/chimeric inputs with clashes.

## Output files per run

```
best_run/
    reactant.pdb           # Relaxed reactant (input PDB format)
    product.pdb            # Relaxed product
    transition_state.pdb   # TS structure
    neb_path.pdb           # Multi-MODEL PDB (scrub in PyMOL)
    energy_profile.png     # Barrier plot (kcal/mol)
    summary.json           # Barriers, timings, metadata
```
