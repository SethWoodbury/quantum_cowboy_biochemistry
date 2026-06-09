> ⚠️ **ARCHIVED (2026-06).** These step-by-step protocols predate the reaction-agnostic
> `cowboy-qc ts-entry` orchestrator and reference removed/renamed entry points. For the
> current workflow see **[`ts_workflow.md`](ts_workflow.md)**,
> **[`optimizers_and_engines.md`](optimizers_and_engines.md)**, and
> **[`extending.md`](extending.md)**. Kept for historical reference only.

# Standard Protocols

Step-by-step protocols for common use cases with QCB.

---

## Protocol 1: Quick Screening (many designs, fast)

**Goal:** Rank ~10-50 theozyme designs by approximate barrier height. Discard obvious failures, identify top candidates for production runs.

**Time per system:** ~60-90 min on A4000
**Accuracy:** +/- 5 kcal/mol (sufficient for ranking)

### Steps

```bash
# 1. Protonate each design (if not already done)
for pdb in designs/*.pdb; do
    apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
        --env "PYTHONPATH=deps/.local_pkgs" \
        /net/software/containers/universal.sif \
        python scripts/protonate_active_site.py "$pdb" \
            -o "${pdb%.pdb}_prot.pdb" --ligand-charge 0 --pH 7.0 --relax-h --strip-h
done

# 2. Run quick NEB on each
for pdb in designs/*_prot.pdb; do
    apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
        --env "PYTHONPATH=deps/.local_pkgs" \
        /net/software/containers/universal.sif \
        python scripts/run_neb_ts.py "$pdb" \
            --model mace-mp \
            --mode quick \
            --constraint-mode ca-only \
            --n-images 9 \
            --md-steps 200 \
            --spring-k 3.0 --spring-fmax 3.0 \
            --skip-freq
done
```

### Settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| mode | quick | NEB only, no climbing image. Fastest. |
| model | mace-mp | Fast, A4000 sufficient |
| n-images | 9 | Rough resolution, adequate for ranking |
| md-steps | 200 | Standard (never go below 200) |
| spring-k | 3.0 | Validated optimal |

### Interpret results

- Barrier < 20 kcal/mol: promising, move to production
- Barrier 20-25 kcal/mol: marginal, may be worth investigating
- Barrier > 30 kcal/mol: likely non-functional, discard
- Barrier < 5 or negative: endpoint generation failed, re-run with different settings

### SLURM array submission

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --mem=32g
#SBATCH --time=02:00:00
#SBATCH --array=1-50
```

---

## Protocol 2: Production Theozyme (gold standard)

**Goal:** Get the best barrier estimate for a single system. Publication quality.

**Time:** ~130-250 min depending on model and system size
**Accuracy:** +/- 2 kcal/mol (validated against experiment)

### Steps

```bash
# 1. Protonate (fresh, strip existing H)
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/protonate_active_site.py input.pdb \
        -o protonated.pdb --ligand-charge 0 --pH 7.0 --relax-h --strip-h

# 2. Production NEB-TS
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/run_neb_ts.py protonated.pdb \
        --model mace-mp \
        --mode standard \
        --constraint-mode ca-only \
        --n-images 15 \
        --md-steps 200 \
        --spring-k 3.0 --spring-fmax 3.0

# 3. (Optional) Refine TS with Sella (only if system < 300 atoms)
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/refine_ts_guess.py outputs/transition_state.pdb \
        --model mace-omol --charge 0

# 4. (Optional) Higher-accuracy single-point with mace-omol
#    Use dual-model: re-run NEB with omol on the pre-converged path
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/run_neb_ts.py protonated.pdb \
        --model mace-omol \
        --model-relax mace-mp \
        --mode standard \
        --constraint-mode ca-only \
        --n-images 15 \
        --md-steps 200 \
        --spring-k 3.0 --spring-fmax 3.0
```

### Settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| mode | standard | CI-NEB gives good TS. Sella optional for <300 atoms. |
| model | mace-mp (or mace-omol) | mace-mp: fast and good. mace-omol: best accuracy. |
| n-images | 15 | Validated optimal resolution |
| md-steps | 200 | Validated optimal |
| spring-k | 3.0 | Validated optimal |
| constraint-mode | ca-only | Best for designed theozymes |

### Quality checks

After the run, verify:
1. **Energy profile** (`energy_profile.png`): smooth bell curve, single maximum
2. **Reaction energy** (ΔE_rxn): should be negative for exothermic reaction
3. **Bond distances** in TS: check P-O_nuc and P-O_lg are intermediate
4. **No negative barrier**: if barrier < 0, endpoints not properly separated
5. **Convergence**: NEB should converge within step limit (check logs)

---

## Protocol 3: Mechanistic Investigation (stepwise vs concerted)

**Goal:** Determine whether the reaction is concerted (single TS) or stepwise (intermediate + two TS).

**Time:** ~3-5 hours total (multiple NEB runs)

### Steps

```bash
# Step 1: High-resolution concerted path (21 images)
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/run_neb_ts.py protonated.pdb \
        --model mace-mp \
        --mode standard \
        --n-images 21 \
        --spring-mode both \
        --md-steps 200 \
        --spring-k 3.0 --spring-fmax 3.0 \
        -o outputs/concerted_21img/

# Step 2: Nucleophile attack only
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/run_neb_ts.py protonated.pdb \
        --model mace-mp \
        --mode standard \
        --n-images 15 \
        --spring-mode nuc-only \
        --md-steps 200 \
        --spring-k 3.0 --spring-fmax 3.0 \
        -o outputs/step_nuc_only/

# Step 3: Leaving group departure (starting from nuc-only product)
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/run_neb_ts.py outputs/step_nuc_only/product.pdb \
        --model mace-mp \
        --mode standard \
        --n-images 15 \
        --spring-mode lg-only \
        --md-steps 200 \
        --spring-k 3.0 --spring-fmax 3.0 \
        --unidirectional --direction forward \
        -o outputs/step_lg_only/
```

### Interpretation

| Observation | Conclusion |
|-------------|-----------|
| Concerted profile: single smooth maximum | Concerted mechanism |
| Concerted profile: shoulder or plateau | Possible stepwise, investigate further |
| max(TS_nuc, TS_lg) < TS_concerted | Stepwise is preferred |
| TS_concerted < max(TS_nuc, TS_lg) | Concerted is preferred |
| Intermediate stable by > 5 kcal/mol below TS | True intermediate exists |

---

## Protocol 4: Chimeric/Docked Structure

**Goal:** Get a barrier from a structure with docking artifacts (clashes, unrelaxed backbone).

**Time:** ~180-240 min on A4000 (longer due to pre-relax)

### Critical: Always use `--pre-relax`

Without pre-relax, clashes inflate barriers by 20-25 kcal/mol (validated: 53 -> 28 kcal/mol on Frankenstein system).

### Steps

```bash
# 1. Protonate
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/protonate_active_site.py chimeric_input.pdb \
        -o chimeric_prot.pdb --ligand-charge -2 --pH 7.0 --relax-h --strip-h

# 2. NEB-TS with pre-relax and potentially chain-specific constraints
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/run_neb_ts.py chimeric_prot.pdb \
        --model mace-mp \
        --mode standard \
        --constraint-mode ca-only \
        --n-images 15 \
        --md-steps 300 \
        --spring-k 3.0 --spring-fmax 3.0 \
        --pre-relax
```

### Settings differences from standard

| Parameter | Standard | Chimeric | Why |
|-----------|---------|----------|-----|
| `--pre-relax` | off | **on** | Removes clash energy before NEB |
| `--md-steps` | 200 | **300** | More equilibration needed for messy inputs |
| `--fix-chains` | none | optional | Fix scaffold chain, free designed chain |

### When to use `--fix-chains`

If your chimeric structure has:
- Chain A: designed active-site residues (should be free to move)
- Chain B: scaffold/template backbone (should be constrained)

```bash
--constraint-mode ca-only --fix-chains B
```

This fixes only chain B CAs, letting chain A (the design) relax freely.

---

## Protocol 5: Full Workflow (Protonation + NEB)

**Goal:** Complete pipeline from a raw PDB (no hydrogens, no charge assignment) to a validated barrier.

### Step-by-step

```bash
# ──────────────────────────────────────────────────────────
# STEP 1: PROTONATE
# Input: raw PDB from RFdiffusion/Rosetta (no H, unknown protonation)
# ──────────────────────────────────────────────────────────

apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/protonate_active_site.py \
        raw_design.pdb \
        -o protonated.pdb \
        --ligand-charge 0 \
        --pH 7.0 \
        --relax-h \
        --strip-h \
        --write-charge-json

# CHECK: Inspect protonated.pdb in PyMOL/ChimeraX
#   - Are metal-coordinating HIS properly deprotonated on the metal side?
#   - Are fragment termini neutral (NH2, C(=O)H)?
#   - Is the net charge reasonable?
# The charge is encoded in the output filename (e.g., _netCHG_0.pdb)

# ──────────────────────────────────────────────────────────
# STEP 2: NEB TRANSITION STATE SEARCH
# ──────────────────────────────────────────────────────────

apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/run_neb_ts.py \
        protonated.pdb \
        --model mace-mp \
        --mode standard \
        --constraint-mode ca-only \
        --n-images 15 \
        --md-steps 200 \
        --spring-k 3.0 --spring-fmax 3.0

# ──────────────────────────────────────────────────────────
# STEP 3: VALIDATE
# ──────────────────────────────────────────────────────────

# 3a. Inspect energy_profile.png
#   - Should be a smooth bell curve
#   - Single maximum (concerted) or two maxima (stepwise)

# 3b. Check summary.json
#   - barrier_fwd_kcal: should be 10-25 for functional enzymes
#   - delta_e_rxn_kcal: should be negative (exothermic)

# 3c. Inspect transition_state.pdb in PyMOL
#   - Bond distances at TS should be intermediate
#   - No gross structural distortion

# 3d. (Optional) Refine TS with Sella
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/refine_ts_guess.py \
        outputs/transition_state.pdb \
        --model mace-omol \
        --charge 0

# ──────────────────────────────────────────────────────────
# STEP 4: (OPTIONAL) DFT SINGLE-POINT VALIDATION
# ──────────────────────────────────────────────────────────

# Generate Gaussian input for DFT single-point on TS geometry
# python -c "from qcb.qm import gaussian; gaussian.write_sp_input('transition_state.pdb', ...)"
```

### Timing breakdown (typical 890-atom system on A4000)

| Step | Time | Notes |
|------|------|-------|
| Protonation | ~10 sec | Fast (mostly I/O) |
| Endpoint generation (spring + relax + MD) | ~40 min | 2 endpoints |
| NEB no-climb (200 steps) | ~50 min | Coarse path optimization |
| CI-NEB climbing (250 steps) | ~40 min | Fine TS localization |
| Total | ~130 min | |

### Output file structure

```
outputs/<system>-<model>/
    reactant.pdb              # Optimized reactant
    product.pdb               # Optimized product
    transition_state.pdb      # TS structure (CI-NEB highest point)
    neb_path.pdb              # Multi-MODEL PDB (all NEB images)
    energy_profile.png        # Barrier plot
    summary.json              # Machine-readable results
    md_reactant.pdb           # MD equilibration trajectory
    md_product.pdb            # MD equilibration trajectory
    technical/                # Raw logs, xyz trajectories
```

---

## Protocol Comparison

| Protocol | Time | Accuracy | GPU | Use when |
|----------|------|----------|-----|----------|
| Quick screening | 60-90 min | +/- 5 kcal/mol | A4000 | Ranking 10-50 designs |
| Production | 130-250 min | +/- 2 kcal/mol | A4000-H200 | Final barrier for a design |
| Mechanistic | 3-5 hours | +/- 2 kcal/mol | A4000 | Investigating mechanism |
| Chimeric | 180-240 min | +/- 3 kcal/mol | A4000 | Docked/chimeric structures |
| Full workflow | 130-260 min | +/- 2 kcal/mol | A4000-H200 | End-to-end from raw PDB |

---

## Tips for Batch Processing

### JupyterHub command assembly

Use the `notebooks/assemble_neb_ts_jobs.ipynb` notebook to:
1. Select PDB files
2. Set parameters
3. Generate SLURM submit scripts
4. Track output status

### Naming convention for outputs

The pipeline auto-names output directories based on input and model:
```
outputs/ZAPP_i1_P1D1-mace-mp/
outputs/ZAPP_i1_P1D1-mace-omol/
outputs/frankenstein-mace-mp-prerelax/
```

### Monitoring running jobs

```bash
# Check SLURM job status
squeue -u $USER

# Check NEB progress (look for barrier estimates in logs)
tail -f logs/NEB_TS_standard_mace-mp_*.stdout | grep "barrier"

# Check output directories
ls outputs/*/summary.json  # completed runs have summary.json
```
