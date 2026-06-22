> ⚠️ **ARCHIVED (2026-06).** This guide documents the legacy `tools/run_neb_ts.py`
> entry point, which is **deprecated**. The current, reaction-agnostic TS workflow is
> `cowboy-qc ts-entry` (path → saddle → partial-Hessian → IRC-like gate). See
> **[`ts_workflow.md`](ts_workflow.md)** (canonical core + entry-point decision tree),
> **[`optimizers_and_engines.md`](optimizers_and_engines.md)** (swappable path/saddle/
> optimizer/engine backends), and **[`extending.md`](extending.md)** (registering new
> models/methods). Kept only for historical reference to the old NEB script interface.

# NEB Transition State Search Guide

Complete guide to the `run_neb_ts.py` pipeline for finding enzyme reaction barriers using MACE ML force fields.

---

## When to Use NEB vs Sella vs Dimer

| Method | Script | Input needed | System size | When to use |
|--------|--------|-------------|-------------|-------------|
| **NEB + CI-NEB** | `run_neb_ts.py` | Reactant + product (or input near TS) | Any | Default. Endpoints generated automatically via springs. |
| **Sella** | `refine_ts_guess.py` | Good TS guess | <300 atoms free | Have a DFT TS guess or high-quality NEB TS. Crashes on large systems. |
| **Dimer** | `qcb.mlff.dimer` | TS guess + displacement vector | Any | Large systems (>300 atoms), no Hessian required. |
| **IRC** | `refine_ts_guess.py --irc` | Validated TS | Any | Confirm TS connects correct reactant/product. |

**Decision tree:**
1. Do you have endpoints (or can generate them)? Use **NEB**.
2. Do you have a TS guess from DFT or NEB, system <300 atoms? Use **Sella** to refine.
3. Large system (>300 atoms) with TS guess? Use **Dimer**.
4. Need to validate a TS? Run **IRC** forward + reverse.

---

## Endpoint Generation Methods

The pipeline generates reactant and product from a single input PDB (which is often near the TS or a Michaelis complex). Four approaches are available:

### 1. Equilibrium Scan (default: bidirectional spring + relax)

The standard method. From the input geometry, apply directional spring constraints to drive bond-breaking/forming, then relax and equilibrate with MD.

```bash
python scripts/run_neb_ts.py input.pdb \
    --spring-k 3.0 --spring-fmax 3.0 \
    --md-steps 200 --md-temp 300
```

**How it works:**
1. Fix all atoms except ligand, apply attractive/repulsive springs
2. Optimize with springs active (drives bonds to target geometry)
3. Remove springs, relax with `ca-only` constraints
4. Short MD equilibration (200 steps Langevin at 300 K)
5. Final polish (LBFGS to fmax < 0.04 eV/A)

Both reactant and product are generated this way (with opposite spring directions).

### 2. Constrained Scan

For cases where the default springs produce poor endpoints. Incrementally scan the reaction coordinate while allowing everything else to relax at each step.

```bash
python scripts/run_neb_ts.py input.pdb \
    --endpoint-method scan --scan-steps 10
```

### 3. Incremental Stretch

Gradually stretch the breaking bond in small increments (0.1-0.2 A per step), relaxing between increments. Prevents sudden geometry distortion.

```bash
python scripts/run_neb_ts.py input.pdb \
    --endpoint-method incremental --stretch-step 0.15
```

### 4. Spring + Relax (single direction)

Use `--unidirectional` when the input is already a good reactant and only the product needs generating:

```bash
python scripts/run_neb_ts.py input.pdb \
    --unidirectional --direction forward
```

### Unidirectional vs Bidirectional

| Mode | When to use | Example |
|------|-------------|---------|
| **Bidirectional** (default) | Input is near TS or unknown geometry | Most theozyme inputs |
| **Unidirectional forward** | Input IS the reactant | You have a well-relaxed Michaelis complex |
| **Unidirectional reverse** | Input IS the product | Starting from post-reaction state |

---

## Spring Modes

Spring constraints define WHICH bonds to drive during endpoint generation. Critical for mechanistic investigation.

### Defined in `BOND_BREAKING_DEFS`

Each ligand type has defined spring targets:

```python
# PTE phosphoester hydrolysis (P-O bond breaking/forming)
"YYL": [
    ("P1", "O1", 1.4, "attractive"),   # nucleophile attack (form bond)
    ("P1", "O5", 3.5, "repulsive"),    # leaving group departure (break bond)
]
```

### Spring mode options

| Mode | What it drives | When to use |
|------|---------------|-------------|
| `both` | Both nucleophile attack AND leaving group departure | Default. Concerted mechanism assumed. |
| `nuc-only` | Only nucleophile attack (P-O_nuc bond forming) | Investigate stepwise: does nuc attack first? |
| `lg-only` | Only leaving group departure (P-O_lg bond breaking) | Investigate stepwise: does LG leave first? |

```bash
# Concerted (default)
python scripts/run_neb_ts.py input.pdb --spring-mode both

# Stepwise: nucleophile only
python scripts/run_neb_ts.py input.pdb --spring-mode nuc-only

# Stepwise: leaving group only
python scripts/run_neb_ts.py input.pdb --spring-mode lg-only
```

### Spring parameters (validated optimal ranges)

| Parameter | Optimal | Range tested | Notes |
|-----------|---------|-------------|-------|
| `--spring-k` | 3.0 | 2-10 | k=2 too soft (endpoints not separated), k=10 too aggressive (over-driven) |
| `--spring-fmax` | 3.0 | 2-6 | Force cap prevents energy blow-up. Match to spring-k. |

```
Sensitivity analysis (ZAPP P1D1 system):
  k=2   -> -0.6 kcal/mol   (too soft, endpoints overlap)
  k=3   -> 16-17 kcal/mol  (optimal)
  k=4   -> 17-19 kcal/mol  (good)
  k=6   -> 17-19 kcal/mol  (slightly aggressive)
  k=10  -> -1.9 kcal/mol   (over-driven, collapses)
```

---

## MD Strategies

MD equilibration after spring-driven relaxation finds the actual local minimum basin. Critical for realistic barriers.

### Available strategies

| Strategy | Flag | Steps | How it works | Best for |
|----------|------|-------|-------------|----------|
| `short` | `--md-strategy short` | 200 | Single NVT Langevin at 300 K | Default. Fast and reliable. |
| `annealing` | `--md-strategy annealing` | 200-400 | Heat to 600K, hold, cool to 300K | Escape shallow local minima |
| `multi-seed` | `--md-strategy multi-seed` | 5x200 | 5 independent MD runs, pick lowest E | Most robust endpoints |
| `long` | `--md-strategy long` | 500-1000 | Extended single NVT | Thorough sampling (risk: over-relaxation) |

### Accuracy vs efficiency tradeoffs

```
                    Time (relative)
short (200 steps)       1x          Good for screening. 95% of production quality.
annealing (300 steps)   2x          Overcomes barriers between shallow minima.
multi-seed (5x200)      5x          Most reliable. Best final endpoints.
long (500+ steps)       3-5x        Risk of over-relaxation: endpoint drifts too far.
```

### Validated sensitivity analysis

```
MD equilibration steps (ZAPP P1D1, mace-mp):
  100 steps -> 0.1 kcal/mol   (FAILURE: insufficient, trapped in shallow min)
  200 steps -> 16-17 kcal/mol  (OPTIMAL: reaches proper basin)
  300 steps -> ~20 kcal/mol    (conservative, slightly over-estimates)
  500 steps -> 6.1 kcal/mol    (OVER-RELAXED: endpoint too deep)
```

**Rule of thumb:** 200 steps for clean inputs, 300 for messy/chimeric inputs, never use 500+.

---

## Constraint Modes

Control which atoms are frozen during optimization and MD.

| Mode | Fixed during OPT/NEB | Fixed during MD | Use case |
|------|---------------------|-----------------|----------|
| `ca-only` (default) | CA atoms only | CA atoms only | Most cases. Sidechains and backbone C/N/O all move. |
| `backbone` | CA, C, N, O | CA only | When backbone needs rigidity but MD still needs flexibility. |
| `backbone-water` | CA, C, N, O + water O | CA only | Preserve water network during optimization. |
| `ca-restrained` | CA fixed + soft harmonic restraints on isolated termini | CA only | Isolated residues with dangling C/N. |
| `none` | Nothing | Nothing | Small theozymes (<50 atoms), gas-phase-like. |

### Chain-specific constraints

For chimeric structures with one designed chain and one scaffold chain:

```bash
# Fix only chain B (scaffold), let chain A (designed) be fully free
python scripts/run_neb_ts.py input.pdb \
    --constraint-mode ca-only --fix-chains B
```

### Which to use

- **Designed theozymes (RFdiffusion output)**: `ca-only` (default)
- **Chimeric (docked) structures**: `ca-only --fix-chains B` or `backbone`
- **Small cluster models (<50 atoms)**: `none`
- **When backbone wobble causes problems**: `backbone`
- **Active sites with important water bridges**: `backbone-water`

---

## Number of NEB Images

The number of interpolated structures along the reaction path.

| N images | Use case | Resolution | Notes |
|----------|----------|-----------|-------|
| 9 | Quick screening | ~0.1 A per image | May miss narrow barriers. Use for ranking many designs. |
| 11 | Economy production | Good | Occasionally misses barrier (seen in our tests). |
| 15 | **Production** (recommended) | ~0.05 A per image | Best balance of accuracy and cost. |
| 21-31 | Mechanistic investigation | Very fine | For distinguishing stepwise vs concerted, locating intermediates. |

### Validated sensitivity

```
NEB images (ZAPP P1D1):
  9  images -> 34.7 kcal/mol  (ROUGH: too few, over-estimates barrier significantly)
  11 images -> variable        (can miss the barrier entirely)
  15 images -> 18-20 kcal/mol  (OPTIMAL: consistent, correct barrier)
```

**Always use 15 for production runs. Use 9 only for initial screening of many systems.**

---

## Model Selection

### Available MACE models on DIGS

| Model | DFT Level | Metals? | Charge? | Min GPU | Best for |
|-------|-----------|---------|---------|---------|----------|
| `mace-mp` | r2SCAN | Yes | No | A4000 (16 GB) | General purpose. Default. Fast. |
| `mace-omol` | wB97M-V | Yes | Yes | A6000 (48 GB) | Best barrier accuracy. Gold standard. |
| `mace-mh --head omol` | wB97M-V | Yes | Yes | A4000 (16 GB) | Compact version of OMOL. |
| `mace-mh --head rgd1_b3lyp` | B3LYP | Yes | Yes | A4000 (16 GB) | TS-trained head. |
| `mace-off` | wB97M-D3BJ | No | No | A4000 (16 GB) | Organic molecules only. NO metals. |
| `mace-polar-[s/m/l]` | Polarizable | Yes | Yes | A4000+ | Long-range electrostatics. |

### Decision tree

1. **Default / fast screening**: `mace-mp` (fits on A4000, handles metals)
2. **Production barriers**: `mace-omol` (best accuracy, needs A6000/L40/H200)
3. **Budget production**: `mace-mh --head omol` (OMOL quality, A4000 fits)
4. **Organic-only (no metals)**: `mace-off` (best for pure organic TS)
5. **Charged metal pockets**: `mace-polar-m` (explicit polarization)

### Dual-model strategy (`--model-relax`)

Use a cheap model for relaxation/MD and an accurate model for the NEB energy surface:

```bash
python scripts/run_neb_ts.py input.pdb \
    --model mace-omol \
    --model-relax mace-mp
```

This runs endpoint generation (relax + MD) with mace-mp (fast, A4000) and the actual NEB path optimization with mace-omol (accurate, needs more VRAM). Saves significant GPU time on the cheap parts.

### Multi-head model heads

The `mace-mh` model contains 7 DFT levels in one checkpoint:

| Head | DFT Level | Notes |
|------|-----------|-------|
| `omol` (default) | wB97M-V | Same training as MACE-OMOL |
| `rgd1_b3lyp` | B3LYP-D3 | Trained on reaction gradient data including TS |
| `matpes_r2scan` | r2SCAN | Same level as MACE-MP |
| `spice_wB97M` | wB97M-D3BJ | Biomolecular training set |
| `ani2x` | wB97X | Small molecule benchmark |

```bash
# Use the B3LYP TS-trained head
python scripts/run_neb_ts.py input.pdb --model mace-mh --head rgd1_b3lyp
```

---

## Stepwise vs Concerted Mechanisms

### Concerted (default)

Both bond-breaking and bond-forming happen simultaneously. One TS connects reactant to product.

```bash
python scripts/run_neb_ts.py input.pdb --spring-mode both --n-images 15
```

### Investigating stepwise mechanisms

A stepwise mechanism has an intermediate between two transition states. To investigate:

**Step 1: Run with more images to resolve intermediates**
```bash
python scripts/run_neb_ts.py input.pdb \
    --spring-mode both --n-images 21
```

If the energy profile shows a shoulder or local minimum, the mechanism may be stepwise.

**Step 2: Run nucleophile-only and LG-only separately**
```bash
# Step A: nucleophilic attack only
python scripts/run_neb_ts.py input.pdb \
    --spring-mode nuc-only --n-images 15 \
    -o outputs/step_a_nuc_attack/

# Step B: leaving group departure only (starting from Step A product)
python scripts/run_neb_ts.py outputs/step_a_nuc_attack/product.pdb \
    --spring-mode lg-only --n-images 15 \
    -o outputs/step_b_lg_departure/
```

**Step 3: Compare barriers**

If TS_A + TS_B < TS_concerted, the mechanism is likely stepwise.
If TS_concerted < max(TS_A, TS_B), the mechanism is concerted.

---

## Validated Results

All PTE theozyme systems tested with the production pipeline.

### Cross-system comparison

| System | Atoms | Model | Barrier (kcal/mol) | Exp ΔG* | Diff | Activity | Settings |
|--------|-------|-------|-------------------|---------|------|----------|----------|
| Native PTE (lit) | ~400 | mace-mp | ~15-16 | 15.5 | - | kcat = 2100 s-1 | Reference |
| ZAPP P1D1 bestHIT | 890 | mace-mp | **19.5** | 21.5 | -2.0 | kcat = 0.008 s-1 | k=4, 250MD, 15img |
| ZAPP P1D1 (ca-only) | 890 | mace-mp | 19.6 | 21.5 | -1.9 | kcat = 0.008 s-1 | k=3, 200MD, 15img |
| Enhanced PTE set3 | 475 | mace-mp | 6.4 | ~15? | - | native-like | k=3, 200MD, 15img |
| Frankenstein ACHE_PTE | 1060 | mace-mp | **27.9** | ~30-35? | - | ~uncatalyzed | k=3, 300MD, 15img, pre-relax |
| Frankenstein (no pre-relax) | 1060 | mace-mp | 53.5 | ~30-35? | - | ~uncatalyzed | k=3, 300MD, 15img |
| Uncatalyzed (literature) | - | - | ~30-35 | 30-35 | - | ~10-8 s-1 | Reference |

### Key findings

1. The pipeline correctly **ranks all systems** across 4 orders of magnitude in activity
2. Barriers are within **~2 kcal/mol** of experiment for the system with known kcat
3. MACE-MP and MACE-OMOL agree on barrier heights (~19-21 kcal/mol for ZAPP P1D1)
4. `--pre-relax` is essential for chimeric inputs (53 -> 28 kcal/mol correction)

### Hyperparameter sensitivity (ZAPP P1D1)

| Parameter | Tested values | Optimal | Critical? |
|-----------|-------------|---------|-----------|
| spring-k | 2, 3, 4, 6, 10 | 3-4 | YES: k=2 and k=10 give nonsense |
| md-steps | 100, 200, 300, 500 | 200 | YES: 100 fails, 500 over-relaxes |
| n-images | 9, 11, 15 | 15 | YES: 9 and 11 can miss the barrier |
| constraint-mode | backbone, ca-only | ca-only | Minor: both give ~19.5 kcal/mol |
| model | mace-mp, mace-omol | both good | Minor: MP=19.5, OMOL=21.3 |
| pre-relax | on/off | off (clean inputs) | CRITICAL for chimeric inputs only |

---

## Common Pitfalls and How to Avoid Them

### 1. Sella crashes on large systems

**Symptom:** `MemoryError` or `LinAlgError` during Sella refinement.

**Cause:** Sella computes a full Hessian. For >200 free atoms this requires too much memory.

**Fix:** Use `--mode standard` (CI-NEB only, no Sella). The CI-NEB TS is usually sufficient.

```bash
# DO NOT use --mode full for systems > 500 atoms
python scripts/run_neb_ts.py input.pdb --mode standard  # safe
```

### 2. Inflated barriers from docking clashes

**Symptom:** Barriers of 50+ kcal/mol that drop to ~28 with pre-relax.

**Cause:** Chimeric/docked structures have steric clashes that lock the backbone in a high-energy state.

**Fix:** Always use `--pre-relax` for docked or chimeric inputs.

```bash
python scripts/run_neb_ts.py chimeric.pdb --pre-relax
```

### 3. Negative or zero barriers

**Symptom:** Barrier is 0 or negative.

**Cause:** Either (a) spring-k too soft/aggressive (endpoints not properly separated), or (b) too few/too many MD steps.

**Fix:** Check spring-k (should be 3-4) and md-steps (should be 200).

### 4. MACE-OMOL OOM on A4000

**Symptom:** `CUDA out of memory` with mace-omol.

**Cause:** MACE-OMOL (extra_large, 1024 channels) needs ~20+ GB VRAM for 890-atom systems.

**Fix:** Use B4000 (32 GB), H200 (80 GB), or use `--model-relax mace-mp` with `--model mace-omol`.

### 5. Poor NEB convergence

**Symptom:** NEB does not converge within step limit, energy profile is jagged.

**Cause:** (a) Poor initial path (endpoints too different), or (b) fmax too tight.

**Fix:**
- Increase `--steps-noclimb` to 300-400
- Loosen `--fmax-neb-noclimb 0.5`
- Try `--md-strategy annealing` for better endpoints

### 6. Wrong ligand detected

**Symptom:** `Expected exactly 1 known ligand, found: []`

**Cause:** The ligand residue name is not in `BOND_BREAKING_DEFS`.

**Fix:** Add a custom entry:
```python
BOND_BREAKING_DEFS['NEW_LIG'] = [
    ('atom_forming', 'atom_nuc', 1.4, 'attractive'),  # new bond
    ('atom_breaking', 'atom_lg', 2.5, 'repulsive'),   # old bond
]
```

### 7. Energy profile shows intermediate (shoulder)

**Symptom:** Energy profile has a bump or plateau rather than clean bell curve.

**Cause:** The mechanism may be stepwise, or the path is not smooth.

**Fix:**
- Increase images to 21 to resolve the intermediate
- If confirmed stepwise, run separate NEB for each step (see Stepwise section)

---

## Full CLI Reference

```
python scripts/run_neb_ts.py INPUT.pdb [OPTIONS]

Required:
  INPUT.pdb                     Input structure (active site cluster with ligand)

Model:
  --model MODEL                 MACE model key or path (default: mace-mp)
  --model-relax MODEL           Separate model for relaxation (cheaper)
  --head HEAD                   Multi-head model head (e.g., rgd1_b3lyp)

Mode:
  --mode {quick,standard,full}  Pipeline mode (default: standard)
      quick    = NEB only (no climbing image, no Sella)
      standard = NEB + climbing-image NEB (recommended)
      full     = NEB + CI-NEB + Sella + frequency validation

Constraints:
  --constraint-mode MODE        ca-only | backbone | backbone-water | ca-restrained | none
  --fix-chains CHAIN [CHAIN...] Only constrain atoms in these chains

NEB:
  --n-images N                  Number of NEB images (default: 15)
  --k-spring K                  NEB spring constant between images (default: 1.0)
  --fmax-neb-noclimb F          Convergence for no-climb phase (default: 0.40 eV/A)
  --steps-noclimb N             Max steps for no-climb (default: 200)
  --fmax-neb-climb F            Convergence for climbing image (default: 0.045 eV/A)
  --steps-climb N               Max steps for climbing (default: 250)

Endpoints:
  --spring-k K                  Bond-driving spring constant (default: 3.0)
  --spring-fmax F               Spring force cap (default: 3.0)
  --spring-mode MODE            both | nuc-only | lg-only (default: both)
  --fmax-end-spring F           Convergence during spring phase (default: 0.10)
  --fmax-end-final F            Final endpoint convergence (default: 0.04)

MD:
  --md-steps N                  MD equilibration steps (default: 200)
  --md-temp T                   MD temperature in K (default: 300)
  --md-strategy STRATEGY        short | annealing | multi-seed | long

Flags:
  --pre-relax                   Pre-relax entire structure before NEB (for chimeric)
  --skip-freq                   Skip frequency validation
  --skip-sella                  Skip Sella even in full mode
  --force-sella                 Force Sella even if >200 free atoms (risky)
  --unidirectional              Only generate one endpoint (input is the other)
  --direction {forward,reverse} Direction for unidirectional mode
  --resume                      Resume from checkpoint
  -o, --output DIR              Output directory
```

> **See also:** [TS search — method selection, pitfalls & sanity guards](ts_search_pitfalls_and_methods.md) — when to use CI-NEB vs GSM/FSM, the Sella-Cartesian collapse, and the automatic `ts_sanity` warnings.
