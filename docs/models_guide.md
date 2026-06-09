# MACE Models Guide

All ML force field models available for QCB on the DIGS cluster, their capabilities, GPU requirements, and when to use each.

---

## Model Summary Table

| Model | Key | DFT Level | Elements | Metals? | Charge? | Size | Best for |
|-------|-----|-----------|----------|---------|---------|------|----------|
| MACE-MP | `mace-mp` | r2SCAN (MatPES + OMAT) | All | Yes | No | Medium | General purpose, default |
| MACE-OMOL | `mace-omol` | wB97M-V | All | Yes | Yes | Extra-Large (1024ch) | Best barrier accuracy |
| MACE-MH | `mace-mh` | 7 heads (see below) | All | Yes | Yes | Medium | Multi-level, versatile |
| MACE-OFF Small | `mace-off-small` | wB97M-D3BJ | H,C,N,O,F,S | No | No | Small | Fast organic screening |
| MACE-OFF Medium | `mace-off-medium` | wB97M-D3BJ | H,C,N,O,F,S | No | No | Medium | Organic molecules |
| MACE-OFF Large | `mace-off` | wB97M-D3BJ | H,C,N,O,F,S | No | No | Large | Best organic accuracy |
| MACE-POLAR-S | `mace-polar-s` | Polarizable | All | Yes | Yes | Small | Long-range, fast |
| MACE-POLAR-M | `mace-polar-m` | Polarizable | All | Yes | Yes | Medium | Long-range, balanced |
| MACE-POLAR-L | `mace-polar-l` | Polarizable | All | Yes | Yes | Large | Long-range, best |
| UMA-SM | `uma-sm` | Multi-task | All | Yes | Yes | Small-Medium | FairChem ecosystem |

---

## Paths on DIGS

```python
# From qcb/config.py — these are the actual file paths on the cluster

MACE_MODELS = {
    # General-purpose (r2SCAN, all elements including metals)
    "mace-mp": "/mnt/projects/ml/mlff/models/mace_mp/MACE-matpes-r2scan-omat-ft.model",

    # Organic molecules (wB97M-D3BJ, NO metals)
    "mace-off-small":  "/mnt/projects/ml/mlff/models/mace_off/MACE-OFF23_small.model",
    "mace-off-medium": "/mnt/projects/ml/mlff/models/mace_off/MACE-OFF23_medium.model",
    "mace-off":        "/mnt/projects/ml/mlff/models/mace_off/MACE-OFF23_large.model",

    # Charge-aware, trained on TS data (wB97M-V, all elements)
    "mace-omol": "/home/gbg222/projects/mace_models/MACE-omol-0-extra-large-1024.model",

    # Multi-head (7 DFT levels in one model)
    "mace-mh": "/home/gbg222/projects/mace_models/mace-mh-0.model",

    # Polarizable — BAKED INTO the main container (deps/quantum_chem.def installs
    # graph_longrange + the PolarMACE fork); loads in-process, no separate venv.
    "mace-polar-s": "/home/gbg222/projects/mace_models/MACE-POLAR-1-S.model",
    "mace-polar-m": "/home/gbg222/projects/mace_models/MACE-POLAR-1-M.model",
    "mace-polar-l": "/home/gbg222/projects/mace_models/MACE-POLAR-1-L.model",

    # FairChem UMA (different calculator interface, NOT MACECalculator)
    "uma-sm": "/mnt/projects/ml/mlff/models/fairchem/UMA/uma_sm.pt",
}
```

---

## GPU Memory Requirements

Measured on the ZAPP P1D1 system (890 atoms) and Frankenstein (1060 atoms):

| Model | ~890 atoms | ~1060 atoms | ~475 atoms | Min GPU |
|-------|-----------|------------|-----------|---------|
| `mace-mp` | ~8 GB | ~10 GB | ~5 GB | **A4000 (16 GB)** |
| `mace-mh` | ~10 GB | ~12 GB | ~6 GB | **A4000 (16 GB)** |
| `mace-off` | ~12 GB | ~14 GB | ~7 GB | **A4000 (16 GB)** |
| `mace-omol` | ~20 GB | ~28 GB | ~12 GB | **A6000 (48 GB)** or B4000 (32 GB) |
| `mace-polar-m` | ~14 GB | ~18 GB | ~8 GB | **A4000 (16 GB)** (tight) |
| `mace-polar-l` | ~22 GB | ~30 GB | ~13 GB | **B4000 (32 GB)** |

### GPU options on DIGS

| GPU | VRAM | Partition | Queue time | Notes |
|-----|------|-----------|-----------|-------|
| A4000 | 16 GB | `gpu` | Fast | Most available. Fits mace-mp, mace-mh. |
| B4000 | 32 GB | `gpu-b4000` | Medium | Fits mace-omol for <600 atoms. |
| A6000 | 48 GB | `gpu` | Medium | Fits mace-omol for ~1000 atoms. |
| L40 | 48 GB | `gpu-bf` | Medium | Same as A6000 capacity. |
| H200 | 80 GB | `gpu-bf` | Slow | Fits everything. Fastest compute. |

### SLURM request examples

```bash
# mace-mp on A4000 (most common)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --mem=32g

# mace-omol on H200
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32g
```

---

## DFT Level of Theory per Model

### MACE-MP

- **Level:** r2SCAN (MatPES training set + OMAT fine-tuning)
- **Functional:** r2SCAN meta-GGA
- **Basis:** Not applicable (plane-wave periodic DFT reference)
- **Strengths:** All elements Z=1-89, good geometries, reasonable barriers
- **Weaknesses:** No explicit charge handling; barriers can be slightly under-estimated

### MACE-OMOL (recommended for barriers)

- **Level:** wB97M-V / def2-TZVPPD
- **Functional:** wB97M-V (range-separated hybrid + VV10 nonlocal correlation)
- **Training set:** OMol25 (organic molecules + metals, includes TS geometries)
- **Strengths:** Best barrier accuracy, charge-aware, trained on TS data
- **Weaknesses:** Large model (1024 channels), needs A6000+ for >500 atoms

### MACE-MH (multi-head)

Seven DFT heads in one model file:

| Head | DFT Level | Training set | Notes |
|------|-----------|-------------|-------|
| `omol` | wB97M-V/def2-TZVPPD | OMol25 | Same quality as MACE-OMOL. **Default.** |
| `rgd1_b3lyp` | B3LYP-D3/def2-SVP | Reaction Gradient Data | TS-optimized, good for barriers |
| `matpes_r2scan` | r2SCAN | MatPES | Same as MACE-MP level |
| `spice_wB97M` | wB97M-D3BJ/def2-TZVPPD | SPICE | Biomolecular focus |
| `ani2x` | wB97X/6-31G* | ANI-2x | Small molecules, fast |
| `tmqm` | TPSSh/def2-SVP | tmQM | Transition metal complexes |
| `dipeptides` | wB97M-D3BJ/def2-TZVPPD | Dipeptides | Protein backbone |

Select a head with:
```bash
python scripts/run_neb_ts.py input.pdb --model mace-mh --head rgd1_b3lyp
```

### MACE-OFF

- **Level:** wB97M-D3BJ / def2-TZVPPD
- **Training set:** OFF23 (organic molecules, drug-like)
- **Elements:** H, C, N, O, F, S only
- **Strengths:** Very accurate for pure organic systems
- **Weaknesses:** NO metals, NO phosphorus -- cannot use for PTE theozymes!

### MACE-POLAR

- **Level:** Polarizable force field with explicit atomic dipoles
- **Physics:** Short-range MACE + long-range electrostatics via graph_electrostatics
- **Strengths:** Proper long-range interactions, important for charged pockets
- **Weaknesses:** Requires `graph_electrostatics` package (only in gbg222's venv)
- **Note:** Run through gbg222's environment, not the universal container directly

### UMA (FairChem)

- **Level:** Multi-task (trained on multiple datasets simultaneously)
- **Calculator:** FairChem `OCPCalculator` (NOT `MACECalculator`)
- **Strengths:** Broad coverage, active development
- **Weaknesses:** Different API, not yet integrated into run_neb_ts.py

---

## When to Use Which Model

### Decision flowchart

```
Does your system have metals?
  |
  +-- NO --> Use mace-off (best organic accuracy)
  |
  +-- YES
        |
        Do you need barrier accuracy <2 kcal/mol?
          |
          +-- NO --> Use mace-mp (fast, A4000)
          |
          +-- YES
                |
                Can you get A6000/H200?
                  |
                  +-- YES --> Use mace-omol (gold standard)
                  |
                  +-- NO --> Use mace-mh --head omol (compact, A4000)
```

### Use case recommendations

| Use case | Model | Why |
|----------|-------|-----|
| Quick screening (many designs) | `mace-mp` | Fast, fits A4000, good enough for ranking |
| Production barriers (publication) | `mace-omol` | Best accuracy, validated against experiment |
| Budget production | `mace-mh --head omol` | OMOL quality on A4000 |
| Organic reactions (no metals) | `mace-off` | Trained specifically on organic TS |
| Charged metal pocket (electrostatics matter) | `mace-polar-m` | Explicit polarization |
| Multi-level comparison | `mace-mh` (multiple heads) | Compare B3LYP vs wB97M in one run |

---

## Dual-Model Strategy (`--model-relax`)

Split computation between a cheap model (relaxation/MD) and an expensive model (NEB energy surface):

```bash
python scripts/run_neb_ts.py input.pdb \
    --model mace-omol \
    --model-relax mace-mp
```

### How it works

| Phase | Model used | Why |
|-------|-----------|-----|
| Endpoint spring-driving | `--model-relax` (cheap) | Geometry only, accuracy less critical |
| Endpoint MD equilibration | `--model-relax` (cheap) | Sampling, not energetics |
| Endpoint final polish | `--model-relax` (cheap) | Geometry refinement |
| NEB path optimization | `--model` (expensive) | Energy differences matter here |
| CI-NEB climbing image | `--model` (expensive) | TS energy critical |
| Sella TS refinement | `--model` (expensive) | Saddle point requires accuracy |

### Benefits

- **GPU efficiency:** Relaxation/MD runs on A4000 with mace-mp (~60% of wall time)
- **Accuracy where it matters:** NEB uses mace-omol for the energy surface
- **Workflow:** Can run relaxation as a separate job on A4000, then NEB on H200

### Caveats

- Endpoints optimized with one PES may not be perfect minima on the other
- Works well because mace-mp and mace-omol agree on geometries (just differ slightly on energetics)
- NOT recommended to mix very different models (e.g., mace-off for relax + mace-omol for NEB)

---

## Model Comparison: Our Validated Results

All tested on ZAPP P1D1 (890 atoms, experimental ΔG* = 21.5 kcal/mol):

| Model | Barrier | Wall time | GPU | Agreement |
|-------|---------|----------|-----|-----------|
| mace-mp (r2SCAN) | 19.5 kcal/mol | ~129 min | A4000 | -2.0 kcal/mol vs exp |
| mace-omol (wB97M-V) | 21.3 kcal/mol | ~240 min | A6000 | -0.2 kcal/mol vs exp |
| mace-mh --head omol | ~20 kcal/mol | ~145 min | A4000 | ~-1.5 kcal/mol vs exp |

**Takeaway:** mace-mp is already within 2 kcal/mol. mace-omol is slightly better but 2x slower and needs bigger GPU. For screening, mace-mp is the clear choice. For publication, use mace-omol.

---

## Special Considerations

### MACE-POLAR requires a different environment

The POLAR models use `graph_electrostatics` which is NOT in the universal container. Use gbg222's venv:

```bash
# Not: apptainer exec ... /net/software/containers/universal.sif python
# Instead:
/home/gbg222/projects/enz-ts/.venv/bin/python scripts/run_neb_ts.py \
    input.pdb --model mace-polar-m
```

### UMA requires FairChem calculator

UMA models use `OCPCalculator` not `MACECalculator`. Currently not integrated into `run_neb_ts.py`. To use:

```python
from fairchem.core import OCPCalculator
calc = OCPCalculator(checkpoint_path="/mnt/projects/ml/mlff/models/fairchem/UMA/uma_sm.pt")
atoms.calc = calc
```

### Auto-download fallback

If a model file is not found at its DIGS path, the script attempts auto-download from HuggingFace:
- `mace-mp` -> `mace.calculators.mace_mp()`
- `mace-omol` -> `mace.calculators.mace_omol(model="extra_large")`
- `mace-off` -> `mace.calculators.mace_off(model="large")`

This works on any machine with internet access but downloads can be large (>1 GB for OMOL).
