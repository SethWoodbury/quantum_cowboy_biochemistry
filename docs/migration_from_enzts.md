# Migrating from enz-ts → qcb

The plan is to deprecate `enz-ts` (the Lars/gbg222 codebase at
`/projects/ml/enzyme_filtering/enz-ts/`) entirely once `cowboy-qc` covers all
the same workflows. enz-ts remains read-only as a reference; we don't
modify it.

## Status table

| enz-ts capability | cowboy-qc status | Plan |
|-------------------|-----------|------|
| MACE-OMOL/MH/OFF model registry | ✓ in `qcb/calc/factory.py` | done |
| **MACE-POLAR-1 S/M/L** (charge-aware) | ✓ added to factory | **done — use `--model mace-polar`** |
| Pre-built PLUMED kernel | ✓ path in `qcb/config.py` | done — auto-detected |
| PLUMED2 source (submodule) | ✓ at `deps/plumed2/` | added; build with `bash deps/plumed2_build.sh` (~15 min) |
| Pure-Python WT-MTD | ✓ `qcb/mlff/metadynamics.py` | works (single CV) |
| Pure-Python OPES | ✓ same file | works |
| FES from PLUMED HILLS | ✓ `qcb/analysis/fes.py` | new — clean rewrite, not a port |
| FES via Tiwary-Parrinello reweighting | ✓ same file | new |
| Umbrella sampling (single window) | (use `cowboy-qc md` + harmonic CV restraint) | works manually |
| Umbrella sampling (multi-window WHAM/MBAR) | ✓ `fes.py:fes_from_umbrella_pymbar` | works (requires `pymbar`) |
| Multi-walker MTD via shared filesystem | ✗ | TODO — add via PLUMED runner |
| Hill merging across walkers | (handled by PLUMED itself) | handled by PLUMED |
| Basin detection / TS identification | ✓ `fes.py:find_basins`, `barrier_between_basins` | works |
| YAML config with biotite expression grammar | ✗ | TODO — port from enz-ts `parse_config.py` |
| QM cluster cutoff + H-capping | partial (`qcb/prep/extract.py`) | TODO — port enz-ts `capping_utils.py` |
| ChimeraX-based protonation | ✗ (we use pdbfixer/reduce/propka) | TODO — port enz-ts `chimera_utils.py` |
| Per-system YAML configs (PTE variants) | ✗ | TODO — port `seth_pte/*.yaml` from `/net/scratch/woodbuse/metad/config/` |
| `Enzyme` class abstraction | ✗ | TODO — port `enzyme_class.py` |
| Production NEB (different from ours) | ✓ `qcb/ops/neb.py` | cowboy-qc has native + tested |
| IRC-from-TS | ✓ `qcb/ops/irc.py` | qcb-only feature (enz-ts doesn't have) |
| FSM/GSM (pysisyphus) | ✓ `qcb/ops/gsm.py` | qcb-only |
| Auto spring constant (Pauling/xTB) | ✓ `qcb/mlff/auto_spring_k.py` | qcb-only |
| Geodesic interpolation | ✓ `qcb/mlff/interpolation.py` | qcb-only |

## What you can do today (fully native qcb)

```bash
# Use MACE-POLAR-1-M for charged systems (ions, phosphates, etc.)
cowboy-qc opt input.pdb --model mace-polar --fmax 0.01

# Pure-Python WT-MTD on bond-difference CV
cowboy-qc mtd input.pdb --model mace-polar \
    --p-idx 162 --nuc-idx 164 --lg-idx 166 \
    --time 50 --temp 300

# Analyze any PLUMED HILLS file (from our MTD or enz-ts)
python -c "from qcb.analysis.fes import analyze_hills_file; analyze_hills_file('runs/walker_0/HILLS')"

# Native TS pipeline (no charge bug)
cowboy-qc ts input.pdb --model mace-polar --strategy cv-spring --fix-preset ca-only
```

## What needs PLUMED2 built (or use prebuilt at `/net/scratch/woodbuse/metad/plumed/`)

- ASE-driven MTD with arbitrary CVs (RMSD, coordination, path, ...)
- Multi-walker MTD via file-based walkers
- OPES with full PLUMED feature set

To use the prebuilt:
```bash
export PLUMED_KERNEL=/net/scratch/woodbuse/metad/plumed/lib/libplumedKernel.so
export LD_LIBRARY_PATH=/net/scratch/woodbuse/metad/plumed/lib:$LD_LIBRARY_PATH
```
Or build our own submodule (~15 min):
```bash
bash deps/plumed2_build.sh
source deps/plumed2/install/setup.sh
```

## What's still in enz-ts only (planned ports)

These are the remaining items to port before enz-ts can be fully retired:

1. **YAML config grammar with biotite expressions** — `enz-ts/src/enzts/utils/parse_config.py`
   - Lets users write: `P1: {expr: "res_YYL & atom_P1", type: position}`
   - Way better than our hardcoded BOND_BREAKING_DEFS dict
   - ETA: ~1 day to port + test

2. **QM cluster cutoff + H-capping** — `enz-ts/src/enzts/utils/capping_utils.py`
   - Cuts an active-site sphere from a full enzyme + caps backbones with H
   - Critical for going from full PDB → cluster model
   - ETA: ~1 day

3. **ChimeraX-based protonation** — `enz-ts/src/enzts/utils/chimera_utils.py`
   - More accurate than our pdbfixer+reduce+propka cocktail
   - Requires ChimeraX kernel (also at `enz-ts/kernels/chimerax/`)
   - ETA: ~1 day

4. **Per-system YAML configs** — `/net/scratch/woodbuse/metad/config/seth_pte/`
   - 8 ready-to-use PTE configs (wBB_wKCX, wBB_wGLU, ZAPP variants, ...)
   - Just copy + adapt the schema
   - ETA: ~1 hour

5. **Multi-walker PLUMED MTD runner** — `enz-ts/src/enzts/utils/plumed.py`
   - Manages parallel walker dirs, hill merging, etc.
   - ETA: ~2 days

Total remaining: ~1 week of focused work. After that, enz-ts can be archived.

## Reference paths

- enz-ts source: `/projects/ml/enzyme_filtering/enz-ts/` (read-only)
- enz-ts venv: `/projects/ml/enzyme_filtering/enz-ts/.venv/` (read-only)
- Your MTD runs: `/net/scratch/woodbuse/metad/runs/seth_pte-*-trial-*/`
- Your configs: `/net/scratch/woodbuse/metad/config/seth_pte/*.yaml`
- Prebuilt PLUMED: `/net/scratch/woodbuse/metad/plumed/lib/libplumedKernel.so`

## Why deprecate at all?

enz-ts works. But:
- It's a separate codebase requiring its own venv (`uv sync`)
- It depends on a private repo (`enz-proj`) for configs
- Two codebases for one job (TS searches) → context-switching cost
- cowboy-qc already has features enz-ts lacks (IRC-from-TS, FSM/GSM, auto-k, native ts, modular CLI)
- One repo means one set of tests, one set of docs, one place to fix bugs
