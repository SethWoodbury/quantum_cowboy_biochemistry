# Dependencies & paths

Single source of truth for paths is `quantum_engine/site.py` — edit there (or
override via env vars). This doc is a map of what lives where on the UW Baker-lab
/ DIGS cluster.

## Containers (the 2-container story)

Apptainer images under `/net/software/containers/users/woodbuse/quantum_chem/`:

| Image | Holds | Why separate |
|-------|-------|--------------|
| `quantum_chem-<date>.sif` (main) | MACE (incl. **POLAR** fork + `graph_longrange`), xTB, pysisyphus, Sella, ASE, the `quantum_engine` deps | the everyday container |
| `uma-<date>.sif` (sidecar) | FairChem UMA | needs numpy 2 / torch 2.8 — conflicts with the main image |

`site.py` auto-picks the newest `quantum_chem-*.sif` via glob. Run things with:

```bash
apptainer exec --nv --bind /home --bind /net \
  /net/software/containers/users/woodbuse/quantum_chem/quantum_chem-<date>.sif \
  python -m quantum_engine.cli <op> ...
```

The package is NOT pip-installed in the image — invoke as
`python -m quantum_engine.cli`.

## Model weights

`/net/databases/huggingface/mlFF_models/` is the **single canonical location** for
every MLFF weight the codebase references (`site._HF_HUB_BASE`); all
`site.MACE_MODELS` paths resolve under it. A new model = an entry there +
(optionally) a `register_energy` family (see [extending.md](extending.md)).

Present + wired (alias → family): the MACE family (`mace-mp`, `mace-off-*`,
`mace-omol` / **`mace-omol25`** [OMol25, charge-aware], `mace-mh`/`mace-mh-1`,
`mace-polar-*`), UMA (`uma-sm`, `uma-s-1p1`, `uma-s-1p2`, `uma-m-1p1` /
**`uma-m`**), `orb-mol-conservative`, `aimnet2-rxn`. `cowboy-qc list-models` shows them.

- UMA checkpoints: download via
  `/net/databases/huggingface/mlFF_models/download_uma_models.sh` (needs an
  `HF_TOKEN`; FAIR Chemistry License). 11 GB for `uma-m-1p1`.
- **eSEN** (`esen-s`, FairChem OMol25-trained) is **wired but not yet downloaded**
  — it routes through the same fairchem-core path as UMA (the UMA sidecar) and
  errors with a clear "checkpoint not on disk" until you add it to the download
  script and populate `models--facebook--esen-*`.

## External binaries (host/cluster, not in the container)

| Tool | `site.py` | Path |
|------|-----------|------|
| ORCA | `ORCA_BIN` | `/net/software/orca/orca_4_1_1_linux_x86-64_openmpi313/orca` (also 4.0.1.2) |
| Gaussian | `GAUSSIAN_ROOT` | `/net/software/gaussian/g16` |
| PLUMED 2 kernel | `PLUMED_KERNEL` | `$QCB_PLUMED_KERNEL` → `/net/software/lab/plumed2-2.10/lib/libplumedKernel.so` |
| CREST | — | `/net/software/lab/crest/bin/crest` |

**ORCA / Gaussian are not in the cowboy-qc container** — QM-native engine runs happen
host-side (a node where ORCA is installed). `cowboy-qc ts-entry --engine orca
--no-execute` prepares the input to `sbatch`.

## Vendored submodules (`deps/`, built into the main container)

- `deps/xtb` (`XTB_BIN`) — GFN0/1/2 + GFN-FF.
- `deps/g-xtb` (`GXTB_BIN`) — g-xTB (distinct from xtb).
- `deps/pysisyphus` — saddle (RS-P-RFO, dimer) + string methods (FSM/GSM).
  Prefer the *installed* pysisyphus; the vendored tree is a fallback only.
- `deps/graph_longrange_src` + `deps/mace_polar_src` — the MACE-POLAR fork
  (`PolarMACE`), installed `--no-deps` editable so it doesn't shadow stock MACE.
- `deps/pyGSM` — SE-GSM (single-ended), for the planned `gsm-se` path method.

## Vendored notebook helpers (`notebooks/lib/`)

`notebook_core.py` + `slurm_submission.py`, copied verbatim from the author's
`~/special_scripts/notebook_functions/`. They power the command/`sbatch`-generator
notebooks. **DIGS-specific**: `slurm_submission.py` hardcodes Baker-lab SLURM
partitions/GRES/GPU classes — edit there for another cluster. See
`notebooks/lib/README.md`.

## Other lab data

`/net/databases` (lab-shared data: PDB/CIF mirrors, M-CSA, HF models),
`/net/software/lab` (lab installs). Scratch: `$SCRATCH` →
`/net/scratch/$USER`.
