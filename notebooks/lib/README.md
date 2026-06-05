# Vendored notebook helpers

`notebook_core.py` and `slurm_submission.py` are **vendored verbatim** from the
author's `~/special_scripts/notebook_functions/` and kept together as an
importable pair. They power the command/`sbatch`-generator notebooks in
`notebooks/` (the notebook builds the `python -m quantum_engine.cli ...` command
strings; the heavy compute runs as SLURM jobs, never in a cell).

## Use

In a notebook's INIT cell:

```python
import sys
sys.path.insert(0, "<repo>/notebooks/lib")   # so `import notebook_core` finds slurm_submission
import notebook_core as nb

WORKING_DIR = nb.resolve_working_dir(_WORKING_DIR_OVERRIDE)
OUTPUT_DIR  = nb.resolve_output_dir(WORKING_DIR, _OUTPUT_DIR_OVERRIDE)
nb.setup_directories(WORKING_DIR, WORKING_SUBDIRS, export_globals=True, globals_dict=globals())
nb.setup_directories(OUTPUT_DIR,  OUTPUT_SUBDIRS,  export_globals=True, globals_dict=globals())
nb.print_initialization(WORKING_DIR, OUTPUT_DIR, project_name=PROJECT_NAME)
```

Public API: `resolve_working_dir` / `resolve_output_dir` / `setup_directories`
(creates subdirs + exports `<NAME>_DIR` globals) / `print_initialization` /
`set_pandas_display`, and the re-exported `submit_cpu` / `submit_array_job`
(SLURM array submit with requeue/resilience/GPU-targeting).

## ⚠️ DIGS-specific

`slurm_submission.py` hardcodes **UW Baker-lab DIGS** SLURM partitions, GRES, and
GPU classes. On another cluster, edit the partition/GRES/queue names there (or
generate your own `sbatch` scripts). This is the one cluster-specific corner of
the otherwise portable pipeline.

These two files are intentionally not modified from upstream — update them by
re-copying from the author's `notebook_functions/` if they change there.
