"""SLURM submission helpers for qcb's HPC workflows.

Two layers:
  * :mod:`.submit_walkers` — the multi-walker MTD sweep submitter
    (chained array jobs, config snapshot, dependency=afterany).
  * :mod:`.job_runner` — generic single-job submit-and-wait used by
    Pipeline Steps that need a GPU node for one-shot work
    (MACE-POLAR-1M refinement, NEB, TS polish). Reads partition
    defaults from ``QCB_SLURM_PARTITION_{GPU,CPU}`` env vars or falls
    back to DIGS-conventional 'gpu'/'cpu' (Seth's L40 partition for GPU).
"""
from quantum_engine.slurm.submit_walkers import (
    submit_walker_sweep,
    run_local_sweep,
    load_config_snapshot,
)
from quantum_engine.slurm.job_runner import (
    JobConfig,
    JobResult,
    slurm_available,
    submit_job,
    wait_for_job,
    submit_and_wait,
    query_job_state,
)

__all__ = [
    # Walker sweep (multi-walker MTD)
    "submit_walker_sweep",
    "run_local_sweep",
    "load_config_snapshot",
    # Generic job runner
    "JobConfig",
    "JobResult",
    "slurm_available",
    "submit_job",
    "wait_for_job",
    "submit_and_wait",
    "query_job_state",
]
