"""Generic SLURM submit-and-wait helper for QCB pipelines.

Where :mod:`.submit_walkers` handles the multi-walker MTD sweep
specifically, this module is the reusable single-job submitter the
pipelines call when a Step needs to land on a GPU node.

Usage pattern in a Pipeline Step:

    from quantum_engine.slurm.job_runner import JobConfig, submit_and_wait
    cfg = JobConfig(
        command=f"{python} -m enz_qc_pipelines.mcsa_theozyme.refine_one ...",
        workdir=outdir / "slurm",
        gpu=True,
        time="04:00:00",
    )
    result = submit_and_wait(cfg)
    if result.exit_code != 0:
        raise RuntimeError(f"GPU job {result.job_id} failed; logs at {result.stderr_path}")

Defaults assume the DIGS L40 partition (Seth's user memory says
'Use L40 on DIGS, not A4000' — A4000 has 13-day queue). Override via
:class:`JobConfig` fields or QCB_SLURM_* env vars.

The poller uses ``squeue --json`` when available (SLURM ≥ 20.11),
falls back to ``sacct``. If neither is on PATH, raises with a clear
hint.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("quantum_engine.slurm.job_runner")

# Terminal job states from squeue/sacct
TERMINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
    "OUT_OF_MEMORY", "BOOT_FAIL", "NODE_FAIL", "PREEMPTED",
}
SUCCESS_STATES = {"COMPLETED"}


# ─────────────────────────────────────────────────────────────────────
# Config + result types
# ─────────────────────────────────────────────────────────────────────

@dataclass
class JobConfig:
    """A SLURM job spec. Only ``command`` + ``workdir`` are required."""
    command: str                              # the shell command to run
    workdir: Path                             # also where logs land
    job_name: str = "qcb_job"
    partition: str | None = None              # falls back to env / L40 default
    account: str | None = None
    gpu: bool = False                         # adds --gres=gpu:1 if True
    n_gpus: int = 1
    cpus: int = 4
    mem: str = "16G"
    time: str = "04:00:00"
    nodes: int = 1
    extra_directives: list[str] = field(default_factory=list)
    env_modules: list[str] = field(default_factory=list)  # `module load <x>` lines
    setup_lines: list[str] = field(default_factory=list)  # extra bash before command
    dependency: str | None = None             # e.g. 'afterok:12345'

    # Optional apptainer wrapping — when set, the script wraps `command`
    # in `apptainer exec [--nv] --bind <binds> <container> <command>`.
    # ``container`` accepts a CONTAINERS key (e.g. "qcb") or absolute
    # .sif path. Binds default to /home + /net + /mnt.
    container: str | None = None
    container_binds: tuple[str, ...] = ("/home", "/net", "/mnt")

    def resolve_partition(self) -> str:
        """Resolve partition: explicit > env > L40 default for GPU else cpu."""
        if self.partition:
            return self.partition
        env_part = os.environ.get(
            "QCB_SLURM_PARTITION_GPU" if self.gpu else "QCB_SLURM_PARTITION_CPU"
        )
        if env_part:
            return env_part
        # DIGS conventional defaults: L40 GPU partition, generic CPU partition
        return "gpu" if self.gpu else "cpu"

    def wrap_command(self) -> str:
        """Return the actual shell command with optional apptainer prefix."""
        if not self.container:
            return self.command
        from quantum_engine.site import apptainer_exec
        return apptainer_exec(
            self.container, self.command,
            gpu=self.gpu, binds=self.container_binds,
        )


@dataclass
class JobResult:
    """Outcome of a submit-and-wait."""
    job_id: int
    state: str                          # final SLURM state
    exit_code: int                      # 0 on COMPLETED, nonzero else
    wall_seconds: float
    stdout_path: Path
    stderr_path: Path
    script_path: Path

    @property
    def succeeded(self) -> bool:
        return self.state in SUCCESS_STATES and self.exit_code == 0


# ─────────────────────────────────────────────────────────────────────
# Capability check
# ─────────────────────────────────────────────────────────────────────

def slurm_available() -> bool:
    """Are sbatch + squeue (or sacct) on PATH?"""
    return shutil.which("sbatch") is not None and (
        shutil.which("squeue") is not None or shutil.which("sacct") is not None
    )


def _require_slurm() -> None:
    if not slurm_available():
        raise RuntimeError(
            "sbatch / squeue not on PATH. Either run on a SLURM head node, "
            "or run the Step locally instead of submitting (most Steps support "
            "an in-process fallback)."
        )


# ─────────────────────────────────────────────────────────────────────
# Sbatch script generation
# ─────────────────────────────────────────────────────────────────────

def _format_sbatch(cfg: JobConfig) -> str:
    workdir = Path(cfg.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    stdout = workdir / f"{cfg.job_name}.out"
    stderr = workdir / f"{cfg.job_name}.err"

    directives = [
        f"#SBATCH -J {cfg.job_name}",
        f"#SBATCH -p {cfg.resolve_partition()}",
        f"#SBATCH -N {cfg.nodes}",
        f"#SBATCH -c {cfg.cpus}",
        f"#SBATCH --mem={cfg.mem}",
        f"#SBATCH -t {cfg.time}",
        f"#SBATCH -o {stdout}",
        f"#SBATCH -e {stderr}",
    ]
    if cfg.account:
        directives.append(f"#SBATCH -A {cfg.account}")
    if cfg.gpu:
        directives.append(f"#SBATCH --gres=gpu:{cfg.n_gpus}")
    if cfg.dependency:
        directives.append(f"#SBATCH --dependency={cfg.dependency}")
    directives.extend(f"#SBATCH {d}" for d in cfg.extra_directives)

    body_lines: list[str] = []
    for mod in cfg.env_modules:
        body_lines.append(f"module load {mod}")
    body_lines.extend(cfg.setup_lines)
    body_lines.append(f"cd {workdir}")
    body_lines.append(cfg.wrap_command())

    return "\n".join([
        "#!/usr/bin/env bash",
        *directives,
        "set -euo pipefail",
        *body_lines,
        "",
    ])


# ─────────────────────────────────────────────────────────────────────
# Submit + poll
# ─────────────────────────────────────────────────────────────────────

def submit_job(cfg: JobConfig) -> tuple[int, Path]:
    """Submit a SLURM job. Returns (job_id, script_path)."""
    _require_slurm()
    workdir = Path(cfg.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    script_text = _format_sbatch(cfg)
    script_path = workdir / f"{cfg.job_name}.sh"
    script_path.write_text(script_text)
    script_path.chmod(0o755)

    log.info(f"sbatch {script_path}")
    r = subprocess.run(
        ["sbatch", "--parsable", str(script_path)],
        capture_output=True, text=True, check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"sbatch failed (exit {r.returncode}):\n  stdout: {r.stdout}\n  stderr: {r.stderr}"
        )
    # --parsable prints just "<jobid>" or "<jobid>;<cluster>"
    job_id = int(r.stdout.strip().split(";")[0])
    log.info(f"  → job {job_id}")
    return job_id, script_path


def query_job_state(job_id: int) -> str:
    """Return the current SLURM state of a job. Tries squeue first
    (faster, only sees active jobs); falls back to sacct (also sees
    completed). Returns 'UNKNOWN' if neither tool can find the job."""
    # squeue first — only knows active/pending
    if shutil.which("squeue"):
        r = subprocess.run(
            ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
            capture_output=True, text=True, check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[0]
    # Fall back to sacct (knows finished)
    if shutil.which("sacct"):
        r = subprocess.run(
            ["sacct", "-j", str(job_id), "--format=State", "--noheader",
             "-X", "-P"],
            capture_output=True, text=True, check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            # "RUNNING" or "COMPLETED" or "FAILED" — first line, strip
            # SLURM's trailing "+" suffix on long state names
            state = r.stdout.strip().splitlines()[0].strip().rstrip("+")
            return state.split()[0]
    return "UNKNOWN"


def query_exit_code(job_id: int) -> int:
    """Sacct ExitCode is 'RC:Signal' — return RC as int. 0 on
    completion-not-found (which usually means the job hasn't reached
    sacct yet)."""
    if not shutil.which("sacct"):
        return 0
    r = subprocess.run(
        ["sacct", "-j", str(job_id), "--format=ExitCode", "--noheader",
         "-X", "-P"],
        capture_output=True, text=True, check=False,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return 0
    code_str = r.stdout.strip().splitlines()[0].split(":")[0]
    try:
        return int(code_str)
    except ValueError:
        return 1


def wait_for_job(
    job_id: int,
    *,
    poll_interval: int = 30,
    timeout: int | None = None,
) -> str:
    """Block until the job reaches a terminal state. Returns the final
    state. If ``timeout`` is set and exceeded, raises TimeoutError."""
    start = time.time()
    while True:
        state = query_job_state(job_id)
        if state in TERMINAL_STATES:
            return state
        elapsed = time.time() - start
        if timeout is not None and elapsed > timeout:
            raise TimeoutError(
                f"job {job_id} still in state {state} after {timeout}s"
            )
        log.debug(f"  job {job_id} state={state} t={elapsed:.0f}s; "
                  f"sleeping {poll_interval}s")
        time.sleep(poll_interval)


def submit_and_wait(
    cfg: JobConfig,
    *,
    poll_interval: int = 30,
    timeout: int | None = None,
) -> JobResult:
    """Submit, poll, return outcome. The orchestrator's standard
    GPU-step pattern."""
    start = time.time()
    job_id, script_path = submit_job(cfg)
    state = wait_for_job(job_id, poll_interval=poll_interval, timeout=timeout)
    exit_code = query_exit_code(job_id) if state == "COMPLETED" else 1
    workdir = Path(cfg.workdir).resolve()
    return JobResult(
        job_id=job_id,
        state=state,
        exit_code=exit_code,
        wall_seconds=time.time() - start,
        stdout_path=workdir / f"{cfg.job_name}.out",
        stderr_path=workdir / f"{cfg.job_name}.err",
        script_path=script_path,
    )
