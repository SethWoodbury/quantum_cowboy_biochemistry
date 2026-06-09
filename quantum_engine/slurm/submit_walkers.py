"""SLURM array submitter for multi-walker metadynamics sweeps.

Generates a single sbatch script per round and chains rounds together
with ``--dependency=afterany``. Each round submits an array of size
``n_walkers``; ``$SLURM_ARRAY_TASK_ID`` becomes the walker id.

Modernisation vs enz-ts ``submit_rounds.sh`` + ``launch_walkers.sh``:

* Pure Python — no shell stage. The sbatch script we generate just
  exec's a Python entry point that calls
  :func:`quantum_engine.ops.mtd_walkers.run_walker`. No env-var smuggling beyond
  the four SLURM provides natively.
* Config is one JSON snapshot pickled to disk at submit time; every
  walker reads the same snapshot. Eliminates the rounds-edit-config
  race in enz-ts.
* Round chaining happens via Python sbatch returns, so we can also do
  *local* multi-round sweeps for development without SLURM (call
  :func:`run_local_sweep` instead).

Public API
----------
- :func:`submit_walker_sweep` — submit a SLURM sweep, return job ids.
- :func:`run_local_sweep` — run all walkers serially on the local node
  (for tests / debugging / single-machine work).
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from quantum_engine.ops.mtd_walkers import WalkerSweepConfig

log = logging.getLogger("quantum_engine.slurm.submit_walkers")


# ─────────────────────────────────────────────────────────────────────
# Config snapshot
# ─────────────────────────────────────────────────────────────────────

def _serialize_config(cfg: WalkerSweepConfig, snapshot_path: Path) -> Path:
    """Pickle the WalkerSweepConfig to a JSON file the workers can read.

    We write JSON not pickle so the file is human-debuggable (the user
    might want to know exactly what each walker thought it was doing).
    Path objects become strings; CV objects must be reconstructable from
    their dict — we delegate that to :func:`_dump_cv`.
    """
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(cfg)
    payload["structure"] = str(cfg.structure)
    payload["output_dir"] = str(cfg.output_dir)
    payload["cv"] = _dump_cv(cfg.cv)
    snapshot_path.write_text(json.dumps(payload, indent=2, default=str))
    return snapshot_path


def _dump_cv(cv: Any) -> dict:
    """Serialise a quantum_engine.mlff.plumed_runner CV instance to a dict."""
    cls = type(cv).__name__
    return {"__class__": cls, **{k: v for k, v in cv.__dict__.items()}}


def load_config_snapshot(snapshot_path: Path) -> WalkerSweepConfig:
    """Reverse of :func:`_serialize_config` — used by walker-side workers."""
    payload = json.loads(Path(snapshot_path).read_text())
    payload["cv"] = _load_cv(payload["cv"])
    payload["structure"] = Path(payload["structure"])
    payload["output_dir"] = Path(payload["output_dir"])
    return WalkerSweepConfig(**payload)


def _load_cv(d: dict) -> Any:
    """JSON → CV instance. Restores tuple fields (JSON roundtrips them as
    lists, which trips Python type hints downstream)."""
    from quantum_engine.mlff import plumed_runner as pr
    cls_name = d.pop("__class__")
    cls = getattr(pr, cls_name)
    fixed = {
        k: tuple(v) if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v)
        else v
        for k, v in d.items()
    }
    return cls(**fixed)


# ─────────────────────────────────────────────────────────────────────
# SLURM submit
# ─────────────────────────────────────────────────────────────────────

_SBATCH_TEMPLATE = """\
#!/usr/bin/env bash
#SBATCH -J {job_name}
#SBATCH -p {partition}
#SBATCH -c {cpus_per_walker}
#SBATCH --mem={mem_per_walker}
#SBATCH -t {walltime}
#SBATCH -a 0-{last_walker}
#SBATCH -o {logs_dir}/r{round_num:03d}_w%a.out
#SBATCH -e {logs_dir}/r{round_num:03d}_w%a.err
{gres_line}{deps_line}
set -euo pipefail
{python_bin} -m quantum_engine.slurm.submit_walkers run-walker \\
    --snapshot {snapshot} \\
    --walker-id "$SLURM_ARRAY_TASK_ID" \\
    --round {round_num}
"""


def _format_sbatch(
    cfg: WalkerSweepConfig,
    snapshot_path: Path,
    round_num: int,
    deps_line: str,
    logs_dir: Path,
    python_bin: str,
) -> str:
    return _SBATCH_TEMPLATE.format(
        job_name=f"{cfg.job_name_prefix}_r{round_num}",
        partition=cfg.partition,
        cpus_per_walker=cfg.cpus_per_walker,
        mem_per_walker=cfg.mem_per_walker,
        walltime=cfg.walltime_per_round,
        last_walker=cfg.n_walkers - 1,
        logs_dir=str(logs_dir),
        round_num=round_num,
        gres_line=f"#SBATCH --gres={cfg.gres}\n" if cfg.gres else "",
        deps_line=deps_line,
        python_bin=python_bin,
        snapshot=str(snapshot_path),
    )


def submit_walker_sweep(
    cfg: WalkerSweepConfig,
    *,
    python_bin: str | None = None,
    dry_run: bool = False,
) -> list[int]:
    """Submit ``cfg.n_rounds`` chained array jobs, one per round.

    Round 1 has no dependency; rounds 2..N each depend on the previous
    via ``--dependency=afterany`` so they only run *after* the prior
    round's array reaches a terminal state (COMPLETED, FAILED,
    CANCELLED, TIMEOUT). ``afterany`` is intentional: walltime-killed
    walkers and crashed walkers should still let the next round run —
    the surviving walkers' shared HILLS keep accumulating, and PLUMED
    reads whatever HILLS files exist. Side effect: if you ``scancel``
    round 1, rounds 2..N will still launch and pick up the partial
    HILLS — explicitly cancel the whole chain with ``scancel`` on every
    job id this function returns.

    A shared-filesystem precheck warns if ``cfg.output_dir`` looks like
    it lives on tmpfs / overlayfs (sub-second visibility for HILLS
    files is required for correct multi-walker bias collection).

    Args:
        cfg: the sweep config.
        python_bin: path to the Python interpreter to use inside the
            job. Defaults to the running interpreter.
        dry_run: if True, write the sbatch scripts but don't submit.

    Returns:
        SLURM job ids (one per round). ``-1`` for ``dry_run``.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _validate_shared_fs(cfg.output_dir)
    logs_dir = cfg.output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    snapshot_path = cfg.output_dir / "sweep_config.json"
    _serialize_config(cfg, snapshot_path)
    python_bin = python_bin or _detect_python()

    job_ids: list[int] = []
    prev_id: int | None = None
    for round_num in range(1, cfg.n_rounds + 1):
        deps_line = (
            f"#SBATCH --dependency=afterany:{prev_id}\n"
            if prev_id is not None and prev_id != -1
            else ""
        )
        script = _format_sbatch(
            cfg, snapshot_path, round_num, deps_line, logs_dir, python_bin
        )
        script_path = logs_dir / f"sbatch_round_{round_num:03d}.sh"
        script_path.write_text(script)
        script_path.chmod(0o755)

        if dry_run:
            log.info(f"[dry-run] would sbatch {script_path}")
            job_ids.append(-1)
            prev_id = -1
            continue

        submit = subprocess.run(
            ["sbatch", str(script_path)],
            check=True, capture_output=True, text=True,
        )
        # sbatch stdout: "Submitted batch job 12345\n"
        try:
            jid = int(submit.stdout.strip().split()[-1])
        except (IndexError, ValueError):
            raise RuntimeError(
                f"could not parse sbatch output: {submit.stdout!r}"
            )
        log.info(f"submitted round {round_num} as job {jid}")
        job_ids.append(jid)
        prev_id = jid

    return job_ids


def run_local_sweep(cfg: WalkerSweepConfig) -> list[dict]:
    """Run the entire sweep on the current node — no SLURM.

    For development, single-machine smoke tests, or running on a
    well-provisioned login node with no scheduler.

    .. warning::

       Walkers run **serially in a single process**. This is *not*
       equivalent to a real multi-walker run: with parallel walkers,
       hills deposited at time t0 by walker A appear in walker B's
       bias at the same wall-clock t0, so the walkers interact through
       collective bias accumulation. Here walker A runs to completion
       before walker B starts — they share the deposited HILLS history
       but the *temporal* coupling is gone. Use this for plumbing
       smoke-tests, not for production-quality sampling.
       (Production work: use :func:`submit_walker_sweep` on SLURM, or
        plug in :class:`multiprocessing.Pool` here yourself.)

    Returns a list of walker summaries (one per (walker, round) pair).
    """
    from quantum_engine.ops.mtd_walkers import run_walker
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _validate_shared_fs(cfg.output_dir)
    snapshot_path = cfg.output_dir / "sweep_config.json"
    _serialize_config(cfg, snapshot_path)

    summaries: list[dict] = []
    for round_num in range(1, cfg.n_rounds + 1):
        for walker_id in range(cfg.n_walkers):
            summaries.append(
                run_walker(
                    cfg, walker_id=walker_id, round_num=round_num,
                    equilibrate=cfg.equilibrate_first_round,
                )
            )
    return summaries


def _validate_shared_fs(path: Path) -> None:
    """Warn (don't fail) if ``path`` looks like a non-shared filesystem.

    Multi-walker PLUMED needs sub-second cross-process visibility on the
    HILLS files. /tmp / overlayfs / tmpfs typically don't share at all
    across nodes; NFS and Lustre do. We can't tell with certainty, so
    this is best-effort.
    """
    p = str(path.resolve())
    bad_prefixes = ("/tmp/", "/var/tmp/", "/dev/shm/")
    if any(p.startswith(pre) for pre in bad_prefixes):
        log.warning(
            f"output_dir {p!r} looks node-local (tmpfs/overlay). "
            "Multi-walker PLUMED needs a shared filesystem (NFS/Lustre) "
            "for HILLS visibility — sweep results may be wrong."
        )


def _detect_python() -> str:
    """Best-effort: where is the Python interpreter that should run on
    the cluster node? Defaults to the running interpreter."""
    import sys
    return sys.executable


# ─────────────────────────────────────────────────────────────────────
# Worker-side entry point (called from the sbatch script)
# ─────────────────────────────────────────────────────────────────────

def _walker_main(argv: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="quantum_engine.slurm.submit_walkers")
    sub = p.add_subparsers(dest="cmd", required=True)
    rw = sub.add_parser("run-walker", help="Run a single walker for one round")
    rw.add_argument("--snapshot", required=True, type=Path)
    rw.add_argument("--walker-id", required=True, type=int)
    rw.add_argument("--round", required=True, type=int)
    args = p.parse_args(argv)

    if args.cmd == "run-walker":
        from quantum_engine.ops.mtd_walkers import run_walker
        cfg = load_config_snapshot(args.snapshot)
        summary = run_walker(
            cfg, walker_id=args.walker_id, round_num=args.round,
            equilibrate=cfg.equilibrate_first_round and args.round == 1,
        )
        log.info(f"walker summary: {summary}")
        return 0
    return 1


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    sys.exit(_walker_main(sys.argv[1:]))
