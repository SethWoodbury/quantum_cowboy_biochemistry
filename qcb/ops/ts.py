"""High-level TS pipeline: composes endpoint generation + NEB + saddle + IRC.

For now this delegates to the proven scripts/run_neb_ts.py code path via
subprocess, keeping backward compatibility. A future refactor will make this
a pure-Python composition of ops/{saddle, irc, neb, mtd, ...}.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("qcb.ops.ts")


def run(
    pdb_file: str | Path,
    outdir: str | Path,
    strategy: str = "legacy",
    model: str = "mace-omol",
    charge: int | None = None,
    extra_args: list[str] | None = None,
    **kwargs,
) -> dict:
    """High-level TS search pipeline.

    This delegates to scripts/run_neb_ts.py which implements the full pipeline
    including endpoint generation, NEB, Sella TS refinement, and frequency
    validation.

    Args:
        pdb_file: input PDB (may be a TS guess or reactant, depending on strategy)
        outdir: output directory
        strategy: "legacy", "irc", "cv-spring", or "mtd"
        model: MACE model alias
        charge: system net charge (if None, inferred from PDB REMARK or filename)
        extra_args: additional CLI args to pass through

    Returns: dict with barrier_fwd_kcal, barrier_rev_kcal, ts_pdb path, etc.
    """
    pdb_file = Path(pdb_file).resolve()
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve().parent.parent.parent / "scripts" / "run_neb_ts.py"

    cmd = [
        sys.executable, str(script),
        str(pdb_file),
        "--outdir", str(outdir),
        "--model", model,
        "--strategy", strategy,
    ]
    if charge is not None:
        cmd.extend(["--charge", str(charge)])
    if extra_args:
        cmd.extend(extra_args)

    log.info(f"ts pipeline: {' '.join(cmd)}")
    # Stream output to user's stdout/stderr so they see progress live
    # (these pipelines can run for hours; silent subprocess is confusing)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        log.error(f"ts pipeline failed with returncode {proc.returncode}")

    return {
        "status": "completed" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "outdir": str(outdir),
        "outputs": {},
    }
