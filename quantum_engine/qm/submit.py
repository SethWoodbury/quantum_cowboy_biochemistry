"""
SLURM job submission helpers.

Generic ``sbatch`` script generation, plus convenience wrappers for
Gaussian 16 and ORCA that set up the correct environment variables.

Cluster conventions follow those in:
  - Indrek's gauss: /home/ikalvet/bin/gauss
  - Indrek's orcasub: /home/ikalvet/bin/orcasub
  - Declan's g16_setup.py: /home/dme5188/scripts/g16/g16_setup.py
"""

from __future__ import annotations

import getpass
import os
import subprocess
from pathlib import Path

from quantum_engine.config import GAUSSIAN_ROOT, SCRATCH_DIR

# ── Defaults ─────────────────────────────────────────────────────────
_USERNAME = getpass.getuser()
_EMAIL = f"{_USERNAME}@uw.edu"
_GAUSS_SCRDIR = f"/net/scratch/{_USERNAME}/temp"
_ORCA_EXEC = "/software/orca/latest/orca"
_MPI_PATH = "/home/ikalvet/software/openmpi-3.1.3"


# ── Generic SLURM script generation ─────────────────────────────────


def write_slurm_script(
    commands: list[str],
    job_name: str,
    output_path: str | Path,
    partition: str = "cpu",
    time: str = "12:00:00",
    mem: str = "8g",
    cpus: int = 8,
    gpu: str | None = None,
    array: str | bool = False,
) -> Path:
    """Write a SLURM submission script.

    Parameters
    ----------
    commands : list[str]
        Shell commands to execute (one per line in the script body).
    job_name : str
        SLURM job name (``--job-name``).
    output_path : str or Path
        File path for the generated ``.sh`` script.
    partition : str
        SLURM partition (default ``"cpu"``).
    time : str
        Wall-clock time limit in ``HH:MM:SS`` (default ``"12:00:00"``).
    mem : str
        Total memory request, e.g. ``"8g"`` (default ``"8g"``).
    cpus : int
        CPUs per task (default 8).
    gpu : str or None
        GPU resource string for ``--gres`` (e.g. ``"gpu:1"``).  If ``None``,
        no GPU is requested.
    array : str or bool
        If a string like ``"1-10"``, sets ``--array``.  If ``True`` but not
        a string, the caller must set it themselves.  ``False`` disables
        (default).

    Returns
    -------
    Path
        Absolute path to the written script.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["#!/bin/bash"]
    lines.append(f"#SBATCH --job-name={job_name}")
    lines.append(f"#SBATCH --partition={partition}")
    lines.append(f"#SBATCH --time={time}")
    lines.append(f"#SBATCH --cpus-per-task={cpus}")
    lines.append(f"#SBATCH --mem={mem}")
    lines.append(f"#SBATCH --mail-type=END,FAIL")
    lines.append(f"#SBATCH --mail-user={_EMAIL}")
    lines.append(f"#SBATCH -o {job_name}_%j.out")
    lines.append(f"#SBATCH -e {job_name}_%j.err")

    if gpu is not None:
        lines.append(f"#SBATCH --gres={gpu}")

    if isinstance(array, str):
        lines.append(f"#SBATCH --array={array}")

    lines.append("")
    lines.append('echo "Job $SLURM_JOB_ID started on: $(hostname -s)"')
    lines.append('echo "Job $SLURM_JOB_ID started on: $(date)"')
    lines.append('echo ""')
    lines.append("")

    for cmd in commands:
        lines.append(cmd)

    lines.append("")
    lines.append('echo ""')
    lines.append('echo "Job $SLURM_JOB_ID finished on: $(date)"')
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n")
    output_path.chmod(0o755)
    return output_path.resolve()


# ── Submit ───────────────────────────────────────────────────────────


def submit_job(script_path: str | Path) -> str:
    """Submit a SLURM script via ``sbatch`` and return the job ID.

    Parameters
    ----------
    script_path : str or Path
        Path to the ``.sh`` submission script.

    Returns
    -------
    str
        The SLURM job ID (e.g. ``"1234567"``).

    Raises
    ------
    RuntimeError
        If ``sbatch`` exits with a non-zero return code.
    """
    script_path = Path(script_path)
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch failed (exit code {result.returncode}):\n{result.stderr}"
        )

    # sbatch output is typically: "Submitted batch job 1234567"
    for word in result.stdout.strip().split():
        if word.isdigit():
            return word

    # Fallback: return the full stdout if we cannot parse it
    return result.stdout.strip()


# ── Gaussian convenience wrapper ─────────────────────────────────────


def write_gaussian_slurm(
    com_files: list[str | Path],
    job_name: str,
    output_dir: str | Path,
    nproc: int = 8,
    mem: str = "16gb",
    time: str = "24:00:00",
    partition: str = "cpu",
) -> Path:
    """Generate a SLURM script that runs one or more Gaussian 16 jobs sequentially.

    Sets ``GAUSS_EXEDIR`` and ``GAUSS_SCRDIR`` environment variables following
    the conventions in Indrek's ``gauss`` and Declan's ``g16_setup.py``.

    Parameters
    ----------
    com_files : list of str or Path
        Gaussian .com input files to run.
    job_name : str
        SLURM job name.
    output_dir : str or Path
        Directory where the .sh script will be written.
    nproc : int
        Number of processors (default 8).
    mem : str
        Memory allocation matching the .com header (default ``"16gb"``).
    time : str
        Wall time (default ``"24:00:00"``).
    partition : str
        SLURM partition (default ``"cpu"``).

    Returns
    -------
    Path
        Absolute path to the generated SLURM script.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / f"{job_name}.sh"

    # Parse memory for SLURM (convert e.g. "16gb" -> "16g")
    mem_lower = mem.lower().replace("gb", "g").replace("mb", "m")
    # Request ~10% more memory from SLURM than Gaussian uses for overhead
    mem_val = int("".join(c for c in mem_lower if c.isdigit()))
    mem_unit = "".join(c for c in mem_lower if c.isalpha())
    slurm_mem = f"{int(mem_val * 1.1)}{mem_unit}"

    commands: list[str] = [
        f"export GAUSS_EXEDIR='{GAUSSIAN_ROOT}/'",
        f"export GAUSS_SCRDIR='{_GAUSS_SCRDIR}/'",
        f"export OMP_NUM_THREADS={nproc}",
        f"export MKL_NUM_THREADS={nproc}",
        f"mkdir -p $GAUSS_SCRDIR",
        "",
        'echo "GAUSS_EXEDIR=$GAUSS_EXEDIR"',
        'echo "GAUSS_SCRDIR=$GAUSS_SCRDIR"',
        'echo ""',
        "",
    ]

    for com in com_files:
        com = Path(com)
        log_name = com.stem + ".log"
        commands.append(f'echo "Running {com.name} ..."')
        commands.append(
            f"/usr/bin/time -v {GAUSSIAN_ROOT}/g16 < {com} > {com.parent / log_name}"
        )
        commands.append("")

    return write_slurm_script(
        commands=commands,
        job_name=job_name,
        output_path=script_path,
        partition=partition,
        time=time,
        mem=slurm_mem,
        cpus=nproc,
    )


# ── ORCA convenience wrapper ────────────────────────────────────────


def write_orca_slurm(
    inp_files: list[str | Path],
    job_name: str,
    output_dir: str | Path,
    nproc: int = 8,
    maxcore: int = 4000,
    time: str = "24:00:00",
    partition: str = "cpu",
) -> Path:
    """Generate a SLURM script that runs one or more ORCA jobs sequentially.

    Sets up the OpenMPI paths following the pattern in Indrek's ``orcasub``.

    Parameters
    ----------
    inp_files : list of str or Path
        ORCA .inp input files to run.
    job_name : str
        SLURM job name.
    output_dir : str or Path
        Directory where the .sh script will be written.
    nproc : int
        Number of parallel processes (default 8).
    maxcore : int
        Memory per core in MB (default 4000).
    time : str
        Wall time (default ``"24:00:00"``).
    partition : str
        SLURM partition (default ``"cpu"``).

    Returns
    -------
    Path
        Absolute path to the generated SLURM script.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / f"{job_name}.sh"

    # Total memory: maxcore * nproc, plus ~10% overhead, converted to GB
    total_mem_mb = maxcore * nproc
    slurm_mem_gb = int(total_mem_mb * 1.1 / 1000) + 1
    slurm_mem = f"{slurm_mem_gb}g"

    commands: list[str] = [
        f"export PATH={_MPI_PATH}/bin:$PATH",
        f"export LD_LIBRARY_PATH={_MPI_PATH}/lib:$LD_LIBRARY_PATH",
        "",
    ]

    for inp in inp_files:
        inp = Path(inp)
        out_name = inp.stem + ".out"
        commands.append(f'echo "Running {inp.name} ..."')
        commands.append(
            f"{_ORCA_EXEC} {inp} > {inp.parent / out_name}"
        )
        commands.append("")

    return write_slurm_script(
        commands=commands,
        job_name=job_name,
        output_path=script_path,
        partition=partition,
        time=time,
        mem=slurm_mem,
        cpus=nproc,
    )
