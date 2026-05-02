"""
xTB (extended tight-binding) workflow helpers.

Generates xTB command lines and runs geometry optimisations via the xTB
binary at ``/home/dme5188/bin/xtb/xtb-6.6.1/bin/xtb``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from quantum_engine.site import XTB_BIN

# ── Supported GFN methods ───────────────────────────────────────────
_GFN_METHODS: dict[str, str] = {
    "gfn0": "--gfn 0",
    "gfn1": "--gfn 1",
    "gfn2": "--gfn 2",
    "gfnff": "--gfnff",
}


def write_xtb_input(
    xyz_path: str | Path,
    charge: int = 0,
    uhf: int = 0,
    method: str = "gfn2",
    solvent: str | None = None,
) -> list[str]:
    """Build the xTB command-line argument list for a given calculation.

    This does **not** run xTB; it returns the argument list so the caller
    can inspect, log, or pass it to ``subprocess.run``.

    Parameters
    ----------
    xyz_path : str or Path
        Path to the input .xyz geometry file.
    charge : int
        Net molecular charge (default 0).
    uhf : int
        Number of unpaired electrons (default 0).
    method : str
        GFN method level. One of ``"gfn0"``, ``"gfn1"``, ``"gfn2"``,
        ``"gfnff"`` (default ``"gfn2"``).
    solvent : str or None
        If given, enables ALPB implicit solvation with the named solvent
        (e.g. ``"water"``, ``"chcl3"``).

    Returns
    -------
    list[str]
        Complete command-line arguments (including the xTB binary path).

    Raises
    ------
    ValueError
        If *method* is not recognised.
    FileNotFoundError
        If the xTB binary or *xyz_path* does not exist.
    """
    xyz_path = Path(xyz_path)
    if not xyz_path.exists():
        raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

    m = method.lower()
    if m not in _GFN_METHODS:
        raise ValueError(
            f"Unknown xTB method '{method}'. "
            f"Supported: {', '.join(_GFN_METHODS)}"
        )

    cmd = [XTB_BIN, str(xyz_path)]
    cmd.extend(_GFN_METHODS[m].split())
    cmd.extend(["--chrg", str(charge)])
    cmd.extend(["--uhf", str(uhf)])

    if solvent is not None:
        cmd.extend(["--alpb", solvent])

    return cmd


def run_xtb_opt(
    xyz_path: str | Path,
    charge: int = 0,
    uhf: int = 0,
    method: str = "gfn2",
    solvent: str | None = None,
    output_dir: str | Path = ".",
    constraints: str | Path | None = None,
) -> Path:
    """Run an xTB geometry optimisation and return the optimised structure.

    The optimisation is executed in *output_dir*.  The input .xyz file is
    copied there first so that xTB output files land in the right place.

    Parameters
    ----------
    xyz_path : str or Path
        Path to the input .xyz file.
    charge : int
        Net molecular charge (default 0).
    uhf : int
        Number of unpaired electrons (default 0).
    method : str
        GFN method level (default ``"gfn2"``).
    solvent : str or None
        Implicit solvent name for ALPB (default ``None``).
    output_dir : str or Path
        Directory where xTB will be executed and output stored (default ``"."``).
    constraints : str, Path, or None
        Path to an xTB constraint file (``--input <file>``).  If ``None``,
        no constraints are applied.

    Returns
    -------
    Path
        Absolute path to the optimised geometry (``xtbopt.xyz``).

    Raises
    ------
    RuntimeError
        If xTB exits with a non-zero return code.
    """
    xyz_path = Path(xyz_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy input into the working directory so outputs land there
    local_xyz = output_dir / xyz_path.name
    if local_xyz.resolve() != xyz_path.resolve():
        shutil.copy2(xyz_path, local_xyz)

    cmd = write_xtb_input(
        local_xyz,
        charge=charge,
        uhf=uhf,
        method=method,
        solvent=solvent,
    )
    cmd.append("--opt")

    if constraints is not None:
        constraints = Path(constraints).resolve()
        if not constraints.exists():
            raise FileNotFoundError(f"Constraint file not found: {constraints}")
        cmd.extend(["--input", str(constraints)])

    result = subprocess.run(
        cmd,
        cwd=output_dir,
        capture_output=True,
        text=True,
    )

    # Write stdout/stderr logs regardless of success
    (output_dir / "xtb_opt.out").write_text(result.stdout)
    if result.stderr:
        (output_dir / "xtb_opt.err").write_text(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"xTB optimisation failed (exit code {result.returncode}).\n"
            f"Working directory: {output_dir}\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr (last 500 chars): {result.stderr[-500:]}"
        )

    opt_xyz = output_dir / "xtbopt.xyz"
    if not opt_xyz.exists():
        raise RuntimeError(
            f"xTB completed but xtbopt.xyz not found in {output_dir}. "
            "Check xtb_opt.out for details."
        )

    return opt_xyz.resolve()
