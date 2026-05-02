"""CREST conformer search — thin wrapper around the vendored binary.

CREST (Conformer-Rotamer Ensemble Sampling Tool) from Grimme group:
https://github.com/crest-lab/crest. We vendor the source as a git
submodule at ``deps/crest`` and ship a build script at
``deps/build_crest.sh`` that installs the conda-forge binary into the
``qcb-xtb`` env (or builds from source if ``BUILD_FROM_SOURCE=1``).

This wrapper expects an ASE :class:`Atoms` object in and returns an
ensemble of conformer Atoms. CREST is a long-running tool (minutes to
hours per system); we run it as a subprocess and parse its output.

API
---
:func:`run_conformer_search` — drop your TS guess in, get out a list of
geometric conformers. The defaults run "fast" (`gfn0` + `--quick`) so
you get something back in a few minutes; the full GFN2-xTB
metadynamics-based search is enabled with ``method="gfn2"``.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from quantum_engine.site import CREST_BIN

log = logging.getLogger("quantum_engine.qm.crest")


def crest_available() -> bool:
    """Is the CREST binary on disk?"""
    return CREST_BIN is not None and os.path.isfile(CREST_BIN)


def _ensure_available() -> str:
    if not crest_available():
        raise FileNotFoundError(
            "CREST binary not found. Run `bash deps/build_crest.sh` to "
            "install (default: conda-forge into the qcb-xtb env). "
            "Override with $QCB_CREST_BIN if you have it elsewhere."
        )
    return CREST_BIN  # type: ignore[return-value]


def run_conformer_search(
    atoms: Any,
    *,
    charge: int = 0,
    uhf: int = 0,
    method: str = "gfn2",
    solvent: str | None = None,
    n_threads: int = 4,
    extra_args: list[str] | None = None,
    workdir: str | Path | None = None,
    timeout_s: int = 7200,
    keep_workdir: bool = False,
) -> dict:
    """Run CREST on ``atoms``; return the parsed conformer ensemble.

    Args:
        atoms: ASE :class:`~ase.Atoms`.
        charge: net charge.
        uhf: number of unpaired electrons (default 0 = closed shell).
        method: ``"gfn2"`` (default — slowest, most accurate),
            ``"gfn1"``, ``"gfnff"``, or ``"gfn0"`` (fastest screen).
        solvent: ALPB solvent name for implicit solvation (e.g.
            ``"water"``); ``None`` = gas phase.
        n_threads: OMP threads. CREST scales OK to ~8.
        extra_args: pass-through CLI flags (e.g. ``["--quick"]`` for a
            fast screen, ``["--noreftopo"]`` to skip reference topology).
        workdir: where CREST runs. None → tempdir wiped on exit
            (set ``keep_workdir=True`` to preserve).
        timeout_s: overall wall limit.

    Returns:
        ``{"conformers": list[ase.Atoms], "energies_Eh": list[float],
           "n_unique": int, "best": ase.Atoms,
           "workdir": Path}`` (workdir present iff ``keep_workdir=True``).
    """
    binary = _ensure_available()
    from ase.io import read as ase_read, write as ase_write

    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="qcb_crest_"))
    else:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

    try:
        in_xyz = workdir / "input.xyz"
        ase_write(str(in_xyz), atoms)

        cmd = [binary, str(in_xyz),
               f"-{method}",
               "-chrg", str(charge), "-uhf", str(uhf),
               "-T", str(n_threads)]
        if solvent:
            cmd += ["--alpb", solvent]
        if extra_args:
            cmd += list(extra_args)

        log.info(f"  CREST: {' '.join(cmd)}")
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", str(n_threads))
        proc = subprocess.run(cmd, cwd=str(workdir), env=env,
                              capture_output=True, text=True,
                              timeout=timeout_s)
        if proc.returncode != 0:
            raise RuntimeError(
                f"CREST exited {proc.returncode}.\n"
                f"  stderr tail: {proc.stderr[-500:]}"
            )

        # CREST writes the conformer ensemble to crest_conformers.xyz
        # and the unique rotamer set to crest_rotamers.xyz. We prefer
        # the deduplicated conformers.
        ensemble_path = workdir / "crest_conformers.xyz"
        if not ensemble_path.is_file():
            raise FileNotFoundError(
                f"crest_conformers.xyz not produced; tail:\n"
                f"{proc.stdout[-500:]}"
            )
        conformers = ase_read(str(ensemble_path), ":")
        if not isinstance(conformers, list):
            conformers = [conformers]
        # Per-conformer energies live in line-2 of each XYZ frame —
        # ASE parses these as info["comment"] but the value is at info["energy"]
        energies_Eh = [c.info.get("energy", float("nan")) for c in conformers]

        return {
            "conformers": conformers,
            "energies_Eh": energies_Eh,
            "n_unique": len(conformers),
            "best": conformers[0],
            "workdir": workdir if keep_workdir else None,
        }
    finally:
        if not keep_workdir:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)
