"""ChemShell QM/MM wrapper — opt-in, manual-install backend.

ChemShell (CCFE / STFC) is the most TS-native open QM/MM framework but
its distribution model is friction-heavy (registration at
chemshell.org, no PyPI / conda-forge / public git mirror). We don't
vendor it as a submodule for that reason. If you want it, install it
manually following the steps in :file:`deps/README.md` ("ChemShell"
section) and point qcb at the binary via:

  * ``$QCB_CHEMSHELL`` env var (highest priority), or
  * ``deps/chemshell/install/bin/chemsh`` (manual extraction here), or
  * ``/net/software/lab/chemshell/bin/chemsh`` (lab-wide install).

If none of those exist, the wrapper raises a :class:`FileNotFoundError`
with the install instructions. For most users, the
:func:`quantum_engine.qm.openmm_xtb` glue (OpenMM + xtb in-process via
ASE) is a more accessible alternative for QM/MM-style work.

Status: SCAFFOLD ONLY. The high-level workflow API is sketched; the
input-file generator and output parser need to be filled in once we
have a binary on disk to test against.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from quantum_engine.site import CHEMSHELL_BIN

log = logging.getLogger("quantum_engine.qm.chemshell")


_INSTALL_HELP = """\
ChemShell is not installed. To install:

  1. Register at https://www.chemshell.org/ (LGPL but requires the form).
  2. Download the latest tarball.
  3. Extract into deps/chemshell/install/  OR  /net/software/lab/chemshell/
     (so the binary lives at .../bin/chemsh).
  4. Re-import quantum_engine; CHEMSHELL_BIN will resolve.

Override the resolution with $QCB_CHEMSHELL=/path/to/chemsh.

Open-source alternative: see :mod:`quantum_engine.qm.openmm_xtb` for
OpenMM-based QM/MM (no registration required).
"""


def chemshell_available() -> bool:
    """Is the ChemShell binary on disk?"""
    return CHEMSHELL_BIN is not None and os.path.isfile(CHEMSHELL_BIN)


def _ensure_available() -> str:
    if not chemshell_available():
        raise FileNotFoundError(_INSTALL_HELP)
    return CHEMSHELL_BIN  # type: ignore[return-value]


def run_qmmm_optimisation(
    atoms: Any,
    *,
    qm_indices: list[int],
    qm_method: str = "xtb",
    mm_forcefield: str = "amber",
    workdir: str | Path | None = None,
    timeout_s: int = 7200,
) -> dict:
    """Run a QM/MM optimisation in ChemShell.

    Args:
        atoms: ASE Atoms with the full QM+MM system.
        qm_indices: 0-based indices of atoms in the QM region.
        qm_method: ``"xtb"`` (default), ``"orca"``, or any
            other QM driver ChemShell knows.
        mm_forcefield: MM partition force field (default Amber via
            DL_POLY).

    Returns: ``{"atoms_optimised", "energy_eV", "qm_energy_eV",
        "mm_energy_eV", "outputs": {...}}``.
    """
    binary = _ensure_available()
    raise NotImplementedError(
        "ChemShell QM/MM wiring is not implemented yet. The binary is "
        f"at {binary!r}; the next step is generating a chemsh.py "
        "input file (qm_partition / mm_partition / runner) and parsing "
        "the .out energies. Sketch in :mod:`quantum_engine.pipelines.steps` "
        "as a future Step. Until that lands, fall back to "
        ":mod:`quantum_engine.qm.openmm_xtb` for QM/MM."
    )


def run_qmmm_ts(
    atoms: Any,
    *,
    qm_indices: list[int],
    qm_method: str = "xtb",
    saddle_method: str = "dimer",
    **kw,
) -> dict:
    """Run a QM/MM TS search via ChemShell's HDLCopt + dimer.

    Same NotImplementedError pattern; left as a stub that points at
    a clear hand-off when the wiring is ready.
    """
    _ensure_available()
    raise NotImplementedError(
        "QM/MM TS via ChemShell — not implemented. See "
        "run_qmmm_optimisation for the input-file pattern to follow."
    )
