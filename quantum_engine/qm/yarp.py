"""YARP adapter — cold-storage stub.

YARP (zhaoqy1996/YARP) was the bond-electron-matrix product enumeration
+ GSM TS-localization workflow. Upstream is dead — last commit
2022-04, **Python 2.7 only**, no PyPI/conda artefact, no maintained
fork. We chose not to vendor it; SCINE Chemoton + ReaDuct cover the
same ground with active maintenance.

If a future user really wants YARP, they can:
  1. ``git submodule add https://github.com/zhaoqy1996/YARP.git deps/yarp``
  2. Build a separate ``qcb-yarp-py27`` micromamba env (do NOT
     contaminate qcb-xtb).
  3. Reference: paper at https://doi.org/10.1038/s43588-022-00256-7

This module exists so callers can target ``quantum_engine.qm.yarp.*``
in pipeline configs without an import error; everything raises with
the install pointer.
"""
from __future__ import annotations

from typing import Any

_INSTALL_HINT = (
    "YARP is upstream-dead (2022-04, Py2.7-only). Use SCINE Chemoton + "
    "ReaDuct instead — same product enumeration + GSM TS workflow with "
    "active maintenance: see quantum_engine.qm.scine. If you really need "
    "YARP, vendor it manually under a separate qcb-yarp-py27 env."
)


def yarp_available() -> bool:
    return False


def enumerate_products(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_INSTALL_HINT)


def localise_ts(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_INSTALL_HINT)
