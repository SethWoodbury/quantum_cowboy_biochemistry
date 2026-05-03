"""molecularGSM adapter (ZimmermanGroup/molecularGSM, MIT).

The C++/MKL implementation of the same Growing String Method as pyGSM,
generally faster on big systems but driven via subprocess (no Python
bindings). Vendored at ``deps/molecularGSM``; built with
``bash deps/build_molecular_gsm.sh`` (CMake + MKL).

Use molecularGSM when:
  * The system is too big for pyGSM's pure-Python loop (>~100 atoms in
    the QM region).
  * You want a third independent implementation of GSM for cross-check.

The CLI takes plain text input (gsm.inpfileq) + xyz files; we wrap the
build/run/parse loop here.

Status: skeletal. Wraps the gsm CLI subprocess.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger("quantum_engine.qm.molecular_gsm")


def molecular_gsm_available() -> bool:
    """Is the gsm binary on disk (per site.py:MOLECULAR_GSM_BIN)?"""
    from quantum_engine.site import MOLECULAR_GSM_BIN
    return MOLECULAR_GSM_BIN is not None


def _require_bin() -> str:
    from quantum_engine.site import MOLECULAR_GSM_BIN
    if MOLECULAR_GSM_BIN is None:
        raise FileNotFoundError(
            "molecularGSM binary not found. Run "
            "`bash deps/build_molecular_gsm.sh` (CMake + MKL build)."
        )
    return MOLECULAR_GSM_BIN


def write_inpfileq(workdir: Path, *,
                   sm_type: str = "GSM",        # "GSM" | "SSM"
                   restart: int = 0,
                   max_opt_steps: int = 3,
                   nnodes: int = 11,
                   isomers_file: str = "isomers.txt",
                   ) -> Path:
    """Generate a minimal gsm.inpfileq under ``workdir``. STUB — full
    template comes once we test on M-CSA 159 vacuum TS."""
    raise NotImplementedError(
        "molecular_gsm.write_inpfileq: stub — see deps/molecularGSM/example/ "
        "for the upstream template format."
    )


def double_ended_gsm(
    reactant: Any,                   # ASE Atoms
    product: Any,                    # ASE Atoms
    *,
    n_nodes: int = 11,
    qm_method: str = "xtb",          # external QM driver script
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    """DE-GSM via molecularGSM. Subprocess-driven; parses gsm output
    files for the TS image. STUB."""
    bin_path = _require_bin()
    raise NotImplementedError(
        f"molecular_gsm.double_ended_gsm: stub — would run {bin_path} "
        "after writing initial000.xyz, gsm.inpfileq, and a QM-driver script."
    )


def single_ended_gsm(
    reactant: Any,
    *,
    driving_coords: list[tuple[str, list[int], float]],
    qm_method: str = "xtb",
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    """SE-GSM (sm_type=SSM) via molecularGSM CLI. STUB."""
    bin_path = _require_bin()
    raise NotImplementedError(
        f"molecular_gsm.single_ended_gsm: stub — see {bin_path} usage."
    )


def _run_gsm_cli(workdir: Path, bin_path: str) -> subprocess.CompletedProcess:
    """Invoke the gsm binary in ``workdir`` and return the completed
    process. Helper for the public wrappers above."""
    return subprocess.run(
        [bin_path], cwd=workdir, check=True, capture_output=True, text=True,
    )
