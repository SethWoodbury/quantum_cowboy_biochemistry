"""pysisyphus adapter (eljost/pysisyphus, GPL-3).

pysisyphus is a Swiss-army knife for reaction-path methods — NEB, GSM,
dimer, IRC, and a battery of TS optimisers including RS-I-RFO, EF, and
Sella-equivalent partial-Hessian methods. Already vendored at
``deps/pysisyphus``.

Where it fits in the QCB pipelines:

  * **NEB / CI-NEB** as alternative to ASE's NEB (pysisyphus tends to
    converge faster on big systems via its trust-radius algorithms).
  * **Dimer** for TS optimisation when a single reactant + initial
    direction guess is all we have (no product, no driving coords).
  * **Restraints** — its SpringConstraint is the cleanest way to tether
    backbone C/N atoms to initial positions in chopped-cluster mode
    (the "ca-restrained" mode from run_neb_ts.py's plan).

Install: pyproject's ``[project.optional-dependencies].opt`` group, or
``pip install -e deps/pysisyphus``.

Status: skeletal. Wraps the upstream Python API; impls land once we
have the M-CSA 159 dry run going.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("quantum_engine.qm.pysisyphus")


def pysisyphus_available() -> bool:
    try:
        import pysisyphus  # noqa: F401
        return True
    except ImportError:
        return False


def _require() -> None:
    if not pysisyphus_available():
        raise ImportError(
            "pysisyphus not installed. `pip install -e deps/pysisyphus` "
            "against qcb-xtb."
        )


def neb_path(
    reactant: Any,                   # ASE Atoms
    product: Any,
    *,
    n_images: int = 11,
    calc: Any | None = None,
    climbing: bool = True,
    fmax: float = 0.05,
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    """CI-NEB via pysisyphus. STUB.
    Returns ``{ts_atoms, ts_energy, images, energies}``."""
    _require()
    raise NotImplementedError(
        "pysisyphus.neb_path: wire to pysisyphus.cos.NEB and "
        "pysisyphus.optimizers.LBFGS. See deps/pysisyphus/examples/."
    )


def dimer_ts(
    atoms: Any,
    *,
    initial_direction: Any | None = None,   # 3N vector or None for random
    calc: Any | None = None,
    fmax: float = 0.01,
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    """Dimer TS optimisation from a single guess + direction. STUB."""
    _require()
    raise NotImplementedError(
        "pysisyphus.dimer_ts: wire to pysisyphus.tsoptimizers.dimer."
    )


def rsirfo_ts(
    atoms: Any,
    *,
    calc: Any | None = None,
    hessian: Any | None = None,
    fmax: float = 0.005,
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    """RS-I-RFO TS optimisation — needs a Hessian (full or partial). STUB."""
    _require()
    raise NotImplementedError(
        "pysisyphus.rsirfo_ts: wire to pysisyphus.tsoptimizers.RSIRFOptimizer."
    )


def irc(
    ts_atoms: Any,
    *,
    calc: Any | None = None,
    direction: str = "both",         # "forward" | "backward" | "both"
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    """IRC descent from a TS. STUB. Returns endpoints + paths in both
    directions when ``direction='both'``."""
    _require()
    raise NotImplementedError(
        "pysisyphus.irc: wire to pysisyphus.irc.EulerPC or DampedVelocityVerlet."
    )


def harmonic_restraints(
    atoms: Any,
    *,
    indices: list[int],
    k: float = 5.0,                  # eV/Å^2
) -> Any:
    """Build a SpringConstraint pinning ``indices`` to current positions.
    Used by chopped-cluster TS searches to prevent isolated residue
    backbones drifting without hard-fixing them. STUB."""
    _require()
    raise NotImplementedError(
        "pysisyphus.harmonic_restraints: wire to "
        "pysisyphus.constraints.HarmonicRestraint."
    )
