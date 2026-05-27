"""Energy minimization (geometry optimization).

Analogous to Gaussian `opt` keyword. Supports LBFGS (default, quasi-Newton),
BFGS, and FIRE (good for noisy ML potentials; Bitzek et al. PRL 2006,
doi:10.1103/PhysRevLett.97.170201).

Backends (selected via ``backend`` kwarg or legacy ``optimizer`` kwarg):

* ``"ase-lbfgs"`` (default — same as legacy ``optimizer="lbfgs"``)
* ``"ase-bfgs"`` / ``"ase-fire"``
* ``"torch-sim-fire"`` (GPU-batched FIRE via torch-sim 0.3.0; only
  works for MACE calcs and unconstrained relaxes)
* ``"torch-sim-lbfgs"`` (stub — needs Python 3.12 container rebuild)

The legacy ``optimizer="lbfgs"|"fire"|"bfgs"`` argument is preserved for
backwards compatibility — existing call sites do NOT need to change.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from quantum_engine.units import EV_TO_KCAL
from ase import Atoms
from ase.optimize import LBFGS, BFGS, FIRE

log = logging.getLogger("quantum_engine.ops.opt")


# Legacy direct map — still here so external code that reads this
# attribute keeps working. Internally we now route through the
# `quantum_engine.opt` factory.
OPTIMIZERS = {"lbfgs": LBFGS, "bfgs": BFGS, "fire": FIRE}


# Map legacy optimizer names → modular backend identifiers.
_LEGACY_TO_BACKEND = {
    "lbfgs": "ase-lbfgs",
    "fire":  "ase-fire",
    "bfgs":  "ase-bfgs",
}


def _resolve_backend(*, backend: str | None, optimizer: str | None) -> str:
    """Reconcile the new ``backend`` kwarg with the legacy ``optimizer`` kwarg.

    Precedence: explicit ``backend`` wins. Otherwise translate
    ``optimizer`` (lbfgs/fire/bfgs) → ase-lbfgs/ase-fire/ase-bfgs.
    Default is ``"ase-lbfgs"``, matching the historical default.
    """
    if backend is not None:
        return backend
    if optimizer is not None:
        key = optimizer.lower()
        if key in _LEGACY_TO_BACKEND:
            return _LEGACY_TO_BACKEND[key]
        # Allow callers to pass full backend strings via ``optimizer=``
        # (we already validate downstream).
        return key
    return "ase-lbfgs"


def run(
    atoms: Atoms,
    calculator=None,
    outdir: str | Path = ".",
    constraint=None,
    optimizer: str = "lbfgs",
    backend: str | None = None,
    fmax: float = 0.05,
    max_steps: int = 500,
    **kwargs,
) -> dict:
    """Geometry optimization.

    Args:
        atoms: ASE Atoms
        calculator: ASE calculator
        outdir: output directory
        constraint: ASE constraint, list of constraints, or None
        optimizer: legacy alias — "lbfgs" (default), "bfgs", or "fire".
                   Translated internally to the matching ``ase-*`` backend.
        backend: NEW — full backend string (e.g. "torch-sim-fire").
                 Wins over ``optimizer`` if both given. ``None`` → keep
                 legacy ``optimizer`` semantics.
        fmax: convergence threshold on max atomic force (eV/Å)
        max_steps: safety cap on iterations
        **kwargs: forwarded to the optimizer constructor (e.g. ``device``,
                  ``dtype`` for torch-sim backends).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if atoms.calc is None:
        if calculator is None:
            raise ValueError("Need a calculator")
        atoms.calc = calculator

    if constraint is not None:
        if not isinstance(constraint, list):
            constraint = [constraint]
        atoms.set_constraint(constraint)

    backend_str = _resolve_backend(backend=backend, optimizer=optimizer)

    # Lazy import (keeps `quantum_engine.ops.opt` cheap if torch_sim isn't
    # available — matters for legacy callers that never touch torch-sim).
    from quantum_engine.opt import make_optimizer
    opt_obj = make_optimizer(
        backend_str,
        fmax=fmax,
        max_steps=max_steps,
        outdir=outdir,
        **kwargs,
    )
    opt_result = opt_obj.run(atoms)

    # Repackage as the legacy dict shape so callers don't have to change.
    result = {
        "status": opt_result.status,
        "converged": opt_result.converged,
        "atoms": opt_result.atoms,
        "energy_eV": float(opt_result.energy_eV),
        "energy_kcal_mol": float(opt_result.energy_kcal_mol),
        "fmax_final": float(opt_result.fmax_final),
        "n_steps": int(opt_result.n_steps),
        "backend": opt_result.backend,
        "wall_time_s": float(opt_result.wall_time_s),
        "outputs": dict(opt_result.outputs),
    }

    (outdir / "opt-summary.json").write_text(json.dumps({
        k: v for k, v in result.items() if k not in ("atoms", "outputs")
    }, indent=2))
    result["outputs"]["summary"] = str(outdir / "opt-summary.json")

    return result
