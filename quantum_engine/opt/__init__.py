"""Modular geometry-optimization backends.

A small factory + base class that lets QCB tools swap between ASE-native
LBFGS/FIRE and torch-sim-based GPU-batched optimizers without rewriting
the calling code.

Existing call sites (``quantum_engine.ops.opt.run``, plus a handful of
``ase.optimize.LBFGS`` direct callers in ``ops/ts.py``, ``ops/scan.py``,
``mlff/irc.py``, etc.) keep their behaviour because the default backend
is ``"ase-lbfgs"`` and the OptResult schema is a strict superset of the
old dict.

Public API
----------
``make_optimizer(backend: str, **cfg) -> Optimizer``
    Factory. Backend strings: ``"ase-lbfgs"`` (default), ``"ase-fire"``,
    ``"ase-bfgs"``, ``"torch-sim-fire"``, ``"torch-sim-lbfgs"``.

``Optimizer.run(atoms, calculator) -> OptResult``
    Run the relaxation. ``atoms`` may already have ``.calc`` set; if not,
    ``calculator`` must be passed.

``list_backends() -> list[str]``
    Discover what's available given the current Python install (e.g.
    ``"torch-sim-*"`` only listed if ``import torch_sim`` works).

Example
-------
>>> from quantum_engine.opt import make_optimizer
>>> opt = make_optimizer("ase-lbfgs", fmax=0.05, max_steps=500,
...                      logfile="opt.log", trajectory="opt.traj")
>>> result = opt.run(atoms, calculator=mace_calc)
>>> print(result.converged, result.fmax_final, result.energy_eV)
"""

from quantum_engine.opt.base import Optimizer, OptResult
from quantum_engine.opt.factory import make_optimizer, list_backends, BACKENDS

__all__ = [
    "Optimizer",
    "OptResult",
    "make_optimizer",
    "list_backends",
    "BACKENDS",
]
