"""Registry-backed factory for minimization ``Optimizer`` instances.

Single source of truth for optimizer-backend strings (``--optimizer``). Backends
live in a :class:`~quantum_engine.registry.Registry`, so adding a NEW minimizer
later is one ``register_optimizer(...)`` call — no edits here:

    from quantum_engine.opt.factory import register_optimizer
    register_optimizer("my-min", lambda: MyOptimizerClass, aliases=("mine",))

Energy-function-agnostic: every backend consumes any ASE calculator EXCEPT the
``torch-sim-*`` ones, which need a torch/MACE model (GPU fast path) — flagged via
``requires_torch_sim`` and filtered by ``list_backends(available_only=True)``.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from quantum_engine.opt.base import Optimizer
from quantum_engine.registry import Registry

log = logging.getLogger("quantum_engine.opt.factory")

# Registry of name -> (factory_fn, requires_torch_sim). factory_fn returns the
# Optimizer *class* lazily (so a missing dep doesn't break `import`).
OPTIMIZERS: Registry = Registry("optimizer")


def register_optimizer(name: str, factory_fn: Callable[[], type], *,
                       requires_torch_sim: bool = False,
                       aliases: tuple[str, ...] = (), overwrite: bool = False):
    """Register a minimizer backend. ``factory_fn`` returns the Optimizer class."""
    OPTIMIZERS.register(name, (factory_fn, requires_torch_sim),
                        aliases=aliases, overwrite=overwrite)
    return factory_fn


# ---- lazy per-backend imports (a missing dep doesn't break `import`) ----
def _ase_lbfgs():
    from quantum_engine.opt.ase_optimizers import AseLbfgsOptimizer  # noqa: PLC0415
    return AseLbfgsOptimizer

def _ase_fire():
    from quantum_engine.opt.ase_optimizers import AseFireOptimizer  # noqa: PLC0415
    return AseFireOptimizer

def _ase_bfgs():
    from quantum_engine.opt.ase_optimizers import AseBfgsOptimizer  # noqa: PLC0415
    return AseBfgsOptimizer

def _ase_precon_lbfgs():
    from ase.optimize.precon import PreconLBFGS  # noqa: F401, PLC0415 — availability gate
    from quantum_engine.opt.ase_optimizers import AsePreconLbfgsOptimizer  # noqa: PLC0415
    return AsePreconLbfgsOptimizer

def _ase_fire2():
    from ase.optimize import FIRE2  # noqa: F401, PLC0415 — availability gate (needs recent ASE)
    from quantum_engine.opt.ase_optimizers import AseFire2Optimizer  # noqa: PLC0415
    return AseFire2Optimizer

def _ts_fire():
    from quantum_engine.opt.torchsim_optimizers import TorchSimFireOptimizer  # noqa: PLC0415
    return TorchSimFireOptimizer

def _ts_lbfgs():
    from quantum_engine.opt.torchsim_optimizers import TorchSimLbfgsOptimizer  # noqa: PLC0415
    return TorchSimLbfgsOptimizer


# ---- built-in backends ----
register_optimizer("ase-lbfgs", _ase_lbfgs, aliases=("lbfgs",))
register_optimizer("ase-fire", _ase_fire, aliases=("fire",))
register_optimizer("ase-bfgs", _ase_bfgs, aliases=("bfgs",))
register_optimizer("ase-precon-lbfgs", _ase_precon_lbfgs, aliases=("precon-lbfgs", "precon"))
register_optimizer("ase-fire2", _ase_fire2, aliases=("fire2",))
register_optimizer("torch-sim-fire", _ts_fire, requires_torch_sim=True)
register_optimizer("torch-sim-lbfgs", _ts_lbfgs, requires_torch_sim=True)


def _backends_snapshot() -> dict[str, tuple[Any, bool]]:
    return {n: OPTIMIZERS.get(n) for n in OPTIMIZERS.names()}


# Module-level dict kept for backward compatibility with code/tests that read
# ``BACKENDS`` directly. Snapshot of the registry (the source of truth) at import
# time; for the live set after dynamic registration use ``list_backends()``.
BACKENDS: dict[str, tuple[Any, bool]] = _backends_snapshot()


def list_backends(*, available_only: bool = False) -> list[str]:
    """Return optimizer backend names. ``available_only=True`` drops backends
    whose deps fail to import (e.g. torch-sim-* without torch_sim, fire2 on old ASE)."""
    if not available_only:
        return OPTIMIZERS.names()
    out = []
    for name in OPTIMIZERS.names():
        factory_fn, needs_ts = OPTIMIZERS.get(name)
        if needs_ts:
            try:
                import torch_sim  # noqa: F401, PLC0415
            except Exception:  # noqa: BLE001
                continue
        try:
            factory_fn()
        except Exception:  # noqa: BLE001
            continue
        out.append(name)
    return out


def make_optimizer(backend: str = "ase-lbfgs", **kwargs: Any) -> Optimizer:
    """Build an Optimizer instance for ``backend`` (name or alias).

    Raises ValueError on unknown backend; ImportError if the backend's deps are
    unavailable (e.g. torch_sim missing, FIRE2 on an old ASE).
    """
    if backend not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer backend {backend!r}. Choices: {OPTIMIZERS.names()}")
    factory_fn, _ = OPTIMIZERS.get(backend)
    cls = factory_fn()
    return cls(**kwargs)


__all__ = ["OPTIMIZERS", "register_optimizer", "make_optimizer", "list_backends",
           "BACKENDS"]
