"""Factory for ``Optimizer`` instances.

Single source of truth for optimizer-backend strings used across QCB
tools (``--optimizer-backend``).
"""
from __future__ import annotations

import logging
from typing import Any

from quantum_engine.opt.base import Optimizer

log = logging.getLogger("quantum_engine.opt.factory")


# Lazy imports per-backend so e.g. torch_sim being unavailable doesn't
# break ``import quantum_engine.opt``.
def _ase_lbfgs():
    from quantum_engine.opt.ase_optimizers import AseLbfgsOptimizer  # noqa: PLC0415
    return AseLbfgsOptimizer

def _ase_fire():
    from quantum_engine.opt.ase_optimizers import AseFireOptimizer  # noqa: PLC0415
    return AseFireOptimizer

def _ase_bfgs():
    from quantum_engine.opt.ase_optimizers import AseBfgsOptimizer  # noqa: PLC0415
    return AseBfgsOptimizer

def _ts_fire():
    from quantum_engine.opt.torchsim_optimizers import TorchSimFireOptimizer  # noqa: PLC0415
    return TorchSimFireOptimizer

def _ts_lbfgs():
    from quantum_engine.opt.torchsim_optimizers import TorchSimLbfgsOptimizer  # noqa: PLC0415
    return TorchSimLbfgsOptimizer


# Backend name → (factory_fn, requires_torch_sim)
BACKENDS: dict[str, tuple[Any, bool]] = {
    "ase-lbfgs":       (_ase_lbfgs,  False),
    "ase-fire":        (_ase_fire,   False),
    "ase-bfgs":        (_ase_bfgs,   False),
    "torch-sim-fire":  (_ts_fire,    True),
    "torch-sim-lbfgs": (_ts_lbfgs,   True),
}


def list_backends(*, available_only: bool = False) -> list[str]:
    """Return optimizer backend names.

    ``available_only=True`` filters out backends whose imports fail
    (e.g. ``torch-sim-*`` when torch_sim isn't installed).
    """
    if not available_only:
        return list(BACKENDS)
    out = []
    for name, (factory_fn, needs_ts) in BACKENDS.items():
        if needs_ts:
            try:
                import torch_sim  # noqa: F401, PLC0415
            except Exception:
                continue
        try:
            factory_fn()
        except Exception:
            continue
        out.append(name)
    return out


def make_optimizer(backend: str = "ase-lbfgs", **kwargs: Any) -> Optimizer:
    """Build an Optimizer instance.

    Args:
        backend: One of ``BACKENDS`` keys. Default ``"ase-lbfgs"``
                 (matches legacy QCB default).
        **kwargs: Forwarded to the optimizer constructor — typically
                  ``fmax``, ``max_steps``, ``outdir``, ``logfile``,
                  ``trajectory``. Backend-specific extras (``device``,
                  ``dtype`` for torch-sim) go through ``**kwargs`` too
                  and end up on ``self.extra``.

    Raises:
        ValueError: unknown backend.
        ImportError: backend deps unavailable (e.g. torch_sim missing).
    """
    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown optimizer backend {backend!r}. "
            f"Choices: {sorted(BACKENDS)}"
        )
    factory_fn, _ = BACKENDS[backend]
    cls = factory_fn()
    return cls(**kwargs)
