"""MACE calculator factory + model registry.

Extracted from scripts/run_neb_ts.py to provide a unified, module-level way
to instantiate calculators. All ops (opt, md, freq, neb, ts, ...) use this.

Example
-------
>>> from quantum_engine.calc import make_calc
>>> atoms.calc = make_calc("mace-omol", charge=0)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger("quantum_engine.calc")


# MACE model paths on DIGS cluster (override via qcb.config if needed)
try:
    from quantum_engine.config import MACE_MODELS as _CONFIGURED_MODELS
except Exception:
    _CONFIGURED_MODELS = None


DEFAULT_MODEL_PATHS = {
    # ── General-purpose foundation models ──
    "mace-mp":           "/mnt/projects/ml/mlff/models/mace_mp/MACE-matpes-r2scan-omat-ft.model",
    "mace-omol":         "/home/gbg222/projects/mace_models/MACE-omol-0-extra-large-1024.model",
    "mace-mh":           "/home/gbg222/projects/mace_models/mace-mh-0.model",
    # ── MACE-OFF (organic only, no metals) ──
    "mace-off-small":    "/mnt/projects/ml/mlff/models/mace_off/MACE-OFF23_small.model",
    "mace-off-medium":   "/mnt/projects/ml/mlff/models/mace_off/MACE-OFF23_medium.model",
    "mace-off":          "/mnt/projects/ml/mlff/models/mace_off/MACE-OFF23_large.model",
    "mace-off-large":    "/mnt/projects/ml/mlff/models/mace_off/MACE-OFF23_large.model",
    # ── MACE-POLAR-1 (charge/electrostatics-aware, recommended for ionic systems) ──
    # See: arXiv:2602.19411 — long-range electrostatics in MACE
    "mace-polar-s":      "/projects/ml/enzyme_filtering/enz-ts/lib/mace_models/MACE-POLAR-1-S.model",
    "mace-polar-m":      "/projects/ml/enzyme_filtering/enz-ts/lib/mace_models/MACE-POLAR-1-M.model",
    "mace-polar":        "/projects/ml/enzyme_filtering/enz-ts/lib/mace_models/MACE-POLAR-1-M.model",  # default = M
    "mace-polar-l":      "/projects/ml/enzyme_filtering/enz-ts/lib/mace_models/MACE-POLAR-1-L.model",
    # ── Legacy ──
    "mace-128":          "/projects/ml/enzyme_filtering/enz-ts/lib/mace_models/2023-12-10-mace-128-L0_energy_epoch-249.model",
}


def list_models() -> dict[str, str]:
    """Return dict of available model aliases and their file paths."""
    models = dict(DEFAULT_MODEL_PATHS)
    if _CONFIGURED_MODELS:
        models.update({k: v for k, v in _CONFIGURED_MODELS.items() if v})
    return models


@dataclass
class CalcSpec:
    """Specification for a calculator (can be persisted to YAML/JSON)."""
    model: str = "mace-omol"
    head: str | None = None          # e.g., "omol" for mace-mh multi-head
    device: str = "cuda"
    default_dtype: str = "float64"
    charge: int | None = None


def make_calc(
    model: str = "mace-omol",
    head: str | None = None,
    device: str = "cuda",
    default_dtype: str = "float64",
    charge: int | None = None,
) -> "Calculator":
    """Instantiate a MACE calculator.

    Args:
        model: Alias ('mace-omol', 'mace-mp', 'mace-mh', 'mace-off', ...)
               or an absolute path to a .model file.
        head: For multi-head models (mace-mh), specify the head.
              E.g., 'omol' for the wB97M-V head.
        device: 'cuda' or 'cpu'
        default_dtype: 'float32' or 'float64'
        charge: System net charge (only used by charge-aware models like mace-omol)

    Returns:
        ASE-compatible MACE calculator
    """
    from mace.calculators import MACECalculator

    models = list_models()

    if os.path.isfile(model):
        model_path = model
    elif model in models:
        model_path = models[model]
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model '{model}' → '{model_path}' not found. "
                f"Update qcb/config.py or pass an absolute path."
            )
    else:
        available = ", ".join(models.keys())
        raise ValueError(f"Unknown model '{model}'. Available: {available}")

    log.info(f"Loading '{model}' from {model_path}" + (f" (head={head})" if head else ""))

    kwargs = dict(
        model_paths=model_path,
        device=device,
        default_dtype=default_dtype,
    )
    if head:
        kwargs["head"] = head
    calc = MACECalculator(**kwargs)

    # Charge-aware models: some accept 'charges' via atoms.info
    if charge is not None:
        # Users still need to set atoms.info["charge"] = charge for the calc to read it
        log.debug(f"  Note: set atoms.info['charge'] = {charge} for charge-aware evaluation")

    return calc


def make_calc_fn(**kwargs) -> Callable[[], "Calculator"]:
    """Return a zero-arg callable that constructs a fresh calculator each call.

    Useful for NEB where each image needs its own calculator instance.
    """
    def _factory():
        return make_calc(**kwargs)
    return _factory
