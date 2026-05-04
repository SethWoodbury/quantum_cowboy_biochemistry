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


# Single source of truth for MACE model paths: quantum_engine.config.MACE_MODELS.
# Edit that file (or override via env vars per the layered-config story) to
# add/move models; this factory just reads.
try:
    from quantum_engine.site import MACE_MODELS
except Exception:
    MACE_MODELS = {}


def list_models() -> dict[str, str]:
    """Return dict of available model aliases and their file paths."""
    return dict(MACE_MODELS)


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
    models = list_models()

    # 1. Direct file path
    if os.path.isfile(model):
        model_path = model
    # 2. Registry hit with the model file present locally
    elif model in models and models[model] and os.path.isfile(models[model]):
        model_path = models[model]
    # 3. HuggingFace auto-download fallback for known model families.
    #    Lets users get going on a fresh machine where the local registry
    #    paths don't exist yet — picks a sensible variant per family.
    else:
        if model in models and models[model]:
            log.warning(f"Registry path missing on disk: {models[model]} — "
                        f"falling back to HuggingFace auto-download")
        try:
            if "omol" in model:
                from mace.calculators import mace_omol
                log.info(f"Loading MACE-OMOL via auto-download")
                return mace_omol(model="extra_large", device=device,
                                 default_dtype=default_dtype)
            if "polar" in model:
                from mace.calculators import mace_polar
                size = "polar-1-m"
                if "-s" in model: size = "polar-1-s"
                elif "-l" in model: size = "polar-1-l"
                log.info(f"Loading MACE-POLAR via auto-download ({size})")
                return mace_polar(model=size, device=device,
                                  default_dtype=default_dtype)
            if "off" in model:
                from mace.calculators import mace_off
                size = "large"
                if "small" in model: size = "small"
                elif "medium" in model: size = "medium"
                log.info(f"Loading MACE-OFF via auto-download ({size})")
                return mace_off(model=size, device=device,
                                default_dtype=default_dtype)
            if "mp" in model:
                from mace.calculators import mace_mp
                log.info(f"Loading MACE-MP via auto-download")
                return mace_mp(device=device, default_dtype=default_dtype)
        except Exception as e:
            log.warning(f"Auto-download failed for {model}: {e}")
        # Last resort: tell the user what we tried.
        available = ", ".join(models.keys())
        raise FileNotFoundError(
            f"Model '{model}' could not be resolved.\n"
            f"  - Not a file: {model}\n"
            f"  - Registry path: "
            f"{models.get(model, '(not in registry)')!r}\n"
            f"  - HuggingFace auto-download family didn't match.\n"
            f"  Available registry keys: {available}"
        )

    log.info(f"Loading '{model}' from {model_path}" + (f" (head={head})" if head else ""))

    from mace.calculators import MACECalculator

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
