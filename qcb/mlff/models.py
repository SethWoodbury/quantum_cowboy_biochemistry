"""
MACE model registry and calculator factory.

Centralizes model loading so all scripts use the same logic.
Import this instead of duplicating model paths in each script.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("qcb.mlff.models")

# Import paths from config (DIGS-specific, edit for other clusters)
try:
    from qcb.config import MACE_MODELS, MH_DEFAULT_HEADS, NEEDS_GBG_VENV
except ImportError:
    # Fallback if qcb package not installed — use inline defaults
    MACE_MODELS = {
        "mace-mp": "/mnt/projects/ml/mlff/models/mace_mp/MACE-matpes-r2scan-omat-ft.model",
        "mace-omol": "/home/gbg222/projects/mace_models/MACE-omol-0-extra-large-1024.model",
        "mace-mh": "/home/gbg222/projects/mace_models/mace-mh-0.model",
        "mace-polar-m": "/home/gbg222/projects/mace_models/MACE-POLAR-1-M.model",
    }
    MH_DEFAULT_HEADS = {"mace-mh": "omol"}
    NEEDS_GBG_VENV = {"mace-polar-s", "mace-polar-m", "mace-polar-l", "mace-polar"}


def get_calculator(
    model_key: str,
    device: str = "cuda",
    dtype: str = "float64",
    head: str | None = None,
):
    """Create a MACE calculator from a model key or file path.

    Args:
        model_key: Registry name ('mace-mp', 'mace-omol', etc.),
                   direct path to .model file, or convenience name.
        device: 'cuda' or 'cpu'
        dtype: 'float64' (accurate) or 'float32' (faster)
        head: For multi-head models, which DFT head to use.

    Returns:
        ASE Calculator (MACECalculator)
    """
    from mace.calculators import MACECalculator

    kwargs = dict(device=device, default_dtype=dtype)

    # Resolve head for multi-head models
    if head is None and model_key in MH_DEFAULT_HEADS:
        head = MH_DEFAULT_HEADS[model_key]
    if head:
        kwargs["head"] = head

    # Direct file path
    if os.path.isfile(model_key):
        log.info(f"Loading MACE from file: {model_key}" + (f" (head={head})" if head else ""))
        return MACECalculator(model_paths=model_key, **kwargs)

    # Try registry
    if model_key in MACE_MODELS:
        path = MACE_MODELS[model_key]
        if path and os.path.isfile(path):
            log.info(f"Loading '{model_key}'" + (f" (head={head})" if head else ""))
            return MACECalculator(model_paths=path, **kwargs)
        if path:
            log.warning(f"Model path not found: {path}")

    # Auto-download from HuggingFace
    try:
        if "omol" in model_key:
            from mace.calculators import mace_omol
            log.info("Loading MACE-OMOL via auto-download...")
            return mace_omol(model="extra_large", device=device, default_dtype=dtype)
        elif "polar" in model_key:
            from mace.calculators import mace_polar
            size = "polar-1-m"
            if "-s" in model_key:
                size = "polar-1-s"
            elif "-l" in model_key:
                size = "polar-1-l"
            log.info(f"Loading MACE-POLAR via auto-download ({size})...")
            return mace_polar(model=size, device=device, default_dtype=dtype)
        elif "off" in model_key:
            from mace.calculators import mace_off
            size = "large"
            if "small" in model_key:
                size = "small"
            elif "medium" in model_key:
                size = "medium"
            return mace_off(model=size, device=device, default_dtype=dtype)
        elif "mp" in model_key:
            from mace.calculators import mace_mp
            log.info("Loading MACE-MP via auto-download...")
            return mace_mp(device=device, default_dtype=dtype)
    except Exception as e:
        log.warning(f"Auto-download failed for {model_key}: {e}")

    raise FileNotFoundError(
        f"Model '{model_key}' not found.\n"
        f"Available: {list(MACE_MODELS.keys())}\n"
        f"Or provide a direct path to a .model file."
    )


def list_available_models() -> dict[str, dict]:
    """List all available models with their status."""
    info = {}
    for name, path in MACE_MODELS.items():
        exists = path and os.path.isfile(path)
        size_mb = os.path.getsize(path) / 1e6 if exists else None
        info[name] = {
            "path": path,
            "available": exists,
            "size_mb": f"{size_mb:.0f}" if size_mb else None,
            "needs_gbg_venv": name in NEEDS_GBG_VENV,
            "default_head": MH_DEFAULT_HEADS.get(name),
        }
    return info
