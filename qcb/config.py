"""
Cluster-specific paths and configuration.

This is the only file that needs editing to port QCB to a different cluster.
All other modules import paths from here.
"""

import os
from pathlib import Path

# ══════════════════════════════════════════════════════════════
#  DIGS cluster paths (Baker Lab, University of Washington)
# ══════════════════════════════════════════════════════════════

# Apptainer containers
CONTAINERS = {
    "universal": "/net/software/containers/universal.sif",
}

# MACE model files on DIGS
MACE_MODELS = {
    # General-purpose (r2SCAN, all elements incl. metals)
    "mace-mp": "/mnt/projects/ml/mlff/models/mace_mp/MACE-matpes-r2scan-omat-ft.model",
    "mace-mp-old": None,  # auto-downloads from HuggingFace

    # Organic molecules (wB97M-D3BJ, NO metals)
    "mace-off-small": "/mnt/projects/ml/mlff/models/mace_off/MACE-OFF23_small.model",
    "mace-off-medium": "/mnt/projects/ml/mlff/models/mace_off/MACE-OFF23_medium.model",
    "mace-off": "/mnt/projects/ml/mlff/models/mace_off/MACE-OFF23_large.model",
    "mace-off-large": "/mnt/projects/ml/mlff/models/mace_off/MACE-OFF23_large.model",

    # Charge-aware, trained on TS data (wB97M-V, all elements)
    "mace-omol": "/home/gbg222/projects/mace_models/MACE-omol-0-extra-large-1024.model",

    # Multi-head (7 DFT levels in one model)
    "mace-mh": "/home/gbg222/projects/mace_models/mace-mh-0.model",

    # Polarizable (needs graph_electrostatics → gbg222 venv only)
    "mace-polar-s": "/home/gbg222/projects/mace_models/MACE-POLAR-1-S.model",
    "mace-polar-m": "/home/gbg222/projects/mace_models/MACE-POLAR-1-M.model",
    "mace-polar-l": "/home/gbg222/projects/mace_models/MACE-POLAR-1-L.model",
    "mace-polar": "/home/gbg222/projects/mace_models/MACE-POLAR-1-M.model",

    # FairChem UMA (different calculator interface)
    "uma-sm": "/mnt/projects/ml/mlff/models/fairchem/UMA/uma_sm.pt",
}

# Multi-head default heads
MH_DEFAULT_HEADS = {
    "mace-mh": "omol",
}

# Models that need gbg222's venv (has graph_electrostatics for POLAR)
NEEDS_GBG_VENV = {"mace-polar-s", "mace-polar-m", "mace-polar-l", "mace-polar"}

# QM software
GAUSSIAN_ROOT = "/net/software/gaussian/g16"
XTB_BIN = "/home/dme5188/bin/xtb/xtb-6.6.1/bin/xtb"
OPENBABEL_BIN = "/home/dme5188/bin/openbabel/bin/obabel"

# Reference scripts from collaborators
GBG_ENZTS_VENV = "/home/gbg222/projects/enz-ts/.venv/bin/python"
GBG_ENZTS_SRC = "/home/gbg222/projects/enz-ts/src"

# Python environments
UNIVERSAL_PYTHON = f"apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net {CONTAINERS['universal']} python"

# Scratch directory for temp files
SCRATCH_DIR = os.environ.get("SCRATCH", f"/net/scratch/{os.environ.get('USER', 'unknown')}")


def get_project_root():
    """Return the root directory of this project."""
    return Path(__file__).parent.parent


def get_deps_dir():
    """Return the deps directory for local package installs (.local_pkgs)."""
    return get_project_root() / "deps"
