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

# xtb is vendored as a git submodule at deps/xtb (see deps/README.md). The
# build script writes the binary + libxtb.so under deps/xtb/install/.
_QCB_ROOT_FOR_XTB = Path(__file__).resolve().parent.parent
XTB_BIN = str(_QCB_ROOT_FOR_XTB / "deps" / "xtb" / "install" / "bin" / "xtb")
# Shared libs that the vendored xtb needs at runtime (libxtb, libquadmath).
# Tools that subprocess xtb should put these on LD_LIBRARY_PATH; they're
# also needed by xtb-python's ASE wrapper if that's installed alongside.
XTB_LIB_DIRS = [
    str(_QCB_ROOT_FOR_XTB / "deps" / "xtb" / "install" / "lib" / "x86_64-linux-gnu"),
    "/home/woodbuse/conda/envs/qcb-xtb/lib",
]

# g-xTB is a *separate* binary that supports an extra `--gxtb` flag for
# Grimme's ωB97M-V/def2-TZVPPD-trained method. Vendored from
# https://github.com/grimme-lab/g-xtb (the repo ships a static linux tarball).
# Distinct from XTB_BIN — that one supports GFN-FF / GFN1 / GFN2 only.
GXTB_BIN = str(_QCB_ROOT_FOR_XTB / "deps" / "g-xtb" / "install"
               / "xtb-6.7.1" / "bin" / "xtb")
GXTB_LIB_DIRS: list[str] = []  # static binary — no extra LD_LIBRARY_PATH needed

OPENBABEL_BIN = "/home/dme5188/bin/openbabel/bin/obabel"

# ChimeraX — for empirical H placement + Gasteiger charges
# Falls back to None if not found; consensus protonation will skip ChimeraX.
def _find_chimera() -> str | None:
    import os as __os
    candidates = [
        "/projects/ml/enzyme_filtering/enz-ts/kernels/chimerax/usr/lib/ucsf-chimerax/bin/ChimeraX",
        "/net/software/chimerax/bin/ChimeraX",
    ]
    for p in candidates:
        if __os.path.isfile(p):
            return p
    from shutil import which
    return which("ChimeraX") or which("chimerax")

CHIMERA_KERNEL = _find_chimera()

# PLUMED 2 — for advanced sampling (MTD, OPES, umbrella, multi-walker)
# Resolution order (deliberately ignores the standard $PLUMED_KERNEL env
# variable; that one is often set in user shells for plain `plumed` CLI
# use and may point at a stale or about-to-be-deleted path).
#   1. $QCB_PLUMED_KERNEL (qcb-specific env override)
#   2. deps/plumed2/install/lib/libplumedKernel.so (vendored submodule)
#   3. /net/software/lab/plumed2-2.10/lib/libplumedKernel.so (lab-shared
#      install — populated by `cp -a deps/plumed2/install
#      /net/software/lab/plumed2-2.10` after a vendored build)
# Build the submodule with `bash deps/plumed2_build.sh`.
import os as _os
_qcb_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_kernel_candidates = [
    _os.environ.get("QCB_PLUMED_KERNEL"),
    _os.path.join(_qcb_root, "deps", "plumed2", "install", "lib", "libplumedKernel.so"),
    "/net/software/lab/plumed2-2.10/lib/libplumedKernel.so",
]
PLUMED_KERNEL = next(
    (p for p in _kernel_candidates if p and _os.path.isfile(p)),
    None,
)

# Reference codebases (read-only, for migration/comparison)
GBG_ENZTS_VENV = "/home/gbg222/projects/enz-ts/.venv/bin/python"
GBG_ENZTS_SRC = "/home/gbg222/projects/enz-ts/src"
ENZTS_PROD = "/projects/ml/enzyme_filtering/enz-ts"  # production enz-ts (gbg222/Lars)

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
