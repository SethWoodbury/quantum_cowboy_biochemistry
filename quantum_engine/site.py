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

# MACE / MLFF models — centralised HF-cache layout at
# /net/databases/huggingface/mlFF_models/ (group `baker`, setgid 2775,
# anyone in the lab can add). The HF naming convention is
# `models--<org>--<name>/<file>`.
#
# Per-alias resolution order:
#   1. `QCB_MACE_<ALIAS>` env override (e.g. QCB_MACE_POLAR=/path/to/file)
#   2. central HF location
#   3. original lab paths (/mnt/projects/... and /home/gbg222/...) — kept
#      as fallback so in-flight runs against old paths don't break.
#
# **MACE-POLAR is internal beta access** (arXiv 2602.19411v1; not a Baker
# Lab model — early access only). Hence the `-beta` suffix in the dir.

_HF_HUB_BASE = "/net/databases/huggingface/mlFF_models"


def _resolve_mace(alias: str, *candidates: str | None) -> str | None:
    """Resolve an MLFF model path by alias.

    Tries env override ``QCB_MACE_<ALIAS_UPPER>`` first, then each
    ``candidates`` path in order. Returns the first existing file. If
    none exist, returns the first non-None candidate (so downstream
    error messages point at the preferred central location). Returns
    None if every candidate is None.
    """
    if not candidates or all(c is None for c in candidates):
        return None
    env_key = f"QCB_MACE_{alias.upper().replace('-', '_')}"
    for p in [os.environ.get(env_key), *candidates]:
        if p and os.path.isfile(p):
            return p
    return next((c for c in candidates if c is not None), None)


MACE_MODELS = {
    # General-purpose (r2SCAN, all elements incl. metals)
    "mace-mp": _resolve_mace("mace-mp",
        f"{_HF_HUB_BASE}/models--ACEsuit--mace-mp-0/MACE-matpes-r2scan-omat-ft.model",
        "/mnt/projects/ml/mlff/models/mace_mp/MACE-matpes-r2scan-omat-ft.model",
    ),
    "mace-mp-old": None,  # auto-downloads from HuggingFace at run-time

    # Organic molecules (wB97M-D3BJ, NO metals)
    "mace-off-small": _resolve_mace("mace-off-small",
        f"{_HF_HUB_BASE}/models--ACEsuit--mace-off-23/MACE-OFF23_small.model",
        "/mnt/projects/ml/mlff/models/mace_off/MACE-OFF23_small.model",
    ),
    "mace-off-medium": _resolve_mace("mace-off-medium",
        f"{_HF_HUB_BASE}/models--ACEsuit--mace-off-23/MACE-OFF23_medium.model",
        "/mnt/projects/ml/mlff/models/mace_off/MACE-OFF23_medium.model",
    ),
    "mace-off": _resolve_mace("mace-off",
        f"{_HF_HUB_BASE}/models--ACEsuit--mace-off-23/MACE-OFF23_large.model",
        "/mnt/projects/ml/mlff/models/mace_off/MACE-OFF23_large.model",
    ),
    "mace-off-large": _resolve_mace("mace-off-large",
        f"{_HF_HUB_BASE}/models--ACEsuit--mace-off-23/MACE-OFF23_large.model",
        "/mnt/projects/ml/mlff/models/mace_off/MACE-OFF23_large.model",
    ),
    # MACE-OFF24 — newer organic-molecule training (still wB97M-D3BJ, no
    # metals — useful for substrate/residue cleanup).
    "mace-off-24": _resolve_mace("mace-off-24",
        f"{_HF_HUB_BASE}/models--ACEsuit--mace-off-24/MACE-OFF24_medium.model",
        None,
    ),
    "mace-off-24-medium": _resolve_mace("mace-off-24-medium",
        f"{_HF_HUB_BASE}/models--ACEsuit--mace-off-24/MACE-OFF24_medium.model",
        None,
    ),

    # Charge-aware, trained on TS data (wB97M-V, all elements)
    "mace-omol": _resolve_mace("mace-omol",
        f"{_HF_HUB_BASE}/models--ACEsuit--mace-omol-0/MACE-omol-0-extra-large-1024.model",
        "/home/gbg222/projects/mace_models/MACE-omol-0-extra-large-1024.model",
    ),

    # Multi-head (7 DFT levels in one model)
    "mace-mh": _resolve_mace("mace-mh",
        f"{_HF_HUB_BASE}/models--ACEsuit--mace-mh-0/mace-mh-0.model",
        "/home/gbg222/projects/mace_models/mace-mh-0.model",
    ),

    # Polarizable — BETA / EARLY ACCESS (arXiv 2602.19411v1, not Baker Lab).
    # Needs graph_electrostatics → gbg222 venv only.
    "mace-polar-s": _resolve_mace("mace-polar-s",
        f"{_HF_HUB_BASE}/models--ACEsuit--mace-polar-1-beta/MACE-POLAR-1-S.model",
        "/home/gbg222/projects/mace_models/MACE-POLAR-1-S.model",
    ),
    "mace-polar-m": _resolve_mace("mace-polar-m",
        f"{_HF_HUB_BASE}/models--ACEsuit--mace-polar-1-beta/MACE-POLAR-1-M.model",
        "/home/gbg222/projects/mace_models/MACE-POLAR-1-M.model",
    ),
    "mace-polar-l": _resolve_mace("mace-polar-l",
        f"{_HF_HUB_BASE}/models--ACEsuit--mace-polar-1-beta/MACE-POLAR-1-L.model",
        "/home/gbg222/projects/mace_models/MACE-POLAR-1-L.model",
    ),
    "mace-polar": _resolve_mace("mace-polar",
        f"{_HF_HUB_BASE}/models--ACEsuit--mace-polar-1-beta/MACE-POLAR-1-M.model",
        "/home/gbg222/projects/mace_models/MACE-POLAR-1-M.model",
    ),

    # FairChem UMA (different calculator interface)
    "uma-sm": _resolve_mace("uma-sm",
        f"{_HF_HUB_BASE}/models--facebook--fairchem-uma-sm/uma_sm.pt",
        "/mnt/projects/ml/mlff/models/fairchem/UMA/uma_sm.pt",
    ),
    # UMA-S-1.2 / UMA-M-1.1 — current FairChem foundation models.
    # GATED on HuggingFace (FAIR Chemistry License v1) — see the README
    # in each placeholder dir for the manual download steps. The
    # _resolve_mace helper returns the central path as the preferred
    # location even when the file is missing, so error messages from
    # make_calc(...) point users straight at the placeholder README.
    "uma-s-1p2": _resolve_mace("uma-s-1p2",
        f"{_HF_HUB_BASE}/models--facebook--fairchem-uma-s-1p2/uma-s-1p2.pt",
        None,
    ),
    "uma-m-1p1": _resolve_mace("uma-m-1p1",
        f"{_HF_HUB_BASE}/models--facebook--fairchem-uma-m-1p1/uma-m-1p1.pt",
        None,
    ),

    # OrbMol (Orbital Materials, OMol25-trained, charge/spin-aware,
    # Zn-capable, Apache-2.0). Conservative variant — forces are
    # gradients of a defined energy surface (matters for saddle search).
    "orb-mol": _resolve_mace("orb-mol",
        f"{_HF_HUB_BASE}/models--orbital-materials--orb-mol-conservative/orb-v3-conservative-omol-20250820.ckpt",
        None,
    ),
    "orb-mol-conservative": _resolve_mace("orb-mol-conservative",
        f"{_HF_HUB_BASE}/models--orbital-materials--orb-mol-conservative/orb-v3-conservative-omol-20250820.ckpt",
        None,
    ),

    # AIMNet2-rxn (Isayev lab, organic-only TS-tuned ensemble — H/C/N/O,
    # 4-member safetensors ensemble + config). Loaded via the aimnet pip
    # package. Path here points at ensemble member 0; the calculator
    # walks the dir for the rest. Encoded as a directory because the
    # safetensors files only resolve as a set with config.json.
    "aimnet2-rxn": _resolve_mace("aimnet2-rxn",
        f"{_HF_HUB_BASE}/models--isayevlab--aimnet2-rxn/ensemble_0.safetensors",
        None,
    ),
}

# Multi-head default heads
MH_DEFAULT_HEADS = {
    "mace-mh": "omol",
}

# Models that need gbg222's venv (has graph_electrostatics for POLAR)
NEEDS_GBG_VENV = {"mace-polar-s", "mace-polar-m", "mace-polar-l", "mace-polar"}

# ── External QM packages ───────────────────────────────────────
# Cluster-installed packages we shell out to. Each path is the binary
# itself (or, for Gaussian, the install root).
GAUSSIAN_ROOT = "/net/software/gaussian/g16"
GAUSSIAN_BIN = f"{GAUSSIAN_ROOT}/g16"

# ORCA — locally installed at the cluster level, multiple versions:
ORCA_BIN = "/net/software/orca/orca_4_1_1_linux_x86-64_openmpi313/orca"
ORCA_VERSIONS = {
    "4.1.1": "/net/software/orca/orca_4_1_1_linux_x86-64_openmpi313/orca",
    "4.0.1.2": "/net/software/orca/orca_4_0_1_2_linux_x86-64_openmpi202/orca",
}

# ── External chemistry workflow tools ──────────────────────────
# Wired below after _REPO_ROOT is defined.

# xtb is vendored as a git submodule at deps/xtb (see deps/README.md). The
# build script writes the binary + libxtb.so under deps/xtb/install/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
XTB_BIN = str(_REPO_ROOT / "deps" / "xtb" / "install" / "bin" / "xtb")
# Shared libs that the vendored xtb needs at runtime (libxtb, libquadmath).
# Tools that subprocess xtb should put these on LD_LIBRARY_PATH; they're
# also needed by xtb-python's ASE wrapper if that's installed alongside.
XTB_LIB_DIRS = [
    str(_REPO_ROOT / "deps" / "xtb" / "install" / "lib" / "x86_64-linux-gnu"),
    "/home/woodbuse/conda/envs/qcb-xtb/lib",
]

# g-xTB is a *separate* binary that supports an extra `--gxtb` flag for
# Grimme's ωB97M-V/def2-TZVPPD-trained method. Vendored from
# https://github.com/grimme-lab/g-xtb (the repo ships a static linux tarball).
# Distinct from XTB_BIN — that one supports GFN-FF / GFN1 / GFN2 only.
GXTB_BIN = str(_REPO_ROOT / "deps" / "g-xtb" / "install"
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
_repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_kernel_candidates = [
    _os.environ.get("QCB_PLUMED_KERNEL"),
    _os.path.join(_repo_root, "deps", "plumed2", "install", "lib", "libplumedKernel.so"),
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


# ─────────────────────────────────────────────────────────────────────
# External chemistry workflow tools (resolved last so _REPO_ROOT exists)
# ─────────────────────────────────────────────────────────────────────

def _resolve_crest() -> str | None:
    """CREST (Grimme conformer search). Vendored at deps/crest with
    deps/build_crest.sh. Override via $QCB_CREST_BIN."""
    candidates = [
        os.environ.get("QCB_CREST_BIN"),
        str(_REPO_ROOT / "deps" / "crest" / "install" / "bin" / "crest"),
        "/net/software/lab/crest/bin/crest",
    ]
    return next((p for p in candidates if p and os.path.isfile(p)), None)


def _resolve_chemshell() -> str | None:
    """ChemShell (CCFE QM/MM TS framework). Override via $QCB_CHEMSHELL."""
    candidates = [
        os.environ.get("QCB_CHEMSHELL"),
        str(_REPO_ROOT / "deps" / "chemshell" / "install" / "bin" / "chemsh"),
        "/net/software/lab/chemshell/bin/chemsh",
    ]
    return next((p for p in candidates if p and os.path.isfile(p)), None)


CREST_BIN = _resolve_crest()
CHEMSHELL_BIN = _resolve_chemshell()


# ─────────────────────────────────────────────────────────────────────
# Lab-shared data + install locations
# ─────────────────────────────────────────────────────────────────────
# Two parallel scratch namespaces on DIGS:
#   * /net/databases — lab-shared scientific data (ChEBI dump, M-CSA
#     JSON, RGD1 archives, MME55 SI, etc.). Read-mostly; we cache here
#     when we have write perms, so the next user doesn't re-download.
#   * /net/software/lab — lab-shared installs (parallel to /net/software,
#     which is admin-managed). Tools we vendor under deps/ get copied
#     here when QCB_INSTALL_LAB_SHARED=1 is set during build.

DATABASES_DIR = "/net/databases"
LAB_SOFTWARE = "/net/software/lab"

# Lab-mounted PDB mirror — hash-bucketed (../pdb/<2-char hash>/<id>.pdb.gz),
# updated weekly by the cluster. Use this instead of the RCSB FTP to keep
# pipelines deterministic + offline-friendly. Falls back to None if the
# mount isn't visible (laptop dev, etc.) — callers should detect and
# fall back to RCSB HTTP.
PDB_MIRROR = "/net/databases/rcsb/pdb" if os.path.isdir("/net/databases/rcsb/pdb") else None
CIF_MIRROR = "/net/databases/rcsb/cif" if os.path.isdir("/net/databases/rcsb/cif") else None


def _resolve_cache_dir(name: str) -> Path:
    """Pick a writable cache dir: prefer DATABASES_DIR/<name>, fall back
    to <repo>/data/<name>. Lazy — creates the dir on first call only."""
    shared = Path(DATABASES_DIR) / name
    try:
        shared.mkdir(parents=True, exist_ok=True)
        return shared
    except (PermissionError, OSError):
        local = _REPO_ROOT / "data" / name
        local.mkdir(parents=True, exist_ok=True)
        return local


def mcsa_cache_dir() -> Path:
    """Cache dir for M-CSA JSON entries (per-entry .json files)."""
    return _resolve_cache_dir("mcsa_cache")


def chebi_cache_dir() -> Path:
    """Cache dir for ChEBI flat-file dump + per-id SMILES lookups."""
    return _resolve_cache_dir("chebi_cache")


def benchmark_cache_dir() -> Path:
    """Cache dir for downloaded TS benchmark datasets (RGD1, MME55, …)."""
    return _resolve_cache_dir("benchmark_cache")


# ─────────────────────────────────────────────────────────────────────
# molecularGSM (ZimmermanGroup) — C++/CMake + MKL source build.
# Vendored at deps/molecularGSM. No pip/conda path; binary-only tool
# driven via subprocess. Build with `bash deps/build_molecular_gsm.sh`.
# ─────────────────────────────────────────────────────────────────────

def _resolve_molecular_gsm() -> str | None:
    candidates = [
        os.environ.get("QCB_MOLECULAR_GSM_BIN"),
        str(_REPO_ROOT / "deps" / "molecularGSM" / "install" / "bin" / "gsm"),
        f"{LAB_SOFTWARE}/molecularGSM/current/bin/gsm",
    ]
    return next((p for p in candidates if p and os.path.isfile(p)), None)


MOLECULAR_GSM_BIN = _resolve_molecular_gsm()
