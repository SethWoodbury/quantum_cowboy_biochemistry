#!/usr/bin/env bash
# Install OpenMM + companion packages (openmm-ml, openmm-plumed,
# openmm-torch) into the qcb-xtb conda env. Same pattern as CREST:
# default to conda-forge (matches the SHA-pinned submodule's tag);
# BUILD_FROM_SOURCE=1 if you need a custom build.
#
# Companions:
#   * openmm-ml      — MACE / ANI / NequIP / TorchANI as OpenMM forces
#   * openmm-plumed  — PLUMED 2 inside an OpenMM simulation (we already
#                       vendor PLUMED; this binds it to OpenMM's loop)
#   * openmm-torch   — generic PyTorch force, lets you plug arbitrary
#                       models in without writing C++
#
# Together they make OpenMM a "swap any force field" engine: classical
# (AMBER/CHARMM/OPLS), ML (MACE/ANI), or QM (xtb via ASE bridge) — all
# under the same MD integrator with PLUMED biasing on top.
#
# Usage:
#   bash deps/build_openmm.sh                  # conda path (default)
#   BUILD_FROM_SOURCE=1 bash deps/build_openmm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/home/woodbuse/conda/envs/qcb-xtb}"
OPENMM_SRC="${ROOT}/deps/openmm"
INSTALL_DIR="${OPENMM_SRC}/install"
BUILD_FROM_SOURCE="${BUILD_FROM_SOURCE:-0}"

if [ "${BUILD_FROM_SOURCE}" = "1" ]; then
    if ! command -v cmake >/dev/null; then
        echo "ERROR: cmake required for source build (apt install cmake)" >&2
        exit 1
    fi
    BUILD_DIR="${BUILD_DIR:-/tmp/qcb_openmm_build_$USER}"
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake "${OPENMM_SRC}" \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
        -DOPENMM_BUILD_PYTHON_WRAPPERS=ON \
        -DCMAKE_BUILD_TYPE=Release
    make -j "${NJOBS:-8}"
    make install
    make PythonInstall
    echo "Source build complete → ${INSTALL_DIR}"
else
    if [ ! -x "${ENV_PREFIX}/bin/conda" ] && ! command -v conda >/dev/null; then
        echo "ERROR: no conda. Set BUILD_FROM_SOURCE=1 or install conda." >&2
        exit 1
    fi
    echo "==> Installing openmm + companions from conda-forge into qcb-xtb…"
    /home/woodbuse/conda/bin/conda install -n qcb-xtb -c conda-forge \
        "openmm=8.5.1" \
        "openmm-ml" \
        "openmm-plumed" \
        -y >&2
    # openmm-torch is GPU-only (cuda runtime in package name); install
    # only if a GPU is present and torch is on the env.
    if "${ENV_PREFIX}/bin/python" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        /home/woodbuse/conda/bin/conda install -n qcb-xtb -c conda-forge openmm-torch -y >&2
    else
        echo "  (skipping openmm-torch: no CUDA torch detected)"
    fi
    echo "Conda install complete → ${ENV_PREFIX}/lib/python*/site-packages/openmm"
    "${ENV_PREFIX}/bin/python" -c "import openmm; print('OpenMM version:', openmm.version.full_version)"
fi
