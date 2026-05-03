#!/usr/bin/env bash
# molecularGSM (ZimmermanGroup) — C++/CMake + MKL source build.
#
# No pip or conda path exists upstream; molecularGSM is a binary-only
# tool driven via subprocess. We vendor it at deps/molecularGSM and
# build into deps/molecularGSM/install/bin/gsm so site.py:MOLECULAR_GSM_BIN
# auto-resolves.
#
# Heads up:
#  * Requires Intel MKL (LAPACK/BLAS). We pull mkl + mkl-devel via
#    conda-forge into the qcb-xtb env so the build is self-contained.
#  * Distinct from pyGSM (Python; deps/pyGSM): molecularGSM is the
#    legacy C++ compile from the same group, generally faster for big
#    runs but lacks Python bindings.
#  * Lab-shared install: QCB_INSTALL_LAB_SHARED=1 copies the binary
#    under /net/software/lab/molecularGSM/<sha>/ and points
#    /net/software/lab/molecularGSM/current at it.
#
# Usage:
#   bash deps/build_molecular_gsm.sh
#   QCB_INSTALL_LAB_SHARED=1 bash deps/build_molecular_gsm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/deps/molecularGSM"
INSTALL="${SRC}/install"
ENV_PREFIX="${ENV_PREFIX:-/home/woodbuse/conda/envs/qcb-xtb}"

if [ ! -d "${SRC}" ] || [ -z "$(ls -A "${SRC}" 2>/dev/null || true)" ]; then
    echo "ERROR: ${SRC} missing or empty." >&2
    echo "       Run 'git submodule update --init deps/molecularGSM'" >&2
    exit 1
fi

if ! command -v cmake >/dev/null; then
    echo "ERROR: cmake required (apt install cmake or conda install cmake)" >&2
    exit 1
fi

# MKL — pull into the env once. molecularGSM CMakeLists looks for MKLROOT.
if [ ! -f "${ENV_PREFIX}/lib/libmkl_rt.so" ]; then
    echo "==> Installing MKL into qcb-xtb (one-time, ~500 MB)…"
    /home/woodbuse/conda/bin/conda install -n qcb-xtb -c conda-forge \
        mkl mkl-devel -y >&2
fi
export MKLROOT="${ENV_PREFIX}"

BUILD_DIR="${BUILD_DIR:-/tmp/qcb_molgsm_build_$USER}"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "==> Configuring…"
cmake "${SRC}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL}" \
    -DCMAKE_BUILD_TYPE=Release

echo "==> Building (j=${NJOBS:-8})…"
make -j "${NJOBS:-8}"
make install

if [ ! -x "${INSTALL}/bin/gsm" ]; then
    echo "ERROR: build finished but ${INSTALL}/bin/gsm missing." >&2
    exit 1
fi
echo "molecularGSM built → ${INSTALL}/bin/gsm"

# Lab-shared install
if [ "${QCB_INSTALL_LAB_SHARED:-0}" = "1" ]; then
    SHA="$(git -C "${SRC}" rev-parse --short HEAD)"
    LAB_DIR="/net/software/lab/molecularGSM/${SHA}"
    mkdir -p "${LAB_DIR}"
    cp -r "${INSTALL}"/* "${LAB_DIR}/"
    ln -sfn "${LAB_DIR}" "/net/software/lab/molecularGSM/current"
    echo "Lab-shared install → ${LAB_DIR} (current → ${LAB_DIR})"
fi
