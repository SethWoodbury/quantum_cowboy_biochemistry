#!/usr/bin/env bash
# Resolve a usable `crest` binary for quantum_engine. Two paths:
#
#   1. (default) install conda-forge crest=3.0.2 into the qcb-xtb env. Fast
#      (~2 min). Binary matches the deps/crest submodule's SHA-pinned tag,
#      so the in-repo source tree stays the canonical "what version are
#      we on" answer.
#
#   2. Source build (BUILD_FROM_SOURCE=1). Uses meson + the qcb-xtb env's
#      gfortran toolchain, mirroring deps/build_xtb.sh. Slower (~30 min)
#      but lets us patch upstream locally.
#
# After either path, the resulting binary is symlinked to
# deps/crest/install/bin/crest so quantum_engine.site.CREST_BIN finds it.
#
# Usage:
#   bash deps/build_crest.sh                    # conda path
#   BUILD_FROM_SOURCE=1 bash deps/build_crest.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/home/woodbuse/conda/envs/qcb-xtb}"
CREST_DIR="${ROOT}/deps/crest"
INSTALL_DIR="${CREST_DIR}/install"
BUILD_FROM_SOURCE="${BUILD_FROM_SOURCE:-0}"

mkdir -p "${INSTALL_DIR}/bin"

if [ "${BUILD_FROM_SOURCE}" = "1" ]; then
    if [ ! -x "${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-gfortran" ]; then
        echo "ERROR: qcb-xtb env's gfortran missing. Set up that env first" >&2
        exit 1
    fi
    export FC="${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-gfortran"
    export CC="${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
    export PATH="${ENV_PREFIX}/bin:${PATH}"
    export LDFLAGS="-L${ENV_PREFIX}/lib -lquadmath -Wl,-rpath,${ENV_PREFIX}/lib"

    BUILD_DIR="${BUILD_DIR:-/tmp/qcb_crest_build_$USER}"
    rm -rf "${BUILD_DIR}"
    cd "${CREST_DIR}"
    meson setup "${BUILD_DIR}" \
        --buildtype release --optimization 2 \
        --prefix "${INSTALL_DIR}" \
        -Dla_backend=openblas
    ninja -C "${BUILD_DIR}" -j "${NJOBS:-8}"
    ninja -C "${BUILD_DIR}" install
    echo "Source build complete: ${INSTALL_DIR}/bin/crest"
else
    # Conda path: install crest=3.0.2 from conda-forge into qcb-xtb.
    if [ ! -x "${ENV_PREFIX}/bin/conda" ] && ! command -v conda >/dev/null; then
        echo "ERROR: no conda binary found. Either install conda or set " >&2
        echo "       BUILD_FROM_SOURCE=1 to build from the submodule." >&2
        exit 1
    fi
    /home/woodbuse/conda/bin/conda install -n qcb-xtb -c conda-forge \
        crest=3.0.2 -y >&2
    src="${ENV_PREFIX}/bin/crest"
    if [ ! -x "${src}" ]; then
        echo "ERROR: conda install completed but ${src} missing" >&2
        exit 1
    fi
    ln -sf "${src}" "${INSTALL_DIR}/bin/crest"
    echo "CREST 3.0.2 installed via conda-forge → ${INSTALL_DIR}/bin/crest"
fi

# Sanity smoke test: does --version run?
"${INSTALL_DIR}/bin/crest" --version 2>&1 | head -3
