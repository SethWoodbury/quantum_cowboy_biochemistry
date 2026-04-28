#!/usr/bin/env bash
# Build xtb from the deps/xtb submodule using the qcb-xtb conda env's
# gfortran/gcc/meson/ninja toolchain. Runs in-tree (build dir + install
# prefix both inside deps/xtb), so nothing leaks outside the project.
#
# Output binary: deps/xtb/install/bin/xtb
#                deps/xtb/install/lib/libxtb.so
#
# Re-run this script after `git submodule update` to rebuild against new
# upstream xtb.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/home/woodbuse/conda/envs/qcb-xtb}"
XTB_SRC="${ROOT}/deps/xtb"
# Build dir lives on local /tmp because /home is NFS-mounted and the clock
# skew (~1-2 s) between the NFS server and this node trips meson's
# "file from the future" guard. Install dir stays in the repo so the binary
# is checkout-relative.
BUILD_DIR="${BUILD_DIR:-/tmp/qcb_xtb_build_$USER}"
INSTALL_DIR="${XTB_SRC}/install"

if [ ! -x "${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-gfortran" ]; then
    echo "ERROR: gfortran not found in ${ENV_PREFIX}/bin"
    echo "Create the env first:"
    echo "  conda create -n qcb-xtb -c conda-forge gfortran_linux-64 gxx_linux-64 \\"
    echo "      gcc_linux-64 meson ninja cmake openblas-devel lapack-devel ase numpy -y"
    exit 1
fi

if [ ! -d "${XTB_SRC}/src" ]; then
    echo "ERROR: ${XTB_SRC} is empty — run 'git submodule update --init deps/xtb'"
    exit 1
fi

# Conda build envs put compiler symlinks under a triplet name; use full paths
export FC="${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-gfortran"
export CC="${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export PKG_CONFIG_PATH="${ENV_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

# gfortran-15 from conda-forge links libgfortran.so against quadmath but
# meson's auto-detected link command doesn't pull -lquadmath in. Force it.
export LDFLAGS="-L${ENV_PREFIX}/lib -lquadmath -Wl,-rpath,${ENV_PREFIX}/lib"

echo "==> FC = $FC"
echo "==> CC = $CC"
echo "==> meson = $(which meson)"
echo "==> ninja = $(which ninja)"

cd "${XTB_SRC}"

# Wipe stale build dir if present (idempotent)
rm -rf "${BUILD_DIR}"

meson setup "${BUILD_DIR}" \
    --buildtype release \
    --optimization 2 \
    --prefix "${INSTALL_DIR}" \
    -Dlapack=openblas \
    -Dopenmp=true

echo "==> Compiling xtb (this may take 5-15 minutes)…"
ninja -C "${BUILD_DIR}" -j "${NJOBS:-8}"

echo "==> Installing to ${INSTALL_DIR}"
ninja -C "${BUILD_DIR}" install

echo
echo "==> Build complete."
echo "    binary  : ${INSTALL_DIR}/bin/xtb"
echo "    library : ${INSTALL_DIR}/lib/libxtb.so"
echo
"${INSTALL_DIR}/bin/xtb" --version | head -3
