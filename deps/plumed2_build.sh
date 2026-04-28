#!/bin/bash
# Build PLUMED2 from the deps/plumed2 git submodule.
#
# Notes:
# - We build with --disable-mpi (single-replica MTD/OPES is what we need;
#   replica-exchange is the only feature lost). Multi-walker via shared
#   filesystem still works without MPI.
# - The resulting libplumedKernel.so is read-only-shareable across users
#   on the cluster (see qcb/config.py PLUMED_KERNEL).
# - Build time on a login node: ~15-20 min.
# - Resulting install dir: ~500 MB.
#
# Usage:
#   bash deps/plumed2_build.sh
#   source deps/plumed2/install/setup.sh   # then in your shell
#
# Or just consume the prebuilt one at:
#   /net/scratch/woodbuse/metad/plumed/lib/libplumedKernel.so
# (qcb/config.py auto-detects this.)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PLUMED_DIR="$SCRIPT_DIR/plumed2"

if [ ! -d "$PLUMED_DIR" ]; then
    echo "ERROR: $PLUMED_DIR not found. Run: git submodule update --init deps/plumed2"
    exit 1
fi

cd "$PLUMED_DIR"

INSTALL_DIR="$PLUMED_DIR/install"

echo "Configuring PLUMED 2 (no MPI, all modules, Python bindings)..."
./configure \
    --prefix="$INSTALL_DIR" \
    --disable-mpi \
    --enable-modules=all \
    --enable-python \
    CXX="${CXX:-g++}" \
    CC="${CC:-gcc}"

echo "Building (parallel, $(nproc) cores)..."
make -j "$(nproc)"

echo "Installing to $INSTALL_DIR..."
make install

# Make world-readable for shared use across cluster users
chmod -R a+rX "$INSTALL_DIR"

# Write a setup.sh users source before running
cat > "$INSTALL_DIR/setup.sh" <<EOF
# Source this file to use PLUMED 2 from the qcb submodule build.
PLUMED_ROOT="$INSTALL_DIR"
export PATH="\$PLUMED_ROOT/bin:\$PATH"
export LD_LIBRARY_PATH="\$PLUMED_ROOT/lib:\$LD_LIBRARY_PATH"
export PLUMED_KERNEL="\$PLUMED_ROOT/lib/libplumedKernel.so"
export PYTHONPATH="\$PLUMED_ROOT/lib/plumed/python:\$PYTHONPATH"
EOF
chmod a+r "$INSTALL_DIR/setup.sh"

echo ""
echo "=========================================="
echo "PLUMED 2 build complete."
echo "Kernel: $INSTALL_DIR/lib/libplumedKernel.so"
echo "Setup:  source $INSTALL_DIR/setup.sh"
echo "=========================================="
