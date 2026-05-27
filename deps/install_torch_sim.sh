#!/usr/bin/env bash
# install_torch_sim.sh — install torch-sim 0.3.0 + its missing runtime
# deps into deps/.local_pkgs (kept out of the apptainer image so we can
# iterate without rebuilding the .sif).
#
# Usage:
#   bash deps/install_torch_sim.sh
#
# Then in code (or via SLURM scripts):
#   PYTHONPATH=<repo>/deps/.local_pkgs:<other paths>
#
# Why pinned to 0.3.0:
#   * 0.4.0+ requires Python 3.12; the QCB container ships Python 3.11.
#   * 0.3.0 is the last release that still publishes a 3.11-compatible
#     wheel. It includes FIRE + gradient_descent, but NOT LBFGS — LBFGS
#     was added in 0.4.x. See docs/optimizer_factory.md for the
#     "torch-sim-lbfgs" stub story.
#
# Why --no-deps:
#   * The container already ships compatible numpy 1.26.4, ase 3.28.0,
#     biotite 1.4.0, torch 2.11+cu130, mace-torch 0.3.15. A bare
#     `pip install torch-sim-atomistic==0.3.0` (no --no-deps) would
#     drag in numpy>=2 + a different ase / pymatgen / etc. and shadow
#     the container's pinned versions, breaking pymatgen and other
#     downstream tools (see slurm-14697934.err for the np.deprecate
#     traceback when this happened during the 2026-05-07 audit).
#   * We hand-pick the only deps that aren't in the container: vesin,
#     vesin-torch (neighbour list), tqdm, h5py, tables.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_PKGS="$REPO_ROOT/deps/.local_pkgs"
CONTAINER="${QCB_CONTAINER:-/net/software/containers/users/woodbuse/quantum_chem/quantum_chem-20260506.sif}"

if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: container not found at $CONTAINER"
    echo "Override with QCB_CONTAINER=<path>"
    exit 1
fi

mkdir -p "$LOCAL_PKGS"

echo "Installing torch-sim 0.3.0 (no deps) into $LOCAL_PKGS"
apptainer exec --nv --bind /home --bind /net "$CONTAINER" \
    pip install --no-cache-dir --target="$LOCAL_PKGS" --no-deps \
        torch-sim-atomistic==0.3.0

echo "Installing missing runtime deps (vesin, tqdm, h5py, tables)"
apptainer exec --nv --bind /home --bind /net "$CONTAINER" \
    pip install --no-cache-dir --target="$LOCAL_PKGS" --no-deps \
        vesin vesin-torch tqdm tables h5py

echo
echo "Smoke test: import torch_sim"
apptainer exec --nv --bind /home --bind /net \
    --env "PYTHONPATH=$LOCAL_PKGS" \
    "$CONTAINER" \
    python -c "
import torch_sim as ts
import torch_sim.runners as r
import torch_sim.optimizers as o
from torch_sim.models.mace import MaceModel
print('torch-sim 0.3.0 OK')
print('  fire available:', hasattr(o, 'fire'))
print('  gradient_descent available:', hasattr(o, 'gradient_descent'))
print('  lbfgs available:', hasattr(o, 'lbfgs'))
"
echo
echo "Done. To use:"
echo "  export PYTHONPATH=\"$LOCAL_PKGS:\$PYTHONPATH\""
echo "  apptainer exec --env PYTHONPATH=$LOCAL_PKGS:... $CONTAINER python ..."
