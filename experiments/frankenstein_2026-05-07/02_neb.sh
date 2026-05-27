#!/bin/bash
#SBATCH --job-name=frank_neb_2026-05-07
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:1
#SBATCH --constraint="A6000|A100|H200|L40|L40S|A5000"
#SBATCH --mem=128g
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/net/scratch/woodbuse/PTE_slurm_logs/frank2_neb-%j.out
#SBATCH --error=/net/scratch/woodbuse/PTE_slurm_logs/frank2_neb-%j.err
set -euo pipefail

REPO=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
SIF=/net/software/containers/universal.sif
QCB="$REPO/tools/qcb"

SCAN_DIR=/net/scratch/woodbuse/frankenstein_AChE_PdPTE_2026-05-07/06_scan
NEB_DIR=/net/scratch/woodbuse/frankenstein_AChE_PdPTE_2026-05-07/07_neb
TPL=/net/scratch/woodbuse/frankenstein_AChE_PdPTE_2026-05-07/05_minimize_mace_mp/relaxed.pdb
mkdir -p "$NEB_DIR"

echo "==> [FRANK_NEB] SLURM job=$SLURM_JOB_ID node=$SLURMD_NODENAME"
nvidia-smi -L | head -1
echo "==> Wall start: $(date -Iseconds)"
echo "==> SCAN_DIR  = $SCAN_DIR"
echo "==> NEB_DIR   = $NEB_DIR"

# Extract first and last frames from scan-trajectory.xyz as PDB endpoints.
# qcb scan writes ASE extxyz; we need PDB so the NEB constraint parsers
# (which need biotite annotations: chain, residue) can work.
TRAJ="$SCAN_DIR/scan-trajectory.xyz"
if [ ! -f "$TRAJ" ]; then
  echo "ERROR: scan trajectory not found at $TRAJ" >&2
  ls -la "$SCAN_DIR" || true
  exit 2
fi

REACT="$NEB_DIR/_endpoint_reactant.pdb"
PROD="$NEB_DIR/_endpoint_product.pdb"

apptainer exec \
  --bind /home/woodbuse --bind /net/scratch --bind /net/software --bind /net/databases \
  --env "PYTHONPATH=$REPO/deps/mace_polar_src:$REPO/deps/graph_longrange_src:$REPO:$REPO/tools" \
  "$SIF" python -c "
from ase.io import read
from quantum_engine.io import load_structure, write_pdb
import csv

frames = read('$TRAJ', ':')
print('scan frames:', len(frames))

# Use the PDB template for chain/residue annotations
atoms_R, bt_R, _ = load_structure('$TPL')
atoms_P, bt_P, _ = load_structure('$TPL')

# Substitute coordinates from first / last scan frames
atoms_R.set_positions(frames[0].get_positions())
atoms_P.set_positions(frames[-1].get_positions())

write_pdb(atoms_R, bt_R, '$REACT', total_charge=-2)
write_pdb(atoms_P, bt_P, '$PROD',  total_charge=-2)
print('wrote', '$REACT', '$PROD')

# Print summary of energies at endpoints
with open('$SCAN_DIR/scan.csv') as f:
    rows = list(csv.DictReader(f))
print('Scan profile (step, coord, E_rel_kcal):')
for r in rows:
    print('  ', r['step'], r['coord_value'], r['energy_rel_kcal'])
"

echo "==> Reactant: $REACT"
echo "==> Product:  $PROD"

apptainer exec --nv \
  --bind /home/woodbuse --bind /net/scratch --bind /net/software --bind /net/databases \
  --env "PYTHONPATH=$REPO/deps/mace_polar_src:$REPO/deps/graph_longrange_src:$REPO:$REPO/tools" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  "$SIF" python "$QCB" neb "$REACT" "$PROD" \
  --model mace-polar-m --device cuda --charge -2 \
  --outdir "$NEB_DIR" \
  --n-images 11 \
  --interpolation geodesic \
  --optimizer fire --max-step 0.10 \
  --fmax-noclimb 0.50 --steps-noclimb 80 \
  --fmax-climb 0.15 --steps-climb 80 \
  --ts-tol-fmax 0.15 \
  --key-bond "P1:YYL,O1:YYL" --key-bond "P1:YYL,O5:YYL" \
  --fix-preset ca-only --free "chain A" "resid 119 120 121 122 123 124" \
  --save-trajectory \
  2>&1 | tee "$NEB_DIR/neb_run.log"

echo "==> Wall end: $(date -Iseconds)"
