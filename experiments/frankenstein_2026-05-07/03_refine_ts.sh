#!/bin/bash
#SBATCH --job-name=frank_refts_2026-05-07
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:1
#SBATCH --constraint="A6000|A100|H200|L40|L40S|A5000"
#SBATCH --mem=128g
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/net/scratch/woodbuse/PTE_slurm_logs/frank2_refts-%j.out
#SBATCH --error=/net/scratch/woodbuse/PTE_slurm_logs/frank2_refts-%j.err
set -euo pipefail

REPO=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
SIF=/net/software/containers/universal.sif
QCB="$REPO/tools/qcb"

NEB_DIR=/net/scratch/woodbuse/frankenstein_AChE_PdPTE_2026-05-07/07_neb
REF_DIR=/net/scratch/woodbuse/frankenstein_AChE_PdPTE_2026-05-07/08_refine_ts
TPL=/net/scratch/woodbuse/frankenstein_AChE_PdPTE_2026-05-07/05_minimize_mace_mp/relaxed.pdb
mkdir -p "$REF_DIR"

echo "==> [FRANK_REFTS] SLURM job=$SLURM_JOB_ID node=$SLURMD_NODENAME"
nvidia-smi -L | head -1
echo "==> Wall start: $(date -Iseconds)"
echo "==> NEB_DIR   = $NEB_DIR"
echo "==> REF_DIR   = $REF_DIR"
echo "==> Reactive atoms (1-based PDB serials): P1=1 O1=4 O5=12"

apptainer exec --nv \
  --bind /home/woodbuse --bind /net/scratch --bind /net/software --bind /net/databases \
  --env "PYTHONPATH=$REPO/deps/mace_polar_src:$REPO/deps/graph_longrange_src:$REPO:$REPO/tools" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  "$SIF" python "$QCB" refine-ts "$TPL" \
  --from-neb "$NEB_DIR" \
  --template-pdb "$TPL" \
  --reactive-atoms 1 4 12 \
  --backend auto \
  --model mace-polar-m --device cuda --charge -2 \
  --outdir "$REF_DIR" \
  --saddle-fmax 0.05 --saddle-max-steps 200 \
  --imag-cm-cutoff -25 --imag-mode-overlap 0.5 --n-imag-expected 1 \
  --fix-preset ca-only --free "chain A" "resid 119 120 121 122 123 124" \
  2>&1 | tee "$REF_DIR/refine_ts_run.log"

echo "==> Wall end: $(date -Iseconds)"
