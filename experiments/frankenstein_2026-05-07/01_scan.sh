#!/bin/bash
#SBATCH --job-name=frank_scan_2026-05-07
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:1
#SBATCH --constraint="A6000|A100|H200|A5000"
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/net/scratch/woodbuse/PTE_slurm_logs/frank2_scan-%j.out
#SBATCH --error=/net/scratch/woodbuse/PTE_slurm_logs/frank2_scan-%j.err
set -euo pipefail

REPO=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
SIF=/net/software/containers/universal.sif
QCB="$REPO/tools/qcb"

INPUT=/net/scratch/woodbuse/frankenstein_AChE_PdPTE_2026-05-07/05_minimize_mace_mp/relaxed.pdb
OUT=/net/scratch/woodbuse/frankenstein_AChE_PdPTE_2026-05-07/06_scan
mkdir -p "$OUT"

echo "==> [FRANK_SCAN] SLURM job=$SLURM_JOB_ID node=$SLURMD_NODENAME"
nvidia-smi -L | head -1
echo "==> Wall start: $(date -Iseconds)"
echo "==> INPUT     = $INPUT"
echo "==> OUT       = $OUT"
echo "==> CHARGE    = -2 (mace-mp converged at fmax 0.009)"
echo "==> MODEL     = mace-mp (cheap scan; refine-ts uses mace-polar-m)"
echo "==> Coord     = bond P1-O5 (0-based ASE indices: P1=0, O5=11)"
echo "==>             FixBondLength only — no FixInternals overhead"
echo "==> Range     = 1.681 -> 3.0 A in 7 steps (covers reactant -> late-product)"
echo "==> Free res chain B: 119 120 121 122 123 124"

# Use qcb scan with --coord bond (simple FixBondLength, much faster than
# scan_along_s.py's FixInternals(bondcombos) which was 2 min/step on L40S).
# 1-based PDB serials: P1=1, O5=12.
apptainer exec --nv \
  --bind /home/woodbuse --bind /net/scratch --bind /net/software --bind /net/databases \
  --env "PYTHONPATH=$REPO/deps/mace_polar_src:$REPO/deps/graph_longrange_src:$REPO:$REPO/tools" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  "$SIF" python "$QCB" scan "$INPUT" \
  --model mace-mp --device cuda --charge -2 \
  --coord bond --indices 0 11 \
  --start 1.681 --end 3.0 --n-steps 7 \
  --fmax 0.10 \
  --outdir "$OUT" \
  --fix-preset ca-only --free "chain A" "resid 119 120 121 122 123 124" \
  2>&1 | tee "$OUT/scan_run.log"

echo "==> Wall end: $(date -Iseconds)"
