#!/bin/bash
#SBATCH --job-name=ts_irc_polar_K1c10_h200
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=07:55:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/ts_irc_polar_KCX_set1_conf10_h200_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/ts_irc_polar_KCX_set1_conf10_h200_%j.stderr
#
# H200 (gpu-train) parallel run of the same TS hunt as the a6000 job. First to
# finish wins; the other can be cancelled. Same input, same flags.
#
# qcb ts --strategy irc → Sella saddle (auto-Cartesian, n_free=220) →
# partial-Hessian imag-mode → damped LBFGS IRC both ways → R + TS + P.
set -eu

cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
mkdir -p logs

INPUT=/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/test/results/04_E_final_conf_10.pdb
OUTDIR=/home/woodbuse/for/rflatent/original_theozymes/pte/qcb_runs/ts_irc_polar_KCX_set1_conf10_h200
MODEL="${MODEL_OVERRIDE:-mace-polar-m}"

echo "=== ts_irc_polar_KCX_set1_conf10_h200  $(date) ==="
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true
echo "input:  $INPUT"
echo "outdir: $OUTDIR"
echo "model:  $MODEL"
mkdir -p "$OUTDIR"

apptainer exec --nv \
    --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python tools/qcb ts "$INPUT" \
        --outdir "$OUTDIR" \
        --strategy irc \
        --model "$MODEL" \
        --device cuda \
        --charge 1 \
        --fix-preset ca-only \
        --p-idx 177 --nuc-idx 180 --lg-idx 188

echo "=== done $(date) ==="
ls -la "$OUTDIR" | head -30
