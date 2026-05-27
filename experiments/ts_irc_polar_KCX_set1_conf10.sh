#!/bin/bash
#SBATCH --job-name=ts_irc_polar_K1c10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32g
#SBATCH --cpus-per-task=8
#SBATCH --time=07:55:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/ts_irc_polar_KCX_set1_conf10_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/ts_irc_polar_KCX_set1_conf10_%j.stderr
#
# First-pass TS hunt for woodbuse — qcb ts --strategy irc on the top conformer
# from the crest_funnel pipeline. Sella saddle (auto-Cartesian since n_free=220
# > 200) → partial-Hessian imaginary mode → damped LBFGS IRC both ways →
# reactant + TS + product.
#
# Input: /home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/test/results/04_E_final_conf_10.pdb
#   229 atoms, charge +1, CAs frozen, P-Onuc=1.66 Å, P-Olg=2.29 Å (5-coord TS)
#
# Reactive-atom indices (from partition.json, no_waters 1-based; same for
# full system since waters are at the end):
#   P1   = 178 (1-based) / 177 (0-based)
#   Onuc = 181 (1-based) / 180 (0-based)
#   Olg  = 189 (1-based) / 188 (0-based)
set -eu

cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
mkdir -p logs

INPUT=/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/test/results/04_E_final_conf_10.pdb
OUTDIR=/home/woodbuse/for/rflatent/original_theozymes/pte/qcb_runs/ts_irc_polar_KCX_set1_conf10
MODEL="${MODEL_OVERRIDE:-mace-polar-m}"   # caller can override via env (e.g. mace-omol)

echo "=== ts_irc_polar_KCX_set1_conf10  $(date) ==="
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
