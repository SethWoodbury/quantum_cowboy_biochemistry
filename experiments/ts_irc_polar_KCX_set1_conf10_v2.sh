#!/bin/bash
#SBATCH --job-name=ts_polar_K1c10_v2
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=48g
#SBATCH --cpus-per-task=8
#SBATCH --time=07:55:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/ts_polar_K1c10_v2_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/ts_polar_K1c10_v2_%j.stderr
#
# v2: pre-relax with mace-polar-m to fmax=0.10 BEFORE Sella saddle. The
# xtb-GFN2-optimized input had fmax=3.25 eV/Å at the mace-polar-m level,
# which made Sella's first step prohibitively expensive (Step 0 stalled
# for 42 min on a6000 in v1). Pre-relax brings the geometry close to
# mace's local minimum so Sella has a clean starting point. CAs frozen
# throughout via $fix; reactive distances NOT restrained during pre-opt
# so the substrate can settle into mace's preferred coordination.
set -eu

cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
mkdir -p logs

INPUT=/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/test/results/04_E_final_conf_10.pdb
ROOT=/home/woodbuse/for/rflatent/original_theozymes/pte/qcb_runs/ts_polar_K1c10_v2
PRE=$ROOT/01_pre_relax
TS=$ROOT/02_ts_irc
mkdir -p "$PRE" "$TS"

echo "=== ts_polar_K1c10_v2  $(date) ==="
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Step 1 — relax with mace-polar-m, CAs frozen, no reactive-distance restraint
echo ">>> Step 1: pre-relax with mace-polar-m"
apptainer exec --nv \
    --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python tools/qcb opt "$INPUT" \
        --outdir "$PRE" \
        --model mace-polar-m \
        --device cuda \
        --charge 1 \
        --fix-preset ca-only \
        --optimizer lbfgs \
        --fmax 0.10 \
        --max-steps 200 \
        --output-pdb "$PRE/relaxed.pdb"

if [ ! -f "$PRE/relaxed.pdb" ]; then
    echo "ERROR: pre-relax did not produce $PRE/relaxed.pdb"; exit 1
fi
echo "<<< Step 1 done"

# Step 2 — qcb ts --strategy irc on the pre-relaxed geometry
echo ">>> Step 2: qcb ts --strategy irc"
apptainer exec --nv \
    --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python tools/qcb ts "$PRE/relaxed.pdb" \
        --outdir "$TS" \
        --strategy irc \
        --model mace-polar-m \
        --device cuda \
        --charge 1 \
        --fix-preset ca-only \
        --p-idx 177 --nuc-idx 180 --lg-idx 188

echo "=== done $(date) ==="
ls -la "$TS"
