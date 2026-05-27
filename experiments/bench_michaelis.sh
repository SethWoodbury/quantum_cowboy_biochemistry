#!/bin/bash
#SBATCH --job-name=bench_mc
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=48g
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/bench_mc_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/bench_mc_%j.stderr
set -eu
cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
mkdir -p logs

INPUT=/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/test/results/04_E_final_conf_10.pdb
OUT=/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/ts_pipeline/benchmark_06h/michaelis_complex

echo "=== bench_mc  $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null

apptainer exec --nv \
    --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python tools/find_michaelis_complex.py \
        --input "$INPUT" \
        --out "$OUT" \
        --model mace-polar-m \
        --device cuda \
        --charge 1 \
        --initial-d-p-onuc 4.5 \
        --target-d-p-olg 1.65 \
        --fmax 0.05 \
        --max-steps 600

echo "=== done $(date) ==="
ls -la "$OUT"
