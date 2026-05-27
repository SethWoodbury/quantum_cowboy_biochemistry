#!/bin/bash
#SBATCH --job-name=bench_scan_s
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=48g
#SBATCH --cpus-per-task=8
#SBATCH --time=03:30:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/bench_scan_s_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/bench_scan_s_%j.stderr
set -eu
cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
mkdir -p logs

INPUT=/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/test/results/04_E_final_conf_10.pdb
OUT=/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/ts_pipeline/benchmark_06h/scan_along_s
mkdir -p "$OUT"

echo "=== bench_scan_s  $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null

apptainer exec --nv \
    --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python tools/scan_along_s.py \
        --input "$INPUT" \
        --out "$OUT" \
        --model mace-polar-m \
        --device cuda \
        --charge 1 \
        --s-min -1.5 --s-max 1.5 --sum-target 4.25 --n-points 11 \
        --fmax 0.05 --max-steps 200

echo "=== done $(date) ==="
ls -la "$OUT"
