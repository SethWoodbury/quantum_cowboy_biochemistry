#!/bin/bash
#SBATCH --job-name=bench_scan_c09
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=48g
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/bench_scan_c09_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/bench_scan_c09_%j.stderr
set -eu
cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
INPUT=/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/test/results/04_E_final_conf_09.pdb
OUT=/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/ts_pipeline/benchmark_06h/scan_conf09
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net     --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs"     /net/software/containers/universal.sif     python tools/scan_along_s.py --input "$INPUT" --out "$OUT"         --model mace-polar-m --device cuda --charge 1         --s-min -1.5 --s-max 1.5 --sum-target 4.25 --n-points 11         --fmax 0.05 --max-steps 200
