#!/bin/bash
#SBATCH --job-name=bench_val_v3
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=48g
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/bench_val_v3_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/bench_val_v3_%j.stderr
set -eu
cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
TS=/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/ts_pipeline/v3_FINAL/TS.pdb
OUT=/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/ts_pipeline/v3_FINAL/validation
mkdir -p "$OUT/freq" "$OUT/irc"

apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python tools/qcb freq "$TS" \
        --outdir "$OUT/freq" --model mace-polar-m --device cuda --charge 1 \
        --indices 177 178 179 180 181 182 183 184 185 186 187 188 189 \
        --delta 0.02 --method central

apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python tools/qcb irc "$TS" \
        --outdir "$OUT/irc" --model mace-polar-m --device cuda --charge 1 \
        --no-refine-ts --step 0.1 --fmax 0.05 --max-steps 400
