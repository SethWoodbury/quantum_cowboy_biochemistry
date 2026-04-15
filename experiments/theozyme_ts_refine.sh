#!/bin/bash
#SBATCH --job-name=ts-refine
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=8
#SBATCH --time=03:59:00
#SBATCH --output=logs/ts_refine_%j.stdout
#SBATCH --error=logs/ts_refine_%j.stderr

cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
echo "Started: $(date) | Node: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null

apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/refine_ts_guess.py \
        tests/test_data/protonation/output_protonated.pdb \
        --charge 1 \
        --model mace-omol \
        --fmax-sella 0.01 \
        --outdir outputs/theozyme_ts_refine_omol

echo "Finished: $(date)"
