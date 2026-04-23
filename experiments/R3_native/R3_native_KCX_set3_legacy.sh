#!/bin/bash
#SBATCH --job-name=R3_native_KCX_set3
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:b4000:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=8
#SBATCH --time=03:59:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/R3_native_KCX_set3_%j.out
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/R3_native_KCX_set3_%j.err
cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
echo "R3 Native: KCX_set3 legacy | $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

# Native qcb ts (NO subprocess, NO charge-state bug)
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs:/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry" \
    /net/software/containers/universal.sif \
    python scripts/qcb ts /home/woodbuse/for/rflatent/original_theozymes/pte/mlff_inputs/PdPTE_KCX_set3_waters__O3nuc_P1_O7lg__netCHG_minus_1.pdb \
        --outdir /home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs_R3_native/KCX_set3_cvspring \
        --model mace-omol \
        --charge -1 \
        --strategy cv-spring \
        --fix-preset ca-only \
        --n-images 15 \
        --interpolation geodesic

echo "Finished: $(date)"
