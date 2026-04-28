#!/bin/bash
#SBATCH --job-name=R2_PdPTE_KCX_set3_water
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=8
#SBATCH --time=11:59:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/R2_PdPTE_KCX_set3_waters__O3nuc_P1_O7lg__netCHG_minus_1_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/R2_PdPTE_KCX_set3_waters__O3nuc_P1_O7lg__netCHG_minus_1_%j.stderr
cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
echo "R2 Gold Standard: PdPTE_KCX_set3_waters__O3nuc_P1_O7lg__netCHG_minus_1 | atoms=293 | charge=-1 | $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null

apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/run_neb_ts.py /home/woodbuse/for/rflatent/original_theozymes/pte/mlff_inputs/PdPTE_KCX_set3_waters__O3nuc_P1_O7lg__netCHG_minus_1.pdb \
        --outdir /home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs_R2/PdPTE_KCX_set3_waters__O3nuc_P1_O7lg__netCHG_minus_1 \
        --model mace-omol  \
        --charge -1 \
        --mode standard \
        --constraint-mode ca-only \
        --n-images 31 \
        --spring-mode nuc-only \
        --endpoint-method auto \
        --md-strategy multi-seed \
        --n-md-seeds 5 \
        --md-steps 1000 \
        --spring-k 3.0 --spring-fmax 3.0 \
        --charge-method auto \
        --skip-freq

echo "Finished: $(date)"
