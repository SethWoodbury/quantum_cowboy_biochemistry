#!/bin/bash
#SBATCH --job-name=neb_PdPTE_GLU_set0_water
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=8
#SBATCH --time=05:59:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/pte_PdPTE_GLU_set0_waters__O1nuc_P1_O5lg__netCHG_plus_1_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/pte_PdPTE_GLU_set0_waters__O1nuc_P1_O5lg__netCHG_plus_1_%j.stderr
cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
echo "NEB: PdPTE_GLU_set0_waters__O1nuc_P1_O5lg__netCHG_plus_1 | atoms=195 | charge=1 | $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/run_neb_ts.py /home/woodbuse/for/rflatent/original_theozymes/pte/mlff_inputs/PdPTE_GLU_set0_waters__O1nuc_P1_O5lg__netCHG_plus_1.pdb \
        --outdir /home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/PdPTE_GLU_set0_waters__O1nuc_P1_O5lg__netCHG_plus_1 \
        --model mace-omol \
        --mode standard \
        --constraint-mode ca-only \
        --n-images 11 \
        --md-steps 200 \
        --spring-k 3.0 --spring-fmax 3.0 \
        --unidirectional \
        --skip-freq
echo "Finished: $(date)"
