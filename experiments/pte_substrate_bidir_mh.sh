#!/bin/bash
#SBATCH --job-name=sub_bidir_mh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=8
#SBATCH --time=05:59:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/pte_sub_bidir_mh_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/pte_sub_bidir_mh_%j.stderr
cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
echo "Substrate BIDIRECTIONAL NEB (MACE-MH omol) | $(date)"
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/run_neb_ts.py /home/woodbuse/for/rflatent/original_theozymes/pte/mlff_inputs/substrate_uncat_hydrolysis_TS__O1nuc_P1_O5lg__netCHG_minus_1.pdb \
        --outdir /home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/substrate_uncat_bidir_mh \
        --model mace-mh --head omol \
        --mode standard \
        --constraint-mode ca-only \
        --n-images 11 \
        --md-steps 200 \
        --spring-k 3.0 --spring-fmax 3.0 \
        --skip-freq
echo "Finished: $(date)"
