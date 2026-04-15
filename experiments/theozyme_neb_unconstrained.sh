#!/bin/bash
#SBATCH --job-name=theo-neb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=8
#SBATCH --time=05:59:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/theo_neb_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/theo_neb_%j.stderr
cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
echo "Started: $(date) | Theozyme NEB fully unconstrained, MACE-OMOL"

apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net     --env "PYTHONPATH=deps/.local_pkgs"     /net/software/containers/universal.sif     python scripts/run_neb_ts.py         tests/test_data/protonation/output_protonated.pdb         --model mace-omol         --mode standard         --constraint-mode none         --n-images 11         --md-steps 200         --spring-k 3.0 --spring-fmax 3.0         --skip-freq         --outdir outputs/theozyme_neb_omol_unconstrained

echo "Finished: $(date)"
