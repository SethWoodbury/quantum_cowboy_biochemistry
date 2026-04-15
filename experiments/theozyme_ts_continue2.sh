#!/bin/bash
#SBATCH --job-name=ts-cont2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=8
#SBATCH --time=05:59:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/ts_cont2_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/ts_cont2_%j.stderr
cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
echo "Started: $(date) | TS direct Sella (6hr, from original PDB)"

# Use the ORIGINAL protonated PDB as input, skip prerelax, 
# Sella will start from scratch but now we know it needs 500+ steps
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net     --env "PYTHONPATH=deps/.local_pkgs"     /net/software/containers/universal.sif     python scripts/refine_ts_guess.py         tests/test_data/protonation/output_protonated.pdb         --charge 1         --model mace-omol         --fmax-sella 0.02         --skip-prerelax         --outdir outputs/theozyme_ts_omol_6hr

echo "Finished: $(date)"
