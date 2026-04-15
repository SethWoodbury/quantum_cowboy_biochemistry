#!/bin/bash
#SBATCH --job-name=frank_v2_ca
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=8
#SBATCH --time=03:59:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/frank_v2_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/frank_v2_%j.stderr
cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
echo "Frank V2: CA-only fixed | Started: $(date)"
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net     --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs"     /net/software/containers/universal.sif     python scripts/run_neb_ts.py data/examples/frankenstein_ACHE_PTE/input.pdb         --outdir outputs/frank_v2_caonly         --model mace-mp --mode standard --constraint-mode ca-only         --n-images 15 --md-steps 300 --spring-k 3.0 --spring-fmax 3.0         --pre-relax --skip-freq
echo "Finished: $(date)"
