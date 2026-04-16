#!/bin/bash
#SBATCH --job-name=frank_s4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=8
#SBATCH --time=05:59:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/frank_s4_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/frank_s4_%j.stderr
cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
echo "Strategy 4: MACE-MH omol head + CA-only chainB + pre-relax | $(date)"
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net     --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs"     /net/software/containers/universal.sif     python scripts/run_neb_ts.py data/examples/frankenstein_ACHE_PTE/input.pdb         --outdir outputs/frank_s4_mh_omol         --model mace-mh --head omol --mode standard         --constraint-mode ca-only --fix-chains B         --n-images 11 --md-steps 300         --spring-k 3.0 --spring-fmax 3.0         --pre-relax --skip-freq
echo "Finished: $(date)"
