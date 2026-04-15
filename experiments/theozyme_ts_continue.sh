#!/bin/bash
#SBATCH --job-name=ts-continue
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=8
#SBATCH --time=05:59:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/ts_continue_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/ts_continue_%j.stderr
cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
echo "Started: $(date) | Continuing TS refinement from partial"

# Use the partial TS as input — already near the saddle point
apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net     --env "PYTHONPATH=deps/.local_pkgs"     /net/software/containers/universal.sif     python scripts/refine_ts_guess.py         outputs/theozyme_ts_direct_sella_omol/transition_state_partial.xyz         --charge 1         --model mace-omol         --fmax-sella 0.01         --skip-prerelax         --outdir outputs/theozyme_ts_continue_omol

echo "Finished: $(date)"
