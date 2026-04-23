#!/bin/bash
#SBATCH --job-name=R3_GLU_set0_irc
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:b4000:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=8
#SBATCH --time=03:59:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/R3_GLU_set0_irc_%j.out
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/R3_GLU_set0_irc_%j.err
cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
echo "R3 Benchmark: R3_GLU_set0_irc | $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net     --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs:/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry"     /net/software/containers/universal.sif     python scripts/qcb ts /home/woodbuse/for/rflatent/original_theozymes/pte/mlff_inputs/PdPTE_GLU_set0_waters__O1nuc_P1_O5lg__netCHG_plus_1.pdb         --outdir /home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs_R3_benchmark/R3_GLU_set0_irc         --model mace-omol         --charge 1         --strategy irc         --fix-preset ca-only         --passthrough --n-images 15 --md-strategy single-shot --md-steps 500             --spring-mode nuc-only --endpoint-method auto --skip-freq

echo "Finished: $(date)"
