#!/bin/bash
#SBATCH --job-name=sample_KCX1_noF132
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=24g
#SBATCH --cpus-per-task=8
#SBATCH --time=03:59:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/sample_KCX1_noF132_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/sample_KCX1_noF132_%j.stderr

set -euo pipefail

cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
mkdir -p logs

INPUT=/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_inputs/without_F132/PdPTE_KCX_set1_waters__O3nuc_P1_O7lg__netCHG_plus_1.pdb
OUTDIR=/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_sampling/without_F132/PdPTE_KCX_set1_mh_omol_restrained

echo "Cheap TS-starter sampling: PdPTE_KCX_set1 without F132 | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs:/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry" \
    /net/software/containers/universal.sif \
    python tools/sample_ts_starters.py "$INPUT" \
        --outdir "$OUTDIR" \
        --model mace-mh --head omol \
        --charge 1 \
        --device cuda --dtype float32 \
        --seeds 8 \
        --md-ps 1.5 \
        --timestep-fs 0.5 \
        --sample-every-fs 75 \
        --temp 300 \
        --anneal-peak 450 \
        --friction 5.0 \
        --reactive-k 10.0 \
        --reactive-fmax 5.0 \
        --ligand-heavy-anchor-k 0.03 \
        --water-o-anchor-k 0.02 \
        --pre-opt-steps 50 \
        --pre-opt-fmax 0.2 \
        --keep-frames 12 \
        --polish-steps 150 \
        --polish-fmax 0.05

echo "Finished: $(date)"
echo "Ranked PDBs: $OUTDIR/ranked_pdbs"
echo "Summary:     $OUTDIR/summary.csv"
