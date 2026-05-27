#!/bin/bash
#SBATCH --job-name=ts_cv_K1c10
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=07:55:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/ts_cv_K1c10_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/ts_cv_K1c10_%j.stderr
#
# v3 — qcb ts --strategy cv-spring on conf_10. The previous direct-Sella
# attempt (v1) stalled because the input fmax was 3.25 eV/Å at the mace-polar-m
# level: Sella's RFO step proposals couldn't find a productive uphill direction
# from that far out, so it defaulted to gradient-following and slid toward
# product (s=+0.63 → s=+1.28).
#
# cv-spring strategy:
#   1. opt to s_reactant = -2.5 (pulls Onuc away, P-Olg intact bond)
#   2. opt to s_product  = +2.5 (Onuc bonded, P-Olg cleaved)
#   3. NEB+CI between the two; CI image converges to a low-fmax saddle
#   4. Sella refines from that CI image — clean starting point this time
# CV: s = d(P,Olg) - d(P,Onuc) using BondDifferenceCVSpring.
#
# Reactive atoms (0-based, into 04_E_final_conf_10.pdb):
#   P1 = 177  (SUB)   Onuc = 180 (OHX:O3)   Olg = 188 (SUB:O7)
set -eu
cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
mkdir -p logs

INPUT=/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/test/results/04_E_final_conf_10.pdb
OUTDIR=/home/woodbuse/for/rflatent/original_theozymes/pte/qcb_runs/ts_cvspring_KCX_set1_conf10
mkdir -p "$OUTDIR"

echo "=== ts_cv_K1c10  $(date) ==="
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

apptainer exec --nv \
    --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python tools/qcb ts "$INPUT" \
        --outdir "$OUTDIR" \
        --strategy cv-spring \
        --model mace-polar-m \
        --device cuda \
        --charge 1 \
        --fix-preset ca-only \
        --p-idx 177 --nuc-idx 180 --lg-idx 188 \
        --cv-s-reactant -2.5 \
        --cv-s-product   2.5 \
        --n-images 11 \
        --interpolation geodesic

echo "=== done $(date) ==="
ls -la "$OUTDIR"
