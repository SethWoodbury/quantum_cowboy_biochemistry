#!/bin/bash
#SBATCH --job-name=ts_polish_v2
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=48g
#SBATCH --cpus-per-task=8
#SBATCH --time=01:55:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/ts_polish_v2_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/ts_polish_v2_%j.stderr
#
# v2 polish — robust against the cv-spring frame-explosion bug:
#   - FixInternals over all 36 CA-CA pair distances → rigid-body scaffold
#     (lets cluster translate/rotate as a unit; preserves backbone geometry)
#   - Reactive distances pinned at LITERATURE TS values (Aubert 2004 / Wong 2007):
#       d(P-Onuc) = 2.00 Å,  d(P-Olg) = 2.25 Å,  s = +0.25 (asymmetric concerted)
#   - mace-polar-m / cuda / charge=+1
#   - Output written via template-based PDB writer (preserves atom names + REMARKs)
#   - StructureValidator runs on input AND output; FAILS THE JOB on any
#     critical check (anchor decoupling, bond stretch >1.5×, atom-count mismatch)
set -eu
cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
mkdir -p logs

INPUT=/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/test/results/04_E_final_conf_10.pdb
OUTDIR=/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/ts_pipeline/v2_polish_KCX_set1_conf10
mkdir -p "$OUTDIR"

echo "=== ts_polish_v2  $(date) ==="
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

apptainer exec --nv \
    --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python tools/polish_ts_v2.py \
        --input "$INPUT" \
        --out "$OUTDIR" \
        --model mace-polar-m \
        --device cuda \
        --charge 1 \
        --target-d-p-onuc 2.00 \
        --target-d-p-olg  2.25 \
        --ca-rigid \
        --fmax 0.05 \
        --max-steps 500

echo "=== done $(date) ==="
ls -la "$OUTDIR"
