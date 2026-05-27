#!/bin/bash
#SBATCH --job-name=ts_freq_irc
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=48g
#SBATCH --cpus-per-task=8
#SBATCH --time=03:55:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/ts_freq_irc_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/ts_freq_irc_%j.stderr
#
# Closes the universal TS-validation loop on the polished TS:
#   (1) qcb freq — partial Hessian on reactive core; n_imaginary should equal 1,
#                  imaginary mode should overlap with d(P-Olg)−d(P-Onuc)
#   (2) qcb irc  — descent from the TS produces reactant + product PDBs and
#                  the IRC connectivity check (do reactant/product match what
#                  the user expected?)
#
# Reactive core for partial Hessian (0-indexed): P, Onuc, Olg + connected SUB Os
#   P1   = 177    O3 OHX = 180 (nucleophile)
#   O7   = 188 (leaving group)   O4 = 182   O5 = 184   O6 = 186
set -eu
cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
mkdir -p logs

TS_PDB=/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/ts_pipeline/v2_polish_KCX_set1_conf10/transition_state.pdb
ROOT=/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/ts_pipeline/v2_validation
mkdir -p "$ROOT"

echo "=== ts_freq_irc  $(date) ==="
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# ----- (1) FREQ — partial Hessian on the reactive core -----
echo ">>> Stage 1: qcb freq (partial Hessian on reactive core)"
apptainer exec --nv \
    --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python tools/qcb freq "$TS_PDB" \
        --outdir "$ROOT/freq" \
        --model mace-polar-m \
        --device cuda \
        --charge 1 \
        --indices 177 178 179 180 181 182 183 184 185 186 187 188 189 \
        --delta 0.02 \
        --method central
echo "<<< Stage 1 done"

# ----- (2) IRC — descent both ways from the polished TS -----
echo ">>> Stage 2: qcb irc (no Sella refinement; descent only)"
apptainer exec --nv \
    --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python tools/qcb irc "$TS_PDB" \
        --outdir "$ROOT/irc" \
        --model mace-polar-m \
        --device cuda \
        --charge 1 \
        --no-refine-ts \
        --step 0.1 --fmax 0.05
echo "<<< Stage 2 done"

echo "=== done $(date) ==="
ls -la "$ROOT" "$ROOT/freq" "$ROOT/irc" 2>/dev/null
