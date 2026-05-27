#!/bin/bash
#SBATCH --job-name=ts_polish_K1
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=48g
#SBATCH --cpus-per-task=8
#SBATCH --time=01:55:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/ts_polish_K1_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/ts_polish_K1_%j.stderr
#
# Constrained polish of the symmetric-concerted TS midpoint:
#   - $fix CAs at source positions (9 atoms)
#   - Pin d(P,Onuc) ≈ 2.19 Å and d(P,Olg) ≈ 2.25 Å (saddle-ridge values)
#   - Relax everything else with mace-polar-m to fmax=0.05
# Output: a clean TS-quality structure for enzyme design.
set -eu
cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
mkdir -p logs

OUT=/home/woodbuse/for/rflatent/original_theozymes/pte/qcb_runs/ts_FINAL/polished_midpoint
mkdir -p "$OUT"

echo "=== ts_polish_K1  $(date) ==="
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

apptainer exec --nv \
    --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python << 'PY'
import sys, os
sys.path.insert(0, '/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry')
import numpy as np
from ase.io import read, write
from ase.constraints import FixAtoms, FixBondLengths
from ase.optimize import LBFGS
from quantum_engine.calc import make_calc

OUT = '/home/woodbuse/for/rflatent/original_theozymes/pte/qcb_runs/ts_FINAL/polished_midpoint'
INPUT = '/home/woodbuse/for/rflatent/original_theozymes/pte/qcb_runs/ts_FINAL/TS_symmetric_midpoint.pdb'

P, ON, OL = 177, 180, 188
ca_idx = [3, 22, 41, 67, 89, 108, 127, 142, 161]

a = read(INPUT)
print(f'loaded {len(a)} atoms')
print(f'starting d(P-Onuc) = {np.linalg.norm(a.positions[P]-a.positions[ON]):.4f} Å')
print(f'starting d(P-Olg)  = {np.linalg.norm(a.positions[P]-a.positions[OL]):.4f} Å')

a.info['charge'] = 1
a.calc = make_calc('mace-polar-m', device='cuda', charge=1)

# Compose constraints: CAs fixed + two bond-length constraints pinned at midpoint values
a.set_constraint([
    FixAtoms(indices=ca_idx),
    FixBondLengths([(P, ON), (P, OL)]),
])
print(f'constraints: FixAtoms (9 CAs) + FixBondLengths (P-Onuc, P-Olg)')

e0 = a.get_potential_energy()
f0 = a.get_forces()
print(f'\ninitial: E = {e0:.4f} eV    fmax = {abs(f0).max():.4f} eV/Å')

opt = LBFGS(a, trajectory=f'{OUT}/polish.traj', logfile=f'{OUT}/polish.log')
opt.run(fmax=0.05, steps=300)

ef = a.get_potential_energy()
ff = a.get_forces()
print(f'\nfinal: E = {ef:.4f} eV    fmax = {abs(ff).max():.4f} eV/Å')
print(f'final d(P-Onuc) = {np.linalg.norm(a.positions[P]-a.positions[ON]):.4f} Å')
print(f'final d(P-Olg)  = {np.linalg.norm(a.positions[P]-a.positions[OL]):.4f} Å')

write(f'{OUT}/transition_state_polished.pdb', a)
print(f'\nwrote: {OUT}/transition_state_polished.pdb')
PY

echo "=== done $(date) ==="
ls -la "$OUT"
