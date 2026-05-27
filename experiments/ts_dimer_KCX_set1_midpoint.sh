#!/bin/bash
#SBATCH --job-name=ts_dimer_K1
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=03:55:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/ts_dimer_K1_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/ts_dimer_K1_%j.stderr
#
# Dimer search from the geometric midpoint of NEB images 5 and 6 (the cv-spring
# band's two highest-energy images, on opposite sides of the s=0 discontinuity).
# Dimer method is gradient-only (no Hessian), robust for >200-atom systems.
# Should find the true saddle near s≈0 (symmetric concerted phosphoryl TS).
set -eu
cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
mkdir -p logs

ROOT=/home/woodbuse/for/rflatent/original_theozymes/pte/qcb_runs/ts_dimer_KCX_set1_midpoint
mkdir -p "$ROOT"

echo "=== ts_dimer_K1  $(date) ==="
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
from ase.io import write, read
from ase.io.trajectory import Trajectory
from ase.constraints import FixAtoms
from quantum_engine.calc import make_calc
from quantum_engine.mlff.dimer import run_dimer_search

ROOT = '/home/woodbuse/for/rflatent/original_theozymes/pte/qcb_runs/ts_dimer_KCX_set1_midpoint'
NEB = '/home/woodbuse/for/rflatent/original_theozymes/pte/qcb_runs/ts_cvspring_KCX_set1_conf10/ts/neb-opt.traj'

# Read final NEB band (last 11 images)
band = list(Trajectory(NEB))[-11:]
img5 = band[5]; img6 = band[6]
P, ON, OL = 177, 180, 188
def s(a):
    return float(np.linalg.norm(a.positions[P]-a.positions[OL]) -
                 np.linalg.norm(a.positions[P]-a.positions[ON]))
print(f'img5: s={s(img5):.3f}   img6: s={s(img6):.3f}')

# Build geometric midpoint
mid = img5.copy()
mid.set_positions(0.5 * (img5.get_positions() + img6.get_positions()))
print(f'midpoint: s={s(mid):.3f}')
print(f'  d(P-Onuc)={np.linalg.norm(mid.positions[P]-mid.positions[ON]):.4f} Å')
print(f'  d(P-Olg) ={np.linalg.norm(mid.positions[P]-mid.positions[OL]):.4f} Å')

# Save the dimer starting geometry
write(f'{ROOT}/dimer_input.pdb', mid)

# Set up calc + CA fix
mid.info['charge'] = 1
calc = make_calc('mace-polar-m', device='cuda', charge=1)
mid.calc = calc

# CA fix (chain A residues 55,57,131,169,201,230,233,254,301; 0-indexed serials 3,22,41,67,89,108,127,142,161)
ca_idx = [3, 22, 41, 67, 89, 108, 127, 142, 161]
mid.set_constraint(FixAtoms(indices=ca_idx))
print(f'fixed {len(ca_idx)} CA atoms')

# initial fmax check
e0 = mid.get_potential_energy()
f0 = mid.get_forces()
print(f'midpoint E = {e0:.4f} eV    fmax = {abs(f0).max():.4f} eV/Å')

# run dimer
ts = run_dimer_search(
    atoms=mid,
    outdir=ROOT,
    fmax=0.05,
    max_steps=500,
    dimer_separation=0.01,
    logfile=f'{ROOT}/dimer.log',
)

write(f'{ROOT}/transition_state.pdb', ts)
e_ts = ts.get_potential_energy()
f_ts = ts.get_forces()
print(f'\nFINAL TS:')
print(f'  E = {e_ts:.4f} eV    fmax = {abs(f_ts).max():.4f} eV/Å')
print(f'  s = {s(ts):.4f}')
print(f'  d(P-Onuc) = {np.linalg.norm(ts.positions[P]-ts.positions[ON]):.4f} Å')
print(f'  d(P-Olg)  = {np.linalg.norm(ts.positions[P]-ts.positions[OL]):.4f} Å')
print(f'  CA max disp from input: {np.linalg.norm(ts.positions[ca_idx]-mid.positions[ca_idx], axis=1).max():.4f} Å (should be ~0)')
PY

echo "=== done $(date) ==="
ls -la "$ROOT"
