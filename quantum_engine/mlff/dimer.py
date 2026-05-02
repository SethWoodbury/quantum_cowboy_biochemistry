"""
Dimer method for transition state search.

The dimer method finds saddle points using only first derivatives (gradients),
making it much cheaper than Hessian-based methods (Sella) for large systems.
It works by translating a "dimer" (two images close together) along the
lowest curvature mode of the potential energy surface.

Best for:
- Systems with >200 atoms where Sella is too expensive
- When you have a TS guess but not clean endpoints
- Refinement after NEB/FSM gives an approximate TS

References:
- Henkelman & Jónsson, J. Chem. Phys. 111, 7010 (1999)
- Improved dimer: Heyden, Bell, Keil, J. Chem. Phys. 123, 224101 (2005)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

log = logging.getLogger("quantum_engine.dimer")


def run_dimer_search(
    atoms: Atoms,
    outdir: str | Path,
    fmax: float = 0.02,
    max_steps: int = 500,
    dimer_separation: float = 0.01,
    logfile: str | None = None,
) -> Atoms:
    """Run the improved dimer method to find a transition state.

    Uses ASE's MinModeAtoms + MinModeTranslate with DimerControl.
    Only needs gradients (no Hessian), so scales well to large systems.

    Args:
        atoms: Starting geometry (should be near a TS)
        outdir: Directory for output files
        fmax: Force convergence criterion (eV/A)
        max_steps: Maximum optimization steps
        dimer_separation: Distance between dimer images (A)
        logfile: Path to log file (None for default)

    Returns:
        Optimized TS structure
    """
    from ase.mep.dimer import DimerControl, MinModeAtoms, MinModeTranslate

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if logfile is None:
        logfile = str(outdir / "dimer.log")

    traj_path = str(outdir / "dimer.traj")

    # Set up dimer control
    dimer_control = DimerControl(
        initial_eigenmode_method="displacement",
        displacement_method="vector",
        dimer_separation=dimer_separation,
        logfile=logfile,
    )

    # Create random initial displacement for the dimer
    # (or could use the imaginary mode from a previous calculation)
    displacement = np.random.randn(len(atoms), 3) * 0.01

    # Set up the dimer atoms object
    dimer_atoms = MinModeAtoms(atoms, dimer_control)
    dimer_atoms.displace(displacement_vector=displacement.ravel())

    # Run the dimer translation (finds the TS)
    e0 = atoms.get_potential_energy()
    log.info(f"Dimer search: {len(atoms)} atoms, E0={e0:.4f} eV")

    optimizer = MinModeTranslate(dimer_atoms, trajectory=traj_path, logfile=logfile)
    optimizer.run(fmax=fmax, steps=max_steps)

    e_ts = atoms.get_potential_energy()
    fmax_final = np.max(np.linalg.norm(atoms.get_forces(), axis=1))
    log.info(f"Dimer converged: E={e_ts:.4f} eV, fmax={fmax_final:.4f} eV/A")

    write(str(outdir / "dimer_ts.xyz"), atoms)

    return atoms


def run_dimer_with_mode(
    atoms: Atoms,
    mode_vector: np.ndarray,
    outdir: str | Path,
    fmax: float = 0.02,
    max_steps: int = 500,
) -> Atoms:
    """Run dimer method with a known initial mode direction.

    Use this when you have the imaginary mode eigenvector from a previous
    frequency calculation or NEB highest-energy image.

    Args:
        atoms: Starting geometry
        mode_vector: Initial guess for the reaction coordinate direction
                     (shape: (N, 3) or (3N,))
        outdir: Directory for output files
        fmax: Force convergence
        max_steps: Max steps
    """
    from ase.mep.dimer import DimerControl, MinModeAtoms, MinModeTranslate

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dimer_control = DimerControl(
        initial_eigenmode_method="displacement",
        displacement_method="vector",
        logfile=str(outdir / "dimer.log"),
    )

    dimer_atoms = MinModeAtoms(atoms, dimer_control)

    # Use the provided mode as the initial dimer displacement
    if mode_vector.ndim == 2:
        mode_vector = mode_vector.ravel()
    mode_vector = mode_vector / np.linalg.norm(mode_vector)
    dimer_atoms.displace(displacement_vector=mode_vector * 0.01)

    optimizer = MinModeTranslate(
        dimer_atoms,
        trajectory=str(outdir / "dimer.traj"),
        logfile=str(outdir / "dimer.log"),
    )
    optimizer.run(fmax=fmax, steps=max_steps)

    write(str(outdir / "dimer_ts.xyz"), atoms)
    return atoms
