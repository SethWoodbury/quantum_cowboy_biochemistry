"""Coordinate scan (Gaussian-style opt=modredundant).

Scans a bond, angle, or dihedral from START to END in N steps, relaxing all
other degrees of freedom at each step. Useful for:
- 1D reaction path surveys (before NEB)
- PMF approximations
- Debugging PES features near a TS guess
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixBondLength, FixInternals
from ase.io import write as ase_write
from ase.optimize import LBFGS

log = logging.getLogger("qcb.ops.scan")

EV_TO_KCAL = 23.0609


def run(
    atoms: Atoms,
    calculator=None,
    outdir: str | Path = ".",
    constraint=None,
    coord_type: str = "bond",
    atom_indices: list[int] = None,
    start: float = None,
    end: float = None,
    n_steps: int = 15,
    relax_other: bool = True,
    fmax: float = 0.05,
    max_opt_steps: int = 300,
    **kwargs,
) -> dict:
    """1D coordinate scan.

    Args:
        atoms, calculator, outdir, constraint: standard
        coord_type: "bond" (requires 2 indices), "angle" (3), "dihedral" (4)
        atom_indices: atom indices defining the coordinate
        start: initial value of the coordinate (Å for bond, degrees for angle/dihedral)
        end: final value
        n_steps: number of scan points (inclusive of start and end)
        relax_other: if True, at each scan point relax all atoms except the
                     scan coordinate (constrained). If False, single-point only.
        fmax: relaxation convergence threshold
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if atoms.calc is None:
        if calculator is None:
            raise ValueError("Need a calculator")
        atoms.calc = calculator

    if coord_type not in ("bond", "angle", "dihedral"):
        raise ValueError(f"Unknown coord_type '{coord_type}'")

    expected_n = {"bond": 2, "angle": 3, "dihedral": 4}[coord_type]
    if atom_indices is None or len(atom_indices) != expected_n:
        raise ValueError(f"{coord_type} scan needs {expected_n} atom indices, got {atom_indices}")

    if start is None or end is None:
        raise ValueError("Must specify start and end")

    values = np.linspace(start, end, n_steps)
    log.info(f"Scan: {coord_type} {atom_indices} from {start} to {end}, {n_steps} steps")

    # Initial constraints from caller (e.g., CA fix)
    base_constraints = []
    if constraint is not None:
        base_constraints = constraint if isinstance(constraint, list) else [constraint]

    frames = []
    energies = []
    relax_steps = []

    for step_idx, target in enumerate(values):
        # Set the coordinate
        if coord_type == "bond":
            # fix=0.5 moves both atoms symmetrically about midpoint
            # (fix=0 would move atom 0 only → translates whole molecule)
            atoms.set_distance(atom_indices[0], atom_indices[1], target, fix=0.5)
            scan_con = FixBondLength(atom_indices[0], atom_indices[1])
        elif coord_type == "angle":
            atoms.set_angle(atom_indices[0], atom_indices[1], atom_indices[2], target)
            # FixInternals for angle constraint
            scan_con = FixInternals(angles_deg=[[target, [atom_indices[0], atom_indices[1], atom_indices[2]]]])
        else:  # dihedral
            atoms.set_dihedral(atom_indices[0], atom_indices[1], atom_indices[2], atom_indices[3], target)
            scan_con = FixInternals(dihedrals_deg=[[target, list(atom_indices)]])

        # Set combined constraints
        atoms.set_constraint(base_constraints + [scan_con])

        n_opt = 0
        if relax_other:
            log_path = outdir / f"opt-step{step_idx:03d}.log"
            opt = LBFGS(atoms, logfile=str(log_path))
            opt.run(fmax=fmax, steps=max_opt_steps)
            n_opt = opt.nsteps if hasattr(opt, "nsteps") else -1

        e = float(atoms.get_potential_energy())
        energies.append(e)
        relax_steps.append(n_opt)
        frame = atoms.copy()
        frame.info["scan_value"] = float(target)
        frame.info["energy_eV"] = e
        frames.append(frame)

        log.info(f"  Step {step_idx+1}/{n_steps}: {coord_type} = {target:.3f}, "
                 f"E = {e:.6f} eV, relax_steps = {n_opt}")

    # Restore base constraints
    atoms.set_constraint(base_constraints)

    # Outputs
    energies = np.array(energies)
    rel_energies_kcal = (energies - energies.min()) * EV_TO_KCAL

    csv_path = outdir / "scan.csv"
    with open(csv_path, "w") as f:
        f.write("step,coord_value,energy_eV,energy_rel_kcal,relax_steps\n")
        for i, (v, e, rel, n) in enumerate(zip(values, energies, rel_energies_kcal, relax_steps)):
            f.write(f"{i},{v:.5f},{e:.8f},{rel:.4f},{n}\n")

    xyz_path = outdir / "scan-trajectory.xyz"
    ase_write(str(xyz_path), frames, format="extxyz")

    # Plot
    png_path = None
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(values, rel_energies_kcal, "o-", markersize=6)
        unit = "Å" if coord_type == "bond" else "degrees"
        ax.set_xlabel(f"{coord_type} ({unit})")
        ax.set_ylabel("Relative energy (kcal/mol)")
        ax.set_title(f"Scan: {coord_type} {atom_indices}")
        ax.grid(alpha=0.3)
        png_path = outdir / "scan.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        log.warning(f"  Plot failed: {e}")

    max_idx = int(np.argmax(energies))
    barrier_kcal = float(rel_energies_kcal[max_idx])

    result = {
        "status": "completed",
        "atoms": atoms,
        "coord_values": values.tolist(),
        "energies_eV": energies.tolist(),
        "relative_energies_kcal": rel_energies_kcal.tolist(),
        "max_energy_idx": max_idx,
        "max_coord_value": float(values[max_idx]),
        "barrier_kcal": barrier_kcal,
        "outputs": {
            "csv": str(csv_path),
            "trajectory": str(xyz_path),
            "plot": str(png_path) if png_path else None,
        },
    }

    (outdir / "scan-summary.json").write_text(json.dumps({
        "coord_type": coord_type, "atom_indices": list(atom_indices),
        "start": start, "end": end, "n_steps": n_steps,
        "barrier_kcal": barrier_kcal, "max_coord_value": result["max_coord_value"],
    }, indent=2))
    result["outputs"]["summary"] = str(outdir / "scan-summary.json")

    log.info(f"Scan done: barrier = {barrier_kcal:.2f} kcal/mol at {coord_type} = {values[max_idx]:.3f}")
    return result
