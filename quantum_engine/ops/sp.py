"""Single-point energy and forces."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from ase import Atoms

log = logging.getLogger("quantum_engine.ops.sp")

EV_TO_KCAL = 23.0609


def run(
    atoms: Atoms,
    calculator=None,
    outdir: str | Path = ".",
    constraint=None,
    **kwargs,
) -> dict:
    """Single-point energy and forces.

    Args:
        atoms: ASE Atoms (caller may have set atoms.calc already)
        calculator: ASE calculator. If atoms.calc is None, assigned here.
        outdir: directory for output files
        constraint: ignored (single-point doesn't apply constraints), but accepted
                    for interface consistency
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if atoms.calc is None:
        if calculator is None:
            raise ValueError("Need a calculator (either atoms.calc or calculator arg)")
        atoms.calc = calculator

    log.info(f"Single-point: {len(atoms)} atoms")
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    fmax = float(np.linalg.norm(forces, axis=1).max())

    # Try optional properties
    try:
        dipole = atoms.get_dipole_moment()
    except Exception:
        dipole = None
    try:
        charges = atoms.get_charges()
    except Exception:
        charges = None

    result = {
        "status": "converged",
        "energy_eV": float(energy),
        "energy_kcal_mol": float(energy * EV_TO_KCAL),
        "atoms": atoms,
        "fmax_eV_per_A": fmax,
        "dipole_debye": dipole.tolist() if dipole is not None else None,
        "charges": charges.tolist() if charges is not None else None,
        "outputs": {},
    }

    # Persist to JSON for later inspection
    summary_path = outdir / "sp.json"
    json_safe = {
        "energy_eV": result["energy_eV"],
        "energy_kcal_mol": result["energy_kcal_mol"],
        "fmax_eV_per_A": fmax,
        "n_atoms": len(atoms),
        "dipole_debye": result["dipole_debye"],
    }
    summary_path.write_text(json.dumps(json_safe, indent=2))
    result["outputs"]["summary"] = str(summary_path)

    log.info(f"  E = {energy:.6f} eV ({energy * EV_TO_KCAL:.3f} kcal/mol), |F|max = {fmax:.4f}")
    return result
