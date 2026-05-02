"""Energy minimization (geometry optimization).

Analogous to Gaussian `opt` keyword. Supports LBFGS (default, quasi-Newton),
BFGS, and FIRE (good for noisy ML potentials; Bitzek et al. PRL 2006,
doi:10.1103/PhysRevLett.97.170201).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.optimize import LBFGS, BFGS, FIRE

log = logging.getLogger("quantum_engine.ops.opt")

EV_TO_KCAL = 23.0609

OPTIMIZERS = {"lbfgs": LBFGS, "bfgs": BFGS, "fire": FIRE}


def run(
    atoms: Atoms,
    calculator=None,
    outdir: str | Path = ".",
    constraint=None,
    optimizer: str = "lbfgs",
    fmax: float = 0.05,
    max_steps: int = 500,
    **kwargs,
) -> dict:
    """Geometry optimization.

    Args:
        atoms: ASE Atoms
        calculator: ASE calculator
        outdir: output directory
        constraint: ASE constraint, list of constraints, or None
        optimizer: "lbfgs" (default), "bfgs", or "fire"
        fmax: convergence threshold on max atomic force (eV/Å)
        max_steps: safety cap on iterations
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if atoms.calc is None:
        if calculator is None:
            raise ValueError("Need a calculator")
        atoms.calc = calculator

    if constraint is not None:
        if not isinstance(constraint, list):
            constraint = [constraint]
        atoms.set_constraint(constraint)

    opt_cls = OPTIMIZERS.get(optimizer.lower())
    if opt_cls is None:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Choices: {list(OPTIMIZERS)}")

    traj_path = outdir / f"opt-{optimizer}.traj"
    log_path = outdir / f"opt-{optimizer}.log"

    log.info(f"Minimization: {optimizer.upper()} on {len(atoms)} atoms, fmax={fmax}")
    dyn = opt_cls(atoms, trajectory=str(traj_path), logfile=str(log_path))
    dyn.run(fmax=fmax, steps=max_steps)

    e_final = atoms.get_potential_energy()
    f_final = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    converged = f_final <= fmax * 1.05  # small tolerance
    n_steps = dyn.nsteps if hasattr(dyn, "nsteps") else -1

    result = {
        "status": "converged" if converged else "not_converged",
        "converged": converged,
        "atoms": atoms,
        "energy_eV": float(e_final),
        "energy_kcal_mol": float(e_final * EV_TO_KCAL),
        "fmax_final": f_final,
        "n_steps": n_steps,
        "outputs": {
            "trajectory": str(traj_path),
            "log": str(log_path),
        },
    }

    (outdir / "opt-summary.json").write_text(json.dumps({
        k: v for k, v in result.items() if k not in ("atoms", "outputs")
    }, indent=2))
    result["outputs"]["summary"] = str(outdir / "opt-summary.json")

    status_str = "✓ converged" if converged else "✗ did not converge"
    log.info(f"  {status_str}: E={e_final:.6f} eV, |F|max={f_final:.4f}, steps={n_steps}")
    return result
