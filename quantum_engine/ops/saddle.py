"""Sella saddle-point optimization (standalone, no IRC).

Wraps qcb.mlff.irc.sella_saddle_search with the standard ops interface.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ase import Atoms

from quantum_engine.mlff.irc import sella_saddle_search
from quantum_engine.units import EV_TO_KCAL

log = logging.getLogger("quantum_engine.ops.saddle")



def run(
    atoms: Atoms,
    calculator=None,
    outdir: str | Path = ".",
    constraint=None,
    fmax: float = 0.02,
    max_steps: int = 500,
    use_internal: bool | None = None,
    **kwargs,
) -> dict:
    """Sella saddle search.

    Use when input is a TS guess and you only want the refined TS (not IRC endpoints).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if atoms.calc is None:
        if calculator is None:
            raise ValueError("Need a calculator")
        atoms.calc = calculator

    ts_atoms, converged = sella_saddle_search(
        atoms, outdir,
        constraint=constraint,
        fmax=fmax, max_steps=max_steps, use_internal=use_internal,
    )
    energy = float(ts_atoms.get_potential_energy())

    return {
        "status": "converged" if converged else "not_converged",
        "converged": converged,
        "atoms": ts_atoms,
        "energy_eV": energy,
        "energy_kcal_mol": energy * EV_TO_KCAL,
        "outputs": {"saddle_xyz": str(outdir / "saddle.xyz")},
    }
