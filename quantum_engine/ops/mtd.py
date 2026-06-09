"""Well-tempered / OPES metadynamics on a weighted bond-distance CV.

Thin orchestration wrapper over :mod:`quantum_engine.mlff.metadynamics`. The CV
is reaction-agnostic — define it either as a single-center bond difference
(``center_idx`` + ``breaking_idx`` + ``forming_idx``) or, for multi-bond /
no-center / forming-only ("click") / breaking-only reactions, as explicit
``breaking_bonds``/``forming_bonds`` lists of atom-index pairs.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ase import Atoms

log = logging.getLogger("quantum_engine.ops.mtd")


def run(
    atoms: Atoms,
    calculator=None,
    outdir: str | Path = ".",
    constraint=None,
    center_idx: int | None = None,
    forming_idx: int | None = None,
    breaking_idx: int | None = None,
    breaking_bonds: list[tuple[int, int]] | None = None,
    forming_bonds: list[tuple[int, int]] | None = None,
    total_time_ps: float = 100.0,
    temperature_K: float = 300.0,
    variant: str = "wt",  # "wt" = well-tempered; "opes" = OPES-MetaD
    basin_windows: dict | None = None,
    **kwargs,
) -> dict:
    """Metadynamics on a weighted bond-distance CV ``s = Σ w·d(i,j)``.

    Args:
        variant: "wt" for classic well-tempered MTD (default),
                 "opes" for OPES-MetaD (Invernizzi & Parrinello JPCL 2020,
                 fewer parameters, faster convergence).
        center_idx/breaking_idx/forming_idx: single-center CV
            (``s = d(center,breaking) − d(center,forming)``).
        breaking_bonds/forming_bonds: general CV terms (lists of atom-index
            pairs) for multi-bond / no-center / forming-only / breaking-only.
        basin_windows: optional ``{label: (lo, hi)}`` to label FES minima; with
            ``None``, reactant/product fall out of the extreme-CV basins (the
            reaction-agnostic default).
    """
    from quantum_engine.mlff.cv_spring import (
        bond_cv_terms_from_bonds, bond_cv_terms_from_roles)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if atoms.calc is None:
        if calculator is None:
            raise ValueError("Need a calculator")
        atoms.calc = calculator

    # Build the reaction-agnostic CV terms from whichever spec was given.
    if center_idx is not None:
        if forming_idx is None or breaking_idx is None:
            raise ValueError("center_idx requires forming_idx and breaking_idx")
        cv_terms = bond_cv_terms_from_roles(center_idx, breaking_idx, forming_idx)
    elif breaking_bonds or forming_bonds:
        cv_terms = bond_cv_terms_from_bonds(breaking_bonds or [], forming_bonds or [])
    else:
        raise ValueError(
            "MTD requires a CV: pass center_idx+forming_idx+breaking_idx, or "
            "breaking_bonds/forming_bonds (atom-index pairs)")

    if variant == "opes":
        from quantum_engine.mlff.metadynamics import run_opes_rescue
        entry = run_opes_rescue
        valid_kwargs = ("timestep_fs", "barrier_kJ_mol", "sigma_A",
                        "pace_steps", "bias_factor", "friction_per_ps")
    else:
        from quantum_engine.mlff.metadynamics import run_metadynamics_rescue
        entry = run_metadynamics_rescue
        valid_kwargs = ("timestep_fs", "bias_height_kJ_mol", "bias_sigma_A",
                        "bias_pace_steps", "bias_factor", "friction_per_ps")

    result = entry(
        atoms,
        cv_terms,
        calculator=atoms.calc,
        outdir=outdir,
        constraint=constraint,
        temperature_K=temperature_K,
        total_time_ps=total_time_ps,
        basin_windows=basin_windows,
        **{k: v for k, v in kwargs.items() if k in valid_kwargs},
    )

    return {
        "status": "completed",
        "atoms": atoms,
        "basins": result.get("basins"),
        "reactant": result.get("reactant"),
        "product": result.get("product"),
        "intermediate": result.get("intermediate"),
        "fes": result.get("fes"),
        "basin_depths_kcal": result.get("basin_depths_kcal"),
        "outputs": {
            "fes": str(outdir / "fes.npy"),
            "cv_trajectory": str(outdir / "cv_trajectory.npy"),
            "hills": str(outdir / "hills.npy"),
            "summary": str(outdir / "mtd_summary.json"),
        },
    }
