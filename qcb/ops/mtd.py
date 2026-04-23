"""Well-tempered metadynamics (wraps qcb.mlff.metadynamics)."""
from __future__ import annotations

import logging
from pathlib import Path

from ase import Atoms

log = logging.getLogger("qcb.ops.mtd")


def run(
    atoms: Atoms,
    calculator=None,
    outdir: str | Path = ".",
    constraint=None,
    p_idx: int = None,
    nuc_idx: int = None,
    lg_idx: int = None,
    total_time_ps: float = 100.0,
    temperature_K: float = 300.0,
    **kwargs,
) -> dict:
    """Well-tempered metadynamics on the bond-difference CV.

    Required atom indices: p_idx (central atom), nuc_idx (nucleophile), lg_idx (leaving group).
    """
    from qcb.mlff.metadynamics import run_metadynamics_rescue

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if atoms.calc is None:
        if calculator is None:
            raise ValueError("Need a calculator")
        atoms.calc = calculator

    if p_idx is None or nuc_idx is None or lg_idx is None:
        raise ValueError("MTD requires p_idx, nuc_idx, lg_idx (atom indices for the CV)")

    result = run_metadynamics_rescue(
        atoms,
        p_idx=p_idx, nuc_idx=nuc_idx, lg_idx=lg_idx,
        calculator=atoms.calc,
        outdir=outdir,
        constraint=constraint,
        temperature_K=temperature_K,
        total_time_ps=total_time_ps,
        **{k: v for k, v in kwargs.items()
           if k in ("timestep_fs", "bias_height_kJ_mol", "bias_sigma_A",
                    "bias_pace_steps", "bias_factor", "friction_per_ps")},
    )

    return {
        "status": "completed",
        "atoms": atoms,
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
