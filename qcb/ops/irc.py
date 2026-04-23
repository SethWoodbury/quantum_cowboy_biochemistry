"""Intrinsic Reaction Coordinate (IRC) descent from a TS.

Wraps qcb.mlff.irc.irc_from_ts_guess (full workflow) or run_irc (descent-only).
"""
from __future__ import annotations

import logging
from pathlib import Path

from ase import Atoms

log = logging.getLogger("qcb.ops.irc")

EV_TO_KCAL = 23.0609


def run(
    atoms: Atoms,
    calculator=None,
    outdir: str | Path = ".",
    constraint=None,
    refine_ts: bool = True,
    saddle_fmax: float = 0.02,
    irc_step: float = 0.1,
    irc_fmax: float = 0.03,
    reactant_hint_bonds: dict | None = None,
    **kwargs,
) -> dict:
    """IRC descent from a TS (or TS guess).

    Args:
        atoms: the TS geometry (or close to it)
        refine_ts: if True, run Sella saddle search first, then IRC (recommended).
                   If False, assume atoms is already a saddle and skip Sella.
        saddle_fmax: Sella convergence threshold (only used if refine_ts=True)
        irc_step: displacement along imaginary mode (Å)
        irc_fmax: convergence for each IRC leg
        reactant_hint_bonds: dict {(i,j): 'bonded'|'broken'} to disambiguate
                            forward vs reverse side. Highly recommended.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if atoms.calc is None:
        if calculator is None:
            raise ValueError("Need a calculator")
        atoms.calc = calculator

    if refine_ts:
        from qcb.mlff.irc import irc_from_ts_guess
        res = irc_from_ts_guess(
            atoms, outdir=outdir, constraint=constraint,
            saddle_fmax=saddle_fmax, irc_step=irc_step, irc_fmax=irc_fmax,
            reactant_hint_bonds=reactant_hint_bonds,
        )
    else:
        from qcb.mlff.irc import run_irc
        res = run_irc(
            atoms, outdir=outdir,
            step_size=irc_step, fmax=irc_fmax, constraint=constraint,
            reactant_hint_bonds=reactant_hint_bonds,
        )
        # Wrap to match irc_from_ts_guess schema
        res = {
            "ts": atoms, "reactant": res.get("reactant"), "product": res.get("product"),
            "imag_freq_cm": res.get("mode_freq"),
            "success": res.get("success"),
        }

    return {
        "status": "converged" if res.get("success") else "not_converged",
        "atoms": res.get("ts"),
        "reactant": res.get("reactant"),
        "product": res.get("product"),
        "ts": res.get("ts"),
        "imag_freq_cm": res.get("imag_freq_cm"),
        "barrier_fwd_kcal": res.get("barrier_fwd_kcal"),
        "barrier_rev_kcal": res.get("barrier_rev_kcal"),
        "outputs": {
            "reactant_xyz": str(outdir / "reactant.xyz"),
            "product_xyz": str(outdir / "product.xyz"),
            "saddle_xyz": str(outdir / "saddle.xyz") if refine_ts else None,
        },
    }
