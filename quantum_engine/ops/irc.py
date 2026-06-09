"""Intrinsic Reaction Coordinate (IRC) descent from a TS.

Wraps quantum_engine.mlff.irc.irc_from_ts_guess (full workflow) or run_irc (descent-only).
"""
from __future__ import annotations

from quantum_engine.units import EV_TO_KCAL

import logging
from pathlib import Path

from ase import Atoms

from quantum_engine.registry import Registry

log = logging.getLogger("quantum_engine.ops.irc")


# IRC-method registry — the fourth plug-and-play axis. Only the MLFF/ASE descent
# exists today; a future QM-native IRC (ORCA/pysisyphus) drops in via one
# ``register_irc(name, runner)`` call. Runners share ``run``'s signature.
IRC_METHODS: Registry = Registry("irc-method")


def register_irc(name: str, runner=None, *, aliases: tuple[str, ...] = (),
                 overwrite: bool = False):
    """Register an IRC backend (decorator or imperative)."""
    return IRC_METHODS.register(name, runner, aliases=aliases, overwrite=overwrite)


def make_irc(method: str = "mlff"):
    """Return the IRC runner registered under ``method`` (name/alias)."""
    if method not in IRC_METHODS:
        raise ValueError(
            f"Unknown IRC method {method!r}. Choices: {IRC_METHODS.names()}")
    return IRC_METHODS.get(method)


def run(
    atoms: Atoms,
    calculator=None,
    outdir: str | Path = ".",
    constraint=None,
    refine_ts: bool = True,
    saddle_fmax: float = 0.02,
    irc_step: float = 0.1,
    irc_fmax: float = 0.05,
    irc_max_steps: int = 400,
    reactant_hint_bonds: dict | None = None,
    *,
    method: str = "mlff",
    **kwargs,
) -> dict:
    """IRC descent from a TS (or TS guess) via the chosen ``method`` backend.

    Args:
        atoms: the TS geometry (or close to it)
        method: IRC backend (default ``"mlff"`` — the ASE/MLFF descent). New
                backends register via :func:`register_irc`.
        refine_ts: if True, run Sella saddle search first, then IRC (recommended).
                   If False, assume atoms is already a saddle and skip Sella.
        saddle_fmax: Sella convergence threshold (only used if refine_ts=True)
        irc_step: displacement along imaginary mode (Å)
        irc_fmax: convergence for each IRC leg
        reactant_hint_bonds: dict {(i,j): 'bonded'|'broken'} to disambiguate
                            forward vs reverse side. Highly recommended.
    """
    runner = make_irc(method)
    return runner(
        atoms, calculator=calculator, outdir=outdir, constraint=constraint,
        refine_ts=refine_ts, saddle_fmax=saddle_fmax, irc_step=irc_step,
        irc_fmax=irc_fmax, irc_max_steps=irc_max_steps,
        reactant_hint_bonds=reactant_hint_bonds, **kwargs)


def _run_mlff_irc(
    atoms: Atoms,
    calculator=None,
    outdir: str | Path = ".",
    constraint=None,
    refine_ts: bool = True,
    saddle_fmax: float = 0.02,
    irc_step: float = 0.1,
    irc_fmax: float = 0.05,
    irc_max_steps: int = 400,
    reactant_hint_bonds: dict | None = None,
    **kwargs,
) -> dict:
    """MLFF/ASE IRC descent (the default backend; see :func:`run`)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if atoms.calc is None:
        if calculator is None:
            raise ValueError("Need a calculator")
        atoms.calc = calculator

    if refine_ts:
        from quantum_engine.mlff.irc import irc_from_ts_guess
        res = irc_from_ts_guess(
            atoms, outdir=outdir, constraint=constraint,
            saddle_fmax=saddle_fmax, irc_step=irc_step, irc_fmax=irc_fmax,
            irc_max_steps=irc_max_steps,
            reactant_hint_bonds=reactant_hint_bonds,
        )
    else:
        from quantum_engine.mlff.irc import run_irc
        res = run_irc(
            atoms, outdir=outdir,
            step_size=irc_step, fmax=irc_fmax, max_steps=irc_max_steps,
            constraint=constraint,
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


register_irc("mlff", _run_mlff_irc, aliases=("ase", "default"))


__all__ = ["run", "make_irc", "register_irc", "IRC_METHODS"]
