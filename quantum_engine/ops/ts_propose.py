"""TS-guess proposers — a new plug-and-play axis (generative + heuristic).

A *proposer* produces a transition-state GUESS directly from a reactant +
product, **skipping path search**. This is the integration point for generative
TS models (diffusion / flow-matching — OA-ReactDiff, React-OT, AEFM, …): they
propose a TS structure in one shot, which then feeds the SAME canonical gate as
any other guess (``ts_entry`` → refine → partial-Hessian → IRC-like validation).
Proposers don't accept a TS — they only suggest one; the saddle+Hessian+overlap+
IRC gate stays the acceptance authority.

A proposer is registered exactly like a path/saddle/energy backend, so adding a
new one (your own model included) is a one-liner + a small adapter:

    from quantum_engine.ops.ts_propose import register_ts_proposer

    @register_ts_proposer("my-model", aliases=("mine",))
    def _my_proposer(reactant, product, *, charge, spin, atom_map, outdir, **kw):
        ts = my_model.predict_ts(reactant, product)        # an ASE Atoms
        return {"ts_guess": ts, "confidence": 0.9, "status": "converged",
                "outputs": {}}

Contract — the runner takes the (atom-mapped) endpoints + run state and returns
``{"ts_guess": Atoms, "confidence": float|None, "status": str, "outputs": dict}``.
Endpoint atom-order consistency (which generative models REQUIRE — they map atom
i of R to atom i of P) is validated here, reusing the path-search safeguard.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ase import Atoms

from quantum_engine.logging_utils import get_logger
from quantum_engine.registry import Registry

log = get_logger("ops.ts_propose")

TS_PROPOSERS: Registry = Registry("ts-proposer")


def register_ts_proposer(name: str, runner=None, *, aliases: tuple[str, ...] = (),
                         overwrite: bool = False):
    """Register a TS-guess proposer (decorator or imperative)."""
    return TS_PROPOSERS.register(name, runner, aliases=aliases, overwrite=overwrite)


def make_ts_proposer(name: str):
    """Return the proposer runner registered under ``name`` (name/alias)."""
    if name not in TS_PROPOSERS:
        raise ValueError(
            f"Unknown TS proposer {name!r}. Choices: {TS_PROPOSERS.names()}")
    return TS_PROPOSERS.get(name)


# ---- built-in reference proposer (no extra deps) -------------------------
def _midpoint(reactant: Atoms, product: Atoms, *, charge: int, spin: int,
              atom_map=None, outdir, interpolation_method: str = "geodesic",
              **kwargs) -> dict:
    """The interpolation midpoint of R→P as a (naive) TS guess.

    A dependency-free reference proposer: it proves the plumbing end-to-end and
    is a sane baseline (the saddle refiner does the real work). A geodesic/IDPP
    midpoint is often a surprisingly serviceable guess for simple reactions.
    """
    from ase.io import write as ase_write  # noqa: PLC0415

    from quantum_engine.mlff.interpolation import interpolate  # noqa: PLC0415
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    images = interpolate(reactant, product, n_images=3, method=interpolation_method)
    ts = images[len(images) // 2].copy()
    ts.info["charge"] = charge
    ts.info["spin"] = spin
    out = outdir / "proposed_ts.xyz"
    ase_write(str(out), ts, format="extxyz")
    log.info("midpoint proposer: TS guess at the %s midpoint of R→P",
             interpolation_method)
    return {"ts_guess": ts, "confidence": None, "status": "converged",
            "outputs": {"ts_guess_xyz": str(out)}}


register_ts_proposer("midpoint", _midpoint, aliases=("geodesic-mid",))


def _react_ot(reactant: Atoms, product: Atoms, *, charge, spin, atom_map, outdir,
              **kwargs) -> dict:
    """React-OT generative proposer (gated — lives in its own sidecar)."""
    from quantum_engine.qm import reactot  # noqa: PLC0415
    return reactot.run(reactant, product, charge=charge, spin=spin,
                       atom_map=atom_map, outdir=outdir, **kwargs)


register_ts_proposer("react-ot", _react_ot, aliases=("reactot",))


def run(method: str, reactant: Atoms, product: Atoms, *, charge: int = 0,
        spin: int = 1, atom_map: dict | None = None, outdir: str | Path = ".",
        check_endpoints: bool = True, **kwargs) -> dict:
    """Propose a TS guess from ``reactant`` + ``product`` via ``method``.

    Validates endpoint atom-order consistency first (generative proposers map
    atom i of R to atom i of P), reusing the path-search safeguard, and reorders
    the product per ``atom_map`` when given. Returns the proposer result dict.
    """
    if check_endpoints:
        from quantum_engine.ops.path_search import (  # noqa: PLC0415
            check_endpoint_consistency, reorder_product_to_match)
        rep = check_endpoint_consistency(reactant, product, atom_map)
        if rep["needs_reorder"] and atom_map is not None:
            log.info("ts_propose: reordering product to the reactant's atom order")
            product = reorder_product_to_match(product, atom_map)
    runner = make_ts_proposer(method)
    log.info("ts_propose.run: method=%s, charge=%d, spin=%d", method, charge, spin)
    return runner(reactant, product, charge=charge, spin=spin, atom_map=atom_map,
                  outdir=Path(outdir), **kwargs)


__all__ = ["TS_PROPOSERS", "register_ts_proposer", "make_ts_proposer", "run"]
