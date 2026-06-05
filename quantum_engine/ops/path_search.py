"""Reaction-path search — the third plug-and-play factory axis.

A thin registry over the double-ended path methods (CI-NEB, FSM, double-ended
GSM), giving them ONE uniform entry point so the orchestrator can swap path
methods without knowing which module implements each:

    from quantum_engine.ops.path_search import run as path_run
    res = path_run("neb", reactant, product, calculator_fn, outdir=..., charge=...)

Like the optimiser/saddle factories, a NEW path method drops in via one
``register_path(name, runner)`` call. Every runner takes a *per-image*
``calculator_fn`` (a zero-arg callable returning a FRESH calculator per image —
NEB/GSM mutate many geometries concurrently, so a shared calculator is wrong),
plus ``outdir``/``constraint``/``charge`` and method-specific ``**kwargs``, and
returns the standard path-result dict (``status``/``images``/``ts``/
``energies_eV``/``barrier_*_kcal``/``outputs``).

Single-ended SE-GSM (reactant-only, driving-coordinate driven) is a planned
backend (via the vendored ``deps/pyGSM``); it is intentionally NOT registered
yet, so ``make_path_method("gsm-se")`` raises a clear "unknown method" listing
the available ones rather than silently faking endpoints.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from ase import Atoms

from quantum_engine.registry import Registry

log = logging.getLogger("quantum_engine.ops.path_search")

PATH_METHODS: Registry = Registry("path-method")


def register_path(name: str, runner=None, *, aliases: tuple[str, ...] = (),
                  overwrite: bool = False):
    """Register a path-method backend (decorator or imperative)."""
    return PATH_METHODS.register(name, runner, aliases=aliases, overwrite=overwrite)


def make_path_method(method: str):
    """Return the path-method runner registered under ``method`` (name/alias)."""
    if method not in PATH_METHODS:
        raise ValueError(
            f"Unknown path method {method!r}. Choices: {PATH_METHODS.names()}")
    return PATH_METHODS.get(method)


# ---- built-in runners (uniform signature) --------------------------------
def _run_neb(reactant: Atoms, product: Atoms, calculator_fn: Callable[[], Any], *,
             outdir, constraint, charge, **kwargs) -> dict:
    from quantum_engine.ops import neb
    return neb.run(reactant, product, calculator_fn, outdir=outdir,
                   constraint=constraint, charge=charge, **kwargs)


def _run_fsm(reactant: Atoms, product: Atoms, calculator_fn: Callable[[], Any], *,
             outdir, constraint, charge, **kwargs) -> dict:
    from quantum_engine.ops import gsm
    if constraint is not None:
        log.warning("path-method 'fsm' does not support ASE constraints; ignoring")
    return gsm.run(reactant, product, calculator_fn, outdir, method="fsm",
                   charge=charge, **kwargs)


def _run_gsm_de(reactant: Atoms, product: Atoms, calculator_fn: Callable[[], Any], *,
                outdir, constraint, charge, **kwargs) -> dict:
    from quantum_engine.ops import gsm
    if constraint is not None:
        log.warning("path-method 'gsm-de' does not support ASE constraints; ignoring")
    return gsm.run(reactant, product, calculator_fn, outdir, method="gsm",
                   charge=charge, **kwargs)


register_path("neb", _run_neb, aliases=("ci-neb",))
register_path("fsm", _run_fsm)
register_path("gsm-de", _run_gsm_de, aliases=("gsm",))


def run(method: str, reactant: Atoms, product: Atoms,
        calculator_fn: Callable[[], Any], *,
        outdir: str | Path = ".", constraint=None, charge: int = 0,
        **kwargs) -> dict:
    """Dispatch a double-ended path search to ``method`` (name/alias).

    Args:
        method: ``"neb"`` (alias ``ci-neb``), ``"fsm"``, or ``"gsm-de"`` (alias
            ``gsm``). Unknown methods raise ValueError listing the choices.
        reactant, product: endpoint ``Atoms`` (basins; atom order must match).
        calculator_fn: zero-arg callable returning a FRESH calculator per image.
        outdir, constraint, charge: forwarded; ``constraint`` is honoured by NEB
            and warned-and-ignored by FSM/GSM (which lack ASE-constraint hooks).
        **kwargs: method-specific knobs (``n_images``, ``k_spring``, ``fmax_*``,
            ``max_nodes`` …) forwarded to the underlying runner.
    """
    runner = make_path_method(method)
    log.info("path_search.run: method=%s, charge=%d", method, charge)
    return runner(reactant, product, calculator_fn,
                  outdir=Path(outdir), constraint=constraint, charge=charge,
                  **kwargs)


__all__ = ["PATH_METHODS", "register_path", "make_path_method", "run"]
