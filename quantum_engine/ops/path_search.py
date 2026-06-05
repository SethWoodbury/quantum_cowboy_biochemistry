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


def check_endpoint_consistency(reactant: Atoms, product: Atoms,
                               atom_map: dict | None = None) -> dict:
    """Validate that two endpoints are index-consistent for double-ended pathing.

    NEB/FSM/GSM interpolate atom ``i`` of the reactant toward atom ``i`` of the
    product, so the atom ORDER and per-index ELEMENTS must match (a mismatch
    interpolates e.g. an O into an H — silent garbage). This guards against that.

    If ``atom_map`` ``{reactant_idx: product_idx}`` is given it must be a bijection
    over all atoms, and the MAPPED elements must match; ``reorder_product_to_match``
    can then bring the product into the reactant's order. Returns a report dict
    ``{ok, n_atoms, mismatches, needs_reorder, atom_map}``; raises ValueError on a
    genuine mismatch (no silent interpolation across mismatched atoms).
    """
    nr, npd = len(reactant), len(product)
    if nr != npd:
        raise ValueError(
            f"endpoint atom-count mismatch: reactant has {nr}, product has {npd}. "
            "Double-ended path methods need matching atoms in matching order.")
    rsym = [str(s) for s in reactant.symbols]
    psym = [str(s) for s in product.symbols]

    if atom_map is not None:
        if sorted(atom_map.keys()) != list(range(nr)) or \
                sorted(atom_map.values()) != list(range(nr)):
            raise ValueError(
                "atom_map must be a bijection over all atom indices "
                f"[0, {nr}); got keys/values that don't cover every atom.")
        mism = [(i, rsym[i], psym[atom_map[i]]) for i in range(nr)
                if rsym[i] != psym[atom_map[i]]]
        if mism:
            raise ValueError(
                f"atom_map maps {len(mism)} atom(s) onto a different element, e.g. "
                f"reactant[{mism[0][0]}]={mism[0][1]} → product={mism[0][2]}. "
                "Fix the mapping.")
        needs_reorder = any(atom_map[i] != i for i in range(nr))
        return {"ok": True, "n_atoms": nr, "mismatches": [],
                "needs_reorder": needs_reorder, "atom_map": dict(atom_map)}

    mism = [(i, rsym[i], psym[i]) for i in range(nr) if rsym[i] != psym[i]]
    if mism:
        raise ValueError(
            f"endpoint element order mismatch at {len(mism)} index(es), e.g. "
            f"reactant[{mism[0][0]}]={mism[0][1]} vs product[{mism[0][0]}]={mism[0][2]}. "
            "Reorder the product to match the reactant, or pass an atom_map "
            "{reactant_idx: product_idx}.")
    return {"ok": True, "n_atoms": nr, "mismatches": [],
            "needs_reorder": False, "atom_map": None}


def reorder_product_to_match(product: Atoms, atom_map: dict) -> Atoms:
    """Return a copy of ``product`` reordered so its atom ``i`` is the partner of
    reactant atom ``i`` (per ``atom_map`` ``{reactant_idx: product_idx}``)."""
    order = [atom_map[i] for i in range(len(product))]
    return product[order]


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
    # spin → multiplicity (2S+1) from the stamped endpoint; explicit mult wins.
    kwargs.setdefault("mult", int(reactant.info.get("spin", 1)))
    return gsm.run(reactant, product, calculator_fn, outdir, method="fsm",
                   charge=charge, **kwargs)


def _run_gsm_de(reactant: Atoms, product: Atoms, calculator_fn: Callable[[], Any], *,
                outdir, constraint, charge, **kwargs) -> dict:
    from quantum_engine.ops import gsm
    if constraint is not None:
        log.warning("path-method 'gsm-de' does not support ASE constraints; ignoring")
    kwargs.setdefault("mult", int(reactant.info.get("spin", 1)))   # spin → 2S+1
    return gsm.run(reactant, product, calculator_fn, outdir, method="gsm",
                   charge=charge, **kwargs)


register_path("neb", _run_neb, aliases=("ci-neb",))
register_path("fsm", _run_fsm)
register_path("gsm-de", _run_gsm_de, aliases=("gsm",))


def run(method: str, reactant: Atoms, product: Atoms,
        calculator_fn: Callable[[], Any], *,
        outdir: str | Path = ".", constraint=None, charge: int = 0,
        atom_map: dict | None = None, check_endpoints: bool = True,
        **kwargs) -> dict:
    """Dispatch a double-ended path search to ``method`` (name/alias).

    Args:
        method: ``"neb"`` (alias ``ci-neb``), ``"fsm"``, or ``"gsm-de"`` (alias
            ``gsm``). Unknown methods raise ValueError listing the choices.
        reactant, product: endpoint ``Atoms`` (basins; atom order must match).
        calculator_fn: zero-arg callable returning a FRESH calculator per image.
        outdir, constraint, charge: forwarded; ``constraint`` is honoured by NEB
            and warned-and-ignored by FSM/GSM (which lack ASE-constraint hooks).
        atom_map: optional ``{reactant_idx: product_idx}``; the product is
            reordered to the reactant's order before pathing.
        check_endpoints: validate endpoint index/element consistency first
            (default True — guards against interpolating mismatched atoms).
        **kwargs: method-specific knobs (``n_images``, ``k_spring``, ``fmax_*``,
            ``max_nodes`` …) forwarded to the underlying runner.
    """
    if check_endpoints:
        rep = check_endpoint_consistency(reactant, product, atom_map)
        if rep["needs_reorder"] and atom_map is not None:
            log.info("path_search: reordering product to the reactant's atom order "
                     "per atom_map")
            product = reorder_product_to_match(product, atom_map)
    runner = make_path_method(method)
    log.info("path_search.run: method=%s, charge=%d", method, charge)
    return runner(reactant, product, calculator_fn,
                  outdir=Path(outdir), constraint=constraint, charge=charge,
                  **kwargs)


__all__ = ["PATH_METHODS", "register_path", "make_path_method", "run",
           "check_endpoint_consistency", "reorder_product_to_match"]
