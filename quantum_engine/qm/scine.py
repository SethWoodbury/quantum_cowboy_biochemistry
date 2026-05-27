"""SCINE Chemoton + ReaDuct adapter.

SCINE is the modern, actively-maintained replacement for YARP. Chemoton
explores reaction networks (BSD-3, qcscine/chemoton); ReaDuct does the
single/double-ended path search and TS optimisation (qcscine/readuct).

We use them at two stages of the enzyme TS-search pipelines:

  * **Reaction enumeration** (alternative to autodE): given a set of
    reactant fragments + active-site cluster, Chemoton enumerates
    plausible elementary steps. Useful as a hypothesis generator for
    metallohydrolases where the nucleophile or proton-transfer path
    isn't obvious. ``enumerate_elementary_steps`` delegates to the
    SteeringWheel-driven facade in ``tools/scine_bridge.py``; for the
    full Chemoton-explore pipeline use the ``qcb chemoton-explore`` CLI.
  * **Path search** (alternative to pyGSM): ReaDuct's double-ended
    B-spline interpolation finds TSs between known reactant and product
    states. Single-ended Newton-trajectory search drives via a chosen
    coordinate.

Install:
    bash deps/build_scine.sh   # pip install pinned wheels into qcb-xtb

The vendored deps/scine_chemoton + deps/scine_readuct submodules are
source-of-truth tracking only — actual binaries come from PyPI.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("quantum_engine.qm.scine")


def scine_available() -> bool:
    """Are scine_chemoton + scine_readuct importable in this env?"""
    try:
        import scine_chemoton  # noqa: F401
        import scine_readuct  # noqa: F401
        return True
    except ImportError:
        return False


def _require() -> None:
    if not scine_available():
        raise ImportError(
            "SCINE not installed. Run `bash deps/build_scine.sh` "
            "(pip-installs scine-chemoton + scine-readuct into qcb-xtb)."
        )


def _ase_to_atom_collection(atoms: Any) -> Any:
    """Convert an ASE ``Atoms`` to ``scine_utilities.AtomCollection``.

    SCINE's AtomCollection stores positions in Bohr; ASE in Å. Caller is
    responsible for matching multiplicities / charge separately when
    constructing tasks.
    """
    import numpy as np
    import scine_utilities as su

    ac = su.AtomCollection()
    positions = atoms.get_positions() * su.BOHR_PER_ANGSTROM
    for sym, pos in zip(atoms.get_chemical_symbols(), positions):
        elem = getattr(su.ElementType, sym.capitalize())
        ac.push_back(su.Atom(elem, pos))
    return ac


def _atom_collection_to_ase(ac: Any) -> Any:
    """Convert a ``scine_utilities.AtomCollection`` back to an ASE ``Atoms``."""
    import numpy as np
    import scine_utilities as su
    from ase import Atoms

    n = ac.size()
    positions = np.array([ac.get_position(i) for i in range(n)]) * su.ANGSTROM_PER_BOHR
    symbols = [str(ac.get_element(i)).split(".")[-1] for i in range(n)]
    return Atoms(symbols=symbols, positions=positions)


def _readuct_settings(backend: str) -> dict:
    """Return ReaDuct task-settings for the requested backend.

    Maps qcb-flavoured backend strings to ReaDuct's
    (program / method / basis_set) tuple. Currently supports
    ``'sparrow'`` (PM6), ``'sparrow-pm7'``, ``'xtb'`` (xtb-GFN2).
    """
    backend = backend.lower()
    if backend in ("sparrow", "sparrow-pm6"):
        return {"program": "sparrow", "method_family": "PM6", "method": "PM6", "basis_set": ""}
    if backend in ("sparrow-pm7", "pm7"):
        return {"program": "sparrow", "method_family": "PM7", "method": "PM7", "basis_set": ""}
    if backend in ("xtb", "xtb-gfn2", "gfn2"):
        return {"program": "xtb", "method_family": "GFN2", "method": "GFN2", "basis_set": ""}
    raise ValueError(
        f"Unknown SCINE backend {backend!r}. Choose among "
        "{'sparrow', 'sparrow-pm7', 'xtb-gfn2'}."
    )


def enumerate_elementary_steps(
    reactants: list[Any],            # list of ASE Atoms
    *,
    active_atoms: list[int] | None = None,
    backend: str = "sparrow",
    workdir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Chemoton — exhaustively enumerate elementary steps (bond-forming/
    breaking moves) starting from ``reactants``. Restrict to the
    ``active_atoms`` subset to keep the search bounded.

    This is a *thin* wrapper. For the full Chemoton-explore pipeline with
    Steering Wheel filtering, MongoDB lifecycle, and PDB round-trip use the
    ``qcb chemoton-explore`` CLI (see ``quantum_engine/ops/chemoton_explore.py``)
    or the bridge directly via ``tools/scine_bridge.py``.

    Returns a list of step dicts, each with reactant/product Atoms +
    proposed driving coordinates.
    """
    _require()
    log.warning(
        "enumerate_elementary_steps: thin wrapper — for full functionality use "
        "`qcb chemoton-explore <input.pdb>` (see "
        "quantum_engine/ops/chemoton_explore.py)."
    )
    raise NotImplementedError(
        "Direct in-memory enumeration is not implemented. Use "
        "`qcb chemoton-explore` or the helper functions in "
        "tools/scine_bridge.py (pdb_to_scine_cluster + run_steered_exploration)."
    )


def double_ended_b_spline(
    reactant: Any,                   # ASE Atoms
    product: Any,                    # ASE Atoms
    *,
    n_images: int = 11,
    backend: str = "xtb",
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    """ReaDuct — double-ended B-spline TS search between known endpoints.

    The B-spline interpolant is the path; the highest image's geometry is
    optimised to a saddle point.

    Returns ``{ts_atoms, ts_energy, path_atoms, path_energies, status}``.
    """
    _require()

    import scine_readuct as readuct
    import scine_utilities as su

    workdir = Path(workdir or tempfile.mkdtemp(prefix="qcb_bspline_"))
    workdir.mkdir(parents=True, exist_ok=True)
    log.info("double_ended_b_spline: backend=%s n_images=%d workdir=%s",
             backend, n_images, workdir)

    r_ac = _ase_to_atom_collection(reactant)
    p_ac = _ase_to_atom_collection(product)

    settings = _readuct_settings(backend)

    # ReaDuct exposes its tasks as ``run_<task>_task(systems, names, **settings)``
    # where ``systems`` is a list of Calculator objects. We construct a fresh
    # calculator pair via scine_utilities's ManagerLocator + calculator_factory.
    try:
        from scine_utilities.core import ModuleManager  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            f"Could not import scine_utilities.core.ModuleManager: {exc}. "
            "ReaDuct tasks need a valid calculator factory; ensure "
            "scine_sparrow / scine_xtb_wrapper is loaded."
        ) from None

    mm = ModuleManager.get_instance()
    program = settings["program"]
    if not mm.has_module(program):
        raise RuntimeError(
            f"SCINE module {program!r} is not loaded; run "
            "`apptainer exec quantum_chem.sif python ...` from inside the "
            "container so the right backend is on LD_LIBRARY_PATH."
        )
    calc = mm.get(program)
    calc.settings.update({k: v for k, v in settings.items() if k != "program"})
    calc.structure = r_ac

    raise NotImplementedError(
        "double_ended_b_spline: B-spline task plumbing pending (use "
        "scine_readuct.run_bspline_task once you have charge/multiplicity "
        "metadata wired through). See "
        "deps/scine_readuct/Readuct/python/scripts/ for the Python entrypoints."
    )


def newton_trajectory_drive(
    reactant: Any,
    *,
    driving_coords: list[tuple[str, list[int], float]],
    backend: str = "xtb",
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    """ReaDuct — single-ended Newton-trajectory scan along the supplied
    driving coordinates. Returns a TS guess at the first maximum found
    plus the full scan path.

    ``driving_coords`` is a list of ``("bond"|"angle"|"dihedral",
    [atom_indices], target_delta)`` tuples (target_delta is signed: positive =
    increase the coordinate, negative = decrease). Currently a thin pass-through
    that hands off to ``scine_readuct.run_nt_task`` — see ReaDuct's docs for the
    full settings dictionary.
    """
    _require()

    import scine_readuct as readuct
    workdir = Path(workdir or tempfile.mkdtemp(prefix="qcb_nt_"))
    workdir.mkdir(parents=True, exist_ok=True)
    log.info("newton_trajectory_drive: backend=%s coords=%s workdir=%s",
             backend, driving_coords, workdir)
    settings = _readuct_settings(backend)

    raise NotImplementedError(
        "newton_trajectory_drive: NT task plumbing pending. The hook is "
        "scine_readuct.run_nt_task(systems, names, association_indices=..., "
        "dissociation_indices=..., total_force_norm=..., "
        "convergence_max_iterations=...); we need a small Calculator factory "
        "wired the same way as double_ended_b_spline. Track the same "
        "downstream pull request that fixes B-spline."
    )


def optimise_ts(
    ts_guess: Any,                   # ASE Atoms near a saddle
    *,
    backend: str = "xtb",
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    """ReaDuct — Berny-style TS optimisation from a guess.

    Returns ``{ts_atoms, ts_energy, freq_imag_cm, status}`` once wired up.
    """
    _require()

    import scine_readuct as readuct
    workdir = Path(workdir or tempfile.mkdtemp(prefix="qcb_tsopt_"))
    workdir.mkdir(parents=True, exist_ok=True)
    log.info("optimise_ts: backend=%s workdir=%s", backend, workdir)

    raise NotImplementedError(
        "optimise_ts: TS-opt task plumbing pending. The hook is "
        "scine_readuct.run_tsopt_task. Use Sella via "
        "`quantum_engine.ops.saddle.run` until ReaDuct plumbing is finished."
    )


def irc_descend(
    ts_atoms: Any,
    *,
    backend: str = "xtb",
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    """ReaDuct IRC — descend from TS in both directions to confirm
    reactant/product connectivity. Returns the two endpoints + paths."""
    _require()

    import scine_readuct as readuct
    workdir = Path(workdir or tempfile.mkdtemp(prefix="qcb_irc_"))
    workdir.mkdir(parents=True, exist_ok=True)
    log.info("irc_descend: backend=%s workdir=%s", backend, workdir)

    raise NotImplementedError(
        "irc_descend: IRC task plumbing pending. The hook is "
        "scine_readuct.run_irc_task. Use ASE-based IRC via "
        "`quantum_engine.ops.irc.run` until ReaDuct plumbing is finished."
    )
