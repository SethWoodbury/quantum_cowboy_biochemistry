"""pyGSM (Zimmerman lab) Growing String Method — DE-GSM + SE-GSM via the ASE LoT.

A SECOND, mature, **pysisyphus-independent** route to growing-string path search,
and the ONLY route to **SE-GSM** (single-ended / reactant-only, driven by
``driving_coords`` — the product need not be known). pyGSM is vendored at
``deps/pyGSM`` (not pip-installed); its ``ASELoT`` drives ANY ASE calculator, so
our MLFFs / xTB plug straight in.

Design (modular + insulated, like ``qm/pysis_string.py``):
- All pyGSM-specific assembly (LoT → PES → topology → internal coords → Molecule →
  optimizer → GSM) lives here, behind two small entry points
  :func:`run_de_gsm` / :func:`run_se_gsm`, each returning the standard
  path-result dict. ``ops/path_search.py`` registers them as ``pygsm-de`` /
  ``gsm-se``.
- Energies/barriers are recomputed via the ASE calculator on the extracted
  geometries (eV), so we never depend on pyGSM's internal unit bookkeeping.
- pyGSM is noisy (prints + scratch files); we run it inside ``outdir`` and
  redirect its stdout, surfacing a clean human log + the standard result.

Units: ``fmax`` is eV/Å; pyGSM converges on gradient in atomic units, so we map
``fmax → fmax × EVANG2AUBOHR`` for its CONV thresholds.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from ase import Atoms

from quantum_engine.logging_utils import get_logger
from quantum_engine.units import EV_TO_KCAL

log = get_logger("qm.pygsm")

# eV/Å -> hartree/bohr (CODATA); pyGSM convergence thresholds are atomic units.
_EVANG2AUBOHR = 0.0194469038


def _ensure_pygsm_on_path() -> None:
    """Put the vendored pyGSM on sys.path (it is not pip-installed)."""
    try:
        import pyGSM  # noqa: F401
        return
    except ImportError:
        pass
    vendored = Path(__file__).resolve().parents[2] / "deps" / "pyGSM"
    if vendored.exists() and str(vendored) not in sys.path:
        sys.path.insert(0, str(vendored))


def pygsm_available() -> tuple[bool, str]:
    """Probe pyGSM + its ASE level-of-theory (cheap, no run)."""
    _ensure_pygsm_on_path()
    try:
        from pyGSM.level_of_theories.ase import ASELoT  # noqa: F401
        from pyGSM.growing_string_methods import DE_GSM, SE_GSM  # noqa: F401
        from pyGSM.molecule import Molecule  # noqa: F401
        return True, "pyGSM available (vendored)"
    except Exception as exc:  # noqa: BLE001
        return False, f"pyGSM unavailable: {exc}"


# ---------------------------------------------------------------------------
# pyGSM object assembly (isolated here; the only version-specific glue)
# ---------------------------------------------------------------------------
def _build_pes(atoms: Atoms, calculator, mult: int):
    from pyGSM.level_of_theories.ase import ASELoT  # noqa: PLC0415
    from pyGSM.potential_energy_surfaces import PES  # noqa: PLC0415
    lot = ASELoT.from_options(
        calculator, geom=[[a.symbol, *a.position] for a in atoms], ID=0)
    return PES.from_options(lot=lot, ad_idx=0, multiplicity=mult)


def _add_driving_bonds_to_topology(topo, driving_coords) -> None:
    """SE-GSM only: ensure each ADD/BREAK driving-coord bond is an edge of the
    topology, so the corresponding internal coordinate exists (else
    ``SE_GSM.get_tangent`` raises 'Distance i-j is not in list'). Driving-coord
    atom indices are 1-based; topology is 0-based."""
    for d in (driving_coords or []):
        tag = str(d[0]).upper()
        if tag in ("ADD", "BREAK") and len(d) >= 3:
            i, j = int(d[1]) - 1, int(d[2]) - 1
            a, b = (i, j) if i < j else (j, i)
            if not (topo.has_edge(a, b) or topo.has_edge(b, a)):
                topo.add_edge(a, b)


def _build_molecule(atoms: Atoms, pes, *, node_id: int, form_hessian: bool,
                    driving_coords=None):
    from pyGSM.coordinate_systems.delocalized_coordinates import (  # noqa: PLC0415
        DelocalizedInternalCoordinates)
    from pyGSM.coordinate_systems.primitive_internals import (  # noqa: PLC0415
        PrimitiveInternalCoordinates)
    from pyGSM.coordinate_systems.topology import Topology  # noqa: PLC0415
    from pyGSM.utilities.elements import ElementData  # noqa: PLC0415
    from pyGSM.molecule import Molecule  # noqa: PLC0415

    et = ElementData()
    elements = [et.from_symbol(s) for s in atoms.get_chemical_symbols()]
    topo = Topology.build_topology(xyz=atoms.get_positions(), atoms=elements)
    _add_driving_bonds_to_topology(topo, driving_coords)   # SE-GSM only
    prim = PrimitiveInternalCoordinates.from_options(
        xyz=atoms.get_positions(), atoms=elements, topology=topo, addtr=True)
    deloc = DelocalizedInternalCoordinates.from_options(
        xyz=atoms.get_positions(), atoms=elements, addtr=True, primitives=prim)
    return Molecule.from_options(
        geom=[[a.symbol, *a.position] for a in atoms], PES=pes,
        coord_obj=deloc, Form_Hessian=form_hessian), prim, topo, elements


def _make_optimizer(fmax: float):
    from pyGSM.optimizers.eigenvector_follow import eigenvector_follow  # noqa: PLC0415
    conv_au = max(float(fmax) * _EVANG2AUBOHR, 1e-5)
    return eigenvector_follow.from_options(
        print_level=0, Linesearch="NoLineSearch", update_hess_in_bg=False,
        conv_Ediff=100.0, conv_gmax=100.0, DMAX=0.1, opt_climb=True), conv_au


def _string_to_result(gsm, calculator_fn, charge: int, outdir: Path, tag: str) -> dict:
    """Extract the pyGSM string → ASE images, recompute energies (eV), assemble."""
    frames: list[Atoms] = []
    for geom in gsm.geometries:
        a = Atoms(symbols=[g[0] for g in geom], positions=[g[1:4] for g in geom])
        a.calc = calculator_fn()
        a.info["charge"] = charge
        frames.append(a)
    if not frames:
        return {"status": "failed", "error": "pyGSM produced no geometries",
                "images": [], "outputs": {}}
    energies = [float(im.get_potential_energy()) for im in frames]
    ts_idx = int(getattr(gsm, "TSnode", int(np.argmax(energies[1:-1])) + 1
                         if len(energies) > 2 else 0))
    ts_idx = max(0, min(ts_idx, len(frames) - 1))

    from ase.io import write as ase_write  # noqa: PLC0415
    xyz = outdir / f"{tag}_path.xyz"
    ase_write(str(xyz), frames, format="extxyz")
    barrier_fwd = (energies[ts_idx] - energies[0]) * EV_TO_KCAL
    log.info("%s done: %d images, TSnode=%d, Δ‡fwd = %.2f kcal/mol",
             tag.upper(), len(frames), ts_idx, barrier_fwd)
    return {
        "status": "converged", "images": frames, "ts": frames[ts_idx],
        "ts_idx": ts_idx, "reactant": frames[0], "product": frames[-1],
        "energies_eV": energies, "barrier_fwd_kcal": barrier_fwd,
        "barrier_rev_kcal": (energies[ts_idx] - energies[-1]) * EV_TO_KCAL,
        "n_images_final": len(frames), "outputs": {"path_xyz": str(xyz)},
    }


@contextlib.contextmanager
def _quiet_in(outdir: Path):
    """Run pyGSM inside ``outdir`` (it writes scratch/) with stdout redirected and
    its (very numerous) numpy DeprecationWarnings suppressed."""
    import warnings  # noqa: PLC0415
    outdir.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    buf = io.StringIO()
    try:
        os.chdir(str(outdir))
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield buf
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def run_de_gsm(reactant: Atoms, product: Atoms, calculator_fn: Callable[[], Any], *,
               outdir: str | Path = ".", charge: int = 0, mult: int = 1,
               n_nodes: int = 11, fmax: float = 0.05, max_iterations: int = 100,
               max_opt_steps: int = 3, **kwargs) -> dict:
    """Double-ended GSM (reactant + product) via pyGSM. Standard path-result dict."""
    ok, reason = pygsm_available()
    if not ok:
        log.warning("pygsm-de unavailable: %s", reason)
        return {"status": "unsupported", "error": reason, "images": [], "outputs": {}}
    from pyGSM.growing_string_methods import DE_GSM  # noqa: PLC0415

    outdir = Path(outdir)
    log.info("pyGSM DE-GSM: %d nodes, fmax=%s eV/Å", n_nodes, fmax)
    calc = calculator_fn()
    try:
        with _quiet_in(outdir):
            pes = _build_pes(reactant, calc, mult)
            mol_r, _prim, _topo, _el = _build_molecule(reactant, pes, node_id=0,
                                                       form_hessian=True)
            from pyGSM.molecule import Molecule  # noqa: PLC0415
            mol_p = Molecule.copy_from_options(
                mol_r, xyz=product.get_positions(), new_node_id=n_nodes - 1,
                copy_wavefunction=False)
            opt, conv = _make_optimizer(fmax)
            gsm = DE_GSM.from_options(
                reactant=mol_r, product=mol_p, nnodes=n_nodes, CONV_TOL=conv,
                CONV_gmax=100.0, CONV_Ediff=100.0, ADD_NODE_TOL=0.1,
                growth_direction=0, optimizer=opt, ID=0, print_level=0, mp_cores=1,
                interp_method="DLC")
            gsm.go_gsm(max_iterations, max_opt_steps, rtype=1)   # rtype=1 = climb
        return _string_to_result(gsm, calculator_fn, charge, outdir, "pygsm_de")
    except Exception as exc:  # noqa: BLE001 — pyGSM internals can be finicky
        log.error("pyGSM DE-GSM failed: %s", exc)
        return {"status": "failed", "error": str(exc), "images": [], "outputs": {}}


def run_se_gsm(reactant: Atoms, calculator_fn: Callable[[], Any],
               driving_coords: Sequence, *, outdir: str | Path = ".",
               charge: int = 0, mult: int = 1, n_nodes: int = 20,
               fmax: float = 0.05, max_iterations: int = 100,
               max_opt_steps: int = 3, **kwargs) -> dict:
    """Single-ended GSM (reactant + driving coordinates; product unknown).

    ``driving_coords`` is pyGSM's grammar, e.g.
    ``[("ADD", i, j), ("BREAK", k, l)]`` (1-based atom indices). Returns the
    standard path-result dict; raises a clear error if no driving coords given.
    """
    if not driving_coords:
        raise ValueError(
            "SE-GSM needs driving_coords (e.g. [('ADD', i, j), ('BREAK', k, l)]) "
            "— the single-ended string is grown along them. None supplied.")
    ok, reason = pygsm_available()
    if not ok:
        log.warning("gsm-se unavailable: %s", reason)
        return {"status": "unsupported", "error": reason, "images": [], "outputs": {}}
    from pyGSM.growing_string_methods import SE_GSM  # noqa: PLC0415

    outdir = Path(outdir)
    log.info("pyGSM SE-GSM: %d nodes, driving=%s", n_nodes, list(driving_coords))
    calc = calculator_fn()
    try:
        with _quiet_in(outdir):
            pes = _build_pes(reactant, calc, mult)
            mol_r, _prim, _topo, _el = _build_molecule(
                reactant, pes, node_id=0, form_hessian=True,
                driving_coords=driving_coords)   # add ADD/BREAK bonds to primitives
            opt, conv = _make_optimizer(fmax)
            gsm = SE_GSM.from_options(
                reactant=mol_r, nnodes=n_nodes, optimizer=opt,
                driving_coords=[list(d) for d in driving_coords], CONV_TOL=conv,
                ADD_NODE_TOL=0.1, ID=0, print_level=0, mp_cores=1, interp_method="DLC")
            gsm.go_gsm(max_iterations, max_opt_steps, rtype=1)
        return _string_to_result(gsm, calculator_fn, charge, outdir, "pygsm_se")
    except Exception as exc:  # noqa: BLE001 — SE-GSM internals are finicky
        log.error("pyGSM SE-GSM failed: %s", exc)
        return {"status": "failed", "error": str(exc), "images": [], "outputs": {}}


__all__ = ["pygsm_available", "run_de_gsm", "run_se_gsm"]
