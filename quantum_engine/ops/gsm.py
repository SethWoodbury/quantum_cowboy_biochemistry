"""Growing String Method (GSM) and Freezing String Method (FSM).

Wraps pysisyphus's ChainOfStates implementations:
- FSM (FreezingString): grows path from both ends, freezes each added node.
  Tolerates imperfect endpoints; never reverts.
- GSM (GrowingString): similar but continues optimizing nodes.

References
----------
- Behn, A.; Zimmerman, P.M.; Bell, A.T.; Head-Gordon, M. "Efficient exploration
  of reaction paths via a freezing string method." J. Chem. Phys. 2011, 135, 224108.
  https://doi.org/10.1063/1.3664901
- Zimmerman, P.M. "Growing string method with interpolation and optimization in
  internal coordinates: method and examples." J. Chem. Phys. 2013, 138, 184102.
  https://doi.org/10.1063/1.4804162
- Wan et al., arXiv:2604.00405 (2026): benchmarks MACE-OMol + FSM at 88-90%
  success vs. 63-71% for CI-NEB on the same 58-reaction dataset.

When to use
-----------
- FSM: reactions where CI-NEB struggles (shallow product basins, concerted
  mechanisms, floppy systems). Faster-to-approximate-TS than NEB.
- GSM: when you want a smoother path than FSM at slightly higher cost.
- Both: require reactant + product endpoints as input (same as NEB).

When NOT to use
---------------
- Input is a single TS guess — use IRC-from-TS instead (`cowboy-qc irc`).
- Input is one endpoint and you don't know the other — use CV spring
  driving (`cowboy-qc ts --strategy cv-spring`) or metadynamics (`cowboy-qc mtd`).
"""
from __future__ import annotations

from quantum_engine.units import EV_TO_KCAL

import logging
from pathlib import Path
from typing import Callable

from ase import Atoms

log = logging.getLogger("quantum_engine.ops.gsm")


# NB: pysisyphus path resolution is handled ONCE, correctly, by
# quantum_engine.qm.pysisyphus._ensure_pysisyphus_on_path() (imported below) —
# it prefers an INSTALLED pysisyphus and only falls back to the vendored
# deps/pysisyphus when none is installed. We deliberately do NOT insert the
# vendored copy here: doing so unconditionally shadowed the working installed
# package with the incomplete vendored tree (missing the generated version.py),
# which made FSM/GSM silently fail with "pysisyphus not available".

# Unit-correct ASE<->pysisyphus glue lives in ONE place (qm/pysisyphus.py) and is
# shared by the saddle backends AND the string methods here — no divergence. See
# the units note there: pysisyphus is atomic-units throughout (bohr / hartree /
# hartree-per-bohr), so geometries go in as bohr and forces come back converted.
from quantum_engine.qm.pysisyphus import (  # noqa: E402
    _atoms_to_geom as _atoms_to_pysis_geom,
    make_pysis_calc_getter as _make_pysis_calc_getter,
)


def _pysis_geom_to_ase(g, calc=None, charge: int = 0) -> Atoms:
    """Final pysisyphus Geometry -> ASE Atoms: coords bohr->Å, symbols re-cased.

    pysisyphus stores element symbols lowercase and coordinates in bohr.
    """
    import numpy as np
    from pysisyphus.constants import BOHR2ANG
    n = len(g.atoms)
    positions = np.asarray(g.coords, dtype=float).reshape(n, 3) * BOHR2ANG
    symbols = [str(s).capitalize() for s in g.atoms]
    a = Atoms(symbols=symbols, positions=positions)
    if calc is not None:
        a.calc = calc
    a.info["charge"] = charge
    return a


def _assemble_string_result(drive: dict, calculator_fn, charge: int, outdir: Path,
                            tag: str) -> dict:
    """Turn a pysis_string driver result (pysisyphus Geometry images) into the
    standard path-result dict. Propagates unsupported/failed cleanly."""
    import numpy as np  # noqa: PLC0415
    if drive.get("status") in ("unsupported", "failed") or not drive.get("images"):
        return {"status": drive.get("status", "failed"),
                "error": drive.get("error"), "images": [], "outputs": {}}

    images = [_pysis_geom_to_ase(g, calc=calculator_fn(), charge=charge)
              for g in drive["images"]]
    energies = [float(im.get_potential_energy()) for im in images]
    ts_idx = int(np.argmax(energies[1:-1])) + 1 if len(energies) > 2 else 0

    # Cheap sanity guards: a string peak can be a misplaced node (a clash / a point on a
    # repulsive wall), and/or geometrically ~identical to an endpoint — i.e. not a TS at
    # all. Emit LOUD warnings here so a suspect guess is scrutinised before refinement.
    from quantum_engine.analysis.ts_sanity import (  # noqa: PLC0415
        ts_endpoint_similarity, flag_spurious_path_peak, emit)
    rel_kcal = ((np.asarray(energies) - float(np.min(energies))) * EV_TO_KCAL).tolist()
    findings = flag_spurious_path_peak(rel_kcal)
    if len(images) >= 3:
        findings += ts_endpoint_similarity(images[ts_idx], images[0], images[-1])
    warnings = emit(findings)

    from ase.io import write as ase_write  # noqa: PLC0415
    xyz = outdir / f"{tag}_path.xyz"
    ase_write(str(xyz), images, format="extxyz")
    barrier_fwd = (energies[ts_idx] - energies[0]) * EV_TO_KCAL
    log.info("%s done: %d images, Δ‡fwd = %.2f kcal/mol", tag.upper(),
             len(images), barrier_fwd)
    return {
        "status": drive["status"], "images": images, "ts": images[ts_idx],
        "ts_idx": ts_idx, "reactant": images[0], "product": images[-1],
        "energies_eV": energies, "barrier_fwd_kcal": barrier_fwd,
        "barrier_rev_kcal": (energies[ts_idx] - energies[-1]) * EV_TO_KCAL,
        "n_images_final": len(images), "warnings": warnings,
        "outputs": {"path_xyz": str(xyz)},
    }


def run_fsm(
    reactant: Atoms,
    product: Atoms,
    calculator_fn: Callable,
    outdir: str | Path = ".",
    charge: int = 0,
    mult: int = 1,
    n_images: int = 15,
    fmax: float = 0.05,
    max_nodes: int = 30,
    freeze_atoms=None,
    **kwargs,
) -> dict:
    """Freezing String Method.

    Args:
        reactant, product: endpoint Atoms
        calculator_fn: zero-arg callable returning a fresh ASE calculator
        outdir: output directory
        charge: system charge
        mult: spin multiplicity (2S+1) — threaded to the pysisyphus calc
        n_images: target number of images in the final path
        fmax: convergence criterion on orthogonal force (eV/Å)
        max_nodes: safety cap on node additions
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log.info("FSM: %d target images, fmax=%s eV/Å", n_images, fmax)
    try:
        geom_r = _atoms_to_pysis_geom(reactant, freeze_atoms=freeze_atoms)
        geom_p = _atoms_to_pysis_geom(product, freeze_atoms=freeze_atoms)
    except ImportError as e:
        log.error("  pysisyphus not available: %s", e)
        return {"status": "failed", "error": str(e), "outputs": {}}
    calc_getter = _make_pysis_calc_getter(calculator_fn, charge=charge, mult=mult)

    from quantum_engine.qm import pysis_string  # noqa: PLC0415
    try:
        drive = pysis_string.drive_freezing_string(
            geom_r, geom_p, calc_getter, max_nodes=max_nodes, fmax=fmax,
            max_cycles=200, out_dir=outdir)
    except Exception as e:  # noqa: BLE001
        log.error("  FSM run failed: %s", e)
        return {"status": "failed", "error": str(e), "outputs": {}}
    return _assemble_string_result(drive, calculator_fn, charge, outdir, "fsm")


def run_gsm(
    reactant: Atoms,
    product: Atoms,
    calculator_fn: Callable,
    outdir: str | Path = ".",
    charge: int = 0,
    mult: int = 1,
    n_images: int = 15,
    fmax: float = 0.045,
    freeze_atoms=None,
    **kwargs,
) -> dict:
    """Growing String Method.

    Similar interface to FSM but produces a fully-optimized string at the end.
    ``mult`` is the spin multiplicity (2S+1), threaded to the pysisyphus calc;
    ``fmax`` is in eV/Å.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log.info("GSM: %d target images, fmax=%s eV/Å", n_images, fmax)
    try:
        geom_r = _atoms_to_pysis_geom(reactant, freeze_atoms=freeze_atoms)
        geom_p = _atoms_to_pysis_geom(product, freeze_atoms=freeze_atoms)
    except ImportError as e:
        log.error("  pysisyphus not available: %s", e)
        return {"status": "failed", "error": str(e), "outputs": {}}
    calc_getter = _make_pysis_calc_getter(calculator_fn, charge=charge, mult=mult)

    from quantum_engine.qm import pysis_string  # noqa: PLC0415
    try:
        drive = pysis_string.drive_growing_string(
            geom_r, geom_p, calc_getter, max_nodes=n_images, fmax=fmax,
            max_cycles=300, out_dir=outdir)
    except Exception as e:  # noqa: BLE001
        log.error("  GSM run failed: %s", e)
        return {"status": "failed", "error": str(e), "outputs": {}}
    return _assemble_string_result(drive, calculator_fn, charge, outdir, "gsm")


def run(
    reactant: Atoms,
    product: Atoms,
    calculator_fn: Callable,
    outdir: str | Path = ".",
    method: str = "fsm",
    **kwargs,
) -> dict:
    """Unified entry point for FSM or GSM."""
    if method == "fsm":
        return run_fsm(reactant, product, calculator_fn, outdir, **kwargs)
    if method == "gsm":
        return run_gsm(reactant, product, calculator_fn, outdir, **kwargs)
    raise ValueError(f"Unknown method '{method}'. Choose 'fsm' or 'gsm'.")
