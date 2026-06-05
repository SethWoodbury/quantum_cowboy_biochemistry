"""AutoNEB — adaptive-image NEB (ASE-native).

A double-ended path method that starts from a small seed band and *adaptively
inserts images* where the path is poorly resolved (high energy / large spacing),
converging the minimum-energy path with fewer total force calls than a fixed-image
CI-NEB. Pure ASE (``ase.mep.AutoNEB``) — no extra dependency.

Wired as a registry path method ``autoneb`` (``ops/path_search.py``); it takes
the standard ``(reactant, product, calculator_fn, ...)`` contract and returns the
standard path-result dict, so it drops in anywhere ``neb``/``gsm-de`` do. Like
the other string/NEB backends, energies are (re)computed via the ASE calculator
in eV and the peak image is the TS *guess* — the saddle+Hessian+IRC gate remains
the acceptance authority.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from ase import Atoms

from quantum_engine.logging_utils import get_logger
from quantum_engine.units import EV_TO_KCAL

log = get_logger("ops.autoneb")


def run(reactant: Atoms, product: Atoms, calculator_fn: Callable[[], Any], *,
        outdir: str | Path = ".", constraint=None, charge: int = 0,
        n_images: int = 11, n_seed: int = 3, n_simul: int = 1,
        fmax: float = 0.05, k: float = 0.1, climb: bool = True,
        optimizer: str = "FIRE", maxsteps: int = 200,
        interpolation_method: str = "geodesic", **kwargs) -> dict:
    """Adaptive-image NEB from ``reactant`` to ``product``.

    Args:
        n_images: maximum number of images AutoNEB may grow to (``n_max``).
        n_seed: initial band size to seed (>=3; AutoNEB inserts the rest).
        n_simul: images optimised simultaneously per AutoNEB cycle.
        fmax, k, climb, optimizer, maxsteps: NEB knobs (climbing image on by default).
        interpolation_method: seed interpolation (``geodesic``/``idpp``/``linear``);
            AutoNEB's own re-interpolation uses idpp.
    Returns the standard path-result dict (status/images/ts/ts_idx/energies_eV/
    barrier_*_kcal/outputs).
    """
    import ase.optimize as ase_opt  # noqa: PLC0415
    from ase.io import write as ase_write  # noqa: PLC0415
    from ase.mep import AutoNEB  # noqa: PLC0415

    from quantum_engine.mlff.interpolation import interpolate  # noqa: PLC0415

    # ASE deprecates passing the optimizer as a string — resolve to the class.
    opt_cls = getattr(ase_opt, optimizer) if isinstance(optimizer, str) else optimizer

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    n_seed = max(3, int(n_seed))
    n_images = max(n_seed, int(n_images))
    log.info("AutoNEB: seed=%d → n_max=%d, fmax=%s, climb=%s", n_seed, n_images,
             fmax, climb)

    # ---- seed band (geodesic) written to <prefix>NNN.traj WITH energies ----
    # AutoNEB reads the seed images back and needs their energies stored (it
    # picks the highest-energy node), so we compute each before writing.
    seed = interpolate(reactant, product, n_seed, method=interpolation_method)
    prefix = str(outdir / "autoneb")
    for i, img in enumerate(seed):
        img = img.copy()
        img.info["charge"] = charge
        if constraint is not None:
            img.set_constraint(constraint)
        img.calc = calculator_fn()
        img.get_potential_energy()          # store energy in the traj
        ase_write(f"{prefix}{i:03d}.traj", img)

    def attach_calculators(images):
        for im in images:
            im.info["charge"] = charge
            if constraint is not None:
                im.set_constraint(constraint)
            im.calc = calculator_fn()

    def _drive(climb_flag: bool):
        neb = AutoNEB(
            attach_calculators, prefix=prefix, n_simul=n_simul, n_max=n_images,
            fmax=fmax, maxsteps=maxsteps, k=k, climb=climb_flag, method="eb",
            optimizer=opt_cls, interpolate_method="idpp",
            iter_folder=str(outdir / "AutoNEB_iter"))
        neb.run()
        return list(getattr(neb, "all_images", []))

    try:
        band = _drive(climb)
    except AssertionError as exc:
        # ASE 3.28's AutoNEB can hit "climb_safe should be true at this point!"
        # on short/easy bands. The non-climbing adaptive band's peak is still a
        # valid TS guess (our saddle refiner does the precise climb), so retry.
        if climb and "climb_safe" in str(exc):
            log.warning("AutoNEB climbing hit the ASE climb_safe assertion; "
                        "retrying without climbing (the peak is still a TS guess).")
            try:
                band = _drive(False)
            except Exception as exc2:  # noqa: BLE001
                log.error("AutoNEB (no-climb retry) failed: %s", exc2)
                return {"status": "failed", "error": str(exc2), "images": [],
                        "outputs": {}}
        else:
            log.error("AutoNEB failed: %s", exc)
            return {"status": "failed", "error": str(exc), "images": [], "outputs": {}}
    except Exception as exc:  # noqa: BLE001 — surface a clean status, never crash
        log.error("AutoNEB failed: %s", exc)
        return {"status": "failed", "error": str(exc), "images": [], "outputs": {}}

    if not band:
        return {"status": "failed", "error": "AutoNEB produced no images",
                "images": [], "outputs": {}}

    images = []
    for im in band:
        a = Atoms(symbols=im.get_chemical_symbols(), positions=im.get_positions())
        a.info["charge"] = charge
        a.calc = calculator_fn()
        images.append(a)
    energies = [float(im.get_potential_energy()) for im in images]
    ts_idx = int(np.argmax(energies[1:-1])) + 1 if len(energies) > 2 else 0

    xyz = outdir / "autoneb_path.xyz"
    ase_write(str(xyz), images, format="extxyz")
    barrier_fwd = (energies[ts_idx] - energies[0]) * EV_TO_KCAL
    log.info("AutoNEB done: %d images, Δ‡fwd = %.2f kcal/mol", len(images), barrier_fwd)
    return {
        "status": "converged", "images": images, "ts": images[ts_idx],
        "ts_idx": ts_idx, "reactant": images[0], "product": images[-1],
        "energies_eV": energies, "barrier_fwd_kcal": barrier_fwd,
        "barrier_rev_kcal": (energies[ts_idx] - energies[-1]) * EV_TO_KCAL,
        "n_images_final": len(images), "outputs": {"path_xyz": str(xyz)},
    }


__all__ = ["run"]
