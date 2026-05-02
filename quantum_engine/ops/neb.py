"""Nudged Elastic Band (NEB) with CI-NEB climbing image.

Accepts reactant and product endpoints and runs a standard NEB → CI-NEB
procedure. Uses geodesic interpolation by default (Zhu et al. JCP 2019,
doi:10.1063/1.5090303).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
from quantum_engine.units import EV_TO_KCAL
from ase import Atoms
from ase.io import read, write
from ase.mep.neb import NEB, NEBOptimizer

from quantum_engine.mlff.interpolation import interpolate

log = logging.getLogger("quantum_engine.ops.neb")



def run(
    reactant: Atoms,
    product: Atoms,
    calculator_fn: Callable,
    outdir: str | Path = ".",
    constraint=None,
    n_images: int = 15,
    k_spring: float = 1.0,
    interpolation_method: str = "geodesic",
    fmax_noclimb: float = 0.40,
    steps_noclimb: int = 200,
    fmax_climb: float = 0.045,
    steps_climb: int = 250,
    charge: int = 0,
    **kwargs,
) -> dict:
    """Run NEB → CI-NEB.

    Args:
        reactant, product: endpoint Atoms objects (with calc set)
        calculator_fn: zero-arg callable that returns a fresh calculator
                       (each image needs its own instance)
        outdir: output directory
        constraint: ASE constraint applied to all images
        n_images: total number of images (including endpoints)
        k_spring: NEB spring constant (eV/Å)
        interpolation_method: "geodesic" (default), "idpp", or "linear"
        fmax_noclimb, steps_noclimb: no-climb phase settings
        fmax_climb, steps_climb: climbing-image phase settings (set steps_climb=0 to skip)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Check for cached result
    climb_file = outdir / "path-neb-climb.xyz"
    if climb_file.exists():
        log.info(f"  NEB climb path found at {climb_file} – loading")
        images = read(str(climb_file), index=":")
        for img in images:
            img.calc = calculator_fn()
            img.info["charge"] = charge
        return _summarize(images, outdir, "cached")

    log.info(f"NEB: {n_images} images, k={k_spring}, interp={interpolation_method}")

    # Interpolate
    images = interpolate(reactant, product, n_images, method=interpolation_method)

    # Assign constraints and calculators
    for img in images:
        if constraint is not None:
            img.set_constraint(constraint if isinstance(constraint, list) else [constraint])
        img.calc = calculator_fn()
        img.info["charge"] = charge

    neb = NEB(images, k=k_spring, allow_shared_calculator=False, method="improvedtangent")

    write(str(outdir / "path-neb-init.xyz"), images)

    # Stage 1: no-climb
    log.info(f"  NEB no-climb: fmax={fmax_noclimb}, max {steps_noclimb} steps")
    optimizer = NEBOptimizer(neb, trajectory=str(outdir / "neb-opt.traj"), method="ODE")
    optimizer.run(fmax=fmax_noclimb, steps=steps_noclimb)
    write(str(outdir / "path-neb-noclimb.xyz"), images)
    log.info("  NEB no-climb complete")

    # Stage 2: climbing image
    if steps_climb > 0:
        neb.climb = True
        log.info(f"  NEB climbing: fmax={fmax_climb}, max {steps_climb} steps")
        optimizer.run(fmax=fmax_climb, steps=steps_climb)
        write(str(climb_file), images)
        log.info("  NEB climbing complete")
    else:
        log.info("  Climbing phase skipped")
        write(str(climb_file), images)

    return _summarize(images, outdir, "converged")


def _summarize(images, outdir, status):
    energies = [float(img.get_potential_energy()) for img in images]
    e_r, e_p = energies[0], energies[-1]
    inner = energies[1:-1]
    ts_idx = int(np.argmax(inner)) + 1 if inner else 0
    e_ts = energies[ts_idx]

    barrier_fwd = (e_ts - e_r) * EV_TO_KCAL
    barrier_rev = (e_ts - e_p) * EV_TO_KCAL
    dE_rxn = (e_p - e_r) * EV_TO_KCAL

    # Save energy profile
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        rel = [(e - e_r) * EV_TO_KCAL for e in energies]
        ax.plot(range(len(rel)), rel, "o-", markersize=6)
        ax.set_xlabel("Image")
        ax.set_ylabel("Relative energy (kcal/mol)")
        ax.set_title(f"NEB path (Δ‡fwd = {barrier_fwd:.1f} kcal/mol)")
        ax.grid(alpha=0.3)
        png = outdir / "neb-profile.png"
        fig.tight_layout()
        fig.savefig(str(png), dpi=150)
        plt.close(fig)
    except Exception:
        png = None

    return {
        "status": status,
        "images": images,
        "ts": images[ts_idx],
        "ts_idx": ts_idx,
        "reactant": images[0],
        "product": images[-1],
        "energies_eV": energies,
        "barrier_fwd_kcal": float(barrier_fwd),
        "barrier_rev_kcal": float(barrier_rev),
        "dE_rxn_kcal": float(dE_rxn),
        "outputs": {
            "path": str(outdir / "path-neb-climb.xyz"),
            "plot": str(png) if png else None,
        },
    }
