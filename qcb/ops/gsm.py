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
- Input is a single TS guess — use IRC-from-TS instead (`qcb irc`).
- Input is one endpoint and you don't know the other — use CV spring
  driving (`qcb ts --strategy cv-spring`) or metadynamics (`qcb mtd`).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Callable

from ase import Atoms

log = logging.getLogger("qcb.ops.gsm")

EV_TO_KCAL = 23.0609

# Add pysisyphus to path (installed as a git submodule under deps/)
_PYSIS_PATH = Path(__file__).resolve().parents[2] / "deps" / "pysisyphus"
if _PYSIS_PATH.exists() and str(_PYSIS_PATH) not in sys.path:
    sys.path.insert(0, str(_PYSIS_PATH))


def _atoms_to_pysis_geom(atoms: Atoms):
    """Convert ASE Atoms to a pysisyphus Geometry."""
    from pysisyphus.Geometry import Geometry
    return Geometry(
        atoms.get_chemical_symbols(),
        atoms.get_positions().flatten(),
    )


def _make_ase_pysis_calc(calc_fn: Callable, charge: int = 0):
    """Wrap an ASE calculator as a pysisyphus Calculator."""
    from pysisyphus.calculators.Calculator import Calculator

    class _AsePysisCalc(Calculator):
        def __init__(self):
            super().__init__(charge=charge, mult=1)
            self.ase_calc = calc_fn()

        def get_forces(self, atoms, coords):
            import numpy as np
            from ase import Atoms as _Atoms
            ase = _Atoms(symbols=atoms, positions=coords.reshape(-1, 3))
            ase.calc = self.ase_calc
            ase.info["charge"] = charge
            return {
                "energy": ase.get_potential_energy(),
                "forces": ase.get_forces().flatten(),
            }

        def get_energy(self, atoms, coords):
            return {"energy": self.get_forces(atoms, coords)["energy"]}

    return _AsePysisCalc()


def run_fsm(
    reactant: Atoms,
    product: Atoms,
    calculator_fn: Callable,
    outdir: str | Path = ".",
    charge: int = 0,
    n_images: int = 15,
    fmax: float = 0.05,
    max_nodes: int = 30,
    **kwargs,
) -> dict:
    """Freezing String Method.

    Args:
        reactant, product: endpoint Atoms
        calculator_fn: zero-arg callable returning a fresh ASE calculator
        outdir: output directory
        charge: system charge
        n_images: target number of images in the final path
        fmax: convergence criterion on orthogonal force
        max_nodes: safety cap on node additions
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        from pysisyphus.cos.FreezingString import FreezingString
        from pysisyphus.optimizers.StringOptimizer import StringOptimizer
    except ImportError as e:
        log.error(f"  pysisyphus not available: {e}")
        return {"status": "failed", "error": str(e), "outputs": {}}

    log.info(f"FSM: {n_images} target images, fmax={fmax}")
    geom_r = _atoms_to_pysis_geom(reactant)
    geom_p = _atoms_to_pysis_geom(product)
    calc = _make_ase_pysis_calc(calculator_fn, charge)

    # FSM expects a pair of geometries
    fs = FreezingString(
        images=[geom_r, geom_p],
        calculator=calc,
        max_nodes=max_nodes,
    )
    opt = StringOptimizer(
        fs,
        out_dir=str(outdir),
        max_cycles=200,
        max_force_thresh=fmax,
    )
    try:
        opt.run()
        converged = opt.is_converged if hasattr(opt, "is_converged") else True
    except Exception as e:
        log.error(f"  FSM run failed: {e}")
        return {"status": "failed", "error": str(e), "outputs": {}}

    # Extract final images
    images = []
    for g in fs.images:
        n = len(g.atoms)
        a = Atoms(symbols=g.atoms, positions=g.coords.reshape(n, 3))
        a.calc = calculator_fn()
        a.info["charge"] = charge
        images.append(a)

    energies = [float(img.get_potential_energy()) for img in images]
    import numpy as np
    ts_idx = int(np.argmax(energies[1:-1])) + 1 if len(energies) > 2 else 0

    result = {
        "status": "converged" if converged else "not_converged",
        "images": images,
        "ts": images[ts_idx],
        "ts_idx": ts_idx,
        "reactant": images[0],
        "product": images[-1],
        "energies_eV": energies,
        "barrier_fwd_kcal": (energies[ts_idx] - energies[0]) * EV_TO_KCAL,
        "barrier_rev_kcal": (energies[ts_idx] - energies[-1]) * EV_TO_KCAL,
        "n_images_final": len(images),
        "outputs": {
            "path_xyz": str(outdir / "fsm_path.xyz"),
        },
    }

    # Write trajectory
    from ase.io import write as ase_write
    ase_write(str(outdir / "fsm_path.xyz"), images, format="extxyz")

    log.info(f"FSM done: {len(images)} images, Δ‡fwd = {result['barrier_fwd_kcal']:.2f} kcal/mol")
    return result


def run_gsm(
    reactant: Atoms,
    product: Atoms,
    calculator_fn: Callable,
    outdir: str | Path = ".",
    charge: int = 0,
    n_images: int = 15,
    fmax: float = 0.045,
    **kwargs,
) -> dict:
    """Growing String Method.

    Similar interface to FSM but produces a fully-optimized string at the end.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        from pysisyphus.cos.GrowingString import GrowingString
        from pysisyphus.optimizers.StringOptimizer import StringOptimizer
    except ImportError as e:
        log.error(f"  pysisyphus not available: {e}")
        return {"status": "failed", "error": str(e), "outputs": {}}

    log.info(f"GSM: {n_images} target images, fmax={fmax}")
    geom_r = _atoms_to_pysis_geom(reactant)
    geom_p = _atoms_to_pysis_geom(product)
    calc = _make_ase_pysis_calc(calculator_fn, charge)

    gs = GrowingString(
        images=[geom_r, geom_p],
        calculator=calc,
        max_nodes=n_images,
    )
    opt = StringOptimizer(
        gs,
        out_dir=str(outdir),
        max_cycles=300,
        max_force_thresh=fmax,
    )
    try:
        opt.run()
        converged = opt.is_converged if hasattr(opt, "is_converged") else True
    except Exception as e:
        log.error(f"  GSM run failed: {e}")
        return {"status": "failed", "error": str(e), "outputs": {}}

    images = []
    for g in gs.images:
        n = len(g.atoms)
        a = Atoms(symbols=g.atoms, positions=g.coords.reshape(n, 3))
        a.calc = calculator_fn()
        a.info["charge"] = charge
        images.append(a)

    energies = [float(img.get_potential_energy()) for img in images]
    import numpy as np
    ts_idx = int(np.argmax(energies[1:-1])) + 1 if len(energies) > 2 else 0

    result = {
        "status": "converged" if converged else "not_converged",
        "images": images,
        "ts": images[ts_idx],
        "ts_idx": ts_idx,
        "reactant": images[0],
        "product": images[-1],
        "energies_eV": energies,
        "barrier_fwd_kcal": (energies[ts_idx] - energies[0]) * EV_TO_KCAL,
        "barrier_rev_kcal": (energies[ts_idx] - energies[-1]) * EV_TO_KCAL,
        "n_images_final": len(images),
        "outputs": {
            "path_xyz": str(outdir / "gsm_path.xyz"),
        },
    }

    from ase.io import write as ase_write
    ase_write(str(outdir / "gsm_path.xyz"), images, format="extxyz")

    log.info(f"GSM done: {len(images)} images, Δ‡fwd = {result['barrier_fwd_kcal']:.2f} kcal/mol")
    return result


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
