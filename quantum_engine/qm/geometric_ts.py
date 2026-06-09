"""geomeTRIC transition-state optimizer (internal-coordinate saddle refinement).

geomeTRIC (Wang & Song) does TS optimization in robust translation-rotation
internal coordinates (TRIC), which is excellent for large / floppy / weakly-bound
systems where Cartesian eigenvector-following (Sella) struggles. Wired as a
SADDLE backend ``geometric`` (``ops/saddle.py``).

geomeTRIC is NOT bundled in the cowboy-qc container; this adapter imports it lazily, so
selecting ``geometric`` without it installed raises a clean ImportError (the
saddle layer surfaces it like any other missing-dep backend). Install with
``pip install geometric``; the integration is unit-correct (verified against a
known xTB HCN saddle) and auto-works once geomeTRIC is importable.

Units: geomeTRIC is atomic units — it passes coordinates in BOHR and expects
energy in HARTREE, gradient in HARTREE/BOHR. ASE is Å / eV / (eV/Å); we convert
at the boundary (the same factors verified for the pysisyphus adapter).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase import Atoms

from quantum_engine.logging_utils import get_logger
from quantum_engine.units import EV_TO_KCAL

log = get_logger("qm.geometric")

# CODATA 2018
_HARTREE_EV = 27.211386245988
_BOHR_ANG = 0.529177210903
_EVANG2AUBOHR = _BOHR_ANG / _HARTREE_EV   # eV/Å -> hartree/bohr  (= 0.0194469)


def geometric_available() -> tuple[bool, str]:
    try:
        import geometric  # noqa: F401, PLC0415
        return True, f"geometric {getattr(geometric, '__version__', '?')}"
    except Exception as exc:  # noqa: BLE001
        return False, f"geometric not installed: {exc}"


def _fmax_to_converge(fmax: float) -> list:
    """Map an fmax (eV/Å) to a geomeTRIC GAU* preset (passed as ['set', NAME])."""
    gmax_au = float(fmax) * _EVANG2AUBOHR
    if gmax_au >= 1.5e-3:
        name = "GAU_LOOSE"
    elif gmax_au >= 2.0e-4:
        name = "GAU"
    else:
        name = "GAU_TIGHT"
    return ["set", name]


def run(atoms: Atoms, calculator=None, *, outdir: str | Path = ".", constraint=None,
        fmax: float = 0.02, max_steps: int = 300, coordsys: str = "tric",
        **kwargs) -> dict:
    """Refine ``atoms`` to a first-order saddle with geomeTRIC (``transition=True``).

    Returns the standard ``saddle.run`` result dict.
    """
    ok, reason = geometric_available()
    if not ok:
        # Surface as ImportError so the saddle layer treats it like any other
        # missing-dependency backend (clean, expected).
        raise ImportError(
            f"saddle backend 'geometric' requires geomeTRIC: {reason}. "
            "Install with `pip install geometric`.")

    import contextlib  # noqa: PLC0415
    import os  # noqa: PLC0415

    from geometric.engine import Engine  # noqa: PLC0415
    from geometric.molecule import Molecule  # noqa: PLC0415
    from geometric.optimize import run_optimizer  # noqa: PLC0415

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    if atoms.calc is None:
        if calculator is None:
            raise ValueError("geometric_ts.run: need calculator or atoms.calc")
        atoms.calc = calculator
    calc = atoms.calc
    syms = atoms.get_chemical_symbols()
    charge = int(atoms.info.get("charge", 0))
    log.info("geomeTRIC TS: %d atoms, coordsys=%s, fmax=%.4f eV/Å", len(atoms),
             coordsys, fmax)

    class _ASEEngine(Engine):
        def calc_new(self, coords, dirname):
            a = Atoms(symbols=syms,
                      positions=np.asarray(coords, float).reshape(-1, 3) * _BOHR_ANG)
            a.info["charge"] = charge
            a.calc = calc
            energy_ha = float(a.get_potential_energy()) / _HARTREE_EV
            gradient = (-np.asarray(a.get_forces(), float).flatten()) * _EVANG2AUBOHR
            return {"energy": energy_ha, "gradient": gradient}

    mol = Molecule()
    mol.elem = list(syms)
    mol.xyzs = [atoms.get_positions()]
    engine = _ASEEngine(mol)

    final_positions = None
    converged = False
    try:
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            out = run_optimizer(
                customengine=engine, transition=True, coordsys=coordsys,
                input=str(outdir / "geometric_ts"), maxiter=max_steps,
                converge=_fmax_to_converge(fmax))
        final_positions = np.asarray(out.xyzs[-1], float)
        converged = True
    except Exception as exc:  # noqa: BLE001 — incl. geomeTRIC non-convergence
        log.warning("geomeTRIC did not converge cleanly (%s); using best geometry", exc)
        # geomeTRIC writes the optimization trajectory; recover the last frame.
        traj = outdir / "geometric_ts_optim.xyz"
        if traj.exists():
            from ase.io import read as ase_read  # noqa: PLC0415
            final_positions = ase_read(str(traj), index=-1).get_positions()
        else:
            final_positions = atoms.get_positions()

    ts_atoms = atoms.copy()
    ts_atoms.set_positions(final_positions)
    ts_atoms.info["charge"] = charge
    ts_atoms.calc = calc
    energy = float(ts_atoms.get_potential_energy())

    from ase.io import write as ase_write  # noqa: PLC0415
    saddle_xyz = outdir / "saddle.xyz"
    ase_write(str(saddle_xyz), ts_atoms, format="extxyz")
    log.info("geomeTRIC TS: %s, E = %.6f eV",
             "converged" if converged else "not converged", energy)
    return {
        "status": "converged" if converged else "not_converged",
        "converged": converged, "atoms": ts_atoms,
        "energy_eV": energy, "energy_kcal_mol": energy * EV_TO_KCAL,
        "backend": "geometric",
        "outputs": {"saddle_xyz": str(saddle_xyz), "outdir": str(outdir)},
    }


__all__ = ["geometric_available", "run"]
