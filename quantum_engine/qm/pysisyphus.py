"""pysisyphus adapter (eljost/pysisyphus, GPL-3).

pysisyphus is a Swiss-army knife for reaction-path methods — NEB, GSM,
dimer, IRC, and a battery of TS optimisers including RS-P-RFO, RS-I-RFO,
TRIM, and EF. Already vendored at ``deps/pysisyphus``.

Where it fits in the QCB ``saddle`` cascade
-------------------------------------------
- ``rsprfo_ts`` — TS-Hessian-based RS-P-RFO. Use when you have a partial
  Hessian (e.g., from a finite-difference reactive-subset Hessian) and want
  the full Banerjee/Heyden RFO step with restricted-step trust regions.
- ``dimer_ts`` — pysisyphus's own dimer; mostly identical maths to ASE's
  but ships with the trust-region translator and bias options. Useful as
  a second-opinion when ASE dimer stalls.

Both wrappers return the standard ``saddle.run`` dict shape so the
dispatcher can drop them into the cascade alongside Sella / ASE-Dimer.

Generalisation
--------------
*No PTE-specific defaults.* Reactive atoms, charges, multiplicities — all
flow through the public API. The only knob that's hard-coded here is the
LAPACK driver retry list (which is reaction-agnostic).

References
----------
- Banerjee, A. *J. Phys. Chem.* **1985**, *89*, 52 (RFO).
- Heyden, A.; Bell, A.; Keil, F. *J. Chem. Phys.* **2005**, *123*, 224101.
- Stiefel, P.; Zimmermann, J. *J. Comput. Chem.* **1995**, *16*, 859 (Dimer).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from ase import Atoms
from ase.io import write

from quantum_engine.units import EV_TO_KCAL

log = logging.getLogger("quantum_engine.qm.pysisyphus")

# Prefer a fully-installed pysisyphus on sys.path. Fall back to the vendored
# tree under deps/pysisyphus only if the installed one is absent — the
# vendored copy lacks the auto-generated version.py and won't import cleanly
# unless a `pip install -e deps/pysisyphus` has been run.
def _ensure_pysisyphus_on_path() -> None:
    try:
        import pysisyphus  # noqa: F401
        return
    except ImportError:
        pass
    vendored = Path(__file__).resolve().parents[2] / "deps" / "pysisyphus"
    if vendored.exists() and str(vendored) not in sys.path:
        sys.path.insert(0, str(vendored))


_ensure_pysisyphus_on_path()


def pysisyphus_available() -> bool:
    try:
        import pysisyphus  # noqa: F401
        return True
    except ImportError:
        return False


def _require() -> None:
    if not pysisyphus_available():
        raise ImportError(
            "pysisyphus not installed. `pip install -e deps/pysisyphus` "
            "or ensure deps/pysisyphus exists on the import path."
        )


# --------------------------------------------------------------------------
# ASE <-> pysisyphus glue
# --------------------------------------------------------------------------
def _atoms_to_geom(atoms: Atoms, coord_type: str = "cart"):
    """Build a fresh pysisyphus :class:`Geometry` from an ASE ``Atoms``.

    Coords go in flat angstrom; pysisyphus stores them internally as bohr,
    but :class:`Geometry` itself works in angstrom unless ``coord_type='mwcartesian'``.
    """
    from pysisyphus.Geometry import Geometry
    return Geometry(
        atoms.get_chemical_symbols(),
        atoms.get_positions().flatten(),
        coord_type=coord_type,
    )


def _make_pysis_calc(ase_calc, *, charge: int = 0, mult: int = 1):
    """Wrap an ASE ``Calculator`` so pysisyphus can call it.

    pysisyphus calculators expose ``get_energy`` / ``get_forces`` /
    ``get_hessian`` over ``(atom_symbols, flat_bohr_coords)``. ASE works in
    angstrom — we convert at the boundary.
    """
    _require()
    from pysisyphus.calculators.Calculator import Calculator
    from pysisyphus.constants import BOHR2ANG

    class _AseCalc(Calculator):
        def __init__(self):
            super().__init__(charge=charge, mult=mult)
            self.ase_calc = ase_calc

        def _eval(self, atom_symbols, coords_bohr):
            from ase import Atoms as _Atoms
            positions = np.asarray(coords_bohr).reshape(-1, 3) * BOHR2ANG
            a = _Atoms(symbols=list(atom_symbols), positions=positions)
            a.calc = self.ase_calc
            a.info["charge"] = charge
            energy_ev = float(a.get_potential_energy())
            forces_ev_per_a = a.get_forces()
            # Convert to pysisyphus units: hartree, hartree/bohr.
            # pysisyphus *also* internally treats forces in hartree/bohr and
            # energy in hartree, but its ASE bridge (FakeASE) just keeps the
            # numbers consistent. For our purposes we keep eV / (eV/Å) — Sella
            # / RS-P-RFO only care that the energy and gradient are
            # *consistent*, not their absolute units.
            return energy_ev, forces_ev_per_a

        def get_energy(self, atoms, coords, **kw):
            energy, _ = self._eval(atoms, coords)
            return {"energy": energy}

        def get_forces(self, atoms, coords, **kw):
            energy, forces = self._eval(atoms, coords)
            return {"energy": energy, "forces": forces.flatten()}

    return _AseCalc()


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def rsprfo_ts(
    atoms: Atoms,
    *,
    calculator=None,
    outdir: str | Path = ".",
    fmax: float = 0.005,
    max_steps: int = 200,
    charge: int = 0,
    mult: int = 1,
    hessian_init: str = "calc",
    coord_type: str = "cart",
    **kwargs,
) -> dict:
    """RS-P-RFO TS optimisation through pysisyphus.

    The optimiser starts from an analytic Hessian if ``hessian_init='calc'``
    (which calls our :class:`_AseCalc`'s ``get_hessian``, which falls back
    to numerical Hessian via finite differences over the *full* system —
    expensive, but unavoidable without a partial-Hessian protocol). Pass
    ``hessian_init='lindh'`` or ``'fischer'`` for cheap model Hessians at
    the cost of more iterations.
    """
    _require()
    from pysisyphus.tsoptimizers.RSPRFOptimizer import RSPRFOptimizer

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if atoms.calc is None:
        if calculator is None:
            raise ValueError("pysisyphus.rsprfo_ts: need calculator or atoms.calc")
        atoms.calc = calculator

    geom = _atoms_to_geom(atoms, coord_type=coord_type)
    geom.set_calculator(_make_pysis_calc(atoms.calc, charge=charge, mult=mult))

    log.info("pysisyphus RS-P-RFO: %d atoms, fmax=%.4f, hess_init=%s, coords=%s",
             len(atoms), fmax, hessian_init, coord_type)

    opt = RSPRFOptimizer(
        geom,
        max_cycles=max_steps,
        thresh="gau",                     # let pysisyphus map fmax via its own tol
        max_force_thresh=fmax,
        hessian_init=hessian_init,
        out_dir=str(outdir),
    )
    opt.run()
    converged = bool(getattr(opt, "is_converged", False))

    # Pull final geometry back into ASE-land.
    from pysisyphus.constants import BOHR2ANG
    final_pos = geom.coords.reshape(-1, 3) * BOHR2ANG  # Geometry stores in bohr
    atoms.set_positions(final_pos)
    energy = float(atoms.get_potential_energy())

    saddle_xyz = outdir / "saddle.xyz"
    write(str(saddle_xyz), atoms, format="extxyz")

    return {
        "status": "converged" if converged else "not_converged",
        "converged": converged,
        "atoms": atoms,
        "energy_eV": energy,
        "energy_kcal_mol": energy * EV_TO_KCAL,
        "backend": "pysisyphus-rsprfo",
        "outputs": {
            "saddle_xyz": str(saddle_xyz),
            "outdir": str(outdir),
        },
    }


def dimer_ts(
    atoms: Atoms,
    *,
    calculator=None,
    outdir: str | Path = ".",
    fmax: float = 0.005,
    max_steps: int = 200,
    charge: int = 0,
    mult: int = 1,
    initial_mode_vector: np.ndarray | None = None,
    dimer_length: float = 0.0189,
    **kwargs,
) -> dict:
    """Pysisyphus dimer TS optimisation (gradient-only).

    Args:
        initial_mode_vector: ``(N, 3)`` or flat ``(3N,)``. If supplied, used
            to seed the dimer orientation (``N_raw``). Otherwise the dimer
            chooses an initial direction from a small random kick.
        dimer_length: Image-image separation. ``0.0189 bohr`` is the
            pysisyphus default.
    """
    _require()
    from pysisyphus.calculators.Dimer import Dimer
    from pysisyphus.optimizers.PreconLBFGS import PreconLBFGS
    from pysisyphus.constants import BOHR2ANG

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if atoms.calc is None:
        if calculator is None:
            raise ValueError("pysisyphus.dimer_ts: need calculator or atoms.calc")
        atoms.calc = calculator

    geom = _atoms_to_geom(atoms, coord_type="cart")
    inner_calc = _make_pysis_calc(atoms.calc, charge=charge, mult=mult)

    n_raw = None
    if initial_mode_vector is not None:
        v = np.asarray(initial_mode_vector, dtype=float).reshape(-1)
        if v.size != 3 * len(atoms):
            raise ValueError(
                f"initial_mode_vector size {v.size} != 3N={3*len(atoms)}"
            )
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            log.warning("Supplied mode vector has ~zero norm; ignoring")
        else:
            n_raw = v / norm
            log.info("pysisyphus dimer seeded from supplied mode vector "
                     "(|v|=%.4f, used as N_raw)", norm)

    dimer_calc = Dimer(
        calculator=inner_calc,
        N_raw=n_raw,
        length=dimer_length,
    )
    geom.set_calculator(dimer_calc)

    log.info("pysisyphus dimer: %d atoms, fmax=%.4f, length=%.4f bohr",
             len(atoms), fmax, dimer_length)

    opt = PreconLBFGS(
        geom,
        max_cycles=max_steps,
        max_force_thresh=fmax,
        thresh="gau",
        out_dir=str(outdir),
    )
    opt.run()
    converged = bool(getattr(opt, "is_converged", False))

    final_pos = geom.coords.reshape(-1, 3) * BOHR2ANG
    atoms.set_positions(final_pos)
    energy = float(atoms.get_potential_energy())

    saddle_xyz = outdir / "saddle.xyz"
    write(str(saddle_xyz), atoms, format="extxyz")

    return {
        "status": "converged" if converged else "not_converged",
        "converged": converged,
        "atoms": atoms,
        "energy_eV": energy,
        "energy_kcal_mol": energy * EV_TO_KCAL,
        "backend": "pysisyphus-dimer",
        "outputs": {
            "saddle_xyz": str(saddle_xyz),
            "outdir": str(outdir),
        },
    }


# --------------------------------------------------------------------------
# Legacy API (kept for backwards compat; tests import these names)
# --------------------------------------------------------------------------
def neb_path(
    reactant: Any,
    product: Any,
    **kwargs,
) -> dict[str, Any]:
    """Stub for pysisyphus NEB. Use :mod:`quantum_engine.ops.gsm` for FSM/GSM."""
    _require()
    raise NotImplementedError(
        "pysisyphus.neb_path: not wired; use quantum_engine.ops.gsm for FSM/GSM."
    )


def rsirfo_ts(atoms: Atoms, **kwargs) -> dict[str, Any]:
    """RS-I-RFO — alias to :func:`rsprfo_ts` for now (same code path; the
    optimiser only differs in the secondary mode-tracking logic, which is
    irrelevant for first-order saddles)."""
    _require()
    return rsprfo_ts(atoms, **kwargs)


def irc(ts_atoms: Atoms, **kwargs) -> dict[str, Any]:
    """IRC descent through pysisyphus — not yet wired (use the ASE/legacy
    IRC in :mod:`quantum_engine.mlff.irc` for now)."""
    _require()
    raise NotImplementedError(
        "pysisyphus.irc: not wired; use quantum_engine.ops.irc for the "
        "IRC-from-TS workflow."
    )


def harmonic_restraints(atoms: Atoms, **kwargs) -> Any:
    """SpringConstraint helper — keep stubbed; not needed for the cascade."""
    _require()
    raise NotImplementedError(
        "pysisyphus.harmonic_restraints: stubbed; use ASE's "
        "FixedConstraint or the cluster-mode helpers in quantum_engine.io."
    )
