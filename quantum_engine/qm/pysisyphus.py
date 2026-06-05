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
# ASE <-> pysisyphus glue (UNIT-RIGOROUS — see units note below)
# --------------------------------------------------------------------------
# pysisyphus is atomic-units throughout: a :class:`Geometry` stores cartesian
# coordinates in BOHR (its constructor does NOT convert — whatever you pass is
# treated as bohr), and calculators are called with coords in BOHR and must
# return energy in HARTREE and forces in HARTREE/BOHR. ASE is Å / eV / (eV/Å).
# We therefore convert at BOTH boundaries:
#   * ASE positions (Å)  -> Geometry coords:  × ANG2BOHR
#   * pysisyphus coords (bohr) -> ASE atoms:   × BOHR2ANG
#   * ASE energy (eV)   -> pysisyphus:         × (1/AU2EV)
#   * ASE forces (eV/Å) -> pysisyphus:         × EVANG2AUBOHR  (= 0.019447)
#   * fmax threshold (eV/Å) -> nearest named ``thresh`` preset (see _fmax_to_thresh)
# This is shared by the saddle backends here AND ops/gsm.py (one adapter, no
# divergence). Getting the model Hessians / trust radii / Gaussian thresholds
# right requires the coords/energy/forces to be in genuine atomic units.

def _atoms_to_geom(atoms: Atoms, coord_type: str = "cart"):
    """Build a fresh pysisyphus :class:`Geometry` (BOHR) from an ASE ``Atoms`` (Å)."""
    from pysisyphus.Geometry import Geometry
    from pysisyphus.constants import ANG2BOHR
    return Geometry(
        atoms.get_chemical_symbols(),
        atoms.get_positions().flatten() * ANG2BOHR,   # Å -> bohr
        coord_type=coord_type,
    )


# pysisyphus optimizers take a NAMED convergence preset (``thresh=``), not a
# scalar force tolerance — there is no ``max_force_thresh`` kwarg. We therefore
# map the caller's fmax (eV/Å) to the preset whose max-force criterion (in
# hartree/bohr) is multiplicatively closest. Presets (max_force, hartree/bohr):
_PYSIS_THRESH_PRESETS: tuple[tuple[str, float], ...] = (
    ("gau_vtight", 2.0e-6),
    ("gau_tight", 1.5e-5),
    ("baker", 3.0e-4),
    ("gau", 4.5e-4),
    ("gau_loose", 2.5e-3),
    ("nwchem_loose", 4.5e-3),
)


def _fmax_to_thresh(fmax_ev_ang: float) -> str:
    """Map an fmax in eV/Å to the closest pysisyphus ``thresh`` preset name."""
    from pysisyphus.constants import EVANG2AUBOHR
    target_au = max(float(fmax_ev_ang) * EVANG2AUBOHR, 1e-12)
    log_t = np.log(target_au)
    name, _ = min(_PYSIS_THRESH_PRESETS,
                  key=lambda kv: abs(np.log(kv[1]) - log_t))
    return name


_ASE_PYSIS_CALC_CLS = None


def _import_pysis_calculator_base():
    """Import pysisyphus's base ``Calculator`` class robustly.

    ``pysisyphus.calculators.__init__`` eagerly imports EVERY calculator,
    including ``Remote`` → ``paramiko`` → ``cryptography``. That paramiko import
    fails with a spurious ``cryptography InternalError: Unknown OpenSSL error``
    whenever another already-loaded library (e.g. torch/MACE in a normal MLFF
    run) has left the OpenSSL error stack dirty. We don't need any of the remote
    calculators, so on failure we load ``Calculator.py`` directly from disk,
    bypassing the package ``__init__`` (its own imports are paramiko-free).
    """
    try:
        from pysisyphus.calculators.Calculator import Calculator
        return Calculator
    except Exception as exc:  # noqa: BLE001 — paramiko/OpenSSL flakiness, etc.
        log.debug("pysisyphus.calculators package import failed (%s); loading "
                  "Calculator.py directly", exc)
    import importlib.util
    import pysisyphus
    calc_py = Path(pysisyphus.__file__).resolve().parent / "calculators" / "Calculator.py"
    spec = importlib.util.spec_from_file_location(
        "quantum_engine._pysis_calculator_base", str(calc_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Calculator


def _ase_pysis_calc_cls():
    """Lazily build (once) the shared ASE→pysisyphus Calculator class.

    Defined inside a function so ``import quantum_engine.qm.pysisyphus`` does not
    require pysisyphus to be importable (only *using* the adapter does).
    """
    global _ASE_PYSIS_CALC_CLS
    if _ASE_PYSIS_CALC_CLS is not None:
        return _ASE_PYSIS_CALC_CLS
    _require()
    Calculator = _import_pysis_calculator_base()
    from pysisyphus.constants import AU2EV, BOHR2ANG, EVANG2AUBOHR
    _ev2au = 1.0 / AU2EV

    class AsePysisCalculator(Calculator):
        """ASE calculator behind pysisyphus's atomic-unit calc interface."""

        def __init__(self, ase_calc, *, charge: int = 0, mult: int = 1):
            super().__init__(charge=charge, mult=mult)
            self.ase_calc = ase_calc
            self._charge = int(charge)

        def _eval(self, atom_symbols, coords_bohr):
            from ase import Atoms as _Atoms
            positions_ang = np.asarray(coords_bohr, dtype=float).reshape(-1, 3) * BOHR2ANG
            # pysisyphus normalises element symbols to lowercase ('o','cl');
            # ASE needs proper case ('O','Cl'). str.capitalize handles 1- and
            # 2-letter symbols correctly.
            symbols = [str(s).capitalize() for s in atom_symbols]
            a = _Atoms(symbols=symbols, positions=positions_ang)
            a.calc = self.ase_calc
            a.info["charge"] = self._charge
            energy_ev = float(a.get_potential_energy())
            forces_eva = np.asarray(a.get_forces(), dtype=float)
            # -> hartree, hartree/bohr
            return energy_ev * _ev2au, forces_eva * EVANG2AUBOHR

        def get_energy(self, atoms, coords, **kw):
            energy, _ = self._eval(atoms, coords)
            return {"energy": energy}

        def get_forces(self, atoms, coords, **kw):
            energy, forces = self._eval(atoms, coords)
            return {"energy": energy, "forces": forces.flatten()}

    _ASE_PYSIS_CALC_CLS = AsePysisCalculator
    return _ASE_PYSIS_CALC_CLS


def make_pysis_calc(ase_calc, *, charge: int = 0, mult: int = 1):
    """Wrap a single ASE ``Calculator`` for pysisyphus (atomic-unit boundary)."""
    return _ase_pysis_calc_cls()(ase_calc, charge=charge, mult=mult)


def make_pysis_calc_getter(calc_fn, *, charge: int = 0, mult: int = 1):
    """Return a zero-arg ``calc_getter`` yielding a FRESH adapter per call.

    pysisyphus string methods (FSM/GSM) instantiate one calculator per image;
    ``calc_fn`` is the QCB per-image ASE-calculator factory.
    """
    cls = _ase_pysis_calc_cls()

    def calc_getter():
        return cls(calc_fn(), charge=charge, mult=mult)

    return calc_getter


def _make_pysis_calc(ase_calc, *, charge: int = 0, mult: int = 1):
    """Back-compat shim — prefer :func:`make_pysis_calc`."""
    return make_pysis_calc(ase_calc, charge=charge, mult=mult)


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
        thresh=_fmax_to_thresh(fmax),     # map eV/Å fmax -> pysisyphus preset
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
        thresh=_fmax_to_thresh(fmax),     # map eV/Å fmax -> pysisyphus preset
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
def rsirfo_ts(atoms: Atoms, **kwargs) -> dict[str, Any]:
    """RS-I-RFO — alias to :func:`rsprfo_ts` for now (same code path; the
    optimiser only differs in the secondary mode-tracking logic, which is
    irrelevant for first-order saddles)."""
    _require()
    return rsprfo_ts(atoms, **kwargs)


