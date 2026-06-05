"""Unit-correctness tests for the shared ASE<->pysisyphus adapter.

pysisyphus is atomic-units throughout (coords in BOHR, energy in HARTREE, forces
in HARTREE/BOHR) and stores element symbols lowercase. The single shared adapter
in ``quantum_engine.qm.pysisyphus`` (used by BOTH the saddle backends and the
FSM/GSM string methods) must convert at every boundary. These tests pin that
down with a finite-difference self-consistency check — the decisive proof that
the energy AND force conversions agree in pysisyphus's own coordinate metric.

Fast (no optimizer run); requires pysisyphus + ASE EMT (both in the container).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pysisyphus")
from ase.build import molecule  # noqa: E402
from ase.calculators.emt import EMT  # noqa: E402

from quantum_engine.qm.pysisyphus import (  # noqa: E402
    _atoms_to_geom,
    _fmax_to_thresh,
    make_pysis_calc,
    make_pysis_calc_getter,
)


def _mol():
    a = molecule("H2CO")        # has C, H, O -> also exercises symbol re-casing
    a.rattle(0.1, seed=3)
    return a


def test_geometry_roundtrip_angstrom_bohr():
    """ASE Å -> pysisyphus Geometry (bohr) -> Å must round-trip to machine eps."""
    from pysisyphus.constants import BOHR2ANG
    atoms = _mol()
    g = _atoms_to_geom(atoms)
    back_ang = np.asarray(g.coords, float).reshape(-1, 3) * BOHR2ANG
    assert np.abs(back_ang - atoms.get_positions()).max() < 1e-9


def test_adapter_force_is_negative_gradient_in_bohr():
    """THE decisive test: the force the adapter returns (hartree/bohr) equals
    -dE/d(coords) (hartree per bohr) by finite difference. If either the energy
    or the force unit conversion were wrong, this fails."""
    from pysisyphus.constants import ANG2BOHR
    atoms = _mol()
    syms = atoms.get_chemical_symbols()
    adapter = make_pysis_calc(EMT(), charge=0, mult=1)
    coords_bohr = atoms.get_positions().flatten() * ANG2BOHR

    _, F = adapter._eval(syms, coords_bohr)
    F = np.asarray(F, float).flatten()

    h = 1e-4  # bohr
    g_fd = np.empty_like(coords_bohr)
    for i in range(coords_bohr.size):
        cp = coords_bohr.copy(); cp[i] += h
        cm = coords_bohr.copy(); cm[i] -= h
        ep = adapter.get_energy(syms, cp)["energy"]
        em = adapter.get_energy(syms, cm)["energy"]
        g_fd[i] = (ep - em) / (2 * h)

    # F == -grad, well below the force magnitude.
    assert np.abs(F + g_fd).max() < 1e-6
    assert np.abs(F).max() > 1e-3   # sanity: non-trivial forces


def test_adapter_force_matches_ase_times_known_factor():
    """Returned force == ASE eV/Å forces × EVANG2AUBOHR (the documented factor)."""
    from pysisyphus.constants import ANG2BOHR, EVANG2AUBOHR
    atoms = _mol()
    syms = atoms.get_chemical_symbols()
    coords_bohr = atoms.get_positions().flatten() * ANG2BOHR
    _, F = make_pysis_calc(EMT(), charge=0, mult=1)._eval(syms, coords_bohr)

    atoms.calc = EMT()
    F_expected = atoms.get_forces().flatten() * EVANG2AUBOHR
    assert np.allclose(np.asarray(F).flatten(), F_expected, atol=1e-10)


def test_adapter_handles_lowercase_symbols_from_pysisyphus():
    """pysisyphus passes lowercase symbols ('o','h','c'); the adapter must
    re-case them for ASE (else KeyError: 'o')."""
    from pysisyphus.constants import ANG2BOHR
    atoms = _mol()
    coords_bohr = atoms.get_positions().flatten() * ANG2BOHR
    adapter = make_pysis_calc(EMT(), charge=0, mult=1)
    # feed deliberately lowercased symbols, as pysisyphus does internally
    lc = [s.lower() for s in atoms.get_chemical_symbols()]
    out = adapter.get_forces(lc, coords_bohr)
    assert np.isfinite(out["energy"]) and np.isfinite(out["forces"]).all()


def test_calc_getter_threads_charge_and_mult():
    getter = make_pysis_calc_getter(lambda: EMT(), charge=-1, mult=3)
    c = getter()
    assert c.mult == 3 and c._charge == -1
    # fresh instance each call (pysisyphus needs one calc per image)
    assert getter() is not c


@pytest.mark.parametrize("fmax,expected", [
    (0.05, "gau"), (0.02, "gau"), (0.005, "baker"),
    (0.1, "gau_loose"), (0.001, "gau_tight"),
])
def test_fmax_to_thresh_mapping(fmax, expected):
    assert _fmax_to_thresh(fmax) == expected
