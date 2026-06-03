"""Tests for qcb opt per-bond constraints (--fix-bond / --restrain-bond).

ASE-only (LennardJones) so they run anywhere, no MLFF/container needed.
"""
from __future__ import annotations

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase import Atoms  # noqa: E402
from ase.calculators.lj import LennardJones  # noqa: E402
from ase.optimize import LBFGS  # noqa: E402

from quantum_engine.ops.bond_constraints import (  # noqa: E402
    HarmonicDistance,
    build_bond_constraints,
)


def _triatomic(d=1.5):
    # three Ar atoms in a line; LJ has a finite minimum so relax is well-defined
    return Atoms("Ar3", positions=[[0, 0, 0], [d, 0, 0], [2 * d, 0, 0]])


def test_harmonic_distance_force_direction():
    # bond stretched beyond r0 -> force pulls the two atoms together
    a = Atoms("Ar2", positions=[[0, 0, 0], [3.0, 0, 0]])
    hd = HarmonicDistance(0, 1, k=5.0, r0=2.0)
    forces = np.zeros((2, 3))
    hd.adjust_forces(a, forces)
    # d=3 > r0=2 -> f_mag>0 -> atom0 pushed +x, atom1 pushed -x (toward each other)
    assert forces[0, 0] > 0 and forces[1, 0] < 0
    assert np.isclose(forces[0, 0], 5.0 * (3.0 - 2.0))
    # compressed below r0 -> forces reverse (push apart)
    b = Atoms("Ar2", positions=[[0, 0, 0], [1.0, 0, 0]])
    f2 = np.zeros((2, 3))
    HarmonicDistance(0, 1, k=5.0, r0=2.0).adjust_forces(b, f2)
    assert f2[0, 0] < 0 and f2[1, 0] > 0


def test_fix_bond_pins_current_length():
    a = _triatomic(d=1.7)
    d0 = a.get_distance(0, 1)
    cons = build_bond_constraints(a, fix_bond=[[0, 1]])
    a.calc = LennardJones()
    a.set_constraint(cons)
    LBFGS(a, logfile=None).run(fmax=0.01, steps=50)
    assert np.isclose(a.get_distance(0, 1), d0, atol=1e-3)


def test_fix_bond_with_target_sets_then_pins():
    a = _triatomic(d=1.7)
    cons = build_bond_constraints(a, fix_bond=[[0, 1, 2.4]])  # set to 2.4 then pin
    assert np.isclose(a.get_distance(0, 1), 2.4, atol=1e-6)  # set_distance applied
    a.calc = LennardJones()
    a.set_constraint(cons)
    LBFGS(a, logfile=None).run(fmax=0.01, steps=50)
    assert np.isclose(a.get_distance(0, 1), 2.4, atol=1e-3)


def test_restrain_bond_pulls_toward_target():
    # start far from target; harmonic restraint should pull bond toward r0
    a = _triatomic(d=3.0)
    cons = build_bond_constraints(a, restrain_bond=[[0, 1, 50.0, 2.0]])
    a.calc = LennardJones()
    a.set_constraint(cons)
    LBFGS(a, logfile=None).run(fmax=0.01, steps=200)
    # not exact (LJ + restraint balance) but should move substantially toward 2.0
    assert a.get_distance(0, 1) < 2.6


def test_fix_bond_composes_with_fixatoms():
    from ase.constraints import FixAtoms
    a = _triatomic(d=1.7)
    d0 = a.get_distance(0, 1)
    base = FixAtoms(indices=[2])               # scaffold: freeze atom 2
    bond_cons = build_bond_constraints(a, fix_bond=[[0, 1]])
    a.calc = LennardJones()
    a.set_constraint([base] + bond_cons)       # FixAtoms first, then bond pin
    pos2 = a.positions[2].copy()
    LBFGS(a, logfile=None).run(fmax=0.01, steps=50)
    assert np.isclose(a.get_distance(0, 1), d0, atol=1e-3)   # bond held
    assert np.allclose(a.positions[2], pos2, atol=1e-6)      # atom 2 frozen


def test_bad_spec_raises():
    a = _triatomic()
    with pytest.raises(SystemExit):
        build_bond_constraints(a, fix_bond=[[0]])
    with pytest.raises(SystemExit):
        build_bond_constraints(a, restrain_bond=[[0, 1, 5.0]])
