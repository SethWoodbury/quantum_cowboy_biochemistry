"""Generalized weighted-bond-distance CV — reaction-agnostic across mechanisms.

The CV is s = Σ_breaking d − Σ_forming d. These tests pin (a) byte-identical
behavior for the SN2-at-P single-center case, (b) correctness for the edge cases
the engine must now handle (multi-bond, no shared center, forming-only "click",
breaking-only dissociation), and (c) the de-hardcoded metadynamics classifier.
"""
from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from quantum_engine.mlff.cv_spring import (
    BondDifferenceCVSpring, bond_cv_terms_from_bonds, bond_cv_terms_from_roles,
    bond_distance_cv)


def _atoms(positions):
    return Atoms("X" * len(positions), positions=positions)


# --------------------------------------------------------------------------
# Core primitive: value + gradient (finite-difference check)
# --------------------------------------------------------------------------
def test_bond_distance_cv_value_and_gradient():
    pos = np.array([[0, 0, 0], [1.9, 0, 0], [-1.6, 0.3, 0], [4, 4, 4]], float)
    terms = [(0, 2, 1.0), (0, 1, -1.0)]   # s = d(0,2) − d(0,1)
    s, grad = bond_distance_cv(pos, terms)
    assert abs(s - (np.linalg.norm(pos[0] - pos[2]) - np.linalg.norm(pos[0] - pos[1]))) < 1e-12
    # finite-difference gradient
    eps = 1e-6
    num = np.zeros_like(pos)
    for a in range(len(pos)):
        for c in range(3):
            p2 = pos.copy(); p2[a, c] += eps
            sp, _ = bond_distance_cv(p2, terms)
            p3 = pos.copy(); p3[a, c] -= eps
            sm, _ = bond_distance_cv(p3, terms)
            num[a, c] = (sp - sm) / (2 * eps)
    assert np.allclose(grad, num, atol=1e-6)


def test_degenerate_term_no_nan():
    # coincident atoms → that term contributes 0 gradient, no NaN
    pos = np.array([[0, 0, 0], [0, 0, 0], [2, 0, 0]], float)
    s, grad = bond_distance_cv(pos, [(0, 1, 1.0), (0, 2, -1.0)])
    assert np.isfinite(s) and np.all(np.isfinite(grad))


# --------------------------------------------------------------------------
# SN2-at-P single-center case is preserved byte-for-byte
# --------------------------------------------------------------------------
def test_single_center_matches_old_convention():
    # center=P=0, breaking=LG=2, forming=nuc=1  → s = d(P,LG) − d(P,nuc)
    pos = np.array([[0, 0, 0], [2.0, 0, 0], [-1.6, 0, 0], [9, 9, 9]], float)
    a = _atoms(pos)
    spring = BondDifferenceCVSpring(center_idx=0, breaking_idx=2, forming_idx=1, s_target=-2.0)
    s_old = np.linalg.norm(pos[0] - pos[2]) - np.linalg.norm(pos[0] - pos[1])
    assert abs(spring.compute_cv(a) - s_old) < 1e-12
    assert s_old < 0   # reactant-like: LG bonded, nuc far


# --------------------------------------------------------------------------
# Edge cases the engine must now generalize to
# --------------------------------------------------------------------------
def test_forming_only_click_is_monotonic():
    """Association/click: only a bond FORMS. s = −d(forming), grows reactant→product."""
    spring = BondDifferenceCVSpring.from_bonds(forming=[(0, 1)])
    far = _atoms([[0, 0, 0], [5.0, 0, 0]])     # reactant: fragments apart
    near = _atoms([[0, 0, 0], [1.5, 0, 0]])    # product: bond formed
    assert spring.compute_cv(far) < spring.compute_cv(near)


def test_breaking_only_dissociation_is_monotonic():
    """Fragmentation: only a bond BREAKS. s = +d(breaking), grows reactant→product."""
    spring = BondDifferenceCVSpring.from_bonds(breaking=[(0, 1)])
    bonded = _atoms([[0, 0, 0], [1.5, 0, 0]])  # reactant
    apart = _atoms([[0, 0, 0], [5.0, 0, 0]])   # product
    assert spring.compute_cv(bonded) < spring.compute_cv(apart)


def test_multibond_no_shared_center():
    """Two breaking + one forming bond, no atom shared between bonds."""
    spring = BondDifferenceCVSpring.from_bonds(breaking=[(0, 1), (2, 3)], forming=[(4, 5)])
    assert len(spring.terms) == 3
    assert sorted(w for *_, w in spring.terms) == [-1.0, 1.0, 1.0]
    pos = np.arange(18, dtype=float).reshape(6, 3)
    expect = (np.linalg.norm(pos[0] - pos[1]) + np.linalg.norm(pos[2] - pos[3])
              - np.linalg.norm(pos[4] - pos[5]))
    assert abs(spring.compute_cv(_atoms(pos)) - expect) < 1e-12


def test_from_bonds_requires_a_bond():
    with pytest.raises(ValueError):
        bond_cv_terms_from_bonds([], [])


def test_from_resolved_reaction_prefers_explicit_cv_then_falls_back():
    from quantum_engine.reaction_spec import ReactionSpec, ResolvedReaction
    pos = np.array([[0, 0, 0], [2.0, 0, 0], [-1.6, 0, 0]], float)
    a = _atoms(pos)
    s_ref = np.linalg.norm(pos[0] - pos[2]) - np.linalg.norm(pos[0] - pos[1])
    # explicit cv = (center, breaking, forming)
    rr = ResolvedReaction(forming=[(0, 1)], breaking=[(0, 2)], reactive=[0, 1, 2],
                          cv_bond_difference=(0, 2, 1), spec=ReactionSpec())
    assert abs(BondDifferenceCVSpring.from_resolved_reaction(rr).compute_cv(a) - s_ref) < 1e-12
    # no explicit cv → derive from breaking(+1)/forming(−1) bond lists
    rr2 = ResolvedReaction(forming=[(0, 1)], breaking=[(0, 2)], reactive=[0, 1, 2],
                           cv_bond_difference=None, spec=ReactionSpec())
    assert abs(BondDifferenceCVSpring.from_resolved_reaction(rr2).compute_cv(a) - s_ref) < 1e-12


# --------------------------------------------------------------------------
# Spring drives toward the target along the CV
# --------------------------------------------------------------------------
def test_spring_force_points_toward_target():
    pos = np.array([[0, 0, 0], [2.0, 0, 0], [-1.6, 0, 0]], float)
    a = _atoms(pos)
    spring = BondDifferenceCVSpring(center_idx=0, breaking_idx=2, forming_idx=1,
                                    s_target=+2.0, k=3.0, fmax=None)
    s0 = spring.compute_cv(a)
    f = np.zeros_like(pos)
    spring.adjust_forces(a, f)
    # take a tiny step along the spring force; CV should move toward the (higher) target
    a2 = _atoms(pos + 1e-3 * f)
    assert spring.compute_cv(a2) > s0


# --------------------------------------------------------------------------
# PLUMED weighted CV mirrors the in-house convention
# --------------------------------------------------------------------------
def test_plumed_weighted_cv_coefficients():
    from quantum_engine.mlff.plumed_runner import DistanceDifferenceCV, WeightedBondCV
    w = WeightedBondCV.from_bonds("s", breaking_bonds=[(0, 1)], forming_bonds=[(0, 2), (3, 4)])
    lines = w.to_plumed()
    assert any("COEFFICIENTS=1,-1,-1" in ln for ln in lines)
    assert sum(1 for ln in lines if "DISTANCE ATOMS=" in ln) == 3
    # 2-bond DistanceDifferenceCV: breaking +1, forming −1
    d = DistanceDifferenceCV("s", atoms_minus=(0, 2), atoms_plus=(0, 1)).to_plumed()
    assert any("COEFFICIENTS=1,-1" in ln for ln in d)


# --------------------------------------------------------------------------
# De-hardcoded metadynamics basin classifier
# --------------------------------------------------------------------------
def test_classify_basin_assumes_no_chemistry_without_windows():
    from quantum_engine.mlff.metadynamics import _classify_basin
    assert _classify_basin(-2.0) is None
    assert _classify_basin(3.0) is None


def test_classify_basin_with_sn2_windows_reproduces_old_labels():
    from quantum_engine.mlff.metadynamics import _classify_basin, SN2_AT_P_BASIN_WINDOWS
    w = SN2_AT_P_BASIN_WINDOWS
    assert _classify_basin(-2.0, w) == "reactant"
    assert _classify_basin(3.0, w) == "product"
    assert _classify_basin(0.5, w) == "intermediate"
    assert _classify_basin(1.7, w) is None   # between intermediate and product → unlabeled
