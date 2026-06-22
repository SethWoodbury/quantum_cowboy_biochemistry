"""TS-guess / reaction-path sanity guards (quantum_engine.analysis.ts_sanity)."""
from __future__ import annotations

import pytest
from ase import Atoms

from quantum_engine.analysis.ts_sanity import (
    heavy_atom_rmsd, ts_endpoint_similarity, flag_spurious_path_peak, flag_flat_or_soft_ts)


def test_soft_imaginary_mode_flagged():
    assert any(s == "WARNING" and "soft" in m.lower()
               for s, m in flag_flat_or_soft_ts(imag_freq_cm=-65.0))


def test_sharp_imaginary_mode_not_soft():
    assert not any("soft" in m.lower() for s, m in flag_flat_or_soft_ts(imag_freq_cm=-520.0))


def test_low_reactive_overlap_flagged():
    fs = flag_flat_or_soft_ts(imag_freq_cm=-600.0, reactive_overlap=0.05)
    assert any(s == "WARNING" and "overlap" in m.lower() for s, m in fs)


def test_shallow_barrier_flagged():
    assert any(s == "WARNING" and "shallow" in m.lower()
               for s, m in flag_flat_or_soft_ts(barrier_kcal=1.5))


def test_clean_sharp_ts_not_flagged():
    fs = flag_flat_or_soft_ts(imag_freq_cm=-480.0, reactive_overlap=0.8, barrier_kcal=18.0)
    assert not any(s == "WARNING" for s, m in fs)


def _lin(d):
    """3 atoms on a line; the middle atom sits at x=d (the reaction coordinate)."""
    return Atoms("OHO", positions=[[0, 0, 0], [d, 0, 0], [3.0, 0, 0]])


def test_rmsd_zero_for_identical():
    a = _lin(1.5)
    assert heavy_atom_rmsd(a, a) == pytest.approx(0.0)


def test_ts_flagged_when_essentially_the_product():
    reactant, product, ts = _lin(0.5), _lin(2.5), _lin(2.49)   # ts ≈ product
    findings = ts_endpoint_similarity(ts, reactant, product)
    assert any(s == "WARNING" and "product" in m.lower() for s, m in findings)


def test_ts_flagged_when_lopsided_toward_reactant():
    reactant, product, ts = _lin(0.5), _lin(2.5), _lin(0.85)   # much closer to reactant
    findings = ts_endpoint_similarity(ts, reactant, product)
    assert any(s == "WARNING" for s, m in findings)


def test_ts_not_flagged_for_genuine_midpoint():
    reactant, product, ts = _lin(0.5), _lin(2.5), _lin(1.5)
    findings = ts_endpoint_similarity(ts, reactant, product)
    assert not any(s == "WARNING" for s, m in findings)


def test_reactive_idx_subsets_the_rmsd():
    reactant, product, ts = _lin(0.5), _lin(2.5), _lin(2.49)
    findings = ts_endpoint_similarity(ts, reactant, product, reactive_idx=[1])
    assert any("product" in m.lower() and s == "WARNING" for s, m in findings)


def test_spurious_lone_spike_flagged():
    findings = flag_spurious_path_peak([0.0, 1.0, 18.0, 1.5, -14.0])
    assert any(s == "WARNING" and "spike" in m.lower() for s, m in findings)


def test_clean_barrier_not_flagged():
    findings = flag_spurious_path_peak([0.0, 2.0, 5.0, 8.0, 6.0, 2.0, -14.0])
    assert not any(s == "WARNING" for s, m in findings)


def test_endpoint_peak_flagged():
    findings = flag_spurious_path_peak([5.0, 2.0, 1.0, -3.0])
    assert any(s == "WARNING" and "endpoint" in m.lower() for s, m in findings)


def test_perp_force_outlier_flagged():
    findings = flag_spurious_path_peak([0.0, 2.0, 7.0, 2.0, -10.0],
                                       perp_forces=[0.02, 0.03, 0.5, 0.03, 0.02])
    assert any("perpendicular force" in m.lower() for s, m in findings)
