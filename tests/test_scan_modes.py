"""Tests for ops.scan_modes — bond-difference / two-sided / auto scans."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from quantum_engine.ops import scan_modes
from quantum_engine.ops.scan_modes import _resolve_auto


def _co2():
    from ase.calculators.emt import EMT
    a = Atoms("OCO", positions=[[-1.16, 0, 0], [0, 0, 0], [1.16, 0, 0]])
    a.calc = EMT()
    return a


# ---- auto-resolution (fast, no compute) ----
def test_resolve_auto_cv_bond_difference():
    rr = SimpleNamespace(cv_bond_difference=(1, 0, 2), forming=[], breaking=[])
    mode, kw = _resolve_auto(rr)
    assert mode == "bond-difference"
    assert kw["diff_bonds"] == ((1, 2), (1, 0))   # ((P,LG),(P,nuc))


def test_resolve_auto_single_changing_bond():
    rr = SimpleNamespace(cv_bond_difference=None, forming=[(3, 4)], breaking=[])
    mode, kw = _resolve_auto(rr)
    assert mode == "single" and kw["atom_indices"] == [3, 4]


def test_resolve_auto_two_changing_bonds():
    rr = SimpleNamespace(cv_bond_difference=None, forming=[(1, 2)], breaking=[(3, 4)])
    mode, kw = _resolve_auto(rr)
    assert mode == "two-sided" and kw["bond_a"] == (1, 2) and kw["bond_b"] == (3, 4)


def test_resolve_auto_errors_without_changing():
    rr = SimpleNamespace(cv_bond_difference=None, forming=[], breaking=[])
    with pytest.raises(ValueError, match="no changing bonds"):
        _resolve_auto(rr)


# ---- input validation ----
def test_run_rejects_bad_mode():
    with pytest.raises(ValueError, match="not in"):
        scan_modes.run(_co2(), mode="nope")


def test_bond_difference_requires_shared_central_atom(tmp_path):
    # bonds (0,1) and (2,3) share NO atom -> error
    a = Atoms("OCOO", positions=[[0, 0, 0], [1.2, 0, 0], [3, 0, 0], [4.2, 0, 0]])
    from ase.calculators.emt import EMT
    a.calc = EMT()
    with pytest.raises(ValueError, match="share exactly one atom"):
        scan_modes.run(a, mode="bond-difference", diff_bonds=((0, 1), (2, 3)),
                       s_start=-0.2, s_end=0.2, n_steps=2, outdir=tmp_path)


# ---- end-to-end (EMT, minimal) ----
def test_single_mode_delegates_and_returns_ts_guess(tmp_path):
    r = scan_modes.run(_co2(), mode="single", coord_type="bond",
                       atom_indices=[1, 0], start=1.05, end=1.30, n_steps=3,
                       outdir=tmp_path, fmax=0.2, max_opt_steps=30)
    assert r["mode"] == "single"
    assert r["ts_guess_idx"] is not None
    assert "ts_guess" in r and len(r["energies_eV"]) == 3


def test_two_sided_runs_and_reports_frames(tmp_path):
    r = scan_modes.run(_co2(), mode="two-sided", bond_a=(1, 0), bond_b=(1, 2),
                       a_start=1.05, a_end=1.30, b_start=1.30, b_end=1.05,
                       n_steps=3, outdir=tmp_path, fmax=0.2, max_opt_steps=30)
    assert r["mode"] == "two-sided"
    assert len(r["frames"]) == 3 and len(r["energies_eV"]) == 3
    # each frame stores its companion bond value
    assert all("bond_b_value" in fr.info for fr in r["frames"])


def test_bond_difference_drives_s_monotonically(tmp_path):
    r = scan_modes.run(_co2(), mode="bond-difference", diff_bonds=((1, 0), (1, 2)),
                       s_start=-0.3, s_end=0.3, n_steps=4, outdir=tmp_path,
                       fmax=0.2, max_opt_steps=40, spring_k=10.0)
    assert r["mode"] == "bond-difference"
    s = r["coord_values"]            # achieved s at each relaxed geometry
    # the restraint may undershoot, but s must increase with the target
    assert s[0] < s[-1]
    assert np.all(np.diff(s) > -1e-3)   # (weakly) monotonically increasing


def test_auto_via_run_resolves_bond_difference(tmp_path):
    rr = SimpleNamespace(cv_bond_difference=(1, 0, 2), forming=[(1, 0)],
                         breaking=[(1, 2)])
    r = scan_modes.run(_co2(), mode="auto", reaction=rr, s_start=-0.2, s_end=0.2,
                       n_steps=3, outdir=tmp_path, fmax=0.2, max_opt_steps=30,
                       spring_k=10.0)
    assert r["mode"] == "bond-difference"


# ---- TS-guess selection (pure function; endpoints are basins in a relaxed scan) ----
def test_pick_ts_guess_interior_maximum():
    # clean basin -> TS -> basin: pick the interior peak, reactant basin = frame 0
    ts, rb, barrierless = scan_modes._pick_ts_guess([0.0, 0.5, 1.2, 0.4, -0.3])
    assert ts == 2 and rb == 0 and not barrierless


def test_pick_ts_guess_monotonic_is_barrierless():
    # strictly downhill -> no interior maximum -> flagged barrierless, max = endpoint
    ts, rb, barrierless = scan_modes._pick_ts_guess([0.0, -0.3, -0.7, -1.1, -1.6])
    assert barrierless and ts == 0


def test_pick_ts_guess_downhill_with_interior_bump_opaa():
    # the real OPAA bond-difference profile (eV): exothermic overall, small interior
    # TS at frame 4 (s~0); the global max is the strained reactant endpoint (frame 0).
    e = [-303002.79969517, -303002.93866199, -303002.95639744, -303002.84919638,
         -303002.81260730, -303003.14275628, -303003.46131093, -303003.51666555,
         -303003.54157152]
    ts, rb, barrierless = scan_modes._pick_ts_guess(e)
    assert ts == 4 and rb == 2 and not barrierless
