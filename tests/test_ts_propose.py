"""Tests for the TS-guess proposer registry + its ts_entry integration."""
from __future__ import annotations

import tempfile

import pytest
from ase import Atoms

from quantum_engine.ops import ts_propose


def _mol(symbols="OCO"):
    return Atoms(symbols, positions=[[-1.16, 0, 0], [0, 0, 0], [1.16, 0, 0]][:len(symbols)])


# ---- registry ----
def test_midpoint_registered():
    assert "midpoint" in ts_propose.TS_PROPOSERS.names()
    assert ts_propose.make_ts_proposer("geodesic-mid") is ts_propose.make_ts_proposer("midpoint")


def test_make_ts_proposer_unknown_raises():
    with pytest.raises(ValueError, match="Unknown TS proposer"):
        ts_propose.make_ts_proposer("nope")


def test_react_ot_registered_and_gated():
    """react-ot is registered; when React-OT isn't installed (the main container),
    it raises a clean ImportError naming the sidecar — never a cryptic error."""
    assert "react-ot" in ts_propose.TS_PROPOSERS.names()
    assert ts_propose.make_ts_proposer("reactot").__name__ == "_react_ot"
    from quantum_engine.qm.reactot import reactot_available
    if reactot_available()[0]:
        pytest.skip("React-OT is installed in this environment")
    with pytest.raises(ImportError, match="sidecar|React-OT"):
        ts_propose.run("react-ot", _mol(), _mol(), charge=0, spin=1)


def test_register_ts_proposer_extensibility():
    built = {}

    @ts_propose.register_ts_proposer("fake", aliases=("f",))
    def _fake(reactant, product, *, charge, spin, atom_map, outdir, **kw):
        built["charge"] = charge
        return {"ts_guess": reactant.copy(), "confidence": 0.5,
                "status": "converged", "outputs": {}}

    try:
        res = ts_propose.run("f", _mol(), _mol(), charge=-2, spin=1)
        assert res["ts_guess"] is not None and built["charge"] == -2
        assert res["confidence"] == 0.5
    finally:
        ts_propose.TS_PROPOSERS.unregister("fake")
    assert "fake" not in ts_propose.TS_PROPOSERS


# ---- midpoint reference proposer ----
def test_midpoint_returns_ts_guess(tmp_path):
    r = _mol(); p = _mol(); p.positions[2, 0] += 0.4   # perturb product
    res = ts_propose.run("midpoint", r, p, charge=0, spin=1, outdir=tmp_path)
    assert res["status"] == "converged"
    ts = res["ts_guess"]
    assert isinstance(ts, Atoms) and len(ts) == 3
    assert ts.info["charge"] == 0 and ts.info["spin"] == 1
    # the midpoint sits between R and P along the changing coordinate
    assert r.get_distance(1, 2) <= ts.get_distance(1, 2) <= p.get_distance(1, 2) + 1e-6


# ---- endpoint-consistency guard (shared with path_search) ----
def test_endpoint_mismatch_raises(tmp_path):
    with pytest.raises(ValueError, match="element order mismatch"):
        ts_propose.run("midpoint", _mol("OCO"), _mol("COO"), charge=0, outdir=tmp_path)


# ---- ts_entry routing: proposer replaces path search ----
def test_ts_entry_uses_proposer_not_path(monkeypatch, tmp_path):
    """When proposer= is set, ts_entry must call ts_propose (not path_search) for
    the guess, then refine + gate as usual."""
    from types import SimpleNamespace

    import quantum_engine.calc as calc_mod
    import quantum_engine.ops.opt as opt_mod
    import quantum_engine.ops.path_search as path_mod
    import quantum_engine.ops.refine_ts as refine_mod
    import quantum_engine.ops.ts_propose as prop_mod
    from quantum_engine.ops import ts_entry
    from quantum_engine.reaction_spec import ReactionSpec, ResolvedReaction, RunContext

    seen = {"path_called": False, "propose_called": False}
    monkeypatch.setattr(calc_mod, "make_calc_fn", lambda **kw: (lambda: object()))
    monkeypatch.setattr(opt_mod, "run", lambda a, c, **kw: {
        "converged": True, "energy_eV": -5.0, "atoms": a})
    monkeypatch.setattr(path_mod, "run", lambda *a, **k: seen.__setitem__("path_called", True))

    def fake_propose(method, r, p, **kw):
        seen["propose_called"] = True
        return {"ts_guess": r, "confidence": 0.9, "status": "converged", "outputs": {}}
    monkeypatch.setattr(prop_mod, "run", fake_propose)

    def fake_refine(atoms, calc, **kw):
        return {"status": "completed", "overall_pass": True, "ts_atoms": atoms,
                "n_imag": 1, "imag_freq_cm": -500.0, "energy_eV": -4.0,
                "report": SimpleNamespace(imag_mode_overlap=0.8),
                "outputs": {"imag_mode": None}}
    monkeypatch.setattr(refine_mod, "run", fake_refine)

    resolved = ResolvedReaction(forming=[(0, 1)], breaking=[(1, 2)], reactive=[0, 1, 2],
                                cv_bond_difference=None, spec=ReactionSpec())
    ctx = RunContext(charge=0, multiplicity=1, model="mace-omol", device="cpu")
    res = ts_entry.run(resolved, ctx, entry="reactant-product",
                       reactant=_mol(), product=_mol(), outdir=tmp_path,
                       proposer="some-generative-model", validate=False)
    assert seen["propose_called"] is True and seen["path_called"] is False
    assert res["status"] == "converged" and res["n_imag"] == 1


def test_ts_entry_none_proposal_fails_cleanly(monkeypatch, tmp_path):
    """A proposer that returns ts_guess=None must yield status='failed' (a clean
    critical-fail gate), NOT crash _refine."""
    import quantum_engine.calc as calc_mod
    import quantum_engine.ops.opt as opt_mod
    import quantum_engine.ops.refine_ts as refine_mod
    import quantum_engine.ops.ts_propose as prop_mod
    from quantum_engine.ops import ts_entry
    from quantum_engine.reaction_spec import ReactionSpec, ResolvedReaction, RunContext

    monkeypatch.setattr(calc_mod, "make_calc_fn", lambda **kw: (lambda: object()))
    monkeypatch.setattr(opt_mod, "run", lambda a, c, **kw: {
        "converged": True, "energy_eV": -5.0, "atoms": a})
    monkeypatch.setattr(prop_mod, "run", lambda *a, **k: {
        "ts_guess": None, "confidence": None, "status": "failed", "outputs": {}})
    # refine must NOT be called when there's no guess
    called = {"refine": False}
    monkeypatch.setattr(refine_mod, "run",
                        lambda *a, **k: called.__setitem__("refine", True))

    resolved = ResolvedReaction(forming=[(0, 1)], breaking=[(1, 2)], reactive=[0, 1, 2],
                                cv_bond_difference=None, spec=ReactionSpec())
    ctx = RunContext(charge=0, multiplicity=1, model="mace-omol", device="cpu")
    res = ts_entry.run(resolved, ctx, entry="reactant-product",
                       reactant=_mol(), product=_mol(), outdir=tmp_path,
                       proposer="broken-model", validate=False)
    assert res["status"] == "failed" and res["critical_fail"] is True
    assert called["refine"] is False     # never tried to refine a None guess
