"""Tests for the TS-guess refiner registry + its ts_entry integration."""
from __future__ import annotations

import pytest
from ase import Atoms

from quantum_engine.ops import ts_refine


def _mol(symbols="OCO"):
    return Atoms(symbols, positions=[[-1.16, 0, 0], [0, 0, 0], [1.16, 0, 0]][:len(symbols)])


# ---- registry ----
def test_identity_registered():
    assert "identity" in ts_refine.TS_REFINERS.names()
    assert ts_refine.make_ts_refiner("noop") is ts_refine.make_ts_refiner("identity")


def test_make_ts_refiner_unknown_raises():
    with pytest.raises(ValueError, match="Unknown TS refiner"):
        ts_refine.make_ts_refiner("nope")


def test_aefm_registered_and_gated():
    """aefm is registered; when AEFM isn't installed (the main container), it
    raises a clean ImportError naming the sidecar — never a cryptic error."""
    assert "aefm" in ts_refine.TS_REFINERS.names()
    assert ts_refine.make_ts_refiner("aefm").__name__ == "_aefm"
    from quantum_engine.qm.aefm import aefm_available
    if aefm_available()[0]:
        pytest.skip("AEFM is installed in this environment")
    with pytest.raises(ImportError, match="sidecar|AEFM"):
        ts_refine.run("aefm", _mol(), charge=0, spin=1)


def test_register_ts_refiner_extensibility():
    seen = {}

    @ts_refine.register_ts_refiner("fake", aliases=("f",))
    def _fake(ts_guess, *, charge, spin, reactant, product, outdir, **kw):
        seen["charge"] = charge
        out = ts_guess.copy()
        out.info["touched"] = True
        return {"ts_guess": out, "confidence": 0.5, "status": "converged",
                "outputs": {}}

    try:
        res = ts_refine.run("f", _mol(), charge=-2, spin=1)
        assert res["ts_guess"].info["touched"] and seen["charge"] == -2
        assert res["confidence"] == 0.5
    finally:
        ts_refine.TS_REFINERS.unregister("fake")
    assert "fake" not in ts_refine.TS_REFINERS


# ---- identity reference refiner (no-op baseline) ----
def test_identity_returns_guess_unchanged(tmp_path):
    g = _mol()
    res = ts_refine.run("identity", g, charge=-1, spin=2, outdir=tmp_path)
    assert res["status"] == "converged"
    ts = res["ts_guess"]
    assert isinstance(ts, Atoms) and len(ts) == 3
    assert ts.info["charge"] == -1 and ts.info["spin"] == 2
    # geometry untouched
    assert (ts.get_positions() == g.get_positions()).all()


# ---- ts_entry routing: refiner polishes the guess before saddle-refine ----
def _mock_core(monkeypatch):
    """Common ts_entry mocks (calc + refine_ts). Returns the `seen` dict."""
    from types import SimpleNamespace

    import quantum_engine.calc as calc_mod
    import quantum_engine.ops.refine_ts as refine_mod

    seen = {"refine_guess": None}
    monkeypatch.setattr(calc_mod, "make_calc_fn", lambda **kw: (lambda: object()))

    def fake_refine(atoms, calc, **kw):
        seen["refine_guess"] = atoms
        return {"status": "completed", "overall_pass": True, "ts_atoms": atoms,
                "n_imag": 1, "imag_freq_cm": -500.0, "energy_eV": -4.0,
                "report": SimpleNamespace(imag_mode_overlap=0.8),
                "outputs": {"imag_mode": None}}
    monkeypatch.setattr(refine_mod, "run", fake_refine)
    return seen


def _resolved_ctx():
    from quantum_engine.reaction_spec import (ReactionSpec, ResolvedReaction,
                                              RunContext)
    resolved = ResolvedReaction(forming=[(0, 1)], breaking=[(1, 2)],
                                reactive=[0, 1, 2], cv_bond_difference=None,
                                spec=ReactionSpec())
    ctx = RunContext(charge=0, multiplicity=1, model="mace-omol", device="cpu")
    return resolved, ctx


def test_ts_entry_uses_refiner(monkeypatch, tmp_path):
    """When refiner= is set, ts_entry must polish the guess (the REFINED guess,
    not the original, flows into saddle-refine)."""
    import quantum_engine.ops.ts_refine as refine_axis
    from quantum_engine.ops import ts_entry

    seen = _mock_core(monkeypatch)

    def fake_refiner(method, ts_guess, **kw):
        out = ts_guess.copy()
        out.info["refined"] = True       # marker the saddle stage should see
        return {"ts_guess": out, "confidence": 0.9, "status": "converged",
                "outputs": {}}
    monkeypatch.setattr(refine_axis, "run", fake_refiner)

    resolved, ctx = _resolved_ctx()
    res = ts_entry.run(resolved, ctx, entry="ts-guess", ts_guess=_mol(),
                       refiner="some-refiner", outdir=tmp_path, validate=False)
    assert res["status"] == "converged"
    assert seen["refine_guess"] is not None
    assert seen["refine_guess"].info.get("refined") is True   # used the refined one


def test_ts_entry_refiner_failure_falls_back(monkeypatch, tmp_path):
    """A refiner that reports status='failed' must NOT break the run — saddle
    refine still runs on the (un-refined) original guess."""
    import quantum_engine.ops.ts_refine as refine_axis
    from quantum_engine.ops import ts_entry

    seen = _mock_core(monkeypatch)
    monkeypatch.setattr(refine_axis, "run", lambda *a, **k: {
        "status": "failed", "ts_guess": None, "confidence": None, "outputs": {}})

    resolved, ctx = _resolved_ctx()
    res = ts_entry.run(resolved, ctx, entry="ts-guess", ts_guess=_mol(),
                       refiner="broken", outdir=tmp_path, validate=False)
    assert res["status"] == "converged"          # pipeline survived
    assert seen["refine_guess"] is not None
    assert seen["refine_guess"].info.get("refined") is None    # original guess


def test_ts_entry_refiner_exception_is_caught(monkeypatch, tmp_path):
    """A refiner that RAISES (e.g. sidecar absent) is caught — non-critical."""
    import quantum_engine.ops.ts_refine as refine_axis
    from quantum_engine.ops import ts_entry

    seen = _mock_core(monkeypatch)

    def boom(*a, **k):
        raise ImportError("AEFM requires its sidecar")
    monkeypatch.setattr(refine_axis, "run", boom)

    resolved, ctx = _resolved_ctx()
    res = ts_entry.run(resolved, ctx, entry="ts-guess", ts_guess=_mol(),
                       refiner="aefm", outdir=tmp_path, validate=False)
    assert res["status"] == "converged"          # did not crash
    assert seen["refine_guess"] is not None       # refine still ran on the guess
