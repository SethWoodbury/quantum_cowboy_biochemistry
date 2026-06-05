"""Tests for the saddle / path / IRC plug-and-play registries.

These never run real saddle/NEB/IRC compute — they monkeypatch the underlying
qm/ops/mlff entry points and assert the factory dispatches to the right runner
with the right arguments (the registry-routing contract). ASE is used only to
build trivial 2-atom Atoms.
"""
from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms


def _h2(charge: int = 0) -> Atoms:
    a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    a.info["charge"] = charge
    return a


# ===========================================================================
# SADDLE
# ===========================================================================
def test_saddle_registry_lists_builtins():
    from quantum_engine.ops.saddle import SADDLE_OPTIMIZERS
    assert {
        "sella", "sella-internal", "dimer",
        "pysisyphus-rsprfo", "pysisyphus-dimer",
    } <= set(SADDLE_OPTIMIZERS.names())


def test_make_saddle_optimizer_alias_and_case_insensitive():
    from quantum_engine.ops.saddle import make_saddle_optimizer, _run_pysis_rsprfo
    assert make_saddle_optimizer("RSPRFO") is _run_pysis_rsprfo      # alias + case
    assert make_saddle_optimizer("pysisyphus-rsprfo") is _run_pysis_rsprfo


def test_make_saddle_optimizer_rejects_unknown_and_auto():
    from quantum_engine.ops.saddle import make_saddle_optimizer
    # "auto" is the cascade handled by run(), not a registry entry.
    with pytest.raises(ValueError, match="Unknown saddle backend"):
        make_saddle_optimizer("auto")
    with pytest.raises(ValueError, match="Unknown saddle backend"):
        make_saddle_optimizer("does-not-exist")


def test_saddle_run_dispatches_to_sella(monkeypatch, tmp_path):
    """saddle.run(backend='sella') must route through the registry to
    qm.sella.run with coords='cart'."""
    import quantum_engine.qm.sella as qm_sella
    from quantum_engine.ops import saddle

    seen = {}

    def fake_run(atoms, calculator, **kw):
        seen.update(kw)
        seen["called"] = True
        return {"status": "converged", "converged": True, "atoms": atoms,
                "energy_eV": -1.0, "backend": "sella", "outputs": {}}

    monkeypatch.setattr(qm_sella, "run", fake_run)
    res = saddle.run(_h2(), calculator=object(), outdir=tmp_path,
                     backend="sella", fmax=0.03, max_steps=7)
    assert seen["called"] and seen["coords"] == "cart"
    assert seen["fmax"] == 0.03 and seen["max_steps"] == 7
    assert res["backend_used"] == "sella"


def test_saddle_run_pysis_injects_charge_and_mult(monkeypatch, tmp_path):
    """The pysisyphus runner injects atoms.info['charge'] AND threads
    atoms.info['spin'] → multiplicity (2S+1)."""
    import quantum_engine.qm.pysisyphus as qm_pysis
    from quantum_engine.ops import saddle

    seen = {}

    def fake_rsprfo(atoms, *, calculator=None, **kw):
        seen.update(kw)
        return {"status": "converged", "converged": True, "atoms": atoms,
                "energy_eV": -1.0, "backend": "pysisyphus-rsprfo", "outputs": {}}

    monkeypatch.setattr(qm_pysis, "rsprfo_ts", fake_rsprfo)
    atoms = _h2(charge=-2)
    atoms.info["spin"] = 3          # triplet
    saddle.run(atoms, calculator=object(), outdir=tmp_path,
               backend="pysisyphus-rsprfo")
    assert seen["charge"] == -2 and seen["mult"] == 3


def test_register_saddle_adds_backend(monkeypatch, tmp_path):
    from quantum_engine.ops.saddle import (
        SADDLE_OPTIMIZERS, register_saddle, make_saddle_optimizer)

    calls = {}

    def my_runner(atoms, **kw):
        calls["hit"] = True
        return {"status": "converged", "converged": True, "atoms": atoms,
                "energy_eV": 0.0, "backend": "my-saddle", "outputs": {}}

    register_saddle("my-saddle", my_runner, aliases=("mine",))
    try:
        assert make_saddle_optimizer("MINE") is my_runner
        from quantum_engine.ops import saddle
        saddle.run(_h2(), calculator=object(), outdir=tmp_path, backend="my-saddle")
        assert calls.get("hit")
    finally:
        SADDLE_OPTIMIZERS.unregister("my-saddle")
    assert "my-saddle" not in SADDLE_OPTIMIZERS


# ===========================================================================
# PATH
# ===========================================================================
def test_path_registry_lists_builtins():
    from quantum_engine.ops.path_search import PATH_METHODS
    assert {"neb", "fsm", "gsm-de", "pygsm-de", "gsm-se"} <= set(PATH_METHODS.names())


def test_make_path_method_alias_and_unknown():
    from quantum_engine.ops.path_search import make_path_method, _run_neb, _run_gsm_de
    assert make_path_method("ci-neb") is _run_neb       # alias
    assert make_path_method("gsm") is _run_gsm_de       # alias
    assert make_path_method("se-gsm").__name__ == "_run_gsm_se"   # pyGSM single-ended
    with pytest.raises(ValueError, match="Unknown path method"):
        make_path_method("nope")


def test_path_run_dispatches_neb(monkeypatch, tmp_path):
    import quantum_engine.ops.neb as neb
    from quantum_engine.ops import path_search

    seen = {}

    def fake_neb_run(reactant, product, calculator_fn, **kw):
        seen.update(kw)
        return {"status": "converged", "ts": reactant, "outputs": {}}

    monkeypatch.setattr(neb, "run", fake_neb_run)
    path_search.run("neb", _h2(), _h2(), (lambda: object()),
                    outdir=tmp_path, charge=3)
    assert seen["charge"] == 3 and seen["constraint"] is None


@pytest.mark.parametrize("method,expect", [("fsm", "fsm"), ("gsm-de", "gsm")])
def test_path_run_dispatches_gsm(monkeypatch, tmp_path, method, expect):
    import quantum_engine.ops.gsm as gsm
    from quantum_engine.ops import path_search

    seen = {}

    def fake_gsm_run(reactant, product, calculator_fn, outdir, method="fsm", **kw):
        seen["method"] = method
        seen.update(kw)
        return {"status": "converged", "ts": reactant, "outputs": {}}

    monkeypatch.setattr(gsm, "run", fake_gsm_run)
    # constraint is passed but must be warned-and-ignored by FSM/GSM, not crash.
    r = _h2(); r.info["spin"] = 2                 # doublet → mult threads through
    path_search.run(method, r, _h2(), (lambda: object()),
                    outdir=tmp_path, charge=-1, constraint=object())
    assert seen["method"] == expect and seen["charge"] == -1 and seen["mult"] == 2


# ===========================================================================
# IRC
# ===========================================================================
def test_irc_registry_default_and_aliases():
    from quantum_engine.ops.irc import IRC_METHODS, make_irc, _run_mlff_irc
    assert "mlff" in IRC_METHODS.names()
    assert make_irc() is _run_mlff_irc            # default
    assert make_irc("ase") is _run_mlff_irc       # alias
    with pytest.raises(ValueError, match="Unknown IRC method"):
        make_irc("nope")


def test_irc_run_dispatches_to_mlff(monkeypatch, tmp_path):
    import quantum_engine.mlff.irc as mlff_irc
    from quantum_engine.ops import irc

    seen = {}

    def fake_from_ts(atoms, **kw):
        seen.update(kw)
        return {"success": True, "ts": atoms, "reactant": atoms, "product": atoms,
                "imag_freq_cm": -480.0, "barrier_fwd_kcal": 12.0,
                "barrier_rev_kcal": 4.0}

    monkeypatch.setattr(mlff_irc, "irc_from_ts_guess", fake_from_ts)
    res = irc.run(_h2(), calculator=object(), outdir=tmp_path,
                  refine_ts=True, irc_step=0.15)
    assert seen["irc_step"] == 0.15
    assert res["status"] == "converged" and res["imag_freq_cm"] == -480.0
