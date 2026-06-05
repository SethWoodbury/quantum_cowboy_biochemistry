"""Tests for ops.ts_entry — the TS orchestrator.

The heavy sub-ops (opt / path_search / refine_ts / imag_mode_displace /
scan_modes) and the calculator factory are monkeypatched, so these exercise the
ORCHESTRATION: entry/rigor validation, branch wiring, the reactive-atom "0:idx"
contract, gate aggregation (critical-fail → failed), rigor presets, and barrier
math — with no MLFF/GPU/saddle compute.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from quantum_engine.reaction_spec import ReactionSpec, ResolvedReaction, RunContext
from quantum_engine.ops import ts_entry


def _atoms(n=3):
    a = Atoms("OCO", positions=[[-1.16, 0, 0], [0, 0, 0], [1.16, 0, 0]])
    return a


def _resolved(cv=True):
    return ResolvedReaction(
        forming=[(0, 1)], breaking=[(1, 2)], reactive=[0, 1, 2],
        cv_bond_difference=(1, 0, 2) if cv else None, spec=ReactionSpec())


def _ctx():
    return RunContext(charge=-1, spin=1, model="mace-omol", device="cpu")


@pytest.fixture
def patched(monkeypatch):
    """Patch every sub-op + the calc factory; record calls in `rec`."""
    rec = {"refine_kwargs": None, "path_kwargs": None, "opt_calls": 0,
           "validate_called": False, "scan_called": False}

    import quantum_engine.calc as calc_mod
    import quantum_engine.ops.opt as opt_mod
    import quantum_engine.ops.path_search as path_mod
    import quantum_engine.ops.refine_ts as refine_mod
    import quantum_engine.ops.imag_mode_displace as imd_mod
    import quantum_engine.ops.scan_modes as sm_mod

    monkeypatch.setattr(calc_mod, "make_calc_fn",
                        lambda **kw: (lambda: object()))

    def fake_opt(atoms, calc, **kw):
        rec["opt_calls"] += 1
        return {"converged": True, "energy_eV": -5.0, "atoms": atoms}
    monkeypatch.setattr(opt_mod, "run", fake_opt)

    def fake_path(method, r, p, calc_fn, **kw):
        rec["path_kwargs"] = {"method": method, **kw}
        imgs = [_atoms(), _atoms(), _atoms()]
        imgs[2].positions[2, 0] += 0.5   # give a non-zero tangent
        return {"status": "converged", "ts": _atoms(), "images": imgs,
                "ts_idx": 1, "barrier_fwd_kcal": 10.0}
    monkeypatch.setattr(path_mod, "run", fake_path)

    def fake_refine(atoms, calc, **kw):
        rec["refine_kwargs"] = kw
        return {"status": "completed", "overall_pass": True, "ts_atoms": atoms,
                "n_imag": 1, "imag_freq_cm": -520.0, "energy_eV": -4.0,
                "report": SimpleNamespace(imag_mode_overlap=0.82),
                "outputs": {"imag_mode": "/tmp/imag_mode.npy"}}
    monkeypatch.setattr(refine_mod, "run", fake_refine)

    def fake_validate(atoms, calc, **kw):
        rec["validate_called"] = True
        return {"overall_pass": True,
                "branches": [{"energy_eV": -5.1, "output_pdb": None},
                             {"energy_eV": -5.3, "output_pdb": None}]}
    monkeypatch.setattr(imd_mod, "run", fake_validate)

    def fake_scan(atoms, calc, **kw):
        rec["scan_called"] = True
        return {"coord_values": [0.0, 2.0], "frames": [_atoms(), _atoms()]}
    monkeypatch.setattr(sm_mod, "run", fake_scan)

    return rec


# ---- validation ----
def test_rejects_bad_entry_and_rigor(tmp_path):
    with pytest.raises(ValueError, match="entry="):
        ts_entry.run(_resolved(), _ctx(), entry="bogus", outdir=tmp_path)
    with pytest.raises(ValueError, match="rigor="):
        ts_entry.run(_resolved(), _ctx(), entry="ts-guess", outdir=tmp_path,
                     ts_guess=_atoms(), rigor="ultra")


# ---- ts-guess branch ----
def test_ts_guess_refines_and_passes(patched, tmp_path):
    res = ts_entry.run(_resolved(), _ctx(), entry="ts-guess", ts_guess=_atoms(),
                       outdir=tmp_path, rigor="standard")
    assert res["status"] == "converged"
    assert res["imag_freq_cm"] == -520.0 and res["n_imag"] == 1
    # reactive atoms passed unambiguously as 0-based "0:idx" tokens
    assert patched["refine_kwargs"]["reactive_atoms"] == ["0:0", "0:1", "0:2"]
    # standard rigor validates
    assert patched["validate_called"] is True


def test_ts_guess_critical_fail_when_two_imag(patched, tmp_path, monkeypatch):
    import quantum_engine.ops.refine_ts as refine_mod

    def two_imag(atoms, calc, **kw):
        return {"status": "failed_validation", "overall_pass": False,
                "ts_atoms": atoms, "n_imag": 2, "imag_freq_cm": -300.0,
                "energy_eV": -4.0,
                "report": SimpleNamespace(imag_mode_overlap=0.1),
                "outputs": {"imag_mode": None}}
    monkeypatch.setattr(refine_mod, "run", two_imag)
    res = ts_entry.run(_resolved(), _ctx(), entry="ts-guess", ts_guess=_atoms(),
                       outdir=tmp_path, rigor="standard")
    assert res["status"] == "failed" and res["critical_fail"] is True
    # validation must be SKIPPED on a critical fail
    assert patched["validate_called"] is False


# ---- reactant-product branch ----
def test_reactant_product_full_flow(patched, tmp_path):
    res = ts_entry.run(_resolved(), _ctx(), entry="reactant-product",
                       reactant=_atoms(), product=_atoms(), outdir=tmp_path,
                       rigor="standard", n_images=9)
    assert res["status"] == "converged"
    assert patched["opt_calls"] == 2                 # both endpoints relaxed
    assert patched["path_kwargs"]["method"] == "neb"
    assert patched["path_kwargs"]["n_images"] == 9   # override honored
    # barrier = (E_ts - E_reactant) * kcal: (-4.0 - -5.0)*23.06 ≈ 23.06
    assert res["barrier_fwd_kcal"] == pytest.approx(23.06, abs=0.2)


def test_reactant_product_requires_both(patched, tmp_path):
    with pytest.raises(ValueError, match="needs reactant= and product="):
        ts_entry.run(_resolved(), _ctx(), entry="reactant-product",
                     reactant=_atoms(), outdir=tmp_path)


# ---- reactant-only branch ----
def test_reactant_only_requires_cv(patched, tmp_path):
    with pytest.raises(ValueError, match="needs a CV bond-difference"):
        ts_entry.run(_resolved(cv=False), _ctx(), entry="reactant-only",
                     reactant=_atoms(), outdir=tmp_path)


def test_reactant_only_drives_then_refines(patched, tmp_path):
    res = ts_entry.run(_resolved(cv=True), _ctx(), entry="reactant-only",
                       reactant=_atoms(), outdir=tmp_path, rigor="draft")
    assert patched["scan_called"] is True            # CV-drive generated product
    assert res["status"] == "converged"


# ---- rigor presets ----
def test_draft_skips_validation(patched, tmp_path):
    ts_entry.run(_resolved(), _ctx(), entry="ts-guess", ts_guess=_atoms(),
                 outdir=tmp_path, rigor="draft")
    assert patched["validate_called"] is False       # draft.validate == False


def test_validate_override_forces_validation(patched, tmp_path):
    ts_entry.run(_resolved(), _ctx(), entry="ts-guess", ts_guess=_atoms(),
                 outdir=tmp_path, rigor="draft", validate=True)
    assert patched["validate_called"] is True


# ---- engine guard ----
def test_engine_not_none_raises_phase6(tmp_path):
    ctx = RunContext(engine="orca", model="dft")
    with pytest.raises(NotImplementedError, match="Phase 6"):
        ts_entry.run(_resolved(), ctx, entry="ts-guess", ts_guess=_atoms(),
                     outdir=tmp_path)


# ---- gates.json emitted ----
def test_gates_json_emitted(patched, tmp_path):
    ts_entry.run(_resolved(), _ctx(), entry="ts-guess", ts_guess=_atoms(),
                 outdir=tmp_path, rigor="standard")
    assert (tmp_path / "gates.json").exists()
