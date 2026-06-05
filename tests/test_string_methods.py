"""Tests for the version-adaptive pysisyphus string-method driver (FSM/GSM).

The capability probe + the FSM "unsupported, don't crash" path are fast (no
compute). The GSM end-to-end run is slow-marked (real EMT optimization).
"""
from __future__ import annotations

import tempfile

import pytest

pytest.importorskip("pysisyphus")
from ase.build import molecule  # noqa: E402

from quantum_engine.qm import pysis_string  # noqa: E402
from quantum_engine.ops import gsm  # noqa: E402


def _mk(seed, dz=0.0):
    a = molecule("H2CO")
    a.positions[0, 2] += dz
    a.rattle(0.02, seed=seed)
    return a


def test_capabilities_reports_methods():
    caps = pysis_string.capabilities()
    assert "version" in caps
    for m in ("gsm", "fsm"):
        ok, reason = caps[m]
        assert isinstance(ok, bool) and isinstance(reason, str) and reason
    # cached (same object on second call)
    assert pysis_string.capabilities() is caps


def test_fsm_unsupported_is_clean_not_a_crash():
    """When FSM is unsupported on the installed pysisyphus, run_fsm returns a
    clear status + actionable error rather than raising. (No compute: the probe
    short-circuits before any force evaluation.)"""
    ok, _ = pysis_string.capabilities()["fsm"]
    with tempfile.TemporaryDirectory() as td:
        res = gsm.run_fsm(_mk(1), _mk(2, 0.3), lambda: _dummy_calc(), outdir=td,
                          charge=0, mult=1, fmax=0.1, max_nodes=4)
    if ok:
        # a future pysisyphus where FSM works
        assert res["status"] in ("converged", "not_converged")
    else:
        assert res["status"] == "unsupported"
        assert "gsm-de" in res["error"] or "neb" in res["error"]


def _dummy_calc():
    # Never actually called on the unsupported FSM path.
    from ase.calculators.emt import EMT
    return EMT()


@pytest.mark.slow
def test_gsm_runs_end_to_end():
    """GSM (GrowingString) drives to a converged string via the driver — the
    endpoint-calculator fix is what makes this work on this pysisyphus."""
    from ase.calculators.emt import EMT
    with tempfile.TemporaryDirectory() as td:
        res = gsm.run_gsm(_mk(1), _mk(2, 0.3), lambda: EMT(), outdir=td,
                          charge=0, mult=1, fmax=0.1, n_images=5)
    assert res["status"] == "converged"
    assert res["n_images_final"] >= 3
    assert res["ts"] is not None and "barrier_fwd_kcal" in res


@pytest.mark.slow
def test_path_search_gsm_de_end_to_end():
    from ase.calculators.emt import EMT
    from quantum_engine.ops import path_search
    with tempfile.TemporaryDirectory() as td:
        res = path_search.run("gsm-de", _mk(1), _mk(2, 0.3), lambda: EMT(),
                              outdir=td, charge=0, n_images=5, fmax=0.1)
    assert res["status"] == "converged" and res["n_images_final"] >= 3
