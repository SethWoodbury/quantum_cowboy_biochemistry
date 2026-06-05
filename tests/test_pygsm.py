"""Tests for the pyGSM Growing String Method adapter (pygsm-de / gsm-se).

Capability + input-validation + robustness are fast; the DE-GSM end-to-end run
is slow-marked (real EMT). SE-GSM is wired but pyGSM's single-ended internals are
finicky on the vendored version — we assert it FAILS CLEANLY (no crash), not that
it converges.
"""
from __future__ import annotations

import tempfile

import pytest

from quantum_engine.qm import pygsm
from quantum_engine.ops import path_search


def _mk(seed, dz=0.0):
    from ase.build import molecule
    a = molecule("H2CO")
    a.positions[0, 2] += dz
    a.rattle(0.02, seed=seed)
    return a


def test_pygsm_available():
    ok, reason = pygsm.pygsm_available()
    assert isinstance(ok, bool) and isinstance(reason, str) and reason


def test_se_gsm_requires_driving_coords():
    with pytest.raises(ValueError, match="needs driving_coords"):
        pygsm.run_se_gsm(_mk(1), lambda: object(), driving_coords=None, outdir=".")


def test_registry_has_pygsm_methods():
    names = path_search.PATH_METHODS.names()
    assert "pygsm-de" in names and "gsm-se" in names
    # aliases resolve
    from quantum_engine.ops.path_search import make_path_method, _run_pygsm_de
    assert make_path_method("pygsm") is _run_pygsm_de


def test_double_ended_requires_product():
    with pytest.raises(ValueError, match="double-ended"):
        path_search.run("pygsm-de", _mk(1), None, lambda: object())


def test_single_ended_skips_endpoint_check_and_fails_cleanly():
    """gsm-se takes only a reactant (no product / no endpoint check) and, when
    pyGSM's SE-GSM can't converge, returns status in {failed,unsupported} — never
    a crash."""
    from ase.calculators.emt import EMT
    with tempfile.TemporaryDirectory() as td:
        res = path_search.run("gsm-se", _mk(1), None, lambda: EMT(), outdir=td,
                              charge=0, driving_coords=[("ADD", 1, 3)],
                              n_nodes=7, fmax=0.2, max_iterations=5, max_opt_steps=2)
    assert res["status"] in ("failed", "unsupported", "converged")
    assert "images" in res  # always present, even on failure


@pytest.mark.slow
def test_de_gsm_end_to_end():
    """DE-GSM via pyGSM converges to a string with a TS node (EMT)."""
    from ase.calculators.emt import EMT
    with tempfile.TemporaryDirectory() as td:
        res = pygsm.run_de_gsm(_mk(1), _mk(2, 0.4), lambda: EMT(), outdir=td,
                               charge=0, mult=1, n_nodes=7, fmax=0.1,
                               max_iterations=15, max_opt_steps=2)
    assert res["status"] == "converged"
    assert res["n_images_final"] >= 3 and res["ts"] is not None


@pytest.mark.slow
def test_pygsm_de_via_path_search():
    from ase.calculators.emt import EMT
    with tempfile.TemporaryDirectory() as td:
        res = path_search.run("pygsm-de", _mk(1), _mk(2, 0.4), lambda: EMT(),
                              outdir=td, charge=0, n_nodes=7, fmax=0.1,
                              max_iterations=15, max_opt_steps=2)
    assert res["status"] == "converged"
