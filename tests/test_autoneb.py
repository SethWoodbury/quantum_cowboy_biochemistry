"""Tests for the AutoNEB adaptive-image path method (ASE-native).

Registry membership is fast; the end-to-end adaptive band (EMT, a real Cu-hop
barrier) is slow-marked.
"""
from __future__ import annotations

import tempfile

import pytest


def test_autoneb_registered():
    from quantum_engine.ops.path_search import PATH_METHODS, make_path_method
    assert "autoneb" in PATH_METHODS.names()
    assert make_path_method("autoneb").__name__ == "_run_autoneb"


def _cu(dx):
    from ase import Atoms
    return Atoms("Cu5", positions=[[0, 0, 0], [2.55, 0, 0], [1.27, 2.2, 0],
                                   [3.8, 2.2, 0], [1.27 + dx, 1.0, 1.8]])


@pytest.mark.slow
def test_autoneb_end_to_end():
    """AutoNEB grows an adaptive band over a real EMT barrier and returns the
    standard path dict with a TS guess at the peak."""
    from ase.calculators.emt import EMT
    from quantum_engine.ops import path_search
    with tempfile.TemporaryDirectory() as td:
        res = path_search.run("autoneb", _cu(0.0), _cu(2.55), lambda: EMT(),
                              outdir=td, charge=0, n_images=6, n_seed=3,
                              fmax=0.2, maxsteps=80, climb=True)
    assert res["status"] == "converged"
    assert res["n_images_final"] >= 3
    assert res["ts"] is not None and res["barrier_fwd_kcal"] > 0
    # standard schema keys present
    for k in ("images", "ts_idx", "reactant", "product", "energies_eV",
              "barrier_rev_kcal", "outputs"):
        assert k in res


@pytest.mark.slow
def test_autoneb_climb_false_band():
    from ase.calculators.emt import EMT
    from quantum_engine.ops import autoneb
    with tempfile.TemporaryDirectory() as td:
        res = autoneb.run(_cu(0.0), _cu(2.55), lambda: EMT(), outdir=td,
                          n_images=6, n_seed=3, fmax=0.2, maxsteps=60, climb=False)
    assert res["status"] == "converged" and res["n_images_final"] >= 3
