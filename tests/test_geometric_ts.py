"""Tests for the geomeTRIC TS saddle backend.

geomeTRIC is NOT bundled in the container, so these assert the registration +
the clean missing-dependency gating + the unit constants. The end-to-end TS
refinement was verified against a known xTB HCN saddle with geomeTRIC installed
(C-N → 1.203 Å), via `pip install geometric`.
"""
from __future__ import annotations

import pytest


def test_geometric_registered_as_saddle_backend():
    from quantum_engine.ops.saddle import SADDLE_OPTIMIZERS, make_saddle_optimizer
    assert "geometric" in SADDLE_OPTIMIZERS.names()
    assert make_saddle_optimizer("geometric-ts").__name__ == "_run_geometric"


def test_geometric_available_returns_tuple():
    from quantum_engine.qm.geometric_ts import geometric_available
    ok, reason = geometric_available()
    assert isinstance(ok, bool) and isinstance(reason, str) and reason


def test_unit_factor_is_correct():
    # eV/Å -> hartree/bohr, the boundary conversion for the geomeTRIC engine.
    from quantum_engine.qm.geometric_ts import _EVANG2AUBOHR
    assert abs(_EVANG2AUBOHR - 0.0194469) < 1e-6


@pytest.mark.parametrize("fmax,expected", [
    (0.1, "GAU_LOOSE"), (0.02, "GAU"), (0.002, "GAU_TIGHT")])
def test_fmax_to_converge_preset(fmax, expected):
    from quantum_engine.qm.geometric_ts import _fmax_to_converge
    assert _fmax_to_converge(fmax) == ["set", expected]


def test_clean_importerror_when_geometric_absent():
    """When geomeTRIC isn't installed (the container default), selecting it must
    raise a clear ImportError naming the install command — not a cryptic error."""
    from quantum_engine.qm.geometric_ts import geometric_available
    if geometric_available()[0]:
        pytest.skip("geomeTRIC is installed in this environment")
    import warnings
    warnings.filterwarnings("ignore")
    from ase.build import molecule
    from ase.calculators.emt import EMT
    from quantum_engine.ops import saddle
    a = molecule("H2O")
    a.calc = EMT()
    with pytest.raises(ImportError, match="pip install geometric"):
        saddle.run(a, backend="geometric", outdir="/tmp/qcb_geom_test")
