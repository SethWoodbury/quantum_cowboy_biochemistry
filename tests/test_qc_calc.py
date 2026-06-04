"""Tests for the semiempirical (xTB/g-xTB) calculator routing + GFN2-metal guard.

These don't run xtb — they test family routing, method validation, and the
charge-awareness guard. A real xtb energy is a container smoke (binary needed).
"""
from __future__ import annotations

import pytest

from quantum_engine.calc.factory import _family_of
from quantum_engine.calc.qc_calc import make_qc_calc


def test_family_routing():
    assert _family_of("gfn2-xtb") == "qc"
    assert _family_of("gfn1-xtb") == "qc"
    assert _family_of("gfn-ff") == "qc"
    assert _family_of("g-xtb") == "qc"
    assert _family_of("xtb") == "qc"
    assert _family_of("mace-omol") == "mace"
    assert _family_of("uma-s-1p1") == "uma"


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown semiempirical method"):
        make_qc_calc("totally-not-xtb")


def test_gfn2_metal_guard():
    ase = pytest.importorskip("ase")
    from ase import Atoms
    znwater = Atoms("ZnOHH", positions=[[0, 0, 0], [2, 0, 0], [2.6, 0.6, 0], [2.6, -0.6, 0]])
    # default: warn but return a calculator
    calc = make_qc_calc("gfn2-xtb", charge=2, spin=1, atoms=znwater)
    assert calc is not None
    # forbid: raise with a charge-aware-MLFF hint
    with pytest.raises(ValueError, match="charge"):
        make_qc_calc("gfn2-xtb", charge=2, spin=1, atoms=znwater, forbid_metal_gfn2=True)


def test_gfn2_organic_ok_no_metal():
    ase = pytest.importorskip("ase")
    from ase import Atoms
    water = Atoms("OHH", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    calc = make_qc_calc("gfn2-xtb", charge=0, spin=1, atoms=water)  # no metal -> no raise
    assert calc is not None
