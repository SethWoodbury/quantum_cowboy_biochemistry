"""Tests for the energy-function family registry (the primary plug-and-play axis).

Pure routing tests — no model weights / GPU. They check that model aliases map
to the right family, that registration order = match priority, and that a NEW
family drops in via one register_energy() call and intercepts dispatch.
"""
from __future__ import annotations

import pytest

from quantum_engine.calc.factory import (
    ENERGY_FAMILIES, _family_of, register_energy, make_calc)


@pytest.mark.parametrize("alias,family", [
    ("gfn2-xtb", "qc"), ("gfn1-xtb", "qc"), ("gfn-ff", "qc"),
    ("g-xtb", "qc"), ("xtb", "qc"),
    ("mace-omol", "mace"), ("mace-polar-m", "mace"), ("mace-mh-1", "mace"),
    ("uma-s-1p1", "uma"), ("UMA-M-1P1", "uma"),
    ("orb-mol", "orb"), ("aimnet2-rxn", "aimnet"),
    ("/abs/path/to/model.model", "mace"),   # absolute path → catch-all mace
])
def test_family_of_routing(alias, family):
    assert _family_of(alias) == family


def test_mace_is_implicit_default_not_registered():
    # mace is NOT a registry entry — it's the fallback when nothing matches, so
    # it can never shadow a newly-registered family appended after it.
    assert "mace" not in ENERGY_FAMILIES.labels()
    assert ENERGY_FAMILIES.labels() == ["uma", "esen", "orb", "aimnet", "qc"]


def test_match_returns_label_and_builder():
    label, builder = ENERGY_FAMILIES.match("uma-s-1p1")
    assert label == "uma" and callable(builder)
    # No predicate matches -> (None, None); make_calc falls back to mace.
    assert ENERGY_FAMILIES.match("something-weird") == (None, None)
    assert _family_of("something-weird") == "mace"


def test_register_energy_new_family_intercepts():
    """A new family registered via the public API must intercept its aliases and
    route make_calc to its builder (one-liner extensibility). Clean up after."""
    built = {}

    def fake_builder(model, *, model_path, registry_path, head, device,
                     default_dtype, charge, spin):
        built["model"] = model
        built["charge"] = charge
        return "FAKE_CALC"

    register_energy("myff", lambda m: m.lower().startswith("myff-"), fake_builder)
    try:
        assert _family_of("myff-test") == "myff"
        out = make_calc("myff-test", device="cpu", charge=-2)
        assert out == "FAKE_CALC" and built["model"] == "myff-test"
        assert built["charge"] == -2
        # non-matching aliases still route to their real families / mace default
        assert _family_of("mace-omol") == "mace"
        assert _family_of("uma-s-1p1") == "uma"
    finally:
        ENERGY_FAMILIES.unregister("myff")
    assert _family_of("myff-test") == "mace"   # registry restored


def test_register_energy_public_api_appends():
    """register_energy() appends a family; verify it lands and clean up."""
    try:
        register_energy("zzz", lambda m: m == "zzz-only", lambda *a, **k: "Z")
        assert "zzz" in ENERGY_FAMILIES.labels()
        assert ENERGY_FAMILIES.match("zzz-only")[0] == "zzz"
    finally:
        ENERGY_FAMILIES.unregister("zzz")
    assert "zzz" not in ENERGY_FAMILIES.labels()
