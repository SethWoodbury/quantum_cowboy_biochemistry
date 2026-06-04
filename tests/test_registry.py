"""Tests for the generic plugin registry (registry.py). Pure-Python."""
from __future__ import annotations

import pytest

from quantum_engine.registry import PredicateRegistry, Registry


def test_register_direct_and_get_case_insensitive():
    r = Registry("optimizer")
    r.register("ASE-LBFGS", object(), aliases=("lbfgs",))
    assert "ase-lbfgs" in r
    assert "LBFGS" in r                     # alias, case-insensitive
    assert r.get("Ase-Lbfgs") is r.get("lbfgs")
    assert r.names() == ["ase-lbfgs"]


def test_register_decorator_form():
    r = Registry("path")

    @r.register("neb", aliases=("ci-neb",))
    class NEB:
        pass

    assert r.get("neb") is NEB
    assert r.get("ci-neb") is NEB


def test_overwrite_guard():
    r = Registry("saddle")
    r.register("sella", 1)
    with pytest.raises(ValueError, match="already registered"):
        r.register("sella", 2)
    r.register("sella", 2, overwrite=True)
    assert r.get("sella") == 2


def test_unknown_lists_available():
    r = Registry("engine")
    r.register("orca", object())
    with pytest.raises(KeyError, match="orca"):
        r.get("nwchem")


def test_unregister_removes_aliases():
    r = Registry("min")
    r.register("fire", object(), aliases=("f",))
    r.unregister("fire")
    assert "fire" not in r and "f" not in r


def test_predicate_registry_first_match_priority():
    p = PredicateRegistry("energy")
    p.register("uma", lambda m: m.startswith("uma-"), "uma_builder")
    p.register("mace", lambda m: True, "mace_builder")   # catch-all last
    assert p.match("uma-s-1p1") == ("uma", "uma_builder")
    assert p.match("mace-omol") == ("mace", "mace_builder")
    assert p.labels() == ["uma", "mace"]


def test_predicate_registry_no_match():
    p = PredicateRegistry("energy")
    p.register("uma", lambda m: m.startswith("uma-"), "x")
    assert p.match("mace-omol") == (None, None)


def test_predicate_registry_bad_predicate_skipped():
    p = PredicateRegistry("energy")
    p.register("bad", lambda m: 1 / 0, "x")              # raises -> skipped
    p.register("ok", lambda m: True, "y")
    assert p.match("anything") == ("ok", "y")
