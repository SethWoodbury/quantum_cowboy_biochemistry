"""Tests for ReactionSpec / RunContext (reaction_spec.py). Pure-Python (template-free tokens)."""
from __future__ import annotations

import pytest

from quantum_engine.reaction_spec import (
    BondSpec, ReactionSpec, RunContext, resolve_atom_token,
)


def test_resolve_atom_token_template_free():
    assert resolve_atom_token(5) == 5
    assert resolve_atom_token("0:7") == 7
    assert resolve_atom_token("3") == 3            # bare int string, no structure


def test_from_dict_list_and_dict_bond_forms():
    spec = ReactionSpec.from_dict({
        "forming_bonds": [[0, 1]],                 # list form
        "breaking_bonds": [{"a": "0:0", "b": "0:2", "r0": 2.4}],  # dict form
        "reactive_atoms": [0, 1, 2],
        "cv": {"kind": "bond_difference", "atoms": [0, 1, 2]},
        "atom_map": {0: 0, 1: 2, 2: 1},
    })
    assert spec.n_changing == 2
    assert spec.breaking_bonds[0].r0 == 2.4
    assert spec.atom_map == {0: 0, 1: 2, 2: 1}


def test_resolve_indices():
    spec = ReactionSpec.from_dict({
        "forming_bonds": [["0:0", "0:1"]],
        "breaking_bonds": [["0:0", "0:2"]],
        "reactive_atoms": ["0:0", "0:1", "0:2"],
        "cv": {"kind": "bond_difference", "atoms": ["0:0", "0:1", "0:2"]},
    })
    r = spec.resolve()                              # no structure needed
    assert r.forming == [(0, 1)]
    assert r.breaking == [(0, 2)]
    assert r.reactive == [0, 1, 2]
    assert r.cv_bond_difference == (0, 1, 2)


def test_validate_rejects_bad_cv():
    with pytest.raises(ValueError, match="bond_difference"):
        ReactionSpec.from_dict({"cv": {"kind": "bond_difference", "atoms": [0, 1]}})
    with pytest.raises(ValueError, match="must declare a 'kind'"):
        ReactionSpec.from_dict({"cv": {"atoms": [0, 1, 2]}})


def test_validate_rejects_bad_driving_coord():
    with pytest.raises(ValueError, match="driving_coord"):
        ReactionSpec.from_dict({"driving_coords": [{"kind": "BREAK"}]})  # no atoms


def test_resolve_token_routes_to_central_resolver_with_template(monkeypatch):
    # With a structure template, resolve_atom_token routes to the central resolver
    # (atom_descriptor.resolve_atom) with the template-built AtomTable, using the
    # 1-based-serial convention for bare ints (the PDB-viewer convention).
    import quantum_engine.io.atom_descriptor as ad
    captured = {}

    def fake_resolve_atom(token, table, *, bare_int="index"):
        captured.update(token=token, table=table, bare_int=bare_int)
        return 42

    monkeypatch.setattr(ad, "resolve_atom", fake_resolve_atom)
    monkeypatch.setattr(ad.AtomTable, "from_biotite", staticmethod(lambda bt: ("TABLE", bt)))
    sentinel_tmpl = object()
    idx = resolve_atom_token("OHX-O3", template=sentinel_tmpl)
    assert idx == 42
    assert captured["token"] == "OHX-O3"
    assert captured["table"] == ("TABLE", sentinel_tmpl)
    assert captured["bare_int"] == "serial"


def test_runcontext_known_and_extra():
    ctx = RunContext.from_dict({"charge": -2, "spin": 1, "model": "mace-polar-m",
                                "engine": "orca", "weird_future_knob": 7})
    assert ctx.charge == -2 and ctx.model == "mace-polar-m" and ctx.engine == "orca"
    assert ctx.extra["weird_future_knob"] == 7
