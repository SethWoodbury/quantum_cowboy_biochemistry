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


def test_resolve_token_calls_shared_resolver_with_correct_arg_order(monkeypatch):
    # Guards against the (tokens, atoms, template) vs (atoms, template, tokens) bug.
    import quantum_engine.ops.refine_ts as rt
    captured = {}

    def fake(atoms, bt_template, reactive_atoms):
        captured["atoms"] = atoms
        captured["template"] = bt_template
        captured["tokens"] = list(reactive_atoms)
        return [42]

    monkeypatch.setattr(rt, "_resolve_reactive_indices", fake)
    sentinel_atoms, sentinel_tmpl = object(), object()
    idx = resolve_atom_token("A:169:NZ", atoms=sentinel_atoms, template=sentinel_tmpl)
    assert idx == 42
    assert captured["atoms"] is sentinel_atoms          # 1st arg = atoms
    assert captured["template"] is sentinel_tmpl         # 2nd arg = template
    assert captured["tokens"] == ["A:169:NZ"]            # 3rd arg = tokens


def test_runcontext_known_and_extra():
    ctx = RunContext.from_dict({"charge": -2, "spin": 1, "model": "mace-polar-m",
                                "engine": "orca", "weird_future_knob": 7})
    assert ctx.charge == -2 and ctx.model == "mace-polar-m" and ctx.engine == "orca"
    assert ctx.extra["weird_future_knob"] == 7
