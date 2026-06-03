"""Tests for the pydantic-v1 qcb config schema (config/schema.py).

Exercises the validators, the operation tagged-union dispatch, and the
cross-reference checks — the parts most at risk in the v2->v1 rewrite.
"""
from __future__ import annotations

import textwrap

import pytest

from quantum_engine.config import schema as S


def _write(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(text))
    return p


def test_valid_config_round_trip(tmp_path):
    cfg = S.load_config(_write(tmp_path, """
        qcb_version: 1
        name: demo
        structure:
          path: /tmp/x.pdb
          charge: -2
          ligand_names: [YYL]
        selectors:
          P1: "residue YYL atoms P1"
          O_nuc: "residue ASP atoms OD2"
        geometry:
          d_form:
            kind: distance
            atoms: [P1, O_nuc]
        constraints:
          - kind: harmonic_restraint
            geom: d_form
            k: 5.0
            r0: 1.65
          - kind: fix_atoms
            selector: P1
        calculator:
          model: mace-mh-1
          head: omol
          device: cpu
        operation:
          kind: ts
          strategy: cv-spring
          cv_atoms: [P1, O_nuc, O_nuc]
    """))
    # selectors normalized from shorthand strings -> SelectorSpec
    assert cfg.selectors["P1"].spec == "residue YYL atoms P1"
    assert S.normalize_selectors(cfg)["O_nuc"] == "residue ASP atoms OD2"
    # operation dispatched to the right subclass via the Literal kind
    assert isinstance(cfg.operation, S.TSOperation)
    assert cfg.operation.strategy == "cv-spring"
    assert cfg.structure.charge == -2


_OP_YAML = {
    "opt": "operation:\n  kind: opt\n  optimizer: fire\n",
    "md": "operation:\n  kind: md\n  total_time_ps: 2.0\n",
    "irc": "operation:\n  kind: irc\n",
    "neb": "operation:\n  kind: neb\n  reactant: /a.pdb\n  product: /b.pdb\n",
    "scan": ("selectors:\n  A: \"atoms CA\"\n  B: \"atoms CB\"\n"
             "geometry:\n  g:\n    kind: distance\n    atoms: [A, B]\n"
             "operation:\n  kind: scan\n  geom: g\n  start: 1.0\n  end: 2.0\n"),
}


@pytest.mark.parametrize("kind,cls", [
    ("opt", S.OptOperation), ("md", S.MDOperation), ("scan", S.ScanOperation),
    ("neb", S.NEBOperation), ("irc", S.IRCOperation),
])
def test_operation_dispatch(tmp_path, kind, cls):
    cfg = S.load_config(_write(tmp_path, "qcb_version: 1\n" + _OP_YAML[kind]))
    assert isinstance(cfg.operation, cls)
    assert cfg.operation.kind == kind


def test_bad_selector_reference_raises(tmp_path):
    with pytest.raises(ValueError, match="undefined selector"):
        S.load_config(_write(tmp_path, """
            qcb_version: 1
            geometry:
              g:
                kind: distance
                atoms: [NOPE, ALSO_NOPE]
        """))


def test_geometry_arity_raises(tmp_path):
    with pytest.raises(ValueError, match="needs 2 selector names"):
        S.load_config(_write(tmp_path, """
            qcb_version: 1
            selectors:
              A: "atoms CA"
            geometry:
              g:
                kind: distance
                atoms: [A]
        """))


def test_bad_selector_token_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown token"):
        S.load_config(_write(tmp_path, """
            qcb_version: 1
            selectors:
              A: "bogustoken CA"
        """))


def test_constraint_kind_field_required(tmp_path):
    with pytest.raises(ValueError, match="requires 'selector'"):
        S.load_config(_write(tmp_path, """
            qcb_version: 1
            constraints:
              - kind: fix_atoms
        """))


def test_operation_without_kind_raises(tmp_path):
    # regression guard: a kind-less operation dict must NOT silently become opt
    with pytest.raises(ValueError, match="must declare a 'kind'"):
        S.load_config(_write(tmp_path, """
            qcb_version: 1
            operation:
              total_time_ps: 50
        """))


def test_defaults_only_config_ok(tmp_path):
    # No structure / operation -> still valid (used as an inheritance base)
    cfg = S.load_config(_write(tmp_path, """
        qcb_version: 1
        calculator:
          model: mace-omol
    """))
    assert cfg.operation is None
    assert cfg.structure is None
    assert cfg.calculator.model == "mace-omol"
