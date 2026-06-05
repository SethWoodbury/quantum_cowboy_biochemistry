"""Smoke tests for the Phase 7 CLI subcommands (reaction-spec / monitor / ts-entry).

These run the real CLI dispatch (no heavy compute): reaction-spec validate+resolve
and monitor are compute-free; ts-entry is checked for arg wiring + the engine
prepare-only path.
"""
from __future__ import annotations

import textwrap

import pytest

from quantum_engine import cli


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text))
    return str(p)


def _co2(tmp_path):
    return _write(tmp_path, "co2.xyz",
                  "3\nCO2\nO -1.16 0 0\nC 0 0 0\nO 1.16 0 0\n")


def _spec(tmp_path):
    return _write(tmp_path, "spec.yaml", """
        reaction:
          forming_bonds: [["0:1", "0:0"]]
          breaking_bonds: [["0:1", "0:2"]]
          reactive_atoms: ["0:0", "0:1", "0:2"]
          cv: {kind: bond_difference, atoms: ["0:1", "0:0", "0:2"]}
    """)


def test_reaction_spec_validate_and_resolve(tmp_path, capsys):
    rc = cli.main(["reaction-spec", _spec(tmp_path), "--structure", _co2(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "status: valid" in out and "'cv_bond_difference': (1, 0, 2)" in out


def test_monitor_reports_bonds(tmp_path, capsys):
    rc = cli.main(["monitor", _co2(tmp_path), "--metals", "--bond", "0,1"])
    out = capsys.readouterr().out
    assert rc == 0 and "'state': 'bonded'" in out


def test_ts_entry_engine_prepare_only(tmp_path, capsys):
    """ts-entry --engine orca --no-execute routes to the QM-native engine and
    writes the ORCA input without running ORCA (which isn't in the container)."""
    rc = cli.main([
        "ts-entry", "--entry", "ts-guess",
        "--reaction-spec", _spec(tmp_path), "--ts-guess", _co2(tmp_path),
        "--engine", "orca", "--model", "b3lyp/def2-SVP", "--no-execute",
        "--outdir", str(tmp_path / "out"),
    ])
    out = capsys.readouterr().out
    assert rc == 0 and "status: prepared" in out
    assert (tmp_path / "out" / "optts.inp").exists()


def test_reaction_spec_validate_only(tmp_path, capsys):
    rc = cli.main(["reaction-spec", _spec(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0 and "status: valid" in out and "resolved" not in out
