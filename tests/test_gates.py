"""Tests for the acceptance-gate mechanism (ops/gates.py). Pure-Python."""
from __future__ import annotations

import json

from quantum_engine.ops.gates import Gate, GateReport, gate_threshold


def test_overall_precedence():
    r = GateReport("s")
    assert r.overall == "PASS"                       # empty → PASS
    r.add(Gate("a", "PASS", ""))
    r.add(Gate("b", "WARN", ""))
    assert r.overall == "WARN"
    r.add(Gate("c", "FAIL", "", fallback="do x"))
    assert r.overall == "FAIL"


def test_critical_fail_depends_on_fallback():
    r = GateReport("s")
    r.add(Gate("a", "FAIL", "", fallback="try dimer"))
    assert r.overall == "FAIL"
    assert r.critical_fail is False                  # recoverable
    r.add(Gate("b", "FAIL", ""))                     # no fallback
    assert r.critical_fail is True


def test_gate_threshold_pass_warn_fail():
    # in range
    assert gate_threshold("x", 1.0, lo=0.0, hi=2.0).status == "PASS"
    # below, no warn band -> FAIL
    g = gate_threshold("x", -1.0, lo=0.0, hi=2.0, fallback="widen")
    assert g.status == "FAIL" and g.fallback == "widen"
    # one-sided upper bound: imaginary freq must be < -50
    assert gate_threshold("imag", -120.0, hi=-50.0).status == "PASS"
    assert gate_threshold("imag", -10.0, hi=-50.0).status == "FAIL"
    # warn band: value just above hi within warn_frac
    assert gate_threshold("x", 2.1, hi=2.0, warn_frac=0.1).status == "WARN"
    assert gate_threshold("x", 2.5, hi=2.0, warn_frac=0.1).status == "FAIL"


def test_gate_threshold_non_numeric():
    g = gate_threshold("x", None, lo=0.0)
    assert g.status == "FAIL"


def test_emit_writes_json(tmp_path):
    r = GateReport("refine_ts")
    r.add(Gate("imag", "PASS", "one imag mode", value=-300.0, threshold=-50.0))
    p = r.emit(tmp_path)
    assert p.name == "gates.json"
    data = json.loads(p.read_text())
    assert data["step"] == "refine_ts"
    assert data["overall"] == "PASS"
    assert data["gates"][0]["name"] == "imag"
