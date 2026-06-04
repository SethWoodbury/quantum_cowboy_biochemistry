"""Tests for the structured logging + per-step record utility. Pure-Python."""
from __future__ import annotations

import json
import logging

import pytest

from quantum_engine import logging_utils as L
from quantum_engine.ops.gates import Gate, GateReport


def test_get_logger_namespaced_and_idempotent():
    a = L.get_logger("ops.opt")
    b = L.get_logger("ops.opt")
    assert a is b
    assert a.name == "quantum_engine.ops.opt"
    root = logging.getLogger("quantum_engine")
    qcb_handlers = [h for h in root.handlers if getattr(h, "_qcb", False)]
    assert len(qcb_handlers) == 1                    # no duplicate handlers


def test_step_writes_completed_json(tmp_path):
    with L.Step("opt", tmp_path, params={"model": "mace-omol", "fmax": 0.05}) as s:
        s.record(energy_eV=-12.3, output=str(tmp_path / "relaxed.pdb"))
    data = json.loads((tmp_path / "opt.json").read_text())
    assert data["stage"] == "opt"
    assert data["status"] == "completed"
    assert data["params"]["model"] == "mace-omol"
    assert data["results"]["energy_eV"] == -12.3
    assert "elapsed_s" in data


def test_step_records_error_and_reraises(tmp_path):
    with pytest.raises(ValueError):
        with L.Step("scan", tmp_path) as s:
            s.record(started=True)
            raise ValueError("boom")
    data = json.loads((tmp_path / "scan.json").read_text())
    assert data["status"] == "error"
    assert "ValueError: boom" in data["results"]["error"]


def test_step_attaches_gate_overall(tmp_path):
    rep = GateReport("refine_ts")
    rep.add(Gate("imag", "WARN", "borderline"))
    with L.Step("refine_ts", tmp_path) as s:
        s.attach_gates(rep)
    data = json.loads((tmp_path / "refine_ts.json").read_text())
    assert data["gates_overall"] == "WARN"
    assert data["gates"][0]["name"] == "imag"


def test_step_no_outdir_skips_json(tmp_path, capsys):
    # outdir=None -> no file, but no crash
    with L.Step("sp", None, params={"a": 1}) as s:
        s.record(ok=True)
    assert not list(tmp_path.glob("*.json"))
