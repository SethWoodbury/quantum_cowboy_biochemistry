"""Acceptance gates — uniform PASS / WARN / FAIL checks for every pipeline step.

Each step builds a :class:`GateReport` of :class:`Gate` checks and emits it to
``gates.json``. The orchestrator (``ts_entry``) reads each step's report and
**aborts only on a critical failure** — a ``FAIL`` with no ``fallback`` — so a
recoverable failure logs its fallback suggestion and the pipeline can branch.

This generalizes the per-check pattern already used by ``ops/refine_ts.py``
(``RefineTSCheck``) so every step speaks the same verdict language.

    from quantum_engine.ops.gates import Gate, GateReport, gate_threshold

    rep = GateReport(step="refine_ts")
    rep.add(gate_threshold("imag_freq_cm", imag, hi=-50.0,
                           detail="one imaginary mode below -50 cm^-1",
                           fallback="re-run saddle with --backend dimer"))
    rep.add(Gate("imag_overlap", "PASS" if ov >= 0.5 else "FAIL",
                 f"{ov:.0%} of imag mode on reactive atoms", value=ov, threshold=0.5))
    rep.emit(outdir)
    if rep.critical_fail: ...
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Status = Literal["PASS", "WARN", "FAIL"]
_RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}


@dataclass
class Gate:
    """One acceptance check.

    fallback: a human-actionable suggestion. A ``FAIL`` *with* a fallback is
    recoverable (the orchestrator can try the fallback); a ``FAIL`` *without*
    one is a critical failure that aborts the step.
    """
    name: str
    status: Status
    detail: str = ""
    value: Any = None
    threshold: Any = None
    fallback: str | None = None


@dataclass
class GateReport:
    step: str
    gates: list[Gate] = field(default_factory=list)

    def add(self, gate: Gate) -> Gate:
        self.gates.append(gate)
        return gate

    @property
    def overall(self) -> Status:
        if not self.gates:
            return "PASS"
        return max((g.status for g in self.gates), key=lambda s: _RANK[s])

    @property
    def critical_fail(self) -> bool:
        """True if any gate FAILed without an actionable fallback."""
        return any(g.status == "FAIL" and not g.fallback for g in self.gates)

    @property
    def failures(self) -> list[Gate]:
        return [g for g in self.gates if g.status == "FAIL"]

    def emit(self, outdir) -> Path:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        p = outdir / "gates.json"
        payload = {"step": self.step, "overall": self.overall,
                   "critical_fail": self.critical_fail,
                   "gates": [asdict(g) for g in self.gates]}
        p.write_text(json.dumps(payload, indent=2, default=str))
        return p


def gate_threshold(name: str, value: float, *, lo: float | None = None,
                   hi: float | None = None, warn_frac: float = 0.0,
                   fallback: str | None = None, detail: str = "") -> Gate:
    """PASS if ``lo <= value <= hi``; WARN if within ``warn_frac`` of a bound;
    else FAIL. Either bound may be None (one-sided). ``warn_frac`` is a fraction
    of the bound magnitude defining a soft margin (0 = no WARN band).
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return Gate(name, "FAIL", detail or f"{name} is not numeric ({value!r})",
                    value=value, threshold=(lo, hi), fallback=fallback)

    below = lo is not None and v < lo
    above = hi is not None and v > hi
    status: Status
    if below or above:
        # soft WARN band just outside the bound
        bound = lo if below else hi
        margin = abs(bound) * warn_frac if bound is not None else 0.0
        if margin and ((below and v >= lo - margin) or (above and v <= hi + margin)):
            status = "WARN"
        else:
            status = "FAIL"
    else:
        status = "PASS"
    bnd = (lo, hi)
    return Gate(name, status, detail or f"{name}={v:g} (allowed {bnd})",
                value=v, threshold=bnd, fallback=fallback if status == "FAIL" else None)


__all__ = ["Gate", "GateReport", "gate_threshold", "Status"]
