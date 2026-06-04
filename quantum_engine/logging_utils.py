"""Shared structured logging + per-step records for quantum_engine.

Goal (project standard): every pipeline step emits BOTH
  (a) an organized, human-readable log — what it's doing, key values, decisions,
      where outputs went — with a consistent ``[stage]`` prefix, level, and timing; and
  (b) a machine-readable ``<stage>.json`` record — parameters, results, timings,
      status, and (optionally) gate verdicts — so failures are trivial to localize.

Usage
-----
    from quantum_engine.logging_utils import get_logger, configure_logging, Step

    configure_logging("INFO")                     # once, e.g. from the CLI
    log = get_logger("quantum_engine.ops.opt")

    with Step("opt", outdir, params={"model": "mace-omol", "fmax": 0.05}) as step:
        step.log.info("relaxing %d atoms", len(atoms))
        ...
        step.record(energy_eV=e, fmax_final=f, output=str(pdb_path))
        # step.attach_gates(gate_report)          # optional PASS/WARN/FAIL block
    # -> writes <outdir>/opt.json and prints a one-line human SUMMARY

The level is taken from the ``--log-level`` flag (via ``configure_logging``) or the
``QCB_LOG_LEVEL`` env var, defaulting to INFO. Set ``QCB_LOG_TIME=0`` to drop the
timestamp prefix (useful in tests / deterministic logs).
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_ROOT = "quantum_engine"
_CONFIGURED = False


def _fmt() -> logging.Formatter:
    if os.environ.get("QCB_LOG_TIME", "1") == "0":
        return logging.Formatter("[%(name)s] %(levelname)s %(message)s")
    return logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s",
                             datefmt="%H:%M:%S")


def configure_logging(level: str | int | None = None) -> None:
    """Configure the ``quantum_engine`` logger once (idempotent).

    level: name ("DEBUG"/"INFO"/...), int, or None → ``QCB_LOG_LEVEL`` env or INFO.
    """
    global _CONFIGURED
    lvl = _resolve_level(level)
    root = logging.getLogger(_ROOT)
    root.setLevel(lvl)
    # Single stream handler; don't duplicate on repeated calls.
    if not any(getattr(h, "_qcb", False) for h in root.handlers):
        h = logging.StreamHandler()
        h.setFormatter(_fmt())
        h._qcb = True  # type: ignore[attr-defined]
        root.addHandler(h)
    else:
        for h in root.handlers:
            if getattr(h, "_qcb", False):
                h.setFormatter(_fmt())
    root.propagate = False
    _CONFIGURED = True


def _resolve_level(level: str | int | None) -> int:
    if level is None:
        level = os.environ.get("QCB_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        return getattr(logging, level.upper(), logging.INFO)
    return int(level)


def get_logger(name: str | None = None, level: str | int | None = None) -> logging.Logger:
    """Return a logger under the ``quantum_engine`` namespace, configuring on first use."""
    if not _CONFIGURED:
        configure_logging(level)
    if not name:
        name = _ROOT
    elif not name.startswith(_ROOT):
        name = f"{_ROOT}.{name}"
    lg = logging.getLogger(name)
    if level is not None:
        lg.setLevel(_resolve_level(level))
    return lg


def _jsonable(v: Any) -> Any:
    """Best-effort conversion to a JSON-serializable value."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "tolist"):           # numpy arrays/scalars
        try:
            return v.tolist()
        except Exception:              # noqa: BLE001
            return str(v)
    return str(v)


class Step:
    """Context manager wrapping one pipeline step with human + machine output.

    Logs a banner (stage + params) on enter; times the block; on exit logs a
    one-line ``SUMMARY`` and writes ``<outdir>/<stage>.json`` (params, records,
    elapsed, status, optional gate verdicts). On exception: logs the error,
    writes the JSON with ``status="error"``, then re-raises.
    """

    def __init__(self, stage: str, outdir=None, *, params: dict | None = None,
                 logger: logging.Logger | None = None, write_json: bool = True):
        self.stage = stage
        self.outdir = Path(outdir) if outdir is not None else None
        self.params = dict(params or {})
        self.log = logger or get_logger(f"step.{stage}")
        self.write_json = write_json
        self._records: dict[str, Any] = {}
        self._gates: Any = None
        self._t0 = 0.0
        self.status = "running"

    def __enter__(self) -> "Step":
        self._t0 = time.time()
        bar = "─" * 60
        self.log.info("%s", bar)
        self.log.info("[%s] START", self.stage)
        for k, v in self.params.items():
            self.log.info("    %s = %s", k, _jsonable(v))
        return self

    def record(self, **kw: Any) -> None:
        """Add machine-readable result key/values to this step's JSON record."""
        for k, v in kw.items():
            self._records[k] = _jsonable(v)

    def attach_gates(self, gate_report: Any) -> None:
        """Attach a gates.GateReport (its overall status flows into the summary)."""
        self._gates = gate_report

    def _emit_json(self) -> Path | None:
        if not (self.write_json and self.outdir is not None):
            return None
        payload = {
            "stage": self.stage,
            "status": self.status,
            "elapsed_s": round(time.time() - self._t0, 3),
            "params": _jsonable(self.params),
            "results": self._records,
        }
        if self._gates is not None:
            payload["gates_overall"] = getattr(self._gates, "overall", None)
            try:
                from dataclasses import asdict
                payload["gates"] = [asdict(g) for g in getattr(self._gates, "gates", [])]
            except Exception:          # noqa: BLE001
                pass
        self.outdir.mkdir(parents=True, exist_ok=True)
        p = self.outdir / f"{self.stage}.json"
        p.write_text(json.dumps(payload, indent=2, default=str))
        return p

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.time() - self._t0
        if exc_type is not None:
            self.status = "error"
            self.log.error("[%s] FAILED after %.2fs: %s: %s",
                           self.stage, elapsed, exc_type.__name__, exc)
            self._records["error"] = f"{exc_type.__name__}: {exc}"
            self._emit_json()
            return False               # re-raise
        if self.status == "running":
            self.status = "completed"
        gate_tag = f" gates={getattr(self._gates, 'overall', None)}" if self._gates is not None else ""
        jp = self._emit_json()
        self.log.info("[%s] SUMMARY status=%s elapsed=%.2fs%s%s",
                      self.stage, self.status, elapsed, gate_tag,
                      f" -> {jp}" if jp else "")
        return False


@contextmanager
def step(stage: str, outdir=None, **kw):
    """Functional alias for :class:`Step` (``with step('opt', outdir) as s:``)."""
    s = Step(stage, outdir, **kw)
    with s:
        yield s


__all__ = ["configure_logging", "get_logger", "Step", "step"]
