"""Version-adaptive driver for pysisyphus STRING methods (FSM / GSM).

pysisyphus's chain-of-states (COS) + optimizer API drifts between versions, and
the freezing/growing string classes are wired subtly differently. This module
isolates **all** of that version-specific glue in ONE place and **probes
capability at runtime**, so the rest of the codebase (``ops/gsm.py`` →
``ops/path_search.py``) is insulated:

- **GSM** (``GrowingString``, a real ``ChainOfStates``) is driven via
  ``StringOptimizer``. The one non-obvious requirement: pysisyphus only attaches
  calculators to the nodes it GROWS, so we must attach one to each ENDPOINT
  geometry ourselves before constructing the COS (else its first force eval hits
  ``NoneType``).
- **FSM** (``FreezingString``) is driven only if the installed pysisyphus exposes
  a compatible driver. In the version probed here it is NOT a ``ChainOfStates``
  (``StringOptimizer`` asserts COS-only) and the generic optimizers don't call
  its ``reparametrize`` (so the string never grows). We therefore PROBE it and,
  when unsupported, return a clear status + a logged warning pointing at
  ``gsm-de`` / ``neb`` — never a cryptic crash. The probe means FSM **auto-enables**
  the day a pysisyphus bump makes ``FreezingString`` StringOptimizer-compatible.

Version dependencies (the only things to revisit on a pysisyphus bump) live in
:func:`_probe` and the two ``drive_*`` functions — each is small and documented.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from quantum_engine.logging_utils import get_logger

log = get_logger("qm.pysis_string")

_CAPS: dict | None = None


def _probe() -> dict:
    """Probe the installed pysisyphus for FSM/GSM drivability (cached + logged).

    Returns ``{"version": str, "gsm": (ok, reason), "fsm": (ok, reason)}``.
    """
    global _CAPS
    if _CAPS is not None:
        return _CAPS
    caps: dict = {"version": "?", "gsm": (False, "unprobed"), "fsm": (False, "unprobed")}
    try:
        import pysisyphus
        caps["version"] = getattr(pysisyphus, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        caps = {"version": "?", "gsm": (False, f"pysisyphus import failed: {exc}"),
                "fsm": (False, f"pysisyphus import failed: {exc}")}
        _CAPS = caps
        return caps

    # GSM: GrowingString must be a ChainOfStates (StringOptimizer-compatible).
    try:
        from pysisyphus.cos.ChainOfStates import ChainOfStates
        from pysisyphus.cos.GrowingString import GrowingString
        if issubclass(GrowingString, ChainOfStates):
            caps["gsm"] = (True, "GrowingString is a ChainOfStates")
        else:
            caps["gsm"] = (False, "GrowingString is not a ChainOfStates")
    except Exception as exc:  # noqa: BLE001
        caps["gsm"] = (False, f"import failed: {exc}")

    # FSM: FreezingString must be StringOptimizer-compatible (a ChainOfStates).
    try:
        from pysisyphus.cos.ChainOfStates import ChainOfStates
        from pysisyphus.cos.FreezingString import FreezingString
        if issubclass(FreezingString, ChainOfStates):
            caps["fsm"] = (True, "FreezingString is a ChainOfStates")
        else:
            caps["fsm"] = (
                False,
                "FreezingString is not a ChainOfStates on pysisyphus "
                f"{caps['version']}; StringOptimizer rejects it and the generic "
                "optimizers don't call its reparametrize() — use 'gsm-de' or 'neb'.")
    except Exception as exc:  # noqa: BLE001
        caps["fsm"] = (False, f"import failed: {exc}")

    log.info("pysisyphus %s string-method capability: GSM=%s FSM=%s",
             caps["version"], caps["gsm"][0], caps["fsm"][0])
    if not caps["fsm"][0]:
        log.warning("FSM unavailable: %s", caps["fsm"][1])
    _CAPS = caps
    return caps


def capabilities() -> dict:
    """Public capability report (probes once, cached)."""
    return _probe()


def _string_optimizer(cos, *, out_dir, max_cycles, thresh):
    from pysisyphus.optimizers.StringOptimizer import StringOptimizer
    return StringOptimizer(cos, out_dir=str(out_dir), max_cycles=max_cycles,
                           thresh=thresh, dump=False)


def drive_growing_string(geom_r, geom_p, calc_getter: Callable, *, max_nodes: int,
                         fmax: float, max_cycles: int, out_dir) -> dict:
    """Drive GSM and return ``{status, converged, images, error}``.

    ``images`` is the final list of pysisyphus ``Geometry`` objects (bohr); the
    caller converts them to ASE.
    """
    ok, reason = _probe()["gsm"]
    if not ok:
        log.warning("GSM unavailable: %s", reason)
        return {"status": "unsupported", "converged": False, "images": [],
                "error": f"GSM unsupported on this pysisyphus: {reason}"}
    from pysisyphus.cos.GrowingString import GrowingString
    from quantum_engine.qm.pysisyphus import _fmax_to_thresh  # noqa: PLC0415

    # pysisyphus only attaches calculators to GROWN nodes — attach to the
    # endpoints ourselves (else the first force eval is on a calc-less geometry).
    geom_r.set_calculator(calc_getter())
    geom_p.set_calculator(calc_getter())
    gs = GrowingString(images=[geom_r, geom_p], calc_getter=calc_getter,
                       max_nodes=max_nodes)
    opt = _string_optimizer(gs, out_dir=out_dir, max_cycles=max_cycles,
                            thresh=_fmax_to_thresh(fmax))
    opt.run()
    return {"status": "converged" if getattr(opt, "is_converged", False)
            else "not_converged",
            "converged": bool(getattr(opt, "is_converged", False)),
            "images": list(gs.images), "error": None}


def drive_freezing_string(geom_r, geom_p, calc_getter: Callable, *, max_nodes: int,
                          fmax: float, max_cycles: int, out_dir) -> dict:
    """Drive FSM if the installed pysisyphus supports it; else a clean
    ``status="unsupported"`` with an actionable message (no crash)."""
    ok, reason = _probe()["fsm"]
    if not ok:
        log.warning("FSM requested but unsupported: %s", reason)
        return {"status": "unsupported", "converged": False, "images": [],
                "error": f"FSM unsupported on this pysisyphus: {reason}"}
    # Supported path (a future pysisyphus where FreezingString is a COS): drive it
    # exactly like GSM via StringOptimizer.
    from pysisyphus.cos.FreezingString import FreezingString
    from quantum_engine.qm.pysisyphus import _fmax_to_thresh  # noqa: PLC0415
    geom_r.set_calculator(calc_getter())
    geom_p.set_calculator(calc_getter())
    fs = FreezingString(images=[geom_r, geom_p], calc_getter=calc_getter,
                        max_nodes=max_nodes)
    opt = _string_optimizer(fs, out_dir=out_dir, max_cycles=max_cycles,
                            thresh=_fmax_to_thresh(fmax))
    opt.run()
    images = list(getattr(fs, "images", []))
    return {"status": "converged" if getattr(opt, "is_converged", False)
            else "not_converged",
            "converged": bool(getattr(opt, "is_converged", False)),
            "images": images, "error": None}


__all__ = ["capabilities", "drive_growing_string", "drive_freezing_string"]
