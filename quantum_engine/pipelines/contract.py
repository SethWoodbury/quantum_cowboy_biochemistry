"""The Step/Pipeline contract.

Three plain-Python primitives, no framework. Together they make the
``quantum_engine.ops`` family composable in Python without the YAML
indirection:

* :class:`Context`  — the mutable shared state passed between Steps:
  the current ``atoms`` and ``calc``, an output directory, and a
  history of every step's :class:`StepResult` keyed by step name.

* :class:`Step` (Protocol) — anything with ``.name: str`` and a
  ``.run(ctx) -> StepResult`` method. Concrete adapters wrapping
  ``quantum_engine.ops.{sp,opt,md,freq,scan,saddle,irc,neb,mtd}.run``
  live in :mod:`quantum_engine.pipelines.steps`.

* :class:`Pipeline` — runs Steps in order, hands each one the same
  Context, swaps in result.atoms when a step opts to advance the
  geometry (Opt, Saddle, IRC), records each step's StepResult, and
  optionally serialises a JSON summary at the end.

Failure semantics: by default a Step exception aborts the pipeline
immediately. A Step can be wrapped with ``Step.optional()`` to
demote its failure to a logged warning (useful for "best-effort"
analyses like a Freq pass that often fails on imaginary-mode hunts).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

log = logging.getLogger("quantum_engine.pipelines")


@dataclass
class StepResult:
    """What a Step returns. The shape mirrors the dict every
    ``quantum_engine.ops.X.run()`` already produces.

    Attributes
    ----------
    name : str
        The Step's ``.name``.
    status : str
        ``"completed" | "skipped" | "failed"``.
    atoms : ase.Atoms | None
        The geometry the step ended on (None if the step doesn't
        advance geometry — e.g. Freq).
    energy_eV : float | None
        Final electronic energy if the step has one.
    outputs : dict[str, str]
        File paths the step wrote (PDB, CIF, npy, etc.). Use
        ``StepResult.outputs["foo"]`` from later steps, not magic
        constants.
    extra : dict[str, Any]
        Anything else the step's existing ``run()`` returned —
        kept verbatim so we don't lose information.
    duration_s : float
        Wall-clock duration. Useful for benchmarking.
    """
    name: str
    status: str = "completed"
    atoms: Any = None             # ase.Atoms — Any-typed to avoid hard ASE dep at type-check time
    energy_eV: float | None = None
    outputs: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0


@dataclass
class Context:
    """Mutable state threaded through a :class:`Pipeline`.

    A pipeline's job is to run ordered Steps that read from / mutate
    ``ctx.atoms`` (and add to ``ctx.history``). Steps that produce a
    new geometry (Opt, Saddle, IRC) should set ``ctx.atoms`` to the
    new geometry; steps that just compute (Sp, Freq) should leave
    ``ctx.atoms`` alone.

    Attributes
    ----------
    atoms : ase.Atoms
        The current geometry. Steps may mutate this via in-place opt
        or Pipeline-driven swap; the contract is "the next step sees
        whatever the previous step left here".
    calc : ase.calculators.Calculator
        ASE calculator. Persisted across steps unless a step explicitly
        replaces it (very rare). Set ``atoms.info["charge"]`` here
        rather than passing charge through every Step.
    outdir : pathlib.Path
        Top-level run directory. Each Step is given a sub-dir
        ``outdir / step.name`` to write into.
    history : dict[str, StepResult]
        Lookup of every completed step's StepResult, keyed by step
        name. Steps reference earlier results by name, e.g.
        ``Saddle(input_step="opt")``.
    metadata : dict[str, Any]
        Free-form bag for cross-step state (e.g. a CV definition the
        first step computed and the second step consumes).
    """
    atoms: Any
    calc: Any = None
    outdir: Path = field(default_factory=lambda: Path("./qcb-run"))
    history: dict[str, StepResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.outdir = Path(self.outdir)


@runtime_checkable
class Step(Protocol):
    """Protocol every pipeline step satisfies.

    Concrete Step classes live in
    :mod:`quantum_engine.pipelines.steps`. Each one wraps an existing
    ``quantum_engine.ops.X.run()`` with parameter validation and
    Context-aware input wiring.

    Implementations should:
      1. Run their underlying op against ``ctx.atoms`` (or an atoms
         object pulled from ``ctx.history[input_step].atoms``).
      2. Advance ``ctx.atoms`` if they produced a new geometry.
      3. Return a :class:`StepResult`.
    """
    name: str

    def run(self, ctx: Context) -> StepResult: ...


# ─────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────

class Pipeline:
    """Run an ordered list of Steps against a Context.

    Args:
        steps: list of :class:`Step` instances. Names must be unique.
        on_step_failure: ``"abort"`` (default), ``"continue"``, or a
            callable ``(step, exc, ctx) -> bool`` returning True to
            keep going. Useful for "best-effort" analyses.
        write_summary: if True, write ``outdir/pipeline_summary.json``
            at the end with status, duration, and outputs of every
            step. Default True.
    """

    def __init__(
        self,
        steps: Iterable[Step],
        *,
        on_step_failure: str | Callable[[Step, Exception, Context], bool] = "abort",
        write_summary: bool = True,
    ) -> None:
        self.steps: list[Step] = list(steps)
        names = [s.name for s in self.steps]
        if len(set(names)) != len(names):
            dupes = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Pipeline has duplicate step names: {dupes}")
        self.on_step_failure = on_step_failure
        self.write_summary = write_summary

    # --------------------------------------------------------------
    def run(self, ctx: Context) -> Context:
        """Execute every step in order; mutate and return ``ctx``.

        Each step gets its own subdirectory under ``ctx.outdir/<name>``;
        Pipeline creates these before calling ``step.run(ctx)``.
        """
        ctx.outdir.mkdir(parents=True, exist_ok=True)
        log.info(f"Pipeline: {len(self.steps)} steps → {ctx.outdir}")

        for i, step in enumerate(self.steps, 1):
            step_dir = ctx.outdir / step.name
            step_dir.mkdir(exist_ok=True)
            log.info(f"  [{i}/{len(self.steps)}] {step.name} → {step_dir}")

            t0 = time.perf_counter()
            try:
                result = step.run(ctx)
            except Exception as e:
                log.exception(f"  step {step.name!r} raised {type(e).__name__}: {e}")
                result = StepResult(name=step.name, status="failed",
                                    extra={"error": f"{type(e).__name__}: {e}"})
                if not self._handle_failure(step, e, ctx):
                    ctx.history[step.name] = result
                    raise
            result.duration_s = time.perf_counter() - t0
            ctx.history[step.name] = result
            log.info(f"     {step.name}: {result.status} ({result.duration_s:.1f}s)")

        if self.write_summary:
            self._write_summary(ctx)
        return ctx

    # --------------------------------------------------------------
    def _handle_failure(self, step: Step, exc: Exception, ctx: Context) -> bool:
        """Return True to swallow the exception and continue."""
        if self.on_step_failure == "abort":
            return False
        if self.on_step_failure == "continue":
            log.warning(f"  step {step.name!r} failed but on_step_failure='continue'; "
                        f"moving on")
            return True
        if callable(self.on_step_failure):
            return bool(self.on_step_failure(step, exc, ctx))
        raise ValueError(
            f"on_step_failure must be 'abort', 'continue', or callable; got "
            f"{self.on_step_failure!r}"
        )

    # --------------------------------------------------------------
    def _write_summary(self, ctx: Context) -> None:
        path = ctx.outdir / "pipeline_summary.json"
        payload = {
            "n_steps": len(self.steps),
            "outdir": str(ctx.outdir),
            "steps": [
                {
                    "name": r.name,
                    "status": r.status,
                    "energy_eV": r.energy_eV,
                    "duration_s": r.duration_s,
                    "outputs": r.outputs,
                }
                for r in ctx.history.values()
            ],
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        log.info(f"Pipeline summary → {path}")
