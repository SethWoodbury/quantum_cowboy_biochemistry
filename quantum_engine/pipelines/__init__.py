"""Composable pipelines: Step + Context + Pipeline.

The contract for chaining quantum_engine ops in Python. Each existing
``quantum_engine.ops.X.run(atoms, calc, outdir, …)`` is wrapped as a
:class:`Step`; a :class:`Pipeline` runs a list of Steps against a shared
:class:`Context` that flows ``atoms`` / ``calc`` / sub-results between
them.

This is the imperative twin of ``qcb run config.yaml`` — same DAG, just
expressed in Python so library users can build their own orchestrators
without YAML.

Quick example
-------------

>>> from quantum_engine.pipelines import Pipeline, Context
>>> from quantum_engine.pipelines.steps import Opt, Saddle, IRC, Freq
>>> ctx = Context(atoms=atoms, calc=calc, outdir=Path("./run"))
>>> result = Pipeline([
...     Opt(fmax=0.05),
...     Saddle(input_step="opt"),
...     IRC(input_step="saddle"),
...     Freq(input_step="saddle"),
... ]).run(ctx)
>>> result.history["saddle"].outputs["ts_pdb"]   # path to TS structure

For full orchestrator pipelines (protein prep → multi-stage TS search →
analysis), see :mod:`enz_qc_pipelines`.
"""
from quantum_engine.pipelines.contract import (
    Context,
    Pipeline,
    Step,
    StepResult,
)

__all__ = [
    "Context",
    "Pipeline",
    "Step",
    "StepResult",
]
