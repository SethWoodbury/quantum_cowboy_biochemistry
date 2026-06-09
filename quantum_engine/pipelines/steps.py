"""Step adapters wrapping the existing ``quantum_engine.ops`` entries.

Each Step in here is a thin dataclass that:

  1. Carries the ``run()`` parameters (``fmax``, ``n_images``, …) as
     fields.
  2. In ``.run(ctx)``, pulls the ASE Atoms from either ``ctx.atoms``
     or ``ctx.history[input_step].atoms``, calls the underlying op,
     and packages the result into a :class:`StepResult`.

Adding a new Step
-----------------
A Step is anything that satisfies the :class:`~contract.Step`
protocol — you don't have to subclass anything. For one-offs, just
write a small ``@dataclass`` with a ``name`` field and a
``run(ctx) -> StepResult`` method. For the common case of "wrap a
``quantum_engine.ops.X.run`` callable", inherit from :class:`_OpStep`
and set ``op_module``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantum_engine.pipelines.contract import Context, StepResult


# ─────────────────────────────────────────────────────────────────────
# Base helpers
# ─────────────────────────────────────────────────────────────────────

def _resolve_atoms(ctx: Context, input_step: str | None) -> Any:
    """If ``input_step`` is given, return that prior step's atoms;
    otherwise return ``ctx.atoms``."""
    if not input_step:
        return ctx.atoms
    if input_step not in ctx.history:
        raise KeyError(
            f"input_step={input_step!r} not in history. "
            f"Known: {list(ctx.history.keys())}"
        )
    a = ctx.history[input_step].atoms
    if a is None:
        raise ValueError(f"step {input_step!r} did not produce an atoms object")
    return a


def _step_outdir(ctx: Context, name: str) -> Path:
    p = ctx.outdir / name
    p.mkdir(exist_ok=True)
    return p


# ─────────────────────────────────────────────────────────────────────
# Concrete steps. Each wraps quantum_engine.ops.X.run(...).
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Sp:
    """Single-point energy. Doesn't change the geometry."""
    name: str = "sp"

    def run(self, ctx: Context) -> StepResult:
        from quantum_engine.ops import sp
        atoms = _resolve_atoms(ctx, None)
        outdir = _step_outdir(ctx, self.name)
        res = sp.run(atoms, ctx.calc, outdir, constraint=ctx.constraint)
        return StepResult(
            name=self.name, status=res.get("status", "completed"),
            atoms=atoms, energy_eV=res.get("energy_eV"),
            outputs=res.get("outputs", {}),
            extra={k: v for k, v in res.items()
                   if k not in {"status", "atoms", "energy_eV", "outputs"}},
        )


@dataclass
class Opt:
    """Energy minimisation. Advances ``ctx.atoms``."""
    name: str = "opt"
    fmax: float = 0.05
    max_steps: int = 200
    optimizer: str = "BFGS"
    input_step: str | None = None

    def run(self, ctx: Context) -> StepResult:
        from quantum_engine.ops import opt
        atoms = _resolve_atoms(ctx, self.input_step)
        outdir = _step_outdir(ctx, self.name)
        res = opt.run(atoms, ctx.calc, outdir,
                      constraint=ctx.constraint,
                      optimizer=self.optimizer, fmax=self.fmax,
                      max_steps=self.max_steps)
        # Opt advances geometry — flow it forward
        if res.get("atoms") is not None:
            ctx.atoms = res["atoms"]
        return StepResult(
            name=self.name, status=res.get("status", "completed"),
            atoms=res.get("atoms"), energy_eV=res.get("energy_eV"),
            outputs=res.get("outputs", {}),
            extra={k: v for k, v in res.items()
                   if k not in {"status", "atoms", "energy_eV", "outputs"}},
        )


@dataclass
class MD:
    """Molecular dynamics. Advances ``ctx.atoms``."""
    name: str = "md"
    timestep_fs: float = 0.5
    total_time_ps: float = 10.0
    temperature_K: float = 300.0
    friction_per_ps: float = 1.0
    ensemble: str = "nvt"
    seed: int | None = None
    input_step: str | None = None

    def run(self, ctx: Context) -> StepResult:
        from quantum_engine.ops import md
        atoms = _resolve_atoms(ctx, self.input_step)
        outdir = _step_outdir(ctx, self.name)
        res = md.run(atoms, ctx.calc, outdir,
                     constraint=ctx.constraint,
                     ensemble=self.ensemble, timestep_fs=self.timestep_fs,
                     total_time_ps=self.total_time_ps, temperature_K=self.temperature_K,
                     friction_per_ps=self.friction_per_ps, seed=self.seed)
        if res.get("atoms") is not None:
            ctx.atoms = res["atoms"]
        return StepResult(
            name=self.name, status=res.get("status", "completed"),
            atoms=res.get("atoms"), energy_eV=res.get("energy_eV"),
            outputs=res.get("outputs", {}),
            extra={k: v for k, v in res.items()
                   if k not in {"status", "atoms", "energy_eV", "outputs"}},
        )


@dataclass
class Saddle:
    """Sella saddle-point search. Advances ``ctx.atoms`` to the TS."""
    name: str = "saddle"
    fmax: float = 0.01
    max_steps: int = 200
    input_step: str | None = None  # commonly "opt" or "neb"

    def run(self, ctx: Context) -> StepResult:
        from quantum_engine.ops import saddle
        atoms = _resolve_atoms(ctx, self.input_step)
        outdir = _step_outdir(ctx, self.name)
        res = saddle.run(atoms, ctx.calc, outdir,
                         constraint=ctx.constraint,
                         fmax=self.fmax, max_steps=self.max_steps)
        if res.get("atoms") is not None:
            ctx.atoms = res["atoms"]
        return StepResult(
            name=self.name, status=res.get("status", "completed"),
            atoms=res.get("atoms"), energy_eV=res.get("energy_eV"),
            outputs=res.get("outputs", {}),
            extra={k: v for k, v in res.items()
                   if k not in {"status", "atoms", "energy_eV", "outputs"}},
        )


@dataclass
class IRC:
    """IRC descent from a TS. Produces (reactant, product) endpoints."""
    name: str = "irc"
    irc_step: float = 0.1
    irc_fmax: float = 0.05
    saddle_fmax: float = 0.01
    reactant_hint_bonds: list | None = None
    full: bool = True       # full = saddle + descent both sides
    input_step: str | None = "saddle"

    def run(self, ctx: Context) -> StepResult:
        from quantum_engine.ops import irc
        atoms = _resolve_atoms(ctx, self.input_step)
        outdir = _step_outdir(ctx, self.name)
        res = irc.run(atoms, ctx.calc, outdir,
                      constraint=ctx.constraint,
                      irc_step=self.irc_step, irc_fmax=self.irc_fmax,
                      saddle_fmax=self.saddle_fmax,
                      reactant_hint_bonds=self.reactant_hint_bonds, full=self.full)
        return StepResult(
            name=self.name, status=res.get("status", "completed"),
            atoms=res.get("ts"), energy_eV=res.get("e_ts_eV"),
            outputs=res.get("outputs", {}),
            extra={k: v for k, v in res.items()
                   if k not in {"status", "atoms", "outputs"}},
        )


@dataclass
class NEB:
    """Nudged Elastic Band. Needs reactant + product."""
    name: str = "neb"
    n_images: int = 11
    fmax: float = 0.05
    max_steps: int = 200
    climb: bool = True
    interpolation: str = "geodesic"
    reactant_step: str | None = None       # "opt_reactant"
    product_step: str | None = None        # "opt_product"
    reactant: Any = None                   # or pass directly
    product: Any = None

    def run(self, ctx: Context) -> StepResult:
        from quantum_engine.ops import neb
        # Resolve endpoints
        r = self.reactant
        p = self.product
        if r is None and self.reactant_step:
            r = ctx.history[self.reactant_step].atoms
        if p is None and self.product_step:
            p = ctx.history[self.product_step].atoms
        if r is None or p is None:
            raise ValueError(
                f"{self.name}: NEB needs both reactant and product — pass via "
                f"reactant=/product= or reactant_step=/product_step="
            )
        outdir = _step_outdir(ctx, self.name)
        res = neb.run(r, p, ctx.calc, outdir,
                      constraint=ctx.constraint,
                      n_images=self.n_images, fmax=self.fmax,
                      max_steps=self.max_steps, climb=self.climb,
                      interpolation=self.interpolation)
        return StepResult(
            name=self.name, status=res.get("status", "completed"),
            atoms=res.get("ts"), energy_eV=res.get("e_ts_eV"),
            outputs=res.get("outputs", {}),
            extra={k: v for k, v in res.items()
                   if k not in {"status", "atoms", "outputs"}},
        )


@dataclass
class Freq:
    """Vibrational frequency analysis. Doesn't advance geometry."""
    name: str = "freq"
    indices: list[int] | None = None
    delta: float = 0.01
    method: str = "central"
    temperature_K: float = 300.0
    input_step: str | None = "saddle"

    def run(self, ctx: Context) -> StepResult:
        from quantum_engine.ops import freq
        atoms = _resolve_atoms(ctx, self.input_step)
        outdir = _step_outdir(ctx, self.name)
        res = freq.run(atoms, ctx.calc, outdir,
                       constraint=ctx.constraint,
                       indices=self.indices, delta=self.delta,
                       method=self.method, temperature_K=self.temperature_K)
        return StepResult(
            name=self.name, status=res.get("status", "completed"),
            atoms=atoms, energy_eV=None,
            outputs=res.get("outputs", {}),
            extra={k: v for k, v in res.items()
                   if k not in {"status", "atoms", "outputs"}},
        )


@dataclass
class MTD:
    """Well-tempered metadynamics. Advances ctx.atoms to last frame."""
    name: str = "mtd"
    center_idx: int | None = None
    forming_idx: int | None = None
    breaking_idx: int | None = None
    total_time_ps: float = 100.0
    temperature_K: float = 300.0
    variant: str = "wt"
    input_step: str | None = None

    def run(self, ctx: Context) -> StepResult:
        from quantum_engine.ops import mtd
        atoms = _resolve_atoms(ctx, self.input_step)
        outdir = _step_outdir(ctx, self.name)
        res = mtd.run(atoms, calculator=ctx.calc, outdir=outdir,
                      constraint=ctx.constraint,
                      center_idx=self.center_idx, forming_idx=self.forming_idx, breaking_idx=self.breaking_idx,
                      total_time_ps=self.total_time_ps,
                      temperature_K=self.temperature_K, variant=self.variant)
        if res.get("atoms") is not None:
            ctx.atoms = res["atoms"]
        return StepResult(
            name=self.name, status=res.get("status", "completed"),
            atoms=res.get("atoms"), energy_eV=None,
            outputs=res.get("outputs", {}),
            extra={k: v for k, v in res.items()
                   if k not in {"status", "atoms", "outputs"}},
        )


__all__ = ["Sp", "Opt", "MD", "Saddle", "IRC", "NEB", "Freq", "MTD"]
