"""Base class + result dataclass for the modular optimizer factory."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class OptResult:
    """Outcome of a geometry optimization.

    Designed to be a strict superset of the dict returned by the legacy
    ``quantum_engine.ops.opt.run`` so existing callers can switch
    transparently:

      * ``status`` ∈ {"converged", "not_converged", "error"}
      * ``converged`` is the boolean form of ``status``
      * ``energy_eV`` / ``energy_kcal_mol`` / ``fmax_final`` / ``n_steps``
        match the legacy keys 1:1
      * ``atoms`` is the relaxed ASE ``Atoms`` object (mutated in place)
      * ``backend`` records which optimizer engine ran
      * ``wall_time_s`` is the run() wall-clock seconds (excluding setup)
      * ``outputs`` is a mapping of artifact name → file path
      * ``extra`` is a dict of backend-specific extras (e.g. torch_sim
        SimState, batched stats, etc.) — never load-bearing for callers
    """
    status: str
    converged: bool
    atoms: Any                       # ase.Atoms (delayed import)
    energy_eV: float
    energy_kcal_mol: float
    fmax_final: float
    n_steps: int
    backend: str
    wall_time_s: float = 0.0
    outputs: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, drop_atoms: bool = True) -> dict:
        """Convert to a plain dict (for JSON/YAML)."""
        d = asdict(self)
        if drop_atoms:
            d.pop("atoms", None)
        return d


class Optimizer:
    """Abstract base for geometry-optimization backends.

    Subclasses must implement ``run()``. The constructor stores common
    knobs (``fmax``, ``max_steps``, ``outdir``, etc.) so callers can
    build the optimizer once and reuse it. Backend-specific kwargs are
    passed through ``**kwargs`` and stored on ``self.extra``.
    """

    name: str = "base"  # subclasses override

    def __init__(
        self,
        *,
        fmax: float = 0.05,
        max_steps: int = 500,
        outdir: str | Path | None = None,
        logfile: str | Path | None = None,
        trajectory: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.fmax = float(fmax)
        self.max_steps = int(max_steps)
        self.outdir = Path(outdir) if outdir is not None else None
        self.logfile = Path(logfile) if logfile is not None else None
        self.trajectory = Path(trajectory) if trajectory is not None else None
        self.extra: dict[str, Any] = dict(kwargs)
        if self.outdir is not None:
            self.outdir.mkdir(parents=True, exist_ok=True)

    def run(self, atoms, calculator=None) -> OptResult:  # noqa: D401
        """Run the optimization. Subclasses override."""
        raise NotImplementedError

    # ── small helpers (shared by ASE-style subclasses) ──
    def _attach_calc(self, atoms, calculator):
        if atoms.calc is None:
            if calculator is None:
                raise ValueError(
                    f"{self.name}: atoms has no .calc and no calculator was passed"
                )
            atoms.calc = calculator
        return atoms
