"""ASE-backed optimizer wrappers.

These are the QCB defaults — ``ase.optimize.LBFGS``, ``FIRE``, ``BFGS``
each lifted into the modular ``Optimizer`` interface. Behaviour matches
the existing ``quantum_engine.ops.opt.run`` so swapping in
``make_optimizer("ase-lbfgs", ...)`` is bit-for-bit equivalent.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from quantum_engine.opt.base import Optimizer, OptResult
from quantum_engine.units import EV_TO_KCAL

log = logging.getLogger("quantum_engine.opt.ase")


class _AseOptimizerBase(Optimizer):
    """Common ASE-optimizer wrapper. Subclasses set ``_ase_cls`` and ``name``."""

    _ase_cls = None          # ase.optimize.{LBFGS,FIRE,BFGS}
    _maxstep_kwarg = None    # ASE optimizers accept different maxstep names
    name = "ase-base"

    def run(self, atoms, calculator=None) -> OptResult:
        atoms = self._attach_calc(atoms, calculator)

        # Output paths — fall back to outdir if logfile/trajectory not given
        # (matches legacy ops.opt.run behaviour: opt-{optimizer}.{log,traj}).
        slug = self.name.replace("ase-", "")
        if self.outdir is not None:
            log_path = self.logfile or (self.outdir / f"opt-{slug}.log")
            traj_path = self.trajectory or (self.outdir / f"opt-{slug}.traj")
        else:
            log_path = self.logfile
            traj_path = self.trajectory

        kwargs: dict[str, Any] = {}
        if log_path is not None:
            kwargs["logfile"] = str(log_path)
        if traj_path is not None:
            kwargs["trajectory"] = str(traj_path)

        # Optional maxstep param (ASE LBFGS / FIRE / BFGS all accept it
        # but with different default scales). Pass through if user set it.
        max_step_aa = self.extra.get("max_step_aa") or self.extra.get("max_step")
        if max_step_aa is not None and self._maxstep_kwarg is not None:
            kwargs[self._maxstep_kwarg] = float(max_step_aa)

        log.info(
            "Minimization: %s on %d atoms, fmax=%.4f, max_steps=%d",
            slug.upper(), len(atoms), self.fmax, self.max_steps,
        )

        t0 = time.perf_counter()
        dyn = self._ase_cls(atoms, **kwargs)
        try:
            dyn.run(fmax=self.fmax, steps=self.max_steps)
            err = None
        except Exception as exc:  # noqa: BLE001 - we want the message in the result
            log.exception("ASE optimizer raised: %s", exc)
            err = exc
        wall = time.perf_counter() - t0

        e_final = float(atoms.get_potential_energy())
        f_final = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
        converged = err is None and f_final <= self.fmax * 1.05
        n_steps = int(getattr(dyn, "nsteps", -1))
        status = "converged" if converged else ("error" if err else "not_converged")

        outputs: dict[str, str] = {}
        if log_path is not None: outputs["log"] = str(log_path)
        if traj_path is not None: outputs["trajectory"] = str(traj_path)

        result = OptResult(
            status=status,
            converged=converged,
            atoms=atoms,
            energy_eV=e_final,
            energy_kcal_mol=e_final * EV_TO_KCAL,
            fmax_final=f_final,
            n_steps=n_steps,
            backend=self.name,
            wall_time_s=wall,
            outputs=outputs,
            extra={"error": str(err)} if err else {},
        )
        log.info(
            "  %s: E=%.6f eV, |F|max=%.4f, steps=%d, %.2fs",
            "✓ converged" if converged else "✗ did not converge",
            e_final, f_final, n_steps, wall,
        )
        return result


class AseLbfgsOptimizer(_AseOptimizerBase):
    name = "ase-lbfgs"
    _maxstep_kwarg = "maxstep"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from ase.optimize import LBFGS  # noqa: PLC0415 - lazy import
        self._ase_cls = LBFGS


class AseFireOptimizer(_AseOptimizerBase):
    name = "ase-fire"
    _maxstep_kwarg = "maxstep"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from ase.optimize import FIRE  # noqa: PLC0415 - lazy import
        self._ase_cls = FIRE


class AseBfgsOptimizer(_AseOptimizerBase):
    name = "ase-bfgs"
    _maxstep_kwarg = "maxstep"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from ase.optimize import BFGS  # noqa: PLC0415 - lazy import
        self._ase_cls = BFGS
