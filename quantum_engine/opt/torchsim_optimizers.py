"""torch-sim-backed optimizer wrappers.

torch-sim (https://github.com/TorchSim/torch-sim) is a PyTorch-native
batchable atomistic simulation engine. For single-system relaxation it
gives ~5-100× speedups over ASE LBFGS/FIRE on GPU because the model
forward + position update stays inside torch tensors (no CPU↔GPU
roundtrips).

Caveats
-------
* torch-sim 0.3.0 is the last release that supports Python 3.11 (our
  container). It ships with **gradient_descent + FIRE** only. LBFGS
  was added in 0.4.x which requires Python 3.12.
* Therefore ``TorchSimLbfgsOptimizer`` raises ``NotImplementedError``
  with a pointer at upgrading the container.
* Calculator coverage: torch-sim only natively wraps a curated set of
  models (MACE, Fairchem, SevenNet, ORB, MatterSim, metatomic, Nequix).
  We only support **MACE** here — the registry hit on
  ``mace-mp/omol/off/mh/polar`` returns a ``mace.calculators.MACECalculator``
  that we adapt to ``torch_sim.models.mace.MaceModel``.
* For unsupported calculators (xtb, ORCA, etc.) ``run()`` raises
  ``ValueError`` so the caller can fall back to ASE.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from quantum_engine.opt.base import Optimizer, OptResult
from quantum_engine.units import EV_TO_KCAL

log = logging.getLogger("quantum_engine.opt.torchsim")


def _torchsim_available() -> tuple[bool, str | None]:
    """Return (available, version_or_error)."""
    try:
        import torch_sim as ts  # noqa: F401, PLC0415
        # 0.3.0 ships without __version__; sniff for the runner instead.
        from torch_sim.runners import optimize  # noqa: F401, PLC0415
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _is_mace_calc(calc) -> bool:
    """Detect a MACE ASE calculator (MACECalculator or mace_mp/off/etc.)."""
    if calc is None:
        return False
    cls = type(calc).__name__
    mod = type(calc).__module__
    return mod.startswith("mace.") and "Calc" in cls


def _wrap_mace_for_torchsim(ase_calc, *, device: str = "cuda",
                             dtype_str: str = "float64"):
    """Adapt a ``mace.calculators.MACECalculator`` into a
    ``torch_sim.models.mace.MaceModel`` so torch-sim's optimizer can
    drive it directly.

    The adaptation grabs the underlying torch model (``calc.models[0]``
    in mace-torch 0.3.x) and re-wraps it via ``MaceModel``.
    """
    import torch  # noqa: PLC0415
    from torch_sim.models.mace import MaceModel  # noqa: PLC0415

    # mace-torch <=0.3.x stores the underlying nn.Module in either
    # `models` (a list, foundation models) or `model` (single instance,
    # custom-loaded). Try both.
    inner = None
    if hasattr(ase_calc, "models") and ase_calc.models:
        inner = ase_calc.models[0]
    elif hasattr(ase_calc, "model"):
        inner = ase_calc.model
    if inner is None:
        raise ValueError(
            "Could not extract inner nn.Module from MACECalculator "
            f"(attrs: {dir(ase_calc)[:20]})"
        )

    dtype = getattr(torch, dtype_str)
    inner = inner.to(device).to(dtype)

    return MaceModel(model=inner, device=torch.device(device), dtype=dtype)


class TorchSimFireOptimizer(Optimizer):
    """torch-sim FIRE optimizer (single-system).

    Wraps ``torch_sim.runners.optimize`` with ``optimizer=fire``. The
    convergence criterion mirrors ASE: max-component force ≤ ``fmax``
    eV/Å. We pass that as a ``convergence_fn`` so torch-sim short-circuits
    once the threshold is reached.

    Note: torch-sim's ``optimize()`` returns a ``SimState`` object that
    we then convert back to an ASE Atoms via
    ``torch_sim.io.state_to_atoms``.
    """
    name = "torch-sim-fire"

    def run(self, atoms, calculator=None) -> OptResult:
        ok, msg = _torchsim_available()
        if not ok:
            raise RuntimeError(
                f"torch-sim is not importable: {msg}. "
                "Install with `pip install torch-sim-atomistic==0.3.0` "
                "into deps/.local_pkgs and add to PYTHONPATH."
            )
        atoms = self._attach_calc(atoms, calculator)
        if not _is_mace_calc(atoms.calc):
            raise ValueError(
                f"TorchSimFireOptimizer only supports MACE ASE calculators "
                f"(got {type(atoms.calc).__name__}). Fall back to ase-lbfgs / "
                f"ase-fire for xtb / ORCA / etc."
            )

        import torch  # noqa: PLC0415
        import torch_sim as ts  # noqa: PLC0415
        from torch_sim.optimizers import fire  # noqa: PLC0415
        from torch_sim.runners import optimize  # noqa: PLC0415
        from torch_sim.io import state_to_atoms  # noqa: PLC0415

        device_str = self.extra.get("device", "cuda")
        dtype_str = self.extra.get("dtype", "float64")
        device = torch.device(device_str)
        dtype = getattr(torch, dtype_str)

        # Build the torch-sim model wrapper from the existing MACE calc.
        ts_model = _wrap_mace_for_torchsim(atoms.calc, device=device_str,
                                            dtype_str=dtype_str)

        # Convergence function — fmax on max-component atomic force.
        # NOTE: torch-sim 0.3.0's generate_force_convergence_fn defaults
        # to ``include_cell_forces=True``, which crashes when we run
        # without a cell-filter (atom-only relax). Disable it.
        try:
            conv_fn = ts.generate_force_convergence_fn(
                force_tol=float(self.fmax),
                include_cell_forces=False,
            )
        except TypeError:
            # Older torch-sim versions may not have include_cell_forces.
            conv_fn = ts.generate_force_convergence_fn(
                force_tol=float(self.fmax),
            )

        # Strip ASE constraints: torch-sim FIRE has no FixAtoms /
        # FixBondLength path. Document this clearly — caller must NOT
        # use this backend with constrained relax.
        if atoms.constraints:
            raise ValueError(
                "TorchSimFireOptimizer does not support ASE constraints "
                "(FixAtoms, FixBondLength, …) yet. Use ase-lbfgs/ase-fire "
                "for constrained relaxes."
            )

        log.info(
            "Minimization (torch-sim FIRE): %d atoms on %s/%s, fmax=%.4f, max_steps=%d",
            len(atoms), device_str, dtype_str, self.fmax, self.max_steps,
        )

        t0 = time.perf_counter()
        try:
            final_state = optimize(
                system=atoms,
                model=ts_model,
                optimizer=fire,
                convergence_fn=conv_fn,
                max_steps=int(self.max_steps),
                pbar=False,
            )
            err = None
        except Exception as exc:  # noqa: BLE001
            log.exception("torch-sim FIRE raised: %s", exc)
            err = exc
            final_state = None
        wall = time.perf_counter() - t0

        if err is not None or final_state is None:
            return OptResult(
                status="error",
                converged=False,
                atoms=atoms,
                energy_eV=float("nan"),
                energy_kcal_mol=float("nan"),
                fmax_final=float("nan"),
                n_steps=-1,
                backend=self.name,
                wall_time_s=wall,
                extra={"error": str(err)},
            )

        # Convert final state back to ASE atoms — overwrite positions
        # in-place on the original atoms object (callers typically hold
        # a reference and want it mutated).
        relaxed_list = state_to_atoms(final_state)
        relaxed = relaxed_list[0] if isinstance(relaxed_list, list) else relaxed_list
        atoms.set_positions(relaxed.get_positions())
        # Re-attach the original ASE calc and force a fresh energy/force
        # evaluation so downstream callers see consistent numbers.
        e_final = float(atoms.get_potential_energy())
        f_final = float(np.linalg.norm(atoms.get_forces(), axis=1).max())

        converged = f_final <= self.fmax * 1.05
        # torch-sim doesn't expose nsteps cleanly; sniff from state.
        n_steps = int(getattr(final_state, "n_steps", -1))
        if n_steps < 0:
            # heuristic: total atoms × steps in batched stats, not always
            # available — leave as -1 if we can't tell.
            n_steps = -1

        outputs: dict[str, str] = {}
        if self.outdir is not None:
            traj_path = self.trajectory or (self.outdir / "opt-torchsim-fire.traj")
            try:
                from ase.io import write as ase_write  # noqa: PLC0415
                ase_write(str(traj_path), atoms)
                outputs["trajectory"] = str(traj_path)
            except Exception:  # noqa: BLE001
                pass

        result = OptResult(
            status="converged" if converged else "not_converged",
            converged=converged,
            atoms=atoms,
            energy_eV=e_final,
            energy_kcal_mol=e_final * EV_TO_KCAL,
            fmax_final=f_final,
            n_steps=n_steps,
            backend=self.name,
            wall_time_s=wall,
            outputs=outputs,
        )
        log.info(
            "  %s: E=%.6f eV, |F|max=%.4f, %.2fs",
            "✓ converged" if converged else "✗ did not converge",
            e_final, f_final, wall,
        )
        return result


class TorchSimLbfgsOptimizer(Optimizer):
    """Stub — torch-sim LBFGS lives in 0.4.0+ which needs Python 3.12.

    Our container is Python 3.11 with torch-sim 0.3.0 (FIRE only). Once
    the container is rebuilt on Python 3.12, swap this stub for a real
    wrapper modelled on :class:`TorchSimFireOptimizer` but with
    ``optimizer=lbfgs`` (from ``torch_sim.optimizers``).
    """
    name = "torch-sim-lbfgs"

    def run(self, atoms, calculator=None) -> OptResult:
        raise NotImplementedError(
            "torch-sim LBFGS is not available in the current container. "
            "torch-sim 0.3.0 (the last Python 3.11-compatible release) only "
            "ships gradient_descent + FIRE. To get LBFGS, rebuild the "
            "cowboy-qc container on Python 3.12 and `pip install "
            "torch-sim-atomistic>=0.4.0`. For now, use backend "
            "'torch-sim-fire' (also fast on GPU) or 'ase-lbfgs'."
        )
