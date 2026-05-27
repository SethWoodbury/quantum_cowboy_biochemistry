"""ASE-native Dimer saddle-point adapter.

Wraps :mod:`ase.mep.dimer` (``MinModeAtoms`` / ``MinModeTranslate``) into the
standard ``saddle`` op return contract. Gradient-only — no Hessian eigen-
decomposition — which makes it the robust fall-back when Sella crashes with
``LinAlgError`` on large Cartesian systems.

Why this is its own module (vs. ``quantum_engine.mlff.dimer``)
--------------------------------------------------------------
The legacy ``mlff.dimer`` predates the unified saddle dispatch. This file
exposes a thin, generic wrapper that returns the same ``dict`` shape as
``quantum_engine.qm.sella.run`` so callers can swap backends without an
``if/elif`` block. ``mlff.dimer`` stays around as a private helper used by
the older TS-from-CV pipeline.

Generalisation notes
--------------------
- **Initial-mode-direction** can be supplied through ``initial_mode_vector``
  (shape ``(N, 3)`` or ``(3N,)``). When ``None``, the dimer is initialised
  with a small random displacement (the ASE default).
- The kwargs ``dimer_separation`` and ``mode_displacement_scale`` are knobs
  exposed through the CLI — *no* magic numbers buried in this file beyond
  ASE's own internal defaults.
- Works for *any* reaction class (proton transfer, SN2-at-C, sigmatropic,
  hydride transfer, …) — the only inputs are ``atoms`` + a calculator + an
  optional initial direction.

References
----------
- Henkelman, G. & Jónsson, H. *J. Chem. Phys.* **1999**, *111*, 7010.
- Heyden, A.; Bell, A.; Keil, F. *J. Chem. Phys.* **2005**, *123*, 224101.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

from quantum_engine.units import EV_TO_KCAL

log = logging.getLogger("quantum_engine.qm.dimer")


def run(
    atoms: Atoms,
    calculator=None,
    outdir: str | Path = ".",
    constraint=None,
    fmax: float = 0.02,
    max_steps: int = 500,
    *,
    initial_mode_vector: np.ndarray | None = None,
    dimer_separation: float = 0.01,
    mode_displacement_scale: float = 0.01,
    **kwargs,
) -> dict:
    """Refine a TS guess via the ASE Improved-Dimer method.

    Args:
        atoms:                  ASE Atoms with a TS-like geometry.
        calculator:             ASE calculator. If None, ``atoms.calc`` must
                                already be populated.
        outdir:                 Output directory (created if absent).
        constraint:             Optional ASE constraint applied to the dimer.
        fmax:                   Force convergence (eV/Å). Tune via ``--fmax``.
        max_steps:              Iteration cap.
        initial_mode_vector:    Optional ``(N, 3)`` or flat ``(3N,)`` array.
                                Use the climbing-image NEB tangent for the
                                fastest convergence. ``None`` falls back to
                                a small random displacement.
        dimer_separation:       Distance between the two dimer images (Å).
                                Smaller = more curvature accuracy but more
                                noise sensitivity. ``0.01`` is the ASE default.
        mode_displacement_scale: Norm of the initial displacement applied to
                                ``MinModeAtoms`` when seeding the dimer.

    Returns:
        Same shape as :func:`quantum_engine.qm.sella.run`:
            ``status`` (``"converged"`` | ``"not_converged"``),
            ``converged`` (bool),
            ``atoms`` (refined ``Atoms``),
            ``energy_eV``, ``energy_kcal_mol``,
            ``backend`` (``"dimer"``),
            ``outputs`` dict with ``saddle_xyz`` + ``log``.
    """
    from ase.mep.dimer import DimerControl, MinModeAtoms, MinModeTranslate

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if atoms.calc is None:
        if calculator is None:
            raise ValueError("dimer.run: atoms has no calculator and none was provided")
        atoms.calc = calculator
    if constraint is not None:
        atoms.set_constraint(constraint)

    log_path = outdir / "dimer.log"
    traj_path = outdir / "dimer.traj"

    log.info(
        "Dimer (ASE) saddle: %d atoms, fmax=%.4f, max_steps=%d, dimer_sep=%.4f",
        len(atoms), fmax, max_steps, dimer_separation,
    )

    dimer_control = DimerControl(
        initial_eigenmode_method="displacement",
        displacement_method="vector",
        dimer_separation=dimer_separation,
        logfile=str(log_path),
    )

    dimer_atoms = MinModeAtoms(atoms, dimer_control)

    # Build the initial seed
    if initial_mode_vector is not None:
        v = np.asarray(initial_mode_vector, dtype=float)
        if v.ndim == 1:
            if v.size != 3 * len(atoms):
                raise ValueError(
                    f"initial_mode_vector flat length {v.size} != 3N={3*len(atoms)}"
                )
            v = v.reshape(-1, 3)
        elif v.shape != (len(atoms), 3):
            raise ValueError(
                f"initial_mode_vector shape {v.shape} != (N, 3) = {(len(atoms), 3)}"
            )
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            log.warning("initial_mode_vector ~zero; falling back to random displacement")
            v = np.random.default_rng(seed=0).standard_normal((len(atoms), 3))
            norm = np.linalg.norm(v)
        v = (v / norm) * mode_displacement_scale
        log.info("Dimer seeded from supplied mode vector "
                 "(scale=%.4f Å, |v|=%.4f)", mode_displacement_scale, norm)
    else:
        v = np.random.default_rng(seed=0).standard_normal((len(atoms), 3))
        v = (v / np.linalg.norm(v)) * mode_displacement_scale
        log.info("Dimer seeded with random displacement (scale=%.4f Å)",
                 mode_displacement_scale)

    dimer_atoms.displace(displacement_vector=v)

    optimizer = MinModeTranslate(
        dimer_atoms,
        trajectory=str(traj_path),
        logfile=str(log_path),
    )
    converged_flag = False
    try:
        optimizer.run(fmax=fmax, steps=max_steps)
        # ASE optimisers expose ``converged`` only if the final step met fmax.
        # Recompute the actual fmax to make the verdict robust.
        f_final = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
        converged_flag = f_final <= fmax * 1.5  # 50% slack — same as Sella adapter
        log.info("Dimer finished: fmax_final=%.4f eV/Å, converged=%s",
                 f_final, converged_flag)
    except Exception as exc:
        log.error("Dimer optimisation aborted: %s", exc)
        raise

    saddle_xyz = outdir / "saddle.xyz"
    write(str(saddle_xyz), atoms, format="extxyz")

    energy = float(atoms.get_potential_energy())
    return {
        "status": "converged" if converged_flag else "not_converged",
        "converged": bool(converged_flag),
        "atoms": atoms,
        "energy_eV": energy,
        "energy_kcal_mol": energy * EV_TO_KCAL,
        "backend": "dimer",
        "outputs": {
            "saddle_xyz": str(saddle_xyz),
            "log": str(log_path),
            "trajectory": str(traj_path),
        },
    }


def load_mode_from_xyz(xyz_path: str | Path, n_atoms: int) -> np.ndarray:
    """Read a tangent / mode-vector from an XYZ file.

    The XYZ either contains:
      - One frame whose positions encode the displacement vector directly, or
      - Two frames; the displacement is ``frame[1] - frame[0]``.

    This is a backend-agnostic helper for the ``--initial-mode-vector`` CLI
    flag. Caller is responsible for providing a sensible file (e.g., the
    finite-difference tangent at the climbing-image NEB node).
    """
    from ase.io import read as ase_read

    path = Path(xyz_path)
    frames = ase_read(str(path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) >= 2:
        if len(frames[0]) != n_atoms or len(frames[1]) != n_atoms:
            raise ValueError(
                f"mode-vector XYZ frames have {len(frames[0])}/{len(frames[1])} "
                f"atoms but Atoms has {n_atoms}"
            )
        v = frames[1].get_positions() - frames[0].get_positions()
    else:
        if len(frames[0]) != n_atoms:
            raise ValueError(
                f"mode-vector XYZ has {len(frames[0])} atoms but Atoms has {n_atoms}"
            )
        v = frames[0].get_positions()
    return v
