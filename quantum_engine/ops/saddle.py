"""Multi-backend saddle-point search with LinAlg-failure auto-cascade.

The user provides a TS guess; we refine it to a true first-order saddle.
Five backends share a uniform call signature so the dispatcher can swap
without touching downstream code:

  ``sella``                 — :mod:`quantum_engine.qm.sella`, Cartesian
                              (Hessian-eigvec following; fast on small
                              systems but can ``LinAlgError`` past ~200 free
                              atoms in pure Cartesian).
  ``sella-internal``        — same code path, internal (TRIC) coords. Robust
                              fall-back when the eigh call blows up in
                              Cartesian.
  ``dimer``                 — :mod:`quantum_engine.qm.dimer`, ASE's improved
                              dimer; gradient-only, no eigendecomposition.
                              The default for the ``refine-ts`` orchestrator
                              because the NEB tangent is a natural seed.
  ``pysisyphus-rsprfo``     — :mod:`quantum_engine.qm.pysisyphus`'s RS-P-RFO.
                              Recommended for small systems where a partial
                              Hessian is affordable and the TS is "well
                              defined" (single dominant imag mode).
  ``pysisyphus-dimer``      — pysisyphus's own dimer + PreconLBFGS. Useful
                              second opinion when ASE's dimer stalls.
  ``auto``                  — try ``sella`` → ``sella-internal`` on
                              :class:`numpy.linalg.LinAlgError` → ``dimer``
                              on second LinAlgError. The failsafe cascade.

All knobs are CLI-exposed; this module hard-codes nothing reaction-specific.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase import Atoms

from quantum_engine.registry import Registry

log = logging.getLogger("quantum_engine.ops.saddle")


# ---------------------------------------------------------------------------
# Saddle-backend registry — the plug-and-play axis. A NEW saddle optimiser drops
# in via one ``register_saddle(name, runner)`` call (no edits to ``_dispatch_one``
# or ``run``). Each runner has the uniform signature
#     runner(atoms, *, calculator, outdir, constraint, fmax, max_steps,
#            initial_mode_vector, eigh_drivers, **extra) -> dict
# returning the standard ``saddle.run`` result dict.
# ---------------------------------------------------------------------------
SADDLE_OPTIMIZERS: Registry = Registry("saddle-optimizer")


def register_saddle(name: str, runner=None, *, aliases: tuple[str, ...] = (),
                    overwrite: bool = False):
    """Register a saddle-optimiser backend (decorator or imperative)."""
    return SADDLE_OPTIMIZERS.register(name, runner, aliases=aliases,
                                      overwrite=overwrite)


def make_saddle_optimizer(backend: str):
    """Return the saddle-backend runner registered under ``backend`` (name/alias).

    ``"auto"`` is handled by :func:`run` (cascade), not here.
    """
    if backend not in SADDLE_OPTIMIZERS:
        raise ValueError(
            f"Unknown saddle backend {backend!r}. "
            f"Choices: {SADDLE_OPTIMIZERS.names() + ['auto']}")
    return SADDLE_OPTIMIZERS.get(backend)


def _run_sella(atoms, *, calculator, outdir, constraint, fmax, max_steps,
               initial_mode_vector, eigh_drivers, **extra) -> dict:
    from quantum_engine.qm import sella as qm_sella
    return qm_sella.run(
        atoms, calculator, outdir=outdir, constraint=constraint,
        fmax=fmax, max_steps=max_steps, coords="cart",
        try_eigh_drivers=eigh_drivers, **extra)


def _run_sella_internal(atoms, *, calculator, outdir, constraint, fmax, max_steps,
                        initial_mode_vector, eigh_drivers, **extra) -> dict:
    from quantum_engine.qm import sella as qm_sella
    return qm_sella.run(
        atoms, calculator, outdir=outdir, constraint=constraint,
        fmax=fmax, max_steps=max_steps, coords="internal",
        try_eigh_drivers=eigh_drivers, **extra)


def _run_dimer(atoms, *, calculator, outdir, constraint, fmax, max_steps,
               initial_mode_vector, eigh_drivers, **extra) -> dict:
    from quantum_engine.qm import dimer as qm_dimer
    return qm_dimer.run(
        atoms, calculator, outdir=outdir, constraint=constraint,
        fmax=fmax, max_steps=max_steps,
        initial_mode_vector=initial_mode_vector, **extra)


def _run_pysis_rsprfo(atoms, *, calculator, outdir, constraint, fmax, max_steps,
                      initial_mode_vector, eigh_drivers, **extra) -> dict:
    from quantum_engine.qm import pysisyphus as qm_pysis
    # spin → multiplicity (2S+1), threaded from the stamped atoms.info; an
    # explicit caller-supplied mult (in **extra) wins.
    extra.setdefault("mult", int(atoms.info.get("spin", 1)))
    return qm_pysis.rsprfo_ts(
        atoms, calculator=calculator, outdir=outdir, fmax=fmax, max_steps=max_steps,
        charge=int(atoms.info.get("charge", 0)), **extra)


def _run_pysis_dimer(atoms, *, calculator, outdir, constraint, fmax, max_steps,
                     initial_mode_vector, eigh_drivers, **extra) -> dict:
    from quantum_engine.qm import pysisyphus as qm_pysis
    extra.setdefault("mult", int(atoms.info.get("spin", 1)))   # spin → 2S+1
    return qm_pysis.dimer_ts(
        atoms, calculator=calculator, outdir=outdir, fmax=fmax, max_steps=max_steps,
        charge=int(atoms.info.get("charge", 0)),
        initial_mode_vector=initial_mode_vector, **extra)


register_saddle("sella", _run_sella)
register_saddle("sella-internal", _run_sella_internal)
register_saddle("dimer", _run_dimer)
register_saddle("pysisyphus-rsprfo", _run_pysis_rsprfo, aliases=("rsprfo", "rs-p-rfo"))
register_saddle("pysisyphus-dimer", _run_pysis_dimer)


# Built-in backend names accepted by ``backend=`` (snapshot for cli choices).
# Runtime validation uses the live registry so dynamically-registered backends
# are accepted too. Keep in sync with cli.py argument choices.
KNOWN_BACKENDS = (
    "sella",
    "sella-internal",
    "dimer",
    "pysisyphus-rsprfo",
    "pysisyphus-dimer",
    "auto",
)

# Order of fall-back when ``backend='auto'``. Each step catches LinAlgError
# from the previous one and tries the next.
AUTO_CASCADE: tuple[str, ...] = ("sella", "sella-internal", "dimer")


def _dispatch_one(
    backend: str,
    atoms: Atoms,
    *,
    calculator,
    outdir: Path,
    constraint,
    fmax: float,
    max_steps: int,
    initial_mode_vector: np.ndarray | None,
    eigh_drivers: Sequence[str] | None,
    extra: dict[str, Any],
) -> dict:
    """Run a single backend once and return its result dict."""
    try:
        runner = SADDLE_OPTIMIZERS.get(backend)
    except KeyError as exc:
        raise ValueError(
            f"Unknown backend {backend!r}; known: "
            f"{SADDLE_OPTIMIZERS.names() + ['auto']}") from exc
    return runner(
        atoms,
        calculator=calculator, outdir=outdir, constraint=constraint,
        fmax=fmax, max_steps=max_steps,
        initial_mode_vector=initial_mode_vector, eigh_drivers=eigh_drivers,
        **extra,
    )


def run(
    atoms: Atoms,
    calculator=None,
    outdir: str | Path = ".",
    constraint=None,
    fmax: float = 0.02,
    max_steps: int = 500,
    *,
    backend: str = "sella",
    initial_mode_vector: np.ndarray | None = None,
    eigh_drivers: Sequence[str] | None = None,
    cascade: Sequence[str] = AUTO_CASCADE,
    **kwargs,
) -> dict:
    """Refine ``atoms`` to a first-order saddle via the chosen backend.

    Args:
        atoms:                  TS guess.
        calculator:             ASE calculator. Optional if ``atoms.calc`` is set.
        outdir:                 Output directory (created if absent).
        constraint:             Optional ASE constraint applied to all atoms.
        fmax, max_steps:        Convergence + iteration cap (eV/Å, count).
        backend:                One of :data:`KNOWN_BACKENDS`.
        initial_mode_vector:    ``(N, 3)`` or flat ``(3N,)`` initial direction
                                for dimer-style backends. Ignored by Sella.
                                The NEB tangent at the climbing image is the
                                canonical choice; load via
                                :func:`quantum_engine.qm.dimer.load_mode_from_xyz`.
        eigh_drivers:           List of LAPACK drivers (``"evd"``, ``"evr"``,
                                ``"evx"``, ``"ev"``) to try in turn when a
                                Sella backend hits :class:`numpy.linalg.LinAlgError`.
                                ``None`` disables retry. Use
                                :data:`quantum_engine.qm.sella.DEFAULT_EIGH_DRIVERS`
                                for the cascade default.
        cascade:                Backends to try in order when
                                ``backend="auto"``. Default exhausts
                                Sella → Sella-internal → Dimer.

    Returns:
        Dict (see :mod:`quantum_engine.qm.dimer` for the canonical shape).
        Adds:
            ``backend_used``        — the backend that actually converged.
            ``backend_attempts``    — chronological log of attempts and
                                      their failure modes (only set in
                                      cascade mode).
    """
    if backend != "auto" and backend not in SADDLE_OPTIMIZERS:
        raise ValueError(
            f"backend={backend!r} not in {SADDLE_OPTIMIZERS.names() + ['auto']}")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Single-backend mode
    if backend != "auto":
        log.info("saddle.run: backend=%s, fmax=%.4f, max_steps=%d",
                 backend, fmax, max_steps)
        result = _dispatch_one(
            backend, atoms,
            calculator=calculator, outdir=outdir, constraint=constraint,
            fmax=fmax, max_steps=max_steps,
            initial_mode_vector=initial_mode_vector,
            eigh_drivers=eigh_drivers,
            extra={k: v for k, v in kwargs.items() if k not in ("op",)},
        )
        result["backend_used"] = result.get("backend", backend)
        return result

    # Auto-cascade mode
    log.info("saddle.run: backend=auto, cascade=%s, fmax=%.4f", cascade, fmax)
    attempts: list[dict[str, Any]] = []
    last_exc: Exception | None = None

    for stage_idx, stage_backend in enumerate(cascade):
        # Each cascade stage gets its own subdirectory so we don't trample logs.
        stage_dir = outdir / f"stage{stage_idx:02d}_{stage_backend.replace('-', '_')}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        atoms_stage = atoms.copy()
        atoms_stage.calc = atoms.calc if atoms.calc is not None else calculator
        atoms_stage.info["charge"] = atoms.info.get("charge", 0)

        log.info("[auto-cascade %d/%d] trying backend=%s in %s",
                 stage_idx + 1, len(cascade), stage_backend, stage_dir)
        try:
            result = _dispatch_one(
                stage_backend, atoms_stage,
                calculator=calculator, outdir=stage_dir, constraint=constraint,
                fmax=fmax, max_steps=max_steps,
                initial_mode_vector=initial_mode_vector,
                eigh_drivers=eigh_drivers,
                extra={k: v for k, v in kwargs.items() if k not in ("op",)},
            )
            attempts.append({
                "backend": stage_backend,
                "stage_dir": str(stage_dir),
                "status": result.get("status"),
                "converged": result.get("converged"),
            })
            if result.get("converged"):
                result["backend_used"] = result.get("backend", stage_backend)
                result["backend_attempts"] = attempts
                log.info("[auto-cascade] success at stage %d (backend=%s)",
                         stage_idx + 1, stage_backend)
                return result
            log.warning("[auto-cascade] backend=%s did not converge "
                        "(status=%s); falling through",
                        stage_backend, result.get("status"))
            last_exc = None
        except np.linalg.LinAlgError as exc:
            log.warning("[auto-cascade] backend=%s raised LinAlgError: %s",
                        stage_backend, exc)
            attempts.append({
                "backend": stage_backend,
                "stage_dir": str(stage_dir),
                "error": "LinAlgError",
                "message": str(exc),
            })
            last_exc = exc
            continue
        except Exception as exc:  # noqa: BLE001
            log.warning("[auto-cascade] backend=%s raised %s: %s — falling through",
                        stage_backend, type(exc).__name__, exc)
            attempts.append({
                "backend": stage_backend,
                "stage_dir": str(stage_dir),
                "error": type(exc).__name__,
                "message": str(exc),
            })
            last_exc = exc
            continue

    # No backend converged: return the last-known atoms with status='not_converged'.
    log.error("[auto-cascade] all backends in %s exhausted without convergence",
              cascade)
    energy = float(atoms.get_potential_energy()) if atoms.calc is not None else 0.0
    return {
        "status": "not_converged",
        "converged": False,
        "atoms": atoms,
        "energy_eV": energy,
        "backend_used": None,
        "backend_attempts": attempts,
        "error": str(last_exc) if last_exc else "all backends failed",
        "outputs": {},
    }
