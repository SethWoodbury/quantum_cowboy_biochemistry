"""Sella saddle-point adapter (Hessian-eigenvector following).

Wraps the existing :func:`quantum_engine.mlff.irc.sella_saddle_search` and adds
two robustness features the unified ``saddle`` dispatch needs:

1. ``coords`` knob — explicit ``"cart"``/``"internal"`` selector. The legacy
   helper auto-detects from the free-atom count, which silently picks the
   crashing mode on 200+ atom systems. Callers should pass ``"internal"`` to
   force TRIC coords (recommended on big systems).
2. ``try_eigh_drivers`` — opt-in retry pass that swaps SciPy's ``eigh``
   driver between ``evd``, ``evx``, ``evr`` if the default ``evd`` raises
   :class:`numpy.linalg.LinAlgError`. (Sella calls
   ``scipy.linalg.eigh(A)`` inside its stepper, which uses LAPACK ``DSYEVD``
   by default; a single bad split in the divide-and-conquer kernel sometimes
   trips the LAPACK guard.)

The adapter returns the same ``dict`` shape as the dimer adapter so the
``saddle`` dispatch can swap backends without an ``if/elif`` block.
"""
from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from ase import Atoms

from quantum_engine.mlff.irc import sella_saddle_search
from quantum_engine.units import EV_TO_KCAL

log = logging.getLogger("quantum_engine.qm.sella")

# Order matters: divide-and-conquer first (fastest), expert/range as fall-backs.
# These names map onto the ``driver`` argument of :func:`scipy.linalg.eigh`.
#
# IMPORTANT: ``evd`` is for the GENERALIZED problem ``A x = lambda B x`` and
# rejects standard problems with "does not accept input b array". Sella calls
# ``eigh(A)`` (the standard form), so on a LinAlgError we MUST filter to
# drivers that accept the standard problem; otherwise we just generate a
# different exception that looks identical to the user.
#
# Drivers valid for the standard problem (no `b` array):
#       ``ev`` (legacy syevd via getrf/syevd path),
#       ``evr`` (relatively-robust representations),
#       ``evx`` (expert: select range/index of eigenvalues),
#       ``evd`` (divide-and-conquer; preferred for full eigenvalue sets).
# Drivers that REQUIRE a generalized problem (b is mandatory):
#       ``gv``, ``gvd``, ``gvx``.
#
# scipy.linalg.eigh dispatch maps:
#   standard:    ev / evr / evx / evd
#   generalized: gv / gvd / gvx
DEFAULT_EIGH_DRIVERS: tuple[str, ...] = ("evd", "evr", "evx", "ev")
STANDARD_PROBLEM_DRIVERS: frozenset[str] = frozenset({"ev", "evr", "evx", "evd"})
GENERALIZED_PROBLEM_DRIVERS: frozenset[str] = frozenset({"gv", "gvd", "gvx"})


def filter_drivers_for_problem(
    drivers: Sequence[str], generalized: bool = False,
) -> list[str]:
    """Filter a driver list to only those valid for the given problem type.

    Sella calls ``eigh(A)`` for a STANDARD eigenvalue problem (no B matrix).
    The cascade in :data:`DEFAULT_EIGH_DRIVERS` accidentally included generic
    names; this filter keeps only drivers that accept the chosen problem.

    Args:
        drivers: candidate driver names (case-insensitive accepted).
        generalized: if ``True``, return drivers valid for the generalized
            problem ``A x = lambda B x``; otherwise the standard problem.

    Returns:
        Ordered, deduplicated list of drivers from ``drivers`` that are
        valid. Names normalised to lower-case. Empty list if nothing is
        valid (caller should disable patching in that case).
    """
    valid = GENERALIZED_PROBLEM_DRIVERS if generalized else STANDARD_PROBLEM_DRIVERS
    seen: set[str] = set()
    out: list[str] = []
    for d in drivers:
        if d is None:
            continue
        norm = str(d).strip().lower()
        if norm and norm in valid and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


@contextlib.contextmanager
def _patched_eigh(driver: str) -> Iterator[None]:
    """Monkey-patch ``scipy.linalg.eigh`` to use a specific LAPACK driver.

    Sella calls ``eigh(A)`` without ``driver=`` kwarg, so we swap the global
    default for the duration of the optimisation. Restored on exit.

    Only standard-problem drivers are accepted: passing a generalized-only
    driver (``gv*``) for a Sella eigh call would raise
    "does not accept input b array" — exactly the bug the cascade is meant
    to *avoid*. The driver is validated up front; an invalid name aborts
    immediately with a clear error.
    """
    import scipy.linalg as sla

    if driver not in STANDARD_PROBLEM_DRIVERS:
        raise ValueError(
            f"_patched_eigh: driver={driver!r} is not valid for the standard "
            f"eigenvalue problem (need one of "
            f"{sorted(STANDARD_PROBLEM_DRIVERS)}). Generalized-only drivers "
            f"like 'gv'/'gvd'/'gvx' need a B matrix and will fail Sella's "
            f"eigh(A) call. Use quantum_engine.qm.sella."
            f"filter_drivers_for_problem() to filter your cascade list."
        )

    original = sla.eigh

    def patched(*args, **kwargs):
        kwargs.setdefault("driver", driver)
        return original(*args, **kwargs)

    sla.eigh = patched  # type: ignore[assignment]
    try:
        yield
    finally:
        sla.eigh = original  # type: ignore[assignment]


def run(
    atoms: Atoms,
    calculator=None,
    outdir: str | Path = ".",
    constraint=None,
    fmax: float = 0.02,
    max_steps: int = 500,
    *,
    coords: str = "auto",
    try_eigh_drivers: Sequence[str] | None = None,
    **kwargs,
) -> dict:
    """Sella saddle search with optional internal-coord and eigh-driver retry.

    Args:
        atoms, calculator, outdir, constraint, fmax, max_steps: standard.
        coords: ``"cart"`` (Cartesian), ``"internal"`` (TRIC), or ``"auto"``
                (delegate to :func:`sella_saddle_search`'s heuristic).
        try_eigh_drivers: If set, retry the optimisation with each LAPACK
                driver in turn whenever Sella raises
                :class:`numpy.linalg.LinAlgError`. ``None`` disables the
                retry. Pass :data:`DEFAULT_EIGH_DRIVERS` for a sensible cascade.

    Returns: see module docstring of :mod:`quantum_engine.qm.dimer`.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if atoms.calc is None:
        if calculator is None:
            raise ValueError("sella.run: atoms has no calculator and none was provided")
        atoms.calc = calculator

    use_internal: bool | None
    if coords == "cart":
        use_internal = False
    elif coords == "internal":
        use_internal = True
    elif coords == "auto":
        use_internal = None  # let the helper auto-detect
    else:
        raise ValueError(f"coords must be 'cart', 'internal', or 'auto'; got {coords!r}")

    drivers: tuple[str | None, ...]
    if try_eigh_drivers:
        # Filter out drivers that don't apply to Sella's standard eigh(A) call
        # (generalized-only ``gv*`` drivers would error with "does not accept
        # input b array", masquerading as the very LinAlgError the retry
        # cascade is trying to recover from).
        filtered = filter_drivers_for_problem(
            list(try_eigh_drivers), generalized=False,
        )
        if not filtered:
            log.warning(
                "try_eigh_drivers=%s contains no valid standard-problem "
                "drivers; running once without patching",
                list(try_eigh_drivers),
            )
            drivers = (None,)
        else:
            drivers = tuple(filtered)
    else:
        drivers = (None,)  # do not patch, single attempt

    last_exc: Exception | None = None
    for drv in drivers:
        ctx = _patched_eigh(drv) if drv else contextlib.nullcontext()
        try:
            with ctx:
                ts_atoms, converged = sella_saddle_search(
                    atoms, outdir,
                    constraint=constraint,
                    fmax=fmax, max_steps=max_steps,
                    use_internal=use_internal,
                )
            energy = float(ts_atoms.get_potential_energy())
            backend = "sella-internal" if use_internal else (
                "sella" if use_internal is False else "sella-auto"
            )
            return {
                "status": "converged" if converged else "not_converged",
                "converged": bool(converged),
                "atoms": ts_atoms,
                "energy_eV": energy,
                "energy_kcal_mol": energy * EV_TO_KCAL,
                "backend": backend,
                "outputs": {
                    "saddle_xyz": str(outdir / "saddle.xyz"),
                    "trajectory": str(outdir / "sella-saddle.traj"),
                    "log": str(outdir / "sella-saddle.log"),
                },
            }
        except np.linalg.LinAlgError as exc:
            last_exc = exc
            log.warning(
                "Sella eigh failed with driver=%s: %s — retrying with next driver",
                drv, exc,
            )
            continue
        except Exception as exc:
            # Bubble up non-LinAlg errors immediately; nothing the driver swap can fix.
            log.error("Sella aborted (non-LinAlg): %s", exc)
            raise

    # If we get here, every driver tried raised LinAlgError. Re-raise so the
    # cascade dispatcher (``saddle.run(backend="auto")``) can fall back to
    # the gradient-only dimer.
    assert last_exc is not None
    raise last_exc
