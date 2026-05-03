"""SCINE Chemoton + ReaDuct adapter.

SCINE is the modern, actively-maintained replacement for YARP. Chemoton
explores reaction networks (BSD-3, qcscine/chemoton); ReaDuct does the
single/double-ended path search and TS optimisation (qcscine/readuct).

We use them at two stages of the enzyme TS-search pipelines:

  * **Reaction enumeration** (alternative to autodE): given a set of
    reactant fragments + active-site cluster, Chemoton enumerates
    plausible elementary steps. Useful as a hypothesis generator for
    metallohydrolases where the nucleophile or proton-transfer path
    isn't obvious.
  * **Path search** (alternative to pyGSM): ReaDuct's double-ended
    B-spline interpolation finds TSs between known reactant and product
    states. Single-ended Newton-trajectory search drives via a chosen
    coordinate.

Install:
    bash deps/build_scine.sh   # pip install pinned wheels into qcb-xtb

The vendored deps/scine_chemoton + deps/scine_readuct submodules are
source-of-truth tracking only — actual binaries come from PyPI.

Status: skeletal. Each function has the public signature locked in so
the pipelines can target these methods, but the implementation drops
into the upstream Python API and is left as TODO until we benchmark on
M-CSA 159 (PTE).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("quantum_engine.qm.scine")


def scine_available() -> bool:
    """Are scine_chemoton + scine_readuct importable in this env?"""
    try:
        import scine_chemoton  # noqa: F401
        import scine_readuct  # noqa: F401
        return True
    except ImportError:
        return False


def _require() -> None:
    if not scine_available():
        raise ImportError(
            "SCINE not installed. Run `bash deps/build_scine.sh` "
            "(pip-installs scine-chemoton + scine-readuct into qcb-xtb)."
        )


def enumerate_elementary_steps(
    reactants: list[Any],            # list of ASE Atoms
    *,
    active_atoms: list[int] | None = None,
    backend: str = "sparrow",
    workdir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Chemoton — exhaustively enumerate elementary steps (bond-forming/
    breaking moves) starting from ``reactants``. Restrict to the
    ``active_atoms`` subset to keep the search bounded.

    Returns a list of step dicts, each with reactant/product Atoms +
    proposed driving coordinates. STUB.
    """
    _require()
    raise NotImplementedError(
        "scine.enumerate_elementary_steps: wire to scine_chemoton's "
        "ElementaryStepGasPhase/Aggregate APIs. "
        "Reference: deps/scine_chemoton/scine_chemoton/gears/elementary_steps/."
    )


def double_ended_b_spline(
    reactant: Any,                   # ASE Atoms
    product: Any,                    # ASE Atoms
    *,
    n_images: int = 11,
    backend: str = "xtb",
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    """ReaDuct — double-ended B-spline TS search between known
    endpoints. The B-spline interpolant is the path; the highest
    image's geometry is optimised to a saddle point. STUB.

    Returns ``{ts_atoms, ts_energy, path_atoms, path_energies}``.
    """
    _require()
    raise NotImplementedError(
        "scine.double_ended_b_spline: wire to scine_readuct.run_bspline_task. "
        "Reference: deps/scine_readuct/Readuct/python/scripts/."
    )


def newton_trajectory_drive(
    reactant: Any,
    *,
    driving_coords: list[tuple[str, list[int], float]],
    # e.g. [("bond", [12, 27], 1.5), ("bond", [27, 31], -1.5)] — value
    # is a target delta in Å/rad; positive = increase, negative = decrease.
    backend: str = "xtb",
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    """ReaDuct — single-ended Newton-trajectory scan along the supplied
    driving coordinates. Returns a TS guess at the first maximum found
    plus the full scan path. STUB.
    """
    _require()
    raise NotImplementedError(
        "scine.newton_trajectory_drive: wire to scine_readuct.run_nt_task."
    )


def optimise_ts(
    ts_guess: Any,                   # ASE Atoms near a saddle
    *,
    backend: str = "xtb",
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    """ReaDuct — Berny-style TS optimisation from a guess. STUB."""
    _require()
    raise NotImplementedError(
        "scine.optimise_ts: wire to scine_readuct.run_tsopt_task."
    )


def irc_descend(
    ts_atoms: Any,
    *,
    backend: str = "xtb",
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    """ReaDuct IRC — descend from TS in both directions to confirm
    reactant/product connectivity. Returns the two endpoints + paths."""
    _require()
    raise NotImplementedError(
        "scine.irc_descend: wire to scine_readuct.run_irc_task."
    )
