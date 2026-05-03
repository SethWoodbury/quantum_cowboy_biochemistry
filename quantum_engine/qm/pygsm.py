"""pyGSM adapter (ZimmermanGroup/pyGSM, MIT).

pyGSM is the Python implementation of the Growing String Method for
reaction-path + TS finding. We use it primarily for **single-ended**
path search inside an enzyme active site — given a Michaelis complex
and a set of driving coordinates (extracted from M-CSA Marvin XML
arrow-pushing), GSM grows a path forward and converges on a TS without
needing a product geometry.

Already vendored at ``deps/pyGSM`` — install with
``pip install -e deps/pyGSM`` against the qcb-xtb env.

Two modes:
  * **Single-ended (SE-GSM)** — reactant + driving coordinates → TS.
    The mode of choice for "re-find this M-CSA mechanism inside the
    enzyme" — driving coords come from the mechanism arrow-pushing.
  * **Double-ended (DE-GSM)** — reactant + product → TS. Useful for
    Stage 2 vacuum TS searches when both endpoints are known from
    autodE/Chemoton enumeration.

Status: skeletal. Public signatures locked in; bodies wire to
``pygsm.coordinate_systems`` + ``pygsm.growing_string_methods`` once we
have a working test case.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("quantum_engine.qm.pygsm")


def pygsm_available() -> bool:
    """Is pyGSM importable?"""
    try:
        import pygsm  # noqa: F401
        return True
    except ImportError:
        return False


def _require() -> None:
    if not pygsm_available():
        raise ImportError(
            "pyGSM not installed. Run `pip install -e deps/pyGSM` "
            "against qcb-xtb (vendored submodule, MIT)."
        )


def single_ended_gsm(
    reactant: Any,                   # ASE Atoms
    *,
    driving_coords: list[tuple[str, list[int], float]],
    calc: Any | None = None,         # ASE calc (xtb / mace)
    n_max_nodes: int = 15,
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    """SE-GSM: grow a path from ``reactant`` by stepping along the
    supplied driving coordinates until a TS is found.

    ``driving_coords`` format::
        [("BOND_FORM", [i, j], target_distance),
         ("BOND_BREAK", [k, l], target_distance),
         ("PROTON_TRANSFER", [donor, h, acceptor], None), ...]

    Returns ``{ts_atoms, ts_energy, path_atoms, path_energies, n_iters}``.
    STUB.
    """
    _require()
    raise NotImplementedError(
        "pygsm.single_ended_gsm: wire to "
        "pygsm.growing_string_methods.SE_GSM. Driving-coord format "
        "needs translation from QCB's ('BOND_FORM', [i,j], dist) tuples "
        "to pyGSM's primitive list."
    )


def double_ended_gsm(
    reactant: Any,
    product: Any,
    *,
    calc: Any | None = None,
    n_nodes: int = 11,
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    """DE-GSM between known endpoints. STUB.
    Returns ``{ts_atoms, ts_energy, path_atoms, path_energies}``.
    """
    _require()
    raise NotImplementedError(
        "pygsm.double_ended_gsm: wire to "
        "pygsm.growing_string_methods.DE_GSM."
    )


def driving_coords_from_marvin(marvin_arrows: list[dict[str, Any]],
                                atom_map: dict[int, int]) -> list[tuple[str, list[int], float]]:
    """Translate a parsed Marvin XML mechanism step's arrow list into
    pyGSM driving coordinates. ``atom_map`` maps Marvin atom IDs to ASE
    indices in the user's concrete substrate.

    Returns the driving-coord tuple list ready for :func:`single_ended_gsm`.
    STUB — requires the Marvin arrow parse from
    :func:`quantum_engine.data.mcsa.parse_marvin_xml` to be complete.
    """
    raise NotImplementedError(
        "pygsm.driving_coords_from_marvin: depends on completed Marvin "
        "arrow parser in quantum_engine.data.mcsa.parse_marvin_xml."
    )
