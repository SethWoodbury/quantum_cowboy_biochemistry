"""QM packages as ASE *calculators* (the per-step / mode-A route).

Complements the QM-native *engine* gateway (``qm/engine.py``): sometimes you want
a DFT package driven by OUR optimizer/saddle/path factories — for uniformity, for
small systems, for constraints the native optimizer doesn't expose, or for method
experiments. ``make_qm_calc`` returns an ``ase.calculators.orca.ORCA`` calculator
so any of our factories can drive ORCA exactly like an MLFF.

ORCA is NOT in the qcb container; the calculator object constructs fine anywhere,
but *evaluating* it needs ORCA on PATH (a host/node). For routine DFT TS work
prefer the native engine (``engine='orca'``) — it's far more efficient than
per-step ASE driving.
"""
from __future__ import annotations

from quantum_engine.logging_utils import get_logger

log = get_logger("qm.calc")


def make_qm_calc(
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    *,
    charge: int = 0,
    mult: int = 1,
    nproc: int = 8,
    maxcore: int = 4000,
    orca_bin: str | None = None,
    solvent: str | None = None,
    extra_keywords: str = "",
    task: str = "engrad",
):
    """Build an ASE ORCA calculator (per-step SCF+gradient for our optimizers).

    Args:
        method/basis: DFT functional + basis (e.g. ``"B3LYP"``, ``"def2-SVP"``).
        charge/mult: net charge + spin multiplicity (2S+1).
        nproc/maxcore: ORCA parallelism + memory-per-core (MB).
        orca_bin: ORCA executable; defaults to ``site.ORCA_BIN``.
        solvent: CPCM solvent name, or None.
        extra_keywords: appended to the ORCA ``!`` line.
        task: ORCA simple-input for an ASE force call — ``"engrad"`` (default).

    Returns:
        An ``ase.calculators.orca.ORCA`` instance (charge/mult baked in).
    """
    if orca_bin is None:
        from quantum_engine import site  # noqa: PLC0415
        orca_bin = getattr(site, "ORCA_BIN", "orca")

    simple = " ".join(x for x in [method, basis, task,
                                  f"CPCM({solvent})" if solvent else "",
                                  extra_keywords.strip()] if x)
    blocks = f"%pal nprocs {nproc} end\n%maxcore {maxcore}"
    log.info("ORCA ASE calc: '! %s'  (charge=%d mult=%d)", simple, charge, mult)

    # ASE >= 3.23 uses OrcaProfile + orcasimpleinput/orcablocks; older ASE uses
    # ORCA(label=, orcasimpleinput=, ...). Support both.
    from ase.calculators.orca import ORCA  # noqa: PLC0415
    try:
        from ase.calculators.orca import OrcaProfile  # noqa: PLC0415
        profile = OrcaProfile(command=str(orca_bin))
        return ORCA(profile=profile, charge=charge, mult=mult,
                    orcasimpleinput=simple, orcablocks=blocks)
    except Exception:  # noqa: BLE001 — older ASE API
        return ORCA(label="orca", charge=charge, mult=mult,
                    orcasimpleinput=simple, orcablocks=blocks)


__all__ = ["make_qm_calc"]
