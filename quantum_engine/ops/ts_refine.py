"""TS-guess refiners — a plug-and-play axis for ML *refinement* of a TS guess.

A *refiner* takes an existing transition-state GUESS and returns an improved one:
``(ts_guess) -> better ts_guess``. This is a DIFFERENT operation from a TS
*proposer* (``ops/ts_propose``), which makes a guess from scratch out of a
reactant + product (``(R, P) -> ts_guess``). Keeping them as separate axes keeps
each contract clean and makes the pipeline composable:

    path search / proposer  →  REFINER  →  saddle + Hessian + IRC gate
       (makes a guess)        (polishes it)     (the acceptance authority)

e.g. you can chain a React-OT *proposer* into an AEFM *refiner* into the QM
saddle gate. A refiner is a learned structure prior, NOT an optimizer: it does
not compute a Hessian and does not guarantee a first-order saddle, so the real
``refine_ts`` → partial-Hessian → IRC-like gate still runs afterward and remains
the authority. The refiner only improves the *quality of the guess* fed into it.

Registering one is a one-liner + a small adapter, exactly like the other axes:

    from quantum_engine.ops.ts_refine import register_ts_refiner

    @register_ts_refiner("my-refiner", aliases=("mine",))
    def _my_refiner(ts_guess, *, charge, spin, reactant, product, outdir, **kw):
        better = my_model.refine(ts_guess)                 # an ASE Atoms
        return {"ts_guess": better, "confidence": 0.9, "status": "converged",
                "outputs": {}}

Contract — the runner takes the TS guess (+ optional reactant/product context for
refiners that use it, + run state) and returns
``{"ts_guess": Atoms, "confidence": float|None, "status": str, "outputs": dict}``.
``reactant``/``product`` are optional: most released refiners (AEFM's shipped
checkpoint included) refine the single guess unconditionally and ignore them, but
the signature carries them for future object-aware (R/P-conditioned) checkpoints.
"""
from __future__ import annotations

from pathlib import Path

from ase import Atoms

from quantum_engine.logging_utils import get_logger
from quantum_engine.registry import Registry

log = get_logger("ops.ts_refine")

TS_REFINERS: Registry = Registry("ts-refiner")


def register_ts_refiner(name: str, runner=None, *, aliases: tuple[str, ...] = (),
                        overwrite: bool = False):
    """Register a TS-guess refiner (decorator or imperative)."""
    return TS_REFINERS.register(name, runner, aliases=aliases, overwrite=overwrite)


def make_ts_refiner(name: str):
    """Return the refiner runner registered under ``name`` (name/alias)."""
    if name not in TS_REFINERS:
        raise ValueError(
            f"Unknown TS refiner {name!r}. Choices: {TS_REFINERS.names()}")
    return TS_REFINERS.get(name)


# ---- built-in reference refiner (no extra deps) --------------------------
def _identity(ts_guess: Atoms, *, charge: int, spin: int, reactant=None,
              product=None, outdir, **kwargs) -> dict:
    """A no-op refiner: returns the guess unchanged.

    Unlike the proposer axis (whose ``midpoint`` baseline does real geometric
    work), there is no sound dependency-free "refine toward a TS" baseline — a
    plain minimizer would walk AWAY from the saddle. So this reference refiner is
    deliberately a no-op: it exercises the registry + ``ts_entry`` plumbing and
    documents the contract, while the real refiners (e.g. AEFM) live behind their
    own sidecars. It is safe to leave in a pipeline (it changes nothing).
    """
    ts = ts_guess.copy()
    ts.info["charge"] = charge
    ts.info["spin"] = spin
    ts.info["multiplicity"] = spin
    log.info("identity refiner: returning the guess unchanged (no-op baseline)")
    return {"ts_guess": ts, "confidence": None, "status": "converged",
            "outputs": {}}


register_ts_refiner("identity", _identity, aliases=("noop",))


def _aefm(ts_guess: Atoms, *, charge, spin, reactant=None, product=None, outdir,
          **kwargs) -> dict:
    """AEFM refiner (gated — lives in its own sidecar)."""
    from quantum_engine.qm import aefm  # noqa: PLC0415
    return aefm.run(ts_guess, charge=charge, spin=spin, reactant=reactant,
                    product=product, outdir=outdir, **kwargs)


register_ts_refiner("aefm", _aefm)


def run(method: str, ts_guess: Atoms, *, charge: int = 0, spin: int = 1,
        reactant: Atoms | None = None, product: Atoms | None = None,
        outdir: str | Path = ".", **kwargs) -> dict:
    """Refine a TS ``ts_guess`` via ``method``. Returns the refiner result dict.

    ``reactant``/``product`` are optional context (ignored by refiners that
    refine the guess unconditionally, e.g. AEFM's shipped checkpoint).
    """
    runner = make_ts_refiner(method)
    log.info("ts_refine.run: method=%s, charge=%d, spin=%d, natoms=%d",
             method, charge, spin, len(ts_guess))
    return runner(ts_guess, charge=charge, spin=spin, reactant=reactant,
                  product=product, outdir=Path(outdir), **kwargs)


__all__ = ["TS_REFINERS", "register_ts_refiner", "make_ts_refiner", "run"]
