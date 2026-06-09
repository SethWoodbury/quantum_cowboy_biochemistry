"""AEFM TS-guess refiner (Darouich et al., JCIM 2025; arXiv 2507.16521).

AEFM = Adaptive Equilibrium Flow Matching. It *refines* a low-fidelity TS guess
into a better one via an "Adaptive Prior" + "Equilibrium Flow Matching" (a learned
fixed-point / ODE-solution operator with Anderson acceleration). It is a learned
structure prior, NOT an optimizer: it computes no energy/force/Hessian and does
not guarantee a first-order saddle. Registered as the TS *refiner* ``aefm``
(``ops/ts_refine``) — its output still feeds the usual refine → Hessian → IRC gate,
which remains the acceptance authority. AEFM only improves the guess.

Domain (HARD limit): the released checkpoints are trained on **Transition1x /
GDB7-TS** — gas-phase, neutral, organic **H, C, N, O** reactions. The model has
**no charge or spin channel** and uses a fully-connected (O(N²)) neighbor list.
It is therefore OUT of domain for metal / charged active sites (the di-Zn
theozyme): this adapter HARD-FAILS on any non-CHNO element, WARNS on a
non-neutral / non-singlet request, and WARNS on large systems.

Packaging: AEFM's stack (schnetpack>2 + torch_geometric + torchdiffeq + e3nn,
python 3.12) conflicts with the main cowboy-qc container, so it lives in its OWN sidecar
(see ``deps/aefm_sidecar.def``); the weights are on Zenodo (open, not gated —
record 16414436). This adapter drives AEFM's own supported ``aefm_sample`` CLI via
subprocess (the documented interface), so it stays free of the AEFM/torch stack
and is robust to AEFM-internal refactors. It raises a clean ImportError when the
``aefm_sample`` entry point is absent (i.e. not in the sidecar), so registering it
never affects anything else; any runtime failure returns ``status="failed"``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from ase import Atoms

from quantum_engine.logging_utils import get_logger

log = get_logger("qm.aefm")

# AEFM (Transition1x / GDB7-TS) is trained on these elements only.
_SUPPORTED_Z = frozenset({1, 6, 7, 8})  # H, C, N, O
# Transition1x molecules are small (≲23 heavy+H atoms); AEFM's all-to-all
# neighbor list is O(N²). Warn past a generous ceiling.
_N_WARN = 60


def aefm_available() -> tuple[bool, str]:
    """True iff AEFM's ``aefm_sample`` console script is on PATH (i.e. the sidecar).

    Probes the entry point (not ``import aefm``) so the check never pulls in the
    heavy schnetpack/torch stack on the host.
    """
    exe = shutil.which("aefm_sample")
    if exe:
        return True, f"aefm_sample at {exe}"
    return False, "AEFM not installed: 'aefm_sample' not on PATH"


def _resolve_model(model: str | None) -> str:
    if model:
        return model
    from quantum_engine import site  # noqa: PLC0415
    m = getattr(site, "AEFM_MODEL", None)
    if not m:
        raise FileNotFoundError(
            "AEFM model not configured. Set site.AEFM_MODEL or pass model=, and "
            "download an AEFM checkpoint from https://zenodo.org/records/16414436.")
    if not Path(m).is_file():
        raise FileNotFoundError(
            f"AEFM model not on disk: {m}. Download an AEFM checkpoint (e.g. "
            "aefm_xtb_ci_neb.pt) from https://zenodo.org/records/16414436.")
    return m


def _check_domain(ts_guess: Atoms, charge: int, spin: int,
                  allow_out_of_domain: bool = False) -> None:
    """Guard AEFM's domain limits before launching the model.

    AEFM's LEFTNet backbone uses an ``nn.Embedding(100, ...)`` so it mechanically
    accepts any element (Zn included) WITHOUT crashing — unlike React-OT, whose
    one-hot encoder is literally {H,C,N,O} and KeyErrors on anything else. BUT the
    released checkpoints are trained only on Transition1x / GDB7-TS (CHNO, gas-phase
    organics), so the embedding rows for metals/heteroatoms are UNTRAINED and the
    refinement is out-of-distribution / unvalidated. By default we refuse; pass
    ``allow_out_of_domain=True`` (CLI ``--allow-out-of-domain``) to try it anyway —
    the QM saddle+Hessian+IRC gate remains the acceptance authority regardless.
    """
    bad = sorted({int(z) for z in ts_guess.numbers if int(z) not in _SUPPORTED_Z})
    if bad:
        from ase.data import chemical_symbols  # noqa: PLC0415
        syms = ", ".join(chemical_symbols[z] for z in bad)
        if not allow_out_of_domain:
            raise ValueError(
                f"AEFM is trained on H/C/N/O only — got out-of-domain element(s): "
                f"{syms}. AEFM CAN mechanically run on these (LEFTNet embeds Z<100) "
                "but the weights are UNTRAINED on them, so the refinement is "
                "unvalidated. Pass allow_out_of_domain=True (--allow-out-of-domain) "
                "to experiment anyway (e.g. on the di-Zn theozyme); the QM gate "
                "still decides.")
        log.warning(
            "AEFM: out-of-domain element(s) %s — the weights are UNTRAINED on these "
            "(trained on CHNO organics); refinement is unvalidated. Proceeding "
            "because allow_out_of_domain=True. The QM saddle+Hessian+IRC gate is "
            "the authority.", syms)
    if charge != 0 or spin != 1:
        log.warning(
            "AEFM has no charge/spin channel (trained neutral, singlet); "
            "charge=%d spin=%d will be IGNORED. The saddle+Hessian+IRC gate "
            "remains the authority.", charge, spin)
    if len(ts_guess) > _N_WARN:
        log.warning(
            "AEFM: %d atoms exceeds the Transition1x size range and uses an "
            "O(N²) neighbor list — refinement may be slow / out of distribution.",
            len(ts_guess))


def run(ts_guess: Atoms, *, charge: int = 0, spin: int = 1, reactant=None,
        product=None, outdir: str | Path = ".", model: str | None = None,
        device: str | None = None, overrides: list[str] | None = None,
        allow_out_of_domain: bool = False, timeout_s: int = 1800, **kwargs) -> dict:
    """AEFM refiner. Returns the standard ts_refine result dict.

    Drives ``aefm_sample globals.model=<ckpt> globals.samples_path=<in.xyz>
    aefmsampler.store_path=<out>`` and reads back ``<out>/samples.xyz``.
    ``reactant``/``product`` are accepted for signature parity but ignored — the
    released checkpoint refines the single guess unconditionally.
    """
    ok, reason = aefm_available()
    if not ok:
        raise ImportError(
            f"TS refiner 'aefm' requires AEFM: {reason}. It lives in its own "
            "sidecar (deps/aefm_sidecar.def) — schnetpack+torch_geometric+python3.12 "
            "conflict with the main container.")

    _check_domain(ts_guess, charge, spin, allow_out_of_domain=allow_out_of_domain)

    from ase.io import read as ase_read, write as ase_write  # noqa: PLC0415

    mdl = _resolve_model(model)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    in_xyz = outdir / "aefm_in.xyz"
    store = (outdir / "aefm_out").resolve()      # absolute: hydra chdirs the run
    store.mkdir(parents=True, exist_ok=True)      # AEFMSampler writes here but won't mkdir it

    # AEFM identifies each sample by atoms.info["rxn"]; extxyz preserves it.
    g = ts_guess.copy()
    g.info["rxn"] = 0
    ase_write(str(in_xyz), g, format="extxyz")

    cmd = [
        "aefm_sample",
        f"globals.model={mdl}",
        f"globals.samples_path={in_xyz.resolve()}",
        f"aefmsampler.store_path={store}",
        # save_trajectory isn't in sample.yaml's struct, so hydra needs '+' to add
        # it (it IS an AEFMSampler ctor kwarg). Skips per-sample trajectory dirs.
        "+aefmsampler.save_trajectory=false",
    ]
    if device:
        cmd.append(f"aefmsampler.sampler.device={device}")
    if overrides:
        cmd.extend(overrides)

    log.info("AEFM: model=%s natoms=%d → %s", mdl, len(ts_guess), store)
    # Pin PWD to the run dir: AEFM's sample.yaml resolves hydra.searchpath from
    # ${oc.env:PWD}; cwd= alone does NOT update the child's PWD, so an unset/stale
    # PWD could fail config compose or point the searchpath at the wrong dir.
    env = {**os.environ, "PWD": str(outdir)}
    try:
        proc = subprocess.run(cmd, cwd=str(outdir), capture_output=True,
                              text=True, timeout=timeout_s, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"aefm_sample exited {proc.returncode}. stderr tail:\n"
                + "\n".join(proc.stderr.strip().splitlines()[-15:]))
        out_xyz = store / "samples.xyz"
        if not out_xyz.is_file():
            raise FileNotFoundError(
                f"AEFM produced no output at {out_xyz} (stderr tail:\n"
                + "\n".join(proc.stderr.strip().splitlines()[-10:]) + ")")
        refined_all = ase_read(str(out_xyz), index=":")
        refined = refined_all[-1]                 # single input → last/only frame
        if len(refined) != len(ts_guess):
            raise ValueError(
                f"AEFM returned {len(refined)} atoms for a {len(ts_guess)}-atom "
                "guess — element/order mismatch.")
        refined.info["charge"] = charge
        refined.info["spin"] = spin
        n_steps = refined.info.get("n_steps")
        out = outdir / "aefm_ts.xyz"
        ase_write(str(out), refined, format="extxyz")
        log.info("AEFM: refined guess in %s fixed-point steps → %s", n_steps, out)
        return {"ts_guess": refined, "confidence": None, "status": "converged",
                "outputs": {"ts_guess_xyz": str(out), "n_steps": n_steps,
                            "aefm_samples": str(out_xyz)}}
    except Exception as exc:  # noqa: BLE001
        log.error("AEFM refinement failed (run inside the sidecar; check the "
                  "checkpoint + aefm_sample): %s", exc)
        return {"status": "failed", "error": str(exc), "ts_guess": None,
                "confidence": None, "outputs": {}}


__all__ = ["aefm_available", "run"]
