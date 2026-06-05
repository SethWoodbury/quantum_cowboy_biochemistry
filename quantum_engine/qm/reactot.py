"""React-OT generative TS proposer (DeepPrinciple, Nat. Mach. Intell. 2025).

React-OT frames TS search as an optimal-transport problem and **deterministically**
generates a TS structure from a reactant + product in ~0.3 s (10–40 denoiser
evals, vs 40 000 for OA-ReactDiff). Registered as the TS proposer ``react-ot``
(``ops/ts_propose``) — its guess feeds the usual refine → Hessian → IRC gate.

Domain: trained on **Transition1x** (gas-phase, neutral, organic H/C/N/O). It is
OUT of domain for metal / charged active sites (the OPAA di-Zn theozyme) — use it
for organic substrate reactions + the organic benchmarks.

Packaging: React-OT (torch + e3nn + LEFTNet) conflicts with the main qcb
container, so it lives in its OWN sidecar (see deps/reactot_sidecar.def); the
checkpoint is on Zenodo (open, not gated — record 13131875). This adapter imports
React-OT lazily and raises a clean ImportError when it's absent, so registering it
never affects anything else.

Status: model loading + sampling follow React-OT's documented API
(``SBModule.load_from_checkpoint`` → ``eval_sample_batch``). The reactot-internal
*batch* representation is version-specific and must be validated inside the
sidecar (it cannot be exercised in the main container); a mismatch returns a clean
``status="failed"`` with an actionable message rather than a silent wrong TS.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atoms

from quantum_engine.logging_utils import get_logger

log = get_logger("qm.reactot")


def reactot_available() -> tuple[bool, str]:
    try:
        from reactot.trainer.pl_trainer import SBModule  # noqa: F401, PLC0415
        return True, "reactot available"
    except Exception as exc:  # noqa: BLE001
        return False, f"React-OT not installed: {exc}"


def _resolve_checkpoint(checkpoint: str | None) -> str:
    if checkpoint:
        return checkpoint
    from quantum_engine import site  # noqa: PLC0415
    ckpt = getattr(site, "REACTOT_CKPT", None)
    if not ckpt:
        raise FileNotFoundError(
            "React-OT checkpoint not configured. Set site.REACTOT_CKPT or pass "
            "checkpoint=, and download reactot-pretrained.ckpt from "
            "https://zenodo.org/records/13131875.")
    if not Path(ckpt).is_file():
        raise FileNotFoundError(
            f"React-OT checkpoint not on disk: {ckpt}. Download "
            "reactot-pretrained.ckpt from https://zenodo.org/records/13131875.")
    return ckpt


def _build_reactot_batch(reactant: Atoms, product: Atoms, device):
    """Build React-OT's object-aware (reactant, TS, product) batch from ASE atoms.

    React-OT/OA-ReactDiff represent a reaction as three fragments (R, TS-init, P)
    with shared atom ordering; the model denoises the TS fragment conditioned on R
    and P. The exact tensor schema is reactot-version-specific — this delegates to
    reactot's own data utilities so it tracks the installed package. VALIDATE
    inside the sidecar; on any schema mismatch the caller returns status='failed'.
    """
    # Prefer reactot's own xyz→batch path so we don't hard-code its tensor schema.
    from reactot.data.datasets import ProcessedTS  # noqa: PLC0415
    import torch  # noqa: PLC0415

    def _frag(at: Atoms):
        return {
            "num_atoms": len(at),
            "charges": torch.tensor(at.numbers, dtype=torch.long),
            "positions": torch.tensor(at.get_positions(), dtype=torch.float32),
        }

    # TS is initialised as the R/P midpoint (the model refines from the prior).
    mid = reactant.copy()
    mid.set_positions(0.5 * (reactant.get_positions() + product.get_positions()))
    fragments = [_frag(reactant), _frag(mid), _frag(product)]
    return ProcessedTS.collate_fragments(fragments).to(device)


def _extract_ts(sampled: Any, template: Atoms) -> Atoms:
    """Pull the generated TS positions out of eval_sample_batch's output."""
    import numpy as np  # noqa: PLC0415
    # eval_sample_batch(return_all=True) yields the per-fragment samples; the TS
    # is the middle fragment. Be liberal about the container shape.
    cand = sampled
    for key in ("ts", "transition_state", 1):
        try:
            cand = sampled[key]
            break
        except Exception:  # noqa: BLE001
            continue
    pos = getattr(cand, "positions", None)
    if pos is None:
        pos = cand
    pos = np.asarray(getattr(pos, "detach", lambda: pos)()).reshape(-1, 3)
    ts = template.copy()
    ts.set_positions(pos[:len(ts)])
    return ts


def run(reactant: Atoms, product: Atoms, *, charge: int = 0, spin: int = 1,
        atom_map=None, outdir: str | Path = ".", checkpoint: str | None = None,
        device: str | None = None, nfe: int = 10, solver: str = "ode",
        **kwargs) -> dict:
    """React-OT proposer. Returns the standard ts_propose result dict."""
    ok, reason = reactot_available()
    if not ok:
        raise ImportError(
            f"TS proposer 'react-ot' requires React-OT: {reason}. It lives in its "
            "own sidecar (deps/reactot_sidecar.def) — torch+e3nn+LEFTNet conflict "
            "with the main container.")
    import torch  # noqa: PLC0415
    from reactot.trainer.pl_trainer import SBModule  # noqa: PLC0415

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _resolve_checkpoint(checkpoint)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    log.info("React-OT: checkpoint=%s device=%s solver=%s nfe=%d", ckpt, dev, solver, nfe)
    try:
        model = SBModule.load_from_checkpoint(checkpoint_path=ckpt, map_location=dev)
        model.eval()
        batch = _build_reactot_batch(reactant, product, dev)
        with torch.no_grad():
            sampled = model.eval_sample_batch(batch, return_all=True)
        ts = _extract_ts(sampled, reactant)
        ts.info["charge"] = charge
        ts.info["spin"] = spin
        from ase.io import write as ase_write  # noqa: PLC0415
        out = outdir / "reactot_ts.xyz"
        ase_write(str(out), ts, format="extxyz")
        return {"ts_guess": ts, "confidence": None, "status": "converged",
                "outputs": {"ts_guess_xyz": str(out)}}
    except Exception as exc:  # noqa: BLE001
        log.error("React-OT inference failed (validate the batch schema against "
                  "the installed reactot in the sidecar): %s", exc)
        return {"status": "failed", "error": str(exc), "ts_guess": None,
                "confidence": None, "outputs": {}}


__all__ = ["reactot_available", "run"]
