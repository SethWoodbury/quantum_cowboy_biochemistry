"""React-OT generative TS proposer (DeepPrinciple, Nat. Mach. Intell. 2025).

React-OT frames TS search as an optimal-transport problem and **deterministically**
generates a TS structure from a reactant + product in ~0.3 s (10–40 denoiser
evals, vs 40 000 for OA-ReactDiff). Registered as the TS proposer ``react-ot``
(``ops/ts_propose``) — its guess feeds the usual refine → Hessian → IRC gate.

Domain (HARD limit): React-OT's released model supports **H, C, N, O only**
(the one-hot encoder in ``reactot.run_model`` has exactly those four elements) and
was trained on **Transition1x** (gas-phase, neutral, organic). It has **no charge
or spin channel**. It is therefore OUT of domain for metal / charged active sites
(the OPAA di-Zn theozyme): this adapter HARD-FAILS on any non-CHNO element and
WARNS on a non-neutral / non-singlet request. Use it for organic substrate
reactions + the organic benchmarks.

Packaging: React-OT (torch 2.2.1 + torch_geometric + LEFTNet) conflicts with the
main cowboy-qc container, so it lives in its OWN sidecar (see ``deps/reactot_sidecar.def``); the
checkpoint is on Zenodo (open, not gated — record 13131875). This adapter imports
React-OT lazily and raises a clean ImportError when it's absent, so registering it
never affects anything else.

Implementation: this drives React-OT's own single-pair inference path —
``reactot.run_model.parse_rxn_xyzs`` (Kabsch-aligns R/P, seeds the TS at the R↔P
midpoint, builds the object-aware 3-fragment representation) followed by
``make_pred`` (``SBModule.load_from_checkpoint`` → ``eval_sample_batch(...,
return_all=True)``), i.e. exactly the internals of the authors' ``pred_ts``. The
batch construction is reactot's own code (not re-implemented here), so it tracks
the installed package. The forward pass still needs torch + the checkpoint and so
can only be exercised inside the sidecar; any failure there returns a clean
``status="failed"`` (never a silent wrong TS).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ase import Atoms

from quantum_engine.logging_utils import get_logger

log = get_logger("qm.reactot")

# React-OT's released model one-hot encodes exactly these elements (see the
# ``encoder`` in ``reactot.run_model.onehot_convert`` and the ``allowed_atom_types``
# guard in ``check_xyz_files``).
_SUPPORTED_Z = frozenset({1, 6, 7, 8})  # H, C, N, O


def reactot_available() -> tuple[bool, str]:
    try:
        from reactot.run_model import make_pred, parse_rxn_xyzs  # noqa: F401, PLC0415
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
            "checkpoint=, and download the React-OT checkpoint from "
            "https://zenodo.org/records/13131875.")
    if not Path(ckpt).is_file():
        raise FileNotFoundError(
            f"React-OT checkpoint not on disk: {ckpt}. Download the React-OT "
            "checkpoint from https://zenodo.org/records/13131875.")
    return ckpt


def _check_domain(reactant: Atoms, charge: int, spin: int) -> None:
    """Guard React-OT's HARD domain limits before any expensive model load."""
    bad = sorted({int(z) for z in reactant.numbers if int(z) not in _SUPPORTED_Z})
    if bad:
        from ase.data import chemical_symbols  # noqa: PLC0415
        syms = ", ".join(chemical_symbols[z] for z in bad)
        raise ValueError(
            f"React-OT supports only H, C, N, O — got unsupported element(s): "
            f"{syms}. The released model has no embedding for these (it is trained "
            "on Transition1x organics) and cannot be applied to metal / "
            "heteroatom systems such as the di-Zn theozyme. Use a path-search "
            "method (neb/gsm) for this reaction instead.")
    if charge != 0 or spin != 1:
        log.warning(
            "React-OT has no charge/spin channel (trained neutral, singlet); "
            "charge=%d spin=%d will be IGNORED by the model. Treat the guess with "
            "caution — the saddle+Hessian+IRC gate remains the authority.",
            charge, spin)


def _make_opt(checkpoint_path: str, *, nfe: int, solver: str, method: str,
              atol: float, rtol: float, order: int, diz: str,
              normalize: bool) -> SimpleNamespace:
    """The ``opt`` namespace React-OT's ``make_pred`` / ``en_sb.sample`` read.

    Fields consumed: ``checkpoint_path``/``nfe`` (make_pred) and
    ``solver``/``method``/``atol``/``rtol``/``order``/``diz``/``normalize``
    (the diffusion sampler). Defaults follow the authors' ``evaluation.py``.
    """
    return SimpleNamespace(
        checkpoint_path=checkpoint_path, nfe=nfe, solver=solver, method=method,
        atol=atol, rtol=rtol, order=order, diz=diz, normalize=normalize)


def run(reactant: Atoms, product: Atoms, *, charge: int = 0, spin: int = 1,
        atom_map=None, outdir: str | Path = ".", checkpoint: str | None = None,
        device: str | None = None, nfe: int = 100, solver: str = "ode",
        method: str = "midpoint", atol: float = 1e-2, rtol: float = 1e-2,
        order: int = 1, diz: str = "linear", normalize: bool = False,
        **kwargs) -> dict:
    """React-OT proposer. Returns the standard ts_propose result dict.

    ``solver="ode"`` + ``method="midpoint"`` is React-OT's deterministic OT-ODE
    path (its headline ~0.3 s mode). The R/P endpoints must share atom ordering
    (enforced upstream by ``ts_propose.run``'s endpoint check).
    """
    ok, reason = reactot_available()
    if not ok:
        raise ImportError(
            f"TS proposer 'react-ot' requires React-OT: {reason}. It lives in its "
            "own sidecar (deps/reactot_sidecar.def) — torch+torch_geometric+LEFTNet "
            "conflict with the main container.")

    _check_domain(reactant, charge, spin)

    import torch  # noqa: PLC0415
    from ase.io import write as ase_write  # noqa: PLC0415
    from reactot.run_model import make_pred, parse_rxn_xyzs  # noqa: PLC0415

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _resolve_checkpoint(checkpoint)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    log.info("React-OT: checkpoint=%s device=%s solver=%s method=%s nfe=%d",
             ckpt, dev, solver, method, nfe)

    try:
        # React-OT's single-pair path reads R/P from xyz files (it Kabsch-aligns
        # them and seeds the TS at the midpoint internally).
        r_path = outdir / "reactot_r.xyz"
        p_path = outdir / "reactot_p.xyz"
        ase_write(str(r_path), reactant, format="xyz")
        ase_write(str(p_path), product, format="xyz")

        opt = _make_opt(ckpt, nfe=nfe, solver=solver, method=method, atol=atol,
                        rtol=rtol, order=order, diz=diz, normalize=normalize)

        representations, rmsd = parse_rxn_xyzs([str(r_path)], [str(p_path)],
                                               device=dev)
        if rmsd is not None and rmsd > 1.0:
            log.warning("React-OT: R/P Kabsch RMSD=%.2f Å is large — double-check "
                        "the endpoints share atom order/identity.", rmsd)

        # make_pred returns (fragments_nodes, (atomic_numbers, r_pos, ts_pos, p_pos)).
        with torch.no_grad():
            _fragments_nodes, outputs = make_pred(representations, dev, opt)
        ts_pos = outputs[2]
        ts_pos = ts_pos.detach().cpu().numpy().reshape(-1, 3)
        if len(ts_pos) != len(reactant):
            raise ValueError(
                f"React-OT returned {len(ts_pos)} atoms for a {len(reactant)}-atom "
                "reactant — batch/element mismatch.")

        ts = reactant.copy()             # preserve the reactant's atom order
        ts.set_positions(ts_pos)
        ts.info["charge"] = charge
        ts.info["spin"] = spin
        out = outdir / "reactot_ts.xyz"
        ase_write(str(out), ts, format="extxyz")
        log.info("React-OT: generated TS guess (%d atoms) → %s", len(ts), out)
        return {"ts_guess": ts, "confidence": None, "status": "converged",
                "outputs": {"ts_guess_xyz": str(out)}}
    except Exception as exc:  # noqa: BLE001
        log.error("React-OT inference failed (run inside the sidecar with torch + "
                  "the checkpoint; validate against the installed reactot): %s", exc)
        return {"status": "failed", "error": str(exc), "ts_guess": None,
                "confidence": None, "outputs": {}}


__all__ = ["reactot_available", "run"]
