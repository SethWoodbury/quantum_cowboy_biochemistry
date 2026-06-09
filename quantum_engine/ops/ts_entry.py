"""``ts_entry`` — the reaction-agnostic TS orchestrator (canonical core).

Three entry points funnel into ONE validated-TS pipeline, driven by a
``ReactionSpec`` (what reacts) + a ``RunContext`` (how to compute it). Nothing
here is specific to any reaction — every atom index comes from the resolved spec.

    entry = "ts-guess"          guess → saddle-refine → (validate)
    entry = "reactant-product"  R,P → path search → peak → saddle-refine → (validate)
    entry = "reactant-only"     R + CV → drive to a product basin → reactant-product

Canonical core (the plan): clean basins → pluggable path search → refine the peak
to a first-order saddle → active-region Hessian (one imaginary mode below a cutoff
with >= overlap on the reactive atoms) → IRC-like return to R/P. ``--rigor-mode``
(draft / standard / publication) scales images / saddle backend / thresholds /
whether to run the IRC-like validation. Path/saddle methods only PROPOSE a TS;
the saddle+Hessian+overlap+IRC-like gate is the acceptance authority.

Every stage emits a human log + a machine ``<stage>.json`` (via ``logging_utils``)
and contributes Gates to one aggregated ``GateReport`` (``gates.json``). A FAIL
with no fallback aborts; a FAIL with a fallback is recorded and the run continues.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from quantum_engine.logging_utils import Step, configure_logging, get_logger
from quantum_engine.ops.gates import Gate, GateReport
from quantum_engine.reaction_spec import ReactionSpec, ResolvedReaction, RunContext
from quantum_engine.units import EV_TO_KCAL

log = get_logger("ops.ts_entry")

ENTRIES = ("ts-guess", "reactant-product", "reactant-only")


@dataclass(frozen=True)
class RigorPreset:
    """Rigor tier — scales cost vs. confidence across the pipeline."""
    label: str
    path_method: str
    n_images: int
    saddle_backend: str
    saddle_fmax: float
    validate: bool                 # run the IRC-like imag-mode validation?
    imag_cm_cutoff: float          # imag freq must be MORE negative than this
    imag_overlap_min: float        # frac of imag-mode amplitude on reactive atoms


RIGOR: dict[str, RigorPreset] = {
    "draft": RigorPreset("draft", "neb", 7, "dimer", 0.05, False, -30.0, 0.30),
    "standard": RigorPreset("standard", "neb", 11, "auto", 0.03, True, -50.0, 0.50),
    "publication": RigorPreset("publication", "neb", 17, "auto", 0.02, True, -50.0, 0.50),
}


def _calc_fn(ctx: RunContext):
    """Per-image ASE-calculator factory from the RunContext (MLFF/xTB path)."""
    if ctx.engine:
        raise NotImplementedError(
            f"engine={ctx.engine!r}: the QM-native engine gateway is Phase 6. "
            "Use engine=None (an ASE calculator: MLFF or xTB) for now.")
    from quantum_engine.calc import make_calc_fn  # noqa: PLC0415
    return make_calc_fn(model=ctx.model, head=ctx.head, device=ctx.device,
                        charge=ctx.charge, spin=ctx.multiplicity)


def _stamp(atoms: Atoms, ctx: RunContext) -> Atoms:
    atoms.info["charge"] = ctx.charge
    atoms.info["spin"] = ctx.multiplicity
    atoms.info["multiplicity"] = ctx.multiplicity
    return atoms


def _path_tangent(images, ts_idx: int):
    """Unit tangent at the climbing image (path-method agnostic), or None."""
    if images is None or not (0 < ts_idx < len(images) - 1):
        return None
    d = images[ts_idx + 1].get_positions() - images[ts_idx - 1].get_positions()
    n = float(np.linalg.norm(d))
    return (d / n) if n > 1e-9 else None


def _relax_basin(atoms, ctx, outdir, label):
    """Clean a basin (R or P) to a minimum; returns (atoms, energy_eV)."""
    from quantum_engine.ops import opt  # noqa: PLC0415
    with Step(f"relax_{label}", outdir, params={"label": label}) as s:
        res = opt.run(_stamp(atoms.copy(), ctx), _calc_fn(ctx)(),
                      outdir=outdir / f"relax_{label}", fmax=0.05, max_steps=500)
        s.record(converged=res["converged"], energy_eV=res["energy_eV"])
    return res["atoms"], float(res["energy_eV"])


def _refine(ts_guess, ctx, resolved, outdir, preset, saddle_backend,
            initial_mode_vector, template) -> dict:
    """Saddle-refine + active-region Hessian validation of a TS guess."""
    from quantum_engine.ops import refine_ts  # noqa: PLC0415
    from quantum_engine.ops.refine_ts import RefineTSCriteria  # noqa: PLC0415
    crit = RefineTSCriteria(n_imag_expected=1, imag_cm_cutoff=preset.imag_cm_cutoff,
                            n_imag_cutoff=preset.imag_cm_cutoff,
                            imag_mode_overlap=preset.imag_overlap_min)
    # resolved.reactive are 0-based ASE indices -> pass unambiguously as "0:idx"
    reactive = [f"0:{i}" for i in resolved.reactive] or None
    if reactive is None:
        raise ValueError("ReactionSpec has no reactive_atoms — required for the "
                         "imaginary-mode overlap gate.")
    return refine_ts.run(
        _stamp(ts_guess.copy(), ctx), _calc_fn(ctx)(),
        outdir=outdir, reactive_atoms=reactive, bt_template=template,
        backend=saddle_backend, initial_mode_vector=initial_mode_vector,
        saddle_fmax=preset.saddle_fmax, criteria=crit, pdb_charge_hint=ctx.charge)


def _saddle_gates(refine_res: dict, preset: RigorPreset) -> list[Gate]:
    n_imag = refine_res.get("n_imag")
    imag = refine_res.get("imag_freq_cm")
    rep = refine_res.get("report")
    overlap = getattr(rep, "imag_mode_overlap", None)
    gates = [
        Gate("saddle_converged",
             "PASS" if refine_res.get("overall_pass") else "FAIL",
             detail="refine_ts overall_pass", value=refine_res.get("overall_pass")),
        Gate("n_imag",
             "PASS" if n_imag == 1 else "FAIL",
             detail="exactly one imaginary mode expected",
             value=n_imag, threshold=1),
        Gate("imag_freq_cm",
             "PASS" if (imag is not None and imag < preset.imag_cm_cutoff) else "FAIL",
             detail=f"imag freq must be < {preset.imag_cm_cutoff} cm^-1",
             value=imag, threshold=preset.imag_cm_cutoff),
    ]
    if overlap is not None:
        gates.append(Gate(
            "imag_mode_overlap",
            "PASS" if overlap >= preset.imag_overlap_min else "FAIL",
            detail=f"reactive-atom overlap >= {preset.imag_overlap_min}",
            value=round(float(overlap), 3), threshold=preset.imag_overlap_min))
    return gates


def run(
    reaction: ReactionSpec | ResolvedReaction,
    ctx: RunContext,
    *,
    entry: str,
    outdir: str | Path,
    reactant: Atoms | None = None,
    product: Atoms | None = None,
    ts_guess: Atoms | None = None,
    rigor: str = "standard",
    path_method: str | None = None,
    proposer: str | None = None,
    refiner: str | None = None,
    saddle_backend: str | None = None,
    n_images: int | None = None,
    validate: bool | None = None,
    template: Any = None,
    relax_endpoints: bool = True,
    monitor: bool = True,
    cv_product_s: float | None = None,
    **kwargs,
) -> dict:
    """Run one TS entry point through the canonical core. See module docstring.

    Args:
        reaction: a ``ReactionSpec`` (resolved here) or an already-``ResolvedReaction``.
        ctx: ``RunContext`` (charge/spin/model/head/device/engine).
        entry: one of ``"ts-guess"`` / ``"reactant-product"`` / ``"reactant-only"``.
        outdir: run directory (created).
        reactant/product/ts_guess: input geometries for the chosen entry.
        rigor: ``"draft"`` / ``"standard"`` / ``"publication"`` (preset).
        path_method/saddle_backend/n_images/validate: per-run overrides of the preset.
        proposer: (reactant-product only) a TS-guess proposer name (e.g. a
            generative model registered via ``register_ts_proposer``, or the
            built-in ``"midpoint"``). When set, it replaces path search — the
            proposer suggests the TS guess directly, which is then refined +
            gated exactly as usual.
        refiner: an optional TS-guess *refiner* name (e.g. ``"aefm"``, registered
            via ``register_ts_refiner``). When set, the guess (from path search,
            a proposer, or a ``ts-guess`` entry) is polished by the refiner BEFORE
            saddle-refine. This is a non-critical stage: if the refiner is
            unavailable or fails, the pipeline falls back to the un-refined guess
            (a WARN gate) and continues — it can never break the run. Compose with
            ``proposer`` for the React-OT → AEFM chain.
        template: optional biotite template (for ``RES:ID:NAME`` atom tokens).
        relax_endpoints: relax R/P to clean basins before pathing (recommended).
        monitor: attach a non-constraining bond/metal report to R/TS/P.
        cv_product_s: for ``reactant-only`` — the product-side CV target ``s``.

    Returns:
        Unified dict: ``status`` ("converged"/"failed"), ``ts``/``reactant``/
        ``product`` Atoms, energies + barriers (kcal/mol), ``imag_freq_cm``,
        ``n_imag``, ``gates`` (the aggregated GateReport.to-dict-ish), and
        ``outputs`` (paths incl. ``gates.json`` / ``ts_entry.json``).
    """
    configure_logging()
    if entry not in ENTRIES:
        raise ValueError(f"entry={entry!r} not in {ENTRIES}")
    if rigor not in RIGOR:
        raise ValueError(f"rigor={rigor!r} not in {sorted(RIGOR)}")

    # QM-native engine gateway: route the whole TS step to the package's own
    # optimizer (e.g. ORCA NEB-TS / OptTS) instead of the ASE-calculator path.
    if ctx.engine:
        configure_logging()
        from quantum_engine.qm.engine import make_engine  # noqa: PLC0415
        log.info("ts_entry: routing to QM-native engine=%s (entry=%s)",
                 ctx.engine, entry)
        engine_kw = dict(kwargs)
        if n_images is not None:            # forward an explicit n_images (M4)
            engine_kw.setdefault("n_images", n_images)
        runner = make_engine(ctx.engine)
        return runner(reaction, ctx, entry=entry, outdir=Path(outdir),
                      reactant=reactant, product=product, ts_guess=ts_guess,
                      **engine_kw)
    preset = RIGOR[rigor]
    path_method = path_method or preset.path_method
    saddle_backend = saddle_backend or preset.saddle_backend
    n_images = n_images or preset.n_images
    do_validate = preset.validate if validate is None else validate

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    report = GateReport(step="ts_entry")

    # ---- resolve the reaction to 0-based indices ----
    anchor = ts_guess if ts_guess is not None else reactant
    if isinstance(reaction, ResolvedReaction):
        resolved = reaction
    else:
        reaction.validate()
        resolved = reaction.resolve(anchor, template)
    log.info("ts_entry: entry=%s rigor=%s path=%s saddle=%s model=%s charge=%d spin=%d",
             entry, rigor, path_method, saddle_backend, ctx.model, ctx.charge, ctx.multiplicity)

    R = R_e = P = P_e = None
    ts_atoms = None
    refine_res = None

    with Step("ts_entry", outdir, params={
            "entry": entry, "rigor": rigor, "path_method": path_method,
            "saddle_backend": saddle_backend, "n_images": n_images,
            "model": ctx.model, "charge": ctx.charge, "spin": ctx.multiplicity}) as top:

        # ---- reactant-only: drive R to a product basin via the CV ----
        if entry == "reactant-only":
            if resolved.cv_bond_difference is None:
                raise ValueError(
                    "entry='reactant-only' needs a CV bond-difference in the "
                    "ReactionSpec (cv: {kind: bond_difference, atoms: [...]}) to "
                    "generate the product endpoint — no silent fake endpoints.")
            if reactant is None:
                raise ValueError("entry='reactant-only' needs reactant=")
            product = _drive_to_product(reactant, ctx, resolved, outdir,
                                        cv_product_s, report)
            entry = "reactant-product"   # fall through

        # ---- clean basins ----
        if entry == "reactant-product":
            if reactant is None or product is None:
                raise ValueError("entry='reactant-product' needs reactant= and product=")
            if relax_endpoints:
                R, R_e = _relax_basin(reactant, ctx, outdir, "reactant")
                P, P_e = _relax_basin(product, ctx, outdir, "product")
            else:
                R, P = reactant.copy(), product.copy()

            amap = getattr(resolved.spec, "atom_map", None)
            if proposer:
                # ---- generative/heuristic TS proposer → TS guess (skip path) ----
                from quantum_engine.ops import ts_propose  # noqa: PLC0415
                with Step("ts_propose", outdir, params={"proposer": proposer}) as s:
                    prop = ts_propose.run(proposer, _stamp(R.copy(), ctx),
                                          _stamp(P.copy(), ctx), charge=ctx.charge,
                                          spin=ctx.multiplicity, atom_map=amap,
                                          outdir=outdir / "propose", **kwargs)
                    s.record(status=prop.get("status"),
                             confidence=prop.get("confidence"))
                ok = prop.get("status") in ("converged", "completed")
                report.add(Gate("ts_propose", "PASS" if ok else "WARN",
                                detail=str(prop.get("status")), value=prop.get("confidence"),
                                fallback="saddle-refine still runs on the proposed guess"))
                ts_guess = prop.get("ts_guess")
                tangent = None
            else:
                # ---- path search → TS guess ----
                from quantum_engine.ops import path_search  # noqa: PLC0415
                with Step("path_search", outdir,
                          params={"method": path_method, "n_images": n_images}) as s:
                    pth = path_search.run(path_method, _stamp(R.copy(), ctx),
                                          _stamp(P.copy(), ctx), _calc_fn(ctx),
                                          outdir=outdir / "path", charge=ctx.charge,
                                          n_images=n_images, atom_map=amap)
                    s.record(status=pth.get("status"),
                             barrier_fwd_kcal=pth.get("barrier_fwd_kcal"))
                pstatus = "PASS" if pth.get("status") in ("converged", "completed", "cached") else "WARN"
                report.add(Gate("path_search", pstatus, detail=str(pth.get("status")),
                                fallback="saddle-refine still runs on the peak guess"))
                ts_guess = pth.get("ts")
                tangent = _path_tangent(pth.get("images"), int(pth.get("ts_idx", -1)))

        elif entry == "ts-guess":
            if ts_guess is None:
                raise ValueError("entry='ts-guess' needs ts_guess=")
            tangent = None
        else:  # pragma: no cover - guarded above
            raise ValueError(entry)

        # ---- optional ML guess refinement (e.g. AEFM) before saddle-refine ----
        # A refiner polishes the guess; it is NON-critical — any failure (incl.
        # the sidecar being absent) falls back to the un-refined guess so it can
        # never break the run. The QM saddle+Hessian+IRC gate below is unchanged.
        if refiner and ts_guess is not None:
            from quantum_engine.ops import ts_refine  # noqa: PLC0415
            with Step("ts_refine", outdir, params={"refiner": refiner}) as s:
                try:
                    ref = ts_refine.run(refiner, _stamp(ts_guess.copy(), ctx),
                                        charge=ctx.charge, spin=ctx.multiplicity,
                                        reactant=R, product=P,
                                        outdir=outdir / "refine_guess", **kwargs)
                except Exception as exc:  # noqa: BLE001 — refiner is non-critical
                    log.warning("ts_refine(%s) errored (%s) — using the un-refined "
                                "guess", refiner, exc)
                    ref = {"status": "failed", "error": str(exc),
                           "ts_guess": None, "confidence": None}
                s.record(status=ref.get("status"), confidence=ref.get("confidence"))
            ok = ref.get("status") in ("converged", "completed")
            report.add(Gate("ts_refine", "PASS" if ok else "WARN",
                            detail=str(ref.get("status")), value=ref.get("confidence"),
                            fallback="saddle-refine runs on the un-refined guess"))
            if ok and ref.get("ts_guess") is not None:
                ts_guess = ref.get("ts_guess")

        # ---- saddle-refine + active-region Hessian ----
        refine_res = None
        ts_atoms = None
        if ts_guess is None:
            # path search / proposer produced no guess — fail cleanly (a critical
            # FAIL gate, no fallback) instead of crashing the refine step.
            report.add(Gate("ts_guess_available", "FAIL",
                            detail="no TS guess was produced (path search or "
                                   "proposer returned none) — nothing to refine"))
        else:
            with Step("refine_ts", outdir, params={"backend": saddle_backend}) as s:
                refine_res = _refine(ts_guess, ctx, resolved, outdir / "refine_ts",
                                     preset, saddle_backend, tangent, template)
                ts_atoms = refine_res.get("ts_atoms")
                s.record(overall_pass=refine_res.get("overall_pass"),
                         n_imag=refine_res.get("n_imag"),
                         imag_freq_cm=refine_res.get("imag_freq_cm"))
            for g in _saddle_gates(refine_res, preset):
                report.add(g)

        # ---- IRC-like validation ----
        val_res = None
        if do_validate and not report.critical_fail and ts_atoms is not None:
            from quantum_engine.ops import imag_mode_displace  # noqa: PLC0415
            imode = refine_res.get("outputs", {}).get("imag_mode")
            with Step("validate_irc_like", outdir) as s:
                val_res = imag_mode_displace.run(
                    _stamp(ts_atoms.copy(), ctx), _calc_fn(ctx)(),
                    outdir=outdir / "validate", imag_mode_path=imode,
                    charge=ctx.charge)
                s.record(overall_pass=val_res.get("overall_pass"))
            report.add(Gate(
                "irc_like_return",
                "PASS" if val_res.get("overall_pass") else "WARN",
                detail="both displaced branches relax to distinct basins",
                value=val_res.get("overall_pass"),
                fallback="TS still has a valid imaginary mode; endpoints unverified"))
            # adopt the validation branch basins as R/P if we had none. The
            # branch energy is authoritative (relaxed-basin energy); the geometry
            # is best-effort (None if no PDB was written) — keep them consistent:
            # only adopt an energy when its geometry actually loaded.
            if R is None and val_res.get("branches"):
                br = val_res["branches"]
                if len(br) >= 2:
                    R = _branch_atoms(val_res, 0); P = _branch_atoms(val_res, 1)
                    if R is not None:
                        R_e = br[0].get("energy_eV")
                    if P is not None:
                        P_e = br[1].get("energy_eV")

        # ---- non-constraining bond/metal report ----
        if monitor and ts_atoms is not None:
            from quantum_engine.ops.bond_monitor import (  # noqa: PLC0415
                bonds_from_reaction, monitor_bonds)
            monitor_bonds(ts_atoms, bonds=bonds_from_reaction(resolved),
                          metals="auto", label="ts", outdir=outdir, write_json=True)

        # ---- aggregate ----
        ts_e = refine_res.get("energy_eV") if refine_res else None
        result = _finalize(entry, report, R, R_e, P, P_e, ts_atoms, ts_e,
                           refine_res, outdir)
        top.attach_gates(report)
        top.record(status=result["status"], barrier_fwd_kcal=result.get("barrier_fwd_kcal"),
                   imag_freq_cm=result.get("imag_freq_cm"))

    report.emit(outdir)
    return result


def _branch_atoms(val_res, i):
    """Best-effort load of a validation branch geometry (or None)."""
    try:
        from ase.io import read  # noqa: PLC0415
        p = val_res["branches"][i].get("output_pdb")
        return read(p) if p else None
    except Exception:  # noqa: BLE001
        return None


def _drive_to_product(reactant, ctx, resolved, outdir, cv_product_s, report):
    """reactant-only: drive R along the CV to a product basin (scan_modes)."""
    from quantum_engine.ops import scan_modes  # noqa: PLC0415
    p, nuc, lg = resolved.cv_bond_difference
    s0 = float(reactant.get_distance(p, lg) - reactant.get_distance(p, nuc))
    s_target = cv_product_s if cv_product_s is not None else abs(s0) + 2.0
    with Step("drive_to_product", outdir,
              params={"cv": [p, nuc, lg], "s_start": s0, "s_end": s_target}) as s:
        sc = scan_modes.run(_stamp(reactant.copy(), ctx), _calc_fn(ctx)(),
                            mode="bond-difference", diff_bonds=((p, lg), (p, nuc)),
                            s_start=s0, s_end=s_target, n_steps=10,
                            outdir=outdir / "drive", fmax=0.05)
        s.record(achieved_s=sc["coord_values"][-1])
    report.add(Gate("drive_to_product", "PASS",
                    detail=f"drove CV s {s0:.2f} → {sc['coord_values'][-1]:.2f}",
                    fallback="endpoint is a CV-driven guess, not a verified basin"))
    return sc["frames"][-1]


def _finalize(entry, report, R, R_e, P, P_e, ts_atoms, ts_e, refine_res, outdir):
    crit = report.critical_fail
    status = "failed" if crit else "converged"
    barrier_fwd = ((ts_e - R_e) * EV_TO_KCAL) if (ts_e is not None and R_e is not None) else None
    barrier_rev = ((ts_e - P_e) * EV_TO_KCAL) if (ts_e is not None and P_e is not None) else None
    dE_rxn = ((P_e - R_e) * EV_TO_KCAL) if (R_e is not None and P_e is not None) else None
    out = {
        "status": status,
        "entry": entry,
        "gates_overall": report.overall,
        "critical_fail": crit,
        "reactant": R, "product": P, "ts": ts_atoms,
        "energy_eV_reactant": R_e, "energy_eV_product": P_e, "energy_eV_ts": ts_e,
        "barrier_fwd_kcal": barrier_fwd, "barrier_rev_kcal": barrier_rev,
        "dE_rxn_kcal": dE_rxn,
        "imag_freq_cm": refine_res.get("imag_freq_cm") if refine_res else None,
        "n_imag": refine_res.get("n_imag") if refine_res else None,
        "outputs": {"gates_json": str(outdir / "gates.json"),
                    "outdir": str(outdir)},
    }
    log.info("ts_entry %s: status=%s gates=%s  Δ‡fwd=%s kcal/mol  imag=%s cm^-1",
             entry, status, report.overall,
             f"{barrier_fwd:.2f}" if barrier_fwd is not None else "n/a",
             out["imag_freq_cm"])
    return out


__all__ = ["run", "RigorPreset", "RIGOR", "ENTRIES"]
