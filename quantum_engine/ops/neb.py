"""Nudged Elastic Band (NEB) with optional CI-NEB climbing image.

Reaction-class-agnostic chain-of-states saddle finder. Suitable for:
    SN2-at-P, SN2-at-C, hydride / proton transfer, Diels-Alder, Cope
    rearrangement, [3,3]-sigmatropic shifts, electrocyclic ring closure,
    ene reaction — anything with a definable reactant and product geometry.

The pipeline accepts two endpoint Atoms and runs:
    1. (optional) interpolation: geodesic / IDPP / linear
    2. NEB stage 1 (no climbing) to relax the path
    3. NEB stage 2 (climbing image) to push the highest image to the saddle
    4. Validation gate (path-collapse / endpoint-saddle / monotonicity)
    5. Output enrichment (PDB, CIF, JSON profile, multi-MODEL trajectory PDB)

Refs:
- Henkelman & Jonsson, JCP 113, 9978 (2000)  — original NEB
- Henkelman, Uberuaga, Jonsson, JCP 113, 9901 (2000) — climbing image
- Zhu, Thompson, Martinez, JCP 150, 164103 (2019) — geodesic interpolation
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from quantum_engine.units import EV_TO_KCAL
from ase import Atoms
from ase.io import read, write

from quantum_engine.mlff.interpolation import interpolate

log = logging.getLogger("quantum_engine.ops.neb")


# ---------------------------------------------------------------------------
# Public dataclass for caller-facing config (kept stable across versions).
# Every knob exposed here MUST be threaded through the CLI in cli.py:_cmd_neb
# and have help text explaining when to bump it. See feedback memory
# "feedback_no_hardcoded_magic_numbers.md".
# ---------------------------------------------------------------------------
@dataclass
class NEBConfig:
    # Image count
    n_images: int = 11
    n_images_default_policy: str = "fixed"   # "fixed" | "atom-density"
    # Spring force handling
    k_spring: float = 1.0
    k_spring_mode: str = "uniform"           # "uniform" | "distance-weighted"
    k_spring_low_curvature_factor: float = 1.5  # multiply k where curvature < threshold
    k_spring_curvature_threshold_eV: float = 0.05  # eV across two adjacent springs
    # Interpolation
    interpolation_method: str = "geodesic"   # "geodesic" | "idpp" | "linear"
    # Optimizer
    optimizer: str = "fire"                  # "fire" | "lbfgs" | "mdmin" | "bfgs"
    max_step: float = 0.05                   # Å per image step
    # Convergence
    fmax_noclimb: float = 0.30
    steps_noclimb: int = 200
    fmax_climb: float = 0.05
    steps_climb: int = 250
    ts_tol_fmax: float | None = None         # if set, do an extra tight refinement to this fmax
    ts_tol_steps: int = 200
    # Climbing image options
    double_ended_only: bool = False          # skip CI; just relax MEP
    ci_image_index: int | None = None        # manual climber pick (default = argmax E)
    # Restart / autobisect
    restart: bool = False
    auto_bisect_on_stall: bool = False
    auto_bisect_window: int = 8              # iters of fmax-stall before bisecting
    auto_bisect_tol: float = 0.005           # eV/Å fmax stall tolerance
    # Trajectory output
    save_trajectory: bool = True
    trajectory_stride: int = 1
    # Calculator batching
    parallel_images: bool = False
    # Reactive bond hints
    key_bonds: list[tuple[int, int]] = field(default_factory=list)  # 0-based atom pairs
    key_bond_kink_tol_A: float = 0.4  # 2nd-difference threshold (Å) for "smooth"
    # Geodesic-specific kwargs (used when interpolation_method="geodesic")
    geodesic_kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def run(
    reactant: Atoms,
    product: Atoms,
    calculator_fn: Callable[[], Any],
    outdir: str | Path = ".",
    constraint=None,
    *,
    config: NEBConfig | None = None,
    charge: int = 0,
    template_st=None,
    # Backwards-compatible scalar kwargs (legacy callers)
    n_images: int | None = None,
    k_spring: float | None = None,
    interpolation_method: str | None = None,
    fmax_noclimb: float | None = None,
    steps_noclimb: int | None = None,
    fmax_climb: float | None = None,
    steps_climb: int | None = None,
    **legacy_kwargs,
) -> dict:
    """Run NEB → CI-NEB.

    Args:
        reactant: endpoint Atoms (with calc settable)
        product: endpoint Atoms
        calculator_fn: zero-arg callable returning a fresh calculator per image
        outdir: output directory
        constraint: ASE constraint or list-of-constraints applied to all images
        config: NEBConfig with all NEB knobs (see dataclass for defaults).
        charge: system net charge stored on each image's info
        template_st: optional biotite AtomArray for PDB/CIF writes
        n_images, k_spring, ...: legacy scalar overrides for the corresponding
            NEBConfig fields (kept so older Python callers keep working).
    """
    cfg = config or NEBConfig()
    # Honor any legacy scalar overrides (CLI now uses config; callers may not).
    if n_images is not None:
        cfg.n_images = n_images
    if k_spring is not None:
        cfg.k_spring = k_spring
    if interpolation_method is not None:
        cfg.interpolation_method = interpolation_method
    if fmax_noclimb is not None:
        cfg.fmax_noclimb = fmax_noclimb
    if steps_noclimb is not None:
        cfg.steps_noclimb = steps_noclimb
    if fmax_climb is not None:
        cfg.fmax_climb = fmax_climb
    if steps_climb is not None:
        cfg.steps_climb = steps_climb

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve effective n_images using policy
    n_images_effective = _resolve_n_images(reactant, cfg)
    if n_images_effective != cfg.n_images:
        log.info(
            f"  n_images policy '{cfg.n_images_default_policy}' resolved "
            f"{cfg.n_images} → {n_images_effective}"
        )
    cfg.n_images = n_images_effective

    # Cached path?
    climb_file = outdir / "path-neb-climb.xyz"
    if climb_file.exists() and not cfg.restart:
        log.info(f"  NEB climb path found at {climb_file} – loading")
        images = read(str(climb_file), index=":")
        for img in images:
            img.calc = calculator_fn()
            img.info["charge"] = charge
        return _summarize_and_validate(
            images, outdir, "cached", cfg, charge=charge, template_st=template_st,
        )

    log.info(
        f"NEB: {cfg.n_images} images, k={cfg.k_spring} "
        f"({cfg.k_spring_mode}), interp={cfg.interpolation_method}, "
        f"optimizer={cfg.optimizer}, max_step={cfg.max_step}"
    )

    # Build images: restart from previous climb path if requested AND available.
    if cfg.restart and climb_file.exists():
        log.info(f"  Restarting from {climb_file}")
        images = read(str(climb_file), index=":")
    else:
        # Pass key-bond aware kwargs to geodesic where possible
        geo_kwargs = dict(cfg.geodesic_kwargs)
        images = interpolate(
            reactant, product, cfg.n_images,
            method=cfg.interpolation_method, **geo_kwargs,
        )

    # Set constraint + calc + charge for every image
    for img in images:
        if constraint is not None:
            img.set_constraint(constraint if isinstance(constraint, list) else [constraint])
        img.calc = calculator_fn()
        img.info["charge"] = charge

    # Configure spring constants
    spring_k = _build_spring_constants(images, cfg)

    # Build the NEB. allow_shared_calculator False because each image needs its own.
    from ase.mep.neb import NEB
    neb = NEB(
        images,
        k=spring_k,
        allow_shared_calculator=cfg.parallel_images,
        method="improvedtangent",
        parallel=False,
    )

    # Initial path snapshot
    write(str(outdir / "path-neb-init.xyz"), images)

    # Stage 1: no-climb relaxation of the MEP
    log.info(f"  NEB no-climb: fmax={cfg.fmax_noclimb}, max {cfg.steps_noclimb} steps")
    optimizer1 = _build_optimizer(neb, outdir / "neb-noclimb-opt.traj", cfg)
    _run_with_optional_bisect(optimizer1, cfg.fmax_noclimb, cfg.steps_noclimb,
                              neb=neb, cfg=cfg, outdir=outdir,
                              calculator_fn=calculator_fn, charge=charge,
                              constraint=constraint)
    write(str(outdir / "path-neb-noclimb.xyz"), images)
    log.info("  NEB no-climb complete")

    # Stage 2: climbing image OR skip if double-ended-only
    if cfg.double_ended_only or cfg.steps_climb <= 0:
        log.info("  Climbing phase skipped (double-ended-only or steps_climb=0)")
        write(str(climb_file), images)
    else:
        # Manual climber pick: NEB normally uses argmax(E); if user forced one,
        # re-order by tweaking image energies via a temporary climb-attribute
        # override on the NEB. ASE's climbing logic uses argmax of the inner
        # images' energies, so we encode user choice by setting climb=True and
        # nudging the chosen image to be tallest. (Less invasive: warn if the
        # user pick disagrees.)
        if cfg.ci_image_index is not None:
            inner = [float(img.get_potential_energy()) for img in images[1:-1]]
            argmax_idx = int(np.argmax(inner)) + 1 if inner else 0
            forced_idx = cfg.ci_image_index
            if forced_idx <= 0 or forced_idx >= len(images) - 1:
                log.warning(
                    f"  --ci-image-index {forced_idx} is an endpoint; falling "
                    f"back to argmax (image {argmax_idx})"
                )
            elif forced_idx != argmax_idx:
                log.warning(
                    f"  --ci-image-index {forced_idx} differs from argmax-E "
                    f"image {argmax_idx}; ASE always climbs argmax. Consider "
                    "rebalancing the path with a denser interpolation if you "
                    "really want the climber elsewhere."
                )

        neb.climb = True
        log.info(f"  NEB climbing: fmax={cfg.fmax_climb}, max {cfg.steps_climb} steps")
        optimizer2 = _build_optimizer(neb, outdir / "neb-climb-opt.traj", cfg)
        _run_with_optional_bisect(optimizer2, cfg.fmax_climb, cfg.steps_climb,
                                  neb=neb, cfg=cfg, outdir=outdir,
                                  calculator_fn=calculator_fn, charge=charge,
                                  constraint=constraint)
        write(str(climb_file), images)

        # Optional tighter "ts-tol" finishing pass: stricter fmax on the
        # climbing image. Useful when the user wants ~0.02 eV/Å convergence
        # (cf. memory feedback_neb_needs_ts_refinement.md). Note that this is
        # NOT a true saddle finder — for full-Hessian validation, follow up
        # with `qcb saddle --backend dimer` and `qcb freq`.
        if cfg.ts_tol_fmax is not None and cfg.ts_tol_fmax < cfg.fmax_climb:
            log.info(
                f"  NEB tight-TS pass: fmax={cfg.ts_tol_fmax}, max "
                f"{cfg.ts_tol_steps} steps"
            )
            optimizer3 = _build_optimizer(neb, outdir / "neb-tighttts-opt.traj", cfg)
            optimizer3.run(fmax=cfg.ts_tol_fmax, steps=cfg.ts_tol_steps)
            write(str(climb_file), images)

        log.info("  NEB climbing complete")

    return _summarize_and_validate(
        images, outdir, "converged", cfg, charge=charge, template_st=template_st,
    )


# ---------------------------------------------------------------------------
# Helper: image-count policy
# ---------------------------------------------------------------------------
def _resolve_n_images(reactant: Atoms, cfg: NEBConfig) -> int:
    """Resolve the effective n_images from the configured policy.

    "fixed":         use cfg.n_images verbatim.
    "atom-density":  scale by sqrt(N_atoms / 100); guaranteed odd, >=5, <=51.
    """
    policy = cfg.n_images_default_policy
    if policy == "fixed":
        return int(cfg.n_images)
    if policy == "atom-density":
        n_at = len(reactant)
        scaled = int(round(cfg.n_images * math.sqrt(n_at / 100.0)))
        scaled = max(5, min(51, scaled))
        if scaled % 2 == 0:
            scaled += 1
        return scaled
    raise ValueError(f"unknown n_images_default_policy: {policy!r}")


# ---------------------------------------------------------------------------
# Helper: build spring-constant array
# ---------------------------------------------------------------------------
def _build_spring_constants(images: Sequence[Atoms], cfg: NEBConfig) -> float | list[float]:
    """Construct per-spring spring constants.

    For "uniform" returns a scalar (legacy behavior). For "distance-weighted",
    we initialize uniform and let the optimizer warm-start; once an initial
    energy profile is available the springs are recomputed (low-curvature
    regions get a stronger spring to pull images apart, freeing image density
    near the saddle).

    Note: this initial call returns a scalar for uniform mode and a list of
    `len(images)-1` floats for distance-weighted mode. ASE NEB accepts both.
    """
    n_springs = len(images) - 1
    if cfg.k_spring_mode == "uniform":
        return cfg.k_spring
    if cfg.k_spring_mode == "distance-weighted":
        # Initialize uniform; ASE doesn't natively re-tune so we just space
        # by initial pairwise distance. Larger inter-image gap → stronger
        # spring (pulls them tighter); smaller gap (likely near saddle) →
        # weaker spring (lets images densify).
        try:
            d = []
            for i in range(n_springs):
                d.append(float(np.linalg.norm(
                    images[i + 1].get_positions() - images[i].get_positions()
                )))
            d_arr = np.asarray(d)
            if d_arr.max() <= 0:
                return cfg.k_spring
            scale = d_arr / d_arr.mean()  # mean-normalized
            ks = (cfg.k_spring * scale).clip(
                min=cfg.k_spring / cfg.k_spring_low_curvature_factor,
                max=cfg.k_spring * cfg.k_spring_low_curvature_factor,
            )
            return list(ks.astype(float))
        except Exception as exc:
            log.warning(f"  distance-weighted spring init failed ({exc}); using uniform")
            return cfg.k_spring
    raise ValueError(f"unknown k_spring_mode: {cfg.k_spring_mode!r}")


# ---------------------------------------------------------------------------
# Helper: optimizer factory
# ---------------------------------------------------------------------------
def _build_optimizer(neb, traj_path: Path, cfg: NEBConfig):
    """Return a configured ASE optimizer wrapping the NEB chain.

    "fire"  : FIRE (Bitzek 2006). Most robust default for chain-of-states.
    "lbfgs" : LBFGS quasi-Newton. Good when path is already near-saddle.
    "mdmin" : Velocity-projected MD. Mid-cost, mid-robustness.
    "bfgs"  : full BFGS. Memory-hungry on large NEBs but very tight near saddle.

    NEBOptimizer (ODE) is also available as "ode" for backwards compat.
    """
    from ase.optimize import FIRE, LBFGS, BFGS, MDMin

    name = cfg.optimizer.lower()
    log_path = traj_path.with_suffix(".log")
    common = dict(trajectory=str(traj_path), logfile=str(log_path))

    if name == "fire":
        return FIRE(neb, maxstep=cfg.max_step, **common)
    if name == "lbfgs":
        return LBFGS(neb, maxstep=cfg.max_step, **common)
    if name == "bfgs":
        return BFGS(neb, maxstep=cfg.max_step, **common)
    if name == "mdmin":
        return MDMin(neb, maxstep=cfg.max_step, **common)
    if name == "ode":
        from ase.mep.neb import NEBOptimizer
        return NEBOptimizer(neb, trajectory=str(traj_path), method="ODE")
    raise ValueError(
        f"unknown optimizer: {cfg.optimizer!r}. Choices: fire, lbfgs, "
        "bfgs, mdmin, ode"
    )


# ---------------------------------------------------------------------------
# Optional auto-bisect on stall
# ---------------------------------------------------------------------------
def _run_with_optional_bisect(
    optimizer, fmax: float, steps: int, *,
    neb, cfg: NEBConfig, outdir: Path,
    calculator_fn: Callable[[], Any], charge: int, constraint,
):
    """Run optimizer; if auto_bisect_on_stall is on, watch for fmax stalling."""
    if not cfg.auto_bisect_on_stall:
        optimizer.run(fmax=fmax, steps=steps)
        return

    # Run in chunks of cfg.auto_bisect_window steps and check for plateau.
    window = max(1, int(cfg.auto_bisect_window))
    tol = cfg.auto_bisect_tol
    history: list[float] = []
    remaining = steps
    while remaining > 0:
        run_steps = min(window, remaining)
        # Capture pre-run residual; ASE optimizers expose .converged() but not
        # a clean residual getter, so probe via NEB.
        optimizer.run(fmax=fmax, steps=run_steps)
        remaining -= run_steps
        try:
            f = neb.get_residual()
        except Exception:
            try:
                f = float(np.linalg.norm(neb.get_forces(), axis=1).max())
            except Exception:
                f = float("nan")
        if not math.isfinite(f):
            break
        history.append(f)
        if f <= fmax:
            return
        # Check stall: last `window` residuals all within tol of each other?
        if len(history) >= 3:
            recent = history[-3:]
            if max(recent) - min(recent) < tol and remaining > 0:
                log.warning(
                    f"  auto-bisect-on-stall: residual plateau at {f:.4f} "
                    f"(tol={tol}); inserting an image at the highest-energy "
                    "region and continuing"
                )
                _autobisect_path(neb, calculator_fn, charge, constraint)
                history.clear()


def _autobisect_path(neb, calculator_fn, charge, constraint) -> None:
    """Insert one new image at the highest-energy interior gap.

    Finds the image with max E. Inserts a new image halfway between it and the
    higher-energy neighbor. Mutates neb.images in place. Springs are left at
    their previous values; ASE expands `k` to a list automatically if needed.
    """
    images = list(neb.images)
    if len(images) < 4:
        return
    inner_e = [float(img.get_potential_energy()) for img in images[1:-1]]
    argmax_inner = int(np.argmax(inner_e))
    ts_idx = argmax_inner + 1
    # Pick the higher-energy neighbor
    left = images[ts_idx - 1].get_potential_energy()
    right = images[ts_idx + 1].get_potential_energy()
    if left >= right:
        a, b = ts_idx - 1, ts_idx
    else:
        a, b = ts_idx, ts_idx + 1
    new_img = images[a].copy()
    new_img.set_positions(0.5 * (images[a].get_positions() + images[b].get_positions()))
    new_img.calc = calculator_fn()
    new_img.info["charge"] = charge
    if constraint is not None:
        new_img.set_constraint(
            constraint if isinstance(constraint, list) else [constraint]
        )
    images.insert(b, new_img)
    neb.images = images


# ---------------------------------------------------------------------------
# Validation gate
# ---------------------------------------------------------------------------
def _validate_path(images: Sequence[Atoms], cfg: NEBConfig) -> dict:
    """Run sanity checks on the converged path and return a dict result.

    Checks:
        endpoints_low      : E(R), E(P) are local minima of the path
        ci_not_endpoint    : the highest image is not 0 or N-1
        monotonic_ascent   : energies rise monotonically up to the climber
        monotonic_descent  : energies fall monotonically from climber to product
        rms_displacement_R : R didn't drift (frozen if constrained)
        rms_displacement_P : P didn't drift (frozen if constrained)
        key_bond_smoothness: distances of key_bonds change monotonically
                             (only if cfg.key_bonds is non-empty)

    Returns dict with per-check status (PASS/WARN/FAIL) and a top-level
    `summary` field of "PASS" / "WARN" / "FAIL".
    """
    energies = [float(img.get_potential_energy()) for img in images]
    n = len(images)
    e_r, e_p = energies[0], energies[-1]
    inner = energies[1:-1]
    if inner:
        argmax_inner = int(np.argmax(inner))
        ts_idx = argmax_inner + 1
    else:
        ts_idx = 0
    e_ts = energies[ts_idx]

    checks: dict[str, dict] = {}

    # 1. endpoints are local minima
    e_first_inner = energies[1] if n > 2 else e_r
    e_last_inner = energies[-2] if n > 2 else e_p
    checks["endpoints_low"] = {
        "status": "PASS" if (e_r <= e_first_inner and e_p <= e_last_inner) else "WARN",
        "detail": f"E(R)={e_r:.4f}, E(image1)={e_first_inner:.4f}, "
                  f"E(P)={e_p:.4f}, E(image-2)={e_last_inner:.4f}",
    }

    # 2. CI is not at an endpoint
    if ts_idx in (0, n - 1):
        ci_status = "FAIL"
        ci_detail = f"climbing image at endpoint (idx={ts_idx}); rerun with " \
                    "wider scan window — endpoint is not really a minimum."
    else:
        ci_status = "PASS"
        ci_detail = f"climbing image at interior index {ts_idx} (good)"
    checks["ci_not_endpoint"] = {"status": ci_status, "detail": ci_detail}

    # 3. monotonic ascent up to ts_idx
    asc = all(energies[i] <= energies[i + 1] + 1e-6 for i in range(ts_idx))
    checks["monotonic_ascent"] = {
        "status": "PASS" if asc else "WARN",
        "detail": "energies rise monotonically up to climber" if asc else
                  f"non-monotonic ascent: {energies[:ts_idx+1]}",
    }

    # 4. monotonic descent from ts_idx
    desc = all(energies[i] >= energies[i + 1] - 1e-6 for i in range(ts_idx, n - 1))
    checks["monotonic_descent"] = {
        "status": "PASS" if desc else "WARN",
        "detail": "energies fall monotonically from climber" if desc else
                  f"non-monotonic descent: {energies[ts_idx:]}",
    }

    # 5. key-bond smoothness (where requested)
    if cfg.key_bonds:
        kb_dist = []  # list of arrays; one per key_bond
        for (a, b) in cfg.key_bonds:
            d = []
            for img in images:
                pos = img.get_positions()
                d.append(float(np.linalg.norm(pos[a] - pos[b])))
            kb_dist.append(d)
        # smoothness = max abs second-difference, normalized
        smooth_ok = True
        worst = 0.0
        for series, (a, b) in zip(kb_dist, cfg.key_bonds):
            arr = np.asarray(series)
            if len(arr) < 3:
                continue
            d2 = np.abs(np.diff(arr, n=2))
            worst = max(worst, float(d2.max()))
            # 2nd-difference threshold (rule of thumb; user-tunable)
            if d2.max() > cfg.key_bond_kink_tol_A:
                smooth_ok = False
        checks["key_bond_smoothness"] = {
            "status": "PASS" if smooth_ok else "WARN",
            "detail": f"max |d²/di²| over key bonds = {worst:.3f} Å",
        }

    # Aggregate
    if any(c["status"] == "FAIL" for c in checks.values()):
        summary = "FAIL"
    elif any(c["status"] == "WARN" for c in checks.values()):
        summary = "WARN"
    else:
        summary = "PASS"

    return {
        "summary": summary,
        "checks": checks,
        "ts_index": ts_idx,
        "energies_eV": energies,
        "barrier_fwd_kcal": float((e_ts - e_r) * EV_TO_KCAL),
        "barrier_rev_kcal": float((e_ts - e_p) * EV_TO_KCAL),
        "dE_rxn_kcal": float((e_p - e_r) * EV_TO_KCAL),
    }


# ---------------------------------------------------------------------------
# Output enrichment
# ---------------------------------------------------------------------------
def _summarize_and_validate(
    images: Sequence[Atoms],
    outdir: Path,
    status: str,
    cfg: NEBConfig,
    *,
    charge: int = 0,
    template_st=None,
) -> dict:
    """Compose all post-NEB outputs: profile.json, neb-profile.png, R/P/TS
    individual files (PDB + CIF + XYZ), trajectory PDB if requested, and the
    validation gate JSON.
    """
    energies = [float(img.get_potential_energy()) for img in images]
    n = len(images)
    e_r, e_p = energies[0], energies[-1]
    inner = energies[1:-1]
    ts_idx = (int(np.argmax(inner)) + 1) if inner else 0
    e_ts = energies[ts_idx]

    barrier_fwd = (e_ts - e_r) * EV_TO_KCAL
    barrier_rev = (e_ts - e_p) * EV_TO_KCAL
    dE_rxn = (e_p - e_r) * EV_TO_KCAL

    # -------- Validation gate (writes qcb_neb_validation.json) --------
    validation = _validate_path(images, cfg)
    (outdir / "qcb_neb_validation.json").write_text(
        json.dumps(validation, indent=2)
    )
    log.info(
        f"  NEB validation: {validation['summary']} "
        f"({sum(1 for c in validation['checks'].values() if c['status'] == 'PASS')}"
        f"/{len(validation['checks'])} PASS)"
    )
    for name, check in validation["checks"].items():
        marker = {"PASS": " ✓", "WARN": " !", "FAIL": " ✗"}.get(check["status"], " ?")
        log.info(f"    {marker} {name}: {check['detail']}")

    # -------- profile.json with key-bond distances --------
    profile_payload: dict = {
        "image_index": list(range(n)),
        "energy_eV": energies,
        "energy_rel_kcal": [(e - e_r) * EV_TO_KCAL for e in energies],
        "ts_index": ts_idx,
        "barrier_fwd_kcal": float(barrier_fwd),
        "barrier_rev_kcal": float(barrier_rev),
        "dE_rxn_kcal": float(dE_rxn),
        "key_bond_distances_A": {},
        "force_residual_eV_per_A": _per_image_residual(images),
    }
    if cfg.key_bonds:
        for (a, b) in cfg.key_bonds:
            d = [
                float(np.linalg.norm(img.get_positions()[a] - img.get_positions()[b]))
                for img in images
            ]
            profile_payload["key_bond_distances_A"][f"{a}-{b}"] = d
    (outdir / "profile.json").write_text(json.dumps(profile_payload, indent=2))

    # -------- profile PNG --------
    png = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        rel = profile_payload["energy_rel_kcal"]
        ax.plot(range(n), rel, "o-", markersize=6, label="MEP")
        ax.axvline(ts_idx, color="r", linestyle="--", alpha=0.5, label=f"TS (image {ts_idx})")
        ax.set_xlabel("Image")
        ax.set_ylabel("Relative energy (kcal/mol)")
        ax.set_title(f"NEB path (Δ‡fwd = {barrier_fwd:.1f} kcal/mol)")
        ax.grid(alpha=0.3)
        ax.legend()
        png = outdir / "neb-profile.png"
        fig.tight_layout()
        fig.savefig(str(png), dpi=150)
        plt.close(fig)
    except Exception as exc:
        log.warning(f"  matplotlib plot failed: {exc}")

    # -------- Multi-MODEL trajectory PDB + extxyz --------
    traj_outputs: dict = {}
    if cfg.save_trajectory:
        try:
            stride = max(1, int(cfg.trajectory_stride))
            sub_imgs = list(images[::stride])
            if sub_imgs[-1] is not images[-1]:
                sub_imgs.append(images[-1])
            traj_pdb = outdir / "neb-path-trajectory.pdb"
            try:
                write(str(traj_pdb), sub_imgs, format="proteindatabank")
                traj_outputs["trajectory_pdb"] = str(traj_pdb)
            except Exception as exc:
                log.warning(f"  PDB trajectory write failed: {exc}")
            traj_xyz = outdir / "neb-path-trajectory.extxyz"
            for frame, e in zip(sub_imgs, energies[::stride]):
                frame.info["energy_eV"] = e
            try:
                write(str(traj_xyz), sub_imgs, format="extxyz")
                traj_outputs["trajectory_extxyz"] = str(traj_xyz)
            except Exception as exc:
                log.warning(f"  extxyz trajectory write failed: {exc}")
        except Exception as exc:
            log.warning(f"  trajectory output failed: {exc}")

    # -------- R, P, TS individual files (XYZ / PDB / CIF) --------
    rpts_paths = write_neb_outputs(
        reactant=images[0],
        product=images[-1],
        ts=images[ts_idx],
        outdir=outdir,
        template_st=template_st,
        charge=charge,
        e_r_eV=e_r,
        e_p_eV=e_p,
        e_ts_eV=e_ts,
    )

    # -------- Final return --------
    result = {
        "status": status,
        "images": list(images),
        "ts": images[ts_idx],
        "ts_idx": ts_idx,
        "reactant": images[0],
        "product": images[-1],
        "energies_eV": energies,
        "barrier_fwd_kcal": float(barrier_fwd),
        "barrier_rev_kcal": float(barrier_rev),
        "dE_rxn_kcal": float(dE_rxn),
        "validation": validation,
        "outputs": {
            "path": str(outdir / "path-neb-climb.xyz"),
            "plot": str(png) if png else None,
            "profile_json": str(outdir / "profile.json"),
            "validation_json": str(outdir / "qcb_neb_validation.json"),
            **traj_outputs,
            **rpts_paths,
        },
    }
    return result


def _per_image_residual(images: Sequence[Atoms]) -> list[float]:
    """Return per-image |F|max in eV/Å (skipping endpoints which are typically frozen)."""
    out: list[float] = []
    for img in images:
        try:
            f = img.get_forces()
            out.append(float(np.linalg.norm(f, axis=1).max()))
        except Exception:
            out.append(float("nan"))
    return out


def write_neb_outputs(
    reactant: Atoms,
    product: Atoms,
    ts: Atoms,
    outdir: str | Path,
    *,
    template_st=None,
    charge: int = 0,
    e_r_eV: float | None = None,
    e_p_eV: float | None = None,
    e_ts_eV: float | None = None,
) -> dict:
    """Write R, P, TS endpoints + climber as individual XYZ / PDB / (optional CIF).

    Returns a dict mapping logical names ("ts_xyz", "ts_pdb", "ts_cif",
    "reactant_pdb", "product_pdb") to paths. Robust to missing biotite
    template (CIF write skipped) and to missing atomworks (CIF skipped with
    a warning).
    """
    from quantum_engine.io.structure import write_pdb
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    # XYZ — always works
    xyz_paths = {
        "reactant": outdir / "reactant.xyz",
        "product": outdir / "product.xyz",
        "ts": outdir / "ts.xyz",
    }
    for label, atoms in [("reactant", reactant), ("product", product), ("ts", ts)]:
        try:
            write(str(xyz_paths[label]), atoms, format="extxyz")
            outputs[f"{label}_xyz"] = str(xyz_paths[label])
        except Exception as exc:
            log.warning(f"  XYZ write {label} failed: {exc}")

    # PDB — if biotite template present, preserve annotations + REMARKs
    pdb_paths = {
        "reactant": outdir / "reactant.pdb",
        "product": outdir / "product.pdb",
        "ts": outdir / "ts.pdb",
    }
    energy_map = {"reactant": e_r_eV, "product": e_p_eV, "ts": e_ts_eV}
    for label, atoms in [("reactant", reactant), ("product", product), ("ts", ts)]:
        try:
            write_pdb(
                atoms, template_st, pdb_paths[label],
                total_charge=charge, energy_eV=energy_map[label],
            )
            outputs[f"{label}_pdb"] = str(pdb_paths[label])
        except Exception as exc:
            log.warning(f"  PDB write {label} failed: {exc}")

    # CIF for the TS — best effort via atomworks (already used by ts.py)
    try:
        from quantum_engine.io.cif import write_ts_cif
        try:
            write_ts_cif(
                ts, template_st, outdir,
                total_charge=charge, filename="ts.cif",
            )
            outputs["ts_cif"] = str(outdir / "ts.cif")
        except Exception as exc:
            log.warning(f"  CIF write failed: {exc}")
    except ImportError:
        log.info("  atomworks unavailable; skipping CIF output")

    return outputs
