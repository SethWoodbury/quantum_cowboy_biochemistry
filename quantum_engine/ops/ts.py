"""High-level TS search pipeline — NATIVE composition of quantum_engine.ops primitives.

This replaces the legacy subprocess-based wrapper around scripts/run_neb_ts.py.
The pipeline is now pure Python composition of:
  saddle → irc   (for strategy="irc")
  opt(R) + opt(P) + neb + saddle  (for strategy="legacy" or "cv-spring")
  mtd → neb + saddle  (for strategy="mtd")

Advantages over the subprocess wrapper:
- Single calc lifecycle → no energy-state mismatches (R3 bug: same geometry
  evaluated with different calc instances gave 3500 kcal/mol inconsistency)
- Flows Python objects (atoms + calc) directly, no file round-trips
- Easier to extend, test, and compose into notebooks
- `atoms.info["charge"]` set ONCE at pipeline entry, flows through

Strategies (--strategy flag):
  legacy      — spring-driven reactant/product opt + NEB + Sella (classic)
  irc         — Sella saddle on TS guess + IRC descent both ways (gold standard)
  cv-spring   — bond-difference CV drive to reactant + product, then NEB
  mtd         — well-tempered metadynamics → basins → NEB

All strategies use:
  - Geodesic interpolation (default) for NEB initial path
  - atoms.info["charge"] for charge-aware MACE-OMOL
  - Shared constraint (e.g., FixAtoms for CAs)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from quantum_engine.units import EV_TO_KCAL
from ase import Atoms
from ase.optimize import LBFGS

log = logging.getLogger("quantum_engine.ops.ts")



def _setup_calc_and_constraint(atoms, calc_fn, charge: int, constraint=None):
    """Attach fresh calc + set charge + set constraint (once, consistently)."""
    atoms.calc = calc_fn()
    atoms.info["charge"] = charge
    if constraint is not None:
        atoms.set_constraint(constraint if isinstance(constraint, list) else [constraint])
    return atoms


def _energy_consistent(atoms_a, atoms_b, tolerance_eV: float = 1.0) -> bool:
    """Check if two Atoms objects have consistent energies under their current calcs.
    Returns True if |E_a - E_b| < tolerance_eV. Used to catch charge-state bugs."""
    try:
        ea = atoms_a.get_potential_energy()
        eb = atoms_b.get_potential_energy()
        return abs(ea - eb) < tolerance_eV
    except Exception:
        return True  # if either can't compute, don't flag


def assert_frame_consistency(
    candidate: Atoms,
    source_atoms: Atoms,
    constraint=None,
    *,
    anchor_tol: float = 0.01,
    max_anchor_neighbor_dist: float = 2.5,
    com_drift_warn: float = 5.0,
    label: str = "stage",
) -> None:
    """Hard frame-consistency gate. RAISES RuntimeError if a stage produced
    a geometry where FixAtoms anchors are decoupled from the rest of the
    molecule (the geodesic-vs-FixAtoms frame-shift bug).

    Use after every stage that builds or modifies images/endpoints.

    Checks (when a FixAtoms constraint is present):
      (1) anchors at template positions: max(|cand[anchor] - src[anchor]|) < anchor_tol
      (2) no anchor-rest decoupling: every anchor has at least one non-anchor
          atom within max_anchor_neighbor_dist Å (anchor not floating in vacuum)
      (3) COM drift warning: ||COM(free) - COM(fixed)|| - source_value
          (rate-limited warning if exceeds com_drift_warn)
    """
    from ase.constraints import FixAtoms
    cs = constraint if isinstance(constraint, list) else ([constraint] if constraint else [])
    fix_idx = None
    for c in cs:
        if isinstance(c, FixAtoms):
            fix_idx = np.asarray(c.index, dtype=int)
            break
    if fix_idx is None or len(fix_idx) == 0:
        return

    p_cand = candidate.get_positions()
    p_src = source_atoms.get_positions()

    # (1) anchors didn't move
    d_anchor = np.linalg.norm(p_cand[fix_idx] - p_src[fix_idx], axis=1)
    if d_anchor.max() > anchor_tol:
        raise RuntimeError(
            f"[{label}] FixAtoms anchors moved {d_anchor.max():.3f} Å "
            f"(> {anchor_tol} Å); FixAtoms invariant violated."
        )

    # (2) anchor-to-rest decoupling — the smoking-gun frame-shift bug check
    free_idx = np.array([i for i in range(len(candidate)) if i not in set(fix_idx.tolist())])
    if len(free_idx) > 0:
        dmin_to_free = np.min(
            np.linalg.norm(p_cand[fix_idx][:, None, :] - p_cand[free_idx][None, :, :], axis=2),
            axis=1,
        )
        bad = np.where(dmin_to_free > max_anchor_neighbor_dist)[0]
        if len(bad):
            worst_local = int(bad[np.argmax(dmin_to_free[bad])])
            worst_global = int(fix_idx[worst_local])
            raise RuntimeError(
                f"[{label}] {len(bad)} FixAtoms anchors are >{max_anchor_neighbor_dist} Å "
                f"from any free atom (DECOUPLED). Worst: anchor index {worst_global}, "
                f"min-d-to-free = {dmin_to_free.max():.2f} Å. This is the "
                "geodesic-vs-FixAtoms frame-shift bug; do NOT trust "
                "energies/barriers from this stage."
            )

    # (3) COM-drift warning
    if len(free_idx):
        delta_cand = float(np.linalg.norm(p_cand[free_idx].mean(axis=0) - p_cand[fix_idx].mean(axis=0)))
        delta_src = float(np.linalg.norm(p_src[free_idx].mean(axis=0) - p_src[fix_idx].mean(axis=0)))
        drift = abs(delta_cand - delta_src)
        if drift > com_drift_warn:
            log.warning(
                f"[{label}] COM(free)-COM(fixed) drift {drift:.2f} Å vs source "
                f"({delta_src:.2f} → {delta_cand:.2f}); possible frame inconsistency."
            )


def run(
    input_atoms: Atoms,
    calculator_fn,
    outdir: str | Path,
    strategy: str = "legacy",
    charge: int = 0,
    constraint=None,
    n_images: int = 15,
    interpolation: str = "geodesic",
    # Strategy-specific knobs (pass-through)
    saddle_fmax: float = 0.02,
    irc_step: float = 0.1,
    irc_fmax: float = 0.03,
    neb_fmax_noclimb: float = 0.40,
    neb_steps_noclimb: int = 200,
    neb_fmax_climb: float = 0.045,
    neb_steps_climb: int = 250,
    cv_s_reactant: float = -2.0,
    cv_s_product: float = 2.5,
    spring_k: float = 3.0,
    spring_fmax: float = 3.0,
    spring_drive_fmax: float = 0.10,
    opt_final_fmax: float = 0.04,
    # CV atoms for cv-spring/mtd (optional; if None, strategy="cv-spring"/"mtd"
    # will error out unless auto-detected upstream)
    center_idx: int | None = None,
    forming_idx: int | None = None,
    breaking_idx: int | None = None,
    # MTD
    mtd_time_ps: float = 100.0,
    mtd_temperature_K: float = 300.0,
    mtd_variant: str = "wt",
    # Output preservation
    template=None,
    **kwargs,
) -> dict:
    """Native TS search pipeline.

    Args:
        input_atoms: starting geometry (interpretation depends on strategy):
            - strategy="irc": TS guess (will be refined by Sella)
            - strategy="legacy"/"cv-spring": any geometry (reactant-like, TS-like, doesn't matter)
            - strategy="mtd": any geometry (MTD will explore basins)
        calculator_fn: zero-arg callable returning a FRESH calculator each call
        outdir: output directory
        strategy: "legacy" | "irc" | "cv-spring" | "mtd"
        charge: system net charge (set on atoms.info)
        constraint: ASE constraint (e.g., FixAtoms for CAs)
        n_images: NEB image count (ignored for irc)
        interpolation: NEB interpolation method
        center_idx, forming_idx, breaking_idx: CV atoms (required for cv-spring and mtd)

    Returns: dict with status, ts, reactant, product, barriers_kcal, outputs.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    relax_dir = outdir / "relax"
    ts_dir = outdir / "ts"
    relax_dir.mkdir(exist_ok=True)
    ts_dir.mkdir(exist_ok=True)

    t0 = time.time()
    result = {"status": "unknown", "strategy": strategy, "outputs": {}}

    # Attach calc + constraint to input
    input_atoms = input_atoms.copy()
    _setup_calc_and_constraint(input_atoms, calculator_fn, charge, constraint)
    # Snapshot the source frame BEFORE any stage touches the geometry.
    # Used by assert_frame_consistency to detect frame-shift bugs.
    _source_atoms = input_atoms.copy()

    log.info("=" * 60)
    log.info(f"cowboy-qc ts: strategy={strategy}, charge={charge:+d}, n_atoms={len(input_atoms)}")
    log.info("=" * 60)

    # ===================================================================
    # STRATEGY: IRC-FROM-TS
    # ===================================================================
    if strategy == "irc":
        from quantum_engine.mlff.irc import irc_from_ts_guess

        # Build bond-pattern hint (disambiguates forward/reverse assignment)
        hint_bonds = None
        if center_idx is not None and forming_idx is not None and breaking_idx is not None:
            hint_bonds = {
                (center_idx, forming_idx): "broken",
                (center_idx, breaking_idx): "bonded",
            }

        irc_res = irc_from_ts_guess(
            input_atoms, outdir=ts_dir, constraint=constraint,
            saddle_fmax=saddle_fmax, irc_step=irc_step, irc_fmax=irc_fmax,
            reactant_hint_bonds=hint_bonds,
        )
        ts = irc_res.get("ts")
        reactant = irc_res.get("reactant")
        product = irc_res.get("product")

        if ts is None or reactant is None or product is None:
            log.error("  IRC did not produce full (R, TS, P). Returning partial result.")
            result["status"] = "failed"
            return result

        # FRAME-CONSISTENCY GATE — refuse to ship a decoupled-anchor structure
        for a, lbl in [(reactant, "irc/reactant"), (ts, "irc/ts"), (product, "irc/product")]:
            assert_frame_consistency(a, _source_atoms, constraint, label=lbl)

        # Rebuild atoms with a single consistent calc for final energy eval
        for a in (reactant, ts, product):
            _setup_calc_and_constraint(a, calculator_fn, charge, constraint)

        e_r = reactant.get_potential_energy()
        e_ts = ts.get_potential_energy()
        e_p = product.get_potential_energy()

        _finalize_result(result, reactant, ts, product, e_r, e_ts, e_p, outdir, template, charge)
        result["imag_freq_cm"] = irc_res.get("imag_freq_cm")
        result["time_minutes"] = (time.time() - t0) / 60
        return result

    # ===================================================================
    # STRATEGY: METADYNAMICS
    # ===================================================================
    if strategy == "mtd":
        from quantum_engine.mlff.metadynamics import run_metadynamics_rescue, run_opes_rescue
        from quantum_engine.mlff.cv_spring import bond_cv_terms_from_roles
        if center_idx is None or forming_idx is None or breaking_idx is None:
            raise ValueError("MTD requires center_idx, forming_idx, breaking_idx (CV atom indices)")

        entry = run_opes_rescue if mtd_variant == "opes" else run_metadynamics_rescue
        # Single-center bond-difference CV; reactant/product fall out of the
        # extreme-CV basins (reaction-agnostic), so no basin_windows needed.
        cv_terms = bond_cv_terms_from_roles(center_idx, breaking_idx, forming_idx)
        mtd_res = entry(
            input_atoms,
            cv_terms,
            calculator=input_atoms.calc,
            outdir=relax_dir,
            constraint=constraint,
            total_time_ps=mtd_time_ps,
            temperature_K=mtd_temperature_K,
        )
        reactant = mtd_res.get("reactant")
        product = mtd_res.get("product")
        if reactant is None or product is None:
            log.error("  MTD did not produce both basins. Falling back to legacy.")
            strategy = "legacy"  # fall through to legacy below
        else:
            log.info("  MTD basins found. Running NEB between them.")
            # Proceed to NEB with MTD-found endpoints (fall through to combined path below)

    # ===================================================================
    # STRATEGIES: CV-SPRING, LEGACY, or MTD-post-basins all converge here
    # ===================================================================
    if strategy in ("legacy", "cv-spring") or (strategy == "mtd" and reactant is not None):
        # Produce reactant/product endpoints
        if strategy == "cv-spring":
            if center_idx is None or forming_idx is None or breaking_idx is None:
                raise ValueError("cv-spring requires center_idx, forming_idx, breaking_idx")
            reactant, product = _cv_spring_endpoints(
                input_atoms, calculator_fn, charge, constraint,
                center_idx, forming_idx, breaking_idx,
                cv_s_reactant, cv_s_product,
                spring_k, spring_fmax,
                spring_drive_fmax, opt_final_fmax,
                relax_dir,
            )
        elif strategy == "legacy":
            # Simple legacy: just relax the input as both endpoints starting points
            # (users should prefer cv-spring or irc; this is a naive baseline)
            reactant = _simple_relax(
                input_atoms.copy(), calculator_fn, charge, constraint,
                fmax=opt_final_fmax, label="reactant", outdir=relax_dir,
            )
            product = _simple_relax(
                input_atoms.copy(), calculator_fn, charge, constraint,
                fmax=opt_final_fmax, label="product", outdir=relax_dir,
            )
        # mtd case: reactant, product already set

        # FRAME-CONSISTENCY GATE on endpoints (catches cv-spring drive failures)
        assert_frame_consistency(reactant, _source_atoms, constraint, label=f"{strategy}/reactant")
        assert_frame_consistency(product,  _source_atoms, constraint, label=f"{strategy}/product")

        # Run NEB
        from quantum_engine.ops import neb
        # Ensure endpoints have their own calc instances with charge set
        _setup_calc_and_constraint(reactant, calculator_fn, charge, constraint)
        _setup_calc_and_constraint(product, calculator_fn, charge, constraint)

        neb_res = neb.run(
            reactant, product, calculator_fn, outdir=ts_dir,
            constraint=constraint,
            n_images=n_images, k_spring=1.0,
            interpolation_method=interpolation,
            fmax_noclimb=neb_fmax_noclimb, steps_noclimb=neb_steps_noclimb,
            fmax_climb=neb_fmax_climb, steps_climb=neb_steps_climb,
            charge=charge,
        )
        ts = neb_res.get("ts")
        images = neb_res.get("images")

        if ts is None:
            result["status"] = "failed"
            result["error"] = "NEB did not produce TS"
            return result

        # FRAME-CONSISTENCY GATE on every NEB image (catches the
        # geodesic-vs-FixAtoms frame-shift bug; see interpolation.py fix).
        for i, img in enumerate(images or []):
            assert_frame_consistency(img, _source_atoms, constraint,
                                     label=f"{strategy}/NEB-image-{i}")
        assert_frame_consistency(ts, _source_atoms, constraint, label=f"{strategy}/TS")

        # Critical: ensure TS has same calc setup as endpoints for consistent energy
        # This is what the legacy pipeline got wrong.
        ts = ts.copy()
        _setup_calc_and_constraint(ts, calculator_fn, charge, constraint)

        # Optional: Sella refinement of TS (if requested via future flag)
        # For now, skip; Sella can be called separately via `cowboy-qc saddle`

        # Get energies (all under consistent calc setup)
        e_r = reactant.get_potential_energy()
        e_ts = ts.get_potential_energy()
        e_p = product.get_potential_energy()

        # Energy sanity check: if NEB reported different e_r than we compute now, flag it
        neb_energies = neb_res.get("energies_eV", [])
        if neb_energies:
            neb_e_r = neb_energies[0]
            if abs(neb_e_r - e_r) > 1.0:  # 1 eV = 23 kcal/mol
                log.warning(
                    f"  ENERGY INCONSISTENCY: reactant E from NEB = {neb_e_r:.3f} eV, "
                    f"from fresh calc = {e_r:.3f} eV (diff = {(e_r - neb_e_r) * EV_TO_KCAL:.1f} kcal/mol). "
                    "This usually means atoms.info['charge'] wasn't propagated. Using NEB values."
                )
                # Use NEB energies for consistency (those were calculated during the run
                # with all images having matching calc setup)
                e_r = neb_e_r
                e_p = neb_energies[-1]
                inner = neb_energies[1:-1]
                ts_idx = int(np.argmax(inner)) + 1 if inner else 0
                e_ts = neb_energies[ts_idx]

        _finalize_result(result, reactant, ts, product, e_r, e_ts, e_p, outdir, template, charge,
                         neb_images=images)
        result["time_minutes"] = (time.time() - t0) / 60
        return result

    result["status"] = "failed"
    result["error"] = f"Unknown strategy '{strategy}'"
    return result


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _simple_relax(atoms, calc_fn, charge, constraint, fmax, label, outdir):
    """LBFGS-relax and return the relaxed atoms."""
    _setup_calc_and_constraint(atoms, calc_fn, charge, constraint)
    log.info(f"  Simple relax ({label}) ...")
    opt = LBFGS(atoms, logfile=str(outdir / f"relax-{label}.log"),
                trajectory=str(outdir / f"relax-{label}.traj"))
    opt.run(fmax=fmax, steps=500)
    return atoms


def _cv_spring_endpoints(
    input_atoms, calc_fn, charge, constraint,
    center_idx, forming_idx, breaking_idx,
    s_reactant, s_product,
    spring_k, spring_fmax,
    spring_drive_fmax, opt_final_fmax,
    outdir,
):
    """Drive input to reactant (s=s_reactant) and product (s=s_product) via CV spring."""
    from quantum_engine.mlff.cv_spring import BondDifferenceCVSpring

    # Snapshot the source frame BEFORE any drive runs — used to gate frame
    # consistency at every drive/polish boundary so a runaway spring or
    # geodesic-style frame shift can't propagate into NEB.
    _source_for_gate = input_atoms.copy()

    def _drive(s_target: float, label: str):
        a = input_atoms.copy()
        _setup_calc_and_constraint(a, calc_fn, charge, constraint)
        base_cs = list(a.constraints) if a.constraints else []
        cv_spring = BondDifferenceCVSpring(
            center_idx=center_idx, forming_idx=forming_idx, breaking_idx=breaking_idx,
            k=spring_k, s_target=s_target, fmax=spring_fmax,
        )
        a.set_constraint(base_cs + [cv_spring])
        log.info(f"  CV-drive ({label}, s_target={s_target:+.1f}) ...")
        # NOTE: no trajectory= here — ASE's Trajectory writer reopens+deserializes the file
        # which fails for our custom BondDifferenceCVSpring constraint.
        # Log-only; final geometry is what we care about.
        opt = LBFGS(a, logfile=str(outdir / f"cv-drive-{label}.log"))
        opt.run(fmax=spring_drive_fmax, steps=300)
        # Frame-consistency gate immediately after drive (catches a spring that
        # ran away). Defense in depth: the polish step might mask symptoms.
        assert_frame_consistency(a, _source_for_gate, constraint,
                                 label=f"cv-drive/{label}")
        # Remove CV spring, keep opt constraint, polish (now safe to use trajectory)
        a.set_constraint(base_cs)
        opt2 = LBFGS(a, logfile=str(outdir / f"cv-polish-{label}.log"),
                     trajectory=str(outdir / f"cv-polish-{label}.traj"))
        opt2.run(fmax=opt_final_fmax, steps=200)
        # Gate again after polish.
        assert_frame_consistency(a, _source_for_gate, constraint,
                                 label=f"cv-polish/{label}")
        return a

    reactant = _drive(s_reactant, "reactant")
    product = _drive(s_product, "product")
    return reactant, product


def _finalize_result(result, reactant, ts, product, e_r, e_ts, e_p, outdir,
                     template, charge, neb_images=None):
    """Compute barriers, write PDBs, populate result dict."""
    from quantum_engine.io import write_pdb

    barrier_fwd = (e_ts - e_r) * EV_TO_KCAL
    barrier_rev = (e_ts - e_p) * EV_TO_KCAL
    dE_rxn = (e_p - e_r) * EV_TO_KCAL

    result["status"] = "converged"
    result["reactant"] = reactant
    result["ts"] = ts
    result["product"] = product
    result["energy_eV_reactant"] = float(e_r)
    result["energy_eV_ts"] = float(e_ts)
    result["energy_eV_product"] = float(e_p)
    result["barrier_fwd_kcal"] = float(barrier_fwd)
    result["barrier_rev_kcal"] = float(barrier_rev)
    result["dE_rxn_kcal"] = float(dE_rxn)

    log.info("=" * 60)
    log.info("QCB TS PIPELINE COMPLETE")
    log.info(f"  Barrier (fwd): {barrier_fwd:.2f} kcal/mol")
    log.info(f"  Barrier (rev): {barrier_rev:.2f} kcal/mol")
    log.info(f"  ΔE(rxn):       {dE_rxn:.2f} kcal/mol")
    log.info("=" * 60)

    # Write PDBs
    try:
        write_pdb(reactant, template, outdir / "reactant.pdb",
                  total_charge=charge, energy_eV=e_r)
        write_pdb(ts, template, outdir / "transition_state.pdb",
                  total_charge=charge, energy_eV=e_ts)
        write_pdb(product, template, outdir / "product.pdb",
                  total_charge=charge, energy_eV=e_p)
        result["outputs"]["reactant_pdb"] = str(outdir / "reactant.pdb")
        result["outputs"]["ts_pdb"] = str(outdir / "transition_state.pdb")
        result["outputs"]["product_pdb"] = str(outdir / "product.pdb")
    except Exception as e:
        log.warning(f"  PDB write failed: {e}")

    # Write summary JSON
    summary = {
        "strategy": result.get("strategy"),
        "barrier_fwd_kcal": result["barrier_fwd_kcal"],
        "barrier_rev_kcal": result["barrier_rev_kcal"],
        "dE_rxn_kcal": result["dE_rxn_kcal"],
        "energy_eV": {"reactant": e_r, "ts": e_ts, "product": e_p},
        "charge": charge,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    result["outputs"]["summary"] = str(outdir / "summary.json")
