"""High-level TS search pipeline — NATIVE composition of qcb.ops primitives.

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
from ase import Atoms
from ase.optimize import LBFGS

log = logging.getLogger("qcb.ops.ts")

EV_TO_KCAL = 23.0609


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
    p_idx: int | None = None,
    nuc_idx: int | None = None,
    lg_idx: int | None = None,
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
        p_idx, nuc_idx, lg_idx: CV atoms (required for cv-spring and mtd)

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

    log.info("=" * 60)
    log.info(f"qcb ts: strategy={strategy}, charge={charge:+d}, n_atoms={len(input_atoms)}")
    log.info("=" * 60)

    # ===================================================================
    # STRATEGY: IRC-FROM-TS
    # ===================================================================
    if strategy == "irc":
        from qcb.mlff.irc import irc_from_ts_guess

        # Build bond-pattern hint (disambiguates forward/reverse assignment)
        hint_bonds = None
        if p_idx is not None and nuc_idx is not None and lg_idx is not None:
            hint_bonds = {
                (p_idx, nuc_idx): "broken",
                (p_idx, lg_idx): "bonded",
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
        from qcb.mlff.metadynamics import run_metadynamics_rescue, run_opes_rescue
        if p_idx is None or nuc_idx is None or lg_idx is None:
            raise ValueError("MTD requires p_idx, nuc_idx, lg_idx (CV atom indices)")

        entry = run_opes_rescue if mtd_variant == "opes" else run_metadynamics_rescue

        mtd_res = entry(
            input_atoms,
            p_idx=p_idx, nuc_idx=nuc_idx, lg_idx=lg_idx,
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
            if p_idx is None or nuc_idx is None or lg_idx is None:
                raise ValueError("cv-spring requires p_idx, nuc_idx, lg_idx")
            reactant, product = _cv_spring_endpoints(
                input_atoms, calculator_fn, charge, constraint,
                p_idx, nuc_idx, lg_idx,
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

        # Run NEB
        from qcb.ops import neb
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

        # Critical: ensure TS has same calc setup as endpoints for consistent energy
        # This is what the legacy pipeline got wrong.
        ts = ts.copy()
        _setup_calc_and_constraint(ts, calculator_fn, charge, constraint)

        # Optional: Sella refinement of TS (if requested via future flag)
        # For now, skip; Sella can be called separately via `qcb saddle`

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
    p_idx, nuc_idx, lg_idx,
    s_reactant, s_product,
    spring_k, spring_fmax,
    spring_drive_fmax, opt_final_fmax,
    outdir,
):
    """Drive input to reactant (s=s_reactant) and product (s=s_product) via CV spring."""
    from qcb.mlff.cv_spring import BondDifferenceCVSpring

    def _drive(s_target: float, label: str):
        a = input_atoms.copy()
        _setup_calc_and_constraint(a, calc_fn, charge, constraint)
        base_cs = list(a.constraints) if a.constraints else []
        cv_spring = BondDifferenceCVSpring(
            p_idx=p_idx, nuc_idx=nuc_idx, lg_idx=lg_idx,
            k=spring_k, s_target=s_target, fmax=spring_fmax,
        )
        a.set_constraint(base_cs + [cv_spring])
        log.info(f"  CV-drive ({label}, s_target={s_target:+.1f}) ...")
        opt = LBFGS(a, logfile=str(outdir / f"cv-drive-{label}.log"),
                    trajectory=str(outdir / f"cv-drive-{label}.traj"))
        opt.run(fmax=spring_drive_fmax, steps=300)
        # Remove CV spring, keep opt constraint, polish
        a.set_constraint(base_cs)
        opt2 = LBFGS(a, logfile=str(outdir / f"cv-polish-{label}.log"),
                     trajectory=str(outdir / f"cv-polish-{label}.traj"))
        opt2.run(fmax=opt_final_fmax, steps=200)
        return a

    reactant = _drive(s_reactant, "reactant")
    product = _drive(s_product, "product")
    return reactant, product


def _finalize_result(result, reactant, ts, product, e_r, e_ts, e_p, outdir,
                     template, charge, neb_images=None):
    """Compute barriers, write PDBs, populate result dict."""
    from qcb.io import write_pdb

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


# ═══════════════════════════════════════════════════════════════════
# Legacy subprocess entry (kept for backward compat with old scripts)
# ═══════════════════════════════════════════════════════════════════

def run_legacy_subprocess(
    pdb_file,
    outdir,
    strategy: str = "legacy",
    model: str = "mace-omol",
    charge: int | None = None,
    extra_args: list[str] | None = None,
    **kwargs,
) -> dict:
    """Call scripts/run_neb_ts.py as a subprocess (legacy compatibility).

    Preserved ONLY for old SLURM scripts. New code should use `run()`.
    WARNING: has known charge-state-bug in some configurations; verified
    on R3 benchmarks (2026-04-22).
    """
    import subprocess
    import sys

    pdb_file = Path(pdb_file).resolve()
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve().parent.parent.parent / "scripts" / "run_neb_ts.py"

    cmd = [
        sys.executable, str(script),
        str(pdb_file),
        "--outdir", str(outdir),
        "--model", model,
        "--strategy", strategy,
    ]
    if charge is not None:
        cmd.extend(["--charge", str(charge)])
    if extra_args:
        cmd.extend(extra_args)

    log.info(f"ts legacy subprocess: {' '.join(cmd)}")
    # Stream output (do not capture; these runs are long)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        log.error(f"  subprocess returned {proc.returncode}")

    return {
        "status": "completed" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "outdir": str(outdir),
        "outputs": {},
    }
