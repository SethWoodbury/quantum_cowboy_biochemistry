"""Dispatch a `QcbConfig` (loaded from YAML) to the right ops module.

This is the keystone of the new config-driven workflow:

    cowboy-qc run myconfig.yaml

…loads, validates, and runs whatever operation is defined inside.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("quantum_engine.ops.run_config")


def _resolve_atom_indices(
    selector_name: str, selectors: dict, atoms, bt_struct
) -> list[int]:
    """Look up a named selector and return matching atom indices."""
    from quantum_engine.select import parse_constraints
    spec = selectors[selector_name]
    spec_str = spec.spec if hasattr(spec, "spec") else spec
    mask = parse_constraints(atoms, bt_struct, [spec_str])
    return np.where(mask)[0].tolist()


def _resolve_geometry_atoms(
    geom_name: str, geometry: dict, selectors: dict, atoms, bt_struct
) -> list[int]:
    """Resolve a GeometrySpec's atom-list to flat indices.

    For distance/angle/dihedral with single-atom selectors, returns one index per
    selector. For distance_diff (s = d(P,LG) − d(P,nuc)), returns [P, nuc, LG] order.
    """
    geom = geometry[geom_name]
    atom_indices = []
    for ref in geom.atoms:
        idxs = _resolve_atom_indices(ref, selectors, atoms, bt_struct)
        if len(idxs) != 1:
            raise ValueError(
                f"Geometry '{geom_name}' uses selector '{ref}' which matches "
                f"{len(idxs)} atoms; geometry CVs need single-atom selectors."
            )
        atom_indices.append(idxs[0])
    return atom_indices


def run(config_path: str | Path, override: dict | None = None) -> dict:
    """Load a YAML config and execute its operation.

    Args:
        config_path: path to a qcb YAML config
        override: optional dict to merge after loading (for CLI overrides)

    Returns: result dict from the called op (status, energy_eV, outputs, ...)
    """
    from quantum_engine.config.schema import load_config
    from quantum_engine.io import load_structure, parse_constraints, build_fix_atoms
    from quantum_engine.calc import make_calc, make_calc_fn

    cfg = load_config(config_path)
    if override:
        # Apply override at the top of cfg (very simple: just for tweaking common fields)
        for k, v in override.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    if cfg.structure is None:
        raise ValueError(
            f"Config '{config_path}' has no `structure:` block — it appears to "
            "be a defaults-only file. Use it via `defaults: [thispath.yaml]` "
            "in another config."
        )
    if cfg.operation is None:
        raise ValueError(f"Config '{config_path}' has no `operation:` block.")

    # ── Resolve structure file (relative to config file unless absolute) ──
    config_path = Path(config_path).resolve()
    struct_path = Path(cfg.structure.path)
    if not struct_path.is_absolute():
        struct_path = (config_path.parent / struct_path).resolve()
    if not struct_path.is_file():
        # Try a few likely fallback locations
        candidates = [
            Path.cwd() / cfg.structure.path,
            config_path.parent / "structures" / cfg.structure.path,
            Path("/net/scratch/woodbuse/metad") / cfg.structure.path,
        ]
        struct_path = next((c for c in candidates if c.is_file()), struct_path)
    log.info(f"Loading structure: {struct_path}")
    atoms, bt_struct, charge_hint = load_structure(struct_path)

    charge = cfg.structure.charge
    if charge_hint is not None and charge_hint != charge:
        log.warning(f"PDB REMARK charge ({charge_hint}) differs from config ({charge}); using config")

    # ── Calculator ──
    cs = cfg.calculator
    log.info(f"Calculator: {cs.model} (head={cs.head}, dtype={cs.dtype}, device={cs.device})")
    atoms.info["charge"] = charge

    # ── Constraints ──
    constraint = None
    if cfg.constraints:
        # Build a unified mask over all fix_atoms / harmonic_init constraints
        # Other constraints (harmonic_restraint, bond_difference_cv) are applied
        # as ASE constraint objects directly.
        fix_mask = np.zeros(len(atoms), dtype=bool)
        ase_constraints = []
        for c in cfg.constraints:
            if c.kind == "fix_atoms":
                spec = cfg.selectors[c.selector]
                spec_str = spec.spec if hasattr(spec, "spec") else spec
                mask = parse_constraints(atoms, bt_struct, [spec_str])
                fix_mask |= mask
            elif c.kind == "harmonic_init":
                # Per-atom soft anchor at initial positions (matches enz-ts spring_init).
                # Each selected atom gets a Hookean restraint to a virtual fixed point
                # at its INITIAL position, with stiffness c.k (eV/Å²) and force cap c.fmax.
                # This lets atoms wiggle locally but prevents large drifts during MD/opt.
                from ase.constraints import Hookean
                spec = cfg.selectors[c.selector]
                spec_str = spec.spec if hasattr(spec, "spec") else spec
                mask = parse_constraints(atoms, bt_struct, [spec_str])
                init_pos = atoms.get_positions()
                n_anchored = 0
                for i in np.where(mask)[0]:
                    # Hookean(a1=int, a2=Sequence_of_3_floats, rt=0, k=...) anchors
                    # atom a1 to a fixed 3D point (a2). rt = equilibrium distance from
                    # the anchor (use 0 for "stay where you started").
                    # Note: ASE's Hookean anchor uses (A·r + B = 0) plane form when
                    # a2 is a 4-tuple, or 3D point when 3-tuple. We use 3-tuple.
                    anchor = tuple(float(x) for x in init_pos[int(i)])
                    ase_constraints.append(Hookean(a1=int(i), a2=anchor, rt=0.0, k=c.k))
                    n_anchored += 1
                log.info(f"  harmonic_init '{c.name}': anchored {n_anchored} atoms "
                         f"with k={c.k} eV/Å² to initial positions")
            elif c.kind == "harmonic_restraint":
                # Build an ASE constraint that restrains a geometry
                idx = _resolve_geometry_atoms(c.geom, cfg.geometry, cfg.selectors, atoms, bt_struct)
                geom = cfg.geometry[c.geom]
                if geom.kind == "distance":
                    from ase.constraints import Hookean
                    ase_constraints.append(Hookean(a1=idx[0], a2=idx[1], rt=c.r0, k=c.k))
                else:
                    log.warning(f"  harmonic_restraint on geom kind={geom.kind} not yet implemented; skipping")
            elif c.kind == "bond_difference_cv":
                from quantum_engine.mlff.cv_spring import BondDifferenceCVSpring
                idx = [_resolve_atom_indices(s, cfg.selectors, atoms, bt_struct)[0] for s in c.atoms]
                ase_constraints.append(BondDifferenceCVSpring(
                    p_idx=idx[0], nuc_idx=idx[1], lg_idx=idx[2],
                    k=c.k, s_target=c.r0, fmax=c.fmax,
                ))

        fa = build_fix_atoms(fix_mask)
        if fa is not None:
            ase_constraints.insert(0, fa)
        constraint = ase_constraints if ase_constraints else None

    # ── Dispatch operation ──
    op = cfg.operation
    outdir = Path(f"qcb-{cfg.name or op.kind}-out")
    outdir.mkdir(parents=True, exist_ok=True)

    if op.kind == "sp":
        from quantum_engine.ops import sp
        atoms.calc = make_calc(cs.model, head=cs.head, device=cs.device,
                               default_dtype=cs.dtype, charge=charge)
        return sp.run(atoms, atoms.calc, outdir, constraint)

    elif op.kind == "opt":
        from quantum_engine.ops import opt
        atoms.calc = make_calc(cs.model, head=cs.head, device=cs.device,
                               default_dtype=cs.dtype, charge=charge)
        return opt.run(atoms, atoms.calc, outdir, constraint,
                       optimizer=op.optimizer, fmax=op.fmax, max_steps=op.max_steps)

    elif op.kind == "md":
        from quantum_engine.ops import md
        atoms.calc = make_calc(cs.model, head=cs.head, device=cs.device,
                               default_dtype=cs.dtype, charge=charge)
        return md.run(atoms, atoms.calc, outdir, constraint,
                      ensemble=op.ensemble, timestep_fs=op.timestep_fs,
                      total_time_ps=op.total_time_ps, temperature_K=op.temperature_K,
                      friction_per_ps=op.friction_per_ps)

    elif op.kind == "freq":
        from quantum_engine.ops import freq
        atoms.calc = make_calc(cs.model, head=cs.head, device=cs.device,
                               default_dtype=cs.dtype, charge=charge)
        indices = None
        if op.indices_selector:
            indices = _resolve_atom_indices(op.indices_selector, cfg.selectors, atoms, bt_struct)
        return freq.run(atoms, atoms.calc, outdir, constraint,
                        indices=indices, delta=op.delta, method=op.method,
                        temperature_K=op.temperature_K)

    elif op.kind == "scan":
        from quantum_engine.ops import scan
        atoms.calc = make_calc(cs.model, head=cs.head, device=cs.device,
                               default_dtype=cs.dtype, charge=charge)
        idx = _resolve_geometry_atoms(op.geom, cfg.geometry, cfg.selectors, atoms, bt_struct)
        return scan.run(atoms, atoms.calc, outdir, constraint,
                        coord_type=cfg.geometry[op.geom].kind, atom_indices=idx,
                        start=op.start, end=op.end, n_steps=op.n_steps,
                        relax_other=op.relax_other)

    elif op.kind == "irc":
        from quantum_engine.ops import irc
        atoms.calc = make_calc(cs.model, head=cs.head, device=cs.device,
                               default_dtype=cs.dtype, charge=charge)
        return irc.run(atoms, atoms.calc, outdir, constraint,
                       refine_ts=op.refine_ts,
                       saddle_fmax=op.saddle_fmax,
                       irc_step=op.irc_step, irc_fmax=op.irc_fmax)

    elif op.kind == "neb":
        from quantum_engine.ops import neb
        calc_fn = make_calc_fn(model=cs.model, head=cs.head, device=cs.device, charge=charge)
        r_atoms, r_bt, _ = load_structure(op.reactant)
        p_atoms, p_bt, _ = load_structure(op.product)
        r_atoms.info["charge"] = charge
        p_atoms.info["charge"] = charge
        r_atoms.calc = calc_fn()
        p_atoms.calc = calc_fn()
        return neb.run(r_atoms, p_atoms, calc_fn, outdir, constraint,
                       n_images=op.n_images, k_spring=op.k_spring,
                       interpolation_method=op.interpolation, charge=charge)

    elif op.kind == "ts":
        from quantum_engine.ops import ts
        calc_fn = make_calc_fn(model=cs.model, head=cs.head, device=cs.device, charge=charge)
        # Resolve cv_atoms if provided (selector names → atom indices)
        p_idx = nuc_idx = lg_idx = None
        if op.cv_atoms and len(op.cv_atoms) >= 3:
            p_idx = _resolve_atom_indices(op.cv_atoms[0], cfg.selectors, atoms, bt_struct)[0]
            nuc_idx = _resolve_atom_indices(op.cv_atoms[1], cfg.selectors, atoms, bt_struct)[0]
            lg_idx = _resolve_atom_indices(op.cv_atoms[2], cfg.selectors, atoms, bt_struct)[0]
        return ts.run(atoms, calc_fn, outdir,
                      strategy=op.strategy, charge=charge, constraint=constraint,
                      n_images=op.n_images, interpolation=op.interpolation,
                      cv_s_reactant=op.cv_s_reactant, cv_s_product=op.cv_s_product,
                      p_idx=p_idx, nuc_idx=nuc_idx, lg_idx=lg_idx,
                      template=bt_struct)

    elif op.kind == "mtd":
        atoms.calc = make_calc(cs.model, head=cs.head, device=cs.device,
                               default_dtype=cs.dtype, charge=charge)
        if op.backend == "plumed":
            from quantum_engine.mlff.plumed_runner import run_bond_difference_mtd
            # The CV from config could be any geometry, but for now we support
            # the bond-difference-CV bond_difference_cv natively. Future: emit
            # PLUMED for arbitrary GeometrySpec.
            geom = cfg.geometry[op.cv]
            if geom.kind != "distance_diff":
                log.warning(f"  PLUMED backend currently only wired for distance_diff CVs; "
                            f"got '{geom.kind}'. Falling back to pure Python.")
                op_backend = "pure_python"
            else:
                idx = _resolve_geometry_atoms(op.cv, cfg.geometry, cfg.selectors, atoms, bt_struct)
                # distance_diff atoms come back as flat list — interpret as
                # [P, LG, P, nuc] from the geometry spec? Simpler: assume
                # geometry.atoms is [P, nuc, LG] for distance_diff
                # (which matches what the schema implies)
                p_idx, nuc_idx, lg_idx = idx[0], idx[1], idx[2]
                return run_bond_difference_mtd(
                    atoms, atoms.calc,
                    p_idx=p_idx, nuc_idx=nuc_idx, lg_idx=lg_idx,
                    outdir=outdir, method=op.variant,
                    sigma_A=op.sigma_A, pace_steps=op.pace_steps,
                    bias_factor=op.bias_factor,
                    total_time_ps=op.total_time_ps,
                    temperature_K=op.temperature_K,
                    friction_per_ps=op.friction_per_ps,
                    constraint=constraint,
                )

        # Pure-Python backend (fallback)
        from quantum_engine.ops import mtd
        idx = _resolve_geometry_atoms(op.cv, cfg.geometry, cfg.selectors, atoms, bt_struct)
        return mtd.run(atoms, atoms.calc, outdir, constraint,
                       p_idx=idx[0], nuc_idx=idx[1], lg_idx=idx[2],
                       total_time_ps=op.total_time_ps, temperature_K=op.temperature_K,
                       variant=op.variant)

    elif op.kind == "umbrella":
        # Multi-window: run sequentially or as separate jobs
        # Here: minimal sequential implementation
        from quantum_engine.mlff.plumed_runner import run_umbrella_window
        atoms.calc = make_calc(cs.model, head=cs.head, device=cs.device,
                               default_dtype=cs.dtype, charge=charge)
        idx = _resolve_geometry_atoms(op.cv, cfg.geometry, cfg.selectors, atoms, bt_struct)
        if cfg.geometry[op.cv].kind != "distance":
            raise ValueError("umbrella op currently supports only `distance` geometries")
        results = []
        for i, c in enumerate(op.centers):
            wd = outdir / f"window_{i:03d}"
            log.info(f"  umbrella window {i+1}/{len(op.centers)}: center={c}")
            r = run_umbrella_window(
                atoms.copy(), atoms.calc, cv_atoms=(idx[0], idx[1]),
                at_A=c, kappa_kJ_per_A2=op.k,
                outdir=wd,
                total_time_ps=op.total_time_ps_per_window,
                temperature_K=op.temperature_K,
                constraint=constraint,
            )
            results.append(r)
        return {
            "status": "completed",
            "n_windows": len(results),
            "outputs": {"windows": [str(outdir / f"window_{i:03d}") for i in range(len(results))]},
        }

    raise ValueError(f"Unknown operation kind: {op.kind}")
