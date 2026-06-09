"""Top-level CLI for quantum_cowboy_biochemistry.

Usage:
    qcb <op> <input> [options]

Ops:
    sp         single-point energy
    opt        energy minimization
    md         molecular dynamics
    freq       vibrational frequencies
    scan       coordinate scan (bond/angle/dihedral)
    saddle     multi-backend saddle-point search (Sella / Dimer / pysisyphus / auto)
    refine-ts  post-NEB / TS-guess validation pipeline (saddle → freq → checks)
    pipeline   chain ops via YAML config (CREST → NEB → refine-ts → freq, etc.)
    irc        IRC descent from a TS (guess)
    neb        nudged elastic band (requires reactant + product)
    mtd        well-tempered metadynamics
    ts         full TS pipeline (composes endpoint generation + NEB + Sella + IRC)

    --- v2 extended TS pipeline ops (composable, modular) ---
    endpoint-release   release reactive-bond constraints from scan endpoints
    scan2d             diagnostic 2-D relaxed scan around a TS guess
    microstates        protonation/tautomer/water microstate enumeration
    validate-ts        tiered Hessian validation (A=reactive, B=expanded, C=Lanczos)
    verify-irc-like    ±imag-mode displacement + relax test
    ts-pipeline-v2     YAML-driven orchestrator chaining all v2 stages

All ops support common flags:
    --model        MACE model alias (default: mace-omol)
    --charge       system net charge (default: inferred from PDB REMARK)
    --head         head for multi-head models (e.g., mace-mh --head omol)
    --device       "cuda" (default) or "cpu"
    --outdir       output directory
    --fix          constraint spec(s): "residue X", "chain B", "atoms CA", "resid 5", "range 0 50", "all", "none"
    --free         remove atoms from fix mask (same syntax)
    --fix-preset   named preset: ca-only, backbone, backbone-water, none
    --log-level    logging level (INFO default)

Examples:
    cowboy-qc opt input.pdb --model mace-omol --fmax 0.01
    cowboy-qc md input.pdb --time 10 --temp 300 --friction 1.0
    cowboy-qc scan input.pdb --coord "bond" --indices "5 12" --range "1.5 3.5 20"
    cowboy-qc saddle ts_guess.pdb
    cowboy-qc irc ts.pdb --step 0.1
    cowboy-qc neb reactant.pdb product.pdb --n-images 15
    cowboy-qc ts input.pdb --strategy irc
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np


def _common_parser_setup(parser: argparse.ArgumentParser, needs_structure: bool = True):
    """Add flags common to all ops."""
    if needs_structure:
        parser.add_argument("input", help="Input structure (PDB, XYZ, or CIF)")
    parser.add_argument("--model", default="mace-omol",
                        help="MACE model alias or path (default: mace-omol)")
    parser.add_argument("--head", default=None,
                        help="Head for multi-head models (e.g., 'omol' for mace-mh)")
    parser.add_argument("--charge", type=int, default=None,
                        help="System net charge (default: inferred from PDB REMARK)")
    parser.add_argument("--multiplicity", dest="spin", type=int, default=None,
                        help="Spin multiplicity M=2S+1 (1=singlet, 2=doublet, 3=triplet; default 1).")
    parser.add_argument("--spin", dest="_spin_S", type=int, default=None,
                        help="Spin quantum number S; converted to multiplicity 2S+1. Prefer --multiplicity.")
    parser.add_argument("--charge-ledger", default=None,
                        help="Path to YAML/JSON charge ledger; sets total + spin "
                             "and propagates to output PDB REMARKs. Validated "
                             "against --charge if both supplied.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--outdir", default=None, help="Output directory")
    parser.add_argument("--fix", nargs="+", default=None,
                        help="Constraint spec(s) to fix (see module docs for grammar)")
    parser.add_argument("--free", nargs="+", default=None,
                        help="Constraint spec(s) to exclude from --fix")
    parser.add_argument("--fix-preset", default=None,
                        choices=["ca-only", "backbone", "backbone-water", "none"],
                        help="Named constraint preset")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def _setup_atoms_and_calc(args):
    """Shared setup: load structure, build calculator, build constraint."""
    from quantum_engine.calc import make_calc
    from quantum_engine.io import load_structure, parse_constraints, build_fix_atoms
    from quantum_engine.ops.charge_ledger import (
        append_remarks_to_pdb, inject_into_atoms, resolve_charge_and_spin,
    )
    from quantum_engine.select import preset_to_specs

    atoms, bt_struct, charge_hint = load_structure(args.input)
    # Warn on charge conflict between CLI flag and PDB REMARK
    if args.charge is not None and charge_hint is not None and args.charge != charge_hint:
        logging.getLogger("quantum_engine.cli").warning(
            f"--charge {args.charge} disagrees with PDB REMARK ({charge_hint}); using CLI value"
        )
    # Charge ledger (optional) takes precedence; CLI flags fall through.
    ledger_path = getattr(args, "charge_ledger", None)
    cli_spin = getattr(args, "spin", None)
    charge, spin, ledger = resolve_charge_and_spin(
        ledger_path=ledger_path,
        cli_charge=args.charge,
        cli_spin=cli_spin,
        pdb_charge_hint=charge_hint,
    )

    # Detect ligand for preset exclusions (first HETATM non-water, non-metal)
    ligand_name = None
    if bt_struct is not None:
        from quantum_engine.select import STANDARD_EXCLUDED_RES
        for rn in bt_struct.res_name:
            if rn not in STANDARD_EXCLUDED_RES and rn not in ("ALA","ARG","ASN","ASP","CYS",
                "GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP",
                "TYR","VAL","KCX"):
                ligand_name = str(rn)
                break

    # spin is forwarded to make_calc as well — ORB / UMA / AIMNet2 need it at
    # construction time, MACE ignores it.
    calc = make_calc(model=args.model, head=args.head, device=args.device,
                     charge=charge, spin=spin)
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    if ledger is not None:
        inject_into_atoms(atoms, ledger)
    atoms.calc = calc

    # Constraints
    fix_specs = list(args.fix or [])
    free_specs = list(args.free or [])
    excluded = set()
    if args.fix_preset:
        preset_specs, preset_excluded = preset_to_specs(args.fix_preset, ligand_name)
        fix_specs = preset_specs + fix_specs
        excluded |= preset_excluded

    if fix_specs or free_specs:
        import numpy as np
        fix_mask = parse_constraints(atoms, bt_struct, fix_specs, excluded) if fix_specs \
                   else np.zeros(len(atoms), dtype=bool)
        if free_specs:
            free_mask = parse_constraints(atoms, bt_struct, free_specs, set())
            fix_mask &= ~free_mask
        constraint = build_fix_atoms(fix_mask)
    else:
        constraint = None

    outdir = Path(args.outdir) if args.outdir else Path(f"qcb-{args.op}-out")

    return atoms, bt_struct, calc, constraint, charge, outdir, ligand_name


def _cmd_sp(args):
    from quantum_engine.ops import sp
    atoms, _, calc, constraint, _, outdir, _ = _setup_atoms_and_calc(args)
    return sp.run(atoms, calc, outdir, constraint)


def _cmd_opt(args):
    from quantum_engine.ops import opt
    atoms, bt, calc, constraint, charge, outdir, _ = _setup_atoms_and_calc(args)
    # Per-bond constraints (--fix-bond / --restrain-bond) go ON TOP of the
    # FixAtoms scaffold built from --fix/--fix-preset.
    bond_specs = getattr(args, "fix_bond", None) or []
    restrain_specs = getattr(args, "restrain_bond", None) or []
    if bond_specs or restrain_specs:
        from quantum_engine.ops.bond_constraints import build_bond_constraints
        bond_cons = build_bond_constraints(atoms, bond_specs, restrain_specs)
        if constraint is None:
            constraint = bond_cons
        else:
            base = constraint if isinstance(constraint, list) else [constraint]
            constraint = base + bond_cons
    res = opt.run(atoms, calc, outdir, constraint,
                  optimizer=args.optimizer, fmax=args.fmax, max_steps=args.max_steps)
    # Optionally write output PDB
    if args.output_pdb:
        from quantum_engine.io import write_pdb
        write_pdb(res["atoms"], bt, args.output_pdb,
                  total_charge=charge, energy_eV=res["energy_eV"])
    return res


def _cmd_md(args):
    from quantum_engine.ops import md
    atoms, _, calc, constraint, _, outdir, _ = _setup_atoms_and_calc(args)
    return md.run(atoms, calc, outdir, constraint,
                  ensemble=args.ensemble, timestep_fs=args.timestep,
                  total_time_ps=args.time, temperature_K=args.temp,
                  friction_per_ps=args.friction, dump_every_fs=args.dump_every,
                  anneal_peak_K=args.anneal_peak, seed=args.seed)


def _cmd_freq(args):
    from quantum_engine.ops import freq
    atoms, _, calc, constraint, _, outdir, _ = _setup_atoms_and_calc(args)
    indices = None
    if args.indices:
        indices = [int(x) for x in args.indices]
    return freq.run(atoms, calc, outdir, constraint,
                    indices=indices, delta=args.delta,
                    method=args.method, temperature_K=args.temp)


def _cmd_scan(args):
    from quantum_engine.ops import scan
    atoms, _, calc, constraint, _, outdir, _ = _setup_atoms_and_calc(args)
    indices = [int(x) for x in args.indices]
    return scan.run(atoms, calc, outdir, constraint,
                    coord_type=args.coord, atom_indices=indices,
                    start=args.start, end=args.end, n_steps=args.n_steps,
                    relax_other=not args.no_relax, fmax=args.fmax)


def _cmd_saddle(args):
    from quantum_engine.ops import saddle
    from quantum_engine.qm.sella import DEFAULT_EIGH_DRIVERS

    atoms, _, calc, constraint, _, outdir, _ = _setup_atoms_and_calc(args)

    initial_mode = None
    if getattr(args, "initial_mode_vector", None):
        from quantum_engine.qm.dimer import load_mode_from_xyz
        initial_mode = load_mode_from_xyz(args.initial_mode_vector, n_atoms=len(atoms))

    eigh_drivers = None
    if getattr(args, "eigh_drivers", None):
        eigh_drivers = [s.strip() for s in args.eigh_drivers.split(",") if s.strip()]
    elif args.backend in ("sella", "sella-internal", "auto"):
        # Default: enable LAPACK-driver retry cascade for Sella backends.
        # Cheap (no overhead unless a LinAlgError actually fires) and turns
        # the most common eigh failure into a recoverable event.
        eigh_drivers = list(DEFAULT_EIGH_DRIVERS)

    return saddle.run(
        atoms, calc, outdir, constraint,
        fmax=args.fmax, max_steps=args.max_steps,
        backend=args.backend,
        initial_mode_vector=initial_mode,
        eigh_drivers=eigh_drivers,
    )


def _cmd_refine_ts(args):
    """`cowboy-qc refine-ts` — post-NEB / TS-guess validation pipeline."""
    from quantum_engine.calc import make_calc
    from quantum_engine.io import load_structure, parse_constraints, build_fix_atoms
    from quantum_engine.ops import refine_ts as refine_ts_op
    from quantum_engine.ops.refine_ts import RefineTSCriteria
    from quantum_engine.qm.sella import DEFAULT_EIGH_DRIVERS
    from quantum_engine.select import preset_to_specs

    outdir = Path(args.outdir) if args.outdir else Path("qcb-refine-ts-out")

    bt_template = None
    charge_hint = None
    initial_mode_vector_path = (
        getattr(args, "initial_mode_vector", None)
        or getattr(args, "neb_tangent", None)
    )
    use_neb_tangent = not getattr(args, "no_neb_tangent", False)
    atoms = None

    if args.from_neb:
        template_pdb = (
            getattr(args, "template_pdb", None)
            or getattr(args, "input", None)
        )
        if template_pdb:
            _, bt_template, charge_hint = load_structure(template_pdb)
    else:
        if not getattr(args, "input", None):
            raise SystemExit("refine-ts: must pass either --from-neb or a TS structure")
        atoms, bt_template, charge_hint = load_structure(args.input)

    from quantum_engine.ops.charge_ledger import resolve_charge_and_spin
    charge, spin, ledger = resolve_charge_and_spin(
        ledger_path=getattr(args, "charge_ledger", None),
        cli_charge=args.charge,
        cli_spin=getattr(args, "spin", None),
        pdb_charge_hint=charge_hint,
    )

    calc = make_calc(model=args.model, head=args.head,
                     device=args.device, charge=charge, spin=spin)

    constraint = None
    fix_specs = list(args.fix or [])
    free_specs = list(args.free or [])
    if args.fix_preset:
        preset_specs, _ = preset_to_specs(args.fix_preset, None)
        fix_specs = preset_specs + fix_specs
    if (fix_specs or free_specs) and atoms is not None:
        fix_mask = parse_constraints(atoms, bt_template, fix_specs, set()) if fix_specs \
                   else np.zeros(len(atoms), dtype=bool)
        if free_specs:
            free_mask = parse_constraints(atoms, bt_template, free_specs, set())
            fix_mask &= ~free_mask
        constraint = build_fix_atoms(fix_mask)

    reactive_atoms = []
    for spec in args.reactive_atoms:
        for piece in str(spec).split(","):
            piece = piece.strip()
            if piece:
                reactive_atoms.append(piece)
    if not reactive_atoms:
        raise SystemExit(
            "refine-ts: --reactive-atoms is required (1-based PDB serials by "
            "default; '0:N' for 0-based ASE indices; 'RES:ID:NAME' if a PDB "
            "template is loaded)"
        )

    freq_indices = None
    if args.freq_indices:
        freq_indices = []
        for spec in args.freq_indices:
            for piece in str(spec).split(","):
                piece = piece.strip()
                if not piece:
                    continue
                if piece.startswith("0:"):
                    freq_indices.append(int(piece[2:]))
                else:
                    freq_indices.append(int(piece) - 1)

    eigh_drivers = None
    if getattr(args, "eigh_drivers", None):
        eigh_drivers = [s.strip() for s in args.eigh_drivers.split(",") if s.strip()]
    elif args.backend in ("sella", "sella-internal", "auto"):
        eigh_drivers = list(DEFAULT_EIGH_DRIVERS)

    criteria = RefineTSCriteria(
        require_converged=not args.allow_unconverged,
        n_imag_expected=args.n_imag_expected,
        imag_cm_cutoff=args.imag_cm_cutoff,
        imag_mode_overlap=args.imag_mode_overlap,
    )

    common_kwargs = dict(
        outdir=outdir,
        reactive_atoms=reactive_atoms,
        bt_template=bt_template,
        constraint=constraint,
        backend=args.backend,
        saddle_fmax=args.saddle_fmax,
        saddle_max_steps=args.saddle_max_steps,
        eigh_drivers=eigh_drivers,
        freq_indices=freq_indices,
        freq_delta=args.freq_delta,
        criteria=criteria,
        pdb_charge_hint=charge,
    )

    if args.from_neb:
        return refine_ts_op.run_from_neb(
            args.from_neb,
            calc,
            pdb_for_template=getattr(args, "template_pdb", None),
            use_neb_tangent=use_neb_tangent,
            initial_mode_vector_path=initial_mode_vector_path,
            **common_kwargs,
        )

    initial_mode_vector = None
    if initial_mode_vector_path:
        from quantum_engine.qm.dimer import load_mode_from_xyz
        initial_mode_vector = load_mode_from_xyz(
            initial_mode_vector_path, n_atoms=len(atoms)
        )
    atoms.calc = calc
    atoms.info["charge"] = charge
    return refine_ts_op.run(
        atoms,
        calculator=calc,
        initial_mode_vector=initial_mode_vector,
        **common_kwargs,
    )


def _cmd_pipeline(args):
    """`cowboy-qc pipeline --config pipe.yaml` — chain ops in subprocess sequence.

    Each step is dispatched as its own ``qcb`` invocation; the config is a
    list of ``{op: <name>, input: <path>, args: {...}}`` blocks. Generic —
    no PTE-specific keys.
    """
    if getattr(args, "print_example", False):
        print(_PIPELINE_EXAMPLE_YAML)
        return {"status": "completed"}

    import subprocess
    import yaml

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise SystemExit(f"--config {cfg_path} not found")
    cfg = yaml.safe_load(cfg_path.read_text()) or {}

    steps = cfg.get("steps") or []
    if not steps:
        raise SystemExit("Pipeline config has no 'steps:' list")

    base_outdir = Path(cfg.get("outdir", "qcb-pipeline-out"))
    base_outdir.mkdir(parents=True, exist_ok=True)
    log_ = logging.getLogger("quantum_engine.cli")

    results: list[dict] = []
    for i, step in enumerate(steps, start=1):
        op_name = step.get("op")
        if not op_name:
            raise SystemExit(f"Step {i} missing 'op:'")
        step_outdir = base_outdir / step.get("name", f"{i:02d}_{op_name}")
        step_outdir.mkdir(parents=True, exist_ok=True)

        cmd = ["cowboy-qc", op_name]
        if "input" in step:
            cmd.append(str(step["input"]))
        for k, v in step.get("args", {}).items():
            flag = "--" + k.replace("_", "-")
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
            elif isinstance(v, list):
                cmd.append(flag)
                cmd.extend(str(x) for x in v)
            else:
                cmd.append(flag)
                cmd.append(str(v))
        cmd.extend(["--outdir", str(step_outdir)])
        log_.info("[pipeline] step %d/%d (%s): %s", i, len(steps), op_name,
                  " ".join(cmd))
        rc = subprocess.call(cmd)
        results.append({"step": i, "op": op_name, "outdir": str(step_outdir),
                        "returncode": rc})
        if rc != 0 and not step.get("continue_on_failure", False):
            log_.error("[pipeline] step %d failed (rc=%d); halting", i, rc)
            return {"status": "failed", "steps": results}

    return {"status": "completed", "steps": results,
            "outdir": str(base_outdir)}


_PIPELINE_EXAMPLE_YAML = """
# Generic chain: any cowboy-qc subcommand can be a step. Configurable per-step.
# Pipeline is intentionally reaction-agnostic: pass your own --reactive-atoms.
outdir: my-pipeline-out
steps:
  - op: opt
    input: reactant.pdb
    args: {model: mace-omol, fmax: 0.05}
  - op: neb
    input: reactant_relaxed.pdb
    args: {n-images: 15}
  - op: refine-ts
    args:
      from-neb: my-pipeline-out/02_neb
      reactive-atoms: ["178,180,189"]
      backend: dimer
""".lstrip()


def _cmd_irc(args):
    from quantum_engine.ops import irc
    atoms, _, calc, constraint, _, outdir, _ = _setup_atoms_and_calc(args)
    return irc.run(atoms, calc, outdir, constraint,
                   refine_ts=not args.no_refine_ts,
                   saddle_fmax=args.saddle_fmax,
                   irc_step=args.step, irc_fmax=args.fmax,
                   irc_max_steps=args.max_steps)


def _cmd_neb(args):
    from quantum_engine.ops import neb
    from quantum_engine.ops.neb import NEBConfig
    from quantum_engine.calc import make_calc_fn
    from quantum_engine.io import load_structure, parse_constraints, build_fix_atoms
    from quantum_engine.select import preset_to_specs, STANDARD_EXCLUDED_RES

    r_atoms, r_bt, r_charge = load_structure(args.reactant)
    p_atoms, p_bt, p_charge = load_structure(args.product)

    from quantum_engine.ops.charge_ledger import (
        inject_into_atoms, resolve_charge_and_spin,
    )
    charge, spin, ledger = resolve_charge_and_spin(
        ledger_path=getattr(args, "charge_ledger", None),
        cli_charge=args.charge,
        cli_spin=getattr(args, "spin", None),
        pdb_charge_hint=r_charge,
    )

    calc_fn = make_calc_fn(model=args.model, head=args.head, device=args.device, charge=charge)
    r_atoms.calc = calc_fn()
    p_atoms.calc = calc_fn()
    r_atoms.info["charge"] = charge
    p_atoms.info["charge"] = charge
    r_atoms.info["spin"] = spin
    p_atoms.info["spin"] = spin
    if ledger is not None:
        inject_into_atoms(r_atoms, ledger)
        inject_into_atoms(p_atoms, ledger)

    # Constraints (applied to all NEB images)
    constraint = None
    fix_specs = list(args.fix or [])
    free_specs = list(args.free or [])
    if args.fix_preset:
        preset_specs, _ = preset_to_specs(args.fix_preset, None)
        fix_specs = preset_specs + fix_specs
    if fix_specs or free_specs:
        import numpy as np
        fix_mask = parse_constraints(r_atoms, r_bt, fix_specs, STANDARD_EXCLUDED_RES) if fix_specs \
                   else np.zeros(len(r_atoms), dtype=bool)
        if free_specs:
            free_mask = parse_constraints(r_atoms, r_bt, free_specs, set())
            fix_mask &= ~free_mask
        constraint = build_fix_atoms(fix_mask)

    outdir = Path(args.outdir) if args.outdir else Path("qcb-neb-out")

    # Resolve --key-bond specs into 0-based atom-index pairs.
    # Each spec is "A,B" where A/B is either a 0-based index, a 1-based PDB
    # serial with trailing 's' (e.g. "23s"), or a NAME:RESNAME selector
    # (e.g. "P:SUB:NUC:HIS"). For simplicity we only support index/serial
    # syntax here — name-based resolution is delegated to the user with
    # `cowboy-qc info` for now.
    key_bonds = _resolve_key_bonds(getattr(args, "key_bond", None) or [],
                                   r_atoms, r_bt)

    cfg = NEBConfig(
        n_images=args.n_images,
        n_images_default_policy=args.n_images_default_policy,
        k_spring=args.k_spring,
        k_spring_mode=args.k_spring_mode,
        interpolation_method=args.interpolation,
        optimizer=args.optimizer,
        max_step=args.max_step,
        fmax_noclimb=args.fmax_noclimb,
        steps_noclimb=args.steps_noclimb,
        fmax_climb=args.fmax_climb,
        steps_climb=args.steps_climb,
        ts_tol_fmax=args.ts_tol_fmax,
        ts_tol_steps=args.ts_tol_steps,
        double_ended_only=args.double_ended_only,
        ci_image_index=args.ci_image_index,
        restart=args.restart,
        auto_bisect_on_stall=args.auto_bisect_on_stall,
        auto_bisect_window=args.auto_bisect_window,
        auto_bisect_tol=args.auto_bisect_tol,
        save_trajectory=args.save_trajectory,
        trajectory_stride=args.trajectory_stride,
        parallel_images=args.parallel_images,
        key_bonds=key_bonds,
        key_bond_kink_tol_A=args.key_bond_kink_tol,
    )

    return neb.run(
        r_atoms, p_atoms, calc_fn, outdir,
        constraint=constraint,
        config=cfg,
        charge=charge,
        template_st=r_bt,
    )


def _resolve_key_bonds(specs, atoms, bt_struct):
    """Parse --key-bond specs into a list of (i, j) 0-based index pairs.

    Spec grammar (per element of the pair):
        N        — 0-based ASE index
        Ns       — 1-based PDB serial (matches biotite's atom_id)
        NAME:RESNAME[:RES_ID] — atom-name + residue (best-effort biotite query)

    Returns list[tuple[int,int]]. Raises SystemExit on unresolvable specs.
    """
    out: list[tuple[int, int]] = []
    if not specs:
        return out
    for spec in specs:
        if "," not in spec:
            raise SystemExit(f"--key-bond '{spec}' must be 'A,B' (got {spec!r})")
        a_raw, b_raw = (s.strip() for s in spec.split(",", 1))
        i = _resolve_atom_token(a_raw, atoms, bt_struct)
        j = _resolve_atom_token(b_raw, atoms, bt_struct)
        out.append((int(i), int(j)))
    return out


def _resolve_atom_token(tok: str, atoms, bt_struct) -> int:
    """Resolve a single atom token into a 0-based ASE index.

    Supports:
      - bare integer "23"           → ASE index 23
      - serial-style "23s"          → 1-based PDB serial 23 → ASE index 22
      - "NAME:RESNAME"              → first match
      - "NAME:RESNAME:RES_ID"       → exact match
    """
    if tok.endswith("s") and tok[:-1].isdigit():
        return int(tok[:-1]) - 1
    if tok.isdigit() or (tok.startswith("-") and tok[1:].isdigit()):
        return int(tok)
    parts = tok.split(":")
    if len(parts) >= 2 and bt_struct is not None:
        name = parts[0].strip()
        resname = parts[1].strip()
        res_id = int(parts[2]) if len(parts) >= 3 and parts[2].strip() else None
        try:
            mask = (bt_struct.atom_name == name) & (bt_struct.res_name == resname)
            if res_id is not None:
                mask = mask & (bt_struct.res_id == res_id)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                raise SystemExit(
                    f"--key-bond token {tok!r}: no atom matches "
                    f"name={name} resname={resname}"
                    + (f" res_id={res_id}" if res_id is not None else "")
                )
            return int(idx[0])
        except SystemExit:
            raise
        except Exception as exc:
            raise SystemExit(f"--key-bond token {tok!r} resolution failed: {exc}")
    raise SystemExit(
        f"--key-bond token {tok!r}: must be an int (ASE idx), 'Ns' "
        "(PDB serial), or 'NAME:RESNAME[:RES_ID]'"
    )


def _cmd_mtd(args):
    from quantum_engine.ops import mtd
    atoms, _, calc, constraint, _, outdir, _ = _setup_atoms_and_calc(args)
    return mtd.run(atoms, calc, outdir, constraint,
                   p_idx=args.p_idx, nuc_idx=args.nuc_idx, lg_idx=args.lg_idx,
                   total_time_ps=args.time, temperature_K=args.temp,
                   variant=args.variant)


def _cmd_gsm(args):
    from quantum_engine.ops import gsm
    from quantum_engine.calc import make_calc_fn
    from quantum_engine.io import load_structure

    r_atoms, _, r_charge = load_structure(args.reactant)
    p_atoms, _, p_charge = load_structure(args.product)
    charge = args.charge if args.charge is not None else (r_charge or 0)
    calc_fn = make_calc_fn(model=args.model, head=args.head, device=args.device, charge=charge)
    r_atoms.calc = calc_fn()
    p_atoms.calc = calc_fn()
    r_atoms.info["charge"] = charge
    p_atoms.info["charge"] = charge

    outdir = Path(args.outdir) if args.outdir else Path(f"qcb-{args.method}-out")

    return gsm.run(r_atoms, p_atoms, calc_fn, outdir, method=args.method,
                   charge=charge, n_images=args.n_images, fmax=args.fmax)


def _cmd_run(args):
    """Load a YAML config and dispatch to the right operation."""
    from quantum_engine.ops.run_config import run as run_config
    return run_config(args.config)


def _cmd_list_models(args):
    """`cowboy-qc list-models` — every MLFF alias the factory knows + which calculator
    backend will be used + whether that backend can load on this machine.

    Single source of truth: :data:`quantum_engine.site.MACE_MODELS` +
    :func:`quantum_engine.calc.factory._family_of`.
    """
    import importlib
    import os
    from quantum_engine.calc import list_models
    from quantum_engine.calc.factory import _family_of
    models = list_models()
    # one importable canary per backend — purely for the availability column
    canaries = {"mace": "mace.calculators", "orb": "orb_models",
                "aimnet": "aimnet", "uma": "fairchem.core"}
    backend_ok: dict = {}
    for fam, mod in canaries.items():
        try:
            importlib.import_module(mod)
            backend_ok[fam] = True
        except Exception:
            backend_ok[fam] = False
    print(f"{'alias':<22}  {'family':<7}  {'file?':<5}  {'pkg?':<5}  path")
    print("-" * 100)
    for k, path in sorted(models.items()):
        fam = _family_of(k)
        if not path:
            file_ok = "  -  "
        else:
            file_ok = "  ✓  " if os.path.isfile(path) else "  ✗  "
        pkg_ok = "  ✓  " if backend_ok.get(fam, False) else "  ✗  "
        if path is None and not args.missing_ok:
            continue
        print(f"{k:<22}  {fam:<7}  {file_ok:<5}  {pkg_ok:<5}  {path or '(none)'}")
    print()
    print("legend: family = calculator backend dispatched by make_calc.")
    print("        file?  = model checkpoint present on this filesystem.")
    print("        pkg?   = Python package for that backend importable here.")
    return {"status": "completed", "n_models": len(models),
            "backends_available": {k: v for k, v in backend_ok.items() if v}}


def _cmd_info(args):
    """`cowboy-qc info <file>` — quick summary of a structure file."""
    from collections import Counter
    from quantum_engine.io import load_structure
    atoms, bt_struct, charge_hint = load_structure(args.input)
    print(f"file:        {args.input}")
    print(f"format:      ASE inferred ({type(atoms).__name__})")
    print(f"n_atoms:     {len(atoms)}")
    if bt_struct is not None:
        chains = sorted(set(bt_struct.chain_id))
        print(f"n_chains:    {len(chains)}  ({chains})")
        residues = set(zip(bt_struct.chain_id, bt_struct.res_id))
        print(f"n_residues:  {len(residues)}")
        hets = sorted(set(bt_struct.res_name[bt_struct.hetero]))
        if hets:
            print(f"hetatms:     {hets}")
    el_counts = Counter(atoms.get_chemical_symbols())
    print(f"composition: {dict(el_counts.most_common())}")
    if charge_hint is not None:
        print(f"charge_hint: {charge_hint:+d}  (from REMARK)")
    return {"status": "completed"}


def _cmd_protonate(args):
    """cowboy-qc protonate — deterministic, staged protonation of a theozyme PDB/CIF.

    The protonator carries its own complete CLI; every argument after
    ``protonate`` is passed straight through to it (run ``cowboy-qc protonate -h``)."""
    from quantum_engine.prep.protonator import main as _protonator_main
    rc = _protonator_main(args.protonate_args)
    return {"status": "completed" if rc == 0 else "failed",
            "returncode": rc}


def _cmd_chemoton_explore(args):
    """`cowboy-qc chemoton-explore` — Steering-Wheel-driven Chemoton exploration.

    Loads the input PDB, optionally pulls extra defaults from a YAML config,
    then dispatches to ``quantum_engine.ops.chemoton_explore.run`` which
    delegates to ``tools/scine_bridge.py``.
    """
    from quantum_engine.ops import chemoton_explore as ce_op

    # Optional config file: overlay defaults under any explicit CLI flags.
    config: dict = {}
    if getattr(args, "config", None):
        cfg_path = Path(args.config)
        if not cfg_path.is_file():
            raise SystemExit(f"--config {cfg_path} not found")
        try:
            import yaml
            config = yaml.safe_load(cfg_path.read_text()) or {}
        except Exception as exc:
            raise SystemExit(f"Failed to load config {cfg_path}: {exc}")
        logging.getLogger("quantum_engine.cli").info(
            "loaded chemoton config from %s (keys=%s)", cfg_path, sorted(config)
        )

    # CLI flags win over config; config wins over hard-coded defaults
    def _resolved(name: str, default):
        cli_val = getattr(args, name, None)
        if cli_val not in (None, [], ""):
            return cli_val
        return config.get(name, default)

    out_dir = Path(args.out) if args.out else Path(_resolved("out_dir", "runs/chemoton"))

    return ce_op.run(
        input_pdb=Path(args.input),
        cluster_spec=_resolved("cluster_spec", "auto"),
        backend=_resolved("backend", "xtb-gfn2"),
        max_bond_modifications=int(_resolved("max_bond_modifications", 2)),
        max_depth=int(_resolved("max_depth", 2)),
        barrier_cap_kcal=float(_resolved("barrier_cap_kcal", 60.0)),
        top_n_export=int(_resolved("top_n_export", 5)),
        mongodb_uri=_resolved("mongodb_uri", "mongodb://localhost:27017/"),
        out_dir=out_dir,
        central_metal=_resolved("central_metal", "Zn"),
        cluster_charge=int(_resolved("charge", 1)),
        cluster_multiplicity=int(_resolved("spin", 1)) + 1
            if _resolved("spin", 0) == 0 and "spin" in config
            else int(_resolved("multiplicity", 1)),
        restart_file=Path(args.restart_file) if getattr(args, "restart_file", None) else None,
        validate_with=getattr(args, "validate_with", None),
        dry_run=bool(getattr(args, "dry_run", False)),
        reactive_atoms=getattr(args, "reactive_atoms", None) or config.get("reactive_atoms"),
    )


def _cmd_endpoint_release(args):
    """`cowboy-qc endpoint-release` — clean reactant/product endpoints before NEB."""
    import sys as _sys
    from pathlib import Path as _Path
    repo = _Path(__file__).resolve().parents[1]
    if str(repo) not in _sys.path:
        _sys.path.insert(0, str(repo))
    from tools.endpoint_release import release_endpoint
    from quantum_engine.ops.charge_ledger import load_ledger

    ledger = load_ledger(args.charge_ledger) if args.charge_ledger else None
    bonds = []
    for spec in args.release_bond or []:
        if "," not in spec:
            raise SystemExit(f"--release-bond {spec!r} must be 'A,B'")
        a, b = (s.strip() for s in spec.split(",", 1))
        bonds.append((a, b))
    res = release_endpoint(
        args.input, args.out,
        release_bonds=bonds,
        boundary_fix_preset=args.boundary_fix_preset,
        fix_specs=args.fix, free_specs=args.free,
        model=args.model, head=args.head, device=args.device,
        charge_ledger=ledger,
        cli_charge=args.charge, cli_spin=args.spin,
        fmax=args.fmax, max_steps=args.max_steps,
        optimizer=args.optimizer,
    )
    return {
        "status": res.status,
        "converged": res.converged,
        "fmax_final": res.fmax_final,
        "energy_eV": res.energy_eV,
        "outputs": {"output_pdb": str(res.output_pdb),
                     "summary_json": str(res.summary_json)},
        "warnings": res.warnings,
    }


def _cmd_scan2d(args):
    """`cowboy-qc scan2d` — diagnostic 2-D relaxed scan around a TS guess."""
    import sys as _sys
    from pathlib import Path as _Path
    repo = _Path(__file__).resolve().parents[1]
    if str(repo) not in _sys.path:
        _sys.path.insert(0, str(repo))
    from tools.scan2d import scan_2d_around, _parse_grid
    from quantum_engine.ops.charge_ledger import load_ledger

    ledger = load_ledger(args.charge_ledger) if args.charge_ledger else None
    bond_a = tuple(s.strip() for s in args.bond_a.split(",", 1))
    bond_b = tuple(s.strip() for s in args.bond_b.split(",", 1))
    if len(bond_a) != 2 or len(bond_b) != 2:
        raise SystemExit("--bond-a / --bond-b must be 'A,B'")
    grid = _parse_grid(args.grid)
    res = scan_2d_around(
        args.input, args.ts_guess,
        out_dir=args.outdir or "qcb-scan2d-out",
        bond_a=bond_a, bond_b=bond_b,
        grid=grid, delta_d=args.delta_d,
        delta_d_a=args.delta_d_a, delta_d_b=args.delta_d_b,
        boundary_fix_preset=args.boundary_fix_preset,
        fix_specs=args.fix, free_specs=args.free,
        model=args.model, head=args.head, device=args.device,
        charge_ledger=ledger,
        cli_charge=args.charge, cli_spin=args.spin,
        fmax=args.fmax, max_steps=args.max_steps,
        write_plot=not args.no_plot,
    )
    return {
        "status": res.status,
        "argmax": res.argmax_indices,
        "argmin": res.argmin_indices,
        "outputs": {"summary": res.summary_path,
                     "plot": res.plot_path},
        "warnings": res.warnings,
    }


def _cmd_microstates(args):
    """`cowboy-qc microstates` — generate labelled protonation/water-orientation ensemble."""
    import sys as _sys
    from pathlib import Path as _Path
    repo = _Path(__file__).resolve().parents[1]
    if str(repo) not in _sys.path:
        _sys.path.insert(0, str(repo))
    from tools.microstate_sampler import (
        sample_microstates, sample_protonation_microstates,
    )
    from quantum_engine.ops.charge_ledger import load_ledger

    ledger = load_ledger(args.charge_ledger) if args.charge_ledger else None

    if getattr(args, "auto_protonation", False) and getattr(args, "protonation_rules", None):
        raise SystemExit(
            "--auto-protonation and --protonation-rules are mutually "
            "exclusive. Pick one."
        )

    use_atom_path = (getattr(args, "auto_protonation", False)
                     or getattr(args, "protonation_rules", None))
    if use_atom_path:
        families = [f.strip().upper() for f in
                    getattr(args, "protonation_families", "HIS,ASP,GLU,LYS,CYS").split(",")
                    if f.strip()]
        mode = "auto" if args.auto_protonation else "rules"
        res = sample_protonation_microstates(
            args.input,
            args.outdir or "qcb-microstates-out",
            mode=mode,
            rules_yaml=getattr(args, "protonation_rules", None),
            families=families,
            max_microstates=getattr(args, "max_microstates", 16),
            metal_cutoff_a=getattr(args, "metal_cutoff_a", 3.5),
            pos_charge_cutoff_a=getattr(args, "pos_charge_cutoff_a", 4.5),
            nh_bond_length=getattr(args, "nh_bond_length", 1.01),
            oh_bond_length=getattr(args, "oh_bond_length", 0.96),
            sh_bond_length=getattr(args, "sh_bond_length", 1.34),
            include_consensus=not getattr(args, "no_include_consensus", False),
            seed=args.seed,
            charge_ledger=ledger,
            cli_charge=args.charge, cli_spin=args.spin,
            relax=args.relax,
            relax_max_steps=args.relax_max_steps,
            relax_fmax=args.relax_fmax,
            model=args.model, head=args.head, device=args.device,
        )
    else:
        if not args.generators:
            raise SystemExit(
                "cowboy-qc microstates: pass --generators (legacy ledger-only), "
                "--auto-protonation, or --protonation-rules path.yaml."
            )
        generators = [g.strip() for g in args.generators.split(",") if g.strip()]
        res = sample_microstates(
            args.input,
            args.outdir or "qcb-microstates-out",
            generators=generators,
            n_water_shuffle=args.n_water_shuffle,
            n_water_translate=args.n_water_translate,
            water_translate_delta_A=args.water_translate_delta,
            seed=args.seed,
            relax=args.relax,
            relax_max_steps=args.relax_max_steps,
            relax_fmax=args.relax_fmax,
            model=args.model, head=args.head, device=args.device,
            charge_ledger=ledger,
            cli_charge=args.charge, cli_spin=args.spin,
            max_variants=args.max_variants,
        )
    return {
        "status": res.status,
        "n_variants": res.n_variants,
        "outputs": {"manifest": str(res.manifest_path),
                     "out_dir": str(res.out_dir)},
        "warnings": res.warnings,
    }


def _cmd_validate_ts(args):
    """`cowboy-qc validate-ts` — tiered Hessian validation of a TS."""
    from quantum_engine.calc import make_calc
    from quantum_engine.io import load_structure
    from quantum_engine.ops import expanded_hessian as eh
    from quantum_engine.ops.expanded_hessian import TSValidationCriteria
    from quantum_engine.ops.charge_ledger import load_ledger, resolve_charge_and_spin

    atoms, bt_struct, charge_hint = load_structure(args.input)
    ledger = load_ledger(args.charge_ledger) if args.charge_ledger else None
    charge, spin, ledger = resolve_charge_and_spin(
        ledger_path=None,
        cli_charge=args.charge,
        cli_spin=args.spin,
        pdb_charge_hint=charge_hint,
    )
    if ledger is None and args.charge_ledger:
        from quantum_engine.ops.charge_ledger import load_ledger as _ll
        ledger = _ll(args.charge_ledger)
        charge = ledger.total
        spin = ledger.spin

    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    atoms.calc = make_calc(model=args.model, head=args.head,
                           device=args.device, charge=charge, spin=spin)

    # Resolve reactive atoms via shared helper from refine_ts
    from quantum_engine.ops.refine_ts import _resolve_reactive_indices
    reactive_specs = []
    for spec in args.reactive_atoms:
        for piece in str(spec).split(","):
            piece = piece.strip()
            if piece:
                reactive_specs.append(piece)
    reactive_idx = _resolve_reactive_indices(atoms, bt_struct, reactive_specs)

    criteria = TSValidationCriteria(
        n_imag_expected=args.n_imag_expected,
        imag_cm_cutoff=args.imag_cm_cutoff,
        imag_mode_min_overlap=args.imag_mode_min_overlap,
        second_imag_cm_cutoff=args.second_imag_cm_cutoff,
        require_no_second_imag=not args.allow_second_imag,
    )
    res_names = [s.strip() for s in (args.active_region_resnames or "").split(",") if s.strip()]
    return eh.run(
        atoms,
        atoms.calc,
        outdir=args.outdir or "qcb-validate-ts-out",
        reactive_indices=reactive_idx,
        bt_struct=bt_struct,
        tier=args.tier,
        active_region_spec=args.active_region,
        active_region_radius_A=args.active_region_radius,
        active_region_res_names=res_names or None,
        delta=args.delta,
        method=args.method,
        criteria=criteria,
        tier_c_n_modes=args.tier_c_n_modes,
        tier_c_max_iters=args.tier_c_max_iters,
    )


def _cmd_verify_irc_like(args):
    """`cowboy-qc verify-irc-like` — ±imag-mode displacement + relax test."""
    from quantum_engine.calc import make_calc
    from quantum_engine.io import load_structure
    from quantum_engine.ops import imag_mode_displace as imd
    from quantum_engine.ops.charge_ledger import load_ledger, resolve_charge_and_spin

    atoms, bt_struct, charge_hint = load_structure(args.input)
    ledger = load_ledger(args.charge_ledger) if args.charge_ledger else None
    charge, spin, ledger = resolve_charge_and_spin(
        ledger_path=None,
        cli_charge=args.charge,
        cli_spin=args.spin,
        pdb_charge_hint=charge_hint,
    )
    if ledger is None and args.charge_ledger:
        ledger = load_ledger(args.charge_ledger)
        charge, spin = ledger.total, ledger.spin
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    atoms.calc = make_calc(model=args.model, head=args.head,
                           device=args.device, charge=charge, spin=spin)

    return imd.run(
        atoms,
        atoms.calc,
        outdir=args.outdir or "qcb-verify-irc-out",
        imag_mode_path=args.imag_mode,
        displacement_A=args.displacement,
        fmax=args.fmax,
        max_steps=args.max_steps,
        optimizer=args.optimizer,
        basin_min_drop_eV=args.basin_min_drop_eV,
        bt_struct=bt_struct,
        charge=charge,
    )


def _cmd_crest_mace(args):
    """`cowboy-qc crest-mace` — run CREST 3 with MACE forces via the daemon wrapper.

    This is a thin shell-out to ``tools/crest_with_mace.sh``. It exists so
    that users discover the capability through ``cowboy-qc --help`` and so that
    pipelines can drive it via the standard subcommand dispatch. The actual
    daemon/socket/TOML logic lives in the bash wrapper; we don't reimplement
    it in Python.
    """
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    wrapper = repo_root / "tools" / "crest_with_mace.sh"
    if not wrapper.is_file():
        return {"status": "failed", "error": f"wrapper not found: {wrapper}"}

    cmd: list[str] = ["bash", str(wrapper), str(args.input)]
    cmd += ["--model", args.model, "--device", args.device, "--dtype", args.dtype]
    if args.head:
        cmd += ["--head", args.head]
    cmd += ["--charge", str(args.charge), "--spin", str(args.spin)]
    if args.workdir:
        cmd += ["--workdir", str(args.workdir)]
    if args.toml:
        cmd += ["--toml", str(args.toml)]
    if args.logfile:
        cmd += ["--logfile", str(args.logfile)]
    if args.crest_args:
        cmd += ["--", *args.crest_args]

    log = logging.getLogger("quantum_engine.cli.crest_mace")
    log.info("invoking: %s", " ".join(cmd))
    env = os.environ.copy()
    # Make sure the daemon uses the same Python interpreter we're running in.
    env.setdefault("QCB_PYTHON", sys.executable)
    proc = subprocess.run(cmd, env=env)
    rc = proc.returncode

    return {
        "status": "completed" if rc == 0 else "failed",
        "exit_code": rc,
        "workdir": args.workdir or "(tempdir; see wrapper output)",
    }


def _cmd_ts_pipeline_v2(args):
    """`cowboy-qc ts-pipeline-v2` — chained orchestrator for the v2 TS workflow."""
    if args.print_example:
        from quantum_engine.ops.ts_pipeline_v2 import _PIPELINE_EXAMPLE_YAML
        print(_PIPELINE_EXAMPLE_YAML)
        return {"status": "completed"}
    from quantum_engine.ops.ts_pipeline_v2 import run_pipeline
    summary = run_pipeline(
        args.config,
        base_outdir=args.outdir,
        resume_from=args.resume_from,
        dry_run=args.dry_run,
        only_stages=(args.only_stages.split(",") if args.only_stages else None),
    )
    return {
        "status": "completed" if summary.overall_pass else "failed",
        "overall_pass": summary.overall_pass,
        "n_stages": len(summary.stages),
        "outdir": summary.base_outdir,
        "outputs": {"summary": str(Path(summary.base_outdir) / "pipeline_summary.json")},
    }


def _cmd_ts(args):
    """Native cowboy-qc ts pipeline: loads structure + calc + constraint and calls ts.run()."""
    from quantum_engine.ops import ts as ts_op
    from quantum_engine.calc import make_calc_fn
    from quantum_engine.io import load_structure, parse_constraints, build_fix_atoms
    from quantum_engine.select import preset_to_specs, STANDARD_EXCLUDED_RES

    # Legacy subprocess mode (opt-in via --legacy-subprocess)
    if getattr(args, "legacy_subprocess", False):
        extra = list(args.passthrough or [])
        if args.fix_preset:
            extra.extend(["--constraint-mode", args.fix_preset])
        if args.head:
            extra.extend(["--head", args.head])
        outdir = Path(args.outdir) if args.outdir else Path("qcb-ts-out")
        return ts_op.run_legacy_subprocess(
            args.input, outdir, strategy=args.strategy,
            model=args.model, charge=args.charge, extra_args=extra,
        )

    # Native pipeline (default)
    atoms, bt_struct, charge_hint = load_structure(args.input)
    if args.charge is not None and charge_hint is not None and args.charge != charge_hint:
        logging.getLogger("quantum_engine.cli").warning(
            f"--charge {args.charge} disagrees with PDB REMARK ({charge_hint}); using CLI value"
        )
    charge = args.charge if args.charge is not None else (charge_hint or 0)

    # Detect ligand name (first non-protein residue)
    ligand_name = None
    if bt_struct is not None:
        protein_res = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU",
                       "LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","KCX"}
        for rn in bt_struct.res_name:
            rns = str(rn)
            if rns not in STANDARD_EXCLUDED_RES and rns not in protein_res:
                ligand_name = rns
                break

    # Constraints
    constraint = None
    fix_specs = list(getattr(args, "fix", []) or [])
    free_specs = list(getattr(args, "free", []) or [])
    if args.fix_preset:
        preset_specs, preset_excluded = preset_to_specs(args.fix_preset, ligand_name)
        fix_specs = preset_specs + fix_specs
        excluded = preset_excluded
    else:
        excluded = STANDARD_EXCLUDED_RES

    if fix_specs or free_specs:
        fix_mask = parse_constraints(atoms, bt_struct, fix_specs, excluded) if fix_specs \
                   else np.zeros(len(atoms), dtype=bool)
        if free_specs:
            free_mask = parse_constraints(atoms, bt_struct, free_specs, set())
            fix_mask &= ~free_mask
        constraint = build_fix_atoms(fix_mask)

    # CONVENIENCE ONLY: auto-detect CV atoms from a KNOWN ligand's reference
    # bond-breaking defs (data/ligand_bonds.py). This is a shortcut for a handful
    # of curated ligands, not a general mechanism — for any other reaction pass
    # --p-idx/--nuc-idx/--lg-idx explicitly (or a ReactionSpec). When a ligand IS
    # named, a mismatch is an explicit error, never a silent None.
    p_idx = nuc_idx = lg_idx = None
    if ligand_name and bt_struct is not None:
        _log = logging.getLogger("quantum_engine.cli")
        from quantum_engine.data import BOND_BREAKING_DEFS
        if ligand_name not in BOND_BREAKING_DEFS:
            _log.warning(
                "ligand %r is not a known convenience ligand (data/ligand_bonds.py: "
                "%s) — not auto-detecting CV atoms; pass --p-idx/--nuc-idx/--lg-idx "
                "explicitly for this reaction.", ligand_name,
                sorted(BOND_BREAKING_DEFS))
        else:
            defs = BOND_BREAKING_DEFS[ligand_name]
            nuc_name = next((d[1] for d in defs if d[3] == "attractive"), None)
            lg_name = next((d[1] for d in defs if d[3] == "repulsive"), None)
            p_name = defs[0][0]

            def _find(atom_name):
                hits = np.where((bt_struct.res_name == ligand_name) &
                                (bt_struct.atom_name == atom_name))[0]
                if len(hits) == 0:
                    raise SystemExit(
                        f"cowboy-qc ts: ligand {ligand_name!r} is known but atom "
                        f"{atom_name!r} was not found in the structure. The "
                        "convenience defs don't match this input — pass "
                        "--p-idx/--nuc-idx/--lg-idx explicitly.")
                return int(hits[0])

            p_idx, nuc_idx, lg_idx = _find(p_name), _find(nuc_name), _find(lg_name)
            _log.info("auto-detected CV atoms for ligand %s: P=%d nuc=%d lg=%d",
                      ligand_name, p_idx, nuc_idx, lg_idx)

    # CLI override of CV indices
    if getattr(args, "p_idx", None) is not None:
        p_idx = args.p_idx
    if getattr(args, "nuc_idx", None) is not None:
        nuc_idx = args.nuc_idx
    if getattr(args, "lg_idx", None) is not None:
        lg_idx = args.lg_idx

    calc_fn = make_calc_fn(model=args.model, head=args.head, device=args.device, charge=charge)
    outdir = Path(args.outdir) if args.outdir else Path("qcb-ts-out")

    return ts_op.run(
        atoms, calc_fn, outdir,
        strategy=args.strategy, charge=charge, constraint=constraint,
        n_images=args.n_images, interpolation=args.interpolation,
        cv_s_reactant=args.cv_s_reactant, cv_s_product=args.cv_s_product,
        p_idx=p_idx, nuc_idx=nuc_idx, lg_idx=lg_idx,
        mtd_time_ps=args.mtd_time_ps,
        template=bt_struct,
    )


def _cmd_ts_entry(args):
    """Reaction-agnostic TS orchestrator (ReactionSpec/RunContext → validated TS)."""
    from quantum_engine.io import load_structure
    from quantum_engine.ops import ts_entry
    from quantum_engine.reaction_spec import ReactionSpec, RunContext

    spec = ReactionSpec.from_yaml(args.reaction_spec)
    spec.validate()
    ctx = RunContext(charge=args.charge or 0, multiplicity=args.spin or 1,
                     model=args.model, head=args.head, engine=args.engine,
                     device=args.device)

    template = None
    geoms: dict[str, object] = {}
    for key, path in (("reactant", args.reactant), ("product", args.product),
                      ("ts_guess", args.ts_guess)):
        if path:
            atoms, bt, _ = load_structure(path)
            geoms[key] = atoms
            if template is None and bt is not None:
                template = bt

    outdir = Path(args.outdir) if args.outdir else Path("qcb-ts-entry-out")
    return ts_entry.run(
        spec, ctx, entry=args.entry, outdir=outdir,
        reactant=geoms.get("reactant"), product=geoms.get("product"),
        ts_guess=geoms.get("ts_guess"), template=template, rigor=args.rigor,
        path_method=args.path_method, proposer=args.proposer, refiner=args.refiner,
        saddle_backend=args.saddle_backend,
        n_images=args.n_images, validate=args.validate,
        cv_product_s=args.cv_product_s,
        execute=args.execute)   # QM-native engine: prepare-only when --no-execute


def _cmd_ts_propose(args):
    """Run ONE TS-guess proposer (e.g. react-ot) and write the guess.

    The first half of the sidecar two-step handoff: run the generative proposer in
    its sidecar to emit a guess, then feed the guess to the main container's
    ``ts-entry --entry ts-guess`` (which has the MLFF for saddle-refine + the gate).
    """
    from ase.io import write as ase_write
    from quantum_engine.io import load_structure
    from quantum_engine.ops import ts_propose

    R = load_structure(args.reactant)[0]
    P = load_structure(args.product)[0]
    outdir = Path(args.outdir) if args.outdir else Path("qcb-ts-propose-out")
    res = ts_propose.run(args.method, R, P, charge=args.charge or 0,
                         spin=args.spin or 1, outdir=outdir)
    ts = res.get("ts_guess")
    if ts is not None and args.out:
        ase_write(args.out, ts, format="extxyz")
        res.setdefault("outputs", {})["out"] = args.out
    return res


def _cmd_ts_refine(args):
    """Run ONE TS-guess refiner (e.g. aefm) and write the refined guess.

    The first half of the sidecar two-step handoff: run the refiner in its sidecar
    to polish a guess, then feed the result to ``ts-entry --entry ts-guess``.
    """
    from ase.io import write as ase_write
    from quantum_engine.io import load_structure
    from quantum_engine.ops import ts_refine

    guess = load_structure(args.ts_guess)[0]
    R = load_structure(args.reactant)[0] if args.reactant else None
    P = load_structure(args.product)[0] if args.product else None
    outdir = Path(args.outdir) if args.outdir else Path("qcb-ts-refine-out")
    res = ts_refine.run(args.method, guess, charge=args.charge or 0,
                        spin=args.spin or 1, reactant=R, product=P, outdir=outdir,
                        allow_out_of_domain=args.allow_out_of_domain)
    ts = res.get("ts_guess")
    if ts is not None and args.out:
        ase_write(args.out, ts, format="extxyz")
        res.setdefault("outputs", {})["out"] = args.out
    return res


def _cmd_monitor(args):
    """Non-constraining bond + metal-coordination report on a structure."""
    from quantum_engine.io import load_structure
    from quantum_engine.ops.bond_monitor import monitor_bonds

    atoms, _bt, _ = load_structure(args.input)
    bonds = [tuple(int(x) for x in b.split(",")) for b in (args.bond or [])]
    outdir = Path(args.outdir) if args.outdir else None
    return monitor_bonds(
        atoms, bonds=bonds, metals=("auto" if args.metals else None),
        label="monitor", outdir=outdir, write_json=outdir is not None)


def _cmd_reaction_spec(args):
    """Validate (and optionally resolve against a structure) a ReactionSpec YAML."""
    from quantum_engine.reaction_spec import ReactionSpec
    spec = ReactionSpec.from_yaml(args.spec)
    spec.validate()
    out: dict[str, object] = {
        "status": "valid",
        "forming_bonds": len(spec.forming_bonds),
        "breaking_bonds": len(spec.breaking_bonds),
        "reactive_atoms": len(spec.reactive_atoms),
        "has_cv": spec.cv is not None,
        "has_atom_map": spec.atom_map is not None,
    }
    if args.structure:
        from quantum_engine.io import load_structure
        atoms, bt, _ = load_structure(args.structure)
        r = spec.resolve(atoms, bt)
        out["resolved"] = {"forming": r.forming, "breaking": r.breaking,
                           "reactive": r.reactive,
                           "cv_bond_difference": r.cv_bond_difference}
    return out


def main(argv=None):
    # Deprecated-alias notice: the CLI was renamed `qcb` -> `cowboy-qc`. Both entry
    # points call this; warn (once) if invoked via the old `qcb` name.
    try:
        _prog = Path(sys.argv[0]).name
    except Exception:  # noqa: BLE001
        _prog = ""
    if _prog == "qcb":
        print("# note: `qcb` is the deprecated alias for `cowboy-qc` — please switch.",
              file=sys.stderr)

    # `protonate` carries a complete standalone CLI of its own. Intercept it
    # before the cowboy-qc argparse so all its flags (and its own -h) work cleanly.
    _av = list(sys.argv[1:] if argv is None else argv)
    if _av and _av[0] == "protonate":
        from quantum_engine.prep.protonator import main as _protonator_main
        return _protonator_main(_av[1:])

    parser = argparse.ArgumentParser(
        prog="cowboy-qc",
        description="quantum_engine — Cowboy Quantum Chemistry toolkit "
                    "(`cowboy-qc` is the CLI; `qcb` is a deprecated alias).",
    )
    sub = parser.add_subparsers(dest="op", required=True)

    # sp
    p_sp = sub.add_parser("sp", help="Single-point energy")
    _common_parser_setup(p_sp)

    # opt
    p_opt = sub.add_parser("opt", help="Energy minimization")
    _common_parser_setup(p_opt)
    p_opt.add_argument("--optimizer", default="lbfgs", choices=["lbfgs", "bfgs", "fire"])
    p_opt.add_argument("--fmax", type=float, default=0.05)
    p_opt.add_argument("--max-steps", type=int, default=500)
    p_opt.add_argument("--output-pdb", default=None, help="Write relaxed structure to PDB")
    p_opt.add_argument("--fix-bond", action="append", nargs="+", metavar="I J [R0]", default=[],
                       help="Hard-pin a bond (0-based ASE indices). With optional R0 (A), set the "
                            "bond to R0 first, then fix it; else fix at the current length. "
                            "Repeatable. Applied on top of --fix/--fix-preset.")
    p_opt.add_argument("--restrain-bond", action="append", nargs=4,
                       metavar=("I", "J", "K", "R0"), default=[],
                       help="Two-sided harmonic bond restraint E=0.5*K*(d-R0)^2 "
                            "(K in eV/A^2, R0 in A; 0-based ASE indices). Repeatable.")

    # md
    p_md = sub.add_parser("md", help="Molecular dynamics")
    _common_parser_setup(p_md)
    p_md.add_argument("--ensemble", default="langevin_nvt", choices=["langevin_nvt", "verlet_nve"])
    p_md.add_argument("--timestep", type=float, default=1.0, help="Timestep in fs (default: 1.0)")
    p_md.add_argument("--time", type=float, default=10.0, help="Total time in ps (default: 10)")
    p_md.add_argument("--temp", type=float, default=300.0, help="Temperature in K (default: 300)")
    p_md.add_argument("--friction", type=float, default=1.0, help="Langevin friction in ps⁻¹")
    p_md.add_argument("--dump-every", type=float, default=10.0, help="Dump interval in fs")
    p_md.add_argument("--anneal-peak", type=float, default=None, help="Peak T (K) for annealing")
    p_md.add_argument("--seed", type=int, default=None)

    # freq
    p_freq = sub.add_parser("freq", help="Vibrational frequencies")
    _common_parser_setup(p_freq)
    p_freq.add_argument("--indices", nargs="+", default=None,
                        help="Atom indices for partial Hessian (default: all free)")
    p_freq.add_argument("--delta", type=float, default=0.02)
    p_freq.add_argument("--method", default="central", choices=["central", "forward"])
    p_freq.add_argument("--temp", type=float, default=298.15, help="Temperature for thermo (K)")

    # scan
    p_scan = sub.add_parser("scan", help="1D coordinate scan")
    _common_parser_setup(p_scan)
    p_scan.add_argument("--coord", required=True, choices=["bond", "angle", "dihedral"])
    p_scan.add_argument("--indices", nargs="+", required=True, help="Atom indices")
    p_scan.add_argument("--start", type=float, required=True)
    p_scan.add_argument("--end", type=float, required=True)
    p_scan.add_argument("--n-steps", type=int, default=15)
    p_scan.add_argument("--no-relax", action="store_true", help="Don't relax other DOFs")
    p_scan.add_argument("--fmax", type=float, default=0.05)

    # saddle
    p_saddle = sub.add_parser(
        "saddle",
        help="Multi-backend saddle-point search (Sella / Dimer / pysisyphus / auto)",
        description=(
            "Refine a TS guess to a first-order saddle. Pick a backend with "
            "--backend; use --backend auto for the failsafe Sella → "
            "Sella-internal → Dimer cascade. The cascade catches "
            "numpy.linalg.LinAlgError from Sella's eigh on >200-atom "
            "Cartesian systems. For dimer-style backends, supply "
            "--initial-mode-vector to seed the eigenmode (e.g., a NEB "
            "tangent XYZ); without it the dimer starts from a random "
            "displacement, which is slower but works for any reaction."
        ),
    )
    _common_parser_setup(p_saddle)
    p_saddle.add_argument("--fmax", type=float, default=0.02,
                          help="Force convergence (eV/Å). Bump to 0.05 for "
                               "noisier ML potentials; 0.005 for tight "
                               "high-quality TS.")
    p_saddle.add_argument("--max-steps", type=int, default=500,
                          help="Iteration cap. Bump if 'not_converged' but "
                               "the trajectory is still descending.")
    p_saddle.add_argument(
        "--backend", default="sella",
        choices=["sella", "sella-internal", "dimer",
                 "pysisyphus-rsprfo", "pysisyphus-dimer", "auto"],
        help=("sella = native Cartesian Sella (default; fast on small "
              "systems, can LinAlgError on >200 atoms). "
              "sella-internal = Sella with TRIC internal coords (robust "
              "fallback). "
              "dimer = ASE improved-dimer (gradient-only, scales). "
              "pysisyphus-rsprfo = RS-P-RFO via pysisyphus (best when a "
              "Hessian is affordable). "
              "pysisyphus-dimer = pysisyphus dimer + PreconLBFGS "
              "(second-opinion to ASE dimer). "
              "auto = Sella → Sella-internal → Dimer cascade with "
              "LinAlgError catching (recommended for any-system robustness)."),
    )
    p_saddle.add_argument(
        "--initial-mode-vector", default=None,
        help=("Path to an XYZ file encoding the initial dimer-eigenmode "
              "direction. Either one frame of displacement vectors or two "
              "frames whose difference IS the displacement. The NEB tangent "
              "XYZ from a previous neb run is the canonical input. Ignored "
              "by Sella backends."),
    )
    p_saddle.add_argument(
        "--eigh-drivers", default=None,
        help=("Comma-separated LAPACK drivers for Sella's eigh (e.g. "
              "'evd,evr,evx,ev'). Each driver is tried in turn on "
              "LinAlgError. Default: enabled with the standard cascade for "
              "Sella backends; disabled for non-Sella backends."),
    )

    # refine-ts: post-NEB validation pipeline
    p_refts = sub.add_parser(
        "refine-ts",
        help="Validate / refine a TS guess: saddle search + partial-Hessian freq",
        description=(
            "Full TS validation pipeline: extract a TS guess (from a NEB "
            "directory, an FSM/GSM directory, or a single structure), refine "
            "it with the chosen saddle backend, run a partial-Hessian "
            "frequency calculation on the reactive subset, and check that "
            "the imaginary mode aligns with the user-supplied reaction "
            "coordinate. Outputs a refined PDB + CIF + summary.json. "
            "Generalisable: pass --reactive-atoms with the *atoms whose "
            "motion defines the reaction coordinate* (1-based PDB serials by "
            "default; '0:N' for 0-based ASE indices; 'RES:ID:NAME' if a PDB "
            "template is loaded)."
        ),
    )
    p_refts.add_argument(
        "input", nargs="?", default=None,
        help=("Optional: TS-guess structure (PDB/XYZ/CIF). If --from-neb is "
              "given, the TS is pulled from the NEB output and this "
              "argument is interpreted as a template-PDB for residue "
              "annotations.")
    )
    # Calculator/output flags shared with other ops. Cannot reuse
    # _common_parser_setup because that helper assumes a required `input`.
    p_refts.add_argument("--model", default="mace-omol",
                         help="MACE model alias or path (default: mace-omol)")
    p_refts.add_argument("--head", default=None,
                         help="Head for multi-head models (e.g. 'omol' for mace-mh)")
    p_refts.add_argument("--charge", type=int, default=None,
                         help="System net charge (default: inferred from PDB REMARK)")
    p_refts.add_argument("--multiplicity", dest="spin", type=int, default=None,
                         help="Spin multiplicity M=2S+1 (1=singlet, 2=doublet, 3=triplet; default 1).")
    p_refts.add_argument("--spin", dest="_spin_S", type=int, default=None,
                         help="Spin quantum number S; converted to multiplicity 2S+1 (prefer --multiplicity).")
    p_refts.add_argument("--charge-ledger", default=None,
                         help="Path to YAML/JSON charge ledger; sets total + spin "
                              "and propagates to output PDB REMARKs.")
    p_refts.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p_refts.add_argument("--outdir", default=None, help="Output directory")
    p_refts.add_argument("--fix", nargs="+", default=None,
                         help="Constraint spec(s) to fix")
    p_refts.add_argument("--free", nargs="+", default=None,
                         help="Constraint spec(s) to exclude from --fix")
    p_refts.add_argument("--fix-preset", default=None,
                         choices=["ca-only", "backbone", "backbone-water", "none"],
                         help="Named constraint preset")
    p_refts.add_argument("--log-level", default="INFO",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p_refts.add_argument(
        "--from-neb", default=None,
        help=("Path to a directory containing path-neb-climb.xyz (or "
              "path-neb-noclimb.xyz) from a previous `cowboy-qc neb` run. The TS "
              "guess is the highest-energy inner image; the central-"
              "difference NEB tangent at that image seeds dimer backends.")
    )
    p_refts.add_argument(
        "--template-pdb", default=None,
        help=("Optional template PDB for writing the refined TS as PDB+CIF "
              "with residue annotations. Atom order MUST match the NEB "
              "images. Defaults to the positional 'input' if given.")
    )
    p_refts.add_argument(
        "--reactive-atoms", nargs="+", required=True,
        help=("Atoms whose motion defines the reaction coordinate. Comma- "
              "or space-separated. Specs: integers = 1-based PDB serials "
              "(default — copy from any PDB viewer); '0:N' = 0-based ASE "
              "index; 'RES:ID:NAME' = lookup via PDB template (e.g. "
              "'HIS:55:NE2'). Pass ALL reactive atoms for concerted reactions "
              "(Diels-Alder, sigmatropic). Examples: '178,180,189' or "
              "'HIS:55:NE2 LIG:1:O3'.")
    )
    p_refts.add_argument(
        "--neb-tangent", default=None,
        help=("Override the NEB-derived tangent with an explicit XYZ "
              "(forwarded to --initial-mode-vector). Useful when the "
              "automatic tangent is misaligned.")
    )
    p_refts.add_argument(
        "--initial-mode-vector", default=None,
        help="Same as --neb-tangent (alias kept for parity with `cowboy-qc saddle`)."
    )
    p_refts.add_argument(
        "--no-neb-tangent", action="store_true",
        help=("Disable seeding dimer backends with the NEB tangent. The "
              "dimer will start from a small random displacement instead.")
    )
    p_refts.add_argument(
        "--backend", default="dimer",
        choices=["sella", "sella-internal", "dimer",
                 "pysisyphus-rsprfo", "pysisyphus-dimer", "auto"],
        help=("Saddle backend (see `cowboy-qc saddle --help` for details). "
              "Default: dimer (uses NEB tangent for free).")
    )
    p_refts.add_argument(
        "--eigh-drivers", default=None,
        help="Comma-separated LAPACK drivers (Sella backends only)."
    )
    p_refts.add_argument("--saddle-fmax", type=float, default=0.02,
                         help="Saddle-search convergence (eV/Å). Bump to 0.05 "
                              "for noisy ML potentials; 0.005 for tight TS.")
    p_refts.add_argument("--saddle-max-steps", type=int, default=500,
                         help="Saddle iteration cap. Bump if cascade still "
                              "descends but plateau hasn't been reached.")
    p_refts.add_argument(
        "--freq-indices", nargs="+", default=None,
        help=("Atoms to include in the partial Hessian. Same spec as "
              "--reactive-atoms (1-based PDB serials by default; '0:N' for "
              "ASE indices). Defaults to reactive atoms + their first "
              "bonded shell, which is usually sufficient.")
    )
    p_refts.add_argument(
        "--freq-delta", type=float, default=0.005,
        help=("Finite-difference step (Å) for the partial Hessian. Smaller "
              "= more accurate frequencies but more sensitive to ML noise. "
              "Bump to 0.02 for noisy potentials.")
    )
    p_refts.add_argument(
        "--imag-cm-cutoff", type=float, default=-50.0,
        help=("Pass criterion: imaginary frequency must be MORE negative "
              "than this value (cm⁻¹). Default -50 catches near-zero modes "
              "as failures.")
    )
    p_refts.add_argument(
        "--imag-mode-overlap", type=float, default=0.5,
        help=("Pass criterion: fraction of the imaginary-mode L2 amplitude "
              "that must be on the reactive atoms. Lower this to 0.3 for "
              "delocalised concerted reactions; raise to 0.8 for very "
              "localised SN2-style transfers.")
    )
    p_refts.add_argument(
        "--n-imag-expected", type=int, default=1,
        help="Expected imaginary-mode count. 1 for a normal first-order "
             "saddle; 2 for second-order saddles (rare)."
    )
    p_refts.add_argument(
        "--allow-unconverged", action="store_true",
        help="Skip the 'saddle converged' check (still records the fmax)."
    )

    # pipeline (orchestrator)
    p_pipe = sub.add_parser(
        "pipeline",
        help="Run a chained pipeline (CREST → scan/NEB → refine-ts → freq …)",
        description=(
            "Generic chain orchestrator: each step is a sub-invocation of "
            "qcb. Reaction-agnostic — supply your own per-step --reactive-"
            "atoms, etc. Use --print-example to see a template config."
        ),
    )
    p_pipe.add_argument("config", nargs="?", default=None,
                        help="Path to a YAML config (omit with --print-example).")
    p_pipe.add_argument("--print-example", action="store_true",
                        help="Print an example pipeline YAML to stdout and exit.")
    p_pipe.add_argument("--log-level", default="INFO")

    # irc
    p_irc = sub.add_parser("irc", help="IRC descent from a TS")
    _common_parser_setup(p_irc)
    p_irc.add_argument("--no-refine-ts", action="store_true", help="Skip Sella refinement")
    p_irc.add_argument("--saddle-fmax", type=float, default=0.02)
    p_irc.add_argument("--step", type=float, default=0.1, help="Displacement along imag mode")
    p_irc.add_argument("--fmax", type=float, default=0.05,
                       help="IRC endpoint LBFGS convergence (eV/Å). 0.03 is often "
                            "too tight for ML potentials; 0.05 is a robust default.")
    p_irc.add_argument("--max-steps", type=int, default=400,
                       help="LBFGS iteration cap per IRC direction.")

    # neb
    p_neb = sub.add_parser(
        "neb",
        help="NEB + CI-NEB chain-of-states saddle finder (any reaction class)",
        description=(
            "Generalizes to: SN2-at-P, SN2-at-C, hydride transfer, proton "
            "transfer, Diels-Alder, Cope rearrangement, sigmatropic shift, "
            "electrocyclic, ene reaction, any reaction with definable R and P "
            "endpoints. Pass --key-bond for each bond that must change "
            "smoothly along the path (geodesic interpolation respects them)."
        ),
    )
    p_neb.add_argument("reactant",
        help="Reactant endpoint structure (PDB, XYZ, or CIF)")
    p_neb.add_argument("product",
        help="Product endpoint structure (PDB, XYZ, or CIF). MUST share atom "
             "indexing with the reactant.")
    p_neb.add_argument("--model", default="mace-omol",
        help="MACE alias or path. Default: mace-omol.")
    p_neb.add_argument("--head", default=None,
        help="Head selector for multi-head models (e.g. 'omol' for mace-mh).")
    p_neb.add_argument("--charge", type=int, default=None,
        help="System net charge. Default: inferred from PDB REMARK or filename.")
    p_neb.add_argument("--multiplicity", dest="spin", type=int, default=None,
        help="Spin multiplicity M=2S+1 (1=singlet, 2=doublet, 3=triplet; default 1).")
    p_neb.add_argument("--spin", dest="_spin_S", type=int, default=None,
        help="Spin quantum number S; converted to multiplicity 2S+1 (prefer --multiplicity).")
    p_neb.add_argument("--charge-ledger", default=None,
        help="Path to YAML/JSON charge ledger; takes precedence over --charge.")
    p_neb.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
        help="Compute device for the calculator. Default: cuda.")
    p_neb.add_argument("--outdir", default=None,
        help="Output directory. Default: ./qcb-neb-out")
    p_neb.add_argument("--fix", nargs="+", default=None,
        help="Constraint specs (residue X / chain Y / range A B / atoms NAME / "
             "all / none). Repeatable.")
    p_neb.add_argument("--free", nargs="+", default=None,
        help="Constraint specs to subtract from --fix.")
    p_neb.add_argument("--fix-preset", default=None,
        choices=["ca-only", "backbone", "backbone-water", "none"],
        help="Named constraint preset applied to every image.")

    # Image count + policy
    p_neb.add_argument("--n-images", type=int, default=11,
        help="Total number of images including endpoints. Default: 11. "
             "Bump for floppy paths or multi-bond reactions; reduce to 7-9 "
             "for very small (<30 atom) molecules to save evals.")
    p_neb.add_argument("--n-images-default-policy",
        default="fixed", choices=["fixed", "atom-density"],
        help="'fixed': use --n-images verbatim. 'atom-density': scale "
             "n_images by sqrt(N_atoms/100) (clamped 5..51, forced odd). "
             "Useful when running across systems with varying size.")

    # Spring forces
    p_neb.add_argument("--k-spring", type=float, default=1.0,
        help="NEB spring constant (eV/Å). Default 1.0. Bump (e.g. 2.0) for "
             "longer paths to keep images from sliding into the saddle; "
             "lower (e.g. 0.5) for short, well-resolved paths.")
    p_neb.add_argument("--k-spring-mode",
        default="uniform", choices=["uniform", "distance-weighted"],
        help="'uniform' (default): same k everywhere. 'distance-weighted': "
             "stronger springs in low-curvature regions and weaker near the "
             "saddle, which lets the path densify around the climber.")

    # Interpolation
    p_neb.add_argument("--interpolation",
        default="geodesic", choices=["geodesic", "idpp", "linear"],
        help="Initial path: 'geodesic' (Zhu 2019, default, best for "
             "enzymes), 'idpp' (Smidstrup 2014, decent fallback), "
             "'linear' (only for tiny rigid molecules).")

    # Optimizer + step
    p_neb.add_argument("--optimizer",
        default="fire", choices=["fire", "lbfgs", "mdmin", "bfgs", "ode"],
        help="ASE optimizer wrapping the NEB chain. 'fire' (default) is the "
             "most robust for chain-of-states (Bitzek PRL 2006). 'lbfgs' is "
             "faster near a converged path. 'ode' is ASE's NEBOptimizer "
             "(legacy default). 'bfgs'/'mdmin' available as alternatives.")
    p_neb.add_argument("--max-step", type=float, default=0.05,
        help="Per-image displacement cap per optimizer step (Å). Default "
             "0.05 (tight). Bump to 0.10 for early stages of a noisy path; "
             "lower to 0.02 for tight-saddle convergence.")

    # Convergence
    p_neb.add_argument("--fmax-noclimb", type=float, default=0.30,
        help="Stage-1 (no-climb) NEB convergence on max image force (eV/Å). "
             "Default 0.30. Loosen to 0.40 to save iterations on rough paths.")
    p_neb.add_argument("--steps-noclimb", type=int, default=200,
        help="Cap on stage-1 iterations. Default 200.")
    p_neb.add_argument("--fmax-climb", type=float, default=0.05,
        help="Stage-2 (climbing) convergence on max image force (eV/Å). "
             "Default 0.05. NEB-CI is rarely tighter than this without "
             "instability — use --ts-tol-fmax for an extra refinement pass.")
    p_neb.add_argument("--steps-climb", type=int, default=250,
        help="Cap on stage-2 iterations. Default 250. Set to 0 with "
             "--double-ended-only to skip CI entirely.")
    p_neb.add_argument("--ts-tol-fmax", type=float, default=None,
        help="Optional tighter NEB-CI fmax for a third stage (eV/Å). "
             "Try 0.02 to push toward true-saddle convergence (still NOT "
             "a substitute for `cowboy-qc saddle` + `cowboy-qc freq`).")
    p_neb.add_argument("--ts-tol-steps", type=int, default=200,
        help="Cap on tight-TS refinement iterations. Default 200.")

    # Climbing-image options
    p_neb.add_argument("--double-ended-only", action="store_true",
        help="Skip CI; only relax the MEP. Use when you want the path "
             "without committing to a saddle (e.g. roughly mapping a "
             "barrier landscape).")
    p_neb.add_argument("--ci-image-index", type=int, default=None,
        help="Manually pick which image becomes the climber. Default: "
             "argmax(E). Warns if the choice disagrees with argmax.")

    # Output / trajectory
    p_neb.add_argument("--save-trajectory", action="store_true", default=True,
        help="Save multi-MODEL trajectory PDB + extxyz of the final path. "
             "Enabled by default.")
    p_neb.add_argument("--no-save-trajectory",
        dest="save_trajectory", action="store_false",
        help="Disable trajectory output.")
    p_neb.add_argument("--trajectory-stride", type=int, default=1,
        help="Stride for trajectory frames (1 = every image). Default 1.")

    # Restart / batching / autobisect
    p_neb.add_argument("--restart", action="store_true",
        help="Restart from a previous NEB run's last frame "
             "(`<outdir>/path-neb-climb.xyz`) — useful for tightening "
             "convergence in a follow-up call.")
    p_neb.add_argument("--auto-bisect-on-stall", action="store_true",
        help="If NEB residual plateaus, insert an extra image at the "
             "highest-energy region and continue. Advanced/optional.")
    p_neb.add_argument("--auto-bisect-window", type=int, default=8,
        help="Iterations of fmax-stall before bisecting. Default 8.")
    p_neb.add_argument("--auto-bisect-tol", type=float, default=0.005,
        help="Residual stall tolerance (eV/Å). Default 0.005.")
    p_neb.add_argument("--parallel-images", action="store_true",
        help="Hint to share calculators across images (only safe with "
             "calculators that support batched eval; MACE typically does NOT). "
             "Off by default; turn on at your own risk.")

    # Multi-bond / key-bond awareness
    p_neb.add_argument("--key-bond", action="append", default=[],
        metavar="ATOMA,ATOMB",
        help="A bond that must change smoothly along the path; "
             "ATOMA/ATOMB are 0-based indices or 1-based PDB serials with "
             "trailing 's' (e.g. '23s'). Repeatable. For SN2-like "
             "reactions: pass forming + breaking bonds. For Diels-Alder: "
             "both new C-C bonds. For sigmatropic: migrating bond + "
             "migration target.")
    p_neb.add_argument("--key-bond-kink-tol", type=float, default=0.4,
        help="Max allowed 2nd-difference of any --key-bond distance along "
             "the path before validation flags 'kinked' (Å). Default 0.4. "
             "Bump for highly distorted MEPs (e.g. metal-coordinated "
             "rearrangements where one bond stretches >0.5 Å in a step).")

    p_neb.add_argument("--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # mtd
    p_mtd = sub.add_parser("mtd", help="Metadynamics (well-tempered or OPES)")
    _common_parser_setup(p_mtd)
    p_mtd.add_argument("--p-idx", type=int, required=True, help="Central atom index")
    p_mtd.add_argument("--nuc-idx", type=int, required=True, help="Nucleophile atom index")
    p_mtd.add_argument("--lg-idx", type=int, required=True, help="Leaving-group atom index")
    p_mtd.add_argument("--time", type=float, default=100.0, help="Total MTD time in ps")
    p_mtd.add_argument("--temp", type=float, default=300.0)
    p_mtd.add_argument("--variant", default="wt", choices=["wt", "opes"],
                        help="wt = well-tempered MTD (default); opes = OPES-MetaD")

    # gsm / fsm
    p_gsm = sub.add_parser("gsm", help="Growing/Freezing String Method (via pysisyphus)")
    p_gsm.add_argument("reactant")
    p_gsm.add_argument("product")
    p_gsm.add_argument("--method", default="fsm", choices=["fsm", "gsm"],
                       help="fsm = Freezing String (faster, 88-90%% success vs CI-NEB 63-71%% per Wan 2026); "
                            "gsm = Growing String (more optimization per node)")
    p_gsm.add_argument("--model", default="mace-omol")
    p_gsm.add_argument("--head", default=None)
    p_gsm.add_argument("--charge", type=int, default=None)
    p_gsm.add_argument("--device", default="cuda")
    p_gsm.add_argument("--outdir", default=None)
    p_gsm.add_argument("--n-images", type=int, default=15)
    p_gsm.add_argument("--fmax", type=float, default=0.05)
    p_gsm.add_argument("--log-level", default="INFO")

    # run — config-driven dispatch
    p_run = sub.add_parser("run", help="Run any operation from a YAML config (recommended)")
    p_run.add_argument("config", help="Path to a qcb YAML config file")
    p_run.add_argument("--log-level", default="INFO")

    # protonate — deterministic, staged protonation of a theozyme PDB/CIF.
    # The protonator owns a complete CLI; everything after `protonate` is
    # passed straight through to it (add_help=False lets `-h` reach it too).
    p_pro = sub.add_parser(
        "protonate", add_help=False,
        help="Deterministic staged protonation of a theozyme PDB/CIF")
    p_pro.add_argument("protonate_args", nargs=argparse.REMAINDER,
                       help="arguments forwarded to the protonator "
                            "(run `cowboy-qc protonate -h` for the full list)")

    # list-models — discoverability
    p_lm = sub.add_parser("list-models",
                          help="List MACE / MLFF model aliases known to the calculator factory")
    p_lm.add_argument("--missing-ok", action="store_true",
                      help="Show entries even when the file isn't on disk")

    # info — quick structure info
    p_info = sub.add_parser("info",
                            help="Quick structural / electronic-state summary of a PDB or XYZ")
    p_info.add_argument("input", help="Path to PDB / XYZ / CIF")

    # ts
    p_ts = sub.add_parser("ts", help="Native TS pipeline (composes saddle/irc/neb/mtd)")
    p_ts.add_argument("input")
    p_ts.add_argument("--outdir", default=None)
    p_ts.add_argument("--model", default="mace-omol")
    p_ts.add_argument("--head", default=None)
    p_ts.add_argument("--charge", type=int, default=None)
    p_ts.add_argument("--device", default="cuda")
    p_ts.add_argument("--strategy", default="legacy",
                      choices=["legacy", "irc", "cv-spring", "mtd"])
    p_ts.add_argument("--fix", nargs="+", default=None,
                      help="Constraint specs (see other ops for grammar)")
    p_ts.add_argument("--free", nargs="+", default=None)
    p_ts.add_argument("--fix-preset", default=None,
                      choices=["ca-only", "backbone", "backbone-water", "none"])
    p_ts.add_argument("--n-images", type=int, default=15)
    p_ts.add_argument("--interpolation", default="geodesic",
                      choices=["geodesic", "idpp", "linear"])
    p_ts.add_argument("--cv-s-reactant", type=float, default=-2.0)
    p_ts.add_argument("--cv-s-product", type=float, default=2.5)
    p_ts.add_argument("--p-idx", type=int, default=None,
                      help="Override: P atom index for CV (auto-detected from ligand)")
    p_ts.add_argument("--nuc-idx", type=int, default=None)
    p_ts.add_argument("--lg-idx", type=int, default=None)
    p_ts.add_argument("--mtd-time-ps", type=float, default=100.0)
    p_ts.add_argument("--legacy-subprocess", action="store_true",
                      help="Use old subprocess wrapper around tools/run_neb_ts.py "
                           "(has known energy-consistency bug; use only for backward compat)")
    p_ts.add_argument("--passthrough", nargs=argparse.REMAINDER,
                      help="(legacy-subprocess only) additional flags to pass to run_neb_ts.py")
    p_ts.add_argument("--log-level", default="INFO")

    # ts-entry — the reaction-agnostic orchestrator (ReactionSpec/RunContext)
    p_tse = sub.add_parser(
        "ts-entry",
        help="Reaction-agnostic TS orchestrator: a ReactionSpec + entry point "
             "→ validated TS (path → saddle → Hessian → IRC-like), gated.")
    p_tse.add_argument("--entry", required=True,
                       choices=["ts-guess", "reactant-product", "reactant-only"])
    p_tse.add_argument("--reaction-spec", required=True,
                       help="Path to a ReactionSpec YAML (forming/breaking bonds, "
                            "cv, reactive_atoms, atom_map).")
    p_tse.add_argument("--reactant", default=None, help="Reactant geometry (PDB/XYZ/CIF)")
    p_tse.add_argument("--product", default=None, help="Product geometry")
    p_tse.add_argument("--ts-guess", default=None, help="TS-guess geometry")
    p_tse.add_argument("--model", default="mace-omol", help="Energy-function alias")
    p_tse.add_argument("--head", default=None)
    p_tse.add_argument("--charge", type=int, default=None)
    p_tse.add_argument("--multiplicity", dest="spin", type=int, default=None, help="Multiplicity 2S+1")
    p_tse.add_argument("--spin", dest="_spin_S", type=int, default=None, help="Spin quantum number S; converted to multiplicity 2S+1 (prefer --multiplicity).")
    p_tse.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p_tse.add_argument("--engine", default=None,
                       help="QM-native engine (e.g. 'orca'); default None = ASE "
                            "calculator path (MLFF / xTB).")
    p_tse.add_argument("--rigor", default="standard",
                       choices=["draft", "standard", "publication"])
    p_tse.add_argument("--path-method", default=None,
                       help="Override path method (neb/fsm/gsm-de).")
    p_tse.add_argument("--proposer", default=None,
                       help="reactant-product: use a TS-guess proposer instead of "
                            "path search (midpoint / react-ot[sidecar]).")
    p_tse.add_argument("--refiner", default=None,
                       help="Optional ML refinement of the TS guess before saddle "
                            "(identity / aefm[sidecar]); non-critical, falls back "
                            "to the un-refined guess. Compose with --proposer.")
    p_tse.add_argument("--saddle-backend", default=None,
                       help="Override saddle backend (sella/dimer/auto/...).")
    p_tse.add_argument("--n-images", type=int, default=None)
    p_tse.add_argument("--validate", action=argparse.BooleanOptionalAction,
                       default=None, help="Force/skip the IRC-like validation "
                                          "(default: per --rigor).")
    p_tse.add_argument("--cv-product-s", type=float, default=None,
                       help="reactant-only: product-side CV target s (Å).")
    p_tse.add_argument("--execute", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="QM-native engine only: --no-execute writes the input "
                            "(e.g. an ORCA NEB-TS job to sbatch) without running it.")
    p_tse.add_argument("--outdir", default=None)
    p_tse.add_argument("--log-level", default="INFO")

    # ts-propose — run ONE TS-guess proposer standalone (sidecar two-step handoff)
    p_tpr = sub.add_parser(
        "ts-propose", help="Run a single TS-guess proposer (e.g. react-ot) and write "
                           "the guess (sidecar step 1 → feed ts-entry --entry ts-guess).")
    p_tpr.add_argument("--method", required=True, help="Proposer (midpoint/react-ot/...)")
    p_tpr.add_argument("--reactant", required=True, help="Reactant geometry")
    p_tpr.add_argument("--product", required=True, help="Product geometry")
    p_tpr.add_argument("--charge", type=int, default=None)
    p_tpr.add_argument("--multiplicity", dest="spin", type=int, default=None, help="Multiplicity 2S+1")
    p_tpr.add_argument("--spin", dest="_spin_S", type=int, default=None, help="Spin quantum number S; converted to multiplicity 2S+1 (prefer --multiplicity).")
    p_tpr.add_argument("--out", default=None, help="Write the guess here (xyz) for the handoff.")
    p_tpr.add_argument("--outdir", default=None)
    p_tpr.add_argument("--log-level", default="INFO")

    # ts-refine — run ONE TS-guess refiner standalone (sidecar two-step handoff)
    p_trf = sub.add_parser(
        "ts-refine", help="Run a single TS-guess refiner (e.g. aefm) and write the "
                          "refined guess (sidecar step 1 → feed ts-entry --entry ts-guess).")
    p_trf.add_argument("--method", required=True, help="Refiner (identity/aefm/...)")
    p_trf.add_argument("--ts-guess", required=True, help="TS-guess geometry to refine")
    p_trf.add_argument("--reactant", default=None, help="Optional R context (refiner-dependent)")
    p_trf.add_argument("--product", default=None, help="Optional P context (refiner-dependent)")
    p_trf.add_argument("--charge", type=int, default=None)
    p_trf.add_argument("--multiplicity", dest="spin", type=int, default=None, help="Multiplicity 2S+1")
    p_trf.add_argument("--spin", dest="_spin_S", type=int, default=None, help="Spin quantum number S; converted to multiplicity 2S+1 (prefer --multiplicity).")
    p_trf.add_argument("--out", default=None, help="Write the refined guess here (xyz).")
    p_trf.add_argument("--allow-out-of-domain", action="store_true",
                       help="Let a refiner run on out-of-training-domain elements (e.g. "
                            "AEFM on a metal site): unvalidated; the QM gate still decides.")
    p_trf.add_argument("--outdir", default=None)
    p_trf.add_argument("--log-level", default="INFO")

    # monitor — non-constraining bond + metal-coordination report
    p_mon = sub.add_parser(
        "monitor", help="Non-constraining bond + metal-coordination report")
    p_mon.add_argument("input", help="Structure (PDB/XYZ/CIF)")
    p_mon.add_argument("--bond", action="append", default=None, metavar="i,j",
                       help="0-based atom index pair to measure (repeatable).")
    p_mon.add_argument("--metals", action="store_true",
                       help="Auto-detect metals + report their coordination shells.")
    p_mon.add_argument("--outdir", default=None, help="If set, also write a JSON report.")
    p_mon.add_argument("--log-level", default="INFO")

    # reaction-spec — validate / resolve a ReactionSpec YAML
    p_rs = sub.add_parser(
        "reaction-spec", help="Validate (and optionally resolve) a ReactionSpec YAML")
    p_rs.add_argument("spec", help="Path to the ReactionSpec YAML")
    p_rs.add_argument("--structure", default=None,
                      help="Resolve atom tokens against this structure (PDB/XYZ/CIF).")
    p_rs.add_argument("--log-level", default="INFO")

    # chemoton-explore — steered Chemoton reaction-network exploration
    p_ce = sub.add_parser("chemoton-explore",
                          help="Steered Chemoton reaction-network exploration "
                               "(Sparrow PM6/PM7 or xtb-GFN2; outputs ranked "
                               "TS-PDBs for downstream MACE refinement)")
    p_ce.add_argument("input", help="Input PDB (active site + ligand + metals)")
    p_ce.add_argument("--config", default=None,
                      help="YAML defaults file (e.g. configs/chemoton_pte.yaml). "
                           "CLI flags override config values.")
    p_ce.add_argument("--reactive-atoms", nargs="+", default=None,
                      help="Hint atom names (e.g. 'Zn1' 'SUB:P1' 'OHX:O3'). "
                           "Currently logged only — filter stack derives "
                           "reactivity from element rules.")
    p_ce.add_argument("--cluster-spec", default="auto",
                      help="Cluster selection grammar (currently 'auto' or "
                           "'site 5.0 LIG SUB OHX KCX'). Defaults to 'auto'.")
    p_ce.add_argument("--backend", default="xtb-gfn2",
                      choices=["sparrow-pm6", "sparrow-pm7", "xtb-gfn2"],
                      help="Cheap-tier backend. Default xtb-gfn2 because the "
                           "scine-sparrow 5.1.0 wheel currently aborts on "
                           "import in our container (free() invalid pointer). "
                           "Switch back to sparrow-pm6 once the wheel is "
                           "fixed.")
    p_ce.add_argument("--max-bond-modifications", type=int, default=2,
                      help="Per-Rearrangement step max bond formations / "
                           "dissociations (default 2).")
    p_ce.add_argument("--max-depth", type=int, default=2,
                      help="Number of expansion rounds beyond the input "
                           "(default 2; only 2 currently fully wired).")
    p_ce.add_argument("--barrier-cap-kcal", type=float, default=60.0,
                      help="Step-3 barrier filter (gas-phase electronic, "
                           "kcal/mol). Default 60.")
    p_ce.add_argument("--top-n-export", type=int, default=5,
                      help="How many lowest-barrier elementary steps to "
                           "write back as PDB (default 5).")
    p_ce.add_argument("--mongodb-uri",
                      default="mongodb://localhost:27017/",
                      help="MongoDB URI. dbname empty → auto-generated. "
                           "Use tools/scine_mongo.sh start to launch a "
                           "per-user mongod.")
    p_ce.add_argument("--central-metal", default="Zn",
                      help="Element symbol of catalytic metal "
                           "(CentralSiteFilter / CentralMetalSelection).")
    p_ce.add_argument("--charge", type=int, default=1,
                      help="Cluster net charge (default +1 for PTE).")
    p_ce.add_argument("--multiplicity", type=int, default=1,
                      help="Cluster spin multiplicity (default singlet=1).")
    p_ce.add_argument("--validate-with", default=None,
                      help="MACE model alias (e.g. 'mace-polar-m'). Hint "
                           "for downstream refinement; not run here.")
    p_ce.add_argument("--out", default=None,
                      help="Output directory (default: runs/chemoton/<runid>/).")
    p_ce.add_argument("--restart-file", default=None,
                      help="Resume from a previous SteeringWheel pickle.")
    p_ce.add_argument("--dry-run", action="store_true",
                      help="Build wheel + check Mongo, but don't run.")
    p_ce.add_argument("-v", "--verbose", action="store_true",
                      help="Set log level to DEBUG.")
    p_ce.add_argument("--log-level", default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # ============================================================
    #  v2 extended TS pipeline subcommands (modular, composable)
    # ============================================================
    # endpoint-release
    p_er = sub.add_parser(
        "endpoint-release",
        help="Release reactive-bond constraints from a scan endpoint and "
             "relax with cluster boundary preserved (clean NEB endpoint).",
        description=(
            "Stage 1.5 of the v2 TS pipeline. Fixes the kinked-endpoint "
            "artefact left over from a 1-D constrained scan: load a R-side "
            "or P-side scan structure, drop any reactive-bond CV "
            "constraints, keep the boundary scaffold (CA-only, backbone, "
            "etc.), and relax to a tighter fmax. The released PDB becomes "
            "the new NEB endpoint. Reaction-agnostic — supply --release-bond "
            "with NAME.RESNAME tokens or 1-based serials."
        ),
    )
    p_er.add_argument("input", help="Scan endpoint PDB (R-side or P-side).")
    p_er.add_argument("--out", required=True,
                       help="Output PDB path for the released endpoint.")
    p_er.add_argument("--release-bond", action="append", default=[],
                       metavar="ATOMA,ATOMB",
                       help="Reactive bond whose CV constraint is dropped. "
                            "Tokens accept 'NAME.RESNAME', "
                            "'NAME.RESNAME.RESID', integer 1-based PDB "
                            "serial, or '0:idx'. Repeatable.")
    p_er.add_argument("--boundary-fix-preset", default=None,
                       choices=["ca-only", "backbone", "backbone-water", "none"],
                       help="Boundary preset (CA-only is the standard).")
    p_er.add_argument("--fix", nargs="+", default=None,
                       help="Extra constraint specs.")
    p_er.add_argument("--free", nargs="+", default=None,
                       help="Subtractive constraint specs.")
    p_er.add_argument("--model", default="mace-omol")
    p_er.add_argument("--head", default=None)
    p_er.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p_er.add_argument("--charge-ledger", default=None,
                       help="Optional ledger.yaml (validated and propagated).")
    p_er.add_argument("--charge", type=int, default=None)
    p_er.add_argument("--multiplicity", dest="spin", type=int, default=None)
    p_er.add_argument("--spin", dest="_spin_S", type=int, default=None, help="Spin quantum number S; converted to multiplicity 2S+1 (prefer --multiplicity).")
    p_er.add_argument("--fmax", type=float, default=0.02,
                       help="Convergence target (eV/Å). Tighter than scan "
                            "defaults — that's the whole point. Bump to 0.03 "
                            "for noisy ML potentials.")
    p_er.add_argument("--max-steps", type=int, default=500)
    p_er.add_argument("--optimizer", default="lbfgs",
                       choices=["lbfgs", "bfgs", "fire"])
    p_er.add_argument("--outdir", default=None,
                       help="(unused — kept for CLI parity)")
    p_er.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # scan2d
    p_s2d = sub.add_parser(
        "scan2d",
        help="2-D relaxed scan around a TS guess (3x3 / 5x5 / NxN grid).",
        description=(
            "Diagnostic for the 1-D constrained-scan path. Sweeps two "
            "reactive bond lengths INDEPENDENTLY on a small grid; the "
            "argmax tells you whether the 1-D scan slices through the "
            "saddle or misses an off-axis saddle. Reaction-agnostic — "
            "the two bonds are user-supplied via --bond-a / --bond-b."
        ),
    )
    p_s2d.add_argument("--input", required=True,
                        help="PDB used for atom indexing (typically same as --ts-guess).")
    p_s2d.add_argument("--ts-guess", required=True,
                        help="TS-guess PDB to centre the grid on.")
    p_s2d.add_argument("--bond-a", required=True, metavar="ATOMA,ATOMB",
                        help="First reactive bond.")
    p_s2d.add_argument("--bond-b", required=True, metavar="ATOMC,ATOMD",
                        help="Second reactive bond.")
    p_s2d.add_argument("--grid", default="3x3",
                        help="Grid shape NxM (default 3x3). 5x5 for finer "
                             "scan when the 1-D path looks suspect.")
    p_s2d.add_argument("--delta-d", type=float, default=0.20,
                        help="Half-width sweep (Å). Bump if argmax lands on "
                             "the grid edge.")
    p_s2d.add_argument("--delta-d-a", type=float, default=None)
    p_s2d.add_argument("--delta-d-b", type=float, default=None)
    p_s2d.add_argument("--boundary-fix-preset", default=None,
                        choices=["ca-only", "backbone", "backbone-water", "none"])
    p_s2d.add_argument("--fix", nargs="+", default=None)
    p_s2d.add_argument("--free", nargs="+", default=None)
    p_s2d.add_argument("--model", default="mace-omol")
    p_s2d.add_argument("--head", default=None)
    p_s2d.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p_s2d.add_argument("--charge-ledger", default=None)
    p_s2d.add_argument("--charge", type=int, default=None)
    p_s2d.add_argument("--multiplicity", dest="spin", type=int, default=None)
    p_s2d.add_argument("--spin", dest="_spin_S", type=int, default=None, help="Spin quantum number S; converted to multiplicity 2S+1 (prefer --multiplicity).")
    p_s2d.add_argument("--fmax", type=float, default=0.05,
                        help="Per-grid-point optimizer fmax. Looser than "
                             "release stage — FixBondLengths holds geometry.")
    p_s2d.add_argument("--max-steps", type=int, default=250)
    p_s2d.add_argument("--no-plot", action="store_true",
                        help="Skip the heatmap PNG.")
    p_s2d.add_argument("--outdir", default=None, help="Output directory.")
    p_s2d.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # microstates
    p_ms = sub.add_parser(
        "microstates",
        help="Generate protonation/tautomer/water microstate ensemble.",
        description=(
            "Targeted microstate sampler — discrete enumeration of HID/HIE/"
            "HIP, ASP/GLU/LYS/CYS protonation states, Zn-bound OH⁻ vs H₂O, "
            "and stochastic water reorientations. Pass --generators to pick "
            "which axes to vary. Reaction-agnostic — residues are "
            "discovered from the input PDB."
        ),
    )
    p_ms.add_argument("--input", required=True, help="Cluster PDB.")
    p_ms.add_argument("--outdir", default=None, help="Output directory.")
    p_ms.add_argument(
        "--generators", required=False, default=None,
        help="Legacy ledger-only generators. Comma-separated subset of: "
             "his, asp_glu, lys, cys, zn_oh, water_shuffle, water_translate. "
             "Required ONLY when neither --auto-protonation nor "
             "--protonation-rules is given."
    )
    # Atom-level H-rewriting paths
    p_ms.add_argument("--auto-protonation", action="store_true",
                       help="Auto-enumerate HIS/ASP/GLU/LYS/CYS protonation "
                            "states with REAL H-atom rewriting "
                            "(consensus + 1-residue-perturbation strategy "
                            "when product exceeds --max-microstates).")
    p_ms.add_argument("--protonation-rules", default=None,
                       help="Path to YAML listing residues + states for "
                            "explicit (non-auto) Cartesian product. "
                            "See tools/microstate_sampler.py for schema.")
    p_ms.add_argument("--protonation-families",
                       default="HIS,ASP,GLU,LYS,CYS",
                       help="Comma list of families for --auto-protonation. "
                            "Default HIS,ASP,GLU,LYS,CYS.")
    p_ms.add_argument("--max-microstates", type=int, default=16,
                       help="Cap on protonation microstates (auto / rules "
                            "modes). Default 16.")
    p_ms.add_argument("--metal-cutoff-a", type=float, default=3.5,
                       help="HIS within this many Å of any metal → "
                            "consensus HID. Default 3.5.")
    p_ms.add_argument("--pos-charge-cutoff-a", type=float, default=4.5,
                       help="Reserved for ASP/GLU positive-neighbor "
                            "heuristic. Default 4.5.")
    p_ms.add_argument("--nh-bond-length", type=float, default=1.01,
                       help="N–H placement target length (Å). "
                            "Default 1.01 (Allen et al. CSD).")
    p_ms.add_argument("--oh-bond-length", type=float, default=0.96,
                       help="O–H placement target length (Å). "
                            "Default 0.96 (Allen et al. CSD).")
    p_ms.add_argument("--sh-bond-length", type=float, default=1.34,
                       help="S–H placement target length (Å). "
                            "Default 1.34 (Allen et al. CSD).")
    p_ms.add_argument("--no-include-consensus", action="store_true",
                       help="Skip emitting all-consensus state (auto only).")
    p_ms.add_argument("--n-water-shuffle", type=int, default=5,
                       help="Variants per water_shuffle invocation.")
    p_ms.add_argument("--n-water-translate", type=int, default=5,
                       help="Variants per water_translate invocation.")
    p_ms.add_argument("--water-translate-delta", type=float, default=0.30,
                       help="Displacement magnitude for water_translate (Å).")
    p_ms.add_argument("--seed", type=int, default=12345,
                       help="PRNG seed for stochastic generators.")
    p_ms.add_argument("--relax", action="store_true",
                       help="MLFF-relax each variant (slow; off by default).")
    p_ms.add_argument("--relax-max-steps", type=int, default=100)
    p_ms.add_argument("--relax-fmax", type=float, default=0.05)
    p_ms.add_argument("--model", default="mace-omol")
    p_ms.add_argument("--head", default=None)
    p_ms.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p_ms.add_argument("--charge-ledger", default=None)
    p_ms.add_argument("--charge", type=int, default=None)
    p_ms.add_argument("--multiplicity", dest="spin", type=int, default=None)
    p_ms.add_argument("--spin", dest="_spin_S", type=int, default=None, help="Spin quantum number S; converted to multiplicity 2S+1 (prefer --multiplicity).")
    p_ms.add_argument("--max-variants", type=int, default=200,
                       help="Cap on emitted variants (defensive default).")
    p_ms.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # validate-ts
    p_vts = sub.add_parser(
        "validate-ts",
        help="Tiered Hessian validation (A=reactive, B=expanded, C=Lanczos).",
        description=(
            "Three tiers of TS validation, each more expensive than the "
            "last. Tier A (~11 atoms) is the standard refine-ts check; "
            "Tier B (~30-60 atoms) covers the full active region; Tier C "
            "(Lanczos) verifies no second imag mode hides anywhere in the "
            "movable subspace. --tier accepts a, b, c, all, or a comma "
            "list (e.g. 'a,b'). Reaction-agnostic — reactive atoms come "
            "from --reactive-atoms, active region from --active-region "
            "(the same select grammar as 'cowboy-qc saddle')."
        ),
    )
    p_vts.add_argument("input", help="TS structure PDB.")
    p_vts.add_argument("--outdir", default=None)
    p_vts.add_argument("--model", default="mace-omol")
    p_vts.add_argument("--head", default=None)
    p_vts.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p_vts.add_argument("--charge-ledger", default=None)
    p_vts.add_argument("--charge", type=int, default=None)
    p_vts.add_argument("--multiplicity", dest="spin", type=int, default=None)
    p_vts.add_argument("--spin", dest="_spin_S", type=int, default=None, help="Spin quantum number S; converted to multiplicity 2S+1 (prefer --multiplicity).")
    p_vts.add_argument("--reactive-atoms", nargs="+", required=True,
                        help="Atoms whose motion defines the reaction "
                             "coordinate (1-based PDB serials, '0:idx', "
                             "or 'RES:ID:NAME').")
    p_vts.add_argument("--tier", default="b",
                        help="Tier(s) to run: 'a', 'b', 'c', 'all', or "
                             "comma list (e.g. 'a,b'). Default 'b'.")
    p_vts.add_argument("--active-region", default=None,
                        help="select-grammar spec defining the Tier B "
                             "atom set. Examples: 'site 5.0 SUB OHX KCX', "
                             "'residue HIS GLU', 'chain A; residue YYE'. "
                             "Multiple specs separated by ';'.")
    p_vts.add_argument("--active-region-radius", type=float, default=None,
                        help="Alternative to --active-region: include all "
                             "atoms within this many Å of any reactive atom.")
    p_vts.add_argument("--active-region-resnames", default=None,
                        help="Alternative to --active-region: comma list of "
                             "residue names to include in Tier B.")
    p_vts.add_argument("--delta", type=float, default=0.005,
                        help="Finite-difference step (Å). Tighter than "
                             "freq default because TS validation needs "
                             "accurate frequencies.")
    p_vts.add_argument("--method", default="central", choices=["central", "forward"])
    p_vts.add_argument("--n-imag-expected", type=int, default=1,
                        help="Expected imag-mode count. 1 for first-order TS.")
    p_vts.add_argument("--imag-cm-cutoff", type=float, default=-50.0,
                        help="Imag freq must be MORE NEGATIVE than this (cm⁻¹).")
    p_vts.add_argument("--imag-mode-min-overlap", type=float, default=0.5,
                        help="Min fraction of imag-mode L2 amplitude on "
                             "reactive atoms. Lower (0.3) for delocalised "
                             "concerted reactions; raise (0.8) for SN2.")
    p_vts.add_argument("--second-imag-cm-cutoff", type=float, default=-10.0,
                        help="Tier B: any second-mode below this cutoff "
                             "(cm⁻¹) flags 'second imag present'. Default "
                             "-10 catches obvious water/ligand drift modes.")
    p_vts.add_argument("--allow-second-imag", action="store_true",
                        help="Don't fail when a second imag is present "
                             "(downgrade to warning).")
    p_vts.add_argument("--tier-c-n-modes", type=int, default=4,
                        help="Number of lowest modes to estimate via Lanczos.")
    p_vts.add_argument("--tier-c-max-iters", type=int, default=50)
    p_vts.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # verify-irc-like
    p_vil = sub.add_parser(
        "verify-irc-like",
        help="±imag-mode displacement + relax test (cheap IRC alternative).",
        description=(
            "Push the TS atoms ±X Å along the imag mode and relax both "
            "branches. A bona-fide saddle has the two branches end in "
            "DIFFERENT basins, both lower in energy than the TS. "
            "Cheaper than a full IRC. Reaction-agnostic — supply the "
            "imag mode as .npy or XYZ via --imag-mode."
        ),
    )
    p_vil.add_argument("input", help="TS structure PDB.")
    p_vil.add_argument("--outdir", default=None)
    p_vil.add_argument("--imag-mode", required=True,
                        help="Path to imag-mode vector (.npy or .xyz).")
    p_vil.add_argument("--model", default="mace-omol")
    p_vil.add_argument("--head", default=None)
    p_vil.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p_vil.add_argument("--charge-ledger", default=None)
    p_vil.add_argument("--charge", type=int, default=None)
    p_vil.add_argument("--multiplicity", dest="spin", type=int, default=None)
    p_vil.add_argument("--spin", dest="_spin_S", type=int, default=None, help="Spin quantum number S; converted to multiplicity 2S+1 (prefer --multiplicity).")
    p_vil.add_argument("--displacement", type=float, default=0.20,
                        help="Displacement magnitude (Å). Default 0.20. "
                             "Bump to 0.30-0.50 for stiff TS.")
    p_vil.add_argument("--fmax", type=float, default=0.05,
                        help="Per-branch optimizer convergence.")
    p_vil.add_argument("--max-steps", type=int, default=200)
    p_vil.add_argument("--optimizer", default="lbfgs",
                        choices=["lbfgs", "bfgs", "fire"])
    p_vil.add_argument("--basin-min-drop-eV", type=float, default=0.005,
                        dest="basin_min_drop_eV",
                        help="Each branch must drop at least this much (eV) "
                             "below the TS to qualify as a basin.")
    p_vil.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # ts-pipeline-v2
    p_tp2 = sub.add_parser(
        "ts-pipeline-v2",
        help="v2 TS pipeline orchestrator (microstates → scans → endpoints "
             "→ NEB → dimer → validate → IRC-like).",
        description=(
            "Chained orchestrator for the v2 extended TS workflow. Reads "
            "a YAML config that lists stages and per-stage knobs; each "
            "stage runs as a sub-CLI invocation. --resume-from skips "
            "earlier stages (assumed already completed). --print-example "
            "shows a template config."
        ),
    )
    p_tp2.add_argument("config", nargs="?", default=None,
                       help="Path to YAML pipeline config (omit with --print-example).")
    p_tp2.add_argument("--outdir", default=None,
                       help="Override config's outdir key.")
    p_tp2.add_argument("--resume-from", default=None,
                       help="Skip stages before this one (e.g. 'endpoint_release').")
    p_tp2.add_argument("--only-stages", default=None,
                       help="Comma-separated whitelist of stage names to run.")
    p_tp2.add_argument("--dry-run", action="store_true",
                       help="Log planned commands without executing.")
    p_tp2.add_argument("--print-example", action="store_true",
                       help="Print example pipeline.yaml and exit.")
    p_tp2.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # crest-mace — CREST 3 conformer search with MACE forces (daemon + generic_sc)
    p_cm = sub.add_parser(
        "crest-mace",
        help="Run CREST 3 with MACE forces via the generic_sc calculator + daemon "
             "(see tools/MACE_CREST_README.md). Additive to xtb-CREST; does not "
             "replace existing cowboy-qc crest paths.",
    )
    p_cm.add_argument("input", help="Input xyz file")
    p_cm.add_argument("--model", default="mace-mp",
                      help="MACE alias from the cowboy-qc factory or absolute .model path")
    p_cm.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p_cm.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p_cm.add_argument("--head", default=None,
                      help="Head selector for multi-head models (e.g. 'omol' for mace-mh)")
    p_cm.add_argument("--charge", type=int, default=0)
    p_cm.add_argument("--spin", type=int, default=0,
                      help="Number of unpaired electrons")
    p_cm.add_argument("--workdir", default=None,
                      help="CREST working directory (default: tempdir)")
    p_cm.add_argument("--toml", default=None,
                      help="Custom TOML config (default: auto-generated)")
    p_cm.add_argument("--logfile", default=None,
                      help="Daemon log path (default: <workdir>/mace_daemon.log)")
    p_cm.add_argument(
        "--crest-args", nargs=argparse.REMAINDER, default=[],
        help="All remaining args after this flag are forwarded to CREST. "
             "Example: --crest-args -gfn2 --quick",
    )

    args = parser.parse_args(argv)

    # Resolve --spin (the spin quantum number S) into the multiplicity field
    # (args.spin holds the MULTIPLICITY M=2S+1 that the handlers/RunContext expect).
    # --multiplicity sets M directly; --spin sets S and is converted here. Giving
    # both is an error. (crest-mace has no _spin_S — its --spin is unpaired electrons.)
    _spin_S = getattr(args, "_spin_S", None)
    if _spin_S is not None:
        if getattr(args, "spin", None) is not None:
            parser.error("pass either --multiplicity (M=2S+1) or --spin (S), not both")
        args.spin = 2 * int(_spin_S) + 1
        print(f"# --spin={_spin_S} interpreted as spin quantum number S "
              f"→ multiplicity {args.spin} (=2S+1); use --multiplicity to set M directly.",
              file=sys.stderr)

    # --verbose flag overrides --log-level for chemoton-explore
    if getattr(args, "verbose", False):
        args.log_level = "DEBUG"

    logging.basicConfig(
        level=getattr(logging, getattr(args, "log_level", "INFO")),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    dispatch = {
        "sp": _cmd_sp, "opt": _cmd_opt, "md": _cmd_md,
        "freq": _cmd_freq, "scan": _cmd_scan,
        "saddle": _cmd_saddle, "irc": _cmd_irc,
        "refine-ts": _cmd_refine_ts,
        "pipeline": _cmd_pipeline,
        "neb": _cmd_neb, "mtd": _cmd_mtd,
        "gsm": _cmd_gsm, "ts": _cmd_ts,
        "ts-entry": _cmd_ts_entry,
        "ts-propose": _cmd_ts_propose,
        "ts-refine": _cmd_ts_refine,
        "monitor": _cmd_monitor,
        "reaction-spec": _cmd_reaction_spec,
        "run": _cmd_run,
        "protonate": _cmd_protonate,
        "list-models": _cmd_list_models,
        "info": _cmd_info,
        "chemoton-explore": _cmd_chemoton_explore,
        # v2 extended TS pipeline subcommands
        "endpoint-release": _cmd_endpoint_release,
        "scan2d": _cmd_scan2d,
        "microstates": _cmd_microstates,
        "validate-ts": _cmd_validate_ts,
        "verify-irc-like": _cmd_verify_irc_like,
        "ts-pipeline-v2": _cmd_ts_pipeline_v2,
        "crest-mace": _cmd_crest_mace,
    }
    handler = dispatch[args.op]
    result = handler(args)

    # Pretty-print key results
    print()
    print("=" * 60)
    print(f"qcb {args.op} result:")
    for k, v in result.items():
        if k in ("atoms", "reactant", "product", "ts", "images", "outputs",
                 "energies_eV", "temperatures_K", "fes", "cv_trajectory",
                 "coord_values", "relative_energies_kcal", "frequencies_cm",
                 "modes", "charges", "dipole_debye", "forces_eV_per_A"):
            continue
        print(f"  {k}: {v}")
    if "outputs" in result:
        print("  outputs:")
        for k, v in result["outputs"].items():
            if v:
                print(f"    {k}: {v}")
    print("=" * 60)

    return 0 if result.get("status", "ok") in (
        "converged", "completed", "cached", "valid", "prepared", "ok") else 1


if __name__ == "__main__":
    sys.exit(main())
