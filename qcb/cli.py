"""Top-level CLI for quantum_cowboy_biochemistry.

Usage:
    qcb <op> <input> [options]

Ops:
    sp       single-point energy
    opt      energy minimization
    md       molecular dynamics
    freq     vibrational frequencies
    scan     coordinate scan (bond/angle/dihedral)
    saddle   Sella saddle-point search
    irc      IRC descent from a TS (guess)
    neb      nudged elastic band (requires reactant + product)
    mtd      well-tempered metadynamics
    ts       full TS pipeline (composes endpoint generation + NEB + Sella + IRC)

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
    qcb opt input.pdb --model mace-omol --fmax 0.01
    qcb md input.pdb --time 10 --temp 300 --friction 1.0
    qcb scan input.pdb --coord "bond" --indices "5 12" --range "1.5 3.5 20"
    qcb saddle ts_guess.pdb
    qcb irc ts.pdb --step 0.1
    qcb neb reactant.pdb product.pdb --n-images 15
    qcb ts input.pdb --strategy irc
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


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
    from qcb.calc import make_calc
    from qcb.io import load_structure, parse_constraints, build_fix_atoms
    from qcb.io.constraints import preset_to_specs

    atoms, bt_struct, charge_hint = load_structure(args.input)
    # Warn on charge conflict between CLI flag and PDB REMARK
    if args.charge is not None and charge_hint is not None and args.charge != charge_hint:
        logging.getLogger("qcb.cli").warning(
            f"--charge {args.charge} disagrees with PDB REMARK ({charge_hint}); using CLI value"
        )
    charge = args.charge if args.charge is not None else (charge_hint or 0)

    # Detect ligand for preset exclusions (first HETATM non-water, non-metal)
    ligand_name = None
    if bt_struct is not None:
        from qcb.io.constraints import STANDARD_EXCLUDED_RES
        for rn in bt_struct.res_name:
            if rn not in STANDARD_EXCLUDED_RES and rn not in ("ALA","ARG","ASN","ASP","CYS",
                "GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP",
                "TYR","VAL","KCX"):
                ligand_name = str(rn)
                break

    calc = make_calc(model=args.model, head=args.head, device=args.device, charge=charge)
    atoms.info["charge"] = charge
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
    from qcb.ops import sp
    atoms, _, calc, constraint, _, outdir, _ = _setup_atoms_and_calc(args)
    return sp.run(atoms, calc, outdir, constraint)


def _cmd_opt(args):
    from qcb.ops import opt
    atoms, bt, calc, constraint, charge, outdir, _ = _setup_atoms_and_calc(args)
    res = opt.run(atoms, calc, outdir, constraint,
                  optimizer=args.optimizer, fmax=args.fmax, max_steps=args.max_steps)
    # Optionally write output PDB
    if args.output_pdb:
        from qcb.io import write_pdb
        write_pdb(res["atoms"], bt, args.output_pdb,
                  total_charge=charge, energy_eV=res["energy_eV"])
    return res


def _cmd_md(args):
    from qcb.ops import md
    atoms, _, calc, constraint, _, outdir, _ = _setup_atoms_and_calc(args)
    return md.run(atoms, calc, outdir, constraint,
                  ensemble=args.ensemble, timestep_fs=args.timestep,
                  total_time_ps=args.time, temperature_K=args.temp,
                  friction_per_ps=args.friction, dump_every_fs=args.dump_every,
                  anneal_peak_K=args.anneal_peak, seed=args.seed)


def _cmd_freq(args):
    from qcb.ops import freq
    atoms, _, calc, constraint, _, outdir, _ = _setup_atoms_and_calc(args)
    indices = None
    if args.indices:
        indices = [int(x) for x in args.indices]
    return freq.run(atoms, calc, outdir, constraint,
                    indices=indices, delta=args.delta,
                    method=args.method, temperature_K=args.temp)


def _cmd_scan(args):
    from qcb.ops import scan
    atoms, _, calc, constraint, _, outdir, _ = _setup_atoms_and_calc(args)
    indices = [int(x) for x in args.indices]
    return scan.run(atoms, calc, outdir, constraint,
                    coord_type=args.coord, atom_indices=indices,
                    start=args.start, end=args.end, n_steps=args.n_steps,
                    relax_other=not args.no_relax, fmax=args.fmax)


def _cmd_saddle(args):
    from qcb.ops import saddle
    atoms, _, calc, constraint, _, outdir, _ = _setup_atoms_and_calc(args)
    return saddle.run(atoms, calc, outdir, constraint,
                      fmax=args.fmax, max_steps=args.max_steps)


def _cmd_irc(args):
    from qcb.ops import irc
    atoms, _, calc, constraint, _, outdir, _ = _setup_atoms_and_calc(args)
    return irc.run(atoms, calc, outdir, constraint,
                   refine_ts=not args.no_refine_ts,
                   saddle_fmax=args.saddle_fmax,
                   irc_step=args.step, irc_fmax=args.fmax)


def _cmd_neb(args):
    from qcb.ops import neb
    from qcb.calc import make_calc_fn
    from qcb.io import load_structure, parse_constraints, build_fix_atoms
    from qcb.io.constraints import preset_to_specs, STANDARD_EXCLUDED_RES

    r_atoms, r_bt, r_charge = load_structure(args.reactant)
    p_atoms, p_bt, p_charge = load_structure(args.product)

    # Reconcile charges: CLI flag wins but warn on disagreement
    if args.charge is not None and r_charge is not None and args.charge != r_charge:
        logging.getLogger("qcb.cli").warning(
            f"--charge {args.charge} disagrees with PDB REMARK ({r_charge}); using CLI value"
        )
    charge = args.charge if args.charge is not None else (r_charge or 0)

    calc_fn = make_calc_fn(model=args.model, head=args.head, device=args.device, charge=charge)
    r_atoms.calc = calc_fn()
    p_atoms.calc = calc_fn()
    r_atoms.info["charge"] = charge
    p_atoms.info["charge"] = charge

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

    return neb.run(r_atoms, p_atoms, calc_fn, outdir, constraint=constraint,
                   n_images=args.n_images, k_spring=args.k_spring,
                   interpolation_method=args.interpolation,
                   fmax_noclimb=args.fmax_noclimb, steps_noclimb=args.steps_noclimb,
                   fmax_climb=args.fmax_climb, steps_climb=args.steps_climb,
                   charge=charge)


def _cmd_mtd(args):
    from qcb.ops import mtd
    atoms, _, calc, constraint, _, outdir, _ = _setup_atoms_and_calc(args)
    return mtd.run(atoms, calc, outdir, constraint,
                   p_idx=args.p_idx, nuc_idx=args.nuc_idx, lg_idx=args.lg_idx,
                   total_time_ps=args.time, temperature_K=args.temp)


def _cmd_ts(args):
    from qcb.ops import ts
    extra = []
    if args.passthrough:
        extra = args.passthrough
    if args.fix_preset:
        extra.extend(["--constraint-mode", args.fix_preset])
    if args.head:
        extra.extend(["--head", args.head])
    charge = args.charge
    outdir = Path(args.outdir) if args.outdir else Path("qcb-ts-out")
    return ts.run(args.input, outdir, strategy=args.strategy,
                  model=args.model, charge=charge, extra_args=extra)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="qcb",
        description="Quantum Cowboy Biochemistry: enzyme computational chemistry toolkit",
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
    p_saddle = sub.add_parser("saddle", help="Sella saddle-point search")
    _common_parser_setup(p_saddle)
    p_saddle.add_argument("--fmax", type=float, default=0.02)
    p_saddle.add_argument("--max-steps", type=int, default=500)

    # irc
    p_irc = sub.add_parser("irc", help="IRC descent from a TS")
    _common_parser_setup(p_irc)
    p_irc.add_argument("--no-refine-ts", action="store_true", help="Skip Sella refinement")
    p_irc.add_argument("--saddle-fmax", type=float, default=0.02)
    p_irc.add_argument("--step", type=float, default=0.1, help="Displacement along imag mode")
    p_irc.add_argument("--fmax", type=float, default=0.03)

    # neb
    p_neb = sub.add_parser("neb", help="NEB + CI-NEB")
    p_neb.add_argument("reactant")
    p_neb.add_argument("product")
    p_neb.add_argument("--model", default="mace-omol")
    p_neb.add_argument("--head", default=None)
    p_neb.add_argument("--charge", type=int, default=None)
    p_neb.add_argument("--device", default="cuda")
    p_neb.add_argument("--outdir", default=None)
    p_neb.add_argument("--fix", nargs="+", default=None,
                        help="Constraint specs to apply to all images")
    p_neb.add_argument("--free", nargs="+", default=None)
    p_neb.add_argument("--fix-preset", default=None,
                        choices=["ca-only", "backbone", "backbone-water", "none"])
    p_neb.add_argument("--n-images", type=int, default=15)
    p_neb.add_argument("--k-spring", type=float, default=1.0)
    p_neb.add_argument("--interpolation", default="geodesic", choices=["geodesic","idpp","linear"])
    p_neb.add_argument("--fmax-noclimb", type=float, default=0.40)
    p_neb.add_argument("--steps-noclimb", type=int, default=200)
    p_neb.add_argument("--fmax-climb", type=float, default=0.045)
    p_neb.add_argument("--steps-climb", type=int, default=250)
    p_neb.add_argument("--log-level", default="INFO")

    # mtd
    p_mtd = sub.add_parser("mtd", help="Well-tempered metadynamics")
    _common_parser_setup(p_mtd)
    p_mtd.add_argument("--p-idx", type=int, required=True, help="Central atom index")
    p_mtd.add_argument("--nuc-idx", type=int, required=True, help="Nucleophile atom index")
    p_mtd.add_argument("--lg-idx", type=int, required=True, help="Leaving-group atom index")
    p_mtd.add_argument("--time", type=float, default=100.0, help="Total MTD time in ps")
    p_mtd.add_argument("--temp", type=float, default=300.0)

    # ts
    p_ts = sub.add_parser("ts", help="Full TS pipeline (wraps scripts/run_neb_ts.py)")
    p_ts.add_argument("input")
    p_ts.add_argument("--outdir", default=None)
    p_ts.add_argument("--model", default="mace-omol")
    p_ts.add_argument("--head", default=None)
    p_ts.add_argument("--charge", type=int, default=None)
    p_ts.add_argument("--device", default="cuda")
    p_ts.add_argument("--strategy", default="legacy",
                      choices=["legacy", "irc", "cv-spring", "mtd"])
    p_ts.add_argument("--fix-preset", default=None,
                      choices=["ca-only", "backbone", "backbone-water", "none"])
    p_ts.add_argument("--passthrough", nargs=argparse.REMAINDER,
                      help="Additional flags to pass to run_neb_ts.py")
    p_ts.add_argument("--log-level", default="INFO")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, getattr(args, "log_level", "INFO")),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    dispatch = {
        "sp": _cmd_sp, "opt": _cmd_opt, "md": _cmd_md,
        "freq": _cmd_freq, "scan": _cmd_scan,
        "saddle": _cmd_saddle, "irc": _cmd_irc,
        "neb": _cmd_neb, "mtd": _cmd_mtd, "ts": _cmd_ts,
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

    return 0 if result.get("status") in ("converged", "completed", "cached") else 1


if __name__ == "__main__":
    sys.exit(main())
