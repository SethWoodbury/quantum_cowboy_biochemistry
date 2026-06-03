#!/usr/bin/env python
"""substrate_rotamer_sample.py — rigid-body rotation of a substrate ligand
about a chosen reactive bond, with constrained energy minimization at each
sample.

Use case: TS-like input where the substrate orientation is wrong (heavy
clashes with active-site residues) but the reactive distances are correct.
We rotate the substrate around the P-Onuc axis (or any specified axis) by N
angles, run a constrained relax for each, and pick the lowest-energy /
fewest-clash pose.

Inputs:
  --input PDB
  --substrate-resname YYL --substrate-chain Z
  --axis-atoms NAME1:NAME2  (default P1:O1)
  --rotate-set NAME [NAME ...]  atoms to rotate about the axis (default:
                                  all substrate atoms NOT on the axis line)
  --keep-fixed NAMES            atoms to NOT rotate (must include axis atoms)
  --angles N (e.g. --angles 12 → 0..330° in 30° increments)
  --p-name --nuc-name --lg-name --nonbridge-names
  --b-flex-resids ... --a-fix-resids ...
  --use-fixbondlength
  --model --device --charge --fmax --max-steps

Outputs (per angle):
  OUTDIR/rot_<angle>/relaxed.pdb
  OUTDIR/rot_<angle>/manifest.json
And a top-level OUTDIR/rotamer_summary.json ranking all angles.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--substrate-resname", default="YYL")
    p.add_argument("--substrate-chain", default="Z")
    p.add_argument("--axis-atoms", default="P1:O1",
                   help="Two atom names defining the rotation axis (substrate).")
    p.add_argument("--p-name", default="P1")
    p.add_argument("--nuc-name", default="O1")
    p.add_argument("--lg-name", default="O5")
    p.add_argument("--nonbridge-names", nargs="*", default=["O3"])

    # Rotation
    p.add_argument("--n-angles", type=int, default=12,
                   help="Number of evenly-spaced angles in [0, 360).")
    p.add_argument("--angle-min", type=float, default=0.0)
    p.add_argument("--angle-max", type=float, default=360.0)

    # Constraint specs (forwarded)
    p.add_argument("--b-flex-resids", nargs="+", type=int, default=[118, 119, 120, 121, 122, 123, 124])
    p.add_argument("--b-flex-chain", default="B")
    p.add_argument("--a-fix-resids", nargs="*", type=int, default=[])
    p.add_argument("--a-fix-chain", default="A")
    p.add_argument("--use-fixbondlength", action="store_true")

    # Calculator + opt
    p.add_argument("--model", default="mace-polar-m")
    p.add_argument("--device", default="cuda")
    p.add_argument("--charge", type=int, default=None)
    p.add_argument("--fmax", type=float, default=0.10)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--clash-cutoff-A", type=float, default=1.8)
    p.add_argument("--clash-cutoff-skip-A", type=float, default=0.5,
                   help="Per-rotamer minimum allowed pre-relax clash distance; "
                        "below this we skip the relax and consider the rotamer "
                        "invalid (atoms overlap too much for MACE to converge).")

    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def configure_logging(level: str):
    logging.basicConfig(level=getattr(logging, level.upper()),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    return logging.getLogger("rotamer")


def find_idx(bt, chain, resname, name):
    for i in range(len(bt)):
        if bt.chain_id[i] == chain and bt.res_name[i] == resname \
                and bt.atom_name[i] == name:
            return i
    raise KeyError(f"{chain}/{resname}/{name} not found")


def rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Right-handed rotation about unit vector axis by theta radians."""
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def count_inter_residue_clashes(positions, atoms_meta, cutoff):
    from scipy.spatial import cKDTree
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=cutoff)
    n = 0; min_d = 1e9
    for i, j in pairs:
        a, b = atoms_meta[i], atoms_meta[j]
        if a["chain"] == b["chain"] and a["resseq"] == b["resseq"]:
            continue
        if a["chain"] == b["chain"] and abs(a["resseq"] - b["resseq"]) == 1 \
                and {a["name"], b["name"]} == {"C", "N"}:
            continue
        if {a["element"].upper(), b["element"].upper()} & {"ZN"} \
                and {a["element"].upper(), b["element"].upper()} & {"O", "N", "S"}:
            continue
        d = float(np.linalg.norm(positions[i] - positions[j]))
        n += 1
        min_d = min(min_d, d)
    return n, min_d


def main(argv=None):
    args = parse_args(argv)
    log = configure_logging(args.log_level)
    args.out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from quantum_engine.io import load_structure, write_pdb
    from ase.constraints import FixAtoms, FixBondLengths
    from quantum_engine.calc import make_calc
    from quantum_engine.ops import opt as opt_op

    # ---- Load
    atoms_in, bt, charge_hint = load_structure(args.input)
    if args.charge is None:
        args.charge = charge_hint if charge_hint is not None else 0
    log.info(f"Atoms: {len(atoms_in)}, charge: {args.charge}")

    # ---- Identify axis + substrate set
    axis_n1, axis_n2 = args.axis_atoms.split(":")
    P_idx = find_idx(bt, args.substrate_chain, args.substrate_resname, args.p_name)
    Nuc_idx = find_idx(bt, args.substrate_chain, args.substrate_resname, args.nuc_name)
    LG_idx = find_idx(bt, args.substrate_chain, args.substrate_resname, args.lg_name)
    NonBr_idx = []
    for n in args.nonbridge_names:
        try:
            NonBr_idx.append(find_idx(bt, args.substrate_chain, args.substrate_resname, n))
        except KeyError:
            log.warning(f"Non-bridging {n!r} not found; skipping")

    axis_a = find_idx(bt, args.substrate_chain, args.substrate_resname, axis_n1)
    axis_b = find_idx(bt, args.substrate_chain, args.substrate_resname, axis_n2)

    sub_indices = [i for i in range(len(atoms_in))
                   if bt.chain_id[i] == args.substrate_chain
                   and bt.res_name[i] == args.substrate_resname]
    log.info(f"Substrate has {len(sub_indices)} atoms")

    # Rotation axis vector
    p0 = atoms_in.get_positions()[axis_a]
    p1 = atoms_in.get_positions()[axis_b]
    axis_vec = p1 - p0

    # Build constraints (same as ts_constrained_relax)
    is_b = bt.chain_id == args.b_flex_chain
    is_a = bt.chain_id == args.a_fix_chain
    is_ca = bt.atom_name == "CA"
    b_flex_set = set(args.b_flex_resids)
    a_fix_set = set(args.a_fix_resids)
    fix_mask = np.zeros(len(atoms_in), dtype=bool)
    for i in range(len(atoms_in)):
        if is_b[i] and is_ca[i] and bt.res_id[i] not in b_flex_set:
            fix_mask[i] = True
        if is_a[i] and is_ca[i] and bt.res_id[i] in a_fix_set:
            fix_mask[i] = True
    fix_constraint = FixAtoms(indices=np.where(fix_mask)[0].tolist()) \
        if fix_mask.sum() > 0 else None
    if args.use_fixbondlength:
        bond_pairs = [(P_idx, Nuc_idx), (P_idx, LG_idx)] + [(P_idx, n) for n in NonBr_idx]
    else:
        bond_pairs = []

    atoms_meta = [{
        "chain": str(bt.chain_id[i]),
        "resseq": int(bt.res_id[i]),
        "resname": str(bt.res_name[i]),
        "name": str(bt.atom_name[i]),
        "element": str(bt.element[i]),
    } for i in range(len(atoms_in))]

    angles = np.linspace(args.angle_min, args.angle_max, args.n_angles, endpoint=False)
    all_results = []
    base_positions = atoms_in.get_positions().copy()

    for ang_deg in angles:
        ang_rad = math.radians(float(ang_deg))
        log.info(f"--- Rotamer angle = {ang_deg:6.1f}° ---")
        # Apply rotation: substrate atoms NOT on the axis are rotated.
        # Axis pivot is the first axis atom (P1).
        new_pos = base_positions.copy()
        R = rotation_matrix(axis_vec, ang_rad)
        for idx in sub_indices:
            if idx == axis_a or idx == axis_b:
                continue   # keep axis atoms fixed
            v = base_positions[idx] - p0
            new_pos[idx] = p0 + R @ v

        # Build new ASE Atoms with these positions
        atoms = atoms_in.copy()
        atoms.set_positions(new_pos)

        # Pre-relax clash check
        n_clash_before, min_d = count_inter_residue_clashes(
            new_pos, atoms_meta, args.clash_cutoff_A)
        log.info(f"  pre-relax clashes: {n_clash_before}, min_dist={min_d:.3f}Å")

        # Skip rotamers that have insanely overlapping atoms (MACE will diverge)
        if min_d < args.clash_cutoff_skip_A:
            log.warning(f"  SKIP: min_dist {min_d:.3f}Å < {args.clash_cutoff_skip_A}Å")
            all_results.append({
                "angle_deg": float(ang_deg),
                "skipped": True,
                "reason": f"min_dist={min_d:.3f}Å too low",
                "clashes_before": n_clash_before,
            })
            continue

        # Calculator + constraints
        atoms.info["charge"] = args.charge
        calc = make_calc(model=args.model, device=args.device, charge=args.charge)
        atoms.calc = calc
        constraints = []
        if fix_constraint is not None:
            constraints.append(fix_constraint)
        if bond_pairs:
            constraints.append(FixBondLengths(bond_pairs))
        if constraints:
            atoms.set_constraint(constraints)

        e0 = float(atoms.get_potential_energy())
        rot_dir = args.out / f"rot_{int(ang_deg):03d}"
        rot_dir.mkdir(parents=True, exist_ok=True)

        try:
            res = opt_op.run(atoms, calc, rot_dir, constraint=None,
                             optimizer="lbfgs", fmax=args.fmax,
                             max_steps=args.max_steps)
        except Exception as e:
            log.error(f"  LBFGS failed: {e}")
            all_results.append({
                "angle_deg": float(ang_deg),
                "skipped": False,
                "lbfgs_failed": True,
                "error": str(e),
            })
            continue

        e_final = float(res["energy_eV"])
        n_clash_after, min_d_after = count_inter_residue_clashes(
            atoms.get_positions(), atoms_meta, args.clash_cutoff_A)
        out_pdb = rot_dir / "relaxed.pdb"
        write_pdb(atoms, bt, out_pdb,
                  total_charge=args.charge, energy_eV=e_final)
        manifest = {
            "angle_deg": float(ang_deg),
            "skipped": False,
            "energy_initial_eV": e0,
            "energy_final_eV": e_final,
            "delta_E_kcal_mol": (e_final - e0) * 23.060541945329334,
            "fmax_final": float(res["fmax_final"]),
            "n_steps": int(res.get("n_steps", -1)),
            "clashes_before": n_clash_before,
            "clashes_after": n_clash_after,
            "min_dist_after_A": min_d_after,
            "out_pdb": str(out_pdb),
        }
        (rot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        all_results.append(manifest)
        log.info(f"  E_final={e_final:.3f} eV, clashes_after={n_clash_after}, "
                 f"min_d_after={min_d_after:.3f}Å")

    # Rank
    all_results.sort(key=lambda r: (r.get("clashes_after", 1e9),
                                     r.get("energy_final_eV", 1e9)))
    summary = {
        "input": str(args.input),
        "out": str(args.out),
        "n_angles": args.n_angles,
        "angle_range_deg": [args.angle_min, args.angle_max],
        "results": all_results,
    }
    (args.out / "rotamer_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"Best (by clash, then energy):")
    for r in all_results[:3]:
        if not r.get("skipped"):
            log.info(f"  angle={r['angle_deg']:.1f}° E={r['energy_final_eV']:.3f} "
                     f"clashes={r['clashes_after']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
