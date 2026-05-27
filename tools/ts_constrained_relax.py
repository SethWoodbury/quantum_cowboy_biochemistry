#!/usr/bin/env python
"""ts_constrained_relax.py — geometry minimization with FixAtoms + FixBondLength.

Generalizable de-clash / TS-prep relaxer. Drop-in companion to qcb opt that
adds frozen-distance constraints on the reactive triplet so a clashing input
can settle without disrupting the TS-like P-Onuc / P-Olg / P-Onon-bridging
geometry.

Multi-stage workflow (--stages flag):
  Stage A: backbone scaffold frozen  (CA + chain B all fixed, chain A free)
  Stage B: active-site free          (only chain B non-flexible CAs fixed)

Inputs:
  --input PDB
  --substrate-resname YYL  --substrate-chain Z
  --p-name P1 --nuc-name O1 --lg-name O5 --nonbridge-names O3
  --b-flex-resids "120 121 122"   chain B residues whose CA stays flexible
  --a-fix-resids ""               chain A residues whose CA stays fixed (else free)
  --hookean-pairs "P1:O1:2.24:1.0 P1:O5:1.68:1.0 P1:O3:1.52:1.0"
                                  optional one-sided Hookean springs
  --use-fixbondlength             freeze the TS bonds rigidly via FixBondLength
  --model mace-polar-m  --device cuda --charge -2
  --fmax 0.10  --max-steps 500
  --out OUTDIR

Output:
  OUTDIR/relaxed.pdb               final geometry
  OUTDIR/opt-lbfgs.log             LBFGS step log
  OUTDIR/opt-lbfgs.traj            trajectory
  OUTDIR/opt-summary.json          parsed summary
  OUTDIR/manifest.json             input echo + clash counts before/after
"""
from __future__ import annotations

import argparse
import json
import logging
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

    # Reactive triplet
    p.add_argument("--substrate-resname", default="YYL")
    p.add_argument("--substrate-chain", default="Z")
    p.add_argument("--p-name", default="P1")
    p.add_argument("--nuc-name", default="O1")
    p.add_argument("--lg-name", default="O5")
    p.add_argument("--nonbridge-names", nargs="*", default=["O3"],
                   help="Names of non-bridging P=O oxygens to freeze (default O3)")

    # Constraint specs
    p.add_argument("--b-flex-resids", nargs="+", type=int, default=[120, 121, 122],
                   help="Chain B residue IDs whose CA is FREE (default 120 121 122)")
    p.add_argument("--b-flex-chain", default="B",
                   help="Chain ID of the protein scaffold whose CAs are mostly fixed")
    p.add_argument("--a-fix-resids", nargs="*", type=int, default=[],
                   help="Chain A residue IDs whose CA stays FIXED (default: all chain A is free)")
    p.add_argument("--a-fix-chain", default="A",
                   help="Chain ID of the metalloenzyme active site")
    p.add_argument("--also-fix-residues-resnames", nargs="*", default=[],
                   help="Additional residue NAMES to fully fix (e.g. nothing).")
    p.add_argument("--use-fixbondlength", action="store_true",
                   help="Freeze the TS-relevant bond distances rigidly via FixBondLength.")
    p.add_argument("--hookean-pairs", nargs="*", default=None,
                   help="Optional Hookean restraints, NAME1:NAME2:r0:k tokens. Note "
                        "ASE Hookean is one-sided (above r0). Use --use-fixbondlength "
                        "for true bidirectional freeze on bonds.")

    # Calculator
    p.add_argument("--model", default="mace-polar-m")
    p.add_argument("--head", default=None)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--charge", type=int, default=None)

    # Optimizer
    p.add_argument("--optimizer", default="lbfgs", choices=["lbfgs", "fire"],
                   help="Legacy optimizer alias (deprecated — prefer "
                        "--optimizer-backend). Kept for backwards compat.")
    p.add_argument("--optimizer-backend", default=None,
                   help="Modular optimizer backend. Wins over --optimizer. "
                        "Choices: ase-lbfgs, ase-fire, ase-bfgs, "
                        "torch-sim-fire, torch-sim-lbfgs (stub). "
                        "torch-sim-fire does NOT support FixBondLength — "
                        "the script will refuse if --use-fixbondlength is on.")
    p.add_argument("--fmax", type=float, default=0.10)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--clash-cutoff-A", type=float, default=1.8)

    # Output
    p.add_argument("--output-pdb-name", default="relaxed.pdb")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def configure_logging(level: str) -> logging.Logger:
    logging.basicConfig(level=getattr(logging, level.upper()),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    return logging.getLogger("ts_constrained_relax")


def find_atom_index(bt_struct, chain: str, resname: str, atom_name: str) -> int:
    """Return atom index in biotite AtomArray with given (chain, resname, name)."""
    for i in range(len(bt_struct)):
        if bt_struct.chain_id[i] == chain and bt_struct.res_name[i] == resname \
                and bt_struct.atom_name[i] == atom_name:
            return i
    raise KeyError(f"Atom {chain}/{resname}/{atom_name} not found")


def find_atom_index_anyres(bt_struct, resname: str, atom_name: str) -> int:
    """Return atom index for first match by (resname, name) ignoring chain."""
    for i in range(len(bt_struct)):
        if bt_struct.res_name[i] == resname and bt_struct.atom_name[i] == atom_name:
            return i
    raise KeyError(f"Atom {resname}/{atom_name} not found")


def count_clashes(positions: np.ndarray, atoms_meta, cutoff: float) -> int:
    """Re-count inter-residue heavy-atom clashes (peptide bonds + Zn-coord excluded)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=cutoff)
    n = 0
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
        n += 1
    return n


def main(argv=None) -> int:
    args = parse_args(argv)
    log = configure_logging(args.log_level)
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- Load
    sys.path.insert(0, "/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry")
    from quantum_engine.io import load_structure, write_pdb
    from ase.constraints import FixAtoms, FixBondLengths, Hookean

    log.info(f"Reading input: {args.input}")
    atoms, bt_struct, charge_hint = load_structure(args.input)
    if args.charge is None:
        if charge_hint is not None:
            args.charge = charge_hint
            log.info(f"Using charge {args.charge} from PDB REMARK")
        else:
            args.charge = 0
            log.warning(f"No charge specified or in REMARK; defaulting to 0")
    log.info(f"Atoms: {len(atoms)}, charge: {args.charge}")

    if bt_struct is None:
        log.error("biotite parsing failed; cannot apply residue-based constraints")
        return 2

    # ---- Reactive atom indices (in ASE Atoms order)
    P_idx = find_atom_index(bt_struct, args.substrate_chain, args.substrate_resname, args.p_name)
    Nuc_idx = find_atom_index(bt_struct, args.substrate_chain, args.substrate_resname, args.nuc_name)
    LG_idx = find_atom_index(bt_struct, args.substrate_chain, args.substrate_resname, args.lg_name)
    NonBr_idx = []
    for nbn in args.nonbridge_names:
        try:
            NonBr_idx.append(find_atom_index(bt_struct, args.substrate_chain,
                                             args.substrate_resname, nbn))
        except KeyError:
            log.warning(f"Non-bridging atom {nbn!r} not found in substrate; skipping")
    coords = atoms.get_positions()
    d_P_Nuc = float(np.linalg.norm(coords[P_idx] - coords[Nuc_idx]))
    d_P_LG = float(np.linalg.norm(coords[P_idx] - coords[LG_idx]))
    log.info(f"Reactive triplet: P={P_idx} Nuc={Nuc_idx} LG={LG_idx}; "
             f"d(P-Nuc)={d_P_Nuc:.3f}, d(P-LG)={d_P_LG:.3f}")
    for k, idx in enumerate(NonBr_idx):
        d = float(np.linalg.norm(coords[P_idx] - coords[idx]))
        log.info(f"Non-bridging P-{args.nonbridge_names[k]}: {d:.3f} Å (idx {idx})")

    # ---- Build constraints
    constraints = []

    # FixAtoms: chain B CAs except flex residues + chain A CAs in --a-fix-resids
    fix_mask = np.zeros(len(atoms), dtype=bool)
    is_b_chain = bt_struct.chain_id == args.b_flex_chain
    is_a_chain = bt_struct.chain_id == args.a_fix_chain
    is_ca = bt_struct.atom_name == "CA"
    b_flex_set = set(args.b_flex_resids)
    a_fix_set = set(args.a_fix_resids)

    for i in range(len(atoms)):
        if is_b_chain[i] and is_ca[i] and bt_struct.res_id[i] not in b_flex_set:
            fix_mask[i] = True
        if is_a_chain[i] and is_ca[i] and bt_struct.res_id[i] in a_fix_set:
            fix_mask[i] = True

    n_fix_b = int((is_b_chain & is_ca).sum())
    n_b_flex = sum(int(((bt_struct.res_id == r) & is_b_chain & is_ca).sum())
                   for r in b_flex_set)
    n_b_fixed = int((fix_mask & is_b_chain & is_ca).sum())
    log.info(f"Fixed CAs in chain {args.b_flex_chain}: {n_b_fixed} / {n_fix_b} total CAs "
             f"(flex residues: {sorted(b_flex_set)})")
    log.info(f"Fixed CAs in chain {args.a_fix_chain}: {int((fix_mask & is_a_chain & is_ca).sum())} "
             f"(specified: {sorted(a_fix_set)})")

    if fix_mask.sum() > 0:
        constraints.append(FixAtoms(indices=np.where(fix_mask)[0].tolist()))
        log.info(f"FixAtoms: {fix_mask.sum()} atoms")

    # FixBondLength: TS triplet
    if args.use_fixbondlength:
        bond_pairs = [(P_idx, Nuc_idx), (P_idx, LG_idx)]
        for nbi in NonBr_idx:
            bond_pairs.append((P_idx, nbi))
        constraints.append(FixBondLengths(bond_pairs))
        log.info(f"FixBondLengths: {bond_pairs}")

    # Hookean restraints (optional, one-sided)
    if args.hookean_pairs:
        for spec in args.hookean_pairs:
            try:
                n1, n2, r0_str, k_str = spec.split(":")
                r0 = float(r0_str)
                k = float(k_str)
                # Find both atoms in substrate
                i1 = find_atom_index(bt_struct, args.substrate_chain,
                                     args.substrate_resname, n1)
                i2 = find_atom_index(bt_struct, args.substrate_chain,
                                     args.substrate_resname, n2)
                constraints.append(Hookean(a1=i1, a2=i2, rt=r0, k=k))
                log.info(f"Hookean: {n1}({i1}) <-> {n2}({i2}) rt={r0:.3f}, k={k}")
            except Exception as e:
                log.error(f"Failed to parse --hookean-pairs '{spec}': {e}")
                return 2

    # ---- Calculator
    from quantum_engine.calc import make_calc
    calc = make_calc(model=args.model, head=args.head, device=args.device,
                     charge=args.charge)
    atoms.info["charge"] = args.charge
    atoms.calc = calc

    # ---- Apply constraints
    if constraints:
        atoms.set_constraint(constraints)

    # ---- Pre-clash count
    atoms_meta = [{
        "chain": str(bt_struct.chain_id[i]),
        "resseq": int(bt_struct.res_id[i]),
        "resname": str(bt_struct.res_name[i]),
        "name": str(bt_struct.atom_name[i]),
        "element": str(bt_struct.element[i]),
    } for i in range(len(atoms))]
    n_clash_before = count_clashes(atoms.get_positions(), atoms_meta, args.clash_cutoff_A)
    log.info(f"Inter-residue clashes BEFORE relax: {n_clash_before} (cutoff {args.clash_cutoff_A}Å)")

    e0 = float(atoms.get_potential_energy())
    log.info(f"Initial energy: {e0:.6f} eV")

    # ---- Run optimization
    from quantum_engine.ops import opt as opt_op

    # Pick backend: explicit --optimizer-backend wins, else translate
    # the legacy --optimizer flag.
    backend = args.optimizer_backend
    if backend is None:
        backend = {"lbfgs": "ase-lbfgs", "fire": "ase-fire"}[args.optimizer]

    # Sanity: torch-sim FIRE has no constraint support, so refuse early
    # if the user combined it with --use-fixbondlength.
    if backend.startswith("torch-sim") and args.use_fixbondlength:
        raise SystemExit(
            f"--optimizer-backend {backend} does not support FixBondLength. "
            f"Use --optimizer-backend ase-lbfgs (default) or drop --use-fixbondlength."
        )

    log.info("Optimizer backend: %s (legacy --optimizer=%s)",
             backend, args.optimizer)
    res = opt_op.run(atoms, calc, args.out, constraint=None,
                     backend=backend, fmax=args.fmax,
                     max_steps=args.max_steps)
    # Note: we already set the constraint on atoms, so passing None to opt avoids overwrite

    e_final = float(res["energy_eV"])
    fmax_final = float(res["fmax_final"])
    log.info(f"Final energy: {e_final:.6f} eV, fmax: {fmax_final:.4f}")

    # Re-compute clash count
    n_clash_after = count_clashes(atoms.get_positions(), atoms_meta, args.clash_cutoff_A)
    log.info(f"Inter-residue clashes AFTER relax: {n_clash_after}")

    # Reactive distances post-relax
    d_P_Nuc_after = float(np.linalg.norm(atoms.positions[P_idx] - atoms.positions[Nuc_idx]))
    d_P_LG_after = float(np.linalg.norm(atoms.positions[P_idx] - atoms.positions[LG_idx]))

    # Output PDB
    out_pdb = args.out / args.output_pdb_name
    write_pdb(atoms, bt_struct, out_pdb,
              total_charge=args.charge, energy_eV=e_final)
    log.info(f"Wrote relaxed PDB: {out_pdb}")

    manifest = {
        "input": str(args.input),
        "out_pdb": str(out_pdb),
        "model": args.model,
        "charge": args.charge,
        "n_atoms": len(atoms),
        "n_atoms_fixed": int(fix_mask.sum()),
        "fix_bond_lengths": args.use_fixbondlength,
        "n_steps": int(res.get("n_steps", -1)),
        "fmax_final": fmax_final,
        "fmax_target": args.fmax,
        "energy_initial_eV": e0,
        "energy_final_eV": e_final,
        "delta_E_eV": e_final - e0,
        "delta_E_kcal_mol": (e_final - e0) * 23.060541945329334,
        "clashes_before": n_clash_before,
        "clashes_after": n_clash_after,
        "clash_cutoff_A": args.clash_cutoff_A,
        "d_P_Nuc_initial": d_P_Nuc,
        "d_P_Nuc_final": d_P_Nuc_after,
        "d_P_LG_initial": d_P_LG,
        "d_P_LG_final": d_P_LG_after,
        "P_idx": P_idx,
        "Nuc_idx": Nuc_idx,
        "LG_idx": LG_idx,
        "converged": bool(res.get("converged", False)),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(f"Wrote manifest: {args.out / 'manifest.json'}")

    return 0 if res.get("converged", False) else 1


if __name__ == "__main__":
    sys.exit(main())
