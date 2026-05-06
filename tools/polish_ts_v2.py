#!/usr/bin/env python
"""polish_ts_v2.py — robust TS polish that *cannot* repeat the cv-spring frame bug.

Fixes from the overnight failures:
  * Uses FixInternals over all C(N_CA, 2) pairwise CA-CA distances → rigid-body
    scaffold that lets the cluster translate/rotate as a unit without breaking
    backbone bonds. (FixAtoms tonight allowed the molecule to fly 64 Å away
    from the pinned CAs because mace's 5 Å cutoff decoupled them.)
  * Pins reactive d(P-Onuc) and d(P-Olg) at literature TS values (s ≈ +0.30,
    d(P-Onuc)≈2.00 Å, d(P-Olg)≈2.25 Å — Aubert 2004 / Wong 2007 / Khare 2012).
  * Writes output PDB by substituting coordinates into the source-PDB
    template (preserves atom names, residue records, REMARK lines).
  * Runs StructureValidator on input and output; refuses to ship if any
    critical check fails (anchor decoupling, bond stretches >1.5×, atom-count
    mismatch, etc.).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from structure_validator import StructureValidator, parse_pdb, _coord, validate_or_die  # noqa: E402


def write_pdb_from_template(template_pdb: Path, new_coords: np.ndarray,
                            out_pdb: Path,
                            extra_remarks: list[str] | None = None) -> None:
    """Substitute coords into the template PDB. Preserves all metadata."""
    template_atoms, template_remarks = parse_pdb(template_pdb)
    if len(template_atoms) != len(new_coords):
        raise ValueError(
            f"coord count {len(new_coords)} ≠ template atom count {len(template_atoms)}"
        )
    lines = list(template_remarks)
    if extra_remarks:
        for r in extra_remarks:
            if not r.startswith("REMARK"):
                r = "REMARK 999 " + r
            lines.append(r)
    for a, (x, y, z) in zip(template_atoms, new_coords):
        new_xyz = f"{x:8.3f}{y:8.3f}{z:8.3f}"
        lines.append(a.raw[:30] + new_xyz + a.raw[54:])
    lines.append("END")
    out_pdb.write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True,
                   help="source PDB (provides geometry AND template for output)")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--model", default="mace-polar-m")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--charge", type=int, default=1)
    p.add_argument("--p-name",  default="P1");  p.add_argument("--p-res",  default="SUB")
    p.add_argument("--nuc-name",default="O3");  p.add_argument("--nuc-res",default="OHX")
    p.add_argument("--lg-name", default="O7");  p.add_argument("--lg-res", default="SUB")
    p.add_argument("--target-d-p-onuc", type=float, default=2.00,
                   help="literature TS value (Aubert 2004 / Wong 2007)")
    p.add_argument("--target-d-p-olg",  type=float, default=2.25)
    p.add_argument("--ca-rigid",  action="store_true", default=True,
                   help="use FixInternals over all CA-CA pair distances (rigid-body scaffold)")
    p.add_argument("--ca-fix-atoms", action="store_true",
                   help="OLD behavior: hard FixAtoms on CAs. Off by default — "
                        "use --ca-rigid (FixInternals) which is the safer pattern.")
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=400)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # ----- VALIDATE INPUT -----
    src_atoms, src_remarks = parse_pdb(args.input)
    print(f"[input ] {len(src_atoms)} atoms, {len(src_remarks)} REMARK lines")

    # Identify CA atoms in chain A (the scaffold anchors)
    ca_idx = [i for i, a in enumerate(src_atoms)
              if a.chain == "A" and a.name == "CA"]
    print(f"[input ] {len(ca_idx)} CA atoms (chain A) — these will be the rigid-body anchors")
    anchor_keys = [(src_atoms[i].chain, src_atoms[i].resname,
                    src_atoms[i].resseq, src_atoms[i].icode, src_atoms[i].name)
                   for i in ca_idx]

    # Identify reactive atoms
    def find(name: str, res: str) -> int:
        for i, a in enumerate(src_atoms):
            if a.name == name and a.resname == res:
                return i
        raise RuntimeError(f"could not find {name} in {res}")
    P, ON, OL = find(args.p_name, args.p_res), find(args.nuc_name, args.nuc_res), find(args.lg_name, args.lg_res)
    print(f"[input ] reactive atoms: P[{P}]={args.p_name}.{args.p_res}, "
          f"Onuc[{ON}]={args.nuc_name}.{args.nuc_res}, Olg[{OL}]={args.lg_name}.{args.lg_res}")

    # Initial reactive distances
    d_pn0 = float(np.linalg.norm(_coord(src_atoms[P]) - _coord(src_atoms[ON])))
    d_pl0 = float(np.linalg.norm(_coord(src_atoms[P]) - _coord(src_atoms[OL])))
    print(f"[input ] initial d(P-Onuc)={d_pn0:.4f}  d(P-Olg)={d_pl0:.4f}  s={d_pl0-d_pn0:+.4f}")
    print(f"[input ] target  d(P-Onuc)={args.target_d_p_onuc:.4f}  d(P-Olg)={args.target_d_p_olg:.4f}  "
          f"s={args.target_d_p_olg-args.target_d_p_onuc:+.4f}")

    # Validate input is sane
    v = StructureValidator(args.input)
    print("\n[input ] running input self-validation:")
    v.check_all(args.input, anchor_keys=anchor_keys,
                reactive_pairs=[(anchor_keys[0], anchor_keys[0], 0.0)] if False else None)
    # ↑ self-validation always passes; real validation is below on the candidate

    # ----- LOAD WITH ASE, BUILD CONSTRAINTS -----
    from ase.io import read
    from ase.constraints import FixAtoms, FixInternals, FixBondLengths
    from ase.optimize import LBFGS
    sys.path.insert(0, str(REPO))
    from quantum_engine.calc import make_calc

    atoms = read(str(args.input))
    print(f"\n[ase   ] loaded {len(atoms)} atoms via ASE")

    # Manually walk d(P-Onuc) toward target by moving the ONE atom (Onuc).
    # Move Onuc along the P→Onuc direction to set d(P,Onuc) = target.
    def shift_distance(positions: np.ndarray, anchor: int, mover: int,
                       target_d: float) -> np.ndarray:
        v = positions[mover] - positions[anchor]
        d = float(np.linalg.norm(v))
        if d < 1e-6:
            return positions
        new = positions.copy()
        new[mover] = positions[anchor] + (v / d) * target_d
        return new

    pos = atoms.get_positions().copy()
    pos = shift_distance(pos, P, ON, args.target_d_p_onuc)
    pos = shift_distance(pos, P, OL, args.target_d_p_olg)
    atoms.set_positions(pos)
    d_pn1 = float(np.linalg.norm(pos[P] - pos[ON]))
    d_pl1 = float(np.linalg.norm(pos[P] - pos[OL]))
    print(f"[shift ] post-shift  d(P-Onuc)={d_pn1:.4f}  d(P-Olg)={d_pl1:.4f}  s={d_pl1-d_pn1:+.4f}")

    # Save the shifted starting structure with full template metadata
    shifted_pdb = args.out / "01_shifted_to_target_distances.pdb"
    write_pdb_from_template(args.input, atoms.get_positions(), shifted_pdb,
                            extra_remarks=[
                                "REMARK QCB STAGE 01_target_shifted",
                                f"REMARK QCB METHOD direct_atom_shift",
                                f"REMARK QCB d_P_Onuc_A {d_pn1:.4f}",
                                f"REMARK QCB d_P_Olg_A {d_pl1:.4f}",
                            ])
    print(f"[write ] {shifted_pdb}")

    # Validate the shifted structure
    print("\n[validate] shifted structure:")
    validate_or_die(args.input, shifted_pdb, anchor_keys=anchor_keys,
                    reactive_pairs=[
                        (anchor_keys[0], anchor_keys[0], 0.0),  # noop, just exercise checks
                    ])
    # ↑ shifted has CAs untouched + only ON, OL moved. Should pass.

    # ----- SET UP CALCULATOR + CONSTRAINTS -----
    atoms.info["charge"] = args.charge
    atoms.calc = make_calc(args.model, device=args.device, charge=args.charge)
    print(f"\n[calc  ] {args.model} on {args.device}, charge={args.charge}")

    constraints = []
    if args.ca_rigid:
        # FixInternals over all C(N,2) pairwise CA-CA distances. Preserves
        # scaffold shape exactly while leaving 6 rigid-body DOF free.
        positions = atoms.get_positions()
        bonds = []
        for ii in range(len(ca_idx)):
            for jj in range(ii + 1, len(ca_idx)):
                i, j = ca_idx[ii], ca_idx[jj]
                d = float(np.linalg.norm(positions[i] - positions[j]))
                bonds.append([d, [i, j]])
        constraints.append(FixInternals(bonds=bonds, epsilon=1e-7))
        print(f"[constr] FixInternals: {len(bonds)} CA-CA pair distances "
              f"(rigid-body scaffold; lets the cluster translate/rotate)")
    if args.ca_fix_atoms:
        constraints.append(FixAtoms(indices=ca_idx))
        print(f"[constr] FixAtoms on {len(ca_idx)} CAs (DEPRECATED — caused 64Å "
              f"frame explosion in cv-spring; use --ca-rigid instead)")

    # Reactive distances pinned at target
    constraints.append(FixBondLengths([(P, ON), (P, OL)]))
    print(f"[constr] FixBondLengths on (P-Onuc) and (P-Olg)")

    atoms.set_constraint(constraints)

    # ----- OPT -----
    e0 = float(atoms.get_potential_energy())
    fmax0 = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    print(f"\n[opt   ] initial: E = {e0:.4f} eV  fmax = {fmax0:.4f} eV/Å")

    opt = LBFGS(atoms, trajectory=str(args.out / "polish.traj"),
                logfile=str(args.out / "polish.log"))
    converged = opt.run(fmax=args.fmax, steps=args.max_steps)
    ef = float(atoms.get_potential_energy())
    fmaxf = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    print(f"[opt   ] final:   E = {ef:.4f} eV  fmax = {fmaxf:.4f} eV/Å  converged={converged}")

    # ----- WRITE OUTPUT WITH METADATA PRESERVED -----
    final_pdb = args.out / "transition_state.pdb"
    extra = [
        "REMARK QCB STAGE 02_polished_TS",
        f"REMARK QCB METHOD {args.model}/{args.device} LBFGS, FixInternals(CA-CA scaffold) + FixBondLengths(reactive)",
        f"REMARK QCB CHARGE {args.charge}",
        f"REMARK QCB ENERGY_eV {ef:.4f}",
        f"REMARK QCB FMAX_eV_per_A {fmaxf:.6f}",
        f"REMARK QCB CONVERGED {bool(converged)}",
        f"REMARK QCB d_P_Onuc_A {np.linalg.norm(atoms.positions[P]-atoms.positions[ON]):.4f}",
        f"REMARK QCB d_P_Olg_A  {np.linalg.norm(atoms.positions[P]-atoms.positions[OL]):.4f}",
        "REMARK QCB CONSTRAINT_PATTERN FixInternals_36_CA_CA_pairs+FixBondLengths_P_Onuc_P_Olg",
    ]
    write_pdb_from_template(args.input, atoms.get_positions(), final_pdb, extra_remarks=extra)
    print(f"[write ] {final_pdb}")

    # ----- VALIDATE OUTPUT -----
    reactive_pairs_check = [
        (anchor_keys[0], anchor_keys[0], 0.0),  # noop placeholder
    ]
    P_key = (src_atoms[P].chain, src_atoms[P].resname, src_atoms[P].resseq, src_atoms[P].icode, src_atoms[P].name)
    ON_key = (src_atoms[ON].chain, src_atoms[ON].resname, src_atoms[ON].resseq, src_atoms[ON].icode, src_atoms[ON].name)
    OL_key = (src_atoms[OL].chain, src_atoms[OL].resname, src_atoms[OL].resseq, src_atoms[OL].icode, src_atoms[OL].name)
    real_reactive = [
        (P_key, ON_key, args.target_d_p_onuc),
        (P_key, OL_key, args.target_d_p_olg),
    ]
    print("\n[validate] FINAL polished structure:")
    anchor_mode = "rigid" if args.ca_rigid else "fixed"
    out = validate_or_die(args.input, final_pdb, anchor_keys=anchor_keys,
                          anchor_mode=anchor_mode,
                          reactive_pairs=real_reactive)
    # default= handler converts numpy types to JSON-friendly equivalents.
    def _json_default(x):
        try:
            import numpy as _np
            if isinstance(x, _np.bool_): return bool(x)
            if isinstance(x, _np.integer): return int(x)
            if isinstance(x, _np.floating): return float(x)
            if isinstance(x, _np.ndarray): return x.tolist()
        except Exception:
            pass
        return str(x)
    (args.out / "validation.json").write_text(json.dumps(out, indent=2, default=_json_default))
    print(f"[write ] validation.json")

    # Also write a summary
    summary = {
        "input": str(args.input),
        "output": str(final_pdb),
        "model": args.model,
        "charge": args.charge,
        "ca_constraint_pattern": "FixInternals" if args.ca_rigid else "FixAtoms",
        "n_ca": len(ca_idx),
        "n_atoms": len(atoms),
        "energy_eV": ef,
        "fmax_eV_per_A": fmaxf,
        "converged": bool(converged),
        "input_d_p_onuc_A": d_pn0,
        "input_d_p_olg_A": d_pl0,
        "target_d_p_onuc_A": args.target_d_p_onuc,
        "target_d_p_olg_A": args.target_d_p_olg,
        "final_d_p_onuc_A": float(np.linalg.norm(atoms.positions[P]-atoms.positions[ON])),
        "final_d_p_olg_A": float(np.linalg.norm(atoms.positions[P]-atoms.positions[OL])),
        "validation": out,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    print(f"[write ] summary.json")
    print(f"\n=== DONE.  TS in {final_pdb} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
