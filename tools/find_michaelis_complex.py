#!/usr/bin/env python
"""find_michaelis_complex.py — relax to a TRUE pre-reactive Michaelis complex.

Releases the FixBondLengths constraint on (P, Onuc) so the hydroxide can
migrate to its equilibrium distance (~4 Å in the literature MC), keeps
P-Olg pinned (the leaving group is still bonded), and lets the rigid-body
CA scaffold + free Zn coordination relax. Output is a candidate MC for
recomputing the activation barrier from a less-strained starting point.

Run from polished TS or from the IRC reactant XYZ; will walk *uphill*
in the bonding direction (Onuc moves AWAY from P) to find the well bottom.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="mace-polar-m")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--charge", type=int, default=1)
    ap.add_argument("--initial-d-p-onuc", type=float, default=4.0,
                    help="seed Onuc this far from P before relaxing")
    ap.add_argument("--target-d-p-olg", type=float, default=1.65,
                    help="leaving-group P-O bond, kept pinned (intact reactant)")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=600)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    from polish_ts_v2 import write_pdb_from_template, parse_pdb
    from ase.io import read
    from ase.constraints import FixInternals, FixBondLengths
    from ase.optimize import LBFGS
    from quantum_engine.calc import make_calc

    src_atoms, _ = parse_pdb(args.input)
    ca_idx = [i for i, a in enumerate(src_atoms) if a.chain == "A" and a.name == "CA"]

    def find(name, res):
        for i, a in enumerate(src_atoms):
            if a.name == name and a.resname == res:
                return i
        raise RuntimeError(f"missing {name} in {res}")
    P  = find("P1", "SUB")
    ON = find("O3", "OHX")
    OL = find("O7", "SUB")
    print(f"P={P} Onuc={ON} Olg={OL}; CA={len(ca_idx)} anchors")

    atoms = read(str(args.input))
    pos = atoms.get_positions().copy()
    # Move Onuc to initial-d distance (Michaelis-complex like)
    v = pos[ON] - pos[P]; d = float(np.linalg.norm(v))
    pos[ON] = pos[P] + (v/d) * args.initial_d_p_onuc
    # Move Olg to target (intact P-O bond)
    v = pos[OL] - pos[P]; d = float(np.linalg.norm(v))
    pos[OL] = pos[P] + (v/d) * args.target_d_p_olg
    atoms.set_positions(pos)
    atoms.info['charge'] = args.charge
    atoms.calc = make_calc(args.model, device=args.device, charge=args.charge)

    # Constraint: rigid CA scaffold + ONLY P-Olg pinned (Onuc free!)
    bonds = [
        [float(np.linalg.norm(pos[ca_idx[i]] - pos[ca_idx[j]])), [ca_idx[i], ca_idx[j]]]
        for i in range(len(ca_idx)) for j in range(i+1, len(ca_idx))
    ]
    atoms.set_constraint([
        FixInternals(bonds=bonds, epsilon=1e-7),
        FixBondLengths([(P, OL)]),  # Olg pinned (intact paraoxon); Onuc FREE
    ])

    e0 = float(atoms.get_potential_energy())
    fmax0 = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    print(f"\ninitial: E={e0:.4f} eV   fmax={fmax0:.4f}   d(P-Onuc)={args.initial_d_p_onuc:.3f} (free)")

    opt = LBFGS(atoms, trajectory=str(args.out / "mc.traj"),
                logfile=str(args.out / "mc.log"))
    converged = opt.run(fmax=args.fmax, steps=args.max_steps)
    ef = float(atoms.get_potential_energy())
    fmaxf = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    d_pn = float(np.linalg.norm(atoms.positions[P] - atoms.positions[ON]))
    d_pl = float(np.linalg.norm(atoms.positions[P] - atoms.positions[OL]))
    print(f"final  : E={ef:.4f} eV   fmax={fmaxf:.4f}   d(P-Onuc)={d_pn:.3f}   d(P-Olg)={d_pl:.3f}   conv={converged}")
    print(f"   ΔE relax = {(ef-e0)*23.0605:+.2f} kcal/mol")

    # Save MC PDB with metadata
    out_pdb = args.out / "michaelis_complex.pdb"
    write_pdb_from_template(args.input, atoms.get_positions(), out_pdb,
                            extra_remarks=[
                                "REMARK QCB STAGE michaelis_complex",
                                f"REMARK QCB METHOD {args.model}/{args.device} LBFGS, FixInternals(CA scaffold) + FixBondLengths(P-Olg only); Onuc free",
                                f"REMARK QCB CHARGE {args.charge}",
                                f"REMARK QCB ENERGY_eV {ef:.4f}",
                                f"REMARK QCB d_P_Onuc_A {d_pn:.4f}",
                                f"REMARK QCB d_P_Olg_A {d_pl:.4f}",
                                f"REMARK QCB FMAX_eV_per_A {fmaxf:.6f}",
                                f"REMARK QCB CONVERGED {bool(converged)}",
                            ])
    summary = {
        "input": str(args.input), "model": args.model, "charge": args.charge,
        "initial_d_p_onuc_A": args.initial_d_p_onuc,
        "target_d_p_olg_A": args.target_d_p_olg,
        "final_d_p_onuc_A": d_pn,
        "final_d_p_olg_A": d_pl,
        "energy_eV": ef,
        "fmax_eV_per_A": fmaxf,
        "converged": bool(converged),
        "energy_drop_kcal": (ef - e0) * 23.0605,
        "mc_pdb": str(out_pdb),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote: {out_pdb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
