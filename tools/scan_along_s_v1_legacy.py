#!/usr/bin/env python
"""scan_along_s.py — relaxed scan along s = d(P-Olg) − d(P-Onuc).

For each scan point s_i, pin (d_P_Onuc, d_P_Olg) at values that yield s_i with
a fixed sum d_P_Onuc + d_P_Olg = sum_target (default 4.25 Å, the literature
"associative" sum). FixInternals(36 CA-CA pairs) + FixBondLengths.

Output: per-point energies and structures, plus the TS guess at energy max.
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
    ap.add_argument("--s-min", type=float, default=-1.5)
    ap.add_argument("--s-max", type=float, default=1.5)
    ap.add_argument("--sum-target", type=float, default=4.25,
                    help="d(P-Onuc) + d(P-Olg) sum held fixed across scan")
    ap.add_argument("--n-points", type=int, default=11)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--p-name", default="P1");  ap.add_argument("--p-res", default="SUB")
    ap.add_argument("--nuc-name", default="O3"); ap.add_argument("--nuc-res", default="OHX")
    ap.add_argument("--lg-name", default="O7");  ap.add_argument("--lg-res", default="SUB")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    from polish_ts_v2 import write_pdb_from_template, parse_pdb
    from structure_validator import _coord
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
    P, ON, OL = find(args.p_name, args.p_res), find(args.nuc_name, args.nuc_res), find(args.lg_name, args.lg_res)
    print(f"P={P} Onuc={ON} Olg={OL}; CA count={len(ca_idx)}")

    s_values = np.linspace(args.s_min, args.s_max, args.n_points)
    print(f"scan s ∈ [{args.s_min:+.2f}, {args.s_max:+.2f}], {args.n_points} points; sum d_P_Onuc+d_P_Olg={args.sum_target}")

    results = []
    for k, s in enumerate(s_values):
        d_pn = (args.sum_target - s) / 2.0
        d_pl = (args.sum_target + s) / 2.0
        if d_pn < 1.4 or d_pl < 1.4:
            print(f"point {k}: d_pn={d_pn:.3f} or d_pl={d_pl:.3f} too short; skipping")
            continue

        print(f"\n--- point {k}: s={s:+.3f} target d_P_Onuc={d_pn:.3f} d_P_Olg={d_pl:.3f}")
        atoms = read(str(args.input))
        # Move ON, OL toward target
        pos = atoms.get_positions().copy()
        for ix, t in [(ON, d_pn), (OL, d_pl)]:
            v = pos[ix] - pos[P]
            d = float(np.linalg.norm(v))
            pos[ix] = pos[P] + (v / d) * t
        atoms.set_positions(pos)
        atoms.info['charge'] = args.charge
        atoms.calc = make_calc(args.model, device=args.device, charge=args.charge)

        ca_positions = atoms.get_positions()
        bonds = []
        for ii in range(len(ca_idx)):
            for jj in range(ii+1, len(ca_idx)):
                i, j = ca_idx[ii], ca_idx[jj]
                bonds.append([float(np.linalg.norm(ca_positions[i] - ca_positions[j])), [i, j]])
        atoms.set_constraint([
            FixInternals(bonds=bonds, epsilon=1e-7),
            FixBondLengths([(P, ON), (P, OL)]),
        ])

        opt = LBFGS(atoms, logfile=str(args.out / f"point_{k:02d}.log"))
        converged = opt.run(fmax=args.fmax, steps=args.max_steps)
        e = float(atoms.get_potential_energy())
        fmax_final = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
        d_pn_final = float(np.linalg.norm(atoms.positions[P] - atoms.positions[ON]))
        d_pl_final = float(np.linalg.norm(atoms.positions[P] - atoms.positions[OL]))

        # write template-based PDB
        pdb_path = args.out / f"point_{k:02d}_s{s:+.3f}.pdb"
        write_pdb_from_template(args.input, atoms.get_positions(), pdb_path,
                                extra_remarks=[
                                    f"REMARK QCB SCAN_POINT {k}",
                                    f"REMARK QCB s_target {s:+.4f}",
                                    f"REMARK QCB d_P_Onuc {d_pn_final:.4f}",
                                    f"REMARK QCB d_P_Olg {d_pl_final:.4f}",
                                    f"REMARK QCB ENERGY_eV {e:.4f}",
                                    f"REMARK QCB FMAX_eV_per_A {fmax_final:.6f}",
                                    f"REMARK QCB CONVERGED {bool(converged)}",
                                ])
        results.append({
            "point": k, "s_target": float(s),
            "d_P_Onuc": d_pn_final, "d_P_Olg": d_pl_final, "s_actual": d_pl_final - d_pn_final,
            "energy_eV": e, "fmax_eV_per_A": fmax_final, "converged": bool(converged),
            "pdb": str(pdb_path),
        })
        print(f"  E={e:.4f} eV  fmax={fmax_final:.4f}  d_pn={d_pn_final:.3f}  d_pl={d_pl_final:.3f}  conv={converged}")

    # Find energy max → TS guess
    if results:
        max_idx = int(np.argmax([r["energy_eV"] for r in results]))
        ts_guess = results[max_idx]
        print(f"\n*** ENERGY MAX: point {ts_guess['point']} s={ts_guess['s_target']:+.3f} E={ts_guess['energy_eV']:.4f} eV ***")
        print(f"    pdb: {ts_guess['pdb']}")

        # Compute relative-energy profile in kcal/mol from min
        min_e = min(r["energy_eV"] for r in results)
        EV = 23.0605
        profile = [{"s": r["s_actual"], "E_rel_kcal": (r["energy_eV"] - min_e) * EV, "pdb": r["pdb"]} for r in results]

        summary = {
            "input": str(args.input),
            "model": args.model,
            "charge": args.charge,
            "sum_target_A": args.sum_target,
            "scan_n_points": args.n_points,
            "ts_guess_index": max_idx,
            "ts_guess_pdb": ts_guess["pdb"],
            "ts_guess_s": ts_guess["s_actual"],
            "ts_guess_d_P_Onuc": ts_guess["d_P_Onuc"],
            "ts_guess_d_P_Olg": ts_guess["d_P_Olg"],
            "ts_guess_energy_eV": ts_guess["energy_eV"],
            "barrier_R_to_TS_kcal": (ts_guess["energy_eV"] - results[0]["energy_eV"]) * EV,
            "barrier_P_to_TS_kcal": (ts_guess["energy_eV"] - results[-1]["energy_eV"]) * EV,
            "scan_results": results,
            "energy_profile_kcal": profile,
        }
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nwrote: {args.out / 'summary.json'}")
        print(f"barrier R(s={results[0]['s_actual']:.2f}) → TS(s={ts_guess['s_actual']:.2f}) = {summary['barrier_R_to_TS_kcal']:.2f} kcal/mol")
    return 0

if __name__ == "__main__":
    sys.exit(main())
