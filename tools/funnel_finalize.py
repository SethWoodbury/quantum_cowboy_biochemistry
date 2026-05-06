#!/usr/bin/env python
"""funnel_finalize.py — collect crest_funnel staged outputs into a clean ``results/``.

Reads the staged output dir (00_input, 10_xtb_preopt, 20_crest,
30_gxtb_minimize, 40_water_relax) produced by ``crest_funnel.py`` and writes::

    results/
      SUMMARY.md              human report
      summary.json            machine-readable
      validation.json         per-conformer pass/fail
      01_source.pdb           original input, REMARKs preserved
      02_xtb_preopt.pdb       Stage A (no waters)
      03_C_refined_conf_NN.pdb  Stage C, per conformer (no waters)
      04_E_final_conf_NN.pdb    Stage E, per conformer (full system) ← deliverable

All PDBs are Kabsch-aligned to source frame using CA atoms (always frozen).
Source REMARKs are preserved + per-stage REMARK QCB lines are appended.
Per-atom Mulliken charges (when available) are written into the B-factor column.

If `--cleanup` is passed, intermediate XYZ files outside ``results/`` are deleted.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np


# ----- minimal PDB read/write -----------------------------------------------
class PdbAtom:
    __slots__ = ("raw", "serial", "name", "resname", "chain", "resseq",
                 "x", "y", "z", "element")

    def __init__(self, line: str):
        self.raw = line.rstrip("\n")
        self.serial = int(line[6:11])
        self.name = line[12:16].strip()
        self.resname = line[17:20].strip()
        self.chain = line[21:22].strip()
        self.resseq = int(line[22:26])
        self.x = float(line[30:38])
        self.y = float(line[38:46])
        self.z = float(line[46:54])
        self.element = line[76:78].strip()


def parse_pdb(path: Path) -> tuple[list[PdbAtom], list[str]]:
    atoms: list[PdbAtom] = []
    remarks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            atoms.append(PdbAtom(line))
        elif line.startswith("REMARK"):
            remarks.append(line.rstrip("\n"))
    return atoms, remarks


def write_enhanced_pdb(template_atoms: list[PdbAtom],
                       coords: np.ndarray,
                       out_path: Path,
                       source_remarks: list[str],
                       new_remarks: list[str],
                       per_atom_bfac: list[float] | None = None) -> None:
    assert len(template_atoms) == len(coords), \
        f"atom-count mismatch: template={len(template_atoms)} coords={len(coords)}"
    lines = []
    lines.extend(source_remarks)
    for r in new_remarks:
        if not r.startswith("REMARK"):
            r = "REMARK 999 " + r
        lines.append(r)
    for i, (a, (x, y, z)) in enumerate(zip(template_atoms, coords)):
        new_xyz = f"{x:8.3f}{y:8.3f}{z:8.3f}"
        line = a.raw[:30] + new_xyz + a.raw[54:80]
        if per_atom_bfac is not None and i < len(per_atom_bfac):
            # rewrite B-factor cols (61-66 1-idx) with the partial charge.
            # PDB format is %6.2f — clamp to that range.
            v = per_atom_bfac[i]
            v = max(-99.99, min(99.99, v))
            bf = f"{v:6.2f}"
            line = line[:60] + bf + line[66:]
        lines.append(line)
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n")


def read_xyz_coords(path: Path) -> np.ndarray:
    t = path.read_text().splitlines()
    n = int(t[0].strip())
    return np.array([list(map(float, ln.split()[1:4])) for ln in t[2:2 + n]])


# ----- Kabsch alignment ------------------------------------------------------
def kabsch_align(coords: np.ndarray, src_coords: np.ndarray,
                 anchor_idx_0: list[int]) -> tuple[np.ndarray, float]:
    P, Q = coords[anchor_idx_0], src_coords[anchor_idx_0]
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, S, Vt = np.linalg.svd(H)
    sign = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, sign]) @ U.T
    aligned = (coords - cP) @ R.T + cQ
    residual = float(np.linalg.norm(aligned[anchor_idx_0] - Q, axis=1).max())
    return aligned, residual


# ----- xtb output parsers ----------------------------------------------------
_RE_TOTAL_E = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s*Eh")


def parse_xtb_charges(charges_path: Path) -> list[float] | None:
    """xtb writes a `charges` file with one Mulliken charge per atom."""
    if not charges_path.exists():
        return None
    return [float(x) for x in charges_path.read_text().split()]


def parse_xtb_total_energy(out_path: Path) -> float:
    if not out_path.exists():
        return float("nan")
    m = _RE_TOTAL_E.search(out_path.read_text())
    return float(m.group(1)) if m else float("nan")


def parse_xtbopt_xyz_energy(xyz_path: Path) -> float:
    if not xyz_path.exists():
        return float("nan")
    try:
        comment = xyz_path.read_text().splitlines()[1]
        return float(comment.split()[1])
    except Exception:
        return float("nan")


# ----- validation ------------------------------------------------------------
def validate_conformer(coords_full: np.ndarray, src_full: np.ndarray,
                       part: dict, n_no_waters: int,
                       d_p_onuc_target: float, d_p_olg_target: float) -> dict:
    fix_idx_0 = [i - 1 for i in part["fix_indices_no_waters_1based"]]
    P0 = part["p_idx_no_waters_1based"] - 1
    O0 = part["onuc_idx_no_waters_1based"] - 1
    L0 = part["olg_idx_no_waters_1based"] - 1

    ca_disp = np.linalg.norm(coords_full[fix_idx_0] - src_full[fix_idx_0], axis=1)
    d_pn = float(np.linalg.norm(coords_full[P0] - coords_full[O0]))
    d_pl = float(np.linalg.norm(coords_full[P0] - coords_full[L0]))

    # water-pocket sanity
    waters_have = coords_full.shape[0] > n_no_waters
    if waters_have:
        waters = coords_full[n_no_waters:]
        nonw_heavy = coords_full[:n_no_waters]   # all of them; we just want any-near
        # nearest non-water atom for each water O (every 3rd atom is the O)
        water_o = waters[::3]
        diff = water_o[:, None, :] - nonw_heavy[None, :, :]
        nearest = np.linalg.norm(diff, axis=2).min(axis=1)
        water_min_dist = float(nearest.min())
        water_max_dist = float(nearest.max())
    else:
        water_min_dist = water_max_dist = float("nan")

    return {
        "ca_max_disp_A": float(ca_disp.max()),
        "ca_rmsd_A": float(np.sqrt((ca_disp ** 2).mean())),
        "d_p_onuc_A": d_pn,
        "d_p_onuc_drift": d_pn - d_p_onuc_target,
        "d_p_olg_A": d_pl,
        "d_p_olg_drift": d_pl - d_p_olg_target,
        "water_o_min_protein_dist_A": water_min_dist,
        "water_o_max_protein_dist_A": water_max_dist,
        "checks": {
            "ca_locked": float(ca_disp.max()) < 0.001,
            "p_onuc_within_0.05A": abs(d_pn - d_p_onuc_target) < 0.05,
            "p_olg_within_0.05A": abs(d_pl - d_p_olg_target) < 0.05,
            "no_water_in_vacuum": (water_min_dist > 1.5 and
                                    water_max_dist < 6.0) if waters_have else True,
            "no_water_clash": water_min_dist > 2.2 if waters_have else True,
        },
    }


# ----- main ------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True,
                    help="crest_funnel output dir (00_input/, 10_xtb_preopt/, …)")
    ap.add_argument("--cleanup", action="store_true",
                    help="after writing results/, delete the staged XYZ files")
    args = ap.parse_args()

    base = args.out
    src_pdb = base / "00_input/source.pdb"
    partition = json.loads((base / "00_input/partition.json").read_text())
    n_no_waters = partition["n_atoms_no_waters"]
    n_water_atoms = partition["n_water_atoms"]
    fix_idx_0 = [i - 1 for i in partition["fix_indices_no_waters_1based"]]

    src_atoms, src_remarks = parse_pdb(src_pdb)
    src_xyz_full = np.array([[a.x, a.y, a.z] for a in src_atoms])
    src_no_waters = src_atoms[:n_no_waters]
    src_xyz_no_waters = src_xyz_full[:n_no_waters]

    results = base / "results"
    results.mkdir(exist_ok=True)

    # 01 source — copy with REMARKs, ENABLE downstream apps to pick it up
    shutil.copy2(src_pdb, results / "01_source.pdb")

    # 02 Stage A
    a_xyz = base / "10_xtb_preopt/preopt.xyz"
    stage_a_meta = None
    if a_xyz.exists():
        coords = read_xyz_coords(a_xyz)
        eA = parse_xtbopt_xyz_energy(a_xyz)
        write_enhanced_pdb(
            src_no_waters, coords, results / "02_xtb_preopt.pdb",
            source_remarks=src_remarks,
            new_remarks=[
                f"REMARK QCB STAGE A_xtb_preopt",
                f"REMARK QCB METHOD GFN2-xTB ALPB-water --opt loose",
                f"REMARK QCB ENERGY_Eh {eA:.6f}",
                f"REMARK QCB CHARGE {partition['charge']}",
                f"REMARK QCB N_ATOMS {len(coords)}",
            ],
        )
        stage_a_meta = {"energy_Eh": eA, "n_atoms": len(coords),
                        "pdb": str(results / "02_xtb_preopt.pdb")}

    # Validation per conformer (Stages C, E)
    summary = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "source_pdb": partition["source"],
        "n_atoms_total": partition["n_atoms_total"],
        "n_no_waters": n_no_waters,
        "n_water_atoms": n_water_atoms,
        "charge": partition["charge"],
        "constraints": {
            "fix_atoms_1based": partition["fix_indices_no_waters_1based"],
            "fix_atom_count": partition["n_fix_atoms_no_waters"],
            "p_idx_1based": partition["p_idx_no_waters_1based"],
            "onuc_idx_1based": partition["onuc_idx_no_waters_1based"],
            "olg_idx_1based": partition["olg_idx_no_waters_1based"],
            "d_p_onuc_target_A": partition["d_p_onuc_A"],
            "d_p_olg_target_A": partition["d_p_olg_A"],
        },
        "stage_A_xtb_preopt": stage_a_meta,
        "conformers": [],
    }
    validation = {"per_conformer": []}

    c_root = base / "30_gxtb_minimize"
    e_root = base / "40_water_relax"

    conformers = sorted(c_root.glob("conf_*")) if c_root.exists() else []
    for c_dir in conformers:
        cid = c_dir.name           # "conf_01"
        cnum = int(cid.split("_")[1])
        c_xyz = c_dir / "xtbopt.xyz"
        if not c_xyz.exists():
            continue
        coords_c = read_xyz_coords(c_xyz)
        coords_c_aligned, _ = kabsch_align(coords_c, src_xyz_no_waters, fix_idx_0)
        e_gfn2 = parse_xtbopt_xyz_energy(c_xyz)
        e_gxtb = parse_xtb_total_energy(c_dir / "gxtb_sp" / "gxtb_sp.out")
        charges_c = parse_xtb_charges(c_dir / "charges")
        valid_c = validate_conformer(
            coords_c_aligned, src_xyz_no_waters, partition, n_no_waters,
            partition["d_p_onuc_A"], partition["d_p_olg_A"],
        )

        write_enhanced_pdb(
            src_no_waters, coords_c_aligned,
            results / f"03_C_refined_{cid}.pdb",
            source_remarks=src_remarks,
            new_remarks=[
                f"REMARK QCB STAGE C_xtb_refined+gxtb_sp",
                f"REMARK QCB METHOD GFN2-xTB tight + g-xTB SP (gas-phase)",
                f"REMARK QCB ENERGY_GFN2_Eh {e_gfn2:.6f}",
                f"REMARK QCB ENERGY_gxTB_SP_Eh {e_gxtb:.6f}",
                f"REMARK QCB CONF {cnum}",
                f"REMARK QCB CHARGE {partition['charge']}",
                f"REMARK QCB CA_max_disp_A {valid_c['ca_max_disp_A']:.4f}",
                f"REMARK QCB d_P_Onuc_A {valid_c['d_p_onuc_A']:.4f}",
                f"REMARK QCB d_P_Olg_A {valid_c['d_p_olg_A']:.4f}",
                "REMARK QCB B_FACTOR_COL=Mulliken_charge_au"
                if charges_c else "REMARK QCB B_FACTOR_COL=unset",
            ],
            per_atom_bfac=charges_c,
        )

        # Stage E full system
        e_dir = e_root / cid if e_root.exists() else None
        e_xyz = (e_dir / "xtbopt.xyz") if e_dir is not None else None
        e_meta = None
        if e_xyz and e_xyz.exists():
            coords_e = read_xyz_coords(e_xyz)
            assert len(coords_e) == n_no_waters + n_water_atoms, \
                f"Stage E XYZ has {len(coords_e)} atoms, expected " \
                f"{n_no_waters + n_water_atoms}"
            coords_e_aligned, _ = kabsch_align(coords_e, src_xyz_full, fix_idx_0)
            e_water = parse_xtbopt_xyz_energy(e_xyz)
            charges_e = parse_xtb_charges(e_dir / "charges")
            valid_e = validate_conformer(
                coords_e_aligned, src_xyz_full, partition, n_no_waters,
                partition["d_p_onuc_A"], partition["d_p_olg_A"],
            )

            write_enhanced_pdb(
                src_atoms, coords_e_aligned,
                results / f"04_E_final_{cid}.pdb",
                source_remarks=src_remarks,
                new_remarks=[
                    f"REMARK QCB STAGE E_water_relax_FINAL",
                    f"REMARK QCB METHOD GFN2-xTB tight ALPB-water; non-water $fix",
                    f"REMARK QCB ENERGY_Eh {e_water:.6f}",
                    f"REMARK QCB CONF {cnum}",
                    f"REMARK QCB CHARGE {partition['charge']}",
                    f"REMARK QCB CA_max_disp_A {valid_e['ca_max_disp_A']:.4f}",
                    f"REMARK QCB d_P_Onuc_A {valid_e['d_p_onuc_A']:.4f}",
                    f"REMARK QCB d_P_Olg_A {valid_e['d_p_olg_A']:.4f}",
                    f"REMARK QCB water_min_protein_dist_A "
                    f"{valid_e['water_o_min_protein_dist_A']:.2f}",
                    "REMARK QCB B_FACTOR_COL=Mulliken_charge_au"
                    if charges_e else "REMARK QCB B_FACTOR_COL=unset",
                ],
                per_atom_bfac=charges_e,
            )
            e_meta = {
                "energy_Eh": e_water,
                "validation": valid_e,
                "pdb": str(results / f"04_E_final_{cid}.pdb"),
            }
        else:
            valid_e = None

        summary["conformers"].append({
            "id": cnum,
            "stage_C": {
                "gfn2_energy_Eh": e_gfn2,
                "gxtb_sp_energy_Eh": e_gxtb,
                "validation": valid_c,
                "pdb": str(results / f"03_C_refined_{cid}.pdb"),
            },
            "stage_E": e_meta,
        })
        validation["per_conformer"].append({
            "id": cnum,
            "stage_C_checks": valid_c["checks"],
            "stage_E_checks": valid_e["checks"] if valid_e else None,
        })

    # Sort conformers by Stage E energy (lowest first); fallback gxtb_sp.
    def _sort_key(c):
        e = (c.get("stage_E") or {}).get("energy_Eh")
        if e is None or math.isnan(e):
            e = c["stage_C"]["gxtb_sp_energy_Eh"]
        return e if not math.isnan(e) else float("inf")

    summary["conformers"].sort(key=_sort_key)

    # ---- write summary.json + validation.json
    (results / "summary.json").write_text(json.dumps(summary, indent=2))
    (results / "validation.json").write_text(json.dumps(validation, indent=2))

    # ---- write SUMMARY.md
    md_lines = [f"# crest_funnel results — {Path(partition['source']).name}", ""]
    md_lines.append(f"Generated: `{summary['generated']}`")
    md_lines.append(f"Source: `{summary['source_pdb']}`")
    md_lines.append(f"Net charge: **{summary['charge']:+d}**     "
                    f"Atoms: {summary['n_atoms_total']} "
                    f"({n_no_waters} non-water + {n_water_atoms} water)")
    md_lines.append("")
    md_lines.append("## Constraints")
    md_lines.append(f"- `$fix atoms:` {summary['constraints']['fix_atom_count']} CA "
                    f"atom(s) at source positions")
    md_lines.append(f"- `$constrain distance:` "
                    f"P–Onuc={summary['constraints']['d_p_onuc_target_A']:.4f} Å, "
                    f"P–Olg={summary['constraints']['d_p_olg_target_A']:.4f} Å, "
                    f"force=1.0 a.u.")
    md_lines.append("")
    md_lines.append("## Stage A pre-opt")
    if stage_a_meta:
        md_lines.append(f"E = `{stage_a_meta['energy_Eh']:.6f} Eh` → "
                        f"`results/02_xtb_preopt.pdb`")
    md_lines.append("")
    md_lines.append("## Conformers (ranked by Stage E energy)")
    md_lines.append("")
    md_lines.append("| rank | conf | E_E_Eh | E_C_gxTB_SP_Eh | CA_Δ_Å | d(P-Onuc) | d(P-Olg) | water_min | checks |")
    md_lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|:--|")
    for rank, c in enumerate(summary["conformers"], start=1):
        e_e = (c.get("stage_E") or {}).get("energy_Eh")
        e_c = c["stage_C"]["gxtb_sp_energy_Eh"]
        v = (c.get("stage_E") or {}).get("validation") or c["stage_C"]["validation"]
        all_checks = v["checks"]
        ok = "✅" if all(all_checks.values()) else "⚠️ " + ",".join(
            k for k, ok in all_checks.items() if not ok)
        md_lines.append(
            f"| {rank} | {c['id']} | "
            f"{e_e if e_e is not None else float('nan'):.4f} | "
            f"{e_c:.4f} | "
            f"{v['ca_max_disp_A']:.4f} | "
            f"{v['d_p_onuc_A']:.4f} | "
            f"{v['d_p_olg_A']:.4f} | "
            f"{v.get('water_o_min_protein_dist_A', float('nan')):.2f} | "
            f"{ok} |"
        )
    md_lines.append("")
    md_lines.append("## Files")
    md_lines.append("- `01_source.pdb` — original input, REMARKs preserved")
    md_lines.append("- `02_xtb_preopt.pdb` — Stage A (no waters)")
    md_lines.append("- `03_C_refined_conf_NN.pdb` — Stage C (no waters)")
    md_lines.append("- `04_E_final_conf_NN.pdb` — Stage E (full system, FINAL)")
    md_lines.append("- `summary.json`, `validation.json` — machine-readable")
    md_lines.append("")
    md_lines.append("## How to view")
    md_lines.append("PyMOL:")
    md_lines.append("```")
    md_lines.append("cd " + str(results))
    md_lines.append("pymol 01_source.pdb 04_E_final_conf_01.pdb")
    md_lines.append("```")
    md_lines.append("B-factor column encodes per-atom Mulliken charge "
                    "(if available); color by it via `spectrum b, blue_white_red`.")

    (results / "SUMMARY.md").write_text("\n".join(md_lines) + "\n")

    print(f"wrote results/ with {len(summary['conformers'])} conformer(s)")
    print(f"  {results / 'SUMMARY.md'}")
    print(f"  {results / 'summary.json'}")
    print(f"  {results / 'validation.json'}")

    if args.cleanup:
        # remove the staged XYZ working files; keep PDBs and summaries
        for stage_dir in ("10_xtb_preopt", "20_crest", "30_gxtb_minimize",
                           "40_water_relax"):
            d = base / stage_dir
            if d.exists():
                for p in d.rglob("*"):
                    if p.is_file() and p.suffix in {".xyz", ".trj", ".log",
                                                     ".out", ".restart",
                                                     ".mdrestart", ".coord"}:
                        p.unlink()
        print("  cleaned intermediate XYZ/log files")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
