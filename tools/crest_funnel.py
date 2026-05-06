#!/usr/bin/env python
"""crest_funnel.py — cheap iterative TS-conformer sampling.

Pipeline (one PDB in, ranked structures out):
  0. Parse PDB; separate waters; identify CA / Zn / P-Onuc-Olg / charge.
  A. GFN2-xTB constrained pre-opt (no waters): fix CA + Zn, restrain P-Onuc,
     P-Olg distances. Stabilizes the starting point.
  B. CREST --nci with the same constraints (no waters). Conformer ensemble.
  C. g-xTB constrained opt of top-N CREST conformers. Re-rank by g-xTB energy.
  D. Re-insert original waters into each top g-xTB conformer.
  E. Water-only relaxation: $fix everything except water atoms, GFN2 opt.
  F. Write final ranked CSV + per-conformer PDB.

Designed for theozyme-style active-site cluster cuts where:
  - protein backbone CAs must NOT move (they encode the scaffold),
  - the reactive P-Onuc / P-Olg distances must stay near the TS guess,
  - waters are flexible bystanders.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _compute_workers(ncpu_total: int, n_jobs: int, min_threads_per_job: int = 4) -> tuple[int, int]:
    """Pack n_jobs across ncpu_total cores: workers × threads_per ≤ ncpu_total."""
    if n_jobs <= 0:
        return 0, 1
    workers = max(1, min(n_jobs, ncpu_total // min_threads_per_job))
    threads = max(1, ncpu_total // workers)
    return workers, threads

# ----- vendored binaries -----------------------------------------------------
QCB_ROOT = Path(__file__).resolve().parents[1]
XTB_BIN = QCB_ROOT / "deps/xtb/install/bin/xtb"
CREST_BIN = QCB_ROOT / "deps/crest/install/bin/crest"
GXTB_BIN = QCB_ROOT / "deps/g-xtb/install/xtb-6.7.1/bin/xtb"

log = logging.getLogger("crest_funnel")


# ----- PDB parsing -----------------------------------------------------------
@dataclass
class PdbAtom:
    serial: int           # original 1-based PDB serial
    record: str           # ATOM or HETATM
    name: str             # atom name, stripped
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    occ: float
    bfac: float
    element: str
    charge_field: str     # raw two-char charge tail, e.g. "2+", "1-", ""


def parse_pdb(path: Path) -> tuple[list[PdbAtom], list[str]]:
    atoms: list[PdbAtom] = []
    remarks: list[str] = []
    with path.open() as fh:
        for line in fh:
            if line.startswith(("ATOM  ", "HETATM")):
                rec = line[:6].strip()
                serial = int(line[6:11])
                name = line[12:16].strip()
                altloc = line[16:17]
                resname = line[17:20].strip()
                chain = line[21:22].strip()
                resseq = int(line[22:26])
                icode = line[26:27]
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                occ = float(line[54:60]) if line[54:60].strip() else 1.0
                bfac = float(line[60:66]) if line[60:66].strip() else 0.0
                element = line[76:78].strip()
                charge = line[78:80].strip() if len(line) >= 80 else ""
                atoms.append(PdbAtom(
                    serial=serial, record=rec, name=name, altloc=altloc,
                    resname=resname, chain=chain, resseq=resseq, icode=icode,
                    x=x, y=y, z=z, occ=occ, bfac=bfac, element=element,
                    charge_field=charge,
                ))
            elif line.startswith("REMARK"):
                remarks.append(line.rstrip("\n"))
    return atoms, remarks


def write_xyz(atoms: list[PdbAtom], out_path: Path, comment: str = "") -> None:
    lines = [f"{len(atoms)}", comment]
    for a in atoms:
        lines.append(f"{a.element:<2s} {a.x:>14.8f} {a.y:>14.8f} {a.z:>14.8f}")
    out_path.write_text("\n".join(lines) + "\n")


# ----- constraint-file generation -------------------------------------------
def write_xtb_constraints(
    out_path: Path,
    fix_atoms: list[int],
    distance_constraints: list[tuple[int, int, float]],
    fix_force: float = 1.0,
) -> None:
    """Write an xtb-format constraint file.

    fix_atoms: 1-based xtb atom indices to freeze entirely.
    distance_constraints: list of (i, j, target_in_angstrom).
    """
    lines = []
    if distance_constraints:
        lines.append("$constrain")
        lines.append(f"  force constant={fix_force:.3f}")
        for i, j, d in distance_constraints:
            lines.append(f"  distance: {i}, {j}, {d:.4f}")
    if fix_atoms:
        lines.append("$fix")
        lines.append(f"  atoms: {_compact_indices(sorted(fix_atoms))}")
    lines.append("$end")
    out_path.write_text("\n".join(lines) + "\n")


def _compact_indices(idx: list[int]) -> str:
    """[1,2,3,5,6,9] -> '1-3,5-6,9'."""
    if not idx:
        return ""
    out = []
    start = prev = idx[0]
    for x in idx[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = x
    out.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(out)


# ----- structure partitioning ------------------------------------------------
@dataclass
class Partition:
    no_waters: list[PdbAtom]            # everything except HOH
    waters: list[PdbAtom]               # only HOH
    fix_indices_no_waters: list[int]    # 1-based, into no_waters list (CA + Zn)
    p_idx_no_waters: int                # 1-based serial of P1 in no_waters
    onuc_idx_no_waters: int             # 1-based serial of nucleophile O
    olg_idx_no_waters: int              # 1-based serial of leaving-group O
    d_p_onuc: float
    d_p_olg: float
    charge: int


def partition_pdb(atoms: list[PdbAtom], charge: int, freeze_zn: bool = False) -> Partition:
    no_waters: list[PdbAtom] = []
    waters: list[PdbAtom] = []
    for a in atoms:
        if a.resname in ("HOH", "WAT"):
            waters.append(a)
        else:
            no_waters.append(a)

    # Build (chain,resseq,resname,name) -> index_in_no_waters (1-based)
    p_idx = onuc_idx = olg_idx = -1
    fix_indices: list[int] = []
    for i, a in enumerate(no_waters, start=1):
        # CA in chain A protein residues
        if a.chain == "A" and a.name == "CA":
            fix_indices.append(i)
        # Zn metals — only if explicitly requested. Default: let the QM method
        # find the metal coordination geometry (both GFN2-xTB and g-xTB
        # parametrize Zn(II) and handle five-coordinate phosphoryl TS well).
        if freeze_zn and a.resname == "ZN2" and a.element == "ZN":
            fix_indices.append(i)
        # P1, O3(nucleophile, in OHX), O7(leaving-group, in SUB)
        if a.resname == "SUB" and a.name == "P1":
            p_idx = i
        if a.resname == "OHX" and a.name == "O3":
            onuc_idx = i
        if a.resname == "SUB" and a.name == "O7":
            olg_idx = i

    if p_idx < 0 or onuc_idx < 0 or olg_idx < 0:
        raise RuntimeError(
            f"failed to find reactive atoms: P1={p_idx}, O3(OHX)={onuc_idx}, "
            f"O7(SUB)={olg_idx}"
        )

    def dist(i: int, j: int) -> float:
        a, b = no_waters[i - 1], no_waters[j - 1]
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    return Partition(
        no_waters=no_waters,
        waters=waters,
        fix_indices_no_waters=fix_indices,
        p_idx_no_waters=p_idx,
        onuc_idx_no_waters=onuc_idx,
        olg_idx_no_waters=olg_idx,
        d_p_onuc=dist(p_idx, onuc_idx),
        d_p_olg=dist(p_idx, olg_idx),
        charge=charge,
    )


def charge_from_filename(path: Path) -> int:
    name = path.stem
    if "netCHG_plus_" in name:
        return int(name.split("netCHG_plus_")[1].split("_")[0])
    if "netCHG_minus_" in name:
        return -int(name.split("netCHG_minus_")[1].split("_")[0])
    raise ValueError(f"cannot infer charge from filename: {path.name}")


# ----- multi-frame XYZ (CREST output) ---------------------------------------
def split_multiframe_xyz(path: Path) -> list[tuple[float, list[str]]]:
    """Return list of (energy_in_Eh, xyz_block_lines_excluding_count_and_comment)."""
    text = path.read_text().splitlines()
    out: list[tuple[float, list[str]]] = []
    i = 0
    while i < len(text):
        if not text[i].strip():
            i += 1
            continue
        n = int(text[i].strip())
        comment = text[i + 1].strip()
        try:
            energy = float(comment.split()[0])
        except Exception:
            energy = float("nan")
        body = text[i + 2 : i + 2 + n]
        out.append((energy, body))
        i += 2 + n
    return out


def write_xyz_block(n_atoms: int, body: list[str], path: Path, comment: str = "") -> None:
    lines = [str(n_atoms), comment, *body]
    path.write_text("\n".join(lines) + "\n")


# ----- subprocess helpers ----------------------------------------------------
def run_cmd(cmd: list[str], cwd: Path, log_path: Path, env: dict | None = None,
            timeout: float | None = None) -> int:
    log.info("run: %s  (cwd=%s)", " ".join(str(c) for c in cmd), cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_path.open("w") as fh:
        fh.write("# " + " ".join(str(c) for c in cmd) + "\n")
        fh.write(f"# cwd={cwd}\n")
        fh.flush()
        proc = subprocess.run(
            [str(c) for c in cmd], cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT,
            env=env, timeout=timeout,
        )
    dt = time.time() - t0
    log.info("  → exit=%d  (%.1fs)", proc.returncode, dt)
    return proc.returncode


# ----- stages ---------------------------------------------------------------
def stage_0_input(src_pdb: Path, out_root: Path, freeze_zn: bool = False) -> Partition:
    log.info("--- Stage 0: parse + partition (freeze_zn=%s)", freeze_zn)
    d = out_root / "00_input"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_pdb, d / "source.pdb")
    atoms, _ = parse_pdb(src_pdb)
    chrg = charge_from_filename(src_pdb)
    part = partition_pdb(atoms, chrg, freeze_zn=freeze_zn)
    write_xyz(part.no_waters, d / "no_waters.xyz",
              comment=f"no_waters charge={part.charge}")
    write_xyz(part.waters, d / "waters.xyz",
              comment=f"waters_only n={len(part.waters)}")
    summary = {
        "source": str(src_pdb),
        "n_atoms_total": len(atoms),
        "n_atoms_no_waters": len(part.no_waters),
        "n_water_atoms": len(part.waters),
        "charge": part.charge,
        "p_idx_no_waters_1based": part.p_idx_no_waters,
        "onuc_idx_no_waters_1based": part.onuc_idx_no_waters,
        "olg_idx_no_waters_1based": part.olg_idx_no_waters,
        "d_p_onuc_A": part.d_p_onuc,
        "d_p_olg_A": part.d_p_olg,
        "n_fix_atoms_no_waters": len(part.fix_indices_no_waters),
        "fix_indices_no_waters_1based": part.fix_indices_no_waters,
    }
    (d / "partition.json").write_text(json.dumps(summary, indent=2))
    log.info("  total=%d  no_waters=%d  waters=%d  charge=%+d",
             len(atoms), len(part.no_waters), len(part.waters), part.charge)
    log.info("  P=%d  Onuc=%d  Olg=%d  d(P-Onuc)=%.3f  d(P-Olg)=%.3f",
             part.p_idx_no_waters, part.onuc_idx_no_waters, part.olg_idx_no_waters,
             part.d_p_onuc, part.d_p_olg)
    return part


def _make_constraints_no_waters(part: Partition, out: Path) -> None:
    write_xtb_constraints(
        out_path=out,
        fix_atoms=part.fix_indices_no_waters,
        distance_constraints=[
            (part.p_idx_no_waters, part.onuc_idx_no_waters, part.d_p_onuc),
            (part.p_idx_no_waters, part.olg_idx_no_waters, part.d_p_olg),
        ],
        fix_force=1.0,
    )


def stage_A_xtb_preopt(out_root: Path, part: Partition, ncpu: int,
                       solvent: str | None = "water") -> Path:
    log.info("--- Stage A: GFN2-xTB constrained pre-opt (no waters)")
    d = out_root / "10_xtb_preopt"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_root / "00_input/no_waters.xyz", d / "input.xyz")
    _make_constraints_no_waters(part, d / "constraints.inp")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = f"{ncpu},1"
    env["MKL_NUM_THREADS"] = str(ncpu)
    env["OMP_STACKSIZE"] = "4G"

    cmd = [str(XTB_BIN), "input.xyz",
           "--gfn", "2",
           "--opt", "loose",
           "--chrg", str(part.charge),
           "--input", "constraints.inp"]
    if solvent:
        cmd += ["--alpb", solvent]

    rc = run_cmd(cmd, cwd=d, log_path=d / "xtb.out", env=env, timeout=3600)
    if rc != 0:
        raise RuntimeError(f"Stage A xtb failed (rc={rc}); see {d/'xtb.out'}")

    # xtb writes xtbopt.xyz on success
    src = d / "xtbopt.xyz"
    if not src.exists():
        raise RuntimeError(f"Stage A: missing xtbopt.xyz under {d}")
    shutil.copy2(src, d / "preopt.xyz")
    log.info("  → %s", d / "preopt.xyz")
    return d / "preopt.xyz"


def stage_B_crest(out_root: Path, part: Partition, ncpu: int,
                  solvent: str | None = "water", preset: str = "mquick",
                  mdlen_ps: float | None = 3.0, rthr: float = 0.05,
                  mddump_fs: int = 200, opt_level: str = "crude",
                  walltime_s: int = 7200) -> Path:
    log.info("--- Stage B: CREST --nci (no waters)  preset=%s mdlen=%s rthr=%s",
             preset, mdlen_ps, rthr)
    d = out_root / "20_crest"
    d.mkdir(parents=True, exist_ok=True)
    # Wipe stale conformer files so a CREST that exits non-zero can't be
    # mistaken for a successful rerun.
    for stale in ("crest_conformers.xyz", "crest_rotamers.xyz", "crest_best.xyz",
                  "crest_dynamics.trj", "crestopt.log"):
        (d / stale).unlink(missing_ok=True)
    shutil.copy2(out_root / "10_xtb_preopt/preopt.xyz", d / "input.xyz")
    _make_constraints_no_waters(part, d / "constraints.inp")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = f"{ncpu},1"
    env["MKL_NUM_THREADS"] = str(ncpu)
    env["OMP_STACKSIZE"] = "4G"

    cmd = [str(CREST_BIN), "input.xyz",
           "--gfn2",
           "--chrg", str(part.charge),
           "--uhf", "0",
           "--cinp", "constraints.inp",
           "--nci",
           "--noreftopo",            # TS geometries have stretched bonds at cutoff edge
           "-rthr", str(rthr),       # looser RMSD → more unique conformers
           "-mddump", str(mddump_fs),# fewer trajectory dumps → fewer opt jobs
           "-O", opt_level,          # multilevel opt level (crude is the cheapest)
           "-T", str(ncpu)]
    if preset and preset != "none":
        cmd += [f"-{preset}"]        # -quick / -squick / -mquick
    if mdlen_ps is not None:
        cmd += ["-mdlen", str(mdlen_ps)]
    if solvent:
        cmd += ["--alpb", solvent]

    rc = run_cmd(cmd, cwd=d, log_path=d / "crest.out", env=env, timeout=walltime_s)
    if rc != 0:
        log.warning("CREST exit=%d (continuing if conformers were written)", rc)

    confs = d / "crest_conformers.xyz"
    if not confs.exists():
        raise RuntimeError(f"Stage B: missing crest_conformers.xyz under {d}")

    frames = split_multiframe_xyz(confs)
    log.info("  CREST returned %d conformers", len(frames))
    with (d / "crest_energies.csv").open("w") as fh:
        w = csv.writer(fh)
        w.writerow(["rank_in_crest", "energy_Eh"])
        for i, (e, _) in enumerate(frames, start=1):
            w.writerow([i, f"{e:.8f}"])
    return confs


def stage_C_gxtb(out_root: Path, part: Partition, top_n: int, ncpu: int,
                 solvent: str | None = "water", per_job_timeout: int = 1800,
                 min_threads_per_job: int = 4) -> Path:
    """Stage C — refine CREST conformers and rerank.

    g-xTB v0.1 silently ignores both ``$fix atoms:`` and ``$constrain atoms:``,
    and CREST's post-MTD multilevel opt strips ``--cinp`` constraints during
    its own ensemble re-optimization. So:

      1. Kabsch-align each CREST conformer back to source frame using CAs.
      2. Snap CA positions to source-frame coords (locks the scaffold).
      3. Run vanilla xtb-GFN2 ``--opt tight`` with ``$fix`` on those CAs
         and ``$constrain distance:`` on the reactive coords (xtb honors both).
      4. Run g-xTB ``--sp`` (gas phase) on the GFN2-optimized geometry to
         get a higher-quality energy for ranking.

    Conformers run in parallel: N workers × M threads ≤ ncpu.
    """
    d = out_root / "30_gxtb_minimize"
    d.mkdir(parents=True, exist_ok=True)

    # Prefer CREST's clustered conformers if there's actual diversity; otherwise
    # subsample from the raw MTD trajectory (CREST's multilevel ensemble opt
    # strips our constraints and collapses everything into one cluster, killing
    # the diversity that the constraint-aware MTD did sample).
    confs_path = out_root / "20_crest/crest_conformers.xyz"
    traj_path = out_root / "20_crest/crest_dynamics.trj"
    confs = split_multiframe_xyz(confs_path) if confs_path.exists() else []
    if len(confs) < 3 and traj_path.exists():
        traj = split_multiframe_xyz(traj_path)
        if len(traj) > len(confs):
            log.info("--- Stage C: CREST returned only %d clustered conformer(s); "
                     "subsampling %d frame(s) from raw MTD trajectory (%d total) "
                     "to recover sampled diversity",
                     len(confs), top_n, len(traj))
            step = max(1, len(traj) // max(1, top_n))
            confs = traj[::step][:top_n]

    n = min(top_n, len(confs))
    workers, threads_per = _compute_workers(ncpu, n, min_threads_per_job=min_threads_per_job)
    log.info("--- Stage C: refine %d conformer(s) — %d worker(s) × %d thread(s)",
             n, workers, threads_per)

    base_env = os.environ.copy()
    base_env["OMP_STACKSIZE"] = "4G"

    # CA anchor set (always frozen) — used to align CREST output back to source frame
    ca_idx_0 = [i - 1 for i, a in enumerate(part.no_waters, start=1)
                if a.chain == "A" and a.name == "CA"]
    src_no_waters_arr = np.array([[a.x, a.y, a.z] for a in part.no_waters])
    src_ca_arr = src_no_waters_arr[ca_idx_0]

    def kabsch_to_source(coords: np.ndarray) -> np.ndarray:
        cP = coords[ca_idx_0].mean(0); cQ = src_ca_arr.mean(0)
        H = (coords[ca_idx_0] - cP).T @ (src_ca_arr - cQ)
        U, S, Vt = np.linalg.svd(H)
        sign = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, sign]) @ U.T
        return (coords - cP) @ R.T + cQ

    def _run_one(k: int) -> dict:
        e_crest, body = confs[k]
        sub = d / f"conf_{k+1:02d}"
        sub.mkdir(parents=True, exist_ok=True)

        # parse element + xyz from CREST conformer body
        elems, xyz = [], []
        for ln in body:
            toks = ln.split()
            elems.append(toks[0])
            xyz.append([float(toks[1]), float(toks[2]), float(toks[3])])
        coords = np.array(xyz)

        # 1) Kabsch align onto source CA frame
        coords = kabsch_to_source(coords)
        # 2) Snap CA atoms exactly to source positions (locks scaffold for $fix)
        ca_drift_before = float(np.linalg.norm(coords[ca_idx_0] - src_ca_arr, axis=1).max())
        coords[ca_idx_0] = src_ca_arr

        # write the prepared starting structure
        with (sub / "input.xyz").open("w") as fh:
            fh.write(f"{len(elems)}\n")
            fh.write(f"crest_rank={k+1} ca_drift_pre_snap={ca_drift_before:.4f}A\n")
            for el, (x, y, z) in zip(elems, coords):
                fh.write(f"{el:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}\n")
        _make_constraints_no_waters(part, sub / "constraints.inp")

        env_local = base_env.copy()
        env_local["OMP_NUM_THREADS"] = f"{threads_per},1"
        env_local["MKL_NUM_THREADS"] = str(threads_per)

        # 3) vanilla xtb-GFN2 constrained opt (honors $fix + $constrain)
        cmd_opt = [str(XTB_BIN), "input.xyz",
                   "--gfn", "2",
                   "--opt", "tight",
                   "--chrg", str(part.charge),
                   "--input", "constraints.inp"]
        if solvent:
            cmd_opt += ["--alpb", solvent]
        rc_opt = run_cmd(cmd_opt, cwd=sub, log_path=sub / "xtb_opt.out",
                         env=env_local, timeout=per_job_timeout)
        opt_xyz = sub / "xtbopt.xyz"
        gfn2_energy = float("nan")
        ok_opt = (rc_opt == 0 and opt_xyz.exists())
        if ok_opt:
            txt = opt_xyz.read_text().splitlines()
            try:
                gfn2_energy = float(txt[1].split()[1])
            except Exception:
                pass

        # 4) g-xTB single point on the GFN2-relaxed geometry (gas-phase, no constraint needed)
        gxtb_sp_energy = float("nan")
        ok_sp = False
        if ok_opt:
            sp_dir = sub / "gxtb_sp"
            sp_dir.mkdir(exist_ok=True)
            shutil.copy2(opt_xyz, sp_dir / "input.xyz")
            cmd_sp = [str(GXTB_BIN), "input.xyz",
                      "--gxtb", "--sp",
                      "--chrg", str(part.charge)]
            rc_sp = run_cmd(cmd_sp, cwd=sp_dir, log_path=sp_dir / "gxtb_sp.out",
                            env=env_local, timeout=600)
            ok_sp = (rc_sp == 0)
            if ok_sp:
                # Pin to "TOTAL ENERGY  -1234.567 Eh" to avoid grabbing iteration counts
                import re as _re
                m = _re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s*Eh",
                               (sp_dir / "gxtb_sp.out").read_text())
                if m:
                    gxtb_sp_energy = float(m.group(1))

        return {
            "conf": k + 1,
            "crest_rank": k + 1,
            "crest_energy_Eh": e_crest,
            "ca_drift_pre_snap_A": ca_drift_before,
            "gxtb_ok": int(ok_opt),
            "gfn2_opt_energy_Eh": gfn2_energy,
            "gxtb_sp_energy_Eh": gxtb_sp_energy,
            "gxtb_sp_ok": int(ok_sp),
            "path": str(opt_xyz) if opt_xyz.exists() else "",
            # keep this column name for downstream Stage E:
            "gxtb_energy_Eh": gxtb_sp_energy if ok_sp else gfn2_energy,
        }

    if workers == 1:
        rows = [_run_one(k) for k in range(n)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_run_one, range(n)))

    rows.sort(key=lambda r: (1 - r["gxtb_ok"], r["gxtb_energy_Eh"]))
    csv_path = d / "ranked.csv"
    with csv_path.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("  → %s", csv_path)
    return csv_path


def _read_xyz(path: Path) -> tuple[list[str], list[tuple[float, float, float]]]:
    txt = path.read_text().splitlines()
    n = int(txt[0])
    elems, xyz = [], []
    for line in txt[2 : 2 + n]:
        toks = line.split()
        elems.append(toks[0])
        xyz.append((float(toks[1]), float(toks[2]), float(toks[3])))
    return elems, xyz


def stage_E_water_relax(out_root: Path, part: Partition, ncpu: int, top_keep: int,
                         solvent: str | None = "water",
                         per_job_timeout: int = 1800,
                         min_threads_per_job: int = 4) -> Path:
    """Re-insert source-frame waters AFTER mapping them into the post-CREST frame
    via a CA-anchored Kabsch transform — otherwise CREST's CMA translation puts
    the waters in vacuum, tens of Å from the protein.
    """
    log.info("--- Stage D+E: re-insert waters and relax")
    d = out_root / "40_water_relax"
    d.mkdir(parents=True, exist_ok=True)

    ranked = list(csv.DictReader((out_root / "30_gxtb_minimize/ranked.csv").open()))
    ranked = [r for r in ranked if r["gxtb_ok"] == "1"][:top_keep]

    n_no_water = len(part.no_waters)
    n_water = len(part.waters)
    fix_atoms = list(range(1, n_no_water + 1))  # freeze everything except waters

    workers, threads_per = _compute_workers(ncpu, len(ranked), min_threads_per_job=4)
    log.info("Stage E: %d conformer(s) — %d worker(s) × %d thread(s)",
             len(ranked), workers, threads_per)

    base_env = os.environ.copy()
    base_env["OMP_STACKSIZE"] = "4G"

    # Anchor set for source-frame -> post-opt-frame alignment.
    # CA atoms are *always* $fix'd, so they're identical in both frames. Using
    # them avoids depending on whether Zn was frozen in the run.
    anchor_idx_0 = [i - 1 for i, a in enumerate(part.no_waters, start=1)
                    if a.chain == "A" and a.name == "CA"]
    src_anchor_arr = np.array([[part.no_waters[i].x, part.no_waters[i].y, part.no_waters[i].z]
                               for i in anchor_idx_0])
    src_water_arr = np.array([[a.x, a.y, a.z] for a in part.waters])

    def kabsch_src_to(target_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cS = src_anchor_arr.mean(0); cT = target_arr.mean(0)
        H = (src_anchor_arr - cS).T @ (target_arr - cT)
        U, S, Vt = np.linalg.svd(H)
        sign = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, sign]) @ U.T
        return R, cS, cT

    def _run_one(r: dict) -> dict:
        opt_xyz = Path(r["path"])
        elems, xyz = _read_xyz(opt_xyz)
        coords_arr = np.array(xyz)
        sub = d / f"conf_{int(r['conf']):02d}"
        sub.mkdir(parents=True, exist_ok=True)

        target_anchor = coords_arr[anchor_idx_0]
        R, cS, cT = kabsch_src_to(target_anchor)
        waters_in_target_frame = (src_water_arr - cS) @ R.T + cT

        anchor_residual = float(np.linalg.norm(
            (src_anchor_arr - cS) @ R.T + cT - target_anchor, axis=1).max())

        combined_lines = [f"{n_no_water + n_water}",
                          f"combined no_water_from={opt_xyz} "
                          f"waters_kabsch_aligned anchor_resid={anchor_residual:.4f}"]
        for el, (x, y, z) in zip(elems, xyz):
            combined_lines.append(f"{el:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}")
        for a, (x, y, z) in zip(part.waters, waters_in_target_frame):
            combined_lines.append(f"{a.element:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}")
        (sub / "combined.xyz").write_text("\n".join(combined_lines) + "\n")

        write_xtb_constraints(
            out_path=sub / "constraints.inp",
            fix_atoms=fix_atoms,
            distance_constraints=[],
            fix_force=1.0,
        )

        env_local = base_env.copy()
        env_local["OMP_NUM_THREADS"] = f"{threads_per},1"
        env_local["MKL_NUM_THREADS"] = str(threads_per)

        cmd = [str(XTB_BIN), "combined.xyz",
               "--gfn", "2",
               "--opt", "tight",
               "--chrg", str(part.charge),
               "--input", "constraints.inp"]
        if solvent:
            cmd += ["--alpb", solvent]

        rc = run_cmd(cmd, cwd=sub, log_path=sub / "xtb.out", env=env_local,
                     timeout=per_job_timeout)
        opt = sub / "xtbopt.xyz"
        e_final = float("nan")
        ok = (rc == 0 and opt.exists())
        if ok:
            txt = opt.read_text().splitlines()
            try:
                e_final = float(txt[1].split()[1])
            except Exception:
                pass

        return {
            "conf": int(r["conf"]),
            "crest_rank": int(r["crest_rank"]),
            "gxtb_energy_Eh": float(r["gxtb_energy_Eh"]),
            "water_relax_ok": int(ok),
            "water_relax_energy_Eh": e_final,
            "relaxed_xyz": str(opt) if opt.exists() else "",
        }

    if workers == 1:
        rows = [_run_one(r) for r in ranked]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_run_one, ranked))

    rows.sort(key=lambda r: (1 - r["water_relax_ok"], r["water_relax_energy_Eh"]))
    csv_path = d / "ranked.csv"
    with csv_path.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # final summary
    final = out_root / "final_ranked.csv"
    with final.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("  → %s", final)
    return csv_path


# ----- driver ---------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdb", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--charge", type=int, default=None,
                   help="override; default infers from filename")
    p.add_argument("--ncpu", type=int, default=max(1, os.cpu_count() or 4))
    p.add_argument("--top-n", type=int, default=10,
                   help="g-xTB minimize this many top CREST conformers")
    p.add_argument("--top-keep", type=int, default=10,
                   help="water-relax this many top g-xTB conformers")
    p.add_argument("--solvent", default="water",
                   help="ALPB solvent or 'none'")
    p.add_argument("--crest-preset", default="mquick",
                   choices=["quick", "squick", "mquick", "none"],
                   help="reduced-settings preset (mquick=fastest)")
    p.add_argument("--crest-mdlen", type=float, default=3.0,
                   help="MTD time in ps PER MTD (CREST runs 6 MTDs)")
    p.add_argument("--crest-rthr", type=float, default=0.05,
                   help="CREST RMSD clustering threshold in Å (default loosened "
                        "from 0.125 to 0.05 to keep more conformers in our "
                        "heavily-constrained TS basin)")
    p.add_argument("--crest-mddump", type=int, default=200,
                   help="trajectory dump interval in fs (larger=fewer frames "
                        "to optimize → faster ensemble-opt phase)")
    p.add_argument("--crest-opt-level", default="crude",
                   choices=["crude", "vloose", "loose", "normal", "tight"],
                   help="multilevel-opt level inside CREST")
    p.add_argument("--crest-walltime", type=int, default=7200,
                   help="seconds before CREST is killed")
    p.add_argument("--min-threads-per-job", type=int, default=4,
                   help="min OMP_NUM_THREADS per parallel xtb/g-xtb worker "
                        "in Stages C+E")
    p.add_argument("--stages", default="0,A,B,C,DE",
                   help="comma list; subset of {0,A,B,C,DE}")
    p.add_argument("--freeze-zn", action="store_true",
                   help="also $fix the Zn atoms (default: only CA frozen, "
                        "Zn allowed to relax)")
    p.add_argument("--cleanup", action="store_true",
                   help="after writing results/, delete intermediate XYZ/log files")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(args.out / "pipeline.log", mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info("=" * 70)
    log.info("crest_funnel  pdb=%s  out=%s", args.pdb, args.out)
    log.info("ncpu=%d  top_n=%d  top_keep=%d  solvent=%s",
             args.ncpu, args.top_n, args.top_keep, args.solvent)

    solvent = None if args.solvent.lower() == "none" else args.solvent
    stages = set(s.strip() for s in args.stages.split(","))

    import dataclasses
    # Stage 0 always runs (we need the partition object); but we keep its files.
    part = stage_0_input(args.pdb, args.out, freeze_zn=args.freeze_zn)
    if args.charge is not None:
        part = dataclasses.replace(part, charge=args.charge)

    # Stage 0 hard validation gates — catch silently-broken parses before we
    # burn an hour of CREST on garbage.
    if not part.fix_indices_no_waters:
        raise RuntimeError("Stage 0: zero $fix anchors. CA-by-chain-A filter "
                           "matched nothing — check chain ID convention.")
    if part.d_p_onuc > 5.0 or part.d_p_olg > 5.0:
        raise RuntimeError(f"Stage 0: reactive distances suspicious "
                           f"d(P-Onuc)={part.d_p_onuc:.2f} d(P-Olg)={part.d_p_olg:.2f} "
                           "— wrong atom indices?")

    if "A" in stages:
        stage_A_xtb_preopt(args.out, part, args.ncpu, solvent=solvent)
    if "B" in stages:
        stage_B_crest(args.out, part, args.ncpu, solvent=solvent,
                      preset=args.crest_preset, mdlen_ps=args.crest_mdlen,
                      rthr=args.crest_rthr, mddump_fs=args.crest_mddump,
                      opt_level=args.crest_opt_level,
                      walltime_s=args.crest_walltime)
    if "C" in stages:
        stage_C_gxtb(args.out, part, top_n=args.top_n, ncpu=args.ncpu,
                     solvent=solvent,
                     min_threads_per_job=args.min_threads_per_job)
    if "DE" in stages:
        stage_E_water_relax(args.out, part, args.ncpu, top_keep=args.top_keep,
                             solvent=solvent,
                             min_threads_per_job=args.min_threads_per_job)

    # Auto-finalize: collect everything into results/ with PDBs + summary.
    try:
        finalize = QCB_ROOT / "tools" / "funnel_finalize.py"
        cmd = [sys.executable, str(finalize), "--out", str(args.out)]
        if args.cleanup:
            cmd.append("--cleanup")
        subprocess.run(cmd, check=True)
    except Exception as e:
        log.warning("auto-finalize failed: %s — run manually: "
                    "python tools/funnel_finalize.py --out %s", e, args.out)

    log.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
