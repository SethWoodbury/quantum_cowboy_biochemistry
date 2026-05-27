#!/usr/bin/env python
"""mlff_benchmark.py — single-snapshot wall-time / accuracy comparison
for MLFF aliases registered in `quantum_engine.site.MACE_MODELS`.

Per model:
  1. Load calc via ``quantum_engine.calc.make_calc(model, charge=...)``.
  2. Single-point evaluation (warmup the kernel + record peak GPU mem).
  3. Optional N-step LBFGS relax with FixBondLength on the P-Nuc / P-LG
     reactive triplet (so geometry stays comparable across models).

Outputs:
  OUTDIR/<model>/relaxed.pdb
  OUTDIR/<model>/manifest.json    (per-model summary)
  OUTDIR/comparison.tsv           (one row per model, all numbers)
  OUTDIR/comparison.json          (same data, machine-readable)

Why a separate script (not just ``qcb opt``)? We want to:
  * benchmark inference speed (sec/atom) on a fixed reference geometry
  * report Zn-O coordination distances + reactive-triplet distances
  * carry on if one model fails to load (skip + log instead of raise)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

log = logging.getLogger("mlff_benchmark")


@dataclass
class ModelBenchRow:
    model: str
    charge: int
    n_atoms: int
    status: str
    error: str = ""
    sp_time_s: float = float("nan")
    energy_sp_eV: float = float("nan")
    fmax_sp: float = float("nan")
    relax_time_s: float = float("nan")
    n_relax_steps: int = -1
    energy_final_eV: float = float("nan")
    fmax_final: float = float("nan")
    s_per_atom: float = float("nan")
    gpu_mem_peak_mb: float = float("nan")
    d_P_Nuc_final: float = float("nan")
    d_P_LG_final: float = float("nan")
    zn_oxygen_min_dist: float = float("nan")
    zn_oxygen_neighbours: str = ""


def parse_args(argv=None):
    p = argparse.ArgumentParser(__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True,
        help="Reference PDB (e.g. Frankenstein deep-min relaxed.pdb)")
    p.add_argument("--out", type=Path, required=True,
        help="Output directory (will be created)")
    p.add_argument("--charge", type=int, required=True,
        help="System net charge (charge-aware calcs need this)")
    p.add_argument("--models", nargs="+", required=True,
        help="MLFF aliases to benchmark (e.g. mace-mp uma-sm orb-mol)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float64",
        choices=["float64", "float32"])
    p.add_argument("--relax-steps", type=int, default=20,
        help="Number of LBFGS relax steps. 0 = skip relax (single-point only).")
    p.add_argument("--fmax", type=float, default=0.05,
        help="LBFGS convergence threshold (eV/Å)")
    p.add_argument("--substrate-resname", default="YYL",
        help="Substrate residue name (used to find P/Onuc/OLG atoms)")
    p.add_argument("--p-name", default="P1")
    p.add_argument("--nuc-name", default="O1")
    p.add_argument("--lg-name", default="O5")
    p.add_argument("--metal-element", default="Zn",
        help="Element symbol for the catalytic metal (Zn or Pd)")
    p.add_argument("--zn-cutoff-A", type=float, default=3.0,
        help="Zn-O distance cutoff for coordination report")
    p.add_argument("--freeze-triplet", action="store_true",
        help="If set, FixBondLength on P-Nuc and P-LG during relax")
    p.add_argument("--warmup", action="store_true",
        help="Run a 1-step throwaway force-eval before timing (reduces "
             "first-call CUDA overhead bias)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _setup_logger(level):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def _load_atoms_with_meta(pdb_path: Path):
    """Load PDB → ase.Atoms + (chain, resname, atom_name) tuples per atom."""
    from quantum_engine.io import load_structure
    atoms, bt_struct, charge_hint = load_structure(str(pdb_path))
    meta = []
    for i in range(len(atoms)):
        meta.append({
            "chain": str(bt_struct.chain_id[i]),
            "resseq": int(bt_struct.res_id[i]),
            "resname": str(bt_struct.res_name[i]),
            "name": str(bt_struct.atom_name[i]),
            "element": str(bt_struct.element[i]),
        })
    return atoms, meta


def _find_idx(meta, *, resname=None, name=None, element=None):
    out = []
    for i, m in enumerate(meta):
        if resname and m["resname"] != resname: continue
        if name and m["name"] != name: continue
        if element and m["element"] != element: continue
        out.append(i)
    return out


def _zn_neighbour_oxygens(atoms, meta, zn_cutoff: float, metal: str):
    """Return list of (zn_idx, o_idx, distance) within cutoff."""
    pos = atoms.get_positions()
    syms = atoms.get_chemical_symbols()
    zn_idx = [i for i, s in enumerate(syms) if s == metal]
    o_idx = [i for i, s in enumerate(syms) if s == "O"]
    out = []
    for z in zn_idx:
        for o in o_idx:
            d = float(np.linalg.norm(pos[z] - pos[o]))
            if d <= zn_cutoff:
                out.append((z, o, d))
    return out


def _torch_gpu_mem_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            torch.cuda.reset_peak_memory_stats()
            return float(mb)
    except Exception:
        pass
    return float("nan")


def _benchmark_one(args, model: str) -> ModelBenchRow:
    """Run one model; never raises (errors are recorded on the row)."""
    log.info("─" * 60)
    log.info("Model: %s", model)
    out_dir = args.out / model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    row = ModelBenchRow(model=model, charge=args.charge, n_atoms=0,
                        status="loading")

    # Reload atoms each model — keeps geometry comparable.
    try:
        atoms, meta = _load_atoms_with_meta(args.input)
        row.n_atoms = len(atoms)
    except Exception as exc:
        row.status = "error"
        row.error = f"PDB load: {exc}"
        return row

    # Charge-aware calcs read atoms.info["charge"] (mace-omol, mace-polar, …).
    atoms.info["charge"] = args.charge

    # ── Build calc ─────────────────────────────────────────────────
    try:
        from quantum_engine.calc import make_calc
        # mace-mh-1 ships multiple heads (omol, mp_pbe, etc.). Default
        # head choice would error out — pick `omol` for charge-aware /
        # all-element coverage (matches mace-mh-0 default in
        # MH_DEFAULT_HEADS).
        head_kw = {}
        if model in ("mace-mh", "mace-mh-1"):
            head_kw["head"] = "omol"
        calc = make_calc(
            model=model,
            charge=args.charge,
            device=args.device,
            default_dtype=args.dtype,
            **head_kw,
        )
        atoms.calc = calc
    except Exception as exc:
        row.status = "load_failed"
        row.error = f"make_calc: {type(exc).__name__}: {exc}"
        log.warning("  load failed: %s", row.error)
        return row

    # Reset peak GPU stats now that the model is loaded.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    # ── Optional warmup (untimed) ──────────────────────────────────
    if args.warmup:
        try:
            _ = atoms.get_potential_energy()
            _ = atoms.get_forces()
        except Exception as exc:
            row.status = "sp_failed"
            row.error = f"warmup: {type(exc).__name__}: {exc}"
            log.warning("  warmup failed: %s", row.error)
            return row

    # ── Single-point timing ────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        e_sp = float(atoms.get_potential_energy())
        f_sp = atoms.get_forces()
        sp_dt = time.perf_counter() - t0
        row.sp_time_s = sp_dt
        row.energy_sp_eV = e_sp
        row.fmax_sp = float(np.linalg.norm(f_sp, axis=1).max())
        row.s_per_atom = sp_dt / max(row.n_atoms, 1)
    except Exception as exc:
        row.status = "sp_failed"
        row.error = f"SP: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        log.warning("  SP failed: %s", row.error)
        return row

    # ── Optional relax ─────────────────────────────────────────────
    relax_atoms = atoms  # mutated in place
    if args.relax_steps > 0:
        try:
            from ase.constraints import FixBondLength
            from quantum_engine.opt import make_optimizer

            constraints = []
            if args.freeze_triplet:
                # Find P / Onuc / OLG atoms in the substrate.
                P_idx = _find_idx(meta, resname=args.substrate_resname,
                                  name=args.p_name)
                Nuc_idx = _find_idx(meta, resname=args.substrate_resname,
                                    name=args.nuc_name)
                LG_idx = _find_idx(meta, resname=args.substrate_resname,
                                   name=args.lg_name)
                if P_idx and Nuc_idx and LG_idx:
                    constraints.append(FixBondLength(P_idx[0], Nuc_idx[0]))
                    constraints.append(FixBondLength(P_idx[0], LG_idx[0]))
                else:
                    log.warning("  triplet atoms not found — running unconstrained")
            if constraints:
                relax_atoms.set_constraint(constraints)

            opt = make_optimizer(
                "ase-lbfgs",
                fmax=args.fmax,
                max_steps=int(args.relax_steps),
                outdir=out_dir,
            )
            t0 = time.perf_counter()
            res = opt.run(relax_atoms)
            row.relax_time_s = time.perf_counter() - t0
            row.n_relax_steps = int(res.n_steps)
            row.energy_final_eV = float(res.energy_eV)
            row.fmax_final = float(res.fmax_final)
        except Exception as exc:
            row.status = "relax_failed"
            row.error = f"relax: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            log.warning("  relax failed: %s", row.error)

    # ── Geometry diagnostics ──────────────────────────────────────
    try:
        P_idx = _find_idx(meta, resname=args.substrate_resname, name=args.p_name)
        Nuc_idx = _find_idx(meta, resname=args.substrate_resname, name=args.nuc_name)
        LG_idx = _find_idx(meta, resname=args.substrate_resname, name=args.lg_name)
        pos = relax_atoms.get_positions()
        if P_idx and Nuc_idx:
            row.d_P_Nuc_final = float(np.linalg.norm(pos[P_idx[0]] - pos[Nuc_idx[0]]))
        if P_idx and LG_idx:
            row.d_P_LG_final = float(np.linalg.norm(pos[P_idx[0]] - pos[LG_idx[0]]))

        zn_neigh = _zn_neighbour_oxygens(relax_atoms, meta, args.zn_cutoff_A,
                                          args.metal_element)
        if zn_neigh:
            zn_neigh.sort(key=lambda t: t[2])
            row.zn_oxygen_min_dist = zn_neigh[0][2]
            row.zn_oxygen_neighbours = ";".join(
                f"{meta[o_idx]['resname']}{meta[o_idx]['resseq']}"
                f":{meta[o_idx]['name']}={d:.3f}A" for _z, o_idx, d in zn_neigh[:5]
            )
    except Exception as exc:
        log.warning("  geometry diagnostics failed: %s", exc)

    row.gpu_mem_peak_mb = _torch_gpu_mem_mb()
    row.status = "ok"

    # ── Persist per-model artefact ─────────────────────────────────
    manifest = asdict(row)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    try:
        from ase.io import write as ase_write
        ase_write(str(out_dir / "relaxed.pdb"), relax_atoms)
    except Exception:
        pass

    log.info("  ✓ %s: SP %.3fs (%.2e s/atom), peak GPU %.1f MB",
             model, row.sp_time_s, row.s_per_atom, row.gpu_mem_peak_mb)
    return row


def main(argv=None):
    args = parse_args(argv)
    _setup_logger(args.log_level)
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[ModelBenchRow] = []
    for model in args.models:
        try:
            row = _benchmark_one(args, model)
        except Exception as exc:
            log.error("Unexpected error on %s: %s", model, exc)
            row = ModelBenchRow(model=model, charge=args.charge, n_atoms=0,
                                status="crash",
                                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        rows.append(row)

    # Comparison TSV
    fields = list(asdict(rows[0]).keys())
    tsv_lines = ["\t".join(fields)]
    for r in rows:
        d = asdict(r)
        tsv_lines.append("\t".join(str(d[k]) for k in fields))
    (args.out / "comparison.tsv").write_text("\n".join(tsv_lines) + "\n")
    (args.out / "comparison.json").write_text(json.dumps(
        [asdict(r) for r in rows], indent=2))

    log.info("=" * 60)
    log.info("DONE — %d models benchmarked", len(rows))
    log.info("Results: %s/comparison.{tsv,json}", args.out)

    return 0 if all(r.status == "ok" for r in rows) else 0  # never fail outright


if __name__ == "__main__":
    sys.exit(main())
