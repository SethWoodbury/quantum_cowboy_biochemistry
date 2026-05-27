#!/usr/bin/env python
"""Generate cheap restrained starting-point ensembles from a TS-guess PDB.

This is a pre-Sella funnel for enzyme/theozyme TS guesses:

* keep protein CAs fixed;
* restrain the two reactive distances near the supplied TS geometry;
* optionally apply weak Cartesian tethers to ligand heavy atoms and water O;
* run several short Langevin/annealing trajectories;
* polish the lowest-scoring frames with the same restraints;
* write ranked PDBs plus a JSON/CSV summary.

The goal is not to find a final TS. It is to cheaply discover whether the
substrate, sidechains, and waters have nearby lower-strain arrangements before
spending time on Sella/IRC.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from ase import Atoms, units
from ase.constraints import FixAtoms
from ase.io import write as ase_write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import FIRE, LBFGS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from quantum_engine.calc import make_calc
from quantum_engine.io import load_structure, write_pdb

log = logging.getLogger("sample_ts_starters")

PROTEIN_RES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "KCX",
}
WATER_RES = {"HOH", "WAT"}
METAL_RES = {"ZN", "ZN2", "MG", "MN", "FE", "CU", "NI", "CO", "CA", "NA", "K"}
NUCLEOPHILE_RES = {"OHX"}
AUXILIARY_RES = {"CO2"}


class HarmonicDistance:
    """Two-sided harmonic distance restraint between two atoms."""

    def __init__(self, i: int, j: int, target: float, k: float, fmax: float | None = None):
        self.i = int(i)
        self.j = int(j)
        self.target = float(target)
        self.k = float(k)
        self.fmax = None if fmax is None else float(fmax)

    def get_removed_dof(self, atoms):
        return 0

    def adjust_positions(self, atoms, new):
        return None

    def adjust_momenta(self, atoms, momenta):
        return None

    def adjust_forces(self, atoms, forces):
        rij = atoms.positions[self.j] - atoms.positions[self.i]
        dist = float(np.linalg.norm(rij))
        if dist < 1e-10:
            return
        delta = dist - self.target
        force_mag = -self.k * delta
        if self.fmax is not None:
            force_mag = float(np.clip(force_mag, -self.fmax, self.fmax))
        fvec = force_mag * rij / dist
        forces[self.j] += fvec
        forces[self.i] -= fvec

    def adjust_potential_energy(self, atoms):
        dist = atoms.get_distance(self.i, self.j)
        delta = dist - self.target
        raw_force = self.k * abs(delta)
        if self.fmax is None or raw_force <= self.fmax:
            return 0.5 * self.k * delta * delta
        d_cap = self.fmax / self.k
        return 0.5 * self.k * d_cap * d_cap + self.fmax * (abs(delta) - d_cap)

    def todict(self):
        return {
            "name": "HarmonicDistance",
            "kwargs": {
                "i": self.i,
                "j": self.j,
                "target": self.target,
                "k": self.k,
                "fmax": self.fmax,
            },
        }


class HarmonicCartesianInit:
    """Weakly tether selected atoms to their initial positions."""

    def __init__(self, indices: Iterable[int], reference_positions, k: float, fmax: float | None = None):
        self.indices = np.array(list(indices), dtype=int)
        self.reference_positions = np.array(reference_positions, dtype=float)
        self.k = float(k)
        self.fmax = None if fmax is None else float(fmax)

    def get_removed_dof(self, atoms):
        return 0

    def adjust_positions(self, atoms, new):
        return None

    def adjust_momenta(self, atoms, momenta):
        return None

    def adjust_forces(self, atoms, forces):
        if len(self.indices) == 0 or self.k <= 0:
            return
        disp = atoms.positions[self.indices] - self.reference_positions[self.indices]
        f = -self.k * disp
        if self.fmax is not None:
            norms = np.linalg.norm(f, axis=1)
            mask = norms > self.fmax
            if np.any(mask):
                f[mask] *= (self.fmax / norms[mask])[:, None]
        forces[self.indices] += f

    def adjust_potential_energy(self, atoms):
        if len(self.indices) == 0 or self.k <= 0:
            return 0.0
        disp = atoms.positions[self.indices] - self.reference_positions[self.indices]
        norms = np.linalg.norm(disp, axis=1)
        if self.fmax is None:
            return 0.5 * self.k * float(np.sum(norms * norms))
        d_cap = self.fmax / self.k
        e = 0.0
        for d in norms:
            if d <= d_cap:
                e += 0.5 * self.k * d * d
            else:
                e += 0.5 * self.k * d_cap * d_cap + self.fmax * (d - d_cap)
        return e

    def todict(self):
        return {
            "name": "HarmonicCartesianInit",
            "kwargs": {
                "indices": self.indices.tolist(),
                "reference_positions": self.reference_positions.tolist(),
                "k": self.k,
                "fmax": self.fmax,
            },
        }


@dataclass
class StructureIndex:
    res_names: np.ndarray
    atom_names: np.ndarray
    chain_ids: np.ndarray
    res_ids: np.ndarray
    elements: np.ndarray
    hetero: np.ndarray


def _bt_index(bt_struct, atoms: Atoms) -> StructureIndex:
    if bt_struct is None:
        raise ValueError("This sampler needs PDB annotations; load a PDB, not XYZ.")
    return StructureIndex(
        res_names=np.array([str(x) for x in bt_struct.res_name]),
        atom_names=np.array([str(x).strip() for x in bt_struct.atom_name]),
        chain_ids=np.array([str(x) for x in bt_struct.chain_id]),
        res_ids=np.array([int(x) for x in bt_struct.res_id]),
        elements=np.array(atoms.get_chemical_symbols()),
        hetero=np.array(bt_struct.hetero, dtype=bool),
    )


def _infer_ligand_name(ix: StructureIndex) -> str:
    candidates: list[str] = []
    for rn in ix.res_names[ix.hetero]:
        if (
            rn in WATER_RES
            or rn in METAL_RES
            or rn in PROTEIN_RES
            or rn in NUCLEOPHILE_RES
            or rn in AUXILIARY_RES
        ):
            continue
        if rn not in candidates:
            candidates.append(rn)
    if len(candidates) != 1:
        raise ValueError(f"Could not infer exactly one ligand residue; candidates={candidates}")
    return candidates[0]


def _infer_reactive_names(path: Path, ligand: str, nuc: str | None, center: str | None, lg: str | None):
    if nuc and center and lg:
        return center, nuc, lg
    m = re.search(r"__([A-Za-z0-9]+)nuc_([A-Za-z0-9]+)_([A-Za-z0-9]+)lg__", path.name)
    if m:
        return center or m.group(2), nuc or m.group(1), lg or m.group(3)
    from quantum_engine.data import BOND_BREAKING_DEFS
    defs = BOND_BREAKING_DEFS.get(ligand)
    if not defs:
        raise ValueError(
            "Could not infer reactive atom names from filename or BOND_BREAKING_DEFS; "
            "pass --center-atom/--nuc-atom/--lg-atom."
        )
    center_name = center or defs[0][0]
    nuc_name = nuc or next((d[1] for d in defs if d[3] == "attractive"), None)
    lg_name = lg or next((d[1] for d in defs if d[3] == "repulsive"), None)
    if not (center_name and nuc_name and lg_name):
        raise ValueError("Reactive atom names are incomplete")
    return center_name, nuc_name, lg_name


def _find_ligand_atom(ix: StructureIndex, ligand: str, atom_name: str) -> int:
    hits = np.where((ix.res_names == ligand) & (ix.atom_names == atom_name))[0]
    if len(hits) == 1:
        return int(hits[0])
    if len(hits) > 1:
        raise ValueError(f"Expected one {ligand}:{atom_name}, found {len(hits)}")

    ignored = WATER_RES | METAL_RES | PROTEIN_RES | AUXILIARY_RES
    fallback = np.where(
        ix.hetero
        & (ix.atom_names == atom_name)
        & ~np.isin(ix.res_names, list(ignored))
    )[0]
    if len(fallback) == 1:
        return int(fallback[0])
    raise ValueError(
        f"Expected one reactive atom named {atom_name} in ligand {ligand} "
        f"or a unique non-protein hetero residue, found {len(fallback)} fallback hits"
    )


def _constraint_indices(ix: StructureIndex, ligand: str, args) -> dict[str, list[int]]:
    heavy = ix.elements != "H"
    protein = np.isin(ix.res_names, list(PROTEIN_RES))
    waters = np.isin(ix.res_names, list(WATER_RES))
    ligand_mask = ix.res_names == ligand

    ca = np.where(protein & (ix.atom_names == "CA"))[0].tolist()
    ligand_heavy = np.where(ligand_mask & heavy)[0].tolist()
    water_o = np.where(waters & (ix.atom_names == "O"))[0].tolist()

    protein_heavy_anchor = []
    if args.protein_heavy_anchor_k > 0:
        protein_heavy_anchor = np.where(protein & heavy & (ix.atom_names != "CA"))[0].tolist()

    return {
        "ca": ca,
        "ligand_heavy": ligand_heavy,
        "water_o": water_o,
        "protein_heavy_anchor": protein_heavy_anchor,
    }


def _build_constraints(atoms: Atoms, idxs: dict[str, list[int]], p_idx: int, nuc_idx: int,
                       lg_idx: int, args) -> list:
    constraints: list = []
    if idxs["ca"]:
        constraints.append(FixAtoms(indices=idxs["ca"]))

    d_p_nuc = atoms.get_distance(p_idx, nuc_idx)
    d_p_lg = atoms.get_distance(p_idx, lg_idx)
    constraints.append(HarmonicDistance(
        p_idx, nuc_idx, d_p_nuc, args.reactive_k, fmax=args.reactive_fmax
    ))
    constraints.append(HarmonicDistance(
        p_idx, lg_idx, d_p_lg, args.reactive_k, fmax=args.reactive_fmax
    ))

    ref_pos = atoms.get_positions()
    if args.ligand_heavy_anchor_k > 0 and idxs["ligand_heavy"]:
        constraints.append(HarmonicCartesianInit(
            idxs["ligand_heavy"], ref_pos, args.ligand_heavy_anchor_k,
            fmax=args.anchor_fmax,
        ))
    if args.water_o_anchor_k > 0 and idxs["water_o"]:
        constraints.append(HarmonicCartesianInit(
            idxs["water_o"], ref_pos, args.water_o_anchor_k,
            fmax=args.anchor_fmax,
        ))
    if args.protein_heavy_anchor_k > 0 and idxs["protein_heavy_anchor"]:
        constraints.append(HarmonicCartesianInit(
            idxs["protein_heavy_anchor"], ref_pos, args.protein_heavy_anchor_k,
            fmax=args.anchor_fmax,
        ))
    return constraints


def _raw_energy(atoms: Atoms) -> float:
    tmp = atoms.copy()
    tmp.calc = atoms.calc
    tmp.info.update(atoms.info)
    tmp.set_constraint()
    return float(tmp.get_potential_energy())


def _metrics(atoms: Atoms, ref_positions: np.ndarray, idxs: dict[str, list[int]],
             p_idx: int, nuc_idx: int, lg_idx: int) -> dict:
    pos = atoms.get_positions()
    ligand_idx = idxs["ligand_heavy"]
    water_idx = idxs["water_o"]
    ligand_rmsd = None
    water_o_rmsd = None
    water_o_max = None
    if ligand_idx:
        d = pos[ligand_idx] - ref_positions[ligand_idx]
        ligand_rmsd = float(np.sqrt(np.mean(np.sum(d * d, axis=1))))
    if water_idx:
        d = pos[water_idx] - ref_positions[water_idx]
        per = np.sqrt(np.sum(d * d, axis=1))
        water_o_rmsd = float(np.sqrt(np.mean(per * per)))
        water_o_max = float(np.max(per))
    d_nuc = float(atoms.get_distance(p_idx, nuc_idx))
    d_lg = float(atoms.get_distance(p_idx, lg_idx))
    return {
        "d_p_nuc_A": d_nuc,
        "d_p_lg_A": d_lg,
        "cv_lg_minus_nuc_A": d_lg - d_nuc,
        "ligand_heavy_rmsd_A": ligand_rmsd,
        "water_o_rmsd_A": water_o_rmsd,
        "water_o_max_A": water_o_max,
    }


def _run_md_seed(seed: int, base_atoms: Atoms, calc, constraints: list, args) -> list[tuple[float, int, Atoms]]:
    atoms = base_atoms.copy()
    atoms.calc = calc
    atoms.info.update(base_atoms.info)
    atoms.set_constraint(constraints)

    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp, rng=rng)
    friction_ase = args.friction * 1e-3 / units.fs
    dyn = Langevin(atoms, args.timestep_fs * units.fs, temperature_K=args.temp,
                   friction=friction_ase, rng=rng)

    n_steps = int(args.md_ps * 1000 / args.timestep_fs)
    sample_interval = max(1, int(args.sample_every_fs / args.timestep_fs))
    frames: list[tuple[float, int, Atoms]] = []

    def snapshot():
        e = float(atoms.get_potential_energy())
        frame = atoms.copy()
        frame.calc = calc
        frame.info.update(atoms.info)
        frames.append((e, dyn.nsteps, frame))

    dyn.attach(snapshot, interval=sample_interval)

    if args.anneal_peak and args.anneal_peak > args.temp:
        half = max(1, n_steps // 2)
        for step in range(n_steps):
            if step < half:
                alpha = (step + 1) / half
                target_t = args.temp + alpha * (args.anneal_peak - args.temp)
            else:
                alpha = (step - half + 1) / max(1, n_steps - half)
                target_t = args.anneal_peak + alpha * (args.temp - args.anneal_peak)
            dyn.set_temperature(target_t)
            dyn.run(1)
    else:
        dyn.run(n_steps)

    snapshot()
    return frames


def _polish_candidate(rank: int, seed: int, step: int, atoms: Atoms, calc, constraints: list, args):
    atoms = atoms.copy()
    atoms.calc = calc
    atoms.info.update({"sample_seed": seed, "sample_step": step})
    atoms.set_constraint(constraints)
    opt_cls = FIRE if args.polish_optimizer == "fire" else LBFGS
    opt = opt_cls(atoms, logfile=None)
    opt.run(fmax=args.polish_fmax, steps=args.polish_steps)
    restrained = float(atoms.get_potential_energy())
    raw = _raw_energy(atoms)
    fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    return atoms, restrained, raw, fmax


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_pdb")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model", default="mace-mh")
    parser.add_argument("--head", default="omol")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--charge", type=int, default=None)
    parser.add_argument("--ligand", default=None)
    parser.add_argument("--center-atom", default=None)
    parser.add_argument("--nuc-atom", default=None)
    parser.add_argument("--lg-atom", default=None)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seed0", type=int, default=1000)
    parser.add_argument("--md-ps", type=float, default=2.0)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--sample-every-fs", type=float, default=100.0)
    parser.add_argument("--temp", type=float, default=300.0)
    parser.add_argument("--anneal-peak", type=float, default=450.0)
    parser.add_argument("--friction", type=float, default=5.0)
    parser.add_argument("--reactive-k", type=float, default=10.0,
                        help="Two-sided harmonic k for P-nuc and P-LG distances (eV/A^2)")
    parser.add_argument("--reactive-fmax", type=float, default=5.0)
    parser.add_argument("--ligand-heavy-anchor-k", type=float, default=0.03,
                        help="Weak initial-position tether on ligand heavy atoms (eV/A^2)")
    parser.add_argument("--water-o-anchor-k", type=float, default=0.02,
                        help="Weak initial-position tether on water oxygens; Hs remain free")
    parser.add_argument("--protein-heavy-anchor-k", type=float, default=0.0,
                        help="Optional weak tether on non-CA protein heavy atoms")
    parser.add_argument("--anchor-fmax", type=float, default=1.0)
    parser.add_argument("--pre-opt-steps", type=int, default=50)
    parser.add_argument("--pre-opt-fmax", type=float, default=0.2)
    parser.add_argument("--keep-frames", type=int, default=12)
    parser.add_argument("--polish-steps", type=int, default=150)
    parser.add_argument("--polish-fmax", type=float, default=0.05)
    parser.add_argument("--polish-optimizer", default="fire", choices=["fire", "lbfgs"])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    t0 = time.time()
    input_pdb = Path(args.input_pdb).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    atoms, bt_struct, charge_hint = load_structure(input_pdb)
    charge = args.charge if args.charge is not None else (charge_hint or 0)
    atoms.info["charge"] = charge
    ix = _bt_index(bt_struct, atoms)
    ligand = args.ligand or _infer_ligand_name(ix)
    center_name, nuc_name, lg_name = _infer_reactive_names(
        input_pdb, ligand, args.nuc_atom, args.center_atom, args.lg_atom
    )
    p_idx = _find_ligand_atom(ix, ligand, center_name)
    nuc_idx = _find_ligand_atom(ix, ligand, nuc_name)
    lg_idx = _find_ligand_atom(ix, ligand, lg_name)
    ref_positions = atoms.get_positions().copy()
    idxs = _constraint_indices(ix, ligand, args)

    log.info("Input: %s", input_pdb)
    log.info("Atoms=%d charge=%+d ligand=%s reactive=(%s:%d, %s:%d, %s:%d)",
             len(atoms), charge, ligand, center_name, p_idx, nuc_name, nuc_idx, lg_name, lg_idx)
    log.info("Initial distances: d(P-nuc)=%.3f A d(P-LG)=%.3f A",
             atoms.get_distance(p_idx, nuc_idx), atoms.get_distance(p_idx, lg_idx))
    log.info("Fixed CAs=%d ligand_heavy_tether=%d water_O_tether=%d protein_heavy_tether=%d",
             len(idxs["ca"]), len(idxs["ligand_heavy"]), len(idxs["water_o"]),
             len(idxs["protein_heavy_anchor"]))

    calc = make_calc(
        model=args.model, head=args.head, device=args.device,
        default_dtype=args.dtype, charge=charge,
    )
    atoms.calc = calc
    constraints = _build_constraints(atoms, idxs, p_idx, nuc_idx, lg_idx, args)
    atoms.set_constraint(constraints)

    if args.pre_opt_steps > 0:
        log.info("Pre-opt: FIRE fmax=%.3f steps=%d", args.pre_opt_fmax, args.pre_opt_steps)
        pre = FIRE(atoms, logfile=str(outdir / "pre_opt.log"))
        pre.run(fmax=args.pre_opt_fmax, steps=args.pre_opt_steps)

    all_frames: list[tuple[float, int, int, Atoms]] = []
    for n in range(args.seeds):
        seed = args.seed0 + n
        log.info("MD seed %d/%d: seed=%d", n + 1, args.seeds, seed)
        frames = _run_md_seed(seed, atoms, calc, constraints, args)
        all_frames.extend((e, seed, step, frame) for e, step, frame in frames)

    all_frames.sort(key=lambda x: x[0])
    selected = all_frames[: max(1, args.keep_frames)]
    traj_frames = [frame for _, _, _, frame in selected]
    if traj_frames:
        ase_write(str(outdir / "selected_prepolish.xyz"), traj_frames, format="extxyz")

    rows = []
    polished_dir = outdir / "ranked_pdbs"
    polished_dir.mkdir(exist_ok=True)
    for rank, (pre_e, seed, step, frame) in enumerate(selected, 1):
        log.info("Polish rank=%02d seed=%d step=%d pre_E=%.6f eV", rank, seed, step, pre_e)
        polished, e_restrained, e_raw, fmax = _polish_candidate(
            rank, seed, step, frame, calc, constraints, args
        )
        metrics = _metrics(polished, ref_positions, idxs, p_idx, nuc_idx, lg_idx)
        pdb_path = polished_dir / f"starter_rank{rank:02d}_seed{seed}_step{step}.pdb"
        write_pdb(polished, bt_struct, pdb_path, total_charge=charge, energy_eV=e_raw)
        xyz_path = polished_dir / f"starter_rank{rank:02d}_seed{seed}_step{step}.xyz"
        ase_write(str(xyz_path), polished, format="extxyz")
        row = {
            "rank": rank,
            "seed": seed,
            "md_step": step,
            "prepolish_restrained_eV": pre_e,
            "polished_restrained_eV": e_restrained,
            "polished_raw_eV": e_raw,
            "fmax_eV_A": fmax,
            "pdb": str(pdb_path),
            "xyz": str(xyz_path),
            **metrics,
        }
        rows.append(row)

    summary = {
        "input_pdb": str(input_pdb),
        "outdir": str(outdir),
        "model": args.model,
        "head": args.head,
        "charge": charge,
        "ligand": ligand,
        "reactive_atoms": {
            "center": {"name": center_name, "index0": p_idx, "serial": p_idx + 1},
            "nuc": {"name": nuc_name, "index0": nuc_idx, "serial": nuc_idx + 1},
            "lg": {"name": lg_name, "index0": lg_idx, "serial": lg_idx + 1},
        },
        "settings": vars(args),
        "elapsed_min": (time.time() - t0) / 60.0,
        "candidates": rows,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (outdir / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    log.info("Done. Wrote %d ranked starters to %s", len(rows), polished_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
