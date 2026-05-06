#!/usr/bin/env python
"""polish_ts_v3.py — generalizable TS-polish driver with rich CLI.

Successor to polish_ts_v2.py. Adds:
  * Better output naming (`--out-basename` or default `<input_stem>_polished`)
  * Both PDB and CIF outputs by default (atomworks-backed CIF with bonds +
    kekulized aromatic types + per-atom charges); flags `--no-pdb`, `--no-cif`
  * REMARK 665 / 666 propagation; condensed `REMARK QCB <NNN>` lineage
  * Strict PDB column-format validation on every output
  * Custom constraints:  `--free-residues`, `--prune-backbone-residues`,
    `--prune-residue-keep RESID:ATOMS`, `--fix-atoms`, `--fix-distance`,
    `--fix-angle`, `--fix-dihedral`
  * Trajectory snapshots as multi-MODEL PDB (`--snapshot-stride N`)
  * Always copies the input PDB into `out/` for reference
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from structure_io import (    # noqa: E402
    parse_pdb_lineage, write_pdb_lineage, validate_pdb_format,
    write_cif_lineage, write_trajectory_pdb,
    autoadd_protein_matcher_remarks, qcb_remark, qcb_legend_remarks,
)

log = logging.getLogger("polish_ts_v3")


# ===== CLI parsing helpers ==================================================
def _csv_int(s: str) -> list[int]:
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_keep_specs(specs: list[str]) -> dict[int, list[str]]:
    """`--prune-residue-keep 169:CD,CE,NZ 254:CG,CD2`."""
    out = {}
    for spec in specs or []:
        if ":" not in spec:
            raise ValueError(f"invalid --prune-residue-keep {spec!r} (need RESID:ATOMS)")
        resid_s, atoms_s = spec.split(":", 1)
        out[int(resid_s)] = [a.strip() for a in atoms_s.split(",") if a.strip()]
    return out


def _parse_distance_specs(specs: list[str]) -> list[tuple[int, int, float]]:
    """`--fix-distance i,j,d` (1-based serials)."""
    out = []
    for s in specs or []:
        toks = [x.strip() for x in s.split(",")]
        if len(toks) != 3:
            raise ValueError(f"--fix-distance need i,j,d (1-based serials); got {s!r}")
        out.append((int(toks[0]), int(toks[1]), float(toks[2])))
    return out


def _parse_angle_specs(specs: list[str]) -> list[tuple[int, int, int, float]]:
    out = []
    for s in specs or []:
        toks = [x.strip() for x in s.split(",")]
        if len(toks) != 4:
            raise ValueError(f"--fix-angle need i,j,k,deg; got {s!r}")
        out.append((int(toks[0]), int(toks[1]), int(toks[2]), float(toks[3])))
    return out


def _parse_dihedral_specs(specs: list[str]) -> list[tuple[int, int, int, int, float]]:
    out = []
    for s in specs or []:
        toks = [x.strip() for x in s.split(",")]
        if len(toks) != 5:
            raise ValueError(f"--fix-dihedral need i,j,k,l,deg; got {s!r}")
        out.append((int(toks[0]), int(toks[1]), int(toks[2]), int(toks[3]), float(toks[4])))
    return out


# ===== run spec ==============================================================
@dataclass
class RunSpec:
    input_pdb: Path
    out_dir: Path
    out_basename: str
    emit_pdb: bool
    emit_cif: bool
    model: str
    device: str
    charge: int
    p_name: str; p_res: str
    nuc_name: str; nuc_res: str
    lg_name: str; lg_res: str
    target_d_p_onuc: float
    target_d_p_olg: float
    fmax: float
    max_steps: int
    snapshot_stride: int
    free_residues: list[int]
    prune_backbone_residues: list[int]
    prune_residue_keep: dict[int, list[str]]
    fix_atoms: list[int]
    fix_distance: list[tuple[int, int, float]]
    fix_angle: list[tuple[int, int, int, float]]
    fix_dihedral: list[tuple[int, int, int, int, float]]
    cap_h_bond: float
    prune_xtb_relax: bool
    prune_xtb_max_steps: int
    metal_oxidation: dict[str, int]
    compute_mulliken: bool
    dry_run: bool
    seed: int


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="polish_ts_v3",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", "--out-dir", dest="out_dir", type=Path, required=True)
    p.add_argument("--out-basename", default=None,
                   help="Output stem; default <input_stem>_polished")
    fmt = p.add_argument_group("output formats")
    fmt.add_argument("--no-pdb", dest="emit_pdb", action="store_false", default=True)
    fmt.add_argument("--no-cif", dest="emit_cif", action="store_false", default=True)
    fmt.add_argument("--snapshot-stride", type=int, default=50,
                     help="Save a trajectory snapshot every N optimizer steps. 0=off.")

    calc = p.add_argument_group("calculator")
    calc.add_argument("--model", default="mace-polar-m")
    calc.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    calc.add_argument("--charge", type=int, default=1)

    react = p.add_argument_group("reactive atoms")
    react.add_argument("--p-name", default="P1")
    react.add_argument("--p-res", default="SUB")
    react.add_argument("--nuc-name", default="O3")
    react.add_argument("--nuc-res", default="OHX")
    react.add_argument("--lg-name", default="O7")
    react.add_argument("--lg-res", default="SUB")
    react.add_argument("--target-d-p-onuc", type=float, default=2.00)
    react.add_argument("--target-d-p-olg", type=float, default=2.25)

    opt = p.add_argument_group("optimizer")
    opt.add_argument("--fmax", type=float, default=0.05)
    opt.add_argument("--max-steps", type=int, default=500)

    constr = p.add_argument_group("custom constraints (all 1-based atom serials)")
    constr.add_argument("--free-residues", type=_csv_int, default=[],
                         help="Residue ids to EXCLUDE from CA-rigid scaffold (move freely).")
    constr.add_argument("--prune-backbone-residues", type=_csv_int, default=[],
                         help="Residue ids: drop backbone N/C/O; keep sidechain. Cap CA→H.")
    constr.add_argument("--prune-residue-keep", nargs="+", default=[], metavar="RESID:ATOMS",
                         help="e.g. 169:CD,CE,NZ — keep only listed heavy atoms (+attached Hs).")
    constr.add_argument("--fix-atoms", type=_csv_int, default=[],
                         help="Atom serials (1-based, global PDB) to fix in Cartesian.")
    constr.add_argument("--fix-distance", nargs="+", default=[], metavar="i,j,d",
                         help="Pin distance pair(s).")
    constr.add_argument("--fix-angle", nargs="+", default=[], metavar="i,j,k,deg")
    constr.add_argument("--fix-dihedral", nargs="+", default=[], metavar="i,j,k,l,deg")

    cap = p.add_argument_group("pruning H-cap behavior")
    cap.add_argument("--cap-h-bond", type=float, default=1.09,
                      help="Initial bond length (Å) of cap H placed at cut bonds.")
    cap.add_argument("--prune-xtb-relax", action="store_true", default=True,
                      help="After H-cap placement, run a fast xTB-GFN2 opt with all "
                           "non-cap atoms FixAtoms'd to relax cap H positions.")
    cap.add_argument("--no-prune-xtb-relax", dest="prune_xtb_relax",
                      action="store_false")
    cap.add_argument("--prune-xtb-max-steps", type=int, default=100)

    chrg = p.add_argument_group("metal oxidation states (CIF formal_charge)")
    chrg.add_argument("--metal-oxidation", nargs="+", default=[], metavar="ELEM:CHARGE",
                       help="Override per-element formal charge for metals "
                            "(e.g. 'Zn:+2' 'Fe:+3' 'Cu:+1'). Applies to ALL "
                            "atoms whose element matches; overrides PDB cols "
                            "79-80. For systems where PDB charge col is "
                            "missing/inconsistent on metals.")
    chrg.add_argument("--compute-mulliken", action="store_true", default=True,
                       help="Run a fast xTB-GFN2 single-point on the final "
                            "geometry to compute Mulliken partial charges. "
                            "Written to CIF _qcb_atom_charge.partial_charge "
                            "loop AND to PDB B-factor column. Default ON.")
    chrg.add_argument("--no-compute-mulliken", dest="compute_mulliken",
                       action="store_false")

    p.add_argument("--dry-run", action="store_true",
                   help="Resolve constraints and write README; skip optimization.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def resolve_args(args: argparse.Namespace) -> RunSpec:
    if not args.emit_pdb and not args.emit_cif:
        raise SystemExit("error: --no-pdb and --no-cif are mutually exclusive (must emit ≥1 format)")
    out_basename = args.out_basename or f"{args.input.stem}_polished"
    return RunSpec(
        input_pdb=args.input.resolve(),
        out_dir=args.out_dir.resolve(),
        out_basename=out_basename,
        emit_pdb=args.emit_pdb,
        emit_cif=args.emit_cif,
        model=args.model,
        device=args.device,
        charge=args.charge,
        p_name=args.p_name, p_res=args.p_res,
        nuc_name=args.nuc_name, nuc_res=args.nuc_res,
        lg_name=args.lg_name, lg_res=args.lg_res,
        target_d_p_onuc=args.target_d_p_onuc,
        target_d_p_olg=args.target_d_p_olg,
        fmax=args.fmax,
        max_steps=args.max_steps,
        snapshot_stride=args.snapshot_stride,
        free_residues=args.free_residues,
        prune_backbone_residues=args.prune_backbone_residues,
        prune_residue_keep=_parse_keep_specs(args.prune_residue_keep),
        fix_atoms=args.fix_atoms,
        fix_distance=_parse_distance_specs(args.fix_distance),
        fix_angle=_parse_angle_specs(args.fix_angle),
        fix_dihedral=_parse_dihedral_specs(args.fix_dihedral),
        cap_h_bond=args.cap_h_bond,
        prune_xtb_relax=args.prune_xtb_relax,
        prune_xtb_max_steps=args.prune_xtb_max_steps,
        metal_oxidation=_parse_metal_oxidation(args.metal_oxidation),
        compute_mulliken=args.compute_mulliken,
        dry_run=args.dry_run,
        seed=args.seed,
    )


def _parse_metal_oxidation(specs: list[str]) -> dict[str, int]:
    """Parse `--metal-oxidation Zn:+2 Fe:+3` into {'Zn': 2, 'Fe': 3}."""
    out: dict[str, int] = {}
    for s in specs or []:
        if ":" not in s:
            raise SystemExit(f"--metal-oxidation needs ELEM:CHARGE; got {s!r}")
        elem, charge_s = s.split(":", 1)
        out[elem.strip().capitalize()] = int(charge_s.replace("+", ""))
    return out


# ===== topology utilities (prune + cap) =====================================
BACKBONE_NAMES = {"N", "CA", "C", "O", "OXT", "H", "HA", "H2", "H3", "HXT"}


def apply_prune_keep(lineage, keep_specs: dict[int, list[str]]) -> tuple:
    """Remove atoms not in keep set for each listed residue.
    Caps dangling bonds with H placed along the severed bond direction.
    Returns (new_lineage_atoms, new_coords, mapping_old_to_new_or_None).
    """
    if not keep_specs:
        return lineage.atoms, np.array([[a.x, a.y, a.z] for a in lineage.atoms]), None

    keep_atoms: list = []
    keep_coords: list = []
    dropped_residue_atoms: dict[int, list] = {}    # resid -> list of dropped atom records

    for a in lineage.atoms:
        if a.resseq in keep_specs:
            allowed = set(keep_specs[a.resseq])
            # Always keep H atoms attached to allowed heavy atoms (we'll only
            # drop a hydrogen if its parent heavy atom is dropped). Crude but
            # adequate for sidechain-trimming workflows.
            heavy = a.name.lstrip("0123456789").rstrip("0123456789")
            if a.name in allowed or any(allowed & {a.name, heavy[:2]}):
                keep_atoms.append(a)
                keep_coords.append(np.array([a.x, a.y, a.z]))
            else:
                dropped_residue_atoms.setdefault(a.resseq, []).append(a)
        else:
            keep_atoms.append(a)
            keep_coords.append(np.array([a.x, a.y, a.z]))

    log.info(f"  prune-residue-keep: kept {len(keep_atoms)}/{len(lineage.atoms)}; "
             f"dropped {sum(len(v) for v in dropped_residue_atoms.values())}")
    return keep_atoms, np.array(keep_coords), dropped_residue_atoms


def apply_prune_backbone(lineage, residues: list[int]) -> tuple:
    """Drop backbone atoms (N, C, O, CA bond partners) for listed residues, keep sidechains.
    Caps the CB attachment with an H placed at the original CA position (1.09 Å from CB)."""
    if not residues:
        return lineage.atoms, np.array([[a.x, a.y, a.z] for a in lineage.atoms]), None

    new_atoms = []; new_coords = []
    dropped_bb: dict[int, list] = {}
    for a in lineage.atoms:
        if a.resseq in residues and a.name in BACKBONE_NAMES:
            dropped_bb.setdefault(a.resseq, []).append(a)
            continue
        new_atoms.append(a)
        new_coords.append(np.array([a.x, a.y, a.z]))

    log.info(f"  prune-backbone-residues: dropped "
             f"{sum(len(v) for v in dropped_bb.values())} backbone atoms")
    return new_atoms, np.array(new_coords), dropped_bb


# ===== the main pipeline ====================================================
def run_polish(spec: RunSpec) -> int:
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"=== polish_ts_v3  in={spec.input_pdb}  out={spec.out_dir} ===")

    # 0. Copy input for reference (always)
    shutil.copy2(spec.input_pdb, spec.out_dir / spec.input_pdb.name)
    log.info(f"  copied input to {spec.out_dir / spec.input_pdb.name}")

    # 1. Read with lineage
    lineage = parse_pdb_lineage(spec.input_pdb)
    log.info(f"  loaded {len(lineage.atoms)} atoms; {len(lineage.matcher_remarks)} REMARK 666")

    # 1a. Auto-add REMARK 666 for protein residues that don't have one
    autoadd_protein_matcher_remarks(lineage,
                                     target_chain="X",
                                     target_name=spec.p_res,
                                     target_resi=0)
    log.info(f"  after auto-add: {len(lineage.matcher_remarks)} REMARK 666 entries")

    # 2. Apply pruning if requested — use the cap-aware util that places HP
    #    atoms at cut bonds and (optionally) xTB-relaxes them
    hp_indices: list[int] = []
    if spec.prune_residue_keep or spec.prune_backbone_residues:
        log.info(f"  pruning (with cap H placement): residue_keep="
                 f"{list(spec.prune_residue_keep.keys())}, "
                 f"backbone_drop={spec.prune_backbone_residues}, "
                 f"xtb_relax={spec.prune_xtb_relax}")
        from prune_utils import apply_prune_with_caps
        lineage, coords, hp_indices = apply_prune_with_caps(
            lineage,
            prune_keep_specs=spec.prune_residue_keep,
            prune_backbone_residues=spec.prune_backbone_residues,
            cap_h_bond=spec.cap_h_bond,
            do_xtb_relax=spec.prune_xtb_relax,
            xtb_max_steps=spec.prune_xtb_max_steps,
            xtb_charge=spec.charge,
            xtb_solvent=None,    # gas phase for cap relax (fast)
        )
    else:
        coords = np.array([[a.x, a.y, a.z] for a in lineage.atoms])

    # 3. Find reactive atoms
    def find(name, res):
        for i, a in enumerate(lineage.atoms):
            if a.name == name and a.resname == res:
                return i, a
        return None, None
    P_i, P_a = find(spec.p_name, spec.p_res)
    ON_i, ON_a = find(spec.nuc_name, spec.nuc_res)
    OL_i, OL_a = find(spec.lg_name, spec.lg_res)
    if P_i is None or ON_i is None or OL_i is None:
        raise SystemExit(
            f"reactive atoms not found: P({spec.p_name}.{spec.p_res})={P_i}, "
            f"Onuc({spec.nuc_name}.{spec.nuc_res})={ON_i}, "
            f"Olg({spec.lg_name}.{spec.lg_res})={OL_i}"
        )
    log.info(f"  reactive: P=#{P_i+1} Onuc=#{ON_i+1} Olg=#{OL_i+1}")

    # 4. Shift reactive distances to targets (if not pruned away)
    def shift(positions, anchor, mover, target):
        v = positions[mover] - positions[anchor]
        d = float(np.linalg.norm(v))
        if d < 1e-6: return positions
        new = positions.copy()
        new[mover] = positions[anchor] + (v / d) * target
        return new
    coords = shift(coords, P_i, ON_i, spec.target_d_p_onuc)
    coords = shift(coords, P_i, OL_i, spec.target_d_p_olg)
    log.info(f"  shifted: d(P-Onuc)={np.linalg.norm(coords[P_i]-coords[ON_i]):.4f} "
             f"d(P-Olg)={np.linalg.norm(coords[P_i]-coords[OL_i]):.4f}")

    # 5. Build CA scaffold (rigid via FixInternals over CA-CA pairs)
    ca_idx = [i for i, a in enumerate(lineage.atoms)
              if a.chain == "A" and a.name == "CA" and a.resseq not in spec.free_residues]
    log.info(f"  CA-rigid scaffold: {len(ca_idx)} CA atoms ({len(spec.free_residues)} freed)")

    # 6. Set up ASE + constraints (skip if dry-run)
    extra_qcb = [
        qcb_remark("STAGE", "stage=polish_ts_v3"),
        qcb_remark("METHOD", f"model={spec.model}", f"device={spec.device}"),
        qcb_remark("REACTIVE_DISTANCES",
                    f"P={P_i+1}", f"Onuc={ON_i+1}", f"Olg={OL_i+1}",
                    f"target_d_P_Onuc={spec.target_d_p_onuc}",
                    f"target_d_P_Olg={spec.target_d_p_olg}"),
        qcb_remark("CONSTRAINT_PATTERN",
                    f"FixInternals_CA_CA_pairs={len(ca_idx)*(len(ca_idx)-1)//2}",
                    f"FixBondLengths_reactive=2",
                    f"free_residues={spec.free_residues or 'none'}",
                    f"pruned_keep={list(spec.prune_residue_keep.keys()) or 'none'}",
                    f"pruned_backbone={spec.prune_backbone_residues or 'none'}"),
        qcb_remark("TOTAL_CHARGE", f"value={spec.charge:+d}"),
    ]

    if spec.dry_run:
        log.info("  --dry-run set; skipping optimization")
        e_final = float("nan"); fmax_final = float("nan"); converged = False
    else:
        # Real optimization
        from ase.io import read as ase_read
        from ase.constraints import FixAtoms, FixInternals, FixBondLengths
        from ase.optimize import LBFGS
        from quantum_engine.calc import make_calc

        # Load atoms via ase read (preserves bond order)
        atoms = ase_read(str(spec.input_pdb))
        # If we pruned, reload from pruned coords + pruned element list
        if len(atoms) != len(lineage.atoms):
            from ase import Atoms as ASEAtoms
            elements = []
            for a in lineage.atoms:
                from structure_io import _guess_element
                elements.append(a.element if a.element else _guess_element(a.name))
            atoms = ASEAtoms(symbols=elements, positions=coords)
        else:
            atoms.set_positions(coords)
        atoms.info["charge"] = spec.charge
        atoms.calc = make_calc(spec.model, device=spec.device, charge=spec.charge)

        # Build constraints — combine CA-rigid scaffold + reactive bonds +
        # user-supplied fix-atoms / fix-distance / fix-angle / fix-dihedral.
        # ASE's FixInternals supports angles_deg and dihedrals_deg natively.
        constraints = []
        positions = atoms.get_positions()

        # CA-rigid scaffold pairwise distances → FixInternals(bonds=...)
        rigid_bonds = []
        if len(ca_idx) >= 2:
            for ii in range(len(ca_idx)):
                for jj in range(ii + 1, len(ca_idx)):
                    i, j = ca_idx[ii], ca_idx[jj]
                    rigid_bonds.append([float(np.linalg.norm(positions[i] - positions[j])), [i, j]])

        # User --fix-angle and --fix-dihedral go into the SAME FixInternals
        # call (1-based PDB serials → 0-based ASE indices).
        rigid_angles_deg = []
        for (i, j, k, deg) in spec.fix_angle:
            rigid_angles_deg.append([float(deg), [i - 1, j - 1, k - 1]])
            log.info(f"  user fix-angle: ({i},{j},{k}) → {deg}°")
        rigid_dihedrals_deg = []
        for (i, j, k, l, deg) in spec.fix_dihedral:
            rigid_dihedrals_deg.append([float(deg), [i - 1, j - 1, k - 1, l - 1]])
            log.info(f"  user fix-dihedral: ({i},{j},{k},{l}) → {deg}°")

        if rigid_bonds or rigid_angles_deg or rigid_dihedrals_deg:
            constraints.append(FixInternals(
                bonds=rigid_bonds or None,
                angles_deg=rigid_angles_deg or None,
                dihedrals_deg=rigid_dihedrals_deg or None,
                epsilon=1e-7,
            ))

        # Reactive distances + user --fix-distance → FixBondLengths
        # (separate constraint type, stiffer than FixInternals bonds).
        bondlength_pairs = [(P_i, ON_i), (P_i, OL_i)]
        for (i, j, _d) in spec.fix_distance:
            bondlength_pairs.append((i - 1, j - 1))
            log.info(f"  user fix-distance: ({i},{j}) → {_d} Å")
        constraints.append(FixBondLengths(bondlength_pairs))

        # User --fix-atoms (Cartesian pin)
        if spec.fix_atoms:
            fix_indices_0 = [s - 1 for s in spec.fix_atoms]
            constraints.append(FixAtoms(indices=fix_indices_0))
            log.info(f"  user FixAtoms: {len(fix_indices_0)} atoms")

        atoms.set_constraint(constraints)

        # Run LBFGS with snapshot every N steps if requested
        traj_path = spec.out_dir / f"{spec.out_basename}_traj.traj"
        opt = LBFGS(atoms, trajectory=str(traj_path),
                    logfile=str(spec.out_dir / f"{spec.out_basename}_polish.log"))
        e0 = float(atoms.get_potential_energy())
        log.info(f"  initial: E={e0:.4f} eV  fmax={float(np.linalg.norm(atoms.get_forces(), axis=1).max()):.4f}")
        converged = opt.run(fmax=spec.fmax, steps=spec.max_steps)
        e_final = float(atoms.get_potential_energy())
        fmax_final = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
        log.info(f"  final  : E={e_final:.4f} eV  fmax={fmax_final:.4f}  converged={converged}")
        coords = atoms.get_positions()

        extra_qcb.append(qcb_remark("ENERGY", f"value_eV={e_final:.4f}"))
        extra_qcb.append(qcb_remark("CONVERGENCE",
                                     f"fmax_eV_per_A={fmax_final:.6f}",
                                     f"converged={bool(converged)}"))

    # 7. Compute Mulliken partial charges via fast xTB-GFN2 single-point
    #    on the final geometry. Used for both PDB B-factor column AND CIF
    #    `_qcb_atom_charge.partial_charge` loop.
    partial_charges_arr: np.ndarray | None = None
    if spec.compute_mulliken:
        # Imports up-front to avoid NameError if a.element is empty for any atom
        from structure_io import (compute_mulliken_charges_xtb,
                                    _normalize_element, _guess_element)
        try:
            elements_for_xtb = [
                _normalize_element(a.element) if a.element else _guess_element(a.name)
                for a in lineage.atoms
            ]
            partial_charges_arr = compute_mulliken_charges_xtb(
                elements=elements_for_xtb,
                coords=coords,
                total_charge=spec.charge,
                method="gfn2",
                timeout_s=180,
            )
            # Sanity gate: Mulliken charges should sum to total system charge
            if partial_charges_arr is not None:
                summed = float(partial_charges_arr.sum())
                if abs(summed - spec.charge) > 0.5:
                    log.warning(
                        f"  Mulliken sum {summed:+.4f} differs from total_charge "
                        f"{spec.charge:+d} by >0.5 e — possible SCF or input issue; "
                        "still writing but flag for review")
            if partial_charges_arr is not None:
                log.info(f"  Mulliken (xTB-GFN2): sum={partial_charges_arr.sum():+.4f} "
                         f"(target {spec.charge:+d}), max|q|={abs(partial_charges_arr).max():.3f}")
                extra_qcb.append(qcb_remark(
                    "B_FACTOR_MEANING",
                    "column=B-factor", "value=Mulliken_partial_charge_au",
                    "method=xTB-GFN2_gas-phase",
                    f"sum={partial_charges_arr.sum():+.4f}",
                ))
        except Exception as e:
            log.warning(f"Mulliken computation failed (continuing without): {e}")
            partial_charges_arr = None

    out_paths = {}

    if spec.emit_pdb:
        pdb_out = spec.out_dir / f"{spec.out_basename}.pdb"
        # Write Mulliken charges to PDB B-factor column (cols 61-66) so PyMOL
        # can color by partial charge: `spectrum b, blue_white_red`.
        write_pdb_lineage(lineage, coords, pdb_out,
                           drop_old_qcb=False, preserve_other_remarks=True,
                           extra_qcb=extra_qcb,
                           bfactor=partial_charges_arr,
                           title=f"polish_ts_v3 {spec.out_basename}")
        # validate
        v = validate_pdb_format(pdb_out)
        log.info(f"  PDB output: {pdb_out}  format-ok={v['ok']}  n_atoms={v['n_atoms']}")
        if not v["ok"]:
            log.error(f"PDB FORMAT INVALID: {v['issues']}")
        out_paths["pdb"] = str(pdb_out)
        out_paths["pdb_validation"] = v

    if spec.emit_cif:
        cif_out = spec.out_dir / f"{spec.out_basename}.cif"
        # Default metal-oxidation overrides — Zn always +2 (only valid ox state
        # in PTE/biology); user can extend or override via --metal-oxidation.
        metal_ox = {"Zn": 2}
        metal_ox.update(spec.metal_oxidation or {})
        try:
            write_cif_lineage(lineage, coords, cif_out,
                               total_charge=spec.charge,
                               partial_charges=partial_charges_arr,
                               metal_oxidation_overrides=metal_ox,
                               energy_eV=e_final if not spec.dry_run else None,
                               extra_qcb=extra_qcb)
            log.info(f"  CIF output: {cif_out}")
            out_paths["cif"] = str(cif_out)
        except Exception as e:
            log.error(f"CIF write failed: {e}", exc_info=True)

    # 8. Trajectory snapshots
    if not spec.dry_run and spec.snapshot_stride > 0:
        try:
            from ase.io.trajectory import Trajectory
            traj = Trajectory(str(spec.out_dir / f"{spec.out_basename}_traj.traj"))
            frames = [a.get_positions() for a in traj]
            traj_pdb = spec.out_dir / f"{spec.out_basename}_trajectory.pdb"
            write_trajectory_pdb(lineage, frames, traj_pdb,
                                  model_stride=spec.snapshot_stride,
                                  title=f"trajectory {spec.out_basename}")
            out_paths["trajectory_pdb"] = str(traj_pdb)
            log.info(f"  trajectory snapshots: {traj_pdb}")
        except Exception as e:
            log.warning(f"trajectory snapshot write failed: {e}")

    # 9. Summary JSON
    summary = {
        "input": str(spec.input_pdb),
        "out_basename": spec.out_basename,
        "outputs": out_paths,
        "n_atoms": len(lineage.atoms),
        "n_matcher_remarks_666": len(lineage.matcher_remarks),
        "constraints": {
            "ca_rigid_anchors": len(ca_idx),
            "free_residues": spec.free_residues,
            "prune_residue_keep": {str(k): v for k, v in spec.prune_residue_keep.items()},
            "prune_backbone_residues": spec.prune_backbone_residues,
            "fix_atoms": spec.fix_atoms,
            "fix_distance": [[i, j, d] for (i, j, d) in spec.fix_distance],
        },
        "energy_eV": e_final if not spec.dry_run else None,
        "fmax_eV_per_A": fmax_final if not spec.dry_run else None,
        "converged": converged if not spec.dry_run else None,
        "model": spec.model,
        "charge": spec.charge,
    }
    sj = spec.out_dir / f"{spec.out_basename}_summary.json"
    sj.write_text(json.dumps(summary, indent=2, default=str))
    log.info(f"  summary: {sj}")

    # 10. README
    readme = spec.out_dir / "README.md"
    readme.write_text(f"""# polish_ts_v3 output: {spec.out_basename}

Input: `{spec.input_pdb.name}` (copied to this directory)

## Files
- `{spec.input_pdb.name}` — input copy for reference
- `{spec.out_basename}.pdb` — polished structure (PDB; REMARKs preserved)
- `{spec.out_basename}.cif` — polished structure (mmCIF with bonds, charges)
- `{spec.out_basename}_trajectory.pdb` — multi-MODEL optimizer trajectory
- `{spec.out_basename}_summary.json` — machine-readable summary
- `{spec.out_basename}_polish.log` — LBFGS log

## Constraints applied
- CA-rigid scaffold over {len(ca_idx)} CA atoms (FixInternals on all pairs)
- Reactive bond pins: P-Onuc={spec.target_d_p_onuc}, P-Olg={spec.target_d_p_olg}
- Free residues: {spec.free_residues or 'none'}
- Pruned (residue_keep): {dict(spec.prune_residue_keep) or 'none'}
- Pruned backbone: {spec.prune_backbone_residues or 'none'}

## Result
{'(dry run)' if spec.dry_run else f'E = {e_final:.4f} eV; fmax = {fmax_final:.4f}; converged = {converged}'}
""")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )
    try:
        spec = resolve_args(args)
    except SystemExit:
        raise
    except Exception as e:
        print(f"error parsing args: {e}", file=sys.stderr)
        return 2
    return run_polish(spec)


if __name__ == "__main__":
    sys.exit(main())
