#!/usr/bin/env python
"""microstate_sampler.py — targeted protonation / tautomer / water sampler.

Why this exists
---------------
A full CREST conformer-funnel on a metalloenzyme cluster is overkill when
the genuine physical uncertainty is **categorical**: HID vs HIE vs HIP for
each catalytic histidine, OH⁻ vs H₂O for the bridging oxo, GLU/ASP/KCX
protonation, and a small set of water-orientation alternatives. This
module enumerates those ENUMERABLE states deterministically, optionally
runs a short MLFF relaxation on each, and emits a labelled ensemble of
PDB files plus a manifest.

Generators (independent, composable; pass via ``--generators``):

    his                — toggle each histidine through {HID, HIE, HIP}
    asp_glu            — toggle each ASP/GLU between deprotonated/neutral
    lys                — toggle each LYS between charged/neutral
    cys                — toggle each CYS between thiol/thiolate
    zn_oh              — toggle Zn-bound oxygen between OH⁻ and H₂O
    water_shuffle      — random rotation of N waters about their O
    water_translate    — translate a chosen water O by ±delta in random
                         directions (useful when water positions are
                         under-determined by the X-ray fit)

All generators are reaction-agnostic — the residues they act on are
discovered from the input PDB, NOT hardcoded to a particular enzyme.

NOTE on H-atom rewriting (added 2026-05-06)
-------------------------------------------
The legacy ``his/asp_glu/lys/cys`` generators emitted **ledger-only deltas**
— variant PDBs had identical coordinates to the input. The new
``--auto-protonation`` and ``--protonation-rules`` paths use
``quantum_engine.ops.protonation`` to ACTUALLY rewrite H atoms (place /
remove HD1/HE2 on HIS, OE2-H on GLU, NZ Hs on LYS, SG-H on CYS) so the
xTB / MACE calculator sees a chemically distinct system per variant.
The legacy generators are preserved for backward compatibility but are
deprecated; new pipelines should use ``--auto-protonation`` or
``--protonation-rules``.

CREST does NOT explore protonation states automatically — its MTD bias
forces preserve covalent connectivity. To explore tautomers / protonation
states, you MUST enumerate variants externally (this module's job) and
feed each variant to CREST as a separate input. The
``--protonation-ensemble`` flag in ``crest_funnel.py`` automates this.

Public API
----------
``sample_microstates(input_pdb, out_dir, generators, **knobs)`` — returns
``MicrostateBatch``. Calls a calculator only if ``--relax`` is set.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quantum_engine.io import load_structure, write_pdb
from quantum_engine.ops.charge_ledger import (
    ChargeLedger, append_remarks_to_pdb, inject_into_atoms,
)
from quantum_engine.ops.protonation import (
    BOND_LENGTH_NH_DEFAULT, BOND_LENGTH_OH_DEFAULT, BOND_LENGTH_SH_DEFAULT,
    apply_residue_states, discover_titratable_residues,
)
from quantum_engine.ops.protonation_grid import (
    DEFAULT_FAMILIES, DEFAULT_MAX_MICROSTATES, METAL_CUTOFF_A_DEFAULT,
    POS_CHARGE_CUTOFF_A_DEFAULT, MicrostateAssignment,
    build_auto_grid, build_rules_grid, load_rules_yaml,
)

log = logging.getLogger("tools.microstate_sampler")


# ----------------------------------------------------------------------------
# Generator registry
# ----------------------------------------------------------------------------
KNOWN_GENERATORS = (
    "his", "asp_glu", "lys", "cys", "zn_oh",
    "water_shuffle", "water_translate",
)

WATER_RESNAMES = {"HOH", "WAT", "H2O", "TIP", "TIP3", "TIP3P"}
PROTEIN_RESNAMES = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","HIE","HID","HIP",
    "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","KCX",
}


@dataclass
class Microstate:
    """One generated variant."""
    label: str
    description: str
    pdb_path: Path
    operations: list[str]
    energy_eV: float | None = None
    converged: bool | None = None


@dataclass
class MicrostateBatch:
    status: str
    n_variants: int
    out_dir: Path
    manifest_path: Path
    microstates: list[Microstate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Per-generator helpers (return a list of (description, modified_atoms_pos,
# modified_resname_or_None, ledger_delta_dict))
# ----------------------------------------------------------------------------
def _residue_indices(bt_struct, res_names: Iterable[str]) -> list[tuple[int, str]]:
    """Return list of (representative_atom_idx, label) for distinct residues
    matching any of ``res_names``. ``label`` is ``RESNAME_RESID_CHAIN``.

    The "representative atom" is the first atom encountered for that residue.
    """
    out: list[tuple[int, str]] = []
    seen: set[tuple[str, int, str]] = set()
    res_set = {r.upper() for r in res_names}
    for i in range(len(bt_struct)):
        rn = str(bt_struct.res_name[i]).upper()
        if rn not in res_set:
            continue
        chain = str(bt_struct.chain_id[i])
        rid = int(bt_struct.res_id[i])
        key = (chain, rid, rn)
        if key in seen:
            continue
        seen.add(key)
        out.append((i, f"{rn}{rid}{chain}"))
    return out


def _gen_his_tautomers(atoms, bt_struct) -> list[dict]:
    """Generate HID/HIE/HIP variants by relabelling histidines.

    For each HIS-like residue (HIS / HIE / HID / HIP), produce three variants
    with the residue name flipped. We DON'T move atoms — the goal is to feed
    the cluster into MACE with the right N/H bookkeeping; the operator is
    expected to add or remove the relevant H atom upstream (e.g. via
    ``qcb protonate``) and use this generator to enumerate the labelled
    options. As a result this generator emits **ledger-only deltas**: the
    tautomer label and a ``+1`` charge delta for HIP.
    """
    his_residues = _residue_indices(bt_struct, {"HIS", "HIE", "HID", "HIP"})
    variants: list[dict] = []
    for atom_idx, label in his_residues:
        for tautomer in ("HID", "HIE", "HIP"):
            variants.append({
                "description": f"his_tautomer:{label}->{tautomer}",
                "rename": {label: tautomer},
                "ledger_delta": {label: 1 if tautomer == "HIP" else 0},
                "atoms_unchanged": True,
            })
    return variants


def _gen_asp_glu_protonation(atoms, bt_struct) -> list[dict]:
    """For each ASP/GLU, emit (deprotonated, neutral) variants."""
    residues = _residue_indices(bt_struct, {"ASP", "GLU"})
    out: list[dict] = []
    for atom_idx, label in residues:
        for state, delta in (("deprotonated", -1), ("neutral", 0)):
            out.append({
                "description": f"asp_glu:{label}={state}",
                "rename": {},
                "ledger_delta": {label: delta},
                "atoms_unchanged": True,
            })
    return out


def _gen_lys_protonation(atoms, bt_struct) -> list[dict]:
    """LYS charged (+1) vs neutral (0)."""
    residues = _residue_indices(bt_struct, {"LYS"})
    out: list[dict] = []
    for atom_idx, label in residues:
        for state, delta in (("charged", +1), ("neutral", 0)):
            out.append({
                "description": f"lys:{label}={state}",
                "rename": {},
                "ledger_delta": {label: delta},
                "atoms_unchanged": True,
            })
    return out


def _gen_cys_protonation(atoms, bt_struct) -> list[dict]:
    """CYS thiol vs thiolate."""
    residues = _residue_indices(bt_struct, {"CYS"})
    out: list[dict] = []
    for atom_idx, label in residues:
        for state, delta in (("thiol", 0), ("thiolate", -1)):
            out.append({
                "description": f"cys:{label}={state}",
                "rename": {},
                "ledger_delta": {label: delta},
                "atoms_unchanged": True,
            })
    return out


def _gen_zn_oh_water(
    atoms, bt_struct, zn_o_cutoff_a: float = 2.5,
) -> list[dict]:
    """For each Zn ion, toggle a candidate bound oxygen between OH⁻ and H₂O.

    Uses the heuristic: any O atom within ``zn_o_cutoff_a`` Å of any Zn is a
    candidate bound oxygen. This is reaction-agnostic — the residue name
    doesn't matter (could be OHX, HOH, BCA, whatever).

    Codex flag 2026-05-07: cutoff was previously hardcoded; now CLI-exposed
    via ``--zn-o-cutoff-a``.
    """
    coords = atoms.get_positions()
    syms = np.asarray(atoms.get_chemical_symbols())
    zn_idxs = np.where(syms == "Zn")[0]
    o_idxs = np.where(syms == "O")[0]
    out: list[dict] = []
    for zn_i in zn_idxs:
        zn_label = f"ZN{zn_i}"
        # Find bound oxygens
        for o_i in o_idxs:
            d = float(np.linalg.norm(coords[zn_i] - coords[o_i]))
            if d > zn_o_cutoff_a:
                continue
            o_label = f"O{o_i}"
            for state, delta in (("OH-", -1), ("H2O", 0)):
                out.append({
                    "description": f"zn_oh:{zn_label}-{o_label}={state}",
                    "rename": {},
                    "ledger_delta": {f"znO_{zn_i}_{o_i}": delta},
                    "atoms_unchanged": True,
                })
    return out


def _gen_water_shuffle(
    atoms, bt_struct, n_variants: int, rng: random.Random,
) -> list[dict]:
    """Generate ``n_variants`` water-orientation variants.

    For each variant: pick a random subset of waters, rotate each water's
    H atoms about its O atom by a random angle. This perturbs the
    water-network topology without changing any chemical state.
    """
    out: list[dict] = []
    if bt_struct is None:
        return out
    # Find waters: residue name in WATER_RESNAMES
    res_names = np.asarray([str(r).upper() for r in bt_struct.res_name])
    is_water = np.isin(res_names, list(WATER_RESNAMES))
    if not is_water.any():
        return out

    coords = atoms.get_positions()
    syms = np.asarray(atoms.get_chemical_symbols())

    # Group waters by (chain, res_id)
    chain = np.asarray(bt_struct.chain_id)
    rid = np.asarray(bt_struct.res_id)
    water_idxs = np.where(is_water)[0]
    waters_by_res: dict[tuple[str, int], list[int]] = {}
    for i in water_idxs:
        key = (str(chain[i]), int(rid[i]))
        waters_by_res.setdefault(key, []).append(int(i))

    for v in range(n_variants):
        new_pos = coords.copy()
        rotated_waters: list[str] = []
        for key, atom_list in waters_by_res.items():
            # 50% chance to rotate this water in this variant
            if rng.random() < 0.5:
                continue
            o_idxs = [j for j in atom_list if syms[j] == "O"]
            h_idxs = [j for j in atom_list if syms[j] == "H"]
            if not o_idxs or not h_idxs:
                continue
            o = o_idxs[0]
            # Random axis + angle
            axis = np.array([rng.gauss(0, 1) for _ in range(3)])
            n = np.linalg.norm(axis)
            if n < 1e-6:
                continue
            axis /= n
            angle = rng.uniform(-np.pi, np.pi)
            R = _rotation_matrix(axis, angle)
            for h in h_idxs:
                rel = new_pos[h] - new_pos[o]
                new_pos[h] = new_pos[o] + R @ rel
            rotated_waters.append(f"{key[0]}{key[1]}")
        out.append({
            "description": f"water_shuffle_v{v}_n={len(rotated_waters)}",
            "rename": {},
            "ledger_delta": {},
            "atoms_unchanged": False,
            "new_positions": new_pos,
            "shuffled": rotated_waters,
        })
    return out


def _gen_water_translate(
    atoms, bt_struct, n_variants: int, delta_A: float, rng: random.Random,
) -> list[dict]:
    """For each variant: pick a random subset of water Os, displace each
    by a random unit vector × ``delta_A`` Å.
    """
    out: list[dict] = []
    if bt_struct is None:
        return out
    res_names = np.asarray([str(r).upper() for r in bt_struct.res_name])
    is_water = np.isin(res_names, list(WATER_RESNAMES))
    if not is_water.any():
        return out
    coords = atoms.get_positions()
    syms = np.asarray(atoms.get_chemical_symbols())
    chain = np.asarray(bt_struct.chain_id)
    rid = np.asarray(bt_struct.res_id)
    water_idxs = np.where(is_water)[0]
    waters_by_res: dict[tuple[str, int], list[int]] = {}
    for i in water_idxs:
        key = (str(chain[i]), int(rid[i]))
        waters_by_res.setdefault(key, []).append(int(i))

    for v in range(n_variants):
        new_pos = coords.copy()
        translated: list[str] = []
        for key, atom_list in waters_by_res.items():
            if rng.random() < 0.5:
                continue
            # translate ALL atoms in this water by the same vector
            vec = np.array([rng.gauss(0, 1) for _ in range(3)])
            n = np.linalg.norm(vec)
            if n < 1e-6:
                continue
            vec = (vec / n) * delta_A
            for j in atom_list:
                new_pos[j] = coords[j] + vec
            translated.append(f"{key[0]}{key[1]}")
        out.append({
            "description": f"water_translate_v{v}_delta={delta_A:.2f}_n={len(translated)}",
            "rename": {},
            "ledger_delta": {},
            "atoms_unchanged": False,
            "new_positions": new_pos,
            "translated": translated,
        })
    return out


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix for unit ``axis`` and ``angle`` radians."""
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


# ----------------------------------------------------------------------------
# Atom-level H-rewriting public entry point (auto / rules path)
# ----------------------------------------------------------------------------
def sample_protonation_microstates(
    input_pdb: str | Path,
    out_dir: str | Path,
    *,
    mode: str,                                # "auto" or "rules"
    rules_yaml: str | Path | None = None,
    families: Iterable[str] = DEFAULT_FAMILIES,
    max_microstates: int = DEFAULT_MAX_MICROSTATES,
    metal_cutoff_a: float = METAL_CUTOFF_A_DEFAULT,
    pos_charge_cutoff_a: float = POS_CHARGE_CUTOFF_A_DEFAULT,
    nh_bond_length: float = BOND_LENGTH_NH_DEFAULT,
    oh_bond_length: float = BOND_LENGTH_OH_DEFAULT,
    sh_bond_length: float = BOND_LENGTH_SH_DEFAULT,
    include_consensus: bool = True,
    seed: int = 12345,                        # for any future random ops
    charge_ledger: ChargeLedger | None = None,
    cli_charge: int | None = None,
    cli_spin: int | None = None,
    relax: bool = False,
    relax_max_steps: int = 100,
    relax_fmax: float = 0.05,
    model: str = "mace-omol",
    head: str | None = None,
    device: str = "cuda",
) -> MicrostateBatch:
    """Generate a labelled microstate ensemble with REAL H-atom rewriting.

    This is the modern path: each variant has H atoms ADDED / REMOVED at the
    targeted residues so the QM / MLFF calculator sees a chemically distinct
    system. Use ``mode="auto"`` for an automatic Cartesian product (capped),
    or ``mode="rules"`` to follow a user YAML.

    Args:
        input_pdb: Cluster PDB to start from.
        out_dir: where ensemble PDBs and manifest are written.
        mode: ``"auto"`` (Cartesian product across all titratable residues)
            or ``"rules"`` (follow ``rules_yaml``).
        rules_yaml: when ``mode='rules'``, path to YAML with the residue
            list. See :func:`quantum_engine.ops.protonation_grid.load_rules_yaml`.
        families: which residue families to include in auto mode.
        max_microstates: defensive cap; auto-mode prunes to consensus +
            single-residue perturbations when full product exceeds this.
        metal_cutoff_a: HIS within this of any metal → consensus HID; CYS
            within this of any metal → consensus CYS-S-.
        pos_charge_cutoff_a: reserved for ASP/GLU near positive charge
            heuristic (not yet flipping consensus, but exposed for callers).
        nh_bond_length, oh_bond_length, sh_bond_length: H-placement
            target distances (Å). Defaults from Allen et al. CSD survey
            (1987 S1).
        include_consensus: include the all-consensus state as the first
            entry in auto mode.
        seed: PRNG seed (currently unused but reserved for future
            stochastic prunings).
        charge_ledger, cli_charge, cli_spin: same semantics as legacy path.
        relax, relax_max_steps, relax_fmax, model, head, device: optional
            MLFF relaxation, same as legacy path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("sample_protonation_microstates: mode=%s, max_microstates=%d, "
             "families=%s", mode, max_microstates, list(families))

    atoms, bt_struct, charge_hint = load_structure(input_pdb)
    if bt_struct is None:
        raise ValueError(
            "sample_protonation_microstates requires a PDB with biotite-"
            "parseable annotations (residue names, chain IDs, atom names)"
        )

    base_charge = charge_ledger.total if charge_ledger is not None else (
        cli_charge if cli_charge is not None else (charge_hint or 0)
    )
    base_spin = charge_ledger.spin if charge_ledger is not None else (
        cli_spin if cli_spin is not None else 1
    )

    if mode == "auto":
        assignments, residues = build_auto_grid(
            atoms, bt_struct,
            families=families,
            max_microstates=max_microstates,
            metal_cutoff_a=metal_cutoff_a,
            pos_charge_cutoff_a=pos_charge_cutoff_a,
            include_consensus=include_consensus,
        )
        log.info("auto-protonation: %d residues found, %d microstates planned",
                 len(residues), len(assignments))
    elif mode == "rules":
        if rules_yaml is None:
            raise ValueError("mode='rules' requires rules_yaml path")
        rules = load_rules_yaml(rules_yaml)
        assignments = build_rules_grid(
            atoms, bt_struct, rules, max_microstates=max_microstates,
        )
        log.info("rules-protonation: %d microstates from %s",
                 len(assignments), rules_yaml)
    else:
        raise ValueError(f"mode must be 'auto' or 'rules', got {mode!r}")

    if not assignments:
        log.warning("sample_protonation_microstates: no assignments generated")
        manifest = {
            "input_pdb": str(input_pdb), "mode": mode,
            "families": list(families),
            "max_microstates": max_microstates,
            "n_variants": 0, "microstates": [],
        }
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return MicrostateBatch(
            status="completed", n_variants=0,
            out_dir=out_dir, manifest_path=manifest_path,
            warnings=["no titratable residues found / no rules"],
        )

    microstates: list[Microstate] = []
    for i, assignment in enumerate(assignments):
        label = f"variant_{i:04d}"
        try:
            new_atoms, new_bt, ledger_delta_dict, total_dq = apply_residue_states(
                atoms, bt_struct,
                residue_states=assignment.assignments,
                nh_bond_length=nh_bond_length,
                oh_bond_length=oh_bond_length,
                sh_bond_length=sh_bond_length,
            )
        except Exception as exc:
            log.error("variant %s (%s) FAILED H-rewrite: %s",
                      label, assignment.label(), exc)
            continue

        var_charge = base_charge + total_dq
        new_atoms.info["charge"] = var_charge
        new_atoms.info["spin"] = base_spin

        # Build a per-variant ledger (extending the input ledger)
        var_ledger: ChargeLedger | None = None
        if charge_ledger is not None:
            new_components = dict(charge_ledger.components)
            for k, dq in ledger_delta_dict.items():
                new_components[k] = new_components.get(k, 0) + int(dq)
            var_ledger = ChargeLedger(
                total=charge_ledger.total + total_dq,
                spin=charge_ledger.spin,
                components=new_components,
                notes={**charge_ledger.notes,
                       "_variant": assignment.description,
                       "_variant_label": label},
                source=f"{charge_ledger.source}@{label}",
            )
            inject_into_atoms(new_atoms, var_ledger)

        e_eV: float | None = None
        converged: bool | None = None
        if relax:
            from quantum_engine.calc import make_calc
            from quantum_engine.ops import opt as opt_op
            new_atoms.calc = make_calc(model=model, head=head, device=device,
                                       charge=var_charge)
            var_outdir = out_dir / f"{label}_relax"
            try:
                rres = opt_op.run(
                    new_atoms, new_atoms.calc, var_outdir,
                    constraint=None, optimizer="lbfgs",
                    fmax=relax_fmax, max_steps=relax_max_steps,
                )
                e_eV = float(rres.get("energy_eV",
                                       rres["atoms"].get_potential_energy()))
                converged = bool(rres.get("converged"))
                new_atoms = rres["atoms"]
            except Exception as exc:
                log.error("variant %s relax failed: %s", label, exc)

        pdb_path = out_dir / f"{label}.pdb"
        write_pdb(new_atoms, new_bt, pdb_path,
                  total_charge=var_charge, energy_eV=e_eV)
        if var_ledger is not None:
            append_remarks_to_pdb(pdb_path, var_ledger)

        ms = Microstate(
            label=label,
            description=assignment.description,
            pdb_path=pdb_path,
            operations=[assignment.label()],
            energy_eV=e_eV,
            converged=converged,
        )
        microstates.append(ms)
        log.info("variant %s: %s charge_delta=%+d total_charge=%+d%s",
                 label, assignment.description, total_dq, var_charge,
                 f" E={e_eV:.4f} eV" if e_eV is not None else "")

    manifest_path = out_dir / "manifest.json"
    manifest = {
        "input_pdb": str(input_pdb),
        "mode": mode,
        "families": list(families),
        "max_microstates": max_microstates,
        "metal_cutoff_a": metal_cutoff_a,
        "pos_charge_cutoff_a": pos_charge_cutoff_a,
        "nh_bond_length_a": nh_bond_length,
        "oh_bond_length_a": oh_bond_length,
        "sh_bond_length_a": sh_bond_length,
        "include_consensus": include_consensus,
        "n_variants": len(microstates),
        "relax": relax,
        "model": model if relax else None,
        "microstates": [
            {
                "label": ms.label,
                "description": ms.description,
                "pdb": str(ms.pdb_path),
                "energy_eV": ms.energy_eV,
                "converged": ms.converged,
                "operations": ms.operations,
            }
            for ms in microstates
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return MicrostateBatch(
        status="completed",
        n_variants=len(microstates),
        out_dir=out_dir,
        manifest_path=manifest_path,
        microstates=microstates,
        warnings=([] if microstates else
                  ["no variants generated"]),
    )


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def sample_microstates(
    input_pdb: str | Path,
    out_dir: str | Path,
    *,
    generators: Sequence[str],
    n_water_shuffle: int = 5,
    n_water_translate: int = 5,
    water_translate_delta_A: float = 0.30,
    zn_o_cutoff_a: float = 2.5,
    seed: int = 12345,
    relax: bool = False,
    relax_max_steps: int = 100,
    relax_fmax: float = 0.05,
    model: str = "mace-omol",
    head: str | None = None,
    device: str = "cuda",
    charge_ledger: ChargeLedger | None = None,
    cli_charge: int | None = None,
    cli_spin: int | None = None,
    max_variants: int = 200,
) -> MicrostateBatch:
    """Generate a labelled microstate ensemble.

    Args:
        input_pdb: Cluster PDB to start from.
        out_dir: where ensemble PDBs and manifest are written.
        generators: list of generator names; subset of :data:`KNOWN_GENERATORS`.
        n_water_shuffle, n_water_translate: number of variants per stochastic
            generator.
        water_translate_delta_A: displacement magnitude for ``water_translate``.
        seed: PRNG seed for reproducibility.
        relax: if True, run an MLFF relax on each variant.
        relax_max_steps, relax_fmax: relax convergence (only used if relax).
        model, head, device: calculator factory args (only used if relax).
        charge_ledger: optional :class:`ChargeLedger`; per-variant charge
            deltas are applied on top of this.
        cli_charge, cli_spin: fall-back charge / spin when no ledger.
        max_variants: cap on total emitted variants (defensive default).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("microstate_sampler: generators=%s, seed=%d, relax=%s",
             list(generators), seed, relax)

    invalid = [g for g in generators if g not in KNOWN_GENERATORS]
    if invalid:
        raise ValueError(
            f"unknown generators: {invalid}; known = {list(KNOWN_GENERATORS)}"
        )

    atoms, bt_struct, charge_hint = load_structure(input_pdb)
    if bt_struct is None:
        raise ValueError(
            "microstate_sampler requires a PDB with biotite-parseable "
            "annotations (residue names, chain IDs)"
        )

    base_charge = charge_ledger.total if charge_ledger is not None else (
        cli_charge if cli_charge is not None else (charge_hint or 0)
    )
    base_spin = charge_ledger.spin if charge_ledger is not None else (
        cli_spin if cli_spin is not None else 1
    )

    rng = random.Random(seed)
    variants: list[dict] = []

    if "his" in generators:
        variants.extend(_gen_his_tautomers(atoms, bt_struct))
    if "asp_glu" in generators:
        variants.extend(_gen_asp_glu_protonation(atoms, bt_struct))
    if "lys" in generators:
        variants.extend(_gen_lys_protonation(atoms, bt_struct))
    if "cys" in generators:
        variants.extend(_gen_cys_protonation(atoms, bt_struct))
    if "zn_oh" in generators:
        variants.extend(_gen_zn_oh_water(atoms, bt_struct,
                                         zn_o_cutoff_a=zn_o_cutoff_a))
    if "water_shuffle" in generators:
        variants.extend(_gen_water_shuffle(atoms, bt_struct, n_water_shuffle, rng))
    if "water_translate" in generators:
        variants.extend(_gen_water_translate(
            atoms, bt_struct, n_water_translate, water_translate_delta_A, rng,
        ))

    if not variants:
        log.warning(
            "microstate_sampler: no variants generated — generators=%s found "
            "no candidate residues / waters in %s", generators, input_pdb,
        )
    if len(variants) > max_variants:
        log.warning("microstate_sampler: capping %d variants to --max-variants=%d",
                    len(variants), max_variants)
        variants = variants[:max_variants]

    microstates: list[Microstate] = []
    calc = None
    if relax:
        from quantum_engine.calc import make_calc
        calc = make_calc(model=model, head=head, device=device, charge=base_charge)

    for i, v in enumerate(variants):
        label = f"variant_{i:04d}"
        var_atoms = atoms.copy()
        if "new_positions" in v:
            var_atoms.set_positions(v["new_positions"])

        # Per-variant ledger and charge.
        # Codex flag 2026-05-07: previously ``ledger_delta`` was ONLY applied
        # when a charge ledger was passed. Without a ledger, the legacy
        # generators emitted N variants all with the same ``base_charge`` —
        # silently wrong for the asp_glu / lys / cys / zn_oh paths which
        # legitimately have different total charges per variant. Apply the
        # delta-sum to ``var_charge`` even without a ledger.
        var_ledger: ChargeLedger | None = None
        ledger_delta_sum = sum(v.get("ledger_delta", {}).values())
        var_charge = base_charge + int(ledger_delta_sum)
        if charge_ledger is not None:
            new_components = dict(charge_ledger.components)
            for k, delta in v.get("ledger_delta", {}).items():
                new_components[k] = new_components.get(k, 0) + int(delta)
            var_ledger = ChargeLedger(
                total=charge_ledger.total + int(ledger_delta_sum),
                spin=charge_ledger.spin,
                components=new_components,
                notes={**charge_ledger.notes, "_variant": v["description"]},
                source=f"{charge_ledger.source}@{label}",
            )
            var_charge = var_ledger.total

        var_atoms.info["charge"] = var_charge
        var_atoms.info["spin"] = base_spin
        if var_ledger is not None:
            inject_into_atoms(var_atoms, var_ledger)

        pdb_path = out_dir / f"{label}.pdb"

        e_eV: float | None = None
        converged: bool | None = None
        if relax:
            from quantum_engine.ops import opt as opt_op
            from quantum_engine.calc import make_calc
            var_atoms.calc = make_calc(model=model, head=head, device=device,
                                       charge=var_charge)
            var_outdir = out_dir / f"{label}_relax"
            try:
                rres = opt_op.run(
                    var_atoms, var_atoms.calc, var_outdir,
                    constraint=None, optimizer="lbfgs",
                    fmax=relax_fmax, max_steps=relax_max_steps,
                )
                e_eV = float(rres.get("energy_eV",
                                       rres["atoms"].get_potential_energy()))
                converged = bool(rres.get("converged"))
                var_atoms = rres["atoms"]
            except Exception as exc:
                log.error("variant %s relax failed: %s", label, exc)

        write_pdb(var_atoms, bt_struct, pdb_path, total_charge=var_charge,
                  energy_eV=e_eV)
        if var_ledger is not None:
            append_remarks_to_pdb(pdb_path, var_ledger)

        ms = Microstate(
            label=label,
            description=v["description"],
            pdb_path=pdb_path,
            operations=[v["description"]],
            energy_eV=e_eV,
            converged=converged,
        )
        microstates.append(ms)
        log.info("variant %s: %s%s", label, v["description"],
                 f"  E={e_eV:.4f} eV" if e_eV is not None else "")

    manifest_path = out_dir / "manifest.json"
    manifest = {
        "input_pdb": str(input_pdb),
        "generators": list(generators),
        "seed": seed,
        "relax": relax,
        "model": model if relax else None,
        "n_variants": len(microstates),
        "microstates": [
            {
                "label": ms.label,
                "description": ms.description,
                "pdb": str(ms.pdb_path),
                "energy_eV": ms.energy_eV,
                "converged": ms.converged,
            }
            for ms in microstates
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return MicrostateBatch(
        status="completed",
        n_variants=len(microstates),
        out_dir=out_dir,
        manifest_path=manifest_path,
        microstates=microstates,
        warnings=([] if microstates else
                  ["no variants generated; check --generators arguments"]),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="microstate_sampler",
        description=(
            "Generate a labelled microstate ensemble (protonation tautomers, "
            "Zn-bound OH/H2O, water reorientations, ...) from a starting "
            "cluster PDB.\n\n"
            "Three modes of operation:\n"
            "  1. Legacy: --generators his,asp_glu,...  (ledger-only deltas; "
            "     atom coordinates UNCHANGED — kept for backward compat).\n"
            "  2. Auto: --auto-protonation  (Cartesian product over all\n"
            "     titratable HIS/ASP/GLU/LYS/CYS, REWRITES H atoms per\n"
            "     variant; capped at --max-microstates with a consensus +\n"
            "     1-residue-perturbation heuristic).\n"
            "  3. Manual: --protonation-rules rules.yaml  (user-specified\n"
            "     residues and states; full Cartesian product; H atoms are\n"
            "     rewritten per variant).\n\n"
            "CREST does NOT explore protonation states automatically (its\n"
            "MTD bias forces preserve covalent connectivity), so for any\n"
            "state-uncertainty study you MUST enumerate variants externally.\n"
            "Use this tool's auto / rules modes upstream of CREST.\n\n"
            "Example 'auto' run:\n"
            "  cowboy-qc microstates --input MC.pdb --outdir variants/\n"
            "                  --auto-protonation --max-microstates 8\n\n"
            "Example 'rules' run:\n"
            "  cowboy-qc microstates --input MC.pdb --outdir variants/\n"
            "                  --protonation-rules rules.yaml\n"
            "where rules.yaml contains:\n"
            "  residues:\n"
            "    - residue: A:201:HIS\n"
            "      states: [HID, HIE, HIP]\n"
            "    - residue: A:169:GLU\n"
            "      states: [GLU-, GLU0]\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Cluster PDB.")
    # ``--outdir`` is an alias for ``--out`` (docstring/examples used the
    # former name; codex flag 2026-05-07).
    p.add_argument("--out", "--outdir", dest="out", required=True,
                   help="Output directory.")
    p.add_argument(
        "--generators", required=False, default=None,
        help=("Legacy ledger-only generators. Comma-separated subset of: "
              + ",".join(KNOWN_GENERATORS) +
              ". Required ONLY when neither --auto-protonation nor "
              "--protonation-rules is given. Examples: 'his,water_shuffle' "
              "or 'his,asp_glu,zn_oh'.")
    )
    # ---- Atom-level H-rewriting (auto / rules paths) -----------------------
    p.add_argument("--auto-protonation", action="store_true",
                   help="Build the auto Cartesian product across all "
                        "titratable residues (HIS / ASP / GLU / LYS / CYS) "
                        "discovered in --input. H atoms are REWRITTEN per "
                        "variant (real chemistry, not just ledger). "
                        "Capped at --max-microstates with a consensus "
                        "+ single-residue perturbation strategy.")
    p.add_argument("--protonation-rules", default=None,
                   help="Path to a YAML file specifying which residues and "
                        "which states to enumerate. Produces a full "
                        "Cartesian product (capped at --max-microstates). "
                        "Mutually exclusive with --auto-protonation.")
    p.add_argument("--protonation-families", default=",".join(DEFAULT_FAMILIES),
                   help=("Comma list of families to include in the auto "
                         "grid. Default: " + ",".join(DEFAULT_FAMILIES) +
                         ". Restrict (e.g. 'HIS,GLU') to keep the grid "
                         "small."))
    p.add_argument("--max-microstates", type=int,
                   default=DEFAULT_MAX_MICROSTATES,
                   help=f"Cap on protonation microstates emitted. Default "
                        f"{DEFAULT_MAX_MICROSTATES}. Auto mode prunes via "
                        "consensus + single-residue perturbations when the "
                        "full product exceeds this. Bump to 64 / 128 for "
                        "exhaustive coverage of small systems.")
    p.add_argument("--metal-cutoff-a", type=float,
                   default=METAL_CUTOFF_A_DEFAULT,
                   help=f"HIS within this many Å of any metal atom uses HID "
                        f"as the consensus tautomer (the standard "
                        f"metal-coordinating tautomer). Default "
                        f"{METAL_CUTOFF_A_DEFAULT:.2f} Å.")
    p.add_argument("--pos-charge-cutoff-a", type=float,
                   default=POS_CHARGE_CUTOFF_A_DEFAULT,
                   help=f"Reserved for ASP/GLU near-positive-charge heuristic. "
                        f"Default {POS_CHARGE_CUTOFF_A_DEFAULT:.2f} Å.")
    p.add_argument("--nh-bond-length", type=float,
                   default=BOND_LENGTH_NH_DEFAULT,
                   help=f"H placement target N–H length (Å). Default "
                        f"{BOND_LENGTH_NH_DEFAULT:.3f} (Allen et al. CSD).")
    p.add_argument("--oh-bond-length", type=float,
                   default=BOND_LENGTH_OH_DEFAULT,
                   help=f"H placement target O–H length (Å). Default "
                        f"{BOND_LENGTH_OH_DEFAULT:.3f} (Allen et al. CSD).")
    p.add_argument("--sh-bond-length", type=float,
                   default=BOND_LENGTH_SH_DEFAULT,
                   help=f"H placement target S–H length (Å). Default "
                        f"{BOND_LENGTH_SH_DEFAULT:.3f} (Allen et al. CSD).")
    p.add_argument("--no-include-consensus", action="store_true",
                   help="Skip emitting the all-consensus assignment as the "
                        "first microstate (auto mode only).")
    p.add_argument("--n-water-shuffle", type=int, default=5,
                   help="Number of water-rotation variants. Default 5.")
    p.add_argument("--n-water-translate", type=int, default=5,
                   help="Number of water-translation variants. Default 5.")
    p.add_argument("--water-translate-delta", type=float, default=0.30,
                   help="Displacement magnitude for water_translate (Å). "
                        "Default 0.30.")
    p.add_argument("--zn-o-cutoff-a", type=float, default=2.5,
                   help="Zn-O distance cutoff for the zn_oh generator: any O "
                        "within this Å of any Zn is a candidate metal-bound "
                        "oxygen. Default 2.5.")
    p.add_argument("--seed", type=int, default=12345,
                   help="PRNG seed (deterministic outputs). Default 12345.")
    p.add_argument("--relax", action="store_true",
                   help="Run a short MLFF relax on each variant. Off by default.")
    p.add_argument("--relax-max-steps", type=int, default=100)
    p.add_argument("--relax-fmax", type=float, default=0.05)
    p.add_argument("--model", default="mace-omol")
    p.add_argument("--head", default=None)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--charge-ledger", default=None,
                   help="Optional charge-ledger.yaml (propagated into variants).")
    p.add_argument("--charge", type=int, default=None)
    p.add_argument("--spin", type=int, default=None)
    p.add_argument("--max-variants", type=int, default=200,
                   help="Defensive cap on emitted variants. Default 200.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.auto_protonation and args.protonation_rules:
        raise SystemExit(
            "--auto-protonation and --protonation-rules are mutually "
            "exclusive. Pick one."
        )
    if not args.auto_protonation and not args.protonation_rules and not args.generators:
        raise SystemExit(
            "Specify either --generators (legacy ledger-only mode), "
            "--auto-protonation (atom-level H-rewriting, automatic grid), "
            "or --protonation-rules path.yaml (atom-level H-rewriting, "
            "user-specified)."
        )

    ledger = None
    if args.charge_ledger:
        from quantum_engine.ops.charge_ledger import load_ledger
        ledger = load_ledger(args.charge_ledger)

    if args.auto_protonation or args.protonation_rules:
        # Atom-level H-rewriting path
        families = [f.strip().upper() for f in args.protonation_families.split(",")
                    if f.strip()]
        mode = "auto" if args.auto_protonation else "rules"
        res = sample_protonation_microstates(
            args.input,
            args.out,
            mode=mode,
            rules_yaml=args.protonation_rules,
            families=families,
            max_microstates=args.max_microstates,
            metal_cutoff_a=args.metal_cutoff_a,
            pos_charge_cutoff_a=args.pos_charge_cutoff_a,
            nh_bond_length=args.nh_bond_length,
            oh_bond_length=args.oh_bond_length,
            sh_bond_length=args.sh_bond_length,
            include_consensus=not args.no_include_consensus,
            seed=args.seed,
            charge_ledger=ledger,
            cli_charge=args.charge,
            cli_spin=args.spin,
            relax=args.relax,
            relax_max_steps=args.relax_max_steps,
            relax_fmax=args.relax_fmax,
            model=args.model,
            head=args.head,
            device=args.device,
        )
    else:
        # Legacy ledger-only path
        generators = [g.strip() for g in args.generators.split(",") if g.strip()]
        res = sample_microstates(
            args.input,
            args.out,
            generators=generators,
            n_water_shuffle=args.n_water_shuffle,
            n_water_translate=args.n_water_translate,
            water_translate_delta_A=args.water_translate_delta,
            zn_o_cutoff_a=args.zn_o_cutoff_a,
            seed=args.seed,
            relax=args.relax,
            relax_max_steps=args.relax_max_steps,
            relax_fmax=args.relax_fmax,
            model=args.model,
            head=args.head,
            device=args.device,
            charge_ledger=ledger,
            cli_charge=args.charge,
            cli_spin=args.spin,
            max_variants=args.max_variants,
        )
    print(json.dumps({
        "status": res.status,
        "n_variants": res.n_variants,
        "manifest": str(res.manifest_path),
        "warnings": res.warnings,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
