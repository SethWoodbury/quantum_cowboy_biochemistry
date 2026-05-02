"""Parse human-readable constraint specifications into ASE constraints.

Grammar
-------
Specs are strings that describe which atoms to fix or free. Multiple specs
can be passed as a list, combined with OR (union). Example:

    --fix "residue HIS ASP" "chain B" "atoms CA CB"
        → fix all atoms in HIS/ASP residues, plus all atoms in chain B,
          plus any CA/CB atoms anywhere.
    --free "residue YYE" "atoms ZN1 ZN2"
        → subtract YYE atoms and Zn atoms from any fix spec.

Supported tokens
----------------
    residue X [Y Z ...]     — by residue name
    resid N [M P ...]       — by residue number (integers)
    chain A [B ...]         — by chain ID
    atoms NAME [NAME ...]   — by PDB atom name (CA, CB, etc.)
    element SYM [SYM ...]   — by chemical symbol (C, N, ...)
    range START END         — by atom index (0-based, inclusive)
    all                     — every atom
    none                    — no atoms

Presets
-------
For convenience, named preset modes (--constraint-mode) are also supported
and internally expand to the spec grammar:
    ca-only         → fix "atoms CA" (excluding ligand, water, metals)
    backbone        → fix "atoms CA C N O" (excluding ligand, water, metals)
    backbone-water  → fix "atoms CA C N O" + water oxygens
    none            → no constraints
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

log = logging.getLogger("quantum_engine.io.constraints")


STANDARD_EXCLUDED_RES = {
    # Residues/molecules typically NEVER constrained (they're the reacting species)
    "HOH", "WAT", "H2O",       # water
    "ZN", "MG", "CA", "FE", "CU", "MN", "NI",  # metals (residue names, not atom names)
}


def parse_constraints(
    atoms: Atoms,
    bt_struct,
    specs: str | list[str] | None = None,
    excluded_residues: Iterable[str] | None = None,
) -> np.ndarray:
    """Parse constraint specs into a boolean mask (True = fixed).

    Args:
        atoms: ASE Atoms object
        bt_struct: biotite AtomArray with per-atom annotations (res_name,
                   chain_id, atom_name). If None, only index-based specs work.
        specs: string or list of strings, each a constraint spec
        excluded_residues: residue names to never include in any fix mask
                          (defaults to water, metals, ligand typically handled by caller)

    Returns:
        Boolean array of length len(atoms), True where atoms should be fixed.
    """
    n = len(atoms)
    mask = np.zeros(n, dtype=bool)

    if specs is None:
        return mask
    if isinstance(specs, str):
        specs = [specs]

    excluded_residues = set(excluded_residues or ())

    for spec in specs:
        spec_mask = _parse_single_spec(atoms, bt_struct, spec)
        mask |= spec_mask

    # Apply exclusions (excluded residues are NEVER fixed)
    if bt_struct is not None and excluded_residues:
        for ex in excluded_residues:
            ex_mask = bt_struct.res_name == ex
            mask &= ~ex_mask

    return mask


def _parse_single_spec(atoms: Atoms, bt_struct, spec: str) -> np.ndarray:
    """Parse one spec string into a mask."""
    n = len(atoms)
    tokens = spec.strip().split()
    if not tokens:
        return np.zeros(n, dtype=bool)

    selector = tokens[0].lower()
    args = tokens[1:]

    if selector == "all":
        return np.ones(n, dtype=bool)
    if selector == "none":
        return np.zeros(n, dtype=bool)

    if selector == "range":
        if len(args) != 2:
            raise ValueError(f"range spec needs START END, got: {spec}")
        start, end = int(args[0]), int(args[1])
        mask = np.zeros(n, dtype=bool)
        mask[start : end + 1] = True
        return mask

    if selector == "element":
        syms = atoms.get_chemical_symbols()
        target = set(args)
        return np.array([s in target for s in syms], dtype=bool)

    # Remaining selectors need biotite annotations
    if bt_struct is None:
        raise ValueError(
            f"Spec '{spec}' requires biotite annotations (residue/chain/atom names). "
            "Load a PDB (not XYZ/CIF) or use 'element' or 'range' selectors."
        )

    if selector == "residue":
        return np.isin(bt_struct.res_name, args)
    if selector == "resid":
        try:
            res_ids = [int(a) for a in args]
        except ValueError:
            raise ValueError(f"resid needs integer IDs, got: {spec}")
        return np.isin(bt_struct.res_id.astype(int), res_ids)
    if selector == "chain":
        return np.isin(bt_struct.chain_id, args)
    if selector == "atoms":
        return np.isin(bt_struct.atom_name, args)

    raise ValueError(f"Unknown constraint selector '{selector}' in spec: {spec}")


def build_fix_atoms(mask: np.ndarray) -> FixAtoms | None:
    """Build an ASE FixAtoms constraint from a boolean mask.

    Returns None if no atoms are fixed (so downstream code can `if constraint is not None`).
    """
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None
    return FixAtoms(indices=indices)


def preset_to_specs(
    preset: str,
    ligand_name: str | None = None,
) -> tuple[list[str], set[str]]:
    """Expand a preset mode name into (fix_specs, excluded_residues).

    Presets:
        ca-only           — fix backbone alpha carbons of protein residues
        backbone          — fix backbone atoms (CA, C, N, O) of protein residues
        backbone-water    — like backbone but also pin water oxygens
        none              — no constraints

    Args:
        preset: preset name
        ligand_name: residue name of the ligand (should not be fixed)

    Returns:
        (fix_specs, excluded_residues)
    """
    excluded = set(STANDARD_EXCLUDED_RES)
    if ligand_name:
        excluded.add(ligand_name)

    if preset == "ca-only":
        return (["atoms CA"], excluded)
    if preset == "backbone":
        return (["atoms CA C N O"], excluded)
    if preset == "backbone-water":
        # Water oxygens intentionally kept (not in excluded)
        excluded_no_water = excluded - {"HOH", "WAT", "H2O"}
        return (["atoms CA C N O"], excluded_no_water)
    if preset == "none":
        return ([], excluded)
    raise ValueError(
        f"Unknown preset '{preset}'. "
        "Choose: ca-only, backbone, backbone-water, none"
    )
