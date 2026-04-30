"""
Backbone H-capping for cluster-cut active sites.

When you carve a cluster out of a protein (e.g. with
:func:`qcb.prep.extract.extract_active_site`), some residues end up with
severed peptide bonds — their backbone N (at the residue's "left" end) or
backbone C (at the "right" end) was previously bonded to a neighbouring
residue that didn't make the cut. For QM/MLFF calculations these dangling
heavy atoms must be capped, otherwise they'll see implicit valence
violations.

This module implements a lightweight **H-cap** strategy: for every dangling
N or C, place a single hydrogen along the CA→terminal axis at standard
N–H / C–H distance. That's appropriate when the enclosing pipeline plans
to freeze backbone atoms (so the cap doesn't need to model the missing
neighbour residue's electronics — it just needs to satisfy valence).

For workflows where backbone atoms move freely, prefer ACE/NME caps (not
yet implemented; would mimic enz-ts ``capping_utils.py``).

Public API
----------
- :func:`cap_backbone_h` — pure function: AtomArray → AtomArray.
- :data:`PROTEIN_RES` — set of canonical 3-letter residue codes.
"""
from __future__ import annotations

import numpy as np
from biotite.structure import AtomArray, array

PROTEIN_RES: set[str] = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def _residue_keys(atoms: AtomArray) -> np.ndarray:
    """Build a structured array of (chain_id, res_id, ins_code) per atom for
    O(N) residue identity comparisons."""
    return np.array(
        list(zip(atoms.chain_id, atoms.res_id, atoms.ins_code)),
        dtype=[("chain", "U4"), ("res_id", "i4"), ("ins", "U1")],
    )


def cap_backbone_h(
    atoms: AtomArray,
    bond_length: float = 1.0,
    cap_name: str = "HCAP",
) -> AtomArray:
    """Add H caps to dangling backbone N or C atoms at chain breaks.

    For every protein residue ``r`` in *atoms*, look for its left neighbour
    ``(r.chain, r.res_id - 1)`` and right neighbour ``(r.chain, r.res_id + 1)``.
    If a neighbour is missing AND the corresponding terminal heavy atom (N
    on the left, C on the right) is present, append a single H along the
    CA→terminal direction at *bond_length*.

    Args:
        atoms: input :class:`biotite.structure.AtomArray`. Not modified.
        bond_length: cap N–H / C–H distance in Å. Default 1.0 Å — chosen
            to be intentionally a bit shorter than equilibrium so that the
            first energy minimization step doesn't kick the cap into a
            valence-violating geometry.
        cap_name: ``atom_name`` to assign to the new caps so downstream
            code can identify (and e.g. freeze) them. Defaults to
            ``"HCAP"``.

    Returns:
        New :class:`AtomArray` with cap H atoms appended at the end. The
        original *atoms* is unchanged. If no caps are needed, the original
        array is returned (after a defensive copy).
    """
    if len(atoms) == 0:
        return atoms.copy()

    # Build (chain, res_id, ins_code) → set lookup once
    res_keys = _residue_keys(atoms)
    present_residues: set[tuple[str, int, str]] = {
        (str(k["chain"]), int(k["res_id"]), str(k["ins"])) for k in res_keys
    }
    is_protein = np.isin(atoms.res_name, list(PROTEIN_RES))

    # Group atom indices per residue for fast position lookup
    res_to_atoms: dict[tuple[str, int, str], dict[str, int]] = {}
    for i in range(len(atoms)):
        if not is_protein[i]:
            continue
        key = (str(atoms.chain_id[i]), int(atoms.res_id[i]), str(atoms.ins_code[i]))
        res_to_atoms.setdefault(key, {})[atoms.atom_name[i].strip()] = i

    cap_records: list[dict] = []
    for key, name_to_idx in res_to_atoms.items():
        chain, rnum, ins = key
        for offset, target_atom in ((-1, "N"), (+1, "C")):
            neighbor = (chain, rnum + offset, ins)
            if neighbor in present_residues:
                continue
            if target_atom not in name_to_idx or "CA" not in name_to_idx:
                continue
            ca_idx = name_to_idx["CA"]
            tgt_idx = name_to_idx[target_atom]
            v = atoms.coord[tgt_idx] - atoms.coord[ca_idx]
            n = float(np.linalg.norm(v))
            if n < 1e-9:
                continue
            cap_pos = atoms.coord[tgt_idx] + (v / n) * bond_length
            cap_records.append({
                "chain_id": chain,
                "res_id": rnum,
                "ins_code": ins,
                "res_name": atoms.res_name[ca_idx],
                "atom_name": cap_name,
                "element": "H",
                "hetero": False,
                "coord": cap_pos,
            })

    if not cap_records:
        return atoms.copy()

    # Build the cap AtomArray and concatenate to the input
    n_caps = len(cap_records)
    caps = AtomArray(n_caps)
    caps.chain_id = np.array([r["chain_id"] for r in cap_records])
    caps.res_id = np.array([r["res_id"] for r in cap_records], dtype=int)
    caps.ins_code = np.array([r["ins_code"] for r in cap_records])
    caps.res_name = np.array([r["res_name"] for r in cap_records])
    caps.atom_name = np.array([r["atom_name"] for r in cap_records])
    caps.element = np.array([r["element"] for r in cap_records])
    caps.hetero = np.array([r["hetero"] for r in cap_records])
    caps.coord = np.array([r["coord"] for r in cap_records])

    return atoms + caps
