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
# Cached tuple form for np.isin: avoids re-coercing the set each call.
_PROTEIN_RES_TUPLE = tuple(sorted(PROTEIN_RES))


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
    *,
    original: AtomArray | None = None,
) -> AtomArray:
    """Add H caps to dangling backbone N or C atoms at chain breaks.

    For every protein residue ``r`` in *atoms*, look for an adjacent
    *protein* residue with ``res_id - 1`` (N-terminal direction) or
    ``res_id + 1`` (C-terminal direction) on the same chain. If no
    adjacent residue is in *original* (or in *atoms* if no original
    provided) AND the corresponding terminal heavy atom (N on the left,
    C on the right) is present, append a single H along the CA→terminal
    direction at *bond_length*.

    The H sits along the existing CA→N (or CA→C) axis extrapolated past
    the heavy atom, giving a linear (180°) cap geometry. That's
    intentional: for backbone-frozen QM/MLFF clusters this is the
    cheapest valence-saturating cap that doesn't require additional
    geometry decisions; the cap atom doesn't model real chemistry.

    Args:
        atoms: input :class:`biotite.structure.AtomArray`. Not modified.
        bond_length: cap N–H / C–H distance in Å. Default 1.0 Å, slightly
            below equilibrium (≈ 1.01 N–H, 1.09 C–H) so the first energy
            minimization step doesn't kick the cap into a clash.
        cap_name: ``atom_name`` to assign to the new caps so downstream
            code can identify (and e.g. freeze) them. Defaults to
            ``"HCAP"``.
        original: full *unchopped* :class:`AtomArray` from which *atoms*
            was extracted. Pass this so we can distinguish a real chain
            break (= cap needed) from a true natural terminus (= no cap,
            because there was nothing to bond to in the parent structure
            either). If ``None``, we fall back to *atoms* itself, which
            will over-cap the genuine N- and C-termini of every chain
            and treat HETATM-adjacent residues as breaks too.

    Returns:
        New :class:`AtomArray` with cap H atoms appended at the end. The
        original *atoms* is unchanged. If no caps are needed, a defensive
        copy of *atoms* is returned.
    """
    if len(atoms) == 0:
        return atoms.copy()

    # Build (chain, res_id) protein-residue sets for both the cluster and,
    # if provided, the parent structure. We ignore ins_code in this set:
    # res_id arithmetic doesn't carry the insertion-code letter, and a
    # split insert cluster like 27/27A/27B is correctly summarised as a
    # single "(chain, 27)" presence.
    cluster_protein_keys: set[tuple[str, int]] = {
        (str(atoms.chain_id[i]), int(atoms.res_id[i]))
        for i in np.where(np.isin(atoms.res_name, _PROTEIN_RES_TUPLE))[0]
    }
    if original is not None:
        original_protein_keys: set[tuple[str, int]] = {
            (str(original.chain_id[i]), int(original.res_id[i]))
            for i in np.where(np.isin(original.res_name, _PROTEIN_RES_TUPLE))[0]
        }
    else:
        original_protein_keys = cluster_protein_keys  # fallback: over-caps termini

    # Group ATOM-array indices by (chain, res_id, ins_code) for the
    # protein subset of `atoms`.
    is_protein_atoms = np.isin(atoms.res_name, _PROTEIN_RES_TUPLE)
    res_to_atoms: dict[tuple[str, int, str], dict[str, int]] = {}
    for i in np.where(is_protein_atoms)[0]:
        key = (str(atoms.chain_id[i]), int(atoms.res_id[i]),
               str(atoms.ins_code[i]))
        res_to_atoms.setdefault(key, {})[atoms.atom_name[i].strip()] = i

    cap_records: list[dict] = []
    for key, name_to_idx in res_to_atoms.items():
        chain, rnum, _ins = key
        for offset, target_atom in ((-1, "N"), (+1, "C")):
            neighbor_key = (chain, rnum + offset)
            # Internal cluster residue → no cap needed.
            if neighbor_key in cluster_protein_keys:
                continue
            # Boundary residue: cap only if there *was* a peptide-bonded
            # neighbour in the parent structure. Without that we'd be
            # adding an HCAP onto a real protein terminus that's already
            # chemically saturated (NH3⁺ / COOH).
            if neighbor_key not in original_protein_keys:
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
                "ins_code": str(atoms.ins_code[ca_idx]),
                "res_name": str(atoms.res_name[ca_idx]),
                "atom_name": cap_name,
                "element": "H",
                # Inherit hetero status from the host residue's CA so the
                # cap doesn't break PDB ATOM/HETATM consistency.
                "hetero": bool(atoms.hetero[ca_idx]),
                "coord": cap_pos,
            })

    if not cap_records:
        return atoms.copy()

    # Build the cap AtomArray with the host array's exact dtypes so we
    # can concatenate without silent string-dtype broadening.
    n_caps = len(cap_records)
    caps = AtomArray(n_caps)
    caps.chain_id = np.array([r["chain_id"] for r in cap_records],
                              dtype=atoms.chain_id.dtype)
    caps.res_id = np.array([r["res_id"] for r in cap_records],
                            dtype=atoms.res_id.dtype)
    caps.ins_code = np.array([r["ins_code"] for r in cap_records],
                              dtype=atoms.ins_code.dtype)
    caps.res_name = np.array([r["res_name"] for r in cap_records],
                              dtype=atoms.res_name.dtype)
    caps.atom_name = np.array([r["atom_name"] for r in cap_records],
                               dtype=atoms.atom_name.dtype)
    caps.element = np.array([r["element"] for r in cap_records],
                             dtype=atoms.element.dtype)
    caps.hetero = np.array([r["hetero"] for r in cap_records], dtype=bool)
    caps.coord = np.array([r["coord"] for r in cap_records], dtype=np.float64)

    return atoms + caps
