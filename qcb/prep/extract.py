"""
Active-site extraction from PDB structures.

Extracts atoms surrounding a ligand (or set of ligands) by distance cutoff,
keeping whole residues intact.  Supports single-radius and multi-zone
extraction for QM/MM-style partitioning, plus:

- *gap-filling*: small (≤ N residues) gaps in the consecutive-residue
  window of each chain are filled in so the cluster doesn't have isolated
  single residues sticking out
- *always-include*: explicit catalytic residues always make the cut
  regardless of distance (useful when the catres list is known a priori
  from a REMARK 666 line)
- pairs cleanly with :func:`qcb.prep.cap.cap_backbone_h` for downstream
  H-capping of the severed peptide bonds
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

log = logging.getLogger("qcb.prep.extract")


def _load_structure(pdb_path: str | Path) -> AtomArray:
    """Read a PDB file and return a single-model AtomArray."""
    pdb_path = Path(pdb_path).resolve()
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    pdb_file = PDBFile.read(str(pdb_path))
    structure = pdb_file.get_structure(model=1)
    return structure


def _get_ligand_mask(structure: AtomArray, ligand_names: list[str]) -> np.ndarray:
    """Return a boolean mask selecting atoms whose res_name is in *ligand_names*."""
    mask = np.isin(structure.res_name, ligand_names)
    if not mask.any():
        available = sorted(set(structure.res_name))
        raise ValueError(
            f"No atoms found for ligand(s) {ligand_names}. "
            f"Available residue names: {available}"
        )
    return mask


def _min_distances_to_selection(
    structure: AtomArray,
    selection_mask: np.ndarray,
) -> np.ndarray:
    """Return per-atom minimum distance to any atom in the selection.

    Args:
        structure: Full atom array.
        selection_mask: Boolean mask for reference atoms (e.g. ligand).

    Returns:
        1-D float array of shape ``(len(structure),)`` with the minimum
        Euclidean distance from each atom to the closest selected atom.
    """
    ref_coords = structure.coord[selection_mask]  # (M, 3)
    all_coords = structure.coord                  # (N, 3)

    # Chunked computation to keep memory reasonable for large structures.
    # For each atom, compute min distance to any reference atom.
    chunk_size = 5000
    n_atoms = len(all_coords)
    min_dists = np.empty(n_atoms, dtype=np.float64)

    for start in range(0, n_atoms, chunk_size):
        end = min(start + chunk_size, n_atoms)
        # (chunk, 1, 3) - (1, M, 3) -> (chunk, M, 3)
        diff = all_coords[start:end, np.newaxis, :] - ref_coords[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=-1))  # (chunk, M)
        min_dists[start:end] = dists.min(axis=1)

    return min_dists


def _whole_residue_mask(
    structure: AtomArray,
    atom_mask: np.ndarray,
) -> np.ndarray:
    """Expand an atom-level boolean mask to include every atom in each
    selected residue (so that no residue is partially included).

    Residue identity is determined by the combination of
    ``(chain_id, res_id, ins_code)``.
    """
    # Build unique residue keys for every atom
    selected_keys: set[tuple[str, int, str]] = set()
    for i in np.where(atom_mask)[0]:
        selected_keys.add((
            structure.chain_id[i],
            int(structure.res_id[i]),
            structure.ins_code[i],
        ))

    whole_mask = np.array([
        (structure.chain_id[i], int(structure.res_id[i]), structure.ins_code[i])
        in selected_keys
        for i in range(len(structure))
    ], dtype=bool)
    return whole_mask


def fill_residue_gaps(
    structure: AtomArray,
    keep_mask: np.ndarray,
    max_gap: int = 2,
) -> np.ndarray:
    """Expand a residue-level keep mask to fill small gaps in the residue-id
    sequence within each chain.

    For each chain, look at the kept *protein* residue ids: if two
    consecutive kept ids satisfy ``0 < (nxt - prev - 1) ≤ max_gap``,
    include all residues with ids in ``(prev, nxt)`` — i.e. fill any gap
    of at most ``max_gap`` residues. This avoids cutting clusters across
    isolated "lonely" residues; a ±1 gap typically means a single residue
    happened to fall just outside the radius but is part of the same
    secondary structure element.

    Only protein-chain residues participate in the analysis; promoted
    atoms are restricted to the protein subset of the chain too, so a
    HETATM with the same `res_id` won't sneak in via gap-fill.

    A warning is logged if a chain has duplicate residue ids (e.g. two
    independent segments numbered identically) — gap-fill would collapse
    them, which is rarely what you want.

    Args:
        structure: full AtomArray.
        keep_mask: atom-level boolean mask (length ``len(structure)``)
            after an initial selection pass.
        max_gap: maximum gap (in residues) that gets filled. Default 2.
            Set to 0 to disable. Mirrors enz-ts ``_fill_in_consec_res``.

    Returns:
        New atom-level boolean mask with small protein-chain gaps filled.
    """
    if max_gap <= 0:
        return keep_mask.copy()

    new_mask = keep_mask.copy()
    protein_mask_all = ~structure.hetero
    chains = np.unique(structure.chain_id)
    for chain in chains:
        chain_mask = structure.chain_id == chain
        protein_in_chain = chain_mask & protein_mask_all
        if not protein_in_chain.any():
            continue

        kept_in_chain = chain_mask & keep_mask & protein_mask_all
        if not kept_in_chain.any():
            continue

        kept_res_ids = np.unique(structure.res_id[kept_in_chain])
        if len(kept_res_ids) < 2:
            continue

        # Sanity check: count residues with this chain id and >1 ins-code
        # span. Duplicate res_ids in the same chain (e.g. two segments
        # of identical numbering) would be merged by the gap-fill below.
        seg_atoms = structure.res_id[protein_in_chain]
        if len(np.unique(seg_atoms)) < seg_atoms.shape[0]:
            # That count is meaningless on its own; the real test is
            # whether res_ids are not strictly increasing within the chain
            sorted_ids = seg_atoms[np.argsort(seg_atoms)]
            if not np.all(np.diff(sorted_ids[sorted_ids != np.roll(sorted_ids, 1)]) >= 0):
                pass  # the count test below is the actual one
        # Cheap test: are kept_res_ids monotonic with respect to atom order?
        # If a chain has duplicates the np.unique won't tell us, so
        # additionally check that the chain's first→last residue sequence
        # is non-decreasing.
        atom_order_ids = structure.res_id[chain_mask]
        if np.any(np.diff(atom_order_ids) < 0):
            log.warning(
                f"chain {chain!r} has non-monotonic res_ids — gap-fill may "
                "merge what the source PDB intended as separate segments"
            )

        to_add: list[int] = []
        for prev, nxt in zip(kept_res_ids[:-1], kept_res_ids[1:]):
            gap = int(nxt) - int(prev) - 1
            if 0 < gap <= max_gap:
                to_add.extend(range(int(prev) + 1, int(nxt)))
        if not to_add:
            continue

        # Crucial: confine the fill to protein atoms of this chain. A
        # bare `chain_mask & np.isin(res_id, to_add)` would sweep in any
        # HETATM that happens to share a res_id (rare but documented in
        # CHEMBL ligand entries).
        fill_mask = protein_in_chain & np.isin(structure.res_id, to_add)
        new_mask |= fill_mask

    return new_mask


def _always_include_mask(
    structure: AtomArray,
    always_include: (
        list[tuple[str, int] | tuple[str, int, str]] | None
    ),
) -> np.ndarray:
    """Boolean mask selecting all atoms in residues listed in
    ``always_include``.

    Each entry can be ``(chain_id, res_id)`` (matches all insertion codes)
    or ``(chain_id, res_id, ins_code)`` (matches a specific insertion).
    """
    if not always_include:
        return np.zeros(len(structure), dtype=bool)
    mask = np.zeros(len(structure), dtype=bool)
    for entry in always_include:
        if len(entry) == 2:
            chain, rid = entry
            mask |= (structure.chain_id == chain) & (structure.res_id == rid)
        elif len(entry) == 3:
            chain, rid, ins = entry
            mask |= (
                (structure.chain_id == chain)
                & (structure.res_id == rid)
                & (structure.ins_code == ins)
            )
        else:
            raise ValueError(
                f"always_include entries must be (chain, res_id) or "
                f"(chain, res_id, ins_code); got {entry!r}"
            )
    return mask


def extract_active_site(
    pdb_path: str | Path,
    ligand_names: list[str],
    radius: float = 5.0,
    include_waters: bool = True,
    always_include: list[tuple[str, int]] | None = None,
    gap_fill: int = 0,
) -> AtomArray:
    """Extract atoms within *radius* angstroms of a ligand, keeping whole residues.

    The ligand itself is always included.  Water molecules (HOH / WAT) within
    the radius are included only when *include_waters* is True.

    Args:
        pdb_path: Path to a PDB file.
        ligand_names: Residue name(s) of the ligand(s) to use as the
            distance reference (e.g. ``["FMN"]`` or ``["SUB", "CO"]``).
        radius: Distance cutoff in angstroms. All residues that have at
            least one atom within this distance of any ligand atom are kept.
        include_waters: Whether to keep water molecules (HOH/WAT) that
            fall within the radius.
        always_include: List of ``(chain_id, res_id)`` tuples that must
            always be in the cut, even if they fall outside the radius.
            Useful for catalytic residues identified ahead of time
            (e.g. from a REMARK 666 line).
        gap_fill: Fill polypeptide-chain residue-id gaps of up to this
            many residues so the cluster doesn't have isolated lonely
            residues. Default 0 (no fill). 1-3 is sensible.

    Returns:
        A biotite :class:`AtomArray` containing the extracted atoms.
    """
    structure = _load_structure(pdb_path)
    ligand_mask = _get_ligand_mask(structure, ligand_names)
    min_dists = _min_distances_to_selection(structure, ligand_mask)

    within_radius = min_dists <= radius
    residue_mask = _whole_residue_mask(structure, within_radius)

    # Always include the ligand itself
    residue_mask |= ligand_mask

    # Always include explicit catres
    residue_mask |= _always_include_mask(structure, always_include)

    # Fill small residue-id gaps so cuts happen at sensible boundaries
    residue_mask = fill_residue_gaps(structure, residue_mask, max_gap=gap_fill)

    # Optionally exclude waters (after gap-fill so we don't accidentally
    # promote a water inside the gap; gap-fill ignores hetero already).
    if not include_waters:
        water_mask = np.isin(structure.res_name, ["HOH", "WAT"])
        residue_mask &= ~water_mask

    return structure[residue_mask]


def extract_by_zones(
    pdb_path: str | Path,
    ligand_names: list[str],
    zones: list[float] | None = None,
    include_waters: bool = True,
    always_include: list[tuple[str, int]] | None = None,
    gap_fill: int = 0,
) -> AtomArray:
    """Zone-based extraction around a ligand.

    Assigns each residue to the innermost zone that contains at least one of
    its atoms.  Zones are defined by cumulative distance cutoffs:

    * zone 0: atoms within ``zones[0]`` angstroms of the ligand
    * zone 1: atoms between ``zones[0]`` and ``zones[1]`` angstroms
    * ...

    The ligand itself is assigned to zone -1.

    The zone index is stored in a custom annotation ``"zone"`` on the
    returned :class:`AtomArray`.

    Args:
        pdb_path: Path to a PDB file.
        ligand_names: Residue name(s) of the ligand(s).
        zones: List of cumulative distance cutoffs in angstroms.
            Defaults to ``[5.0, 10.0]``.
        include_waters: Whether to keep water molecules that fall within the
            outermost zone.
        always_include: List of ``(chain, res_id)`` (or
            ``(chain, res_id, ins_code)``) tuples that must be in the
            extraction regardless of distance — typically catres from a
            REMARK 666 line.
        gap_fill: Fill polypeptide-chain residue-id gaps of up to this many
            residues. Default 0 (no fill).

    Returns:
        A biotite :class:`AtomArray` with an integer annotation ``zone``
        indicating which zone each atom belongs to (-1 for ligand atoms,
        0 for the innermost zone, 1 for the next, etc.).
    """
    if zones is None:
        zones = [5.0, 10.0]

    if not zones or any(z <= 0 for z in zones):
        raise ValueError("zones must be a non-empty list of positive floats")
    zones = sorted(zones)

    structure = _load_structure(pdb_path)
    ligand_mask = _get_ligand_mask(structure, ligand_names)
    min_dists = _min_distances_to_selection(structure, ligand_mask)

    outer_radius = zones[-1]
    within_outer = min_dists <= outer_radius
    residue_mask = _whole_residue_mask(structure, within_outer)

    # Always include ligand
    residue_mask |= ligand_mask

    # Always include explicit catres
    residue_mask |= _always_include_mask(structure, always_include)

    # Fill small residue-id gaps inside the outer zone
    residue_mask = fill_residue_gaps(structure, residue_mask, max_gap=gap_fill)

    # Optionally exclude waters
    if not include_waters:
        water_mask = np.isin(structure.res_name, ["HOH", "WAT"])
        residue_mask &= ~water_mask

    extracted = structure[residue_mask]
    extracted_dists = min_dists[residue_mask]
    extracted_ligand = ligand_mask[residue_mask]

    # Assign zone annotations
    zone_arr = np.full(len(extracted), len(zones), dtype=int)  # beyond last zone

    # Work from outermost to innermost so inner zones overwrite outer
    for zone_idx in range(len(zones) - 1, -1, -1):
        cutoff = zones[zone_idx]
        zone_arr[extracted_dists <= cutoff] = zone_idx

    # Ligand atoms get zone -1
    zone_arr[extracted_ligand] = -1

    # Expand atom-level zones to whole residues (use minimum zone per residue)
    residue_keys_to_min_zone: dict[tuple[str, int, str], int] = {}
    for i in range(len(extracted)):
        key = (
            extracted.chain_id[i],
            int(extracted.res_id[i]),
            extracted.ins_code[i],
        )
        if key not in residue_keys_to_min_zone:
            residue_keys_to_min_zone[key] = zone_arr[i]
        else:
            residue_keys_to_min_zone[key] = min(
                residue_keys_to_min_zone[key], zone_arr[i]
            )

    # Write back uniform zone per residue
    for i in range(len(extracted)):
        key = (
            extracted.chain_id[i],
            int(extracted.res_id[i]),
            extracted.ins_code[i],
        )
        zone_arr[i] = residue_keys_to_min_zone[key]

    extracted.set_annotation("zone", zone_arr)
    return extracted
