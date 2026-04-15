"""
Active-site extraction from PDB structures.

Extracts atoms surrounding a ligand (or set of ligands) by distance cutoff,
keeping whole residues intact.  Supports single-radius and multi-zone
extraction for QM/MM-style partitioning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile


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


def extract_active_site(
    pdb_path: str | Path,
    ligand_names: list[str],
    radius: float = 5.0,
    include_waters: bool = True,
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

    # Optionally exclude waters
    if not include_waters:
        water_mask = np.isin(structure.res_name, ["HOH", "WAT"])
        residue_mask &= ~water_mask

    return structure[residue_mask]


def extract_by_zones(
    pdb_path: str | Path,
    ligand_names: list[str],
    zones: list[float] | None = None,
    include_waters: bool = True,
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
