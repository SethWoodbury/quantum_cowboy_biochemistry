"""
Geometry analysis for enzyme active site structures.

Measure bond distances, compare structures via RMSD, and track bond
evolution along NEB reaction paths.  Uses biotite for PDB parsing and
numpy for distance calculations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io


# ══════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════

AtomPair = Union[tuple[str, str, str, str], tuple[int, int]]
"""Either (resname1, atomname1, resname2, atomname2) or (idx1, idx2)."""


def _load_structure(pdb_path: Union[str, Path]) -> struc.AtomArray:
    """Load a PDB file and return the first model as an AtomArray."""
    pdb_file = pdb_io.PDBFile.read(str(pdb_path))
    return pdb_file.get_structure(model=1)


def _load_all_models(pdb_path: Union[str, Path]) -> struc.AtomArrayStack:
    """Load a multi-MODEL PDB file and return all models as an AtomArrayStack.

    Falls back to a single-model stack if only one model is present.
    """
    pdb_file = pdb_io.PDBFile.read(str(pdb_path))
    structure = pdb_file.get_structure()
    if isinstance(structure, struc.AtomArray):
        # Single model -- wrap in a stack for uniform handling
        return struc.stack([structure])
    return structure


def _resolve_atom_index(st: struc.AtomArray, spec: Union[tuple[str, str], int]) -> int:
    """Resolve an atom specification to an integer index.

    Parameters
    ----------
    st : AtomArray
        The structure to search.
    spec : (resname, atomname) or int
        If int, returned directly (after bounds check).
        If (resname, atomname), find the first matching atom.

    Returns
    -------
    int
        Zero-based atom index.
    """
    if isinstance(spec, int):
        if spec < 0 or spec >= len(st):
            raise IndexError(f"Atom index {spec} out of range [0, {len(st)})")
        return spec
    resname, atomname = spec
    mask = (st.res_name == resname) & (st.atom_name == atomname)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        raise ValueError(
            f"No atom found matching res_name={resname!r}, atom_name={atomname!r}"
        )
    if len(indices) > 1:
        # Take the first match but warn via the returned value -- callers
        # can refine with residue IDs if needed.
        pass
    return int(indices[0])


def _resolve_pair(
    st: struc.AtomArray, pair: AtomPair
) -> tuple[int, int, str]:
    """Resolve an atom pair specification to (idx1, idx2, label).

    Parameters
    ----------
    st : AtomArray
        Structure for name-based lookups.
    pair : AtomPair
        Either ``(resname1, atomname1, resname2, atomname2)`` or ``(idx1, idx2)``.

    Returns
    -------
    idx1 : int
    idx2 : int
    label : str
        Human-readable label for the bond.
    """
    if len(pair) == 4:
        resname1, atomname1, resname2, atomname2 = pair
        idx1 = _resolve_atom_index(st, (resname1, atomname1))
        idx2 = _resolve_atom_index(st, (resname2, atomname2))
        label = f"{resname1}:{atomname1}-{resname2}:{atomname2}"
    elif len(pair) == 2:
        idx1, idx2 = int(pair[0]), int(pair[1])
        label = f"atom{idx1}-atom{idx2}"
    else:
        raise ValueError(
            f"atom_pair must be (resname1, atomname1, resname2, atomname2) "
            f"or (idx1, idx2), got {pair!r}"
        )
    return idx1, idx2, label


def _distance(coords: np.ndarray, i: int, j: int) -> float:
    """Euclidean distance between two atoms."""
    return float(np.linalg.norm(coords[i] - coords[j]))


# ══════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════

def measure_bond_distances(
    pdb_path: Union[str, Path],
    atom_pairs: list[AtomPair],
) -> dict[str, float]:
    """Measure specific bond distances in a PDB structure.

    Parameters
    ----------
    pdb_path : str or Path
        Path to a PDB file.
    atom_pairs : list of AtomPair
        Each element is either
        ``(resname1, atomname1, resname2, atomname2)`` to select atoms by
        residue name and atom name, or ``(idx1, idx2)`` to select by
        zero-based atom index.

    Returns
    -------
    dict[str, float]
        Mapping of ``"RES1:ATOM1-RES2:ATOM2"`` (or ``"atomI-atomJ"``) to
        distance in Angstroms.
    """
    st = _load_structure(pdb_path)
    results: dict[str, float] = {}
    for pair in atom_pairs:
        idx1, idx2, label = _resolve_pair(st, pair)
        results[label] = _distance(st.coord, idx1, idx2)
    return results


def compare_structures(
    pdb1: Union[str, Path],
    pdb2: Union[str, Path],
    align: bool = True,
) -> dict[str, float]:
    """Compute RMSD between two structures (all-atom and CA-only).

    Parameters
    ----------
    pdb1, pdb2 : str or Path
        Paths to PDB files.  Must contain the same number of atoms in the
        same order for all-atom RMSD.
    align : bool, optional
        If True (default), optimally superimpose *pdb2* onto *pdb1* before
        computing RMSD (Kabsch algorithm via biotite).

    Returns
    -------
    dict with keys:
        - ``"rmsd_all"`` : float -- all-atom RMSD in Angstroms
        - ``"rmsd_ca"``  : float -- CA-only RMSD (None if no CA atoms)
        - ``"n_atoms"``  : int   -- number of atoms (all-atom)
        - ``"n_ca"``     : int   -- number of CA atoms
    """
    st1 = _load_structure(pdb1)
    st2 = _load_structure(pdb2)

    if len(st1) != len(st2):
        raise ValueError(
            f"Structures have different atom counts: {len(st1)} vs {len(st2)}. "
            "Cannot compute RMSD without atom correspondence."
        )

    # All-atom RMSD
    if align:
        fitted, _ = struc.superimpose(st1, st2)
        rmsd_all = struc.rmsd(st1, fitted)
    else:
        rmsd_all = struc.rmsd(st1, st2)

    # CA-only RMSD
    ca_mask1 = st1.atom_name == "CA"
    ca_mask2 = st2.atom_name == "CA"
    n_ca = int(np.sum(ca_mask1))
    rmsd_ca = None

    if n_ca > 0 and np.sum(ca_mask1) == np.sum(ca_mask2):
        st1_ca = st1[ca_mask1]
        st2_ca = st2[ca_mask2]
        if align:
            fitted_ca, _ = struc.superimpose(st1_ca, st2_ca)
            rmsd_ca = float(struc.rmsd(st1_ca, fitted_ca))
        else:
            rmsd_ca = float(struc.rmsd(st1_ca, st2_ca))

    return {
        "rmsd_all": float(rmsd_all),
        "rmsd_ca": rmsd_ca,
        "n_atoms": len(st1),
        "n_ca": n_ca,
    }


def track_bonds_along_path(
    neb_path_pdb: Union[str, Path],
    atom_pairs: list[AtomPair],
) -> dict[str, list[float]]:
    """Track bond distances across NEB images in a multi-MODEL PDB.

    Parameters
    ----------
    neb_path_pdb : str or Path
        Multi-MODEL PDB file (e.g. ``neb_path.pdb`` from ``run_neb_ts.py``).
    atom_pairs : list of AtomPair
        Bond specifications (see :func:`measure_bond_distances`).

    Returns
    -------
    dict[str, list[float]]
        Mapping of bond label to a list of distances (one per NEB image),
        in Angstroms.
    """
    stack = _load_all_models(neb_path_pdb)

    # Resolve pair indices using the first frame
    resolved: list[tuple[int, int, str]] = []
    for pair in atom_pairs:
        resolved.append(_resolve_pair(stack[0], pair))

    results: dict[str, list[float]] = {label: [] for _, _, label in resolved}

    for model_idx in range(stack.stack_depth()):
        coords = stack[model_idx].coord
        for idx1, idx2, label in resolved:
            results[label].append(_distance(coords, idx1, idx2))

    return results


def plot_bond_evolution(
    neb_path_pdb: Union[str, Path],
    atom_pairs: list[AtomPair],
    output_png: Union[str, Path, None] = None,
) -> None:
    """Plot bond distances vs NEB image index.

    Parameters
    ----------
    neb_path_pdb : str or Path
        Multi-MODEL PDB file from a NEB run.
    atom_pairs : list of AtomPair
        Bond specifications to track and plot.
    output_png : str, Path, or None
        If provided, save the figure to this path.  If None, call
        ``plt.show()`` interactively.
    """
    import matplotlib.pyplot as plt

    bond_data = track_bonds_along_path(neb_path_pdb, atom_pairs)
    n_images = max(len(v) for v in bond_data.values()) if bond_data else 0

    fig, ax = plt.subplots(figsize=(8, 5))
    image_indices = list(range(n_images))

    for label, distances in bond_data.items():
        ax.plot(image_indices, distances, "o-", label=label, markersize=5)

    ax.set_xlabel("NEB image index")
    ax.set_ylabel("Distance (A)")
    ax.set_title("Bond evolution along NEB path")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_png is not None:
        fig.savefig(str(output_png), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
