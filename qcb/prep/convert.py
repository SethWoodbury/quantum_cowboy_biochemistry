"""
Format conversions between biotite, ASE, PDB, XYZ, and Gaussian input files.

Provides round-trip conversion so that structures can move freely between
biotite (for annotation-rich manipulation), ASE (for calculators and
optimization), and text-based QM input formats.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from biotite.structure import AtomArray, Atom
from biotite.structure.io.pdb import PDBFile


# ---------------------------------------------------------------------------
#  biotite <-> ASE
# ---------------------------------------------------------------------------

def biotite_to_ase(st: AtomArray):
    """Convert a biotite AtomArray to an ASE Atoms object.

    Custom biotite annotations (chain_id, res_id, res_name, atom_name, etc.)
    are stored in ``atoms.info["biotite_annotations"]`` so they can be
    recovered by :func:`ase_to_biotite`.

    Args:
        st: A biotite :class:`AtomArray`.

    Returns:
        An :class:`ase.Atoms` object with coordinates, elements, and
        biotite metadata stored in ``atoms.info``.
    """
    from ase import Atoms as ASEAtoms

    elements = [e.capitalize() for e in st.element]
    atoms = ASEAtoms(symbols=elements, positions=st.coord.copy())

    # Store biotite annotations for round-tripping
    annotations: dict[str, list] = {}
    for name in st.get_annotation_categories():
        annotations[name] = list(st.get_annotation(name))
    atoms.info["biotite_annotations"] = annotations

    return atoms


def ase_to_biotite(atoms) -> AtomArray:
    """Convert an ASE Atoms object back to a biotite AtomArray.

    If the ASE object carries ``atoms.info["biotite_annotations"]`` (as
    produced by :func:`biotite_to_ase`), those annotations are restored.
    Otherwise only element and coordinate information is preserved.

    Args:
        atoms: An :class:`ase.Atoms` object.

    Returns:
        A biotite :class:`AtomArray`.
    """
    n = len(atoms)
    st = AtomArray(n)
    st.coord = np.array(atoms.get_positions(), dtype=np.float32)
    st.element = np.array([s.upper() for s in atoms.get_chemical_symbols()])

    # Restore saved annotations if available
    saved = atoms.info.get("biotite_annotations", {})
    if saved:
        for name, values in saved.items():
            if name in ("coord", "element"):
                continue  # already set
            try:
                st.set_annotation(name, np.array(values))
            except Exception:
                # Annotation may not be compatible; skip gracefully
                pass
    else:
        # Minimal defaults so the AtomArray is usable
        st.atom_name = np.array([el for el in st.element])
        st.res_name = np.full(n, "UNK")
        st.chain_id = np.full(n, "A")
        st.res_id = np.ones(n, dtype=int)
        st.ins_code = np.full(n, "")
        st.hetero = np.zeros(n, dtype=bool)
        st.b_factor = np.zeros(n, dtype=np.float32)
        st.occupancy = np.ones(n, dtype=np.float32)

    return st


# ---------------------------------------------------------------------------
#  PDB <-> XYZ
# ---------------------------------------------------------------------------

def pdb_to_xyz(pdb_path: str | Path, xyz_path: str | Path) -> Path:
    """Convert a PDB file to XYZ format.

    The XYZ file contains the number of atoms, a comment line with the
    source filename, and then one ``element x y z`` line per atom.

    Args:
        pdb_path: Path to the input PDB file.
        xyz_path: Path for the output XYZ file.

    Returns:
        Resolved path to the written XYZ file.
    """
    pdb_path = Path(pdb_path).resolve()
    xyz_path = Path(xyz_path).resolve()

    structure = _read_pdb(pdb_path)

    with open(xyz_path, "w") as fh:
        fh.write(f"{len(structure)}\n")
        fh.write(f"Converted from {pdb_path.name}\n")
        for i in range(len(structure)):
            el = structure.element[i].capitalize()
            x, y, z = structure.coord[i]
            fh.write(f"{el:<4s} {x:14.8f} {y:14.8f} {z:14.8f}\n")

    return xyz_path


def xyz_to_pdb(
    xyz_path: str | Path,
    template_pdb: str | Path,
    out_path: str | Path,
) -> Path:
    """Convert an XYZ file back to PDB, using a template for atom metadata.

    Coordinates are taken from the XYZ file; atom names, residue names,
    chain IDs, and all other annotations come from the template PDB.
    The number of atoms in the XYZ and template must match.

    Args:
        xyz_path: Path to the input XYZ file.
        template_pdb: Path to the template PDB that provides atom-level
            annotations (atom names, residue info, etc.).
        out_path: Path for the output PDB file.

    Returns:
        Resolved path to the written PDB file.
    """
    xyz_path = Path(xyz_path).resolve()
    template_pdb = Path(template_pdb).resolve()
    out_path = Path(out_path).resolve()

    template = _read_pdb(template_pdb)
    xyz_coords, xyz_elements = _read_xyz(xyz_path)

    if len(xyz_coords) != len(template):
        raise ValueError(
            f"Atom count mismatch: XYZ has {len(xyz_coords)} atoms, "
            f"template PDB has {len(template)} atoms."
        )

    # Update template coordinates
    result = template.copy()
    result.coord = xyz_coords

    pdb_file = PDBFile()
    pdb_file.set_structure(result)
    pdb_file.write(str(out_path))

    return out_path


# ---------------------------------------------------------------------------
#  Gaussian input writer
# ---------------------------------------------------------------------------

def write_gaussian_input(
    atoms: AtomArray,
    output_path: str | Path,
    charge: int,
    multiplicity: int = 1,
    method: str = "B3LYP",
    basis: str = "6-31G*",
    title: str = "QCB generated input",
    extra_keywords: str = "",
    memory: str = "4GB",
    nproc: int = 8,
) -> Path:
    """Write a Gaussian .com input file from a biotite AtomArray.

    Args:
        atoms: Biotite AtomArray with the molecular geometry.
        output_path: Path for the output .com file.
        charge: Net formal charge of the system.
        multiplicity: Spin multiplicity (default 1 = singlet).
        method: QM method string (e.g. ``"B3LYP"``, ``"wB97XD"``).
        basis: Basis set string (e.g. ``"6-31G*"``, ``"def2-SVP"``).
        title: Title line for the Gaussian input.
        extra_keywords: Additional Gaussian keywords to append to the
            route line (e.g. ``"opt freq"``).
        memory: Memory allocation (e.g. ``"4GB"``).
        nproc: Number of processors.

    Returns:
        Resolved path to the written .com file.
    """
    output_path = Path(output_path).resolve()

    route = f"# {method}/{basis}"
    if extra_keywords:
        route += f" {extra_keywords}"

    lines: list[str] = []
    lines.append(f"%mem={memory}")
    lines.append(f"%nproc={nproc}")
    lines.append(route)
    lines.append("")
    lines.append(title)
    lines.append("")
    lines.append(f"{charge} {multiplicity}")

    for i in range(len(atoms)):
        el = atoms.element[i].capitalize()
        x, y, z = atoms.coord[i]
        lines.append(f"{el:<4s} {x:14.8f} {y:14.8f} {z:14.8f}")

    lines.append("")  # blank line after coordinates
    lines.append("")  # Gaussian requires trailing blank line

    with open(output_path, "w") as fh:
        fh.write("\n".join(lines))

    return output_path


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _read_pdb(pdb_path: Path) -> AtomArray:
    """Read a PDB file into a single-model AtomArray."""
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    pdb_file = PDBFile.read(str(pdb_path))
    return pdb_file.get_structure(model=1)


def _read_xyz(xyz_path: Path) -> tuple[np.ndarray, list[str]]:
    """Read an XYZ file and return (coords, elements).

    Args:
        xyz_path: Path to the XYZ file.

    Returns:
        Tuple of (coords array of shape (N, 3), list of element symbols).
    """
    if not xyz_path.exists():
        raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

    with open(xyz_path) as fh:
        lines = fh.readlines()

    n_atoms = int(lines[0].strip())
    # Line 1 is a comment; atom data starts at line 2
    coords = []
    elements = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        elements.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return np.array(coords, dtype=np.float32), elements
