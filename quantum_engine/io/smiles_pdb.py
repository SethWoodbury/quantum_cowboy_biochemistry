"""SMILES → PDB / XYZ → PDB helpers.

Two utilities used by Stage 1 (ParseReaction) and Stage 2 (autodE
adapter) of the enzyme_ts_design pipeline so reactant / TS / product
3D structures land as PDBs alongside the XYZ outputs — PyMOL renders
PDB with proper bond inference and per-atom labelling, while XYZ is
just element + coords.

These helpers also live in :mod:`tools.smiles_to_pdb` as a one-shot
script for backfilling visualisations of older runs.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("quantum_engine.io.smiles_pdb")


def smiles_to_pdb(smiles: str, output: Path | str, *,
                  residue_prefix: str = "LIG",
                  uff_iters: int = 200,
                  random_seed: int = 42) -> Path:
    """Embed a SMILES string into 3D and write a PDB.

    Multi-component SMILES (``A.B.C``) are kept together in one mol —
    each fragment gets the same residue but distinct atom indices, which
    matches how vacuum reactant / product complexes look in PyMOL.

    The UFF cleanup is *not* an optimisation; it just declashes the
    initial embed. Real optimisation happens downstream (autodE / xTB /
    MLFF / DFT).

    Returns the output path.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    output = Path(output)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    if AllChem.EmbedMolecule(mol, params) < 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) < 0:
            raise RuntimeError(f"ETKDG embedding failed for {smiles!r}")

    if uff_iters > 0:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=uff_iters)
        except Exception:
            pass  # UFF can fail on weird ligands; PDB still useful

    # Set monomer info so PyMOL gets readable atom names + residues
    rname = residue_prefix.ljust(3)[:3]
    for i, atom in enumerate(mol.GetAtoms()):
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName(rname)
        info.SetName(f" {atom.GetSymbol():<2}{i:>1}"[:4])
        info.SetResidueNumber(1)
        info.SetChainId("A")
        atom.SetMonomerInfo(info)

    output.parent.mkdir(parents=True, exist_ok=True)
    Chem.MolToPDBFile(mol, str(output))
    return output


def xyz_to_pdb(xyz_path: Path | str, output: Path | str, *,
               reference_smiles: str | None = None,
               residue_prefix: str = "LIG") -> Path:
    """Convert an XYZ file to PDB. ASE's PDB writer is element-only — if
    we have a ``reference_smiles`` for the same molecule we re-build
    bond connectivity via RDKit's atom-mapped order so PyMOL shows
    proper bonds. Otherwise we fall back to ASE's bond-by-distance
    inference.
    """
    from ase.io import read as ase_read
    from ase.io import write as ase_write

    xyz_path = Path(xyz_path)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if reference_smiles is None:
        atoms = ase_read(str(xyz_path))
        ase_write(str(output), atoms, format="proteindatabank")
        return output

    # Try the RDKit-bond path: parse SMILES, add Hs, override coords
    # from the XYZ.
    try:
        from rdkit import Chem
        from rdkit.Geometry import Point3D
    except ImportError:
        atoms = ase_read(str(xyz_path))
        ase_write(str(output), atoms, format="proteindatabank")
        return output

    mol = Chem.MolFromSmiles(reference_smiles)
    if mol is None:
        log.warning(f"smiles parse failed for {reference_smiles!r}; "
                    "falling back to element-only PDB")
        atoms = ase_read(str(xyz_path))
        ase_write(str(output), atoms, format="proteindatabank")
        return output
    mol = Chem.AddHs(mol)
    atoms = ase_read(str(xyz_path))

    if mol.GetNumAtoms() != len(atoms):
        log.warning(
            f"atom count mismatch — SMILES has {mol.GetNumAtoms()}, "
            f"XYZ has {len(atoms)}; falling back to element-only PDB"
        )
        ase_write(str(output), atoms, format="proteindatabank")
        return output

    # Trust the order matches (autodE preserves SMILES atom order in
    # its XYZ writes after embedding); override coordinates.
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(atoms.get_positions()):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)

    rname = residue_prefix.ljust(3)[:3]
    for i, atom in enumerate(mol.GetAtoms()):
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName(rname)
        info.SetName(f" {atom.GetSymbol():<2}{i:>1}"[:4])
        info.SetResidueNumber(1)
        info.SetChainId("A")
        atom.SetMonomerInfo(info)

    Chem.MolToPDBFile(mol, str(output))
    return output
