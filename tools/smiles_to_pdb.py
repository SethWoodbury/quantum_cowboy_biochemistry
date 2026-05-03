"""SMILES → PDB via RDKit ETKDGv3 + UFF cleanup.

Single-shot utility for the vacuum-TS visualisation gap: when a
pipeline run only emitted XYZ (or only an atom-mapped SMILES JSON),
this script materialises a real 3D structure and writes a PDB so the
user can open it in PyMOL.

Behaviour:
  * Splits '.'-joined multi-component SMILES into separate fragments
    (each fragment gets its own residue chain in the PDB).
  * Adds explicit hydrogens via RDKit, then embeds in 3D with
    ETKDGv3 (current default for biological-relevant geometries).
  * Performs a 200-step UFF cleanup (cheap; just to remove atom
    overlaps from the embed). NOT a real optimisation — autodE / xTB /
    MLFF do that downstream.
  * Writes one PDB per input SMILES. Atoms are labelled C/O/H/etc.
    by element; residue name is "LIG" (or "MOL") so PyMOL colours
    cleanly.

Usage:
    python tools/smiles_to_pdb.py <smiles> --out output.pdb
    python tools/smiles_to_pdb.py "CCOP(=O)(OCC)Oc1ccc([N+](=O)[O-])cc1.O" --out paraoxon_water.pdb

Or for the parse_reaction.json output:
    python tools/smiles_to_pdb.py --reaction-json runs/.../reaction.json --outdir runs/.../pdbs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def smiles_to_pdb(smiles: str, output: Path, *, residue_prefix: str = "LIG") -> None:
    """Embed a SMILES into 3D and write a PDB."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    # Embed — ETKDGv3 is the default for drug-like/biomolecule geometries
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) < 0:
        # Fallback: try the older ETKDG with random coords
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) < 0:
            raise RuntimeError(f"ETKDG embedding failed for {smiles!r}")
    # Quick UFF cleanup — not a real optimisation, just declashing
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass  # UFF can fail on weird ligands; PDB output still useful

    # Set residue name + atom names so PyMOL renders cleanly
    for i, atom in enumerate(mol.GetAtoms()):
        info = atom.GetPDBResidueInfo() or Chem.AtomPDBResidueInfo()
        info.SetResidueName(residue_prefix.ljust(3)[:3])
        # Atom name = element + index (e.g. "C1", "H12")
        info.SetName(f" {atom.GetSymbol():<2}{i:>1}"[:4])
        info.SetResidueNumber(1)
        info.SetChainId("A")
        atom.SetMonomerInfo(info)

    output.parent.mkdir(parents=True, exist_ok=True)
    Chem.MolToPDBFile(mol, str(output))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("smiles", nargs="?", default=None, help="SMILES string")
    g.add_argument("--reaction-json", type=Path,
                   help="Path to a parse_reaction-style JSON; emits R + P PDBs")
    p.add_argument("--out", type=Path, help="Output PDB (single-SMILES mode)")
    p.add_argument("--outdir", type=Path,
                   help="Output dir (multi-PDB / reaction-json mode)")
    p.add_argument("--prefix", default="LIG", help="PDB residue name (3 chars)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.smiles:
        if args.out is None:
            print("--out required when passing a SMILES", file=sys.stderr)
            return 2
        smiles_to_pdb(args.smiles, args.out, residue_prefix=args.prefix)
        print(f"wrote {args.out}")
        return 0

    # reaction-json mode
    if args.outdir is None:
        print("--outdir required with --reaction-json", file=sys.stderr)
        return 2
    payload = json.loads(args.reaction_json.read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)
    for tag in ("reactant_smiles", "product_smiles"):
        smi = payload.get(tag)
        if not smi:
            continue
        out = args.outdir / f"{tag.replace('_smiles', '')}.pdb"
        smiles_to_pdb(smi, out, residue_prefix="LIG")
        print(f"wrote {out}  ← {smi}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main())
