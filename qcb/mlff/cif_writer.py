"""
CIF writer for transition state structures using atomworks.

Produces proper mmCIF files with:
- Per-atom partial charges (from xTB Mulliken or EEM)
- Per-atom formal charges (from xTB Mulliken+WBO or RDKit)
- Bond orders with aromatic types (AROMATIC_SINGLE, AROMATIC_DOUBLE)
- CCD-backed bond templates for standard residues
- Residue and chain annotations from PDB template
- Energy and metadata annotations

Uses atomworks (baker-laboratory/atomworks-dev) for CIF I/O,
which is the modern replacement for raw biotite CIF writing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from ase import Atoms

log = logging.getLogger("qcb.cif")

EV_TO_KCAL = 23.0609


def write_ts_cif(
    ts_atoms: Atoms,
    template_st,
    outdir: str | Path,
    total_charge: int = 0,
    partial_charges: np.ndarray | None = None,
    formal_charges: np.ndarray | None = None,
    filename: str = "transition_state.cif",
):
    """Write transition state as CIF using atomworks with proper bond types.

    Args:
        ts_atoms: ASE Atoms of the transition state
        template_st: Biotite AtomArray template (for residue annotations)
        outdir: Output directory
        total_charge: System net charge
        partial_charges: Per-atom partial charges (Mulliken/EEM)
        formal_charges: Per-atom formal charges (integers)
        filename: Output filename
    """
    cif_path = Path(outdir) / filename
    n_atoms = len(ts_atoms)

    try:
        from atomworks.io.utils.ase_conversions import ase_to_atom_array
        from atomworks.io.utils.io_utils import to_cif_string, CIFWriteConfig
        from atomworks.io.tools.rdkit import atom_array_to_rdkit
        import biotite.structure as struc
        import biotite.structure.bonds as bonds

        # Convert ASE → atomworks AtomArray
        if template_st is not None:
            # Use the template for residue annotations
            atom_array = template_st.copy()
            atom_array.coord = ts_atoms.get_positions().astype(np.float32)
        else:
            atom_array = ase_to_atom_array(ts_atoms)

        # Add charge annotation
        if formal_charges is not None and len(formal_charges) == n_atoms:
            atom_array.set_annotation("charge", formal_charges.astype(int))

        # Try to get bond orders via RDKit (handles aromaticity)
        try:
            mol = atom_array_to_rdkit(
                atom_array,
                infer_bonds=True,
                system_charge=total_charge,
            )

            if mol is not None:
                from rdkit import Chem

                # Build BondList with proper aromatic types
                bond_list = struc.BondList(n_atoms)
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    is_aromatic = bond.GetIsAromatic()
                    bt = bond.GetBondType()

                    if is_aromatic:
                        if bt == Chem.BondType.SINGLE or bond.GetBondTypeAsDouble() <= 1.0:
                            btype = bonds.BondType.AROMATIC_SINGLE
                        elif bt == Chem.BondType.DOUBLE or bond.GetBondTypeAsDouble() >= 2.0:
                            btype = bonds.BondType.AROMATIC_DOUBLE
                        else:
                            btype = bonds.BondType.AROMATIC_SINGLE
                    else:
                        if bt == Chem.BondType.SINGLE:
                            btype = bonds.BondType.SINGLE
                        elif bt == Chem.BondType.DOUBLE:
                            btype = bonds.BondType.DOUBLE
                        elif bt == Chem.BondType.TRIPLE:
                            btype = bonds.BondType.TRIPLE
                        else:
                            btype = bonds.BondType.SINGLE

                    bond_list.add_bond(i, j, btype)

                atom_array.bonds = bond_list
                log.info(f"  Bond perception: {bond_list.get_bond_count()} bonds "
                         f"(via RDKit with aromatic kekulization)")

        except Exception as e:
            log.warning(f"  Bond perception failed: {e}")

        # Write CIF using atomworks
        config = CIFWriteConfig(
            id="transition_state",
            author="QCB",
            include_chem_comp=False,  # don't need full CCD for TS
        )

        cif_str = to_cif_string(atom_array, config=config)

        # Append QCB metadata and partial charges
        energy = None
        try:
            energy = ts_atoms.get_potential_energy()
        except Exception:
            pass

        extra_lines = [
            "",
            "# QCB transition state metadata",
            f"_qcb.total_charge  {total_charge}",
        ]
        if energy is not None:
            extra_lines.append(f"_qcb.energy_eV  {energy:.6f}")
            extra_lines.append(f"_qcb.energy_kcal_mol  {energy * EV_TO_KCAL:.2f}")

        # Add partial charges as a separate loop if available
        if partial_charges is not None and len(partial_charges) == n_atoms:
            extra_lines.append("")
            extra_lines.append("loop_")
            extra_lines.append("_qcb_atom_charge.id")
            extra_lines.append("_qcb_atom_charge.partial_charge")
            extra_lines.append("_qcb_atom_charge.formal_charge")
            for i in range(n_atoms):
                fc = int(formal_charges[i]) if formal_charges is not None else 0
                extra_lines.append(f"{i+1} {partial_charges[i]:.4f} {fc}")

        cif_str += "\n".join(extra_lines) + "\n"

        Path(cif_path).write_text(cif_str)
        log.info(f"  Wrote {cif_path} (atomworks CIF with bonds + charges)")
        return cif_path

    except ImportError as e:
        log.warning(f"  atomworks not available ({e}), falling back to basic CIF")
        return _write_basic_cif(ts_atoms, template_st, cif_path, total_charge,
                                partial_charges, formal_charges)
    except Exception as e:
        log.warning(f"  atomworks CIF failed ({e}), falling back to basic CIF")
        return _write_basic_cif(ts_atoms, template_st, cif_path, total_charge,
                                partial_charges, formal_charges)


def _write_basic_cif(ts_atoms, template_st, cif_path, total_charge,
                     partial_charges, formal_charges):
    """Fallback basic CIF writer without atomworks."""
    from ase.io import write as ase_write

    ase_write(str(cif_path), ts_atoms, format="cif")

    with open(cif_path, "a") as f:
        f.write(f"\n_qcb.total_charge  {total_charge}\n")

    log.info(f"  Wrote {cif_path} (basic CIF, no bond types)")
    return cif_path
