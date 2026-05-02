"""Static data tables shipped with quantum_engine.

This subpackage holds reference dicts / lookup tables that are
*data*, not behaviour: ligand bond-breaking templates, atomic
covalent radii, etc. Importable from anywhere; no side effects.
"""
from quantum_engine.data.ligand_bonds import BOND_BREAKING_DEFS

__all__ = ["BOND_BREAKING_DEFS"]
