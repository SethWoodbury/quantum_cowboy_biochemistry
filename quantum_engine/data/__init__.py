"""Static data tables + external-database loaders shipped with quantum_engine.

This subpackage holds two kinds of thing:

  * **Static reference dicts** — ligand bond-breaking templates, etc.
    (no side effects, importable anywhere).
  * **External-database loaders** — :mod:`.mcsa` (Mechanism + Catalytic
    Site Atlas) and :mod:`.chebi` (compound SMILES). These fetch from
    EBI on demand, cache to ``site.mcsa_cache_dir()`` /
    ``site.chebi_cache_dir()`` (typically
    ``/net/databases/{mcsa,chebi}_cache/`` on DIGS, with local fallback).
"""
from quantum_engine.data.ligand_bonds import BOND_BREAKING_DEFS
from quantum_engine.data.mcsa import (
    MCSAEntry,
    CatalyticResidue,
    Mechanism,
    MechanismStep,
    fetch_entry as fetch_mcsa_entry,
    PTM_CODES,
    METAL_CODES,
)
from quantum_engine.data.chebi import lookup_smiles as lookup_chebi_smiles

__all__ = [
    "BOND_BREAKING_DEFS",
    "MCSAEntry",
    "CatalyticResidue",
    "Mechanism",
    "MechanismStep",
    "fetch_mcsa_entry",
    "PTM_CODES",
    "METAL_CODES",
    "lookup_chebi_smiles",
]
