"""
Structure preparation: active site extraction, protonation, charge calculation, capping.
"""

# Canonical protonation is the deterministic, staged engine in
# ``quantum_engine.prep.protonator`` (CLI: ``cowboy-qc protonate``). The older
# multi-method consensus arbiter (protonate_consensus / protonate_chimera)
# was retired in the 2026-06 cleanup; see archive/pre-cleanup to recover it.
# ``protonate`` (PROPKA pKa prediction) is kept — protonator stage5 and
# ``charge.py`` both use ``get_pka_dict``.
from quantum_engine.prep.protonate import get_pka_dict, assign_protonation_states
from quantum_engine.prep.extract import (
    extract_active_site,
    extract_by_zones,
    fill_residue_gaps,
)
from quantum_engine.prep.cap import cap_backbone_h, PROTEIN_RES
from quantum_engine.prep.charge import calculate_net_charge
from quantum_engine.prep.convert import (
    biotite_to_ase,
    ase_to_biotite,
    pdb_to_xyz,
    xyz_to_pdb,
    write_gaussian_input,
)

__all__ = [
    # propka pKa prediction
    "get_pka_dict",
    "assign_protonation_states",
    # extraction / charges / I/O
    "extract_active_site",
    "extract_by_zones",
    "fill_residue_gaps",
    "cap_backbone_h",
    "PROTEIN_RES",
    "calculate_net_charge",
    "biotite_to_ase",
    "ase_to_biotite",
    "pdb_to_xyz",
    "xyz_to_pdb",
    "write_gaussian_input",
]
