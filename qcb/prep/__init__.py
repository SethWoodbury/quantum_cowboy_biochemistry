"""
Structure preparation: active site extraction, protonation, charge calculation, capping.
"""

from qcb.prep.protonate import get_pka_dict, assign_protonation_states
from qcb.prep.protonate_chimera import (
    add_hydrogens_chimera,
    add_hydrogens_with_charges,
    parse_pqr_charges,
)
from qcb.prep.protonate_consensus import (
    consensus_protonate,
    ConsensusResult,
    MethodResult,
)
from qcb.prep.extract import (
    extract_active_site,
    extract_by_zones,
    fill_residue_gaps,
)
from qcb.prep.cap import cap_backbone_h, PROTEIN_RES
from qcb.prep.charge import calculate_net_charge
from qcb.prep.convert import (
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
    # ChimeraX
    "add_hydrogens_chimera",
    "add_hydrogens_with_charges",
    "parse_pqr_charges",
    # Consensus arbiter
    "consensus_protonate",
    "ConsensusResult",
    "MethodResult",
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
