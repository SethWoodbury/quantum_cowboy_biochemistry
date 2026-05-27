#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_prediction_to_ref_pdb_and_copy_lig.py

Created: 2026-01-15 by Seth Woodbury (woodbuse@uw.edu)
Updated: 2026-01-15 - Fixed HETATM stripping, improved sidechain optimization

Description:
    Advanced PDB alignment script for realigning AlphaFold3 predictions with designed
    enzyme structures. Uses multiple alignment strategies with iterative outlier removal
    and sidechain dihedral optimization to recapitulate catalytic geometries.

Features:
    - Multiple alignment strategies with iterative outlier removal
    - Per-residue sidechain dihedral optimization using reference-guided search
    - NCAA (non-canonical amino acid) handling for AF3 predictions
    - Proper HETATM handling (ALL stripped from prediction, reference added back)
    - HIS tautomer detection (HIS vs HIS_D)
    - Comprehensive metrics tracking and CSV output
    - All REMARK lines copied from reference
    - Configurable thresholds

Usage:
    python align_prediction_to_ref_pdb_and_copy_lig.py \\
        --ref_pdb reference.pdb \\
        --pdb_for_alignment prediction.pdb \\
        --output_dir ./aligned_outputs \\
        --catres_subset 1,2,3,4,5,6,7,8,9,10 \\
        --ptm_from_remark666 "A/LYS/6:KCX" \\
        --verbose
"""

import os
import sys
import argparse
import tempfile
import shutil
import time
import copy
import warnings
import re
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd

# BioPython imports
try:
    from Bio.PDB import PDBParser, PDBIO, Superimposer, Structure, Model, Chain
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Residue import Residue
    from Bio.PDB.vectors import Vector, calc_dihedral, rotaxis
except ImportError:
    print("ERROR: BioPython is not installed. Install with: pip install biopython")
    sys.exit(1)

# PyRosetta imports (for hydrogen addition)
try:
    import pyrosetta
    import pyrosetta.rosetta
    import pyrosetta.distributed.io
    HAS_PYROSETTA = True
except ImportError:
    print("WARNING: PyRosetta not available. Hydrogen addition will be skipped.")
    HAS_PYROSETTA = False

# Biotite imports (for TM-align style superposition)
try:
    import biotite.structure as bs
    import biotite.structure.io.pdb as bpdb
    from biotite.structure import superimpose_structural_homologs
    HAS_BIOTITE = True
except ImportError:
    print("WARNING: Biotite not available. TM-align strategies will be skipped.")
    HAS_BIOTITE = False


# ============================================================================
# NCAA (NON-CANONICAL AMINO ACID) DEFINITIONS
# ============================================================================

# Hardcoded dictionary of common NCAAs and their atoms to cut
NCAA_ATOMS_TO_CUT = {
    'KCX': ['CX', 'OQ1', 'OQ2'],           # Carboxy-lysine
    'MLY': ['CM', 'CE', 'NZ'],              # N-dimethyl-lysine
    'ALY': ['N1'],                          # N-acetyl-lysine
    'MLZ': ['CM'],                          # N-methyl-lysine
    'M3L': ['CM', 'CN'],                    # N-trimethyl-lysine
    'HYP': ['OD1'],                         # Hydroxyproline
    'SEP': ['P', 'O1P', 'O2P', 'O3P'],     # Phosphoserine
    'TPO': ['P', 'O1P', 'O2P', 'O3P'],     # Phosphothreonine
    'PTR': ['P', 'O1P', 'O2P', 'O3P'],     # Phosphotyrosine
    'CSO': ['OD'],                          # S-hydroxycysteine
    'CSD': ['OD1', 'OD2'],                  # Cysteinesulfinic acid
    'OCS': ['O1', 'O2'],                    # Cysteinesulfonic acid
    'MSE': ['SE'],                          # Selenomethionine
}


# ============================================================================
# SYMMETRIC ATOM PAIRS - Chemically equivalent atoms
# ============================================================================

# Symmetric atom pairs for RMSD and lDDT calculations
# These atoms are chemically equivalent due to rotational symmetry:
# - PHE/TYR: 180° ring flip exchanges CD1↔CD2 and CE1↔CE2
# - ASP/GLU: carboxylate oxygen swap
# - ARG: guanidinium terminal nitrogens
# - LEU/VAL: branched aliphatic carbons
SYMMETRIC_ATOM_PAIRS = {
    "PHE": [("CD1", "CD2"), ("CE1", "CE2")],
    "TYR": [("CD1", "CD2"), ("CE1", "CE2")],
    "ASP": [("OD1", "OD2")],
    "GLU": [("OE1", "OE2")],
    "ARG": [("NH1", "NH2")],
    "LEU": [("CD1", "CD2")],
    "VAL": [("CG1", "CG2")],
}


# ============================================================================
# CHI ANGLE DEFINITIONS - Complete atom connectivity for proper rotation
# ============================================================================

# For each residue type, define:
# - chi_atoms: the 4 atoms defining each chi angle dihedral
# - atoms_to_rotate: atoms that move when rotating around that chi angle
CHI_DEFINITIONS = {
    'ARG': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'CD'),
            ('CB', 'CG', 'CD', 'NE'),
            ('CG', 'CD', 'NE', 'CZ')
        ],
        'downstream_atoms': [
            ['CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE', 'HH11', 'HH12', 'HH21', 'HH22'],
            ['CD', 'NE', 'CZ', 'NH1', 'NH2', 'HG2', 'HG3', 'HD2', 'HD3', 'HE', 'HH11', 'HH12', 'HH21', 'HH22'],
            ['NE', 'CZ', 'NH1', 'NH2', 'HD2', 'HD3', 'HE', 'HH11', 'HH12', 'HH21', 'HH22'],
            ['CZ', 'NH1', 'NH2', 'HE', 'HH11', 'HH12', 'HH21', 'HH22']
        ]
    },
    'ASN': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'OD1')
        ],
        'downstream_atoms': [
            ['CG', 'OD1', 'ND2', 'HB2', 'HB3', 'HD21', 'HD22'],
            ['OD1', 'ND2', 'HD21', 'HD22']
        ]
    },
    'ASP': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'OD1')
        ],
        'downstream_atoms': [
            ['CG', 'OD1', 'OD2', 'HB2', 'HB3'],
            ['OD1', 'OD2']
        ]
    },
    'CYS': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'SG')
        ],
        'downstream_atoms': [
            ['SG', 'HB2', 'HB3', 'HG']
        ]
    },
    'GLN': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'CD'),
            ('CB', 'CG', 'CD', 'OE1')
        ],
        'downstream_atoms': [
            ['CG', 'CD', 'OE1', 'NE2', 'HB2', 'HB3', 'HG2', 'HG3', 'HE21', 'HE22'],
            ['CD', 'OE1', 'NE2', 'HG2', 'HG3', 'HE21', 'HE22'],
            ['OE1', 'NE2', 'HE21', 'HE22']
        ]
    },
    'GLU': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'CD'),
            ('CB', 'CG', 'CD', 'OE1')
        ],
        'downstream_atoms': [
            ['CG', 'CD', 'OE1', 'OE2', 'HB2', 'HB3', 'HG2', 'HG3'],
            ['CD', 'OE1', 'OE2', 'HG2', 'HG3'],
            ['OE1', 'OE2']
        ]
    },
    'HIS': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'ND1')
        ],
        'downstream_atoms': [
            ['CG', 'ND1', 'CD2', 'CE1', 'NE2', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2'],
            ['ND1', 'CD2', 'CE1', 'NE2', 'HD1', 'HD2', 'HE1', 'HE2']
        ]
    },
    'ILE': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG1'),
            ('CA', 'CB', 'CG1', 'CD1')
        ],
        'downstream_atoms': [
            ['CG1', 'CG2', 'CD1', 'HB', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23', 'HD11', 'HD12', 'HD13'],
            ['CD1', 'HG12', 'HG13', 'HD11', 'HD12', 'HD13']
        ]
    },
    'LEU': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'CD1')
        ],
        'downstream_atoms': [
            ['CG', 'CD1', 'CD2', 'HB2', 'HB3', 'HG', 'HD11', 'HD12', 'HD13', 'HD21', 'HD22', 'HD23'],
            ['CD1', 'CD2', 'HG', 'HD11', 'HD12', 'HD13', 'HD21', 'HD22', 'HD23']
        ]
    },
    'LYS': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'CD'),
            ('CB', 'CG', 'CD', 'CE'),
            ('CG', 'CD', 'CE', 'NZ')
        ],
        'downstream_atoms': [
            ['CG', 'CD', 'CE', 'NZ', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE2', 'HE3', 'HZ1', 'HZ2', 'HZ3'],
            ['CD', 'CE', 'NZ', 'HG2', 'HG3', 'HD2', 'HD3', 'HE2', 'HE3', 'HZ1', 'HZ2', 'HZ3'],
            ['CE', 'NZ', 'HD2', 'HD3', 'HE2', 'HE3', 'HZ1', 'HZ2', 'HZ3'],
            ['NZ', 'HE2', 'HE3', 'HZ1', 'HZ2', 'HZ3']
        ]
    },
    'MET': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'SD'),
            ('CB', 'CG', 'SD', 'CE')
        ],
        'downstream_atoms': [
            ['CG', 'SD', 'CE', 'HB2', 'HB3', 'HG2', 'HG3', 'HE1', 'HE2', 'HE3'],
            ['SD', 'CE', 'HG2', 'HG3', 'HE1', 'HE2', 'HE3'],
            ['CE', 'HE1', 'HE2', 'HE3']
        ]
    },
    'PHE': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'CD1')
        ],
        'downstream_atoms': [
            ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HZ'],
            ['CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'HD1', 'HD2', 'HE1', 'HE2', 'HZ']
        ]
    },
    'PRO': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'CD')
        ],
        'downstream_atoms': [
            ['CG', 'CD', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3'],
            ['CD', 'HG2', 'HG3', 'HD2', 'HD3']
        ]
    },
    'SER': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'OG')
        ],
        'downstream_atoms': [
            ['OG', 'HB2', 'HB3', 'HG']
        ]
    },
    'THR': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'OG1')
        ],
        'downstream_atoms': [
            ['OG1', 'CG2', 'HB', 'HG1', 'HG21', 'HG22', 'HG23']
        ]
    },
    'TRP': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'CD1')
        ],
        'downstream_atoms': [
            ['CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'HB2', 'HB3', 'HD1', 'HE1', 'HE3', 'HZ2', 'HZ3', 'HH2'],
            ['CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'HD1', 'HE1', 'HE3', 'HZ2', 'HZ3', 'HH2']
        ]
    },
    'TYR': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG'),
            ('CA', 'CB', 'CG', 'CD1')
        ],
        'downstream_atoms': [
            ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HH'],
            ['CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'HD1', 'HD2', 'HE1', 'HE2', 'HH']
        ]
    },
    'VAL': {
        'chi_atoms': [
            ('N', 'CA', 'CB', 'CG1')
        ],
        'downstream_atoms': [
            ['CG1', 'CG2', 'HB', 'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23']
        ]
    }
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AlignmentMetrics:
    """Store metrics for a single alignment strategy."""
    strategy_name: str
    converged_rmsd: float
    n_iterations: int
    n_atoms_final: int
    all_backbone_rmsd: float
    ca_rmsd: float
    catres_backbone_rmsd: float
    catres_ca_rmsd: float
    catres_subset_backbone_rmsd: float
    catres_subset_ca_rmsd: float
    catres_subset_all_atom_rmsd_before_opt: float
    catres_subset_all_atom_rmsd_after_opt: float
    n_sidechain_opt_iterations: int
    sidechain_opt_improvement: float
    # New fields for enhanced metrics
    catres_subset_lddt: float = 0.0  # lDDT between catres_subset atoms only
    tm_score: float = float('nan')  # TM-score (NaN for non-TM-align strategies; computed only for global_TMalign)


@dataclass
class CatalyticResidue:
    """Information about a catalytic residue from REMARK 666."""
    chain: str
    resname: str
    resnum: int
    catres_index: int


@dataclass
class PTMSpec:
    """Specification for a post-translational modification."""
    chain: str
    canonical_resname: str
    catres_index: int
    ncaa_resname: str
    atoms_to_cut: List[str]


# ============================================================================
# NCAA PARSING
# ============================================================================

def parse_ptm_spec(spec_str: str) -> PTMSpec:
    """Parse PTM specification string."""
    if ':' not in spec_str:
        raise ValueError(f"Invalid PTM spec format: {spec_str}")

    left, right = spec_str.split(':', 1)
    parts = left.split('/')
    if len(parts) != 3:
        raise ValueError(f"Invalid PTM spec: {left}")

    chain = parts[0].strip()
    canonical_resname = parts[1].strip()
    catres_index = int(parts[2].strip())

    if '-' in right:
        ncaa_resname, atoms_str = right.split('-', 1)
        ncaa_resname = ncaa_resname.strip()
        atoms_to_cut = [a.strip() for a in atoms_str.split(',')]
    else:
        ncaa_resname = right.strip()
        atoms_to_cut = NCAA_ATOMS_TO_CUT.get(ncaa_resname, []).copy()
        if not atoms_to_cut:
            print(f"WARNING: No default atoms for NCAA {ncaa_resname}")

    return PTMSpec(chain, canonical_resname, catres_index, ncaa_resname, atoms_to_cut)


def parse_ptm_specs(spec_list: List[str]) -> List[PTMSpec]:
    """Parse multiple PTM specifications."""
    return [parse_ptm_spec(s) for s in spec_list]


# ============================================================================
# NCAA AND HETATM PROCESSING
# ============================================================================

def process_prediction_pdb(
    pdb_path: str,
    output_path: str,
    ptm_specs: List[PTMSpec],
    catalytic_residues: List[CatalyticResidue],
    verbose: bool = False
) -> None:
    """
    Process prediction PDB:
    1. Handle NCAA residues (convert to canonical, cut atoms)
    2. STRIP ALL HETATM from prediction (they will be added from reference later)
    """
    if verbose:
        print(f"\nProcessing prediction PDB...")
        print(f"  PTM specs: {len(ptm_specs)}")

    # Build mapping from catres_index to residue info
    catres_map = {cr.catres_index: cr for cr in catalytic_residues}

    # Build PTM mapping: (chain, resnum) -> PTMSpec
    ptm_map = {}
    for ptm_spec in ptm_specs:
        if ptm_spec.catres_index in catres_map:
            catres = catres_map[ptm_spec.catres_index]
            key = (catres.chain, catres.resnum)
            ptm_map[key] = ptm_spec
            if verbose:
                print(f"  PTM: Catres #{ptm_spec.catres_index} ({catres.chain} {catres.resnum}) "
                      f"{ptm_spec.canonical_resname} -> {ptm_spec.ncaa_resname}")
                print(f"    Atoms to cut: {ptm_spec.atoms_to_cut}")

    # Process PDB file
    with open(pdb_path, 'r') as f:
        lines = f.readlines()

    output_lines = []
    hetatm_removed = 0
    atoms_cut = 0
    ncaa_converted = 0

    for line in lines:
        # STRIP ALL HETATM - they will be added from reference later
        if line.startswith("HETATM"):
            # Check if this is an NCAA that should be converted
            try:
                chain = line[21].strip()
                resnum = int(line[22:26].strip())
                atom_name = line[12:16].strip()
                key = (chain, resnum)

                if key in ptm_map:
                    ptm_spec = ptm_map[key]
                    # Cut specified atoms
                    if atom_name in ptm_spec.atoms_to_cut:
                        atoms_cut += 1
                        if verbose:
                            print(f"    Cutting atom {atom_name} from {chain} {resnum}")
                        continue

                    # Convert HETATM to ATOM AND change residue name from NCAA to canonical
                    # PDB format: columns 0-5 = record type, 17-19 = residue name
                    new_line = "ATOM  " + line[6:17] + f"{ptm_spec.canonical_resname:>3}" + line[20:]
                    output_lines.append(new_line)
                    ncaa_converted += 1
                    if verbose and ncaa_converted == 1:
                        print(f"    Converting {ptm_spec.ncaa_resname} -> {ptm_spec.canonical_resname}")
                    continue
            except (ValueError, IndexError):
                pass

            # All other HETATM are removed
            hetatm_removed += 1
            if verbose and hetatm_removed <= 10:
                resname = line[17:20].strip() if len(line) > 20 else "???"
                print(f"    Removing HETATM: {resname}")
            continue

        # Keep ATOM and other lines
        if line.startswith(("ATOM", "TER", "END")):
            output_lines.append(line)
        elif line.startswith("REMARK") or line.startswith("HEADER"):
            # Keep some header lines but we'll replace with ref later
            pass
        else:
            output_lines.append(line)

    # Write output
    with open(output_path, 'w') as f:
        f.writelines(output_lines)

    if verbose:
        print(f"\nPrediction PDB processing complete:")
        print(f"  HETATM removed: {hetatm_removed}")
        print(f"  NCAA atoms cut: {atoms_cut}")
        print(f"  NCAA atoms converted to ATOM: {ncaa_converted}")


# ============================================================================
# SYMMETRIC ATOM HANDLING
# ============================================================================

def _get_symmetric_atom_names(resname: str) -> Set[str]:
    """Get set of all symmetric atom names for a residue type."""
    if resname not in SYMMETRIC_ATOM_PAIRS:
        return set()
    names = set()
    for pair in SYMMETRIC_ATOM_PAIRS[resname]:
        names.add(pair[0])
        names.add(pair[1])
    return names


def _apply_swap_to_name(name: str, resname: str) -> str:
    """Apply symmetric swap to an atom name. Returns swapped name or original if not symmetric."""
    if resname not in SYMMETRIC_ATOM_PAIRS:
        return name
    for a, b in SYMMETRIC_ATOM_PAIRS[resname]:
        if name == a:
            return b
        elif name == b:
            return a
    return name


def _determine_best_swap_for_residue(
    ref_res: Residue,
    mob_res: Residue,
    verbose: bool = False
) -> bool:
    """
    Determine whether to swap symmetric atoms using per-residue Kabsch alignment.

    Aligns on non-symmetric atoms only, then checks which assignment
    (original vs swapped) gives lower RMSD for the symmetric atoms.

    Args:
        ref_res: Reference residue (Biopython Residue object)
        mob_res: Mobile residue (Biopython Residue object)
        verbose: Print debug info

    Returns:
        True if swapping gives better RMSD, False otherwise
    """
    resname = ref_res.resname
    if resname not in SYMMETRIC_ATOM_PAIRS:
        return False

    symmetric_names = _get_symmetric_atom_names(resname)

    # Get non-symmetric heavy atoms present in both
    ref_atoms = {a.name: a for a in ref_res.get_atoms() if a.element != 'H'}
    mob_atoms = {a.name: a for a in mob_res.get_atoms() if a.element != 'H'}

    common_nonsym = [n for n in ref_atoms if n in mob_atoms and n not in symmetric_names]

    if len(common_nonsym) < 3:
        # Not enough atoms to align, default to no swap
        return False

    # Get coords for non-symmetric atoms
    ref_nonsym_coords = np.array([ref_atoms[n].coord for n in common_nonsym])
    mob_nonsym_coords = np.array([mob_atoms[n].coord for n in common_nonsym])

    # Center coords
    ref_centroid = np.mean(ref_nonsym_coords, axis=0)
    mob_centroid = np.mean(mob_nonsym_coords, axis=0)
    ref_centered = ref_nonsym_coords - ref_centroid
    mob_centered = mob_nonsym_coords - mob_centroid

    # Kabsch rotation
    H = mob_centered.T @ ref_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T

    # Apply rotation to all mobile coords
    def transform_coord(coord):
        return R @ (coord - mob_centroid) + ref_centroid

    # Get symmetric atoms present in both
    common_sym = [n for n in ref_atoms if n in mob_atoms and n in symmetric_names]
    if not common_sym:
        return False

    # Calculate RMSD for original assignment
    rmsd_original = 0.0
    for name in common_sym:
        ref_coord = ref_atoms[name].coord
        mob_coord_transformed = transform_coord(mob_atoms[name].coord)
        rmsd_original += np.sum((ref_coord - mob_coord_transformed) ** 2)

    # Calculate RMSD for swapped assignment
    rmsd_swapped = 0.0
    for name in common_sym:
        swapped_name = _apply_swap_to_name(name, resname)
        if swapped_name in mob_atoms:
            ref_coord = ref_atoms[name].coord
            mob_coord_transformed = transform_coord(mob_atoms[swapped_name].coord)
            rmsd_swapped += np.sum((ref_coord - mob_coord_transformed) ** 2)
        else:
            # Swapped atom not present, use original
            ref_coord = ref_atoms[name].coord
            mob_coord_transformed = transform_coord(mob_atoms[name].coord)
            rmsd_swapped += np.sum((ref_coord - mob_coord_transformed) ** 2)

    should_swap = rmsd_swapped < rmsd_original

    if verbose and should_swap:
        print(f"      {resname} {ref_res.id[1]}: swapping symmetric atoms "
              f"(RMSD orig={np.sqrt(rmsd_original/len(common_sym)):.3f} vs swap={np.sqrt(rmsd_swapped/len(common_sym)):.3f})")

    return should_swap


def resolve_symmetric_atoms(
    ref_structure: Structure.Structure,
    mobile_structure: Structure.Structure,
    catalytic_residues: List['CatalyticResidue'],
    verbose: bool = False
) -> Dict[Tuple[str, int], bool]:
    """
    Determine best symmetric atom assignment for each catalytic residue.

    Called ONCE after alignment/optimization, before any metrics are calculated.
    All subsequent RMSD, lDDT, etc. calculations use this resolved mapping.

    Args:
        ref_structure: Reference BioPython structure
        mobile_structure: Mobile (aligned) BioPython structure
        catalytic_residues: List of CatalyticResidue objects
        verbose: Print debug info

    Returns:
        Dict mapping (chain, resnum) -> should_swap (bool)
    """
    swap_map = {}

    # Build residue lookup for both structures
    ref_res_dict = {}
    mob_res_dict = {}

    for model in ref_structure:
        for chain in model:
            for res in chain:
                if res.id[0] == ' ':  # Standard residue
                    ref_res_dict[(chain.id, res.id[1])] = res

    for model in mobile_structure:
        for chain in model:
            for res in chain:
                if res.id[0] == ' ':  # Standard residue
                    mob_res_dict[(chain.id, res.id[1])] = res

    if verbose:
        print("\n  Resolving symmetric atoms for catalytic residues...")

    for catres in catalytic_residues:
        key = (catres.chain, catres.resnum)
        resname = catres.resname

        # Check if this residue has symmetric atoms
        if resname not in SYMMETRIC_ATOM_PAIRS:
            swap_map[key] = False
            continue

        # Get residues from both structures
        ref_res = ref_res_dict.get(key)
        mob_res = mob_res_dict.get(key)

        if ref_res is None or mob_res is None:
            swap_map[key] = False
            continue

        # Determine best swap
        should_swap = _determine_best_swap_for_residue(ref_res, mob_res, verbose)
        swap_map[key] = should_swap

        if verbose and should_swap:
            print(f"    {resname} {catres.chain}/{catres.resnum}: using swapped assignment")

    return swap_map


# ============================================================================
# PDB UTILITIES
# ============================================================================

def separate_protein_and_hetatm(pdb_content: List[str]) -> Tuple[List[str], List[str]]:
    """Separate protein and HETATM lines."""
    protein_lines = []
    hetatm_lines = []
    for line in pdb_content:
        if line.startswith("HETATM"):
            hetatm_lines.append(line)
        elif line.startswith(("ATOM", "TER")):
            protein_lines.append(line)
    return protein_lines, hetatm_lines


def renumber_pdb_atoms(pdb_lines: List[str], start_number: int = 1) -> List[str]:
    """Renumber all ATOM and HETATM lines sequentially."""
    renumbered = []
    atom_num = start_number
    for line in pdb_lines:
        if line.startswith(("ATOM", "HETATM")):
            new_line = line[:6] + f"{atom_num:5d}" + line[11:]
            renumbered.append(new_line)
            atom_num += 1
        else:
            renumbered.append(line)
    return renumbered


def build_his_tautomer_map_from_raw_pdb(pdb_path: str, debug: bool = False) -> Dict[Tuple[str, int], str]:
    """Build HIS tautomer map from raw PDB file."""
    his_map = {}
    his_atoms = {}

    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                atom_name = line[12:16].strip()
                resn = line[17:20].strip()
                chain = line[21].strip()
                resno = int(line[22:26].strip())
            except (ValueError, IndexError):
                continue

            if resn not in ["HIS", "HIE", "HID", "HIP"]:
                continue

            key = (chain, resno)
            if key not in his_atoms:
                his_atoms[key] = set()
            his_atoms[key].add(atom_name)

    for key, atoms in his_atoms.items():
        has_hd1 = "HD1" in atoms
        has_he2 = "HE2" in atoms

        if has_hd1 and not has_he2:
            his_map[key] = "HIS_D"
        else:
            his_map[key] = "HIS"

    return his_map


def extract_all_remark_lines(pdb_path: str) -> List[str]:
    """Extract all REMARK, HEADER, HETNAM, LINK lines from PDB."""
    headers = []
    seen = set()
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(("HEADER", "REMARK", "HETNAM", "LINK", "CONECT")):
                if line not in seen:
                    headers.append(line)
                    seen.add(line)
    return headers


def parse_remark666_lines(pdb_path: str) -> Tuple[List[str], List[CatalyticResidue]]:
    """Parse REMARK 666 lines from a PDB file."""
    remark_lines = []
    catalytic_residues = []
    seen_catres = set()

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("REMARK 666"):
                remark_lines.append(line)

                if "MATCH TEMPLATE" in line and "MATCH MOTIF" in line:
                    parts = line.split()
                    try:
                        motif_idx = parts.index("MOTIF")
                        chain = parts[motif_idx + 1]
                        resname = parts[motif_idx + 2]
                        resnum = int(parts[motif_idx + 3])
                        catres_index = int(parts[motif_idx + 4])

                        key = (chain, resnum, catres_index)
                        if key not in seen_catres:
                            catalytic_residues.append(
                                CatalyticResidue(chain, resname, resnum, catres_index)
                            )
                            seen_catres.add(key)
                    except (ValueError, IndexError):
                        print(f"WARNING: Could not parse REMARK 666 line: {line.strip()}")

    catalytic_residues.sort(key=lambda x: x.catres_index)
    return remark_lines, catalytic_residues


def filter_catres_by_subset(
    catalytic_residues: List[CatalyticResidue],
    subset_indices: Optional[List[int]]
) -> List[CatalyticResidue]:
    """Filter catalytic residues to only include those in the subset."""
    if subset_indices is None:
        return catalytic_residues

    max_index = max(cr.catres_index for cr in catalytic_residues) if catalytic_residues else 0
    for idx in subset_indices:
        if idx > max_index or idx < 1:
            print(f"WARNING: catres_subset index {idx} out of range (max={max_index})")

    return [cr for cr in catalytic_residues if cr.catres_index in subset_indices]


def hetatm_lines_to_heavy_coords(
    hetatm_lines: List[str],
    skip_resnames: Optional[Set[str]] = None,
) -> np.ndarray:
    """
    Parse HETATM PDB lines into an (N, 3) numpy array of HEAVY-ATOM coordinates.

    Used by the ligand-aware sidechain optimiser (Phase 2): the returned coords
    serve as the "ligand" set against which catres sidechain rotamers are
    clash-penalised. Skips waters by default (HOH, WAT, TIP3, TIP4); skips H atoms
    by element column (cols 76-78) with name-fallback. Generalisable: caller can
    pass a custom skip-set (e.g. to exclude crystallographic ions).

    Returns empty (0, 3) array if no qualifying atoms found.
    """
    if skip_resnames is None:
        skip_resnames = {'HOH', 'WAT', 'TIP3', 'TIP4', 'TIP', 'SOL', 'DOD'}

    coords = []
    for line in hetatm_lines:
        if not line.startswith('HETATM'):
            continue
        if len(line) < 54:
            continue
        resname = line[17:20].strip()
        if resname in skip_resnames:
            continue
        # Element column (cols 76-78). If missing or whitespace, fall back to
        # atom name (cols 12-16) — first non-digit, non-space character.
        elem = line[76:78].strip() if len(line) >= 78 else ''
        if not elem:
            atom_name = line[12:16].strip()
            # Strip leading digits (e.g., "1HB" -> "H")
            elem = atom_name.lstrip('0123456789')[:1].upper() if atom_name else ''
        if elem.upper() == 'H':
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        coords.append((x, y, z))
    if not coords:
        return np.empty((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def extract_hetatm_lines(pdb_path: str) -> List[str]:
    """Extract all HETATM lines from a PDB file."""
    hetatm_lines = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("HETATM"):
                hetatm_lines.append(line)
    return hetatm_lines


# ============================================================================
# RMSD CALCULATIONS
# ============================================================================

def get_backbone_atoms(residue: Residue) -> List[Atom]:
    """Get backbone atoms (N, CA, C, O) from a residue."""
    atoms = []
    for atom_name in ['N', 'CA', 'C', 'O']:
        if atom_name in residue:
            atoms.append(residue[atom_name])
    return atoms


def get_ca_atom(residue: Residue) -> Optional[Atom]:
    """Get CA atom from a residue."""
    return residue['CA'] if 'CA' in residue else None


def get_all_heavy_atoms(residue: Residue) -> List[Atom]:
    """Get all non-hydrogen atoms from a residue."""
    return [atom for atom in residue.get_atoms() if atom.element != 'H']


def calculate_rmsd(atoms1: List[Atom], atoms2: List[Atom]) -> float:
    """Calculate RMSD between two lists of atoms."""
    if len(atoms1) != len(atoms2) or len(atoms1) == 0:
        return np.inf

    coords1 = np.array([atom.coord for atom in atoms1])
    coords2 = np.array([atom.coord for atom in atoms2])

    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def calculate_rmsd_with_residues(
    residues1: List[Residue],
    residues2: List[Residue],
    atom_selector: str = 'ca'
) -> Tuple[float, List[Atom], List[Atom]]:
    """Calculate RMSD between residue lists."""
    if len(residues1) != len(residues2):
        return np.inf, [], []

    atoms1 = []
    atoms2 = []

    for res1, res2 in zip(residues1, residues2):
        if atom_selector == 'ca':
            a1 = get_ca_atom(res1)
            a2 = get_ca_atom(res2)
            if a1 and a2:
                atoms1.append(a1)
                atoms2.append(a2)
        elif atom_selector == 'backbone':
            bb1 = get_backbone_atoms(res1)
            bb2 = get_backbone_atoms(res2)
            if len(bb1) == len(bb2) and len(bb1) > 0:
                atoms1.extend(bb1)
                atoms2.extend(bb2)
        elif atom_selector == 'all_heavy':
            heavy1 = get_all_heavy_atoms(res1)
            heavy2 = get_all_heavy_atoms(res2)
            atom_dict1 = {a.name: a for a in heavy1}
            atom_dict2 = {a.name: a for a in heavy2}
            common_atoms = set(atom_dict1.keys()) & set(atom_dict2.keys())
            for atom_name in sorted(common_atoms):
                atoms1.append(atom_dict1[atom_name])
                atoms2.append(atom_dict2[atom_name])

    return calculate_rmsd(atoms1, atoms2), atoms1, atoms2


def calculate_rmsd_with_symmetry(
    residues1: List[Residue],
    residues2: List[Residue],
    catalytic_residues: List['CatalyticResidue'],
    swap_map: Dict[Tuple[str, int], bool],
    atom_selector: str = 'all_heavy'
) -> Tuple[float, List[Atom], List[Atom]]:
    """
    Calculate RMSD between residue lists with symmetric atom handling.

    Uses pre-resolved swap_map to determine atom naming for symmetric residues.

    Args:
        residues1: Reference residues
        residues2: Mobile residues
        catalytic_residues: List of CatalyticResidue objects for chain/resnum lookup
        swap_map: Pre-computed mapping of (chain, resnum) -> should_swap
        atom_selector: 'ca', 'backbone', or 'all_heavy'

    Returns:
        Tuple of (rmsd, atoms_list1, atoms_list2)
    """
    if len(residues1) != len(residues2):
        return np.inf, [], []

    atoms1 = []
    atoms2 = []

    for res1, res2, catres in zip(residues1, residues2, catalytic_residues):
        key = (catres.chain, catres.resnum)
        should_swap = swap_map.get(key, False)
        resname = res1.resname

        if atom_selector == 'ca':
            a1 = get_ca_atom(res1)
            a2 = get_ca_atom(res2)
            if a1 and a2:
                atoms1.append(a1)
                atoms2.append(a2)

        elif atom_selector == 'backbone':
            bb1 = get_backbone_atoms(res1)
            bb2 = get_backbone_atoms(res2)
            if len(bb1) == len(bb2) and len(bb1) > 0:
                atoms1.extend(bb1)
                atoms2.extend(bb2)

        elif atom_selector == 'all_heavy':
            heavy1 = get_all_heavy_atoms(res1)
            heavy2 = get_all_heavy_atoms(res2)
            atom_dict1 = {a.name: a for a in heavy1}
            atom_dict2 = {a.name: a for a in heavy2}

            # If swapping, remap the atom names for structure2
            if should_swap and resname in SYMMETRIC_ATOM_PAIRS:
                atom_dict2_swapped = {}
                for name, atom in atom_dict2.items():
                    swapped_name = _apply_swap_to_name(name, resname)
                    atom_dict2_swapped[swapped_name] = atom
                atom_dict2 = atom_dict2_swapped

            common_atoms = set(atom_dict1.keys()) & set(atom_dict2.keys())
            for atom_name in sorted(common_atoms):
                atoms1.append(atom_dict1[atom_name])
                atoms2.append(atom_dict2[atom_name])

    return calculate_rmsd(atoms1, atoms2), atoms1, atoms2


def calculate_catres_subset_lddt(
    ref_structure: Structure.Structure,
    mobile_structure: Structure.Structure,
    catres_subset: List['CatalyticResidue'],
    swap_map: Dict[Tuple[str, int], bool],
    thresholds: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    inclusion_radius: float = 15.0,
    verbose: bool = False
) -> float:
    """
    Calculate lDDT for catalytic residue subset.

    IMPORTANT: Only considers atom pairs BETWEEN residues in catres_subset.
    This measures how well the inter-residue geometry of the catalytic site
    is preserved, ignoring interactions with the rest of the protein.

    Args:
        ref_structure: Reference BioPython structure
        mobile_structure: Mobile (aligned) BioPython structure
        catres_subset: List of CatalyticResidue objects defining the subset
        swap_map: Pre-computed mapping of (chain, resnum) -> should_swap
        thresholds: Distance thresholds for lDDT calculation
        inclusion_radius: Max distance in reference to consider for lDDT
        verbose: Print debug info

    Returns:
        lDDT score (0.0 to 1.0)
    """
    if not catres_subset:
        return float('nan')

    # Build residue lookup for both structures
    ref_res_dict = {}
    mob_res_dict = {}

    for model in ref_structure:
        for chain in model:
            for res in chain:
                if res.id[0] == ' ':
                    ref_res_dict[(chain.id, res.id[1])] = res

    for model in mobile_structure:
        for chain in model:
            for res in chain:
                if res.id[0] == ' ':
                    mob_res_dict[(chain.id, res.id[1])] = res

    # Collect all atoms from catres_subset with their resolved names
    # Each entry: (residue_key, atom_name, ref_coord, mob_coord)
    atom_data = []

    for catres in catres_subset:
        key = (catres.chain, catres.resnum)
        resname = catres.resname
        should_swap = swap_map.get(key, False)

        ref_res = ref_res_dict.get(key)
        mob_res = mob_res_dict.get(key)

        if ref_res is None or mob_res is None:
            continue

        ref_heavy = {a.name: a.coord for a in ref_res.get_atoms() if a.element != 'H'}
        mob_heavy = {a.name: a.coord for a in mob_res.get_atoms() if a.element != 'H'}

        # Apply swap to mobile atom names if needed
        if should_swap and resname in SYMMETRIC_ATOM_PAIRS:
            mob_heavy_swapped = {}
            for name, coord in mob_heavy.items():
                swapped_name = _apply_swap_to_name(name, resname)
                mob_heavy_swapped[swapped_name] = coord
            mob_heavy = mob_heavy_swapped

        # Get common atoms
        common_names = set(ref_heavy.keys()) & set(mob_heavy.keys())
        for atom_name in common_names:
            atom_data.append({
                'res_key': key,
                'atom_name': atom_name,
                'ref_coord': ref_heavy[atom_name],
                'mob_coord': mob_heavy[atom_name]
            })

    if len(atom_data) < 2:
        return float('nan')

    # Calculate all pairwise distances between atoms from DIFFERENT residues
    preserved_count = 0
    total_comparisons = 0

    for i in range(len(atom_data)):
        for j in range(i + 1, len(atom_data)):
            # Only consider pairs from DIFFERENT residues
            if atom_data[i]['res_key'] == atom_data[j]['res_key']:
                continue

            ref_dist = np.linalg.norm(
                np.array(atom_data[i]['ref_coord']) - np.array(atom_data[j]['ref_coord'])
            )

            # Only consider pairs within inclusion radius in reference
            if ref_dist > inclusion_radius:
                continue

            mob_dist = np.linalg.norm(
                np.array(atom_data[i]['mob_coord']) - np.array(atom_data[j]['mob_coord'])
            )

            # Check each threshold
            for thresh in thresholds:
                total_comparisons += 1
                if abs(ref_dist - mob_dist) < thresh:
                    preserved_count += 1

    if total_comparisons == 0:
        return float('nan')

    lddt = preserved_count / total_comparisons

    if verbose:
        print(f"    lDDT: {lddt:.4f} ({preserved_count}/{total_comparisons} preserved, "
              f"{len(atom_data)} atoms from {len(catres_subset)} residues)")

    return lddt


# ============================================================================
# RESIDUE MATCHING
# ============================================================================

def match_residues_by_chain_and_number(
    structure1: Structure.Structure,
    structure2: Structure.Structure
) -> List[Tuple[Residue, Residue]]:
    """Match residues between two structures."""
    res_dict1 = {}
    res_dict2 = {}

    for model in structure1:
        for chain in model:
            for res in chain:
                if res.id[0] == ' ':
                    res_dict1[(chain.id, res.id[1])] = res

    for model in structure2:
        for chain in model:
            for res in chain:
                if res.id[0] == ' ':
                    res_dict2[(chain.id, res.id[1])] = res

    common_keys = set(res_dict1.keys()) & set(res_dict2.keys())
    return [(res_dict1[k], res_dict2[k]) for k in sorted(common_keys)]


def get_catalytic_residues_from_structure(
    structure: Structure.Structure,
    catalytic_residues: List[CatalyticResidue]
) -> List[Residue]:
    """Extract catalytic residues from structure."""
    res_dict = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':
                    res_dict[(chain.id, residue.id[1])] = residue

    catres_list = []
    for catres in catalytic_residues:
        key = (catres.chain, catres.resnum)
        if key in res_dict:
            catres_list.append(res_dict[key])
        else:
            print(f"WARNING: Catalytic residue {catres.chain} {catres.resnum} not found")

    return catres_list


# ============================================================================
# TM-ALIGN SUPERPOSITION
# ============================================================================

def tmalign_superimpose(
    ref_structure: Structure.Structure,
    mobile_structure: Structure.Structure,
    residue_subset: Optional[List['CatalyticResidue']] = None,
    verbose: bool = False
) -> Tuple[float, int]:
    """
    Perform TM-align style superposition using biotite.

    Args:
        ref_structure: Reference BioPython structure
        mobile_structure: Mobile BioPython structure (will be modified in-place)
        residue_subset: Optional list of CatalyticResidue to restrict alignment
        verbose: Print debug info

    Returns:
        Tuple of (tm_score, n_aligned_residues)
        Returns (0.0, 0) if biotite not available or alignment fails
    """
    if not HAS_BIOTITE:
        if verbose:
            print("  WARNING: Biotite not available, cannot perform TM-align")
        return 0.0, 0

    try:
        # Convert BioPython structures to biotite AtomArrays
        # We need to write to temp files and read back with biotite
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            ref_temp = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            mob_temp = f.name

        # Save structures
        io = PDBIO()
        io.set_structure(ref_structure)
        io.save(ref_temp)
        io.set_structure(mobile_structure)
        io.save(mob_temp)

        # Load with biotite
        ref_file = bpdb.PDBFile.read(ref_temp)
        mob_file = bpdb.PDBFile.read(mob_temp)

        ref_array = ref_file.get_structure(model=1)
        mob_array = mob_file.get_structure(model=1)

        # Filter to CA atoms for superposition
        ref_ca_mask = (ref_array.atom_name == "CA") & (ref_array.element != "")
        mob_ca_mask = (mob_array.atom_name == "CA") & (mob_array.element != "")

        ref_ca = ref_array[ref_ca_mask]
        mob_ca = mob_array[mob_ca_mask]

        if residue_subset is not None:
            # Filter to specified residues
            subset_keys = {(cr.chain, cr.resnum) for cr in residue_subset}

            ref_subset_mask = np.array([
                (ref_ca.chain_id[i], ref_ca.res_id[i]) in subset_keys
                for i in range(len(ref_ca))
            ])
            mob_subset_mask = np.array([
                (mob_ca.chain_id[i], mob_ca.res_id[i]) in subset_keys
                for i in range(len(mob_ca))
            ])

            ref_ca = ref_ca[ref_subset_mask]
            mob_ca = mob_ca[mob_subset_mask]

        if len(ref_ca) == 0 or len(mob_ca) == 0:
            if verbose:
                print("  WARNING: No CA atoms found for TM-align")
            os.unlink(ref_temp)
            os.unlink(mob_temp)
            return 0.0, 0

        # Perform TM-align style superposition
        # biotite's superimpose_structural_homologs uses a TM-align inspired algorithm
        # Returns: (fitted, transform, fixed_indices, mobile_indices)
        mob_superimposed, transformation, fixed_indices, mobile_indices = superimpose_structural_homologs(
            ref_ca, mob_ca
        )

        # Number of aligned residues (those within d0 threshold)
        n_aligned = len(fixed_indices)

        # Calculate TM-score
        # TM-score = (1/L) * sum_i(1/(1 + (d_i/d0)^2))
        # where L = length of target, d0 = 1.24 * cuberoot(L - 15) - 1.8
        L = len(ref_ca)
        if L > 15:
            d0 = 1.24 * np.cbrt(L - 15) - 1.8
        else:
            d0 = 0.5  # Minimum d0

        # Calculate distances after superposition for the aligned residues
        if n_aligned > 0:
            aligned_ref_coords = ref_ca.coord[fixed_indices]
            aligned_mob_coords = mob_superimposed.coord[mobile_indices]
            distances = np.sqrt(np.sum((aligned_ref_coords - aligned_mob_coords) ** 2, axis=1))
            tm_score = np.sum(1 / (1 + (distances / d0) ** 2)) / L
        else:
            tm_score = float('nan')  # FIX (Bug C): NaN signals "alignment failed", not "perfect" or "0"

        if verbose:
            print(f"  TM-align: TM-score={tm_score:.4f}, {n_aligned}/{len(ref_ca)} CA atoms aligned, d0={d0:.2f}")

        # Apply transformation to ALL atoms in mobile structure
        # Biotite's AffineTransformation has: rotation (3D), center_translation, target_translation
        # The full transform is: R @ (coord - center_translation) + target_translation
        # We need to extract these and apply to BioPython structure

        # Get transformation components
        rot_matrix = transformation.rotation[0]  # Extract 2D rotation from 3D array
        center_trans = transformation.center_translation[0] if transformation.center_translation.ndim > 1 else transformation.center_translation
        target_trans = transformation.target_translation[0] if transformation.target_translation.ndim > 1 else transformation.target_translation

        # FIX (Bug B): biotite stores center_translation = -mob_centroid (already
        # negated) and AffineTransformation.apply() does `mobile + center_translation`
        # i.e. it ADDS center_trans (not subtract). Previous code did `-` which moved
        # the structure by 2 * mob_centroid in the wrong direction, producing TM-score
        # 0.99 (computed on biotite's internal fit) but BioPython RMSD ~5.9 Å.
        # The mathematically correct form: R @ (coord + center_trans) + target_trans.
        for atom in mobile_structure.get_atoms():
            old_coord = np.array(atom.coord)
            centered = old_coord + center_trans
            rotated = rot_matrix @ centered
            new_coord = rotated + target_trans
            atom.coord = new_coord

        # Cleanup
        os.unlink(ref_temp)
        os.unlink(mob_temp)

        return tm_score, n_aligned

    except Exception as e:
        if verbose:
            print(f"  WARNING: TM-align failed: {e}")
            import traceback
            traceback.print_exc()
        return 0.0, 0


# ============================================================================
# ITERATIVE ALIGNMENT
# ============================================================================

class IterativeAligner:
    """Performs iterative alignment with outlier removal."""

    def __init__(
        self,
        ref_structure: Structure.Structure,
        mobile_structure: Structure.Structure,
        atom_selector: str = 'ca',
        rmsd_threshold: float = 0.5,
        max_iterations: int = 50,
        convergence_threshold: float = 0.001,
        min_atoms: int = 10,
        verbose: bool = True
    ):
        self.ref_structure = ref_structure
        self.mobile_structure = mobile_structure
        self.atom_selector = atom_selector
        self.rmsd_threshold = rmsd_threshold
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.min_atoms = min_atoms
        self.verbose = verbose
        self.superimposer = Superimposer()

    def align_with_outlier_removal(
        self,
        ref_residues: List[Residue],
        mobile_residues: List[Residue]
    ) -> Tuple[float, int, List[int]]:
        """Perform iterative alignment with outlier removal."""
        if len(ref_residues) != len(mobile_residues):
            print(f"ERROR: Residue count mismatch: ref={len(ref_residues)}, mob={len(mobile_residues)}")
            return np.inf, 0, []

        if len(ref_residues) == 0:
            return np.inf, 0, []

        n_residues = len(ref_residues)
        active_indices = list(range(n_residues))
        prev_rmsd = np.inf
        rmsd = np.inf

        for iteration in range(self.max_iterations):
            ref_res_active = [ref_residues[i] for i in active_indices]
            mob_res_active = [mobile_residues[i] for i in active_indices]

            rmsd, ref_atoms, mob_atoms = calculate_rmsd_with_residues(
                ref_res_active, mob_res_active, self.atom_selector
            )

            if len(ref_atoms) == 0:
                return np.inf, iteration + 1, active_indices

            if self.verbose:
                print(f"  Iter {iteration+1}: RMSD={rmsd:.4f}Å ({len(ref_atoms)} atoms, {len(active_indices)} res)")

            if abs(prev_rmsd - rmsd) < self.convergence_threshold:
                if self.verbose:
                    print(f"  Converged after {iteration+1} iterations")
                break

            prev_rmsd = rmsd

            self.superimposer.set_atoms(ref_atoms, mob_atoms)
            self.superimposer.apply(self.mobile_structure.get_atoms())

            # FIX (Bug A): recompute RMSD AFTER applying the Kabsch transform.
            # Otherwise on early-exit paths (esp. small catres-only sets that hit
            # "no outliers" on iter 0) we return the PRE-fit RMSD, which can be huge
            # while the structure is actually well-fitted. The variable `rmsd` now
            # reflects post-Kabsch state for use in convergence check + return.
            rmsd_post, _, _ = calculate_rmsd_with_residues(
                ref_res_active, mob_res_active, self.atom_selector
            )
            rmsd = rmsd_post

            # Calculate per-residue deviations
            per_res_dev = []
            for i, (ref_res, mob_res) in enumerate(zip(ref_res_active, mob_res_active)):
                if self.atom_selector == 'ca':
                    ref_ca = get_ca_atom(ref_res)
                    mob_ca = get_ca_atom(mob_res)
                    if ref_ca and mob_ca:
                        dev = np.linalg.norm(ref_ca.coord - mob_ca.coord)
                    else:
                        dev = 0.0
                else:
                    ref_bb = get_backbone_atoms(ref_res)
                    mob_bb = get_backbone_atoms(mob_res)
                    if len(ref_bb) > 0 and len(mob_bb) > 0:
                        dev = calculate_rmsd(ref_bb, mob_bb)
                    else:
                        dev = 0.0
                per_res_dev.append((active_indices[i], dev))

            outliers = [idx for idx, dev in per_res_dev if dev > self.rmsd_threshold]

            if not outliers:
                if self.verbose:
                    print(f"  No outliers, done")
                break

            worst = max(outliers, key=lambda idx: next(d for i, d in per_res_dev if i == idx))
            active_indices.remove(worst)

            if len(active_indices) < self.min_atoms:
                if self.verbose:
                    print(f"  Stopped at {len(active_indices)} residues (min={self.min_atoms})")
                break

        return rmsd, iteration + 1, active_indices


# ============================================================================
# SIDECHAIN OPTIMIZATION - THOROUGH CONVERGENCE-BASED VERSION
# ============================================================================

# Common rotamer chi angles for each residue type (from Dunbrack library)
# Values in degrees
ROTAMER_LIBRARY = {
    'ARG': [
        (62, 180, 65, 85), (62, 180, 65, -175), (62, 180, 180, 85), (62, 180, 180, 180),
        (-67, 180, 65, 85), (-67, 180, 65, -175), (-67, 180, 180, 85), (-67, 180, 180, 180),
        (-177, 65, 65, 85), (-177, 65, 65, -175), (-177, 180, 65, 85), (-177, 180, 180, 85),
    ],
    'ASN': [
        (62, -10), (62, 30), (-65, -40), (-65, 120), (-174, -20), (-174, 30),
    ],
    'ASP': [
        (62, 10), (62, -70), (-70, -15), (-70, 65), (-177, 0), (-177, 65),
    ],
    'CYS': [
        (62,), (-65,), (-176,),
    ],
    'GLN': [
        (62, 180, 20), (62, 180, -40), (-67, 180, 0), (-67, 180, -40),
        (-174, 65, 20), (-174, 180, 0), (-65, -65, -40), (-65, -65, 100),
    ],
    'GLU': [
        (62, 180, -20), (62, 180, -60), (-67, 180, -20), (-67, -65, -60),
        (-177, 65, -10), (-177, 180, 0), (-65, -65, -40), (-65, -65, -100),
    ],
    'HIS': [
        (62, -75), (62, 80), (-65, -70), (-65, 165), (-177, -75), (-177, 80),
    ],
    'ILE': [
        (62, 170), (62, -60), (-65, 170), (-65, -60), (-57, -60), (-170, 170),
    ],
    'LEU': [
        (62, 80), (-65, 175), (-177, 65), (-65, -65), (-90, 65),
    ],
    'LYS': [
        (62, 180, 68, 180), (62, 180, 180, 65), (-67, 180, 68, 180), (-67, 180, 180, 65),
        (-177, 68, 68, 180), (-62, -68, 180, 65), (-67, 180, 180, 180), (-177, 180, 68, 180),
    ],
    'MET': [
        (62, 180, 75), (62, 180, -75), (-67, 180, 75), (-67, 180, -75),
        (-177, 65, 75), (-177, 180, 75), (-65, -65, 103), (-65, -65, -75),
    ],
    'PHE': [
        (62, 90), (-65, -85), (-177, 80),
    ],
    'PRO': [
        (30, -40), (-30, 30),
    ],
    'SER': [
        (62,), (-65,), (-176,),
    ],
    'THR': [
        (62,), (-65,), (-175,),
    ],
    'TRP': [
        (62, 90), (62, -90), (-65, -90), (-65, 90), (-177, -90), (-177, 90),
    ],
    'TYR': [
        (62, 90), (-65, -85), (-177, 80), (-65, 90),
    ],
    'VAL': [
        (62,), (-60,), (-175,), (175,),
    ],
}


# SP3 carbon atoms that can have bond angle flexibility
SP3_CARBONS = {
    'ARG': ['CB', 'CG', 'CD'],
    'ASN': ['CB'],
    'ASP': ['CB'],
    'CYS': ['CB'],
    'GLN': ['CB', 'CG'],
    'GLU': ['CB', 'CG'],
    'HIS': ['CB'],
    'ILE': ['CB', 'CG1'],
    'LEU': ['CB', 'CG'],
    'LYS': ['CB', 'CG', 'CD', 'CE'],
    'MET': ['CB', 'CG'],
    'PHE': ['CB'],
    'PRO': ['CB', 'CG'],
    'SER': ['CB'],
    'THR': ['CB'],
    'TRP': ['CB'],
    'TYR': ['CB'],
    'VAL': ['CB'],
}


class SidechainDihedralOptimizer:
    """
    Thorough, convergence-based sidechain optimization.

    Strategy:
    1. Global search phase: Full 360° search or rotamer library sampling
    2. Local refinement: Iteratively refine with decreasing grid sizes
    3. Convergence-based: Continue until no improvement above threshold
    4. Multi-pass: Account for chi angle coupling by multiple passes
    5. Optional SP3 bond angle flexibility (±3° from ideal 109.5°)
    """

    def __init__(
        self,
        ref_structure: Structure.Structure,
        mobile_structure: Structure.Structure,
        catalytic_residues: List[CatalyticResidue],
        verbose: bool = True,
        sp3_angle_flexibility: bool = False,
        sp3_angle_tolerance: float = 3.0,
        # NEW (Phase 2 — ligand-aware torsion search):
        ligand_coords: Optional[np.ndarray] = None,
        # Per user (2026-05-12): clash penalty should be LIGHT — chi-RMSD to ref
        # is the dominant signal we care about; downstream sequence redesign will
        # resolve remaining clashes anyway. Default 1.0 (down from 10.0).
        clash_weight: float = 1.0,
        clash_cutoff: float = 2.0,
        # NEW (Phase 2 — diagnostic + PTM-aware):
        diagnostic_ligand_coords: Optional[np.ndarray] = None,
        ptm_bonded_residues: Optional[Set[Tuple[str, int]]] = None,
        # NEW (Phase 2 — reference-proximity filter):
        # Per user: clashes already present in ref_pdb between catres and ligand
        # are DESIRED contacts (H-bonds, metal coord) — exempt them. Default
        # cutoff bumped to 3.5 Å to catch typical H-bonds (2.6-3.5 Å) and metal
        # coord (2.0-2.3 Å) AS EXPECTED, not penalised.
        ref_proximity_cutoff: float = 3.5,
    ):
        """
        Ligand-awareness:
          ligand_coords: np.ndarray of shape (N, 3) — coordinates of reference HETATM
                        heavy atoms (ligand, cofactors). When provided, the cost
                        function penalises catres sidechain atoms that overlap the
                        ligand. None disables (legacy RMSD-only behaviour).
          diagnostic_ligand_coords: same shape, but used ONLY for per-residue CSV
                        clash counts — never gated. Lets you compare legacy vs.
                        ligand-aware runs side-by-side on the SAME clash diagnostic.
                        Defaults to ligand_coords if not supplied.
          ptm_bonded_residues: set of (chain, resnum) tuples for PTM residues with
                        intrinsic covalent bonds to the ligand (e.g. KCX with its
                        CO2 group in the substrate residue). These residues are
                        EXCLUDED from clash detection — both in the cost function
                        and in the diagnostic CSV — because the NZ-CX bond would
                        otherwise register as a false-positive clash.
          clash_weight: penalty multiplier on Σ max(0, cutoff - d)^2 (eV-style soft).
                        Default 10.0 — heavy enough to dominate paste-in clashes (~few Å^2)
                        but small enough to not swamp the chi-RMSD signal once clashes resolve.
          clash_cutoff: Å. Any sidechain heavy atom within this distance of any
                        ligand/other-catres heavy atom contributes to the penalty.
                        Default 2.0 — short enough that proper H-bond contacts (2.6–3.5 Å)
                        and Zn-coordination (2.0–2.3 Å) are NOT penalised. Tighten for
                        non-metal systems if needed.

        All clash penalty terms collapse to 0.0 when ligand_coords is None AND
        _current_other_catres_coords is None (set per-call in optimize_single_residue),
        making this a strict superset of the legacy RMSD-only optimiser.
        """
        self.ref_structure = ref_structure
        self.mobile_structure = mobile_structure
        self.catalytic_residues = catalytic_residues
        self.verbose = verbose
        self.sp3_angle_flexibility = sp3_angle_flexibility
        self.sp3_angle_tolerance = sp3_angle_tolerance

        # Phase 2 ligand-aware fields
        self.ligand_coords = ligand_coords
        self.clash_weight = clash_weight
        self.clash_cutoff = clash_cutoff
        # Diagnostic-only ligand coords: used in per-residue CSV regardless of whether
        # the cost function uses ligand_coords. Lets you A/B legacy vs ligand-aware
        # and still see clash counts in the legacy run. Falls back to ligand_coords.
        self.diagnostic_ligand_coords = (
            diagnostic_ligand_coords if diagnostic_ligand_coords is not None
            else ligand_coords
        )
        # PTM-bonded residues: (chain, resnum) pairs whose sidechain has a known
        # covalent bond to the ligand (e.g., KCX NZ-CX). Skipped in clash detection.
        self.ptm_bonded_residues = ptm_bonded_residues or set()
        # Reference-proximity filter: pairs (catres sidechain atom, ligand atom)
        # within ref_proximity_cutoff Å in the REFERENCE geometry are considered
        # "expected close contacts" (Zn-N coordination, established H-bonds, etc.)
        # and are EXCLUDED from clash penalty/count calculations. This generalises
        # PTM filtering to all coordination/H-bond contacts present in the design.
        self.ref_proximity_cutoff = ref_proximity_cutoff
        # set per-call in optimize_single_residue; default None means clash-vs-others off
        self._current_other_catres_coords: Optional[np.ndarray] = None
        # Phase 2: per-residue diagnostic collector (appended in optimize_single_residue).
        # Each entry: dict with chain/resnum/resname/chi_rmsd_initial/final/clash counts/etc.
        self.per_residue_diagnostics: List[Dict] = []

        # Pre-compute reference-proximity exemption masks (one per catres) so per-eval
        # cost is just a numpy multiply. Indexed by (chain, resnum) — must match the
        # mob_res.full_id[2], mob_res.id[1] keys used during optimization.

        self.ref_catres = get_catalytic_residues_from_structure(ref_structure, catalytic_residues)
        self.mob_catres = get_catalytic_residues_from_structure(mobile_structure, catalytic_residues)

        # Reference-proximity exemption masks: dict (chain, resnum) -> bool array of
        # shape (n_sidechain_heavy, n_ligand). True = pair was close in reference =
        # expected coordination/H-bond = SKIP in clash detection.
        self._ref_exempt_mask_ligand: Dict[Tuple[str, int], np.ndarray] = {}
        if self.diagnostic_ligand_coords is not None and self.diagnostic_ligand_coords.size > 0:
            for ref_cr in self.ref_catres:
                key = (ref_cr.full_id[2], ref_cr.id[1])
                sc_ref = self._sidechain_heavy_coords(ref_cr)
                if sc_ref.size == 0:
                    continue
                diff = sc_ref[:, None, :] - self.diagnostic_ligand_coords[None, :, :]
                dists = np.sqrt(np.sum(diff ** 2, axis=-1))
                self._ref_exempt_mask_ligand[key] = (dists < self.ref_proximity_cutoff)

    def _exempt_mask_for(self, mob_res: Residue) -> Optional[np.ndarray]:
        """Look up the (n_sidechain, n_ligand) reference-proximity exempt mask for
        this residue. Returns None if no mask available (atom-count mismatch, etc.)
        Atom ordering must be consistent between ref/mobile for the mask to align."""
        key = (mob_res.full_id[2], mob_res.id[1])
        mask = self._ref_exempt_mask_ligand.get(key)
        if mask is None:
            return None
        # Sanity: mob_res sidechain count must match ref's. If a chi rotation has
        # somehow changed the heavy-atom count this would break the mask alignment,
        # so we drop the mask defensively.
        n_sc_mob = int(self._sidechain_heavy_coords(mob_res).shape[0])
        if mask.shape[0] != n_sc_mob:
            return None
        return mask

    def get_chi_angle(self, residue: Residue, chi_idx: int) -> Optional[float]:
        """Calculate current chi angle for a residue."""
        resname = residue.resname
        if resname not in CHI_DEFINITIONS:
            return None

        chi_atoms_list = CHI_DEFINITIONS[resname]['chi_atoms']
        if chi_idx >= len(chi_atoms_list):
            return None

        atom_names = chi_atoms_list[chi_idx]
        try:
            atoms = [residue[name] for name in atom_names]
            vectors = [Vector(atom.coord) for atom in atoms]
            return calc_dihedral(*vectors)
        except KeyError:
            return None

    def set_chi_angle(self, residue: Residue, chi_idx: int, target_angle: float) -> bool:
        """Set chi angle by rotating downstream atoms."""
        resname = residue.resname
        if resname not in CHI_DEFINITIONS:
            return False

        chi_def = CHI_DEFINITIONS[resname]
        if chi_idx >= len(chi_def['chi_atoms']):
            return False

        atom_names = chi_def['chi_atoms'][chi_idx]
        downstream_names = chi_def['downstream_atoms'][chi_idx]

        try:
            dihedral_atoms = [residue[name] for name in atom_names]
            vectors = [Vector(atom.coord) for atom in dihedral_atoms]
            current_angle = calc_dihedral(*vectors)
            rotation_angle = target_angle - current_angle

            axis_start = np.array(dihedral_atoms[1].coord)
            axis_end = np.array(dihedral_atoms[2].coord)
            axis = axis_end - axis_start
            axis = axis / np.linalg.norm(axis)

            from scipy.spatial.transform import Rotation as R
            rot = R.from_rotvec(rotation_angle * axis)
            rotation_matrix = rot.as_matrix()

            for atom in residue.get_atoms():
                if atom.name in downstream_names:
                    coord = np.array(atom.coord) - axis_start
                    new_coord = rotation_matrix @ coord
                    atom.coord = new_coord + axis_start

            return True
        except KeyError:
            return False

    def get_all_chi_angles(self, residue: Residue) -> List[Optional[float]]:
        """Get all chi angles for a residue."""
        resname = residue.resname
        if resname not in CHI_DEFINITIONS:
            return []
        n_chi = len(CHI_DEFINITIONS[resname]['chi_atoms'])
        return [self.get_chi_angle(residue, i) for i in range(n_chi)]

    def set_all_chi_angles(self, residue: Residue, angles: List[float]) -> None:
        """Set all chi angles for a residue."""
        for chi_idx, angle in enumerate(angles):
            if angle is not None:
                self.set_chi_angle(residue, chi_idx, angle)

    def get_bond_angle(self, residue: Residue, atom1_name: str, atom2_name: str, atom3_name: str) -> Optional[float]:
        """Calculate bond angle between three atoms (atom2 is the central atom)."""
        try:
            atom1 = residue[atom1_name]
            atom2 = residue[atom2_name]
            atom3 = residue[atom3_name]

            v1 = np.array(atom1.coord) - np.array(atom2.coord)
            v2 = np.array(atom3.coord) - np.array(atom2.coord)

            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)

            cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            return np.arccos(cos_angle)
        except (KeyError, ZeroDivisionError):
            return None

    def adjust_sp3_bond_angle(
        self,
        residue: Residue,
        central_atom_name: str,
        target_angle_deviation: float  # In radians, deviation from 109.5°
    ) -> bool:
        """
        Adjust the bond angle at an SP3 carbon by moving downstream atoms.

        This applies a small angular displacement to all atoms downstream of the
        central SP3 carbon to simulate bond angle flexibility.
        """
        resname = residue.resname
        if resname not in SP3_CARBONS or central_atom_name not in SP3_CARBONS[resname]:
            return False

        if resname not in CHI_DEFINITIONS:
            return False

        try:
            central_atom = residue[central_atom_name]
            central_coord = np.array(central_atom.coord)

            # Find upstream atom (toward backbone)
            chi_atoms_list = CHI_DEFINITIONS[resname]['chi_atoms']
            upstream_atom_name = None
            downstream_atoms = set()

            for chi_idx, chi_atoms in enumerate(chi_atoms_list):
                if central_atom_name in chi_atoms:
                    atom_pos = chi_atoms.index(central_atom_name)
                    if atom_pos > 0:
                        upstream_atom_name = chi_atoms[atom_pos - 1]
                    # Get downstream atoms from this chi and all subsequent
                    for ds_idx in range(chi_idx, len(CHI_DEFINITIONS[resname]['downstream_atoms'])):
                        downstream_atoms.update(CHI_DEFINITIONS[resname]['downstream_atoms'][ds_idx])
                    break

            if not upstream_atom_name or not downstream_atoms:
                return False

            upstream_atom = residue[upstream_atom_name]
            upstream_coord = np.array(upstream_atom.coord)

            # Define rotation axis perpendicular to the bond
            bond_vector = central_coord - upstream_coord
            bond_vector = bond_vector / np.linalg.norm(bond_vector)

            # Create perpendicular vector for rotation
            perp = np.array([1, 0, 0])
            if abs(np.dot(bond_vector, perp)) > 0.9:
                perp = np.array([0, 1, 0])
            rotation_axis = np.cross(bond_vector, perp)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

            # Apply rotation to downstream atoms
            from scipy.spatial.transform import Rotation as R
            rot = R.from_rotvec(target_angle_deviation * rotation_axis)
            rotation_matrix = rot.as_matrix()

            for atom in residue.get_atoms():
                if atom.name in downstream_atoms and atom.name != central_atom_name:
                    coord = np.array(atom.coord) - central_coord
                    new_coord = rotation_matrix @ coord
                    atom.coord = new_coord + central_coord

            return True
        except (KeyError, ZeroDivisionError):
            return False

    def optimize_sp3_angles(
        self,
        ref_res: Residue,
        mob_res: Residue
    ) -> Tuple[float, int]:
        """
        Optimize SP3 bond angles within tolerance to minimize RMSD.
        Returns (best_rmsd, n_evaluations).
        """
        if not self.sp3_angle_flexibility:
            rmsd, _ = self.calculate_residue_cost(ref_res, mob_res)  # Phase 2: ligand-aware cost (RMSD + clash penalty); falls back to RMSD if no ligand_coords
            return rmsd, 0

        resname = mob_res.resname
        if resname not in SP3_CARBONS:
            rmsd, _ = self.calculate_residue_cost(ref_res, mob_res)  # Phase 2: ligand-aware cost (RMSD + clash penalty); falls back to RMSD if no ligand_coords
            return rmsd, 0

        best_rmsd, _ = self.calculate_residue_cost(ref_res, mob_res)  # SP3 init: use ligand-aware cost
        n_evals = 1

        # Store original coordinates
        original_coords = {atom.name: atom.coord.copy() for atom in mob_res.get_atoms()}

        # Try small adjustments to each SP3 carbon
        tolerance_rad = np.radians(self.sp3_angle_tolerance)
        angle_steps = np.linspace(-tolerance_rad, tolerance_rad, 5)  # 5 steps including 0

        for sp3_atom in SP3_CARBONS[resname]:
            for angle_adj in angle_steps:
                if abs(angle_adj) < 0.001:  # Skip zero adjustment
                    continue

                # Restore original coords before trying new adjustment
                for atom in mob_res.get_atoms():
                    if atom.name in original_coords:
                        atom.coord = original_coords[atom.name].copy()

                if self.adjust_sp3_bond_angle(mob_res, sp3_atom, angle_adj):
                    rmsd, _ = self.calculate_residue_cost(ref_res, mob_res)  # Phase 2: ligand-aware cost (RMSD + clash penalty); falls back to RMSD if no ligand_coords
                    n_evals += 1

                    if rmsd < best_rmsd:
                        best_rmsd = rmsd
                        # Save as new best
                        original_coords = {atom.name: atom.coord.copy() for atom in mob_res.get_atoms()}

        # Restore best configuration
        for atom in mob_res.get_atoms():
            if atom.name in original_coords:
                atom.coord = original_coords[atom.name].copy()

        return best_rmsd, n_evals

    def calculate_residue_rmsd(self, ref_res: Residue, mob_res: Residue) -> Tuple[float, int]:
        """Calculate all-atom RMSD for a single residue pair."""
        ref_heavy = get_all_heavy_atoms(ref_res)
        mob_heavy = get_all_heavy_atoms(mob_res)

        ref_dict = {a.name: a for a in ref_heavy}
        mob_dict = {a.name: a for a in mob_heavy}

        common = set(ref_dict.keys()) & set(mob_dict.keys())
        if not common:
            return np.inf, 0

        ref_coords = np.array([ref_dict[n].coord for n in sorted(common)])
        mob_coords = np.array([mob_dict[n].coord for n in sorted(common)])

        diff = ref_coords - mob_coords
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        return rmsd, len(common)

    # =====================================================================
    # Phase 2 (2026-05-12) — Ligand-aware cost function
    # =====================================================================
    # The legacy `calculate_residue_rmsd` is the chi-target-similarity term.
    # The new `calculate_residue_cost` adds two clash penalty terms:
    #   - against any reference ligand HETATM heavy atoms (constant during opt)
    #   - against the heavy atoms of OTHER catres sidechains (frozen per call)
    # Falls back to RMSD-only when no ligand/other-catres provided.
    # =====================================================================

    def _sidechain_heavy_coords(self, mob_res: Residue) -> np.ndarray:
        """Return Nx3 array of CB-and-beyond heavy atom coords for this residue."""
        coords = []
        for atom in mob_res.get_atoms():
            # Skip backbone N/CA/C/O and any hydrogens
            if atom.name in ('N', 'CA', 'C', 'O'):
                continue
            elem = (atom.element or '').strip()
            if elem == 'H':
                continue
            coords.append(atom.coord)
        if not coords:
            return np.empty((0, 3), dtype=float)
        return np.asarray(coords, dtype=float)

    def _soft_clash_penalty(self, sidechain_coords: np.ndarray,
                            other_coords: np.ndarray,
                            exempt_mask: Optional[np.ndarray] = None) -> float:
        """Sum of (cutoff - d)^2 for any pair (sidechain_atom, other_atom) with d < cutoff,
        EXCEPT pairs marked True in exempt_mask (reference-proximity exemption: pairs
        already close in the reference are expected coord/H-bond, not real clashes).
        Vectorised pairwise. Falls back to 0.0 if either set is empty."""
        if sidechain_coords.size == 0 or other_coords.size == 0:
            return 0.0
        # broadcast: (Nsc, 1, 3) - (1, Noth, 3) -> (Nsc, Noth, 3)
        diff = sidechain_coords[:, None, :] - other_coords[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        overlap = self.clash_cutoff - dists
        np.maximum(overlap, 0.0, out=overlap)
        if exempt_mask is not None and exempt_mask.shape == overlap.shape:
            overlap = overlap * (~exempt_mask)
        return float(np.sum(overlap ** 2))

    def _count_hard_clashes(self, sidechain_coords: np.ndarray,
                            other_coords: np.ndarray,
                            cutoff: Optional[float] = None,
                            exempt_mask: Optional[np.ndarray] = None) -> int:
        """Number of (sidechain_atom, other_atom) pairs with d < cutoff (default
        self.clash_cutoff), EXCEPT pairs marked True in exempt_mask (ref-proximity).
        Used for diagnostic per-residue CSV output — distinct from the soft penalty
        above, which is the optimisation cost contribution."""
        if sidechain_coords.size == 0 or other_coords.size == 0:
            return 0
        cut = self.clash_cutoff if cutoff is None else float(cutoff)
        diff = sidechain_coords[:, None, :] - other_coords[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        hard = (dists < cut)
        if exempt_mask is not None and exempt_mask.shape == hard.shape:
            hard = hard & (~exempt_mask)
        return int(np.sum(hard))

    def _diagnose_residue_state(self, mob_res: Residue) -> Dict:
        """Snapshot of clash-related diagnostics for the current state of mob_res.
        Used twice in optimize_single_residue (before opt + after opt) to build
        the per-residue CSV that shows whether ligand-aware opt actually reduced
        ligand-residue overlap. Cheap (~one numpy broadcast per call).

        Uses self.diagnostic_ligand_coords (always populated, even when the cost
        function ignores ligand info) so legacy vs ligand-aware runs produce
        directly comparable clash counts.

        Reference-proximity exemption (Phase 2): pairs close in the reference
        (Zn-coord, H-bonds) are not counted as clashes. This is consistent with
        the user's stated priority: "look at the ref_pdb and see where the
        catalytic residue clashes are with the ligand and then ignore those
        (cause that means they are desirable contacts)".

        PTM-bonded residues (e.g. KCX) report 0 clashes because their bond into
        the ligand is intrinsic chemistry, not a steric problem.
        """
        # PTM-bonded residue: skip clash detection entirely (would be all false-positive)
        is_ptm = self._is_ptm_bonded(mob_res)

        sc = self._sidechain_heavy_coords(mob_res)
        diag_lig = self.diagnostic_ligand_coords
        diag_other = self._current_other_catres_coords
        # Reference-proximity exemption mask (pairs close in ref are desired contacts)
        exempt_lig = self._exempt_mask_for(mob_res)

        # Also report RAW (unfiltered) counts for transparency — useful to see
        # which contacts are being exempted as "expected" vs "real clash".
        raw_lig = (
            self._count_hard_clashes(sc, diag_lig, exempt_mask=None)
            if diag_lig is not None else 0
        )

        return {
            'n_sidechain_heavy': int(sc.shape[0]),
            'is_ptm_bonded': bool(is_ptm),
            # Filtered counts (PTM excluded, ref-proximity excluded) — the "real" clashes
            'clashes_vs_ligand':
                0 if is_ptm else (
                    self._count_hard_clashes(sc, diag_lig, exempt_mask=exempt_lig)
                    if diag_lig is not None else 0
                ),
            'clashes_vs_other_catres':
                0 if is_ptm else (
                    self._count_hard_clashes(sc, diag_other) if diag_other is not None else 0
                ),
            # Raw (unfiltered) ligand count — pre-exemption. The DIFF (raw - filtered)
            # tells you how many "expected contacts" got filtered = desired contacts preserved.
            'clashes_vs_ligand_raw': raw_lig,
            'soft_penalty':
                0.0 if is_ptm else (
                    (self._soft_clash_penalty(sc, diag_lig, exempt_mask=exempt_lig)
                        if diag_lig is not None else 0.0) +
                    (self._soft_clash_penalty(sc, diag_other) if diag_other is not None else 0.0)
                ),
        }

    def _is_ptm_bonded(self, mob_res: Residue) -> bool:
        """True if this residue has a known covalent bond into the ligand (e.g. KCX
        NZ-CX into YYE). Such residues must be excluded from clash detection because
        the bonded distance (~1.5 Å) would otherwise register as a clash."""
        key = (mob_res.full_id[2], mob_res.id[1])
        return key in self.ptm_bonded_residues

    def calculate_residue_cost(self, ref_res: Residue, mob_res: Residue) -> Tuple[float, int]:
        """
        Cost-to-minimise for a single residue: chi-target RMSD + clash penalties.

        cost = RMSD(chi_target) + clash_weight * (clash_vs_ligand + clash_vs_other_catres)

        Backwards-compatible: when neither ligand_coords nor _current_other_catres_coords
        is set, returns plain RMSD (== calculate_residue_rmsd behaviour). The optimiser
        therefore degrades cleanly to legacy mode when invoked without ligand info.

        PTM-bonded residues (e.g. KCX with NZ-CX bond to YYE substrate) are skipped
        from BOTH clash terms because their covalent bond to the ligand would
        otherwise show up as a perpetual clash that no chi angle can resolve.
        """
        rmsd, n_atoms = self.calculate_residue_rmsd(ref_res, mob_res)
        if self.ligand_coords is None and self._current_other_catres_coords is None:
            return rmsd, n_atoms

        # Skip clash detection for PTM-bonded residues — preserves chi-RMSD search.
        if self._is_ptm_bonded(mob_res):
            return rmsd, n_atoms

        sc_coords = self._sidechain_heavy_coords(mob_res)
        # Reference-proximity exemption mask for the (sidechain, ligand) pair set.
        # Pairs close in reference (Zn-coord, H-bonds) are NOT penalised.
        exempt_lig = self._exempt_mask_for(mob_res)

        clash = 0.0
        if self.ligand_coords is not None:
            clash += self._soft_clash_penalty(sc_coords, self.ligand_coords, exempt_mask=exempt_lig)
        if self._current_other_catres_coords is not None:
            # No exemption mask for inter-catres clashes (the relative geometry of
            # the other residues changes between calls; a static ref-mask doesn't
            # generalise. Future work: build inter-catres ref-mask too.)
            clash += self._soft_clash_penalty(sc_coords, self._current_other_catres_coords)

        return rmsd + self.clash_weight * clash, n_atoms

    def _snapshot_other_catres_coords(self, mob_res: Residue) -> np.ndarray:
        """Build a frozen Nx3 array of CB-and-beyond heavy atoms across all OTHER catres.
        Called once at the start of `optimize_single_residue` so the per-evaluation cost
        is fast (just numpy broadcasting against this snapshot)."""
        coords = []
        target_chain = mob_res.full_id[2]
        target_resnum = mob_res.id[1]
        for cr in self.mob_catres:
            if cr.full_id[2] == target_chain and cr.id[1] == target_resnum:
                continue  # skip the residue we're optimising
            for a in cr.get_atoms():
                if a.name in ('N', 'CA', 'C', 'O'):
                    continue
                elem = (a.element or '').strip()
                if elem == 'H':
                    continue
                coords.append(a.coord)
        if not coords:
            return np.empty((0, 3), dtype=float)
        return np.asarray(coords, dtype=float)

    def global_search_single_chi(
        self,
        ref_res: Residue,
        mob_res: Residue,
        chi_idx: int,
        step_degrees: float = 30.0
    ) -> Tuple[float, float, int]:
        """
        Global search for a single chi angle over full 360°.
        Returns (best_angle, best_rmsd, n_evaluations).
        """
        best_rmsd = np.inf
        best_angle = 0.0
        n_evals = 0

        angles = np.arange(-180, 180, step_degrees)
        for angle_deg in angles:
            angle_rad = np.radians(angle_deg)
            self.set_chi_angle(mob_res, chi_idx, angle_rad)
            rmsd, _ = self.calculate_residue_cost(ref_res, mob_res)  # Phase 2: ligand-aware cost (RMSD + clash penalty); falls back to RMSD if no ligand_coords
            n_evals += 1

            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_angle = angle_rad

        return best_angle, best_rmsd, n_evals

    def local_refine_single_chi(
        self,
        ref_res: Residue,
        mob_res: Residue,
        chi_idx: int,
        center_angle: float,
        half_range: float = 20.0,
        step_degrees: float = 2.0
    ) -> Tuple[float, float, int]:
        """
        Local refinement around a center angle.
        Returns (best_angle, best_rmsd, n_evaluations).
        """
        best_rmsd = np.inf
        best_angle = center_angle
        n_evals = 0

        half_range_rad = np.radians(half_range)
        step_rad = np.radians(step_degrees)
        offsets = np.arange(-half_range_rad, half_range_rad + step_rad, step_rad)

        for offset in offsets:
            test_angle = center_angle + offset
            self.set_chi_angle(mob_res, chi_idx, test_angle)
            rmsd, _ = self.calculate_residue_cost(ref_res, mob_res)  # Phase 2: ligand-aware cost (RMSD + clash penalty); falls back to RMSD if no ligand_coords
            n_evals += 1

            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_angle = test_angle

        return best_angle, best_rmsd, n_evals

    def optimize_single_residue(
        self,
        ref_res: Residue,
        mob_res: Residue,
        convergence_threshold: float = 0.001,
        max_passes: int = 10
    ) -> Tuple[float, float, int, List[float]]:
        """
        Thorough optimization of a single residue using multi-phase approach.

        Phase 1: Try reference chi angles directly
        Phase 2: Try all rotamers from library
        Phase 3: Full 360° global search on each chi
        Phase 4: Iterative local refinement until convergence

        Returns (initial_rmsd, final_rmsd, n_evaluations, final_chi_angles)
        """
        # Phase 2 (2026-05-12): freeze a snapshot of OTHER catres sidechain heavy atoms
        # for the duration of this single-residue optimisation. The cost function picks
        # this up via self._current_other_catres_coords so each chi evaluation costs
        # just one numpy broadcast against the (constant-during-this-call) snapshot.
        self._current_other_catres_coords = self._snapshot_other_catres_coords(mob_res)

        initial_rmsd, n_atoms = self.calculate_residue_rmsd(ref_res, mob_res)
        resname = mob_res.resname

        # Phase 2 diagnostic — snapshot initial clash state (sidechain-vs-ligand,
        # sidechain-vs-other-catres). Written to per-residue CSV at end of strategy.
        diag_initial = self._diagnose_residue_state(mob_res)

        def _emit_diagnostic(final_rmsd, final_chi):
            """Helper to append per-residue diagnostic record before each return."""
            diag_final = self._diagnose_residue_state(mob_res)
            self.per_residue_diagnostics.append({
                'chain': mob_res.full_id[2],
                'resnum': mob_res.id[1],
                'resname': resname,
                'n_atoms': int(n_atoms),
                'n_sidechain_heavy': diag_initial['n_sidechain_heavy'],
                'rmsd_initial': float(initial_rmsd),
                'rmsd_final': float(final_rmsd),
                'rmsd_improvement': float(initial_rmsd - final_rmsd),
                'clashes_vs_ligand_initial': diag_initial['clashes_vs_ligand'],
                'clashes_vs_ligand_final': diag_final['clashes_vs_ligand'],
                'clashes_vs_other_catres_initial': diag_initial['clashes_vs_other_catres'],
                'clashes_vs_other_catres_final': diag_final['clashes_vs_other_catres'],
                'soft_penalty_initial': diag_initial['soft_penalty'],
                'soft_penalty_final': diag_final['soft_penalty'],
                'chi_final_deg': [float(np.degrees(a)) if a is not None else None
                                  for a in (final_chi or [])],
            })

        if resname not in CHI_DEFINITIONS:
            _emit_diagnostic(initial_rmsd, [])
            self._current_other_catres_coords = None  # clear snapshot before return
            return initial_rmsd, initial_rmsd, 0, []

        n_chi = len(CHI_DEFINITIONS[resname]['chi_atoms'])
        if n_chi == 0:
            _emit_diagnostic(initial_rmsd, [])
            self._current_other_catres_coords = None
            return initial_rmsd, initial_rmsd, 0, []

        ref_chi = self.get_all_chi_angles(ref_res)
        mob_chi = self.get_all_chi_angles(mob_res)
        total_evals = 0

        if self.verbose:
            print(f"      {resname} {mob_res.id[1]}: {n_chi} chi, {n_atoms} atoms")
            print(f"        Ref chi:  {[f'{np.degrees(a):.1f}°' if a else 'N/A' for a in ref_chi]}")
            print(f"        Mob chi:  {[f'{np.degrees(a):.1f}°' if a else 'N/A' for a in mob_chi]}")
            print(f"        Initial RMSD: {initial_rmsd:.4f} Å")

        best_rmsd = initial_rmsd
        best_chi = mob_chi.copy()

        # PHASE 1: Try reference chi angles directly
        if all(a is not None for a in ref_chi[:n_chi]):
            self.set_all_chi_angles(mob_res, ref_chi)
            rmsd, _ = self.calculate_residue_cost(ref_res, mob_res)  # Phase 2: ligand-aware cost (RMSD + clash penalty); falls back to RMSD if no ligand_coords
            total_evals += 1
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_chi = ref_chi.copy()
                if self.verbose:
                    print(f"        Phase 1 (ref angles): RMSD {rmsd:.4f} Å")

        # PHASE 2: Try rotamer library
        if resname in ROTAMER_LIBRARY:
            for rotamer in ROTAMER_LIBRARY[resname]:
                if len(rotamer) != n_chi:
                    continue
                test_chi = [np.radians(a) for a in rotamer]
                self.set_all_chi_angles(mob_res, test_chi)
                rmsd, _ = self.calculate_residue_cost(ref_res, mob_res)  # Phase 2: ligand-aware cost (RMSD + clash penalty); falls back to RMSD if no ligand_coords
                total_evals += 1
                if rmsd < best_rmsd:
                    best_rmsd = rmsd
                    best_chi = test_chi.copy()

            if self.verbose:
                print(f"        Phase 2 (rotamers): best RMSD {best_rmsd:.4f} Å")

        # Apply current best before global search
        self.set_all_chi_angles(mob_res, best_chi)

        # PHASE 3: Full 360° global search on each chi (coarse)
        for chi_idx in range(n_chi):
            angle, rmsd, n_evals = self.global_search_single_chi(
                ref_res, mob_res, chi_idx, step_degrees=30.0
            )
            total_evals += n_evals

            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_chi[chi_idx] = angle
                # Apply and continue to next chi
                self.set_chi_angle(mob_res, chi_idx, angle)
            else:
                # Restore best angle
                self.set_chi_angle(mob_res, chi_idx, best_chi[chi_idx])

        if self.verbose:
            print(f"        Phase 3 (global 30°): best RMSD {best_rmsd:.4f} Å")

        # PHASE 4: Iterative local refinement with convergence
        prev_rmsd = best_rmsd
        for pass_num in range(max_passes):
            improved_this_pass = False

            # Refine each chi angle with progressively finer grids
            for grid_size in [10.0, 5.0, 2.0, 1.0]:
                for chi_idx in range(n_chi):
                    center = best_chi[chi_idx]
                    if center is None:
                        continue

                    angle, rmsd, n_evals = self.local_refine_single_chi(
                        ref_res, mob_res, chi_idx,
                        center_angle=center,
                        half_range=grid_size * 2,
                        step_degrees=grid_size
                    )
                    total_evals += n_evals

                    if rmsd < best_rmsd - convergence_threshold:
                        best_rmsd = rmsd
                        best_chi[chi_idx] = angle
                        self.set_chi_angle(mob_res, chi_idx, angle)
                        improved_this_pass = True
                    else:
                        # Restore best angle
                        self.set_chi_angle(mob_res, chi_idx, best_chi[chi_idx])

            # Check convergence
            improvement = prev_rmsd - best_rmsd
            if self.verbose and improved_this_pass:
                print(f"        Pass {pass_num+1}: RMSD {best_rmsd:.4f} Å (Δ={improvement:.4f})")

            if improvement < convergence_threshold:
                if self.verbose:
                    print(f"        Converged after {pass_num+1} passes")
                break

            prev_rmsd = best_rmsd

        # Final application of chi angles
        self.set_all_chi_angles(mob_res, best_chi)

        # PHASE 5: SP3 bond angle optimization (if enabled)
        if self.sp3_angle_flexibility:
            pre_sp3_rmsd = best_rmsd
            best_rmsd, sp3_evals = self.optimize_sp3_angles(ref_res, mob_res)
            total_evals += sp3_evals
            if self.verbose:
                sp3_improvement = pre_sp3_rmsd - best_rmsd
                print(f"        Phase 5 (SP3 angles ±{self.sp3_angle_tolerance}°): "
                      f"RMSD {best_rmsd:.4f} Å (Δ={sp3_improvement:.4f})")

        final_rmsd, _ = self.calculate_residue_rmsd(ref_res, mob_res)

        if self.verbose:
            print(f"        Final RMSD: {final_rmsd:.4f} Å (improvement: {initial_rmsd - final_rmsd:.4f} Å)")
            print(f"        Final chi: {[f'{np.degrees(a):.1f}°' if a else 'N/A' for a in best_chi]}")
            print(f"        Total evaluations: {total_evals}")

        # Phase 2: emit per-residue diagnostic + clear the per-call snapshot
        _emit_diagnostic(final_rmsd, best_chi)
        self._current_other_catres_coords = None

        return initial_rmsd, final_rmsd, total_evals, best_chi

    def optimize_all_residues(
        self,
        n_cycles: int = 3,
        fine_grid_degrees: float = 5.0  # Kept for API compatibility, not used
    ) -> Tuple[float, float, int]:
        """
        Optimize all catalytic residues with convergence-based approach.

        The optimization runs multiple global cycles over all residues,
        as optimizing one residue may affect the optimal configuration of others
        (through steric interactions or cumulative RMSD effects).

        Returns (initial_total_rmsd, final_total_rmsd, total_evaluations)
        """
        if len(self.ref_catres) != len(self.mob_catres):
            print(f"ERROR: Residue count mismatch: ref={len(self.ref_catres)}, mob={len(self.mob_catres)}")
            return np.inf, np.inf, 0

        if len(self.ref_catres) == 0:
            print("ERROR: No catalytic residues found")
            return np.inf, np.inf, 0

        # Calculate initial total RMSD
        initial_rmsd, _, _ = calculate_rmsd_with_residues(
            self.ref_catres, self.mob_catres, 'all_heavy'
        )

        if self.verbose:
            print(f"\n  === SIDECHAIN OPTIMIZATION (Convergence-Based) ===")
            print(f"  Initial catres all-atom RMSD: {initial_rmsd:.4f} Å")
            print(f"  Optimizing {len(self.ref_catres)} residues...")
            print(f"  Using: rotamer library + global search + iterative refinement")

        total_evals = 0
        per_residue_improvements = []
        per_residue_details = []

        # Multiple global cycles for inter-residue effects
        for global_cycle in range(n_cycles):
            if self.verbose:
                print(f"\n  --- Global Cycle {global_cycle + 1}/{n_cycles} ---")

            cycle_start_rmsd, _, _ = calculate_rmsd_with_residues(
                self.ref_catres, self.mob_catres, 'all_heavy'
            )

            for res_idx, (ref_res, mob_res) in enumerate(zip(self.ref_catres, self.mob_catres)):
                if self.verbose:
                    print(f"\n    Residue {res_idx + 1}/{len(self.ref_catres)}:")

                init_rmsd, final_rmsd, n_evals, final_chi = self.optimize_single_residue(
                    ref_res, mob_res,
                    convergence_threshold=0.001,
                    max_passes=10
                )
                total_evals += n_evals

                if global_cycle == 0:
                    per_residue_improvements.append(init_rmsd - final_rmsd)
                    per_residue_details.append({
                        'resname': mob_res.resname,
                        'resnum': mob_res.id[1],
                        'init_rmsd': init_rmsd,
                        'final_rmsd': final_rmsd,
                        'improvement': init_rmsd - final_rmsd,
                        'final_chi': [np.degrees(a) if a else None for a in final_chi]
                    })

            cycle_end_rmsd, _, _ = calculate_rmsd_with_residues(
                self.ref_catres, self.mob_catres, 'all_heavy'
            )

            cycle_improvement = cycle_start_rmsd - cycle_end_rmsd
            if self.verbose:
                print(f"\n  Cycle {global_cycle + 1} improvement: {cycle_improvement:.4f} Å")
                print(f"  Current total RMSD: {cycle_end_rmsd:.4f} Å")

            # Early stopping if no significant improvement
            if global_cycle > 0 and cycle_improvement < 0.01:
                if self.verbose:
                    print(f"  Stopping early - minimal improvement")
                break

        # Calculate final total RMSD
        final_rmsd, _, _ = calculate_rmsd_with_residues(
            self.ref_catres, self.mob_catres, 'all_heavy'
        )

        if self.verbose:
            print(f"\n  === OPTIMIZATION SUMMARY ===")
            print(f"  Initial catres all-atom RMSD: {initial_rmsd:.4f} Å")
            print(f"  Final catres all-atom RMSD:   {final_rmsd:.4f} Å")
            print(f"  Total improvement: {initial_rmsd - final_rmsd:.4f} Å")
            print(f"  Total evaluations: {total_evals}")
            print(f"\n  Per-residue details:")
            for detail in per_residue_details:
                print(f"    {detail['resname']} {detail['resnum']}: "
                      f"{detail['init_rmsd']:.3f} -> {detail['final_rmsd']:.3f} Å "
                      f"(Δ={detail['improvement']:.3f})")
                if detail['final_chi']:
                    chi_str = ', '.join([f"χ{i+1}={a:.1f}°" if a else "N/A"
                                        for i, a in enumerate(detail['final_chi'])])
                    print(f"      Final: {chi_str}")

        return initial_rmsd, final_rmsd, total_evals


# ============================================================================
# WINNER SELECTION LOGIC
# ============================================================================

def select_winner(
    df: pd.DataFrame,
    tiebreaker_threshold: float = 0.1,
    verbose: bool = True
) -> str:
    """
    Select the winning alignment strategy using hierarchical criteria.

    Priority order:
    1. catres_subset_all_atom_rmsd_after_opt (primary)
    2. catres_subset_ca_rmsd (tiebreaker 1)
    3. all_backbone_rmsd (tiebreaker 2)

    Two values are considered "too close to call" if they differ by less than
    tiebreaker_threshold (default 0.1 Å).

    Returns the strategy name of the winner.
    """
    if df.empty:
        return ""

    # Sort by primary metric
    sorted_df = df.sort_values('catres_subset_all_atom_rmsd_after_opt').reset_index(drop=True)

    if verbose:
        print("\n  Winner selection:")
        print(f"    Tiebreaker threshold: {tiebreaker_threshold} Å")

    # Check if there's a clear winner on primary metric
    if len(sorted_df) == 1:
        winner = sorted_df.loc[0, 'strategy']
        if verbose:
            print(f"    Single strategy, winner: {winner}")
        return winner

    best_primary = sorted_df.loc[0, 'catres_subset_all_atom_rmsd_after_opt']

    # Find all strategies within threshold of the best
    candidates_mask = (sorted_df['catres_subset_all_atom_rmsd_after_opt'] - best_primary) <= tiebreaker_threshold
    candidates = sorted_df[candidates_mask]

    if verbose:
        print(f"    Primary metric (catres_all_atom): best={best_primary:.4f} Å")
        print(f"    Candidates within threshold: {list(candidates['strategy'])}")

    if len(candidates) == 1:
        winner = candidates.iloc[0]['strategy']
        if verbose:
            print(f"    Clear winner on primary metric: {winner}")
        return winner

    # Tiebreaker 1: catres_subset_ca_rmsd
    candidates = candidates.sort_values('catres_subset_ca_rmsd').reset_index(drop=True)
    best_secondary = candidates.loc[0, 'catres_subset_ca_rmsd']
    secondary_mask = (candidates['catres_subset_ca_rmsd'] - best_secondary) <= tiebreaker_threshold
    candidates = candidates[secondary_mask]

    if verbose:
        print(f"    Secondary metric (catres_ca): best={best_secondary:.4f} Å")
        print(f"    Candidates within threshold: {list(candidates['strategy'])}")

    if len(candidates) == 1:
        winner = candidates.iloc[0]['strategy']
        if verbose:
            print(f"    Winner after tiebreaker 1: {winner}")
        return winner

    # Tiebreaker 2: all_backbone_rmsd
    candidates = candidates.sort_values('all_backbone_rmsd').reset_index(drop=True)
    winner = candidates.iloc[0]['strategy']

    if verbose:
        best_tertiary = candidates.loc[0, 'all_backbone_rmsd']
        print(f"    Tertiary metric (backbone): best={best_tertiary:.4f} Å")
        print(f"    Final winner: {winner}")

    return winner


# ============================================================================
# MAIN ALIGNMENT WORKFLOW
# ============================================================================

class MultiStrategyAligner:
    """Main class that orchestrates multiple alignment strategies."""

    def __init__(
        self,
        ref_pdb_path: str,
        mobile_pdb_path: str,
        output_dir: str,
        catres_subset: Optional[List[int]] = None,
        ptm_specs: Optional[List[PTMSpec]] = None,
        outlier_rmsd_threshold: float = 0.5,
        convergence_threshold: float = 0.001,
        max_iterations: int = 50,
        min_residues: int = 10,
        sidechain_cycles: int = 3,
        sidechain_fine_grid: float = 5.0,
        enable_sidechain_opt: bool = True,
        sp3_angle_flexibility: bool = False,
        sp3_angle_tolerance: float = 3.0,
        keep_all_outputs: bool = False,
        save_csv: bool = False,
        winner_threshold: float = 0.1,
        verbose: bool = True,
        # Phase 2 (2026-05-12) — ligand-aware torsion search
        ligand_aware_opt: bool = True,
        clash_weight: float = 1.0,
        clash_cutoff: float = 2.0,
        ref_proximity_cutoff: float = 3.5,
    ):
        self.ref_pdb_path = ref_pdb_path
        self.mobile_pdb_path = mobile_pdb_path
        self.output_dir = output_dir
        self.catres_subset_indices = catres_subset
        self.ptm_specs = ptm_specs or []
        self.outlier_rmsd_threshold = outlier_rmsd_threshold
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.min_residues = min_residues
        self.sidechain_cycles = sidechain_cycles
        self.sidechain_fine_grid = sidechain_fine_grid
        self.enable_sidechain_opt = enable_sidechain_opt
        self.sp3_angle_flexibility = sp3_angle_flexibility
        self.sp3_angle_tolerance = sp3_angle_tolerance
        self.keep_all_outputs = keep_all_outputs
        self.save_csv = save_csv
        self.winner_threshold = winner_threshold
        self.verbose = verbose
        # Phase 2 attributes (consumed by run_single_strategy at optimizer construction)
        self.ligand_aware_opt = ligand_aware_opt
        self.clash_weight = clash_weight
        self.clash_cutoff = clash_cutoff
        self.ref_proximity_cutoff = ref_proximity_cutoff

        # Store aligned structures and their paths
        self.aligned_structures: Dict[str, Structure.Structure] = {}
        self.output_paths: Dict[str, str] = {}

        self.mobile_basename = Path(mobile_pdb_path).stem
        os.makedirs(output_dir, exist_ok=True)

        # Parse REMARK 666 from reference
        self.all_remark_lines = extract_all_remark_lines(ref_pdb_path)
        _, self.all_catalytic_residues = parse_remark666_lines(ref_pdb_path)

        if not self.all_catalytic_residues:
            print("WARNING: No catalytic residues found in REMARK 666!")
        else:
            print(f"\nFound {len(self.all_catalytic_residues)} catalytic residues:")
            for cr in self.all_catalytic_residues:
                print(f"  #{cr.catres_index}: {cr.chain} {cr.resname} {cr.resnum}")

        # Get subset
        self.catres_subset = filter_catres_by_subset(self.all_catalytic_residues, catres_subset)

        if catres_subset:
            print(f"\nUsing catres_subset with {len(self.catres_subset)} residues")

        # Process prediction PDB - NCAA handling + strip ALL HETATM
        print(f"\nProcessing prediction PDB (NCAA + HETATM removal)...")
        self.mobile_processed = tempfile.NamedTemporaryFile(
            mode='w', suffix='.pdb', delete=False
        )
        self.mobile_processed.close()

        process_prediction_pdb(
            mobile_pdb_path,
            self.mobile_processed.name,
            self.ptm_specs,
            self.all_catalytic_residues,
            verbose=verbose
        )

        # Parse structures
        parser = PDBParser(QUIET=True)
        self.ref_structure = parser.get_structure('ref', ref_pdb_path)
        self.mobile_structure_original = parser.get_structure('mobile', self.mobile_processed.name)

        # HETATM from reference
        self.ref_hetatm_lines = extract_hetatm_lines(ref_pdb_path)
        print(f"\nFound {len(self.ref_hetatm_lines)} HETATM lines in reference")

        # Phase 2 (2026-05-12): parse HETATM heavy-atom coordinates for the
        # ligand-aware sidechain optimiser. Waters are excluded by default. The
        # resulting array is passed to SidechainDihedralOptimizer so that catres
        # rotamer search penalises sidechain atoms that overlap the ligand.
        self.ref_ligand_coords = hetatm_lines_to_heavy_coords(self.ref_hetatm_lines)
        print(f"  → {len(self.ref_ligand_coords)} heavy ligand atoms parsed for clash-penalty (waters excluded)")

        # HIS tautomer map
        self.his_tautomer_map = build_his_tautomer_map_from_raw_pdb(ref_pdb_path)

        self.metrics_list: List[AlignmentMetrics] = []

    def _ptm_bonded_set(self) -> Set[Tuple[str, int]]:
        """Resolve each PTMSpec to its (chain, resnum) pair so the SidechainDihedralOptimizer
        can skip clash detection for these residues (KCX-NZ ↔ CX-in-YYE is a covalent
        bond, not a steric clash). Looks up the resnum in self.all_catalytic_residues
        by matching catres_index."""
        if not self.ptm_specs:
            return set()
        out: Set[Tuple[str, int]] = set()
        all_catres = getattr(self, 'all_catalytic_residues', None) or []
        idx_map = {cr.catres_index: cr for cr in all_catres}
        for spec in self.ptm_specs:
            cr = idx_map.get(spec.catres_index)
            if cr is None:
                continue
            out.add((cr.chain, cr.resnum))
        return out

    def __del__(self):
        """Cleanup temp files."""
        if hasattr(self, 'mobile_processed') and os.path.exists(self.mobile_processed.name):
            os.unlink(self.mobile_processed.name)

    def run_all_strategies(self) -> pd.DataFrame:
        """Run all alignment strategies."""
        # RMSD-based strategies (participate in winner selection)
        strategies = [
            ('all_backbone_rmsd', 'backbone', 'all'),
            ('ca_rmsd', 'ca', 'all'),
            ('catres_backbone_rmsd', 'backbone', 'catres'),
            ('catres_ca_rmsd', 'ca', 'catres'),
        ]

        # FIX (strategy redundancy): only run catres_subset_* if user supplied
        # an actual subset via --catres_subset. Otherwise self.catres_subset ==
        # self.all_catalytic_residues, making catres_subset_* rows byte-identical
        # to catres_* rows — pure wasted CPU.
        if self.catres_subset_indices is not None:
            strategies.append(('catres_subset_backbone_rmsd', 'backbone', 'catres_subset'))
            strategies.append(('catres_subset_ca_rmsd', 'ca', 'catres_subset'))
        else:
            print("\nINFO: --catres_subset not supplied; skipping catres_subset_* strategies "
                  "(would be duplicates of catres_* strategies)")

        # TM-align strategy (global only - catres-specific TM-align doesn't work well)
        if HAS_BIOTITE:
            strategies.append(('global_TMalign', 'tmalign', 'all'))
        else:
            print("\nWARNING: Biotite not available, skipping TM-align strategies")

        for strategy_name, atom_selector, residue_set in strategies:
            print(f"\n{'='*80}")
            print(f"STRATEGY: {strategy_name}")
            print(f"{'='*80}")

            try:
                metrics = self.run_single_strategy(strategy_name, atom_selector, residue_set)
                if metrics:
                    self.metrics_list.append(metrics)

                    # Store the aligned structure
                    self.aligned_structures[strategy_name] = copy.deepcopy(self.mobile_structure_aligned)

                    # Save to temporary location with strategy-specific name
                    output_pdb = os.path.join(
                        self.output_dir,
                        f"{self.mobile_basename}_aligned_{strategy_name}.pdb"
                    )
                    self.output_paths[strategy_name] = output_pdb
                    self.save_aligned_structure(
                        self.mobile_structure_aligned,
                        output_pdb,
                        strategy_name
                    )
            except Exception as e:
                print(f"ERROR in {strategy_name}: {e}")
                import traceback
                traceback.print_exc()

        if not self.metrics_list:
            # No successful strategies - still clean up any partial outputs
            if not self.keep_all_outputs and self.output_paths:
                print("\nNo successful strategies. Cleaning up partial outputs...")
                self._cleanup_strategy_files(exclude_path=None)
            return pd.DataFrame()

        df = self.create_metrics_dataframe()
        winner_strategy = ""

        try:
            # Select winner
            winner_strategy = select_winner(df, self.winner_threshold, self.verbose)

            if winner_strategy:
                self.finalize_outputs(winner_strategy)
            else:
                # No winner selected - clean up all files unless keep_all
                print("\nWARNING: No winner could be selected")
                if not self.keep_all_outputs:
                    self._cleanup_strategy_files(exclude_path=None)

        except Exception as e:
            print(f"\nERROR during output finalization: {e}")
            import traceback
            traceback.print_exc()
            # Attempt cleanup even on error
            if not self.keep_all_outputs:
                print("Attempting cleanup after error...")
                try:
                    self._cleanup_strategy_files(exclude_path=None)
                except Exception as cleanup_error:
                    print(f"Cleanup also failed: {cleanup_error}")

        # Save CSV only if requested
        if self.save_csv:
            csv_path = os.path.join(self.output_dir, f"{self.mobile_basename}_alignment_metrics.csv")
            df.to_csv(csv_path, index=False)
            print(f"\nMetrics saved to: {csv_path}")

        self.print_summary(df, winner_strategy)
        return df

    def finalize_outputs(self, winner_strategy: str) -> None:
        """
        Finalize outputs: rename winner to _aligned.pdb, delete others unless keep_all.
        """
        winner_path = self.output_paths.get(winner_strategy)
        if not winner_path or not os.path.exists(winner_path):
            print(f"WARNING: Winner path not found: {winner_path}")
            # Still try to clean up other files
            if not self.keep_all_outputs:
                self._cleanup_strategy_files(exclude_path=None)
            return

        # Final path for winner
        final_winner_path = os.path.join(
            self.output_dir,
            f"{self.mobile_basename}_aligned.pdb"
        )

        try:
            # If final path already exists, remove it
            if os.path.exists(final_winner_path) and final_winner_path != winner_path:
                os.remove(final_winner_path)

            # Move winner to final location (more atomic than copy+delete)
            shutil.move(winner_path, final_winner_path)
            print(f"\nWinner ({winner_strategy}) saved as: {final_winner_path}")

            # Update output_paths so cleanup doesn't try to delete the moved file
            self.output_paths[winner_strategy] = final_winner_path

        except Exception as e:
            print(f"ERROR: Failed to move winner file: {e}")
            # Try copy as fallback
            try:
                shutil.copy2(winner_path, final_winner_path)
                print(f"\nWinner ({winner_strategy}) copied to: {final_winner_path}")
            except Exception as e2:
                print(f"ERROR: Failed to copy winner file: {e2}")
                return

        # Clean up strategy-specific files (unless keep_all)
        if not self.keep_all_outputs:
            self._cleanup_strategy_files(exclude_path=final_winner_path)

    def _cleanup_strategy_files(self, exclude_path: Optional[str] = None) -> None:
        """Remove all strategy-specific output files except the excluded path."""
        removed_count = 0
        failed_count = 0

        for strategy_name, output_path in self.output_paths.items():
            # Skip if it's the excluded path (winner)
            if exclude_path and os.path.abspath(output_path) == os.path.abspath(exclude_path):
                continue

            if not os.path.exists(output_path):
                continue

            try:
                os.remove(output_path)
                removed_count += 1
                if self.verbose:
                    print(f"  Removed: {output_path}")
            except Exception as e:
                failed_count += 1
                print(f"  WARNING: Could not remove {output_path}: {e}")

        if failed_count > 0:
            print(f"  WARNING: Failed to remove {failed_count} files")
        elif removed_count > 0 and self.verbose:
            print(f"  Cleaned up {removed_count} strategy files")

    def run_single_strategy(
        self,
        strategy_name: str,
        atom_selector: str,
        residue_set: str
    ) -> Optional[AlignmentMetrics]:
        """Run a single alignment strategy."""
        mobile_structure = copy.deepcopy(self.mobile_structure_original)

        # Determine residues for alignment
        if residue_set == 'catres_subset':
            ref_residues = get_catalytic_residues_from_structure(self.ref_structure, self.catres_subset)
            mobile_residues = get_catalytic_residues_from_structure(mobile_structure, self.catres_subset)
            catres_for_opt = self.catres_subset
            tmalign_residue_subset = self.catres_subset
        elif residue_set == 'catres':
            ref_residues = get_catalytic_residues_from_structure(self.ref_structure, self.all_catalytic_residues)
            mobile_residues = get_catalytic_residues_from_structure(mobile_structure, self.all_catalytic_residues)
            catres_for_opt = self.all_catalytic_residues
            tmalign_residue_subset = self.all_catalytic_residues
        else:
            matched = match_residues_by_chain_and_number(self.ref_structure, mobile_structure)
            ref_residues = [r1 for r1, r2 in matched]
            mobile_residues = [r2 for r1, r2 in matched]
            catres_for_opt = self.catres_subset if self.catres_subset else self.all_catalytic_residues
            tmalign_residue_subset = None  # Use all residues for global TM-align

        if not ref_residues or not mobile_residues or len(ref_residues) != len(mobile_residues):
            print(f"ERROR: Invalid residues for {strategy_name}")
            return None

        print(f"\nAligning {len(ref_residues)} residues with {atom_selector} method")

        # Variables for alignment results
        converged_rmsd = 0.0
        n_iterations = 0
        kept_indices = []
        tm_score = float('nan')  # FIX (Bug C): NaN default for non-TM-align strategies; overwritten by global_TMalign branch below if applicable

        # Alignment - either Kabsch (iterative) or TM-align
        if atom_selector == 'tmalign':
            # TM-align superposition
            tm_score, n_aligned = tmalign_superimpose(
                self.ref_structure,
                mobile_structure,
                residue_subset=tmalign_residue_subset,
                verbose=self.verbose
            )
            if n_aligned == 0:
                print(f"ERROR: TM-align failed")
                return None

            converged_rmsd = 0.0  # TM-align doesn't use RMSD convergence
            n_iterations = 1
            kept_indices = list(range(len(ref_residues)))
            print(f"\nTM-align: TM-score={tm_score:.4f}, aligned {n_aligned} CA atoms")
        else:
            # Standard Kabsch alignment with iterative outlier removal
            aligner = IterativeAligner(
                self.ref_structure,
                mobile_structure,
                atom_selector=atom_selector,
                rmsd_threshold=self.outlier_rmsd_threshold,
                max_iterations=self.max_iterations,
                convergence_threshold=self.convergence_threshold,
                min_atoms=self.min_residues,
                verbose=self.verbose
            )

            converged_rmsd, n_iterations, kept_indices = aligner.align_with_outlier_removal(
                ref_residues, mobile_residues
            )

            if np.isinf(converged_rmsd):
                print(f"ERROR: Alignment failed")
                return None

            print(f"\nConverged: {converged_rmsd:.4f}Å after {n_iterations} iters ({len(kept_indices)}/{len(ref_residues)} res)")

        self.mobile_structure_aligned = mobile_structure

        # Calculate standard metrics (without symmetry handling yet)
        all_matched = match_residues_by_chain_and_number(self.ref_structure, mobile_structure)
        ref_all = [r1 for r1, r2 in all_matched]
        mob_all = [r2 for r1, r2 in all_matched]

        all_backbone_rmsd, _, _ = calculate_rmsd_with_residues(ref_all, mob_all, 'backbone')
        ca_rmsd, _, _ = calculate_rmsd_with_residues(ref_all, mob_all, 'ca')

        ref_catres = get_catalytic_residues_from_structure(self.ref_structure, self.all_catalytic_residues)
        mob_catres = get_catalytic_residues_from_structure(mobile_structure, self.all_catalytic_residues)

        catres_backbone_rmsd, _, _ = calculate_rmsd_with_residues(ref_catres, mob_catres, 'backbone')
        catres_ca_rmsd, _, _ = calculate_rmsd_with_residues(ref_catres, mob_catres, 'ca')

        opt_catres = self.catres_subset if self.catres_subset else self.all_catalytic_residues
        ref_catres_subset = get_catalytic_residues_from_structure(self.ref_structure, opt_catres)
        mob_catres_subset = get_catalytic_residues_from_structure(mobile_structure, opt_catres)

        catres_subset_backbone_rmsd, _, _ = calculate_rmsd_with_residues(ref_catres_subset, mob_catres_subset, 'backbone')
        catres_subset_ca_rmsd, _, _ = calculate_rmsd_with_residues(ref_catres_subset, mob_catres_subset, 'ca')

        print(f"\nMetrics after alignment:")
        print(f"  All backbone RMSD: {all_backbone_rmsd:.4f} Å")
        print(f"  CA RMSD: {ca_rmsd:.4f} Å")
        print(f"  Catres backbone RMSD: {catres_backbone_rmsd:.4f} Å")
        print(f"  Catres CA RMSD: {catres_ca_rmsd:.4f} Å")
        if atom_selector == 'tmalign':
            print(f"  TM-score: {tm_score:.4f}")

        # Sidechain optimization (optional)
        if self.enable_sidechain_opt:
            optimizer = SidechainDihedralOptimizer(
                self.ref_structure,
                mobile_structure,
                catres_for_opt,
                verbose=self.verbose,
                sp3_angle_flexibility=self.sp3_angle_flexibility,
                sp3_angle_tolerance=self.sp3_angle_tolerance,
                # Phase 2 (2026-05-12): ligand-aware torsion search. Pass parsed
                # reference HETATM heavy-atom coords so rotamer evaluation includes
                # a clash penalty against the ligand AND against other catres
                # sidechains. --no_ligand_aware_opt disables this (legacy RMSD-only).
                ligand_coords=(self.ref_ligand_coords if self.ligand_aware_opt else None),
                clash_weight=self.clash_weight,
                clash_cutoff=self.clash_cutoff,
                # diagnostic ligand coords are ALWAYS set so legacy runs still
                # produce comparable clash counts in the per-residue CSV.
                diagnostic_ligand_coords=self.ref_ligand_coords,
                # Map each PTM spec to (chain, resnum) of its catres so clash
                # detection skips known covalent bonds into the ligand (e.g. KCX-NZ-CX).
                ptm_bonded_residues=self._ptm_bonded_set(),
                # Reference-proximity exemption: pairs close in ref are desired contacts
                ref_proximity_cutoff=self.ref_proximity_cutoff,
            )

            rmsd_before_opt, rmsd_after_opt, n_opt_iter = optimizer.optimize_all_residues(
                n_cycles=self.sidechain_cycles,
                fine_grid_degrees=self.sidechain_fine_grid
            )
            improvement = rmsd_before_opt - rmsd_after_opt

            # Phase 2 diagnostic: per-residue CSV with initial/final clash counts.
            # Useful for verifying that ligand-aware optimisation actually reduces
            # ligand-residue overlap (vs. legacy RMSD-only which is ligand-blind).
            if optimizer.per_residue_diagnostics:
                per_res_csv = os.path.join(
                    self.output_dir,
                    f"{self.mobile_basename}_aligned_{strategy_name}_per_residue.csv"
                )
                try:
                    per_res_df = pd.DataFrame(optimizer.per_residue_diagnostics)
                    per_res_df.to_csv(per_res_csv, index=False)
                    # Print summary unconditionally (per user "verbose everything" preference)
                    # — show only the FINAL pass per residue (deduplicate by (chain,resnum))
                    final_per_res = per_res_df.drop_duplicates(
                        subset=['chain', 'resnum'], keep='last'
                    )
                    total_clash_before = int(
                        final_per_res['clashes_vs_ligand_initial'].sum() +
                        final_per_res['clashes_vs_other_catres_initial'].sum()
                    )
                    total_clash_after = int(
                        final_per_res['clashes_vs_ligand_final'].sum() +
                        final_per_res['clashes_vs_other_catres_final'].sum()
                    )
                    soft_before = float(final_per_res['soft_penalty_initial'].sum())
                    soft_after = float(final_per_res['soft_penalty_final'].sum())
                    print(f"\n  Per-residue diagnostic CSV: {per_res_csv}")
                    print(f"  Hard-clash count (sum across catres, cutoff={optimizer.clash_cutoff:.2f} Å):")
                    print(f"    initial: {total_clash_before}  →  final: {total_clash_after}")
                    print(f"  Soft-clash penalty sum: {soft_before:.4f} → {soft_after:.4f}")
                except Exception as e:
                    print(f"  WARNING: per-residue CSV write failed: {e}")
        else:
            # Just calculate RMSD without optimization
            rmsd_before_opt, _, _ = calculate_rmsd_with_residues(
                ref_catres_subset, mob_catres_subset, 'all_heavy'
            )
            rmsd_after_opt = rmsd_before_opt
            n_opt_iter = 0
            improvement = 0.0
            print(f"\n  Sidechain optimization DISABLED")
            print(f"  Catres all-atom RMSD: {rmsd_before_opt:.4f} Å")

        # ============================================================
        # SYMMETRIC ATOM RESOLUTION (happens ONCE, after optimization)
        # ============================================================
        if self.verbose:
            print(f"\n  Resolving symmetric atoms...")

        swap_map = resolve_symmetric_atoms(
            self.ref_structure,
            mobile_structure,
            opt_catres,
            verbose=self.verbose
        )

        # Recalculate catres_subset all-atom RMSD with symmetry handling
        rmsd_after_opt_symmetric, _, _ = calculate_rmsd_with_symmetry(
            ref_catres_subset,
            mob_catres_subset,
            opt_catres,
            swap_map,
            atom_selector='all_heavy'
        )

        if self.verbose:
            print(f"  Catres all-atom RMSD (with symmetry): {rmsd_after_opt_symmetric:.4f} Å")
            if abs(rmsd_after_opt_symmetric - rmsd_after_opt) > 0.001:
                print(f"    (was {rmsd_after_opt:.4f} Å without symmetry handling)")

        # Use symmetric RMSD as the final metric
        rmsd_after_opt = rmsd_after_opt_symmetric

        # ============================================================
        # lDDT CALCULATION (uses resolved symmetric atom mapping)
        # ============================================================
        if self.verbose:
            print(f"\n  Calculating lDDT for catres_subset...")

        catres_subset_lddt = calculate_catres_subset_lddt(
            self.ref_structure,
            mobile_structure,
            opt_catres,
            swap_map,
            verbose=self.verbose
        )

        if self.verbose:
            print(f"  Catres subset lDDT: {catres_subset_lddt:.4f}")

        return AlignmentMetrics(
            strategy_name=strategy_name,
            converged_rmsd=converged_rmsd,
            n_iterations=n_iterations,
            n_atoms_final=len(kept_indices),
            all_backbone_rmsd=all_backbone_rmsd,
            ca_rmsd=ca_rmsd,
            catres_backbone_rmsd=catres_backbone_rmsd,
            catres_ca_rmsd=catres_ca_rmsd,
            catres_subset_backbone_rmsd=catres_subset_backbone_rmsd,
            catres_subset_ca_rmsd=catres_subset_ca_rmsd,
            catres_subset_all_atom_rmsd_before_opt=rmsd_before_opt,
            catres_subset_all_atom_rmsd_after_opt=rmsd_after_opt,
            n_sidechain_opt_iterations=n_opt_iter,
            sidechain_opt_improvement=improvement,
            catres_subset_lddt=catres_subset_lddt,
            tm_score=tm_score
        )

    def save_aligned_structure(
        self,
        structure: Structure.Structure,
        output_path: str,
        strategy_name: str
    ) -> None:
        """Save aligned structure with HETATM from reference."""
        io = PDBIO()
        io.set_structure(structure)

        temp_pdb = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
        io.save(temp_pdb.name)
        temp_pdb.close()

        with open(temp_pdb.name, 'r') as f:
            content = f.readlines()

        protein_lines, _ = separate_protein_and_hetatm(content)

        # Add hydrogens with PyRosetta
        if HAS_PYROSETTA:
            try:
                if self.verbose:
                    print(f"  Adding hydrogens with PyRosetta...")

                pose = pyrosetta.pose_from_file(temp_pdb.name)

                # Fix HIS tautomers
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            if residue.resname == 'HIS':
                                key = (chain.id, residue.id[1])
                                if key in self.his_tautomer_map:
                                    his_type = self.his_tautomer_map[key]
                                    seqpos = pose.pdb_info().pdb2pose(chain.id, residue.id[1])
                                    if seqpos > 0:
                                        mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
                                        mutres.set_res_name(his_type)
                                        mutres.set_target(seqpos)
                                        mutres.set_preserve_atom_coords(True)
                                        mutres.apply(pose)

                pdb_string = pyrosetta.distributed.io.to_pdbstring(pose)
                protein_with_h_lines = [line + '\n' for line in pdb_string.split('\n')
                                       if line.startswith(('ATOM', 'TER'))]
            except Exception as e:
                print(f"  WARNING: Could not add hydrogens: {e}")
                protein_with_h_lines = protein_lines
        else:
            protein_with_h_lines = protein_lines

        # Build final PDB
        final_lines = []
        final_lines.extend(self.all_remark_lines)
        final_lines.append(f"REMARK   Aligned using strategy: {strategy_name}\n")
        final_lines.extend(protein_with_h_lines)
        final_lines.extend(self.ref_hetatm_lines)
        # Strip any CONECT records from the final output PDB
        final_lines = [ln for ln in final_lines if not ln.startswith("CONECT")]
        final_lines = renumber_pdb_atoms(final_lines, start_number=1)
        if not any(line.startswith('END') for line in final_lines):
            final_lines.append('END\n')

        with open(output_path, 'w') as f:
            f.writelines(final_lines)

        os.unlink(temp_pdb.name)
        print(f"\nSaved: {output_path}")

    def create_metrics_dataframe(self) -> pd.DataFrame:
        """Convert metrics to DataFrame."""
        data = []
        for m in self.metrics_list:
            data.append({
                'strategy': m.strategy_name,
                'converged_rmsd': m.converged_rmsd,
                'n_iterations': m.n_iterations,
                'n_atoms_final': m.n_atoms_final,
                'all_backbone_rmsd': m.all_backbone_rmsd,
                'ca_rmsd': m.ca_rmsd,
                'catres_backbone_rmsd': m.catres_backbone_rmsd,
                'catres_ca_rmsd': m.catres_ca_rmsd,
                'catres_subset_backbone_rmsd': m.catres_subset_backbone_rmsd,
                'catres_subset_ca_rmsd': m.catres_subset_ca_rmsd,
                'catres_subset_all_atom_rmsd_before_opt': m.catres_subset_all_atom_rmsd_before_opt,
                'catres_subset_all_atom_rmsd_after_opt': m.catres_subset_all_atom_rmsd_after_opt,
                'n_sidechain_opt_iterations': m.n_sidechain_opt_iterations,
                'sidechain_opt_improvement': m.sidechain_opt_improvement,
                'catres_subset_lddt': m.catres_subset_lddt,
                'tm_score': m.tm_score
            })
        return pd.DataFrame(data)

    def print_summary(self, df: pd.DataFrame, winner_strategy: str = "") -> None:
        """Print summary with winner information."""
        print(f"\n{'='*80}")
        print("SUMMARY OF ALL STRATEGIES")
        print(f"{'='*80}\n")

        best_catres_idx = df['catres_subset_all_atom_rmsd_after_opt'].idxmin()
        best_catres_strategy = df.loc[best_catres_idx, 'strategy']
        best_catres_rmsd = df.loc[best_catres_idx, 'catres_subset_all_atom_rmsd_after_opt']

        best_ca_idx = df['ca_rmsd'].idxmin()
        best_ca_strategy = df.loc[best_ca_idx, 'strategy']
        best_ca_rmsd = df.loc[best_ca_idx, 'ca_rmsd']

        # Find best lDDT
        best_lddt_idx = df['catres_subset_lddt'].idxmax()
        best_lddt_strategy = df.loc[best_lddt_idx, 'strategy']
        best_lddt = df.loc[best_lddt_idx, 'catres_subset_lddt']

        print("BEST STRATEGIES BY METRIC:")
        print(f"  Best catres all-atom RMSD: {best_catres_strategy} ({best_catres_rmsd:.4f} Å)")
        print(f"  Best CA RMSD: {best_ca_strategy} ({best_ca_rmsd:.4f} Å)")
        print(f"  Best catres lDDT: {best_lddt_strategy} ({best_lddt:.4f})")

        # Show TM-align strategy details if present
        tmalign_rows = df[df['strategy'].str.contains('TMalign', case=False)]
        if not tmalign_rows.empty:
            print("\nTM-ALIGN STRATEGY:")
            for _, row in tmalign_rows.iterrows():
                print(f"  {row['strategy']}: TM-score={row['tm_score']:.4f}, "
                      f"catres_RMSD={row['catres_subset_all_atom_rmsd_after_opt']:.4f} Å, "
                      f"lDDT={row['catres_subset_lddt']:.4f}")

        print("\nFULL METRICS TABLE:")
        print(df.to_string(index=False))

        if winner_strategy:
            winner_row = df[df['strategy'] == winner_strategy].iloc[0]
            print(f"\n{'='*80}")
            print(f"SELECTED WINNER: {winner_strategy}")
            print(f"{'='*80}")
            print(f"  Catres all-atom RMSD (optimized): {winner_row['catres_subset_all_atom_rmsd_after_opt']:.4f} Å")
            print(f"  Catres CA RMSD: {winner_row['catres_subset_ca_rmsd']:.4f} Å")
            print(f"  All backbone RMSD: {winner_row['all_backbone_rmsd']:.4f} Å")
            print(f"  CA RMSD: {winner_row['ca_rmsd']:.4f} Å")
            print(f"  Catres subset lDDT: {winner_row['catres_subset_lddt']:.4f}")
            print(f"  Sidechain improvement: {winner_row['sidechain_opt_improvement']:.4f} Å")
            print(f"{'='*80}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Advanced PDB alignment with NCAA handling and sidechain optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage:
    python %(prog)s --ref_pdb ref.pdb --pdb_for_alignment pred.pdb --output_dir ./out

  With NCAA handling:
    python %(prog)s --ref_pdb ref.pdb --pdb_for_alignment pred.pdb --output_dir ./out \\
        --ptm_from_remark666 "A/LYS/6:KCX"

  Full options:
    python %(prog)s --ref_pdb ref.pdb --pdb_for_alignment pred.pdb --output_dir ./out \\
        --catres_subset "1,5,6,11" --ptm_from_remark666 "A/LYS/6:KCX" \\
        --sp3_angle_flexibility --keep_all --save_csv --verbose
"""
    )

    # Required arguments
    parser.add_argument('--ref_pdb', required=True, help='Reference PDB file path')
    parser.add_argument('--pdb_for_alignment', required=True, help='PDB file to align (e.g., AF3 prediction)')
    parser.add_argument('--output_dir', required=True, help='Output directory for aligned structures')

    # Catalytic residue arguments
    parser.add_argument('--catres_subset', type=str, default=None,
                        help='Comma-separated catres indices to use for optimization (e.g., "1,5,11")')
    parser.add_argument('--ptm_from_remark666', nargs='+', default=None,
                        help='PTM specifications: "CHAIN/RESNAME/CATRES_IDX:NCAA[-ATOMS]" (e.g., "A/LYS/6:KCX")')

    # Alignment thresholds
    parser.add_argument('--outlier_threshold', type=float, default=0.5,
                        help='RMSD threshold for outlier removal (default: 0.5 Å)')
    parser.add_argument('--convergence_threshold', type=float, default=0.001,
                        help='Convergence threshold for iterative alignment (default: 0.001 Å)')
    parser.add_argument('--max_iterations', type=int, default=50,
                        help='Maximum alignment iterations (default: 50)')
    parser.add_argument('--min_residues', type=int, default=10,
                        help='Minimum residues to keep during outlier removal (default: 10)')

    # Sidechain optimization
    parser.add_argument('--no_sidechain_opt', action='store_true',
                        help='Disable sidechain optimization entirely')
    parser.add_argument('--sidechain_cycles', type=int, default=3,
                        help='Number of global sidechain optimization cycles (default: 3)')
    parser.add_argument('--sidechain_fine_grid', type=float, default=5.0,
                        help='Fine grid step size in degrees (default: 5.0)')
    parser.add_argument('--sp3_angle_flexibility', action='store_true',
                        help='Enable SP3 C-C bond angle flexibility (±3° from 109.5°)')
    parser.add_argument('--sp3_angle_tolerance', type=float, default=3.0,
                        help='SP3 bond angle tolerance in degrees (default: 3.0)')

    # Phase 2 (2026-05-12) — Ligand-aware sidechain optimisation
    parser.add_argument('--no_ligand_aware_opt', action='store_true',
                        help='Disable ligand-aware torsion search (Phase 2). When set, '
                             'sidechain optimisation uses the legacy RMSD-only cost, '
                             'ignoring ligand-residue and inter-catres clashes. '
                             'Useful for A/B comparison with the legacy behaviour.')
    parser.add_argument('--clash_cutoff', type=float, default=2.0,
                        help='Clash cutoff in Angstrom for the ligand-aware cost function. '
                             'Atom pairs within this distance contribute (cutoff - d)^2 to '
                             'the soft penalty. Default 2.0 preserves typical metal-coord '
                             'distances (Zn-O ~2.0-2.3 A); tighten/loosen for non-metal sites.')
    parser.add_argument('--clash_weight', type=float, default=1.0,
                        help='Soft-clash penalty weight in the cost = RMSD + w * sum_pair penalty. '
                             'Default 1.0 (LIGHT). Per design priority, chi-RMSD to reference '
                             'dominates; clashes will be resolved later by sequence redesign. '
                             'Bump up if you want aggressive clash-avoidance during alignment.')
    parser.add_argument('--ref_proximity_cutoff', type=float, default=3.5,
                        help='Angstrom cutoff for the reference-proximity filter. Any (catres '
                             'sidechain atom, ligand atom) pair within this distance IN THE '
                             'REFERENCE is considered a desired contact (H-bond, metal coord, '
                             'covalent PTM bond) and is EXEMPTED from clash penalty. Default '
                             '3.5 catches typical H-bonds (2.6-3.5 A) and metal coord (2.0-2.3 A).')

    # Winner selection
    parser.add_argument('--winner_threshold', type=float, default=0.1,
                        help='Threshold for tiebreaker in winner selection (default: 0.1 Å)')

    # Output options
    parser.add_argument('--keep_all', action='store_true',
                        help='Keep all strategy output files (default: only keep winner)')
    parser.add_argument('--save_csv', action='store_true',
                        help='Save metrics to CSV file (default: no CSV output)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output with per-residue optimization details')

    args = parser.parse_args()

    # Parse catres_subset
    catres_subset = None
    if args.catres_subset:
        catres_subset = [int(x.strip()) for x in args.catres_subset.split(',')]

    # Parse PTM specs
    ptm_specs = None
    if args.ptm_from_remark666:
        ptm_specs = parse_ptm_specs(args.ptm_from_remark666)
        print(f"\nParsed {len(ptm_specs)} PTM specifications:")
        for spec in ptm_specs:
            print(f"  #{spec.catres_index}: {spec.chain}/{spec.canonical_resname} -> {spec.ncaa_resname}")
            print(f"    Atoms to cut: {spec.atoms_to_cut}")

    # Initialize PyRosetta
    if HAS_PYROSETTA:
        print("Initializing PyRosetta...")
        pyrosetta.init("-mute all -run:preserve_header")

    # Print configuration
    print(f"\n{'='*80}")
    print("MULTI-STRATEGY ALIGNMENT AND OPTIMIZATION")
    print(f"{'='*80}")
    print(f"\nReference: {args.ref_pdb}")
    print(f"Mobile: {args.pdb_for_alignment}")
    print(f"Output: {args.output_dir}")
    print(f"\nAlignment settings:")
    print(f"  Outlier threshold: {args.outlier_threshold} Å")
    print(f"  Convergence threshold: {args.convergence_threshold} Å")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Min residues: {args.min_residues}")
    print(f"\nSidechain optimization:")
    if args.no_sidechain_opt:
        print(f"  DISABLED")
    else:
        print(f"  Global cycles: {args.sidechain_cycles}")
        print(f"  SP3 angle flexibility: {args.sp3_angle_flexibility}")
        if args.sp3_angle_flexibility:
            print(f"  SP3 tolerance: ±{args.sp3_angle_tolerance}°")
    print(f"\nOutput options:")
    print(f"  Keep all outputs: {args.keep_all}")
    print(f"  Save CSV: {args.save_csv}")
    print(f"  Winner threshold: {args.winner_threshold} Å")

    start_time = time.time()

    aligner = MultiStrategyAligner(
        ref_pdb_path=args.ref_pdb,
        mobile_pdb_path=args.pdb_for_alignment,
        output_dir=args.output_dir,
        catres_subset=catres_subset,
        ptm_specs=ptm_specs,
        outlier_rmsd_threshold=args.outlier_threshold,
        convergence_threshold=args.convergence_threshold,
        max_iterations=args.max_iterations,
        min_residues=args.min_residues,
        sidechain_cycles=args.sidechain_cycles,
        sidechain_fine_grid=args.sidechain_fine_grid,
        enable_sidechain_opt=not args.no_sidechain_opt,
        sp3_angle_flexibility=args.sp3_angle_flexibility,
        sp3_angle_tolerance=args.sp3_angle_tolerance,
        keep_all_outputs=args.keep_all,
        save_csv=args.save_csv,
        winner_threshold=args.winner_threshold,
        verbose=args.verbose,
        # Phase 2 ligand-aware sidechain opt
        ligand_aware_opt=not args.no_ligand_aware_opt,
        clash_weight=args.clash_weight,
        clash_cutoff=args.clash_cutoff,
        ref_proximity_cutoff=args.ref_proximity_cutoff,
    )

    df = aligner.run_all_strategies()

    print(f"\n\nTotal time: {time.time() - start_time:.2f}s")
    print(f"Output: {args.output_dir}")


if __name__ == '__main__':
    main()
