#!/usr/bin/env python3
"""
active_site_relax.py — Lightweight MLFF active-site relax after AF3-to-design alignment.

Phase 3 of the AF3-to-design pipeline:
  align_prediction_to_ref_pdb_and_copy_lig.py  →  active_site_relax.py  →  active_site_metrics.py

Goal: take a paste-in PDB (catres sidechains rotamer-optimised, ligand transplanted from
reference), do a fast constrained energy minimisation to resolve any residual clashes
WITHOUT disturbing the catalytic TS-like geometry.

Strategy:
  - Catres SIDECHAINS free (CB→tip + sidechain Hs). Backbone N/CA/C/O/HA fixed.
  - Ligand (HETATM resname X) as a 6-DOF RIGID BODY:
       FixCom on ligand atoms freezes translation,
       Hookean springs on every intra-ligand bond at high k freeze internal geometry.
  - Everything else (non-catres protein) FROZEN via FixAtoms.
  - MACE-MP universal (charge-agnostic, Zn-aware) by default; mace-polar-m, mace-mh-1,
    aimnet2-rxn, xtb-gfn2 selectable. ASE LBFGS for optimisation.

Optional features:
  --crop / --crop-radius     : Crop to a sphere around the ligand for speed
  --ptm-residues             : Post-process Rosetta-protonated PTMs (e.g. LYS→KCX strips
                                 2 of 3 NZ Hs; CO2 atoms stay in ligand residue)
  --charge-ligand            : User-supplied ligand net charge; protein charge auto-computed
                                 from residue protonation states (ASP/GLU/LYS/ARG/HIS/KCX)

Output (in --out OUTDIR):
  <basename>_relaxed.pdb          : final geometry (full system)
  <basename>_opt.log              : LBFGS step log
  <basename>_opt.traj             : ASE trajectory
  <basename>_summary.json         : config + energies + clash counts before/after
  <basename>_charge_breakdown.json: per-residue charge contributions (if charge-aware MLFF)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict

import numpy as np

# Make qcb importable
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ase import Atoms
from ase.io import read as ase_read, write as ase_write
from ase.constraints import FixAtoms, Hookean
from ase.optimize import LBFGS, FIRE

# qcb factories
from quantum_engine.calc.factory import make_calc

# --------------------------------------------------------------------------
# Residue charge table (default protonation states; PTMs handled separately)
# --------------------------------------------------------------------------
DEFAULT_RESIDUE_CHARGE = {
    'ASP': -1, 'GLU': -1,                 # ionic, deprotonated
    'LYS': +1, 'ARG': +1,                 # ionic, protonated
    # HIS: 0 by default (HID/HIE) — auto-detected via atom H presence
    # KCX: -1 (carboxylated Lys; CO2 group lives in ligand, NZ stripped to 1H here)
    'KCX': -1,
    # Phosphorylated residues (default deprotonated, configurable)
    'SEP': -2, 'TPO': -2, 'PTR': -2,
}

# Atoms always considered backbone (never relaxed for catres)
BACKBONE_ATOMS = {'N', 'CA', 'C', 'O', 'HA', 'H', '1H', '2H', '3H', 'H1', 'H2', 'H3', 'OXT'}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # I/O
    p.add_argument('--input', type=Path, required=True,
                   help='Aligned PDB (output of align_prediction_to_ref_pdb_and_copy_lig.py)')
    p.add_argument('--out', type=Path, required=True,
                   help='Output directory')

    # Active-site selection
    p.add_argument('--cat-residues', type=str, default=None,
                   help='Catres spec "A:HIS:55,A:HIS:184,A:LYS:64,...". '
                        'If omitted, auto-parse from REMARK 666 lines in the input PDB.')
    p.add_argument('--ligand-resname', type=str, default=None,
                   help='Ligand HETATM resname (e.g., YYE, YYL, SUB). Auto-detected '
                        'if exactly one non-water HETATM resname is present.')
    p.add_argument('--ligand-chain', type=str, default=None,
                   help='Ligand chain ID. Auto-detected if --ligand-resname yields '
                        'atoms in only one chain.')

    # Cropping
    p.add_argument('--crop', action='store_true',
                   help='Crop to a sphere around the ligand (default OFF).')
    p.add_argument('--crop-radius', type=float, default=10.0,
                   help='Å around ANY ligand atom (default 10).')
    p.add_argument('--no-crop-h-cap', action='store_true',
                   help='Skip H-cap on peptide-bond cuts during crop (default: cap with H).')
    p.add_argument('--clash-cutoff', type=float, default=1.8,
                   help='Heavy-atom clash distance cutoff (Å) for pre/post-relax diagnostic. '
                        'Default 1.8.')
    p.add_argument('--nz-h-bond-cutoff', type=float, default=1.3,
                   help='Distance (Å) within which an H is considered NZ-bonded for KCX strip. '
                        'Default 1.3.')

    # PTM handling
    p.add_argument('--ptm-residues', type=str, default=None,
                   help='Post-Rosetta PTM adjustments. Format: '
                        '"A/LYS/3:KCX,B/SER/7:SEP". For each entry, strips/adds '
                        'H atoms to match the PTM target while keeping the resname.')

    # Charge
    p.add_argument('--charge-ligand', type=int, default=None,
                   help='Net charge of the ligand complex (HETATM atoms). Required if '
                        '--model uses charge (mace-mh-1 head=omol, mace-polar-*, xtb-gfn2). '
                        'Protein contribution is auto-computed from residue Hs.')

    # MLFF
    p.add_argument('--model', type=str, default='mace-mp',
                   help='MACE/UMA/AIMNET2 model alias. Default mace-mp (universal, '
                        'charge-agnostic, Zn-aware). '
                        'Charge-aware options: mace-mh-1 (with --head omol), mace-polar-{s,m,l}.')
    p.add_argument('--head', type=str, default=None,
                   help='Head for multi-head models (e.g., "omol" for mace-mh-1).')
    p.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                   help='Compute device (default cuda).')

    # Optimizer
    p.add_argument('--optimizer', type=str, default='lbfgs', choices=['lbfgs', 'fire'],
                   help='ASE optimizer (default lbfgs).')
    p.add_argument('--fmax', type=float, default=0.10,
                   help='Convergence threshold in eV/Å (default 0.10 — loose for fast '
                        'clash resolution while preserving TS-like geometry).')
    p.add_argument('--max-steps', type=int, default=200,
                   help='Optimizer max steps (default 200).')

    # Ligand freedom — defaults: ALL ligand atoms FIXED.
    p.add_argument('--ligand-free-atoms', type=str, default=None,
                   help='Comma-separated ligand atom specs that are ALLOWED to move, '
                        'e.g. "B:YYE:209:O3,B:YYE:209:P1". Format CHAIN:RESN:RESI:NAME. '
                        'All other ligand atoms stay fixed. If omitted, the entire ligand '
                        'is frozen.')

    # Graft back to the full (uncropped) input PDB
    p.add_argument('--no-graft-back', action='store_true',
                   help='Skip writing a full-protein "_relaxed_grafted.pdb" alongside the '
                        'cropped one. By default, when --crop is on, we also produce a full '
                        'PDB with the relaxed coordinates spliced over the input.')

    # CPU threading — default ON to use whatever cores are available
    p.add_argument('--cpu-threads', type=int, default=None,
                   help='Number of CPU threads for torch/MKL/OpenMP. Default: os.cpu_count(). '
                        'Ignored when --device cuda.')

    p.add_argument('--log-level', type=str, default='INFO',
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    return p.parse_args(argv)


# --------------------------------------------------------------------------
# PDB writing helper — ASE's writer drops chain ID and uses non-standard atom-name
# alignment. We write our own using the per-atom metadata to preserve REMARK 666
# residue identity downstream.
# --------------------------------------------------------------------------
def write_pdb_with_metadata(atoms: Atoms, meta: List[Dict], out_path: Path) -> None:
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    with open(out_path, 'w') as f:
        for i, m in enumerate(meta):
            rec = 'HETATM' if m['is_hetatm'] else 'ATOM  '
            serial = (i + 1) % 100000
            atom_name = m['atom_name']
            # PDB atom-name column 13-16 alignment (PDB v3.3):
            # - 4-char names occupy cols 13-16 left-aligned.
            # - 3-char names whose first char is alphabetic (a standard element)
            #   are indented by one space → col 14 start.
            # - 3-char names whose first char is a digit (e.g. '1HZ', '2HB') stay
            #   left-aligned at col 13 (one trailing space).
            if len(atom_name) == 4:
                name_field = atom_name
            elif atom_name and atom_name[0].isdigit():
                name_field = f"{atom_name:<4s}"
            else:
                name_field = f" {atom_name:<3s}"
            resname = m['resname'][:3].rjust(3)
            chain = (m['chain'] or 'A')[0]
            resnum = m['resnum']
            element = symbols[i].rjust(2)
            x, y, z = pos[i]
            f.write(
                f"{rec}{serial:5d} {name_field} {resname} {chain}{resnum:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element}\n"
            )
        f.write("END\n")


# --------------------------------------------------------------------------
# PDB parsing helpers (ASE-based; preserves residue/chain info in atoms.arrays)
# --------------------------------------------------------------------------
def load_pdb_with_metadata(pdb_path: Path) -> Tuple[Atoms, List[Dict]]:
    """Load PDB into ASE Atoms + parallel list of per-atom metadata dicts.

    Metadata fields: atom_name, resname, chain, resnum, is_hetatm.
    We re-parse the PDB lines directly because ASE's reader drops some info
    (atom_name) into atomtypes which is fine but inconsistent. This wrapper
    gives a single source of truth for indexing.
    """
    atoms = ase_read(pdb_path, format='proteindatabank')
    # Re-parse lines to get atom_name + chain + is_hetatm
    meta = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            if len(line) < 54:
                continue
            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21]
            try:
                resnum = int(line[22:26])
            except ValueError:
                resnum = -1
            meta.append({
                'atom_name': atom_name,
                'resname': resname,
                'chain': chain,
                'resnum': resnum,
                'is_hetatm': line.startswith('HETATM'),
            })
    if len(meta) != len(atoms):
        # Best-effort: alignment may be off if there are multi-MODEL frames or weird records.
        # Truncate to the shorter for downstream safety.
        n = min(len(meta), len(atoms))
        if n != len(meta):
            meta = meta[:n]
        if n != len(atoms):
            atoms = atoms[:n]
    return atoms, meta


def parse_cat_residues_spec(spec: str) -> List[Tuple[str, str, int]]:
    """'A:HIS:55,A:HIS:184,A:LYS:64' → [('A','HIS',55), ('A','HIS',184), ('A','LYS',64)]"""
    out = []
    for entry in (spec or '').split(','):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(':')
        if len(parts) != 3:
            raise ValueError(f"Bad --cat-residues entry {entry!r}; expected 'CHAIN:RESNAME:RESNUM'")
        out.append((parts[0], parts[1], int(parts[2])))
    return out


def parse_cat_residues_from_remark666(pdb_path: Path) -> List[Tuple[str, str, int]]:
    """Extract (chain, resname, resnum) from each REMARK 666 line in the PDB.
    Format: REMARK 666 MATCH TEMPLATE ... MATCH MOTIF <chain> <resn> <resi> ..."""
    out = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('REMARK 666'):
                continue
            # Look for "MATCH MOTIF <chain> <resn> <resi>"
            try:
                idx = line.index('MATCH MOTIF')
                rest = line[idx + len('MATCH MOTIF'):].split()
                if len(rest) >= 3:
                    out.append((rest[0], rest[1], int(rest[2])))
            except (ValueError, IndexError):
                continue
    return out


def parse_ptm_spec(spec: str) -> List[Dict]:
    """'A/LYS/3:KCX' → [{'chain':'A','canonical':'LYS','catres_idx':3,'target':'KCX'}, ...]"""
    out = []
    for entry in (spec or '').split(','):
        entry = entry.strip()
        if not entry:
            continue
        # Format: CHAIN/RESNAME/CATRES_IDX:TARGET
        try:
            lhs, target = entry.split(':')
            chain, resname, idx_str = lhs.split('/')
            out.append({
                'chain': chain,
                'canonical': resname,
                'catres_idx': int(idx_str),
                'target': target,
            })
        except ValueError:
            raise ValueError(f"Bad --ptm-residues entry {entry!r}; expected 'CHAIN/RESNAME/IDX:TARGET'")
    return out


# --------------------------------------------------------------------------
# Atom indexing
# --------------------------------------------------------------------------
def find_ligand_indices(meta: List[Dict], resname: Optional[str], chain: Optional[str]) -> Tuple[List[int], str, str]:
    """Find ligand atom indices by resname (and optionally chain).
    If resname is None, auto-detect: pick the single non-water HETATM resname."""
    waters = {'HOH', 'WAT', 'TIP3', 'TIP4', 'SOL', 'DOD'}
    if resname is None:
        het_resnames = sorted({m['resname'] for m in meta if m['is_hetatm'] and m['resname'] not in waters})
        if not het_resnames:
            raise ValueError("No non-water HETATM residues found; specify --ligand-resname")
        if len(het_resnames) > 1:
            raise ValueError(f"Multiple HETATM resnames found {het_resnames}; specify --ligand-resname")
        resname = het_resnames[0]
    indices = [i for i, m in enumerate(meta)
               if m['is_hetatm'] and m['resname'] == resname and (chain is None or m['chain'] == chain)]
    if not indices:
        raise ValueError(f"No HETATM atoms with resname={resname} chain={chain}")
    chains_found = {meta[i]['chain'] for i in indices}
    if chain is None and len(chains_found) > 1:
        raise ValueError(f"Ligand resname {resname} present in multiple chains {chains_found}; specify --ligand-chain")
    chain = chain or next(iter(chains_found))
    return indices, resname, chain


def find_catres_atoms(meta: List[Dict], cat_residues: List[Tuple[str, str, int]]) -> Dict[Tuple[str, int], List[int]]:
    """For each (chain, resname, resnum) catres, return its list of atom indices in `meta`."""
    by_key = {(c, r): [] for c, _, r in cat_residues}
    for i, m in enumerate(meta):
        key = (m['chain'], m['resnum'])
        if key in by_key:
            by_key[key].append(i)
    return by_key


def catres_sidechain_indices(meta: List[Dict], catres_atoms: Dict[Tuple[str, int], List[int]]) -> List[int]:
    """Sidechain heavy + H atoms (CB and beyond). Backbone N/CA/C/O/HA stay fixed."""
    out = []
    for key, atom_idxs in catres_atoms.items():
        for i in atom_idxs:
            if meta[i]['atom_name'] not in BACKBONE_ATOMS:
                out.append(i)
    return sorted(out)


# --------------------------------------------------------------------------
# PTM post-processing (e.g. LYS → KCX: strip 2 of 3 NZ Hs)
# --------------------------------------------------------------------------
def apply_ptm_kcx(atoms: Atoms, meta: List[Dict],
                  cat_residues: List[Tuple[str, str, int]],
                  ptm_specs: List[Dict],
                  log: logging.Logger,
                  nz_bond_cutoff: float = 1.3) -> Tuple[Atoms, List[Dict]]:
    """For each PTM spec mapping a LYS catres → KCX, strip 2 of 3 NZ Hs so the
    protein-side carries 1 H on NZ (matching the KCX-NZ-CX bond pattern, where
    the CO2 group lives in the ligand HETATM and contributes the other -1 charge).
    Returns updated (atoms, meta) with those Hs removed.
    """
    if not ptm_specs:
        return atoms, meta

    drop_indices: List[int] = []
    for spec in ptm_specs:
        if spec['target'] != 'KCX' or spec['canonical'] != 'LYS':
            log.info(f"PTM {spec} not handled by KCX adjuster; pass-through")
            continue
        # Find the corresponding (chain, resname, resnum) from cat_residues by index
        # (1-indexed catres_idx maps to position in cat_residues list)
        idx = spec['catres_idx'] - 1
        if idx < 0 or idx >= len(cat_residues):
            log.warning(f"PTM catres_idx {spec['catres_idx']} out of range; skipping")
            continue
        chain, resname, resnum = cat_residues[idx]
        if resname != 'LYS':
            log.warning(f"PTM target KCX requires canonical LYS at idx {spec['catres_idx']}, got {resname}; skipping")
            continue
        # Collect H atoms bonded to NZ in this residue
        # (H detection uses ASE chemical symbol, not atom_name — atom names like
        # '1HZ' don't start with 'H' but are still hydrogens.)
        nz_idx = None
        h_idx = []
        symbols = atoms.get_chemical_symbols()
        for i, m in enumerate(meta):
            if m['chain'] == chain and m['resnum'] == resnum:
                if m['atom_name'] == 'NZ':
                    nz_idx = i
                elif symbols[i] == 'H' and m['atom_name'] not in BACKBONE_ATOMS:
                    h_idx.append(i)
        if nz_idx is None:
            log.warning(f"No NZ atom found for LYS {chain}:{resnum}; skipping KCX strip")
            continue
        # Find H atoms within ~1.2 Å of NZ
        nz_coord = atoms.positions[nz_idx]
        nz_hs = []
        for hi in h_idx:
            d = np.linalg.norm(atoms.positions[hi] - nz_coord)
            if d < nz_bond_cutoff:
                nz_hs.append((d, hi))
        nz_hs.sort()
        if len(nz_hs) < 2:
            log.warning(f"LYS {chain}:{resnum} has only {len(nz_hs)} Hs near NZ; expected ≥2 to strip for KCX")
            continue
        # Drop the two closest Hs (leave 1; closest distance ones are most clearly bonded)
        # (Rosetta-protonated LYS NZ has 3 Hs at ~1.01 Å; KCX has 1 H at ~1.01 Å)
        to_drop = [hi for _, hi in nz_hs[:2]]  # keep the THIRD H (or whichever remains)
        # Note: if Rosetta wrote exactly 3 Hs, we keep the one with the largest NZ-H distance
        # (least sterically constrained); arbitrary but defensible.
        # Actually keep the FIRST: standard convention is HZ1/HZ2/HZ3 — keep HZ1.
        # For simplicity, drop the two FARTHEST and keep the closest:
        nz_hs.sort()
        to_drop = [hi for _, hi in nz_hs[1:3]]  # drop second + third closest
        drop_indices.extend(to_drop)
        log.info(f"PTM KCX: LYS {chain}:{resnum} — dropping {len(to_drop)} of {len(nz_hs)} NZ-Hs "
                 f"(atom names: {[meta[i]['atom_name'] for i in to_drop]})")

    if not drop_indices:
        return atoms, meta

    drop_set = set(drop_indices)
    keep = [i for i in range(len(atoms)) if i not in drop_set]
    new_atoms = atoms[keep]
    new_meta = [meta[i] for i in keep]
    return new_atoms, new_meta


# --------------------------------------------------------------------------
# Charge calculation (from residue protonation)
# --------------------------------------------------------------------------
def compute_protein_charge(meta: List[Dict], cat_residues: List[Tuple[str, str, int]],
                           ptm_chain_resnum: Set[Tuple[str, int]],
                           log: logging.Logger) -> Tuple[int, Dict]:
    """Sum residue contributions. HIS is auto-detected (HID/HIE/HIP by H atom presence)."""
    # Group atoms by (chain, resnum) → (resname, atom_names)
    by_res: Dict[Tuple[str, int], Dict] = {}
    for m in meta:
        if m['is_hetatm']:
            continue
        key = (m['chain'], m['resnum'])
        if key not in by_res:
            by_res[key] = {'resname': m['resname'], 'atom_names': []}
        by_res[key]['atom_names'].append(m['atom_name'])

    breakdown = []
    total = 0
    for key, info in by_res.items():
        resname = info['resname']
        atom_names = set(info['atom_names'])
        if key in ptm_chain_resnum:
            # PTM Lys → KCX (-1 per user instruction)
            q = DEFAULT_RESIDUE_CHARGE.get('KCX', -1)
            note = f"PTM (originally {resname}, treated as KCX)"
        elif resname == 'HIS':
            # Detect HID/HIE/HIP via H atoms on ND1/NE2
            has_hd1 = 'HD1' in atom_names
            has_he2 = 'HE2' in atom_names
            if has_hd1 and has_he2:
                q, note = +1, "HIS (HIP — both Hs)"
            elif has_hd1:
                q, note = 0, "HIS (HID — δ-proton)"
            elif has_he2:
                q, note = 0, "HIS (HIE — ε-proton)"
            else:
                q, note = 0, "HIS (deprotonated — both Ns lack H; rare)"
        else:
            q = DEFAULT_RESIDUE_CHARGE.get(resname, 0)
            note = f"{resname} default"
        if q != 0:
            breakdown.append({
                'chain': key[0], 'resnum': key[1], 'resname': resname,
                'charge': q, 'note': note,
            })
        total += q
    log.info(f"Protein net charge (from residue protonation): {total:+d}")
    return total, {'total_protein_charge': int(total), 'nonzero_residues': breakdown}


# --------------------------------------------------------------------------
# Geometry: cropping + clash detection + bond detection
# --------------------------------------------------------------------------
def count_clashes(positions: np.ndarray, meta: List[Dict], cutoff: float = 1.8) -> int:
    """Count atom pairs within cutoff, excluding same-residue pairs and peptide bonds."""
    from scipy.spatial import cKDTree
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=cutoff)
    n = 0
    for i, j in pairs:
        mi, mj = meta[i], meta[j]
        # Skip same residue
        if mi['chain'] == mj['chain'] and mi['resnum'] == mj['resnum']:
            continue
        # Skip peptide bond C(i)-N(i+1)
        if mi['chain'] == mj['chain'] and abs(mi['resnum'] - mj['resnum']) == 1:
            names = {mi['atom_name'], mj['atom_name']}
            if names <= {'C', 'N'}:
                continue
        n += 1
    return n


def detect_ligand_bonds(positions: np.ndarray, ligand_indices: List[int],
                        cutoff: float = 1.7) -> List[Tuple[int, int, float]]:
    """Pairs of ligand atoms with d < cutoff are presumed bonded.
    Returns list of (i, j, ref_distance) tuples for Hookean spring placement."""
    from scipy.spatial import cKDTree
    if not ligand_indices:
        return []
    lig_set = set(ligand_indices)
    lig_pos = positions[ligand_indices]
    local_tree = cKDTree(lig_pos)
    local_pairs = local_tree.query_pairs(r=cutoff)
    out = []
    for li, lj in local_pairs:
        # local index → global index
        gi = ligand_indices[li]
        gj = ligand_indices[lj]
        d = float(np.linalg.norm(positions[gi] - positions[gj]))
        out.append((gi, gj, d))
    return out


def crop_to_ligand(atoms: Atoms, meta: List[Dict], ligand_indices: List[int],
                   radius: float, log: logging.Logger,
                   cap_with_h: bool = True) -> Tuple[Atoms, List[Dict], List[int]]:
    """Crop to residues with any atom within radius of any ligand atom.

    When cap_with_h=True, every kept residue whose N- or C-side peptide neighbour
    has been DROPPED gets a hydrogen placed along the original C(prev)–N or
    N(next)–C bond direction (~1.01 Å for N-cap, ~1.09 Å for C-cap). The cap H
    keeps the backbone fragment chemically valid for MLFF so boundary forces
    don't dominate.

    Returns (cropped_atoms, cropped_meta, cropped_ligand_indices).
    """
    from scipy.spatial import cKDTree
    pos = atoms.get_positions()
    syms = atoms.get_chemical_symbols()
    lig_pos = pos[ligand_indices]
    tree = cKDTree(pos)
    within = set()
    for lp in lig_pos:
        within.update(tree.query_ball_point(lp, r=radius))
    res_in = set()
    for i in within:
        res_in.add((meta[i]['chain'], meta[i]['resnum']))
    keep = [i for i, m in enumerate(meta) if (m['chain'], m['resnum']) in res_in]
    keep_set = set(keep)
    log.info(f"Crop: kept {len(keep)} of {len(atoms)} atoms ({len(res_in)} residues) within {radius} Å of ligand")

    # ---- H-cap at peptide-bond cuts (simple MVP)
    # Build per-residue C/N atom lookup for the FULL input, plus track which
    # (chain, resnum) tuples we are keeping. For each kept residue, check
    # whether resnum-1 and resnum+1 (same chain) are also kept; if not, the
    # cut peptide bond gets an H.
    cap_positions: List[np.ndarray] = []
    cap_meta: List[Dict] = []
    if cap_with_h:
        by_res: Dict[Tuple[str, int], Dict[str, int]] = {}
        for i, m in enumerate(meta):
            if m['is_hetatm']:
                continue
            by_res.setdefault((m['chain'], m['resnum']), {})[m['atom_name']] = i
        for (chain, resnum), atoms_in_res in by_res.items():
            if (chain, resnum) not in res_in:
                continue
            n_idx = atoms_in_res.get('N')
            c_idx = atoms_in_res.get('C')
            # N-side cut: if prev residue is NOT kept and we have N + prev's C
            if n_idx is not None:
                prev_atoms = by_res.get((chain, resnum - 1), {})
                prev_c = prev_atoms.get('C')
                if prev_c is not None and (chain, resnum - 1) not in res_in:
                    # H along N → (prev C) direction at 1.01 Å
                    n_pos = pos[n_idx]
                    direction = pos[prev_c] - n_pos
                    norm = np.linalg.norm(direction)
                    if norm > 1e-6:
                        cap_positions.append(n_pos + direction / norm * 1.01)
                        cap_meta.append({
                            'atom_name': 'HN0', 'resname': meta[n_idx]['resname'],
                            'chain': chain, 'resnum': resnum, 'is_hetatm': False,
                        })
            # C-side cut: if next residue is NOT kept and we have C + next's N
            if c_idx is not None:
                next_atoms = by_res.get((chain, resnum + 1), {})
                next_n = next_atoms.get('N')
                if next_n is not None and (chain, resnum + 1) not in res_in:
                    c_pos = pos[c_idx]
                    direction = pos[next_n] - c_pos
                    norm = np.linalg.norm(direction)
                    if norm > 1e-6:
                        cap_positions.append(c_pos + direction / norm * 1.09)
                        cap_meta.append({
                            'atom_name': 'HC0', 'resname': meta[c_idx]['resname'],
                            'chain': chain, 'resnum': resnum, 'is_hetatm': False,
                        })
        if cap_meta:
            log.info(f"Crop H-cap: added {len(cap_meta)} H atoms at peptide-bond cuts")

    new_atoms = atoms[keep]
    new_meta = [meta[i] for i in keep]
    new_ligand_indices = []
    old_to_new = {old: new for new, old in enumerate(keep)}
    for old in ligand_indices:
        if old in keep_set:
            new_ligand_indices.append(old_to_new[old])

    # Append H caps as additional atoms (extend the ASE Atoms object)
    if cap_positions:
        from ase import Atoms as _Atoms
        cap_atoms = _Atoms(symbols=['H'] * len(cap_positions),
                           positions=np.array(cap_positions))
        new_atoms = new_atoms + cap_atoms
        new_meta = new_meta + cap_meta

    return new_atoms, new_meta, new_ligand_indices


# --------------------------------------------------------------------------
# Constraint builder
# --------------------------------------------------------------------------
def build_constraints(atoms: Atoms, meta: List[Dict],
                      catres_sidechain_idx: List[int],
                      ligand_indices: List[int],
                      ligand_free_idx: List[int],
                      log: logging.Logger) -> List:
    """FixAtoms model:
       - FREE: catres sidechain heavy + H atoms, plus user-selected ligand atoms
         passed via --ligand-free-atoms (ligand_free_idx). Everything else fixed.
       - Ligand atoms NOT in ligand_free_idx are LOCKED. No Hookean springs — atoms
         are either pinned or free, no in-between. This keeps the ligand pose
         honest and reproducible.
    """
    n = len(atoms)
    free = set(catres_sidechain_idx) | set(ligand_free_idx)
    fixed = sorted(i for i in range(n) if i not in free)
    ligand_locked = sorted(set(ligand_indices) - set(ligand_free_idx))
    log.info(f"Constraints: {len(fixed)} atoms FIXED, "
             f"{len(catres_sidechain_idx)} catres sidechain atoms FREE, "
             f"{len(ligand_free_idx)} ligand atoms FREE "
             f"({len(ligand_locked)}/{len(ligand_indices)} ligand atoms LOCKED).")
    return [FixAtoms(indices=fixed)]


def parse_ligand_free_atoms(spec: Optional[str], meta: List[Dict],
                            log: logging.Logger) -> List[int]:
    """Parse "B:YYE:209:O3,B:YYE:209:P1" → atom indices into the cropped meta."""
    if not spec:
        return []
    out: List[int] = []
    for chunk in spec.split(','):
        parts = chunk.strip().split(':')
        if len(parts) != 4:
            raise ValueError(f"--ligand-free-atoms entry {chunk!r} must be CHAIN:RESN:RESI:NAME")
        c, rn, ri, nm = parts
        ri_i = int(ri)
        match = None
        for i, m in enumerate(meta):
            if (m['chain'] == c and m['resname'] == rn
                    and m['resnum'] == ri_i and m['atom_name'] == nm):
                match = i
                break
        if match is None:
            log.warning(f"--ligand-free-atoms {chunk!r}: no matching atom in cropped system; skipping")
            continue
        out.append(match)
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger('active_site_relax')

    args.out.mkdir(parents=True, exist_ok=True)
    basename = args.input.stem

    # ---- CPU threading: pin OpenMP/MKL/torch to use available cores
    if args.device == 'cpu':
        import os as _os
        n_cores = args.cpu_threads or _os.cpu_count() or 1
        _os.environ['OMP_NUM_THREADS'] = str(n_cores)
        _os.environ['MKL_NUM_THREADS'] = str(n_cores)
        _os.environ['NUMEXPR_MAX_THREADS'] = str(n_cores)
        _os.environ['OPENBLAS_NUM_THREADS'] = str(n_cores)
        try:
            import torch  # noqa: WPS433
            torch.set_num_threads(n_cores)
            torch.set_num_interop_threads(max(1, n_cores // 2))
        except Exception as e:
            log.warning(f"Could not set torch thread count: {e}")
        log.info(f"CPU threading: requesting {n_cores} threads (omp/mkl/numexpr/openblas/torch)")

    # ---- Load
    log.info(f"Loading {args.input}")
    atoms, meta = load_pdb_with_metadata(args.input)
    log.info(f"  {len(atoms)} atoms loaded")
    # Stash the FULL pre-relax atoms+meta for graft-back below
    full_atoms_initial = atoms.copy()
    full_meta_initial = [dict(m) for m in meta]

    # ---- Catres
    if args.cat_residues:
        cat_residues = parse_cat_residues_spec(args.cat_residues)
    else:
        cat_residues = parse_cat_residues_from_remark666(args.input)
    log.info(f"Catres ({len(cat_residues)}): {cat_residues}")
    if not cat_residues:
        log.error("No catalytic residues found (REMARK 666 missing or --cat-residues empty)")
        return 1

    # ---- Ligand
    ligand_indices, ligand_resname, ligand_chain = find_ligand_indices(
        meta, args.ligand_resname, args.ligand_chain
    )
    log.info(f"Ligand: {ligand_resname} chain {ligand_chain}, {len(ligand_indices)} atoms")

    # ---- PTM post-processing
    ptm_specs = parse_ptm_spec(args.ptm_residues) if args.ptm_residues else []
    if ptm_specs:
        atoms, meta = apply_ptm_kcx(atoms, meta, cat_residues, ptm_specs, log,
                                    nz_bond_cutoff=args.nz_h_bond_cutoff)
        # Re-index ligand after atom drops
        ligand_indices, _, _ = find_ligand_indices(meta, ligand_resname, ligand_chain)

    # ---- Optional crop
    if args.crop:
        atoms, meta, ligand_indices = crop_to_ligand(
            atoms, meta, ligand_indices, args.crop_radius, log,
            cap_with_h=not args.no_crop_h_cap,
        )

    # ---- Identify catres sidechain atom indices
    catres_atoms = find_catres_atoms(meta, cat_residues)
    catres_sc_idx = catres_sidechain_indices(meta, catres_atoms)
    log.info(f"Catres sidechain atoms (FREE for opt): {len(catres_sc_idx)}")

    # ---- Charge
    ptm_chain_resnum = set()
    for spec in ptm_specs:
        if spec['target'] == 'KCX' and 1 <= spec['catres_idx'] <= len(cat_residues):
            chain, _, resnum = cat_residues[spec['catres_idx'] - 1]
            ptm_chain_resnum.add((chain, resnum))
    protein_charge, charge_breakdown = compute_protein_charge(meta, cat_residues, ptm_chain_resnum, log)

    # Fail-fast: charge-aware calculators need an explicit ligand charge
    CHARGE_AWARE_MODELS = ('mace-mh-1', 'mace-polar-', 'xtb', 'aimnet2', 'uma-')
    is_charge_aware = any(args.model.startswith(p) for p in CHARGE_AWARE_MODELS)
    if is_charge_aware and args.charge_ligand is None:
        log.error(f"Model '{args.model}' is charge-aware; --charge-ligand is required "
                  f"(silent default to 0 would risk a wrong total charge).")
        return 1
    total_charge = protein_charge + (args.charge_ligand or 0)
    log.info(f"Total system charge: protein {protein_charge:+d} + ligand {args.charge_ligand or 0:+d} = {total_charge:+d}")

    # ---- Calculator
    log.info(f"Loading MLFF: {args.model} (head={args.head}, device={args.device})")
    calc = make_calc(args.model, head=args.head, device=args.device, charge=total_charge)
    atoms.calc = calc

    # ---- Constraints
    ligand_free_idx = parse_ligand_free_atoms(args.ligand_free_atoms, meta, log)
    constraints = build_constraints(
        atoms, meta, catres_sc_idx, ligand_indices,
        ligand_free_idx=ligand_free_idx,
        log=log,
    )
    atoms.set_constraint(constraints)

    # ---- Clash counts before
    pos_before = atoms.get_positions().copy()
    clashes_before = count_clashes(pos_before, meta, cutoff=args.clash_cutoff)
    log.info(f"Hard clashes (cutoff 1.8 Å, non-bonded, cross-residue): {clashes_before}")

    # ---- Energy before
    e_before = float(atoms.get_potential_energy())
    log.info(f"Initial energy: {e_before:.6f} eV")

    # ---- Optimize
    opt_log = args.out / f"{basename}_opt.log"
    opt_traj = args.out / f"{basename}_opt.traj"
    Opt = LBFGS if args.optimizer == 'lbfgs' else FIRE
    opt = Opt(atoms, logfile=str(opt_log), trajectory=str(opt_traj))
    log.info(f"Optimising with {args.optimizer} (fmax={args.fmax}, max_steps={args.max_steps})")
    converged = opt.run(fmax=args.fmax, steps=args.max_steps)
    n_steps = opt.nsteps

    # ---- Stats after
    e_after = float(atoms.get_potential_energy())
    pos_after = atoms.get_positions()
    clashes_after = count_clashes(pos_after, meta, cutoff=args.clash_cutoff)
    fmax_final = float(np.max(np.linalg.norm(atoms.get_forces(), axis=-1)))

    # Catres sidechain displacement
    sc_disp = np.linalg.norm(pos_after[catres_sc_idx] - pos_before[catres_sc_idx], axis=-1)
    sc_max = float(sc_disp.max()) if len(sc_disp) else 0.0
    sc_mean = float(sc_disp.mean()) if len(sc_disp) else 0.0

    # Ligand RMSD (rigid body should be small)
    if ligand_indices:
        lig_disp = np.linalg.norm(pos_after[ligand_indices] - pos_before[ligand_indices], axis=-1)
        lig_mean = float(np.sqrt(np.mean(lig_disp**2)))
        lig_max = float(lig_disp.max())
    else:
        lig_mean = lig_max = 0.0

    log.info(f"Converged: {converged} in {n_steps} steps; ΔE = {e_after - e_before:.4f} eV")
    log.info(f"fmax_final = {fmax_final:.4f} eV/Å")
    log.info(f"Catres sidechain disp: max={sc_max:.4f} Å, mean={sc_mean:.4f} Å")
    log.info(f"Ligand atom disp: mean={lig_mean:.4f} Å, max={lig_max:.4f} Å "
             f"(expect ~0 except for --ligand-free-atoms)")
    log.info(f"Clashes: {clashes_before} → {clashes_after}")

    # ---- Outputs
    out_pdb = args.out / f"{basename}_relaxed.pdb"
    write_pdb_with_metadata(atoms, meta, out_pdb)
    log.info(f"Wrote {out_pdb}  (active-site only)")

    # ---- Graft-back: splice relaxed coords over the FULL input PDB
    grafted_pdb = None
    if not args.no_graft_back:
        # Build (chain, resnum, atom_name) → relaxed-position lookup
        relaxed_pos = atoms.get_positions()
        relaxed_lookup: Dict[Tuple[str, int, str], np.ndarray] = {}
        for i, m in enumerate(meta):
            relaxed_lookup[(m['chain'], m['resnum'], m['atom_name'])] = relaxed_pos[i]
        # Walk the FULL initial structure; replace coords where present in relaxed
        grafted_pos = full_atoms_initial.get_positions().copy()
        n_replaced = 0
        for i, m in enumerate(full_meta_initial):
            key = (m['chain'], m['resnum'], m['atom_name'])
            if key in relaxed_lookup:
                grafted_pos[i] = relaxed_lookup[key]
                n_replaced += 1
        # KCX H-strip: drop any atom from the initial that NO LONGER exists in the
        # relaxed set when the residue was PTM-converted, so the grafted PDB
        # matches the (chain,resnum) atom inventory of the relaxed cropped region.
        # Build the set of (chain, resnum, atom_name) that survived the relax for
        # residues touched by PTM specs.
        ptm_touched = set()
        for spec in ptm_specs:
            if 1 <= spec['catres_idx'] <= len(cat_residues):
                ch, _, rn = cat_residues[spec['catres_idx'] - 1]
                ptm_touched.add((ch, rn))
        if ptm_touched:
            relaxed_keys = set(relaxed_lookup.keys())
            keep = []
            for i, m in enumerate(full_meta_initial):
                if (m['chain'], m['resnum']) in ptm_touched:
                    if (m['chain'], m['resnum'], m['atom_name']) not in relaxed_keys:
                        continue
                keep.append(i)
            grafted_meta = [full_meta_initial[i] for i in keep]
            grafted_atoms = full_atoms_initial[keep]
            grafted_atoms.set_positions(grafted_pos[keep])
        else:
            grafted_meta = full_meta_initial
            grafted_atoms = full_atoms_initial.copy()
            grafted_atoms.set_positions(grafted_pos)
        grafted_pdb = args.out / f"{basename}_relaxed_grafted.pdb"
        write_pdb_with_metadata(grafted_atoms, grafted_meta, grafted_pdb)
        log.info(f"Wrote {grafted_pdb}  (full input with relaxed active-site spliced in, "
                 f"{n_replaced}/{len(full_meta_initial)} atoms updated)")

    summary = {
        'input': str(args.input),
        'out': str(args.out),
        'n_atoms_in': len(pos_before),
        'n_atoms_after_ptm_crop': len(atoms),
        'cat_residues': [{'chain': c, 'resname': r, 'resnum': n} for c, r, n in cat_residues],
        'ligand': {'resname': ligand_resname, 'chain': ligand_chain, 'n_atoms': len(ligand_indices)},
        'ptm_specs': ptm_specs,
        'cropped': bool(args.crop),
        'crop_radius_A': args.crop_radius if args.crop else None,
        'model': args.model, 'head': args.head, 'device': args.device,
        'optimizer': args.optimizer,
        'fmax_target': args.fmax,
        'max_steps': args.max_steps,
        'ligand_free_atoms': args.ligand_free_atoms,
        'graft_back': not args.no_graft_back,
        'charge_ligand': args.charge_ligand,
        'protein_charge': protein_charge,
        'total_charge': total_charge,
        'energy_initial_eV': e_before,
        'energy_final_eV': e_after,
        'delta_E_kcal_per_mol': (e_after - e_before) * 23.06035,
        'fmax_final_eV_per_A': fmax_final,
        'converged': bool(converged),
        'n_optimizer_steps': int(n_steps),
        'clashes_before': int(clashes_before),
        'clashes_after': int(clashes_after),
        'catres_sidechain_max_disp_A': sc_max,
        'catres_sidechain_mean_disp_A': sc_mean,
        'ligand_rmsd_A': lig_mean,
        'ligand_max_disp_A': lig_max,
        'outputs': {
            'relaxed_pdb': str(out_pdb),
            'opt_log': str(opt_log),
            'opt_traj': str(opt_traj),
        },
    }
    with open(args.out / f"{basename}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    with open(args.out / f"{basename}_charge_breakdown.json", 'w') as f:
        json.dump(charge_breakdown, f, indent=2)
    log.info(f"Wrote summary + charge breakdown")

    return 0 if converged else 2


if __name__ == '__main__':
    sys.exit(main())
