#!/usr/bin/env python
"""active_site_metrics.py — Diagnostic metrics comparing reference vs final PDB.

Phase 4 of the AF3-to-design pipeline:
  align_prediction_to_ref_pdb_and_copy_lig.py
    → active_site_relax.py
    → active_site_metrics.py   ← this file

Computes:
  1. Global RMSDs (Kabsch-aligned heavy-atom): whole-protein, backbone-only, catres-all-atom
  2. Per-cat-residue: heavy-atom RMSD, per-chi-angle error (degrees), sidechain RMS displacement
  3. Cat-residue ↔ ligand: closest heavy-atom distance ref vs final; H-bond donor/acceptor geometry
  4. Cat-residue ↔ cat-residue: minimum sidechain heavy-atom distance ref vs final
  5. Reactive-atom-triplet (P / Onuc / Olg): d(P-Onuc), d(P-Olg), Onuc-P-Olg angle
  6. Clash count (heavy-atom only, non-bonded cross-residue, cutoff 1.8 Å)

Outputs (next to <final_pdb>):
  - <basename>_metrics.csv     — flat per-metric table
  - <basename>_metrics.json    — full structured dump including flags
  - <basename>_metrics_REPORT.md — markdown summary with flagged drifts

Generalizable: cat residues come from REMARK 666 or CLI; ligand resname auto-detected
or specified; reactive atoms via CLI flag. No PTE-specific hardcoded paths/residues.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Reuse parsing utilities from active_site_relax.py
sys.path.insert(0, str(Path(__file__).parent))
from active_site_relax import (  # noqa: E402
    load_pdb_with_metadata,
    parse_cat_residues_spec,
    parse_cat_residues_from_remark666,
    find_ligand_indices,
)

# Standard sidechain chi-angle atom-name quadruplets
# (None marks "no such chi for this residue")
CHI_DEFINITIONS: Dict[str, List[Tuple[str, str, str, str]]] = {
    'ARG': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD'),
            ('CB', 'CG', 'CD', 'NE'), ('CG', 'CD', 'NE', 'CZ')],
    'ASN': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'OD1')],
    'ASP': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'OD1')],
    'CYS': [('N', 'CA', 'CB', 'SG')],
    'GLN': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD'),
            ('CB', 'CG', 'CD', 'OE1')],
    'GLU': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD'),
            ('CB', 'CG', 'CD', 'OE1')],
    'HIS': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'ND1')],
    'HID': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'ND1')],
    'HIE': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'ND1')],
    'HIP': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'ND1')],
    'ILE': [('N', 'CA', 'CB', 'CG1'), ('CA', 'CB', 'CG1', 'CD1')],
    'KCX': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD'),
            ('CB', 'CG', 'CD', 'CE'), ('CG', 'CD', 'CE', 'NZ')],
    'LEU': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD1')],
    'LYS': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD'),
            ('CB', 'CG', 'CD', 'CE'), ('CG', 'CD', 'CE', 'NZ')],
    'MET': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'SD'),
            ('CB', 'CG', 'SD', 'CE')],
    'PHE': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD1')],
    'SER': [('N', 'CA', 'CB', 'OG')],
    'THR': [('N', 'CA', 'CB', 'OG1')],
    'TRP': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD1')],
    'TYR': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD1')],
    'VAL': [('N', 'CA', 'CB', 'CG1')],
    # PTMs
    'SEP': [('N', 'CA', 'CB', 'OG')],
    'TPO': [('N', 'CA', 'CB', 'OG1')],
    'PTR': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD1')],
}

BACKBONE_ATOMS = {'N', 'CA', 'C', 'O', 'OXT'}  # mainchain heavy atoms

# H-bond donor/acceptor classification for distance + angle metric (rough heuristic)
HBOND_DONOR_HEAVY = {  # residue → set of heavy donors
    'SER': {'OG'}, 'THR': {'OG1'}, 'TYR': {'OH'}, 'CYS': {'SG'},
    'ASN': {'ND2'}, 'GLN': {'NE2'},
    'ARG': {'NE', 'NH1', 'NH2'}, 'LYS': {'NZ'}, 'KCX': {'NZ'},
    'HIS': {'ND1', 'NE2'}, 'HID': {'ND1'}, 'HIE': {'NE2'}, 'HIP': {'ND1', 'NE2'},
    'TRP': {'NE1'},
}
HBOND_ACCEPTOR_HEAVY = {
    'ASP': {'OD1', 'OD2'}, 'GLU': {'OE1', 'OE2'},
    'ASN': {'OD1'}, 'GLN': {'OE1'},
    'SER': {'OG'}, 'THR': {'OG1'}, 'TYR': {'OH'},
    'HIS': {'ND1', 'NE2'}, 'HID': {'NE2'}, 'HIE': {'ND1'},
    'CYS': {'SG'}, 'MET': {'SD'},
}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ref', type=Path, required=True,
                   help='Reference PDB (theozyme template, with REMARK 666).')
    p.add_argument('--final', type=Path, required=True,
                   help='Final PDB (e.g. output of active_site_relax.py).')
    p.add_argument('--out', type=Path, default=None,
                   help='Output directory. Default: parent of --final.')
    p.add_argument('--cat-residues', type=str, default=None,
                   help='Comma-separated CHAIN:RESNAME:RESNUM. Default: parse REMARK 666 from --ref.')
    p.add_argument('--ligand-resname', type=str, default=None,
                   help='Ligand 3-letter resname. Default: first HETATM resname.')
    p.add_argument('--ligand-chain', type=str, default=None,
                   help='Ligand chain ID. Default: auto-detect (single-chain ligand).')
    p.add_argument('--reactive-atoms', type=str, default=None,
                   help='Reactive triplet, format "CHAIN:RESN:RESI:NAME,...". '
                        'Three atoms: P-center, Onuc, Olg. e.g. "B:YYE:209:P1,B:YYE:209:O3,B:YYE:209:O7"')
    p.add_argument('--clash-cutoff', type=float, default=1.8,
                   help='Heavy-atom clash distance threshold (Å). Default 1.8.')
    p.add_argument('--drift-threshold', type=float, default=0.5,
                   help='Distance drift (Å) above which a metric is flagged. Default 0.5.')
    p.add_argument('--angle-drift-threshold', type=float, default=5.0,
                   help='Angle drift (deg) above which a metric is flagged. Default 5.0.')
    p.add_argument('--hbond-cutoff', type=float, default=3.5,
                   help='Donor-acceptor distance threshold for H-bond presence (Å). Default 3.5.')
    p.add_argument('--global-rmsd-flag-A', type=float, default=1.0,
                   help='Whole-protein heavy RMSD that triggers a global-drift flag (Å). Default 1.0.')
    p.add_argument('--catres-rmsd-flag-A', type=float, default=0.5,
                   help='Catres all-heavy RMSD that triggers a catalytic-geometry flag (Å). Default 0.5.')
    p.add_argument('--reactive-distance-flag-A', type=float, default=0.05,
                   help='Reactive-triplet bond-distance drift flag threshold (Å). Default 0.05.')
    p.add_argument('--clash-growth-flag', type=int, default=2,
                   help='Allowed clash count growth before flagging (#). Default 2.')
    p.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rotation R, translation t) that maps P onto Q minimising RMSD.

    P_aligned = (P - P_centroid) @ R + Q_centroid
    """
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    det = np.linalg.det(Vt.T @ U.T)
    d = 1.0 if det >= 0 else -1.0  # explicit branch — np.sign(0) → 0 would yield singular D
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    t = Q.mean(axis=0) - P.mean(axis=0) @ R
    return R, t


def apply_kabsch(P: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return P @ R + t


def rmsd(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))


def dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute dihedral angle p0-p1-p2-p3 in degrees, range (-180, 180]."""
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    nb1 = np.linalg.norm(b1)
    if nb1 < 1e-9:
        return float('nan')
    b1n = b1 / nb1
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle a-b-c in degrees."""
    ba = a - b
    bc = c - b
    cos_t = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-12)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_t)))


def angular_difference(a_deg: float, b_deg: float) -> float:
    """Minimum signed angle difference in degrees, [-180, 180]."""
    d = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    return float(d)


# --------------------------------------------------------------------------
# Atom indexing helpers
# --------------------------------------------------------------------------
def build_residue_atom_map(meta: List[Dict]) -> Dict[Tuple[str, int], Dict[str, int]]:
    """Group atom indices by (chain, resnum) → atom_name → idx."""
    out: Dict[Tuple[str, int], Dict[str, int]] = {}
    for i, m in enumerate(meta):
        key = (m['chain'], m['resnum'])
        out.setdefault(key, {})[m['atom_name']] = i
    return out


def heavy_atom_indices(meta: List[Dict], symbols: List[str], exclude_hetatm: bool = True) -> List[int]:
    out = []
    for i, m in enumerate(meta):
        if exclude_hetatm and m['is_hetatm']:
            continue
        if symbols[i] == 'H':
            continue
        out.append(i)
    return out


def backbone_indices(meta: List[Dict]) -> List[int]:
    return [i for i, m in enumerate(meta)
            if not m['is_hetatm'] and m['atom_name'] in BACKBONE_ATOMS]


def catres_all_atom_indices(meta: List[Dict], symbols: List[str],
                            cat_residues: List[Tuple[str, str, int]]) -> List[int]:
    """All HEAVY atoms (backbone + sidechain) of each catres."""
    catset = {(c, r) for c, _, r in cat_residues}
    return [i for i, m in enumerate(meta)
            if (m['chain'], m['resnum']) in catset and symbols[i] != 'H']


def catres_sidechain_heavy_indices(meta: List[Dict], symbols: List[str],
                                   chain: str, resnum: int) -> List[int]:
    """Heavy atoms beyond CB (CB + sidechain tip)."""
    return [i for i, m in enumerate(meta)
            if m['chain'] == chain and m['resnum'] == resnum
            and m['atom_name'] not in BACKBONE_ATOMS
            and symbols[i] != 'H']


# --------------------------------------------------------------------------
# Alignment of FINAL onto REF, by SHARED heavy atoms (chain, resnum, atom_name)
# --------------------------------------------------------------------------
def shared_protein_atoms(ref_meta: List[Dict], ref_syms: List[str],
                         fin_meta: List[Dict], fin_syms: List[str]) -> List[Tuple[int, int]]:
    """Build pairs of (ref_idx, fin_idx) for shared (chain, resnum, atom_name)
    protein heavy atoms. Used to compute Kabsch."""
    ref_lookup = {}
    for i, m in enumerate(ref_meta):
        if m['is_hetatm']:
            continue
        if ref_syms[i] == 'H':
            continue
        ref_lookup[(m['chain'], m['resnum'], m['atom_name'])] = i
    pairs = []
    for j, m in enumerate(fin_meta):
        if m['is_hetatm']:
            continue
        if fin_syms[j] == 'H':
            continue
        key = (m['chain'], m['resnum'], m['atom_name'])
        i = ref_lookup.get(key)
        if i is not None:
            pairs.append((i, j))
    return pairs


# --------------------------------------------------------------------------
# Per-cat-residue metrics
# --------------------------------------------------------------------------
def per_residue_chi_errors(ref_meta: List[Dict], ref_pos: np.ndarray,
                           fin_meta: List[Dict], fin_pos: np.ndarray,
                           ref_amap: Dict, fin_amap: Dict,
                           chain: str, resname: str, resnum: int,
                           log: logging.Logger) -> List[Optional[float]]:
    """Return per-chi error |Δchi| in degrees, with the canonical (in-protein) resname.

    Returns a list of length len(CHI_DEFINITIONS[resname]); entries are None when
    a required atom is missing from EITHER ref or final.
    """
    defs = CHI_DEFINITIONS.get(resname)
    if defs is None:
        return []
    ref_atoms = ref_amap.get((chain, resnum), {})
    fin_atoms = fin_amap.get((chain, resnum), {})
    out: List[Optional[float]] = []
    for quad in defs:
        try:
            ri = [ref_atoms[name] for name in quad]
            fi = [fin_atoms[name] for name in quad]
        except KeyError as e:
            log.debug(f"Chi missing atom {e} for {chain}:{resname}:{resnum} (quad {quad})")
            out.append(None)
            continue
        chi_ref = dihedral(*[ref_pos[i] for i in ri])
        chi_fin = dihedral(*[fin_pos[i] for i in fi])
        out.append(abs(angular_difference(chi_ref, chi_fin)))
    return out


def per_residue_metrics(ref_meta, ref_pos, ref_syms,
                        fin_meta, fin_pos, fin_syms,
                        cat_residues, log) -> List[Dict]:
    """Per-cat-residue: heavy-atom RMSD (Kabsch-aligned at this point), per-chi error,
    sidechain heavy RMS displacement."""
    ref_amap = build_residue_atom_map(ref_meta)
    fin_amap = build_residue_atom_map(fin_meta)
    out = []
    for chain, resname, resnum in cat_residues:
        # All-atom heavy RMSD (no per-residue re-alignment; just direct delta after global Kabsch)
        ref_at = ref_amap.get((chain, resnum), {})
        fin_at = fin_amap.get((chain, resnum), {})
        shared = [(ref_at[n], fin_at[n]) for n in ref_at if n in fin_at]
        # heavy-only
        shared_heavy = [(ri, fi) for ri, fi in shared
                        if ref_syms[ri] != 'H' and fin_syms[fi] != 'H']
        sc_shared_heavy = [(ri, fi) for ri, fi in shared_heavy
                           if ref_meta[ri]['atom_name'] not in BACKBONE_ATOMS]
        if shared_heavy:
            A = np.array([ref_pos[ri] for ri, _ in shared_heavy])
            B = np.array([fin_pos[fi] for _, fi in shared_heavy])
            all_rmsd = rmsd(A, B)
            n_all = len(shared_heavy)
        else:
            all_rmsd = float('nan')
            n_all = 0
        if sc_shared_heavy:
            A = np.array([ref_pos[ri] for ri, _ in sc_shared_heavy])
            B = np.array([fin_pos[fi] for _, fi in sc_shared_heavy])
            sc_rmsd = rmsd(A, B)
            n_sc = len(sc_shared_heavy)
        else:
            sc_rmsd = float('nan')
            n_sc = 0

        chi_errs = per_residue_chi_errors(ref_meta, ref_pos, fin_meta, fin_pos,
                                          ref_amap, fin_amap, chain, resname, resnum, log)
        out.append({
            'chain': chain, 'resname': resname, 'resnum': resnum,
            'all_atom_heavy_rmsd_A': all_rmsd, 'n_all_atom': n_all,
            'sidechain_heavy_rmsd_A': sc_rmsd, 'n_sidechain': n_sc,
            'chi_errors_deg': chi_errs,
        })
    return out


# --------------------------------------------------------------------------
# Cat-residue ↔ ligand metrics
# --------------------------------------------------------------------------
def catres_to_ligand_geometry(ref_meta, ref_pos, ref_syms,
                              fin_meta, fin_pos, fin_syms,
                              cat_residues, ligand_chain, ligand_resname,
                              hbond_cutoff: float, log) -> List[Dict]:
    """For each cat-residue, identify the heavy atom closest to ligand in REF.
    Report its distance to ligand in ref vs final.
    Also report any donor/acceptor H-bond geometry (distance + angle via CA-N-acceptor).
    """
    def _is_lig(m, chain_arg):
        if not m['is_hetatm']:
            return False
        if m['resname'] != ligand_resname:
            return False
        if chain_arg is not None and m['chain'] != chain_arg:
            return False
        return True

    ref_lig = np.array([ref_pos[i] for i, m in enumerate(ref_meta)
                        if _is_lig(m, ligand_chain) and ref_syms[i] != 'H'])
    fin_lig = np.array([fin_pos[i] for i, m in enumerate(fin_meta)
                        if _is_lig(m, ligand_chain) and fin_syms[i] != 'H'])
    ref_amap = build_residue_atom_map(ref_meta)
    fin_amap = build_residue_atom_map(fin_meta)
    out = []
    for chain, resname, resnum in cat_residues:
        # Heavy sidechain atoms in ref + their indices
        ref_sc = catres_sidechain_heavy_indices(ref_meta, ref_syms, chain, resnum)
        if not ref_sc or len(ref_lig) == 0:
            out.append({
                'chain': chain, 'resname': resname, 'resnum': resnum,
                'closest_atom_name': None, 'ref_min_dist_A': None, 'final_min_dist_A': None,
                'drift_A': None, 'hbonds': [],
            })
            continue
        # For each sidechain atom: min distance to any ligand heavy atom in ref
        ref_dists = []
        for ri in ref_sc:
            d2 = np.sum((ref_lig - ref_pos[ri]) ** 2, axis=1)
            ref_dists.append((np.sqrt(d2.min()), ri))
        ref_dists.sort()
        best_d_ref, best_ri = ref_dists[0]
        best_name = ref_meta[best_ri]['atom_name']
        # Final: same-name atom, min distance to any final-ligand heavy atom
        fin_at = fin_amap.get((chain, resnum), {})
        if best_name in fin_at and len(fin_lig) > 0:
            fi = fin_at[best_name]
            d2 = np.sum((fin_lig - fin_pos[fi]) ** 2, axis=1)
            best_d_fin = float(np.sqrt(d2.min()))
        else:
            best_d_fin = None
        # H-bond donor/acceptor analysis: for any donor/acceptor heavy in this catres,
        # find closest ligand heavy and report dist; flag if it broke
        hbonds = []
        donors = HBOND_DONOR_HEAVY.get(resname, set())
        acceptors = HBOND_ACCEPTOR_HEAVY.get(resname, set())
        for atom_name in (donors | acceptors):
            ri = ref_amap.get((chain, resnum), {}).get(atom_name)
            fi = fin_amap.get((chain, resnum), {}).get(atom_name)
            if ri is None or fi is None or len(ref_lig) == 0 or len(fin_lig) == 0:
                continue
            d_ref = float(np.sqrt(np.min(np.sum((ref_lig - ref_pos[ri]) ** 2, axis=1))))
            d_fin = float(np.sqrt(np.min(np.sum((fin_lig - fin_pos[fi]) ** 2, axis=1))))
            if d_ref < hbond_cutoff:
                hbonds.append({
                    'atom_name': atom_name,
                    'role': 'donor' if atom_name in donors else 'acceptor',
                    'ref_dist_A': d_ref, 'final_dist_A': d_fin,
                    'broken': d_fin > hbond_cutoff,
                })
        drift = abs(best_d_fin - best_d_ref) if best_d_fin is not None else None
        out.append({
            'chain': chain, 'resname': resname, 'resnum': resnum,
            'closest_atom_name': best_name,
            'ref_min_dist_A': float(best_d_ref),
            'final_min_dist_A': best_d_fin,
            'drift_A': drift,
            'hbonds': hbonds,
        })
    return out


# --------------------------------------------------------------------------
# Cat-residue ↔ cat-residue metrics
# --------------------------------------------------------------------------
def catres_to_catres_geometry(ref_meta, ref_pos, ref_syms,
                              fin_meta, fin_pos, fin_syms,
                              cat_residues, log) -> List[Dict]:
    """Pairwise minimum sidechain heavy-atom distance per cat-residue pair, ref vs final."""
    out = []
    n = len(cat_residues)
    for i in range(n):
        for j in range(i + 1, n):
            ci, ri_n, rri = cat_residues[i]
            cj, rj_n, rrj = cat_residues[j]
            ref_si = catres_sidechain_heavy_indices(ref_meta, ref_syms, ci, rri)
            ref_sj = catres_sidechain_heavy_indices(ref_meta, ref_syms, cj, rrj)
            fin_si = catres_sidechain_heavy_indices(fin_meta, fin_syms, ci, rri)
            fin_sj = catres_sidechain_heavy_indices(fin_meta, fin_syms, cj, rrj)
            d_ref = d_fin = None
            if ref_si and ref_sj:
                A = np.array([ref_pos[k] for k in ref_si])
                B = np.array([ref_pos[k] for k in ref_sj])
                d_ref = float(np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1).min()))
            if fin_si and fin_sj:
                A = np.array([fin_pos[k] for k in fin_si])
                B = np.array([fin_pos[k] for k in fin_sj])
                d_fin = float(np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1).min()))
            drift = abs(d_fin - d_ref) if (d_ref is not None and d_fin is not None) else None
            out.append({
                'pair': f"{ci}/{ri_n}{rri} - {cj}/{rj_n}{rrj}",
                'ref_min_sidechain_dist_A': d_ref,
                'final_min_sidechain_dist_A': d_fin,
                'drift_A': drift,
            })
    return out


# --------------------------------------------------------------------------
# Reactive-atom-triplet
# --------------------------------------------------------------------------
def parse_reactive_atoms(spec: str) -> List[Tuple[str, str, int, str]]:
    out = []
    for chunk in spec.split(','):
        parts = chunk.strip().split(':')
        if len(parts) != 4:
            raise ValueError(f"Bad --reactive-atoms entry {chunk!r}; expected CHAIN:RESN:RESI:NAME")
        c, rn, ri, nm = parts
        out.append((c, rn, int(ri), nm))
    if len(out) != 3:
        raise ValueError(f"--reactive-atoms must specify exactly 3 atoms (P, Onuc, Olg); got {len(out)}")
    return out


def reactive_triplet_geometry(meta, pos, triplet) -> Optional[Dict]:
    coords = []
    for c, rn, ri, nm in triplet:
        idx = None
        for i, m in enumerate(meta):
            if m['chain'] == c and m['resnum'] == ri and m['atom_name'] == nm and m['resname'] == rn:
                idx = i
                break
        if idx is None:
            return None
        coords.append(pos[idx])
    P, Onuc, Olg = coords
    return {
        'd_P_Onuc_A': float(np.linalg.norm(Onuc - P)),
        'd_P_Olg_A': float(np.linalg.norm(Olg - P)),
        'angle_Onuc_P_Olg_deg': angle(Onuc, P, Olg),
    }


# --------------------------------------------------------------------------
# Clash count
# --------------------------------------------------------------------------
def count_clashes(meta, pos, symbols, cutoff: float = 1.8) -> int:
    """Heavy-atom clash count between cross-residue, non-bonded heavy pairs."""
    heavy = [i for i, s in enumerate(symbols) if s != 'H']
    n = len(heavy)
    if n == 0:
        return 0
    coords = pos[heavy]
    keys = [(meta[i]['chain'], meta[i]['resnum']) for i in heavy]
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    cutoff2 = cutoff ** 2
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            if keys[i] == keys[j]:
                continue
            if d2[i, j] < cutoff2:
                cnt += 1
    return cnt


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    log = logging.getLogger('active_site_metrics')

    if not args.ref.exists():
        log.error(f"--ref {args.ref} does not exist"); return 1
    if not args.final.exists():
        log.error(f"--final {args.final} does not exist"); return 1
    out_dir = args.out or args.final.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    basename = args.final.stem

    log.info(f"REF:   {args.ref}")
    log.info(f"FINAL: {args.final}")
    ref_atoms, ref_meta = load_pdb_with_metadata(args.ref)
    fin_atoms, fin_meta = load_pdb_with_metadata(args.final)
    ref_syms = ref_atoms.get_chemical_symbols()
    fin_syms = fin_atoms.get_chemical_symbols()
    ref_pos = ref_atoms.get_positions()
    fin_pos = fin_atoms.get_positions()
    log.info(f"REF atoms: {len(ref_atoms)} | FINAL atoms: {len(fin_atoms)}")

    # Cat residues
    if args.cat_residues:
        cat_residues = parse_cat_residues_spec(args.cat_residues)
    else:
        cat_residues = parse_cat_residues_from_remark666(args.ref)
    if not cat_residues:
        log.error("No catalytic residues found (REMARK 666 missing in --ref, or --cat-residues empty)")
        return 1
    log.info(f"Cat residues ({len(cat_residues)}): {cat_residues}")

    # Ligand resname
    if args.ligand_resname:
        ligand_resname = args.ligand_resname
    else:
        # default: first HETATM resname in ref
        hetatm_names = [m['resname'] for m in ref_meta if m['is_hetatm']]
        if not hetatm_names:
            log.error("No HETATM in --ref; specify --ligand-resname")
            return 1
        ligand_resname = hetatm_names[0]
    if args.ligand_chain:
        ligand_chain = args.ligand_chain
    else:
        ligand_chain_candidates = sorted({m['chain'] for m in ref_meta
                                          if m['is_hetatm'] and m['resname'] == ligand_resname})
        if len(ligand_chain_candidates) > 1:
            log.warning(f"Ligand resname {ligand_resname} appears in {len(ligand_chain_candidates)} "
                        f"chains: {ligand_chain_candidates}. Using first; pass --ligand-chain to override.")
        ligand_chain = ligand_chain_candidates[0] if ligand_chain_candidates else None
    log.info(f"Ligand: {ligand_resname} chain {ligand_chain}")

    # ---- Kabsch alignment of FINAL onto REF using SHARED protein heavy atoms
    pairs = shared_protein_atoms(ref_meta, ref_syms, fin_meta, fin_syms)
    if len(pairs) < 4:
        log.error(f"Only {len(pairs)} shared protein heavy atoms — too few to Kabsch")
        return 1
    P_ref = np.array([ref_pos[i] for i, _ in pairs])
    P_fin = np.array([fin_pos[j] for _, j in pairs])
    R, t = kabsch(P_fin, P_ref)
    fin_pos_aln = apply_kabsch(fin_pos, R, t)
    pre_rmsd = rmsd(P_fin, P_ref)
    post_rmsd = rmsd(apply_kabsch(P_fin, R, t), P_ref)
    log.info(f"Kabsch on {len(pairs)} shared protein heavy atoms: "
             f"pre-align RMSD {pre_rmsd:.3f} Å → post-align {post_rmsd:.3f} Å")

    # ---- GLOBAL metrics (on shared heavy atoms, after Kabsch)
    ref_heavy = heavy_atom_indices(ref_meta, ref_syms, exclude_hetatm=True)
    fin_heavy = heavy_atom_indices(fin_meta, fin_syms, exclude_hetatm=True)
    # All-heavy whole-protein RMSD: shared atoms in heavy set
    ref_lookup_h = {(ref_meta[i]['chain'], ref_meta[i]['resnum'], ref_meta[i]['atom_name']): i for i in ref_heavy}
    fin_lookup_h = {(fin_meta[j]['chain'], fin_meta[j]['resnum'], fin_meta[j]['atom_name']): j for j in fin_heavy}
    shared_keys = set(ref_lookup_h) & set(fin_lookup_h)
    A = np.array([ref_pos[ref_lookup_h[k]] for k in shared_keys])
    B = np.array([fin_pos_aln[fin_lookup_h[k]] for k in shared_keys])
    global_heavy_rmsd = rmsd(A, B) if len(A) else float('nan')

    # Backbone-only RMSD
    ref_bb = backbone_indices(ref_meta)
    fin_bb = backbone_indices(fin_meta)
    ref_lookup_bb = {(ref_meta[i]['chain'], ref_meta[i]['resnum'], ref_meta[i]['atom_name']): i for i in ref_bb}
    fin_lookup_bb = {(fin_meta[j]['chain'], fin_meta[j]['resnum'], fin_meta[j]['atom_name']): j for j in fin_bb}
    shared_bb = set(ref_lookup_bb) & set(fin_lookup_bb)
    A = np.array([ref_pos[ref_lookup_bb[k]] for k in shared_bb])
    B = np.array([fin_pos_aln[fin_lookup_bb[k]] for k in shared_bb])
    backbone_rmsd = rmsd(A, B) if len(A) else float('nan')

    # Catres-only all-atom heavy RMSD
    catres_heavy_idx_ref = catres_all_atom_indices(ref_meta, ref_syms, cat_residues)
    catres_heavy_idx_fin = catres_all_atom_indices(fin_meta, fin_syms, cat_residues)
    ref_lookup_cat = {(ref_meta[i]['chain'], ref_meta[i]['resnum'], ref_meta[i]['atom_name']): i for i in catres_heavy_idx_ref}
    fin_lookup_cat = {(fin_meta[j]['chain'], fin_meta[j]['resnum'], fin_meta[j]['atom_name']): j for j in catres_heavy_idx_fin}
    shared_cat = set(ref_lookup_cat) & set(fin_lookup_cat)
    A = np.array([ref_pos[ref_lookup_cat[k]] for k in shared_cat])
    B = np.array([fin_pos_aln[fin_lookup_cat[k]] for k in shared_cat])
    catres_rmsd = rmsd(A, B) if len(A) else float('nan')

    log.info(f"Global heavy RMSD: {global_heavy_rmsd:.3f} Å ({len(shared_keys)} atoms)")
    log.info(f"Backbone RMSD:    {backbone_rmsd:.3f} Å ({len(shared_bb)} atoms)")
    log.info(f"Cat-res heavy RMSD: {catres_rmsd:.3f} Å ({len(shared_cat)} atoms)")

    # ---- Per-residue metrics
    per_res = per_residue_metrics(ref_meta, ref_pos, ref_syms,
                                  fin_meta, fin_pos_aln, fin_syms,
                                  cat_residues, log)
    # ---- Cat-residue ↔ ligand
    cat_lig = catres_to_ligand_geometry(ref_meta, ref_pos, ref_syms,
                                        fin_meta, fin_pos_aln, fin_syms,
                                        cat_residues, ligand_chain, ligand_resname,
                                        args.hbond_cutoff, log)
    # ---- Cat-residue ↔ cat-residue
    cat_cat = catres_to_catres_geometry(ref_meta, ref_pos, ref_syms,
                                        fin_meta, fin_pos_aln, fin_syms,
                                        cat_residues, log)
    # ---- Reactive triplet
    reactive_ref = reactive_fin = None
    triplet_spec = None
    if args.reactive_atoms:
        try:
            triplet_spec = parse_reactive_atoms(args.reactive_atoms)
            reactive_ref = reactive_triplet_geometry(ref_meta, ref_pos, triplet_spec)
            reactive_fin = reactive_triplet_geometry(fin_meta, fin_pos_aln, triplet_spec)
        except ValueError as e:
            log.warning(f"Reactive triplet parse error: {e}")

    # ---- Clash counts
    ref_clashes = count_clashes(ref_meta, ref_pos, ref_syms, cutoff=args.clash_cutoff)
    fin_clashes = count_clashes(fin_meta, fin_pos, fin_syms, cutoff=args.clash_cutoff)
    log.info(f"Clash count (heavy, cross-res, <{args.clash_cutoff} Å): "
             f"REF={ref_clashes}, FINAL={fin_clashes}")

    # ---- Flags
    flags = []
    if global_heavy_rmsd > args.global_rmsd_flag_A:
        flags.append(f"GLOBAL heavy RMSD {global_heavy_rmsd:.2f} Å > {args.global_rmsd_flag_A} — large drift")
    if catres_rmsd > args.catres_rmsd_flag_A:
        flags.append(f"CATRES heavy RMSD {catres_rmsd:.2f} Å > {args.catres_rmsd_flag_A} — catalytic geometry shifted")
    for r in per_res:
        if r['sidechain_heavy_rmsd_A'] is not None and not np.isnan(r['sidechain_heavy_rmsd_A']) \
                and r['sidechain_heavy_rmsd_A'] > args.drift_threshold:
            flags.append(f"{r['chain']}:{r['resname']}{r['resnum']} sidechain RMSD "
                         f"{r['sidechain_heavy_rmsd_A']:.2f} Å > {args.drift_threshold}")
    for r in cat_lig:
        if r['drift_A'] is not None and r['drift_A'] > args.drift_threshold:
            flags.append(f"{r['chain']}:{r['resname']}{r['resnum']} "
                         f"closest-atom drift {r['drift_A']:.2f} Å > {args.drift_threshold}")
        for h in r['hbonds']:
            if h['broken']:
                flags.append(f"{r['chain']}:{r['resname']}{r['resnum']}/{h['atom_name']} "
                             f"H-bond broken (ref {h['ref_dist_A']:.2f} → final {h['final_dist_A']:.2f} Å)")
    for p in cat_cat:
        if p['drift_A'] is not None and p['drift_A'] > args.drift_threshold:
            flags.append(f"Pair [{p['pair']}] min-sidechain drift "
                         f"{p['drift_A']:.2f} Å > {args.drift_threshold}")
    if reactive_ref and reactive_fin:
        for k in ('d_P_Onuc_A', 'd_P_Olg_A'):
            d = abs(reactive_fin[k] - reactive_ref[k])
            if d > args.reactive_distance_flag_A:
                flags.append(f"Reactive {k} drift {d:.3f} Å > {args.reactive_distance_flag_A}")
        a = abs(reactive_fin['angle_Onuc_P_Olg_deg'] - reactive_ref['angle_Onuc_P_Olg_deg'])
        if a > args.angle_drift_threshold:
            flags.append(f"Reactive Onuc-P-Olg angle drift {a:.1f}° > {args.angle_drift_threshold}°")
    if fin_clashes > ref_clashes + args.clash_growth_flag:
        flags.append(f"Clash count grew from REF={ref_clashes} to FINAL={fin_clashes} (>{args.clash_growth_flag})")

    # ---- Outputs
    summary = {
        'ref_pdb': str(args.ref),
        'final_pdb': str(args.final),
        'cat_residues': [{'chain': c, 'resname': r, 'resnum': n} for c, r, n in cat_residues],
        'ligand_resname': ligand_resname,
        'ligand_chain': ligand_chain,
        'kabsch': {'n_shared': len(pairs), 'pre_rmsd_A': pre_rmsd, 'post_rmsd_A': post_rmsd},
        'global': {
            'heavy_rmsd_A': global_heavy_rmsd,
            'backbone_rmsd_A': backbone_rmsd,
            'catres_heavy_rmsd_A': catres_rmsd,
            'n_shared_heavy': len(shared_keys),
            'n_shared_backbone': len(shared_bb),
            'n_shared_catres_heavy': len(shared_cat),
        },
        'per_residue': per_res,
        'cat_to_ligand': cat_lig,
        'cat_to_cat': cat_cat,
        'reactive': {'spec': args.reactive_atoms, 'ref': reactive_ref, 'final': reactive_fin},
        'clashes': {'ref': ref_clashes, 'final': fin_clashes, 'cutoff_A': args.clash_cutoff},
        'flags': flags,
        'thresholds': {
            'drift_threshold_A': args.drift_threshold,
            'angle_drift_threshold_deg': args.angle_drift_threshold,
            'hbond_cutoff_A': args.hbond_cutoff,
            'clash_cutoff_A': args.clash_cutoff,
        },
    }

    json_path = out_dir / f"{basename}_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Wrote {json_path}")

    csv_path = out_dir / f"{basename}_metrics.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['category', 'key', 'ref', 'final', 'drift', 'note'])
        w.writerow(['global', 'heavy_rmsd_A', '', f"{global_heavy_rmsd:.4f}", '', f"{len(shared_keys)} atoms"])
        w.writerow(['global', 'backbone_rmsd_A', '', f"{backbone_rmsd:.4f}", '', f"{len(shared_bb)} atoms"])
        w.writerow(['global', 'catres_heavy_rmsd_A', '', f"{catres_rmsd:.4f}", '', f"{len(shared_cat)} atoms"])
        for r in per_res:
            label = f"{r['chain']}/{r['resname']}{r['resnum']}"
            w.writerow(['per_residue', f"{label}/all_atom_heavy_rmsd_A", '',
                        f"{r['all_atom_heavy_rmsd_A']:.4f}" if not np.isnan(r['all_atom_heavy_rmsd_A']) else '',
                        '', f"n={r['n_all_atom']}"])
            w.writerow(['per_residue', f"{label}/sidechain_heavy_rmsd_A", '',
                        f"{r['sidechain_heavy_rmsd_A']:.4f}" if not np.isnan(r['sidechain_heavy_rmsd_A']) else '',
                        '', f"n={r['n_sidechain']}"])
            for k, err in enumerate(r['chi_errors_deg']):
                w.writerow(['per_residue', f"{label}/chi{k+1}_err_deg", '',
                            f"{err:.2f}" if err is not None else '', '', ''])
        for r in cat_lig:
            label = f"{r['chain']}/{r['resname']}{r['resnum']}"
            w.writerow(['cat_to_ligand', f"{label}/closest_dist_A",
                        f"{r['ref_min_dist_A']:.4f}" if r['ref_min_dist_A'] is not None else '',
                        f"{r['final_min_dist_A']:.4f}" if r['final_min_dist_A'] is not None else '',
                        f"{r['drift_A']:.4f}" if r['drift_A'] is not None else '',
                        f"closest atom: {r['closest_atom_name']}"])
            for h in r['hbonds']:
                w.writerow(['cat_to_ligand_hbond', f"{label}/{h['atom_name']}",
                            f"{h['ref_dist_A']:.4f}", f"{h['final_dist_A']:.4f}",
                            f"{abs(h['final_dist_A'] - h['ref_dist_A']):.4f}",
                            f"{h['role']}{', BROKEN' if h['broken'] else ''}"])
        for p in cat_cat:
            w.writerow(['cat_to_cat', p['pair'],
                        f"{p['ref_min_sidechain_dist_A']:.4f}" if p['ref_min_sidechain_dist_A'] is not None else '',
                        f"{p['final_min_sidechain_dist_A']:.4f}" if p['final_min_sidechain_dist_A'] is not None else '',
                        f"{p['drift_A']:.4f}" if p['drift_A'] is not None else '', ''])
        if reactive_ref and reactive_fin:
            for k in ('d_P_Onuc_A', 'd_P_Olg_A', 'angle_Onuc_P_Olg_deg'):
                w.writerow(['reactive', k,
                            f"{reactive_ref[k]:.4f}", f"{reactive_fin[k]:.4f}",
                            f"{abs(reactive_fin[k] - reactive_ref[k]):.4f}", ''])
        w.writerow(['clashes', f"count_<{args.clash_cutoff}A", str(ref_clashes), str(fin_clashes),
                    str(fin_clashes - ref_clashes), 'heavy, cross-residue'])
    log.info(f"Wrote {csv_path}")

    md_path = out_dir / f"{basename}_metrics_REPORT.md"
    with open(md_path, 'w') as f:
        f.write(f"# Active-site metrics report\n\n")
        f.write(f"- **Reference:** `{args.ref}`\n")
        f.write(f"- **Final:**     `{args.final}`\n\n")
        f.write(f"## Global RMSDs (after Kabsch on {len(pairs)} shared protein heavy atoms)\n\n")
        f.write(f"| Metric | RMSD (Å) | N atoms |\n|---|---|---|\n")
        f.write(f"| Heavy whole-protein | {global_heavy_rmsd:.3f} | {len(shared_keys)} |\n")
        f.write(f"| Backbone only | {backbone_rmsd:.3f} | {len(shared_bb)} |\n")
        f.write(f"| Catres all-heavy | {catres_rmsd:.3f} | {len(shared_cat)} |\n\n")
        f.write(f"## Per-cat-residue\n\n")
        f.write(f"| Residue | All-heavy RMSD (Å) | Sidechain RMSD (Å) | Chi errors (deg) |\n|---|---|---|---|\n")
        for r in per_res:
            chi_str = ', '.join(f"{e:.1f}" if e is not None else 'n/a' for e in r['chi_errors_deg'])
            f.write(f"| {r['chain']}/{r['resname']}{r['resnum']} | "
                    f"{r['all_atom_heavy_rmsd_A']:.3f} | {r['sidechain_heavy_rmsd_A']:.3f} | {chi_str or '—'} |\n")
        f.write(f"\n## Cat-residue ↔ ligand\n\n")
        f.write(f"| Residue | Closest atom | Ref dist (Å) | Final dist (Å) | Drift (Å) |\n|---|---|---|---|---|\n")
        for r in cat_lig:
            rd = f"{r['ref_min_dist_A']:.3f}" if r['ref_min_dist_A'] is not None else '—'
            fd = f"{r['final_min_dist_A']:.3f}" if r['final_min_dist_A'] is not None else '—'
            dr = f"{r['drift_A']:.3f}" if r['drift_A'] is not None else '—'
            f.write(f"| {r['chain']}/{r['resname']}{r['resnum']} | {r['closest_atom_name']} | {rd} | {fd} | {dr} |\n")
        any_hb = any(r['hbonds'] for r in cat_lig)
        if any_hb:
            f.write(f"\n### Donor/acceptor H-bonds to ligand (heavy-atom distance only)\n\n")
            f.write(f"| Residue | Atom | Role | Ref (Å) | Final (Å) | Broken |\n|---|---|---|---|---|---|\n")
            for r in cat_lig:
                for h in r['hbonds']:
                    f.write(f"| {r['chain']}/{r['resname']}{r['resnum']} | {h['atom_name']} | "
                            f"{h['role']} | {h['ref_dist_A']:.3f} | {h['final_dist_A']:.3f} | "
                            f"{'YES' if h['broken'] else 'no'} |\n")
        f.write(f"\n## Cat-residue ↔ cat-residue\n\n")
        f.write(f"| Pair | Ref min sidechain dist (Å) | Final (Å) | Drift (Å) |\n|---|---|---|---|\n")
        for p in cat_cat:
            rd = f"{p['ref_min_sidechain_dist_A']:.3f}" if p['ref_min_sidechain_dist_A'] is not None else '—'
            fd = f"{p['final_min_sidechain_dist_A']:.3f}" if p['final_min_sidechain_dist_A'] is not None else '—'
            dr = f"{p['drift_A']:.3f}" if p['drift_A'] is not None else '—'
            f.write(f"| {p['pair']} | {rd} | {fd} | {dr} |\n")
        if reactive_ref and reactive_fin:
            f.write(f"\n## Reactive-atom triplet\n\n")
            f.write(f"| Quantity | Ref | Final | Drift |\n|---|---|---|---|\n")
            for k in ('d_P_Onuc_A', 'd_P_Olg_A', 'angle_Onuc_P_Olg_deg'):
                f.write(f"| {k} | {reactive_ref[k]:.3f} | {reactive_fin[k]:.3f} | "
                        f"{abs(reactive_fin[k] - reactive_ref[k]):.3f} |\n")
        f.write(f"\n## Clashes (heavy, cross-residue, <{args.clash_cutoff} Å)\n\n")
        f.write(f"REF: {ref_clashes} | FINAL: {fin_clashes} | Δ {fin_clashes - ref_clashes}\n\n")
        f.write(f"## Flags\n\n")
        if not flags:
            f.write("None.\n")
        else:
            for fl in flags:
                f.write(f"- {fl}\n")
    log.info(f"Wrote {md_path}")
    log.info(f"Flags: {len(flags)}")
    for fl in flags:
        log.info(f"  ⚠ {fl}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
