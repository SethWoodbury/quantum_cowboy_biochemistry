"""protonation.py — atom-level H rewriting for residue protonation states.

Why this exists
---------------
``microstate_sampler.py`` historically emitted ``ledger-only`` deltas for the
HIS/ASP/GLU/LYS/CYS protonation generators: the variant PDB had identical
atomic coordinates as the input, only the charge bookkeeping changed. That is
silently wrong — the QM/xTB calculator sees the SAME number of H atoms in
each variant, so a "HID" and a "HIE" variant carrying the same coordinates
are identical SCF problems and cannot rank differently.

This module fixes that by rewriting H atoms at the residue level:

    * HIS:  HID (only HD1 on Nδ), HIE (only HE2 on Nε), HIP (both, +1 charge).
    * GLU/ASP: GLU0 / ASP0 (neutral COOH, place HE2 / HD2 on the protonated
      carboxylate oxygen); GLU- / ASP- (anionic, no carboxylate H).
    * LYS:  LYS+ (NH3+ — three Hs on Nζ); LYS0 (NH2 — two Hs on Nζ).
    * CYS:  CYS-SH (HG on Sγ); CYS-S- (no HG).

H placement uses idealized geometry (1.0 Å N–H, 0.96 Å O–H, 1.34 Å S–H bond
lengths from Allen et al. CSD bond-length surveys, J. Chem. Soc. Perkin
Trans. 2 1987 S1, and Cordero et al. (2008) covalent radii compendium —
single-bond medians). The H is placed along a residue-specific lone-pair
direction computed from the nearest heavy atoms (e.g. for HIS Nδ, in the
plane of the imidazole ring opposite to the bisector of CG–Nδ–CE1).

The implementation is reaction-agnostic and discovers the residues from the
input PDB. No PTE-specific defaults.

Public API
----------
``apply_protonation(atoms, bt_struct, residue_state, ...)`` — return
``(new_atoms, new_bt_struct, charge_delta_int)`` for ONE residue, ONE state
transition. The caller composes multiple calls for multi-residue products.

``apply_residue_states(atoms, bt_struct, residue_states_dict, ...)`` —
apply the whole product as a single transformation (returns combined
result with summed charge delta + ledger delta dict).

CLI
---
Most users never call this module directly; instead they drive it through
``tools/microstate_sampler.py --auto-protonation`` or
``--protonation-rules rules.yaml``.

Geometry rule references
------------------------
* N–H bond length (~1.01 Å in primary amines, ~1.00 Å in protonated
  imidazole, ~1.03 Å in protonated amines): Allen et al., J. Chem. Soc.
  Perkin Trans. 2 1987 S1; Cordero et al. Dalton Trans. 2008, 2832.
  Default: 1.01 Å.
* O–H bond length (~0.96 Å in alcohols, hydroxyls): Allen et al. 1987 S1;
  consistent with crystallographic neutron diffraction. Default: 0.96 Å.
* S–H bond length (~1.34 Å in thiols): Allen et al. 1987 S1. Default: 1.34 Å.
* H–N–H angle in protonated NH3+ ≈ 109.5° (sp3 tetrahedral).
* H–N–C angle in HID/HIE imidazole ≈ 126° (sp2 trigonal, derived from the
  ring planarity).
* H–O–C angle in COOH carboxylic acid ≈ 110° (close to sp3, but COOH oxygen
  is sp2-leaning). We use 109.5° as a defensible default.

Each placement target distance is exposed as a CLI argument so users can
override (e.g. for force-field-specific geometries that need 1.04 Å).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

log = logging.getLogger("quantum_engine.ops.protonation")


# ----------------------------------------------------------------------------
# Bond-length defaults (Å). Cited references in module docstring.
# ----------------------------------------------------------------------------
BOND_LENGTH_NH_DEFAULT: float = 1.01      # Allen 1987 S1
BOND_LENGTH_OH_DEFAULT: float = 0.96      # Allen 1987 S1, neutron-derived
BOND_LENGTH_SH_DEFAULT: float = 1.34      # Allen 1987 S1
TETRAHEDRAL_ANGLE_DEG: float = 109.4712   # arccos(-1/3) = 109.4712°


# ----------------------------------------------------------------------------
# Residue-state vocabulary (canonical labels)
# ----------------------------------------------------------------------------
# HIS family:
#   HID — neutral, proton on Nδ (HD1)
#   HIE — neutral, proton on Nε (HE2)
#   HIP — protonated, both HD1 and HE2 (charge +1)
HIS_STATES: tuple[str, ...] = ("HID", "HIE", "HIP")

# ASP family:
#   ASP-  — anionic, no proton on Oδ1/Oδ2 (default, charge -1)
#   ASP0  — neutral COOH, proton on Oδ2 (HD2)
ASP_STATES: tuple[str, ...] = ("ASP-", "ASP0")

# GLU family:
#   GLU-  — anionic, no proton on Oε1/Oε2 (default, charge -1)
#   GLU0  — neutral COOH, proton on Oε2 (HE2)
GLU_STATES: tuple[str, ...] = ("GLU-", "GLU0")

# LYS family:
#   LYS+  — NH3+, three Hs on Nζ (HZ1, HZ2, HZ3) (charge +1, default at pH7)
#   LYS0  — NH2, two Hs on Nζ (HZ1, HZ2)
LYS_STATES: tuple[str, ...] = ("LYS+", "LYS0")

# CYS family:
#   CYS-SH  — protonated thiol, HG on Sγ
#   CYS-S-  — thiolate, no HG (charge -1)
CYS_STATES: tuple[str, ...] = ("CYS-SH", "CYS-S-")


# Total ledger charge for each state (vs reference protonation state).
# Reference: HIS (HIE) neutral, ASP-/GLU- anionic, LYS+ cationic, CYS-SH
# neutral. The delta is computed as
#   ledger_charge_for(state) - ledger_charge_for(reference)
STATE_CHARGE: dict[str, int] = {
    "HID": 0,
    "HIE": 0,
    "HIP": +1,
    "ASP-": -1,
    "ASP0": 0,
    "GLU-": -1,
    "GLU0": 0,
    "LYS+": +1,
    "LYS0": 0,
    "CYS-SH": 0,
    "CYS-S-": -1,
}


# Residue-name family lookup (input residue name -> family / states tuple).
RESIDUE_FAMILY: dict[str, tuple[str, tuple[str, ...]]] = {
    "HIS": ("HIS", HIS_STATES),
    "HIE": ("HIS", HIS_STATES),
    "HID": ("HIS", HIS_STATES),
    "HIP": ("HIS", HIS_STATES),
    "ASP": ("ASP", ASP_STATES),
    "ASH": ("ASP", ASP_STATES),  # AMBER neutral-asp name
    "GLU": ("GLU", GLU_STATES),
    "GLH": ("GLU", GLU_STATES),  # AMBER neutral-glu name
    "LYS": ("LYS", LYS_STATES),
    "LYN": ("LYS", LYS_STATES),  # AMBER neutral-lys name
    "CYS": ("CYS", CYS_STATES),
    "CYM": ("CYS", CYS_STATES),  # AMBER thiolate-cys name
}


# Map from STATE label -> RESNAME the variant should carry (for downstream
# tools that key on residue name).
STATE_TO_RESNAME: dict[str, str] = {
    "HID": "HID",
    "HIE": "HIE",
    "HIP": "HIP",
    "ASP-": "ASP",
    "ASP0": "ASH",
    "GLU-": "GLU",
    "GLU0": "GLH",
    "LYS+": "LYS",
    "LYS0": "LYN",
    "CYS-SH": "CYS",
    "CYS-S-": "CYM",
}


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("cannot normalize zero-length vector")
    return v / n


def _place_along_lone_pair(
    heavy_pos: np.ndarray,
    neighbor_positions: Sequence[np.ndarray],
    bond_length: float,
) -> np.ndarray:
    """Place H along the bisector of the angle opposite to the heavy atom's
    bonded neighbors (a defensible lone-pair direction for sp2/sp3 atoms).

    For an sp2 imidazole nitrogen with two ring neighbors, this points the H
    out of the ring plane along the in-plane lone pair. For carboxylate
    oxygen with one C bonded neighbor (and a lone-pair direction we derive
    from the two oxygens of the carboxylate), the caller should pass the
    appropriate neighbor list.

    Geometry: H position = heavy_pos - bond_length * normalized(mean(neighbors - heavy_pos)).

    The minus sign means we point AWAY from the bonded neighbors, into the
    lone-pair direction. For one neighbor this gives a perfect 180°; for two
    coplanar neighbors a perfect bisector.
    """
    if not neighbor_positions:
        raise ValueError("need at least one neighbor for lone-pair placement")
    direction = np.zeros(3)
    for n in neighbor_positions:
        direction += _normalize(n - heavy_pos)
    direction /= len(neighbor_positions)
    direction = _normalize(direction)
    return heavy_pos - bond_length * direction


def _place_carboxyl_h(
    ox2_pos: np.ndarray,
    cx_pos: np.ndarray,
    ox1_pos: np.ndarray,
    bond_length: float,
    angle_deg: float = 109.5,
) -> np.ndarray:
    """Place a carboxylate H on ``ox2_pos`` (the to-be-protonated O) at the
    correct C-O-H ≈ 109.5° angle, in the carboxylate plane (CX, OX1, OX2),
    on the side AWAY from ``ox1_pos`` (the carbonyl C=O partner).

    The earlier implementation called ``_place_along_lone_pair(ox2,[cx,ox1])``
    which (since OX1 is NOT bonded to OX2 and lies ~120° off the C-O bond
    in the carboxylate plane) produced a near-linear ~165° C-O-H geometry
    instead of the documented sp3-like 109.5°. This routine fixes that by
    explicitly constructing the in-plane perpendicular and rotating.
    """
    oc_unit = _normalize(cx_pos - ox2_pos)
    plane_normal = np.cross(cx_pos - ox2_pos, ox1_pos - ox2_pos)
    plane_norm = float(np.linalg.norm(plane_normal))
    if plane_norm < 1e-9:
        # Degenerate (collinear OX1/OX2/CX) — fall back to lone-pair direction.
        return ox2_pos - bond_length * oc_unit
    plane_normal = plane_normal / plane_norm
    side = _normalize(np.cross(plane_normal, oc_unit))
    # Pick side AWAY from OX1
    if float(np.dot(side, ox1_pos - ox2_pos)) > 0.0:
        side = -side
    ang = float(np.deg2rad(angle_deg))
    h_dir = _normalize(np.cos(ang) * oc_unit + np.sin(ang) * side)
    return ox2_pos + bond_length * h_dir


def _place_tetrahedral_third_h(
    n_pos: np.ndarray,
    h_existing: Sequence[np.ndarray],
    c_pos: np.ndarray,
    bond_length: float,
) -> np.ndarray:
    """Given an sp3 nitrogen with 1 C neighbor and 2 existing H atoms, place
    the third H to complete a tetrahedral geometry (used for LYS+ from LYS0).

    We compute the vector that, summed with c_pos - n_pos and the two
    existing N–H bond vectors, gives zero (i.e. balanced tetrahedron).
    Then renormalize to ``bond_length``.
    """
    if len(h_existing) != 2:
        raise ValueError("expected exactly 2 existing H atoms")
    v_sum = (c_pos - n_pos) + (h_existing[0] - n_pos) + (h_existing[1] - n_pos)
    direction = -_normalize(v_sum)
    return n_pos + bond_length * direction


# ----------------------------------------------------------------------------
# Helpers for biotite AtomArray manipulation
# ----------------------------------------------------------------------------
def _residue_atom_indices(
    bt_struct, chain: str, res_id: int,
) -> list[int]:
    """Return atom indices belonging to (chain, res_id) in bt_struct."""
    chains = np.asarray(bt_struct.chain_id)
    rids = np.asarray(bt_struct.res_id)
    return list(np.where((chains == chain) & (rids == int(res_id)))[0])


def _find_atom_by_name(
    bt_struct, chain: str, res_id: int, atom_names: Sequence[str],
) -> int | None:
    """Return atom idx for the first matching atom name in (chain, res_id)
    or ``None`` if none of the names exist."""
    idxs = _residue_atom_indices(bt_struct, chain, res_id)
    for i in idxs:
        name = str(bt_struct.atom_name[i]).strip().upper()
        for an in atom_names:
            if name == an.upper():
                return i
    return None


def _delete_atoms(atoms, bt_struct, indices_to_remove: Iterable[int]):
    """Return new (atoms, bt_struct) with the given indices removed."""
    if not indices_to_remove:
        return atoms, bt_struct
    keep_mask = np.ones(len(atoms), dtype=bool)
    for i in indices_to_remove:
        keep_mask[i] = False
    new_atoms = atoms[keep_mask]
    new_bt = bt_struct[keep_mask]
    return new_atoms, new_bt


def _append_h_atom(
    atoms, bt_struct,
    *,
    position: np.ndarray,
    atom_name: str,
    res_name: str,
    res_id: int,
    chain_id: str,
):
    """Return new (atoms, bt_struct) with an H atom appended."""
    import biotite.structure as struc
    from ase import Atoms

    new_h = struc.array([struc.Atom(
        coord=np.asarray(position, dtype=np.float32),
        atom_name=atom_name,
        res_name=res_name,
        res_id=int(res_id),
        chain_id=chain_id,
        element="H",
        hetero=False,
    )])
    new_bt = bt_struct + new_h

    # Append a corresponding ASE atom
    new_pos = np.vstack([atoms.get_positions(), position[np.newaxis, :]])
    new_syms = list(atoms.get_chemical_symbols()) + ["H"]
    new_atoms = Atoms(symbols=new_syms, positions=new_pos)
    # Preserve atoms.info passthrough
    new_atoms.info.update(atoms.info)
    return new_atoms, new_bt


def _rename_residue(bt_struct, chain: str, res_id: int, new_resname: str):
    """In-place rename residue (chain, res_id) to new_resname in bt_struct.

    Returns a copy with the rename applied.
    """
    out = bt_struct.copy()
    chains = np.asarray(out.chain_id)
    rids = np.asarray(out.res_id)
    mask = (chains == chain) & (rids == int(res_id))
    out.res_name[mask] = new_resname
    return out


# ----------------------------------------------------------------------------
# Per-family transformation routines
# ----------------------------------------------------------------------------
@dataclass
class _ResidueAtomInfo:
    """Helper container for the heavy-atom indices of a target residue."""
    chain: str
    res_id: int
    res_name: str
    indices_by_name: dict[str, int]

    def pos(self, atoms, name: str) -> np.ndarray:
        return atoms.get_positions()[self.indices_by_name[name]]


def _gather_residue_info(
    atoms, bt_struct, chain: str, res_id: int,
) -> _ResidueAtomInfo:
    idxs = _residue_atom_indices(bt_struct, chain, res_id)
    if not idxs:
        raise ValueError(f"no atoms for {chain}:{res_id} in input")
    indices_by_name: dict[str, int] = {}
    res_name = None
    for i in idxs:
        name = str(bt_struct.atom_name[i]).strip().upper()
        indices_by_name[name] = i
        if res_name is None:
            res_name = str(bt_struct.res_name[i]).strip().upper()
    return _ResidueAtomInfo(
        chain=chain, res_id=int(res_id),
        res_name=res_name or "",
        indices_by_name=indices_by_name,
    )


def _strip_residue_protons(
    atoms, bt_struct, info: _ResidueAtomInfo, h_names: Sequence[str],
):
    """Remove any H atoms with name in h_names from the residue."""
    targets: list[int] = []
    for name in h_names:
        if name.upper() in info.indices_by_name:
            targets.append(info.indices_by_name[name.upper()])
    return _delete_atoms(atoms, bt_struct, targets)


def _apply_his(
    atoms, bt_struct,
    *, chain: str, res_id: int, target_state: str,
    nh_bond_length: float,
):
    """Rewrite H atoms on a histidine to reach target_state ∈ {HID,HIE,HIP}."""
    if target_state not in HIS_STATES:
        raise ValueError(f"target_state for HIS must be one of {HIS_STATES}, got {target_state}")
    info = _gather_residue_info(atoms, bt_struct, chain, res_id)
    # Strip both ND1-H and NE2-H if present
    out_atoms, out_bt = _strip_residue_protons(
        atoms, bt_struct, info, ("HD1", "HE2"),
    )
    # Re-gather after stripping
    info = _gather_residue_info(out_atoms, out_bt, chain, res_id)

    if "ND1" not in info.indices_by_name or "NE2" not in info.indices_by_name:
        raise ValueError(
            f"HIS {chain}:{res_id} missing ND1/NE2 — cannot place protons"
        )
    nd1 = info.pos(out_atoms, "ND1")
    ne2 = info.pos(out_atoms, "NE2")
    # Ring neighbors of ND1: CG and CE1; of NE2: CD2 and CE1.
    cg = info.pos(out_atoms, "CG") if "CG" in info.indices_by_name else None
    cd2 = info.pos(out_atoms, "CD2") if "CD2" in info.indices_by_name else None
    ce1 = info.pos(out_atoms, "CE1") if "CE1" in info.indices_by_name else None

    if target_state in ("HID", "HIP"):
        # Place HD1 on ND1 along the in-plane lone-pair direction
        if cg is None or ce1 is None:
            raise ValueError(
                f"HIS {chain}:{res_id} missing CG/CE1 — cannot place HD1"
            )
        h_pos = _place_along_lone_pair(nd1, [cg, ce1], nh_bond_length)
        out_atoms, out_bt = _append_h_atom(
            out_atoms, out_bt,
            position=h_pos, atom_name="HD1",
            res_name=STATE_TO_RESNAME[target_state],
            res_id=res_id, chain_id=chain,
        )
    if target_state in ("HIE", "HIP"):
        if cd2 is None or ce1 is None:
            raise ValueError(
                f"HIS {chain}:{res_id} missing CD2/CE1 — cannot place HE2"
            )
        h_pos = _place_along_lone_pair(ne2, [cd2, ce1], nh_bond_length)
        out_atoms, out_bt = _append_h_atom(
            out_atoms, out_bt,
            position=h_pos, atom_name="HE2",
            res_name=STATE_TO_RESNAME[target_state],
            res_id=res_id, chain_id=chain,
        )

    # Rename whole residue to target state's RESNAME
    out_bt = _rename_residue(out_bt, chain, res_id, STATE_TO_RESNAME[target_state])
    return out_atoms, out_bt


def _apply_carboxylate(
    atoms, bt_struct,
    *, chain: str, res_id: int, target_state: str,
    family: str,                   # "ASP" or "GLU"
    oh_bond_length: float,
):
    """Rewrite carboxylate H for ASP/GLU (target_state ∈ ASP_STATES / GLU_STATES)."""
    info = _gather_residue_info(atoms, bt_struct, chain, res_id)
    if family == "ASP":
        cx_name, ox1_name, ox2_name = "CG", "OD1", "OD2"
        cb_name = "CB"
        h_atom_name = "HD2"
        all_h_names = ("HD2", "HD1")
    else:
        cx_name, ox1_name, ox2_name = "CD", "OE1", "OE2"
        cb_name = "CG"
        h_atom_name = "HE2"
        all_h_names = ("HE2", "HE1")

    out_atoms, out_bt = _strip_residue_protons(atoms, bt_struct, info, all_h_names)
    info = _gather_residue_info(out_atoms, out_bt, chain, res_id)

    # Carboxylate validation: BOTH the protonated and deprotonated paths
    # require CX/OX1/OX2 to exist; otherwise we silently rename a malformed
    # residue. Validate up-front (codex flag, 2026-05-07).
    for needed in (cx_name, ox1_name, ox2_name):
        if needed not in info.indices_by_name:
            raise ValueError(
                f"{family} {chain}:{res_id} missing {needed} — cannot rewrite "
                f"to {target_state}"
            )

    if target_state.endswith("0"):
        # Need to place H on Oε2 (or Oδ2). Use the explicit carboxyl-plane
        # placement (109.5° C-O-H, anti to OX1) — see _place_carboxyl_h
        # docstring. Older implementations used _place_along_lone_pair with
        # both [CX, OX1] which produced a near-linear ~165° geometry.
        cx = info.pos(out_atoms, cx_name)
        ox2 = info.pos(out_atoms, ox2_name)
        ox1 = info.pos(out_atoms, ox1_name)
        h_pos = _place_carboxyl_h(ox2, cx, ox1, oh_bond_length)
        out_atoms, out_bt = _append_h_atom(
            out_atoms, out_bt,
            position=h_pos, atom_name=h_atom_name,
            res_name=STATE_TO_RESNAME[target_state],
            res_id=res_id, chain_id=chain,
        )

    out_bt = _rename_residue(out_bt, chain, res_id, STATE_TO_RESNAME[target_state])
    return out_atoms, out_bt


def _apply_lys(
    atoms, bt_struct,
    *, chain: str, res_id: int, target_state: str,
    nh_bond_length: float,
):
    """Rewrite Nζ Hs for LYS (LYS+ → 3 Hs, LYS0 → 2 Hs)."""
    info = _gather_residue_info(atoms, bt_struct, chain, res_id)
    out_atoms, out_bt = _strip_residue_protons(
        atoms, bt_struct, info, ("HZ1", "HZ2", "HZ3"),
    )
    info = _gather_residue_info(out_atoms, out_bt, chain, res_id)
    if "NZ" not in info.indices_by_name or "CE" not in info.indices_by_name:
        raise ValueError(
            f"LYS {chain}:{res_id} missing NZ/CE — cannot place lysine protons"
        )
    nz = info.pos(out_atoms, "NZ")
    ce = info.pos(out_atoms, "CE")
    # Place the first 2 Hs symmetrically off the lone-pair direction (the
    # one opposite NZ–CE), in any plane perpendicular to NZ-CE.
    bond_dir = _normalize(nz - ce)
    # Build orthonormal basis (e1, e2) perpendicular to bond_dir
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, bond_dir)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = _normalize(seed - np.dot(seed, bond_dir) * bond_dir)
    e2 = np.cross(bond_dir, e1)

    # Tetrahedral angle from bond_dir: cos(theta) = -1/3 → theta = 109.4712°.
    # H direction = cos(109.4712 - 90)*bond_dir + sin(...)*(e1 or e2).
    # Equivalently for tetrahedral NH3 with one N–C bond, the three N–H bonds
    # sit at theta = 109.4712° from the N–C vector. The H position is
    # n_pos + bond_length * (sin(theta - 90°) * bond_dir + cos(...) * e_perp)
    # which simplifies to:
    cos_t = -1.0 / 3.0  # cos(tetrahedral)
    sin_t = float(np.sqrt(1.0 - cos_t**2))
    # H sits at bond_length * (cos_t * bond_dir + sin_t * e_perp)
    # where bond_dir points FROM CE-to-NZ, but we want the H pointing AWAY
    # from CE, so we use -cos_t = 1/3 along bond_dir? Equivalently, since
    # N–C and N–H both point AWAY from N, and the angle between them is
    # tetrahedral 109.5°, we have:
    #   H_offset = bond_length * (-(-1/3) * bond_dir + sin_t * e_perp)
    # Simpler: place H at three positions evenly spaced 120° around the
    # axis (CE → NZ), each tilted from the axis by (180 - 109.47) ≈ 70.53°.
    # tilt_axial component along bond_dir is (cos(tilt)) * bond_length
    # = cos(70.53°) * bond_length = 1/3 * bond_length.
    axial = (1.0 / 3.0) * nh_bond_length
    radial = sin_t * nh_bond_length

    h_positions: list[np.ndarray] = []
    for idx in range(3 if target_state == "LYS+" else 2):
        phi = idx * (2 * np.pi / 3)
        h_perp = radial * (np.cos(phi) * e1 + np.sin(phi) * e2)
        h_pos = nz + axial * bond_dir + h_perp
        h_positions.append(h_pos)

    h_names = ["HZ1", "HZ2", "HZ3"][:len(h_positions)]
    for hp, hn in zip(h_positions, h_names):
        out_atoms, out_bt = _append_h_atom(
            out_atoms, out_bt,
            position=hp, atom_name=hn,
            res_name=STATE_TO_RESNAME[target_state],
            res_id=res_id, chain_id=chain,
        )

    out_bt = _rename_residue(out_bt, chain, res_id, STATE_TO_RESNAME[target_state])
    return out_atoms, out_bt


def _apply_cys(
    atoms, bt_struct,
    *, chain: str, res_id: int, target_state: str,
    sh_bond_length: float,
):
    """Rewrite Sγ-H for CYS (CYS-SH → with HG, CYS-S- → without)."""
    info = _gather_residue_info(atoms, bt_struct, chain, res_id)
    # Validate sidechain heavy atoms BEFORE any rename/strip — symmetric to
    # the carboxylate validation above (codex flag, 2026-05-07).
    if "SG" not in info.indices_by_name or "CB" not in info.indices_by_name:
        raise ValueError(
            f"CYS {chain}:{res_id} missing SG/CB — cannot rewrite to {target_state}"
        )
    out_atoms, out_bt = _strip_residue_protons(atoms, bt_struct, info, ("HG",))
    info = _gather_residue_info(out_atoms, out_bt, chain, res_id)

    if target_state == "CYS-SH":
        sg = info.pos(out_atoms, "SG")
        cb = info.pos(out_atoms, "CB")
        # H along lone pair: opposite the C-S bond
        h_pos = _place_along_lone_pair(sg, [cb], sh_bond_length)
        out_atoms, out_bt = _append_h_atom(
            out_atoms, out_bt,
            position=h_pos, atom_name="HG",
            res_name=STATE_TO_RESNAME[target_state],
            res_id=res_id, chain_id=chain,
        )

    out_bt = _rename_residue(out_bt, chain, res_id, STATE_TO_RESNAME[target_state])
    return out_atoms, out_bt


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def apply_protonation(
    atoms, bt_struct,
    *,
    chain: str, res_id: int, target_state: str,
    nh_bond_length: float = BOND_LENGTH_NH_DEFAULT,
    oh_bond_length: float = BOND_LENGTH_OH_DEFAULT,
    sh_bond_length: float = BOND_LENGTH_SH_DEFAULT,
):
    """Rewrite atoms / bt_struct so that residue (chain, res_id) is in target_state.

    Args:
        atoms: ASE Atoms.
        bt_struct: biotite AtomArray (must have atom_name annotation).
        chain: chain ID of target residue.
        res_id: residue ID (PDB number).
        target_state: one of HIS_STATES ∪ ASP_STATES ∪ GLU_STATES ∪ LYS_STATES
            ∪ CYS_STATES.
        nh_bond_length, oh_bond_length, sh_bond_length: bond-length defaults
            for the H placements (Å).

    Returns:
        ``(new_atoms, new_bt_struct, charge_delta)`` — charge_delta is the
        signed change in this residue's contribution to the total cluster
        charge (e.g. HIS HID → HIP yields +1; GLU GLU- → GLU0 yields +1;
        ASP ASP0 → ASP- yields -1).

    Notes:
        - Reaction-agnostic: residue is identified ONLY by (chain, res_id);
          residue name in the input is used to determine which family of
          states is valid.
        - All H-rewrites preserve the rest of the structure — no atom outside
          the target residue is moved.
        - If the input residue already has the requested state's H pattern,
          this is still safe — the routine is fail-soft and idempotent.
    """
    info = _gather_residue_info(atoms, bt_struct, chain, res_id)
    src_resname = info.res_name
    fam_lookup = RESIDUE_FAMILY.get(src_resname)
    if fam_lookup is None:
        raise ValueError(
            f"Residue {chain}:{res_id} resname={src_resname!r} not in "
            f"protonation family vocabulary {list(RESIDUE_FAMILY)}"
        )
    family, valid_states = fam_lookup
    if target_state not in valid_states:
        raise ValueError(
            f"target_state={target_state!r} not valid for "
            f"residue {src_resname} (family {family}); valid: {valid_states}"
        )

    # Compute charge delta vs the SOURCE residue's implied state.
    src_state = _resname_to_state(src_resname)
    charge_delta = STATE_CHARGE[target_state] - STATE_CHARGE[src_state]

    if family == "HIS":
        out_atoms, out_bt = _apply_his(
            atoms, bt_struct,
            chain=chain, res_id=res_id, target_state=target_state,
            nh_bond_length=nh_bond_length,
        )
    elif family == "ASP":
        out_atoms, out_bt = _apply_carboxylate(
            atoms, bt_struct,
            chain=chain, res_id=res_id, target_state=target_state,
            family="ASP", oh_bond_length=oh_bond_length,
        )
    elif family == "GLU":
        out_atoms, out_bt = _apply_carboxylate(
            atoms, bt_struct,
            chain=chain, res_id=res_id, target_state=target_state,
            family="GLU", oh_bond_length=oh_bond_length,
        )
    elif family == "LYS":
        out_atoms, out_bt = _apply_lys(
            atoms, bt_struct,
            chain=chain, res_id=res_id, target_state=target_state,
            nh_bond_length=nh_bond_length,
        )
    elif family == "CYS":
        out_atoms, out_bt = _apply_cys(
            atoms, bt_struct,
            chain=chain, res_id=res_id, target_state=target_state,
            sh_bond_length=sh_bond_length,
        )
    else:  # pragma: no cover
        raise RuntimeError(f"unhandled family {family}")

    return out_atoms, out_bt, charge_delta


def apply_residue_states(
    atoms, bt_struct,
    residue_states: dict[tuple[str, int], str],
    *,
    nh_bond_length: float = BOND_LENGTH_NH_DEFAULT,
    oh_bond_length: float = BOND_LENGTH_OH_DEFAULT,
    sh_bond_length: float = BOND_LENGTH_SH_DEFAULT,
):
    """Apply a whole set of (chain, res_id) → target_state assignments at once.

    Args:
        atoms, bt_struct: input structure.
        residue_states: mapping from (chain, res_id) → target_state label.
        nh_bond_length, oh_bond_length, sh_bond_length: passed through.

    Returns:
        ``(new_atoms, new_bt, ledger_delta_dict, total_charge_delta)``
        where ``ledger_delta_dict`` maps the residue label
        ``"<RESNAME><RESID><CHAIN>"`` to the per-residue charge delta.
    """
    out_atoms = atoms
    out_bt = bt_struct
    ledger_delta: dict[str, int] = {}
    total_delta = 0
    for (chain, res_id), target_state in residue_states.items():
        info = _gather_residue_info(out_atoms, out_bt, chain, res_id)
        src_resname = info.res_name
        out_atoms, out_bt, dq = apply_protonation(
            out_atoms, out_bt,
            chain=chain, res_id=res_id, target_state=target_state,
            nh_bond_length=nh_bond_length,
            oh_bond_length=oh_bond_length,
            sh_bond_length=sh_bond_length,
        )
        label = f"{src_resname}{int(res_id)}{chain}->{target_state}"
        ledger_delta[label] = dq
        total_delta += dq
    return out_atoms, out_bt, ledger_delta, total_delta


# ----------------------------------------------------------------------------
# Source-resname → canonical state inference (for charge-delta computation)
# ----------------------------------------------------------------------------
def _resname_to_state(resname: str) -> str:
    """Infer the canonical STATE label for an input residue name.

    Defaults: HIS → HIE (most common neutral-tautomer convention at pH7),
    ASP → ASP-, GLU → GLU-, LYS → LYS+, CYS → CYS-SH.
    """
    rn = resname.strip().upper()
    return {
        "HIS": "HIE", "HIE": "HIE", "HID": "HID", "HIP": "HIP",
        "ASP": "ASP-", "ASH": "ASP0",
        "GLU": "GLU-", "GLH": "GLU0",
        "LYS": "LYS+", "LYN": "LYS0",
        "CYS": "CYS-SH", "CYM": "CYS-S-",
    }.get(rn, "")


# ----------------------------------------------------------------------------
# Residue discovery
# ----------------------------------------------------------------------------
@dataclass
class DiscoveredResidue:
    chain: str
    res_id: int
    res_name: str
    family: str
    valid_states: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{self.chain}:{self.res_id}:{self.res_name}"


def discover_titratable_residues(
    bt_struct,
    families: Iterable[str] | None = None,
) -> list[DiscoveredResidue]:
    """List all titratable residues in bt_struct.

    Args:
        bt_struct: biotite AtomArray.
        families: subset of ``("HIS","ASP","GLU","LYS","CYS")`` to include;
            ``None`` means all five.

    Returns:
        List of :class:`DiscoveredResidue` (deduplicated by chain+res_id).
    """
    fam_filter = {f.upper() for f in families} if families else {
        "HIS", "ASP", "GLU", "LYS", "CYS",
    }
    out: list[DiscoveredResidue] = []
    seen: set[tuple[str, int]] = set()
    chains = np.asarray(bt_struct.chain_id)
    rids = np.asarray(bt_struct.res_id)
    rnames = np.asarray(bt_struct.res_name)
    for i in range(len(bt_struct)):
        rn = str(rnames[i]).strip().upper()
        if rn not in RESIDUE_FAMILY:
            continue
        family, valid = RESIDUE_FAMILY[rn]
        if family not in fam_filter:
            continue
        chain = str(chains[i])
        rid = int(rids[i])
        key = (chain, rid)
        if key in seen:
            continue
        seen.add(key)
        out.append(DiscoveredResidue(
            chain=chain, res_id=rid, res_name=rn,
            family=family, valid_states=valid,
        ))
    return out
