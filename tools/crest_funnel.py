#!/usr/bin/env python
"""crest_funnel.py — cheap iterative TS-conformer sampling.

Pipeline (one PDB in, ranked structures out):
  0. Parse PDB; separate waters; identify CA / Zn / P-Onuc-Olg / charge.
  A. GFN2-xTB constrained pre-opt (no waters): fix CA + Zn, restrain P-Onuc,
     P-Olg distances. Stabilizes the starting point.
  B. CREST --nci with the same constraints (no waters). Conformer ensemble.
  C. g-xTB constrained opt of top-N CREST conformers. Re-rank by g-xTB energy.
  D. Re-insert original waters into each top g-xTB conformer.
  E. Water-only relaxation: $fix everything except water atoms, GFN2 opt.
  F. Write final ranked CSV + per-conformer PDB.

Designed for theozyme-style active-site cluster cuts where:
  - protein backbone CAs must NOT move (they encode the scaffold),
  - the reactive P-Onuc / P-Olg distances must stay near the TS guess,
  - waters are flexible bystanders.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ----- xtb convergence targets (Eh/Bohr, RMS gradient) ----------------------
# Hardcoded from xtb 6.7.x defaults — see xtb/src/optimize.f90. These are the
# *RMS gradient* convergence thresholds for each --opt level. We expose them
# as a module-level table so callers (Strategy E adaptive-extension logic) can
# look up "how close to convergence is this run" without re-deriving the
# numbers. xtb's own settings live in code; we mirror them here for a
# read-only diagnostic.
XTB_OPT_TARGET_GRAD_RMS: dict[str, float] = {
    "crude":   1.0e-2,
    "vloose":  1.0e-3,
    "loose":   5.0e-3,
    "normal":  1.0e-3,
    "tight":   1.0e-4,
    "vtight":  2.0e-5,
    "extreme": 5.0e-6,
}

# Default poll interval for the progress sidecar thread: slow enough not to hammer
# the disk, fast enough that a 10-min check window has ~50 samples to read.
PROGRESS_POLL_INTERVAL_S_DEFAULT = 12.0
# Window size (number of recent steps) used to compute the trending-down slope.
PROGRESS_TREND_WINDOW_DEFAULT = 5
# Slope threshold (per-step log10 of grad RMS) below which we call a run
# "trending down". -0.05 ≈ 11% reduction per step in gradient magnitude.
PROGRESS_TREND_SLOPE_THRESHOLD = -0.05


# ----- post-CREST geometry filter (Stage B → Stage C safety net) ------------
# CREST's MTD bias forces sometimes push atoms into unphysical clashes during
# conformer sampling. The post-MTD ensemble optimization (-O crude default)
# does NOT always relax these out (e.g. observed 1.073 A N-CA bond on PTE
# KCX_set3 conf_01, atoms 184-187 — the BACKBONE N-CA of HIS 257, which has
# d=1.475 A in the source PDB but was perturbed to 1.073 A by CREST's MTD).
# Stage C's downstream xtb then spends 30+ minutes failing to converge SCF on
# this pathological geometry. We catch these between Stage B and Stage C with
# a heavy-heavy short-bond filter.
#
# REVISED FILTER STRATEGY (2026-05-06):
#   1. Map post-CREST atoms back to the source PDB by element + Kabsch-aligned
#      proximity. For each violated bond, look up the corresponding bond
#      distance in the SOURCE PDB. Use that as the repair target. This is
#      automatically bond-order-aware (single vs double vs triple vs aromatic)
#      because we copy the source's actual chemistry.
#   2. ONLY when no source bond is found (e.g. brand-new contacts created by
#      MTD sampling between previously-non-bonded atoms), fall back to the
#      element-pair single-bond table below.
#   3. Reactive atoms (P, Onuc, Olg, etc.) are PROTECTED — bonds involving
#      them are never auto-repaired or rejected, just logged. Reasoning: TS
#      geometries legitimately have unusual P-Onuc / P-Olg distances.
#   4. Source-distance comparison: if a violated bond exists in source AND
#      post_d > source_d * shrink_tolerance (default 0.7), the deviation is
#      tolerated (probably a normal vibration, not an MTD artifact).
#
# Element-pair bond-length table for the FALLBACK path only (single-bond
# covalent distances, in A; symmetric A-B == B-A). ORGANIC SINGLE BONDS ONLY:
# metals are deliberately EXCLUDED because metal-ligand coordination
# distances vary too widely with coordination number, charge, and ligand
# (Zn-O can be anywhere from 1.95 to 2.6 A). For metal bonds, the filter
# logs a warning and lets xtb sort it out — CREST distortions on metals
# indicate sampling problems beyond what a simple repair can fix.
DEFAULT_REPAIR_BOND_LENGTHS: dict[tuple[str, str], float] = {
    # Carbon backbone single bonds
    ("C", "C"): 1.54, ("C", "N"): 1.47, ("C", "O"): 1.43, ("C", "S"): 1.81,
    ("C", "P"): 1.84, ("C", "H"): 1.09, ("C", "F"): 1.35, ("C", "Cl"): 1.77,
    ("C", "Br"): 1.94, ("C", "I"): 2.14,
    # Nitrogen single bonds
    ("N", "N"): 1.45, ("N", "O"): 1.40, ("N", "H"): 1.01, ("N", "S"): 1.71,
    ("N", "P"): 1.71,
    # Oxygen single bonds
    ("O", "O"): 1.48, ("O", "H"): 0.96, ("O", "S"): 1.57,
    # Phosphorus / Sulfur (NOT bond-order aware — fallback only;
    # source-PDB lookup handles P=O double bonds correctly when present)
    ("P", "O"): 1.61, ("P", "S"): 2.10, ("P", "P"): 2.21, ("P", "H"): 1.42,
    ("S", "S"): 2.05, ("S", "H"): 1.34,
    # NOTE: metal entries (Zn-X, Mg-X, Mn-X, Fe-X, Cu-X, Ca-X, Ni-X, Co-X)
    # were REMOVED from this table on 2026-05-06. Metal-ligand coordination
    # distances vary too widely (e.g. Zn-O 1.95-2.60 A depending on
    # coordination/charge/ligand) for a one-size-fits-all repair target. The
    # source-PDB lookup path handles real metal bonds via the actual source
    # distance; the filter falls back to logging-only when a metal bond is
    # not in the source (i.e. a brand-new contact that xtb should resolve).
}

# Covalent radii (Å), median values from Cordero et al. (2008). Used as the
# fallback for atom pairs not in DEFAULT_REPAIR_BOND_LENGTHS — kept here as
# a hardcoded table so we do not require an ASE install at runtime.
COVALENT_RADII_A: dict[str, float] = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
    "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Mo": 1.54, "Pd": 1.39,
    "Ag": 1.45, "Cd": 1.44, "Sn": 1.39, "I": 1.39,
}

# Short-bond detection: covalent_radius(A) + covalent_radius(B) typically
# gives a single-bond reference. We treat atoms within this fraction of that
# reference as "bonded" when computing the connection-count heuristic used to
# decide which atom of a clashing pair to MOVE during repair (the one with
# fewer connections is more peripheral, so it moves).
CONNECTIVITY_TOLERANCE_FRACTION: float = 1.30
# Default heavy-heavy short-bond cutoff in Å. CREST's pathological clashes
# routinely produce sub-1.10 Å bonds (e.g. observed 1.073 Å N-CA on PTE
# KCX_set3 conf_01 — a backbone HIS 257 N-CA peptide bond, source distance
# 1.475 Å, perturbed by MTD to 1.073 Å). We set the default to 1.10 Å —
# strictly ABOVE the observed CREST artifact (1.073 Å) so it triggers, but
# strictly below the tightest legitimate carbonyl C=O (~1.20 Å) and the
# tightest triple bonds (C≡N ≈ 1.16 Å, C≡C ≈ 1.20 Å). Choosing 1.10 catches
# all real-world CREST clashes we have seen. False positives on triple
# bonds are then RESOLVED by the source-distance comparison
# (POST_CREST_SOURCE_SHRINK_TOLERANCE_DEFAULT below): if the source has the
# same bond at, say, 1.16 Å (C≡N), we tolerate 1.10 Å as within the shrink
# tolerance instead of repairing it.
# History: spec considered 0.95 Å (too lax — misses 1.073 Å bug), 1.05 Å
# (still misses 1.073 Å), 1.10 Å (catches the bug + relies on source-
# distance comparison to avoid triple-bond false-positives).
POST_CREST_BOND_CUTOFF_DEFAULT: float = 1.10
# Default tolerance for source-distance comparison during the source-PDB
# lookup path. If post_d / source_d >= tolerance, the deviation is treated
# as a normal vibration (TOLERATED). Otherwise, the deviation is an MTD
# artifact (REPAIRED to source_d, or rejected per user mode).
# 0.7 means "post-CREST bond can be up to 30% shorter than source before we
# call it an MTD artifact" — covers normal harmonic vibrations (typically
# <5%) with comfortable margin.
# CAVEAT for the observed PTE KCX_set3 conf_01 case: that 1.073 Å N-CA bond
# (source 1.475 Å) has ratio 0.727 — JUST ABOVE this 0.7 default tolerance,
# so the source-comparison path TOLERATES it. To force repair on cases like
# this without affecting legitimate vibrations, raise the tolerance to 0.75
# (or interpret the 1.073 → 1.475 jump as legitimately bad and rely on the
# downstream xtb optimizer to fail-loudly with a salvageable partial). The
# trade-off: tightening the tolerance increases false-positives on real
# vibrations near 0.7-0.75. We default to 0.7 because xtb's salvage-on-
# timeout (introduced in tasks #25 / #26) catches the failed-SCF case
# elegantly, so we prefer to err toward keeping more conformers.
POST_CREST_SOURCE_SHRINK_TOLERANCE_DEFAULT: float = 0.7
# Default for how many iterative repair passes are allowed. Each pass fixes
# the *worst* (shortest) bad bond; cascading clashes (atom k now too close
# to atom l after fixing i-j) get caught on the next pass.
POST_CREST_REPAIR_MAX_PASSES_DEFAULT: int = 5


# Elements treated as "metal" for the purposes of EXCLUDING from the
# fallback element-pair lookup. If either atom of a violated bond is a
# metal AND the source-PDB lookup misses, we log a warning and pass
# through (no auto-repair).
METAL_ELEMENTS: frozenset[str] = frozenset({
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "Mo",
    "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs", "Ba", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
    # Lanthanides + actinides (rare but parametrized in some xtb inputs)
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
    "Yb", "Lu", "Th", "U",
})


def _is_metal(elem: str) -> bool:
    """Return True if ``elem`` (a 1- or 2-char element symbol) is treated
    as a metal for the post-CREST filter's bond-repair fallback."""
    return elem in METAL_ELEMENTS


def _covalent_radius_or_default(element: str, default: float = 0.75) -> float:
    """Return covalent radius in Å for ``element``; fall back to ``default``."""
    return COVALENT_RADII_A.get(element, default)


def _repair_bond_target_length_organic(elem_a: str, elem_b: str) -> float | None:
    """Look up the FALLBACK target single-bond length (Å) for an A-B pair.

    Symmetric A-B == B-A. Returns ``None`` (NOT a default) when EITHER atom
    is a metal — this enforces the user-requested rule that metals are not
    auto-repaired via the fallback element-pair table. Callers must handle
    ``None`` explicitly (typically: log a warning and skip the repair).

    For non-metal pairs not in ``DEFAULT_REPAIR_BOND_LENGTHS``, falls back
    to ``cov_radius(A) + cov_radius(B)``, which matches CCDC-tabulated
    values for most organic single bonds within ~10 %.
    """
    if _is_metal(elem_a) or _is_metal(elem_b):
        return None
    key1 = (elem_a, elem_b)
    key2 = (elem_b, elem_a)
    if key1 in DEFAULT_REPAIR_BOND_LENGTHS:
        return DEFAULT_REPAIR_BOND_LENGTHS[key1]
    if key2 in DEFAULT_REPAIR_BOND_LENGTHS:
        return DEFAULT_REPAIR_BOND_LENGTHS[key2]
    return (_covalent_radius_or_default(elem_a)
            + _covalent_radius_or_default(elem_b))


def _parse_xyz_body(body: list[str]) -> tuple[list[str], np.ndarray]:
    """Parse an XYZ body (list of 'El x y z' lines, no count/comment header)
    into element list and (N, 3) coordinate array."""
    elems: list[str] = []
    xyz: list[list[float]] = []
    for ln in body:
        toks = ln.split()
        if len(toks) < 4:
            continue
        elems.append(toks[0])
        xyz.append([float(toks[1]), float(toks[2]), float(toks[3])])
    return elems, np.asarray(xyz, dtype=float)


def _xyz_body_from_arrays(elems: list[str], coords: np.ndarray) -> list[str]:
    """Inverse of ``_parse_xyz_body`` — produce 'El x y z' lines."""
    out: list[str] = []
    for el, (x, y, z) in zip(elems, coords):
        out.append(f"{el:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}")
    return out


def _detect_short_heavy_heavy_bonds(
    elems: list[str], coords: np.ndarray, cutoff_a: float,
) -> list[tuple[float, int, int]]:
    """Return list of ``(distance_A, i, j)`` heavy-heavy pairs with d < cutoff.

    H atoms are SKIPPED on both sides — H-X bonds (≈0.96–1.42 Å) routinely
    fall under modest cutoffs and are not the artifact this filter targets.
    Sorted ascending by distance (worst-first).
    """
    pairs: list[tuple[float, int, int]] = []
    n = len(elems)
    for i in range(n):
        if elems[i] == "H":
            continue
        for j in range(i + 1, n):
            if elems[j] == "H":
                continue
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < cutoff_a:
                pairs.append((d, i, j))
    pairs.sort()
    return pairs


# ----- source-PDB cross-frame mapping for bond-order-aware repair -----------
def _kabsch_align_src_into_post(
    src_coords: np.ndarray, post_coords: np.ndarray,
    src_anchor_idx: list[int], post_anchor_idx: list[int],
) -> np.ndarray:
    """Align ``src_coords`` into ``post_coords``'s frame using the supplied
    anchor indices (Kabsch). Returns a fresh ndarray of the same shape as
    ``src_coords``.

    Anchors must correspond row-by-row (src_anchor_idx[k] aligns with
    post_anchor_idx[k]). At least 3 non-colinear anchors are required;
    fewer raises ``ValueError``.
    """
    if (len(src_anchor_idx) != len(post_anchor_idx)
            or len(src_anchor_idx) < 3):
        raise ValueError(
            f"_kabsch_align_src_into_post: need >=3 paired anchors; got "
            f"len(src)={len(src_anchor_idx)}, len(post)={len(post_anchor_idx)}"
        )
    src_anchor_pts = src_coords[src_anchor_idx]
    post_anchor_pts = post_coords[post_anchor_idx]
    cP = src_anchor_pts.mean(0)
    cQ = post_anchor_pts.mean(0)
    H = (src_anchor_pts - cP).T @ (post_anchor_pts - cQ)
    U, S, Vt = np.linalg.svd(H)
    sign = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, sign]) @ U.T
    return (src_coords - cP) @ R.T + cQ


def _build_post_to_source_atom_map(
    post_elems: list[str], post_coords: np.ndarray,
    src_elems: list[str], src_coords_aligned: np.ndarray,
    *, max_match_distance_a: float = 0.75,
) -> list[int | None]:
    """Map each post-CREST atom to its closest same-element source-PDB atom.

    Implementation: for each post atom (post_idx, post_elem), find the
    source atom of the same element with the smallest Euclidean distance
    in the aligned frame. If that distance exceeds ``max_match_distance_a``
    (default 0.75 Å — a conservative threshold; backbone atoms typically
    drift <0.3 Å during CREST sampling, side chains <0.5 Å), we treat the
    post atom as UNMAPPED (returns ``None`` at that index). Brand-new
    contacts created by MTD between previously-distant atoms therefore
    cannot spuriously match.

    Returns a list of length ``len(post_elems)``; entries are 0-based source
    indices or ``None``. The mapping is many-to-one in principle (multiple
    post atoms could nominally match one source atom) but in practice
    backbone scaffolds are well-aligned and collisions are rare; we DO NOT
    enforce uniqueness because the caller only needs to look up bond pairs,
    and even if two post atoms map to the same source atom the lookup
    degrades gracefully (it just returns the same source bond).
    """
    n_post = len(post_elems)
    n_src = len(src_elems)
    out: list[int | None] = [None] * n_post
    if n_src == 0:
        return out
    src_coords_arr = np.asarray(src_coords_aligned)
    # Group source atoms by element for fast lookup
    src_by_elem: dict[str, list[int]] = {}
    for k, el in enumerate(src_elems):
        src_by_elem.setdefault(el, []).append(k)
    for i, post_el in enumerate(post_elems):
        candidates = src_by_elem.get(post_el, [])
        if not candidates:
            continue
        cand_coords = src_coords_arr[candidates]
        dists = np.linalg.norm(cand_coords - post_coords[i], axis=1)
        j_local = int(np.argmin(dists))
        if float(dists[j_local]) <= max_match_distance_a:
            out[i] = candidates[j_local]
    return out


def _lookup_source_bond_distance(
    post_i: int, post_j: int,
    post_to_src_map: list[int | None],
    src_coords: np.ndarray,
) -> float | None:
    """Given two post-CREST atom indices and the post→source map, return
    the Euclidean distance of the corresponding source-PDB pair, or
    ``None`` if either atom did not map to a source atom.

    The returned distance reflects whatever bond order is present in the
    SOURCE (single, double, aromatic, triple, coordination). This is the
    primary bond-order-aware repair target.
    """
    si = post_to_src_map[post_i]
    sj = post_to_src_map[post_j]
    if si is None or sj is None:
        return None
    return float(np.linalg.norm(src_coords[si] - src_coords[sj]))


# ----- reactive atom protection ---------------------------------------------
def _parse_reactive_atoms_spec(
    spec: str | None,
    pdb_atoms: list["PdbAtom"] | None,
) -> set[int]:
    """Parse the ``--post-crest-reactive-atoms`` flag into a 0-based atom
    index set (indices into the no-waters atom list).

    Tokens accepted (comma-separated):
      - Bare integer N: 1-based index into the no-waters atom list (i.e.
        the same numbering the CREST conformer uses).
      - ``NAME.RESNAME`` (case-insensitive): selects atoms whose name and
        residue name match (e.g. ``P1.SUB``, ``O3.OHX``, ``O7.SUB``).
        Requires ``pdb_atoms`` to be non-None.

    Returns a set of 0-based indices. Empty string / None returns empty set
    (no protection).
    """
    out: set[int] = set()
    if spec is None or not spec.strip():
        return out
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    for tok in tokens:
        if "." in tok:
            if pdb_atoms is None:
                log.warning("reactive-atoms: ignoring NAME.RESNAME token %r "
                            "(no PdbAtom list available)", tok)
                continue
            name_part, _, res_part = tok.partition(".")
            name_l, res_l = name_part.strip().upper(), res_part.strip().upper()
            matched_any = False
            for k, a in enumerate(pdb_atoms):
                if a.name.upper() == name_l and a.resname.upper() == res_l:
                    out.add(k)
                    matched_any = True
            if not matched_any:
                log.warning("reactive-atoms: token %r matched zero atoms",
                            tok)
        else:
            try:
                serial_1b = int(tok)
            except ValueError:
                log.warning("reactive-atoms: cannot parse token %r — "
                            "expected int (1-based) or NAME.RESNAME", tok)
                continue
            if serial_1b < 1:
                log.warning("reactive-atoms: 1-based serial %d is invalid",
                            serial_1b)
                continue
            out.add(serial_1b - 1)
    return out


def _connection_counts(
    elems: list[str], coords: np.ndarray,
    tolerance_fraction: float = CONNECTIVITY_TOLERANCE_FRACTION,
) -> np.ndarray:
    """Heuristic connection count per atom — used to decide which member of a
    clashing pair to MOVE (the one with fewer connections). Reference distance
    is ``(cov_rad(A) + cov_rad(B)) * tolerance_fraction``."""
    n = len(elems)
    counts = np.zeros(n, dtype=int)
    radii = np.array([_covalent_radius_or_default(e) for e in elems])
    for i in range(n):
        for j in range(i + 1, n):
            ref = (radii[i] + radii[j]) * tolerance_fraction
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d <= ref:
                counts[i] += 1
                counts[j] += 1
    return counts


def _repair_short_bonds(
    elems: list[str], coords: np.ndarray, cutoff_a: float,
    max_passes: int,
    log_prefix: str = "",
    *,
    source_target_lookup: "callable | None" = None,
    reactive_atoms: set[int] | None = None,
    source_shrink_tolerance: float = POST_CREST_SOURCE_SHRINK_TOLERANCE_DEFAULT,
) -> tuple[np.ndarray, list[dict], int, list[tuple[float, int, int]]]:
    """Iteratively repair short heavy-heavy bonds by pushing the more-peripheral
    atom of each clashing pair away along the bond axis.

    Repair-target priority (per bond):
      1. Source PDB lookup (if ``source_target_lookup`` provided): use the
         source's actual A-B distance as the target. Bond-order-aware
         because we copy the source's chemistry. Skipped only when
         post_d > source_d * source_shrink_tolerance (treated as a
         tolerable vibration; bond is REMOVED from the bad-bond list this
         pass).
      2. Element-pair fallback (organic atoms only): used when the source
         lookup returns ``None``. Falls through to LOG-AND-SKIP for metal
         pairs (no auto-repair).
      3. Reactive-atom protection: bonds where EITHER atom is in
         ``reactive_atoms`` are LOGGED but NEVER repaired.

    Returns ``(repaired_coords, repair_log, passes_used, residual_bad_bonds)``.
    ``residual_bad_bonds`` is the list of bonds STILL below ``cutoff_a``
    after ``max_passes`` AND deemed "should-have-been-repaired" (i.e. NOT
    tolerated, NOT reactive, NOT metal-without-source). Empty on success.

    ``source_target_lookup`` is a callable ``(post_i, post_j) -> float | None``;
    ``None`` means no source bond was found.
    """
    if reactive_atoms is None:
        reactive_atoms = set()
    coords = coords.copy()
    repair_log: list[dict] = []
    passes_used = 0
    residual: list[tuple[float, int, int]] = []
    # Per-conformer set of (i, j) pairs we have already declared "tolerated"
    # (post-CREST shorter than source but within shrink tolerance) or "reactive"
    # or "metal-no-source". Once tolerated, we will not consider it again
    # for repair, but it IS still reported via the per-conformer log.
    handled_pairs: set[tuple[int, int]] = set()
    for pass_idx in range(max_passes):
        bad_all = _detect_short_heavy_heavy_bonds(elems, coords, cutoff_a)
        # Filter out pairs we've already decided to leave alone.
        bad = [(d, i, j) for (d, i, j) in bad_all
               if (i, j) not in handled_pairs]
        if not bad:
            residual = []
            break
        passes_used = pass_idx + 1
        d_bad, i, j = bad[0]
        elem_i, elem_j = elems[i], elems[j]

        # Reactive-atom protection: log only, never repair.
        if i in reactive_atoms or j in reactive_atoms:
            log.warning(
                "REACTIVE-PROTECTED: %satom_i(%s#%d) - atom_j(%s#%d) at "
                "%.3f A (cutoff %.3f A) — leaving unchanged because at "
                "least one atom is in --post-crest-reactive-atoms",
                log_prefix, elem_i, i + 1, elem_j, j + 1, d_bad, cutoff_a,
            )
            handled_pairs.add((i, j))
            repair_log.append({
                "pass": pass_idx + 1,
                "atom_i": int(i + 1), "atom_j": int(j + 1),
                "elem_i": elem_i, "elem_j": elem_j,
                "d_before_A": d_bad,
                "action": "reactive_protected",
            })
            continue

        # Determine repair target.
        target: float | None = None
        target_source: str = "unknown"
        if source_target_lookup is not None:
            src_d = source_target_lookup(i, j)
            if src_d is not None:
                # If the post-CREST distance is within shrink tolerance of
                # the source distance, treat as a tolerable vibration and
                # skip repair.
                if d_bad >= src_d * source_shrink_tolerance:
                    log.info(
                        "TOLERATE: %satom_i(%s#%d) - atom_j(%s#%d) "
                        "post_d=%.3f A vs source_d=%.3f A "
                        "(ratio=%.2f >= shrink_tol=%.2f); leaving unchanged",
                        log_prefix, elem_i, i + 1, elem_j, j + 1,
                        d_bad, src_d, d_bad / src_d, source_shrink_tolerance,
                    )
                    handled_pairs.add((i, j))
                    repair_log.append({
                        "pass": pass_idx + 1,
                        "atom_i": int(i + 1), "atom_j": int(j + 1),
                        "elem_i": elem_i, "elem_j": elem_j,
                        "d_before_A": d_bad, "source_d_A": src_d,
                        "action": "tolerated_vibration",
                    })
                    continue
                target = src_d
                target_source = "source_pdb"

        if target is None:
            # No source bond found; try the organic fallback. If either atom
            # is a metal, this returns None — we then log and skip.
            fallback = _repair_bond_target_length_organic(elem_i, elem_j)
            if fallback is None:
                log.warning(
                    "METAL-NO-REPAIR: %satom_i(%s#%d) - atom_j(%s#%d) at "
                    "%.3f A — at least one atom is a metal AND no source-PDB "
                    "bond was found; logging only (xtb will sort it out)",
                    log_prefix, elem_i, i + 1, elem_j, j + 1, d_bad,
                )
                handled_pairs.add((i, j))
                repair_log.append({
                    "pass": pass_idx + 1,
                    "atom_i": int(i + 1), "atom_j": int(j + 1),
                    "elem_i": elem_i, "elem_j": elem_j,
                    "d_before_A": d_bad,
                    "action": "metal_no_source_no_repair",
                })
                continue
            target = fallback
            target_source = "fallback_organic"

        # Connection counts based on the CURRENT (pre-pass) geometry — atoms
        # with fewer connections are "more peripheral" and move preferentially.
        conns = _connection_counts(elems, coords)
        if conns[j] <= conns[i]:
            mover, anchor = j, i
        else:
            mover, anchor = i, j
        vec = coords[mover] - coords[anchor]
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            # Degenerate overlap — push along +x to break symmetry, then let
            # xtb relax the rest. Anything is better than divide-by-zero.
            unit = np.array([1.0, 0.0, 0.0])
        else:
            unit = vec / norm
        coords[mover] = coords[anchor] + unit * target
        repair_log.append({
            "pass": pass_idx + 1,
            "atom_i": int(i + 1),
            "atom_j": int(j + 1),
            "elem_i": elem_i,
            "elem_j": elem_j,
            "d_before_A": d_bad,
            "d_after_A": target,
            "target_source": target_source,
            "moved_atom": int(mover + 1),
            "anchor_atom": int(anchor + 1),
            "action": "repaired",
        })
        log.warning(
            "REPAIR(%s): %satom_i(%s#%d) - atom_j(%s#%d) bumped %.3f->%.3f A "
            "(pass %d, moved=#%d)",
            target_source, log_prefix, elem_i, i + 1, elem_j, j + 1,
            d_bad, target, pass_idx + 1, mover + 1,
        )
    else:
        # Loop exhausted without break → still bad bonds. Recompute the
        # residual list, EXCLUDING handled_pairs (those are not "should-
        # have-been-repaired" residuals, they are explicitly accepted).
        all_residual = _detect_short_heavy_heavy_bonds(elems, coords, cutoff_a)
        residual = [(d, i, j) for (d, i, j) in all_residual
                    if (i, j) not in handled_pairs]
    return coords, repair_log, passes_used, residual


def _classify_bad_bonds(
    bad: list[tuple[float, int, int]],
    elems: list[str],
    *,
    source_target_lookup: "callable | None",
    reactive_atoms: set[int],
    source_shrink_tolerance: float,
) -> tuple[list[tuple[float, int, int]], list[dict]]:
    """Pre-classify a bad-bond list into (real_bad, accepted_bonds).

    real_bad: bonds that should be rejected/repaired (not tolerated, not
        reactive-protected, not metal-without-source).
    accepted_bonds: dicts describing each tolerated/reactive/metal-no-repair
        case (for diagnostic logging in reject/keep/log modes; repair mode
        re-derives these inside _repair_short_bonds).

    The four "accepted" categories (i.e. bonds that are FLAGGED but NOT
    treated as MTD artifacts):

      1. ``reactive_protected``: at least one atom is in
         ``reactive_atoms`` (e.g. P, Onuc, Olg). Per spec, bonds at
         reactive atoms are intentionally allowed to take TS-like values.
      2. ``tolerated_vibration``: source PDB has the bond at source_d
         and post_d/source_d >= source_shrink_tolerance, so the deviation
         is a normal vibration.
      3. ``metal_no_source_no_repair``: at least one atom is a metal AND
         the source-PDB lookup did not find a corresponding bond (brand-
         new contact created by MTD). Metal coordination distances vary
         too widely for a fallback table; xtb will resolve.
    """
    real_bad: list[tuple[float, int, int]] = []
    accepted: list[dict] = []
    for d, i, j in bad:
        elem_i, elem_j = elems[i], elems[j]
        if i in reactive_atoms or j in reactive_atoms:
            accepted.append({
                "atom_i": int(i + 1), "atom_j": int(j + 1),
                "elem_i": elem_i, "elem_j": elem_j,
                "d_A": d, "action": "reactive_protected",
            })
            continue
        src_d: float | None = None
        if source_target_lookup is not None:
            src_d = source_target_lookup(i, j)
            if src_d is not None and d >= src_d * source_shrink_tolerance:
                accepted.append({
                    "atom_i": int(i + 1), "atom_j": int(j + 1),
                    "elem_i": elem_i, "elem_j": elem_j,
                    "d_A": d, "source_d_A": src_d,
                    "action": "tolerated_vibration",
                })
                continue
        # Metal exclusion: if no source bond AND at least one atom is a
        # metal, log-only (do NOT classify as real_bad).
        if src_d is None and (_is_metal(elem_i) or _is_metal(elem_j)):
            accepted.append({
                "atom_i": int(i + 1), "atom_j": int(j + 1),
                "elem_i": elem_i, "elem_j": elem_j,
                "d_A": d, "action": "metal_no_source_no_repair",
            })
            continue
        real_bad.append((d, i, j))
    return real_bad, accepted


def post_crest_geometry_filter(
    confs: list[tuple[float, list[str]]],
    *,
    bond_cutoff_a: float = POST_CREST_BOND_CUTOFF_DEFAULT,
    bad_bond_mode: str = "reject",
    max_repair_passes: int = POST_CREST_REPAIR_MAX_PASSES_DEFAULT,
    pre_repair_writer: "callable | None" = None,
    source_elems: list[str] | None = None,
    source_coords: np.ndarray | None = None,
    source_anchor_post_idx: list[int] | None = None,
    source_anchor_src_idx: list[int] | None = None,
    same_atom_order: bool = False,
    reactive_atoms: set[int] | None = None,
    source_shrink_tolerance: float = POST_CREST_SOURCE_SHRINK_TOLERANCE_DEFAULT,
    max_match_distance_a: float = 0.75,
) -> tuple[list[tuple[float, list[str]]], dict]:
    """Apply post-CREST short-bond filter to a candidate conformer list.

    Parameters
    ----------
    confs
        List of ``(energy_in_Eh, body_lines)`` tuples. ``body_lines`` are
        'El x y z' strings, one per atom, NO count/comment header.
    bond_cutoff_a
        Heavy-heavy distance below which a bond is considered candidate-
        artifact. Default ``POST_CREST_BOND_CUTOFF_DEFAULT`` (1.10 Å).
    bad_bond_mode
        One of:
          - ``'reject'`` (default): drop conformers with bad bonds.
          - ``'repair'``: iteratively push the more-peripheral atom of each
            clashing pair out to the source-PDB target distance (or, if not
            found, the element-pair organic fallback). Falls back to
            ``'reject'`` for that conformer if residual bad bonds remain
            after ``max_repair_passes``.
          - ``'keep'``: warn but do not modify (legacy passthrough).
          - ``'log'``: same as keep but emit a structured DIAGNOSTIC line
            per bad bond (useful when investigating CREST output).
    max_repair_passes
        Iterative repair passes; each fixes the worst (shortest) bad bond
        on the current geometry. Cascading clashes are caught on later
        passes.
    pre_repair_writer
        Optional callable ``f(conf_index_0based, body_pre, body_post)`` for
        persisting the pre-repair geometry alongside the repaired one.
    source_elems, source_coords
        Source PDB elements and coordinates (typically the no-waters atom
        list) for the bond-order-aware repair path. When ``None``, the
        filter falls back to element-pair-only logic (organic fallback +
        metal-no-repair). Otherwise, each conformer's atoms are mapped to
        source atoms by element + Kabsch-aligned proximity, and the source's
        actual bond distance is used as the repair target.
    source_anchor_post_idx, source_anchor_src_idx
        Index pairs (CREST-frame post atoms ↔ source-frame atoms) used for
        Kabsch alignment between the post and source frames. Both lists
        must have the same length and ≥ 3 entries (typically all CA atoms).
        Required when ``source_coords`` is provided AND ``same_atom_order``
        is False.
    same_atom_order
        When True, ``source_elems`` and ``source_coords`` are assumed to
        be in the SAME ORDER as the post-CREST conformer (i.e. ``post_atom[k]``
        corresponds to ``source_atom[k]``, with identical lengths). In this
        case the post→source map is simply the identity and NO Kabsch
        alignment is performed (we trust the caller to know the frames
        align). This is the common case for the qcb crest_funnel
        pipeline because the conformer is generated from ``part.no_waters``,
        whose atom order is preserved through CREST.
    reactive_atoms
        Set of 0-based post-frame atom indices that are PROTECTED — bonds
        involving any of these atoms are LOGGED but never repaired or
        rejected. Use to preserve TS-like reactive distances (e.g. P-Onuc,
        P-Olg). Default empty set (no protection).
    source_shrink_tolerance
        When the source PDB has a bond at ``source_d`` and the post-CREST
        bond is at ``post_d``, treat as a tolerated normal vibration if
        ``post_d / source_d >= source_shrink_tolerance``. Otherwise treat
        as an MTD artifact (repair to source_d, or reject). Default 0.7.
    max_match_distance_a
        Maximum aligned-frame Euclidean distance for a post atom to be
        matched to its closest same-element source atom. Brand-new contacts
        (post atom far from any aligned source atom) remain unmapped and
        fall through to the element-pair fallback. Default 0.75 Å.

    Returns
    -------
    (filtered_confs, summary_dict)
    """
    if bad_bond_mode not in ("reject", "repair", "keep", "log"):
        raise ValueError(
            f"post_crest_geometry_filter: unknown bad_bond_mode={bad_bond_mode!r}; "
            "expected 'reject' | 'repair' | 'keep' | 'log'"
        )
    if reactive_atoms is None:
        reactive_atoms = set()

    have_source = (source_coords is not None and source_elems is not None)
    if have_source and not same_atom_order:
        # Need anchor pairs for Kabsch.
        if (source_anchor_post_idx is None
                or source_anchor_src_idx is None
                or len(source_anchor_post_idx) < 3):
            log.warning(
                "post_crest_geometry_filter: source_coords supplied but "
                "no anchor pairs (or <3); falling back to element-pair-"
                "only logic. Pass same_atom_order=True to skip Kabsch."
            )
            have_source = False
    if have_source and same_atom_order and source_elems is not None:
        # Validate length matches the first conf for an early error.
        if confs and len(_parse_xyz_body(confs[0][1])[0]) != len(source_elems):
            raise ValueError(
                f"post_crest_geometry_filter: same_atom_order=True but "
                f"len(post_elems)={len(_parse_xyz_body(confs[0][1])[0])} "
                f"!= len(source_elems)={len(source_elems)}. Either drop "
                "same_atom_order or fix the input."
            )

    out: list[tuple[float, list[str]]] = []
    n_total = len(confs)
    n_clean = 0
    n_rejected = 0
    n_repaired = 0
    n_repair_failed = 0
    n_tolerated = 0
    n_reactive_protected = 0
    n_metal_no_source = 0
    per_conf_diag: list[dict] = []
    for k, (energy, body) in enumerate(confs):
        elems, coords = _parse_xyz_body(body)

        # Build source lookup callable for this conformer.
        #
        # Two paths:
        #   - same_atom_order=True (the common case): post_atom[k] ↔
        #     source_atom[k] is an identity map. We use the source coords
        #     DIRECTLY (no Kabsch) — the source PDB and the CREST-output
        #     conformer are already in the same lab frame because CREST
        #     just strips waters and constraint-optimizes; non-water atom
        #     ordering and coordinate frame are preserved.
        #   - same_atom_order=False: Kabsch-align source into post frame
        #     using the supplied anchor pairs, then match each post atom
        #     to its closest same-element source atom.
        source_target_lookup: "callable | None" = None
        if have_source:
            try:
                src_arr = np.asarray(source_coords)
                if same_atom_order:
                    # Validate per-conformer (defensive; outer check
                    # handled the first conformer only)
                    if len(elems) != len(source_elems):
                        raise ValueError(
                            f"same_atom_order=True but conf has "
                            f"{len(elems)} atoms vs source has "
                            f"{len(source_elems)}"
                        )
                    post_to_src_map: list[int | None] = list(range(len(elems)))
                    src_for_lookup = src_arr
                    n_mapped = len(elems)
                    log.info(
                        "conf %d: identity post→source map (same_atom_order)",
                        k + 1,
                    )
                else:
                    src_for_lookup = _kabsch_align_src_into_post(
                        src_arr, coords,
                        src_anchor_idx=source_anchor_src_idx,
                        post_anchor_idx=source_anchor_post_idx,
                    )
                    post_to_src_map = _build_post_to_source_atom_map(
                        elems, coords, source_elems, src_for_lookup,
                        max_match_distance_a=max_match_distance_a,
                    )
                    n_mapped = sum(1 for x in post_to_src_map if x is not None)
                    log.info(
                        "conf %d: post→source map built (%d/%d atoms "
                        "matched, max_match_distance=%.2f A)",
                        k + 1, n_mapped, len(elems), max_match_distance_a,
                    )

                def _lookup(post_i: int, post_j: int,
                            _m=post_to_src_map,
                            _s=src_for_lookup) -> float | None:
                    return _lookup_source_bond_distance(post_i, post_j, _m, _s)

                source_target_lookup = _lookup
            except Exception as exc:
                log.warning(
                    "conf %d: source-PDB lookup setup failed (%s); using "
                    "element-pair-only logic for this conformer",
                    k + 1, exc,
                )
                source_target_lookup = None

        bad = _detect_short_heavy_heavy_bonds(elems, coords, bond_cutoff_a)
        if not bad:
            out.append((energy, body))
            n_clean += 1
            continue

        # Pre-classify (only matters for non-repair modes which would otherwise
        # treat ALL short bonds as "bad" without source-distance awareness).
        real_bad, accepted = _classify_bad_bonds(
            bad, elems,
            source_target_lookup=source_target_lookup,
            reactive_atoms=reactive_atoms,
            source_shrink_tolerance=source_shrink_tolerance,
        )
        n_acc_react = sum(1 for a in accepted if a["action"] == "reactive_protected")
        n_acc_tol = sum(1 for a in accepted if a["action"] == "tolerated_vibration")
        n_acc_metal = sum(1 for a in accepted
                          if a["action"] == "metal_no_source_no_repair")
        n_reactive_protected += n_acc_react
        n_tolerated += n_acc_tol
        n_metal_no_source += n_acc_metal

        # Always emit a structured log of accepted-but-flagged pairs.
        for acc in accepted:
            if acc["action"] == "reactive_protected":
                log.info(
                    "REACTIVE-PROTECTED: conf %d atom_i=%s#%d atom_j=%s#%d "
                    "d=%.3f A cutoff=%.3f A (in --post-crest-reactive-atoms)",
                    k + 1, acc["elem_i"], acc["atom_i"],
                    acc["elem_j"], acc["atom_j"], acc["d_A"], bond_cutoff_a,
                )
            elif acc["action"] == "tolerated_vibration":
                log.info(
                    "TOLERATE: conf %d atom_i=%s#%d atom_j=%s#%d post_d=%.3f A "
                    "vs source_d=%.3f A (ratio=%.2f >= shrink_tol=%.2f)",
                    k + 1, acc["elem_i"], acc["atom_i"],
                    acc["elem_j"], acc["atom_j"], acc["d_A"],
                    acc["source_d_A"], acc["d_A"] / acc["source_d_A"],
                    source_shrink_tolerance,
                )
            elif acc["action"] == "metal_no_source_no_repair":
                log.warning(
                    "METAL-NO-SOURCE: conf %d atom_i=%s#%d atom_j=%s#%d "
                    "d=%.3f A cutoff=%.3f A — metal-ligand bond with no "
                    "source-PDB match; logging only (xtb will resolve)",
                    k + 1, acc["elem_i"], acc["atom_i"],
                    acc["elem_j"], acc["atom_j"], acc["d_A"], bond_cutoff_a,
                )

        if not real_bad:
            # Everything was tolerable / reactive-protected.
            out.append((energy, body))
            n_clean += 1
            per_conf_diag.append({
                "conf": k + 1, "mode": "kept_after_classification",
                "n_flagged": len(bad),
                "n_tolerated": n_acc_tol,
                "n_reactive_protected": n_acc_react,
                "n_metal_no_source": n_acc_metal,
                "worst_flagged_d_A": bad[0][0],
            })
            continue

        # Bad-bond handling (real_bad is non-empty here).
        if bad_bond_mode == "keep":
            log.warning(
                "post-CREST filter (keep): conf %d has %d real bad bond(s) "
                "below %.3f A (worst: %s%d-%s%d at %.3f A); passthrough",
                k + 1, len(real_bad), bond_cutoff_a,
                elems[real_bad[0][1]], real_bad[0][1] + 1,
                elems[real_bad[0][2]], real_bad[0][2] + 1, real_bad[0][0],
            )
            out.append((energy, body))
            n_clean += 1
            per_conf_diag.append({
                "conf": k + 1, "mode": "keep",
                "n_real_bad": len(real_bad),
                "n_tolerated": n_acc_tol,
                "n_reactive_protected": n_acc_react,
                "n_metal_no_source": n_acc_metal,
                "worst_d_A": real_bad[0][0],
            })
            continue
        if bad_bond_mode == "log":
            for d_bad, i, j in real_bad:
                log.warning(
                    "DIAGNOSTIC post-CREST short-bond: conf=%d atom_i=%s#%d "
                    "atom_j=%s#%d d=%.4f A cutoff=%.3f A",
                    k + 1, elems[i], i + 1, elems[j], j + 1,
                    d_bad, bond_cutoff_a,
                )
            out.append((energy, body))
            n_clean += 1
            per_conf_diag.append({
                "conf": k + 1, "mode": "log",
                "n_real_bad": len(real_bad),
                "n_tolerated": n_acc_tol,
                "n_reactive_protected": n_acc_react,
                "n_metal_no_source": n_acc_metal,
                "worst_d_A": real_bad[0][0],
            })
            continue
        if bad_bond_mode == "reject":
            log.warning(
                "post-CREST filter (reject): conf %d dropped — %d real bad "
                "bond(s) below %.3f A (worst: %s#%d-%s#%d at %.3f A); "
                "%d tolerated, %d reactive-protected",
                k + 1, len(real_bad), bond_cutoff_a,
                elems[real_bad[0][1]], real_bad[0][1] + 1,
                elems[real_bad[0][2]], real_bad[0][2] + 1, real_bad[0][0],
                n_acc_tol, n_acc_react,
            )
            n_rejected += 1
            per_conf_diag.append({
                "conf": k + 1, "mode": "reject",
                "n_real_bad": len(real_bad),
                "n_tolerated": n_acc_tol,
                "n_reactive_protected": n_acc_react,
                "n_metal_no_source": n_acc_metal,
                "worst_d_A": real_bad[0][0],
            })
            continue
        # repair mode
        repaired_coords, repair_log, passes_used, residual = _repair_short_bonds(
            elems, coords, bond_cutoff_a, max_repair_passes,
            log_prefix=f"conf {k + 1:02d}: ",
            source_target_lookup=source_target_lookup,
            reactive_atoms=reactive_atoms,
            source_shrink_tolerance=source_shrink_tolerance,
        )
        if residual:
            log.warning(
                "post-CREST filter (repair): conf %d still has %d real bad "
                "bond(s) after %d pass(es) — falling through to reject",
                k + 1, len(residual), max_repair_passes,
            )
            n_repair_failed += 1
            per_conf_diag.append({
                "conf": k + 1, "mode": "repair_failed",
                "passes_used": passes_used,
                "residual_bad": len(residual),
                "worst_d_A": residual[0][0],
                "n_tolerated": n_acc_tol,
                "n_reactive_protected": n_acc_react,
                "n_metal_no_source": n_acc_metal,
            })
            continue
        new_body = _xyz_body_from_arrays(elems, repaired_coords)
        if pre_repair_writer is not None:
            try:
                pre_repair_writer(k, body, new_body)
            except Exception as exc:  # pragma: no cover — defensive
                log.warning("post-CREST filter (repair): conf %d pre-repair "
                            "writer failed: %s", k + 1, exc)
        out.append((energy, new_body))
        n_repaired += 1
        n_actual_repairs = sum(1 for r in repair_log if r.get("action") == "repaired")
        per_conf_diag.append({
            "conf": k + 1, "mode": "repaired",
            "passes_used": passes_used,
            "n_bonds_fixed": n_actual_repairs,
            "n_tolerated": n_acc_tol,
            "n_reactive_protected": n_acc_react,
            "n_metal_no_source": n_acc_metal,
            "repair_log": repair_log,
            "worst_d_before_A": (repair_log[0]["d_before_A"]
                                 if repair_log else None),
        })

    summary = {
        "n_total": n_total,
        "n_clean_passthrough": n_clean,
        "n_rejected": n_rejected,
        "n_repaired": n_repaired,
        "n_repair_failed": n_repair_failed,
        "n_survivors": len(out),
        "n_tolerated_pairs": n_tolerated,
        "n_reactive_protected_pairs": n_reactive_protected,
        "n_metal_no_source_pairs": n_metal_no_source,
        "bond_cutoff_a": bond_cutoff_a,
        "source_shrink_tolerance": source_shrink_tolerance,
        "max_match_distance_a": max_match_distance_a,
        "have_source_lookup": have_source,
        "n_reactive_atoms": len(reactive_atoms),
        "mode": bad_bond_mode,
        "max_repair_passes": max_repair_passes,
        "per_conformer": per_conf_diag,
    }
    return out, summary


def _compute_workers(ncpu_total: int, n_jobs: int, min_threads_per_job: int = 4) -> tuple[int, int]:
    """Pack n_jobs across ncpu_total cores: workers × threads_per ≤ ncpu_total."""
    if n_jobs <= 0:
        return 0, 1
    workers = max(1, min(n_jobs, ncpu_total // min_threads_per_job))
    threads = max(1, ncpu_total // workers)
    return workers, threads

# ----- vendored binaries -----------------------------------------------------
QCB_ROOT = Path(__file__).resolve().parents[1]
XTB_BIN = QCB_ROOT / "deps/xtb/install/bin/xtb"
CREST_BIN = QCB_ROOT / "deps/crest/install/bin/crest"
GXTB_BIN = QCB_ROOT / "deps/g-xtb/install/xtb-6.7.1/bin/xtb"

log = logging.getLogger("crest_funnel")


# ----- PDB parsing -----------------------------------------------------------
@dataclass
class PdbAtom:
    serial: int           # original 1-based PDB serial
    record: str           # ATOM or HETATM
    name: str             # atom name, stripped
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    occ: float
    bfac: float
    element: str
    charge_field: str     # raw two-char charge tail, e.g. "2+", "1-", ""


def parse_pdb(path: Path) -> tuple[list[PdbAtom], list[str]]:
    atoms: list[PdbAtom] = []
    remarks: list[str] = []
    with path.open() as fh:
        for line in fh:
            if line.startswith(("ATOM  ", "HETATM")):
                rec = line[:6].strip()
                serial = int(line[6:11])
                name = line[12:16].strip()
                altloc = line[16:17]
                resname = line[17:20].strip()
                chain = line[21:22].strip()
                resseq = int(line[22:26])
                icode = line[26:27]
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                occ = float(line[54:60]) if line[54:60].strip() else 1.0
                bfac = float(line[60:66]) if line[60:66].strip() else 0.0
                element = line[76:78].strip()
                charge = line[78:80].strip() if len(line) >= 80 else ""
                atoms.append(PdbAtom(
                    serial=serial, record=rec, name=name, altloc=altloc,
                    resname=resname, chain=chain, resseq=resseq, icode=icode,
                    x=x, y=y, z=z, occ=occ, bfac=bfac, element=element,
                    charge_field=charge,
                ))
            elif line.startswith("REMARK"):
                remarks.append(line.rstrip("\n"))
    return atoms, remarks


def write_xyz(atoms: list[PdbAtom], out_path: Path, comment: str = "") -> None:
    lines = [f"{len(atoms)}", comment]
    for a in atoms:
        lines.append(f"{a.element:<2s} {a.x:>14.8f} {a.y:>14.8f} {a.z:>14.8f}")
    out_path.write_text("\n".join(lines) + "\n")


# ----- CLI parsing helpers (matching polish_ts_v3 / scan_along_s) -----------
def _csv_int(s: str) -> list[int]:
    """Parse '131,254' → [131, 254]; empty/whitespace → []. Mirrors
    polish_ts_v3._csv_int so --free-residues / --prune-backbone-residues syntax
    is identical across all four pipeline tools."""
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_keep_specs(specs: list[str]) -> dict[int, list[str]]:
    """Parse repeated --prune-residue-keep RESID:ATOMS tokens into
    {resid: [atom_names]}. Mirrors polish_ts_v3._parse_keep_specs.

    Examples:
      ['169:CD,CE,NZ']           → {169: ['CD', 'CE', 'NZ']}
      ['131:', '169:CG,CD,OE1']  → {131: [], 169: ['CG', 'CD', 'OE1']}
                                    (empty list = drop ALL heavy atoms)
    """
    out: dict[int, list[str]] = {}
    for spec in specs or []:
        if ":" not in spec:
            raise ValueError(
                f"invalid --prune-residue-keep {spec!r} (need RESID:ATOMS, "
                "e.g. '169:CD,CE,NZ' or '131:' to drop all)"
            )
        resid_s, atoms_s = spec.split(":", 1)
        out[int(resid_s)] = [a.strip() for a in atoms_s.split(",") if a.strip()]
    return out


def _crest_atoms_from_lineage(lineage_atoms, coords) -> list[PdbAtom]:
    """Convert a structure_io.StructureLineage's atom list (post-prune) +
    coords ndarray into the local crest_funnel PdbAtom dataclass list (which
    is structurally similar but distinct — local class has 'record' field).

    The lineage atom carries chain/resseq/resname/name/element and a 'raw'
    line; we use those to construct the crest_funnel PdbAtom. Charge field
    on cap-H atoms is empty.
    """
    out: list[PdbAtom] = []
    for i, (la, xyz) in enumerate(zip(lineage_atoms, coords), start=1):
        # lineage 'raw' line — use HETATM if it doesn't start with ATOM
        rec = "ATOM" if (la.raw or "").startswith("ATOM  ") else \
              ("HETATM" if (la.raw or "").startswith("HETATM") else "ATOM")
        out.append(PdbAtom(
            serial=i, record=rec, name=la.name, altloc=la.altloc,
            resname=la.resname, chain=la.chain, resseq=la.resseq,
            icode=la.icode,
            x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
            occ=la.occ, bfac=la.bfac, element=la.element,
            charge_field=la.charge_field,
        ))
    return out


def apply_pruning_to_pdb(
    src_pdb: Path,
    out_pdb: Path,
    *,
    prune_residue_keep: dict[int, list[str]] | None = None,
    prune_backbone_residues: list[int] | None = None,
    cap_h_bond: float = 1.09,
    do_xtb_relax: bool = True,
    xtb_max_steps: int = 100,
    xtb_charge: int = 0,
) -> tuple[Path, int]:
    """Apply --prune-residue-keep / --prune-backbone-residues to ``src_pdb``,
    place H-caps at cut bonds, optionally xTB-relax those H-caps, and write
    a new PDB at ``out_pdb``.

    Reuses ``prune_utils.apply_prune_with_caps`` for chemistry parity with
    polish_ts_v3.

    Returns (out_pdb, n_atoms_after_prune). If no pruning was requested the
    src is just copied to dst and the original atom count is returned.
    """
    prune_residue_keep = prune_residue_keep or {}
    prune_backbone_residues = list(prune_backbone_residues or [])
    if not prune_residue_keep and not prune_backbone_residues:
        if src_pdb.resolve() != out_pdb.resolve():
            shutil.copy2(src_pdb, out_pdb)
        atoms_chk, _ = parse_pdb(src_pdb)
        return out_pdb, len(atoms_chk)

    # Defer-import so crest_funnel doesn't pay the structure_io cost when
    # pruning is not used.
    QCB_TOOLS = Path(__file__).resolve().parent
    if str(QCB_TOOLS) not in sys.path:
        sys.path.insert(0, str(QCB_TOOLS))
    from structure_io import parse_pdb_lineage, write_pdb_lineage  # type: ignore
    from prune_utils import apply_prune_with_caps  # type: ignore

    lineage = parse_pdb_lineage(src_pdb)
    new_lineage, new_coords, hp_indices = apply_prune_with_caps(
        lineage,
        prune_keep_specs=prune_residue_keep,
        prune_backbone_residues=prune_backbone_residues,
        cap_h_bond=cap_h_bond,
        do_xtb_relax=do_xtb_relax,
        xtb_max_steps=xtb_max_steps,
        xtb_charge=xtb_charge,
        xtb_solvent=None,    # gas phase for cap relax
    )
    write_pdb_lineage(new_lineage, new_coords, out_pdb,
                      drop_old_qcb=False, preserve_other_remarks=True,
                      title=f"crest_funnel pruned (HP-caps={len(hp_indices)})")
    return out_pdb, len(new_lineage.atoms)


# ----- constraint-file generation -------------------------------------------
def write_xtb_constraints(
    out_path: Path,
    fix_atoms: list[int],
    distance_constraints: list[tuple[int, int, float]],
    fix_force: float = 1.0,
) -> None:
    """Write an xtb-format constraint file.

    fix_atoms: 1-based xtb atom indices to freeze entirely.
    distance_constraints: list of (i, j, target_in_angstrom).
    """
    lines = []
    if distance_constraints:
        lines.append("$constrain")
        lines.append(f"  force constant={fix_force:.3f}")
        for i, j, d in distance_constraints:
            lines.append(f"  distance: {i}, {j}, {d:.4f}")
    if fix_atoms:
        lines.append("$fix")
        lines.append(f"  atoms: {_compact_indices(sorted(fix_atoms))}")
    lines.append("$end")
    out_path.write_text("\n".join(lines) + "\n")


def _compact_indices(idx: list[int]) -> str:
    """[1,2,3,5,6,9] -> '1-3,5-6,9'."""
    if not idx:
        return ""
    out = []
    start = prev = idx[0]
    for x in idx[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = x
    out.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(out)


# ----- structure partitioning ------------------------------------------------
@dataclass
class Partition:
    no_waters: list[PdbAtom]            # everything except HOH
    waters: list[PdbAtom]               # only HOH
    fix_indices_no_waters: list[int]    # 1-based, into no_waters list (CA + Zn)
    p_idx_no_waters: int                # 1-based serial of P1 in no_waters
    onuc_idx_no_waters: int             # 1-based serial of nucleophile O
    olg_idx_no_waters: int              # 1-based serial of leaving-group O
    d_p_onuc: float
    d_p_olg: float
    charge: int


def partition_pdb(atoms: list[PdbAtom], charge: int,
                   freeze_zn: bool = False,
                   free_residues: list[int] | set[int] | None = None) -> Partition:
    """Partition the input atoms into no-waters / waters and identify $fix
    anchors (CAs and optionally Zn) plus the reactive triplet.

    Args:
        atoms: parsed PDB atom list.
        charge: net molecular charge of the no-waters cluster.
        freeze_zn: also $fix Zn atoms (default False — let xtb relax Zn).
        free_residues: chain-A residue ids to EXCLUDE from the CA-rigid scaffold
            (their CAs are removed from ``fix_indices_no_waters``). Identical
            semantics to scan_along_s and polish_ts_v3 ``--free-residues``.
    """
    free_residues_set: set[int] = set(free_residues or [])

    no_waters: list[PdbAtom] = []
    waters: list[PdbAtom] = []
    for a in atoms:
        if a.resname in ("HOH", "WAT"):
            waters.append(a)
        else:
            no_waters.append(a)

    # Build (chain,resseq,resname,name) -> index_in_no_waters (1-based)
    p_idx = onuc_idx = olg_idx = -1
    fix_indices: list[int] = []
    for i, a in enumerate(no_waters, start=1):
        # CA in chain A protein residues — UNLESS in --free-residues
        if a.chain == "A" and a.name == "CA" and a.resseq not in free_residues_set:
            fix_indices.append(i)
        # Zn metals — only if explicitly requested. Default: let the QM method
        # find the metal coordination geometry (both GFN2-xTB and g-xTB
        # parametrize Zn(II) and handle five-coordinate phosphoryl TS well).
        if freeze_zn and a.resname == "ZN2" and a.element == "ZN":
            fix_indices.append(i)
        # P1, O3(nucleophile, in OHX), O7(leaving-group, in SUB)
        if a.resname == "SUB" and a.name == "P1":
            p_idx = i
        if a.resname == "OHX" and a.name == "O3":
            onuc_idx = i
        if a.resname == "SUB" and a.name == "O7":
            olg_idx = i

    if p_idx < 0 or onuc_idx < 0 or olg_idx < 0:
        raise RuntimeError(
            f"failed to find reactive atoms: P1={p_idx}, O3(OHX)={onuc_idx}, "
            f"O7(SUB)={olg_idx}"
        )

    def dist(i: int, j: int) -> float:
        a, b = no_waters[i - 1], no_waters[j - 1]
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    return Partition(
        no_waters=no_waters,
        waters=waters,
        fix_indices_no_waters=fix_indices,
        p_idx_no_waters=p_idx,
        onuc_idx_no_waters=onuc_idx,
        olg_idx_no_waters=olg_idx,
        d_p_onuc=dist(p_idx, onuc_idx),
        d_p_olg=dist(p_idx, olg_idx),
        charge=charge,
    )


def charge_from_filename(path: Path) -> int:
    name = path.stem
    if "netCHG_plus_" in name:
        return int(name.split("netCHG_plus_")[1].split("_")[0])
    if "netCHG_minus_" in name:
        return -int(name.split("netCHG_minus_")[1].split("_")[0])
    raise ValueError(f"cannot infer charge from filename: {path.name}")


# ----- multi-frame XYZ (CREST output) ---------------------------------------
def split_multiframe_xyz(path: Path) -> list[tuple[float, list[str]]]:
    """Return list of (energy_in_Eh, xyz_block_lines_excluding_count_and_comment)."""
    text = path.read_text().splitlines()
    out: list[tuple[float, list[str]]] = []
    i = 0
    while i < len(text):
        if not text[i].strip():
            i += 1
            continue
        n = int(text[i].strip())
        comment = text[i + 1].strip()
        try:
            energy = float(comment.split()[0])
        except Exception:
            energy = float("nan")
        body = text[i + 2 : i + 2 + n]
        out.append((energy, body))
        i += 2 + n
    return out


def write_xyz_block(n_atoms: int, body: list[str], path: Path, comment: str = "") -> None:
    lines = [str(n_atoms), comment, *body]
    path.write_text("\n".join(lines) + "\n")


# ----- subprocess helpers ----------------------------------------------------
def run_cmd(cmd: list[str], cwd: Path, log_path: Path, env: dict | None = None,
            timeout: float | None = None) -> int:
    log.info("run: %s  (cwd=%s)", " ".join(str(c) for c in cmd), cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_path.open("w") as fh:
        fh.write("# " + " ".join(str(c) for c in cmd) + "\n")
        fh.write(f"# cwd={cwd}\n")
        fh.flush()
        proc = subprocess.run(
            [str(c) for c in cmd], cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT,
            env=env, timeout=timeout,
        )
    dt = time.time() - t0
    log.info("  → exit=%d  (%.1fs)", proc.returncode, dt)
    return proc.returncode


def run_cmd_graceful(cmd: list[str], cwd: Path, log_path: Path,
                     env: dict | None = None, timeout: float | None = None,
                     sigterm_grace_s: float = 30.0) -> tuple[int, bool]:
    """Run a subprocess with a graceful timeout: on timeout send SIGTERM (so xtb
    has a chance to flush ``xtbopt.xyz``), wait ``sigterm_grace_s`` for it to
    exit, then escalate to SIGKILL. Process group is created so all xtb child
    threads die together — no orphan workers eating CPU after the parent goes.

    Returns (exit_code, timed_out). exit_code is -SIGTERM/-SIGKILL on a
    timeout-driven kill; ``timed_out`` is True iff we hit the wall clock.
    The caller is expected to inspect any partial output files (e.g. xtbopt.xyz)
    if they want to salvage a partial result.
    """
    log.info("run: %s  (cwd=%s)", " ".join(str(c) for c in cmd), cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    timed_out = False
    rc = -1
    with log_path.open("w") as fh:
        fh.write("# " + " ".join(str(c) for c in cmd) + "\n")
        fh.write(f"# cwd={cwd}\n")
        fh.flush()
        proc = subprocess.Popen(
            [str(c) for c in cmd], cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT,
            env=env, start_new_session=True,
        )
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            log.warning("  timeout after %.1fs — sending SIGTERM (grace %.0fs) "
                        "to let xtb flush partial xtbopt.xyz",
                        time.time() - t0, sigterm_grace_s)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                rc = proc.wait(timeout=sigterm_grace_s)
            except subprocess.TimeoutExpired:
                log.warning("  process did not exit on SIGTERM; escalating to SIGKILL")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    rc = proc.wait(timeout=sigterm_grace_s)
                except subprocess.TimeoutExpired:
                    rc = -signal.SIGKILL
            fh.write(f"\n# TIMEOUT: wall time {timeout}s exceeded; "
                     f"signaled SIGTERM, exit={rc}\n")
            fh.flush()
    dt = time.time() - t0
    log.info("  → exit=%d  timed_out=%s  (%.1fs)", rc, timed_out, dt)
    return rc, timed_out


def _read_last_gradient_norm(log_path: Path) -> float:
    """Parse the LAST 'gradient norm' line from an xtb log. Returns inf if not
    found. Used to evaluate whether a timed-out optimization is 'good enough'
    to salvage as a partially-converged geometry.
    """
    if not log_path.exists():
        return float("inf")
    try:
        text = log_path.read_text()
    except OSError:
        return float("inf")
    last = None
    for m in re.finditer(r"gradient norm\s*:\s*([0-9.]+)\s*Eh", text):
        last = m.group(1)
    if last is None:
        return float("inf")
    try:
        return float(last)
    except ValueError:
        return float("inf")


# Compact xtbopt.log format (xtb 6.7.x) per geom step:
#   energy: -348.466 gnorm: 0.596 xtb: 6.7.1 (eab23de) iter: 1
# We tolerate scientific notation, stray whitespace, and partial/garbled lines
# (sidecar may read mid-write). Anything that doesn't match falls through.
_XTBOPT_LOG_LINE_RE = re.compile(
    r"^\s*energy:\s*(-?\d+\.\d+(?:[eE][+-]?\d+)?)"
    r"\s+gnorm:\s*(\d+\.\d+(?:[eE][+-]?\d+)?)"
    r".*?iter:\s*(\d+)\s*$"
)
# Verbose xtb.out per-step format (output by every "ANCOPT/RFO" cycle):
#   gradient norm :     0.0023456 Eh/α   predicted   ...
_XTBOUT_GNORM_RE = re.compile(
    r"gradient norm\s*:\s*(\d+\.\d+(?:[eE][+-]?\d+)?)"
)


def parse_xtbopt_log(path: Path,
                     opt_level: str | None = None,
                     trend_window: int = PROGRESS_TREND_WINDOW_DEFAULT,
                     trend_slope_threshold: float = PROGRESS_TREND_SLOPE_THRESHOLD,
                     ) -> dict:
    """Parse an xtbopt.log (compact geom-step format) and return a progress
    summary dict.

    Robust to:
      - missing/empty file (returns step_count=0 with safe defaults)
      - mid-write garbled lines (skipped silently — re.match handles them)
      - missing iter index (we just skip those lines)
      - non-numeric tokens (caught by the regex's float pattern)

    ``opt_level`` is used to look up the target RMS gradient (Eh/Bohr) from
    ``XTB_OPT_TARGET_GRAD_RMS``. If None or unknown, falls back to xtb's
    'normal' default (1e-3 Eh/Bohr).

    Returns dict with the following keys (all guaranteed present, even on
    empty input — values become 0 / None / inf as appropriate):
        step_count: int
        energies: list[float]
        grad_max:  list[float]   (NB: xtbopt.log only stores gnorm, so we
                                  duplicate the gnorm column here for
                                  schema-compatibility with the spec.)
        grad_rms:  list[float]
        delta_E:   list[float]
        current_grad_rms: float
        current_grad_max: float
        current_step: int
        rate_of_decrease: float  (slope of log10(grad_rms) over last N steps;
                                  more negative = converging faster)
        trending_down: bool
        target_grad_rms: float
        ratio_to_target: float   (current / target; <1 = converged)
    """
    target = XTB_OPT_TARGET_GRAD_RMS.get(
        opt_level or "", XTB_OPT_TARGET_GRAD_RMS["normal"])
    empty = {
        "step_count": 0,
        "energies": [],
        "grad_max": [],
        "grad_rms": [],
        "delta_E": [],
        "current_grad_rms": float("inf"),
        "current_grad_max": float("inf"),
        "current_step": 0,
        "rate_of_decrease": 0.0,
        "trending_down": False,
        "target_grad_rms": target,
        "ratio_to_target": float("inf"),
    }
    if not path.exists():
        return empty
    try:
        # Use errors="replace" so a half-written line doesn't crash decoding.
        text = path.read_text(errors="replace")
    except OSError:
        return empty
    energies: list[float] = []
    gnorms: list[float] = []
    iters: list[int] = []
    for line in text.splitlines():
        m = _XTBOPT_LOG_LINE_RE.match(line)
        if not m:
            continue
        try:
            e = float(m.group(1))
            g = float(m.group(2))
            it = int(m.group(3))
        except (ValueError, TypeError):
            continue
        energies.append(e)
        gnorms.append(g)
        iters.append(it)
    if not gnorms:
        return empty
    # delta_E per step (first step's delta is 0)
    deltas = [0.0] + [energies[i] - energies[i - 1]
                      for i in range(1, len(energies))]
    # Trend: log10(grad_rms) slope over the last `trend_window` points.
    rate = 0.0
    if len(gnorms) >= 2:
        n_tail = min(trend_window, len(gnorms))
        tail = gnorms[-n_tail:]
        if all(g > 0 for g in tail):
            xs = np.arange(n_tail, dtype=float)
            ys = np.log10(np.array(tail, dtype=float))
            # least-squares slope: m = cov(x,y)/var(x)
            x_mean = xs.mean()
            y_mean = ys.mean()
            denom = float(((xs - x_mean) ** 2).sum())
            if denom > 0:
                rate = float(((xs - x_mean) * (ys - y_mean)).sum() / denom)
    current = gnorms[-1]
    return {
        "step_count": len(gnorms),
        "energies": energies,
        "grad_max": list(gnorms),  # gnorm proxy; xtbopt.log doesn't expose max separately
        "grad_rms": list(gnorms),
        "delta_E": deltas,
        "current_grad_rms": current,
        "current_grad_max": current,
        "current_step": iters[-1] if iters else len(gnorms),
        "rate_of_decrease": rate,
        "trending_down": rate < trend_slope_threshold,
        "target_grad_rms": target,
        "ratio_to_target": current / target if target > 0 else float("inf"),
    }


def run_cmd_with_progress_monitor(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    *,
    env: dict | None = None,
    timeout: float | None = None,
    sigterm_grace_s: float = 30.0,
    opt_level: str | None = None,
    progress_check_fraction: float = 0.6,
    progress_grad_ratio: float = 2.0,
    salvage_grad_ratio: float = 1.5,
    max_extensions: int = 1,
    extension_factor: float = 1.0,
    poll_interval_s: float = PROGRESS_POLL_INTERVAL_S_DEFAULT,
    progress_log_path: Path | None = None,
) -> tuple[int, bool, int, bool, float | None]:
    """Strategy E adaptive timeout + progress monitor for an xtb optimization.

    Behaviour:
      1. Spawn ``cmd`` in its own process group.
      2. A sidecar thread polls ``cwd/xtbopt.log`` every ``poll_interval_s``.
      3. At ``progress_check_fraction × timeout`` elapsed (default 60%): if
         the current grad-RMS is within ``progress_grad_ratio × target`` AND
         the run is trending down, extend the deadline by
         ``extension_factor × original_timeout`` (default 1.0× = double the
         total budget). Up to ``max_extensions`` extensions per call.
      4. On hard kill (SIGTERM → SIGKILL), if a partial xtbopt.xyz exists
         and the final grad-RMS is within ``salvage_grad_ratio × target``,
         the geometry is treated as salvaged.
      5. Per-step convergence history is written to ``progress_log_path`` if
         provided (one JSON object per call).

    Returns ``(rc, timed_out, extensions_used, salvaged, final_grad_ratio)``.
    ``final_grad_ratio`` is ``None`` if no xtbopt.log was readable.

    Backwards-compatibility note: with default flags and an opt_level whose
    target leaves ``progress_grad_ratio`` and ``salvage_grad_ratio`` matching
    the previous task #25 behaviour, this is a strict superset — call sites
    can still call ``run_cmd_graceful`` and ignore the new args entirely.
    """
    log.info("run: %s  (cwd=%s)", " ".join(str(c) for c in cmd), cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    extensions_used = 0
    timed_out = False
    salvaged = False
    rc = -1
    convergence_history: list[dict] = []

    if timeout is None:
        # No adaptive logic without a base timeout — just run.
        rc = run_cmd(cmd, cwd, log_path, env=env, timeout=None)
        return rc, False, 0, False, None

    base_timeout = float(timeout)
    deadline = t0 + base_timeout
    check_at = t0 + progress_check_fraction * base_timeout

    stop_event = threading.Event()
    state_lock = threading.Lock()
    state = {
        "deadline": deadline,
        "checked": False,
        "extensions_used": 0,
        "last_progress": None,  # last parsed progress dict
    }

    xtbopt_log = cwd / "xtbopt.log"

    def _sidecar() -> None:
        """Polls xtbopt.log; once past check_at, decides whether to extend."""
        while not stop_event.is_set():
            now = time.time()
            try:
                progress = parse_xtbopt_log(xtbopt_log, opt_level=opt_level)
            except Exception as exc:  # never crash the worker
                log.debug("progress sidecar parse error: %s", exc)
                progress = None
            if progress is not None:
                with state_lock:
                    state["last_progress"] = progress
                    if progress["step_count"] > 0:
                        convergence_history.append({
                            "wall_s": round(now - t0, 2),
                            "step": progress["current_step"],
                            "grad_rms": progress["current_grad_rms"],
                            "ratio_to_target": progress["ratio_to_target"],
                            "trending_down": progress["trending_down"],
                            "rate_of_decrease": progress["rate_of_decrease"],
                        })
            with state_lock:
                already_checked = state["checked"]
                cur_deadline = state["deadline"]
                ext_count = state["extensions_used"]
            if (not already_checked) and now >= check_at:
                with state_lock:
                    state["checked"] = True
                if progress is None or progress["step_count"] == 0:
                    log.info("Stage progress check (%s): no usable xtbopt.log "
                             "yet — falling through to hard timeout",
                             cwd.name)
                else:
                    ratio = progress["ratio_to_target"]
                    trending = progress["trending_down"]
                    rate = progress["rate_of_decrease"]
                    if (ratio < progress_grad_ratio and trending
                            and ext_count < max_extensions):
                        new_deadline = cur_deadline + extension_factor * base_timeout
                        with state_lock:
                            state["deadline"] = new_deadline
                            state["extensions_used"] = ext_count + 1
                        log.info(
                            "%s: extending timeout %.0fs -> %.0fs "
                            "(grad_ratio=%.2f, rate=%+.3f/step, trending_down=True)",
                            cwd.name, cur_deadline - t0, new_deadline - t0,
                            ratio, rate)
                    else:
                        log.info(
                            "%s: NOT extending (grad_ratio=%.2f, rate=%+.3f/step, "
                            "trending_down=%s, ext_used=%d/%d)",
                            cwd.name, ratio, rate, trending,
                            ext_count, max_extensions)
            if stop_event.wait(timeout=poll_interval_s):
                return

    sidecar = threading.Thread(target=_sidecar, name=f"progress-{cwd.name}",
                               daemon=True)
    with log_path.open("w") as fh:
        fh.write("# " + " ".join(str(c) for c in cmd) + "\n")
        fh.write(f"# cwd={cwd}\n")
        fh.flush()
        proc = subprocess.Popen(
            [str(c) for c in cmd], cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT,
            env=env, start_new_session=True,
        )
        sidecar.start()
        try:
            # Loop in small slices so a sidecar-driven extension takes effect.
            poll_slice = max(1.0, poll_interval_s / 2.0)
            while True:
                with state_lock:
                    cur_deadline = state["deadline"]
                remaining = cur_deadline - time.time()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(cmd, timeout=cur_deadline - t0)
                wait_for = min(poll_slice, remaining)
                try:
                    rc = proc.wait(timeout=wait_for)
                    break  # process exited normally (or with non-zero rc)
                except subprocess.TimeoutExpired:
                    # Re-check the (possibly extended) deadline and loop again.
                    continue
        except subprocess.TimeoutExpired:
            timed_out = True
            log.warning("  timeout after %.1fs — sending SIGTERM (grace %.0fs) "
                        "to let xtb flush partial xtbopt.xyz",
                        time.time() - t0, sigterm_grace_s)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                rc = proc.wait(timeout=sigterm_grace_s)
            except subprocess.TimeoutExpired:
                log.warning("  process did not exit on SIGTERM; escalating to SIGKILL")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    rc = proc.wait(timeout=sigterm_grace_s)
                except subprocess.TimeoutExpired:
                    rc = -signal.SIGKILL
            fh.write(f"\n# TIMEOUT: wall time exceeded; signaled SIGTERM, exit={rc}\n")
            fh.flush()
        finally:
            stop_event.set()
            sidecar.join(timeout=poll_interval_s + 2.0)
    dt = time.time() - t0
    with state_lock:
        extensions_used = state["extensions_used"]

    # Final salvage check: re-parse xtbopt.log one more time post-mortem.
    final_progress = parse_xtbopt_log(xtbopt_log, opt_level=opt_level)
    final_grad_ratio: float | None = None
    if final_progress["step_count"] > 0:
        final_grad_ratio = final_progress["ratio_to_target"]
    if timed_out and (cwd / "xtbopt.xyz").exists() and final_grad_ratio is not None:
        if final_grad_ratio < salvage_grad_ratio:
            salvaged = True
            log.info("%s: SALVAGED partial geometry (grad_ratio=%.2f < %.2f)",
                     cwd.name, final_grad_ratio, salvage_grad_ratio)

    if progress_log_path is not None:
        try:
            payload = {
                "cwd": str(cwd),
                "cmd": [str(c) for c in cmd],
                "timeout_base_s": base_timeout,
                "extensions_used": extensions_used,
                "wall_s": round(dt, 2),
                "rc": rc,
                "timed_out": bool(timed_out),
                "salvaged": bool(salvaged),
                "final_grad_ratio_to_target": final_grad_ratio,
                "final_step_count": final_progress["step_count"],
                "final_grad_rms": final_progress["current_grad_rms"]
                                  if final_progress["step_count"] > 0 else None,
                "target_grad_rms": final_progress["target_grad_rms"],
                "opt_level": opt_level,
                "progress_check_fraction": progress_check_fraction,
                "progress_grad_ratio": progress_grad_ratio,
                "salvage_grad_ratio": salvage_grad_ratio,
                "max_extensions": max_extensions,
                "extension_factor": extension_factor,
                "convergence_history": convergence_history,
            }
            progress_log_path.parent.mkdir(parents=True, exist_ok=True)
            progress_log_path.write_text(json.dumps(payload, indent=2,
                                                    default=lambda o: None))
        except OSError as exc:
            log.warning("could not write progress log %s: %s",
                        progress_log_path, exc)

    log.info("  → exit=%d  timed_out=%s  ext=%d  salvaged=%s  "
             "final_grad_ratio=%s  (%.1fs)",
             rc, timed_out, extensions_used, salvaged,
             f"{final_grad_ratio:.3f}" if final_grad_ratio is not None else "None",
             dt)
    return rc, timed_out, extensions_used, salvaged, final_grad_ratio


# ----- stages ---------------------------------------------------------------
def stage_0_input(
    src_pdb: Path,
    out_root: Path,
    freeze_zn: bool = False,
    *,
    free_residues: list[int] | None = None,
    prune_residue_keep: dict[int, list[str]] | None = None,
    prune_backbone_residues: list[int] | None = None,
    cap_h_bond: float = 1.09,
    prune_xtb_relax: bool = True,
    prune_xtb_max_steps: int = 100,
    charge_override: int | None = None,
) -> Partition:
    """Parse the source PDB + (optionally) prune residues with H-cap placement,
    then partition into no_waters / waters.

    The original input is preserved as ``00_input/original_source.pdb``.
    The (possibly pruned) cluster is written as ``00_input/source.pdb`` —
    that is the file every subsequent stage (A, B, C, DE) consumes.

    Args:
        src_pdb: input PDB with ATOM/HETATM + REMARK records.
        out_root: top-level output directory.
        freeze_zn: also $fix Zn metals (default False).
        free_residues: chain-A residue ids to exclude from the CA-rigid
            ``$fix atoms`` block (their CAs are not pinned).
        prune_residue_keep: ``{resid: [atom_names]}`` — keep only those
            heavy atoms; drop the rest. H-caps are placed at cut bonds.
            Empty list drops ALL heavy atoms in the residue.
        prune_backbone_residues: residue ids to backbone-strip
            (drop N/C/O/HXT/H/HA/H2/H3, keep sidechain). H-caps placed
            at cut bonds.
        cap_h_bond: initial bond length (Å) of cap H atoms placed at cut
            bonds. Default 1.09 Å (typical C-H).
        prune_xtb_relax: after H-cap placement, run a quick GFN2-xTB
            partial-relax of just the cap H atoms (everything else fixed).
            Default True.
        prune_xtb_max_steps: max optimizer steps for the cap-H relax.
        charge_override: if not None, use this charge instead of inferring
            from the filename (used when the prune changes the net charge).
    """
    free_residues = list(free_residues or [])
    prune_residue_keep = dict(prune_residue_keep or {})
    prune_backbone_residues = list(prune_backbone_residues or [])
    log.info("--- Stage 0: parse + partition "
             "(freeze_zn=%s, free_residues=%s, prune_keep=%s, "
             "prune_backbone=%s)",
             freeze_zn, sorted(free_residues),
             list(prune_residue_keep.keys()), prune_backbone_residues)
    d = out_root / "00_input"
    d.mkdir(parents=True, exist_ok=True)

    # Always preserve the original input — stamp lineage even if no prune.
    shutil.copy2(src_pdb, d / "original_source.pdb")

    # Charge: filename inference unless explicitly overridden.
    if charge_override is not None:
        chrg = int(charge_override)
    else:
        chrg = charge_from_filename(src_pdb)

    # Step 1: optional prune (H-caps). When both prune flags are empty this
    # just copies the input to source.pdb and returns the original atom count.
    pruned_pdb = d / "source.pdb"
    final_pdb_for_partition, n_after_prune = apply_pruning_to_pdb(
        src_pdb=src_pdb,
        out_pdb=pruned_pdb,
        prune_residue_keep=prune_residue_keep,
        prune_backbone_residues=prune_backbone_residues,
        cap_h_bond=cap_h_bond,
        do_xtb_relax=prune_xtb_relax,
        xtb_max_steps=prune_xtb_max_steps,
        xtb_charge=chrg,
    )
    if prune_residue_keep or prune_backbone_residues:
        log.info("  pruning + H-caps: %d atoms after prune (was %d before)",
                 n_after_prune,
                 len(parse_pdb(d / "original_source.pdb")[0]))

    atoms, _ = parse_pdb(final_pdb_for_partition)
    part = partition_pdb(atoms, chrg, freeze_zn=freeze_zn,
                         free_residues=free_residues)
    write_xyz(part.no_waters, d / "no_waters.xyz",
              comment=f"no_waters charge={part.charge}")
    write_xyz(part.waters, d / "waters.xyz",
              comment=f"waters_only n={len(part.waters)}")
    summary = {
        "source": str(src_pdb),
        "n_atoms_total": len(atoms),
        "n_atoms_no_waters": len(part.no_waters),
        "n_water_atoms": len(part.waters),
        "charge": part.charge,
        "p_idx_no_waters_1based": part.p_idx_no_waters,
        "onuc_idx_no_waters_1based": part.onuc_idx_no_waters,
        "olg_idx_no_waters_1based": part.olg_idx_no_waters,
        "d_p_onuc_A": part.d_p_onuc,
        "d_p_olg_A": part.d_p_olg,
        "n_fix_atoms_no_waters": len(part.fix_indices_no_waters),
        "fix_indices_no_waters_1based": part.fix_indices_no_waters,
        # NEW lineage: free + prune knobs are now persisted so finalize / debug
        # can verify what the partition actually was.
        "free_residues": sorted(free_residues),
        "prune_residue_keep": {str(k): v for k, v in prune_residue_keep.items()},
        "prune_backbone_residues": prune_backbone_residues,
    }
    (d / "partition.json").write_text(json.dumps(summary, indent=2))
    log.info("  total=%d  no_waters=%d  waters=%d  charge=%+d",
             len(atoms), len(part.no_waters), len(part.waters), part.charge)
    log.info("  P=%d  Onuc=%d  Olg=%d  d(P-Onuc)=%.3f  d(P-Olg)=%.3f",
             part.p_idx_no_waters, part.onuc_idx_no_waters, part.olg_idx_no_waters,
             part.d_p_onuc, part.d_p_olg)
    if free_residues:
        log.info("  --free-residues %s: %d residue(s) excluded from "
                 "$fix-CA scaffold (final fix_atoms=%d)",
                 sorted(free_residues), len(free_residues),
                 len(part.fix_indices_no_waters))
    return part


def _make_constraints_no_waters(part: Partition, out: Path) -> None:
    write_xtb_constraints(
        out_path=out,
        fix_atoms=part.fix_indices_no_waters,
        distance_constraints=[
            (part.p_idx_no_waters, part.onuc_idx_no_waters, part.d_p_onuc),
            (part.p_idx_no_waters, part.olg_idx_no_waters, part.d_p_olg),
        ],
        fix_force=1.0,
    )


# ----- optional pre-CREST relaxation (cleans up the source geometry) --------
# Pre-CREST relax options. The 'mace-*' choices route through ASE + the
# in-process MLFF calculator. The 'xtb-*' choices route through the vendored
# xtb binary so we don't pay the ASE startup cost for trivial cleanups.
PRE_CREST_RELAX_CHOICES = ("none", "xtb-loose", "xtb-tight",
                            "mace-mp", "mace-polar-m")

# Stage E (water-reinserted relax) backend options. xtb is the historical
# default (vacuum or ALPB); the 'mace-*' choices run an in-process ASE LBFGS
# with the corresponding MACE calculator (vacuum only — MACE has no implicit
# solvent model). All three respect the same FixAtoms-on-protein-heavy-atoms
# constraint so only the reinserted waters move.
STAGE_E_METHOD_CHOICES = ("xtb", "mace-mp", "mace-polar-m")

# Stage C (g-xTB / GFN2 minimize + SP rerank) backend options. ``xtb``
# (default, historical) routes through the vendored xtb binary with full
# Strategy-E adaptive-timeout / salvage stack, ``$fix`` over CA atoms, and
# ``$constrain distance:`` on the reactive P-Onuc / P-Olg pair, with optional
# ALPB solvent. ``mace-mp`` / ``mace-polar-m`` run an in-process ASE LBFGS
# with the corresponding MACE calculator (vacuum only — MACE has no implicit
# solvent model) and apply the equivalent FixAtoms + FixBondLengths constraints
# (matches stage_pre_crest_relax). Stage C still runs the g-xTB single-point
# rerank on the optimized geometry regardless of which backend produced it,
# so downstream ranking is apples-to-apples.
STAGE_C_METHOD_CHOICES = ("xtb", "mace-mp", "mace-polar-m")

# Module-level unit conversion constants. Promoted from local scope per
# codex 2026-05-07 review (#5) to prevent drift between Stage C / Stage E
# MACE branches and any future ASE-based stages.
EV_TO_EH = 1.0 / 27.211386245988          # 1 eV in Hartree
EV_PER_A_TO_EH_PER_BOHR = 0.019446904502  # 1 eV/Å in Eh/Bohr


def _xyz_uhf(xyz_path: Path, charge: int) -> int:
    """Return uhf such that (n_electrons - uhf) is even.

    xtb refuses to run when (--chrg, --uhf) imply an inconsistent electron
    parity; pruning a backbone residue can flip the parity of the cluster
    even if the formal charge is unchanged. This helper inspects the actual
    atomic composition of ``xyz_path`` (FIRST frame only — multi-frame XYZ
    files are tolerated) and picks uhf=0 or uhf=1 accordingly.
    """
    # Periodic table H..U (Z 1..92) — covers everything xtb supports plus a
    # bit of headroom. Capitalised "two-letter" forms only; we normalise the
    # input to title-case before lookup.
    z_table = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
        "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
        "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
        "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
        "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
        "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
        "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
        "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
        "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
        "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
        "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
        "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
        "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
    }
    lines = xyz_path.read_text().splitlines()
    try:
        n = int(lines[0].strip())
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"_xyz_uhf: malformed XYZ header in {xyz_path}: {exc}") from exc
    nel_neutral = 0
    # Read EXACTLY the first frame's n atoms (codex 2026-05-07 review).
    for line in lines[2:2 + n]:
        toks = line.split()
        if len(toks) < 4:
            continue
        sym = toks[0]
        # Normalise: e.g. "ZN" -> "Zn", "h" -> "H". Strip trailing digits
        # that some PDB-derived XYZ writers leave behind ("C1" -> "C").
        sym = "".join(ch for ch in sym if ch.isalpha())
        sym = sym[0].upper() + sym[1:].lower() if len(sym) > 1 else sym.upper()
        if sym not in z_table:
            raise ValueError(
                f"_xyz_uhf: unknown element symbol {sym!r} in {xyz_path}; "
                "extend z_table with the missing element."
            )
        nel_neutral += z_table[sym]
    nel = nel_neutral - charge
    return nel & 1


def stage_pre_crest_relax(
    out_root: Path,
    part: Partition,
    *,
    method: str = "xtb-loose",
    fmax: float = 0.05,
    max_steps: int = 200,
    ncpu: int = 1,
    solvent: str | None = "water",
    timeout_s: int = 1800,
    device: str = "cuda",
    optimizer_backend: str = "ase-lbfgs",
) -> Path:
    """Optional pre-CREST relaxation of the no-waters cluster.

    Runs BEFORE Stage A to give CREST a cleaner starting geometry. Uses the
    SAME constraint pattern as Stage A — ``$fix`` over CA atoms (minus
    --free-residues), ``$constrain distance:`` on the reactive
    P-Onuc / P-Olg pair. The relax updates the input that Stage A reads,
    so all downstream stages see the relaxed geometry.

    Args:
        out_root: top-level pipeline output directory.
        part: Stage-0 partition (carries fix_indices, reactive triplet, charge).
        method: relaxation backend (see ``PRE_CREST_RELAX_CHOICES``):
            'xtb-loose' — fast GFN2 with `--opt loose`
            'xtb-tight' — slower GFN2 with `--opt tight`
            'mace-mp' / 'mace-polar-m' — ASE LBFGS with MACE calculator
            'none' (or empty) — no-op; returns the existing 00_input/no_waters.xyz
        fmax: ASE-side fmax for MACE backends (eV/Å).
        max_steps: ASE-side max optimizer steps for MACE backends.
        ncpu: threads for xtb / mace.
        solvent: ALPB solvent (xtb backends) or None for gas phase.
        timeout_s: subprocess timeout for xtb backends (seconds).

    Side-effects:
        Writes 05_pre_crest_relax/relaxed.xyz on success and OVERWRITES
        00_input/no_waters.xyz with the relaxed geometry, so Stage A
        consumes the cleaner starting structure.

    Returns:
        Path to the relaxed XYZ.
    """
    method = (method or "none").lower()
    src_xyz = out_root / "00_input/no_waters.xyz"
    if method in ("none", ""):
        log.info("--- Pre-CREST relax: SKIPPED (method=%s)", method)
        return src_xyz
    if method not in PRE_CREST_RELAX_CHOICES:
        raise ValueError(
            f"unknown --pre-crest-relax {method!r}; choose from "
            f"{PRE_CREST_RELAX_CHOICES}"
        )

    log.info("--- Pre-CREST relax: method=%s fmax=%.4f max_steps=%d",
             method, fmax, max_steps)
    d = out_root / "05_pre_crest_relax"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_xyz, d / "input.xyz")
    _make_constraints_no_waters(part, d / "constraints.inp")

    if method.startswith("xtb-"):
        opt_level = "loose" if method == "xtb-loose" else "tight"
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = f"{ncpu},1"
        env["MKL_NUM_THREADS"] = str(ncpu)
        env["OMP_STACKSIZE"] = "4G"
        uhf_pre = _xyz_uhf(d / "input.xyz", part.charge)
        cmd = [str(XTB_BIN), "input.xyz",
               "--gfn", "2",
               "--opt", opt_level,
               "--cycles", str(max_steps),
               "--chrg", str(part.charge),
               "--uhf", str(uhf_pre),
               "--input", "constraints.inp"]
        if solvent:
            cmd += ["--alpb", solvent]
        # NOTE: run_cmd raises subprocess.TimeoutExpired on timeout — we catch
        # it explicitly so the "CONTINUING with original geometry" path
        # actually fires, instead of bubbling up to the wrapper. (codex flag
        # 2026-05-07).
        try:
            rc = run_cmd(cmd, cwd=d, log_path=d / "xtb.out", env=env,
                         timeout=timeout_s)
        except subprocess.TimeoutExpired:
            log.warning("pre-CREST xtb relax TIMED OUT after %ds; CONTINUING "
                        "with the ORIGINAL geometry (Stage A will still run on "
                        "the untouched input). See %s",
                        timeout_s, d / "xtb.out")
            return src_xyz
        if rc != 0:
            log.warning("pre-CREST xtb relax exited rc=%d; CONTINUING with the "
                        "ORIGINAL geometry (Stage A will still run on the "
                        "untouched input). See %s", rc, d / "xtb.out")
            return src_xyz
        relaxed = d / "xtbopt.xyz"
        if not relaxed.exists():
            log.warning("pre-CREST xtb relax left no xtbopt.xyz; CONTINUING "
                        "with the original geometry.")
            return src_xyz

        # Capture before/after energies for reporting (easy to diff)
        before_e = _read_xyz_comment_energy(d / "input.xyz")
        after_e = _read_xyz_comment_energy(relaxed)
        delta_kcal = (
            (after_e - before_e) * _HARTREE_TO_KCAL
            if (before_e is not None and after_e is not None) else None
        )
        log.info("  before E = %s   after E = %s  ΔE = %s kcal/mol",
                 f"{before_e:+.6f}" if before_e is not None else "n/a",
                 f"{after_e:+.6f}" if after_e is not None else "n/a",
                 f"{delta_kcal:+.3f}" if delta_kcal is not None else "n/a")

        shutil.copy2(relaxed, d / "relaxed.xyz")
        # Overwrite no_waters.xyz so Stage A picks up the relaxed geometry
        shutil.copy2(relaxed, src_xyz)
        log.info("  → %s (also updated %s)", d / "relaxed.xyz", src_xyz)
        return d / "relaxed.xyz"

    if method.startswith("mace-"):
        # ASE in-process path. Defer-import so non-MACE pipeline runs don't
        # pay the heavy import cost.
        from ase.io import read as ase_read, write as ase_write
        from ase.constraints import FixAtoms, FixBondLengths
        from ase.optimize import LBFGS
        from quantum_engine.calc import make_calc

        atoms = ase_read(str(d / "input.xyz"))
        atoms.info["charge"] = part.charge
        atoms.calc = make_calc(method, device=device, charge=part.charge)
        constraints = []
        # CA fix → 0-based indices (xtb is 1-based; ase is 0-based)
        ca0 = [i - 1 for i in part.fix_indices_no_waters]
        if ca0:
            constraints.append(FixAtoms(indices=ca0))
        # Reactive distances pinned at current values
        bondlength_pairs = [
            (part.p_idx_no_waters - 1, part.onuc_idx_no_waters - 1),
            (part.p_idx_no_waters - 1, part.olg_idx_no_waters - 1),
        ]
        constraints.append(FixBondLengths(bondlength_pairs))
        atoms.set_constraint(constraints)

        e0 = float(atoms.get_potential_energy())
        # The pre-CREST relax always carries FixBondLengths constraints,
        # which torch-sim FIRE can't honour. Refuse the swap silently
        # and stay on ASE LBFGS — never lose the constrained geometry.
        backend = optimizer_backend
        if backend.startswith("torch-sim") and constraints:
            log.warning(
                "  pre-crest-relax optimizer-backend=%s does not "
                "support FixBondLengths; falling back to ase-lbfgs.",
                backend,
            )
            backend = "ase-lbfgs"
        if backend == "ase-lbfgs":
            opt = LBFGS(atoms, logfile=str(d / "ase.log"))
            converged = opt.run(fmax=fmax, steps=max_steps)
        else:
            from quantum_engine.opt import make_optimizer
            opt_obj = make_optimizer(
                backend, fmax=fmax, max_steps=max_steps,
                logfile=d / "ase.log",
            )
            res = opt_obj.run(atoms)
            converged = res.converged
        e_final = float(atoms.get_potential_energy())
        log.info("  initial E = %.4f eV  final E = %.4f eV  "
                 "ΔE = %+.4f eV  converged=%s  backend=%s",
                 e0, e_final, e_final - e0, converged, backend)
        ase_write(str(d / "relaxed.xyz"), atoms, format="xyz")
        shutil.copy2(d / "relaxed.xyz", src_xyz)
        log.info("  → %s (also updated %s)", d / "relaxed.xyz", src_xyz)
        return d / "relaxed.xyz"

    raise RuntimeError(f"pre-crest-relax: unhandled method {method!r}")


def _read_xyz_comment_energy(xyz_path: Path) -> float | None:
    """Best-effort: pull a hartree energy from the comment line of an XYZ file
    (xtb writes ``<energy_in_Eh>  ...`` or ``energy: <Eh>``)."""
    try:
        toks = xyz_path.read_text().splitlines()[1].split()
        for t in toks:
            try:
                return float(t)
            except ValueError:
                continue
    except Exception:
        pass
    return None


def stage_A_xtb_preopt(out_root: Path, part: Partition, ncpu: int,
                       solvent: str | None = "water",
                       xtb_preopt_timeout: int = 3600) -> Path:
    log.info("--- Stage A: GFN2-xTB constrained pre-opt (no waters)")
    d = out_root / "10_xtb_preopt"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_root / "00_input/no_waters.xyz", d / "input.xyz")
    _make_constraints_no_waters(part, d / "constraints.inp")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = f"{ncpu},1"
    env["MKL_NUM_THREADS"] = str(ncpu)
    env["OMP_STACKSIZE"] = "4G"

    uhf = _xyz_uhf(d / "input.xyz", part.charge)
    log.info("  parity check: charge=%+d  uhf=%d", part.charge, uhf)
    cmd = [str(XTB_BIN), "input.xyz",
           "--gfn", "2",
           "--opt", "loose",
           "--chrg", str(part.charge),
           "--uhf", str(uhf),
           "--input", "constraints.inp"]
    if solvent:
        cmd += ["--alpb", solvent]

    rc = run_cmd(cmd, cwd=d, log_path=d / "xtb.out", env=env,
                  timeout=xtb_preopt_timeout)
    if rc != 0:
        raise RuntimeError(f"Stage A xtb failed (rc={rc}); see {d/'xtb.out'}")

    # xtb writes xtbopt.xyz on success
    src = d / "xtbopt.xyz"
    if not src.exists():
        raise RuntimeError(f"Stage A: missing xtbopt.xyz under {d}")
    shutil.copy2(src, d / "preopt.xyz")
    log.info("  → %s", d / "preopt.xyz")
    return d / "preopt.xyz"


def stage_B_crest(out_root: Path, part: Partition, ncpu: int,
                  solvent: str | None = "water", preset: str = "mquick",
                  mdlen_ps: float | None = 3.0, rthr: float = 0.05,
                  mddump_fs: int = 200, opt_level: str = "crude",
                  walltime_s: int = 7200) -> Path:
    log.info("--- Stage B: CREST --nci (no waters)  preset=%s mdlen=%s rthr=%s",
             preset, mdlen_ps, rthr)
    d = out_root / "20_crest"
    d.mkdir(parents=True, exist_ok=True)
    # Wipe stale conformer files so a CREST that exits non-zero can't be
    # mistaken for a successful rerun.
    for stale in ("crest_conformers.xyz", "crest_rotamers.xyz", "crest_best.xyz",
                  "crest_dynamics.trj", "crestopt.log"):
        (d / stale).unlink(missing_ok=True)
    shutil.copy2(out_root / "10_xtb_preopt/preopt.xyz", d / "input.xyz")
    _make_constraints_no_waters(part, d / "constraints.inp")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = f"{ncpu},1"
    env["MKL_NUM_THREADS"] = str(ncpu)
    env["OMP_STACKSIZE"] = "4G"

    uhf_b = _xyz_uhf(d / "input.xyz", part.charge)
    cmd = [str(CREST_BIN), "input.xyz",
           "--gfn2",
           "--chrg", str(part.charge),
           "--uhf", str(uhf_b),
           "--cinp", "constraints.inp",
           "--nci",
           "--noreftopo",            # TS geometries have stretched bonds at cutoff edge
           "-rthr", str(rthr),       # looser RMSD → more unique conformers
           "-mddump", str(mddump_fs),# fewer trajectory dumps → fewer opt jobs
           "-O", opt_level,          # multilevel opt level (crude is the cheapest)
           "-T", str(ncpu)]
    if preset and preset != "none":
        cmd += [f"-{preset}"]        # -quick / -squick / -mquick
    if mdlen_ps is not None:
        cmd += ["-mdlen", str(mdlen_ps)]
    if solvent:
        cmd += ["--alpb", solvent]

    rc = run_cmd(cmd, cwd=d, log_path=d / "crest.out", env=env, timeout=walltime_s)
    if rc != 0:
        log.warning("CREST exit=%d (continuing if conformers were written)", rc)

    confs = d / "crest_conformers.xyz"
    if not confs.exists():
        raise RuntimeError(f"Stage B: missing crest_conformers.xyz under {d}")

    frames = split_multiframe_xyz(confs)
    log.info("  CREST returned %d conformers", len(frames))
    with (d / "crest_energies.csv").open("w") as fh:
        w = csv.writer(fh)
        w.writerow(["rank_in_crest", "energy_Eh"])
        for i, (e, _) in enumerate(frames, start=1):
            w.writerow([i, f"{e:.8f}"])
    return confs


_HARTREE_TO_KCAL = 627.5094740631  # CODATA hartree -> kcal/mol


def _apply_outlier_cutoff(rows: list[dict], energy_key: str, ok_key: str,
                          cutoff_kcal: float | None, label: str) -> list[dict]:
    """Drop rows whose energy is more than ``cutoff_kcal`` kcal/mol above the
    minimum among successfully-optimized rows. Mutates ``rows`` in place by
    setting their ``ok_key`` to 0 and adds a ``"<energy_key>_outlier_drop": 1``
    flag. Returns the same list. Logs each drop. ``cutoff_kcal=None`` is a
    no-op.
    """
    if cutoff_kcal is None or cutoff_kcal <= 0:
        return rows
    energies_ok = [r[energy_key] for r in rows
                   if int(r.get(ok_key, 0)) == 1
                   and isinstance(r[energy_key], float)
                   and not math.isnan(r[energy_key])]
    if not energies_ok:
        return rows
    e_min = min(energies_ok)
    n_dropped = 0
    for r in rows:
        e = r.get(energy_key, float("nan"))
        if not isinstance(e, float) or math.isnan(e):
            continue
        if int(r.get(ok_key, 0)) != 1:
            continue
        delta_kcal = (e - e_min) * _HARTREE_TO_KCAL
        if delta_kcal > cutoff_kcal:
            log.info("  [%s] outlier-drop conf=%s ΔE=%.2f kcal/mol > %.2f cutoff",
                     label, r.get("conf", "?"), delta_kcal, cutoff_kcal)
            r[ok_key] = 0
            r[f"{energy_key}_outlier_drop"] = 1
            n_dropped += 1
    if n_dropped:
        log.info("  [%s] outlier rejection dropped %d conformer(s)",
                 label, n_dropped)
    return rows


def _stage_c_run_mace_one(
    sub: Path,
    elems: list[str],
    coords: np.ndarray,
    part: Partition,
    *,
    method: str,
    fmax: float,
    max_steps: int,
    salvage_gradnorm_max: float,
    device: str,
    label: str,
    conf_id: int | None = None,
    tier: int | None = None,
    optimizer_backend: str = "ase-lbfgs",
) -> tuple[int, bool, bool, float, int, bool, float | None]:
    """ASE+MACE counterpart to :func:`_stage_c_run_xtb_one`.

    Drives one Stage C constrained optimization with the requested MACE
    calculator. Mirrors the xtb branch's contract:

    * Same constraint topology as :func:`stage_pre_crest_relax`'s MACE
      branch: ``FixAtoms`` over ``part.fix_indices_no_waters`` (1-based →
      0-based) plus ``FixBondLengths`` over the reactive P-Onuc / P-Olg
      pair.

      **CONSTRAINT-TARGET SUBTLETY** (codex 2026-05-07 review #1): the
      xtb branch uses ``$constrain distance: P-Onuc, <part.d_p_onuc>``
      i.e. pins distances at the SOURCE-PDB targets. ASE
      ``FixBondLengths`` has no equivalent target-distance form; it pins
      at the CURRENT geometry's distance. For Stage C, where the input
      geometry is a CREST-sampled conformer (potentially far from
      source), this means MACE pins at the CREST-distorted distance, not
      the source target. This is acceptable as long as the post-CREST
      geometry filter (``post_crest_bond_cutoff``) has rejected anything
      pathological — which it does by default. If you need true
      source-target pinning, switch to ``--stage-c-method xtb``.
    * Writes a faux ``xtbopt.xyz`` with line 2 =
      ``energy: <Eh>  source=mace_stage_c  charge=<+/-N>`` so the rest of
      the funnel (g-xTB SP rerank, ranking, finalize) stays unchanged
      and diagnostic tools see the cluster's formal charge.
    * Force-based salvage criterion mirrors the xtb gradient-ratio salvage:
      we convert the final per-atom max force from eV/Å to Eh/Bohr (via
      ``EV_PER_A_TO_EH_PER_BOHR``) and compare against
      ``salvage_gradnorm_max`` (default 0.05 Eh/Bohr ≈ 2.57 eV/Å — see
      Stage E for the same reasoning, and codex review #2 for the
      caveat that this is ~50× looser than the LBFGS fmax target).

    Returns ``(rc, timed_out, ok_opt, gfn2_energy_Eh, extensions_used,
    salvaged, final_grad_ratio)`` — same tuple shape as the xtb path so
    the caller doesn't need a new branch. ``timed_out`` is always
    ``False`` because LBFGS doesn't have a wall-clock timeout in this
    branch (the converged/max_steps distinction is reported via the
    ``salvaged`` flag instead). NOTE: ``per_job_timeout`` is NOT
    enforced here — pathological ASE LBFGS hangs are bounded only by
    ``max_steps``. (codex 2026-05-07 review #6.)

    No ALPB: MACE has no implicit solvent. The caller's
    ``stage_C_gxtb`` issues a WARNING when both
    ``--stage-c-method=mace-*`` and ``--solvent`` are passed.
    """
    from ase.io import read as ase_read, write as ase_write
    from ase.constraints import FixAtoms, FixBondLengths
    from ase.optimize import LBFGS
    from quantum_engine.calc import make_calc

    sub.mkdir(parents=True, exist_ok=True)
    # Write the input geometry exactly the way the xtb branch does so
    # downstream tooling that expects ``input.xyz`` keeps working.
    with (sub / "input.xyz").open("w") as fh:
        fh.write(f"{len(elems)}\n")
        fh.write(f"{label}\n")
        for el, (x, y, z) in zip(elems, coords):
            fh.write(f"{el:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}\n")

    # Defaults for the failure path (mirrors xtb branch returns).
    rc = 0
    timed_out = False
    extensions_used = 0
    salvaged = False
    final_grad_ratio: float | None = None
    gfn2_energy = float("nan")
    ok_opt = False

    try:
        atoms = ase_read(str(sub / "input.xyz"))
        atoms.info["charge"] = part.charge
        atoms.calc = make_calc(method, device=device, charge=part.charge)

        # FixAtoms over the protein-CA scaffold (matches the xtb branch's
        # ``$fix atoms:`` constraint via _make_constraints_no_waters →
        # part.fix_indices_no_waters). 1-based → 0-based conversion.
        constraints = []
        ca0 = [i - 1 for i in part.fix_indices_no_waters]
        if ca0:
            constraints.append(FixAtoms(indices=ca0))
        # Pin reactive P-Onuc / P-Olg distances at the source-PDB values
        # (same data path as _make_constraints_no_waters); FixBondLengths
        # locks them at the current geometry's distance. Equivalent in
        # spirit to xtb's ``$constrain distance:`` lines.
        bondlength_pairs = [
            (part.p_idx_no_waters - 1, part.onuc_idx_no_waters - 1),
            (part.p_idx_no_waters - 1, part.olg_idx_no_waters - 1),
        ]
        constraints.append(FixBondLengths(bondlength_pairs))
        atoms.set_constraint(constraints)

        e0_eV = float(atoms.get_potential_energy())
        # Stage C MACE constraints (FixAtoms + FixBondLengths) are not
        # supported by torch-sim FIRE — silently fall back to ASE LBFGS
        # so the geometry stays valid. Same pattern as Stage E.
        backend = optimizer_backend
        if backend.startswith("torch-sim") and constraints:
            log.warning(
                "  [%s] stage_c MACE optimizer-backend=%s does not "
                "support ASE constraints; falling back to ase-lbfgs.",
                label, backend,
            )
            backend = "ase-lbfgs"
        if backend == "ase-lbfgs":
            opt_obj = LBFGS(atoms, logfile=str(sub / "ase.log"))
            converged = opt_obj.run(fmax=fmax, steps=max_steps)
        else:
            from quantum_engine.opt import make_optimizer
            mod_opt = make_optimizer(
                backend, fmax=fmax, max_steps=max_steps,
                logfile=sub / "ase.log",
            )
            res = mod_opt.run(atoms)
            converged = res.converged
        e_final_eV = float(atoms.get_potential_energy())

        # Convert eV → Hartree to match the xtb branch's units (Stage C
        # ranking is in Eh and the g-xTB SP rerank also reports Eh).
        gfn2_energy = e_final_eV * EV_TO_EH

        # Persist a faux xtbopt.xyz with the right comment-line format so
        # the downstream g-xTB SP path and rank/select code don't need a
        # new branch. Format mirrors xtb: line 2 = ``energy: <Eh>  ...``.
        # Include charge in the comment so diagnostic tools (and a later
        # human re-running g-xTB on the geometry) don't misread the file
        # as neutral. (codex 2026-05-07 review #4.)
        ase_write(str(sub / "relaxed.xyz"), atoms, format="xyz")
        rel_lines = (sub / "relaxed.xyz").read_text().splitlines()
        rel_lines[1] = (
            f" energy: {gfn2_energy:.10f}  source=mace_stage_c  "
            f"charge={part.charge:+d}  method={method}"
        )
        (sub / "xtbopt.xyz").write_text("\n".join(rel_lines) + "\n")

        # Force-based salvage criterion: mirror the xtb gradient-ratio
        # salvage. Convert eV/Å → Eh/Bohr (×EV_PER_A_TO_EH_PER_BOHR) so
        # the threshold can be compared directly against
        # ``salvage_gradnorm_max`` (default 0.05 Eh/Bohr ≈ 2.57 eV/Å).
        forces = atoms.get_forces()
        fmax_final_eV_per_A = float(np.linalg.norm(forces, axis=1).max())
        fmax_final_EhBohr = fmax_final_eV_per_A * EV_PER_A_TO_EH_PER_BOHR
        salvage_threshold_eVpA = (
            salvage_gradnorm_max / EV_PER_A_TO_EH_PER_BOHR
            if salvage_gradnorm_max > 0 else 0.0
        )

        if converged:
            ok_opt = True
        elif (salvage_gradnorm_max > 0
              and fmax_final_eV_per_A <= salvage_threshold_eVpA):
            ok_opt = True
            salvaged = True
            log.info("  Stage C salvage (MACE %s) @ %s: "
                     "fmax_final=%.3f eV/Å (≤ %.3f eV/Å) after %d/%d steps",
                     method, sub.name,
                     fmax_final_eV_per_A, salvage_threshold_eVpA,
                     max_steps, max_steps)
        else:
            ok_opt = False
            log.warning("  Stage C [MACE %s] %s NOT converged "
                        "(fmax_final=%.3f eV/Å > salvage=%.3f eV/Å)",
                        method, sub.name,
                        fmax_final_eV_per_A, salvage_threshold_eVpA)

        # final_grad_ratio: same convention as xtb branch (last_gnorm /
        # salvage threshold). Caller logs it; downstream ranking ignores it.
        if salvage_gradnorm_max > 0:
            final_grad_ratio = fmax_final_EhBohr / salvage_gradnorm_max

        log.info("  Stage C [MACE %s] %s  E0=%.4f eV  Ef=%.4f eV  "
                 "ΔE=%+.4f eV  converged=%s  fmax=%.3f eV/Å  ok=%s",
                 method, sub.name, e0_eV, e_final_eV,
                 e_final_eV - e0_eV, converged,
                 fmax_final_eV_per_A, ok_opt)
    except Exception as exc:  # pragma: no cover — defensive
        log.exception("Stage C [MACE %s] %s crashed: %s",
                      method, sub.name, exc)
        rc = -1
        ok_opt = False

    # Persist a tiny sidecar progress.json so finalize/diagnostic tools
    # that look for one don't crash. (xtb branch writes one via the
    # adaptive-timeout monitor; MACE doesn't, so we synthesize a stub.)
    try:
        payload: dict = {
            "stage": "C",
            "backend": method,
            "salvaged": bool(salvaged),
            "final_grad_ratio": (final_grad_ratio
                                  if final_grad_ratio is not None
                                  else None),
            "ok": int(bool(ok_opt)),
        }
        if conf_id is not None:
            payload["conf_id"] = int(conf_id)
        if tier is not None:
            payload["tier"] = int(tier)
        (sub / "progress.json").write_text(
            json.dumps(payload, indent=2, default=lambda o: None))
    except OSError as exc:
        log.debug("could not write Stage C MACE progress.json: %s", exc)

    return (rc, timed_out, ok_opt, gfn2_energy,
            extensions_used, salvaged, final_grad_ratio)


def _stage_c_run_xtb_one(
    sub: Path,
    elems: list[str],
    coords: np.ndarray,
    part: Partition,
    *,
    solvent: str | None,
    opt_level: str,
    timeout: int,
    sigterm_grace_s: float,
    salvage_gradnorm_max: float,
    base_env: dict,
    threads_per: int,
    label: str,
    progress_check_fraction: float = 0.6,
    progress_grad_ratio: float = 2.0,
    salvage_grad_ratio: float = 1.5,
    max_extensions: int = 0,
    extension_factor: float = 1.0,
    progress_log_filename: str = "progress.json",
    conf_id: int | None = None,
    tier: int | None = None,
) -> tuple[int, bool, bool, float, int, bool, float | None]:
    """Drive one constrained xtb-GFN2 opt under Strategy-E adaptive timeout,
    then decide whether the result is fully-converged, partially-salvaged,
    or dead.

    Returns ``(rc, timed_out, ok_opt, gfn2_energy, extensions_used,
    salvaged, final_grad_ratio)``.

    ``ok_opt`` is True when xtbopt.xyz exists AND either xtb returned 0, OR
    the partial geometry was salvaged via the Strategy-E grad-ratio rule
    (default ``salvage_grad_ratio=1.5``), OR the legacy gradient-norm rule
    (``salvage_gradnorm_max``, retained for backwards-compatible callers).

    With ``max_extensions=0`` (the function default) Strategy E reduces to
    the previous task #25 behaviour: single hard timeout + simple salvage.
    Setting ``max_extensions>=1`` enables adaptive extension based on the
    sidecar progress monitor.
    """
    sub.mkdir(parents=True, exist_ok=True)
    with (sub / "input.xyz").open("w") as fh:
        fh.write(f"{len(elems)}\n")
        fh.write(f"{label}\n")
        for el, (x, y, z) in zip(elems, coords):
            fh.write(f"{el:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}\n")
    _make_constraints_no_waters(part, sub / "constraints.inp")

    env_local = base_env.copy()
    env_local["OMP_NUM_THREADS"] = f"{threads_per},1"
    env_local["MKL_NUM_THREADS"] = str(threads_per)

    uhf_c = _xyz_uhf(sub / "input.xyz", part.charge)
    cmd_opt = [str(XTB_BIN), "input.xyz",
               "--gfn", "2",
               "--opt", opt_level,
               "--chrg", str(part.charge),
               "--uhf", str(uhf_c),
               "--input", "constraints.inp"]
    if solvent:
        cmd_opt += ["--alpb", solvent]
    progress_log_path = sub / progress_log_filename
    rc_opt, timed_out, extensions_used, salvaged, final_grad_ratio = (
        run_cmd_with_progress_monitor(
            cmd_opt, cwd=sub, log_path=sub / "xtb_opt.out",
            env=env_local, timeout=timeout, sigterm_grace_s=sigterm_grace_s,
            opt_level=opt_level,
            progress_check_fraction=progress_check_fraction,
            progress_grad_ratio=progress_grad_ratio,
            salvage_grad_ratio=salvage_grad_ratio,
            max_extensions=max_extensions,
            extension_factor=extension_factor,
            progress_log_path=progress_log_path,
        )
    )
    # Annotate progress.json with conformer / tier metadata for post-hoc
    # diagnostics (matches Deliverable 4 spec).
    if (conf_id is not None or tier is not None) and progress_log_path.exists():
        try:
            payload = json.loads(progress_log_path.read_text())
            if conf_id is not None:
                payload["conf_id"] = int(conf_id)
            if tier is not None:
                payload["tier"] = int(tier)
            progress_log_path.write_text(json.dumps(payload, indent=2,
                                                    default=lambda o: None))
        except (OSError, ValueError, TypeError) as exc:
            log.debug("could not annotate progress log: %s", exc)

    opt_xyz = sub / "xtbopt.xyz"
    gfn2_energy = float("nan")
    ok_opt = (rc_opt == 0 and opt_xyz.exists())
    if not ok_opt and opt_xyz.exists():
        # Strategy-E salvage (preferred) — if the adaptive monitor decided to
        # salvage based on grad RATIO, accept it regardless of absolute gnorm.
        if salvaged:
            log.info("  Stage C salvage (Strategy E) @ %s: grad_ratio=%.3f < %.2f",
                     sub.name,
                     final_grad_ratio if final_grad_ratio is not None else float("nan"),
                     salvage_grad_ratio)
            ok_opt = True
        else:
            # Legacy absolute-gnorm fallback (task #25 behaviour). Kept so
            # callers passing the old kwarg only still get a salvage path.
            last_gn = _read_last_gradient_norm(sub / "xtb_opt.out")
            if last_gn <= salvage_gradnorm_max:
                log.info("  salvaging partial opt @ %s: last grad norm=%.4f Eh/a0 "
                         "<= %.4f threshold (timed_out=%s, rc=%d)",
                         sub.name, last_gn, salvage_gradnorm_max, timed_out, rc_opt)
                ok_opt = True
    if ok_opt:
        try:
            txt = opt_xyz.read_text().splitlines()
            gfn2_energy = float(txt[1].split()[1])
        except (IndexError, ValueError):
            # Energy header wasn't written — partial save before xtb wrote the
            # comment line. Treat as failure for ranking purposes, but keep
            # the geometry on disk for diagnostics.
            ok_opt = False
            log.warning("  %s: xtbopt.xyz exists but no energy in comment line; "
                        "treating as failed for ranking", sub.name)
    return (rc_opt, timed_out, ok_opt, gfn2_energy,
            extensions_used, salvaged, final_grad_ratio)


def stage_C_gxtb(out_root: Path, part: Partition, top_n: int, ncpu: int,
                 solvent: str | None = "water", per_job_timeout: int = 900,
                 gxtb_sp_timeout: int = 600,
                 min_threads_per_job: int = 4,
                 stage_c_opt_level: str = "tight",
                 stage_c_tiers: str = "single",
                 stage_c_tier1_opt_level: str = "loose",
                 stage_c_tier1_timeout: int = 600,
                 stage_c_tier2_opt_level: str = "tight",
                 stage_c_tier2_timeout: int = 1800,
                 stage_c_tier1_keep_top: int = 5,
                 stage_c_energy_outlier_cutoff_kcal: float | None = None,
                 stage_c_sigterm_grace_s: float = 30.0,
                 stage_c_salvage_gradnorm_max: float = 0.05,
                 min_stage_c_survivors: int = 1,
                 stage_c_progress_check_fraction: float = 0.6,
                 stage_c_progress_grad_ratio: float = 2.0,
                 stage_c_salvage_grad_ratio: float = 1.5,
                 stage_c_max_extensions: int = 0,
                 stage_c_extension_factor: float = 1.0,
                 stage_c_method: str = "xtb",
                 stage_c_mace_fmax: float = 0.05,
                 stage_c_mace_max_steps: int = 200,
                 stage_c_device: str = "cuda",
                 optimizer_backend: str = "ase-lbfgs",
                 post_crest_bond_cutoff_a: float = POST_CREST_BOND_CUTOFF_DEFAULT,
                 post_crest_bad_bond_mode: str = "reject",
                 post_crest_min_survivors: int = 1,
                 post_crest_repair_max_passes: int = POST_CREST_REPAIR_MAX_PASSES_DEFAULT,
                 post_crest_source_shrink_tolerance: float = POST_CREST_SOURCE_SHRINK_TOLERANCE_DEFAULT,
                 post_crest_reactive_atoms_spec: str | None = "",
                 post_crest_max_match_distance_a: float = 0.75,
                 ) -> Path:
    """Stage C — refine CREST conformers and rerank.

    g-xTB v0.1 silently ignores both ``$fix atoms:`` and ``$constrain atoms:``,
    and CREST's post-MTD multilevel opt strips ``--cinp`` constraints during
    its own ensemble re-optimization. So:

      1. Kabsch-align each CREST conformer back to source frame using CAs.
      2. Snap CA positions to source-frame coords (locks the scaffold).
      3. Run vanilla xtb-GFN2 with ``$fix`` on those CAs and ``$constrain
         distance:`` on the reactive coords (xtb honors both). Solvent stays
         ``--alpb water``.
      4. Run g-xTB ``--sp`` (gas phase) on the GFN2-optimized geometry to
         get a higher-quality energy for ranking.

    Conformers run in parallel: N workers × M threads ≤ ncpu. Per-conformer
    timeouts are ENFORCED PER CONFORMER — one stuck worker can no longer abort
    the whole batch. Timed-out workers are skipped (and partial xtbopt.xyz
    geometries are salvaged when their last gradient norm is below
    ``stage_c_salvage_gradnorm_max``). After the parallel batch finishes, the
    funnel aborts only if FEWER than ``min_stage_c_survivors`` conformers are
    usable.

    ``stage_c_tiers='two'`` enables a two-tier optimization: Tier 1 runs ALL
    conformers at the cheap ``stage_c_tier1_opt_level`` level (default loose,
    short timeout) to filter outliers; the lowest ``stage_c_tier1_keep_top``
    survivors are then re-optimized at ``stage_c_tier2_opt_level`` (default
    tight). Both tiers keep ``--alpb water``.

    Post-CREST geometry filter: BEFORE per-conformer prep, every candidate
    is checked for unphysical heavy-heavy short bonds (CREST's MTD bias
    forces sometimes push atoms into clashes that the post-MTD multilevel
    opt does not relax). The filter is controlled by:

      * ``post_crest_bad_bond_mode`` (``'reject' | 'repair' | 'keep' | 'log'``)
      * ``post_crest_bond_cutoff_a`` (default 1.10 Å) — heavy-heavy distance
        below which a bond is candidate-suspect.
      * ``post_crest_source_shrink_tolerance`` (default 0.7) — when the
        source PDB has the same bond at ``source_d``, treat post_d as a
        tolerable vibration if ``post_d / source_d >= shrink_tolerance``.
      * ``post_crest_reactive_atoms_spec`` (default ""): comma-separated list
        of 1-based serials or ``NAME.RESNAME`` tokens (e.g. ``P1.SUB,O3.OHX``)
        identifying atoms whose bonds must be PRESERVED (not auto-repaired
        or rejected). Use to protect TS-like reactive distances.

    The filter prefers source-PDB bond distances as repair targets (via
    Kabsch alignment + element + nearest-neighbor matching), which makes
    repair bond-order-aware (single/double/triple/aromatic). Falls back to
    an organic element-pair table when no source bond is found. METAL
    bonds without a source match are LOGGED ONLY (no auto-repair).

    Backend selection (``stage_c_method``):
        * ``"xtb"`` (default, historical) — GFN2-xTB via the vendored
          binary, honours ``solvent`` for ALPB and the full Strategy-E
          adaptive-timeout / salvage stack. Preserves byte-identical
          behaviour for callers that don't pass ``--stage-c-method``.
        * ``"mace-mp"`` / ``"mace-polar-m"`` — ASE LBFGS with the matching
          MACE calculator. No ALPB (vacuum only). Reuses the SAME
          FixAtoms-on-CA + FixBondLengths-on-reactive-pair constraint
          pattern as the xtb path (via :func:`_make_constraints_no_waters`
          for xtb / :func:`_stage_c_run_mace_one` for MACE). Final energy
          is converted eV → Hartree so the g-xTB SP rerank and ranking
          stay apples-to-apples. Two-tier mode is supported in
          ``mace-*`` as well: tier 1 / tier 2 both go through the MACE
          backend (the ``opt_level`` arg is ignored).
    """
    method_c = (stage_c_method or "xtb").lower()
    if method_c not in STAGE_C_METHOD_CHOICES:
        raise ValueError(
            f"unknown --stage-c-method {method_c!r}; choose from "
            f"{STAGE_C_METHOD_CHOICES}"
        )
    if method_c.startswith("mace-") and solvent:
        log.warning("Stage C: --stage-c-method=%s does not support implicit "
                    "solvent; ignoring --solvent=%s and running in vacuum.",
                    method_c, solvent)
    d = out_root / "30_gxtb_minimize"
    d.mkdir(parents=True, exist_ok=True)

    # Prefer CREST's clustered conformers if there's actual diversity; otherwise
    # subsample from the raw MTD trajectory (CREST's multilevel ensemble opt
    # strips our constraints and collapses everything into one cluster, killing
    # the diversity that the constraint-aware MTD did sample).
    confs_path = out_root / "20_crest/crest_conformers.xyz"
    traj_path = out_root / "20_crest/crest_dynamics.trj"
    confs = split_multiframe_xyz(confs_path) if confs_path.exists() else []
    if len(confs) < 3 and traj_path.exists():
        traj = split_multiframe_xyz(traj_path)
        if len(traj) > len(confs):
            log.info("--- Stage C: CREST returned only %d clustered conformer(s); "
                     "subsampling %d frame(s) from raw MTD trajectory (%d total) "
                     "to recover sampled diversity",
                     len(confs), top_n, len(traj))
            step = max(1, len(traj) // max(1, top_n))
            confs = traj[::step][:top_n]

    # Slice to top-N BEFORE the geometry filter so we don't waste cycles
    # checking conformers we'd never have run anyway. The filter then runs
    # ONCE on the consolidated candidate list (Deliverable 4 — single
    # invocation, between Stage B clustering / subsampling and Stage C
    # ThreadPoolExecutor dispatch).
    confs = confs[:top_n]
    n_pre_filter = len(confs)

    # Persist pre-repair XYZs alongside the repaired ones so the user can
    # diff-inspect what the filter changed. Each conformer gets
    # ``30_gxtb_minimize/conf_NN/input_pre_repair.xyz`` (we re-use the
    # existing per-conformer subdirectory layout).
    def _write_pre_repair(k0: int, body_pre: list[str], _body_post: list[str]) -> None:
        sub_pre = d / f"conf_{k0 + 1:02d}"
        sub_pre.mkdir(parents=True, exist_ok=True)
        write_xyz_block(
            len(body_pre), body_pre, sub_pre / "input_pre_repair.xyz",
            comment=f"pre_repair conf={k0 + 1} (CREST-original geometry)",
        )

    # Pre-build source-PDB lookup data (no-waters atoms; CA atoms as Kabsch
    # anchors). The post (CREST) frame contains exactly part.no_waters atoms
    # in the same order, so the anchor indices into post and src lists are
    # IDENTICAL and equal to the CA atom indices. We pass both because the
    # API's signature is general — caller could in principle supply anchor
    # lists with different orderings.
    src_elems_for_filter = [a.element for a in part.no_waters]
    src_coords_for_filter = np.array(
        [[a.x, a.y, a.z] for a in part.no_waters]
    )
    ca_idx_for_filter = [
        i for i, a in enumerate(part.no_waters)
        if a.chain == "A" and a.name == "CA"
    ]
    # Reactive-atom set (0-based indices into the no_waters atom list).
    # User-supplied serials/NAME.RESNAME tokens AUGMENT a small default set
    # of obvious reactive atoms (P1, Onuc, Olg) — explicitly named here so
    # downstream Stage C never tries to "repair" the TS coordinate.
    base_reactive: set[int] = {
        part.p_idx_no_waters - 1,
        part.onuc_idx_no_waters - 1,
        part.olg_idx_no_waters - 1,
    }
    extra_reactive = _parse_reactive_atoms_spec(
        post_crest_reactive_atoms_spec, part.no_waters,
    )
    reactive_atoms_set = base_reactive | extra_reactive
    log.info(
        "post-CREST reactive-atom protection: %d atom(s) protected "
        "(base P/Onuc/Olg=%d, user-supplied=%d via spec=%r)",
        len(reactive_atoms_set), len(base_reactive), len(extra_reactive),
        post_crest_reactive_atoms_spec or "",
    )

    confs, filter_summary = post_crest_geometry_filter(
        confs,
        bond_cutoff_a=post_crest_bond_cutoff_a,
        bad_bond_mode=post_crest_bad_bond_mode,
        max_repair_passes=post_crest_repair_max_passes,
        pre_repair_writer=(_write_pre_repair
                           if post_crest_bad_bond_mode == "repair" else None),
        source_elems=src_elems_for_filter,
        source_coords=src_coords_for_filter,
        # The conformer atom order matches part.no_waters, so the
        # post→source map is the identity (no Kabsch needed).
        same_atom_order=True,
        reactive_atoms=reactive_atoms_set,
        source_shrink_tolerance=post_crest_source_shrink_tolerance,
        max_match_distance_a=post_crest_max_match_distance_a,
    )
    log.info(
        "post-CREST geometry filter: mode=%s cutoff=%.3f A shrink_tol=%.2f "
        "source_lookup=%s reactive_atoms=%d  in=%d clean=%d repaired=%d "
        "rejected=%d repair_failed=%d tolerated_pairs=%d "
        "reactive_protected_pairs=%d metal_no_source_pairs=%d  out=%d",
        filter_summary["mode"], filter_summary["bond_cutoff_a"],
        filter_summary["source_shrink_tolerance"],
        filter_summary["have_source_lookup"],
        filter_summary["n_reactive_atoms"],
        n_pre_filter,
        filter_summary["n_clean_passthrough"], filter_summary["n_repaired"],
        filter_summary["n_rejected"], filter_summary["n_repair_failed"],
        filter_summary["n_tolerated_pairs"],
        filter_summary["n_reactive_protected_pairs"],
        filter_summary["n_metal_no_source_pairs"],
        filter_summary["n_survivors"],
    )
    # Persist the per-conformer filter summary for downstream diagnostics.
    try:
        (d / "post_crest_filter_summary.json").write_text(
            json.dumps(filter_summary, indent=2, default=lambda o: None))
    except OSError as exc:
        log.debug("could not write post_crest_filter_summary.json: %s", exc)

    if len(confs) < post_crest_min_survivors:
        raise RuntimeError(
            f"post-CREST geometry filter dropped too many conformers: "
            f"only {len(confs)} survivor(s) (< post_crest_min_survivors="
            f"{post_crest_min_survivors}). Mode='{post_crest_bad_bond_mode}', "
            f"cutoff={post_crest_bond_cutoff_a:.3f} A, "
            f"rejected={filter_summary['n_rejected']}, "
            f"repair_failed={filter_summary['n_repair_failed']}. "
            f"Actionable: (1) re-run with --post-crest-bad-bond-mode repair "
            f"to fix instead of drop; (2) lower "
            f"--post-crest-source-shrink-tolerance below "
            f"{post_crest_source_shrink_tolerance:.2f} to be more permissive "
            f"about post-CREST bonds shorter than source; (3) widen "
            f"--post-crest-reactive-atoms to protect more atoms whose bonds "
            f"are intentionally TS-distorted; (4) raise "
            f"--post-crest-bond-cutoff above {post_crest_bond_cutoff_a:.2f} "
            f"A only as a last resort (the source-PDB bond-order check is "
            f"the right way to handle triple bonds and exotic chemistry); "
            f"or (5) re-sample with tighter --crest-rthr / less aggressive "
            f"metadynamics.")

    n = min(top_n, len(confs))
    workers, threads_per = _compute_workers(ncpu, n, min_threads_per_job=min_threads_per_job)
    log.info("--- Stage C: refine %d conformer(s) — %d worker(s) × %d thread(s) "
             "  tiers=%s", n, workers, threads_per, stage_c_tiers)

    base_env = os.environ.copy()
    base_env["OMP_STACKSIZE"] = "4G"

    # CA anchor set (always frozen) — used to align CREST output back to source frame
    ca_idx_0 = [i - 1 for i, a in enumerate(part.no_waters, start=1)
                if a.chain == "A" and a.name == "CA"]
    src_no_waters_arr = np.array([[a.x, a.y, a.z] for a in part.no_waters])
    src_ca_arr = src_no_waters_arr[ca_idx_0]

    def kabsch_to_source(coords: np.ndarray) -> np.ndarray:
        cP = coords[ca_idx_0].mean(0); cQ = src_ca_arr.mean(0)
        H = (coords[ca_idx_0] - cP).T @ (src_ca_arr - cQ)
        U, S, Vt = np.linalg.svd(H)
        sign = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, sign]) @ U.T
        return (coords - cP) @ R.T + cQ

    # Pre-prepare element/coords for each conformer once.
    conf_inputs: list[tuple[float, list[str], np.ndarray, float]] = []
    for k in range(n):
        e_crest, body = confs[k]
        elems, xyz = [], []
        for ln in body:
            toks = ln.split()
            elems.append(toks[0])
            xyz.append([float(toks[1]), float(toks[2]), float(toks[3])])
        coords = kabsch_to_source(np.array(xyz))
        ca_drift = float(np.linalg.norm(coords[ca_idx_0] - src_ca_arr, axis=1).max())
        coords[ca_idx_0] = src_ca_arr  # snap CA atoms to source positions
        conf_inputs.append((e_crest, elems, coords, ca_drift))

    if stage_c_tiers == "two":
        tier1_workers, tier1_threads = _compute_workers(
            ncpu, n, min_threads_per_job=min_threads_per_job)
        log.info("Stage C tier1: %d conformer(s) method=%s opt=%s timeout=%ds  "
                 "%d worker(s) × %d thread(s)",
                 n, method_c, stage_c_tier1_opt_level, stage_c_tier1_timeout,
                 tier1_workers, tier1_threads)
    else:
        tier1_workers, tier1_threads = workers, threads_per
        log.info("Stage C single-tier: method=%s opt=%s timeout=%ds  "
                 "%d worker(s) × %d thread(s)",
                 method_c, stage_c_opt_level, per_job_timeout,
                 workers, threads_per)

    # MACE inference is GPU-bound; running multiple ASE workers concurrently
    # against one GPU just contend for the device. Force serial dispatch in
    # that branch (workers=1) — matches stage_pre_crest_relax behaviour.
    if method_c.startswith("mace-"):
        if tier1_workers > 1:
            log.info("Stage C [MACE %s]: forcing serial worker dispatch "
                     "(was %d) — MACE inference is single-GPU bound",
                     method_c, tier1_workers)
        tier1_workers = 1
        if workers > 1:
            workers = 1

    def _run_tier1(k: int) -> dict:
        """Tier 1 (or single-tier when ``stage_c_tiers='single'``):
        constrained opt at the cheap or final level. Backend dispatched on
        ``method_c`` (xtb branch salvages partial xtbopt.xyz when SIGTERM
        hits with small gradient; MACE branch salvages on max_steps when
        the final force is below the same threshold)."""
        e_crest, elems, coords, ca_drift = conf_inputs[k]
        sub = d / f"conf_{k+1:02d}"
        if stage_c_tiers == "two":
            sub = sub / "tier1"
        opt_level = (stage_c_tier1_opt_level if stage_c_tiers == "two"
                     else stage_c_opt_level)
        timeout_s = (stage_c_tier1_timeout if stage_c_tiers == "two"
                     else per_job_timeout)
        try:
            if method_c == "xtb":
                (rc_opt, timed_out, ok_opt, gfn2_energy,
                 extensions_used, salvaged,
                 final_grad_ratio) = _stage_c_run_xtb_one(
                    sub=sub, elems=elems, coords=coords, part=part,
                    solvent=solvent, opt_level=opt_level, timeout=timeout_s,
                    sigterm_grace_s=stage_c_sigterm_grace_s,
                    salvage_gradnorm_max=stage_c_salvage_gradnorm_max,
                    base_env=base_env, threads_per=tier1_threads,
                    label=f"crest_rank={k+1} tier1 "
                          f"ca_drift_pre_snap={ca_drift:.4f}A",
                    progress_check_fraction=stage_c_progress_check_fraction,
                    progress_grad_ratio=stage_c_progress_grad_ratio,
                    salvage_grad_ratio=stage_c_salvage_grad_ratio,
                    max_extensions=stage_c_max_extensions,
                    extension_factor=stage_c_extension_factor,
                    conf_id=k + 1,
                    tier=1,
                )
            else:
                (rc_opt, timed_out, ok_opt, gfn2_energy,
                 extensions_used, salvaged,
                 final_grad_ratio) = _stage_c_run_mace_one(
                    sub=sub, elems=elems, coords=coords, part=part,
                    method=method_c,
                    fmax=stage_c_mace_fmax,
                    max_steps=stage_c_mace_max_steps,
                    salvage_gradnorm_max=stage_c_salvage_gradnorm_max,
                    device=stage_c_device,
                    label=f"crest_rank={k+1} tier1 "
                          f"ca_drift_pre_snap={ca_drift:.4f}A",
                    conf_id=k + 1,
                    tier=1,
                    optimizer_backend=optimizer_backend,
                )
        except Exception as exc:  # pragma: no cover — defensive: never let a
            # worker exception take out the rest of the batch.
            log.exception("Stage C: conformer %d crashed unexpectedly: %s",
                          k + 1, exc)
            rc_opt, timed_out, ok_opt, gfn2_energy = -1, False, False, float("nan")
            extensions_used, salvaged, final_grad_ratio = 0, False, None

        if timed_out and not ok_opt:
            log.warning("Stage C: conformer %d skipped (timeout after %ds, "
                        "ext=%d, final_grad_ratio=%s)",
                        k + 1, timeout_s, extensions_used,
                        f"{final_grad_ratio:.3f}" if final_grad_ratio is not None
                        else "None")

        opt_xyz = sub / "xtbopt.xyz"
        return {
            "conf": k + 1,
            "crest_rank": k + 1,
            "crest_energy_Eh": e_crest,
            "ca_drift_pre_snap_A": ca_drift,
            "tier1_ok": int(ok_opt),
            "tier1_timed_out": int(timed_out),
            "tier1_opt_level": opt_level,
            "tier1_energy_Eh": gfn2_energy,
            "tier1_path": str(opt_xyz) if opt_xyz.exists() else "",
            "tier1_extensions_used": extensions_used,
            "tier1_salvaged": int(salvaged),
            "tier1_final_grad_ratio": (final_grad_ratio
                                       if final_grad_ratio is not None
                                       else float("nan")),
            # filled in by tier-2 / single-tier post-processing below:
            "gxtb_ok": int(ok_opt),
            "timeout": int(timed_out),
            "gfn2_opt_energy_Eh": gfn2_energy,
            "gxtb_sp_energy_Eh": float("nan"),
            "gxtb_sp_ok": 0,
            "path": str(opt_xyz) if opt_xyz.exists() else "",
            "gxtb_energy_Eh": gfn2_energy,
        }

    if tier1_workers <= 1:
        rows = [_run_tier1(k) for k in range(n)]
    else:
        with ThreadPoolExecutor(max_workers=tier1_workers) as pool:
            rows = list(pool.map(_run_tier1, range(n)))

    # Outlier rejection on tier-1 energies — applied to BOTH single-tier and
    # two-tier runs (single-tier just acts as the final tier here).
    _apply_outlier_cutoff(rows, energy_key="tier1_energy_Eh",
                          ok_key="tier1_ok",
                          cutoff_kcal=stage_c_energy_outlier_cutoff_kcal,
                          label="Stage C tier1")

    if stage_c_tiers == "two":
        # Pick survivors: top-K by tier-1 energy among those that succeeded.
        survivors = [r for r in rows if int(r["tier1_ok"]) == 1
                     and isinstance(r["tier1_energy_Eh"], float)
                     and not math.isnan(r["tier1_energy_Eh"])]
        survivors.sort(key=lambda r: r["tier1_energy_Eh"])
        keep_k = min(stage_c_tier1_keep_top, len(survivors))
        keep_confs = {r["conf"] for r in survivors[:keep_k]}
        log.info("Stage C tier1 → tier2: %d/%d survived; keeping top-%d for "
                 "tier2 (%s)", len(survivors), n, keep_k,
                 sorted(keep_confs))

        n_tier2 = len(keep_confs)
        tier2_workers, tier2_threads = _compute_workers(
            ncpu, n_tier2, min_threads_per_job=min_threads_per_job)
        log.info("Stage C tier2: %d conformer(s) method=%s opt=%s timeout=%ds  "
                 "%d worker(s) × %d thread(s)",
                 n_tier2, method_c, stage_c_tier2_opt_level,
                 stage_c_tier2_timeout, tier2_workers, tier2_threads)
        if method_c.startswith("mace-") and tier2_workers > 1:
            log.info("Stage C tier2 [MACE %s]: forcing serial dispatch",
                     method_c)
            tier2_workers = 1

        def _run_tier2(r: dict) -> None:
            k = r["conf"] - 1
            _, elems, _, ca_drift = conf_inputs[k]
            # Read the tier-1 optimized geometry as the tier-2 starting point.
            t1_xyz = Path(r["tier1_path"])
            try:
                _, t1_coords = _read_xyz(t1_xyz)
            except (FileNotFoundError, IndexError, ValueError) as exc:
                log.warning("Stage C tier2: conf %d cannot read tier1 xyz: %s",
                            r["conf"], exc)
                r["gxtb_ok"] = 0
                r["timeout"] = 0
                r["path"] = ""
                r["gxtb_energy_Eh"] = float("nan")
                r["gfn2_opt_energy_Eh"] = float("nan")
                return
            coords = np.array(t1_coords)
            # Re-snap CA atoms in case tier1 left tiny drift (e.g. SIGTERM
            # mid-step) — strictly conservative.
            coords[ca_idx_0] = src_ca_arr
            sub = d / f"conf_{r['conf']:02d}/tier2"
            try:
                if method_c == "xtb":
                    (rc_opt, timed_out, ok_opt, gfn2_energy,
                     extensions_used, salvaged,
                     final_grad_ratio) = _stage_c_run_xtb_one(
                        sub=sub, elems=elems, coords=coords, part=part,
                        solvent=solvent, opt_level=stage_c_tier2_opt_level,
                        timeout=stage_c_tier2_timeout,
                        sigterm_grace_s=stage_c_sigterm_grace_s,
                        salvage_gradnorm_max=stage_c_salvage_gradnorm_max,
                        base_env=base_env, threads_per=tier2_threads,
                        label=f"crest_rank={r['conf']} tier2 from_tier1",
                        progress_check_fraction=stage_c_progress_check_fraction,
                        progress_grad_ratio=stage_c_progress_grad_ratio,
                        salvage_grad_ratio=stage_c_salvage_grad_ratio,
                        max_extensions=stage_c_max_extensions,
                        extension_factor=stage_c_extension_factor,
                        conf_id=int(r["conf"]),
                        tier=2,
                    )
                else:
                    (rc_opt, timed_out, ok_opt, gfn2_energy,
                     extensions_used, salvaged,
                     final_grad_ratio) = _stage_c_run_mace_one(
                        sub=sub, elems=elems, coords=coords, part=part,
                        method=method_c,
                        fmax=stage_c_mace_fmax,
                        max_steps=stage_c_mace_max_steps,
                        salvage_gradnorm_max=stage_c_salvage_gradnorm_max,
                        device=stage_c_device,
                        label=f"crest_rank={r['conf']} tier2 from_tier1",
                        conf_id=int(r["conf"]),
                        tier=2,
                        optimizer_backend=optimizer_backend,
                    )
            except Exception as exc:  # pragma: no cover — defensive
                log.exception("Stage C tier2: conf %d crashed: %s",
                              r["conf"], exc)
                rc_opt, timed_out, ok_opt, gfn2_energy = -1, False, False, float("nan")
                extensions_used, salvaged, final_grad_ratio = 0, False, None

            if timed_out and not ok_opt:
                log.warning("Stage C: conformer %d (tier2) skipped "
                            "(timeout after %ds, ext=%d)",
                            r["conf"], stage_c_tier2_timeout, extensions_used)

            opt_xyz = sub / "xtbopt.xyz"
            r["gxtb_ok"] = int(ok_opt)
            r["timeout"] = int(timed_out)
            r["gfn2_opt_energy_Eh"] = gfn2_energy
            r["path"] = str(opt_xyz) if opt_xyz.exists() else r["tier1_path"]
            r["gxtb_energy_Eh"] = gfn2_energy if ok_opt else float("nan")
            r["tier2_opt_level"] = stage_c_tier2_opt_level
            r["tier2_energy_Eh"] = gfn2_energy
            r["tier2_timed_out"] = int(timed_out)
            r["tier2_extensions_used"] = extensions_used
            r["tier2_salvaged"] = int(salvaged)
            r["tier2_final_grad_ratio"] = (final_grad_ratio
                                           if final_grad_ratio is not None
                                           else float("nan"))

        # Mark non-survivors as failed for ranking (their energies are still in
        # the CSV under tier1_energy_Eh for diagnostics).
        for r in rows:
            if r["conf"] not in keep_confs:
                r["gxtb_ok"] = 0
                # Don't overwrite r["timeout"] from tier1; it documents WHY
                # this conformer didn't make tier2.

        survivor_rows = [r for r in rows if r["conf"] in keep_confs]
        if tier2_workers <= 1:
            for r in survivor_rows:
                _run_tier2(r)
        else:
            with ThreadPoolExecutor(max_workers=tier2_workers) as pool:
                list(pool.map(_run_tier2, survivor_rows))

    # g-xTB single-point on each surviving GFN2-relaxed geometry.
    sp_workers, sp_threads = _compute_workers(
        ncpu, sum(1 for r in rows if int(r["gxtb_ok"]) == 1),
        min_threads_per_job=min_threads_per_job)
    sp_workers = max(1, sp_workers)

    def _run_sp(r: dict) -> None:
        if int(r["gxtb_ok"]) != 1:
            return
        opt_xyz = Path(r["path"])
        if not opt_xyz.exists():
            return
        sp_dir = opt_xyz.parent / "gxtb_sp"
        sp_dir.mkdir(exist_ok=True)
        shutil.copy2(opt_xyz, sp_dir / "input.xyz")
        env_local = base_env.copy()
        env_local["OMP_NUM_THREADS"] = f"{sp_threads},1"
        env_local["MKL_NUM_THREADS"] = str(sp_threads)
        uhf_sp = _xyz_uhf(sp_dir / "input.xyz", part.charge)
        cmd_sp = [str(GXTB_BIN), "input.xyz", "--gxtb", "--sp",
                  "--chrg", str(part.charge),
                  "--uhf", str(uhf_sp)]
        try:
            rc_sp = run_cmd(cmd_sp, cwd=sp_dir, log_path=sp_dir / "gxtb_sp.out",
                            env=env_local, timeout=gxtb_sp_timeout)
        except subprocess.TimeoutExpired:
            log.warning("Stage C: g-xTB SP for conf %d timed out (%ds); "
                        "keeping GFN2 energy", r["conf"], gxtb_sp_timeout)
            r["gxtb_sp_ok"] = 0
            return
        if rc_sp == 0:
            text = (sp_dir / "gxtb_sp.out").read_text()
            m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s*Eh", text)
            if m:
                r["gxtb_sp_energy_Eh"] = float(m.group(1))
                r["gxtb_sp_ok"] = 1
                # If g-xTB single-point landed cleanly, prefer it for ranking.
                r["gxtb_energy_Eh"] = r["gxtb_sp_energy_Eh"]

    sp_rows = [r for r in rows if int(r["gxtb_ok"]) == 1]
    if sp_workers <= 1:
        for r in sp_rows:
            _run_sp(r)
    else:
        with ThreadPoolExecutor(max_workers=sp_workers) as pool:
            list(pool.map(_run_sp, sp_rows))

    # Survival check — bail out only if literally nothing usable came back.
    survivors = sum(1 for r in rows if int(r["gxtb_ok"]) == 1)
    n_timed_out = sum(1 for r in rows if int(r.get("timeout", 0)) == 1)
    log.info("Stage C summary: %d/%d conformer(s) survived (%d timed out)",
             survivors, n, n_timed_out)
    if survivors < min_stage_c_survivors:
        raise RuntimeError(
            f"Stage C: only {survivors} conformer(s) succeeded "
            f"(< min_stage_c_survivors={min_stage_c_survivors}). "
            f"Timeouts: {n_timed_out}/{n}. Consider raising "
            f"--per-job-timeout or switching to --stage-c-tiers two.")

    # NaN-safe sort: failed/NaN energies go to the end.
    def _rank_key(r: dict) -> tuple[int, float]:
        e = r.get("gxtb_energy_Eh", float("nan"))
        if not isinstance(e, float) or math.isnan(e):
            return (1, float("inf"))
        return (1 - int(r["gxtb_ok"]), e)
    rows.sort(key=_rank_key)
    # Normalize rows to a consistent set of columns (in case tier-1-only or
    # tier-1-then-tier-2 produced different keys).
    all_keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for key in r:
            if key not in seen:
                all_keys.append(key)
                seen.add(key)
    csv_path = d / "ranked.csv"
    with csv_path.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in all_keys})
    log.info("  → %s", csv_path)
    return csv_path


def _read_xyz(path: Path) -> tuple[list[str], list[tuple[float, float, float]]]:
    txt = path.read_text().splitlines()
    n = int(txt[0])
    elems, xyz = [], []
    for line in txt[2 : 2 + n]:
        toks = line.split()
        elems.append(toks[0])
        xyz.append((float(toks[1]), float(toks[2]), float(toks[3])))
    return elems, xyz


def stage_E_water_relax(out_root: Path, part: Partition, ncpu: int, top_keep: int,
                         solvent: str | None = "water",
                         per_job_timeout: int = 900,
                         min_threads_per_job: int = 4,
                         stage_e_opt_level: str = "loose",
                         stage_e_sigterm_grace_s: float = 30.0,
                         stage_e_salvage_gradnorm_max: float = 0.05,
                         min_stage_e_survivors: int = 1,
                         stage_e_progress_check_fraction: float = 0.6,
                         stage_e_progress_grad_ratio: float = 2.0,
                         stage_e_salvage_grad_ratio: float = 1.5,
                         stage_e_max_extensions: int = 0,
                         stage_e_extension_factor: float = 1.0,
                         stage_e_method: str = "xtb",
                         stage_e_mace_fmax: float = 0.05,
                         stage_e_mace_max_steps: int = 200,
                         stage_e_device: str = "cuda",
                         optimizer_backend: str = "ase-lbfgs") -> Path:
    """Re-insert source-frame waters AFTER mapping them into the post-CREST frame
    via a CA-anchored Kabsch transform — otherwise CREST's CMA translation puts
    the waters in vacuum, tens of Å from the protein.

    Backend selection (``stage_e_method``):
        * ``"xtb"`` (default, historical) — GFN2-xTB via the vendored binary,
          honours ``solvent`` for ALPB and the full Stage-E grad-monitor /
          salvage stack.
        * ``"mace-mp"`` / ``"mace-polar-m"`` — ASE LBFGS with the matching
          MACE calculator. No ALPB (vacuum only). Reuses the same FixAtoms
          constraint (heavy atoms 1..n_no_water frozen, waters free) and
          writes a faux ``xtbopt.xyz`` so the rest of the pipeline (energy
          parse, finalize) is untouched.
    """
    method = (stage_e_method or "xtb").lower()
    if method not in STAGE_E_METHOD_CHOICES:
        raise ValueError(
            f"unknown --stage-e-method {method!r}; choose from "
            f"{STAGE_E_METHOD_CHOICES}"
        )
    if method.startswith("mace-") and solvent:
        log.warning("Stage E: --stage-e-method=%s does not support implicit "
                    "solvent; ignoring --solvent=%s and running in vacuum.",
                    method, solvent)
    log.info("--- Stage D+E: re-insert waters and relax  (method=%s)", method)
    d = out_root / "40_water_relax"
    d.mkdir(parents=True, exist_ok=True)

    ranked = list(csv.DictReader((out_root / "30_gxtb_minimize/ranked.csv").open()))
    ranked = [r for r in ranked if r["gxtb_ok"] == "1"][:top_keep]

    n_no_water = len(part.no_waters)
    n_water = len(part.waters)
    fix_atoms = list(range(1, n_no_water + 1))  # freeze everything except waters

    workers, threads_per = _compute_workers(ncpu, len(ranked), min_threads_per_job=4)
    log.info("Stage E: %d conformer(s) — %d worker(s) × %d thread(s)",
             len(ranked), workers, threads_per)

    base_env = os.environ.copy()
    base_env["OMP_STACKSIZE"] = "4G"

    # Anchor set for source-frame -> post-opt-frame alignment.
    # CA atoms are *always* $fix'd, so they're identical in both frames. Using
    # them avoids depending on whether Zn was frozen in the run.
    anchor_idx_0 = [i - 1 for i, a in enumerate(part.no_waters, start=1)
                    if a.chain == "A" and a.name == "CA"]
    src_anchor_arr = np.array([[part.no_waters[i].x, part.no_waters[i].y, part.no_waters[i].z]
                               for i in anchor_idx_0])
    src_water_arr = np.array([[a.x, a.y, a.z] for a in part.waters])

    def kabsch_src_to(target_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cS = src_anchor_arr.mean(0); cT = target_arr.mean(0)
        H = (src_anchor_arr - cS).T @ (target_arr - cT)
        U, S, Vt = np.linalg.svd(H)
        sign = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, sign]) @ U.T
        return R, cS, cT

    def _run_one(r: dict) -> dict:
        opt_xyz = Path(r["path"])
        elems, xyz = _read_xyz(opt_xyz)
        coords_arr = np.array(xyz)
        sub = d / f"conf_{int(r['conf']):02d}"
        sub.mkdir(parents=True, exist_ok=True)

        target_anchor = coords_arr[anchor_idx_0]
        R, cS, cT = kabsch_src_to(target_anchor)
        waters_in_target_frame = (src_water_arr - cS) @ R.T + cT

        anchor_residual = float(np.linalg.norm(
            (src_anchor_arr - cS) @ R.T + cT - target_anchor, axis=1).max())

        combined_lines = [f"{n_no_water + n_water}",
                          f"combined no_water_from={opt_xyz} "
                          f"waters_kabsch_aligned anchor_resid={anchor_residual:.4f}"]
        for el, (x, y, z) in zip(elems, xyz):
            combined_lines.append(f"{el:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}")
        for a, (x, y, z) in zip(part.waters, waters_in_target_frame):
            combined_lines.append(f"{a.element:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}")
        (sub / "combined.xyz").write_text("\n".join(combined_lines) + "\n")

        # Defaults the MACE branch will need to fill in if it's the active path.
        opt = sub / "xtbopt.xyz"
        timed_out = False
        extensions_used = 0
        salvaged_e = False
        final_grad_ratio_e: float | None = None
        rc = 0
        e_final = float("nan")

        if method == "xtb":
            write_xtb_constraints(
                out_path=sub / "constraints.inp",
                fix_atoms=fix_atoms,
                distance_constraints=[],
                fix_force=1.0,
            )

            env_local = base_env.copy()
            env_local["OMP_NUM_THREADS"] = f"{threads_per},1"
            env_local["MKL_NUM_THREADS"] = str(threads_per)

            uhf_e = _xyz_uhf(sub / "combined.xyz", part.charge)
            cmd = [str(XTB_BIN), "combined.xyz",
                   "--gfn", "2",
                   "--opt", stage_e_opt_level,
                   "--chrg", str(part.charge),
                   "--uhf", str(uhf_e),
                   "--input", "constraints.inp"]
            if solvent:
                cmd += ["--alpb", solvent]

            progress_log_path = sub / "progress.json"
            try:
                (rc, timed_out, extensions_used, salvaged_e,
                 final_grad_ratio_e) = run_cmd_with_progress_monitor(
                    cmd, cwd=sub, log_path=sub / "xtb.out", env=env_local,
                    timeout=per_job_timeout,
                    sigterm_grace_s=stage_e_sigterm_grace_s,
                    opt_level=stage_e_opt_level,
                    progress_check_fraction=stage_e_progress_check_fraction,
                    progress_grad_ratio=stage_e_progress_grad_ratio,
                    salvage_grad_ratio=stage_e_salvage_grad_ratio,
                    max_extensions=stage_e_max_extensions,
                    extension_factor=stage_e_extension_factor,
                    progress_log_path=progress_log_path,
                )
            except Exception as exc:  # pragma: no cover — defensive: don't let one
                # crashing worker take out the rest of the batch.
                log.exception("Stage E: conformer %s crashed: %s",
                              r["conf"], exc)
                rc, timed_out = -1, False
                extensions_used, salvaged_e = 0, False
                final_grad_ratio_e = None
            # Annotate progress.json with conf metadata.
            if progress_log_path.exists():
                try:
                    payload = json.loads(progress_log_path.read_text())
                    payload["conf_id"] = int(r["conf"])
                    payload["stage"] = "E"
                    progress_log_path.write_text(json.dumps(
                        payload, indent=2, default=lambda o: None))
                except (OSError, ValueError, TypeError) as exc:
                    log.debug("could not annotate Stage E progress log: %s",
                              exc)
            ok = (rc == 0 and opt.exists())
            # Salvage partial: prefer Strategy E grad-ratio salvage (already
            # decided by the monitor); fall back to absolute-gnorm rule for
            # backwards compatibility.
            if not ok and opt.exists():
                if salvaged_e:
                    log.info("  Stage E salvage (Strategy E) conf %s "
                             "grad_ratio=%.3f", r["conf"],
                             final_grad_ratio_e if final_grad_ratio_e is not None
                             else float("nan"))
                    ok = True
                else:
                    last_gn = _read_last_gradient_norm(sub / "xtb.out")
                    if last_gn <= stage_e_salvage_gradnorm_max:
                        log.info("  Stage E: salvaging partial conf %s "
                                 "(grad norm=%.4f Eh/a0 <= %.4f)",
                                 r["conf"], last_gn,
                                 stage_e_salvage_gradnorm_max)
                        ok = True
            if ok:
                try:
                    txt = opt.read_text().splitlines()
                    e_final = float(txt[1].split()[1])
                except (IndexError, ValueError):
                    ok = False
                    log.warning("Stage E: conf %s xtbopt.xyz exists but no "
                                "energy header; treating as failed", r["conf"])
            if timed_out and not ok:
                log.warning("Stage E: conformer %s skipped (timeout after "
                            "%ds, ext=%d)",
                            r["conf"], per_job_timeout, extensions_used)
        else:
            # ASE+MACE branch: in-process LBFGS with FixAtoms over the protein
            # heavy atoms (fix_atoms is 1-based; ASE wants 0-based). We do
            # NOT pin reactive bond lengths — Stage E is supposed to let the
            # waters re-pack and the active-site geometry settle, like the
            # xtb path does. Vacuum only (MACE has no implicit solvent).
            from ase.io import read as ase_read, write as ase_write
            from ase.constraints import FixAtoms
            from ase.optimize import LBFGS
            from quantum_engine.calc import make_calc

            t_start = time.time()
            try:
                atoms = ase_read(str(sub / "combined.xyz"))
                atoms.info["charge"] = part.charge
                atoms.calc = make_calc(method, device=stage_e_device,
                                       charge=part.charge)
                # FixAtoms over the no-water heavy atoms — same semantics as
                # the xtb branch's fix_atoms (which uses 1-based 1..n_no_water).
                fix0 = list(range(n_no_water))
                atoms.set_constraint(FixAtoms(indices=fix0))

                e0_eV = float(atoms.get_potential_energy())
                # Stage E carries a FixAtoms (heavy-atom) constraint only —
                # no FixBondLength. torch-sim FIRE supports this case (it
                # only refuses if any constraint is active in our wrapper).
                # For now, keep the safe fallback: only use torch-sim-fire
                # if explicitly requested AND no constraints are active.
                # Default stays ase-lbfgs.
                backend_e = optimizer_backend
                if backend_e.startswith("torch-sim"):
                    log.warning(
                        "  Stage E [MACE %s] optimizer-backend=%s skips "
                        "FixAtoms constraints — falling back to ase-lbfgs.",
                        method, backend_e,
                    )
                    backend_e = "ase-lbfgs"
                if backend_e == "ase-lbfgs":
                    opt_obj = LBFGS(atoms, logfile=str(sub / "ase.log"))
                    converged = opt_obj.run(fmax=stage_e_mace_fmax,
                                            steps=stage_e_mace_max_steps)
                else:
                    from quantum_engine.opt import make_optimizer
                    mod_opt = make_optimizer(
                        backend_e,
                        fmax=stage_e_mace_fmax,
                        max_steps=stage_e_mace_max_steps,
                        logfile=sub / "ase.log",
                    )
                    res = mod_opt.run(atoms)
                    converged = res.converged
                e_final_eV = float(atoms.get_potential_energy())
                # Convert to Hartree to match the xtb branch's units; the
                # rest of the funnel reads ``water_relax_energy_Eh`` from
                # this column (Stage C is in Eh too, so apples to apples).
                e_final = e_final_eV * EV_TO_EH

                # Persist a faux ``xtbopt.xyz`` so funnel_finalize and the
                # downstream rank/select code don't need a new branch.
                # Match the xtb format: line2 = "<placeholder> <energy_Eh>".
                ase_write(str(sub / "relaxed.xyz"), atoms, format="xyz")
                rel_lines = (sub / "relaxed.xyz").read_text().splitlines()
                rel_lines[1] = f" energy: {e_final:.10f}  source=mace_stage_e"
                opt.write_text("\n".join(rel_lines) + "\n")

                # Acceptance criterion (mirrors the xtb gradient-ratio
                # salvage): if LBFGS hit max-steps without satisfying fmax,
                # check the final max-force. We convert eV/Å -> Eh/Bohr
                # (via EV_PER_A_TO_EH_PER_BOHR) so we can compare directly
                # against ``stage_e_salvage_gradnorm_max`` (default
                # 0.05 Eh/Bohr ≈ 2.57 eV/Å — generous enough that most
                # "near-converged" MACE LBFGS runs squeak through).
                EVPA_TO_EHBOHR = EV_PER_A_TO_EH_PER_BOHR
                forces = atoms.get_forces()
                fmax_final_eV_per_A = float(
                    np.linalg.norm(forces, axis=1).max())
                fmax_final_EhBohr = fmax_final_eV_per_A * EVPA_TO_EHBOHR
                salvage_threshold_eVpA = (stage_e_salvage_gradnorm_max
                                          / EVPA_TO_EHBOHR)
                if converged:
                    ok = True
                elif fmax_final_eV_per_A <= salvage_threshold_eVpA:
                    ok = True
                    salvaged_e = True
                    log.info("  Stage E [MACE %s] salvage conf %s  "
                             "fmax_final=%.3f eV/Å (≤ %.3f eV/Å)  "
                             "after %d/%d steps",
                             method, r["conf"],
                             fmax_final_eV_per_A,
                             salvage_threshold_eVpA,
                             stage_e_mace_max_steps,
                             stage_e_mace_max_steps)
                else:
                    ok = False
                    log.warning("  Stage E [MACE %s] conf %s NOT "
                                "converged (fmax_final=%.3f eV/Å > "
                                "salvage=%.3f eV/Å)",
                                method, r["conf"],
                                fmax_final_eV_per_A,
                                salvage_threshold_eVpA)
                final_grad_ratio_e = (fmax_final_EhBohr
                                       / max(stage_e_salvage_gradnorm_max,
                                             1e-12))
                log.info("  Stage E [MACE %s] conf %s  "
                         "E0=%.4f eV  Ef=%.4f eV  ΔE=%+.4f eV  "
                         "converged=%s  fmax=%.3f eV/Å  ok=%s  wall=%.1fs",
                         method, r["conf"], e0_eV, e_final_eV,
                         e_final_eV - e0_eV, converged,
                         fmax_final_eV_per_A, ok,
                         time.time() - t_start)
            except Exception as exc:  # pragma: no cover — defensive
                log.exception("Stage E [MACE %s] conformer %s crashed: %s",
                              method, r["conf"], exc)
                ok = False
                rc = -1

        return {
            "conf": int(r["conf"]),
            "crest_rank": int(r["crest_rank"]),
            "gxtb_energy_Eh": float(r["gxtb_energy_Eh"]),
            "water_relax_ok": int(ok),
            "water_relax_timed_out": int(timed_out),
            "water_relax_energy_Eh": e_final,
            "water_relax_extensions_used": extensions_used,
            "water_relax_salvaged": int(salvaged_e),
            "water_relax_final_grad_ratio": (final_grad_ratio_e
                                             if final_grad_ratio_e is not None
                                             else float("nan")),
            "relaxed_xyz": str(opt) if opt.exists() else "",
        }

    if workers == 1:
        rows = [_run_one(r) for r in ranked]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_run_one, ranked))

    # Survival check — abort only if literally nothing usable came back.
    survivors = sum(1 for r in rows if int(r["water_relax_ok"]) == 1)
    n_timed_out = sum(1 for r in rows if int(r.get("water_relax_timed_out", 0)) == 1)
    log.info("Stage E summary: %d/%d conformer(s) survived (%d timed out)",
             survivors, len(rows), n_timed_out)
    if rows and survivors < min_stage_e_survivors:
        raise RuntimeError(
            f"Stage E: only {survivors} conformer(s) succeeded "
            f"(< min_stage_e_survivors={min_stage_e_survivors}). "
            f"Timeouts: {n_timed_out}/{len(rows)}. Consider raising "
            f"--per-job-timeout.")

    # NaN-safe sort: failed conformers (or NaN energies) sink to the bottom.
    def _stage_e_rank_key(r: dict) -> tuple[int, float]:
        e = r.get("water_relax_energy_Eh", float("nan"))
        if not isinstance(e, float) or math.isnan(e):
            return (1, float("inf"))
        return (1 - int(r["water_relax_ok"]), e)
    rows.sort(key=_stage_e_rank_key)
    csv_path = d / "ranked.csv"
    with csv_path.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # final summary
    final = out_root / "final_ranked.csv"
    with final.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("  → %s", final)
    return csv_path


# ----- protonation ensemble wrapper ------------------------------------------
def _run_protonation_ensemble_wrapper(args, p) -> int:
    """Drive the funnel over a protonation-state ensemble.

    Resolves the ensemble (auto / manual rules), generates per-variant PDBs
    via ``microstate_sampler.sample_protonation_microstates``, then re-runs
    the FULL crest_funnel main path on each variant in sequence, writing
    per-variant outputs into ``--out / variant_<NN>/``. Finally collects
    the per-variant final_ranked.csv files, picks the lowest g-xTB energy
    from each, and writes ``--out / ensemble_ranked.csv`` ordered by
    energy.

    The non-ensemble path of main() is unchanged.
    """
    # Parse ensemble mode
    pe = args.protonation_ensemble
    rules_yaml = None
    if pe == "auto":
        mode = "auto"
    elif pe.startswith("manual:"):
        mode = "rules"
        rules_yaml = pe[len("manual:"):]
        if not rules_yaml:
            raise SystemExit(
                "--protonation-ensemble manual: requires a rules YAML path "
                "(e.g. --protonation-ensemble manual:rules.yaml)"
            )
    else:
        raise SystemExit(
            f"--protonation-ensemble must be 'none', 'auto', or "
            f"'manual:rules.yaml'; got {pe!r}"
        )

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Generate the protonation variants. We import here to avoid pulling
    # the protonation deps at module-import time for non-ensemble runs.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools.microstate_sampler import sample_protonation_microstates  # noqa: WPS433
    from quantum_engine.ops.charge_ledger import load_ledger as _load_ledger

    ledger = None
    if getattr(args, "charge_ledger", None):
        ledger = _load_ledger(args.charge_ledger)

    variant_dir = out_root / "00_protonation_variants"
    log.info("[protonation-ensemble] generating variants in %s "
             "(mode=%s, max_microstates=%d, keep_top=%d)",
             variant_dir, mode, args.protonation_max_microstates,
             args.protonation_keep_top)
    batch = sample_protonation_microstates(
        args.pdb,
        variant_dir,
        mode=mode,
        rules_yaml=rules_yaml,
        max_microstates=args.protonation_max_microstates,
        metal_cutoff_a=args.protonation_metal_cutoff_a,
        nh_bond_length=args.protonation_nh_bond,
        oh_bond_length=args.protonation_oh_bond,
        sh_bond_length=args.protonation_sh_bond,
        charge_ledger=ledger,
        cli_charge=args.charge,
    )
    if batch.n_variants == 0:
        raise SystemExit(
            "[protonation-ensemble] zero variants generated. Check the "
            "input PDB has titratable residues, or supply rules YAML."
        )

    log.info("[protonation-ensemble] %d variants generated; running CREST "
             "funnel on each", batch.n_variants)

    # Per-variant funnel runs: re-invoke ourselves as a subprocess with the
    # variant PDB and a per-variant output dir. We strip --protonation-* and
    # --max-microstates flags so we don't recurse.
    cleaned_no_io: list[str] = []
    skip_next = False
    PROTONATION_FLAGS = {
        "--protonation-ensemble",
        "--protonation-max-microstates",
        "--max-microstates",
        "--protonation-keep-top",
        "--protonation-metal-cutoff-a",
        "--protonation-nh-bond",
        "--protonation-oh-bond",
        "--protonation-sh-bond",
        "--charge-ledger",
        "--charge",       # per-variant charge is computed below
        "--pdb", "--out",
    }
    for tok in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        # Handle "--flag=value" form: just check the prefix
        eq_split = tok.split("=", 1)
        flag_part = eq_split[0]
        if flag_part in PROTONATION_FLAGS:
            if len(eq_split) == 1:
                # Followed by a separate value token; skip both
                skip_next = True
            continue
        cleaned_no_io.append(tok)

    variant_results: list[dict] = []
    for ms in batch.microstates:
        var_outdir = out_root / f"variant_{ms.label[-4:]}"
        # Extract per-variant total charge from the variant PDB REMARK lines
        # (sample_protonation_microstates writes 'REMARK   2 QCB TOTAL_CHARGE +N').
        var_charge = args.charge
        try:
            for line in Path(ms.pdb_path).read_text().splitlines():
                if "QCB TOTAL_CHARGE" in line:
                    parts = line.split()
                    var_charge = int(parts[-1])
                    break
        except Exception as exc:
            log.warning("could not infer per-variant charge for %s: %s",
                        ms.pdb_path, exc)
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--pdb", str(ms.pdb_path),
            "--out", str(var_outdir),
        ]
        if var_charge is not None:
            cmd.extend(["--charge", str(int(var_charge))])
        cmd += cleaned_no_io
        log.info("[protonation-ensemble] running funnel on %s -> %s "
                 "(charge=%s)", ms.pdb_path, var_outdir, var_charge)
        rc = subprocess.run(cmd, check=False).returncode
        # Read best energy: prefer Stage E (water_relax) > Stage C (gxtb) >
        # Stage A (xtb_preopt). Fall back to per-stage CSV when DE is omitted
        # via --stages (codex flag, 2026-05-07).
        best_energy_eh: float = float("nan")
        for csv_rel in (
            "final_ranked.csv",
            "30_gxtb_minimize/ranked.csv",
            "40_water_relax/ranked.csv",
        ):
            csv_path = var_outdir / csv_rel
            if not csv_path.is_file():
                continue
            try:
                with csv_path.open() as fh:
                    rdr = csv.DictReader(fh)
                    for row in rdr:
                        e = (row.get("water_relax_energy_Eh")
                             or row.get("gxtb_energy_Eh")
                             or row.get("energy_Eh"))
                        if e:
                            try:
                                best_energy_eh = float(e)
                                break
                            except ValueError:
                                pass
            except Exception as exc:
                log.warning("could not parse %s: %s", csv_path, exc)
            if not math.isnan(best_energy_eh):
                break
        variant_results.append({
            "variant_label": ms.label,
            "variant_description": ms.description,
            "variant_pdb": str(ms.pdb_path),
            "variant_outdir": str(var_outdir),
            "best_energy_Eh": best_energy_eh,
            "funnel_rc": int(rc),
        })

    # Rank variants by best-energy
    def _rank(r: dict) -> tuple[int, float]:
        e = r.get("best_energy_Eh", float("nan"))
        if not isinstance(e, float) or math.isnan(e):
            return (1, float("inf"))
        return (0, e)
    variant_results.sort(key=_rank)
    keep_top = max(1, int(args.protonation_keep_top))
    top_variants = variant_results[:keep_top]

    summary_path = out_root / "ensemble_ranked.csv"
    with summary_path.open("w") as fh:
        if variant_results:
            w = csv.DictWriter(fh, fieldnames=list(variant_results[0].keys()))
            w.writeheader()
            w.writerows(variant_results)
    log.info("[protonation-ensemble] ensemble summary -> %s", summary_path)
    log.info("[protonation-ensemble] top %d variants by g-xTB energy:",
             keep_top)
    for i, v in enumerate(top_variants):
        log.info("  %d. %s  E=%s Eh  (%s)",
                 i + 1, v["variant_label"], v["best_energy_Eh"],
                 v["variant_description"])

    # Also write a JSON for programmatic consumers
    (out_root / "ensemble_summary.json").write_text(json.dumps({
        "mode": mode, "rules_yaml": rules_yaml,
        "n_variants": len(variant_results),
        "keep_top": keep_top,
        "ensemble_ranked": variant_results,
        "top_variants": top_variants,
    }, indent=2))

    # Surface variant failures as a non-zero exit code so SLURM jobs / orchestrators
    # don't silently treat a totally failed ensemble as "completed". Codex flag,
    # 2026-05-07. We only return non-zero when EVERY variant produced no usable
    # energy (rc != 0 OR best_energy_Eh is NaN); a partial success still returns 0.
    n_total = len(variant_results)
    n_useful = sum(
        1 for v in variant_results
        if int(v.get("funnel_rc", 1)) == 0
        and isinstance(v.get("best_energy_Eh"), float)
        and not math.isnan(v["best_energy_Eh"])
    )
    if n_total > 0 and n_useful == 0:
        log.error("[protonation-ensemble] ALL %d variant funnel runs failed "
                  "(no usable energies). Returning exit code 2.", n_total)
        return 2
    if n_total > 0 and n_useful < n_total:
        log.warning("[protonation-ensemble] %d/%d variants failed; %d salvaged. "
                    "Continuing (partial success).",
                    n_total - n_useful, n_total, n_useful)
    return 0


# ----- driver ---------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdb", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--charge", type=int, default=None,
                   help="override; default infers from filename")
    p.add_argument("--ncpu", type=int, default=max(1, os.cpu_count() or 4))
    p.add_argument("--top-n", type=int, default=10,
                   help="g-xTB minimize this many top CREST conformers")
    p.add_argument("--top-keep", type=int, default=10,
                   help="water-relax this many top g-xTB conformers")
    p.add_argument("--solvent", default="water",
                   help="ALPB solvent name (e.g. 'water', 'methanol') or "
                        "'none' for VACUUM/gas-phase. Applied uniformly to "
                        "all xtb/CREST invocations: pre-CREST relax (Stage "
                        "0.5), Stage A xtb-preopt, Stage B CREST MTD/opt, "
                        "Stage C g-xTB SP refine, and Stage E water-relax. "
                        "(Cap-H relaxes during pruning are always vacuum.) "
                        "Default 'water' for backwards compat with existing "
                        "PTE pipelines; pass '--solvent none' for "
                        "gas-phase preopt of bare active-site clusters.")
    p.add_argument("--crest-preset", default="mquick",
                   choices=["quick", "squick", "mquick", "none"],
                   help="reduced-settings preset (mquick=fastest)")
    p.add_argument("--crest-mdlen", type=float, default=3.0,
                   help="MTD time in ps PER MTD (CREST runs 6 MTDs)")
    p.add_argument("--crest-rthr", type=float, default=0.5,
                   help="CREST RMSD clustering threshold in Å. Default tightened "
                        "from CREST upstream defaults to 0.5 to reduce "
                        "post-CREST geometry artifacts (fewer, cleaner distinct "
                        "conformers; see --post-crest-bond-cutoff for the "
                        "safety net). CLI-overridable in either direction: "
                        "drop to 0.05 to keep more near-degenerate conformers "
                        "in a heavily-constrained TS basin, or push higher "
                        "(0.5–1.0) to be more aggressive about deduplication.")
    p.add_argument("--crest-mddump", type=int, default=200,
                   help="trajectory dump interval in fs (larger=fewer frames "
                        "to optimize → faster ensemble-opt phase)")
    p.add_argument("--crest-opt-level", default="loose",
                   choices=["crude", "vloose", "loose", "normal", "tight"],
                   help="multilevel-opt level inside CREST. Default tightened "
                        "from CREST upstream 'crude' to 'loose' to give the "
                        "post-MTD ensemble opt a slightly longer relax pass — "
                        "this reduces the rate at which unphysical short bonds "
                        "leak through to Stage C (see --post-crest-bond-cutoff "
                        "for the safety net). CLI-overridable: drop back to "
                        "'crude' if Stage B walltime is the bottleneck, or "
                        "push to 'normal'/'tight' if you want even cleaner "
                        "geometries before the filter.")
    p.add_argument("--xtb-preopt-timeout", type=int, default=3600,
                   help="Stage A xtb-preopt subprocess timeout in seconds. "
                        "Bump if your input is large or solvent is slow. "
                        "Default 3600 (1 hour).")
    p.add_argument("--gxtb-sp-timeout", type=int, default=600,
                   help="Stage C g-xTB single-point timeout (per-conformer) in "
                        "seconds. Default 600 (10 min).")
    p.add_argument("--per-job-timeout", type=int, default=900,
                   help="Per-conformer xtb refinement timeout (Stage C single-tier "
                        "and Stage E parallel workers), seconds. On timeout the "
                        "worker is SIGTERM'd (so xtb can flush its partial "
                        "xtbopt.xyz), gets stage_*_sigterm_grace_s before SIGKILL, "
                        "and is SKIPPED — it does NOT abort the whole funnel. If "
                        "the partial geometry has a small final gradient norm "
                        "(<= --stage-c-salvage-gradnorm-max), it is salvaged. "
                        "Default 900 (15 min); on a 200-atom solvated cluster, "
                        "tight+alpb-water that takes >15 min is almost certainly "
                        "stuck on bad geometry. Raise for unusually large "
                        "systems.")
    p.add_argument("--stage-c-opt-level", default="tight",
                   choices=["crude", "vloose", "loose", "normal", "tight",
                            "vtight", "extreme"],
                   help="xtb --opt level for Stage C SINGLE-TIER mode "
                        "(--stage-c-tiers single, default). Default 'tight' to "
                        "match the user's high-resolution requirement. Bump down "
                        "to 'normal'/'loose' only if Stage C is too slow AND "
                        "downstream ranking quality is acceptable. (For two-tier "
                        "mode see --stage-c-tier1-opt-level / "
                        "--stage-c-tier2-opt-level.)")
    p.add_argument("--stage-c-tiers", default="single",
                   choices=["single", "two"],
                   help="Stage C optimization strategy. 'single' (default) runs "
                        "one parallel pass at --stage-c-opt-level over all top-N "
                        "CREST conformers, with per-conformer skip-on-timeout. "
                        "'two' runs a CHEAP pre-pass (loose, short timeout) on "
                        "all top-N to filter outliers + bad geometries, then a "
                        "TIGHT pass on the lowest-energy --stage-c-tier1-keep-top "
                        "survivors. Two-tier saves wall time when many CREST "
                        "conformers are bad starting points; both tiers KEEP "
                        "--alpb water. Default 'single' for backwards "
                        "compatibility.")
    p.add_argument("--stage-c-tier1-opt-level", default="loose",
                   choices=["crude", "vloose", "loose", "normal", "tight",
                            "vtight", "extreme"],
                   help="xtb --opt level for Stage C TIER 1 (cheap pre-pass) "
                        "when --stage-c-tiers two. Default 'loose'.")
    p.add_argument("--stage-c-tier2-opt-level", default="tight",
                   choices=["crude", "vloose", "loose", "normal", "tight",
                            "vtight", "extreme"],
                   help="xtb --opt level for Stage C TIER 2 (tight pass on "
                        "survivors) when --stage-c-tiers two. Default 'tight' to "
                        "preserve the user's high-resolution final-stage "
                        "requirement.")
    p.add_argument("--stage-c-tier1-timeout", type=int, default=600,
                   help="Per-conformer Stage C TIER 1 timeout in seconds. "
                        "Default 600 (10 min). Skip-on-timeout applies; partial "
                        "salvage applies. Used only when --stage-c-tiers two.")
    p.add_argument("--stage-c-tier2-timeout", type=int, default=1800,
                   help="Per-conformer Stage C TIER 2 timeout in seconds. "
                        "Default 1800 (30 min). Skip-on-timeout applies; partial "
                        "salvage applies. Used only when --stage-c-tiers two.")
    p.add_argument("--stage-c-tier1-keep-top", type=int, default=5,
                   help="When --stage-c-tiers two: number of lowest-energy "
                        "tier-1 survivors to promote to tier 2. Default 5.")
    p.add_argument("--stage-c-energy-outlier-cutoff", type=float, default=None,
                   help="If set (kcal/mol), drop Stage C tier-1 / single-tier "
                        "conformers whose post-opt GFN2 energy is more than this "
                        "above the lowest-energy success. Default None (off). "
                        "Typical: 15.0 to filter clear bad-geometry outliers "
                        "without nuking interesting near-degenerate basins.")
    p.add_argument("--stage-c-sigterm-grace-s", type=float, default=30.0,
                   help="When a Stage C conformer hits its per-job timeout, give "
                        "the xtb process this many seconds to respond to SIGTERM "
                        "(and flush a partial xtbopt.xyz) before SIGKILL. "
                        "Default 30.")
    p.add_argument("--stage-c-salvage-gradnorm-max", type=float, default=0.05,
                   help="Last-iter gradient norm threshold (Eh/Bohr) below which "
                        "a SIGTERM'd partial xtbopt.xyz is treated as "
                        "successfully relaxed. Default 0.05 ≈ xtb 'normal' "
                        "convergence; raise to 0.1 to be more permissive, or "
                        "0.01 for tight only. Set to 0 to disable salvage. "
                        "Note: this is the legacy ABSOLUTE-gnorm rule. The "
                        "Strategy-E grad-RATIO rule "
                        "(--stage-c-salvage-grad-ratio) takes precedence when "
                        "the adaptive monitor accepts a salvage; this absolute "
                        "rule is still a fallback so backwards-compat callers "
                        "have a salvage path.")
    p.add_argument("--stage-c-progress-check-fraction", type=float, default=0.6,
                   help="Strategy E (adaptive timeout): when this FRACTION of "
                        "the base Stage C tier-1 / single-tier timeout has "
                        "elapsed, the sidecar reads xtbopt.log to decide "
                        "whether to extend. Default 0.6 (60%%). Set <=0 to "
                        "disable adaptive extension.")
    p.add_argument("--stage-c-progress-grad-ratio", type=float, default=2.0,
                   help="Strategy E: extension fires only if the current RMS "
                        "gradient is within this MULTIPLE of the opt-level's "
                        "convergence target. Default 2.0 (within 2× of "
                        "convergence). Lower = stricter = fewer extensions.")
    p.add_argument("--stage-c-salvage-grad-ratio", type=float, default=1.5,
                   help="Strategy E: salvage cutoff. After hard kill, if the "
                        "last RMS gradient is within this multiple of target, "
                        "the partial xtbopt.xyz is accepted as 'good enough'. "
                        "Default 1.5 (within 1.5× of full convergence). Wider "
                        "than --stage-c-progress-grad-ratio because we are "
                        "now committed: better to keep a near-converged "
                        "geometry than retry from scratch.")
    p.add_argument("--stage-c-max-extensions", type=int, default=1,
                   help="Strategy E: maximum number of timeout extensions "
                        "permitted per conformer per tier. Default 1 — i.e. "
                        "'cut off truly stuck calls in 10 min, give "
                        "converging ones up to 20 min'. Set to 0 to disable "
                        "extensions entirely (back to task #25 behaviour).")
    p.add_argument("--stage-c-extension-factor", type=float, default=1.0,
                   help="Strategy E: each extension adds this MULTIPLE of the "
                        "base timeout to the deadline. Default 1.0 → with "
                        "max-extensions=1, total budget can grow to 2× base "
                        "timeout. Bump to 1.5 if you want 'up to 25 min from "
                        "10 min base'.")
    p.add_argument("--min-stage-c-survivors", type=int, default=1,
                   help="Abort the funnel only if FEWER than this many Stage C "
                        "conformers succeeded after parallel skip-on-timeout. "
                        "Default 1 (any survivor proceeds).")
    p.add_argument("--stage-c-method", choices=STAGE_C_METHOD_CHOICES,
                   default="xtb",
                   help="Backend for Stage C g-xTB / GFN2 minimize. "
                        "'xtb' (default, historical) honours --solvent for "
                        "ALPB and the full Strategy-E grad-monitor / salvage "
                        "stack; 'mace-mp' / 'mace-polar-m' run an in-process "
                        "ASE LBFGS in vacuum (MACE has no implicit solvent). "
                        "Both backends apply the SAME constraint set "
                        "($fix on CA atoms minus --free-residues; "
                        "FixBondLengths on the reactive P-Onuc / P-Olg pair). "
                        "Default 'xtb' preserves byte-identical historical "
                        "behaviour.")
    p.add_argument("--stage-c-mace-fmax", type=float, default=0.05,
                   help="ASE fmax (eV/Å) for --stage-c-method mace-* backends. "
                        "Default 0.05.")
    p.add_argument("--stage-c-mace-max-steps", type=int, default=200,
                   help="ASE max optimizer steps for --stage-c-method mace-* "
                        "backends. Default 200. Bump to 400 for harder "
                        "active-site clusters.")
    p.add_argument("--stage-c-device", default="cuda",
                   choices=["cuda", "cpu"],
                   help="Device for --stage-c-method mace-* backends. "
                        "Default 'cuda'.")
    # Post-CREST geometry filter (Stage B → Stage C safety net) ---------------
    p.add_argument("--post-crest-bond-cutoff", type=float,
                   default=POST_CREST_BOND_CUTOFF_DEFAULT,
                   help="Heavy-heavy bond cutoff in Å below which to FLAG "
                        "a CREST-output conformer's bond as candidate-"
                        "suspect. Default "
                        f"{POST_CREST_BOND_CUTOFF_DEFAULT:.3f} Å, chosen ABOVE "
                        "the observed CREST 1.073 Å artifact (HIS 257 N-CA "
                        "in PTE KCX_set3 conf_01). Triple bonds (C≡N ~1.16 Å, "
                        "C≡C ~1.20 Å) ARE above this cutoff and would not "
                        "flag, but if a source-PDB bond exists the source-"
                        "distance comparison will tolerate it via "
                        "--post-crest-source-shrink-tolerance regardless. "
                        "H-X bonds are skipped entirely (heavy-heavy only). "
                        "Lower below 1.0 only as a last resort: combined "
                        "with the source-PDB lookup the default 1.10 Å "
                        "already does the right thing for triple bonds AND "
                        "catches the observed 1.073 Å artifact.")
    p.add_argument("--post-crest-bad-bond-mode",
                   default="reject",
                   choices=["reject", "repair", "keep", "log"],
                   help="Behavior when a candidate conformer has heavy-heavy "
                        "bonds shorter than --post-crest-bond-cutoff. "
                        "'reject' (default, conservative): drop the conformer. "
                        "'repair': iteratively push the more-peripheral atom "
                        "of each clashing pair out to an element-pair single-"
                        "bond target distance; saves the original geometry as "
                        "30_gxtb_minimize/conf_NN/input_pre_repair.xyz so you "
                        "can diff. xtb will relax the rest in Stage C. 'keep': "
                        "legacy passthrough (just WARNING-log the bad bond). "
                        "'log': passthrough but emit a structured DIAGNOSTIC "
                        "line per bad bond — useful when investigating what "
                        "CREST is producing without changing pipeline output. "
                        "EXAMPLES: '--post-crest-bad-bond-mode reject' (the "
                        "default; safest); '--post-crest-bad-bond-mode repair "
                        "--post-crest-bond-cutoff 1.05' (recover salvageable "
                        "near-clashes); '--post-crest-bad-bond-mode log' "
                        "(diagnose without touching the run).")
    p.add_argument("--post-crest-min-survivors", type=int, default=1,
                   help="Minimum conformers required AFTER the post-CREST "
                        "geometry filter. Default 1. If fewer survive, "
                        "Stage C raises a RuntimeError with actionable "
                        "advice (switch to repair mode, raise the cutoff, "
                        "or re-sample with tighter --crest-rthr). Tighten "
                        "to e.g. 3 if you depend on diversity for downstream "
                        "ranking; loosen to 0 to make the filter advisory "
                        "(combined with mode='log').")
    p.add_argument("--post-crest-repair-max-passes", type=int,
                   default=POST_CREST_REPAIR_MAX_PASSES_DEFAULT,
                   help="Iterative repair passes per conformer when "
                        "--post-crest-bad-bond-mode=repair. Each pass fixes "
                        "the worst (shortest) remaining bad bond on the "
                        "current geometry, so cascading clashes (atom k now "
                        "too close to atom l after fixing i-j) get caught on "
                        f"later passes. Default {POST_CREST_REPAIR_MAX_PASSES_DEFAULT}. "
                        "If residual bad bonds remain after this many passes, "
                        "the conformer is rejected (falls through to reject "
                        "behavior with a log line).")
    p.add_argument("--post-crest-source-shrink-tolerance", type=float,
                   default=POST_CREST_SOURCE_SHRINK_TOLERANCE_DEFAULT,
                   help="Source-PDB bond-distance tolerance for the post-"
                        "CREST filter. When the source PDB has the same "
                        "atom pair bonded at distance source_d, treat the "
                        "post-CREST distance post_d as a normal vibration "
                        "(NO repair, NO reject) when post_d/source_d >= "
                        f"this tolerance. Default {POST_CREST_SOURCE_SHRINK_TOLERANCE_DEFAULT:.2f} "
                        "(post-CREST bond can be up to 30%% shorter than "
                        "source before it is treated as an MTD artifact). "
                        "Raise toward 0.85 to be stricter — catches the "
                        "observed PTE KCX_set3 conf_01 N-CA case (ratio "
                        "0.727); raise to 1.0 to flag any sub-source bond. "
                        "Lower toward 0.5 to be more permissive on real "
                        "vibrations. Only takes effect when a source-PDB "
                        "bond IS found for the offending pair; if not, the "
                        "element-pair fallback table is used regardless of "
                        "this value. Note: at default 0.70 the observed "
                        "1.073 Å N-CA artifact (source 1.475 Å, ratio "
                        "0.727) is TOLERATED — the downstream xtb relaxation "
                        "(with task #25/#26 timeout-skip + salvage) is "
                        "responsible for handling failed-SCF cases. Set to "
                        "0.75 to force-repair such cases via the source-"
                        "PDB distance.")
    p.add_argument("--post-crest-reactive-atoms", default="",
                   help="Comma-separated list of atom serials (1-based, "
                        "after waters are stripped) and/or NAME.RESNAME "
                        "tokens (e.g. 'P1.SUB,O3.OHX,O7.SUB') identifying "
                        "atoms whose bonds should be PRESERVED — i.e. never "
                        "auto-repaired or rejected by the post-CREST filter, "
                        "only LOGGED. Use to protect TS-like reactive "
                        "distances at unusual ranges (e.g. P-Onuc 2.0 A, "
                        "P-Olg 2.3 A in a TS geometry). Default empty (in "
                        "addition to the always-protected reactive triple "
                        "P-Onuc-Olg). EXAMPLES: '--post-crest-reactive-atoms "
                        "\"P1.SUB,O3.OHX,O7.SUB,O8.SUB\"' protects P + both "
                        "Os + the bridging carbonyl-O; '--post-crest-"
                        "reactive-atoms \"42,43,44\"' uses raw 1-based "
                        "indices into the no-waters atom list.")
    p.add_argument("--post-crest-max-match-distance", type=float, default=0.75,
                   help="Maximum aligned-frame Euclidean distance (Å) for a "
                        "post-CREST atom to be matched to its closest same-"
                        "element source-PDB atom during the bond-order-"
                        "aware repair lookup. Brand-new contacts (post atom "
                        "far from any aligned source atom) remain unmapped "
                        "and fall through to the element-pair fallback "
                        "table. Default 0.75 Å (backbone atoms typically "
                        "drift <0.3 Å during CREST sampling, side chains "
                        "<0.5 Å, so 0.75 is a comfortable safety margin). "
                        "Raise above 1.0 only if your source PDB and CREST "
                        "output differ in unusual ways.")
    p.add_argument("--stage-e-opt-level", default="loose",
                   choices=["crude", "vloose", "loose", "normal", "tight",
                            "vtight", "extreme"],
                   help="xtb --opt level for Stage E water-reinserted relax. "
                        "Default 'loose'.")
    p.add_argument("--stage-e-sigterm-grace-s", type=float, default=30.0,
                   help="SIGTERM-to-SIGKILL grace period for Stage E timeouts, "
                        "seconds. Default 30.")
    p.add_argument("--stage-e-salvage-gradnorm-max", type=float, default=0.05,
                   help="Stage E equivalent of --stage-c-salvage-gradnorm-max. "
                        "Default 0.05 Eh/Bohr.")
    p.add_argument("--stage-e-progress-check-fraction", type=float, default=0.6,
                   help="Stage E equivalent of "
                        "--stage-c-progress-check-fraction. Default 0.6.")
    p.add_argument("--stage-e-progress-grad-ratio", type=float, default=2.0,
                   help="Stage E equivalent of "
                        "--stage-c-progress-grad-ratio. Default 2.0.")
    p.add_argument("--stage-e-salvage-grad-ratio", type=float, default=1.5,
                   help="Stage E equivalent of --stage-c-salvage-grad-ratio. "
                        "Default 1.5.")
    p.add_argument("--stage-e-max-extensions", type=int, default=1,
                   help="Stage E equivalent of --stage-c-max-extensions. "
                        "Default 1.")
    p.add_argument("--stage-e-extension-factor", type=float, default=1.0,
                   help="Stage E equivalent of --stage-c-extension-factor. "
                        "Default 1.0.")
    p.add_argument("--min-stage-e-survivors", type=int, default=1,
                   help="Stage E equivalent of --min-stage-c-survivors. "
                        "Default 1.")
    p.add_argument("--stage-e-method", choices=STAGE_E_METHOD_CHOICES,
                   default="xtb",
                   help="Backend for Stage E water-reinserted relax. "
                        "'xtb' (default, historical) honours --solvent for "
                        "ALPB and the full grad-monitor / salvage stack; "
                        "'mace-mp' / 'mace-polar-m' run an in-process ASE "
                        "LBFGS in vacuum (MACE has no implicit solvent).")
    p.add_argument("--stage-e-mace-fmax", type=float, default=0.05,
                   help="ASE fmax (eV/Å) for --stage-e-method mace-* backends. "
                        "Default 0.05.")
    p.add_argument("--stage-e-mace-max-steps", type=int, default=200,
                   help="ASE max optimizer steps for --stage-e-method mace-* "
                        "backends. Default 200.")
    p.add_argument("--stage-e-device", default="cuda",
                   choices=["cuda", "cpu"],
                   help="Device for --stage-e-method mace-* backends. "
                        "Default 'cuda'.")
    p.add_argument("--optimizer-backend", default="ase-lbfgs",
                   help="Modular optimizer backend for the MACE branches "
                        "of pre-CREST relax, Stage C, and Stage E. "
                        "Choices: ase-lbfgs (default, identical to legacy), "
                        "ase-fire, ase-bfgs, torch-sim-fire, "
                        "torch-sim-lbfgs (stub). torch-sim-* automatically "
                        "falls back to ase-lbfgs if any active constraint "
                        "is incompatible (FixAtoms, FixBondLength).")
    p.add_argument("--crest-walltime", type=int, default=7200,
                   help="seconds before CREST is killed")
    p.add_argument("--min-threads-per-job", type=int, default=4,
                   help="min OMP_NUM_THREADS per parallel xtb/g-xtb worker "
                        "in Stages C+E")
    p.add_argument("--stages", default="0,A,B,C,DE",
                   help="comma list; subset of {0,A,B,C,DE}")
    p.add_argument("--freeze-zn", action="store_true",
                   help="also $fix the Zn atoms (default: only CA frozen, "
                        "Zn allowed to relax)")
    p.add_argument("--cleanup", action="store_true",
                   help="after writing results/, delete intermediate XYZ/log files")

    # ---- free-residues + prune (mirrors polish_ts_v3 / scan_along_s) ----
    constr = p.add_argument_group("free-residues + prune (uniform CLI)")
    constr.add_argument(
        "--free-residues", type=_csv_int, default=[],
        help="Comma-separated chain-A residue ids to EXCLUDE from the "
             "$fix-CA scaffold (their CAs are NOT pinned during Stages A/B/C). "
             "Identical syntax to scan_along_s and polish_ts_v3. "
             "Example: '--free-residues 131,169' frees TRP131 + GLU169 in PTE.",
    )
    constr.add_argument(
        "--prune-residue-keep", nargs="+", default=[], metavar="RESID:ATOMS",
        help="Multi-value: pass MANY tokens after a single flag (NOT repeated "
             "flags — argparse nargs='+' overwrites on repeat). "
             "Each token: RESID:ATOMS = keep ONLY listed heavy atoms (plus "
             "their attached H atoms) of residue RESID; drop the rest. "
             "H-caps placed at cut bonds. Empty atom list (e.g. '131:') "
             "drops ALL heavy atoms in the residue. "
             "Example (correct): '--prune-residue-keep 169:CD,CE,NZ 254:CG,CD2'. "
             "Example (WRONG): '--prune-residue-keep 169:... --prune-residue-keep "
             "254:...' — only the second flag's tokens survive.",
    )
    constr.add_argument(
        "--prune-backbone-residues", type=_csv_int, default=[],
        help="Comma-separated residue ids: drop backbone N/C/O/HXT/H/HA "
             "atoms; keep sidechain. H-caps placed at cut bonds. "
             "Example: '--prune-backbone-residues 131,169'.",
    )
    constr.add_argument(
        "--cap-h-bond", type=float, default=1.09,
        help="Initial bond length (Å) of cap H placed at cut bonds during "
             "pruning. Default 1.09 (typical C-H).",
    )
    constr.add_argument(
        "--prune-xtb-relax", action="store_true", default=True,
        help="After H-cap placement, run a fast xTB-GFN2 partial-relax of "
             "JUST the cap H atoms (everything else FixAtoms'd). Default ON. "
             "Use --no-prune-xtb-relax to skip.",
    )
    constr.add_argument("--no-prune-xtb-relax", dest="prune_xtb_relax",
                         action="store_false")
    constr.add_argument(
        "--prune-xtb-max-steps", type=int, default=100,
        help="Max optimizer steps for the cap-H xTB relax. Default 100.",
    )

    # ---- pre-CREST relaxation (optional cleanup before Stage A) ----
    pre = p.add_argument_group("pre-CREST relaxation (optional)")
    pre.add_argument(
        "--pre-crest-relax",
        choices=PRE_CREST_RELAX_CHOICES, default="none",
        help="Optionally run a constrained relaxation on the no-waters cluster "
             "BEFORE Stage A pre-opt. Uses the SAME constraint pattern as "
             "Stage A — fix CAs (minus --free-residues) + pin reactive "
             "distances. Useful when the source PDB has imperfect cap-H or "
             "sidechain placements that would slow CREST's MTD. Choices: "
             "'none' (default; current behavior), 'xtb-loose' / 'xtb-tight' "
             "(xtb GFN2 partial relax), 'mace-mp' / 'mace-polar-m' (ASE+MACE).",
    )
    pre.add_argument(
        "--pre-crest-fmax", type=float, default=0.05,
        help="ASE fmax (eV/Å) for --pre-crest-relax mace-* backends. "
             "Default 0.05.",
    )
    pre.add_argument(
        "--pre-crest-max-steps", type=int, default=200,
        help="Max optimizer steps for the pre-CREST relax. Default 200.",
    )
    pre.add_argument(
        "--pre-crest-timeout", type=int, default=1800,
        help="Subprocess timeout (seconds) for xtb-* pre-CREST relax. "
             "Default 1800 (30 min).",
    )
    pre.add_argument(
        "--pre-crest-device", default="cuda", choices=["cuda", "cpu"],
        help="Device for --pre-crest-relax mace-* backends. Default 'cuda'. "
             "Use 'cpu' on nodes without a GPU. (Has no effect on xtb-* "
             "backends.)",
    )

    # ---- Protonation-state ensemble (multi-variant CREST runs) -------------
    p.add_argument("--protonation-ensemble", default="none",
                   help=(
                       "Run CREST on a protonation-state ENSEMBLE rather "
                       "than a single starting structure. Values:\n"
                       "  'none' (default) — current behavior, single PDB.\n"
                       "  'auto' — auto-enumerate HIS/ASP/GLU/LYS/CYS "
                       "tautomer/protonation variants of --pdb (capped at "
                       "--protonation-max-microstates), run the FULL CREST "
                       "funnel on each variant in sequence, rank by g-xTB "
                       "energy, keep --protonation-keep-top.\n"
                       "  'manual:rules.yaml' — same, but use explicit "
                       "residue + state lists from rules.yaml (see "
                       "tools/microstate_sampler.py for schema).\n\n"
                       "Why: CREST's MTD bias forces preserve covalent "
                       "connectivity, so CREST cannot explore tautomers / "
                       "protonation states by itself — they MUST be "
                       "enumerated externally and fed in as separate "
                       "inputs. This wrapper does that automation."
                   ))
    p.add_argument("--protonation-max-microstates", "--max-microstates",
                   dest="protonation_max_microstates",
                   type=int, default=8,
                   help="Cap on protonation variants to enumerate. Default "
                        "8 — balanced for the ~hours-per-variant CREST "
                        "funnel cost. Bump to 16 / 32 for exhaustive "
                        "coverage. Auto mode prunes via consensus + "
                        "1-residue perturbations when the full Cartesian "
                        "product exceeds this. Also accepts "
                        "--max-microstates as an alias.")
    p.add_argument("--protonation-keep-top", type=int, default=3,
                   help="Number of top-energy protonation variants to "
                        "retain in the final summary. Default 3.")
    p.add_argument("--protonation-metal-cutoff-a", type=float, default=3.5,
                   help="HIS within this many Å of any metal uses HID as "
                        "the consensus tautomer (auto mode only). "
                        "Default 3.5.")
    p.add_argument("--protonation-nh-bond", type=float, default=1.01,
                   help="N-H placement target length (Å). Default 1.01 "
                        "(Allen et al. CSD).")
    p.add_argument("--protonation-oh-bond", type=float, default=0.96,
                   help="O-H placement target length (Å). Default 0.96 "
                        "(Allen et al. CSD).")
    p.add_argument("--protonation-sh-bond", type=float, default=1.34,
                   help="S-H placement target length (Å). Default 1.34 "
                        "(Allen et al. CSD).")
    p.add_argument("--charge-ledger", default=None,
                   help="Optional charge-ledger.yaml propagated into "
                        "protonation variants (delta charges accumulate "
                        "per variant). Optional; CLI --charge is honored "
                        "otherwise.")

    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )

    args.out.mkdir(parents=True, exist_ok=True)

    # ----- Charged-vacuum warning ------------------------------------------
    # Empirical lesson (PTE iteration campaign 2026-05-06, MEMORY.md):
    # vacuum xtb on charged active-site clusters has been observed to
    # consistently fail at Stage C with 0/N survivors regardless of opt
    # level (loose / normal / tight all blow up SCF or fail to converge
    # before the per-job timeout). The cause is the unscreened Coulomb
    # field of the net charge in vacuum; ALPB(water) screens that and
    # makes the geometry tractable. We warn — but DO NOT BLOCK — so the
    # user can override for diagnostic runs.
    _solvent_str = (args.solvent or "").lower()
    _charge_val = args.charge if args.charge is not None else 0
    if _solvent_str == "none" and int(_charge_val) != 0:
        log.warning("=" * 70)
        log.warning("CHARGED-VACUUM CONFIGURATION DETECTED")
        log.warning("  --solvent none  AND  --charge=%+d", int(_charge_val))
        log.warning("Empirical observation (PTE 2026-05-06 campaign): vacuum "
                    "xtb on charged active-site clusters has been observed to "
                    "fail at Stage C (0/N survivors) regardless of opt level.")
        log.warning("Suggested fix: drop --solvent none and let the pipeline "
                    "use ALPB water (the default) for charged systems.")
        log.warning("Continuing anyway — this is a warning, not a block.")
        log.warning("=" * 70)
    # -----------------------------------------------------------------------

    # If the user requested an ensemble, dispatch to the wrapper BEFORE the
    # legacy single-PDB pipeline runs. The wrapper handles per-variant
    # CREST runs internally.
    if args.protonation_ensemble and args.protonation_ensemble != "none":
        return _run_protonation_ensemble_wrapper(args, p)

    fh = logging.FileHandler(args.out / "pipeline.log", mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info("=" * 70)
    log.info("crest_funnel  pdb=%s  out=%s", args.pdb, args.out)
    log.info("ncpu=%d  top_n=%d  top_keep=%d  solvent=%s",
             args.ncpu, args.top_n, args.top_keep, args.solvent)

    solvent = None if args.solvent.lower() == "none" else args.solvent
    stages = set(s.strip() for s in args.stages.split(","))

    import dataclasses
    # Stage 0 always runs (we need the partition object); but we keep its files.
    prune_keep_specs = _parse_keep_specs(args.prune_residue_keep)
    part = stage_0_input(
        args.pdb, args.out,
        freeze_zn=args.freeze_zn,
        free_residues=args.free_residues,
        prune_residue_keep=prune_keep_specs,
        prune_backbone_residues=args.prune_backbone_residues,
        cap_h_bond=args.cap_h_bond,
        prune_xtb_relax=args.prune_xtb_relax,
        prune_xtb_max_steps=args.prune_xtb_max_steps,
        charge_override=args.charge,
    )

    # Stage 0 hard validation gates — catch silently-broken parses before we
    # burn an hour of CREST on garbage.
    if not part.fix_indices_no_waters:
        raise RuntimeError("Stage 0: zero $fix anchors. CA-by-chain-A filter "
                           "matched nothing — check chain ID convention "
                           "(or, if --free-residues is set, you may have "
                           "freed every CA; reduce that list).")
    if part.d_p_onuc > 5.0 or part.d_p_olg > 5.0:
        raise RuntimeError(f"Stage 0: reactive distances suspicious "
                           f"d(P-Onuc)={part.d_p_onuc:.2f} d(P-Olg)={part.d_p_olg:.2f} "
                           "— wrong atom indices?")

    # Optional pre-CREST relax: fast cleanup of the input geometry before
    # Stage A. Same constraint pattern as Stage A. CLI default is 'none' so
    # behavior with no flags matches the legacy pipeline byte-for-byte.
    if args.pre_crest_relax and args.pre_crest_relax != "none":
        stage_pre_crest_relax(
            args.out, part,
            method=args.pre_crest_relax,
            fmax=args.pre_crest_fmax,
            max_steps=args.pre_crest_max_steps,
            ncpu=args.ncpu,
            solvent=solvent,
            timeout_s=args.pre_crest_timeout,
            device=args.pre_crest_device,
            optimizer_backend=args.optimizer_backend,
        )

    if "A" in stages:
        stage_A_xtb_preopt(args.out, part, args.ncpu, solvent=solvent,
                           xtb_preopt_timeout=args.xtb_preopt_timeout)
    if "B" in stages:
        stage_B_crest(args.out, part, args.ncpu, solvent=solvent,
                      preset=args.crest_preset, mdlen_ps=args.crest_mdlen,
                      rthr=args.crest_rthr, mddump_fs=args.crest_mddump,
                      opt_level=args.crest_opt_level,
                      walltime_s=args.crest_walltime)
    if "C" in stages:
        stage_C_gxtb(args.out, part, top_n=args.top_n, ncpu=args.ncpu,
                     solvent=solvent,
                     per_job_timeout=args.per_job_timeout,
                     gxtb_sp_timeout=args.gxtb_sp_timeout,
                     min_threads_per_job=args.min_threads_per_job,
                     stage_c_opt_level=args.stage_c_opt_level,
                     stage_c_tiers=args.stage_c_tiers,
                     stage_c_tier1_opt_level=args.stage_c_tier1_opt_level,
                     stage_c_tier1_timeout=args.stage_c_tier1_timeout,
                     stage_c_tier2_opt_level=args.stage_c_tier2_opt_level,
                     stage_c_tier2_timeout=args.stage_c_tier2_timeout,
                     stage_c_tier1_keep_top=args.stage_c_tier1_keep_top,
                     stage_c_energy_outlier_cutoff_kcal=args.stage_c_energy_outlier_cutoff,
                     stage_c_sigterm_grace_s=args.stage_c_sigterm_grace_s,
                     stage_c_salvage_gradnorm_max=args.stage_c_salvage_gradnorm_max,
                     min_stage_c_survivors=args.min_stage_c_survivors,
                     stage_c_progress_check_fraction=args.stage_c_progress_check_fraction,
                     stage_c_progress_grad_ratio=args.stage_c_progress_grad_ratio,
                     stage_c_salvage_grad_ratio=args.stage_c_salvage_grad_ratio,
                     stage_c_max_extensions=args.stage_c_max_extensions,
                     stage_c_extension_factor=args.stage_c_extension_factor,
                     stage_c_method=args.stage_c_method,
                     stage_c_mace_fmax=args.stage_c_mace_fmax,
                     stage_c_mace_max_steps=args.stage_c_mace_max_steps,
                     stage_c_device=args.stage_c_device,
                     optimizer_backend=args.optimizer_backend,
                     post_crest_bond_cutoff_a=args.post_crest_bond_cutoff,
                     post_crest_bad_bond_mode=args.post_crest_bad_bond_mode,
                     post_crest_min_survivors=args.post_crest_min_survivors,
                     post_crest_repair_max_passes=args.post_crest_repair_max_passes,
                     post_crest_source_shrink_tolerance=args.post_crest_source_shrink_tolerance,
                     post_crest_reactive_atoms_spec=args.post_crest_reactive_atoms,
                     post_crest_max_match_distance_a=args.post_crest_max_match_distance)
    if "DE" in stages:
        stage_E_water_relax(args.out, part, args.ncpu, top_keep=args.top_keep,
                             solvent=solvent,
                             per_job_timeout=args.per_job_timeout,
                             min_threads_per_job=args.min_threads_per_job,
                             stage_e_opt_level=args.stage_e_opt_level,
                             stage_e_sigterm_grace_s=args.stage_e_sigterm_grace_s,
                             stage_e_salvage_gradnorm_max=args.stage_e_salvage_gradnorm_max,
                             min_stage_e_survivors=args.min_stage_e_survivors,
                             stage_e_progress_check_fraction=args.stage_e_progress_check_fraction,
                             stage_e_progress_grad_ratio=args.stage_e_progress_grad_ratio,
                             stage_e_salvage_grad_ratio=args.stage_e_salvage_grad_ratio,
                             stage_e_max_extensions=args.stage_e_max_extensions,
                             stage_e_extension_factor=args.stage_e_extension_factor,
                             stage_e_method=args.stage_e_method,
                             stage_e_mace_fmax=args.stage_e_mace_fmax,
                             stage_e_mace_max_steps=args.stage_e_mace_max_steps,
                             stage_e_device=args.stage_e_device,
                             optimizer_backend=args.optimizer_backend)

    # Auto-finalize: collect everything into results/ with PDBs + summary.
    try:
        finalize = QCB_ROOT / "tools" / "funnel_finalize.py"
        cmd = [sys.executable, str(finalize), "--out", str(args.out)]
        if args.cleanup:
            cmd.append("--cleanup")
        subprocess.run(cmd, check=True)
    except Exception as e:
        log.warning("auto-finalize failed: %s — run manually: "
                    "python tools/funnel_finalize.py --out %s", e, args.out)

    log.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
