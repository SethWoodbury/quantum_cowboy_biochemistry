"""reactive_autodetect.py — heuristic identification of the reactive triplet
(electrophile / nucleophile / leaving group) for SN2-like single-bond-forming /
single-bond-breaking reactions.

Used by both `scan_along_s.py` (1D-CV scan) and `polish_ts_v3.py` (TS polish).
Generalizes beyond the original PTE-specific defaults so the pipeline can be
applied to any organic / metalloenzyme reaction with a single recognizable
electrophilic center in a small ligand residue.

For multi-bond reactions (Diels-Alder, Cope, sigmatropic, electrocyclic) this
module is INADEQUATE — see `docs/multi_bond_design.md` for a roadmap toward
generic multi-bond CV scans + NEB-CI fallback.
"""
from __future__ import annotations

import numpy as np


# Common ligand residue names where we expect to find the reactive center.
# Extend this set as new substrates show up in PDB inputs.
LIGAND_RESNAMES = {"SUB", "LIG", "SBT", "UNL", "UNK", "MOL", "DRG", "INH"}

# Atoms that can serve as reactive electrophiles.
# (Phosphorus, carbon, sulfur, silicon, boron, aluminum.)
ELECTROPHILE_ELEMENTS = {"P", "C", "S", "Si", "B", "Al"}

# Atoms that can serve as nucleophile / leaving-group donors.
LIGAND_NEIGHBOR_ELEMENTS = {"O", "N", "S", "Cl", "Br", "I", "F", "P"}


DEFAULT_LG_CUTOFF_A = 2.6


def _autodetect_reactive_triplet(
    src_atoms,
    lg_cutoff_a: float = DEFAULT_LG_CUTOFF_A,
) -> tuple[int, int, int, list[str]]:
    """Heuristically infer (P, Nuc, LG) atom indices from a PDB atom list.

    `src_atoms` must be an iterable of objects with attributes:
      - x, y, z (float Å)
      - name, resname, resseq, chain, element

    Parameters
    ----------
    lg_cutoff_a:
        Distance cutoff (Å) for considering a same-residue O/N/halide neighbor
        of the electrophile as a candidate leaving group. The standard
        covalent-bond cutoff is ~2.2 Å; the default 2.6 Å widens this so
        already-partially-broken bonds in TS-like inputs are still detected.
        Bump higher (e.g. 3.0) for very loose late-TS poses; tighten
        (e.g. 2.2) when the input is a true reactant-state geometry.

    Logic:
      1. Find the electrophilic center: an atom in a ligand-like residue
         (resname ∈ LIGAND_RESNAMES) with element in ELECTROPHILE_ELEMENTS
         and 3-5 covalent O/N/halide neighbors. Single match required.
      2. The leaving group is the longest-bonded same-residue O/N/halide
         substituent within ``lg_cutoff_a`` Å (the bond that would break
         first under attack).
      3. The nucleophile is the nearest atom in a DIFFERENT residue at
         distance ∈ [1.4, 4.0] Å, element in LIGAND_NEIGHBOR_ELEMENTS.

    Returns (P_idx, Nuc_idx, LG_idx, log_lines). Raises ValueError if no
    unique triplet can be inferred. log_lines is a list of human-readable
    strings explaining what was inferred and why.
    """
    coords = np.array([[a.x, a.y, a.z] for a in src_atoms])
    log: list[str] = []

    # Step 1: find candidate electrophilic centers
    candidates: list[tuple[int, int]] = []   # (atom_idx, n_neighbors)
    for i, a in enumerate(src_atoms):
        if a.resname not in LIGAND_RESNAMES:
            continue
        if a.element not in ELECTROPHILE_ELEMENTS:
            continue
        # count covalent O/N/halide neighbors in the same residue
        neighs = []
        for j, b in enumerate(src_atoms):
            if i == j:
                continue
            if b.resseq != a.resseq or b.chain != a.chain:
                continue
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < 2.2 and b.element in LIGAND_NEIGHBOR_ELEMENTS:
                neighs.append((d, j, b))
        if 3 <= len(neighs) <= 5:
            candidates.append((i, len(neighs)))
    if not candidates:
        raise ValueError(
            f"auto-detect failed: no electrophilic center found in any "
            f"ligand-like residue ({sorted(LIGAND_RESNAMES)}). Pass "
            "--p-name/--p-res explicitly."
        )
    if len(candidates) > 1:
        log.append(
            f"WARNING: {len(candidates)} candidate electrophilic centers; "
            f"picking first ({src_atoms[candidates[0][0]].name})"
        )
    P = candidates[0][0]
    P_a = src_atoms[P]
    log.append(f"electrophile=auto P={P_a.name}.{P_a.resname}{P_a.resseq}")

    # Step 2: pick the longest-bonded same-residue substituent as LG.
    # Default cutoff 2.6 Å (not the standard 2.2 Å) to include LG bonds
    # that are already partially broken in TS-like input geometries — this
    # is the whole point of detecting "the bond that breaks first". Caller
    # can override via `lg_cutoff_a`.
    same_res_neighs = []
    for j, b in enumerate(src_atoms):
        if j == P:
            continue
        if b.resseq != P_a.resseq or b.chain != P_a.chain:
            continue
        d = float(np.linalg.norm(coords[P] - coords[j]))
        if d < lg_cutoff_a and b.element in LIGAND_NEIGHBOR_ELEMENTS:
            same_res_neighs.append((d, j, b))
    if not same_res_neighs:
        raise ValueError(
            f"electrophile {P_a.name}.{P_a.resname}{P_a.resseq} has no "
            f"covalent neighbors within {lg_cutoff_a} Å; can't auto-pick LG"
        )
    same_res_neighs.sort(reverse=True)   # longest first
    OL = same_res_neighs[0][1]
    OL_a = src_atoms[OL]
    log.append(
        f"leaving_group=auto LG={OL_a.name}.{OL_a.resname}{OL_a.resseq} "
        f"(longest-bonded at {same_res_neighs[0][0]:.3f}Å, cutoff={lg_cutoff_a:.2f}Å)"
    )

    # Step 3: pick the nearest external nucleophile (O/N/S/halide in a
    # different residue, within 4 Å)
    candidates_nuc = []
    for j, b in enumerate(src_atoms):
        if j == P:
            continue
        if b.resseq == P_a.resseq and b.chain == P_a.chain:
            continue
        if b.element not in LIGAND_NEIGHBOR_ELEMENTS:
            continue
        d = float(np.linalg.norm(coords[P] - coords[j]))
        if 1.4 <= d <= 4.0:
            candidates_nuc.append((d, j, b))
    if not candidates_nuc:
        raise ValueError(
            f"no nucleophile candidate found within 4 Å of electrophile "
            f"{P_a.name}.{P_a.resname}{P_a.resseq}"
        )
    candidates_nuc.sort()
    ON = candidates_nuc[0][1]
    ON_a = src_atoms[ON]
    log.append(
        f"nucleophile=auto Nuc={ON_a.name}.{ON_a.resname}{ON_a.resseq} "
        f"(nearest external at {candidates_nuc[0][0]:.3f}Å)"
    )
    return P, ON, OL, log


def auto_sum_target(src_atoms, P: int, ON: int, OL: int) -> float:
    """Compute d(P-ON) + d(P-OL) from input geometry.

    Used as the default `--sum-target` for the 1D scan along
    s = d(P-OL) − d(P-ON), making the scan center on the user's TS guess.
    """
    coords = np.array([[a.x, a.y, a.z] for a in src_atoms])
    return float(
        np.linalg.norm(coords[P] - coords[ON])
        + np.linalg.norm(coords[P] - coords[OL])
    )
