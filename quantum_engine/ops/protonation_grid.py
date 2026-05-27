"""protonation_grid.py — automated and rules-driven microstate enumeration.

Produces a list of "residue → state" assignments (each one a dict) from
either:

  1. An automatic Cartesian product over all titratable residues found in
     the input structure (capped at ``max_microstates``), or
  2. A user-supplied rules YAML that explicitly lists which residues to
     vary and what states to try.

The product entries are then consumed by ``apply_residue_states`` to emit
N variant PDBs.

CLI surface
-----------
* ``--auto-protonation`` (microstate_sampler) → triggers
  :func:`build_auto_grid` with sensible defaults.
* ``--protonation-rules path.yaml`` → triggers :func:`build_rules_grid`.

Heuristic for the auto grid (when ``max_microstates`` is hit)
-------------------------------------------------------------
The full Cartesian product of N HIS × M (ASP/GLU/LYS/CYS) blows up fast
(3 ** n_HIS × 2 ** n_anion × 2 ** n_LYS × 2 ** n_CYS). When that exceeds
``max_microstates``, we prune by selecting microstates closest to a
"consensus" protonation:

* HIS → HID near a metal (within --metal-cutoff Å of any metal); else HIE
* ASP/GLU → ASP-/GLU- (anionic, default at pH 7) unless the residue's
  carboxylate has a positively-charged neighbor within --pos-cutoff Å, in
  which case both states are kept (it is more likely to be neutral).
* LYS → LYS+ (cationic, default at pH 7)
* CYS → CYS-SH (protonated default)

Then we generate microstates by toggling ONE residue at a time AROUND the
consensus, in order of how close that residue is to the geometric centre
(active-site residues toggled first), keeping the cap.

This gives a reasonable diversity of "consensus + 1 perturbation" without
combinatorial explosion. For full coverage of small systems, raise
``--max-microstates`` (e.g. 64 or 128).

The heuristic is reaction-agnostic — no PTE-specific names — and the
metal/positive cutoffs are CLI-exposed.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from quantum_engine.ops.protonation import (
    DiscoveredResidue,
    HIS_STATES, ASP_STATES, GLU_STATES, LYS_STATES, CYS_STATES,
    discover_titratable_residues,
)

log = logging.getLogger("quantum_engine.ops.protonation_grid")


# ----------------------------------------------------------------------------
# Defaults (CLI-exposed by callers)
# ----------------------------------------------------------------------------
METAL_CUTOFF_A_DEFAULT: float = 3.5     # Å; HIS within this of a metal coordinates it
POS_CHARGE_CUTOFF_A_DEFAULT: float = 4.5
DEFAULT_FAMILIES: tuple[str, ...] = ("HIS", "ASP", "GLU", "LYS", "CYS")
DEFAULT_MAX_MICROSTATES: int = 16

# Set of element symbols treated as "metal" for HIS-near-metal heuristic.
METAL_SYMBOLS: frozenset[str] = frozenset({
    "Zn", "Mg", "Ca", "Mn", "Fe", "Cu", "Ni", "Co", "Mo", "Cd",
})


@dataclass
class MicrostateAssignment:
    """One product entry: a complete (residue → state) assignment."""
    assignments: dict[tuple[str, int], str]   # (chain, res_id) -> state
    description: str
    is_consensus: bool = False
    distance_from_consensus: int = 0          # number of residues differing from consensus

    def label(self, max_len: int = 80) -> str:
        parts = [
            f"{c}:{r}={s}" for (c, r), s in sorted(self.assignments.items())
        ]
        joined = ",".join(parts)
        if len(joined) > max_len:
            joined = joined[: max_len - 3] + "..."
        return joined


# ----------------------------------------------------------------------------
# Consensus heuristic
# ----------------------------------------------------------------------------
def _residue_centroid(atoms, bt_struct, chain: str, res_id: int) -> np.ndarray:
    chains = np.asarray(bt_struct.chain_id)
    rids = np.asarray(bt_struct.res_id)
    mask = (chains == chain) & (rids == int(res_id))
    return atoms.get_positions()[mask].mean(axis=0)


def _residue_atom_index_by_name(
    bt_struct, chain: str, res_id: int, atom_name: str,
) -> int | None:
    """Return the int index of the atom named ``atom_name`` in residue
    (chain, res_id), or ``None`` if not found."""
    chains = np.asarray(bt_struct.chain_id)
    rids = np.asarray(bt_struct.res_id)
    names = np.asarray(bt_struct.atom_name)
    target = atom_name.upper()
    mask = (chains == chain) & (rids == int(res_id))
    idxs = np.where(mask)[0]
    for i in idxs:
        if str(names[i]).strip().upper() == target:
            return int(i)
    return None


def _nearest_distance(atoms, bt_struct, chain: str, res_id: int,
                      target_indices: Sequence[int]) -> float:
    """Min distance from any atom of (chain, res_id) to any atom in target_indices.

    Returns ``np.inf`` when target_indices is empty.
    """
    if not target_indices:
        return float("inf")
    chains = np.asarray(bt_struct.chain_id)
    rids = np.asarray(bt_struct.res_id)
    mask = (chains == chain) & (rids == int(res_id))
    src_pos = atoms.get_positions()[mask]
    tgt_pos = atoms.get_positions()[target_indices]
    if len(src_pos) == 0:
        return float("inf")
    diff = src_pos[:, np.newaxis, :] - tgt_pos[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    return float(d.min())


def consensus_state(
    atoms,
    bt_struct,
    residue: DiscoveredResidue,
    *,
    metal_cutoff_a: float = METAL_CUTOFF_A_DEFAULT,
    pos_charge_cutoff_a: float = POS_CHARGE_CUTOFF_A_DEFAULT,
) -> str:
    """Return the consensus protonation state for the residue, given context."""
    syms = np.asarray(atoms.get_chemical_symbols())
    metal_idxs = [int(i) for i, s in enumerate(syms) if s in METAL_SYMBOLS]
    chains = np.asarray(bt_struct.chain_id)
    rids = np.asarray(bt_struct.res_id)
    rnames = np.asarray(bt_struct.res_name)

    if residue.family == "HIS":
        # If HIS is within metal_cutoff_a of any metal, prefer the tautomer
        # whose protonated N is the OPPOSITE of the metal-binding N (since
        # the metal binds an unprotonated N). Choose by which ring N (ND1
        # vs NE2) is closer to the nearest metal:
        #   - ND1 closer  ⇒ ND1 binds metal (unprotonated) ⇒ HIE (H on NE2)
        #   - NE2 closer  ⇒ NE2 binds metal (unprotonated) ⇒ HID (H on ND1)
        # Fix: codex flag 2026-05-07 (was always HID, wrong half the time).
        if metal_idxs:
            nd1_idx = _residue_atom_index_by_name(
                bt_struct, residue.chain, residue.res_id, "ND1",
            )
            ne2_idx = _residue_atom_index_by_name(
                bt_struct, residue.chain, residue.res_id, "NE2",
            )
            if nd1_idx is not None and ne2_idx is not None:
                pos = atoms.get_positions()
                metal_pos = pos[metal_idxs]
                d_nd1 = float(np.linalg.norm(metal_pos - pos[nd1_idx], axis=1).min())
                d_ne2 = float(np.linalg.norm(metal_pos - pos[ne2_idx], axis=1).min())
                if min(d_nd1, d_ne2) <= metal_cutoff_a:
                    return "HIE" if d_nd1 < d_ne2 else "HID"
        # Fall back to the centroid-based check
        d_metal = _nearest_distance(atoms, bt_struct, residue.chain, residue.res_id,
                                    metal_idxs)
        if d_metal <= metal_cutoff_a:
            # No N atoms in residue (shouldn't happen for HIS but be defensive)
            return "HIE"
        return "HIE"

    if residue.family in ("ASP", "GLU"):
        # Default: anionic at pH 7. If a positively-charged residue (LYS, ARG,
        # HIP) or a metal is within pos_charge_cutoff_a, the carboxylate is
        # likely involved in a salt-bridge; keep ANIONIC default still
        # (consensus does not flip — but the variant explorer will toggle
        # this residue, so we still see GLU0 in the grid).
        return f"{residue.family}-"

    if residue.family == "LYS":
        return "LYS+"

    if residue.family == "CYS":
        # Default: thiol. Thiolate (CYS-S-) only reasonable when bound to a
        # metal — if so, prefer that.
        d_metal = _nearest_distance(atoms, bt_struct, residue.chain, residue.res_id,
                                    metal_idxs)
        if d_metal <= metal_cutoff_a:
            return "CYS-S-"
        return "CYS-SH"

    return residue.valid_states[0]   # safe fallback


# ----------------------------------------------------------------------------
# Auto grid
# ----------------------------------------------------------------------------
def build_auto_grid(
    atoms,
    bt_struct,
    *,
    families: Iterable[str] = DEFAULT_FAMILIES,
    max_microstates: int = DEFAULT_MAX_MICROSTATES,
    metal_cutoff_a: float = METAL_CUTOFF_A_DEFAULT,
    pos_charge_cutoff_a: float = POS_CHARGE_CUTOFF_A_DEFAULT,
    include_consensus: bool = True,
) -> tuple[list[MicrostateAssignment], list[DiscoveredResidue]]:
    """Generate the auto Cartesian product (or pruned subset).

    Args:
        atoms, bt_struct: input structure (need atom positions + biotite
            annotations).
        families: which residue families to include in the product.
        max_microstates: defensive cap; if the full product exceeds this,
            we use the consensus + 1-residue-perturbation strategy described
            in the module docstring.
        metal_cutoff_a: HIS within this of a metal → consensus HID.
        pos_charge_cutoff_a: (currently unused for ASP/GLU; reserved for a
            future "near positive charge" → consensus neutral-COOH rule).
        include_consensus: include the all-consensus state as the first entry.

    Returns:
        ``(assignments, residues)`` — the residues list is included so callers
        can log / report which residues were considered.
    """
    residues = discover_titratable_residues(bt_struct, families=families)
    if not residues:
        log.warning(
            "build_auto_grid: no titratable residues found in input "
            "(families=%s)", list(families),
        )
        return [], residues

    consensus = {
        (r.chain, r.res_id): consensus_state(
            atoms, bt_struct, r,
            metal_cutoff_a=metal_cutoff_a,
            pos_charge_cutoff_a=pos_charge_cutoff_a,
        )
        for r in residues
    }

    # Compute the full product size (defensive)
    product_size = 1
    for r in residues:
        product_size *= len(r.valid_states)

    log.info(
        "build_auto_grid: %d titratable residues, "
        "full product size = %d, max_microstates = %d",
        len(residues), product_size, max_microstates,
    )

    if product_size <= max_microstates:
        return _full_product(residues, consensus, include_consensus), residues

    # Prune via consensus + 1-perturbation strategy
    return _consensus_plus_perturbations(
        atoms, bt_struct, residues, consensus,
        max_microstates=max_microstates,
        include_consensus=include_consensus,
    ), residues


def _full_product(
    residues: list[DiscoveredResidue],
    consensus: dict[tuple[str, int], str],
    include_consensus: bool,
) -> list[MicrostateAssignment]:
    """Emit the full Cartesian product of residue states (small systems)."""
    product_iter = itertools.product(*[r.valid_states for r in residues])
    out: list[MicrostateAssignment] = []
    for combo in product_iter:
        assigns = {(r.chain, r.res_id): s for r, s in zip(residues, combo)}
        diff = sum(
            1 for k, v in assigns.items() if consensus.get(k) != v
        )
        is_cons = (diff == 0)
        out.append(MicrostateAssignment(
            assignments=assigns,
            description="auto_full_product:" + _describe(residues, combo),
            is_consensus=is_cons,
            distance_from_consensus=diff,
        ))
    if include_consensus:
        # Move consensus to front (already there if found)
        out.sort(key=lambda m: (not m.is_consensus, m.distance_from_consensus))
    else:
        # Drop the consensus entries entirely. (codex flag 2026-05-07: prior
        # behavior was to leave them in but un-sorted, which still emitted them.)
        out = [m for m in out if not m.is_consensus]
    return out


def _describe(residues: list[DiscoveredResidue], combo: Sequence[str]) -> str:
    parts = [f"{r.chain}:{r.res_id}{r.res_name}={s}" for r, s in zip(residues, combo)]
    return ",".join(parts)


def _consensus_plus_perturbations(
    atoms,
    bt_struct,
    residues: list[DiscoveredResidue],
    consensus: dict[tuple[str, int], str],
    *,
    max_microstates: int,
    include_consensus: bool,
) -> list[MicrostateAssignment]:
    """Build a pruned grid: consensus + single-residue perturbations.

    Order: consensus first (1 entry), then single-toggle perturbations
    sorted by residue's geometric centrality (residues closest to centroid
    get toggled first — those are likely active-site).
    """
    out: list[MicrostateAssignment] = []
    centroid = atoms.get_positions().mean(axis=0)

    if include_consensus:
        out.append(MicrostateAssignment(
            assignments=dict(consensus),
            description="auto_consensus:" + _describe(residues, [
                consensus[(r.chain, r.res_id)] for r in residues
            ]),
            is_consensus=True,
            distance_from_consensus=0,
        ))

    # Compute centrality (smaller distance = more central)
    centrality = []
    for r in residues:
        rc = _residue_centroid(atoms, bt_struct, r.chain, r.res_id)
        centrality.append((r, float(np.linalg.norm(rc - centroid))))
    centrality.sort(key=lambda t: t[1])

    # 1-residue toggles
    for r, _ in centrality:
        cons_state = consensus[(r.chain, r.res_id)]
        for alt in r.valid_states:
            if alt == cons_state:
                continue
            new_assigns = dict(consensus)
            new_assigns[(r.chain, r.res_id)] = alt
            assignment = MicrostateAssignment(
                assignments=new_assigns,
                description=f"auto_perturb:{r.chain}:{r.res_id}{r.res_name}={alt}",
                is_consensus=False,
                distance_from_consensus=1,
            )
            out.append(assignment)
            if len(out) >= max_microstates:
                return out

    # 2-residue toggles (only if we have room)
    for (r1, _), (r2, _) in itertools.combinations(centrality, 2):
        if len(out) >= max_microstates:
            return out
        cons1 = consensus[(r1.chain, r1.res_id)]
        cons2 = consensus[(r2.chain, r2.res_id)]
        for alt1 in r1.valid_states:
            if alt1 == cons1:
                continue
            for alt2 in r2.valid_states:
                if alt2 == cons2:
                    continue
                new_assigns = dict(consensus)
                new_assigns[(r1.chain, r1.res_id)] = alt1
                new_assigns[(r2.chain, r2.res_id)] = alt2
                out.append(MicrostateAssignment(
                    assignments=new_assigns,
                    description=(
                        f"auto_perturb2:"
                        f"{r1.chain}:{r1.res_id}{r1.res_name}={alt1};"
                        f"{r2.chain}:{r2.res_id}{r2.res_name}={alt2}"
                    ),
                    is_consensus=False,
                    distance_from_consensus=2,
                ))
                if len(out) >= max_microstates:
                    return out

    return out


# ----------------------------------------------------------------------------
# Rules YAML
# ----------------------------------------------------------------------------
def load_rules_yaml(path: str | Path) -> list[dict[str, Any]]:
    """Load a rules YAML from disk.

    Expected schema::

        residues:
          - residue: A:201:HIS
            states: [HID, HIE, HIP]
          - residue: A:169:GLU
            states: [GLU-, GLU0]

    Where the ``residue`` field is ``CHAIN:RES_ID:RESNAME`` (RESNAME is
    optional but recommended for documentation; if present it must match
    the input PDB or a warning is logged).

    Returns the parsed list of residue rules.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"protonation rules YAML not found: {p}")
    text = p.read_text()
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML required for --protonation-rules YAML") from exc
    cfg = yaml.safe_load(text) or {}
    rules = cfg.get("residues") or []
    if not isinstance(rules, list):
        raise ValueError(f"rules YAML {p}: 'residues' must be a list")
    return rules


def build_rules_grid(
    atoms,
    bt_struct,
    rules: list[dict[str, Any]],
    *,
    max_microstates: int = DEFAULT_MAX_MICROSTATES,
) -> list[MicrostateAssignment]:
    """Build the Cartesian product from a parsed rules YAML."""
    if not rules:
        return []
    parsed: list[tuple[tuple[str, int], list[str]]] = []
    seen_keys: set[tuple[str, int]] = set()
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError(f"each rules entry must be a dict, got {rule!r}")
        spec = rule.get("residue")
        states = rule.get("states") or []
        if not spec:
            raise ValueError(f"rule missing 'residue' key: {rule!r}")
        if not states:
            raise ValueError(f"rule {spec!r} missing 'states'")
        chain, rid, *rest = spec.split(":")
        rid_i = int(rid)
        key = (chain, rid_i)
        # codex flag 2026-05-07: duplicate residue entries silently collapsed
        # in the assigns dict, making product size and applied charge deltas
        # inconsistent with the manifest. Reject up-front.
        if key in seen_keys:
            raise ValueError(
                f"build_rules_grid: duplicate residue {chain}:{rid_i} in rules; "
                f"each (chain, res_id) may appear at most once."
            )
        seen_keys.add(key)
        parsed.append((key, [str(s) for s in states]))
    # Sanity-check residues exist in the input
    chains = np.asarray(bt_struct.chain_id)
    rids = np.asarray(bt_struct.res_id)
    for (chain, rid), _states in parsed:
        mask = (chains == chain) & (rids == rid)
        if not mask.any():
            log.warning(
                "build_rules_grid: residue %s:%d not found in input PDB",
                chain, rid,
            )
    # Cartesian product
    out: list[MicrostateAssignment] = []
    keys = [k for k, _ in parsed]
    state_lists = [s for _, s in parsed]
    for combo in itertools.product(*state_lists):
        assigns = {k: s for k, s in zip(keys, combo)}
        out.append(MicrostateAssignment(
            assignments=assigns,
            description="rules:" + ",".join(
                f"{k[0]}:{k[1]}={s}" for k, s in zip(keys, combo)
            ),
            is_consensus=False,
        ))
        if len(out) >= max_microstates:
            log.warning(
                "build_rules_grid: capped at %d microstates "
                "(YAML product exceeded cap)", max_microstates,
            )
            break
    return out
