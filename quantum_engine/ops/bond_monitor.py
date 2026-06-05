"""Non-constraining bond + metal-coordination *report* for an ASE ``Atoms``.

Pure observation — this NEVER adds a constraint or touches forces. It measures
the distances the caller cares about (typically a ReactionSpec's changing bonds)
and the coordination shell of any metal, classifies each bond as
bonded/stretched/broken via covalent radii, and (optionally) reports the change
vs a reference geometry. Use it as a post-relaxation sanity hook (did the bond we
expected to form actually form? did a metal lose a ligand?) and as machine-
readable provenance.

Reaction-agnostic: you pass the bond pairs and (optionally) the metal indices;
nothing about a specific reaction is baked in. Metals can be auto-detected from
the elements present, but the *bonds to monitor* are always the caller's.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from ase import Atoms
from ase.data import covalent_radii

from quantum_engine.calc.qc_calc import _METALS
from quantum_engine.logging_utils import get_logger

log = get_logger("ops.bond_monitor")


def _cov(atoms: Atoms, i: int) -> float:
    return float(covalent_radii[atoms.numbers[i]])


def _classify(ratio: float, bonded_max: float, broken_min: float) -> str:
    if ratio <= bonded_max:
        return "bonded"
    if ratio >= broken_min:
        return "broken"
    return "stretched"


def monitor_bonds(
    atoms: Atoms,
    *,
    bonds: Iterable[Sequence[int]] | None = None,
    metals: Sequence[int] | str | None = "auto",
    reference: Atoms | dict | None = None,
    bonded_max: float = 1.3,
    broken_min: float = 1.8,
    metal_tol: float = 1.4,
    label: str | None = None,
    outdir: str | Path | None = None,
    write_json: bool = False,
) -> dict:
    """Report monitored bond distances + metal coordination (non-constraining).

    Args:
        atoms: the geometry to inspect (not modified).
        bonds: iterable of ``(i, j)`` 0-based index pairs to measure. ``None`` →
            no explicit bonds (metal report only).
        metals: ``"auto"`` (detect metal elements), an explicit list of metal
            atom indices, or ``None`` (skip the metal report).
        reference: optional ``Atoms`` (same ordering) OR a ``{(i, j): dist_Å}``
            dict; when given, each monitored bond gets a ``delta_A`` (current −
            reference).
        bonded_max: a bond is ``bonded`` if ``dist / (rcov_i + rcov_j) <=`` this.
        broken_min: a bond is ``broken`` if that ratio ``>=`` this (between →
            ``stretched``).
        metal_tol: a ligand atom ``j`` is in metal ``m``'s first shell if
            ``dist(m, j) <= metal_tol * (rcov_m + rcov_j)``.
        label: tag stored in the report (e.g. ``"post_relax_reactant"``).
        outdir / write_json: if both set, also write ``<label|bonds>.json``.

    Returns:
        A machine-readable dict (see module docstring) — ``bonds`` list, ``metals``
        list, plus ``label``/``n_atoms``.
    """
    bonds = [tuple(int(x) for x in b) for b in (bonds or [])]

    # reference distances for deltas
    ref_dist: dict[tuple[int, int], float] = {}
    if isinstance(reference, Atoms):
        for (i, j) in bonds:
            ref_dist[(i, j)] = float(reference.get_distance(i, j))
    elif isinstance(reference, dict):
        for k, v in reference.items():
            ref_dist[tuple(int(x) for x in k)] = float(v)

    bond_reports: list[dict[str, Any]] = []
    for (i, j) in bonds:
        d = float(atoms.get_distance(i, j))
        rcov = _cov(atoms, i) + _cov(atoms, j)
        ratio = d / rcov if rcov > 0 else float("inf")
        rep = {
            "i": i, "j": j,
            "symbols": f"{atoms.symbols[i]}-{atoms.symbols[j]}",
            "distance_A": round(d, 4),
            "rcov_sum_A": round(rcov, 4),
            "ratio": round(ratio, 4),
            "state": _classify(ratio, bonded_max, broken_min),
            "delta_A": (round(d - ref_dist[(i, j)], 4)
                        if (i, j) in ref_dist else None),
        }
        bond_reports.append(rep)

    # ---- metals ----
    metal_reports: list[dict[str, Any]] = []
    if metals is not None:
        if isinstance(metals, str) and metals == "auto":
            metal_idx = [k for k in range(len(atoms))
                         if str(atoms.symbols[k]) in _METALS]
        else:
            metal_idx = [int(m) for m in metals]   # type: ignore[union-attr]

        for m in metal_idx:
            ligands = []
            for k in range(len(atoms)):
                if k == m:
                    continue
                d = float(atoms.get_distance(m, k))
                cutoff = metal_tol * (_cov(atoms, m) + _cov(atoms, k))
                if d <= cutoff:
                    ligands.append({"index": k, "symbol": str(atoms.symbols[k]),
                                    "distance_A": round(d, 4)})
            ligands.sort(key=lambda x: x["distance_A"])
            metal_reports.append({
                "index": m, "symbol": str(atoms.symbols[m]),
                "coordination_number": len(ligands),
                "ligands": ligands,
            })

    report = {
        "label": label,
        "n_atoms": len(atoms),
        "bonds": bond_reports,
        "metals": metal_reports,
    }

    # human log
    if bond_reports:
        for b in bond_reports:
            dtxt = f", Δ={b['delta_A']:+.3f}Å" if b["delta_A"] is not None else ""
            log.info("  bond %s (%d-%d): %.3f Å (ratio %.2f → %s)%s",
                     b["symbols"], b["i"], b["j"], b["distance_A"],
                     b["ratio"], b["state"], dtxt)
    for mr in metal_reports:
        log.info("  metal %s[%d]: CN=%d (%s)",
                 mr["symbol"], mr["index"], mr["coordination_number"],
                 ", ".join(f"{l['symbol']}{l['index']}@{l['distance_A']:.2f}"
                           for l in mr["ligands"]))

    if outdir is not None and write_json:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        fn = outdir / f"{label or 'bond_monitor'}.json"
        fn.write_text(json.dumps(report, indent=2))
        report["_json"] = str(fn)

    return report


def bonds_from_reaction(resolved) -> list[tuple[int, int]]:
    """Convenience: the changing bonds (forming + breaking) of a
    ``ResolvedReaction`` as 0-based ``(i, j)`` pairs, for monitoring."""
    pairs: list[tuple[int, int]] = []
    pairs.extend(tuple(b) for b in getattr(resolved, "forming", []))
    pairs.extend(tuple(b) for b in getattr(resolved, "breaking", []))
    return pairs


__all__ = ["monitor_bonds", "bonds_from_reaction"]
