"""Cheap sanity guards for transition-state guesses and reaction-path peaks.

Born from a real failure on the OPAA di-Zn theozyme: a Growing-String peak that was
geometrically ~identical to the *product* was reported as a transition state (it wasn't
— it refined straight into the cleaved-product basin). These checks are deliberately
cheap (geometry + energies only, no extra force calls) and emit LOUD warnings so a
suspect TS guess gets scrutiny *before* an expensive saddle refinement.

Two independent checks:

* :func:`ts_endpoint_similarity` — a real TS sits BETWEEN reactant and product. If a
  "TS" is essentially one of the endpoints (small RMSD, or vastly closer to one end than
  the other), it is almost certainly a basin, not a saddle.
* :func:`flag_spurious_path_peak` — a string/NEB peak that is a lone tall "spike" above
  both neighbours, or whose perpendicular force is an outlier, is likely a misplaced
  interpolation node (a clash / a point on a repulsive wall), not the true MEP maximum.

Both return ``list[(severity, message)]`` with severity in ``{"WARNING", "INFO"}``;
:func:`emit` logs them (WARNINGs as a banner) and returns JSON-friendly dicts.
"""
from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import numpy as np

log = logging.getLogger("quantum_engine.analysis.ts_sanity")

Finding = Tuple[str, str]   # (severity, message)


def heavy_atom_rmsd(a, b, indices: Sequence[int] | None = None) -> float:
    """RMSD (Å) between two SAME-ORDERED geometries that share a frame (no superposition).

    ``indices``: restrict to a subset (e.g. the reactive atoms), which is the most
    discriminating comparison for a reaction.
    """
    pa = np.asarray(a.get_positions(), float)
    pb = np.asarray(b.get_positions(), float)
    if indices is not None:
        idx = list(indices)
        pa, pb = pa[idx], pb[idx]
    d = pa - pb
    return float(np.sqrt((d * d).sum() / max(len(pa), 1)))


def ts_endpoint_similarity(ts, reactant, product, *, reactive_idx: Sequence[int] | None = None,
                           abs_tol: float = 0.25, ratio_tol: float = 0.30) -> List[Finding]:
    """Warn when a TS *guess* is geometrically ~indistinguishable from an endpoint.

    Args:
        ts, reactant, product: ASE ``Atoms`` in the same atom order.
        reactive_idx: optional reactive-atom indices — RMSD over just these is the
            sharpest test (whole-cluster RMSD can be dominated by floppy spectators).
        abs_tol: RMSD (Å) below which the TS is "essentially identical" to an endpoint.
        ratio_tol: if the TS is within this fraction as far from one endpoint as the
            other, it's lopsided toward that endpoint (suspect).
    """
    out: List[Finding] = []
    rr = heavy_atom_rmsd(ts, reactant, reactive_idx)
    rp = heavy_atom_rmsd(ts, product, reactive_idx)
    tag = "reactive-atom" if reactive_idx is not None else "all-atom"
    for name, r in (("reactant", rr), ("product", rp)):
        if r < abs_tol:
            out.append(("WARNING",
                f"TS guess is {tag} RMSD {r:.3f} Å from the {name} — essentially identical. "
                f"This is almost certainly the {name.upper()} BASIN, not a transition state. "
                f"Be very skeptical: a real TS lies between reactant and product. Refine with a "
                f"CLIMBING method (dimer/sella-internal) and require exactly one imaginary mode."))
    lo, hi = sorted((rr, rp))
    if hi > 1e-6 and lo / hi < ratio_tol and lo < abs_tol * 3 and not out:
        near = "reactant" if rr < rp else "product"
        out.append(("WARNING",
            f"TS guess is {tag} RMSD {rr:.3f} Å (reactant) / {rp:.3f} Å (product) — only "
            f"{lo / hi:.0%} as far from the {near} as from the other endpoint. A genuine TS sits "
            f"between the two; this strongly resembles the {near.upper()} side. Treat as suspect."))
    out.append(("INFO", f"TS-guess {tag} RMSD: {rr:.3f} Å (reactant) / {rp:.3f} Å (product)."))
    return out


def flag_spurious_path_peak(rel_energies_kcal: Sequence[float], *,
                            perp_forces: Sequence[float] | None = None,
                            spike_kcal: float = 8.0) -> List[Finding]:
    """Flag a string/NEB peak that's likely a misplaced node, not a real MEP maximum.

    Args:
        rel_energies_kcal: per-image energies relative to the lowest image (kcal/mol).
        perp_forces: optional per-image perpendicular force (eV/Å); an outlier at the
            peak means that node hasn't converged onto the MEP, so its energy is unreliable.
        spike_kcal: a peak that towers > this above BOTH neighbours is a lone "spike".
    """
    out: List[Finding] = []
    e = np.asarray(rel_energies_kcal, float)
    n = e.size
    if n < 3:
        return out
    pk = int(np.argmax(e))
    if pk in (0, n - 1):
        out.append(("WARNING",
            f"Reaction-path peak is an ENDPOINT (image {pk}) — no interior barrier was found; "
            f"this is not a transition state. The path may be barrierless, or the range is wrong."))
        return out
    drop_l, drop_r = e[pk] - e[pk - 1], e[pk] - e[pk + 1]
    if drop_l > spike_kcal and drop_r > spike_kcal:
        out.append(("WARNING",
            f"Reaction-path peak (image {pk}, {e[pk]:.1f} kcal/mol) is a lone SPIKE — it towers "
            f"{drop_l:.1f}/{drop_r:.1f} kcal/mol above both neighbours. That is the signature of a "
            f"MISPLACED interpolation node (a clash, or a point on a repulsive wall), not the true "
            f"MEP maximum. Re-run with geodesic interpolation / more images, or prefer CI-NEB."))
    if perp_forces is not None:
        pf = np.asarray(perp_forces, float)
        med = float(np.median(pf)) if pf.size else 0.0
        if pf.size == n and np.isfinite(pf).all() and med > 0 and pf[pk] > 3 * med:
            out.append(("WARNING",
                f"Reaction-path peak (image {pk}) perpendicular force {pf[pk]:.3f} eV/Å is a "
                f"{pf[pk] / med:.0f}× outlier — the peak node is NOT converged onto the MEP, so its "
                f"reported barrier is unreliable."))
    return out


def flag_flat_or_soft_ts(*, imag_freq_cm: float | None = None,
                         reactive_overlap: float | None = None,
                         barrier_kcal: float | None = None,
                         soft_cm: float = 200.0, overlap_min: float = 0.5,
                         shallow_kcal: float = 4.0) -> List[Finding]:
    """Warn when a TS / barrier is FLAT, SHALLOW, or the imaginary mode is not the reaction
    coordinate — the trio that means "do not trust this barrier at the MLFF level".

    These arise when a metalloenzyme active site genuinely flattens the barrier OR
    (indistinguishably from energies alone) when the MLFF under-resolves a sharp saddle.
    Either way the user must know the TS is not sharply defined and should confirm with
    DFT/QM-MM and an uncatalyzed-reaction reference.
    """
    out: List[Finding] = []
    if imag_freq_cm is not None and imag_freq_cm < 0 and abs(imag_freq_cm) < soft_cm:
        out.append(("WARNING",
            f"SOFT imaginary mode ({imag_freq_cm:.0f} cm⁻¹, |nu| < {soft_cm:.0f}) — the barrier is FLAT/shallow and the "
            f"transition state is NOT sharply defined. A genuine sharp SN2 / phosphoryl-transfer TS is usually several "
            f"hundred cm⁻¹. Treat the barrier as qualitative; validate with DFT/QM-MM."))
    if reactive_overlap is not None and reactive_overlap < overlap_min:
        out.append(("WARNING",
            f"The imaginary mode has LOW reactive-atom overlap ({reactive_overlap:.2f} < {overlap_min:.2f}) — it is "
            f"likely a peripheral/spurious motion, NOT the reaction coordinate. This is NOT a validated transition state "
            f"(a small partial Hessian misses delocalized metalloenzyme modes — check mode CHARACTER, not just its sign)."))
    if barrier_kcal is not None and barrier_kcal < shallow_kcal:
        out.append(("WARNING",
            f"SHALLOW barrier ({barrier_kcal:.1f} < {shallow_kcal:.1f} kcal/mol). At the MLFF level this is EITHER a "
            f"genuinely efficient catalyst OR the model under-estimating a sharp saddle — indistinguishable from energies "
            f"alone. Confirm with DFT/QM-MM and an UNCATALYZED-reaction reference (rate enhancement) before trusting it."))
    return out


def emit(findings: List[Finding], logger: logging.Logger | None = None) -> List[dict]:
    """Log findings (WARNINGs as an attention-grabbing banner) → JSON-friendly dicts."""
    logger = logger or log
    msgs = []
    for sev, m in findings:
        if sev == "WARNING":
            logger.warning("\n" + "!" * 78 + f"\n[ts-sanity] {m}\n" + "!" * 78)
        else:
            logger.info("[ts-sanity] %s", m)
        msgs.append({"severity": sev, "message": m})
    return msgs
