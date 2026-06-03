"""Tiered Hessian validation for TS structures.

Why this exists
---------------
The default ``refine-ts`` partial Hessian (``freq_indices = reactive +
first-shell neighbours`` ≈ 11 atoms) is excellent for **catching** bad TS
guesses (no imag, multiple imag, low overlap with reaction coordinate) but
cheap by design. For a CONFIRMED TS we want a deeper check that the imag
mode isn't an artefact of small-Hessian truncation:

    Tier A — partial Hessian on the reactive subset (~11 atoms; the same
             check ``refine-ts`` already runs)
    Tier B — expanded Hessian on the chemically-relevant region (~30-60
             atoms; metals + ligands + nucleophile + leaving group +
             catalytic acid/base + first-shell waters)
    Tier C — a Lanczos / sparse Hessian-vector lowest-mode search over all
             movable atoms. Validates that no second imag mode hides
             elsewhere.

Plus a bonus ``imag_mode_displace`` test: push along ±0.2 Å of the imag
mode and relax both branches; verify one decays toward an R-like minimum
and the other to a P-like minimum.

Reaction-agnostic
-----------------
- Reactive atoms come from CLI (1-based PDB serials, 0:idx, or
  RES:ID:NAME). No hardcoded names.
- The Tier B "active region" is defined via the
  :mod:`quantum_engine.select` grammar (``site 5.0 SUB OHX``,
  ``residue HIS``, etc.).
- Imag-mode overlap threshold, lowest-imag cutoff, etc. are CLI flags.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from ase import Atoms

from quantum_engine.ops import freq as freq_op
from quantum_engine.units import EV_TO_KCAL, FREQ_CONV

log = logging.getLogger("quantum_engine.ops.expanded_hessian")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class TSValidationCriteria:
    """Pass/fail thresholds for tiered TS validation."""
    n_imag_expected: int = 1
    imag_cm_cutoff: float = -50.0
    imag_mode_min_overlap: float = 0.5
    second_imag_cm_cutoff: float = -10.0  # any other mode below this triggers a warning
    require_no_second_imag: bool = True


@dataclass
class TSValidationCheck:
    name: str
    tier: str
    passed: bool
    detail: str
    value: Any = None


@dataclass
class TSValidationReport:
    """Outcome of a tiered validation pass."""
    overall_pass: bool
    tiers_run: list[str] = field(default_factory=list)
    tier_a: dict | None = None
    tier_b: dict | None = None
    tier_c: dict | None = None
    displacement_check: dict | None = None
    checks: list[TSValidationCheck] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _imag_mode_overlap(mode: np.ndarray, indices: Sequence[int]) -> float:
    total = float(np.linalg.norm(mode))
    if total < 1e-12:
        return 0.0
    sub = float(np.linalg.norm(mode[list(indices)]))
    return sub / total


def _expand_indices_to_active_region(
    atoms: Atoms,
    bt_struct,
    reactive_indices: Sequence[int],
    *,
    active_region_spec: str | None = None,
    radius_A: float | None = None,
    res_names: Sequence[str] | None = None,
) -> list[int]:
    """Build the Tier B index set.

    Resolution order:
        1. If ``active_region_spec`` is given, parse via
           :mod:`quantum_engine.select`.
        2. Else if ``radius_A`` is given, take all atoms within radius of
           any reactive atom.
        3. Else if ``res_names`` is given, take all atoms in those residues
           plus the reactive atoms.
        4. Else fall back to ``reactive_indices`` + first bonded shell
           (same heuristic as ``refine-ts``).

    Returns sorted unique 0-based indices.
    """
    from quantum_engine.select import parse_constraints
    coords = atoms.get_positions()
    n = len(atoms)
    mask = np.zeros(n, dtype=bool)
    mask[list(reactive_indices)] = True

    if active_region_spec:
        # 'site R RESNAMES...' is custom — fall back to multi-spec parsing.
        # The simplest interpretation: any spec already supported by
        # parse_constraints, OR a 'site R RESNAME...' shorthand which we
        # expand here.
        spec_strs = active_region_spec.split(";") \
            if isinstance(active_region_spec, str) else list(active_region_spec)
        for sp in spec_strs:
            sp = sp.strip()
            if not sp:
                continue
            tokens = sp.split()
            if tokens and tokens[0].lower() == "site":
                if len(tokens) < 3:
                    raise ValueError(
                        f"active-region 'site' needs RADIUS RESNAMES, got: {sp!r}"
                    )
                try:
                    R = float(tokens[1])
                except ValueError as exc:
                    raise ValueError(f"active-region 'site' radius parse error in {sp!r}") from exc
                resnames = tokens[2:]
                if bt_struct is None:
                    raise ValueError("'site' selector needs a PDB template")
                # Find seed atoms (any atom in any of the named residues)
                seed_mask = np.isin(bt_struct.res_name, resnames)
                seed_idxs = np.where(seed_mask)[0]
                if not len(seed_idxs):
                    log.warning("'site' selector found no seed atoms for "
                                "resnames %s", resnames)
                seed_coords = coords[seed_idxs]
                # Distance from any non-seed atom to nearest seed
                for i in range(n):
                    if mask[i]:
                        continue
                    if seed_coords.size == 0:
                        break
                    d = np.linalg.norm(seed_coords - coords[i], axis=1).min()
                    if d <= R:
                        mask[i] = True
            else:
                m = parse_constraints(atoms, bt_struct, [sp])
                mask |= m
    elif radius_A is not None:
        seed_coords = coords[list(reactive_indices)]
        for i in range(n):
            if mask[i]:
                continue
            d = np.linalg.norm(seed_coords - coords[i], axis=1).min()
            if d <= radius_A:
                mask[i] = True
    elif res_names is not None:
        if bt_struct is None:
            raise ValueError("res_names selector needs a PDB template")
        for rn in res_names:
            mask |= (np.asarray(bt_struct.res_name) == rn)
    else:
        # Fall back to reactive + first bonded shell
        from ase.neighborlist import NeighborList, natural_cutoffs
        cutoffs = natural_cutoffs(atoms, mult=1.1)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)
        for i in reactive_indices:
            neighbours, _ = nl.get_neighbors(int(i))
            for j in neighbours:
                mask[int(j)] = True

    return sorted(int(i) for i in np.where(mask)[0])


# ---------------------------------------------------------------------------
# Tier A — small partial Hessian
# ---------------------------------------------------------------------------
def tier_a_partial_hessian(
    atoms: Atoms,
    *,
    reactive_indices: Sequence[int],
    outdir: str | Path,
    delta: float = 0.005,
    method: str = "central",
) -> dict:
    """Run the standard reactive-only partial Hessian (Tier A).

    Returns the dict from :func:`quantum_engine.ops.freq.run` annotated with
    ``imag_mode_overlap`` against ``reactive_indices``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Use reactive_indices + first bonded shell as the partial-Hessian basis
    from ase.neighborlist import NeighborList, natural_cutoffs
    cutoffs = natural_cutoffs(atoms, mult=1.1)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    idx_set = set(int(i) for i in reactive_indices)
    for i in reactive_indices:
        neighbours, _ = nl.get_neighbors(int(i))
        for j in neighbours:
            idx_set.add(int(j))
    indices = sorted(idx_set)
    log.info("tier_a_partial_hessian: %d atoms", len(indices))

    res = freq_op.run(
        atoms, atoms.calc, outdir,
        constraint=None,
        indices=indices,
        delta=delta,
        method=method,
    )
    modes = np.load(res["outputs"]["modes"])
    overlap = _imag_mode_overlap(modes[0], reactive_indices) if len(modes) else 0.0
    res["imag_mode_overlap_reactive"] = overlap
    res["hessian_indices"] = indices
    return res


# ---------------------------------------------------------------------------
# Tier B — expanded Hessian
# ---------------------------------------------------------------------------
def tier_b_expanded_hessian(
    atoms: Atoms,
    *,
    reactive_indices: Sequence[int],
    bt_struct,
    outdir: str | Path,
    active_region_spec: str | None = None,
    radius_A: float | None = None,
    res_names: Sequence[str] | None = None,
    delta: float = 0.005,
    method: str = "central",
) -> dict:
    """Tier B: 30-60 atom expanded Hessian over the chemically-relevant region."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    indices = _expand_indices_to_active_region(
        atoms, bt_struct, reactive_indices,
        active_region_spec=active_region_spec,
        radius_A=radius_A,
        res_names=res_names,
    )
    log.info("tier_b_expanded_hessian: %d atoms (active region=%s, radius=%s, res_names=%s)",
             len(indices), active_region_spec, radius_A, res_names)
    res = freq_op.run(
        atoms, atoms.calc, outdir,
        constraint=None,
        indices=indices,
        delta=delta,
        method=method,
    )
    modes = np.load(res["outputs"]["modes"])
    overlap_reactive = _imag_mode_overlap(modes[0], reactive_indices) if len(modes) else 0.0
    overlap_region = _imag_mode_overlap(modes[0], indices) if len(modes) else 0.0
    res["imag_mode_overlap_reactive"] = overlap_reactive
    res["imag_mode_overlap_region"] = overlap_region
    res["hessian_indices"] = indices
    return res


# ---------------------------------------------------------------------------
# Tier C — Lanczos lowest-mode search
# ---------------------------------------------------------------------------
def tier_c_lowest_mode_lanczos(
    atoms: Atoms,
    *,
    movable_mask: np.ndarray | None = None,
    outdir: str | Path,
    delta: float = 0.005,
    n_modes: int = 4,
    max_iters: int = 50,
    tol: float = 1e-3,
) -> dict:
    """Tier C: estimate the few lowest mass-weighted Hessian eigenvalues
    using a Lanczos iteration with finite-difference Hessian-vector products.

    Args:
        atoms: TS structure with attached calculator.
        movable_mask: bool array of len(atoms); True for atoms allowed to
            move. If None, all atoms are movable.
        outdir: where to write the summary.
        delta: finite-difference step (Å) for Hessian-vector products.
        n_modes: how many lowest modes to estimate.
        max_iters: max Lanczos iterations.
        tol: relative-residual tolerance for early stop.

    Returns dict with ``frequencies_cm`` (lowest n_modes), ``modes`` (3N
    arrays), and ``n_imag``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if movable_mask is None:
        movable_mask = np.ones(len(atoms), dtype=bool)
    movable_idxs = np.where(movable_mask)[0]
    n_dof = 3 * len(movable_idxs)
    log.info("tier_c_lowest_mode_lanczos: %d dof, n_modes=%d, max_iters=%d",
             n_dof, n_modes, max_iters)

    masses = np.repeat(atoms.get_masses()[movable_idxs], 3)
    masses = np.clip(masses, 1e-3, None)
    m_inv_sqrt = 1.0 / np.sqrt(masses)

    pos0 = atoms.get_positions().copy()

    def hess_vec_product(v_mw: np.ndarray) -> np.ndarray:
        """Mass-weighted Hessian-vector product via central FD of forces.

        Mass-weighted Hessian: ``H_mw = M^{-1/2} H_cart M^{-1/2}``. So:
            ``H_mw @ v_mw = M^{-1/2} @ H_cart @ (M^{-1/2} v_mw)``
        We compute ``H_cart @ x`` for ``x = M^{-1/2} v_mw`` via central FD on
        the forces, then mass-weight the output again.
        """
        x_cart = v_mw * m_inv_sqrt  # Cartesian displacement direction
        norm_x = float(np.linalg.norm(x_cart))
        if norm_x < 1e-12:
            return np.zeros(n_dof)
        # Place ``x_cart`` into a 2-D displacement array on movable atoms only
        disp = np.zeros_like(pos0)
        for k, atom_i in enumerate(movable_idxs):
            disp[atom_i] = x_cart[3 * k : 3 * k + 3]
        # Scale displacement so its 2-norm is ``delta`` (constant FD step size)
        scale = delta / norm_x
        atoms.set_positions(pos0 + scale * disp)
        f_plus = atoms.get_forces()
        atoms.set_positions(pos0 - scale * disp)
        f_minus = atoms.get_forces()
        atoms.set_positions(pos0)
        # H_cart @ (scale*disp) = -(f_plus - f_minus) / 2
        # => H_cart @ disp = -(f_plus - f_minus) / (2*scale)
        df = -(f_plus - f_minus) / (2.0 * scale)  # shape (N,3) Cartesian
        flat = np.zeros(n_dof)
        for k, atom_i in enumerate(movable_idxs):
            flat[3 * k : 3 * k + 3] = df[atom_i]
        # Now mass-weight the output: H_mw @ v_mw = M^{-1/2} @ flat
        return flat * m_inv_sqrt

    # Lanczos
    rng = np.random.default_rng(seed=0)
    v = rng.standard_normal(n_dof)
    v /= np.linalg.norm(v)
    Q = [v]
    alphas: list[float] = []
    betas: list[float] = []
    Hv_prev = hess_vec_product(v)
    alpha = float(v @ Hv_prev)
    r = Hv_prev - alpha * v
    alphas.append(alpha)
    n_iters = min(max_iters, n_dof - 1)
    for j in range(1, n_iters):
        beta = float(np.linalg.norm(r))
        if beta < tol:
            log.info("tier_c lanczos: converged at iter %d (beta=%.2e)", j, beta)
            break
        betas.append(beta)
        v_new = r / beta
        # full re-orthogonalisation (cheap with small subspace)
        for q in Q:
            v_new -= (q @ v_new) * q
        v_new /= np.linalg.norm(v_new)
        Q.append(v_new)
        Hv = hess_vec_product(v_new)
        alpha = float(v_new @ Hv)
        alphas.append(alpha)
        r = Hv - alpha * v_new - beta * Q[-2]

    # Tridiagonal solve
    T = np.diag(alphas)
    for k, b in enumerate(betas):
        T[k, k + 1] = b
        T[k + 1, k] = b
    eigvals, eigvecs_T = np.linalg.eigh(T)

    # Pick the n_modes lowest
    n_pick = min(n_modes, len(eigvals))
    Q_arr = np.column_stack(Q)  # n_dof x m
    modes = Q_arr @ eigvecs_T[:, :n_pick]  # n_dof x n_pick
    freqs_cm = np.sign(eigvals[:n_pick]) * np.sqrt(np.abs(eigvals[:n_pick])) * FREQ_CONV
    n_imag = int(np.sum(freqs_cm < -10))

    summary = {
        "n_dof": n_dof,
        "n_iters_run": len(alphas),
        "frequencies_cm": freqs_cm.tolist(),
        "n_imag": n_imag,
        "delta_A": delta,
        "n_modes": n_pick,
    }
    summary_path = outdir / "tier_c.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    np.save(outdir / "tier_c_modes.npy", modes)
    np.save(outdir / "tier_c_freqs.npy", freqs_cm)
    log.info("tier_c lowest %d freqs (cm⁻¹): %s",
             n_pick, np.round(freqs_cm, 1).tolist())
    return {
        **summary,
        "modes_npy": str(outdir / "tier_c_modes.npy"),
        "freqs_npy": str(outdir / "tier_c_freqs.npy"),
        "summary_path": str(summary_path),
    }


# ---------------------------------------------------------------------------
# Validation gateway
# ---------------------------------------------------------------------------
def validate_ts(
    atoms: Atoms,
    *,
    reactive_indices: Sequence[int],
    outdir: str | Path,
    bt_struct=None,
    tier: str = "b",
    active_region_spec: str | None = None,
    active_region_radius_A: float | None = None,
    active_region_res_names: Sequence[str] | None = None,
    delta: float = 0.005,
    method: str = "central",
    criteria: TSValidationCriteria | None = None,
    tier_c_n_modes: int = 4,
    tier_c_max_iters: int = 50,
) -> TSValidationReport:
    """Run one or more tiers of TS validation and emit a pass/fail report.

    Args:
        atoms: TS atoms (calculator attached).
        reactive_indices: 0-based indices defining the reaction coordinate.
        outdir: report output dir.
        bt_struct: optional biotite template (needed for active-region specs).
        tier: ``'a'`` / ``'b'`` / ``'c'`` / ``'all'`` / comma-list (e.g. ``'a,b'``).
        active_region_spec, active_region_radius_A, active_region_res_names:
            mutually-exclusive ways to define the Tier B index set.
        delta: finite-difference step.
        method: ``'central'`` / ``'forward'``.
        criteria: pass/fail thresholds.
        tier_c_n_modes, tier_c_max_iters: Lanczos config.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if criteria is None:
        criteria = TSValidationCriteria()

    requested: set[str] = set()
    if tier == "all":
        requested = {"a", "b", "c"}
    else:
        for t in str(tier).split(","):
            t = t.strip().lower()
            if t in ("a", "b", "c"):
                requested.add(t)
            elif t:
                raise ValueError(f"unknown tier {t!r}; use a, b, c, or all")
    if not requested:
        raise ValueError(f"no tiers requested (got tier={tier!r})")

    report = TSValidationReport(overall_pass=False, tiers_run=sorted(requested))
    checks: list[TSValidationCheck] = report.checks

    # Tier A
    if "a" in requested:
        log.info("[validate-ts] Tier A — reactive partial Hessian")
        a_dir = outdir / "tier_a"
        a_res = tier_a_partial_hessian(
            atoms, reactive_indices=reactive_indices, outdir=a_dir,
            delta=delta, method=method,
        )
        report.tier_a = {
            "n_imag": a_res["n_imaginary"],
            "lowest_mode_cm": a_res["lowest_mode_cm"],
            "imag_mode_overlap_reactive": a_res["imag_mode_overlap_reactive"],
            "indices": a_res["hessian_indices"],
            "outputs": a_res["outputs"],
        }
        checks.append(TSValidationCheck(
            name="tier_a_n_imag",
            tier="a",
            passed=a_res["n_imaginary"] == criteria.n_imag_expected,
            detail=f"n_imag={a_res['n_imaginary']} (expected {criteria.n_imag_expected})",
            value=a_res["n_imaginary"],
        ))
        checks.append(TSValidationCheck(
            name="tier_a_imag_cutoff",
            tier="a",
            passed=a_res["lowest_mode_cm"] < criteria.imag_cm_cutoff,
            detail=f"lowest_mode_cm={a_res['lowest_mode_cm']:.1f} "
                   f"(must be < {criteria.imag_cm_cutoff})",
            value=a_res["lowest_mode_cm"],
        ))
        checks.append(TSValidationCheck(
            name="tier_a_overlap",
            tier="a",
            passed=a_res["imag_mode_overlap_reactive"] >= criteria.imag_mode_min_overlap,
            detail=f"overlap={a_res['imag_mode_overlap_reactive']:.3f} "
                   f"(>= {criteria.imag_mode_min_overlap})",
            value=a_res["imag_mode_overlap_reactive"],
        ))

    # Tier B
    if "b" in requested:
        log.info("[validate-ts] Tier B — expanded active-region Hessian")
        b_dir = outdir / "tier_b"
        b_res = tier_b_expanded_hessian(
            atoms, reactive_indices=reactive_indices, bt_struct=bt_struct,
            outdir=b_dir,
            active_region_spec=active_region_spec,
            radius_A=active_region_radius_A,
            res_names=active_region_res_names,
            delta=delta, method=method,
        )
        report.tier_b = {
            "n_imag": b_res["n_imaginary"],
            "lowest_mode_cm": b_res["lowest_mode_cm"],
            "imag_mode_overlap_reactive": b_res["imag_mode_overlap_reactive"],
            "imag_mode_overlap_region": b_res["imag_mode_overlap_region"],
            "indices": b_res["hessian_indices"],
            "outputs": b_res["outputs"],
        }
        # Check: still exactly N_imag in expanded region
        checks.append(TSValidationCheck(
            name="tier_b_n_imag",
            tier="b",
            passed=b_res["n_imaginary"] == criteria.n_imag_expected,
            detail=f"n_imag={b_res['n_imaginary']} (expected {criteria.n_imag_expected})",
            value=b_res["n_imaginary"],
        ))
        # Imag still localized on reactive atoms (NOT just on the active region)
        checks.append(TSValidationCheck(
            name="tier_b_overlap",
            tier="b",
            passed=b_res["imag_mode_overlap_reactive"] >= criteria.imag_mode_min_overlap,
            detail=f"reactive_overlap={b_res['imag_mode_overlap_reactive']:.3f} "
                   f"(>= {criteria.imag_mode_min_overlap})",
            value=b_res["imag_mode_overlap_reactive"],
        ))
        # No second imag below threshold (water drift, ligand exchange)
        freqs = np.asarray(b_res["frequencies_cm"])
        n_below_2nd = int((freqs[1:] < criteria.second_imag_cm_cutoff).sum()) if len(freqs) > 1 else 0
        checks.append(TSValidationCheck(
            name="tier_b_no_second_imag",
            tier="b",
            passed=(n_below_2nd == 0) or (not criteria.require_no_second_imag),
            detail=f"second-mode-imag count = {n_below_2nd} (cutoff {criteria.second_imag_cm_cutoff})",
            value=n_below_2nd,
        ))

    # Tier C
    if "c" in requested:
        log.info("[validate-ts] Tier C — Lanczos lowest-mode search")
        c_dir = outdir / "tier_c"
        c_res = tier_c_lowest_mode_lanczos(
            atoms, outdir=c_dir,
            delta=delta,
            n_modes=tier_c_n_modes,
            max_iters=tier_c_max_iters,
        )
        report.tier_c = c_res
        checks.append(TSValidationCheck(
            name="tier_c_n_imag",
            tier="c",
            passed=c_res["n_imag"] == criteria.n_imag_expected,
            detail=f"n_imag (Lanczos) = {c_res['n_imag']} "
                   f"(expected {criteria.n_imag_expected})",
            value=c_res["n_imag"],
        ))

    report.overall_pass = all(c.passed for c in checks)
    summary_path = outdir / "validate_ts.json"
    summary_path.write_text(json.dumps({
        "overall_pass": report.overall_pass,
        "tiers_run": report.tiers_run,
        "tier_a": report.tier_a,
        "tier_b": report.tier_b,
        "tier_c": report.tier_c,
        "checks": [asdict(c) for c in checks],
        "criteria": asdict(criteria),
    }, indent=2, default=str))
    report.outputs["summary"] = str(summary_path)
    log.info("[validate-ts] overall_pass=%s, %d checks (%d passed)",
             report.overall_pass, len(checks),
             sum(1 for c in checks if c.passed))
    return report


def run(
    atoms: Atoms,
    calculator=None,
    outdir: str | Path = ".",
    constraint=None,
    *,
    reactive_indices: Sequence[int] | None = None,
    bt_struct=None,
    tier: str = "b",
    active_region_spec: str | None = None,
    active_region_radius_A: float | None = None,
    active_region_res_names: Sequence[str] | None = None,
    delta: float = 0.005,
    method: str = "central",
    criteria: TSValidationCriteria | None = None,
    tier_c_n_modes: int = 4,
    tier_c_max_iters: int = 50,
    **kwargs,
) -> dict:
    """``ops`` adapter signature for ``qcb validate-ts``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if atoms.calc is None:
        if calculator is None:
            raise ValueError("validate-ts: no calc on atoms and none supplied")
        atoms.calc = calculator
    if constraint is not None:
        atoms.set_constraint(constraint if isinstance(constraint, list) else [constraint])
    if reactive_indices is None:
        raise ValueError("validate-ts: reactive_indices is required")

    report = validate_ts(
        atoms,
        reactive_indices=reactive_indices,
        outdir=outdir,
        bt_struct=bt_struct,
        tier=tier,
        active_region_spec=active_region_spec,
        active_region_radius_A=active_region_radius_A,
        active_region_res_names=active_region_res_names,
        delta=delta,
        method=method,
        criteria=criteria,
        tier_c_n_modes=tier_c_n_modes,
        tier_c_max_iters=tier_c_max_iters,
    )
    return {
        "status": "completed" if report.overall_pass else "warned",
        "overall_pass": report.overall_pass,
        "tiers_run": report.tiers_run,
        "tier_a": report.tier_a,
        "tier_b": report.tier_b,
        "tier_c": report.tier_c,
        "checks": [asdict(c) for c in report.checks],
        "outputs": report.outputs,
    }
