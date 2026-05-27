"""Post-NEB / post-FSM TS validation pipeline.

Why this exists
---------------
NEB-CI and FSM produce a *guess* TS — a structure with the right basin
behaviour and a single dominant maximum on the path, but typically with
~0.05 eV/Å residual gradient and no eigenvalue check. A real TS needs:

1. Force-converged saddle point from a TS-aware optimiser
   (Sella, Dimer, RS-P-RFO).
2. Exactly one imaginary frequency.
3. The imaginary mode aligns with the user's reaction coordinate (i.e.,
   the supplied reactive atom triplet / set actually moves on that mode).
4. The mode is below an empirical noise threshold (e.g., < -50 cm⁻¹).

This op chains:

    extract NEB-CI TS  →  saddle.run()  →  freq.run() (partial Hessian)
                       →  diagnostics  →  summary.json

Generalisation requirements
---------------------------
- *No PTE-specific defaults.* Reactive atoms are always supplied by the
  caller (1-based PDB serial numbers or 0-based indices).
- Works for any reaction class — the reaction-coordinate check uses the
  user-supplied ``reactive_atoms`` and (optionally) reactant/product
  bond-pattern hints.
- The ``imag_cm_cutoff`` and ``imag_mode_overlap`` thresholds are knobs;
  defaults are deliberately loose so they pass for any well-shaped TS
  but flag pathological cases.
- For multi-bond / concerted reactions (Diels-Alder, sigmatropic), pass
  *all* reactive atoms in one ``reactive_atoms`` list — the overlap
  check uses the L2 amplitude of the imaginary mode restricted to those
  atoms relative to the full mode.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase import Atoms
from ase.io import read, write

from quantum_engine.ops import freq as freq_op
from quantum_engine.ops import saddle as saddle_op
from quantum_engine.units import EV_TO_KCAL

log = logging.getLogger("quantum_engine.ops.refine_ts")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _load_climb_path(neb_outdir: Path) -> tuple[list[Atoms], int, np.ndarray | None]:
    """Locate a NEB climbing-image XYZ and pick the TS image.

    Returns ``(images, ts_idx, tangent)`` where ``tangent`` is the central
    finite-difference tangent ``image[ts+1].pos - image[ts-1].pos`` (in Å).
    The tangent direction is the natural seed for the Dimer eigenmode.
    """
    candidates = [
        neb_outdir / "path-neb-climb.xyz",
        neb_outdir / "path-neb-noclimb.xyz",
        neb_outdir / "neb-final.xyz",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError(
            f"No NEB path file in {neb_outdir} (looked for "
            f"{', '.join(p.name for p in candidates)})"
        )
    log.info("refine-ts: loading NEB path from %s", src)
    images = read(str(src), index=":")
    if not isinstance(images, list):
        images = [images]

    # Pick the TS image as the inner-image energy maximum.
    energies = [float(img.info.get("energy", img.get_potential_energy()))
                if img.calc is not None or "energy" in img.info else 0.0
                for img in images]
    inner = energies[1:-1]
    if not inner:
        raise ValueError(f"NEB path in {src} has fewer than 3 images")
    ts_idx = int(np.argmax(inner)) + 1
    log.info("refine-ts: TS image = %d/%d (E = %.2f kcal/mol)",
             ts_idx, len(images) - 1, energies[ts_idx] * EV_TO_KCAL)

    # Central-difference tangent (same convention as the climbing-image method).
    tangent = None
    if 0 < ts_idx < len(images) - 1:
        tangent = (images[ts_idx + 1].get_positions()
                   - images[ts_idx - 1].get_positions())
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-8:
            log.warning("refine-ts: tangent ~0; falling back to None")
            tangent = None
        else:
            log.info("refine-ts: NEB tangent computed (|t|=%.4f Å)", norm)

    return images, ts_idx, tangent


def _resolve_reactive_indices(
    atoms: Atoms,
    bt_template,
    reactive_atoms: Sequence[int | str],
) -> list[int]:
    """Translate user-supplied reactive-atom specs to 0-based ASE indices.

    Accepted formats:
      - integer or numeric string: 1-based PDB serial number (preferred for
        users who copy from a PDB viewer).
      - ``"0:NAME"``: 0-based ASE index.
      - ``"RES:ID:ATOM"``: residue-name + residue-id + atom-name (requires
        the biotite template). Example: ``"SUB:1:P1"`` — note this lookup
        is *user-driven*, not hardcoded.

    The 1-based serial assumption matches the PDB spec; ``ase.io.read`` of a
    PDB preserves order, so PDB serial == ASE index + 1.
    """
    out: list[int] = []
    for spec in reactive_atoms:
        if isinstance(spec, int):
            out.append(spec - 1)  # interpret int as 1-based
            continue
        s = str(spec).strip()
        if s.startswith("0:"):
            out.append(int(s[2:]))
            continue
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            out.append(int(s) - 1)
            continue
        # RES:ID:ATOM lookup
        if bt_template is None:
            raise ValueError(
                f"reactive_atoms spec {s!r} requires a PDB template — pass "
                "atomic indices instead, or load the structure from PDB."
            )
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"reactive_atoms spec {s!r} not in 'RES:ID:ATOM' form; "
                "use integers (1-based PDB serial) or '0:idx'"
            )
        res_name, res_id, atom_name = parts
        try:
            res_id_int = int(res_id)
        except ValueError as exc:
            raise ValueError(f"residue id in {s!r} must be int") from exc
        idxs = np.where(
            (np.asarray(bt_template.res_name) == res_name)
            & (np.asarray(bt_template.res_id) == res_id_int)
            & (np.asarray(bt_template.atom_name) == atom_name)
        )[0]
        if not len(idxs):
            raise ValueError(f"reactive_atoms spec {s!r} not found in template")
        out.append(int(idxs[0]))

    out_unique = sorted(set(out))
    if not out_unique:
        raise ValueError("reactive_atoms list is empty after parsing")
    if any(i < 0 or i >= len(atoms) for i in out_unique):
        raise ValueError(
            f"reactive_atoms resolved to out-of-range indices: "
            f"{out_unique} (n_atoms={len(atoms)})"
        )
    log.info("refine-ts: reactive atoms -> 0-based indices %s "
             "(elements %s)", out_unique,
             [atoms.get_chemical_symbols()[i] for i in out_unique])
    return out_unique


def _imag_mode_overlap(
    mode: np.ndarray,
    reactive_indices: Sequence[int],
) -> float:
    """L2 amplitude of ``mode`` restricted to ``reactive_indices`` over the
    total amplitude. Returns a value in ``[0, 1]``: 1 means all motion is
    on the reactive atoms, 0 means none.

    ``mode`` is shape ``(N, 3)`` and assumed unit-normalised. We normalise
    just to be safe.
    """
    total = float(np.linalg.norm(mode))
    if total < 1e-12:
        return 0.0
    sub = float(np.linalg.norm(mode[list(reactive_indices)]))
    return sub / total


@dataclass
class RefineTSCriteria:
    """Pass/fail thresholds for TS validation."""
    require_converged: bool = True
    n_imag_expected: int = 1
    imag_cm_cutoff: float = -50.0  # imag freq must be MORE NEGATIVE than this
    imag_mode_overlap: float = 0.5  # frac of imag-mode amplitude on reactive atoms


@dataclass
class RefineTSCheck:
    name: str
    passed: bool
    detail: str
    value: Any = None


@dataclass
class RefineTSReport:
    """Structured pass/fail summary for a refined TS."""
    overall_pass: bool
    backend_used: str | None
    energy_eV: float
    energy_kcal_mol: float
    fmax_final: float | None
    n_imag: int | None
    imag_freq_cm: float | None
    imag_mode_overlap: float | None
    reactive_indices: list[int]
    saddle_attempts: list[dict[str, Any]] = field(default_factory=list)
    checks: list[RefineTSCheck] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def run(
    atoms: Atoms,
    calculator,
    *,
    outdir: str | Path,
    reactive_atoms: Sequence[int | str],
    bt_template=None,
    constraint=None,
    backend: str = "dimer",
    initial_mode_vector: np.ndarray | None = None,
    saddle_fmax: float = 0.02,
    saddle_max_steps: int = 500,
    eigh_drivers: Sequence[str] | None = None,
    cascade: Sequence[str] = saddle_op.AUTO_CASCADE,
    freq_indices: Sequence[int] | None = None,
    freq_delta: float = 0.005,
    freq_method: str = "central",
    criteria: RefineTSCriteria | None = None,
    write_pdb_cif: bool = True,
    pdb_charge_hint: int | None = None,
    **kwargs,
) -> dict:
    """End-to-end TS refinement + frequency validation.

    Args:
        atoms:                  TS guess Atoms (already has .calc or pass
                                ``calculator``).
        calculator:             ASE calculator factory (or instance).
        outdir:                 Output directory.
        reactive_atoms:         Indices / serials / names of atoms whose
                                motion must dominate the imaginary mode.
        bt_template:            Optional biotite AtomArray (only needed if
                                ``reactive_atoms`` uses ``RES:ID:NAME``
                                spec or ``write_pdb_cif=True``).
        constraint:             Optional ASE constraint applied during
                                refine + freq.
        backend:                One of :data:`saddle_op.KNOWN_BACKENDS`.
        initial_mode_vector:    Seed direction for dimer-style backends.
                                Use the NEB tangent (loaded automatically
                                when ``--from-neb`` is passed at the CLI).
        saddle_fmax,
        saddle_max_steps:       Saddle-search convergence + iteration cap.
        eigh_drivers:           LAPACK driver retry list for Sella backends
                                (see :mod:`quantum_engine.qm.sella`).
        cascade:                Auto-cascade ordering when ``backend='auto'``.
        freq_indices:           Atom indices for the partial Hessian.
                                Defaults to ``reactive_atoms`` plus their
                                first-shell neighbours (see implementation).
        freq_delta:             Finite-difference step (Å). Tighter than
                                the ASE default ``0.02`` because we want
                                accurate frequencies, not just gradients.
        freq_method:            ``"central"`` (default) or ``"forward"``.
        criteria:               Pass/fail thresholds; uses the dataclass
                                defaults if omitted.
        write_pdb_cif:          If True, write a refined PDB + CIF using
                                the same template the input came from
                                (skipped if ``bt_template`` is None).
        pdb_charge_hint:        Charge written to the PDB REMARK header.

    Returns:
        Dict including ``ts_atoms``, ``saddle_result``, ``freq_result``,
        ``report`` (a :class:`RefineTSReport`), and ``outputs`` paths.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if criteria is None:
        criteria = RefineTSCriteria()

    if atoms.calc is None:
        if calculator is None:
            raise ValueError("refine_ts: no calc on atoms and none supplied")
        atoms.calc = calculator

    reactive_idx = _resolve_reactive_indices(atoms, bt_template, reactive_atoms)
    atoms.info["reactive_atoms"] = reactive_idx

    # ------------------------------------------------------------------
    # Step 1: saddle refinement
    # ------------------------------------------------------------------
    saddle_dir = outdir / "saddle"
    log.info("refine-ts: STEP 1 — saddle refinement (backend=%s)", backend)
    saddle_result = saddle_op.run(
        atoms,
        calculator=calculator,
        outdir=saddle_dir,
        constraint=constraint,
        fmax=saddle_fmax,
        max_steps=saddle_max_steps,
        backend=backend,
        initial_mode_vector=initial_mode_vector,
        eigh_drivers=eigh_drivers,
        cascade=cascade,
    )

    ts_atoms = saddle_result.get("atoms", atoms)
    converged = bool(saddle_result.get("converged", False))
    backend_used = saddle_result.get("backend_used")
    saddle_attempts = saddle_result.get("backend_attempts", [])

    # Re-attach calculator (cascade may have spawned a fresh atoms object)
    if ts_atoms.calc is None:
        ts_atoms.calc = calculator if calculator is not None else atoms.calc
    if constraint is not None:
        ts_atoms.set_constraint(constraint
                                if isinstance(constraint, list) else [constraint])

    fmax_final = float(np.linalg.norm(ts_atoms.get_forces(), axis=1).max())
    energy_eV = float(ts_atoms.get_potential_energy())

    log.info("refine-ts: saddle %s, fmax=%.4f eV/Å, E=%.2f kcal/mol, backend=%s",
             "converged" if converged else "NOT converged",
             fmax_final, energy_eV * EV_TO_KCAL, backend_used)

    # ------------------------------------------------------------------
    # Step 2: partial-Hessian frequency analysis
    # ------------------------------------------------------------------
    if freq_indices is None:
        # Default: reactive atoms + their bonded neighbours (out to ~2 Å).
        # This keeps the Hessian small but resolves modes that mix into the
        # reaction coordinate.
        from ase.neighborlist import NeighborList, natural_cutoffs
        cutoffs = natural_cutoffs(ts_atoms, mult=1.1)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(ts_atoms)
        neighbour_set: set[int] = set(reactive_idx)
        for i in reactive_idx:
            neighbours, _ = nl.get_neighbors(i)
            neighbour_set.update(int(j) for j in neighbours)
        freq_indices = sorted(neighbour_set)
        log.info("refine-ts: auto-selected %d freq indices "
                 "(%d reactive + %d bonded shell)",
                 len(freq_indices), len(reactive_idx),
                 len(freq_indices) - len(reactive_idx))
    else:
        freq_indices = sorted(set(int(i) for i in freq_indices))

    log.info("refine-ts: STEP 2 — partial Hessian (%d atoms, delta=%.4f Å)",
             len(freq_indices), freq_delta)
    freq_dir = outdir / "freq"
    freq_result = freq_op.run(
        ts_atoms, calculator=calculator,
        outdir=freq_dir,
        constraint=constraint,
        indices=freq_indices,
        delta=freq_delta,
        method=freq_method,
    )

    # ------------------------------------------------------------------
    # Step 3: diagnostics + pass/fail
    # ------------------------------------------------------------------
    freqs_cm = np.asarray(freq_result["frequencies_cm"])
    n_imag = int(np.sum(freqs_cm < -10))
    imag_freq_cm = float(freqs_cm[0]) if len(freqs_cm) else float("nan")
    modes = np.load(freq_result["outputs"]["modes"])
    overlap = _imag_mode_overlap(modes[0], reactive_idx) if len(modes) else 0.0

    checks: list[RefineTSCheck] = []
    checks.append(RefineTSCheck(
        name="saddle_converged",
        passed=converged or not criteria.require_converged,
        detail=f"backend={backend_used}, fmax_final={fmax_final:.4f} eV/Å",
        value=fmax_final,
    ))
    checks.append(RefineTSCheck(
        name="n_imag_match",
        passed=n_imag == criteria.n_imag_expected,
        detail=f"observed {n_imag} imag modes (< -10 cm⁻¹), "
               f"expected {criteria.n_imag_expected}",
        value=n_imag,
    ))
    checks.append(RefineTSCheck(
        name="imag_below_cutoff",
        passed=imag_freq_cm < criteria.imag_cm_cutoff,
        detail=f"imag = {imag_freq_cm:.1f} cm⁻¹, "
               f"cutoff {criteria.imag_cm_cutoff:.0f}",
        value=imag_freq_cm,
    ))
    checks.append(RefineTSCheck(
        name="imag_mode_overlap",
        passed=overlap >= criteria.imag_mode_overlap,
        detail=f"reactive-subset amplitude / total = {overlap:.3f}, "
               f"required >= {criteria.imag_mode_overlap:.2f}",
        value=overlap,
    ))
    overall = all(c.passed for c in checks)

    # ------------------------------------------------------------------
    # Step 4: write artefacts
    # ------------------------------------------------------------------
    outputs: dict[str, str] = {}

    refined_xyz = outdir / "ts_refined.xyz"
    write(str(refined_xyz), ts_atoms, format="extxyz")
    outputs["ts_xyz"] = str(refined_xyz)

    # Imaginary mode displacement vectors (NumPy + per-atom CSV for inspection)
    np.save(outdir / "imag_mode.npy", modes[0] if len(modes) else np.zeros((len(ts_atoms), 3)))
    outputs["imag_mode"] = str(outdir / "imag_mode.npy")

    if write_pdb_cif and bt_template is not None and len(bt_template) == len(ts_atoms):
        from quantum_engine.io import write_pdb
        pdb_path = outdir / "ts_refined.pdb"
        write_pdb(ts_atoms, bt_template, pdb_path,
                  total_charge=pdb_charge_hint, energy_eV=energy_eV)
        outputs["ts_pdb"] = str(pdb_path)

        try:
            from quantum_engine.io.cif import write_ts_cif
            cif_path = write_ts_cif(
                ts_atoms, bt_template, outdir,
                total_charge=int(pdb_charge_hint or 0),
                filename="ts_refined.cif",
            )
            if cif_path is not None:
                outputs["ts_cif"] = str(cif_path)
        except Exception as exc:
            log.warning("CIF write failed (%s); PDB still written", exc)

    report = RefineTSReport(
        overall_pass=overall,
        backend_used=backend_used,
        energy_eV=energy_eV,
        energy_kcal_mol=energy_eV * EV_TO_KCAL,
        fmax_final=fmax_final,
        n_imag=n_imag,
        imag_freq_cm=imag_freq_cm,
        imag_mode_overlap=overlap,
        reactive_indices=list(reactive_idx),
        saddle_attempts=saddle_attempts,
        checks=checks,
        outputs=outputs,
    )

    summary_path = outdir / "summary.json"
    summary_payload = {
        "overall_pass": overall,
        "backend_used": backend_used,
        "fmax_final": fmax_final,
        "energy_eV": energy_eV,
        "energy_kcal_mol": energy_eV * EV_TO_KCAL,
        "n_imag": n_imag,
        "imag_freq_cm": imag_freq_cm,
        "imag_mode_overlap": overlap,
        "reactive_indices": list(reactive_idx),
        "freq_indices": list(freq_indices),
        "saddle_attempts": saddle_attempts,
        "checks": [asdict(c) for c in checks],
        "outputs": outputs,
        "criteria": asdict(criteria),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, default=float))
    outputs["summary"] = str(summary_path)
    report.outputs = outputs

    log.info("refine-ts: %s (n_imag=%d, |imag|=%.0f cm⁻¹, overlap=%.2f)",
             "PASS" if overall else "FAIL",
             n_imag, abs(imag_freq_cm), overlap)

    return {
        "status": "completed" if overall else "failed_validation",
        "overall_pass": overall,
        "ts_atoms": ts_atoms,
        "saddle_result": saddle_result,
        "freq_result": freq_result,
        "report": report,
        "outputs": outputs,
        "energy_eV": energy_eV,
        "energy_kcal_mol": energy_eV * EV_TO_KCAL,
        "imag_freq_cm": imag_freq_cm,
        "n_imag": n_imag,
        "backend_used": backend_used,
    }


def run_from_neb(
    neb_outdir: str | Path,
    calculator,
    *,
    outdir: str | Path,
    reactive_atoms: Sequence[int | str],
    bt_template=None,
    pdb_for_template: str | Path | None = None,
    backend: str = "dimer",
    use_neb_tangent: bool = True,
    initial_mode_vector_path: str | Path | None = None,
    constraint=None,
    **refine_kwargs,
) -> dict:
    """Convenience wrapper: pull TS guess + tangent from a NEB run, then refine.

    Args:
        neb_outdir:                 Path to a directory containing
                                    ``path-neb-climb.xyz`` (or noclimb fallback).
        calculator:                 ASE calculator instance / factory.
        outdir:                     Where to write refined artefacts.
        reactive_atoms:             Same spec as :func:`run`.
        bt_template:                Optional biotite template; if absent and
                                    ``pdb_for_template`` is set, parsed from there.
        pdb_for_template:           Optional PDB whose annotations should be
                                    preserved on the refined TS PDB output.
                                    The PDB *must* have the same atom order
                                    as the NEB images.
        backend:                    Saddle-backend choice (default ``dimer``
                                    because the NEB tangent seeds it for free).
        use_neb_tangent:            If True (default), seed dimer-style
                                    backends with the central-difference
                                    tangent at the climbing image.
        initial_mode_vector_path:   Override the tangent with a user file
                                    (XYZ; see :func:`quantum_engine.qm.dimer.load_mode_from_xyz`).
        **refine_kwargs:            Forwarded to :func:`run`.
    """
    neb_outdir = Path(neb_outdir)
    images, ts_idx, tangent = _load_climb_path(neb_outdir)
    ts_guess = images[ts_idx].copy()
    if calculator is not None and not callable(calculator):
        ts_guess.calc = calculator
    elif callable(calculator):
        ts_guess.calc = calculator()

    # Resolve template
    if bt_template is None and pdb_for_template is not None:
        from quantum_engine.io import load_structure
        _, bt_template, charge_hint_pdb = load_structure(pdb_for_template)
        refine_kwargs.setdefault("pdb_charge_hint", charge_hint_pdb)

    # Resolve initial-mode vector
    initial_mode_vector = None
    if initial_mode_vector_path:
        from quantum_engine.qm.dimer import load_mode_from_xyz
        initial_mode_vector = load_mode_from_xyz(
            initial_mode_vector_path, n_atoms=len(ts_guess)
        )
    elif use_neb_tangent and tangent is not None:
        initial_mode_vector = tangent

    return run(
        ts_guess,
        calculator=calculator,
        outdir=outdir,
        reactive_atoms=reactive_atoms,
        bt_template=bt_template,
        constraint=constraint,
        backend=backend,
        initial_mode_vector=initial_mode_vector,
        **refine_kwargs,
    )
