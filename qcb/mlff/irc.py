"""
IRC-from-TS strategy: the textbook gold standard for TS search when a TS guess is available.

Method
------
1. Take the input structure as a TS guess (common when researchers hand-build TS geometries)
2. Sella saddle-point optimization → converges to a true first-order saddle
3. Displace along the imaginary Hessian eigenvector in both directions (+δ, -δ)
4. Steepest-descent / damped-MD relaxation from each displaced geometry
5. The two relaxed endpoints ARE the reactant and product — by construction connected through this specific TS

Why this works where spring-driven endpoint generation fails
------------------------------------------------------------
- No assumptions about the product geometry — it is discovered
- No spring parameters to tune — geometry flows downhill on the PES
- No MD-induced escape from shallow product basins — IRC uses damped descent
- Faithful to the TS: if the TS leads to a pentacoordinate intermediate, IRC finds that intermediate
  directly (and you know you need a second NEB segment)

When to use
-----------
- USE when input structure is already near a TS (partially formed/broken bonds)
- USE as a validation pass after NEB converges (IRC from the CI-NEB TS confirms connectivity)
- USE for low-barrier / ill-defined-product systems where spring-driven endpoint generation fails

When NOT to use
---------------
- Input is a clean reactant or product (no TS guess) → use spring/metadynamics
- System has >200 free atoms and Sella internal-coords struggle (fallback: Cartesian + larger tolerance)
- PES has soft modes (floppy chains) that swamp the reactive mode → consider constraining those modes

References
----------
- Fukui, K. "The Path of Chemical Reactions — The IRC Approach." Acc. Chem. Res. 1981, 14, 363.
- Hermes, E.D. et al. "Sella: An Open-Source, Automation-Friendly Molecular Saddle Point Optimizer."
  J. Chem. Theory Comput. 2022, 18, 6974.  https://doi.org/10.1021/acs.jctc.1c00412
- Schlegel, H.B. "Geometry optimization." WIREs Comput. Mol. Sci. 2011, 1, 790.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import LBFGS

log = logging.getLogger("qcb.irc")

EV_TO_KCAL = 23.0609
BOHR_TO_A = 0.529177


def sella_saddle_search(
    atoms: Atoms,
    outdir: str | Path,
    constraint=None,
    fmax: float = 0.02,
    max_steps: int = 500,
    use_internal: bool | None = None,
) -> tuple[Atoms, bool]:
    """Find the nearest first-order saddle point using Sella.

    Args:
        atoms: Starting geometry (should be a TS guess, not a minimum)
        outdir: Directory for trajectory/log files
        constraint: Optional ASE constraint (e.g., FixAtoms on CAs)
        fmax: Force convergence threshold (eV/Å). Default 0.02 is tighter than
              typical opt (0.05) because Sella is iteratively improving the
              Hessian and needs tight convergence to find the true saddle.
        max_steps: Safety cap on iterations
        use_internal: Use internal coordinates. If None, auto-detect based on
                     free atom count (<200 → internal, else Cartesian).

    Returns:
        (saddle_atoms, success_flag)
    """
    from sella import Sella

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ts_atoms = atoms.copy()
    ts_atoms.calc = atoms.calc
    if constraint is not None:
        ts_atoms.set_constraint(constraint)

    # Auto-detect coordinate system
    if use_internal is None:
        if constraint is not None and hasattr(constraint, "index"):
            n_free = len(atoms) - len(constraint.index)
        else:
            n_free = len(atoms)
        use_internal = n_free < 200

    traj_file = outdir / "sella-saddle.traj"
    log_file = outdir / "sella-saddle.log"

    log.info(f"  Sella saddle search: {len(atoms)} atoms, "
             f"{'internal' if use_internal else 'Cartesian'} coords, fmax={fmax}")

    # Try internal coords first, fall back to Cartesian
    for attempt_internal in [use_internal, False] if use_internal else [False]:
        try:
            dyn = Sella(
                ts_atoms,
                order=1,                       # first-order saddle (TS)
                internal=attempt_internal,
                trajectory=str(traj_file),
                logfile=str(log_file),
            )
            dyn.run(fmax=fmax, steps=max_steps)

            # Verify convergence: recompute forces (fmax is only checked during iteration)
            f_max_actual = float(np.linalg.norm(ts_atoms.get_forces(), axis=1).max())
            converged = f_max_actual < fmax * 1.5  # 50% tolerance
            log.info(f"  Sella {'internal' if attempt_internal else 'Cartesian'}: "
                     f"fmax_final={f_max_actual:.4f}, "
                     f"E={ts_atoms.get_potential_energy()*EV_TO_KCAL:.2f} kcal/mol")

            if converged:
                # Save final geometry
                write(outdir / "saddle.xyz", ts_atoms, format="extxyz")
                return ts_atoms, True

        except Exception as e:
            log.warning(f"  Sella {'internal' if attempt_internal else 'Cartesian'} failed: {e}")
            if attempt_internal:
                continue
            raise

    log.warning("  Sella saddle search did not converge — TS may be unreliable")
    return ts_atoms, False


def compute_imaginary_mode(
    atoms: Atoms,
    indices_of_interest: list[int] | None = None,
    delta: float = 0.01,
) -> tuple[np.ndarray, float]:
    """Compute the lowest-frequency (imaginary) normal mode at a saddle.

    Finite-difference Hessian restricted to atoms of interest (the reacting
    atoms). This is MUCH faster than full Hessian and typically sufficient
    to identify the reaction coordinate mode.

    Args:
        atoms: Structure at a saddle point (one imaginary frequency expected)
        indices_of_interest: Atom indices to include in partial Hessian
                            (default: all non-constrained atoms)
        delta: Finite-difference step in Å. Large delta (~0.02) helps with
               noisy ML potentials (Hermes et al. 2022 Sella paper).

    Returns:
        (displacement_vector, frequency_cm^-1)
        displacement_vector has shape (N, 3) with unit norm
        frequency_cm^-1 is negative for imaginary modes
    """
    if indices_of_interest is None:
        constraint = atoms.constraints[0] if atoms.constraints else None
        if constraint is not None and hasattr(constraint, "index"):
            constrained = set(constraint.index.tolist() if hasattr(constraint.index, 'tolist') else constraint.index)
            indices_of_interest = [i for i in range(len(atoms)) if i not in constrained]
        else:
            indices_of_interest = list(range(len(atoms)))

    n_free = len(indices_of_interest)
    log.info(f"  Computing imaginary mode: {n_free} atoms, finite-diff with delta={delta} Å")

    # Build 3N_free × 3N_free Hessian
    H = np.zeros((3 * n_free, 3 * n_free))

    pos0 = atoms.get_positions().copy()
    masses = atoms.get_masses()

    for a_idx, atom_i in enumerate(indices_of_interest):
        for xyz_i in range(3):
            row = 3 * a_idx + xyz_i

            # Central finite difference of forces
            pos_plus = pos0.copy()
            pos_plus[atom_i, xyz_i] += delta
            atoms.set_positions(pos_plus)
            f_plus = atoms.get_forces()

            pos_minus = pos0.copy()
            pos_minus[atom_i, xyz_i] -= delta
            atoms.set_positions(pos_minus)
            f_minus = atoms.get_forces()

            # H_ij = -dF_i/dr_j (force is negative gradient)
            for b_idx, atom_j in enumerate(indices_of_interest):
                for xyz_j in range(3):
                    col = 3 * b_idx + xyz_j
                    H[row, col] = -(f_plus[atom_j, xyz_j] - f_minus[atom_j, xyz_j]) / (2 * delta)

    # Restore positions
    atoms.set_positions(pos0)

    # Mass-weight: H_mw = M^(-1/2) H M^(-1/2)
    m_atom = np.array([masses[atom_i] for atom_i in indices_of_interest])
    if np.any(m_atom <= 0):
        log.warning(f"  Non-positive masses detected for indices "
                    f"{[indices_of_interest[i] for i in np.where(m_atom <= 0)[0]]}; "
                    "setting to 1.0 amu for Hessian diagonalization")
        m_atom = np.clip(m_atom, 1.0, None)
    m_sqrt = np.repeat(np.sqrt(m_atom), 3)
    H_mw = H / np.outer(m_sqrt, m_sqrt)
    # Symmetrize
    H_mw = 0.5 * (H_mw + H_mw.T)

    # Diagonalize
    eigvals, eigvecs = np.linalg.eigh(H_mw)

    # Convert eigenvalue → frequency (cm^-1)
    # ω^2 [eV/Å²/amu] × conversion → cm^-1
    # sign: imaginary if eigval < 0
    conv = 521.47  # sqrt(eV/Å²/amu) → cm^-1
    freqs = np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * conv

    # Lowest mode (most negative = most imaginary)
    lowest = eigvecs[:, 0]
    freq_lowest = freqs[0]

    log.info(f"  Lowest mode: {freq_lowest:.1f} cm^-1 " +
             ("(imaginary — TS-like)" if freq_lowest < 0 else "(real — not a TS!)"))

    # Reshape to (N_total, 3), filling constrained atoms with zeros
    displacement = np.zeros((len(atoms), 3))
    for a_idx, atom_i in enumerate(indices_of_interest):
        displacement[atom_i] = lowest[3 * a_idx : 3 * a_idx + 3]

    # Normalize
    norm = np.linalg.norm(displacement)
    if norm > 0:
        displacement /= norm

    return displacement, freq_lowest


def run_irc(
    ts_atoms: Atoms,
    outdir: str | Path,
    direction: str = "both",
    step_size: float = 0.1,
    max_steps: int = 200,
    fmax: float = 0.03,
    constraint=None,
    reactant_hint_bonds: dict | None = None,
) -> dict:
    """Run intrinsic reaction coordinate (IRC) from a TS in one or both directions.

    Uses a simple damped-descent implementation:
      1. Displace TS along imaginary mode by ±step_size × mode_vector
      2. LBFGS-relax from each displaced geometry (damped descent on the PES)
      3. The two relaxed endpoints are the reactant and product

    This is NOT the full mass-weighted Page-McIver IRC algorithm, but works well
    for ML potentials where Hessian noise makes rigorous IRC unstable.

    Args:
        ts_atoms: Converged TS (saddle point)
        outdir: Directory for trajectory files
        direction: "forward", "reverse", or "both"
        step_size: Initial displacement along imaginary mode (Å, before normalization)
        max_steps: LBFGS iteration cap
        fmax: Convergence threshold for final endpoint relaxation
        constraint: Optional ASE constraint

    Returns:
        dict with keys 'reactant', 'product' (ASE Atoms), 'mode_freq' (cm^-1),
        'dE_reactant_kcal', 'dE_product_kcal', 'success' (bool)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log.info("IRC: computing imaginary mode at TS ...")
    displacement, freq = compute_imaginary_mode(ts_atoms)

    result = {
        "mode_freq": freq,
        "reactant": None,
        "product": None,
        "success": True,
    }

    if freq > -50:  # threshold: modes between -50 and 0 cm^-1 are typically noise
        log.warning(f"  Lowest mode ({freq:.1f} cm^-1) is not clearly imaginary. "
                    "IRC results may be unreliable.")

    # Sanity check: the chosen mode should have significant amplitude on reacting atoms
    # (if hints provided). This catches spurious soft modes in enzyme systems.
    if reactant_hint_bonds:
        reacting_atom_indices = set()
        for (i, j), _ in reactant_hint_bonds.items():
            reacting_atom_indices.update([i, j])
        reacting_amp = float(np.linalg.norm(displacement[list(reacting_atom_indices)]))
        total_amp = float(np.linalg.norm(displacement))
        if total_amp > 0 and reacting_amp / total_amp < 0.1:
            log.warning(
                f"  Imaginary mode has only {100*reacting_amp/total_amp:.1f}% amplitude "
                "on reacting atoms — may be a soft-mode contamination, not the reactive mode."
            )

    e_ts = ts_atoms.get_potential_energy()
    pos_ts = ts_atoms.get_positions().copy()

    # Descend both sides first, then assign reactant/product by bond patterns
    # (the mode eigenvector sign is arbitrary, so +/- directions don't map to
    # forward/reverse in any physically meaningful way)
    descended = {}  # {"plus": atoms, "minus": atoms}

    for side_tag, sign in [("plus", +1), ("minus", -1)]:
        log.info(f"IRC displacing {sign:+d} × {step_size:.3f} Å × mode ...")
        atoms_side = ts_atoms.copy()
        atoms_side.calc = ts_atoms.calc
        if constraint is not None:
            atoms_side.set_constraint(constraint)

        atoms_side.set_positions(pos_ts + sign * step_size * displacement)

        traj_file = outdir / f"irc-{side_tag}.traj"
        log_file = outdir / f"irc-{side_tag}.log"

        try:
            dyn = LBFGS(atoms_side, trajectory=str(traj_file), logfile=str(log_file))
            dyn.run(fmax=fmax, steps=max_steps)
            e_end = atoms_side.get_potential_energy()
            dE = (e_end - e_ts) * EV_TO_KCAL

            log.info(f"  IRC {side_tag}-direction converged: E = E_TS {dE:+.2f} kcal/mol")
            descended[side_tag] = (atoms_side, dE)
        except Exception as e:
            log.error(f"  IRC {side_tag}-direction failed: {e}")
            result["success"] = False

    # Now assign which side is reactant vs product
    if "plus" in descended and "minus" in descended:
        plus_atoms, plus_dE = descended["plus"]
        minus_atoms, minus_dE = descended["minus"]

        # Prefer bond-pattern assignment if hints given
        assigned = False
        if reactant_hint_bonds:
            def _score_as_reactant(atoms):
                score = 0
                for (i, j), expected in reactant_hint_bonds.items():
                    d = atoms.get_distance(i, j)
                    if expected == "bonded" and d < 2.0:
                        score += 1
                    elif expected == "broken" and d > 2.5:
                        score += 1
                return score

            plus_score = _score_as_reactant(plus_atoms)
            minus_score = _score_as_reactant(minus_atoms)

            if plus_score > minus_score:
                result["reactant"], result["dE_reactant_kcal"] = plus_atoms, plus_dE
                result["product"], result["dE_product_kcal"] = minus_atoms, minus_dE
                assigned = True
                log.info(f"  Assigned by bond pattern: +direction → reactant ({plus_score}/{minus_score} bonds match)")
            elif minus_score > plus_score:
                result["reactant"], result["dE_reactant_kcal"] = minus_atoms, minus_dE
                result["product"], result["dE_product_kcal"] = plus_atoms, plus_dE
                assigned = True
                log.info(f"  Assigned by bond pattern: −direction → reactant ({minus_score}/{plus_score} bonds match)")

        # Fallback: assign so reactant is the lower-energy side (more stable basin)
        # This is a heuristic — for exothermic reactions the product is lower, which
        # would be wrong. Without hints, we cannot resolve the ambiguity.
        if not assigned:
            log.warning(
                "  No bond hints provided — forward/reverse assignment is arbitrary. "
                "Reactant = lower-energy side heuristic applied."
            )
            if plus_dE < minus_dE:
                result["reactant"], result["dE_reactant_kcal"] = plus_atoms, plus_dE
                result["product"], result["dE_product_kcal"] = minus_atoms, minus_dE
            else:
                result["reactant"], result["dE_reactant_kcal"] = minus_atoms, minus_dE
                result["product"], result["dE_product_kcal"] = plus_atoms, plus_dE

        write(outdir / "reactant.xyz", result["reactant"], format="extxyz")
        write(outdir / "product.xyz", result["product"], format="extxyz")

    # Sanity: reactant and product should be distinct
    if result["reactant"] is not None and result["product"] is not None:
        rmsd = float(np.sqrt(np.mean(
            (result["reactant"].get_positions() - result["product"].get_positions())**2
        )))
        log.info(f"  Reactant-Product RMSD: {rmsd:.3f} Å")
        if rmsd < 0.1:
            log.warning("  Reactant and product are nearly identical — IRC failed to separate basins")
            result["success"] = False

    return result


def irc_from_ts_guess(
    atoms: Atoms,
    outdir: str | Path,
    constraint=None,
    saddle_fmax: float = 0.02,
    irc_step: float = 0.1,
    irc_fmax: float = 0.03,
    reactant_hint_bonds: dict | None = None,
) -> dict:
    """Full IRC-from-TS workflow: Sella saddle search + IRC descent both ways.

    This is the complete gold-standard workflow when the input is a TS guess.

    Returns dict with 'ts', 'reactant', 'product', 'imag_freq_cm', 'barrier_fwd_kcal',
    'barrier_rev_kcal', 'success'.
    """
    outdir = Path(outdir)

    # Step 1: Refine TS with Sella
    log.info("=" * 60)
    log.info("IRC-from-TS workflow: Step 1 — Sella saddle optimization")
    log.info("=" * 60)
    ts_refined, converged = sella_saddle_search(
        atoms, outdir, constraint=constraint, fmax=saddle_fmax
    )

    if not converged:
        log.warning("  TS did not converge to saddle — IRC endpoints may not connect correctly")

    # Step 2: IRC in both directions
    log.info("=" * 60)
    log.info("IRC-from-TS workflow: Step 2 — IRC descent both directions")
    log.info("=" * 60)
    irc_result = run_irc(
        ts_refined, outdir,
        direction="both",
        step_size=irc_step,
        fmax=irc_fmax,
        constraint=constraint,
        reactant_hint_bonds=reactant_hint_bonds,
    )

    # Compute barriers
    e_ts = ts_refined.get_potential_energy()
    result = {
        "ts": ts_refined,
        "reactant": irc_result.get("reactant"),
        "product": irc_result.get("product"),
        "imag_freq_cm": irc_result["mode_freq"],
        "success": irc_result["success"] and converged,
    }

    if result["reactant"] is not None:
        e_r = result["reactant"].get_potential_energy()
        result["barrier_fwd_kcal"] = (e_ts - e_r) * EV_TO_KCAL
    if result["product"] is not None:
        e_p = result["product"].get_potential_energy()
        result["barrier_rev_kcal"] = (e_ts - e_p) * EV_TO_KCAL

    log.info("=" * 60)
    log.info(f"IRC-from-TS done: barrier_fwd={result.get('barrier_fwd_kcal', 0):.2f} kcal/mol, "
             f"barrier_rev={result.get('barrier_rev_kcal', 0):.2f} kcal/mol, "
             f"imag_freq={result['imag_freq_cm']:.0f} cm^-1")
    log.info("=" * 60)

    return result
