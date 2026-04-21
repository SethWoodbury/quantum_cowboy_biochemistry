"""
Endpoint generation methods for NEB reactant/product geometries.

Provides four strategies for generating endpoint geometries from a
transition state guess, replacing the previous hardcoded spring approach:

1. equilibrium_scan — Covalent radii + 1D energy scan
2. constrained_scan — Constrained optimization at fixed bond distances
3. incremental_stretch — Stepwise bond stretching with intermediate relaxation
4. spring_then_relax — Spring driving followed by unconstrained relaxation
"""

from __future__ import annotations

import copy
import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms, FixBondLength
from ase.data import atomic_numbers, covalent_radii
from ase.io import write
from ase.optimize import FIRE, LBFGS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def get_smart_bond_targets(
    atoms: Atoms,
    ligand_name: str,
    bond_defs: List[Dict],
) -> List[Dict]:
    """Replace hardcoded target distances with covalent-radii-based values.

    For 'attractive' (bond forming): target = sum of covalent radii * 1.0
    For 'repulsive' (bond breaking): target = sum of covalent radii * 2.5

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object with current geometry.
    ligand_name : str
        Identifier for the ligand (used in logging).
    bond_defs : list of dict
        Each dict must contain:
            - 'idx1': int, first atom index
            - 'idx2': int, second atom index
            - 'type': str, either 'attractive' or 'repulsive'
        Optionally may already contain 'target' which will be overwritten.

    Returns
    -------
    list of dict
        Updated bond definitions with computed 'target' distances (Angstrom).
    """
    updated = []
    for bond in bond_defs:
        bd = dict(bond)
        idx1, idx2 = bd["idx1"], bd["idx2"]
        z1 = atoms.numbers[idx1]
        z2 = atoms.numbers[idx2]
        r_sum = covalent_radii[z1] + covalent_radii[z2]

        if bd["type"] == "attractive":
            bd["target"] = r_sum * 1.0
        elif bd["type"] == "repulsive":
            bd["target"] = r_sum * 2.5
        else:
            raise ValueError(
                f"Unknown bond type '{bd['type']}' for bond {idx1}-{idx2}. "
                "Expected 'attractive' or 'repulsive'."
            )

        logger.info(
            "Ligand %s: bond %d-%d (%s) target = %.3f A "
            "(covalent radii sum = %.3f A)",
            ligand_name,
            idx1,
            idx2,
            bd["type"],
            bd["target"],
            r_sum,
        )
        updated.append(bd)
    return updated


def _attach_calculator(atoms: Atoms, calc_fn: Callable) -> Atoms:
    """Attach a fresh calculator from a factory function."""
    atoms.calc = calc_fn()
    return atoms


def _get_covalent_sum(atoms: Atoms, idx1: int, idx2: int) -> float:
    """Return sum of covalent radii for two atoms."""
    return covalent_radii[atoms.numbers[idx1]] + covalent_radii[atoms.numbers[idx2]]


# ---------------------------------------------------------------------------
# Method 1: equilibrium_scan
# ---------------------------------------------------------------------------


def get_equilibrium_bond_length(
    atoms: Atoms,
    idx1: int,
    idx2: int,
    calc_fn: Callable,
    distances: Optional[Sequence[float]] = None,
    fmax: float = 0.05,
    steps: int = 200,
) -> Tuple[float, np.ndarray]:
    """Scan a bond distance to find the equilibrium length on the MLFF surface.

    Fixes the bond at each candidate distance, relaxes all other degrees of
    freedom, and returns the distance that yields the lowest energy.

    Parameters
    ----------
    atoms : Atoms
        Starting geometry (will not be modified).
    idx1 : int
        Index of first atom in the bond.
    idx2 : int
        Index of second atom in the bond.
    calc_fn : callable
        Factory function that returns a fresh ASE calculator.
    distances : sequence of float, optional
        Bond distances (Angstrom) to scan. Defaults to
        [1.4, 1.5, 1.6, 1.7, 1.8] for a typical P-O bond.
    fmax : float
        Force convergence for constrained optimizations.
    steps : int
        Maximum optimization steps per scan point.

    Returns
    -------
    tuple of (float, np.ndarray)
        Best distance (Angstrom) and array of energies at each scan point.
    """
    if distances is None:
        distances = [1.4, 1.5, 1.6, 1.7, 1.8]

    distances = np.asarray(distances, dtype=float)
    energies = np.full(len(distances), np.inf)

    logger.info(
        "Equilibrium scan for bond %d-%d over distances: %s",
        idx1,
        idx2,
        distances,
    )

    for i, d in enumerate(distances):
        scan_atoms = atoms.copy()
        _attach_calculator(scan_atoms, calc_fn)

        # Set the bond to the target distance
        _set_bond_distance(scan_atoms, idx1, idx2, d)

        # Constrain this bond and relax everything else
        scan_atoms.set_constraint(FixBondLength(idx1, idx2))

        try:
            opt = LBFGS(scan_atoms, logfile=None)
            opt.run(fmax=fmax, steps=steps)
            energies[i] = scan_atoms.get_potential_energy()
            logger.debug(
                "  d=%.3f A -> E=%.6f eV (converged=%s)",
                d,
                energies[i],
                opt.converged(),
            )
        except Exception as e:
            logger.warning("  d=%.3f A -> optimization failed: %s", d, e)
            energies[i] = np.inf

    best_idx = int(np.argmin(energies))
    best_distance = float(distances[best_idx])
    logger.info(
        "Equilibrium bond length for %d-%d: %.3f A (E=%.6f eV)",
        idx1,
        idx2,
        best_distance,
        energies[best_idx],
    )
    return best_distance, energies


def equilibrium_scan(
    atoms: Atoms,
    forming_bond: Tuple[int, int],
    breaking_bond: Tuple[int, int],
    calc_fn: Callable,
    forming_distances: Optional[Sequence[float]] = None,
    breaking_distances: Optional[Sequence[float]] = None,
    fmax: float = 0.05,
    steps: int = 200,
) -> Tuple[Atoms, Atoms]:
    """Generate reactant and product endpoints via 1D equilibrium scans.

    For the forming bond: scans short distances to find the equilibrium.
    For the breaking bond: scans toward dissociation to find the plateau.

    Parameters
    ----------
    atoms : Atoms
        Transition state guess geometry.
    forming_bond : tuple of (int, int)
        Atom indices for the bond being formed.
    breaking_bond : tuple of (int, int)
        Atom indices for the bond being broken.
    calc_fn : callable
        Factory function returning a fresh ASE calculator.
    forming_distances : sequence of float, optional
        Distances to scan for the forming bond.
    breaking_distances : sequence of float, optional
        Distances to scan for the breaking bond. Defaults to scanning
        from current distance to 4.0 A.
    fmax : float
        Force convergence criterion.
    steps : int
        Maximum optimization steps per scan point.

    Returns
    -------
    tuple of (Atoms, Atoms)
        (reactant, product) endpoint geometries.
    """
    if breaking_distances is None:
        current_d = atoms.get_distance(breaking_bond[0], breaking_bond[1])
        breaking_distances = np.linspace(current_d, 4.0, 8)

    # Find equilibrium for forming bond (product has this bond short)
    eq_forming, _ = get_equilibrium_bond_length(
        atoms, forming_bond[0], forming_bond[1], calc_fn,
        distances=forming_distances, fmax=fmax, steps=steps,
    )

    # Find dissociation plateau for breaking bond
    _, breaking_energies = get_equilibrium_bond_length(
        atoms, breaking_bond[0], breaking_bond[1], calc_fn,
        distances=breaking_distances, fmax=fmax, steps=steps,
    )
    # Dissociation limit: find where energy plateaus (smallest gradient)
    if len(breaking_distances) > 2:
        gradients = np.abs(np.gradient(breaking_energies))
        # Exclude points with inf energy
        valid = np.isfinite(breaking_energies)
        if valid.sum() > 1:
            gradients[~valid] = np.inf
            plateau_idx = int(np.argmin(gradients[1:]) + 1)
            eq_breaking = float(breaking_distances[plateau_idx])
        else:
            eq_breaking = 4.0
    else:
        eq_breaking = 4.0

    logger.info(
        "Equilibrium scan targets: forming=%.3f A, breaking=%.3f A",
        eq_forming,
        eq_breaking,
    )

    # Generate product: forming bond at equilibrium, breaking bond stretched
    product = constrained_endpoint(
        atoms,
        bond_pairs=[forming_bond, breaking_bond],
        target_distances=[eq_forming, eq_breaking],
        calc_fn=calc_fn,
        fmax=fmax,
        steps=steps,
    )

    # Generate reactant: forming bond stretched, breaking bond at equilibrium
    eq_breaking_short, _ = get_equilibrium_bond_length(
        atoms, breaking_bond[0], breaking_bond[1], calc_fn,
        distances=forming_distances, fmax=fmax, steps=steps,
    )
    reactant = constrained_endpoint(
        atoms,
        bond_pairs=[forming_bond, breaking_bond],
        target_distances=[eq_breaking, eq_breaking_short],
        calc_fn=calc_fn,
        fmax=fmax,
        steps=steps,
    )

    return reactant, product


# ---------------------------------------------------------------------------
# Method 2: constrained_scan
# ---------------------------------------------------------------------------


def constrained_endpoint(
    atoms: Atoms,
    bond_pairs: List[Tuple[int, int]],
    target_distances: List[float],
    calc_fn: Callable,
    fmax: float = 0.05,
    steps: int = 300,
) -> Atoms:
    """Generate an endpoint by constrained optimization at fixed bond distances.

    Fixes specified bonds at target distances and minimizes all remaining
    degrees of freedom using LBFGS.

    Parameters
    ----------
    atoms : Atoms
        Starting geometry (will not be modified).
    bond_pairs : list of tuple of (int, int)
        Atom index pairs to constrain.
    target_distances : list of float
        Target distances (Angstrom) for each bond pair.
    calc_fn : callable
        Factory function returning a fresh ASE calculator.
    fmax : float
        Force convergence criterion (eV/A).
    steps : int
        Maximum optimization steps.

    Returns
    -------
    Atoms
        Optimized endpoint geometry.

    Raises
    ------
    RuntimeError
        If optimization fails to converge and produces unreasonable geometry.
    """
    if len(bond_pairs) != len(target_distances):
        raise ValueError(
            f"Number of bond pairs ({len(bond_pairs)}) must match "
            f"number of target distances ({len(target_distances)})."
        )

    endpoint = atoms.copy()
    _attach_calculator(endpoint, calc_fn)

    # Set each bond to its target distance
    for (idx1, idx2), target_d in zip(bond_pairs, target_distances):
        _set_bond_distance(endpoint, idx1, idx2, target_d)
        logger.info(
            "Constraining bond %d-%d at %.3f A", idx1, idx2, target_d
        )

    # Apply bond length constraints
    constraints = [FixBondLength(i, j) for (i, j) in bond_pairs]
    endpoint.set_constraint(constraints)

    # Run constrained optimization
    try:
        opt = LBFGS(endpoint, logfile=None)
        converged = opt.run(fmax=fmax, steps=steps)
        if not converged:
            logger.warning(
                "Constrained optimization did not converge within %d steps "
                "(final fmax may exceed %.4f eV/A).",
                steps,
                fmax,
            )
    except Exception as e:
        logger.error("Constrained optimization failed: %s", e)
        raise RuntimeError(f"Constrained endpoint optimization failed: {e}")

    # Remove constraints from the returned atoms
    endpoint.set_constraint()
    logger.info(
        "Constrained endpoint generated. Energy = %.6f eV",
        endpoint.get_potential_energy(),
    )
    return endpoint


# ---------------------------------------------------------------------------
# Method 3: incremental_stretch
# ---------------------------------------------------------------------------


def incremental_stretch(
    atoms: Atoms,
    idx1: int,
    idx2: int,
    target_distance: float,
    calc_fn: Callable,
    n_steps: int = 10,
    fmax: float = 0.05,
    opt_steps: int = 200,
) -> Tuple[Atoms, np.ndarray]:
    """Incrementally stretch/compress a bond, minimizing at each step.

    Traces the minimum energy path along the bond coordinate by moving
    in small increments and relaxing all other degrees of freedom at each
    step. The MLFF naturally finds the product/reactant geometry.

    Parameters
    ----------
    atoms : Atoms
        Starting geometry (will not be modified).
    idx1 : int
        Index of first atom in the bond.
    idx2 : int
        Index of second atom in the bond.
    target_distance : float
        Final target bond distance (Angstrom).
    calc_fn : callable
        Factory function returning a fresh ASE calculator.
    n_steps : int
        Number of intermediate steps between current and target distance.
    fmax : float
        Force convergence criterion per step.
    opt_steps : int
        Maximum optimization steps per intermediate point.

    Returns
    -------
    tuple of (Atoms, np.ndarray)
        Final optimized geometry and array of (distance, energy) pairs
        for the scan profile.
    """
    current_atoms = atoms.copy()
    current_distance = current_atoms.get_distance(idx1, idx2)
    step_size = (target_distance - current_distance) / n_steps

    logger.info(
        "Incremental stretch: bond %d-%d from %.3f to %.3f A in %d steps "
        "(step size = %.3f A)",
        idx1,
        idx2,
        current_distance,
        target_distance,
        n_steps,
        step_size,
    )

    profile = []

    for step in range(1, n_steps + 1):
        intermediate_d = current_distance + step * step_size
        _set_bond_distance(current_atoms, idx1, idx2, intermediate_d)
        _attach_calculator(current_atoms, calc_fn)

        # Constrain the bond at this intermediate distance
        current_atoms.set_constraint(FixBondLength(idx1, idx2))

        try:
            opt = LBFGS(current_atoms, logfile=None)
            opt.run(fmax=fmax, steps=opt_steps)
            energy = current_atoms.get_potential_energy()
            logger.debug(
                "  Step %d/%d: d=%.3f A, E=%.6f eV, converged=%s",
                step,
                n_steps,
                intermediate_d,
                energy,
                opt.converged(),
            )
        except Exception as e:
            logger.warning(
                "  Step %d/%d: optimization failed at d=%.3f A: %s",
                step,
                n_steps,
                intermediate_d,
                e,
            )
            energy = np.nan

        profile.append((intermediate_d, energy))

        # Remove constraint for next iteration (positions are kept)
        current_atoms.set_constraint()

    # Final unconstrained relaxation at the target distance
    _attach_calculator(current_atoms, calc_fn)
    try:
        opt = LBFGS(current_atoms, logfile=None)
        opt.run(fmax=fmax, steps=opt_steps)
        final_energy = current_atoms.get_potential_energy()
        logger.info(
            "Final unconstrained relaxation: E=%.6f eV, d(final)=%.3f A",
            final_energy,
            current_atoms.get_distance(idx1, idx2),
        )
    except Exception as e:
        logger.warning("Final relaxation failed: %s", e)

    profile_array = np.array(profile)
    return current_atoms, profile_array


# ---------------------------------------------------------------------------
# Method 4: spring_then_relax
# ---------------------------------------------------------------------------


def spring_then_relax(
    atoms: Atoms,
    spring_defs: List[Dict],
    calc_fn: Callable,
    spring_k: float = 3.0,
    fmax_spring: float = 0.1,
    fmax_relax: float = 0.03,
    spring_steps: int = 300,
    relax_steps: int = 500,
) -> Atoms:
    """Spring-drive then fully relax to the natural MLFF geometry.

    Uses algorithmic target distances derived from covalent radii:
      - Formed bond target: sum of covalent radii
      - Broken bond target: 2.5 * sum of covalent radii

    After the spring-driven phase, performs a longer unconstrained relaxation
    so the MLFF finds the natural endpoint geometry.

    Parameters
    ----------
    atoms : Atoms
        Starting geometry (will not be modified).
    spring_defs : list of dict
        Each dict must contain:
            - 'idx1': int, first atom index
            - 'idx2': int, second atom index
            - 'target': float, target distance (Angstrom)
            - 'type': str, 'attractive' or 'repulsive'
    calc_fn : callable
        Factory function returning a fresh ASE calculator.
    spring_k : float
        Spring constant (eV/A^2).
    fmax_spring : float
        Force convergence for spring-driven phase.
    fmax_relax : float
        Force convergence for final unconstrained relaxation.
    spring_steps : int
        Maximum steps for spring-driven phase.
    relax_steps : int
        Maximum steps for unconstrained relaxation.

    Returns
    -------
    Atoms
        Final relaxed endpoint geometry.
    """
    driven = atoms.copy()
    _attach_calculator(driven, calc_fn)

    logger.info("Spring-driven phase (k=%.2f eV/A^2):", spring_k)
    for sd in spring_defs:
        logger.info(
            "  Bond %d-%d: target=%.3f A (%s)",
            sd["idx1"],
            sd["idx2"],
            sd["target"],
            sd["type"],
        )

    # Spring-driven phase: iteratively nudge bonds toward targets
    for iteration in range(spring_steps):
        forces_on_target = True
        for sd in spring_defs:
            idx1, idx2 = sd["idx1"], sd["idx2"]
            target = sd["target"]
            current_d = driven.get_distance(idx1, idx2)
            displacement = target - current_d

            if abs(displacement) < 0.01:
                continue

            forces_on_target = False

            # Compute spring force magnitude
            force_mag = spring_k * displacement

            # Direction vector from idx1 to idx2
            pos = driven.get_positions()
            vec = pos[idx2] - pos[idx1]
            dist = np.linalg.norm(vec)
            if dist < 1e-10:
                continue
            direction = vec / dist

            # Move atoms symmetrically
            move = direction * force_mag * 0.05  # damped step
            pos[idx1] -= move * 0.5
            pos[idx2] += move * 0.5
            driven.set_positions(pos)

        if forces_on_target:
            logger.info(
                "  Spring driving converged at iteration %d", iteration
            )
            break

        # Periodically relax with fixed target bonds
        if (iteration + 1) % 20 == 0:
            constraints = [
                FixBondLength(sd["idx1"], sd["idx2"]) for sd in spring_defs
            ]
            driven.set_constraint(constraints)
            _attach_calculator(driven, calc_fn)
            try:
                opt = LBFGS(driven, logfile=None)
                opt.run(fmax=fmax_spring, steps=50)
            except Exception as e:
                logger.debug("  Intermediate relaxation issue: %s", e)
            driven.set_constraint()

    spring_energy = driven.get_potential_energy()
    logger.info("Spring-driven phase complete. E=%.6f eV", spring_energy)

    # Unconstrained relaxation phase
    logger.info(
        "Unconstrained relaxation (fmax=%.4f, max_steps=%d)...",
        fmax_relax,
        relax_steps,
    )
    _attach_calculator(driven, calc_fn)
    driven.set_constraint()  # Remove all constraints

    try:
        opt = LBFGS(driven, logfile=None)
        converged = opt.run(fmax=fmax_relax, steps=relax_steps)
        final_energy = driven.get_potential_energy()
        logger.info(
            "Unconstrained relaxation complete. E=%.6f eV, converged=%s",
            final_energy,
            converged,
        )
    except Exception as e:
        logger.error("Unconstrained relaxation failed: %s", e)
        raise RuntimeError(f"Spring-then-relax final relaxation failed: {e}")

    # Log final bond distances for verification
    for sd in spring_defs:
        final_d = driven.get_distance(sd["idx1"], sd["idx2"])
        logger.info(
            "  Final bond %d-%d: %.3f A (target was %.3f A, diff=%.3f A)",
            sd["idx1"],
            sd["idx2"],
            final_d,
            sd["target"],
            abs(final_d - sd["target"]),
        )

    return driven


# ---------------------------------------------------------------------------
# Internal geometry manipulation
# ---------------------------------------------------------------------------


def _set_bond_distance(
    atoms: Atoms, idx1: int, idx2: int, target: float
) -> None:
    """Set the distance between two atoms by moving them symmetrically.

    Both atoms are displaced equally along the bond axis so the midpoint
    is preserved.

    Parameters
    ----------
    atoms : Atoms
        Atoms object to modify in place.
    idx1 : int
        Index of first atom.
    idx2 : int
        Index of second atom.
    target : float
        Desired distance in Angstrom.
    """
    pos = atoms.get_positions()
    vec = pos[idx2] - pos[idx1]
    current_dist = np.linalg.norm(vec)

    if current_dist < 1e-10:
        logger.warning(
            "Atoms %d and %d are nearly coincident; cannot set bond distance.",
            idx1,
            idx2,
        )
        return

    direction = vec / current_dist
    midpoint = (pos[idx1] + pos[idx2]) / 2.0

    pos[idx1] = midpoint - direction * (target / 2.0)
    pos[idx2] = midpoint + direction * (target / 2.0)
    atoms.set_positions(pos)
