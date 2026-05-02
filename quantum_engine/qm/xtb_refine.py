"""
xTB endpoint refinement: validate reactant/product geometries with a second QM method.

Why this matters
----------------
After any endpoint generation method (spring-driven, CV-driven, metadynamics),
the resulting "reactant" and "product" geometries reflect whatever potential
energy surface was used (usually MACE). Two failure modes are possible:

1. **MACE hallucination**: the geometry sits in a local minimum that is an
   artifact of MACE's approximation to the true PES — no such minimum exists
   at the true QM level.

2. **Shallow/absent product basin**: the "product" from spring driving is not
   actually a minimum. Any perturbation (MD, unconstrained opt) would cause
   it to roll back toward reactant.

xTB (GFN2-xTB in particular) is a semi-empirical QM method that:
- Runs ~1-10 seconds per geometry opt for 100-300 atom systems
- Uses actual electronic structure (not fit to training data)
- Is mechanistically independent from MACE (different error modes)

The xTB refinement step:
1. Takes a candidate endpoint from MACE
2. Runs unconstrained xTB GFN2 geometry optimization (with CA constraints preserved)
3. If xTB relaxes it to essentially the same geometry → real minimum, accept
4. If xTB relaxes it back to the reactant → **product basin doesn't exist, diagnostic!**
5. If xTB finds a different nearby minimum → update the endpoint

This is an independent QM sanity check. If xTB disagrees significantly with
MACE, that's a red flag for MACE extrapolation in the reactive region.

When to use
-----------
- As a sanity check after spring-driven endpoint generation, especially for the product
- When spring target distances produced a geometry that looks suspicious (P-O too long, etc.)
- For validation of any MACE-discovered minimum that will be reported as a result

When NOT to use
---------------
- Systems with >~400 atoms (xTB SCF becomes slow/unstable, large basis demands)
- Systems with unusual bonding (transition metal complexes beyond first-row TM)
- Final production validation (use DFT single-points instead for publication)

Limitations
-----------
- GFN2 has known issues with charge-transfer states and multi-reference character
- xTB optimization with constraints uses internal coordinates, some combinations fail
- Requires xTB binary installed at the location in `qcb.config`

References
----------
- Bannwarth, C.; Ehlert, S.; Grimme, S. "GFN2-xTB — An Accurate and Broadly Parametrized
  Self-Consistent Tight-Binding Quantum Chemical Method." J. Chem. Theory Comput. 2019,
  15, 1652. https://doi.org/10.1021/acs.jctc.8b01176
- Grimme, S. "Exploration of Chemical Compound, Conformer, and Reaction Space with Meta-
  Dynamics Simulations Based on Tight-Binding Quantum Chemical Calculations." J. Chem.
  Theory Comput. 2019, 15, 2847.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from quantum_engine.units import EV_TO_KCAL
from ase import Atoms
from ase.io import read, write

log = logging.getLogger("quantum_engine.xtb_refine")


def _resolve_xtb_binary() -> str | None:
    """Find xTB binary from quantum_engine.config, PATH, or None."""
    # Prefer qcb.config (cluster-specific)
    try:
        from quantum_engine.config import XTB_BIN as _configured
        if _configured and os.path.isfile(_configured):
            return _configured
    except Exception:
        pass
    # Fallback to PATH
    from shutil import which
    found = which("xtb")
    return found


XTB_BIN = _resolve_xtb_binary()


def _write_xtb_input(
    atoms: Atoms,
    workdir: Path,
    constraint_indices: list[int] | None = None,
) -> tuple[Path, Path]:
    """Write xyz file and constraints file for xTB.

    Returns (xyz_path, constraints_path_or_None)
    """
    xyz_path = workdir / "input.xyz"
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    with open(xyz_path, "w") as f:
        f.write(f"{len(atoms)}\n\n")
        for sym, (x, y, z) in zip(symbols, positions):
            f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")

    constraints_path = None
    if constraint_indices is not None and len(constraint_indices) > 0:
        # xTB uses 1-indexed atom IDs; constraint type: fix position
        constraints_path = workdir / "xcontrol.inp"
        with open(constraints_path, "w") as f:
            f.write("$fix\n")
            # Comma-separated list, 1-indexed
            indices_1idx = ",".join(str(i + 1) for i in constraint_indices)
            f.write(f"    atoms: {indices_1idx}\n")
            f.write("$end\n")

    return xyz_path, constraints_path


def _parse_xtb_opt_output(workdir: Path, n_atoms: int) -> tuple[np.ndarray | None, float | None, int]:
    """Parse xTB optimization output.

    Returns (optimized_positions, energy_eV, n_scf_cycles)
    """
    opt_file = workdir / "xtbopt.xyz"
    if not opt_file.exists():
        return None, None, 0

    lines = opt_file.read_text().strip().split("\n")
    if len(lines) < n_atoms + 2:
        return None, None, 0

    # Parse energy from comment line: "energy: -X.XXXXX gnorm: Y.YYY xtb: Z.Z.Z"
    from ase.units import Hartree
    energy_eV: float | None = None
    m = re.search(r"energy:\s*([-\d.]+)", lines[1])
    if m:
        energy_Ha = float(m.group(1))
        energy_eV = energy_Ha * Hartree  # ase.units.Hartree = 27.211386...

    # Parse positions
    positions = []
    for line in lines[2:n_atoms + 2]:
        parts = line.split()
        if len(parts) >= 4:
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if len(positions) != n_atoms:
        return None, energy_eV, 0

    # Count SCF cycles from log
    n_scf = 0
    opt_log = workdir / "xtbopt.log"
    if opt_log.exists():
        n_scf = len([l for l in opt_log.read_text().split("\n") if "SCF" in l])

    return np.array(positions), energy_eV, n_scf


def xtb_optimize(
    atoms: Atoms,
    total_charge: int,
    outdir: str | Path,
    constraint_indices: list[int] | None = None,
    gfn: int = 2,
    max_scf_cycles: int = 500,
    convergence: str = "normal",
    timeout_s: int | None = None,
) -> tuple[Atoms | None, float | None]:
    """Run xTB GFN geometry optimization with optional atom constraints.

    Args:
        atoms: Starting geometry
        total_charge: Total system charge
        outdir: Working directory (will be cleaned up after)
        constraint_indices: List of atom indices (0-based) to hold fixed
        gfn: GFN variant (0, 1, 2). GFN2 is default and most accurate for organics.
        max_scf_cycles: SCF iteration cap
        convergence: "loose", "normal", "tight", "vtight"
        timeout_s: Abort after this many seconds (None = auto based on size)

    Returns:
        (optimized_atoms_or_None, energy_eV_or_None). None if xTB failed.
    """
    if XTB_BIN is None or not os.path.isfile(XTB_BIN):
        log.warning(f"  xTB binary not found (tried qcb.config and PATH)")
        return None, None

    n_atoms = len(atoms)
    if n_atoms > 500:
        log.warning(f"  xTB with {n_atoms} atoms may be slow/unstable")

    if timeout_s is None:
        timeout_s = min(3600, max(120, n_atoms * 5))

    outdir = Path(outdir)
    workdir = outdir / ".xtb_refine"
    workdir.mkdir(parents=True, exist_ok=True)

    xyz_path, constraints_path = _write_xtb_input(atoms, workdir, constraint_indices)

    cmd = [
        XTB_BIN, xyz_path.name,
        "--gfn", str(gfn),
        "--chrg", str(total_charge),
        "--opt", convergence,
    ]
    if constraints_path is not None:
        cmd.extend(["--input", constraints_path.name])

    log.info(f"  xTB GFN{gfn} opt: {n_atoms} atoms, {len(constraint_indices or [])} fixed, "
             f"charge={total_charge}, timeout={timeout_s}s")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_s, cwd=str(workdir),
        )
    except subprocess.TimeoutExpired:
        log.warning(f"  xTB opt timed out ({timeout_s}s)")
        return None, None

    # Parse output
    positions, energy_eV, _ = _parse_xtb_opt_output(workdir, n_atoms)

    if positions is None:
        log.warning(f"  xTB opt did not produce xtbopt.xyz — preserved workdir at {workdir} for diagnosis")
        tail = (result.stderr or result.stdout)[-800:]
        log.debug(f"  xTB output tail:\n{tail}")
        # Preserve workdir on failure for post-mortem
        return None, None

    opt_atoms = atoms.copy()
    opt_atoms.set_positions(positions)

    rmsd = float(np.sqrt(np.mean((atoms.get_positions() - positions) ** 2)))
    if energy_eV is not None:
        log.info(f"  xTB opt done: E = {energy_eV * EV_TO_KCAL:.2f} kcal/mol, "
                 f"RMSD from input = {rmsd:.3f} Å")
    else:
        log.info(f"  xTB opt done: RMSD from input = {rmsd:.3f} Å (energy parse failed)")

    shutil.rmtree(workdir, ignore_errors=True)
    return opt_atoms, energy_eV


def refine_endpoint(
    atoms: Atoms,
    total_charge: int,
    outdir: str | Path,
    label: str = "endpoint",
    constraint=None,
    expected_bonds: dict | None = None,
) -> dict:
    """Refine a single endpoint with xTB and characterize any geometry changes.

    Args:
        atoms: Candidate endpoint (e.g., from spring-driven generation)
        total_charge: System charge
        outdir: Working directory
        label: Tag for logging ("reactant", "product", etc.)
        constraint: ASE FixAtoms constraint to preserve (only indices are used)
        expected_bonds: Dict {(atom1_idx, atom2_idx): ('bonded'|'broken')} to
                       check after refinement

    Returns:
        dict with:
          - 'atoms': refined atoms (or original if xTB failed)
          - 'energy_eV': xTB final energy (or None)
          - 'rmsd_from_input': RMSD of refined vs input
          - 'bond_checks': {pair: (distance_A, expected, passed)}
          - 'verdict': "accepted" | "rejected_collapsed" | "rejected_xtb_failed"
          - 'diagnostics': string with human-readable summary
    """
    log.info(f"  xTB refinement: {label}")

    constraint_indices = None
    if constraint is not None and hasattr(constraint, "index"):
        idx = constraint.index
        if hasattr(idx, "tolist"):
            idx = idx.tolist()
        constraint_indices = list(idx)

    refined, energy = xtb_optimize(atoms, total_charge, outdir, constraint_indices)

    result = {
        "atoms": atoms if refined is None else refined,
        "energy_eV": energy,
        "rmsd_from_input": None,
        "bond_checks": {},
        "verdict": "rejected_xtb_failed" if refined is None else "accepted",
        "diagnostics": "",
    }

    if refined is None:
        result["diagnostics"] = "xTB optimization failed or timed out"
        return result

    rmsd = float(np.sqrt(np.mean((atoms.get_positions() - refined.get_positions()) ** 2)))
    result["rmsd_from_input"] = rmsd

    # Check expected bonds
    diag_lines = [f"xTB-refined {label}: E = {energy * EV_TO_KCAL:.2f} kcal/mol, RMSD = {rmsd:.3f} Å"]

    if expected_bonds:
        all_passed = True
        for (i, j), expected in expected_bonds.items():
            d = refined.get_distance(i, j)
            if expected == "bonded":
                passed = d < 2.0
            elif expected == "broken":
                passed = d > 2.5
            else:
                passed = True  # "any"
            result["bond_checks"][(i, j)] = (d, expected, passed)
            status = "✓" if passed else "✗"
            diag_lines.append(f"  {status} bond ({i},{j}): {d:.2f} Å (expected {expected})")
            if not passed:
                all_passed = False

        if not all_passed:
            # Determine if this is a "collapsed" endpoint (bonds not as expected)
            collapsed_bonds = [
                (i, j, d) for (i, j), (d, exp, ok) in result["bond_checks"].items()
                if not ok and exp == "bonded"
            ]
            broken_bonds = [
                (i, j, d) for (i, j), (d, exp, ok) in result["bond_checks"].items()
                if not ok and exp == "broken"
            ]
            if collapsed_bonds or broken_bonds:
                result["verdict"] = "rejected_collapsed"
                diag_lines.append(
                    f"  → {label} did not converge to expected geometry; "
                    f"{len(collapsed_bonds)} bonds missing, {len(broken_bonds)} unbroken"
                )

    result["diagnostics"] = "\n".join(diag_lines)
    log.info(result["diagnostics"])

    return result


def validate_endpoint_pair(
    reactant: Atoms,
    product: Atoms,
    total_charge: int,
    outdir: str | Path,
    constraint=None,
    p_idx: int | None = None,
    nuc_idx: int | None = None,
    lg_idx: int | None = None,
) -> dict:
    """Run xTB on both endpoints, flag inconsistencies.

    Particularly useful to detect the "product collapsed to reactant" failure mode.

    Args:
        reactant, product: Candidate endpoints
        total_charge: System charge
        outdir: Working directory
        constraint: FixAtoms constraint
        p_idx, nuc_idx, lg_idx: (optional) atom indices for P, nucleophile, leaving group
                                — enables bond-checking diagnostics

    Returns:
        dict with 'reactant' and 'product' refinement results, plus 'inconsistent' flag.
    """
    outdir = Path(outdir)

    # Build expected bonds
    r_bonds = {}
    p_bonds = {}
    if p_idx is not None and nuc_idx is not None:
        r_bonds[(p_idx, nuc_idx)] = "broken"
        p_bonds[(p_idx, nuc_idx)] = "bonded"
    if p_idx is not None and lg_idx is not None:
        r_bonds[(p_idx, lg_idx)] = "bonded"
        p_bonds[(p_idx, lg_idx)] = "broken"

    r_result = refine_endpoint(
        reactant, total_charge, outdir / "r",
        label="reactant", constraint=constraint,
        expected_bonds=r_bonds,
    )
    p_result = refine_endpoint(
        product, total_charge, outdir / "p",
        label="product", constraint=constraint,
        expected_bonds=p_bonds,
    )

    # Check if product collapsed to reactant (RMSD should be non-trivial)
    inconsistent = False
    if r_result["atoms"] is not None and p_result["atoms"] is not None:
        rmsd_rp = float(np.sqrt(np.mean(
            (r_result["atoms"].get_positions() - p_result["atoms"].get_positions()) ** 2
        )))
        if rmsd_rp < 0.3:
            log.warning(
                f"  *** Reactant and product are nearly identical after xTB refinement "
                f"(RMSD = {rmsd_rp:.3f} Å). Product basin likely does not exist at this level. ***"
            )
            inconsistent = True

    if p_result["verdict"] == "rejected_collapsed":
        inconsistent = True

    return {
        "reactant": r_result,
        "product": p_result,
        "inconsistent": inconsistent,
    }
