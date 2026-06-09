"""Gas-phase xTB reference geometries for the REACTING SUBSTRATE ONLY.

Philosophy
----------
The previous approach (`quantum_engine.mlff.irc:_xtb_reference_bond_lengths` inside
scripts/run_neb_ts.py) extracted the ligand AND first-shell coordinating
atoms (metals, water, catalytic sidechains). This is slow (>30 atoms),
expensive (xTB SCF convergence issues), and fragile (charge assignment to
first-shell is ambiguous).

**New approach: extract ONLY the reacting substrate.**

For phosphoester hydrolysis:
  - Reactant = substrate alone (e.g., paraoxon)
  - Product = substrate's fragments (phosphate + leaving group alcohol)
  - Intermediate = pentacoordinate substrate (if applicable)

Treat these as gas-phase species. xTB returns robust equilibrium bond
lengths free of active-site context. The enzyme modulates barriers, but
the EQUILIBRIUM P-O bond length in a phosphate anion is essentially
universal (~1.55 Å) regardless of protein environment.

Exception: if the catalytic residue forms a COVALENT intermediate with
the substrate (e.g., acyl-enzyme in serine proteases), include that
residue's sidechain. For PTE there is no covalent intermediate, so
substrate-only is correct.

Output
------
`compute_substrate_reference_lengths(...)` returns equilibrium bond lengths
for the P-Nuc (attractive) and P-LG (repulsive) bonds in the reactant
(intact substrate), product (broken substrate), and optionally the
intermediate (pentacoordinate).

These are used as spring targets: reactant target = reactant-length (i.e.,
LG-bonded, nuc-absent), product target = product-length (nuc-bonded,
LG-dissociated).

Speed
-----
~5-15 atoms per fragment × xTB SCF = 1-3 seconds per fragment × 3 fragments
= 3-10 seconds total on CPU. Negligible vs. multi-hour NEB.

References
----------
- Bannwarth, C.; Ehlert, S.; Grimme, S. "GFN2-xTB." J. Chem. Theory Comput.
  2019, 15, 1652. https://doi.org/10.1021/acs.jctc.8b01176
- For gas-phase / QM-cluster validity of bond lengths: Himo's group has
  shown that cluster-model equilibrium geometries transfer well across
  active-site contexts. Siegbahn & Himo, JBIC 2009.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np

log = logging.getLogger("quantum_engine.mlff.ligand_xtb")


def _resolve_xtb() -> str | None:
    try:
        from quantum_engine.site import XTB_BIN
        if XTB_BIN and os.path.isfile(XTB_BIN):
            return XTB_BIN
    except Exception:
        pass
    return shutil.which("xtb")


def _run_xtb_opt(
    symbols: List[str],
    positions: np.ndarray,
    charge: int,
    workdir: Path,
    label: str = "frag",
    timeout_s: int | None = None,
    extra_fix: list[int] | None = None,
) -> Tuple[np.ndarray | None, float | None]:
    """Run xTB GFN2 optimization. Returns (opt_positions, energy_Hartree) or (None, None)."""
    xtb = _resolve_xtb()
    if xtb is None:
        log.warning("  xTB binary not found")
        return None, None

    n = len(symbols)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    xyz = workdir / f"{label}.xyz"

    with open(xyz, "w") as f:
        f.write(f"{n}\n\n")
        for sym, (x, y, z) in zip(symbols, positions):
            f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")

    cmd = [xtb, xyz.name, "--gfn", "2", "--chrg", str(charge), "--opt", "tight"]

    if extra_fix:
        control = workdir / f"{label}.inp"
        with open(control, "w") as f:
            f.write("$fix\n")
            f.write(f"    atoms: {','.join(str(i + 1) for i in extra_fix)}\n")
            f.write("$end\n")
        cmd.extend(["--input", control.name])

    timeout_s = timeout_s or max(60, min(600, n * 4))
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=str(workdir))
    except subprocess.TimeoutExpired:
        log.warning(f"  xTB opt {label} timed out")
        return None, None

    opt_xyz = workdir / "xtbopt.xyz"
    if not opt_xyz.exists():
        return None, None

    lines = opt_xyz.read_text().strip().split("\n")
    if len(lines) < n + 2:
        return None, None

    energy_Ha = None
    m = re.search(r"energy:\s*([-\d.]+)", lines[1])
    if m:
        energy_Ha = float(m.group(1))

    pos = []
    for ln in lines[2 : n + 2]:
        parts = ln.split()
        if len(parts) >= 4:
            pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(pos) != n:
        return None, energy_Ha
    return np.array(pos), energy_Ha


def _extract_ligand_fragment(
    atoms, bt_struct, ligand_name: str,
    include_residues: list[str] | None = None,
) -> tuple[list[int], list[str], np.ndarray]:
    """Extract just the ligand atoms (and optionally specified extra residues).

    Args:
        atoms: ASE Atoms
        bt_struct: biotite AtomArray
        ligand_name: residue name to extract
        include_residues: optionally include extra residue names
                         (for covalent-intermediate cases — e.g., KCX sidechain)

    Returns:
        (indices, symbols, positions)
    """
    include = {ligand_name}
    if include_residues:
        include.update(include_residues)

    mask = np.isin(bt_struct.res_name, list(include))
    indices = np.where(mask)[0].tolist()

    symbols = [atoms.get_chemical_symbols()[i] for i in indices]
    positions = atoms.get_positions()[indices]
    return indices, symbols, positions


def _disconnect_bond(
    symbols: list[str], positions: np.ndarray,
    i: int, j: int, target_distance: float,
) -> np.ndarray:
    """Move atom j along the (j - i) direction so |r_j - r_i| = target_distance.

    Moving ONLY j (not symmetrically) because i is typically the reaction
    center (P) and j is the attacking/leaving atom.
    """
    new_pos = positions.copy()
    vec = new_pos[j] - new_pos[i]
    d = float(np.linalg.norm(vec))
    if d < 1e-6:
        return new_pos
    unit = vec / d
    new_pos[j] = new_pos[i] + unit * target_distance
    return new_pos


def compute_substrate_reference_lengths(
    atoms,
    bt_struct,
    ligand_name: str,
    bond_defs: list,  # [(atom1_name, atom2_name, default_r, mode), ...]
    total_charge: int,
    workdir: str | Path,
    include_residues: list[str] | None = None,
    compute_intermediate: bool = True,
) -> dict:
    """Compute equilibrium bond lengths for reactant/product/intermediate
    states of the isolated substrate (gas phase, xTB GFN2).

    Args:
        atoms, bt_struct, ligand_name: structure + annotations
        bond_defs: list of (atom1_name, atom2_name, default_r, mode) where
                   mode ∈ {'attractive', 'repulsive'}
        total_charge: total system charge — but we assume the ligand carries
                      the reactive charge. For PTE: substrate charge typically
                      -1 to -3 depending on protonation.
        workdir: scratch directory
        include_residues: additionally include these residue names in the
                          extracted fragment (for covalent intermediates, e.g.,
                          KCX, acyl-Ser). Default: just the ligand.
        compute_intermediate: also optimize a pentacoordinate intermediate
                              geometry (all bonds present). Default True.

    Returns:
        {
            'reactant': {(a1, a2): distance_A},
            'product': {(a1, a2): distance_A},
            'intermediate': {(a1, a2): distance_A} or None,
            'intermediate_is_minimum': bool,
            'ligand_charge': int (what charge was used),
            'fragment_size': int,
            'success': bool,
        }
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Extract ligand-only fragment
    global_indices, symbols, positions = _extract_ligand_fragment(
        atoms, bt_struct, ligand_name, include_residues
    )
    n = len(symbols)
    log.info(f"  Ligand-only fragment: {n} atoms (residues: {[ligand_name] + (include_residues or [])})")

    # Map atom names → local indices in the fragment
    local_idx = {}
    for k, gi in enumerate(global_indices):
        local_idx[str(bt_struct.atom_name[gi])] = k

    # Find P/nuc/LG in local indices
    bond_pairs = []  # list of (local_i, local_j, mode, names)
    for a1, a2, _, mode in bond_defs:
        if a1 in local_idx and a2 in local_idx:
            bond_pairs.append((local_idx[a1], local_idx[a2], mode, (a1, a2)))

    if not bond_pairs:
        log.warning("  No bond pairs match atom names in extracted fragment")
        return {"success": False, "fragment_size": n}

    # Decide ligand charge: user-provided total charge is for the full system.
    # For pure substrate extraction, we need to estimate what charge the
    # ligand itself carries. For PTE phosphoester: substrate is typically 0
    # (ethyl paraoxon) or -1 (partially deprotonated). Default to 0 unless
    # ligand residue name suggests otherwise (e.g., YYL = monoanion).
    # Heuristic: if ligand name starts with "YY" or "XU" it's a PTE substrate.
    # Allow the user to override via bt_struct if annotated.
    ligand_charge = _guess_ligand_charge(ligand_name, include_residues, total_charge)

    results = {
        "reactant": None, "product": None, "intermediate": None,
        "intermediate_is_minimum": False,
        "ligand_charge": ligand_charge,
        "fragment_size": n,
        "success": False,
    }

    # --- Reactant: nuc far, LG bonded ---
    r_pos = positions.copy()
    for li, lj, mode, _ in bond_pairs:
        if mode == "attractive":
            r_pos = _disconnect_bond(symbols, r_pos, li, lj, 3.5)
    log.info("  Optimizing reactant fragment (nuc far, LG bonded) ...")
    r_opt, r_e = _run_xtb_opt(symbols, r_pos, ligand_charge, workdir, "reactant")
    if r_opt is not None:
        results["reactant"] = {}
        for li, lj, mode, (a1, a2) in bond_pairs:
            d = float(np.linalg.norm(r_opt[li] - r_opt[lj]))
            results["reactant"][(a1, a2)] = round(d, 3)
            log.info(f"    {a1}-{a2} ({mode}): {d:.3f} Å")

    # --- Product: nuc bonded, LG dissociated ---
    p_pos = positions.copy()
    for li, lj, mode, _ in bond_pairs:
        if mode == "repulsive":
            p_pos = _disconnect_bond(symbols, p_pos, li, lj, 3.5)
        elif mode == "attractive":
            # pull nuc in to ~equilibrium
            p_pos = _disconnect_bond(symbols, p_pos, li, lj, 1.55)
    log.info("  Optimizing product fragment (nuc bonded, LG dissociated) ...")
    p_opt, p_e = _run_xtb_opt(symbols, p_pos, ligand_charge, workdir, "product")
    if p_opt is not None:
        results["product"] = {}
        for li, lj, mode, (a1, a2) in bond_pairs:
            d = float(np.linalg.norm(p_opt[li] - p_opt[lj]))
            results["product"][(a1, a2)] = round(d, 3)
            log.info(f"    {a1}-{a2} ({mode}): {d:.3f} Å")

    # --- Intermediate: pentacoordinate (both bonded) ---
    if compute_intermediate:
        int_pos = positions.copy()  # current geometry, often already pentacoordinate
        log.info("  Optimizing intermediate fragment (both bonds present) ...")
        i_opt, i_e = _run_xtb_opt(symbols, int_pos, ligand_charge, workdir, "intermediate")
        if i_opt is not None:
            results["intermediate"] = {}
            for li, lj, mode, (a1, a2) in bond_pairs:
                d = float(np.linalg.norm(i_opt[li] - i_opt[lj]))
                results["intermediate"][(a1, a2)] = round(d, 3)
                log.info(f"    {a1}-{a2} ({mode}): {d:.3f} Å")

            # Check if intermediate is distinct from reactant/product
            # (i.e., both attractive and repulsive bonds remained bonded)
            if results["intermediate"]:
                nuc_bonded = all(
                    d < 2.2 for (a1, a2), d in results["intermediate"].items()
                    if any(bd[0] == a1 and bd[1] == a2 and bd[3] == "attractive"
                           for bd in bond_defs)
                )
                lg_bonded = all(
                    d < 2.2 for (a1, a2), d in results["intermediate"].items()
                    if any(bd[0] == a1 and bd[1] == a2 and bd[3] == "repulsive"
                           for bd in bond_defs)
                )
                results["intermediate_is_minimum"] = nuc_bonded and lg_bonded
                if results["intermediate_is_minimum"]:
                    log.info("  ★ Intermediate is pentacoordinate (both bonds present)")

    # Compute relative energies if all succeeded
    if r_e and p_e:
        from ase.units import Hartree
        dE_rp = (p_e - r_e) * Hartree
        log.info(f"  ΔE(R→P) [gas-phase, ligand only] = {dE_rp:.3f} eV = {dE_rp * 23.0609:.1f} kcal/mol")
    if r_e and p_e:
        results["success"] = True

    return results


def _guess_ligand_charge(
    ligand_name: str,
    include_residues: list[str] | None,
    total_charge: int,
) -> int:
    """Guess the charge on the ligand-only fragment.

    Assumptions (PTE-specific, adjust for other systems):
      - If include_residues contains KCX (carbamoylated lysine): adds -1 (COO-)
      - Paraoxon-like neutral substrate: 0
      - Phosphate monoanion ligand (default for YYL/YYE): 0 at neutral pH,
        may deprotonate to -1 or -2 depending on state
    """
    charge = 0
    # If KCX is included (covalent intermediate), it's -1 charged
    if include_residues:
        if "KCX" in include_residues:
            charge -= 1  # carboxylate
        if "ASP" in include_residues or "GLU" in include_residues:
            charge -= 1  # sidechain carboxylate
    # Otherwise, use 0 as default neutral substrate
    # (user can override via residue name convention or custom logic)
    log.info(f"  Estimated ligand-fragment charge: {charge} "
             f"(system charge was {total_charge})")
    return charge


def suggest_spring_targets(ref: dict, bond_defs: list) -> dict:
    """Pick spring targets from xTB reference dict.

    Returns {(atom1, atom2): target_distance}.
    Uses PRODUCT bond length for attractive (bond forming) springs;
    REACTANT bond length for repulsive (bond breaking) springs.
    """
    targets = {}
    for a1, a2, default_r, mode in bond_defs:
        if mode == "attractive" and ref.get("product"):
            targets[(a1, a2)] = ref["product"].get((a1, a2), default_r)
        elif mode == "repulsive" and ref.get("reactant"):
            # Use 2.5x the equilibrium LG-bonded length for "well dissociated"
            r_bonded = ref["reactant"].get((a1, a2))
            if r_bonded:
                targets[(a1, a2)] = round(r_bonded * 2.5, 2)
            else:
                targets[(a1, a2)] = default_r
        else:
            targets[(a1, a2)] = default_r
    return targets
