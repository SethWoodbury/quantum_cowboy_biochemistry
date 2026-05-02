from quantum_engine.units import EV_TO_KCAL
#!/usr/bin/env python
"""
DEPRECATED ENTRY POINT — use `scripts/qcb ts` or `python -m qcb.cli ts` instead.
================================================================================

This script is the full TS-search pipeline (legacy/irc/cv-spring/mtd strategies
+ NEB + Sella + frequencies). It is still fully functional and all existing
SLURM scripts continue to work. However, new users should use the modular qcb
CLI which provides the same pipeline plus standalone operations:

    qcb ts input.pdb --strategy irc             # this script's equivalent
    qcb opt input.pdb                            # just geometry optimization
    qcb md input.pdb --time 10 --temp 300        # molecular dynamics
    qcb freq input.pdb                           # vibrational frequencies
    qcb scan input.pdb --coord bond ...          # Gaussian-style coordinate scan
    qcb saddle input.pdb                         # Sella only
    qcb irc input.pdb                            # IRC only
    qcb neb reactant.pdb product.pdb             # NEB only
    qcb mtd input.pdb --p-idx N --nuc-idx M ...  # metadynamics
    qcb sp input.pdb                             # single-point energy

See `docs/strategies.md` and `qcb --help` for details.

Full pipeline (what THIS script does):
  1. Load PDB, auto-detect ligand, extract formal charge
  2. Set up backbone constraints
  3. Endpoint generation (strategy-dependent)
  4. NEB path optimisation + CI-NEB climbing image
  5. Sella TS refinement (optional)
  6. Vibrational frequency validation (optional)

Designed for PTE theozyme active-site clusters (YYE / YYL / YYF / PT4 ligands)
but adaptable to other enzyme systems.

Usage
-----
  python run_neb_ts.py path/to/input.pdb                    # auto-detect model
  python run_neb_ts.py path/to/input.pdb --model mace-mp    # specific model
  python run_neb_ts.py path/to/input.pdb --strategy irc     # IRC-from-TS strategy

Authors: synthesised from lschaaf / gbg222 / seth pipelines, March 2026.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

# Ensure project root is on sys.path so qcb package is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np

# ──────────────────────────────────────────────────────────────
# ASE
# ──────────────────────────────────────────────────────────────
from ase import Atoms, units
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.mep.neb import NEB, NEBOptimizer, NEBTools
from ase.optimize import LBFGS
from matplotlib import pyplot as plt

# ──────────────────────────────────────────────────────────────
# Sella – saddle-point optimiser
# ──────────────────────────────────────────────────────────────
from sella import Sella

# ──────────────────────────────────────────────────────────────
# Biotite – structural biology I/O (PDB parsing with annotations)
# ──────────────────────────────────────────────────────────────
import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io

# ──────────────────────────────────────────────────────────────
# MACE calculator
# ──────────────────────────────────────────────────────────────
from mace.calculators import MACECalculator

# ──────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("neb-ts")

# Energy conversion: ASE uses eV internally, enzymology uses kcal/mol


# ══════════════════════════════════════════════════════════════
#  BUNDLED UTILITIES  (portable – no external enzts dependency)
# ══════════════════════════════════════════════════════════════


class SpringConstraint:
    """Harmonic spring between two atoms (repulsive / attractive / both).

    Ported from enzts.utils.preprocessing.SpringConstraint with force-cap
    support for energy-conserving clipping.
    """

    def __init__(self, a1, a2, k, rt, mode="both", fmax=None):
        assert mode in ("repulsive", "attractive", "both")
        self.a1 = a1
        self.a2 = a2
        self.k = k
        self.rt = rt
        self.mode = mode
        self.fmax = fmax

    def get_removed_dof(self, atoms):
        return 0

    def adjust_positions(self, atoms, new):
        pass

    def adjust_momenta(self, atoms, momenta):
        pass

    def adjust_forces(self, atoms, forces):
        d = atoms.positions[self.a2] - atoms.positions[self.a1]
        r = np.linalg.norm(d)
        dx = self.rt - r  # positive → atoms too close

        if (self.mode == "repulsive" and dx <= 0) or (
            self.mode == "attractive" and dx >= 0
        ):
            return

        uv = d / r
        F = self.k * dx
        if self.fmax is not None:
            F = np.clip(F, -self.fmax, self.fmax)

        fv = F * uv
        forces[self.a1] -= fv
        forces[self.a2] += fv

    def adjust_potential_energy(self, atoms):
        r = atoms.get_distance(self.a1, self.a2)
        dx = self.rt - r

        if (self.mode == "repulsive" and dx <= 0) or (
            self.mode == "attractive" and dx >= 0
        ):
            return 0.0

        raw = self.k * abs(dx)
        if self.fmax is None or raw <= self.fmax:
            return 0.5 * self.k * dx**2

        cap_d = self.fmax / self.k
        excess = abs(dx) - cap_d
        return 0.5 * self.k * cap_d**2 + self.fmax * excess

    def todict(self):
        return {
            "name": "SpringConstraint",
            "kwargs": {
                "a1": self.a1,
                "a2": self.a2,
                "k": self.k,
                "rt": self.rt,
                "mode": self.mode,
                "fmax": self.fmax,
            },
        }

    def __repr__(self):
        return (
            f"SpringConstraint({self.a1}, {self.a2}, k={self.k}, "
            f"rt={self.rt}, mode={self.mode!r})"
        )


def biotite_to_ase(st):
    """Convert a biotite AtomArray → ASE Atoms, preserving annotations."""
    symbols = [e.capitalize() for e in st.element]
    at = Atoms(symbols=symbols, positions=st.coord)
    at.arrays.update(st._annot)
    return at


def ase_to_pdb_string(atoms, template_st):
    """Convert ASE Atoms back to PDB format using the original biotite template.

    Takes positions from the ASE atoms and atom labeling/residue info from
    the original biotite AtomArray so the output PDB matches the input format.
    """
    out = template_st.copy()
    out.coord = atoms.get_positions().astype(np.float32)
    pdb_file = pdb_io.PDBFile()
    pdb_file.set_structure(out)
    return str(pdb_file)


def write_result_pdb(atoms, template_st, path):
    """Write a single structure as PDB with input-matching formatting.

    Always includes REMARK QCB lines with charge, energy, and provenance.
    """
    charge = atoms.info.get("charge", 0)
    energy = None
    try:
        energy = atoms.get_potential_energy()
    except Exception:
        pass

    pdb_str = ase_to_pdb_string(atoms, template_st)
    lines = pdb_str.strip().split("\n")

    # Insert QCB REMARK lines after any existing REMARK/HEADER lines
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith(("REMARK", "HEADER", "USER")):
            insert_idx = i + 1
        elif line.startswith(("ATOM", "HETATM", "TER")):
            break

    remarks = [
        f"REMARK QCB TOTAL_CHARGE {charge:+d}",
    ]
    if energy is not None:
        remarks.append(f"REMARK QCB ENERGY_KCAL {energy * EV_TO_KCAL:.2f}")
    basename = os.path.basename(path).replace(".pdb", "")
    remarks.append(f"REMARK QCB STRUCTURE {basename}")

    for i, r in enumerate(remarks):
        lines.insert(insert_idx + i, r)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"  Wrote {path} (charge={charge:+d})")


def write_trajectory_pdb(atoms_list, template_st, path, energies=None):
    """Write multiple structures as multi-MODEL PDB (viewable in PyMOL/ChimeraX).

    Each frame becomes a MODEL/ENDMDL block. If energies are provided they
    are written as REMARK lines.
    """
    lines = []
    for i, atoms in enumerate(atoms_list):
        out = template_st.copy()
        out.coord = atoms.get_positions().astype(np.float32)
        pdb_file = pdb_io.PDBFile()
        pdb_file.set_structure(out)
        model_lines = str(pdb_file).strip().split("\n")

        lines.append(f"MODEL     {i+1:4d}")
        if energies is not None and i < len(energies):
            e_kcal = energies[i] * EV_TO_KCAL
            lines.append(f"REMARK   energy_kcal_mol {e_kcal:.2f}")
        # Keep only ATOM/HETATM/TER lines from the PDB
        for l in model_lines:
            if l.startswith(("ATOM", "HETATM", "TER")):
                lines.append(l)
        lines.append("ENDMDL")
    lines.append("END")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"  Wrote {path} ({len(atoms_list)} models)")


def extract_net_charge(path):
    """Extract net formal charge from multiple sources (priority order):

    1. PDB REMARK QCB TOTAL_CHARGE line (from protonation pipeline)
    2. Filename pattern netCHG_plus3 / netCHG_minus2 / netCHG_0
    3. Returns None if neither found (caller should error, not default to 0)
    """
    # Try PDB REMARK first
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("REMARK QCB TOTAL_CHARGE"):
                    val = line.split()[-1]
                    charge = int(val.replace("+", ""))
                    log.info(f"  Charge from PDB REMARK: {charge:+d}")
                    return charge
                if not line.startswith(("REMARK", "HEADER", "USER")):
                    break  # stop after header section
    except Exception:
        pass

    # Try filename
    name = os.path.basename(path)
    m = re.search(r"net[Cc][Hh][Gg]_?(plus|minus)?_?(\d+)", name)
    if m is not None:
        sign_str, val = m.group(1), int(m.group(2))
        charge = -val if sign_str == "minus" else val
        log.info(f"  Charge from filename: {charge:+d}")
        return charge

    return None


# ══════════════════════════════════════════════════════════════
#  BOND-BREAKING DEFINITIONS (per ligand type)
# ══════════════════════════════════════════════════════════════
#
# Each entry: (reactive_atom, partner_atom, target_distance, mode)
#   attractive → pull atoms together (new bond forming)
#   repulsive  → push atoms apart   (old bond breaking)
#
# For PTE phosphoester hydrolysis the mechanism is:
#   Nucleophilic attack on P by bridging hydroxide (O3/O1)
#   Departure of aryl leaving group (O7/O5)
#
def _xtb_optimize_fragment(symbols, positions, charge, workdir, label="fragment"):
    """Run xTB GFN2 geometry optimization on a fragment, return optimized positions.

    Returns (opt_positions, energy) or (None, None) on failure.
    """
    import shutil

    XTB_BIN = "/home/dme5188/bin/xtb/xtb-6.6.1/bin/xtb"
    if not os.path.isfile(XTB_BIN):
        return None, None

    tmpdir = Path(workdir) / f".xtb_ref_{label}"
    tmpdir.mkdir(exist_ok=True)

    n = len(symbols)
    with open(tmpdir / "frag.xyz", "w") as f:
        f.write(f"{n}\n\n")
        for sym, (x, y, z) in zip(symbols, positions):
            f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")

    timeout_s = min(600, max(60, n * 3))
    try:
        result = subprocess.run(
            [XTB_BIN, "frag.xyz", "--gfn", "2", "--chrg", str(charge), "--opt", "tight"],
            capture_output=True, text=True, timeout=timeout_s, cwd=str(tmpdir),
        )
    except subprocess.TimeoutExpired:
        log.warning(f"    xTB opt timed out for {label}")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None, None

    opt_file = tmpdir / "xtbopt.xyz"
    energy = None
    if opt_file.exists():
        lines = opt_file.read_text().strip().split("\n")
        # Parse energy from comment line
        if len(lines) >= 2:
            import re as _re
            e_match = _re.search(r"energy:\s*([-\d.]+)", lines[1])
            if e_match:
                energy = float(e_match.group(1))  # Hartree
        opt_pos = []
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4:
                opt_pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if len(opt_pos) == n:
            shutil.rmtree(tmpdir, ignore_errors=True)
            return np.array(opt_pos), energy

    shutil.rmtree(tmpdir, ignore_errors=True)
    return None, None


def _xtb_reference_bond_lengths(atoms, st, ligand_name, bond_defs, total_charge, workdir):
    """Use xTB to compute equilibrium bond lengths for reactant, product, and intermediate.

    Extracts the ligand + first-shell coordinating atoms, builds three geometries:
      - reactant: nuc far, LG bonded (input geometry with nuc pulled away)
      - product: nuc bonded, LG far (input geometry with LG pulled away)
      - intermediate: both nuc and LG bonded (pentacoordinate, if applicable)

    Optimizes each with xTB GFN2 and returns equilibrium distances.

    Returns dict: {
        'product': {(atom1, atom2): distance, ...},
        'reactant': {(atom1, atom2): distance, ...},
        'intermediate': {(atom1, atom2): distance, ...} or None,
        'intermediate_is_minimum': bool,
    }
    or None if xTB fails.
    """
    from ase.data import covalent_radii

    log.info("  Running xTB reference geometry optimizations ...")

    # Extract ligand atoms
    ligand_mask = st.res_name == ligand_name
    lig_indices = np.where(ligand_mask)[0]
    if len(lig_indices) == 0:
        return None

    # Also include first-shell atoms (bonded to ligand via covalent criteria)
    all_indices = set(lig_indices.tolist())
    for li in lig_indices:
        zi = atoms.get_atomic_numbers()[li]
        for j in range(len(atoms)):
            if j in all_indices:
                continue
            zj = atoms.get_atomic_numbers()[j]
            d = atoms.get_distance(li, j)
            r_bond = (covalent_radii[zi] + covalent_radii[zj]) * 1.3
            if d < r_bond:
                all_indices.add(j)

    frag_indices = sorted(all_indices)
    # Map global → fragment indices
    global_to_frag = {g: f for f, g in enumerate(frag_indices)}

    frag_symbols = [atoms.get_chemical_symbols()[i] for i in frag_indices]
    frag_positions = atoms.get_positions()[frag_indices]

    # Identify bond atom indices in fragment space
    bond_pairs = {}  # {(aname1, aname2): (frag_idx1, frag_idx2, mode)}
    for aname1, aname2, _, mode in bond_defs:
        idx1 = np.where((st.res_name == ligand_name) & (st.atom_name == aname1))[0]
        idx2 = np.where((st.res_name == ligand_name) & (st.atom_name == aname2))[0]
        if len(idx1) == 1 and len(idx2) == 1:
            gi1, gi2 = int(idx1[0]), int(idx2[0])
            if gi1 in global_to_frag and gi2 in global_to_frag:
                bond_pairs[(aname1, aname2)] = (
                    global_to_frag[gi1], global_to_frag[gi2], mode
                )

    if not bond_pairs:
        return None

    # Estimate fragment charge (rough: use total charge, since ligand is the charged part)
    frag_charge = total_charge

    results = {}

    # ── Intermediate: optimize current geometry as-is (pentacoordinate if applicable) ──
    log.info("    Optimizing intermediate (current geometry) ...")
    int_pos, int_e = _xtb_optimize_fragment(
        frag_symbols, frag_positions, frag_charge, workdir, "intermediate"
    )
    if int_pos is not None:
        results["intermediate"] = {}
        for (a1, a2), (fi1, fi2, mode) in bond_pairs.items():
            d = float(np.linalg.norm(int_pos[fi1] - int_pos[fi2]))
            results["intermediate"][(a1, a2)] = round(d, 3)
            log.info(f"      {a1}-{a2}: {d:.3f} A")
    else:
        results["intermediate"] = None

    # ── Product: pull LG away, push nuc close, then optimize ──
    log.info("    Building product geometry (nuc bonded, LG dissociated) ...")
    prod_pos = frag_positions.copy()
    for (a1, a2), (fi1, fi2, mode) in bond_pairs.items():
        d_vec = prod_pos[fi2] - prod_pos[fi1]
        d_cur = np.linalg.norm(d_vec)
        uv = d_vec / d_cur
        z1 = atoms.get_atomic_numbers()[frag_indices[fi1]]
        z2 = atoms.get_atomic_numbers()[frag_indices[fi2]]
        r_cov = covalent_radii[z1] + covalent_radii[z2]

        if mode == "attractive":
            # Nuc should be bonded in product — place at 0.95 * cov radii
            target = r_cov * 0.95
            if d_cur > target * 1.5:
                prod_pos[fi2] = prod_pos[fi1] + uv * target
        elif mode == "repulsive":
            # LG should be far in product — place at 3.5 A
            target = 3.5
            if d_cur < target * 0.8:
                prod_pos[fi2] = prod_pos[fi1] + uv * target

    log.info("    Optimizing product ...")
    prod_opt, prod_e = _xtb_optimize_fragment(
        frag_symbols, prod_pos, frag_charge, workdir, "product"
    )
    if prod_opt is not None:
        results["product"] = {}
        for (a1, a2), (fi1, fi2, mode) in bond_pairs.items():
            d = float(np.linalg.norm(prod_opt[fi1] - prod_opt[fi2]))
            results["product"][(a1, a2)] = round(d, 3)
            log.info(f"      {a1}-{a2}: {d:.3f} A")
    else:
        results["product"] = None

    # ── Reactant: pull nuc away, keep LG bonded ──
    log.info("    Building reactant geometry (nuc dissociated, LG bonded) ...")
    react_pos = frag_positions.copy()
    for (a1, a2), (fi1, fi2, mode) in bond_pairs.items():
        d_vec = react_pos[fi2] - react_pos[fi1]
        d_cur = np.linalg.norm(d_vec)
        uv = d_vec / d_cur
        z1 = atoms.get_atomic_numbers()[frag_indices[fi1]]
        z2 = atoms.get_atomic_numbers()[frag_indices[fi2]]
        r_cov = covalent_radii[z1] + covalent_radii[z2]

        if mode == "attractive":
            # Nuc should be far in reactant
            target = 3.5
            if d_cur < target * 0.8:
                react_pos[fi2] = react_pos[fi1] + uv * target
        elif mode == "repulsive":
            # LG should be bonded in reactant — place at 0.95 * cov radii
            target = r_cov * 0.95
            if d_cur > target * 1.5:
                react_pos[fi2] = react_pos[fi1] + uv * target

    log.info("    Optimizing reactant ...")
    react_opt, react_e = _xtb_optimize_fragment(
        frag_symbols, react_pos, frag_charge, workdir, "reactant"
    )
    if react_opt is not None:
        results["reactant"] = {}
        for (a1, a2), (fi1, fi2, mode) in bond_pairs.items():
            d = float(np.linalg.norm(react_opt[fi1] - react_opt[fi2]))
            results["reactant"][(a1, a2)] = round(d, 3)
            log.info(f"      {a1}-{a2}: {d:.3f} A")
    else:
        results["reactant"] = None

    # ── Check if intermediate is a distinct minimum ──
    int_is_min = False
    if int_e is not None and prod_e is not None and react_e is not None:
        # Intermediate is a real minimum if it's lower than both endpoints
        # and geometrically distinct (nuc AND LG are bonded)
        if results.get("intermediate"):
            nuc_bonded = all(
                results["intermediate"].get((a1, a2), 99) < 2.5
                for (a1, a2), (_, _, mode) in bond_pairs.items()
                if mode == "attractive"
            )
            lg_bonded = all(
                results["intermediate"].get((a1, a2), 99) < 2.5
                for (a1, a2), (_, _, mode) in bond_pairs.items()
                if mode == "repulsive"
            )
            int_is_min = nuc_bonded and lg_bonded
            if int_is_min:
                log.info(f"    *** Intermediate is pentacoordinate (nuc+LG both bonded) ***")
                log.info(f"    Energies (Ha): reactant={react_e:.6f}, "
                         f"intermediate={int_e:.6f}, product={prod_e:.6f}")
    results["intermediate_is_minimum"] = int_is_min

    return results


def get_smart_bond_targets(atoms, st, ligand_name, bond_defs, total_charge=0, workdir="/tmp"):
    """Determine spring target distances from xTB reference optimizations.

    Runs xTB GFN2 on isolated ligand fragments to get true equilibrium bond
    lengths for reactant, product, and (optionally) intermediate states.
    This captures resonance, hyperconjugation, and coordination effects that
    covalent radii miss.

    Falls back to covalent radii if xTB is unavailable or fails.

    Returns (smart_defs, ref_info) where ref_info contains xTB results
    for downstream use (e.g., deciding 1-step vs 2-step NEB).
    """
    from ase.data import covalent_radii

    # Try xTB reference optimization
    ref = _xtb_reference_bond_lengths(
        atoms, st, ligand_name, bond_defs, total_charge, workdir
    )

    smart_defs = []
    for atom_name1, atom_name2, default_target, mode in bond_defs:
        idx1 = np.where((st.res_name == ligand_name) & (st.atom_name == atom_name1))[0]
        idx2 = np.where((st.res_name == ligand_name) & (st.atom_name == atom_name2))[0]

        if len(idx1) == 1 and len(idx2) == 1:
            z1 = atoms.get_atomic_numbers()[int(idx1[0])]
            z2 = atoms.get_atomic_numbers()[int(idx2[0])]
            r_cov = covalent_radii[z1] + covalent_radii[z2]

            if mode == "attractive":
                # Use xTB product reference if available
                xtb_target = None
                if ref and ref.get("product"):
                    xtb_target = ref["product"].get((atom_name1, atom_name2))

                if xtb_target is not None and xtb_target < r_cov * 1.2:
                    target = xtb_target
                    log.info(f"  Target: {atom_name1}-{atom_name2} ({mode}): "
                             f"xTB_product={target:.3f} A "
                             f"(cov_radii={r_cov:.2f}, default={default_target:.1f})")
                else:
                    target = round(r_cov, 2)
                    log.info(f"  Target: {atom_name1}-{atom_name2} ({mode}): "
                             f"cov_radii={r_cov:.2f} A (xTB ref unavailable, "
                             f"default={default_target:.1f})")
            else:
                # Repulsive: use xTB reactant reference or 2.5x cov radii
                xtb_target = None
                if ref and ref.get("reactant"):
                    xtb_target = ref["reactant"].get((atom_name1, atom_name2))

                if xtb_target is not None and xtb_target > r_cov * 1.5:
                    target = xtb_target
                    log.info(f"  Target: {atom_name1}-{atom_name2} ({mode}): "
                             f"xTB_reactant={target:.3f} A "
                             f"(cov_radii×2.5={r_cov*2.5:.2f}, default={default_target:.1f})")
                else:
                    target = round(r_cov * 2.5, 2)
                    log.info(f"  Target: {atom_name1}-{atom_name2} ({mode}): "
                             f"cov_radii×2.5={target:.2f} A (xTB ref unavailable, "
                             f"default={default_target:.1f})")

            smart_defs.append((atom_name1, atom_name2, target, mode))
        else:
            smart_defs.append((atom_name1, atom_name2, default_target, mode))

    # Log intermediate finding
    if ref and ref.get("intermediate_is_minimum"):
        log.info("  *** xTB predicts pentacoordinate INTERMEDIATE — consider 2-step NEB ***")
        if ref.get("intermediate"):
            for (a1, a2), d in ref["intermediate"].items():
                log.info(f"    Intermediate {a1}-{a2}: {d:.3f} A")

    return smart_defs, ref


BOND_BREAKING_DEFS = {
    # PTE phosphoester substrates
    # These are DEFAULT targets — get_smart_bond_targets() overrides them
    # with covalent-radii-based values when --endpoint-method auto is used.
    "YYL": [
        ("P1", "O1", 1.4, "attractive"),   # nucleophile forms bond
        ("P1", "O5", 3.5, "repulsive"),    # leaving group fully dissociates
    ],
    "YYE": [
        ("P1", "O3", 1.4, "attractive"),
        ("P1", "O7", 3.5, "repulsive"),
    ],
    "YYF": [
        ("P1", "O3", 1.4, "attractive"),
        ("P1", "O7", 3.5, "repulsive"),
    ],
    "XUW": [
        ("P1", "O3", 1.4, "attractive"),
        ("P1", "O7", 3.5, "repulsive"),
    ],
    "YZW": [
        ("P1", "O1", 1.4, "attractive"),
        ("P1", "O5", 3.5, "repulsive"),
    ],
    # Non-PTE
    "PT4": [
        ("C7", "C8", 2.5, "repulsive"),
        ("C7", "C5", 1.4, "attractive"),
    ],
}

# ══════════════════════════════════════════════════════════════
#  MODEL REGISTRY — delegated to quantum_engine.calc
# ══════════════════════════════════════════════════════════════
# This script kept its own copy of the model registry historically;
# both have now been merged into quantum_engine.config.MACE_MODELS,
# read by quantum_engine.calc.factory.make_calc().
from quantum_engine.config import MACE_MODELS as MODEL_PATHS  # back-compat alias
from quantum_engine.calc import make_calc as _make_calc

MH_DEFAULT_HEADS = {
    "mace-mh": "omol",  # wB97M-V level, same training as OMOL
}
NEEDS_GBG_VENV = {"mace-polar-s", "mace-polar-m", "mace-polar-l", "mace-polar"}


def get_calculator(model_key, device="cuda", dtype="float64", head=None):
    """Construct a MACE calculator. Thin wrapper around the canonical
    :func:`quantum_engine.calc.make_calc` (which handles file paths,
    registry lookup, and HuggingFace auto-download)."""
    if head is None and model_key in MH_DEFAULT_HEADS:
        head = MH_DEFAULT_HEADS[model_key]
    return _make_calc(model=model_key, head=head, device=device,
                      default_dtype=dtype)


# ══════════════════════════════════════════════════════════════
#  STRUCTURE LOADING
# ══════════════════════════════════════════════════════════════


def load_structure(pdb_path):
    """Load PDB → biotite AtomArray → ASE Atoms with annotations.

    Returns (ase_atoms, biotite_structure, metadata_dict).
    """
    log.info(f"Loading structure from {pdb_path}")
    pdb_file = pdb_io.PDBFile.read(pdb_path)
    st = pdb_file.get_structure(model=1)
    log.info(f"  {len(st)} atoms, residues: {np.unique(st.res_name).tolist()}")

    ase_atoms = biotite_to_ase(st)

    # Extract charge (priority: CLI --charge > PDB REMARK > filename > error)
    charge = extract_net_charge(pdb_path)
    if charge is not None:
        ase_atoms.info["charge"] = charge
    else:
        log.warning("  *** Could not determine charge from PDB or filename ***")
        log.warning("  *** Use --charge N to specify, or protonate with protonate_active_site.py ***")
        ase_atoms.info["charge"] = 0
        charge = 0

    # Auto-detect ligand
    known_ligands = ["YYL", "YYE", "YYF", "PT4", "XUW", "YZW"]
    found = [r for r in known_ligands if r in np.unique(st.res_name)]
    if len(found) != 1:
        raise ValueError(
            f"Expected exactly 1 known ligand ({known_ligands}), found: {found}. "
            f"All residues: {np.unique(st.res_name).tolist()}"
        )
    ligand_name = found[0]
    log.info(f"  Ligand detected: {ligand_name}")

    meta = {
        "pdb_path": str(pdb_path),
        "n_atoms": len(st),
        "charge": charge,
        "ligand": ligand_name,
        "residues": np.unique(st.res_name).tolist(),
        "elements": np.unique(st.element).tolist(),
    }
    return ase_atoms, st, meta


# ══════════════════════════════════════════════════════════════
#  CONSTRAINT SETUP
# ══════════════════════════════════════════════════════════════


def setup_constraints(st, ligand_name, mode="ca-only", fix_chains=None):
    """Set up constraints for optimisation and MD phases.

    Modes:
      ca-only       CA fixed. Sidechains, backbone C/N/O, waters all free. (default)
      backbone      CA/C/N/O fixed during opt; CA-only during MD. Waters free.
      backbone-water  Like backbone but also pins water O during opt.
      ca-restrained CA fixed + harmonic restraints on C/N of isolated residues.
      none          No constraints at all.

    Args:
      fix_chains: Optional list of chain IDs. If specified, constraints are ONLY
                  applied to atoms in these chains. Atoms in other chains are free.
                  E.g., fix_chains=["B"] fixes only chain B, leaving chain A free.

    Returns (opt_constraint, md_constraint, extra_constraints).
    """
    # Never constrain these residues (in any mode)
    never_fix_res = [ligand_name, "HOH", "WAT", "ZN"]

    # Chain filter: if fix_chains specified, only constrain those chains
    if fix_chains:
        chain_mask = np.isin(st.chain_id, fix_chains)
        log.info(f"  Chain filter: constraining only chain(s) {fix_chains}")
    else:
        chain_mask = np.ones(len(st), dtype=bool)  # all chains

    if mode == "ca-only":
        opt_mask = (st.atom_name == "CA") & ~np.isin(st.res_name, never_fix_res) & chain_mask
        md_mask = opt_mask.copy()
        extra_constraints = []

    elif mode == "backbone":
        opt_mask = (
            np.isin(st.atom_name, ["CA", "C", "N", "O"])
            & ~np.isin(st.res_name, never_fix_res) & chain_mask
        )
        md_mask = (st.atom_name == "CA") & ~np.isin(st.res_name, never_fix_res) & chain_mask
        extra_constraints = []

    elif mode == "backbone-water":
        never_fix_no_water = [ligand_name, "ZN"]
        opt_mask = (
            np.isin(st.atom_name, ["CA", "C", "N", "O"])
            & ~np.isin(st.res_name, never_fix_no_water) & chain_mask
        )
        md_mask = (st.atom_name == "CA") & ~np.isin(st.res_name, never_fix_res) & chain_mask
        extra_constraints = []

    elif mode == "ca-restrained":
        opt_mask = (st.atom_name == "CA") & ~np.isin(st.res_name, never_fix_res) & chain_mask
        md_mask = opt_mask.copy()
        # Add harmonic restraints on backbone C/N of terminal/isolated residues
        extra_constraints = _build_terminal_restraints(st, ligand_name)

    elif mode == "none":
        opt_mask = np.zeros(len(st), dtype=bool)
        md_mask = np.zeros(len(st), dtype=bool)
        extra_constraints = []

    else:
        raise ValueError(
            f"Unknown constraint mode '{mode}'. "
            f"Options: ca-only, backbone, backbone-water, ca-restrained, none"
        )

    n_opt = int(np.sum(opt_mask))
    n_md = int(np.sum(md_mask))

    opt_c = FixAtoms(indices=np.where(opt_mask)[0]) if n_opt > 0 else None
    md_c = FixAtoms(indices=np.where(md_mask)[0]) if n_md > 0 else None

    log.info(f"  Constraint mode: {mode}")
    log.info(f"    Opt/NEB: {n_opt} / {len(st)} atoms fixed")
    log.info(f"    MD:      {n_md} / {len(st)} atoms fixed")
    if extra_constraints:
        log.info(f"    + {len(extra_constraints)} harmonic restraints on isolated backbone")

    return opt_c, md_c, extra_constraints


def _build_terminal_restraints(st, ligand_name, k=5.0, fmax=2.0):
    """Build harmonic restraints for backbone C/N on residues at chain breaks.

    Finds residues whose C or N are not peptide-bonded to a neighbor
    (terminal residues or fragments from cropping) and adds soft spring
    restraints to keep them near their initial positions.
    """
    restraints = []
    never_fix_res = [ligand_name, "HOH", "WAT", "ZN"]

    # Get unique (chain, res_id) pairs
    res_ids = np.unique(st.res_id[~np.isin(st.res_name, never_fix_res)])

    for rid in res_ids:
        res_mask = st.res_id == rid
        if np.isin(st.res_name[res_mask][0], never_fix_res):
            continue

        # Check if this residue has a neighbor (rid-1 or rid+1)
        has_prev = np.any(st.res_id == rid - 1)
        has_next = np.any(st.res_id == rid + 1)

        if not has_prev:
            # N-terminal / isolated: restrain N
            n_idx = np.where(res_mask & (st.atom_name == "N"))[0]
            for idx in n_idx:
                restraints.append(
                    SpringConstraint(int(idx), int(idx), k=k, rt=0.0, mode="both", fmax=fmax)
                )

        if not has_next:
            # C-terminal / isolated: restrain C
            c_idx = np.where(res_mask & (st.atom_name == "C"))[0]
            for idx in c_idx:
                restraints.append(
                    SpringConstraint(int(idx), int(idx), k=k, rt=0.0, mode="both", fmax=fmax)
                )

    return restraints


# Keep backward-compatible alias
def setup_backbone_constraints(st, ligand_name):
    """Legacy wrapper. Use setup_constraints() instead."""
    opt_c, md_c, _ = setup_constraints(st, ligand_name, mode="ca-only")
    mask = (st.atom_name == "CA") & ~np.isin(st.res_name, [ligand_name, "HOH", "WAT", "ZN"])
    return opt_c, mask


def setup_bond_breaking_constraints(st, ligand_name, spring_k=6.0, spring_fmax=5.0):
    """Create spring constraints for driving the reactant → product transformation.

    Returns a list of SpringConstraint objects and metadata about the bonds.
    """
    if ligand_name not in BOND_BREAKING_DEFS:
        raise ValueError(
            f"No bond-breaking definition for ligand '{ligand_name}'. "
            f"Known: {list(BOND_BREAKING_DEFS.keys())}. "
            f"Add a custom entry to BOND_BREAKING_DEFS."
        )

    defs = BOND_BREAKING_DEFS[ligand_name]
    constraints = []
    bond_info = []

    for atom_name1, atom_name2, target_r, mode in defs:
        idx1 = np.where(
            (st.res_name == ligand_name) & (st.atom_name == atom_name1)
        )[0]
        idx2 = np.where(
            (st.res_name == ligand_name) & (st.atom_name == atom_name2)
        )[0]

        if len(idx1) != 1 or len(idx2) != 1:
            raise ValueError(
                f"Could not uniquely find atoms {atom_name1} (found {len(idx1)}) "
                f"or {atom_name2} (found {len(idx2)}) in ligand {ligand_name}"
            )

        i1, i2 = int(idx1[0]), int(idx2[0])
        sc = SpringConstraint(i1, i2, k=spring_k, rt=target_r, mode=mode, fmax=spring_fmax)
        constraints.append(sc)

        info = {
            "atom1": atom_name1,
            "atom2": atom_name2,
            "idx1": i1,
            "idx2": i2,
            "target_r": target_r,
            "mode": mode,
        }
        bond_info.append(info)
        log.info(
            f"  Spring: {atom_name1}({i1}) ↔ {atom_name2}({i2}), "
            f"r₀={target_r:.1f} Å, mode={mode}"
        )

    return constraints, bond_info


# ══════════════════════════════════════════════════════════════
#  STEP 1 – RELAX START STATE
# ══════════════════════════════════════════════════════════════


def _drive_and_relax(
    atoms, st, ligand_name, opt_constraint, md_constraint, outdir, label,
    spring_defs, spring_k, spring_fmax,
    fmax_spring, fmax_final, md_steps, md_temp,
    md_strategy="short", n_md_seeds=5, anneal_peak=600.0,
):
    """Drive bonds with springs, relax, run MD, and polish.

    Shared logic for generating both reactant and product from the input PDB.

    Args:
        spring_defs: list of (atom1, atom2, target_r, mode) — determines direction
        label: 'start' or 'end' — for filenames
    """
    outfile = os.path.join(outdir, f"relaxation-{label}.xyz")

    if os.path.isfile(outfile):
        log.info(f"  {label} relaxation found at {outfile} – loading")
        loaded = read(outfile, index=-1)
        atoms.set_positions(loaded.get_positions())
        return atoms.copy()

    log.info(f"  Generating {label} state via spring-driven relaxation ...")

    # Phase A: Fix all non-ligand atoms, apply directional spring constraints
    fix_all_but_lig = FixAtoms(indices=np.where(~np.isin(st.res_name, [ligand_name]))[0])
    springs = []
    for atom_name1, atom_name2, target_r, mode in spring_defs:
        idx1 = int(np.where((st.res_name == ligand_name) & (st.atom_name == atom_name1))[0][0])
        idx2 = int(np.where((st.res_name == ligand_name) & (st.atom_name == atom_name2))[0][0])
        sc = SpringConstraint(idx1, idx2, k=spring_k, rt=target_r, mode=mode, fmax=spring_fmax)
        springs.append(sc)
        cur_r = atoms.get_distance(idx1, idx2)
        log.info(f"    {atom_name1}({idx1})–{atom_name2}({idx2}): {cur_r:.2f} → {target_r:.1f} Å ({mode})")

    atoms.constraints = [fix_all_but_lig] + springs

    opt = LBFGS(atoms, logfile=os.path.join(outdir, f"opt-{label}-spring.log"))
    opt.run(fmax=fmax_spring, steps=500)
    log.info(f"  Spring-constrained relax done (fmax={fmax_spring})")

    # Phase B: Switch to opt constraints (release springs, sidechains free)
    atoms.constraints = [opt_constraint] if opt_constraint else []
    opt2 = LBFGS(atoms, logfile=os.path.join(outdir, f"opt-{label}-relax.log"))
    opt2.run(fmax=fmax_spring, steps=300)
    log.info("  Opt-constrained relax done")

    # Phase C: MD equilibration (strategy-dependent)
    if md_steps > 0:
        atoms.constraints = [md_constraint] if md_constraint else []
        md_snapshots = [atoms.copy()]

        if md_strategy == "short":
            # Classic single short MD
            log.info(f"  MD strategy=short: {md_steps} steps at {md_temp} K ...")
            md = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=md_temp,
                          friction=0.01, fixcm=False)
            def _save_md_snap():
                md_snapshots.append(atoms.copy())
            md.attach(_save_md_snap, interval=max(1, md_steps // 20))
            MaxwellBoltzmannDistribution(atoms, temperature_K=md_temp)
            md.run(md_steps)

        elif md_strategy == "annealing":
            # Simulated annealing: heat → peak → cool
            log.info(f"  MD strategy=annealing: {md_temp}K → {anneal_peak}K → {md_temp}K "
                     f"({md_steps} steps total) ...")
            ramp_steps = md_steps // 3
            hold_steps = md_steps // 3
            cool_steps = md_steps - ramp_steps - hold_steps

            md = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=md_temp,
                          friction=0.01, fixcm=False)
            MaxwellBoltzmannDistribution(atoms, temperature_K=md_temp)

            # Heat
            for step in range(ramp_steps):
                t = md_temp + (anneal_peak - md_temp) * step / ramp_steps
                md.set_temperature(temperature_K=t)
                md.run(1)
            # Hold at peak
            md.set_temperature(temperature_K=anneal_peak)
            md.run(hold_steps)
            md_snapshots.append(atoms.copy())
            # Cool
            for step in range(cool_steps):
                t = anneal_peak - (anneal_peak - md_temp) * step / cool_steps
                md.set_temperature(temperature_K=t)
                md.run(1)
            log.info(f"  Annealing complete")

        elif md_strategy == "multi-seed":
            # Multiple independent short MDs, pick lowest energy
            log.info(f"  MD strategy=multi-seed: {n_md_seeds} seeds × {md_steps} steps ...")
            best_energy = float("inf")
            best_atoms = atoms.copy()
            start_positions = atoms.get_positions().copy()

            for seed in range(n_md_seeds):
                atoms.set_positions(start_positions)  # reset to same starting point
                md = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=md_temp,
                              friction=0.01, fixcm=False)
                MaxwellBoltzmannDistribution(atoms, temperature_K=md_temp)
                md.run(md_steps)

                # Quick relax to find the minimum this seed reached
                atoms.constraints = [opt_constraint] if opt_constraint else []
                opt_seed = LBFGS(atoms, logfile=os.devnull)
                opt_seed.run(fmax=0.1, steps=100)
                e = atoms.get_potential_energy()

                if e < best_energy:
                    best_energy = e
                    best_atoms = atoms.copy()
                    log.info(f"    Seed {seed+1}/{n_md_seeds}: E={e*EV_TO_KCAL:.1f} kcal/mol ← new best")
                else:
                    log.info(f"    Seed {seed+1}/{n_md_seeds}: E={e*EV_TO_KCAL:.1f} kcal/mol")

                atoms.constraints = [md_constraint] if md_constraint else []

            atoms.set_positions(best_atoms.get_positions())
            log.info(f"  Best seed: E={best_energy*EV_TO_KCAL:.1f} kcal/mol")

        elif md_strategy == "long":
            # Single long MD (md_steps should be 5000+ for this)
            log.info(f"  MD strategy=long: {md_steps} steps at {md_temp} K ...")
            md = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=md_temp,
                          friction=0.01, fixcm=False)
            def _save_md_snap_long():
                md_snapshots.append(atoms.copy())
            md.attach(_save_md_snap_long, interval=max(1, md_steps // 50))
            MaxwellBoltzmannDistribution(atoms, temperature_K=md_temp)
            md.run(md_steps)

        md_snapshots.append(atoms.copy())
        write(os.path.join(outdir, f"md-traj-{label}.xyz"), md_snapshots, format="xyz")

        # Re-apply opt constraints for post-MD relaxation
        atoms.constraints = [opt_constraint] if opt_constraint else []
        opt3 = LBFGS(atoms, logfile=os.path.join(outdir, f"opt-{label}-post-md.log"))
        opt3.run(fmax=fmax_spring, steps=300)
        log.info("  Post-MD relax done")

    # Phase D: Final tight relaxation (opt constraints)
    atoms.constraints = [opt_constraint] if opt_constraint else []
    opt_final = LBFGS(atoms, logfile=os.path.join(outdir, f"opt-{label}-polish.log"))
    opt_final.run(fmax=fmax_final, steps=500)
    log.info(f"  {label} state polished (fmax={fmax_final})")

    write(outfile, atoms)
    log.info(f"  Saved → {outfile}")
    return atoms.copy()


def generate_endpoints(
    atoms, st, ligand_name, opt_constraint, md_constraint, outdir,
    spring_k=6.0, spring_fmax=5.0,
    fmax_spring=0.10, fmax_final=0.04,
    md_steps=100, md_temp=300.0,
    unidirectional=False,
    spring_mode="both",
    md_strategy="short", n_md_seeds=5, anneal_peak=600.0,
):
    """Generate reactant and product states for NEB.

    If bidirectional (default): drives bonds BOTH ways from input.
    If unidirectional: relaxes input as start, drives bonds forward for product only.

    spring_mode controls which bonds are driven:
      both = drive nucleophile AND leaving group (may bias concerted mechanism)
      nuc-only = only attract nucleophile (LG responds naturally to energy surface)
      lg-only = only push leaving group (nuc responds naturally)
    """
    if ligand_name not in BOND_BREAKING_DEFS:
        raise ValueError(
            f"No bond-breaking definition for ligand '{ligand_name}'. "
            f"Known: {list(BOND_BREAKING_DEFS.keys())}."
        )

    fwd_defs_full = BOND_BREAKING_DEFS[ligand_name]

    # Filter spring definitions based on spring_mode
    if spring_mode == "nuc-only":
        fwd_defs = [d for d in fwd_defs_full if d[3] == "attractive"]
        log.info(f"  Spring mode: nuc-only (only driving nucleophile, LG responds naturally)")
    elif spring_mode == "lg-only":
        fwd_defs = [d for d in fwd_defs_full if d[3] == "repulsive"]
        log.info(f"  Spring mode: lg-only (only driving leaving group, nuc responds naturally)")
    else:
        fwd_defs = fwd_defs_full
        log.info(f"  Spring mode: both (driving nucleophile AND leaving group)")

    # Reverse the spring modes to drive TOWARD reactant:
    #   attractive (forming bond) → repulsive (keep that bond broken)
    #   repulsive (breaking bond) → attractive (keep that bond intact)
    # Also swap the target distances to reasonable reactant values
    mode_flip = {"attractive": "repulsive", "repulsive": "attractive", "both": "both"}
    rev_defs = []
    for atom1, atom2, target_r, mode in fwd_defs:
        if mode == "attractive":
            # This bond forms in product → push it apart for reactant
            rev_defs.append((atom1, atom2, target_r + 1.5, mode_flip[mode]))
        elif mode == "repulsive":
            # This bond breaks in product → keep it short for reactant
            rev_defs.append((atom1, atom2, target_r - 0.5, mode_flip[mode]))
        else:
            rev_defs.append((atom1, atom2, target_r, mode))

    if not unidirectional:
        log.info("Step 1: Generating reactant (start) state ...")
        log.info("  Driving bonds TOWARD reactant geometry (reversing reaction)")
        atoms_start = atoms.copy()
        atoms_start.calc = atoms.calc
        atoms_start.info = atoms.info.copy()
        start = _drive_and_relax(
            atoms_start, st, ligand_name, opt_constraint, md_constraint, outdir, "start",
            spring_defs=rev_defs, spring_k=spring_k, spring_fmax=spring_fmax,
            fmax_spring=fmax_spring, fmax_final=fmax_final,
            md_steps=md_steps, md_temp=md_temp,
            md_strategy=md_strategy, n_md_seeds=n_md_seeds, anneal_peak=anneal_peak,
        )
    else:
        log.info("Step 1: Relaxing input as reactant (unidirectional mode) ...")
        atoms_start = atoms.copy()
        atoms_start.calc = atoms.calc
        atoms_start.info = atoms.info.copy()
        atoms_start.constraints = [opt_constraint] if opt_constraint else []
        opt = LBFGS(atoms_start, logfile=os.path.join(outdir, "opt-start-relax.log"))
        opt.run(fmax=fmax_final, steps=500)
        start = atoms_start.copy()
        outfile = os.path.join(outdir, "relaxation-start.xyz")
        write(outfile, start)
        log.info(f"  Reactant relaxed (fmax={fmax_final})")

    log.info("Step 2: Generating product (end) state ...")
    log.info("  Driving bonds TOWARD product geometry (forward reaction)")
    atoms_end = atoms.copy()
    atoms_end.calc = atoms.calc
    atoms_end.info = atoms.info.copy()
    end = _drive_and_relax(
        atoms_end, st, ligand_name, opt_constraint, md_constraint, outdir, "end",
        spring_defs=fwd_defs, spring_k=spring_k, spring_fmax=spring_fmax,
        fmax_spring=fmax_spring, fmax_final=fmax_final,
        md_steps=md_steps, md_temp=md_temp,
        md_strategy=md_strategy, n_md_seeds=n_md_seeds, anneal_peak=anneal_peak,
    )

    return start, end


# ══════════════════════════════════════════════════════════════
#  STEP 3 – NEB PATH OPTIMISATION
# ══════════════════════════════════════════════════════════════


def run_neb(
    atoms_template,
    start_atoms,
    end_atoms,
    opt_constraint,
    calc_fn,
    charge,
    outdir,
    n_images=15,
    k_spring=1.0,
    fmax_noclimb=0.40,
    steps_noclimb=200,
    fmax_climb=0.045,
    steps_climb=250,
    interpolation_method="geodesic",
):
    """Run NEB (regular → climbing-image) and return the optimised images."""

    climb_file = os.path.join(outdir, "path-neb-climb.xyz")
    if os.path.isfile(climb_file):
        log.info(f"  NEB climb path found at {climb_file} – loading")
        images = read(climb_file, index=":")
        for img in images:
            img.calc = calc_fn()
            img.info["charge"] = charge
        return images

    log.info(f"Step 3: NEB optimisation ({n_images} images, k={k_spring}) ...")

    # Build image list
    images = [start_atoms.copy()]
    for _ in range(n_images - 2):
        img = start_atoms.copy()
        images.append(img)
    images.append(end_atoms.copy())

    # Assign constraints and calculators
    for img in images:
        img.constraints = [opt_constraint] if opt_constraint else []
        img.calc = calc_fn()
        img.info["charge"] = charge

    # NEB object
    neb = NEB(images, k=k_spring, allow_shared_calculator=False, method="improvedtangent")

    # Interpolation: use qcb.mlff.interpolation unified entry point
    # (supports geodesic, idpp, linear with automatic fallback)
    try:
        from quantum_engine.mlff.interpolation import interpolate as _qcb_interpolate
        log.info(f"  Interpolating NEB images: method={interpolation_method}")
        new_images = _qcb_interpolate(start_atoms, end_atoms, n_images, method=interpolation_method)
        # Copy positions into our pre-allocated images (preserves calc + constraints)
        for i, new_img in enumerate(new_images):
            images[i].set_positions(new_img.get_positions())
    except Exception as e:
        log.warning(f"  qcb interpolation failed ({e}), falling back to legacy")
        try:
            _do_geodesic_interpolation(images)
            log.info("  Using legacy geodesic interpolation")
        except Exception as e2:
            log.info(f"  Legacy geodesic also failed ({e2}), using IDPP")
            neb.interpolate(method="idpp", apply_constraint=True)

    write(os.path.join(outdir, "path-neb-init.xyz"), images)

    # Stage 1: NEB without climbing image
    log.info(f"  NEB no-climb: fmax={fmax_noclimb}, max {steps_noclimb} steps")
    traj = os.path.join(outdir, "neb-opt.traj")
    optimizer = NEBOptimizer(neb, trajectory=traj, method="ODE")
    optimizer.run(fmax=fmax_noclimb, steps=steps_noclimb)

    _save_neb_plot(images, outdir, "path-neb-noclimb")
    write(os.path.join(outdir, "path-neb-noclimb.xyz"), images)
    log.info("  NEB no-climb complete")

    # Stage 2: Climbing image NEB (skipped in quick mode)
    if steps_climb > 0:
        neb.climb = True
        log.info(f"  NEB climbing: fmax={fmax_climb}, max {steps_climb} steps")
        optimizer.run(fmax=fmax_climb, steps=steps_climb)

        _save_neb_plot(images, outdir, "neb-climb")
        write(os.path.join(outdir, "path-neb-climb.xyz"), images)
        log.info("  NEB climbing complete")
    else:
        log.info("  Climbing image skipped (quick mode)")
        write(os.path.join(outdir, "path-neb-climb.xyz"), images)

    return images


def _do_geodesic_interpolation(images):
    """Interpolate NEB images using geodesic interpolation in internal coords.

    Uses the Geodesic class from geodesic-interpolate which works in
    redundant internal coordinates (respects bond connectivity).
    Much better than IDPP for reactions with significant bond changes.
    """
    from geodesic_interpolate.geodesic import Geodesic
    from geodesic_interpolate.interpolation import redistribute
    import numpy as np

    start_pos = images[0].get_positions()
    end_pos = images[-1].get_positions()
    symbols = images[0].get_chemical_symbols()
    n_images = len(images)

    # Stack start + end as initial "path"
    raw_path = np.array([start_pos, end_pos])

    # Run geodesic interpolation
    geodesic = Geodesic.from_atoms(
        symbols, raw_path, scaling=1.7, threshold=0.002, friction=0.01
    )

    # Redistribute to get n_images evenly spaced along the geodesic
    try:
        smoothed = geodesic.path  # optimized path (may have variable # of points)
        final_path = redistribute(symbols, smoothed, n_images, tol=0.003)
    except Exception:
        # Fallback: just use the raw geodesic path and interpolate linearly
        final_path = np.linspace(start_pos, end_pos, n_images)

    # Apply to NEB images (skip first and last = endpoints)
    for i in range(1, n_images - 1):
        if i < len(final_path):
            images[i].set_positions(final_path[i])


def _save_neb_plot(images, outdir, basename):
    """Save an energy profile plot in kcal/mol for the current NEB band."""
    try:
        energies = [img.get_potential_energy() for img in images]
        e_ref = energies[0]
        e_kcal = [(e - e_ref) * EV_TO_KCAL for e in energies]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(len(e_kcal)), e_kcal, "bo-", markersize=8, linewidth=2)
        ax.set_xlabel("Image", fontsize=13)
        ax.set_ylabel("Relative Energy (kcal/mol)", fontsize=13)
        ax.set_title(basename.replace("-", " ").title(), fontsize=14)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        # Annotate barrier
        ts_idx = int(np.argmax(e_kcal[1:-1])) + 1
        if e_kcal[ts_idx] > 0:
            ax.annotate(
                f"{e_kcal[ts_idx]:.1f} kcal/mol",
                xy=(ts_idx, e_kcal[ts_idx]),
                xytext=(ts_idx + 1, e_kcal[ts_idx] + 2),
                fontsize=11,
                arrowprops=dict(arrowstyle="->", color="red"),
                color="red",
            )

        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{basename}.png"), dpi=200)
        plt.close(fig)
    except Exception as e:
        log.warning(f"  Could not generate NEB plot: {e}")


# ══════════════════════════════════════════════════════════════
#  STEP 4 – SELLA TS REFINEMENT
# ══════════════════════════════════════════════════════════════


def refine_ts_sella(
    images,
    calc_fn,
    charge,
    opt_constraint,
    outdir,
    fmax=0.02,
    max_steps=1000,
):
    """Refine the highest-energy NEB image with Sella saddle-point optimiser.

    Key improvement over earlier versions: uses internal=True for dramatically
    better convergence (Sella + internal coords outperforms Cartesian by ~2×).
    """
    ts_file = os.path.join(outdir, "path-after-sella-ts.xyz")

    if os.path.isfile(ts_file):
        log.info(f"  Sella TS found at {ts_file} – loading")
        return read(ts_file)

    log.info("Step 4: Sella TS refinement ...")

    # Find highest-energy image (skip endpoints)
    energies = [img.get_potential_energy() for img in images[1:-1]]
    ts_idx = int(np.argmax(energies)) + 1
    ts_guess = images[ts_idx].copy()
    ts_guess.calc = calc_fn()
    ts_guess.info["charge"] = charge
    ts_guess.constraints = [opt_constraint] if opt_constraint else []

    e_ts = energies[np.argmax(energies)]
    e_start = images[0].get_potential_energy()
    log.info(
        f"  TS guess: image {ts_idx}, barrier ≈ {(e_ts - e_start) * EV_TO_KCAL:.1f} kcal/mol"
    )

    # Run Sella — use internal coordinates only for small systems (<200 free atoms)
    # because internal coord construction scales O(N^2-N^3) and can OOM for large clusters
    sella_traj = os.path.join(outdir, "neb-sella-opt.traj")
    sella_log = os.path.join(outdir, "sella-opt.log")

    n_free = len(ts_guess) - sum(
        len(c.index) for c in ts_guess.constraints if isinstance(c, FixAtoms)
    )
    use_internal = n_free < 200
    log.info(
        f"  {n_free} free atoms → using {'internal' if use_internal else 'Cartesian'} coords"
    )

    try:
        dyn = Sella(
            ts_guess,
            internal=use_internal,
            order=1,
            trajectory=sella_traj,
            logfile=sella_log,
        )
        dyn.run(fmax=fmax, steps=max_steps)
        log.info(f"  Sella converged (fmax={fmax})")
    except Exception as e:
        if use_internal:
            log.warning(f"  Sella internal coords failed ({e}), retrying Cartesian ...")
            ts_guess = images[ts_idx].copy()
            ts_guess.calc = calc_fn()
            ts_guess.info["charge"] = charge
            ts_guess.constraints = [opt_constraint] if opt_constraint else []
            dyn = Sella(
                ts_guess,
                internal=False,
                order=1,
                trajectory=sella_traj,
                logfile=sella_log,
            )
            dyn.run(fmax=fmax, steps=max_steps)
            log.info(f"  Sella converged (Cartesian fallback, fmax={fmax})")
        else:
            raise

    ts_guess.info["sella_energy"] = ts_guess.get_potential_energy()
    write(ts_file, ts_guess)
    log.info(f"  Saved TS → {ts_file}")

    # Also re-evaluate full path energies and save
    try:
        for img in images:
            if img.calc is None:
                img.calc = calc_fn()
                img.info["charge"] = charge
        _save_neb_plot(images, outdir, "neb-sella-opt")
        write(os.path.join(outdir, "path-after-sella.xyz"), images, format="extxyz")
    except Exception:
        pass

    return ts_guess


# ══════════════════════════════════════════════════════════════
#  STEP 5 – FREQUENCY VALIDATION
# ══════════════════════════════════════════════════════════════


def validate_ts(ts_atoms, opt_constraint, outdir):
    """Run frequency analysis to verify exactly one imaginary frequency.

    For enzyme active-site clusters with fixed backbone, only free atoms
    are included in the Hessian.
    """
    log.info("Step 5: Frequency validation ...")

    try:
        from ase.vibrations import Vibrations
    except ImportError:
        log.warning("  ase.vibrations not available – skipping validation")
        return None

    # Determine free atoms (not fixed by backbone constraint)
    fixed_indices = set()
    for c in ts_atoms.constraints:
        if isinstance(c, FixAtoms):
            fixed_indices.update(c.index)
    free_indices = [i for i in range(len(ts_atoms)) if i not in fixed_indices]

    if len(free_indices) > 200:
        log.warning(
            f"  {len(free_indices)} free atoms – frequency calculation may be slow. "
            "Consider using a smaller active-site model or skipping with --skip-freq"
        )

    vib_dir = os.path.join(outdir, "vib_analysis")
    os.makedirs(vib_dir, exist_ok=True)

    vib = Vibrations(ts_atoms, indices=free_indices, name=os.path.join(vib_dir, "vib"), delta=0.01)
    vib.run()

    freqs = vib.get_frequencies()
    # Imaginary frequencies appear as complex numbers in ASE
    imag_freqs = [f for f in freqs if np.iscomplex(f) and abs(f.imag) > 10]
    real_freqs = [f.real for f in freqs if not np.iscomplex(f) or abs(f.imag) < 10]

    freq_file = os.path.join(outdir, "frequencies.dat")
    with open(freq_file, "w") as fh:
        fh.write("# Vibrational frequencies (cm⁻¹)\n")
        fh.write(f"# Imaginary: {len(imag_freqs)}\n")
        for f in sorted(freqs, key=lambda x: x.real):
            if np.iscomplex(f) and abs(f.imag) > 10:
                fh.write(f"  {f.imag:10.1f}i\n")
            else:
                fh.write(f"  {f.real:10.1f}\n")

    if len(imag_freqs) == 1:
        log.info(
            f"  TS VALIDATED: 1 imaginary frequency at {imag_freqs[0].imag:.1f}i cm⁻¹"
        )
    elif len(imag_freqs) == 0:
        log.warning("  NO imaginary frequencies – this may be a minimum, not a TS")
    else:
        log.warning(
            f"  {len(imag_freqs)} imaginary frequencies found – "
            "TS may not be fully converged or has extra soft modes from constraints"
        )

    vib.summary(log=os.path.join(outdir, "vib_summary.txt"))
    log.info(f"  Frequencies saved → {freq_file}")

    return {
        "n_imaginary": len(imag_freqs),
        "imaginary_freqs": [f.imag for f in imag_freqs],
        "lowest_real": min(real_freqs) if real_freqs else None,
    }


# ══════════════════════════════════════════════════════════════
#  OUTPUT ORGANIZATION
# ══════════════════════════════════════════════════════════════


def _organize_outputs(outdir, relax_dir, ts_dir, template_st, start, end, ts, images,
                      system_name=None, charge_method="auto"):
    """Organize outputs into user-facing PDBs and a technical/ subdirectory.

    Top-level outputs (what the user cares about):
      reactant.pdb         - relaxed reactant state
      product.pdb          - relaxed product state
      transition_state.pdb - the TS structure
      neb_path.pdb         - multi-MODEL PDB of the full NEB path
      energy_profile.png   - barrier plot
      summary.json         - barriers, timings

    technical/ subdirectory (raw files for debugging):
      relax/               - optimization logs, trajectories
      ts/                  - NEB xyz files, .traj files, Sella logs
    """
    import shutil

    tech_dir = os.path.join(outdir, "technical")
    os.makedirs(tech_dir, exist_ok=True)

    # Move relax/ and ts/ into technical/
    for subdir in ["relax", "ts"]:
        src = os.path.join(outdir, subdir)
        dst = os.path.join(tech_dir, subdir)
        if os.path.isdir(src) and not os.path.isdir(dst):
            shutil.move(src, dst)

    # Deterministic basename for all outputs (prevents overwrites when combining dirs)
    prefix = f"{system_name}_" if system_name else ""

    # Write user-facing PDBs
    write_result_pdb(start, template_st, os.path.join(outdir, f"{prefix}reactant.pdb"))
    write_result_pdb(end, template_st, os.path.join(outdir, f"{prefix}product.pdb"))
    write_result_pdb(ts, template_st, os.path.join(outdir, f"{prefix}transition_state.pdb"))

    # Write CIF for transition state (includes charge annotation)
    _write_ts_cif(ts, template_st, outdir, charge_method=charge_method,
                  filename=f"{prefix}transition_state.cif")

    # NEB path as multi-MODEL PDB
    if images:
        neb_energies = []
        for img in images:
            try:
                neb_energies.append(img.get_potential_energy())
            except Exception:
                neb_energies.append(0.0)
        write_trajectory_pdb(
            images, template_st,
            os.path.join(outdir, f"{prefix}neb_path.pdb"),
            energies=neb_energies,
        )

    # MD trajectories — read from technical/ if .traj files exist
    _write_md_traj_pdbs(outdir, tech_dir, template_st)

    # Copy energy profile plot to top level
    for png_name in ["neb-climb.png", "path-neb-noclimb.png"]:
        src = os.path.join(tech_dir, "ts", png_name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(outdir, f"{prefix}energy_profile.png"))
            break

    log.info("  Output organized:")
    log.info(f"    {outdir}/reactant.pdb")
    log.info(f"    {outdir}/product.pdb")
    log.info(f"    {outdir}/transition_state.pdb")
    log.info(f"    {outdir}/neb_path.pdb")
    log.info(f"    {outdir}/energy_profile.png")
    log.info(f"    {outdir}/technical/  (raw logs, traj, xyz)")


def _compute_formal_charges(atoms, template_st, total_charge, outdir=None):
    """Derive formal integer charges from electronic structure.

    Priority:
    1. xTB Mulliken charges + Wiberg bond orders �� formal charges
       (most rigorous: actual electronic structure + bond order analysis)
    2. RDKit DetermineBondOrders from 3D geometry
       (good: handles nitro, phosphate, carboxylate, aromatics)
    3. Mulliken → round to nearest integer
       (fallback)

    The xTB approach is preferred because it solves the actual electronic
    structure, while RDKit only guesses bonds from distances.
    """
    n_atoms = len(atoms)
    symbols = atoms.get_chemical_symbols()

    # Method 1: xTB Mulliken + WBO → formal charges
    XTB_BIN = "/home/dme5188/bin/xtb/xtb-6.6.1/bin/xtb"
    if os.path.isfile(XTB_BIN) and n_atoms < 500:
        try:
            tmpdir = Path(outdir or "/tmp") / ".xtb_formal"
            tmpdir.mkdir(exist_ok=True)

            # Write XYZ
            positions = atoms.get_positions()
            with open(tmpdir / "ts.xyz", "w") as f:
                f.write(f"{n_atoms}\n\n")
                for sym, (x, y, z) in zip(symbols, positions):
                    f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")

            timeout_s = min(600, max(60, n_atoms * 2))
            result = subprocess.run(
                [XTB_BIN, "ts.xyz", "--gfn", "2", "--chrg", str(total_charge), "--sp"],
                capture_output=True, text=True, timeout=timeout_s, cwd=str(tmpdir),
            )

            charges_file = tmpdir / "charges"
            wbo_file = tmpdir / "wbo"

            if charges_file.exists() and wbo_file.exists():
                mulliken = np.array([float(l) for l in charges_file.read_text().strip().split("\n")])

                # Parse WBO for bond order analysis
                bond_orders = {}  # {(i,j): wbo}
                for line in wbo_file.read_text().strip().split("\n"):
                    parts = line.split()
                    if len(parts) >= 3:
                        i, j = int(parts[0]) - 1, int(parts[1]) - 1
                        bo = float(parts[2])
                        if bo > 0.3:  # only significant bonds
                            bond_orders[(min(i, j), max(i, j))] = bo

                # Derive formal charges from Mulliken + valence analysis
                # Strategy: atoms with |Mulliken| > 0.4 are likely charged
                # Refine using bond orders to distinguish -1 from 0
                formal = np.zeros(n_atoms)

                for i in range(n_atoms):
                    q = mulliken[i]
                    z = atoms.get_atomic_numbers()[i]

                    # Standard valence expectations
                    expected_valence = {1: 1, 6: 4, 7: 3, 8: 2, 15: 5, 16: 2, 30: 2}
                    exp_v = expected_valence.get(z, 4)

                    # Sum of bond orders around this atom
                    total_bo = sum(bo for (a, b), bo in bond_orders.items() if a == i or b == i)

                    # Formal charge = valence electrons - lone pairs - bonding electrons
                    # Simplified: if total bond order < expected valence, atom likely has charge
                    if z == 8:  # oxygen
                        if total_bo < 1.5 and q < -0.3:
                            formal[i] = -1  # O⁻ (single bond, negative)
                        elif total_bo > 2.5:
                            formal[i] = +1  # rare: O⁺
                    elif z == 7:  # nitrogen
                        if total_bo > 3.5 and q > 0.2:
                            formal[i] = +1  # N⁺ (4 bonds, like NH4+ or nitro)
                        elif total_bo < 2.5 and q < -0.3:
                            formal[i] = -1  # N⁻
                    elif z == 15:  # phosphorus
                        pass  # P is usually formally 0 in phosphoesters

                # Adjust to match total charge
                fc_sum = int(formal.sum())
                diff = total_charge - fc_sum
                if diff != 0:
                    # Assign remaining charge to atoms with largest |Mulliken|
                    sorted_by_q = np.argsort(mulliken) if diff < 0 else np.argsort(-mulliken)
                    for idx in sorted_by_q:
                        if diff == 0:
                            break
                        if diff > 0 and formal[idx] == 0 and mulliken[idx] > 0.3:
                            formal[idx] = +1
                            diff -= 1
                        elif diff < 0 and formal[idx] == 0 and mulliken[idx] < -0.3:
                            formal[idx] = -1
                            diff += 1

                n_charged = int(np.sum(formal != 0))
                log.info(f"  xTB-derived formal charges: sum={int(formal.sum()):+d}, "
                         f"{n_charged} charged atoms, {len(bond_orders)} bonds")

                for i in range(n_atoms):
                    if formal[i] != 0:
                        resinfo = ""
                        if template_st is not None and i < len(template_st):
                            resinfo = (f" ({template_st.res_name[i]} "
                                       f"{template_st.chain_id[i]}:{template_st.res_id[i]} "
                                       f"{template_st.atom_name[i]})")
                        log.info(f"    {symbols[i]}({i}){resinfo}: {int(formal[i]):+d} "
                                 f"(Mulliken={mulliken[i]:.2f}, BO={sum(bo for (a,b),bo in bond_orders.items() if a==i or b==i):.1f})")

                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
                return formal

        except Exception as e:
            log.warning(f"  xTB formal charge derivation failed: {e}")

    # Method 2: RDKit DetermineBondOrders
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
        from rdkit.Geometry import Point3D

        mol = Chem.RWMol()
        positions = atoms.get_positions()
        for sym in symbols:
            mol.AddAtom(Chem.Atom(sym))
        conf = Chem.Conformer(n_atoms)
        for i, (x, y, z) in enumerate(positions):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        mol.AddConformer(conf)

        rdDetermineBonds.DetermineConnectivity(mol)
        rdDetermineBonds.DetermineBondOrders(mol, charge=total_charge)

        charges = np.array([mol.GetAtomWithIdx(i).GetFormalCharge()
                           for i in range(n_atoms)], dtype=float)

        log.info(f"  RDKit formal charges: sum={int(charges.sum()):+d}, "
                 f"{int(np.sum(charges != 0))} charged atoms")
        return charges

    except Exception as e:
        log.warning(f"  RDKit formal charges failed: {e}")

    # Method 3: Mulliken → round
    log.warning("  Falling back to Mulliken→round for formal charges")
    mulliken = _compute_per_atom_charges(atoms, total_charge, outdir or "/tmp",
                                          method="auto", template_st=template_st)
    charges = np.round(mulliken)
    diff = int(total_charge - charges.sum())
    if diff > 0:
        for _ in range(abs(diff)):
            charges[np.argmin(mulliken - charges)] += 1
    elif diff < 0:
        for _ in range(abs(diff)):
            charges[np.argmax(mulliken - charges)] -= 1

    return charges


def _compute_per_atom_charges(atoms, total_charge, outdir, method="auto", template_st=None):
    """Compute per-atom charges.

    Methods:
      auto: xTB Mulliken → OpenBabel EEM → electronegativity (cascade)
      mulliken: xTB Mulliken only (fails if xTB unavailable)
      eem: OpenBabel EEM only (fast, any size)
      formal: Integer charges from residue identity (instant)

    Returns numpy array of per-atom charges summing to total_charge.
    """
    if method == "formal":
        return _compute_formal_charges(atoms, template_st, total_charge)

    if method == "eem":
        # Skip xTB, go straight to EEM
        pass  # fall through to EEM section below after xTB block

    XTB_BIN = "/home/dme5188/bin/xtb/xtb-6.6.1/bin/xtb"
    n_atoms = len(atoms)

    # For "auto" and "mulliken", try xTB first
    if method == "eem":
        pass  # skip xTB, go to EEM below
    elif not os.path.isfile(XTB_BIN):
        log.warning("  xTB not found — using uniform charge distribution")
        return np.full(n_atoms, total_charge / n_atoms)

    tmpdir = Path(outdir) / ".xtb_charges"
    tmpdir.mkdir(exist_ok=True)

    # Write XYZ
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    xyz_file = tmpdir / "ts.xyz"
    with open(xyz_file, "w") as f:
        f.write(f"{n_atoms}\n\n")
        for sym, (x, y, z) in zip(symbols, positions):
            f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")

    # Try xTB GFN2, then GFN1 (lighter), then electronegativity fallback
    # GFN2 timing: ~5s for 154 atoms, ~30s for 300 atoms, ~15-30 min for 1000 atoms
    # Scale timeout generously: 2s per atom, min 120s, max 1800s (30 min)
    timeout_s = min(1800, max(120, n_atoms * 2))

    for method in ["2", "1"]:
        cmd = [XTB_BIN, "ts.xyz", "--sp", "--chrg", str(total_charge), "--gfn", method]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout_s, cwd=str(tmpdir),
            )
        except subprocess.TimeoutExpired:
            log.warning(f"  xTB GFN{method} timed out ({timeout_s}s) for {n_atoms} atoms")
            continue

        charges_file = tmpdir / "charges"
        if charges_file.exists():
            charges = np.array([float(l) for l in charges_file.read_text().strip().split("\n")])
            if len(charges) == n_atoms and abs(charges.sum() - total_charge) < 0.1:
                log.info(f"  xTB GFN{method} charges: sum={charges.sum():.3f}, "
                         f"range=[{charges.min():.3f}, {charges.max():.3f}]")
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
                return charges
            else:
                # Wrong number of charges or sum mismatch — try next method
                charges_file.unlink()

        log.warning(f"  xTB GFN{method} failed for {n_atoms} atoms, trying next method ...")

    log.warning(f"  xTB methods failed or timed out — trying OpenBabel EEM charges ...")
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Fallback: OpenBabel EEM (Electronegativity Equalization Method)
    # Sub-second for any system size, physically motivated
    OBABEL = "/home/dme5188/bin/openbabel/bin/obabel"
    if os.path.isfile(OBABEL):
        try:
            tmpdir2 = Path(outdir) / ".obabel_charges"
            tmpdir2.mkdir(exist_ok=True)

            # Write PDB for OpenBabel
            from ase.io import write as ase_write
            pdb_tmp = tmpdir2 / "ts.pdb"
            ase_write(str(pdb_tmp), atoms, format="proteindatabank")

            mol2_tmp = tmpdir2 / "ts.mol2"
            result = subprocess.run(
                [OBABEL, str(pdb_tmp), "-O", str(mol2_tmp), "--partialcharge", "eem"],
                capture_output=True, text=True, timeout=30,
            )

            if mol2_tmp.exists():
                # Parse charges from mol2
                eem_charges = []
                reading = False
                for line in mol2_tmp.read_text().splitlines():
                    if "@<TRIPOS>ATOM" in line:
                        reading = True
                        continue
                    if "@<TRIPOS>" in line and reading:
                        break
                    if reading:
                        parts = line.split()
                        if len(parts) >= 9:
                            eem_charges.append(float(parts[8]))

                if len(eem_charges) == n_atoms:
                    charges = np.array(eem_charges)
                    # Shift to match net charge (EEM assumes neutral)
                    charges += (total_charge - charges.sum()) / n_atoms
                    log.info(f"  OpenBabel EEM charges: sum={charges.sum():.3f}, "
                             f"range=[{charges.min():.3f}, {charges.max():.3f}]")
                    shutil.rmtree(tmpdir2, ignore_errors=True)
                    return charges

            shutil.rmtree(tmpdir2, ignore_errors=True)
        except Exception as e:
            log.warning(f"  OpenBabel EEM failed: {e}")

    # Final fallback: electronegativity-based distribution
    log.warning(f"  Using electronegativity-based charge distribution (last resort)")
    electroneg = {1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 15: 2.19, 16: 2.58, 30: 1.65}
    en_values = np.array([electroneg.get(z, 2.0) for z in atoms.get_atomic_numbers()])
    en_shifted = en_values - en_values.mean()
    if np.abs(en_shifted).sum() > 0:
        charges = total_charge * en_shifted / np.abs(en_shifted).sum()
        charges += (total_charge - charges.sum()) / n_atoms
    else:
        charges = np.full(n_atoms, total_charge / n_atoms)
    return charges


def _run_xtb_singlepoint(atoms, total_charge, workdir):
    """Run xTB single-point, return (mulliken_charges, wiberg_bond_orders).

    Returns (None, None) if xTB fails or is unavailable.
    WBO dict is {(i,j): float} with i < j.
    """
    import shutil

    XTB_BIN = "/home/dme5188/bin/xtb/xtb-6.6.1/bin/xtb"
    n_atoms = len(atoms)

    if not os.path.isfile(XTB_BIN) or n_atoms >= 500:
        return None, None

    tmpdir = Path(workdir) / ".xtb_cif"
    tmpdir.mkdir(exist_ok=True)

    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    with open(tmpdir / "ts.xyz", "w") as f:
        f.write(f"{n_atoms}\n\n")
        for sym, (x, y, z) in zip(symbols, positions):
            f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")

    timeout_s = min(1800, max(120, n_atoms * 2))
    try:
        subprocess.run(
            [XTB_BIN, "ts.xyz", "--gfn", "2", "--chrg", str(total_charge), "--sp"],
            capture_output=True, text=True, timeout=timeout_s, cwd=str(tmpdir),
        )
    except subprocess.TimeoutExpired:
        log.warning(f"  xTB timed out ({timeout_s}s) for {n_atoms} atoms")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None, None

    charges_file = tmpdir / "charges"
    wbo_file = tmpdir / "wbo"

    mulliken = None
    bond_orders = None

    if charges_file.exists():
        vals = [float(l) for l in charges_file.read_text().strip().split("\n")]
        if len(vals) == n_atoms:
            mulliken = np.array(vals)

    if wbo_file.exists():
        bond_orders = {}
        for line in wbo_file.read_text().strip().split("\n"):
            parts = line.split()
            if len(parts) >= 3:
                i, j = int(parts[0]) - 1, int(parts[1]) - 1
                bo = float(parts[2])
                if bo > 0.3:
                    bond_orders[(min(i, j), max(i, j))] = bo

    shutil.rmtree(tmpdir, ignore_errors=True)
    return mulliken, bond_orders


def _derive_formal_from_mulliken_wbo(atoms, mulliken, bond_orders, total_charge, template_st=None):
    """Derive formal integer charges from Mulliken charges + Wiberg bond orders."""
    n_atoms = len(atoms)
    symbols = atoms.get_chemical_symbols()
    formal = np.zeros(n_atoms)

    for i in range(n_atoms):
        q = mulliken[i]
        z = atoms.get_atomic_numbers()[i]
        total_bo = sum(bo for (a, b), bo in bond_orders.items() if a == i or b == i)

        if z == 8:  # oxygen
            if total_bo < 1.5 and q < -0.3:
                formal[i] = -1
            elif total_bo > 2.5:
                formal[i] = +1
        elif z == 7:  # nitrogen
            if total_bo > 3.5 and q > 0.2:
                formal[i] = +1
            elif total_bo < 2.5 and q < -0.3:
                formal[i] = -1

    # Adjust to match total charge
    fc_sum = int(formal.sum())
    diff = total_charge - fc_sum
    if diff != 0:
        sorted_by_q = np.argsort(mulliken) if diff < 0 else np.argsort(-mulliken)
        for idx in sorted_by_q:
            if diff == 0:
                break
            if diff > 0 and formal[idx] == 0 and mulliken[idx] > 0.3:
                formal[idx] = +1
                diff -= 1
            elif diff < 0 and formal[idx] == 0 and mulliken[idx] < -0.3:
                formal[idx] = -1
                diff += 1

    n_charged = int(np.sum(formal != 0))
    log.info(f"  Formal charges: sum={int(formal.sum()):+d}, "
             f"{n_charged} charged atoms")

    for i in range(n_atoms):
        if formal[i] != 0:
            resinfo = ""
            if template_st is not None and i < len(template_st):
                resinfo = (f" ({template_st.res_name[i]} "
                           f"{template_st.chain_id[i]}:{template_st.res_id[i]} "
                           f"{template_st.atom_name[i]})")
            total_bo = sum(bo for (a, b), bo in bond_orders.items() if a == i or b == i)
            log.info(f"    {symbols[i]}({i}){resinfo}: {int(formal[i]):+d} "
                     f"(Mulliken={mulliken[i]:.2f}, BO={total_bo:.1f})")
    return formal


def _write_ts_cif(ts_atoms, template_st, outdir, charge_method="auto",
                  filename="transition_state.cif"):
    """Write transition state as CIF with per-atom charges, bond types, and energy.

    Runs xTB once to get both Mulliken charges and Wiberg bond orders.
    Uses atomworks CIF writer (with proper bond types) if available,
    falls back to simple text CIF otherwise.
    """
    try:
        cif_path = os.path.join(outdir, filename)
        charge = ts_atoms.info.get("charge", 0)
        n_atoms = len(ts_atoms)

        # Single xTB run for both charges and bond orders
        log.info("  Computing charges + bond orders (xTB single-point) ...")
        mulliken, wbo = _run_xtb_singlepoint(ts_atoms, charge, outdir)

        # Partial charges
        if mulliken is not None:
            per_atom_charges = mulliken
            log.info(f"  xTB Mulliken charges: sum={mulliken.sum():.3f}, "
                     f"range=[{mulliken.min():.3f}, {mulliken.max():.3f}]")
        else:
            log.info("  xTB unavailable, using fallback charges")
            per_atom_charges = _compute_per_atom_charges(
                ts_atoms, charge, outdir, method=charge_method, template_st=template_st
            )

        # Formal charges
        if mulliken is not None and wbo is not None:
            formal_charges = _derive_formal_from_mulliken_wbo(
                ts_atoms, mulliken, wbo, charge, template_st=template_st
            )
        else:
            formal_charges = _compute_formal_charges(
                ts_atoms, template_st, charge, outdir=outdir
            )

        # Try atomworks CIF writer (proper bond types from WBO)
        try:
            from quantum_engine.io.cif import write_ts_cif
            result = write_ts_cif(
                ts_atoms, template_st, outdir,
                total_charge=charge,
                partial_charges=per_atom_charges,
                formal_charges=formal_charges,
                wiberg_bond_orders=wbo,
                filename=filename,
            )
            if result:
                log.info(f"  Wrote {result} (atomworks CIF with bonds + charges)")
                return
        except ImportError:
            log.info("  atomworks not available, using simple CIF format")
        except Exception as e:
            log.warning(f"  atomworks CIF failed ({e}), using simple CIF format")

        # Fallback: simple text CIF
        energy = None
        try:
            energy = ts_atoms.get_potential_energy()
        except Exception:
            pass

        symbols = ts_atoms.get_chemical_symbols()
        positions = ts_atoms.get_positions()
        has_template = template_st is not None

        with open(cif_path, "w") as f:
            f.write("data_transition_state\n")
            f.write(f"_qcb.total_charge  {charge}\n")
            f.write(f"_qcb.charge_method  {charge_method}\n")
            if energy is not None:
                f.write(f"_qcb.energy_eV  {energy:.6f}\n")
                f.write(f"_qcb.energy_kcal_mol  {energy * EV_TO_KCAL:.2f}\n")
            f.write("\n")

            f.write("loop_\n")
            f.write("_atom_site.id\n")
            f.write("_atom_site.type_symbol\n")
            if has_template:
                f.write("_atom_site.label_atom_id\n")
                f.write("_atom_site.label_comp_id\n")
                f.write("_atom_site.label_asym_id\n")
                f.write("_atom_site.label_seq_id\n")
            f.write("_atom_site.Cartn_x\n")
            f.write("_atom_site.Cartn_y\n")
            f.write("_atom_site.Cartn_z\n")
            f.write("_atom_site.partial_charge\n")
            f.write("_atom_site.pdbx_formal_charge\n")

            for i in range(n_atoms):
                parts = [str(i + 1), symbols[i]]
                if has_template and i < len(template_st):
                    parts.extend([
                        template_st.atom_name[i],
                        template_st.res_name[i],
                        template_st.chain_id[i],
                        str(template_st.res_id[i]),
                    ])
                parts.extend([
                    f"{positions[i][0]:.4f}",
                    f"{positions[i][1]:.4f}",
                    f"{positions[i][2]:.4f}",
                    f"{per_atom_charges[i]:.4f}",
                    f"{int(formal_charges[i])}",
                ])
                f.write(" ".join(parts) + "\n")

        log.info(f"  Wrote {cif_path} (simple CIF with charges, {n_atoms} atoms)")
    except Exception as e:
        log.warning(f"  Could not write CIF: {e}")


def _write_md_traj_pdbs(outdir, tech_dir, template_st):
    """Convert saved MD trajectory xyz files to multi-MODEL PDBs."""
    relax_tech = os.path.join(tech_dir, "relax")
    if not os.path.isdir(relax_tech):
        return

    label_map = {"start": "reactant", "end": "product"}
    for label, nice_name in label_map.items():
        xyz_f = os.path.join(relax_tech, f"md-traj-{label}.xyz")
        if not os.path.isfile(xyz_f):
            continue
        try:
            frames = read(xyz_f, index=":", format="xyz")
            write_trajectory_pdb(
                frames, template_st,
                os.path.join(outdir, f"md_{nice_name}.pdb"),
            )
        except Exception as e:
            log.warning(f"  Could not convert MD trajectory for {label}: {e}")


# ══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════


def run_pipeline(args):
    """Execute the full NEB-TS pipeline."""
    t0 = time.time()
    pdb_path = args.pdb_file

    # ── System name ──
    system_name = Path(pdb_path).stem
    model_tag = Path(args.model).stem if os.path.isfile(args.model) else args.model

    # ── Output directory ──
    if args.outdir:
        outdir_base = args.outdir
    else:
        outdir_base = os.path.join("outputs", f"{system_name}-{model_tag}")
    relax_dir = os.path.join(outdir_base, "relax")
    ts_dir = os.path.join(outdir_base, "ts")
    os.makedirs(relax_dir, exist_ok=True)
    os.makedirs(ts_dir, exist_ok=True)

    # Set up file logging
    fh = logging.FileHandler(os.path.join(outdir_base, "pipeline.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    # ── Resolve mode into flags ──
    mode = args.mode
    do_climb = mode in ("standard", "full")
    do_sella = mode == "full" and not args.skip_sella
    do_freq = mode == "full" and not args.skip_freq
    if args.force_sella:
        do_sella = True
    if args.skip_sella:
        do_sella = False
    if args.skip_freq:
        do_freq = False

    log.info("=" * 60)
    log.info("NEB-TS Pipeline")
    log.info(f"  Input:  {pdb_path}")
    log.info(f"  Model:  {args.model}")
    log.info(f"  Mode:   {mode} (climb={do_climb}, sella={do_sella}, freq={do_freq})")
    log.info(f"  Output: {outdir_base}")
    log.info("=" * 60)

    # ── Load structure ──
    ase_atoms, bt_struct, meta = load_structure(pdb_path)
    ligand = meta["ligand"]

    # Resolve charge: CLI override > auto-detected
    if args.charge is not None:
        charge = args.charge
        ase_atoms.info["charge"] = charge
        log.info(f"  Charge from --charge flag: {charge:+d}")
    else:
        charge = meta["charge"]
    log.info(f"  System net charge: {charge:+d}")

    # ── Calculator factory ──
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("No GPU detected – calculations will be SLOW")

    head = getattr(args, "head", None)
    model_relax = getattr(args, "model_relax", None)

    def make_calc(for_neb=False):
        """Create calculator. Uses --model-relax for relaxation/MD, --model for NEB/TS."""
        if not for_neb and model_relax:
            return get_calculator(model_relax, device=device, dtype="float64", head=head)
        return get_calculator(args.model, device=device, dtype="float64", head=head)

    if model_relax:
        log.info(f"  Using '{model_relax}' for relaxation/MD, '{args.model}' for NEB/TS")

    ase_atoms.calc = make_calc()

    # ── Pre-relaxation (for messy inputs with clashes) ──
    if args.pre_relax:
        log.info("Pre-relaxation: relaxing ENTIRE structure (no constraints) to resolve clashes ...")
        opt_pre = LBFGS(ase_atoms, logfile=os.path.join(relax_dir, "opt-pre-relax.log"))
        opt_pre.run(fmax=0.5, steps=200)
        log.info(f"  Coarse pre-relax done")
        opt_pre.run(fmax=0.1, steps=300)
        log.info(f"  Fine pre-relax done (fmax=0.1)")
        # Update the biotite template with pre-relaxed positions
        bt_struct.coord = ase_atoms.get_positions().astype(np.float32)

    # ── Constraints ──
    opt_c, md_c, extra_constraints = setup_constraints(
        bt_struct, ligand, mode=args.constraint_mode,
        fix_chains=args.fix_chains,
    )

    # ── Override bond targets with xTB-optimized reference values ──
    endpoint_method = args.endpoint_method
    xtb_ref = None
    if endpoint_method == "auto" and ligand in BOND_BREAKING_DEFS:
        log.info("  Computing reference bond lengths (xTB fragment optimization) ...")
        smart_defs, xtb_ref = get_smart_bond_targets(
            ase_atoms, bt_struct, ligand, BOND_BREAKING_DEFS[ligand],
            total_charge=charge, workdir=relax_dir,
        )
        BOND_BREAKING_DEFS[ligand] = smart_defs

    # ══════════════════════════════════════════════════════════════
    # Strategy dispatcher: choose overall TS search approach
    # ══════════════════════════════════════════════════════════════
    strategy = getattr(args, "strategy", "legacy")
    early_return_result = None  # for strategies that skip NEB entirely (e.g. IRC)
    start = None
    end = None

    def _get_cv_atom_indices():
        """Resolve (P, nuc, LG) atom indices from the ligand bond-breaking defs.
        Returns (idx_p, idx_nuc, idx_lg) or raises if atoms can't be located."""
        if ligand not in BOND_BREAKING_DEFS:
            raise ValueError(f"Ligand '{ligand}' not in BOND_BREAKING_DEFS — "
                             "cannot derive CV atom indices. Add the ligand "
                             "definition to BOND_BREAKING_DEFS in run_neb_ts.py.")
        defs = BOND_BREAKING_DEFS[ligand]
        nuc_name = next((d[1] for d in defs if d[3] == "attractive"), None)
        lg_name = next((d[1] for d in defs if d[3] == "repulsive"), None)
        p_name = defs[0][0]
        if not (nuc_name and lg_name):
            raise ValueError(f"Ligand '{ligand}' bond defs missing attractive or repulsive entries")

        def _find(aname):
            match = np.where((bt_struct.res_name == ligand) & (bt_struct.atom_name == aname))[0]
            if len(match) != 1:
                raise ValueError(f"Atom '{aname}' not found uniquely in residue '{ligand}' "
                                 f"(got {len(match)} matches)")
            return int(match[0])

        idx_p, idx_nuc, idx_lg = _find(p_name), _find(nuc_name), _find(lg_name)
        # Basic sanity: distinct indices, geometry plausible
        if len({idx_p, idx_nuc, idx_lg}) != 3:
            raise ValueError("P, nucleophile, and leaving-group must be distinct atoms")
        d_p_nuc = float(ase_atoms.get_distance(idx_p, idx_nuc))
        d_p_lg = float(ase_atoms.get_distance(idx_p, idx_lg))
        if d_p_nuc > 6.0 or d_p_lg > 6.0:
            log.warning(f"  CV atoms far from P: d(P-nuc)={d_p_nuc:.2f}, d(P-LG)={d_p_lg:.2f} — "
                        "check atom assignments")
        return idx_p, idx_nuc, idx_lg, nuc_name, lg_name

    # --- IRC-from-TS strategy: input is a TS guess, bypass endpoint generation ---
    if strategy == "irc":
        log.info("=" * 60)
        log.info("STRATEGY: IRC-from-TS (Sella saddle + IRC descent)")
        log.info("=" * 60)
        from quantum_engine.mlff.irc import irc_from_ts_guess

        ase_atoms.calc = make_calc(for_neb=True)
        ase_atoms.info["charge"] = charge

        # Build bond-pattern hint so reactant/product side assignment is not arbitrary
        hint_bonds = None
        try:
            idx_p, idx_nuc, idx_lg, _, _ = _get_cv_atom_indices()
            hint_bonds = {
                (idx_p, idx_nuc): "broken",   # nuc far in reactant
                (idx_p, idx_lg): "bonded",    # LG intact in reactant
            }
        except Exception as e:
            log.info(f"  No bond hints for IRC side assignment ({e})")

        irc_result = irc_from_ts_guess(
            ase_atoms, outdir=ts_dir, constraint=opt_c,
            saddle_fmax=args.fmax_sella,
            irc_step=args.irc_step,
            irc_fmax=args.fmax_end_final,
            reactant_hint_bonds=hint_bonds,
        )

        start = irc_result.get("reactant")
        end = irc_result.get("product")
        if start is None or end is None:
            log.error("  IRC failed to produce both endpoints — falling back to legacy spring")
            strategy = "legacy"
            start = end = None
        else:
            early_return_result = {
                "ts": irc_result["ts"],
                "reactant": start,
                "product": end,
                "barrier_fwd_kcal": irc_result.get("barrier_fwd_kcal"),
                "barrier_rev_kcal": irc_result.get("barrier_rev_kcal"),
                "imag_freq_cm": irc_result.get("imag_freq_cm"),
            }

    # --- MTD-rescue strategy ---
    if strategy == "mtd" and early_return_result is None:
        log.info("=" * 60)
        log.info("STRATEGY: Metadynamics rescue (WT-MTD on bond-difference CV)")
        log.info("=" * 60)
        try:
            from quantum_engine.mlff.metadynamics import run_metadynamics_rescue
            ase_atoms.calc = make_calc(for_neb=True)
            ase_atoms.info["charge"] = charge
            idx_p, idx_nuc, idx_lg, _, _ = _get_cv_atom_indices()
            mtd_result = run_metadynamics_rescue(
                ase_atoms,
                p_idx=idx_p, nuc_idx=idx_nuc, lg_idx=idx_lg,
                calculator=ase_atoms.calc,
                outdir=relax_dir,
                constraint=opt_c,
                total_time_ps=args.mtd_time_ps,
            )
            start = mtd_result.get("reactant")
            end = mtd_result.get("product")
        except Exception as e:
            log.error(f"  MTD failed: {e} — falling back to legacy spring")
            start = end = None

        if start is None or end is None:
            log.error("  MTD did not find both basins — falling back to legacy spring")
            strategy = "legacy"
            start = end = None

    # --- CV-spring strategy ---
    if strategy == "cv-spring" and early_return_result is None:
        log.info("=" * 60)
        log.info("STRATEGY: Bond-difference CV spring (More O'Ferrall-Jencks coord)")
        log.info("=" * 60)
        try:
            from quantum_engine.mlff.cv_spring import BondDifferenceCVSpring
            idx_p, idx_nuc, idx_lg, _, _ = _get_cv_atom_indices()

            def _cv_drive(s_target, label):
                a = ase_atoms.copy()
                a.calc = make_calc(for_neb=False)
                a.info["charge"] = charge
                cv_spring = BondDifferenceCVSpring(
                    p_idx=idx_p, nuc_idx=idx_nuc, lg_idx=idx_lg,
                    k=args.spring_k, s_target=s_target, fmax=args.spring_fmax,
                )
                a.set_constraint([opt_c, cv_spring] if opt_c else [cv_spring])
                from ase.optimize import LBFGS as _LBFGS
                log.info(f"  CV-drive ({label}, s_target={s_target:+.1f}) ...")
                opt = _LBFGS(a, logfile=os.path.join(relax_dir, f"cv-drive-{label}.log"))
                opt.run(fmax=args.fmax_end_spring, steps=300)
                a.set_constraint([opt_c] if opt_c else [])
                opt2 = _LBFGS(a, logfile=os.path.join(relax_dir, f"cv-polish-{label}.log"))
                opt2.run(fmax=args.fmax_end_final, steps=200)
                return a

            start = _cv_drive(args.cv_s_reactant, "reactant")
            end = _cv_drive(args.cv_s_product, "product")
        except Exception as e:
            log.error(f"  CV-spring strategy failed: {e} — falling back to legacy spring")
            strategy = "legacy"
            start = end = None

    # --- Legacy: spring-driven endpoints (explicit guard: only run if no other strategy set start/end) ---
    legacy_needed = early_return_result is None and (start is None or end is None)
    if legacy_needed and endpoint_method in ("spring", "auto"):
        start, end = generate_endpoints(
            ase_atoms, bt_struct, ligand, opt_c, md_c, relax_dir,
            spring_k=args.spring_k, spring_fmax=args.spring_fmax,
            fmax_spring=args.fmax_end_spring, fmax_final=args.fmax_end_final,
            md_steps=args.md_steps, md_temp=args.md_temp,
            unidirectional=args.unidirectional,
            spring_mode=args.spring_mode,
            md_strategy=args.md_strategy,
            n_md_seeds=args.n_md_seeds,
            anneal_peak=args.anneal_peak,
        )
        legacy_needed = False
    elif legacy_needed and endpoint_method == "constrained-scan":
        try:
            from quantum_engine.mlff.endpoint_generation import constrained_endpoint
            log.info("  Using constrained-scan endpoint generation ...")
            start = constrained_endpoint(
                ase_atoms.copy(), bt_struct, ligand, "reactant",
                make_calc, relax_dir, opt_c, fmax=args.fmax_end_final,
            )
            end = constrained_endpoint(
                ase_atoms.copy(), bt_struct, ligand, "product",
                make_calc, relax_dir, opt_c, fmax=args.fmax_end_final,
            )
        except ImportError:
            log.warning("  qcb.mlff.endpoint_generation not found, falling back to spring method")
            start, end = generate_endpoints(
                ase_atoms, bt_struct, ligand, opt_c, md_c, relax_dir,
                spring_k=args.spring_k, spring_fmax=args.spring_fmax,
                fmax_spring=args.fmax_end_spring, fmax_final=args.fmax_end_final,
                md_steps=args.md_steps, md_temp=args.md_temp,
            )
    elif legacy_needed and endpoint_method == "incremental-stretch":
        try:
            from quantum_engine.mlff.endpoint_generation import incremental_stretch_endpoints
            log.info("  Using incremental-stretch endpoint generation ...")
            start, end = incremental_stretch_endpoints(
                ase_atoms.copy(), bt_struct, ligand,
                make_calc, relax_dir, opt_c, n_steps=10, fmax=args.fmax_end_final,
            )
        except ImportError:
            log.warning("  qcb.mlff.endpoint_generation not found, falling back to spring method")
            start, end = generate_endpoints(
                ase_atoms, bt_struct, ligand, opt_c, md_c, relax_dir,
                spring_k=args.spring_k, spring_fmax=args.spring_fmax,
                fmax_spring=args.fmax_end_spring, fmax_final=args.fmax_end_final,
                md_steps=args.md_steps, md_temp=args.md_temp,
            )
    # Safety check: if we're not in early-return (IRC) mode, we must have endpoints
    if early_return_result is None and (start is None or end is None):
        raise RuntimeError(
            f"Endpoint generation failed: start={start is not None}, end={end is not None}. "
            f"Strategy={strategy}, endpoint_method={endpoint_method}. "
            "Check logs above for the failing step."
        )

    # Switch to primary model (--model) for energy evaluation and NEB/TS
    if early_return_result is None:
        start.calc = make_calc(for_neb=True)
        start.info["charge"] = charge
        end.calc = make_calc(for_neb=True)
        end.info["charge"] = charge

        # ── Optional: xTB endpoint refinement (validates MACE endpoints with semi-empirical QM) ──
        if getattr(args, "refine_xtb", False):
            from quantum_engine.qm.xtb_refine import validate_endpoint_pair
            log.info("  Running xTB endpoint refinement ...")
            p_name = BOND_BREAKING_DEFS[ligand][0][0] if ligand in BOND_BREAKING_DEFS else None
            nuc_name = next((d[1] for d in BOND_BREAKING_DEFS.get(ligand, []) if d[3] == "attractive"), None)
            lg_name = next((d[1] for d in BOND_BREAKING_DEFS.get(ligand, []) if d[3] == "repulsive"), None)

            idx_p = idx_nuc = idx_lg = None
            if p_name and nuc_name and lg_name:
                idx_p = int(np.where((bt_struct.res_name == ligand) & (bt_struct.atom_name == p_name))[0][0])
                idx_nuc = int(np.where((bt_struct.res_name == ligand) & (bt_struct.atom_name == nuc_name))[0][0])
                idx_lg = int(np.where((bt_struct.res_name == ligand) & (bt_struct.atom_name == lg_name))[0][0])

            xtb_check = validate_endpoint_pair(
                start, end, charge, outdir=os.path.join(relax_dir, "xtb_refine"),
                constraint=opt_c, p_idx=idx_p, nuc_idx=idx_nuc, lg_idx=idx_lg,
            )
            if xtb_check["inconsistent"]:
                log.warning("  *** xTB disagrees with MACE endpoints — geometries may be artifacts ***")
            # Replace with xTB-refined geometries if user opted into update
            if getattr(args, "refine_xtb_replace", False):
                if xtb_check["reactant"]["atoms"] is not None:
                    start.set_positions(xtb_check["reactant"]["atoms"].get_positions())
                if xtb_check["product"]["atoms"] is not None:
                    end.set_positions(xtb_check["product"]["atoms"].get_positions())
                log.info("  Endpoints replaced with xTB-refined geometries")

    # ── Energy check ──
    if early_return_result is None:
        e_start = start.get_potential_energy()
        e_end = end.get_potential_energy()
        log.info(f"  E(start) = {e_start:.4f} eV ({e_start * EV_TO_KCAL:.1f} kcal/mol)")
        log.info(f"  E(end)   = {e_end:.4f} eV ({e_end * EV_TO_KCAL:.1f} kcal/mol)")
        log.info(f"  ΔE(rxn)  = {(e_end - e_start) * EV_TO_KCAL:.1f} kcal/mol")

    # ── Endpoint geometry validation ──
    if early_return_result is None and ligand in BOND_BREAKING_DEFS:
        defs = BOND_BREAKING_DEFS[ligand]
        log.info("  Endpoint bond distance validation:")
        valid = True
        for atom1, atom2, target_r, mode in defs:
            idx1 = np.where((bt_struct.res_name == ligand) & (bt_struct.atom_name == atom1))[0]
            idx2 = np.where((bt_struct.res_name == ligand) & (bt_struct.atom_name == atom2))[0]
            if len(idx1) == 1 and len(idx2) == 1:
                i1, i2 = int(idx1[0]), int(idx2[0])
                d_start = start.get_distance(i1, i2)
                d_end = end.get_distance(i1, i2)
                ok_start = (mode == "attractive" and d_start > 2.5) or (mode == "repulsive" and d_start < 1.8)
                ok_end = (mode == "attractive" and d_end < 1.8) or (mode == "repulsive" and d_end > 2.5)
                status_s = "✓" if ok_start else "✗"
                status_e = "✓" if ok_end else "✗"
                log.info(f"    {atom1}-{atom2} ({mode}): start={d_start:.2f} {status_s}  end={d_end:.2f} {status_e}")
                if not ok_start or not ok_end:
                    valid = False
        if not valid:
            log.warning("  *** ENDPOINT VALIDATION FAILED: bonds not properly separated! ***")
            log.warning("  *** NEB barrier may be unreliable. Check input geometry. ***")

    # ── Step 3: NEB (always uses primary --model) ──
    def make_neb_calc():
        return make_calc(for_neb=True)

    if early_return_result is not None:
        # IRC strategy: skip NEB, use IRC result directly
        log.info("  NEB skipped — using IRC-from-TS result")
        ts = early_return_result["ts"]
        images = [start, ts, end]  # minimal 3-image path for output compatibility
    else:
        images = run_neb(
            ase_atoms, start, end, opt_c, make_neb_calc, charge, ts_dir,
            n_images=args.n_images, k_spring=args.k_spring,
            fmax_noclimb=args.fmax_neb_noclimb, steps_noclimb=args.steps_noclimb,
            fmax_climb=args.fmax_neb_climb if do_climb else 999,
            steps_climb=args.steps_climb if do_climb else 0,
            interpolation_method=args.interpolation,
        )

    # ── Step 4: Sella TS (optional) ──
    if early_return_result is not None:
        # IRC strategy already refined TS with Sella saddle search
        ts = early_return_result["ts"]
        log.info("  Sella already completed in IRC-from-TS strategy")
    elif do_sella:
        ts = refine_ts_sella(
            images, make_neb_calc, charge, opt_c, ts_dir,
            fmax=args.fmax_sella, max_steps=args.steps_sella,
        )
    else:
        # Use highest-energy NEB image directly as TS estimate
        inner_E = [img.get_potential_energy() for img in images[1:-1]]
        ts_idx = int(np.argmax(inner_E)) + 1
        ts = images[ts_idx].copy()
        ts.calc = make_calc(for_neb=True)
        ts.info["charge"] = charge
        log.info(f"  Using CI-NEB image {ts_idx} as TS (Sella skipped)")
        write(os.path.join(ts_dir, "ts-neb-highest.xyz"), ts)

    # ── Step 5: Frequency validation (optional) ──
    freq_result = None
    if do_freq:
        freq_result = validate_ts(ts, opt_c, ts_dir)

    # ── Summary ──
    elapsed = time.time() - t0
    e_ts = ts.get_potential_energy()
    barrier = e_ts - e_start

    log.info("")
    log.info("=" * 60)
    barrier_fwd_kcal = barrier * EV_TO_KCAL
    barrier_rev_kcal = (e_ts - e_end) * EV_TO_KCAL
    dE_rxn_kcal = (e_end - e_start) * EV_TO_KCAL

    log.info("PIPELINE COMPLETE")
    log.info(f"  Time:          {elapsed/60:.1f} min")
    log.info(f"  Barrier (fwd): {barrier_fwd_kcal:.1f} kcal/mol")
    log.info(f"  Barrier (rev): {barrier_rev_kcal:.1f} kcal/mol")
    log.info(f"  ΔE(rxn):       {dE_rxn_kcal:.1f} kcal/mol")
    if freq_result:
        log.info(f"  Imaginary freq: {freq_result['n_imaginary']}")
    log.info(f"  TS structure:  {os.path.join(ts_dir, 'path-after-sella-ts.xyz')}")
    log.info(f"  NEB path:      {os.path.join(ts_dir, 'path-neb-climb.xyz')}")
    log.info("=" * 60)

    # ── Organize outputs: user-facing PDBs + technical subdirectory ──
    log.info("Organizing outputs ...")
    _organize_outputs(
        outdir_base, relax_dir, ts_dir, bt_struct, start, end, ts, images,
        system_name=system_name,
        charge_method=args.charge_method,
    )

    # Save summary JSON
    summary = {
        "system": system_name,
        "model": model_tag,
        "mode": mode,
        "n_atoms": meta["n_atoms"],
        "charge": charge,
        "ligand": ligand,
        "barrier_fwd_kcal": barrier_fwd_kcal,
        "barrier_rev_kcal": barrier_rev_kcal,
        "dE_rxn_kcal": dE_rxn_kcal,
        "e_start_eV": e_start,
        "e_end_eV": e_end,
        "e_ts_eV": e_ts,
        "elapsed_min": elapsed / 60,
        "freq_validation": freq_result,
    }
    with open(os.path.join(outdir_base, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(
        description="NEB Transition State Search for Enzyme Active Sites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_neb_ts.py input.pdb
  python run_neb_ts.py input.pdb --model mace-mp
  python run_neb_ts.py input.pdb --model /path/to/custom.model --n-images 11
  python run_neb_ts.py input.pdb --skip-freq   # skip slow frequency calc
        """,
    )
    p.add_argument("pdb_file", help="Input PDB file (enzyme active-site cluster)")

    # Model
    p.add_argument(
        "--model", default="mace-mp",
        help="MACE model for NEB/TS: mace-mp, mace-omol, mace-off[-small/-medium], "
             "mace-mh, mace-polar[-s/-m/-l], or path to .model file (default: mace-mp)",
    )
    p.add_argument(
        "--model-relax", default=None,
        help="Cheaper model for relaxation/MD phases only (e.g. mace-mp). "
             "If set, --model is used only for NEB/TS. Saves GPU time on expensive models.",
    )
    p.add_argument(
        "--head", default=None,
        help="For multi-head models (mace-mh): which DFT head to use. "
             "Options: rgd1_b3lyp, matpes_r2scan, omol, spice_wB97M, etc. "
             "(default: omol for mace-mh)",
    )

    # Output
    p.add_argument("--outdir", default=None, help="Output directory (auto-generated if omitted)")

    # Charge (explicit override — highest priority)
    p.add_argument("--charge", type=int, default=None,
                   help="System net charge (overrides PDB REMARK and filename). "
                        "If not specified, reads from PDB REMARK QCB TOTAL_CHARGE "
                        "or filename netCHG pattern.")
    p.add_argument("--charge-method", default="auto",
                   choices=["auto", "mulliken", "eem", "formal"],
                   help="Per-atom charge method for CIF output: "
                        "auto = xTB Mulliken → EEM → electronegativity (default); "
                        "mulliken = xTB only; eem = OpenBabel EEM only (fast); "
                        "formal = integer charges from residue identity (instant)")

    # NEB parameters
    p.add_argument("--n-images", type=int, default=15, help="Number of NEB images (default: 15)")
    p.add_argument("--k-spring", type=float, default=1.0, help="NEB spring constant (default: 1.0)")
    p.add_argument("--fmax-neb-noclimb", type=float, default=0.40, help="NEB no-climb fmax (default: 0.40)")
    p.add_argument("--steps-noclimb", type=int, default=200, help="NEB no-climb max steps (default: 200)")
    p.add_argument("--fmax-neb-climb", type=float, default=0.045, help="NEB climbing fmax (default: 0.045)")
    p.add_argument("--steps-climb", type=int, default=250, help="NEB climbing max steps (default: 250)")

    # Sella parameters
    p.add_argument("--fmax-sella", type=float, default=0.02, help="Sella TS fmax (default: 0.02)")
    p.add_argument("--steps-sella", type=int, default=1000, help="Sella max steps (default: 1000)")

    # Relaxation parameters (used for both endpoints)
    p.add_argument("--fmax-end-spring", type=float, default=0.10, help="Spring-driven relaxation fmax (default: 0.10)")
    p.add_argument("--fmax-end-final", type=float, default=0.04, help="Final polish fmax for both endpoints (default: 0.04)")
    p.add_argument("--spring-k", type=float, default=3.0, help="Bond-breaking spring constant (default: 3.0; tested: k=3 optimal, k=2 too soft, k=6+ too aggressive)")
    p.add_argument("--spring-fmax", type=float, default=3.0, help="Bond-breaking spring force cap (default: 3.0)")

    # MD parameters
    p.add_argument("--md-steps", type=int, default=200, help="Langevin MD steps for endpoint equilibration (default: 200, critical for proper minima)")
    p.add_argument("--md-temp", type=float, default=300.0, help="MD temperature in K (default: 300)")
    p.add_argument(
        "--md-strategy", default="short",
        choices=["short", "annealing", "multi-seed", "long"],
        help="MD equilibration strategy: "
             "short = single 200-step MD (default, fast); "
             "annealing = heat to --anneal-peak then cool (better exploration); "
             "multi-seed = N independent short MDs, pick lowest energy (best for conformational sampling); "
             "long = single long MD (--md-steps controls length, use 5000+ for thorough)"
    )
    p.add_argument("--n-md-seeds", type=int, default=5,
                   help="Number of independent MD seeds for multi-seed strategy (default: 5)")
    p.add_argument("--anneal-peak", type=float, default=600.0,
                   help="Peak temperature for simulated annealing in K (default: 600)")

    # Spring mode for endpoint generation
    p.add_argument(
        "--spring-mode", default="both",
        choices=["both", "nuc-only", "lg-only"],
        help="Which bonds to drive with springs during endpoint generation: "
             "both = drive nucleophile AND leaving group (default, may bias concerted); "
             "nuc-only = only attract nucleophile to P (LG responds naturally); "
             "lg-only = only push leaving group away (nuc responds naturally). "
             "For unbiased mechanistic investigation, use nuc-only or lg-only."
    )

    # Endpoint generation method
    p.add_argument(
        "--endpoint-method", default="spring",
        choices=["spring", "constrained-scan", "incremental-stretch", "auto"],
        help="How to generate reactant/product endpoints: "
             "spring = spring-driven relaxation (current, fast but biased); "
             "constrained-scan = fix bond at target distance, minimize rest (rigorous); "
             "incremental-stretch = incrementally change bond in steps (traces MEP); "
             "auto = use covalent radii for smart targets + spring method. "
             "For gold-standard: constrained-scan or incremental-stretch."
    )

    # Mode
    p.add_argument(
        "--mode", default="standard",
        choices=["quick", "standard", "full"],
        help="Pipeline depth: "
             "quick = NEB only (no climb, no Sella, fastest screening), "
             "standard = NEB + CI-NEB (default, good TS estimate from climbing image), "
             "full = NEB + CI-NEB + Sella + freq validation (publication quality)"
    )

    # Flags (override mode defaults)
    p.add_argument("--skip-freq", action="store_true", help="Skip frequency validation")
    p.add_argument("--skip-sella", action="store_true", help="Skip Sella refinement (use CI-NEB TS)")
    p.add_argument("--force-sella", action="store_true", help="Force Sella even in quick/standard mode")
    p.add_argument("--pre-relax", action="store_true",
                   help="Pre-relax entire structure with NO constraints before NEB. "
                        "Use for messy inputs with clashes (e.g. docked/chimeric structures)")
    p.add_argument(
        "--constraint-mode", default="ca-only",
        choices=["ca-only", "backbone", "backbone-water", "ca-restrained", "none"],
        help="Constraint strategy: "
             "ca-only = fix only CA (default, sidechains/waters free); "
             "backbone = fix CA/C/N/O during opt, CA-only during MD; "
             "backbone-water = like backbone but also pin water O during opt; "
             "ca-restrained = CA fixed + harmonic restraints on isolated residue termini; "
             "none = no constraints"
    )
    p.add_argument(
        "--fix-chains", nargs="*", default=None,
        help="Only apply constraints to atoms in these chain(s). "
             "Atoms in other chains are completely free. "
             "E.g., --fix-chains B fixes only chain B backbone, leaving chain A free."
    )
    p.add_argument("--unidirectional", action="store_true",
                   help="Only drive bonds FORWARD (to product). Use the relaxed input as "
                        "reactant instead of reverse-driving. Best for inputs that are "
                        "already near reactant geometry or chimeric structures where "
                        "reverse driving creates bad states.")

    # ── New strategy-level flags (see docs/strategies.md) ──
    p.add_argument(
        "--strategy", default="legacy",
        choices=["legacy", "irc", "cv-spring", "mtd"],
        help="Overall TS search strategy (default: legacy). "
             "legacy = spring-driven endpoints + NEB (original pipeline); "
             "irc = Sella saddle + IRC descent both ways (gold standard when input is TS guess); "
             "cv-spring = bond-difference CV spring (More O'Ferrall-Jencks coord); "
             "mtd = well-tempered metadynamics rescue (when all else fails, 1-4 GPU-hours)"
    )
    p.add_argument(
        "--interpolation", default="geodesic",
        choices=["geodesic", "idpp", "linear"],
        help="NEB image interpolation method (default: geodesic). "
             "geodesic = Zhu et al. JCP 2019 (recommended, avoids atom clashes); "
             "idpp = image-dependent pair potential (ASE built-in); "
             "linear = Cartesian linear (fast but breaks for bond-forming reactions)"
    )
    p.add_argument(
        "--refine-xtb", action="store_true",
        help="After endpoint generation, run xTB GFN2 optimization on both endpoints "
             "as an independent QM sanity check. Detects MACE hallucinations and "
             "collapsed product basins. Cost: minutes. Default: off."
    )
    p.add_argument(
        "--refine-xtb-replace", action="store_true",
        help="If --refine-xtb is set, replace MACE endpoints with xTB-refined "
             "geometries (rather than just reporting the comparison). Default: off."
    )
    p.add_argument(
        "--irc-step", type=float, default=0.1,
        help="Displacement along imaginary mode for IRC (Å, default: 0.1). "
             "Larger = faster convergence but may miss shallow intermediates."
    )
    p.add_argument(
        "--cv-s-reactant", type=float, default=-2.0,
        help="CV-spring: target s = d(P-LG) - d(P-nuc) for reactant (Å). "
             "Default: -2.0 (strong nuc dissociation, LG intact)"
    )
    p.add_argument(
        "--cv-s-product", type=float, default=2.5,
        help="CV-spring: target s for product (Å). "
             "Default: +2.5 (nuc bonded, LG dissociated)"
    )
    p.add_argument(
        "--mtd-time-ps", type=float, default=100.0,
        help="Metadynamics total simulation time (ps). Default: 100 ps (1-4 GPU-hours)"
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
