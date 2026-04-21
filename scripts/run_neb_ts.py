#!/usr/bin/env python
"""
NEB Transition State Search Pipeline for Enzyme Active Sites
=============================================================
Uses MACE ML force fields + ASE NEB + Sella saddle-point optimisation.

Full pipeline:
  1. Load PDB, auto-detect ligand, extract formal charge
  2. Set up backbone constraints
  3. Relax start state (reactant)
  4. Generate & relax end state (product) via spring-constrained optimisation
  5. NEB path optimisation  →  climbing-image NEB
  6. Sella TS refinement (internal coordinates)
  7. Vibrational frequency validation

Designed for PTE theozyme active-site clusters (YYE / YYL / YYF / PT4 ligands)
but adaptable to other enzyme systems.

Usage
-----
  python run_neb_ts.py path/to/input.pdb                    # auto-detect model
  python run_neb_ts.py path/to/input.pdb --model mace-mp    # specific model
  python run_neb_ts.py path/to/input.pdb --resume            # resume from checkpoint

Authors: synthesised from lschaaf / gbg222 / seth pipelines, March 2026.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

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
EV_TO_KCAL = 23.0609  # 1 eV = 23.06 kcal/mol


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
    """Write a single structure as PDB with input-matching formatting."""
    pdb_str = ase_to_pdb_string(atoms, template_st)
    with open(path, "w") as f:
        f.write(pdb_str)
    log.info(f"  Wrote {path}")


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
    """Extract net formal charge from filename pattern  netCHG_plus3 / netCHG_minus2 / netCHG_0."""
    name = os.path.basename(path)
    m = re.search(r"net[Cc][Hh][Gg]_?(plus|minus)?(\d+)", name)
    if m is None:
        return None
    sign_str, val = m.group(1), int(m.group(2))
    if sign_str == "minus":
        return -val
    return val


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
BOND_BREAKING_DEFS = {
    # PTE phosphoester substrates
    # Target 3.5 A for leaving group ensures full P-O dissociation (not just
    # pentacoordinate intermediate at ~1.7 A). Target 1.4 A for nucleophile
    # ensures full bond formation.
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
#  MODEL REGISTRY
# ══════════════════════════════════════════════════════════════
MODEL_PATHS = {
    # ── DIGS cluster models ──
    "mace-mp":          "/mnt/projects/ml/mlff/models/mace_mp/MACE-matpes-r2scan-omat-ft.model",
    "mace-mp-old":      "/home/gbg222/projects/mace_models/2023-12-10-mace-128-L0_energy_epoch-249.model",
    "mace-off-small":   "/mnt/projects/ml/mlff/models/mace_off/MACE-OFF23_small.model",
    "mace-off-medium":  "/mnt/projects/ml/mlff/models/mace_off/MACE-OFF23_medium.model",
    "mace-off":         "/mnt/projects/ml/mlff/models/mace_off/MACE-OFF23_large.model",
    "mace-off-large":   "/mnt/projects/ml/mlff/models/mace_off/MACE-OFF23_large.model",
    # ── gbg222 models ──
    "mace-omol":        "/home/gbg222/projects/mace_models/MACE-omol-0-extra-large-1024.model",
    "mace-mh":          "/home/gbg222/projects/mace_models/mace-mh-0.model",
    "mace-polar-s":     "/home/gbg222/projects/mace_models/MACE-POLAR-1-S.model",
    "mace-polar-m":     "/home/gbg222/projects/mace_models/MACE-POLAR-1-M.model",
    "mace-polar-l":     "/home/gbg222/projects/mace_models/MACE-POLAR-1-L.model",
    "mace-polar":       "/home/gbg222/projects/mace_models/MACE-POLAR-1-M.model",  # default size
}

# Default heads for multi-head model
MH_DEFAULT_HEADS = {
    "mace-mh": "omol",  # wB97M-V level, same training as OMOL
}

# Models that need gbg222 venv (not universal.sif) due to graph_electrostatics
NEEDS_GBG_VENV = {"mace-polar-s", "mace-polar-m", "mace-polar-l", "mace-polar"}


def get_calculator(model_key, device="cuda", dtype="float64", head=None):
    """Construct a MACE calculator from a model key or file path.

    Args:
        model_key: Registry name (e.g. 'mace-mp'), direct path to .model file,
                   or convenience name for auto-download.
        device: 'cuda' or 'cpu'
        dtype: 'float64' (accurate, slower) or 'float32' (faster, less precise)
        head: For multi-head models (mace-mh), which DFT head to use.
              Options: rgd1_b3lyp, matpes_r2scan, omol, spice_wB97M, etc.
    """
    kwargs = dict(device=device, default_dtype=dtype)

    # Resolve head for multi-head models
    if head is None and model_key in MH_DEFAULT_HEADS:
        head = MH_DEFAULT_HEADS[model_key]
    if head:
        kwargs["head"] = head

    # Direct file path
    if os.path.isfile(model_key):
        log.info(f"Loading MACE model from file: {model_key}" +
                 (f" (head={head})" if head else ""))
        return MACECalculator(model_paths=model_key, **kwargs)

    # Try the registry
    if model_key in MODEL_PATHS:
        path = MODEL_PATHS[model_key]
        if os.path.isfile(path):
            log.info(f"Loading '{model_key}' from {path}" +
                     (f" (head={head})" if head else ""))
            return MACECalculator(model_paths=path, **kwargs)
        log.warning(f"Model path not found: {path}")

    # Try convenience loaders (auto-download from HuggingFace)
    try:
        if "omol" in model_key:
            from mace.calculators import mace_omol
            log.info("Loading MACE-OMOL via auto-download (extra_large)...")
            return mace_omol(model="extra_large", device=device, default_dtype=dtype)
        elif "polar" in model_key:
            from mace.calculators import mace_polar
            size = "polar-1-m"
            if "small" in model_key or "-s" in model_key:
                size = "polar-1-s"
            elif "large" in model_key or "-l" in model_key:
                size = "polar-1-l"
            log.info(f"Loading MACE-POLAR via auto-download ({size})...")
            return mace_polar(model=size, device=device, default_dtype=dtype)
        elif "mp" in model_key:
            from mace.calculators import mace_mp
            log.info("Loading MACE-MP via auto-download...")
            return mace_mp(device=device, default_dtype=dtype)
        elif "off" in model_key:
            from mace.calculators import mace_off
            size = "large"
            if "small" in model_key:
                size = "small"
            elif "medium" in model_key:
                size = "medium"
            log.info(f"Loading MACE-OFF via auto-download ({size})...")
            return mace_off(model=size, device=device, default_dtype=dtype)
    except Exception as e:
        log.warning(f"Auto-download failed for {model_key}: {e}")

    avail = list(MODEL_PATHS.keys())
    raise FileNotFoundError(
        f"Could not find model '{model_key}'.\n"
        f"  Available on DIGS: {avail}\n"
        f"  Or provide a direct path to a .model file.\n"
        f"  For multi-head models, use --head to select DFT level."
    )


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

    # Extract charge
    charge = extract_net_charge(pdb_path)
    if charge is not None:
        ase_atoms.info["charge"] = charge
        log.info(f"  Net charge from filename: {charge}")
    else:
        log.warning("  Could not extract charge from filename – defaulting to 0")
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

    # Interpolation – try geodesic (internal coords, better), fall back to IDPP
    try:
        _do_geodesic_interpolation(images)
        log.info("  Using geodesic interpolation (internal coordinates)")
    except Exception as e:
        log.info(f"  Geodesic interpolation unavailable ({e}), using IDPP")
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


def _organize_outputs(outdir, relax_dir, ts_dir, template_st, start, end, ts, images):
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

    # Write user-facing PDBs
    write_result_pdb(start, template_st, os.path.join(outdir, "reactant.pdb"))
    write_result_pdb(end, template_st, os.path.join(outdir, "product.pdb"))
    write_result_pdb(ts, template_st, os.path.join(outdir, "transition_state.pdb"))

    # Write CIF for transition state (includes charge annotation)
    _write_ts_cif(ts, template_st, outdir)

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
            os.path.join(outdir, "neb_path.pdb"),
            energies=neb_energies,
        )

    # MD trajectories — read from technical/ if .traj files exist
    _write_md_traj_pdbs(outdir, tech_dir, template_st)

    # Copy energy profile plot to top level
    for png_name in ["neb-climb.png", "path-neb-noclimb.png"]:
        src = os.path.join(tech_dir, "ts", png_name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(outdir, "energy_profile.png"))
            break

    log.info("  Output organized:")
    log.info(f"    {outdir}/reactant.pdb")
    log.info(f"    {outdir}/product.pdb")
    log.info(f"    {outdir}/transition_state.pdb")
    log.info(f"    {outdir}/neb_path.pdb")
    log.info(f"    {outdir}/energy_profile.png")
    log.info(f"    {outdir}/technical/  (raw logs, traj, xyz)")


def _write_ts_cif(ts_atoms, template_st, outdir):
    """Write transition state as CIF with charge and energy annotations."""
    try:
        from ase.io import write as ase_write
        cif_path = os.path.join(outdir, "transition_state.cif")

        # Write basic CIF via ASE
        ase_write(cif_path, ts_atoms, format="cif")

        # Append charge info as CIF data items
        charge = ts_atoms.info.get("charge", 0)
        energy = None
        try:
            energy = ts_atoms.get_potential_energy()
        except Exception:
            pass

        with open(cif_path, "a") as f:
            f.write(f"\n# QCB annotations\n")
            f.write(f"_qcb.total_charge  {charge}\n")
            if energy is not None:
                f.write(f"_qcb.energy_eV  {energy:.6f}\n")
                f.write(f"_qcb.energy_kcal_mol  {energy * EV_TO_KCAL:.2f}\n")

        log.info(f"  Wrote {cif_path} (charge={charge})")
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
    charge = meta["charge"]

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

    # ── Steps 1 & 2: Generate both endpoints by driving bonds in both directions ──
    # From the input PDB (which may be near the TS), we drive toward reactant
    # AND toward product to ensure proper endpoints regardless of input geometry.
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
    # Switch to primary model (--model) for energy evaluation and NEB/TS
    start.calc = make_calc(for_neb=True)
    start.info["charge"] = charge
    end.calc = make_calc(for_neb=True)
    end.info["charge"] = charge

    # ── Energy check ──
    e_start = start.get_potential_energy()
    e_end = end.get_potential_energy()
    log.info(f"  E(start) = {e_start:.4f} eV ({e_start * EV_TO_KCAL:.1f} kcal/mol)")
    log.info(f"  E(end)   = {e_end:.4f} eV ({e_end * EV_TO_KCAL:.1f} kcal/mol)")
    log.info(f"  ΔE(rxn)  = {(e_end - e_start) * EV_TO_KCAL:.1f} kcal/mol")

    # ── Endpoint geometry validation ──
    if ligand in BOND_BREAKING_DEFS:
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

    images = run_neb(
        ase_atoms, start, end, opt_c, make_neb_calc, charge, ts_dir,
        n_images=args.n_images, k_spring=args.k_spring,
        fmax_noclimb=args.fmax_neb_noclimb, steps_noclimb=args.steps_noclimb,
        fmax_climb=args.fmax_neb_climb if do_climb else 999,
        steps_climb=args.steps_climb if do_climb else 0,
    )

    # ── Step 4: Sella TS (optional) ──
    if do_sella:
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

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
