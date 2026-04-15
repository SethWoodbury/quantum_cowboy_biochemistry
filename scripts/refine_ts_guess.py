#!/usr/bin/env python3
"""
Refine a transition state guess using ML force fields.

For systems where you already have a TS guess (from DFT, docking, or NEB),
this script refines it directly using Sella saddle-point optimization,
then validates with IRC (intrinsic reaction coordinate).

Workflow:
  1. FIRE relaxation (robust, handles strained geometries)
  2. LBFGS polish (fast convergence near minimum)
  3. Sella saddle-point optimization (internal coordinates for <200 atoms)
  4. IRC forward + reverse (verify TS connects reactant ↔ product)
  5. Frequency analysis (confirm exactly 1 imaginary frequency)

Designed for small systems (<300 atoms) like theozymes and active site clusters
where you can afford Sella with internal coordinates and full IRC.

Usage:
  python refine_ts_guess.py input.pdb --charge 1 --model mace-omol
  python refine_ts_guess.py input.pdb --charge 1 --model mace-omol --skip-irc
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from ase import Atoms, units
from ase.io import read, write
from ase.optimize import FIRE, LBFGS
from ase.constraints import FixAtoms

from matplotlib import pyplot as plt
from sella import Sella, IRC

import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io

from mace.calculators import MACECalculator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("refine-ts")

EV_TO_KCAL = 23.0609

# ══════════════════════════════════════════════════════════════
#  MODEL LOADING (reuse from run_neb_ts.py)
# ══════════════════════════════════════════════════════════════

MODEL_PATHS = {
    "mace-mp": "/mnt/projects/ml/mlff/models/mace_mp/MACE-matpes-r2scan-omat-ft.model",
    "mace-omol": "/home/gbg222/projects/mace_models/MACE-omol-0-extra-large-1024.model",
    "mace-mh": "/home/gbg222/projects/mace_models/mace-mh-0.model",
    "mace-polar-m": "/home/gbg222/projects/mace_models/MACE-POLAR-1-M.model",
}

MH_DEFAULT_HEADS = {"mace-mh": "omol"}


def get_calculator(model_key, device="cuda", dtype="float64", head=None):
    kwargs = dict(device=device, default_dtype=dtype)
    if head is None and model_key in MH_DEFAULT_HEADS:
        head = MH_DEFAULT_HEADS[model_key]
    if head:
        kwargs["head"] = head

    if os.path.isfile(model_key):
        return MACECalculator(model_paths=model_key, **kwargs)

    if model_key in MODEL_PATHS and os.path.isfile(MODEL_PATHS[model_key]):
        log.info(f"Loading '{model_key}'" + (f" (head={head})" if head else ""))
        return MACECalculator(model_paths=MODEL_PATHS[model_key], **kwargs)

    # Auto-download
    try:
        if "omol" in model_key:
            from mace.calculators import mace_omol
            return mace_omol(model="extra_large", device=device, default_dtype=dtype)
        elif "mp" in model_key:
            from mace.calculators import mace_mp
            return mace_mp(device=device, default_dtype=dtype)
    except Exception as e:
        log.warning(f"Auto-download failed: {e}")

    raise FileNotFoundError(f"Model '{model_key}' not found")


# ══════════════════════════════════════════════════════════════
#  STRUCTURE I/O
# ══════════════════════════════════════════════════════════════


def load_pdb(pdb_path):
    """Load PDB → ASE Atoms + biotite template."""
    pdb_file = pdb_io.PDBFile.read(str(pdb_path))
    st = pdb_file.get_structure(model=1)
    symbols = [e.capitalize() for e in st.element]
    atoms = Atoms(symbols=symbols, positions=st.coord)
    atoms.arrays.update(st._annot)
    return atoms, st


def write_pdb(atoms, template_st, path):
    """Write ASE Atoms as PDB using template for atom naming."""
    out = template_st.copy()
    out.coord = atoms.get_positions().astype(np.float32)
    pdb_file = pdb_io.PDBFile()
    pdb_file.set_structure(out)
    Path(path).write_text(str(pdb_file))


# ══════════════════════════════════════════════════════════════
#  STEP 1: FIRE + LBFGS PRE-RELAXATION
# ══════════════════════════════════════════════════════════════


def prerelax(atoms, outdir, fmax_fire=0.5, fmax_lbfgs=0.1, max_steps_fire=100, max_steps_lbfgs=200):
    """Two-stage pre-relaxation: FIRE (robust) → LBFGS (efficient).

    FIRE handles strained starting geometries better than LBFGS alone.
    We stop early — just get to a reasonable geometry, not a minimum.
    """
    log.info("Step 1: Pre-relaxation (FIRE → LBFGS) ...")

    e0 = atoms.get_potential_energy()
    fmax0 = np.max(np.linalg.norm(atoms.get_forces(), axis=1))
    log.info(f"  Initial: E={e0 * EV_TO_KCAL:.1f} kcal/mol, fmax={fmax0:.3f} eV/A")

    # FIRE stage
    fire = FIRE(atoms, logfile=os.path.join(outdir, "prerelax_fire.log"),
                trajectory=os.path.join(outdir, "prerelax_fire.traj"))
    fire.run(fmax=fmax_fire, steps=max_steps_fire)
    e1 = atoms.get_potential_energy()
    log.info(f"  After FIRE: E={e1 * EV_TO_KCAL:.1f} kcal/mol")

    # LBFGS stage
    lbfgs = LBFGS(atoms, logfile=os.path.join(outdir, "prerelax_lbfgs.log"),
                  trajectory=os.path.join(outdir, "prerelax_lbfgs.traj"))
    lbfgs.run(fmax=fmax_lbfgs, steps=max_steps_lbfgs)
    e2 = atoms.get_potential_energy()
    fmax2 = np.max(np.linalg.norm(atoms.get_forces(), axis=1))
    log.info(f"  After LBFGS: E={e2 * EV_TO_KCAL:.1f} kcal/mol, fmax={fmax2:.3f} eV/A")

    return atoms


# ══════════════════════════════════════════════════════════════
#  STEP 2: SELLA SADDLE-POINT OPTIMIZATION
# ══════════════════════════════════════════════════════════════


def sella_ts(atoms, outdir, fmax=0.01, max_steps=1000, internal=True):
    """Refine TS guess using Sella with internal coordinates.

    For <200 atoms, internal=True gives ~2x faster convergence.
    Uses order=1 (first-order saddle point = transition state).
    """
    log.info(f"Step 2: Sella TS optimization (internal={internal}, fmax={fmax}) ...")

    e0 = atoms.get_potential_energy()
    log.info(f"  Starting E={e0 * EV_TO_KCAL:.1f} kcal/mol")

    traj_path = os.path.join(outdir, "sella_ts.traj")
    log_path = os.path.join(outdir, "sella_ts.log")

    try:
        dyn = Sella(atoms, internal=internal, order=1,
                    trajectory=traj_path, logfile=log_path)
        dyn.run(fmax=fmax, steps=max_steps)
    except Exception as e:
        if internal:
            log.warning(f"  Sella internal failed ({e}), retrying Cartesian ...")
            dyn = Sella(atoms, internal=False, order=1,
                        trajectory=traj_path, logfile=log_path)
            dyn.run(fmax=fmax, steps=max_steps)
        else:
            raise

    e_ts = atoms.get_potential_energy()
    fmax_final = np.max(np.linalg.norm(atoms.get_forces(), axis=1))
    log.info(f"  TS energy: {e_ts * EV_TO_KCAL:.1f} kcal/mol, fmax={fmax_final:.4f} eV/A")

    return atoms


# ══════════════════════════════════════════════════════════════
#  STEP 3: IRC (INTRINSIC REACTION COORDINATE)
# ══════════════════════════════════════════════════════════════


def run_irc(atoms, calc_fn, outdir, fmax=0.02, max_steps=200):
    """Run IRC forward and reverse from the TS to find reactant and product.

    After IRC finds the approximate endpoint, FIRE+LBFGS relaxation is applied
    to fully optimize the reactant and product geometries in their basins.
    """
    log.info("Step 3: IRC (intrinsic reaction coordinate) ...")

    results = {}
    for direction in ["forward", "reverse"]:
        label = "reactant" if direction == "forward" else "product"
        log.info(f"  IRC {direction} → {label} ...")
        irc_atoms = atoms.copy()
        irc_atoms.calc = calc_fn()

        traj_path = os.path.join(outdir, f"irc_{direction}.traj")
        log_path = os.path.join(outdir, f"irc_{direction}.log")

        try:
            irc = IRC(irc_atoms, trajectory=traj_path, logfile=log_path,
                      dx=0.1, eta=1e-4, gamma=0.4)
            irc.run(fmax=fmax, steps=max_steps, direction=direction)

            e_irc = irc_atoms.get_potential_energy()
            log.info(f"    IRC endpoint: E={e_irc * EV_TO_KCAL:.1f} kcal/mol")

            # Relax the IRC endpoint to a proper minimum (FIRE → LBFGS)
            log.info(f"    Relaxing {label} in basin (FIRE → LBFGS) ...")
            irc_atoms.calc = calc_fn()

            fire = FIRE(irc_atoms, logfile=os.path.join(outdir, f"{label}_fire.log"))
            fire.run(fmax=0.1, steps=100)

            lbfgs = LBFGS(irc_atoms, logfile=os.path.join(outdir, f"{label}_lbfgs.log"))
            lbfgs.run(fmax=0.02, steps=300)

            e_relaxed = irc_atoms.get_potential_energy()
            fmax_relaxed = np.max(np.linalg.norm(irc_atoms.get_forces(), axis=1))
            log.info(f"    {label} relaxed: E={e_relaxed * EV_TO_KCAL:.1f} kcal/mol, "
                     f"fmax={fmax_relaxed:.4f} eV/A")

            results[direction] = irc_atoms.copy()
        except Exception as e:
            log.warning(f"    IRC {direction} failed: {e}")
            results[direction] = None

    return results


# ══════════════════════════════════════════════════════════════
#  STEP 4: FREQUENCY ANALYSIS
# ══════════════════════════════════════════════════════════════


def frequency_analysis(atoms, outdir):
    """Compute vibrational frequencies to confirm exactly 1 imaginary."""
    log.info("Step 4: Frequency analysis ...")

    from ase.vibrations import Vibrations

    vib_dir = os.path.join(outdir, "vib")
    vib = Vibrations(atoms, name=vib_dir, delta=0.01)
    vib.run()

    freqs = vib.get_frequencies()
    imag = [f for f in freqs if np.iscomplex(f) and abs(f.imag) > 10]
    real = [f.real for f in freqs if not np.iscomplex(f) or abs(f.imag) < 10]

    # Write frequencies
    with open(os.path.join(outdir, "frequencies.dat"), "w") as fh:
        fh.write("# Vibrational frequencies (cm-1)\n")
        for f in sorted(freqs, key=lambda x: x.real):
            if np.iscomplex(f) and abs(f.imag) > 10:
                fh.write(f"  {f.imag:10.1f}i\n")
            else:
                fh.write(f"  {f.real:10.1f}\n")

    if len(imag) == 1:
        log.info(f"  TS CONFIRMED: 1 imaginary frequency at {imag[0].imag:.1f}i cm-1")
    elif len(imag) == 0:
        log.warning("  NO imaginary frequencies — this is a minimum, not a TS!")
    else:
        log.warning(f"  {len(imag)} imaginary frequencies — higher-order saddle point")
        for f in imag:
            log.warning(f"    {f.imag:.1f}i cm-1")

    return {"n_imaginary": len(imag), "imaginary_freqs": [f.imag for f in imag]}


# ══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════


def main():
    p = argparse.ArgumentParser(description="Refine a TS guess with MLFF + Sella + IRC")
    p.add_argument("input_pdb", help="Input PDB (TS guess)")
    p.add_argument("--charge", type=int, required=True, help="Total system charge")
    p.add_argument("--model", default="mace-omol", help="MACE model (default: mace-omol)")
    p.add_argument("--head", default=None, help="Multi-head model head")
    p.add_argument("--outdir", default=None, help="Output directory")
    p.add_argument("--fmax-prerelax", type=float, default=0.1, help="Pre-relax fmax (default: 0.1)")
    p.add_argument("--fmax-sella", type=float, default=0.01, help="Sella fmax (default: 0.01)")
    p.add_argument("--skip-prerelax", action="store_true", help="Skip FIRE+LBFGS pre-relaxation")
    p.add_argument("--skip-irc", action="store_true", help="Skip IRC validation")
    p.add_argument("--skip-freq", action="store_true", help="Skip frequency analysis")
    args = p.parse_args()

    t0 = time.time()
    input_pdb = Path(args.input_pdb)
    system_name = input_pdb.stem

    # Output directory
    if args.outdir:
        outdir = args.outdir
    else:
        model_tag = Path(args.model).stem if os.path.isfile(args.model) else args.model
        outdir = f"outputs/ts_refine_{system_name}_{model_tag}"
    os.makedirs(outdir, exist_ok=True)

    # Set up logging to file
    fh = logging.FileHandler(os.path.join(outdir, "refine.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    log.info("=" * 60)
    log.info("TS Refinement Pipeline")
    log.info(f"  Input:  {input_pdb}")
    log.info(f"  Model:  {args.model}")
    log.info(f"  Charge: {args.charge:+d}")
    log.info(f"  Output: {outdir}")
    log.info("=" * 60)

    # Load structure
    atoms, template_st = load_pdb(input_pdb)
    atoms.info["charge"] = args.charge
    log.info(f"  {len(atoms)} atoms loaded")

    # Calculator
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def make_calc():
        return get_calculator(args.model, device=device, dtype="float64", head=args.head)

    atoms.calc = make_calc()

    # Step 1: Pre-relaxation
    if not args.skip_prerelax:
        prerelax(atoms, outdir, fmax_lbfgs=args.fmax_prerelax)
        write_pdb(atoms, template_st, os.path.join(outdir, "prerelaxed.pdb"))
        write(os.path.join(outdir, "prerelaxed.xyz"), atoms)

    # Step 2: Sella TS optimization
    use_internal = len(atoms) < 200
    ts = sella_ts(atoms, outdir, fmax=args.fmax_sella, internal=use_internal)
    write_pdb(ts, template_st, os.path.join(outdir, "transition_state.pdb"))
    write(os.path.join(outdir, "transition_state.xyz"), ts)
    e_ts = ts.get_potential_energy()

    # Step 3: IRC
    irc_results = {}
    if not args.skip_irc:
        irc_results = run_irc(ts, make_calc, outdir)
        for direction, irc_atoms in irc_results.items():
            if irc_atoms is not None:
                label = "reactant" if direction == "forward" else "product"
                write_pdb(irc_atoms, template_st, os.path.join(outdir, f"{label}.pdb"))
                write(os.path.join(outdir, f"{label}.xyz"), irc_atoms)

    # Step 4: Frequency analysis
    freq_result = None
    if not args.skip_freq:
        freq_result = frequency_analysis(ts, outdir)

    # Summary
    elapsed = time.time() - t0

    e_reactant = irc_results.get("forward")
    e_product = irc_results.get("reverse")

    log.info("")
    log.info("=" * 60)
    log.info("TS REFINEMENT COMPLETE")
    log.info(f"  Time: {elapsed / 60:.1f} min")
    log.info(f"  E(TS): {e_ts * EV_TO_KCAL:.1f} kcal/mol")
    if e_reactant is not None:
        e_r = e_reactant.get_potential_energy()
        log.info(f"  E(reactant): {e_r * EV_TO_KCAL:.1f} kcal/mol")
        log.info(f"  Barrier (fwd): {(e_ts - e_r) * EV_TO_KCAL:.1f} kcal/mol")
    if e_product is not None:
        e_p = e_product.get_potential_energy()
        log.info(f"  E(product): {e_p * EV_TO_KCAL:.1f} kcal/mol")
        log.info(f"  Barrier (rev): {(e_ts - e_p) * EV_TO_KCAL:.1f} kcal/mol")
    if freq_result:
        log.info(f"  Imaginary frequencies: {freq_result['n_imaginary']}")
    log.info(f"  TS structure: {outdir}/transition_state.pdb")
    log.info("=" * 60)

    # Save summary
    summary = {
        "system": system_name,
        "model": args.model,
        "charge": args.charge,
        "n_atoms": len(atoms),
        "e_ts_kcal": e_ts * EV_TO_KCAL,
        "elapsed_min": elapsed / 60,
    }
    if e_reactant is not None:
        summary["e_reactant_kcal"] = e_reactant.get_potential_energy() * EV_TO_KCAL
        summary["barrier_fwd_kcal"] = (e_ts - e_reactant.get_potential_energy()) * EV_TO_KCAL
    if e_product is not None:
        summary["e_product_kcal"] = e_product.get_potential_energy() * EV_TO_KCAL
        summary["barrier_rev_kcal"] = (e_ts - e_product.get_potential_energy()) * EV_TO_KCAL
    if freq_result:
        summary["freq_validation"] = freq_result

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
