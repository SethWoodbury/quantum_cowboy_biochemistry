#!/usr/bin/env python
"""
Generate CIF files with proper bond types for completed NEB-TS runs.

Runs xTB single-point for Mulliken charges + Wiberg bond orders,
then uses atomworks to write CIF with proper aromatic bond types.

This is a post-processing step — no GPU needed, runs on CPU.

Usage:
  python scripts/generate_cifs.py /path/to/output_dir
  python scripts/generate_cifs.py /path/to/parent_dir --all
"""

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate-cifs")

XTB_BIN = "/home/dme5188/bin/xtb/xtb-6.6.1/bin/xtb"
EV_TO_KCAL = 23.0609


def run_xtb_singlepoint(atoms, total_charge, workdir):
    """Run xTB GFN2 single-point, return (mulliken, wbo_dict)."""
    n_atoms = len(atoms)
    if not os.path.isfile(XTB_BIN):
        log.warning("  xTB not found at %s", XTB_BIN)
        return None, None
    if n_atoms >= 500:
        log.warning("  System too large for xTB (%d atoms)", n_atoms)
        return None, None

    tmpdir = Path(workdir) / ".xtb_cif_gen"
    tmpdir.mkdir(exist_ok=True)

    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    with open(tmpdir / "ts.xyz", "w") as f:
        f.write(f"{n_atoms}\n\n")
        for sym, (x, y, z) in zip(symbols, positions):
            f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")

    timeout_s = min(1800, max(120, n_atoms * 2))
    try:
        result = subprocess.run(
            [XTB_BIN, "ts.xyz", "--gfn", "2", "--chrg", str(total_charge), "--sp"],
            capture_output=True, text=True, timeout=timeout_s, cwd=str(tmpdir),
        )
    except subprocess.TimeoutExpired:
        log.warning("  xTB timed out (%ds)", timeout_s)
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None, None

    mulliken = None
    bond_orders = None

    charges_file = tmpdir / "charges"
    wbo_file = tmpdir / "wbo"

    if charges_file.exists():
        vals = [float(l) for l in charges_file.read_text().strip().split("\n")]
        if len(vals) == n_atoms:
            mulliken = np.array(vals)
            log.info("  xTB Mulliken charges: sum=%.3f, range=[%.3f, %.3f]",
                     mulliken.sum(), mulliken.min(), mulliken.max())

    if wbo_file.exists():
        bond_orders = {}
        for line in wbo_file.read_text().strip().split("\n"):
            parts = line.split()
            if len(parts) >= 3:
                i, j = int(parts[0]) - 1, int(parts[1]) - 1
                bo = float(parts[2])
                if bo > 0.3:
                    bond_orders[(min(i, j), max(i, j))] = bo
        log.info("  xTB WBO: %d bonds detected", len(bond_orders))

    shutil.rmtree(tmpdir, ignore_errors=True)
    return mulliken, bond_orders


def derive_formal_charges(atoms, mulliken, bond_orders, total_charge):
    """Derive formal integer charges from Mulliken + WBO."""
    n_atoms = len(atoms)
    formal = np.zeros(n_atoms)

    for i in range(n_atoms):
        q = mulliken[i]
        z = atoms.get_atomic_numbers()[i]
        total_bo = sum(bo for (a, b), bo in bond_orders.items() if a == i or b == i)

        if z == 8:
            if total_bo < 1.5 and q < -0.3:
                formal[i] = -1
            elif total_bo > 2.5:
                formal[i] = +1
        elif z == 7:
            if total_bo > 3.5 and q > 0.2:
                formal[i] = +1
            elif total_bo < 2.5 and q < -0.3:
                formal[i] = -1

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
    log.info("  Formal charges: sum=%+d, %d charged atoms",
             int(formal.sum()), n_charged)
    return formal


def load_template(pdb_path):
    """Load biotite AtomArray template from PDB."""
    try:
        import biotite.structure.io.pdb as pdb_io
        pdb_file = pdb_io.PDBFile.read(str(pdb_path))
        return pdb_io.get_structure(pdb_file, model=1)
    except Exception as e:
        log.warning("  Could not load biotite template: %s", e)
        return None


def extract_charge_from_pdb(pdb_path):
    """Extract total charge from PDB REMARK line."""
    with open(pdb_path) as f:
        for line in f:
            if "TOTAL_CHARGE" in line:
                m = re.search(r"TOTAL_CHARGE\s+([-\d]+)", line)
                if m:
                    return int(m.group(1))
    # Try from filename
    m = re.search(r"netCHG_(plus|minus)_(\d+)", str(pdb_path))
    if m:
        sign = 1 if m.group(1) == "plus" else -1
        return sign * int(m.group(2))
    return 0


def generate_cif(ts_pdb, outdir, cif_filename=None):
    """Generate CIF for a transition state PDB."""
    if not ts_pdb.exists():
        log.warning("  No TS PDB found at %s", ts_pdb)
        return False

    log.info("Processing: %s", ts_pdb.name)

    # Load atoms and template
    ts_atoms = ase_read(str(ts_pdb))
    template_st = load_template(ts_pdb)
    total_charge = extract_charge_from_pdb(ts_pdb)
    ts_atoms.info["charge"] = total_charge

    log.info("  %d atoms, charge=%+d", len(ts_atoms), total_charge)

    # Run xTB
    mulliken, wbo = run_xtb_singlepoint(ts_atoms, total_charge, outdir)

    if mulliken is None:
        log.warning("  Skipping CIF (xTB failed)")
        return False

    # Derive formal charges
    formal = derive_formal_charges(ts_atoms, mulliken, wbo or {}, total_charge)

    # Determine output filename
    if cif_filename is None:
        cif_filename = ts_pdb.stem + ".cif"

    # Try atomworks CIF writer
    try:
        from quantum_engine.mlff.cif_writer import write_ts_cif
        result = write_ts_cif(
            ts_atoms, template_st, str(outdir),
            total_charge=total_charge,
            partial_charges=mulliken,
            formal_charges=formal,
            wiberg_bond_orders=wbo,
            filename=cif_filename,
        )
        if result:
            log.info("  Wrote %s (atomworks CIF with bonds + charges)", result)
            return True
    except ImportError:
        log.info("  atomworks not available, using simple CIF")
    except Exception as e:
        log.warning("  atomworks CIF failed (%s), using simple CIF", e)

    # Fallback: simple text CIF
    cif_path = outdir / cif_filename
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
        f.write(f"_qcb.total_charge  {total_charge}\n")
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

        n_atoms = len(ts_atoms)
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
                f"{mulliken[i]:.4f}",
                f"{int(formal[i])}",
            ])
            f.write(" ".join(parts) + "\n")

    log.info("  Wrote %s (simple CIF with charges)", cif_path)
    return True


def process_directory(outdir):
    """Find TS PDB in an output directory and generate CIF."""
    outdir = Path(outdir)
    if not outdir.is_dir():
        return False

    # Find transition_state.pdb (may have system_name prefix)
    ts_files = sorted(outdir.glob("*transition_state.pdb"))
    if not ts_files:
        log.info("  No transition_state.pdb in %s", outdir.name)
        return False

    success = True
    for ts_pdb in ts_files:
        cif_name = ts_pdb.stem + ".cif"
        if not generate_cif(ts_pdb, outdir, cif_name):
            success = False

    return success


def main():
    parser = argparse.ArgumentParser(description="Generate CIF files for NEB-TS results")
    parser.add_argument("path", help="Output directory or parent directory")
    parser.add_argument("--all", action="store_true",
                        help="Process all subdirectories")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing CIF files")
    args = parser.parse_args()

    path = Path(args.path)

    if args.all:
        dirs = sorted([d for d in path.iterdir() if d.is_dir()
                       and d.name not in ("technical", "logs")])
    else:
        dirs = [path]

    n_success = 0
    n_total = 0
    for d in dirs:
        # Skip if CIF already exists and not overwriting
        existing_cifs = list(d.glob("*.cif"))
        if existing_cifs and not args.overwrite:
            log.info("Skipping %s (CIF exists, use --overwrite)", d.name)
            continue

        n_total += 1
        if process_directory(d):
            n_success += 1

    log.info("\nGenerated CIFs for %d / %d directories", n_success, n_total)


if __name__ == "__main__":
    main()
