"""Structure loading and writing utilities.

Provides a consistent interface for PDB/XYZ/CIF input and PDB output,
preserving biotite annotations (residue names, chain IDs, etc.) alongside
the ASE Atoms object.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

log = logging.getLogger("qcb.io")

EV_TO_KCAL = 23.0609


def load_structure(path: str | Path) -> tuple[Atoms, "AtomArray | None", int | None]:
    """Load a structure file, return (ase_atoms, biotite_atomarray, charge_hint).

    - For PDB: uses biotite to preserve residue/chain annotations
    - For XYZ/CIF: uses ASE; biotite_atomarray will be None
    - Charge hint is extracted from REMARK QCB TOTAL_CHARGE or from filename
      pattern netCHG_plus_N / netCHG_minus_N

    Returns:
        ase_atoms: ASE Atoms object (always)
        bt_struct: biotite AtomArray (for PDB with annotations), else None
        charge_hint: int from PDB REMARK or filename, else None
    """
    path = Path(path)
    ase_atoms = read(str(path))

    bt_struct = None
    charge_hint = None
    if path.suffix.lower() == ".pdb":
        try:
            import biotite.structure.io.pdb as pdb_io
            pdb_file = pdb_io.PDBFile.read(str(path))
            bt_struct = pdb_io.get_structure(pdb_file, model=1)
        except Exception as e:
            log.warning(f"biotite parse failed ({e}); continuing with ASE only")

        # Extract charge from REMARK
        with open(path) as f:
            for line in f:
                if "TOTAL_CHARGE" in line:
                    m = re.search(r"TOTAL_CHARGE\s+([-\d]+)", line)
                    if m:
                        charge_hint = int(m.group(1))
                    break

    # Charge from filename fallback
    if charge_hint is None:
        m = re.search(r"netCHG_(plus|minus)_(\d+)", path.name)
        if m:
            sign = 1 if m.group(1) == "plus" else -1
            charge_hint = sign * int(m.group(2))

    return ase_atoms, bt_struct, charge_hint


def write_pdb(
    atoms: Atoms,
    template: "AtomArray | None",
    path: str | Path,
    total_charge: int | None = None,
    energy_eV: float | None = None,
):
    """Write an ASE Atoms to PDB.

    If a biotite template is provided, residue names/chain IDs/atom names are
    preserved. Otherwise falls back to ASE's writer. Adds REMARK lines with
    QCB metadata (charge, energy).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if template is not None and len(template) == len(atoms):
        _write_pdb_with_template(atoms, template, path, total_charge, energy_eV)
    else:
        from ase.io import write as ase_write
        ase_write(str(path), atoms, format="proteindatabank")
        _append_remarks(path, total_charge, energy_eV)


def _write_pdb_with_template(
    atoms: Atoms, template, path: Path,
    total_charge: int | None, energy_eV: float | None,
):
    """Write PDB using biotite template for annotations and ASE for coordinates.

    Uses biotite's PDB writer which handles the atom-name alignment rules
    (single-letter elements get padded " CA ", two-letter elements left-flush "FE  ").
    """
    import biotite.structure.io.pdb as pdb_io

    # Update template coordinates from ASE
    updated = template.copy()
    updated.coord = atoms.get_positions().astype(np.float32)

    pdb_file = pdb_io.PDBFile()
    pdb_file.set_structure(updated)

    # biotite writes to str; prepend our REMARK lines
    tmp = path.with_suffix(".tmp.pdb")
    pdb_file.write(str(tmp))

    existing = tmp.read_text()
    header = []
    if total_charge is not None:
        header.append(f"REMARK   2 QCB TOTAL_CHARGE {total_charge:+d}")
    if energy_eV is not None:
        header.append(f"REMARK   2 QCB ENERGY_EV {energy_eV:.6f}")
        header.append(f"REMARK   2 QCB ENERGY_KCAL {energy_eV * EV_TO_KCAL:.2f}")

    if header:
        # Insert after HEADER line (if present) or at the top
        lines = existing.split("\n")
        if lines and lines[0].startswith("HEADER"):
            out = [lines[0]] + header + lines[1:]
        else:
            out = header + lines
        path.write_text("\n".join(out))
    else:
        path.write_text(existing)
    tmp.unlink()


def _append_remarks(path: Path, total_charge: int | None, energy_eV: float | None):
    """Append REMARK lines to an existing PDB."""
    if total_charge is None and energy_eV is None:
        return
    with open(path) as f:
        content = f.read()
    remarks = []
    if total_charge is not None:
        remarks.append(f"REMARK QCB TOTAL_CHARGE {total_charge:+d}")
    if energy_eV is not None:
        remarks.append(f"REMARK QCB ENERGY_EV {energy_eV:.6f}")
    # Insert after first line
    lines = content.split("\n")
    new = [lines[0]] + remarks + lines[1:]
    path.write_text("\n".join(new))


def write_trajectory_xyz(
    frames: list[Atoms],
    path: str | Path,
    energies: list[float] | None = None,
):
    """Write a multi-frame extended-XYZ trajectory."""
    from ase.io import write as ase_write
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if energies is not None:
        for frame, e in zip(frames, energies):
            frame.info["energy_eV"] = e
    ase_write(str(path), frames, format="extxyz")
