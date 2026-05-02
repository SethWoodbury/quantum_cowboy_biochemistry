"""
ORCA input file generation and modification.

Generates ``.inp`` input files for ORCA, with support for CPCM implicit
solvation and parallel execution.  Modification of existing input files
follows the same ``%maxcore`` / ``nprocs`` pattern used by Indrek's
``orcasub`` script (``/home/ikalvet/bin/orcasub``).
"""

from __future__ import annotations

from pathlib import Path

# ── Supported job type -> ORCA keyword mapping ──────────────────────
_JOB_TYPE_KEYWORDS: dict[str, str] = {
    "opt": "Opt",
    "opt+freq": "Opt Freq",
    "freq": "Freq",
    "ts": "OptTS",
    "sp": "",
}


def _read_xyz_block(path: Path) -> list[str]:
    """Read atom coordinate lines from an .xyz or .pdb file.

    Returns lines in ``ELEMENT  X  Y  Z`` format (no trailing newline).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".xyz":
        raw = path.read_text().splitlines()
        atoms: list[str] = []
        for line in raw[2:]:
            parts = line.split()
            if len(parts) >= 4:
                atoms.append(f"  {parts[0]:<2s}  {parts[1]:>14s}  {parts[2]:>14s}  {parts[3]:>14s}")
        return atoms

    if suffix == ".pdb":
        atoms = []
        for line in path.read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")):
                element = line[76:78].strip() if len(line) >= 78 else line[12:16].strip()[:2]
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atoms.append(f"  {element:<2s}  {x:14.8f}  {y:14.8f}  {z:14.8f}")
        return atoms

    raise ValueError(f"Unsupported file format '{suffix}'. Expected .xyz or .pdb.")


# ── Public API ───────────────────────────────────────────────────────


def write_orca_input(
    xyz_or_pdb: str | Path,
    output_inp: str | Path,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    job_type: str = "opt",
    nproc: int = 8,
    maxcore: int = 4000,
    extra_keywords: str = "",
    solvent: str | None = None,
) -> Path:
    """Generate an ORCA ``.inp`` input file from an XYZ or PDB structure.

    Parameters
    ----------
    xyz_or_pdb : str or Path
        Path to the source geometry (.xyz or .pdb).
    output_inp : str or Path
        Destination path for the .inp file.
    charge : int
        Molecular charge (default 0).
    multiplicity : int
        Spin multiplicity (default 1).
    method : str
        DFT functional or method keyword (default ``"B3LYP"``).
    basis : str
        Basis set (default ``"def2-SVP"``).
    job_type : str
        Calculation type. One of ``"opt"``, ``"opt+freq"``, ``"freq"``,
        ``"ts"``, ``"sp"``.
    nproc : int
        Number of parallel processes (default 8).
    maxcore : int
        Maximum memory per core in MB (default 4000).
    extra_keywords : str
        Additional keywords appended to the ``!`` line.
    solvent : str or None
        If given, enables CPCM solvation with the specified solvent
        (e.g. ``"water"``, ``"toluene"``).

    Returns
    -------
    Path
        Absolute path to the written .inp file.

    Raises
    ------
    ValueError
        If *job_type* is not recognised.
    FileNotFoundError
        If *xyz_or_pdb* does not exist.
    """
    xyz_or_pdb = Path(xyz_or_pdb)
    output_inp = Path(output_inp)

    if not xyz_or_pdb.exists():
        raise FileNotFoundError(f"Structure file not found: {xyz_or_pdb}")

    jt = job_type.lower()
    if jt not in _JOB_TYPE_KEYWORDS:
        raise ValueError(
            f"Unknown job_type '{job_type}'. "
            f"Supported: {', '.join(_JOB_TYPE_KEYWORDS)}"
        )

    atom_lines = _read_xyz_block(xyz_or_pdb)
    if not atom_lines:
        raise ValueError(f"No atom coordinates found in {xyz_or_pdb}")

    # Build the ! keyword line
    kw_parts = [method, basis]
    job_kw = _JOB_TYPE_KEYWORDS[jt]
    if job_kw:
        kw_parts.append(job_kw)
    if solvent is not None:
        kw_parts.append(f"CPCM({solvent})")
    if extra_keywords.strip():
        kw_parts.append(extra_keywords.strip())
    keyword_line = "! " + " ".join(kw_parts)

    lines: list[str] = [
        f"{keyword_line}\n",
        f"%maxcore {maxcore}\n",
        "%pal\n",
        f"  nprocs {nproc}\n",
        "end\n",
        "\n",
        f"* xyz {charge} {multiplicity}\n",
    ]
    for atom in atom_lines:
        lines.append(atom + "\n")
    lines.append("*\n")
    lines.append("\n")

    output_inp.parent.mkdir(parents=True, exist_ok=True)
    output_inp.write_text("".join(lines))
    return output_inp.resolve()


def modify_orca_input(
    inp_file: str | Path,
    nproc: int | None = None,
    maxcore: int | None = None,
) -> Path:
    """Modify resource headers (%maxcore, nprocs) in an existing ORCA .inp file.

    Follows the same pattern as Indrek's ``fixInfile()`` in
    ``/home/ikalvet/bin/orcasub``.

    Parameters
    ----------
    inp_file : str or Path
        Path to the ORCA .inp file to modify in-place.
    nproc : int or None
        New value for ``nprocs`` inside the ``%pal`` block.
    maxcore : int or None
        New value for ``%maxcore`` (memory per core in MB).

    Returns
    -------
    Path
        Absolute path to the modified file.
    """
    inp_file = Path(inp_file)
    data = inp_file.read_text().splitlines(keepends=True)

    new_lines: list[str] = []
    for line in data:
        stripped = line.strip().lower()
        if stripped.startswith("%maxcore") and maxcore is not None:
            new_lines.append(f"%maxcore {maxcore}\n")
        elif stripped.startswith("nprocs") and nproc is not None:
            new_lines.append(f"  nprocs {nproc}\n")
        else:
            new_lines.append(line)

    inp_file.write_text("".join(new_lines))
    return inp_file.resolve()
