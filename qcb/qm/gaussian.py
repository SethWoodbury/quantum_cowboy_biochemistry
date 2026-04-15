"""
Gaussian 16 input file generation, modification, and output parsing.

Generates .com input files for Gaussian 16, modifies resource headers in
existing files, and extracts optimized geometries from .log output files.

Reference scripts:
  - Indrek's gauss: /home/ikalvet/bin/gauss  (header fixup + SLURM submission)
  - Declan's g16_setup.py: /home/dme5188/scripts/g16/g16_setup.py  (output->input)
"""

from __future__ import annotations

import re
from pathlib import Path

from qcb.config import GAUSSIAN_ROOT

# ── Supported job type → route keyword mapping ──────────────────────
_JOB_TYPE_KEYWORDS: dict[str, str] = {
    "opt": "Opt",
    "opt+freq": "Opt Freq",
    "freq": "Freq",
    "ts": "Opt=(TS,CalcFC,NoEigenTest)",
    "irc": "IRC=(CalcFC,MaxPoints=50)",
    "irc_forward": "IRC=(CalcFC,Forward,MaxPoints=50)",
    "irc_reverse": "IRC=(CalcFC,Reverse,MaxPoints=50)",
    "sp": "",
}


def _read_xyz_block(path: Path) -> tuple[list[str], int]:
    """Read atom lines from an .xyz or .pdb file.

    Parameters
    ----------
    path : Path
        Path to a .xyz or .pdb file.

    Returns
    -------
    lines : list[str]
        Atom lines in ``ELEMENT  X  Y  Z`` format, each ending with ``\\n``.
    n_atoms : int
        Number of atoms read.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".xyz":
        raw = path.read_text().splitlines()
        # Standard XYZ: first line = natoms, second line = comment, rest = atoms
        atom_lines: list[str] = []
        for line in raw[2:]:
            parts = line.split()
            if len(parts) >= 4:
                atom_lines.append(f"{parts[0]:<2s}  {parts[1]:>14s}  {parts[2]:>14s}  {parts[3]:>14s}\n")
        return atom_lines, len(atom_lines)

    if suffix == ".pdb":
        atom_lines = []
        for line in path.read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")):
                element = line[76:78].strip() if len(line) >= 78 else line[12:16].strip()[:2]
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atom_lines.append(f"{element:<2s}  {x:14.8f}  {y:14.8f}  {z:14.8f}\n")
        return atom_lines, len(atom_lines)

    raise ValueError(f"Unsupported file format '{suffix}'. Expected .xyz or .pdb.")


# ── Public API ───────────────────────────────────────────────────────


def write_gaussian_input(
    xyz_or_pdb: str | Path,
    output_com: str | Path,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "B3LYP",
    basis: str = "6-31+G(d)",
    job_type: str = "opt",
    nproc: int = 8,
    mem: str = "16gb",
    extra_keywords: str = "",
    title: str = "QCB job",
) -> Path:
    """Generate a Gaussian 16 ``.com`` input file from an XYZ or PDB structure.

    Parameters
    ----------
    xyz_or_pdb : str or Path
        Path to the source geometry (.xyz or .pdb).
    output_com : str or Path
        Destination path for the .com file.
    charge : int
        Molecular charge (default 0).
    multiplicity : int
        Spin multiplicity (default 1).
    method : str
        DFT functional or method keyword (default ``"B3LYP"``).
    basis : str
        Basis set (default ``"6-31+G(d)"``).
    job_type : str
        Calculation type. One of ``"opt"``, ``"opt+freq"``, ``"freq"``,
        ``"ts"``, ``"irc"``, ``"irc_forward"``, ``"irc_reverse"``, ``"sp"``.
    nproc : int
        Number of shared processors (default 8).
    mem : str
        Memory allocation, e.g. ``"16gb"`` (default ``"16gb"``).
    extra_keywords : str
        Additional route-line keywords appended after the job type.
    title : str
        Title line written to the .com file (default ``"QCB job"``).

    Returns
    -------
    Path
        Absolute path to the written .com file.

    Raises
    ------
    ValueError
        If *job_type* is not recognised.
    FileNotFoundError
        If *xyz_or_pdb* does not exist.
    """
    xyz_or_pdb = Path(xyz_or_pdb)
    output_com = Path(output_com)

    if not xyz_or_pdb.exists():
        raise FileNotFoundError(f"Structure file not found: {xyz_or_pdb}")

    jt = job_type.lower()
    if jt not in _JOB_TYPE_KEYWORDS:
        raise ValueError(
            f"Unknown job_type '{job_type}'. "
            f"Supported: {', '.join(_JOB_TYPE_KEYWORDS)}"
        )

    atom_lines, _ = _read_xyz_block(xyz_or_pdb)
    if not atom_lines:
        raise ValueError(f"No atom coordinates found in {xyz_or_pdb}")

    chk_name = output_com.stem + ".chk"
    job_kw = _JOB_TYPE_KEYWORDS[jt]

    # Build route line
    route_parts = [f"#p {method}/{basis}"]
    if job_kw:
        route_parts.append(job_kw)
    if extra_keywords.strip():
        route_parts.append(extra_keywords.strip())
    route_line = " ".join(route_parts)

    lines: list[str] = [
        f"%NProcShared={nproc}\n",
        f"%Mem={mem.upper()}\n",
        f"%Chk={chk_name}\n",
        f"{route_line}\n",
        "\n",
        f"{title}\n",
        "\n",
        f"{charge} {multiplicity}\n",
    ]
    lines.extend(atom_lines)
    # Gaussian requires a blank line after coordinates and at EOF
    lines.append("\n")
    lines.append("\n")

    output_com.parent.mkdir(parents=True, exist_ok=True)
    output_com.write_text("".join(lines))
    return output_com.resolve()


def modify_gaussian_input(
    com_file: str | Path,
    nproc: int | None = None,
    mem: str | None = None,
    chk: str | None = None,
) -> Path:
    """Modify resource headers (%NProcShared, %Mem, %Chk) in an existing .com file.

    Only the headers whose corresponding argument is not ``None`` are changed.
    Follows the same logic as Indrek's ``fixInfile()`` in ``/home/ikalvet/bin/gauss``.

    Parameters
    ----------
    com_file : str or Path
        Path to the Gaussian .com file to modify in-place.
    nproc : int or None
        New value for ``%NProcShared``.
    mem : str or None
        New value for ``%Mem`` (e.g. ``"32gb"``).
    chk : str or None
        New value for ``%Chk`` (e.g. ``"newjob.chk"``).

    Returns
    -------
    Path
        Absolute path to the modified file.
    """
    com_file = Path(com_file)
    data = com_file.read_text().splitlines(keepends=True)

    new_lines: list[str] = []
    for line in data:
        low = line.lower()
        if low.startswith("%nproc") and nproc is not None:
            new_lines.append(f"%NProcShared={nproc}\n")
        elif low.startswith("%mem") and mem is not None:
            new_lines.append(f"%Mem={mem.upper()}\n")
        elif low.startswith("%chk") and chk is not None:
            new_lines.append(f"%Chk={chk}\n")
        else:
            new_lines.append(line)

    com_file.write_text("".join(new_lines))
    return com_file.resolve()


def gaussian_output_to_xyz(
    log_file: str | Path,
    output_xyz: str | Path,
) -> Path:
    """Extract the last (optimised) geometry from a Gaussian .log file.

    Scans for ``Standard orientation:`` blocks and takes the final one.

    Parameters
    ----------
    log_file : str or Path
        Path to the Gaussian .log / .out file.
    output_xyz : str or Path
        Destination .xyz file.

    Returns
    -------
    Path
        Absolute path to the written .xyz file.

    Raises
    ------
    RuntimeError
        If no ``Standard orientation`` block is found in the log file.
    """
    log_file = Path(log_file)
    output_xyz = Path(output_xyz)

    # Atomic-number -> element symbol (covers first 103 elements)
    _ELEMENTS = {
        1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N",
        8: "O", 9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si",
        15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
        21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe",
        27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge",
        33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr",
        39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru",
        45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
        51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba",
        57: "La", 58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm",
        63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er",
        69: "Tm", 70: "Yb", 71: "Lu", 72: "Hf", 73: "Ta", 74: "W",
        75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg",
        81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn",
        87: "Fr", 88: "Ra", 89: "Ac", 90: "Th", 91: "Pa", 92: "U",
        93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf",
        99: "Es", 100: "Fm", 101: "Md", 102: "No", 103: "Lr",
    }

    text = log_file.read_text()

    # Find all "Standard orientation:" blocks
    orient_pattern = re.compile(
        r"Standard orientation:.*?\n"
        r"\s*-+\n"            # dashes
        r"\s*Center.*?\n"     # header
        r"\s*Number.*?\n"     # sub-header
        r"\s*-+\n"            # dashes
        r"(.*?)\n"            # atom data (captured)
        r"\s*-+",             # closing dashes
        re.DOTALL,
    )
    matches = list(orient_pattern.finditer(text))
    if not matches:
        raise RuntimeError(
            f"No 'Standard orientation' block found in {log_file}. "
            "The calculation may not have completed."
        )

    # Parse the last block
    block = matches[-1].group(1)
    atoms: list[str] = []
    for line in block.strip().splitlines():
        parts = line.split()
        if len(parts) >= 6:
            atomic_num = int(parts[1])
            x, y, z = parts[3], parts[4], parts[5]
            elem = _ELEMENTS.get(atomic_num, "X")
            atoms.append(f"{elem:<2s}  {x:>14s}  {y:>14s}  {z:>14s}")

    output_xyz.parent.mkdir(parents=True, exist_ok=True)
    with output_xyz.open("w") as fh:
        fh.write(f"{len(atoms)}\n")
        fh.write(f"Extracted from {log_file.name}\n")
        for atom in atoms:
            fh.write(atom + "\n")

    return output_xyz.resolve()
