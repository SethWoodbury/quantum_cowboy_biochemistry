"""
ORCA input file generation and modification.

Generates ``.inp`` input files for ORCA, with support for CPCM implicit
solvation and parallel execution.  Modification of existing input files
follows the same ``%maxcore`` / ``nprocs`` pattern used by Indrek's
``orcasub`` script (``/home/ikalvet/bin/orcasub``).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger("quantum_engine.qm.orca")

# Hartree -> eV (CODATA 2018)
_HARTREE_EV = 27.211386245988

# ── Supported job type -> ORCA keyword mapping ──────────────────────
_JOB_TYPE_KEYWORDS: dict[str, str] = {
    "opt": "Opt",
    "opt+freq": "Opt Freq",
    "freq": "Freq",
    "ts": "OptTS",
    "ts+freq": "OptTS Freq",
    "neb-ts": "NEB-TS",
    "neb-ci": "NEB-CI",
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


def write_orca_nebts_input(
    reactant_xyz: str | Path,
    product_xyz: str | Path,
    output_inp: str | Path,
    *,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    n_images: int = 8,
    freq: bool = True,
    nproc: int = 8,
    maxcore: int = 4000,
    extra_keywords: str = "",
    solvent: str | None = None,
) -> Path:
    """Generate an ORCA native **NEB-TS** input (reactant + product endpoints).

    NEB-TS launches a climbing-image NEB then optimises the highest image to a
    first-order saddle; with ``freq=True`` ORCA runs an analytic frequency job on
    the TS (so n_imag/imag-mode come for free). The product geometry is written
    next to the input as ``<stem>_product.xyz`` and referenced via ``%neb``.
    """
    reactant_xyz, product_xyz = Path(reactant_xyz), Path(product_xyz)
    output_inp = Path(output_inp)
    output_inp.parent.mkdir(parents=True, exist_ok=True)

    atom_lines = _read_xyz_block(reactant_xyz)
    if not atom_lines:
        raise ValueError(f"No atom coordinates found in {reactant_xyz}")

    # ORCA reads the product endpoint from a referenced xyz file.
    prod_lines = _read_xyz_block(product_xyz)
    if not prod_lines:
        raise ValueError(f"No atom coordinates found in {product_xyz}")
    prod_path = output_inp.with_name(output_inp.stem + "_product.xyz")
    prod_path.write_text(
        f"{len(prod_lines)}\nproduct endpoint\n"
        + "\n".join(l.strip() for l in prod_lines) + "\n")

    job_kw = _JOB_TYPE_KEYWORDS["neb-ts"]
    kw_parts = [method, basis, job_kw]
    if freq:
        kw_parts.append("Freq")
    if solvent is not None:
        kw_parts.append(f"CPCM({solvent})")
    if extra_keywords.strip():
        kw_parts.append(extra_keywords.strip())

    lines = [
        "! " + " ".join(kw_parts) + "\n",
        f"%maxcore {maxcore}\n",
        "%pal\n", f"  nprocs {nproc}\n", "end\n",
        "%neb\n",
        f'  NEB_END_XYZFILE "{prod_path.name}"\n',   # ORCA 4.1.1 endpoint keyword
        f"  NImages {n_images}\n",
        "end\n",
        "\n",
        f"* xyz {charge} {multiplicity}\n",
    ]
    lines += [a + "\n" for a in atom_lines]
    lines += ["*\n", "\n"]
    output_inp.write_text("".join(lines))
    return output_inp.resolve()


# ── Output parsing (ORCA 4.x) ────────────────────────────────────────
def parse_final_energy(out_text: str) -> float | None:
    """Last ``FINAL SINGLE POINT ENERGY`` in Hartree (None if absent)."""
    matches = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", out_text)
    return float(matches[-1]) if matches else None


def parse_frequencies(out_text: str) -> list[float]:
    """Vibrational frequencies (cm^-1, signed) from the last ``VIBRATIONAL
    FREQUENCIES`` block. Imaginary modes are negative."""
    blocks = out_text.split("VIBRATIONAL FREQUENCIES")
    if len(blocks) < 2:
        return []
    freqs: list[float] = []
    for line in blocks[-1].splitlines():
        # lines look like:  "   6:      -523.45 cm**-1"
        m = re.match(r"\s*\d+:\s+(-?\d+\.\d+)\s*cm\*\*-1", line)
        if m:
            freqs.append(float(m.group(1)))
        elif freqs and line.strip() and "cm**-1" not in line and ":" not in line:
            break
    return freqs


def parse_last_geometry(out_text: str) -> tuple[list[str], list[list[float]]] | None:
    """Last ``CARTESIAN COORDINATES (ANGSTROEM)`` block -> (symbols, positions)."""
    idx = out_text.rfind("CARTESIAN COORDINATES (ANGSTROEM)")
    if idx < 0:
        return None
    syms: list[str] = []
    pos: list[list[float]] = []
    for line in out_text[idx:].splitlines()[2:]:
        parts = line.split()
        if len(parts) != 4:
            break
        try:
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError:
            break
        syms.append(parts[0]); pos.append(xyz)
    return (syms, pos) if syms else None


def n_imaginary(freqs: list[float], cutoff: float = -10.0) -> int:
    """Count modes more negative than ``cutoff`` cm^-1 (default -10)."""
    return sum(1 for f in freqs if f < cutoff)


def imaginary_frequency(freqs: list[float], cutoff: float = -10.0) -> float | None:
    """The most-negative frequency (cm^-1) IF it is below ``cutoff`` (default
    -10), else None — so a small rot/trans contaminant near 0 isn't mistaken for
    a TS mode. Consistent with :func:`n_imaginary`."""
    if not freqs:
        return None
    lo = min(freqs)
    return lo if lo < cutoff else None


def find_nebts_ts_xyz(outdir: str | Path, stem: str) -> Path | None:
    """Locate ORCA 4.1.1's converged NEB-TS geometry.

    ORCA writes the converged TS to ``<basename>.xyz`` and the highest-energy
    image to ``<basename>_HEI-NEB_converged.xyz`` — those are the real names
    (verified against the 4.1.1 binary)."""
    outdir = Path(outdir)
    for pat in (f"{stem}.xyz", f"{stem}_HEI-NEB_converged.xyz",
                f"{stem}_CI-NEB_converged.xyz"):
        p = outdir / pat
        if p.exists():
            return p
    hits = sorted(outdir.glob("*_HEI-NEB_converged.xyz")) or \
        sorted(outdir.glob("*_CI-NEB_converged.xyz"))
    return hits[-1] if hits else None


def run_orca(inp_file: str | Path, *, orca_bin: str | None = None,
             outfile: str | Path | None = None, timeout: int | None = None) -> dict:
    """Run ORCA on ``inp_file`` (full path required by ORCA for MPI) → result.

    Returns ``{returncode, out_path, energy_eV, frequencies_cm, n_imag,
    imag_freq_cm, geometry}``. ``orca_bin`` defaults to ``site.ORCA_BIN``. ORCA
    is NOT in the cowboy-qc container — run this where ORCA is installed (a host/node).
    """
    import subprocess  # noqa: PLC0415
    inp_file = Path(inp_file).resolve()
    if orca_bin is None:
        from quantum_engine import site  # noqa: PLC0415
        orca_bin = getattr(site, "ORCA_BIN", "orca")
    out_path = Path(outfile) if outfile else inp_file.with_suffix(".out")
    log.info("running ORCA: %s %s > %s", orca_bin, inp_file.name, out_path.name)
    with open(out_path, "w") as fh:
        proc = subprocess.run([str(orca_bin), str(inp_file)], stdout=fh,
                              stderr=subprocess.STDOUT, timeout=timeout,
                              cwd=str(inp_file.parent))
    text = out_path.read_text(errors="replace")
    freqs = parse_frequencies(text)
    e_ha = parse_final_energy(text)
    return {
        "returncode": proc.returncode,
        "out_path": str(out_path),
        "energy_eV": e_ha * _HARTREE_EV if e_ha is not None else None,
        "frequencies_cm": freqs,
        "n_imag": n_imaginary(freqs),
        "imag_freq_cm": imaginary_frequency(freqs),
        "geometry": parse_last_geometry(text),
    }
