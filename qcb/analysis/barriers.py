"""
Barrier comparison and energy analysis for NEB-TS runs.

Load summary.json files from completed NEB runs, compare barriers across
systems/models, and convert between energy barriers and rate constants
via the Eyring equation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Union

# ══════════════════════════════════════════════════════════════
#  Physical constants
# ══════════════════════════════════════════════════════════════

kB = 1.380649e-23   # Boltzmann constant, J/K
h = 6.62607e-34     # Planck constant, J*s
R = 1.987e-3        # Gas constant, kcal/mol/K


# ══════════════════════════════════════════════════════════════
#  I/O
# ══════════════════════════════════════════════════════════════

def load_neb_summary(output_dir: Union[str, Path]) -> dict:
    """Load summary.json from a NEB-TS run directory.

    Parameters
    ----------
    output_dir : str or Path
        Directory containing ``summary.json`` (the top-level output dir
        produced by ``run_neb_ts.py``).

    Returns
    -------
    dict
        Parsed contents of summary.json.  Expected keys include
        ``system``, ``model``, ``barrier_fwd_kcal``, ``barrier_rev_kcal``,
        ``dE_rxn_kcal``, ``elapsed_min``, ``freq_validation``, etc.

    Raises
    ------
    FileNotFoundError
        If summary.json does not exist in *output_dir*.
    """
    summary_path = Path(output_dir) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No summary.json found in {output_dir}. "
            "Is this a completed NEB-TS output directory?"
        )
    with open(summary_path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════
#  Comparison
# ══════════════════════════════════════════════════════════════

def compare_barriers(output_dirs: list[Union[str, Path]]) -> list[dict]:
    """Load multiple NEB-TS runs and return a comparison table.

    Parameters
    ----------
    output_dirs : list of str or Path
        Directories containing ``summary.json`` files.

    Returns
    -------
    list of dict
        One dict per run with standardised keys for easy tabulation::

            {
                "dir": str,
                "system": str,
                "model": str,
                "barrier_fwd_kcal": float,
                "barrier_rev_kcal": float,
                "dE_rxn_kcal": float,
                "kcat_fwd": float,       # s-1, at 300 K
                "kcat_rev": float,       # s-1, at 300 K
                "elapsed_min": float,
                "n_atoms": int,
            }

        Missing keys in the source JSON are filled with ``None``.
    """
    rows: list[dict] = []
    for d in output_dirs:
        summary = load_neb_summary(d)
        fwd = summary.get("barrier_fwd_kcal")
        rev = summary.get("barrier_rev_kcal")
        rows.append({
            "dir": str(d),
            "system": summary.get("system"),
            "model": summary.get("model"),
            "barrier_fwd_kcal": fwd,
            "barrier_rev_kcal": rev,
            "dE_rxn_kcal": summary.get("dE_rxn_kcal"),
            "kcat_fwd": barrier_to_kcat(fwd) if fwd is not None else None,
            "kcat_rev": barrier_to_kcat(rev) if rev is not None else None,
            "elapsed_min": summary.get("elapsed_min"),
            "n_atoms": summary.get("n_atoms"),
        })
    return rows


# ══════════════════════════════════════════════════════════════
#  Eyring equation conversions
# ══════════════════════════════════════════════════════════════

def barrier_to_kcat(barrier_kcal: float, T: float = 300.0) -> float:
    """Convert an activation barrier to a rate constant via the Eyring equation.

    .. math::

        k_{cat} = \\frac{k_B T}{h} \\exp\\!\\left(-\\frac{\\Delta G^\\ddagger}{RT}\\right)

    Parameters
    ----------
    barrier_kcal : float
        Activation free energy in kcal/mol.
    T : float, optional
        Temperature in Kelvin (default 300 K).

    Returns
    -------
    float
        Rate constant in s-1.
    """
    prefactor = (kB * T) / h          # s-1
    exponent = -barrier_kcal / (R * T)
    return prefactor * math.exp(exponent)


def kcat_to_barrier(kcat: float, T: float = 300.0) -> float:
    """Convert a rate constant to an activation barrier via the Eyring equation.

    Inverse of :func:`barrier_to_kcat`.

    Parameters
    ----------
    kcat : float
        Rate constant in s-1.
    T : float, optional
        Temperature in Kelvin (default 300 K).

    Returns
    -------
    float
        Activation free energy in kcal/mol.
    """
    if kcat <= 0:
        raise ValueError(f"kcat must be positive, got {kcat}")
    prefactor = (kB * T) / h
    return -R * T * math.log(kcat / prefactor)


# ══════════════════════════════════════════════════════════════
#  Reporting
# ══════════════════════════════════════════════════════════════

def _fmt(val, fmt: str = ".1f") -> str:
    """Format a value, returning '---' for None."""
    if val is None:
        return "---"
    return f"{val:{fmt}}"


def _fmt_sci(val) -> str:
    """Format a value in scientific notation, returning '---' for None."""
    if val is None:
        return "---"
    return f"{val:.2e}"


def print_barrier_report(output_dirs: list[Union[str, Path]]) -> None:
    """Pretty-print a barrier comparison across multiple NEB-TS runs.

    Prints a formatted table to stdout with forward/reverse barriers,
    reaction energies, estimated kcat values, and run times.

    Parameters
    ----------
    output_dirs : list of str or Path
        Directories containing ``summary.json`` files.
    """
    rows = compare_barriers(output_dirs)
    if not rows:
        print("No runs to compare.")
        return

    # Header
    print()
    print("=" * 100)
    print("  NEB-TS Barrier Comparison Report")
    print("=" * 100)
    print()

    # Column headers
    header = (
        f"{'System':<20s} {'Model':<16s} "
        f"{'Fwd (kcal)':<12s} {'Rev (kcal)':<12s} {'dE_rxn':<10s} "
        f"{'kcat_fwd':<12s} {'kcat_rev':<12s} "
        f"{'Time(min)':<10s} {'N_atoms':<8s}"
    )
    print(header)
    print("-" * 100)

    for row in rows:
        line = (
            f"{(row['system'] or '???'):<20s} "
            f"{(row['model'] or '???'):<16s} "
            f"{_fmt(row['barrier_fwd_kcal']):<12s} "
            f"{_fmt(row['barrier_rev_kcal']):<12s} "
            f"{_fmt(row['dE_rxn_kcal']):<10s} "
            f"{_fmt_sci(row['kcat_fwd']):<12s} "
            f"{_fmt_sci(row['kcat_rev']):<12s} "
            f"{_fmt(row['elapsed_min']):<10s} "
            f"{_fmt(row['n_atoms'], 'd') if row['n_atoms'] else '---':<8s}"
        )
        print(line)

    print("-" * 100)
    print(f"  {len(rows)} run(s) compared  |  kcat estimated via Eyring equation at 300 K")
    print("=" * 100)
    print()
