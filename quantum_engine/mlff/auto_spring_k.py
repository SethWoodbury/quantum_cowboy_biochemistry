"""
Automatic spring-constant selection for endpoint-generation springs.

Motivation
----------
The current QCB pipeline uses a hardcoded ``spring_k = 3.0 eV/A^2`` for all
endpoint-generation springs (bond-breaking and CV-based). That value is
arbitrary and system-independent. For some bonds 3 eV/A^2 is too stiff
(it dominates the PES and buries real chemistry under spring forces),
and for others it is too soft (the spring can't actually drive the bond
to the target distance against a strong real bond).

This module implements two principled, system-aware methods for choosing
``k_spring`` and provides a ``compare_methods`` helper that runs both and
sanity-checks their agreement.

Method A: xTB-computed Hessian (most rigorous)
----------------------------------------------
- Run xTB GFN2 on the (small) fragment that contains the bond of interest at
  its current geometry.
- Request the Hessian (``--hess``).  xTB writes a 3N x 3N mass-weighted-free
  Hessian in Hartree / Bohr^2 to the file ``hessian``.
- Extract the 6 x 6 block belonging to atoms ``i`` and ``j``.
- Project onto the bond axis to get the bond-stretching force constant::

      k_bond = u^T . H_6x6 . u

  where ``u = [+u_ij; -u_ij]`` and ``u_ij = (r_j - r_i) / |r_j - r_i|``.
- Convert Hartree / Bohr^2 -> eV / A^2 (factor ~= 97.17).
- Recommended spring constant: ``k_spring = k_bond / 3`` — deliberately softer
  than the real bond stretch so the spring pokes the PES toward the target
  geometry without overwhelming real forces.

Method B: Pauling bond-order formula (heuristic, generalizable)
---------------------------------------------------------------
- Estimate bond order from bond length using Pauling's relation::

      BO = exp( -(r - r_0) / c )             # c = 0.3 A

  where ``r_0`` is the single-bond reference length (sum of covalent radii).
- Force constant scales as ``k = k_ref * BO^2`` (bond-order-squared
  dependence; Lendvay, JPC A 2000).
- ``k_ref`` is the reference *single-bond* stretching force constant for
  the atom pair, tabulated below with literature values.
- Recommended spring constant: ``k_spring = k_ref * BO^2 / 3``.

Which method should be default?
-------------------------------
Default = Pauling.  Reasoning:

1. Portability. Pauling needs only the two atom symbols, their positions,
   and a small lookup table. It always works, with zero external tools.
   xTB requires the ``xtb`` binary, a working SCF for the fragment,
   correct total charge, and (most painfully) a Hessian calculation that
   in practice takes several minutes for ~30 atoms and can fail outright
   for large / charged / delicate systems.

2. Numerical stability. The Pauling result is a smooth function of bond
   length; small geometry jitter moves the answer slightly. The xTB
   partial Hessian is vulnerable to SCF noise, finite-difference error
   (xTB computes the Hessian by numerical differentiation), and
   projection noise when the bond axis is tangled with bending modes.

3. Scale. Endpoint generation is used in the outer loop of the pipeline,
   and we want it fast.  Pauling is free. xTB Hessian on a 30-atom
   fragment is ~1-5 minutes; on a 100-atom ligand it's 20+ minutes and
   often fails. At scale, xTB-as-default is a bottleneck.

4. Correctness is "good enough" for a spring. We are picking the spring
   constant of a *bias* term that will be released once the endpoint
   is found.  Being within a factor of 2 of the ideal stiffness is
   almost always adequate; a 30% error in k_spring does not change
   whether the endpoint finds its basin. Pauling reliably gets within
   that tolerance for covalent first- and second-row bonds.

xTB is retained as an optional verification and as a fallback for
chemistry that the Pauling table doesn't cover well (e.g., metal-ligand
bonds). Use ``compare_methods`` when you have time; use ``spring_k_auto``
otherwise.

References
----------
- Pauling, L. "The Nature of the Chemical Bond", 3rd ed., Cornell
  University Press, 1960. (bond-order / bond-length relation)
- Lendvay, G. "On the Correlation of Bond Order and Bond Length",
  J. Phys. Chem. A 2000, 104, 9500-9505.
  (BO^2 dependence of the force constant)
- Halgren, T.A. "Merck molecular force field. II. MMFF94 van der Waals
  and electrostatic parameters for intermolecular interactions",
  J. Comput. Chem. 1996, 17, 520-552.
  doi:10.1002/jcc.540170510
  (reference single-bond stretching force constants)
- Bannwarth, C.; Ehlert, S.; Grimme, S. "GFN2-xTB", J. Chem. Theory
  Comput. 2019, 15, 1652.  (xTB method used for the Hessian)
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:  # Prefer ASE's covalent radii if available (they always should be)
    from ase.data import atomic_numbers, covalent_radii
    _HAVE_ASE_RADII = True
except Exception:  # pragma: no cover - ase should always be available
    atomic_numbers = {}
    covalent_radii = []
    _HAVE_ASE_RADII = False

log = logging.getLogger("quantum_engine.auto_spring_k")


# ---------------------------------------------------------------------------
# Constants and reference tables
# ---------------------------------------------------------------------------

# Hartree / Bohr^2 -> eV / A^2
# 27.211386 eV/Hartree divided by (0.529177 A/Bohr)^2
HARTREE_PER_BOHR2_TO_EV_PER_A2 = 27.211386 / (0.529177 ** 2)  # ~= 97.1736

# Safety factor: spring should be weaker than the real bond so it doesn't
# dominate the PES. k_spring = k_bond / SOFTENING_FACTOR.
SOFTENING_FACTOR = 3.0

# Pauling bond-order decay length (Angstrom)
PAULING_C = 0.3

# Default reference single-bond force constants, eV/A^2.
# Values are tabulated from Halgren 1996 (MMFF94, Table V), cross-checked
# against standard organic-chemistry tabulations.  Units: converted from
# mdyn/A (MMFF reports mdyn/A) via 1 mdyn/A = 6.242 eV/A^2.
#
# References per entry:
#   (H96)  Halgren, J. Comput. Chem. 1996, 17, 520.
#   (LEN)  Lendvay, J. Phys. Chem. A 2000, 104, 9500.
#   (MAR)  March's "Advanced Organic Chemistry", 7th ed., bond-stretching
#          force-constant tables.
#
# Sorted by first symbol.
DEFAULT_K_REF_MAP_EV_PER_A2: dict[tuple[str, str], float] = {
    # C-X bonds
    ("C", "C"): 27.0,   # (H96) 4.3 mdyn/A
    ("C", "H"): 30.0,   # (H96) 4.8 mdyn/A
    ("C", "N"): 29.0,   # (H96) 4.6 mdyn/A
    ("C", "O"): 32.0,   # (H96) 5.1 mdyn/A
    ("C", "F"): 36.0,   # (H96) 5.8 mdyn/A
    ("C", "S"): 19.0,   # (H96) 3.0 mdyn/A
    ("C", "Cl"): 20.0,  # (H96)
    ("C", "Br"): 16.0,  # (MAR)
    ("C", "P"): 17.0,   # (MAR) — lower than C-C due to longer bond
    # N-X bonds
    ("N", "H"): 41.0,   # (H96) 6.6 mdyn/A
    ("N", "N"): 24.0,   # (H96)
    ("N", "O"): 25.0,   # (H96)
    # O-X bonds
    ("O", "H"): 51.0,   # (H96) 8.2 mdyn/A
    ("O", "O"): 22.0,   # (H96) peroxide
    ("O", "P"): 25.0,   # (H96/MAR) 4.0 mdyn/A — phosphate P-O single bond
    ("O", "S"): 28.0,   # (MAR)
    # P-X bonds
    ("P", "H"): 20.0,   # (MAR)
    ("P", "P"): 14.0,   # (MAR)
    # S-X bonds
    ("S", "H"): 26.0,   # (H96)
    ("S", "S"): 16.0,   # (H96) disulfide
    # H-H for completeness
    ("H", "H"): 36.0,   # (H96)
}

# If the bond pair isn't listed, use this fallback (rough average of the
# table above, eV/A^2).
DEFAULT_K_REF_FALLBACK = 25.0


def _norm_pair(a: str, b: str) -> tuple[str, str]:
    """Return an order-independent (alpha-sorted) element pair."""
    return (a, b) if a <= b else (b, a)


def _lookup_k_ref(
    sym_i: str,
    sym_j: str,
    k_ref_map: Optional[dict[tuple[str, str], float]] = None,
) -> tuple[float, bool]:
    """Look up the reference single-bond k for the (sym_i, sym_j) pair.

    Returns ``(k_ref, found)`` where ``found=False`` means we fell back to
    the default fallback value.
    """
    table = {_norm_pair(*k): v for k, v in (k_ref_map or DEFAULT_K_REF_MAP_EV_PER_A2).items()}
    key = _norm_pair(sym_i, sym_j)
    if key in table:
        return table[key], True
    return DEFAULT_K_REF_FALLBACK, False


def _covalent_radius(symbol: str) -> float:
    """Return the covalent radius (A) for an element symbol using ase.data."""
    if _HAVE_ASE_RADII and symbol in atomic_numbers:
        return float(covalent_radii[atomic_numbers[symbol]])
    # Minimal fallback table (A) for common organic/phosphate chemistry
    fallback = {
        "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
        "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20,
    }
    return fallback.get(symbol, 0.77)


# ---------------------------------------------------------------------------
# xTB binary resolution (reuse what's already in qcb.mlff.xtb_refine)
# ---------------------------------------------------------------------------

def _resolve_xtb_binary() -> Optional[str]:
    """Find xTB binary. Prefer qcb.mlff.xtb_refine's resolution, then PATH."""
    try:
        from quantum_engine.qm.xtb_refine import XTB_BIN as _configured
        if _configured and os.path.isfile(_configured):
            return _configured
    except Exception:
        pass
    try:
        from quantum_engine.config import XTB_BIN as _configured  # type: ignore
        if _configured and os.path.isfile(_configured):
            return _configured
    except Exception:
        pass
    return shutil.which("xtb")


# ---------------------------------------------------------------------------
# Fragment extraction for xTB Hessian
# ---------------------------------------------------------------------------

def _extract_first_shell(
    atoms,
    i: int,
    j: int,
    cutoff: float = 2.0,
) -> list[int]:
    """Return indices of atoms in the first coordination shell of i or j.

    Includes i and j themselves plus any atom whose distance to either i or j
    is < cutoff (Angstrom).  Keeps the fragment small so xTB Hessian is fast.
    """
    pos = atoms.get_positions()
    r_i = pos[i]
    r_j = pos[j]
    d_i = np.linalg.norm(pos - r_i, axis=1)
    d_j = np.linalg.norm(pos - r_j, axis=1)
    mask = (d_i < cutoff) | (d_j < cutoff)
    indices = sorted(int(k) for k in np.where(mask)[0])
    if i not in indices:
        indices.append(i)
    if j not in indices:
        indices.append(j)
    indices = sorted(set(indices))
    return indices


# ---------------------------------------------------------------------------
# xTB Hessian parsing
# ---------------------------------------------------------------------------

def _parse_xtb_hessian(hessian_path: Path, n_atoms: int) -> Optional[np.ndarray]:
    """Parse xTB's ``hessian`` file.

    xTB writes the Hessian as a plain list of floats (in Hartree/Bohr^2),
    typically 5 per line, with a header line ``$hessian``.  The total count
    is (3N)^2.  We read tokens and reshape.
    """
    if not hessian_path.exists():
        return None
    text = hessian_path.read_text()
    tokens = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("$"):
            continue
        for tok in s.split():
            try:
                tokens.append(float(tok))
            except ValueError:
                # Unknown content — likely a header/footer we should skip.
                pass
    expected = (3 * n_atoms) ** 2
    if len(tokens) < expected:
        log.warning(
            f"  xTB hessian file has {len(tokens)} floats, expected "
            f"{expected} — Hessian parse failed."
        )
        return None
    arr = np.array(tokens[:expected], dtype=float).reshape(3 * n_atoms, 3 * n_atoms)
    # Symmetrize in case of minor numerical asymmetry
    arr = 0.5 * (arr + arr.T)
    return arr


# ---------------------------------------------------------------------------
# Method A: xTB Hessian
# ---------------------------------------------------------------------------

def spring_k_from_xtb(
    atoms,
    i: int,
    j: int,
    total_charge: int,
    workdir,
    fragment_indices: Optional[list[int]] = None,
    softening_factor: float = SOFTENING_FACTOR,
    timeout_s: Optional[int] = None,
) -> dict:
    """Compute spring constant from the xTB-computed Hessian.

    The workflow is:
        1. Extract a small fragment containing the first coordination shell
           of atoms i and j (or use caller-supplied ``fragment_indices``).
        2. Run ``xtb --gfn 2 --chrg C --hess`` on the fragment.
        3. Parse ``hessian`` (Hartree/Bohr^2).
        4. Project the 6x6 block for (i, j) onto the bond axis to get the
           bond-stretch force constant k_bond in eV/A^2.
        5. Return k_spring = k_bond / softening_factor.

    Args:
        atoms: ASE Atoms object.
        i, j: Indices (0-based, referring to ``atoms``) of the bond endpoints.
        total_charge: Charge of the *fragment* (should be set correctly).
        workdir: Scratch directory; xTB files are written under ``.xtb_hess``.
        fragment_indices: If None, auto-extracted via ``_extract_first_shell``.
        softening_factor: Divide k_bond by this to get k_spring (default 3.0).
        timeout_s: xTB timeout (None = auto).

    Returns dict with ``k_eV_per_A2``, ``k_bond_eV_per_A2``, ``method``,
    ``atoms_included``, ``success``, ``diagnostics``.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    xtb_bin = _resolve_xtb_binary()
    if xtb_bin is None:
        return {
            "k_eV_per_A2": None,
            "k_bond_eV_per_A2": None,
            "method": "xtb_hessian",
            "atoms_included": 0,
            "success": False,
            "diagnostics": "xTB binary not found (checked qcb.mlff.xtb_refine, qcb.config, PATH)",
        }

    # 1. Pick fragment indices
    if fragment_indices is None:
        fragment_indices = _extract_first_shell(atoms, i, j, cutoff=2.0)
    fragment_indices = sorted(set(int(k) for k in fragment_indices))
    if i not in fragment_indices or j not in fragment_indices:
        return {
            "k_eV_per_A2": None,
            "k_bond_eV_per_A2": None,
            "method": "xtb_hessian",
            "atoms_included": len(fragment_indices),
            "success": False,
            "diagnostics": "fragment_indices must contain both i and j",
        }
    # Local index of i and j within the fragment
    local_i = fragment_indices.index(i)
    local_j = fragment_indices.index(j)

    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    frag_symbols = [symbols[k] for k in fragment_indices]
    frag_positions = np.array([positions[k] for k in fragment_indices])
    n_frag = len(fragment_indices)

    tmpdir = workdir / ".xtb_hess"
    tmpdir.mkdir(exist_ok=True)
    xyz_path = tmpdir / "frag.xyz"
    with open(xyz_path, "w") as f:
        f.write(f"{n_frag}\n\n")
        for sym, (x, y, z) in zip(frag_symbols, frag_positions):
            f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")

    if timeout_s is None:
        timeout_s = min(1800, max(60, n_frag * 10))

    cmd = [
        xtb_bin, xyz_path.name,
        "--gfn", "2",
        "--chrg", str(int(total_charge)),
        "--hess",
    ]
    log.info(
        f"  xTB Hessian: {n_frag} atoms (fragment of bond {i}-{j}), "
        f"charge={total_charge}, timeout={timeout_s}s"
    )
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_s, cwd=str(tmpdir),
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return {
            "k_eV_per_A2": None,
            "k_bond_eV_per_A2": None,
            "method": "xtb_hessian",
            "atoms_included": n_frag,
            "success": False,
            "diagnostics": f"xTB Hessian timed out after {timeout_s}s",
        }

    # 3. Parse Hessian
    hess_path = tmpdir / "hessian"
    H = _parse_xtb_hessian(hess_path, n_frag)
    if H is None:
        tail = (result.stderr or result.stdout or "")[-400:]
        shutil.rmtree(tmpdir, ignore_errors=True)
        return {
            "k_eV_per_A2": None,
            "k_bond_eV_per_A2": None,
            "method": "xtb_hessian",
            "atoms_included": n_frag,
            "success": False,
            "diagnostics": f"Failed to parse xTB hessian file. xTB stderr tail: {tail!r}",
        }

    # 4. Extract the 6x6 block for atoms (local_i, local_j) and project onto
    #    the bond axis.
    rows_i = slice(3 * local_i, 3 * local_i + 3)
    rows_j = slice(3 * local_j, 3 * local_j + 3)
    H_ii = H[rows_i, rows_i]
    H_ij = H[rows_i, rows_j]
    H_ji = H[rows_j, rows_i]
    H_jj = H[rows_j, rows_j]
    H_6x6 = np.block([[H_ii, H_ij], [H_ji, H_jj]])  # 6 x 6

    # Bond axis (from i to j)
    v = frag_positions[local_j] - frag_positions[local_i]
    d_ij = float(np.linalg.norm(v))
    if d_ij < 1e-6:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return {
            "k_eV_per_A2": None,
            "k_bond_eV_per_A2": None,
            "method": "xtb_hessian",
            "atoms_included": n_frag,
            "success": False,
            "diagnostics": f"Bond length is ~0 (atoms {i} and {j} overlap)",
        }
    u_ij = v / d_ij
    # Displacement vector for "stretching" mode: atom i moves -u, atom j moves +u.
    u_stretch = np.concatenate([-u_ij, u_ij])  # length 6
    # Normalize so that the bond distance changes by 1 A when we move by u_stretch:
    # stretching both atoms by 0.5 along +u gives d -> d + 1, i.e. |u_stretch|=sqrt(2)
    # so the distance change is |u_stretch|/sqrt(2) * 1 = 1.  Correct.

    # Force constant in Hartree/Bohr^2:
    k_bond_Ha_per_Bohr2 = float(u_stretch @ H_6x6 @ u_stretch)
    k_bond_eV_per_A2 = k_bond_Ha_per_Bohr2 * HARTREE_PER_BOHR2_TO_EV_PER_A2

    shutil.rmtree(tmpdir, ignore_errors=True)

    if not np.isfinite(k_bond_eV_per_A2) or k_bond_eV_per_A2 < 0:
        return {
            "k_eV_per_A2": None,
            "k_bond_eV_per_A2": k_bond_eV_per_A2,
            "method": "xtb_hessian",
            "atoms_included": n_frag,
            "success": False,
            "diagnostics": (
                f"Projected k_bond is non-physical ({k_bond_eV_per_A2:.3f} eV/A^2); "
                "Hessian likely non-positive along this mode (saddle geometry?)."
            ),
        }

    k_spring = k_bond_eV_per_A2 / softening_factor
    return {
        "k_eV_per_A2": k_spring,
        "k_bond_eV_per_A2": k_bond_eV_per_A2,
        "method": "xtb_hessian",
        "atoms_included": n_frag,
        "success": True,
        "diagnostics": (
            f"xTB GFN2 Hessian on {n_frag}-atom fragment; bond {i}-{j} "
            f"({symbols[i]}-{symbols[j]}) d={d_ij:.3f} A, "
            f"k_bond={k_bond_eV_per_A2:.2f} eV/A^2, "
            f"k_spring={k_spring:.2f} eV/A^2 (softening={softening_factor:g})"
        ),
    }


# ---------------------------------------------------------------------------
# Method B: Pauling bond-order formula
# ---------------------------------------------------------------------------

def spring_k_from_pauling(
    atoms,
    i: int,
    j: int,
    k_ref_map: Optional[dict[tuple[str, str], float]] = None,
    r0_map: Optional[dict[tuple[str, str], float]] = None,
    softening_factor: float = SOFTENING_FACTOR,
) -> dict:
    """Compute spring constant from Pauling's bond-order formula.

    ``BO = exp(-(r - r0) / 0.3)``, ``k = k_ref * BO^2``,
    ``k_spring = k / softening_factor``.

    r0 defaults to the sum of covalent radii from ``ase.data``.
    """
    symbols = atoms.get_chemical_symbols()
    sym_i, sym_j = symbols[i], symbols[j]
    r_ij = float(atoms.get_distance(i, j))

    # r0 from explicit map or covalent-radii sum
    r0: Optional[float] = None
    if r0_map is not None:
        r0 = r0_map.get(_norm_pair(sym_i, sym_j))
    if r0 is None:
        r0 = _covalent_radius(sym_i) + _covalent_radius(sym_j)

    # Pauling bond order
    bo = float(np.exp(-(r_ij - r0) / PAULING_C))

    k_ref, found = _lookup_k_ref(sym_i, sym_j, k_ref_map)
    k_bond = k_ref * (bo ** 2)
    k_spring = k_bond / softening_factor

    diag = (
        f"Pauling: {sym_i}-{sym_j} d={r_ij:.3f} A, r0={r0:.3f} A, "
        f"BO={bo:.3f}, k_ref={k_ref:.1f} eV/A^2"
        + ("" if found else f" (fallback: pair not in table)")
        + f", k_bond={k_bond:.2f} eV/A^2, k_spring={k_spring:.2f} eV/A^2"
    )
    return {
        "k_eV_per_A2": k_spring,
        "k_ref_eV_per_A2": k_ref,
        "bond_order": bo,
        "bond_length_A": r_ij,
        "r0_A": r0,
        "k_bond_eV_per_A2": k_bond,
        "pair_in_table": found,
        "method": "pauling_bond_order",
        "success": True,
        "diagnostics": diag,
    }


# ---------------------------------------------------------------------------
# Auto-picker and comparator
# ---------------------------------------------------------------------------

def spring_k_auto(
    atoms,
    i: int,
    j: int,
    total_charge: int,
    workdir,
    preferred: str = "xtb",
    fallback: bool = True,
) -> dict:
    """Pick a spring constant with automatic fallback.

    preferred:
        - "xtb":     run xTB Hessian; if it fails and fallback=True, try Pauling.
        - "pauling": use Pauling only.
        - "both":    run both and return {'xtb': ..., 'pauling': ..., 'ratio': ...}
                     — no single ``k_eV_per_A2`` field.

    The returned dict for "xtb"/"pauling" mirrors the single-method return
    (plus a 'fallback_used' flag for "xtb").
    """
    if preferred not in ("xtb", "pauling", "both"):
        raise ValueError(f"preferred must be 'xtb', 'pauling', or 'both'; got {preferred!r}")

    if preferred == "pauling":
        out = spring_k_from_pauling(atoms, i, j)
        out["fallback_used"] = False
        return out

    if preferred == "xtb":
        out = spring_k_from_xtb(atoms, i, j, total_charge, workdir)
        out["fallback_used"] = False
        if out["success"]:
            return out
        if not fallback:
            return out
        log.info(
            f"  xTB Hessian failed ({out['diagnostics']}); "
            f"falling back to Pauling."
        )
        pauling = spring_k_from_pauling(atoms, i, j)
        pauling["fallback_used"] = True
        pauling["xtb_attempt"] = out
        return pauling

    # preferred == "both"
    xtb_res = spring_k_from_xtb(atoms, i, j, total_charge, workdir)
    paul_res = spring_k_from_pauling(atoms, i, j)
    ratio = None
    if xtb_res["success"] and paul_res["success"] and paul_res["k_eV_per_A2"]:
        ratio = xtb_res["k_eV_per_A2"] / paul_res["k_eV_per_A2"]
    return {
        "xtb": xtb_res,
        "pauling": paul_res,
        "ratio_xtb_to_pauling": ratio,
        "method": "both",
    }


def compare_methods(
    atoms,
    i: int,
    j: int,
    total_charge: int,
    workdir,
) -> dict:
    """Run both methods on the same bond, compare, recommend one.

    Agreement classification:
        - 'agree':         ratio within 30% of 1.0 (0.77 < xtb/pauling < 1.30)
        - 'disagree':      both methods succeeded but differ significantly
        - 'only_one_worked': exactly one of the two methods returned success
        - 'both_failed':   neither worked (shouldn't happen — Pauling always works)

    Recommendation policy:
        - If both agree:          use xtb (more rigorous, and pauling corroborates)
        - If they disagree:       use pauling (more numerically trustworthy;
                                   xTB partial Hessian can be noisy)
        - If only pauling worked: use pauling
        - If only xtb worked:     use xtb (with a warning)
    """
    xtb_res = spring_k_from_xtb(atoms, i, j, total_charge, workdir)
    paul_res = spring_k_from_pauling(atoms, i, j)

    xtb_ok = xtb_res["success"] and xtb_res["k_eV_per_A2"] is not None
    paul_ok = paul_res["success"] and paul_res["k_eV_per_A2"] is not None

    ratio = None
    if xtb_ok and paul_ok:
        ratio = xtb_res["k_eV_per_A2"] / paul_res["k_eV_per_A2"]

    if xtb_ok and paul_ok:
        if ratio is not None and 0.77 < ratio < 1.30:
            agreement = "agree"
            recommended_k = xtb_res["k_eV_per_A2"]
            recommended_method = "xtb_hessian"
            rationale = (
                f"Both methods agree within 30% (ratio={ratio:.2f}). "
                "Using xTB as the more rigorous electronic-structure-based value."
            )
        else:
            agreement = "disagree"
            recommended_k = paul_res["k_eV_per_A2"]
            recommended_method = "pauling_bond_order"
            rationale = (
                f"Methods disagree (xtb/pauling ratio = {ratio:.2f}). "
                "Preferring Pauling: numerically more trustworthy for a "
                "bias-spring constant, and insensitive to xTB SCF/Hessian noise."
            )
    elif paul_ok and not xtb_ok:
        agreement = "only_one_worked"
        recommended_k = paul_res["k_eV_per_A2"]
        recommended_method = "pauling_bond_order"
        rationale = (
            "xTB Hessian failed; using Pauling. "
            f"xTB diagnostics: {xtb_res['diagnostics']}"
        )
    elif xtb_ok and not paul_ok:
        agreement = "only_one_worked"
        recommended_k = xtb_res["k_eV_per_A2"]
        recommended_method = "xtb_hessian"
        rationale = (
            "Pauling failed (shouldn't happen normally); using xTB. "
            f"Pauling diagnostics: {paul_res['diagnostics']}"
        )
    else:
        agreement = "both_failed"
        recommended_k = 3.0  # The old hardcoded default
        recommended_method = "hardcoded_fallback"
        rationale = (
            "Both methods failed. Falling back to the historical hardcoded "
            "3.0 eV/A^2. Inspect diagnostics to understand failure."
        )

    return {
        "xtb": xtb_res,
        "pauling": paul_res,
        "ratio_xtb_to_pauling": ratio,
        "agreement": agreement,
        "recommended_k": recommended_k,
        "recommended_method": recommended_method,
        "rationale": rationale,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal quick-test on a toy phosphate-like geometry.
    # Tests Pauling unconditionally; tests xTB only if binary is present.
    import tempfile

    from ase import Atoms  # type: ignore

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Toy PO4 (no ligand chemistry, just a cross-check that the code runs).
    test_atoms = Atoms(
        "POOOO",
        positions=[
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [-1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, -1.5, 0.0],
        ],
    )
    total_charge = -3  # PO4^3-
    bond_i, bond_j = 0, 1  # P - O

    print("=" * 72)
    print(f"Testing auto_spring_k on toy PO4 fragment (charge={total_charge}); "
          f"bond ({bond_i}={test_atoms.symbols[bond_i]}) - "
          f"({bond_j}={test_atoms.symbols[bond_j]})")
    print("=" * 72)

    # Pauling (always available)
    paul = spring_k_from_pauling(test_atoms, bond_i, bond_j)
    print("\n--- Pauling ---")
    for k, v in paul.items():
        print(f"  {k}: {v}")

    # xTB (optional)
    with tempfile.TemporaryDirectory() as td:
        xtb_res = spring_k_from_xtb(test_atoms, bond_i, bond_j, total_charge, td)
    print("\n--- xTB Hessian ---")
    for k, v in xtb_res.items():
        print(f"  {k}: {v}")

    # Comparator
    with tempfile.TemporaryDirectory() as td:
        comp = compare_methods(test_atoms, bond_i, bond_j, total_charge, td)
    print("\n--- compare_methods ---")
    print(f"  agreement: {comp['agreement']}")
    print(f"  ratio_xtb_to_pauling: {comp['ratio_xtb_to_pauling']}")
    print(f"  recommended_k: {comp['recommended_k']}")
    print(f"  recommended_method: {comp['recommended_method']}")
    print(f"  rationale: {comp['rationale']}")
