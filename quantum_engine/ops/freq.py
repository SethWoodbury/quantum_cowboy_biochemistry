"""Vibrational frequency analysis via finite-difference Hessian.

For partial Hessian (selected atoms), much cheaper than full system Hessian
while still resolving reactive modes correctly. See Wilson, Decius, Cross
"Molecular Vibrations" (1955) for the classical treatment.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from ase import Atoms

log = logging.getLogger("quantum_engine.ops.freq")

EV_TO_KCAL = 23.0609
# Conversion factor sqrt(eV/Å²/amu) → cm⁻¹
FREQ_CONV = 521.47083


def run(
    atoms: Atoms,
    calculator=None,
    outdir: str | Path = ".",
    constraint=None,
    indices: list[int] | None = None,
    delta: float = 0.02,
    method: str = "central",
    temperature_K: float = 298.15,
    **kwargs,
) -> dict:
    """Compute vibrational frequencies via finite-difference Hessian.

    Args:
        atoms, calculator, outdir, constraint: standard
        indices: atom indices to include in partial Hessian.
                 If None, include all non-constrained atoms.
        delta: finite-difference step in Å (default 0.02; large for ML potential noise)
        method: "central" (two-point FD) or "forward" (one-point FD, faster)
        temperature_K: temperature for thermodynamic corrections (H, S, G)

    Returns: dict with:
        frequencies_cm: np.ndarray of frequencies (cm⁻¹), negative = imaginary
        n_imaginary: count of imaginary modes
        modes: normal modes as (n_modes, n_atoms, 3) array
        zpe_eV: zero-point energy
        thermo: dict with H, S, G contributions (eV)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if atoms.calc is None:
        if calculator is None:
            raise ValueError("Need a calculator")
        atoms.calc = calculator

    if constraint is not None:
        if not isinstance(constraint, list):
            constraint = [constraint]
        atoms.set_constraint(constraint)

    # Resolve indices
    if indices is None:
        # Use all non-constrained atoms
        if atoms.constraints:
            c = atoms.constraints[0]
            if hasattr(c, "index"):
                idx_set = set(c.index.tolist() if hasattr(c.index, 'tolist') else c.index)
                indices = [i for i in range(len(atoms)) if i not in idx_set]
            else:
                indices = list(range(len(atoms)))
        else:
            indices = list(range(len(atoms)))

    n_free = len(indices)
    log.info(f"Frequencies: partial Hessian over {n_free} atoms, delta={delta}, method={method}")

    # Build Hessian by finite differences
    H = np.zeros((3 * n_free, 3 * n_free))
    pos0 = atoms.get_positions().copy()

    for a_idx, atom_i in enumerate(indices):
        for xyz_i in range(3):
            row = 3 * a_idx + xyz_i
            pos_plus = pos0.copy()
            pos_plus[atom_i, xyz_i] += delta
            atoms.set_positions(pos_plus)
            f_plus = atoms.get_forces()

            if method == "central":
                pos_minus = pos0.copy()
                pos_minus[atom_i, xyz_i] -= delta
                atoms.set_positions(pos_minus)
                f_minus = atoms.get_forces()
            else:
                atoms.set_positions(pos0)
                f_minus = atoms.get_forces()

            denom = (2 * delta) if method == "central" else delta
            for b_idx, atom_j in enumerate(indices):
                for xyz_j in range(3):
                    col = 3 * b_idx + xyz_j
                    H[row, col] = -(f_plus[atom_j, xyz_j] - f_minus[atom_j, xyz_j]) / denom

    atoms.set_positions(pos0)  # restore

    # Mass-weight
    masses = atoms.get_masses()
    m_atom = np.array([masses[i] for i in indices])
    m_atom = np.clip(m_atom, 1e-3, None)
    m_sqrt = np.repeat(np.sqrt(m_atom), 3)
    H_mw = H / np.outer(m_sqrt, m_sqrt)
    H_mw = 0.5 * (H_mw + H_mw.T)

    eigvals, eigvecs = np.linalg.eigh(H_mw)
    frequencies_cm = np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * FREQ_CONV

    # Reshape modes to (n_modes, n_atoms, 3) with zeros on constrained atoms
    n_modes = len(eigvals)
    modes = np.zeros((n_modes, len(atoms), 3))
    for k in range(n_modes):
        v = eigvecs[:, k]
        for a_idx, atom_i in enumerate(indices):
            modes[k, atom_i] = v[3 * a_idx : 3 * a_idx + 3]
    # Normalize each mode
    for k in range(n_modes):
        norm = np.linalg.norm(modes[k])
        if norm > 0:
            modes[k] /= norm

    n_imaginary = int(np.sum(frequencies_cm < -10))  # real modes <10 cm⁻¹ are usually rigid-body

    # Zero-point energy (sum over real positive modes)
    # ZPE = 0.5 * hv = 0.5 * (freq_cm * 1.23984e-4 eV)
    real_freqs = frequencies_cm[frequencies_cm > 10]  # ignore near-zero and imaginary
    zpe_eV = float(0.5 * np.sum(real_freqs * 1.23984e-4))  # cm⁻¹ → eV

    # Thermodynamic corrections (harmonic approximation, ideal gas)
    thermo = _thermochemistry(real_freqs, temperature_K)

    result = {
        "status": "completed",
        "atoms": atoms,
        "frequencies_cm": frequencies_cm.tolist(),
        "n_imaginary": n_imaginary,
        "lowest_mode_cm": float(frequencies_cm[0]),
        "zpe_eV": zpe_eV,
        "zpe_kcal_mol": zpe_eV * EV_TO_KCAL,
        "thermo_eV": thermo,
        "temperature_K": temperature_K,
        "outputs": {},
    }

    # Persist
    summary_path = outdir / "freq-summary.json"
    summary_path.write_text(json.dumps({
        k: v for k, v in result.items() if k not in ("atoms", "outputs")
    }, indent=2))
    result["outputs"]["summary"] = str(summary_path)

    # Write modes.npy for downstream IRC
    np.save(outdir / "frequencies_cm.npy", frequencies_cm)
    np.save(outdir / "modes.npy", modes)
    result["outputs"]["frequencies"] = str(outdir / "frequencies_cm.npy")
    result["outputs"]["modes"] = str(outdir / "modes.npy")

    log.info(f"  {len(frequencies_cm)} modes, {n_imaginary} imaginary, "
             f"lowest = {frequencies_cm[0]:.1f} cm⁻¹, ZPE = {zpe_eV:.4f} eV")
    return result


def _thermochemistry(real_freqs_cm: np.ndarray, T: float) -> dict:
    """Compute thermal corrections (harmonic, ideal gas) at temperature T (K).

    Returns dict of H_vib, S_vib, G_vib in eV.
    """
    kB = 8.617333262145e-5  # eV/K
    hc_over_kB = 1.438777  # cm·K

    if len(real_freqs_cm) == 0:
        return {"H_vib_eV": 0, "S_vib_eV_per_K": 0, "G_vib_eV": 0}

    # Vibrational temperature T_v = h*c*freq / kB
    T_v = hc_over_kB * real_freqs_cm
    ratio = T_v / T

    # H_vib = sum kB*T_v * [0.5 + 1/(exp(T_v/T) - 1)]  (includes ZPE)
    H_vib = np.sum(kB * T_v * (0.5 + 1 / (np.exp(ratio) - 1)))
    # S_vib = sum kB * [ (T_v/T)/(exp(T_v/T)-1) - ln(1 - exp(-T_v/T)) ]
    S_vib = np.sum(kB * (ratio / (np.exp(ratio) - 1) - np.log(1 - np.exp(-ratio))))
    G_vib = H_vib - T * S_vib

    return {
        "H_vib_eV": float(H_vib),
        "S_vib_eV_per_K": float(S_vib),
        "G_vib_eV": float(G_vib),
    }
