"""Unit conversion constants used throughout quantum_engine.

Single source of truth — every module in the codebase that converts
between eV / kJ/mol / kcal/mol / Hartree should import from here, not
redeclare the literal. This stops the "EV_TO_KCAL = 23.0609 in 15
files" footgun.

Where ASE provides a constant we re-export it under the name we use
in qcb code (and pin the literal so this module is self-contained
even if ASE is absent during static analysis).
"""

# Energy: 1 eV = 23.060547... kcal/mol = 96.485332... kJ/mol = 0.036749405... Eh
EV_TO_KCAL: float = 23.060547830619026
EV_TO_KJ: float = 96.48533212331003
EV_TO_HARTREE: float = 0.03674930495120813

# kJ/mol → kcal/mol  (1 kcal == 4.184 kJ exactly, by definition)
KJ_TO_KCAL: float = 1.0 / 4.184
KCAL_TO_KJ: float = 4.184

# Boltzmann constant in commonly-used forms.
KB_KJ_PER_MOL_K: float = 8.314462618e-3
KB_EV_PER_K: float = 8.617333262145e-5

# Vibrational frequency: sqrt(eV/Å²/amu) → cm⁻¹. Used by finite-difference
# Hessian frequency analysis (ops.freq, ops.expanded_hessian).
FREQ_CONV: float = 521.47083

# Inverse for completeness (rarely used):
KCAL_TO_EV: float = 1.0 / EV_TO_KCAL
KJ_TO_EV: float = 1.0 / EV_TO_KJ

__all__ = [
    "EV_TO_KCAL", "EV_TO_KJ", "EV_TO_HARTREE",
    "KJ_TO_KCAL", "KCAL_TO_KJ",
    "KCAL_TO_EV", "KJ_TO_EV",
    "KB_KJ_PER_MOL_K", "KB_EV_PER_K",
    "FREQ_CONV",
]
