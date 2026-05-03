"""
Quantum chemistry adapters.

Eager-imported (always work, no optional deps): Gaussian / ORCA input
generation + parsing, xTB ASE wrapper, SLURM submit helpers.

Lazy adapters — import directly from their module to avoid pulling
optional deps into the parent import chain. Each exposes a cheap
``<tool>_available()`` check before invocation:

  * ``quantum_engine.qm.crest``         — Grimme CREST conformer search
  * ``quantum_engine.qm.chemshell``     — ChemShell QM/MM (registration-walled)
  * ``quantum_engine.qm.scine``         — SCINE Chemoton + ReaDuct
  * ``quantum_engine.qm.pygsm``         — pyGSM single/double-ended GSM
  * ``quantum_engine.qm.pysisyphus``    — pysisyphus NEB / dimer / IRC
  * ``quantum_engine.qm.molecular_gsm`` — molecularGSM (C++ binary)
  * ``quantum_engine.qm.yarp``          — YARP stub (upstream dead — use scine)
"""

from quantum_engine.qm.gaussian import (
    gaussian_output_to_xyz,
    modify_gaussian_input,
    write_gaussian_input,
)
from quantum_engine.qm.orca import (
    modify_orca_input,
    write_orca_input,
)
from quantum_engine.qm.submit import (
    submit_job,
    write_gaussian_slurm,
    write_orca_slurm,
    write_slurm_script,
)
from quantum_engine.qm.xtb import (
    run_xtb_opt,
    write_xtb_input,
)

__all__ = [
    # Gaussian
    "write_gaussian_input",
    "modify_gaussian_input",
    "gaussian_output_to_xyz",
    # ORCA
    "write_orca_input",
    "modify_orca_input",
    # xTB
    "write_xtb_input",
    "run_xtb_opt",
    # SLURM
    "write_slurm_script",
    "submit_job",
    "write_gaussian_slurm",
    "write_orca_slurm",
]
