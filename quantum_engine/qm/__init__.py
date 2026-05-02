"""
Quantum chemistry: Gaussian, ORCA, xTB input generation and SLURM submission.
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
