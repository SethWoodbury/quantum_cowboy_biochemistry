"""
quantum_engine — modular tools for enzyme quantum chemistry.
============================================================

Standalone tools that compose into pipelines. Each subpackage is
independently importable; the same primitives drive the qcb CLI, the
declarative ``qcb run config.yaml`` runner, and the orchestrator
pipelines under :mod:`enz_qc_pipelines`.

Subpackages
-----------
  prep       Active-site extraction, capping, protonation, charge calc
  ops        Atomic-scale operations (sp, opt, md, freq, neb, irc, mtd, ts, …)
  qm         External QM job generation + parsing (Gaussian, ORCA, xTB)
  mlff       In-process ML force field calculators + algorithms
             (metadynamics, interpolation, dimer, cv-spring, …)
  analysis   Free-energy surface, KDE, barriers, geometry scoring
  io         Structure / config I/O (PDB, CIF, YAML schema, selectors)
  calc       Calculator factory (single source of model paths)
  slurm      SLURM array submitters (multi-walker MTD, …)
  pipelines  Step / Pipeline contract for chaining ops in Python

The CLI binary is ``qcb`` (see ``pyproject.toml`` ``[project.scripts]``).
"""

__version__ = "0.2.0"
