"""Molecular mechanics — OpenMM-driven MD with swappable force fields.

Where the rest of cowboy-qc leans on ASE's per-step Python loop for MD
(``quantum_engine.ops.md``, ``quantum_engine.ops.mtd``), this
subpackage uses OpenMM's CUDA-accelerated kernels. Force-field choice
is decoupled from the integrator:

* Classical:  AMBER / CHARMM / OPLS / GAFF (via OpenMM's built-in
  parameter sets or :mod:`openmmforcefields`).
* ML:         MACE / ANI / NequIP via ``openmm-ml``; arbitrary PyTorch
  models via ``openmm-torch``.
* QM/MM:      xtb (in-process via ASE) coupled into OpenMM as a custom
  force; PLUMED biasing on top via ``openmm-plumed``.

The same Pipeline ``Step`` adapters in
:mod:`quantum_engine.pipelines.steps` work whether the underlying MD
runs through ASE or OpenMM — pick the engine based on speed
requirements and force field.

See :mod:`.openmm` for the wrapper. Install via
``bash deps/build_openmm.sh``.
"""
