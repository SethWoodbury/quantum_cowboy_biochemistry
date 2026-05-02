"""
Machine-learning force fields and the algorithms that drive them.

This subpackage holds two kinds of code:

  1. **Calc-agnostic primitives** that happen to be used downstream of an
     MLFF: :mod:`metadynamics`, :mod:`interpolation`, :mod:`cv_spring`,
     :mod:`dimer`. They work with any ASE calculator.

  2. **MLFF-specific helpers**: charge-aware MACE wrappers, post-MD CIF
     writing, etc.

Calculator instantiation lives in :mod:`quantum_engine.calc.factory`
(single source of truth via :data:`quantum_engine.config.MACE_MODELS`).
For Sella saddle-search etc. used by ``ops/saddle.py``, see
:mod:`quantum_engine.mlff.irc`.
"""
