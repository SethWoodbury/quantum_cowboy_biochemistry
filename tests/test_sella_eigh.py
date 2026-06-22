"""Regression tests for the Sella eigh-driver monkey-patch.

The bug: ``_patched_eigh`` forced a standard-only LAPACK driver (evd/evr/evx/ev)
onto *every* ``scipy.linalg.eigh`` call. Sella also makes GENERALIZED
``eigh(A, B)`` calls, and forcing a standard driver onto those raises
``ValueError: "evd" does not accept input b array for standard eigenvalue
problems`` — which aborted the saddle search on a real di-Zn cluster.
"""
from __future__ import annotations

import numpy as np
import pytest

sla = pytest.importorskip("scipy.linalg")

from quantum_engine.qm.sella import _patched_eigh


def test_patched_eigh_forces_driver_on_standard_problem():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    with _patched_eigh("evd"):
        w, _ = sla.eigh(A)               # standard eigh(A) -> driver evd
    assert np.allclose(sorted(w), sorted(np.linalg.eigvalsh(A)))


def test_patched_eigh_passes_through_generalized_problem():
    # Forcing the standard-only driver onto a generalized eigh(A, B) used to
    # raise; it must now fall back to SciPy's default generalized driver.
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    B = np.array([[2.0, 0.0], [0.0, 1.0]])
    w_ref, _ = sla.eigh(A, B)
    with _patched_eigh("evd"):
        w, _ = sla.eigh(A, B)            # must NOT raise
    assert np.allclose(sorted(w), sorted(w_ref))


def test_patched_eigh_generalized_via_kwarg():
    A = np.array([[4.0, 0.5], [0.5, 1.0]])
    B = np.array([[1.0, 0.0], [0.0, 2.0]])
    w_ref, _ = sla.eigh(A, b=B)
    with _patched_eigh("evr"):
        w, _ = sla.eigh(A, b=B)          # b passed by keyword -> still generalized
    assert np.allclose(sorted(w), sorted(w_ref))
