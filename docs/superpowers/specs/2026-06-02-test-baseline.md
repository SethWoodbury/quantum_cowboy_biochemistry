# Test Baseline — before cleanup (2026-06-02)

Captured before any cleanup changes, so regressions can be distinguished from pre-existing failures.

- **Container:** `quantum_chem-20260506.sif`, dev tree mounted via `PYTHONPATH=<repo>`.
- **Collection:** 146 tests collected across `tests/` + `tools/tests/`.
- **Fast core run** (`tests/test_smoke.py tests/test_opt_factory.py`): **25 passed, 2 failed**.
- Full suite not run end-to-end (some tests need GPU / network / slow xTB). The fast core + import
  checks are the regression guard for this deletion-heavy cleanup; re-run after Phases 2 and 3.

## Pre-existing failures (NOT regressions — do not attribute to cleanup)

1. `test_smoke.py::test_subpackage_imports`
   - Cause: `quantum_engine/config/schema.py:115` `from pydantic import field_validator` (pydantic **v2** API),
     but the container ships pydantic **v1** (`cannot import name 'field_validator'`).
   - Impact: `import quantum_engine.config` fails inside the container; the `qcb run <yaml>` declarative
     path is therefore broken in the container. **Deferred** to the notebook/env track (needs pydantic v2
     in the container, or a v1-compatible schema). Out of cleanup scope.

2. `test_smoke.py::test_make_calc_unknown_raises`
   - Cause: stale assertion. Test expects regex `Available registry keys`; the calc factory now raises
     `MACE model '...' not available locally and auto-download did not match any family (...)`.
   - Impact: cosmetic/test-only. Could be refreshed opportunistically when editing the calc factory.

## Success bar for cleanup
After each code-touching phase, the fast core run must show **no more than these same 2 failures**
(no new red), and `import quantum_engine[.subpkg]` must stay clean for non-config subpackages.
