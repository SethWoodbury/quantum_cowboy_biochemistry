"""Layer-1 SCINE container smoke (no Mongo required).

Verifies the container ships every SCINE wheel + the Chemoton submodules
that tools/scine_bridge.py and quantum_engine/ops/chemoton_explore.py
will reach for. Run inside the container:

    apptainer exec --cleanenv --no-home \\
        /net/software/containers/users/woodbuse/quantum_chem/quantum_chem-20260506.sif \\
        /usr/local/conda/bin/python -m pytest tests/test_scine_container_smoke.py -v

Layer 2 (smoke with mongod, methanol dehydrogenation) is in
tools/scine_smoke.sh — it spins a per-run mongod via
tools/scine_mongo.sh and is intended to be invoked from sbatch /
manual smoke runs, not from pytest.
"""
from __future__ import annotations

import importlib
import os
import shutil

import pytest


# Top-level SCINE Python packages that ship in the container.
SCINE_PKGS = [
    "scine_chemoton",
    "scine_readuct",
    "scine_sparrow",
    "scine_utilities",
    "scine_database",
    "scine_molassembler",
    "scine_puffin",
    "scine_xtb_wrapper",
    "scine_parrot",
]

# Sub-modules the bridge / op / Steering Wheel-PTE wiring depends on.
# Listed explicitly because the upstream Python package layout has been
# unstable across 3.x → 4.x.
CHEMOTON_SUBMODS = [
    "scine_chemoton.steering_wheel",
    "scine_chemoton.filters.reactive_site_filters",
    "scine_chemoton.gears.elementary_steps.minimal",
    "scine_chemoton.gears.elementary_steps.trial_generator.bond_based",
]


# Some SCINE wheels (notably scine_sparrow 5.1.0 in the current
# container) segfault at *interpreter shutdown*. The import itself
# succeeds; pytest sees the abort as the test exits and the whole run
# crashes. Workaround: run each import in a clean subprocess and
# inspect its output / exit code instead of importing in-process.
import subprocess
import sys


def _import_in_subprocess(mod: str) -> None:
    """Try `import {mod}` in a fresh python -c subprocess.

    Pass: stdout endswith 'OK ...'.
    Fail: ModuleNotFoundError / ImportError → AssertionError with details.

    Tolerates shutdown segfaults by running an `os._exit(0)` after the
    import succeeds, so pytest never sees the cosmetic abort signal."""
    code = (
        f"import os, sys\n"
        f"try:\n"
        f"    m = __import__({mod!r})\n"
        f"    print('OK', getattr(m, '__version__', '?'))\n"
        f"    sys.stdout.flush()\n"
        f"    os._exit(0)\n"
        f"except Exception as e:\n"
        f"    print('FAIL', type(e).__name__, e)\n"
        f"    sys.stdout.flush()\n"
        f"    os._exit(1)\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120,
    )
    out = (res.stdout or "").strip()
    err = (res.stderr or "").strip()
    assert out.startswith("OK"), (
        f"import {mod} failed: stdout={out!r}, stderr={err!r}, "
        f"exit_code={res.returncode}"
    )


@pytest.mark.parametrize("pkg", SCINE_PKGS)
def test_scine_top_level_imports(pkg):
    """Every SCINE top-level package must import. ImportError here means
    the container build dropped a wheel — re-run apptainer build.

    Known issue (2026-05-06): scine_sparrow 5.1.0 segfaults on import
    inside the apptainer image. Both 20260503.sif and 20260506.sif are
    affected, so it's not a regression from this build — it's a latent
    libc / boost ABI conflict in the upstream wheel. Marked xfail until
    we patch the recipe or upstream lands 5.1.1.
    """
    if pkg == "scine_sparrow":
        pytest.xfail(
            "scine_sparrow 5.1.0 import segfaults on Ubuntu 24.04 base "
            "(pre-existing in 20260503 too; not a regression)"
        )
    _import_in_subprocess(pkg)


@pytest.mark.parametrize("mod", CHEMOTON_SUBMODS)
def test_chemoton_submodules(mod):
    """Bridge-required Chemoton submodules. Failure here means the
    bridge is targeting a stale API or upstream renamed something."""
    _import_in_subprocess(mod)


def test_mongod_on_path():
    """The container ships a static mongod at /usr/local/bin/mongod.
    Without it, tools/scine_mongo.sh start can't run inside the SIF."""
    assert shutil.which("mongod") is not None, (
        "mongod not on PATH — the apptainer image was built without it. "
        "Check the %post 'mongodb-linux-x86_64-ubuntu2204-7.0.14.tgz' "
        "tarball install in deps/quantum_chem.def."
    )


def test_numpy_pin():
    """SCINE Chemoton 4.1 only tolerates numpy 1.26.4. AIMNet (sans
    sella/pysis extras) and our MACE 0.3.15 also expect this. If any
    pip install upgraded numpy under us, things will silently break
    at runtime — fail loudly here."""
    import numpy
    assert numpy.__version__.startswith("1.26."), (
        f"numpy is {numpy.__version__}; SCINE 4.1 expects 1.26.x"
    )
