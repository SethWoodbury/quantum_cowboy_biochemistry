"""Smoke tests — minimal coverage of the public surface.

These run on import + a tiny H2O system and don't need any MLFF model
or external binary. They exist to catch the kind of regression we hit
in the recent reshuffles (a missing import, a renamed module, a
duplicate registry getting out of sync, an empty pipeline call sig).

Run:  pytest tests/test_smoke.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────────────────────────────
# Test 1 — every public subpackage imports cleanly
# ─────────────────────────────────────────────────────────────────────

def test_subpackage_imports():
    """Each top-level quantum_engine subpackage must import without
    side effects or missing-import errors. Catches the kind of bug
    where renaming a module leaves a stale reference in __init__."""
    import quantum_engine
    assert quantum_engine.__version__

    # Each subpackage importable
    from quantum_engine import (        # noqa: F401
        analysis, calc, config, data, io,
        mlff, ops, pipelines, prep, qm, select, slurm, units,
    )


def test_units_constants_consistent():
    """Sanity-check the EV_TO_KCAL value used in ~15 files."""
    from quantum_engine.units import EV_TO_KCAL, EV_TO_KJ, KJ_TO_KCAL
    # 1 eV = 23.06... kcal/mol, kJ/kcal = 4.184 exactly
    assert abs(EV_TO_KCAL - 23.060547) < 1e-3
    assert abs(EV_TO_KJ * KJ_TO_KCAL - EV_TO_KCAL) < 1e-6


# ─────────────────────────────────────────────────────────────────────
# Test 2 — Pipeline contract
# ─────────────────────────────────────────────────────────────────────

def test_pipeline_no_steps():
    """An empty pipeline should run cleanly and produce an empty
    history."""
    from quantum_engine.pipelines import Pipeline, Context
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        ctx = Context(atoms=None, calc=None, outdir=Path(td))
        Pipeline([], write_summary=False).run(ctx)
        assert ctx.history == {}


def test_pipeline_step_protocol():
    """A custom Step satisfying the protocol should plug in unchanged."""
    from quantum_engine.pipelines import Pipeline, Context, StepResult
    import tempfile

    class _Counter:
        name = "count"
        def __init__(self): self.n = 0
        def run(self, ctx):
            self.n += 1
            return StepResult(name=self.name, status="completed",
                              extra={"n": self.n})

    with tempfile.TemporaryDirectory() as td:
        ctx = Context(atoms=None, calc=None, outdir=Path(td))
        s = _Counter()
        Pipeline([s], write_summary=True).run(ctx)
        assert ctx.history["count"].extra["n"] == 1
        assert (Path(td) / "pipeline_summary.json").is_file()


def test_pipeline_duplicate_step_names_rejected():
    from quantum_engine.pipelines import Pipeline, StepResult

    class _S:
        def __init__(self, name): self.name = name
        def run(self, ctx): return StepResult(name=self.name)

    with pytest.raises(ValueError, match="duplicate step names"):
        Pipeline([_S("a"), _S("a")])


def test_orchestrator_skeleton_assembles():
    """The 9-stage active_site_ts orchestrator should at least
    construct (the stubs raise NotImplementedError when run, that's
    fine for the smoke test)."""
    from enz_qc_pipelines.active_site_ts import build_active_site_ts_pipeline
    p = build_active_site_ts_pipeline(cv_indices=(0, 1, 2))
    assert len(p.steps) == 9
    expected = [
        "protonate", "vacuum_neb_guess", "crest_conformers",
        "rigid_dock", "sidechain_torsion_opt", "pick_best_pose",
        "mtd_pocket", "irc", "barrier_report",
    ]
    assert [s.name for s in p.steps] == expected


# ─────────────────────────────────────────────────────────────────────
# Test 3 — calculator factory
# ─────────────────────────────────────────────────────────────────────

def test_list_models_non_empty():
    """list_models() should return the registry from
    quantum_engine.config.MACE_MODELS — single source."""
    from quantum_engine.calc import list_models
    from quantum_engine.config import MACE_MODELS
    models = list_models()
    assert models, "expected at least one model in registry"
    # Every key in the canonical config should appear.
    assert set(MACE_MODELS.keys()) <= set(models.keys())


def test_make_calc_unknown_raises():
    """A bogus model name should raise FileNotFoundError with a
    helpful 'Available registry keys' message."""
    from quantum_engine.calc import make_calc
    with pytest.raises(FileNotFoundError, match="Available registry keys"):
        make_calc(model="not_a_real_model_xyz", device="cpu")


# ─────────────────────────────────────────────────────────────────────
# Test 4 — selectors / constraints
# ─────────────────────────────────────────────────────────────────────

def test_select_module_imports():
    """`from quantum_engine.io import parse_constraints` (the
    backwards-compat path) and `from quantum_engine.select import
    parse_constraints` (the new canonical path) must both work."""
    from quantum_engine.io import parse_constraints as a
    from quantum_engine.select import parse_constraints as b
    assert a is b


def test_bond_breaking_defs_loaded():
    """The ligand bond-breaking dict (extracted from tools/run_neb_ts.py
    in Phase 2) should be loadable via the proper module path."""
    from quantum_engine.data import BOND_BREAKING_DEFS
    assert "YYE" in BOND_BREAKING_DEFS
    yye = BOND_BREAKING_DEFS["YYE"]
    assert isinstance(yye, list) and yye
    a, b, dist, direction = yye[0]
    assert isinstance(a, str) and isinstance(b, str)
    assert direction in ("attractive", "repulsive")


# ─────────────────────────────────────────────────────────────────────
# Test 5 — CLI entry point
# ─────────────────────────────────────────────────────────────────────

def test_cli_help_runs():
    """`qcb --help` should print the top-level usage without error."""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "quantum_engine.cli", "--help"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH":
             str(Path(__file__).resolve().parent.parent)},
    )
    # argparse exits 0 on --help
    assert result.returncode == 0, result.stderr
    assert "qcb" in result.stdout
    # Spot-check a few subcommands appear
    for sub in ("sp", "opt", "md", "neb", "mtd", "ts", "protonate"):
        assert sub in result.stdout, f"missing subcommand {sub!r}"
