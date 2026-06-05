"""Tests for ``quantum_engine.opt`` — the modular optimizer factory.

These don't run any actual minimization (no MACE / GPU needed) — they
just exercise the factory + result schema + legacy-translation paths.
"""
from __future__ import annotations

import pytest


def test_list_backends_includes_all_known():
    from quantum_engine.opt import list_backends, BACKENDS
    backends = set(list_backends())
    # Registry-backed: the core + modern backends must all be present (order is
    # alphabetical now, so check membership, not a fixed list).
    assert {
        "ase-lbfgs", "ase-fire", "ase-bfgs",
        "ase-precon-lbfgs", "ase-fire2",
        "torch-sim-fire", "torch-sim-lbfgs",
    } <= backends
    assert backends == set(BACKENDS.keys())


def test_make_optimizer_each_backend_constructs():
    from quantum_engine.opt import make_optimizer, list_backends
    for name in list_backends():
        try:
            opt = make_optimizer(name, fmax=0.05, max_steps=10)
        except ImportError:
            # Backend dep absent in this env (e.g. FIRE2 on an older ASE) —
            # construction surfaces it; that's expected, skip it here.
            continue
        assert opt.name == name
        assert opt.fmax == 0.05
        assert opt.max_steps == 10


def test_make_optimizer_rejects_unknown_backend():
    from quantum_engine.opt import make_optimizer
    with pytest.raises(ValueError, match="Unknown optimizer backend"):
        make_optimizer("does-not-exist")


def test_make_optimizer_is_case_insensitive_and_alias_aware():
    """Registry lookups are case-insensitive + alias-aware (a UX improvement
    over the old case-sensitive dict)."""
    from quantum_engine.opt import make_optimizer
    assert make_optimizer("ASE-LBFGS").name == "ase-lbfgs"   # case-insensitive
    assert make_optimizer("lbfgs").name == "ase-lbfgs"       # alias


def test_register_optimizer_adds_new_backend():
    """A brand-new minimizer drops in via one register_optimizer() call — the
    core extensibility promise. Clean up after so we don't leak global state."""
    from quantum_engine.opt.factory import (
        OPTIMIZERS, register_optimizer, make_optimizer, list_backends)
    from quantum_engine.opt.ase_optimizers import AseLbfgsOptimizer

    class _MyMin(AseLbfgsOptimizer):
        name = "my-min"

    register_optimizer("my-min", lambda: _MyMin, aliases=("mine",))
    try:
        assert "my-min" in list_backends()
        assert make_optimizer("my-min").name == "my-min"
        assert make_optimizer("MINE").name == "my-min"   # alias + case-insensitive
    finally:
        OPTIMIZERS.unregister("my-min")
    assert "my-min" not in OPTIMIZERS


def test_torch_sim_lbfgs_stub_raises_on_run():
    from quantum_engine.opt import make_optimizer
    opt = make_optimizer("torch-sim-lbfgs", fmax=0.05, max_steps=10)
    with pytest.raises(NotImplementedError, match="Python 3.12"):
        opt.run(None)


def test_optresult_schema_matches_legacy_keys():
    """OptResult.to_dict() must include the keys the legacy
    ``quantum_engine.ops.opt.run`` callers read."""
    from quantum_engine.opt import OptResult
    r = OptResult(
        status="converged", converged=True, atoms=None,
        energy_eV=-1.0, energy_kcal_mol=-23.06,
        fmax_final=0.04, n_steps=42, backend="ase-lbfgs",
        wall_time_s=12.3, outputs={"log": "/tmp/x.log"},
    )
    d = r.to_dict()
    for key in ("status", "converged", "energy_eV", "energy_kcal_mol",
                "fmax_final", "n_steps", "outputs"):
        assert key in d, f"OptResult.to_dict() missing legacy key {key!r}"
    # New keys we added — must also be present
    assert "backend" in d
    assert "wall_time_s" in d


def test_legacy_translation_in_ops_opt():
    """Legacy ``optimizer="lbfgs"|"fire"|"bfgs"`` must translate to the
    matching ``ase-*`` backend, default ``ase-lbfgs``."""
    from quantum_engine.ops.opt import _resolve_backend
    assert _resolve_backend(backend=None, optimizer="lbfgs") == "ase-lbfgs"
    assert _resolve_backend(backend=None, optimizer="LBFGS") == "ase-lbfgs"
    assert _resolve_backend(backend=None, optimizer="fire") == "ase-fire"
    assert _resolve_backend(backend=None, optimizer="bfgs") == "ase-bfgs"
    # Default
    assert _resolve_backend(backend=None, optimizer=None) == "ase-lbfgs"
    # Explicit backend wins
    assert _resolve_backend(
        backend="torch-sim-fire", optimizer="lbfgs") == "torch-sim-fire"
    # Unknown legacy name → passes through (caller validates downstream)
    assert _resolve_backend(
        backend=None, optimizer="ase-fire") == "ase-fire"
