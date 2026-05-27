"""Tests for ``quantum_engine.opt`` — the modular optimizer factory.

These don't run any actual minimization (no MACE / GPU needed) — they
just exercise the factory + result schema + legacy-translation paths.
"""
from __future__ import annotations

import pytest


def test_list_backends_includes_all_known():
    from quantum_engine.opt import list_backends, BACKENDS
    backends = list_backends()
    assert backends == [
        "ase-lbfgs", "ase-fire", "ase-bfgs",
        "torch-sim-fire", "torch-sim-lbfgs",
    ]
    assert set(backends) == set(BACKENDS.keys())


def test_make_optimizer_each_backend_constructs():
    from quantum_engine.opt import make_optimizer, list_backends
    for name in list_backends():
        opt = make_optimizer(name, fmax=0.05, max_steps=10)
        assert opt.name == name
        assert opt.fmax == 0.05
        assert opt.max_steps == 10


def test_make_optimizer_rejects_unknown_backend():
    from quantum_engine.opt import make_optimizer
    with pytest.raises(ValueError, match="Unknown optimizer backend"):
        make_optimizer("ase-LBFGS")  # case-sensitive
    with pytest.raises(ValueError, match="Unknown optimizer backend"):
        make_optimizer("does-not-exist")


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
