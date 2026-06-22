"""SO3LR energy-family routing (quantum_engine.calc.factory + site registry).

SO3LR is a JAX/orbax MLFF that runs in its OWN sidecar, so it is not importable
in the main (torch) test env. These tests check the *routing* — family dispatch,
the alias registry, non-shadowing of other families, and the actionable sidecar
ImportError — NOT a live force evaluation (that needs the so3lr sidecar + a GPU).
"""
from __future__ import annotations

import pytest

from quantum_engine.calc.factory import (
    ENERGY_FAMILIES, _family_of, list_models, make_calc)

SO3LR_ALIASES = ["so3lr", "so3lr-s", "so3lr-m", "so3lr-l"]


@pytest.mark.parametrize("alias", SO3LR_ALIASES)
def test_so3lr_routes_to_so3lr_family(alias):
    assert _family_of(alias) == "so3lr"


def test_so3lr_family_label_registered():
    assert "so3lr" in ENERGY_FAMILIES.labels()


def test_so3lr_does_not_shadow_other_families():
    # Registering the so3lr predicate must not steal aliases from the families
    # already present (so3lr's predicate is startswith("so3lr"), which is disjoint).
    assert _family_of("mace-omol") == "mace"
    assert _family_of("uma-s-1p1") == "uma"
    assert _family_of("orb-mol") == "orb"
    assert _family_of("esen-sm-conserving") == "esen"
    assert _family_of("aimnet2-rxn") == "aimnet"


@pytest.mark.parametrize("alias", SO3LR_ALIASES)
def test_so3lr_in_model_registry(alias):
    models = list_models()
    assert alias in models, f"{alias} missing from site.MACE_MODELS"
    # Registered as the params.pkl INSIDE the staged v2-beta per-size workdir, so
    # the factory's os.path.isfile() resolution + "missing on disk" errors work;
    # _make_so3lr derives the enclosing workdir from this.
    assert models[alias].endswith(f"/{alias}/params.pkl"), models[alias]
    assert "so3lr-v2-beta" in models[alias]


def test_make_calc_so3lr_raises_actionable_sidecar_error():
    # so3lr (JAX) isn't importable in the torch test env, so make_calc must raise
    # a clear ImportError that names the sidecar / build recipe — NOT a bare
    # ModuleNotFoundError the user can't act on.
    with pytest.raises(ImportError) as exc:
        make_calc("so3lr-m", device="cpu")
    msg = str(exc.value)
    assert "SO3LR" in msg
    assert "so3lr_sidecar.def" in msg or "sidecar" in msg.lower()
