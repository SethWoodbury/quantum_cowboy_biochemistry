"""Phase 5 de-hardcoding: suggest_cv_targets + endpoint-consistency safeguard."""
from __future__ import annotations

import pytest
from ase import Atoms

from quantum_engine.mlff.cv_spring import CV_TARGET_HEURISTICS, suggest_cv_targets
from quantum_engine.ops.path_search import (
    check_endpoint_consistency, reorder_product_to_match)


# ---- suggest_cv_targets: no silent reaction-specific default ----
def test_suggest_cv_targets_explicit_wins():
    assert suggest_cv_targets(-1.0, 1.0) == (-1.0, 1.0)


def test_suggest_cv_targets_reaction_type_heuristic():
    r, p = suggest_cv_targets(reaction_type="sn2-at-phosphorus")
    assert (r, p) == CV_TARGET_HEURISTICS["sn2-at-phosphorus"]
    # explicit value overrides the heuristic for that side only
    assert suggest_cv_targets(reactant_s=-3.0, reaction_type="sn2-at-carbon") == (
        -3.0, CV_TARGET_HEURISTICS["sn2-at-carbon"][1])


def test_suggest_cv_targets_requires_explicit_or_type():
    with pytest.raises(ValueError, match="No reaction-specific default"):
        suggest_cv_targets()              # the old silent -2.0/+2.5 is gone
    with pytest.raises(ValueError, match="No reaction-specific default"):
        suggest_cv_targets(reactant_s=-2.0)   # only one side, no type


def test_suggest_cv_targets_unknown_type_raises():
    with pytest.raises(ValueError, match="unknown reaction_type"):
        suggest_cv_targets(reaction_type="diels-alder")


# ---- endpoint-consistency safeguard for double-ended path methods ----
def _mol(symbols, n=None):
    n = n if n is not None else len(symbols)
    return Atoms(symbols, positions=[[i, 0, 0] for i in range(n)])


def test_endpoints_consistent_ok():
    rep = check_endpoint_consistency(_mol("OCO"), _mol("OCO"))
    assert rep["ok"] and not rep["needs_reorder"] and rep["n_atoms"] == 3


def test_endpoints_atom_count_mismatch():
    with pytest.raises(ValueError, match="atom-count mismatch"):
        check_endpoint_consistency(_mol("OCO"), _mol("OC"))


def test_endpoints_element_order_mismatch():
    with pytest.raises(ValueError, match="element order mismatch"):
        check_endpoint_consistency(_mol("OCO"), _mol("COO"))


def test_atom_map_must_be_bijection():
    with pytest.raises(ValueError, match="bijection"):
        check_endpoint_consistency(_mol("OCO"), _mol("OCO"),
                                   atom_map={0: 0, 1: 1})   # missing index 2


def test_atom_map_mapped_element_mismatch():
    # map reactant O(0)->product idx1 which is C -> element mismatch
    with pytest.raises(ValueError, match="different element"):
        check_endpoint_consistency(_mol("OCO"), _mol("OCO"),
                                   atom_map={0: 1, 1: 0, 2: 2})


def test_atom_map_valid_permutation_flags_reorder():
    # product is OCO but atoms 0 and 2 (both O) swapped — valid bijection
    rep = check_endpoint_consistency(_mol("OCO"), _mol("OCO"),
                                     atom_map={0: 2, 1: 1, 2: 0})
    assert rep["ok"] and rep["needs_reorder"] is True


def test_reorder_product_to_match():
    p = Atoms("OCN", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    out = reorder_product_to_match(p, {0: 2, 1: 1, 2: 0})
    assert [str(s) for s in out.symbols] == ["N", "C", "O"]
    assert list(out.positions[0]) == [2, 0, 0]


# ---- ligand_bonds reference table is well-formed + has non-PTE examples ----
def test_ligand_bonds_has_non_pte_examples_and_shape():
    from quantum_engine.data import BOND_BREAKING_DEFS
    assert {"MCL", "EST"} <= set(BOND_BREAKING_DEFS)   # generic non-PTE examples
    for code, defs in BOND_BREAKING_DEFS.items():
        for a, b, dist, direction in defs:
            assert isinstance(a, str) and isinstance(b, str)
            assert isinstance(dist, (int, float))
            assert direction in ("attractive", "repulsive")
