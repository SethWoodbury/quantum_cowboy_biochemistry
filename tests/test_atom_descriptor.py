"""Flexible atom-descriptor resolution — built on the real OPAA di-Zn active site."""
from __future__ import annotations

import pytest

from quantum_engine.io.atom_descriptor import (
    AtomTable, best_descriptor, resolve_atom_descriptor)


# Mirror the user's HETATM block (+ a SUB substrate carrying P1/O7).
#  idx  chain resid resname name
#  0    A     519   ZN      ZN     <- two zincs, same resname, different resid
#  1    A     520   ZN      ZN
#  2    N     998   OHX     O3     <- bridging hydroxide (unique resname)
#  3    N     998   OHX     H9
#  4    Z     999   SUB     P1     <- substrate phosphorus
#  5    Z     999   SUB     O7     <- substrate leaving-group oxygen
TABLE = AtomTable(
    chains=["A", "A", "N", "N", "Z", "Z"],
    resids=[519, 520, 998, 998, 999, 999],
    resnames=["ZN", "ZN", "OHX", "OHX", "SUB", "SUB"],
    names=["ZN", "ZN", "O3", "H9", "P1", "O7"],
)


@pytest.mark.parametrize("token,expected", [
    ("OHX-O3", 2),          # RESNAME-ATOM (unique resname)
    ("SUB-P1", 4),
    ("SUB-O7", 5),
    ("N998-O3", 2),         # <Chain><ResNo>-ATOM
    ("A519-ZN", 0),
    ("A520-ZN", 1),
    ("ZN519-ZN", 0),        # <RESNAME><ResNo>-ATOM (disambiguates the two zincs)
    ("ZN520-ZN", 1),
    ("A:519:ZN", 0),        # explicit CHAIN:RESID:ATOM
    ("2", 2),               # raw 0-based index
    (4, 4),
    ("ohx-o3", 2),          # case-insensitive
])
def test_resolves_unique(token, expected):
    assert resolve_atom_descriptor(token, TABLE) == expected


@pytest.mark.parametrize("token,expected", [
    ("serial:1", 0),        # 1-based PDB serial (file order)
    ("serial:5", 4),        # SUB-P1
    ("0:4", 4),             # explicit 0-based index
    ("SUB:999:P1", 4),      # RESNAME:RESID:ATOM colon form
    ("A:519:ZN", 0),        # CHAIN:RESID:ATOM colon form
])
def test_resolve_atom_superset_forms(token, expected):
    from quantum_engine.io.atom_descriptor import resolve_atom
    assert resolve_atom(token, TABLE) == expected


def test_bare_int_policy_index_vs_serial():
    from quantum_engine.io.atom_descriptor import resolve_atom
    assert resolve_atom("4", TABLE, bare_int="index") == 4      # 0-based
    assert resolve_atom("5", TABLE, bare_int="serial") == 4      # 1-based serial -> idx 4


def test_ambiguous_two_zincs_raises():
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        resolve_atom_descriptor("ZN-ZN", TABLE)   # which zinc?


def test_no_match_raises():
    with pytest.raises(ValueError, match="matched no atom"):
        resolve_atom_descriptor("OHX-O9", TABLE)


def test_out_of_range_index_raises():
    with pytest.raises(ValueError, match="out of range"):
        resolve_atom_descriptor("99", TABLE)


def test_best_descriptor_prefers_concise_unique():
    # OHX O3: resname is unique → the short RESNAME-ATOM form
    assert best_descriptor(TABLE, 2) == "OHX-O3"
    # zinc: RESNAME-ATOM "ZN-ZN" is ambiguous → falls back to RESNAME+ResNo
    assert best_descriptor(TABLE, 0) == "ZN519-ZN"
    assert best_descriptor(TABLE, 1) == "ZN520-ZN"


def test_every_atom_has_a_unique_descriptor():
    # the HETATM-uniqueness guarantee the protonator enforces
    for i in range(len(TABLE)):
        d = best_descriptor(TABLE, i)
        assert d is not None and resolve_atom_descriptor(d, TABLE) == i


def test_genuinely_unaddressable_atom_returns_none():
    # two atoms identical in chain+resid+name → neither has a unique descriptor
    dup = AtomTable(chains=["A", "A"], resids=[1, 1], resnames=["X", "X"], names=["C", "C"])
    assert best_descriptor(dup, 0) is None
