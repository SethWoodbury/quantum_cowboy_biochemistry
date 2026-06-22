"""protonator: non-protein charge is never assumed + canonical PDB columns.

Built on the real OPAA di-Zn active site (2x Zn(II), bridging OHX hydroxide,
SUB substrate) — the non-protein charge there is +2+2-1+0 = +3.
"""
from __future__ import annotations

import numpy as np
import pytest

from quantum_engine.prep.protonator import (
    Atom, ProtonationState, _atom_line, nonprotein_charge_total,
    report_hetatm_descriptors)


def _het(name, resname, chain, resid, element, xyz=(0.0, 0.0, 0.0), charge=""):
    return Atom(record="HETATM", name=name, resname=resname, chain=chain,
                resid=resid, icode="", xyz=np.array(xyz, float),
                element=element, charge=charge)


def _opaa():
    return ProtonationState(atoms=[
        _het("ZN", "ZN", "A", 519, "ZN", (17.162, 32.401, 10.627)),
        _het("ZN", "ZN", "A", 520, "ZN", (16.025, 29.925, 8.582)),
        _het("O3", "OHX", "N", 998, "O", (17.074, 31.622, 8.987), charge="1-"),
        _het("H9", "OHX", "N", 998, "H", (17.755, 30.885, 8.904)),
        _het("P1", "SUB", "Z", 999, "P", (15.0, 30.0, 9.0)),
        _het("O7", "SUB", "Z", 999, "O", (14.0, 29.0, 9.0)),
    ])


# ---- charge: NEVER assumed ----
def test_no_input_means_no_nonprotein_charge():
    total, per, warn = nonprotein_charge_total(_opaa(), {})
    assert total is None                 # not assumed, not measured

def test_per_resname_sums_over_instances():
    total, _, _ = nonprotein_charge_total(_opaa(), {"ZN": 2, "OHX": -1, "SUB": 0})
    assert total == 3                    # +2 +2 −1 +0 (two Zn instances)

def test_partial_declaration_errors_never_assumes_zero():
    with pytest.raises(SystemExit, match="UNDECLARED"):
        nonprotein_charge_total(_opaa(), {"ZN": 2})   # OHX, SUB left out

def test_total_override_wins_over_dict():
    total, _, _ = nonprotein_charge_total(
        _opaa(), {"ZN": 2, "OHX": -1, "SUB": 0}, nonprotein_charge_override=3)
    assert total == 3
    # override alone (no dict) also works
    assert nonprotein_charge_total(_opaa(), {}, nonprotein_charge_override=3)[0] == 3


# ---- canonical PDB columns (1-indexed spec; Python slices are col-1) ----
def test_pdb_columns_zinc_and_oxygen_canonical():
    atoms = _opaa().atoms
    zn = _atom_line(atoms[0], 260)
    assert zn[0:6] == "HETATM"
    assert zn[6:11] == "  260"      # serial, cols 7-11
    assert zn[12:16] == "ZN  "      # 2-char element → name left-justified at col 13
    assert zn[17:20] == " ZN"       # resname, cols 18-20
    assert zn[21] == "A"            # chain, col 22
    assert zn[22:26] == " 519"      # resSeq, cols 23-26
    assert zn[76:78] == "ZN"        # element, cols 77-78
    o3 = _atom_line(atoms[2], 262)
    assert o3[12:16] == " O3 "      # 1-char element → name at col 14
    assert o3[76:78] == " O"        # element O right-justified
    assert o3[78:80] == "1-"        # charge field preserved, cols 79-80


def test_every_hetatm_atom_is_addressable():
    st = _opaa()
    report_hetatm_descriptors(st)
    assert not any("NO unique" in w for w in st.warnings)
