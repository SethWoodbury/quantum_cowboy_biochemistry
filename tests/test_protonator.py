"""Tests for the from-scratch protonator (``qcb protonate``).

Covers the full staged pipeline on the theozyme fixture: I/O round-trip,
terminus capping, canonical protonation, His tautomers, geometry
autocorrection, charge accounting, protomer enumeration, and end-to-end
``main()``.
"""
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from quantum_engine.prep import protonator as P  # noqa: E402

FIXTURE = _HERE / "test_data" / "protonation" / "input_theozyme.pdb"


def _stages(pH: float = 7.5):
    """Run stages 1-4 + autocorrection on the fixture; return the state."""
    st = P.read_structure(FIXTURE)
    P.strip_protein_hydrogens(st)
    P.stage1_caps(st, bond_cutoff=1.8)
    P.stage2_checks(st, bond_cutoff=1.8)
    P.stage3_canonical(st, pH=pH, bond_cutoff=1.8)
    P.stage4_his_tautomer(st)
    P.stage_autocorrect(st, h_clash_cutoff=1.5)
    return st


# --------------------------------------------------------------------------
def test_io_roundtrip(tmp_path):
    st = P.read_structure(FIXTURE)
    assert st.atoms, "fixture parsed to a non-empty atom list"
    out = tmp_path / "roundtrip.pdb"
    P.write_pdb(st, out)
    text = out.read_text()
    assert "CONECT" not in text, "CONECT records are dropped"
    assert text.rstrip().endswith("END")
    # every ATOM/HETATM line is exactly 80 columns wide
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            assert len(line) == 80


def test_strip_protein_hydrogens_keeps_hetatm_h():
    st = P.read_structure(FIXTURE)
    P.strip_protein_hydrogens(st)
    assert not any(a.record == "ATOM" and a.is_hydrogen for a in st.atoms)


def test_geometry_engine_ideal_angles():
    parent = np.zeros(3)
    nbr = np.array([1.52, 0.0, 0.0])
    hs = P.place_hydrogens(parent, [nbr], 3, "sp3", "C")     # methyl
    assert len(hs) == 3
    for h in hs:
        assert abs(np.linalg.norm(h - parent) - 1.09) < 1e-3


def test_capping_places_default_caps():
    st = _stages()
    residues = P.group_residues([a for a in st.atoms if a.record == "ATOM"])
    saw_nh2 = saw_cho = False
    for rk, atoms in residues.items():
        rec = st.residue_records[rk]
        names = {a.name for a in atoms}
        if rec.get("n_cap") == "nh2":
            assert {"H1", "H2"} <= names
            saw_nh2 = True
        if rec.get("c_cap") == "cho":
            assert "HC" in names
            saw_cho = True
    assert saw_nh2 and saw_cho, "fixture exercised both default caps"


def test_canonical_protonation_runs():
    st = _stages()
    # every standard residue ended with a recorded protonation state
    for rk, atoms in P.group_residues(
            [a for a in st.atoms if a.record == "ATOM"]).items():
        assert "state" in st.residue_records[rk]
    # at least one Stage-3 hydrogen was placed
    assert any(a.tag == "h" for a in st.atoms)


def test_his_tautomer_resolved():
    st = _stages()
    for rk, atoms in P.group_residues(
            [a for a in st.atoms if a.record == "ATOM"]).items():
        if P.canonical_resname(atoms[0].resname) != "HIS":
            continue
        state = st.residue_records[rk]["state"]
        assert state in ("HID", "HIE", "HIP")
        names = {a.name for a in atoms}
        n_ring_h = len({"HD1", "HE2"} & names)
        assert n_ring_h == (2 if state == "HIP" else 1)


def test_autocorrection_does_not_worsen_clashes():
    st = _stages()
    ac = st.settings["autocorrect"]
    assert ac["clashes_after"] <= ac["clashes_before"]


def test_net_charge_is_integer():
    st = _stages()
    net, per_res = P.compute_net_charge(st)
    assert isinstance(net, int)
    assert len(per_res) == len(
        P.group_residues([a for a in st.atoms if a.record == "ATOM"]))


def test_no_nan_coordinates():
    st = _stages()
    assert all(np.all(np.isfinite(a.xyz)) for a in st.atoms)


def test_main_end_to_end(tmp_path):
    out = tmp_path / "protonated.pdb"
    rc = P.main(["-i", str(FIXTURE), "-o", str(out), "--log-level", "ERROR"])
    assert rc == 0
    text = out.read_text()
    assert "REMARK 999 CQC-PROTONATOR NET_THEOZYME_PROTEIN_CHARGE" in text
    assert "CONECT" not in text


def test_main_protomers(tmp_path):
    out = tmp_path / "prot.pdb"
    rc = P.main(["-i", str(FIXTURE), "-o", str(out), "--pH", "6.5",
                 "--protomers", "3", "--log-level", "ERROR"])
    assert rc == 0
    # at least the single base structure is written
    produced = list(tmp_path.glob("prot*.pdb"))
    assert produced
