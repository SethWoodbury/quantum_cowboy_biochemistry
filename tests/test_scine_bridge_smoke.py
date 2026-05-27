"""Layer-1 smoke tests for ``tools/scine_bridge.py``.

Goal: catch import / signature regressions WITHOUT requiring MongoDB or any
running mongod. We exercise the lightest possible code paths:

  * ``scine_available()`` returns a bool
  * ``pdb_to_scine_cluster`` round-trips a tiny synthetic Zn + ligand PDB
  * ``make_scine_model`` constructs a usable ``db.Model``
  * ``make_pte_reactive_filter`` returns a non-empty filter array

Skipped automatically (with ``pytest.skip``) when SCINE isn't importable.

Run:  pytest tests/test_scine_bridge_smoke.py -v
"""
from __future__ import annotations

import sys
import tempfile
from collections import Counter
from pathlib import Path

import pytest

# Ensure repo root + tools/ are importable.
_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO, _REPO / "tools"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


@pytest.fixture(scope="module")
def bridge():
    """Import the bridge once per module; skip if SCINE unavailable."""
    pytest.importorskip("scine_chemoton")
    pytest.importorskip("scine_database")
    pytest.importorskip("scine_utilities")
    import scine_bridge as sb
    if not sb.scine_available():
        pytest.skip("scine_bridge.scine_available() returned False")
    return sb


@pytest.fixture(scope="module")
def tiny_pdb(tmp_path_factory):
    """Build a minimal 7-atom test PDB: 1 Zn, 1 OH⁻, 1 phosphate-like ligand.

    Roughly mimics PTE's bridging hydroxide + paraoxon fragment, just enough
    for cluster extraction to find ≥1 protein and ≥1 hetero residue.
    """
    pdb_text = (
        # A protein ALA in chain A, residue 100
        "ATOM      1  N   ALA A 100       0.000   0.000   0.000  1.00  0.00           N  \n"
        "ATOM      2  CA  ALA A 100       1.500   0.000   0.000  1.00  0.00           C  \n"
        "ATOM      3  C   ALA A 100       2.500   1.000   0.000  1.00  0.00           C  \n"
        "ATOM      4  O   ALA A 100       3.500   0.700   0.000  1.00  0.00           O  \n"
        # HETATMs near the protein
        "HETATM    5  ZN  ZN  A 200       2.000   2.000   0.000  1.00  0.00          ZN  \n"
        "HETATM    6  O3  OHX A 201       3.000   2.500   0.000  1.00  0.00           O  \n"
        "HETATM    7  P1  SUB A 202       4.000   2.000   0.000  1.00  0.00           P  \n"
        "END\n"
    )
    p = tmp_path_factory.mktemp("scine_smoke") / "tiny.pdb"
    p.write_text(pdb_text)
    return p


def test_scine_available_returns_bool(bridge):
    """scine_available() must return True (we already passed importorskip)."""
    assert bridge.scine_available() is True


def test_pdb_to_scine_cluster_auto(bridge, tiny_pdb):
    """Auto-cluster picks up the 3 HETATMs + protein within 5 Å."""
    ac, idx_map = bridge.pdb_to_scine_cluster(tiny_pdb, cluster_residues="auto")
    n = ac.size()
    # Expect all 7 atoms within 5 Å of HETATM core (entire test PDB)
    assert n >= 3, f"expected at least Zn+O+P, got {n} atoms"
    assert len(idx_map) == n, "atom_index_map size mismatch"
    # All cluster_idx → original PDB serial mappings should be valid
    for cluster_idx, orig_serial in idx_map.items():
        assert 0 <= cluster_idx < n
        assert orig_serial >= 1, f"PDB serials are 1-indexed; got {orig_serial}"


def test_pdb_to_scine_cluster_element_histogram(bridge, tiny_pdb):
    """Element histogram should reflect the synthetic PDB."""
    ac, _ = bridge.pdb_to_scine_cluster(tiny_pdb, cluster_residues="auto")
    elements = [str(ac.get_element(i)).split(".")[-1] for i in range(ac.size())]
    counts = Counter(elements)
    assert counts.get("Zn", 0) == 1, f"want exactly 1 Zn, got {counts}"
    # P from the SUB residue
    assert counts.get("P", 0) == 1, f"want 1 P, got {counts}"
    # Should contain at least the OH O + a protein O if cluster captures both
    assert counts.get("O", 0) >= 1, f"want ≥1 O, got {counts}"


def test_pdb_to_scine_cluster_index_map_round_trip(bridge, tiny_pdb):
    """Round-trip: cluster atom 0 should map to a PDB serial that exists."""
    ac, idx_map = bridge.pdb_to_scine_cluster(tiny_pdb, cluster_residues="auto")
    pdb_text = tiny_pdb.read_text()
    pdb_serials = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            pdb_serials.append(int(line[6:11]))
    serial_set = set(pdb_serials)
    for cluster_idx, orig_serial in idx_map.items():
        assert orig_serial in serial_set, \
            f"cluster idx {cluster_idx} maps to serial {orig_serial} not in PDB"


def test_pdb_to_scine_cluster_explicit_residues(bridge, tiny_pdb):
    """Explicit residue list returns just those residues + all HETATMs."""
    # Request just residue 100 (ALA); HETATMs are still always retained
    ac, idx_map = bridge.pdb_to_scine_cluster(tiny_pdb, cluster_residues=[100])
    # 4 ALA atoms + 3 HETATMs = 7
    assert ac.size() == 7


def test_make_scine_model_pm6(bridge):
    """sparrow-pm6 backend constructs a valid Model."""
    m = bridge.make_scine_model(backend="sparrow-pm6")
    assert hasattr(m, "method_family")
    assert hasattr(m, "program")
    # Method names are case-sensitive: SCINE returns "PM6"
    assert m.method_family.upper() == "PM6"


def test_make_scine_model_unknown_raises(bridge):
    """Bogus backend name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown backend"):
        bridge.make_scine_model(backend="not-a-real-backend")


def test_make_pte_reactive_filter_returns_filter_array(bridge):
    """The PTE filter stack constructor returns a usable filter array."""
    f = bridge.make_pte_reactive_filter(central_metal="Zn")
    # Should be a ReactiveSiteFilterAndArray, which is callable as a filter.
    # We can't easily test reaction filtering without DB plumbing, but we
    # can at least confirm the returned object has the right interface.
    assert f is not None
    name = type(f).__name__
    assert "ReactiveSiteFilter" in name or "AndArray" in name, \
        f"unexpected type {name}"


def test_required_versions_constant_shape(bridge):
    """_REQUIRED_SCINE_VERSIONS keys must include the core packages."""
    versions = bridge._REQUIRED_SCINE_VERSIONS
    for pkg in ("scine_chemoton", "scine_readuct", "scine_utilities", "scine_database"):
        assert pkg in versions, f"missing version pin for {pkg}"


def test_chemoton_explore_op_imports(bridge):
    """The op-level entry must import without side effects."""
    from quantum_engine.ops import chemoton_explore as ce
    # The run signature should be callable
    assert callable(ce.run)


def test_cli_chemoton_subcommand_present():
    """`qcb chemoton-explore --help` must work."""
    import subprocess
    import os
    repo = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["python", "-m", "quantum_engine.cli", "chemoton-explore", "--help"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(repo)},
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    out = result.stdout
    for flag in ("--backend", "--max-bond-modifications", "--barrier-cap-kcal",
                 "--top-n-export", "--mongodb-uri", "--dry-run"):
        assert flag in out, f"missing flag {flag} in --help; got {out[:600]}"
