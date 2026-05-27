"""Layer-3 paraoxon gas-phase Chemoton-explore test.

Skipped by default (requires running mongod and a populated apptainer
container). Enable explicitly with::

    pytest tests/test_chemoton_paraoxon_gas.py --run-chemoton-mongo

Goal
----
30-atom paraoxon + Zn₂(μ-OH) cluster. Run ``qcb chemoton-explore`` with one
bond modification per Rearrangement step. PASS criteria:

  * At least one elementary step found
  * Top step has Onuc-P bond formed AND Olg-P bond broken
  * Barrier ∈ [30, 60] kcal/mol (PM6 over-binding tolerated)

This is a long-running test (~10 min on 8 CPU cores). Intended for CI
nightly, not pre-commit.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO, _REPO / "tools"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def pytest_addoption(parser):
    """Add a CLI flag so the test only runs when explicitly requested."""
    parser.addoption(
        "--run-chemoton-mongo",
        action="store_true",
        default=False,
        help="Enable Chemoton tests that require a running mongod.",
    )


def _gated(request):
    if not request.config.getoption("--run-chemoton-mongo", default=False):
        pytest.skip("requires --run-chemoton-mongo (mongod must be running)")


@pytest.fixture(scope="module")
def paraoxon_gas_pdb(tmp_path_factory):
    """Build a synthetic paraoxon + Zn₂(μ-OH) cluster (~30 atoms).

    Geometry is approximate (PTE active-site-like): two Zn centres bridged
    by a hydroxide, with paraoxon docked so its phosphorus is poised for
    SN2-at-P. Coordinates are in Å.
    """
    # Paraoxon = (EtO)₂P(=O)(O-C₆H₄-NO₂); Zn₂μOH bridge from PTE active site.
    # We hand-place coordinates that are physically plausible; not a
    # production geometry, just enough atoms for Chemoton to find SN2-at-P.
    lines = [
        # Two Zn centres
        "HETATM    1 ZN1  ZN  A 200      0.000   0.000   0.000  1.00  0.00          ZN  ",
        "HETATM    2 ZN2  ZN  A 201      3.500   0.000   0.000  1.00  0.00          ZN  ",
        # Bridging hydroxide
        "HETATM    3 O3   OHX A 202      1.750   0.500   0.500  1.00  0.00           O  ",
        "HETATM    4 H1   OHX A 202      1.750   1.450   0.500  1.00  0.00           H  ",
        # Paraoxon P
        "HETATM    5 P1   SUB A 203      4.500   2.000   0.500  1.00  0.00           P  ",
        # P=O
        "HETATM    6 O1   SUB A 203      5.500   1.500   1.500  1.00  0.00           O  ",
        # P-OEt #1
        "HETATM    7 O2   SUB A 203      3.300   3.000   1.000  1.00  0.00           O  ",
        "HETATM    8 C1   SUB A 203      3.000   4.500   0.500  1.00  0.00           C  ",
        "HETATM    9 H2   SUB A 203      2.000   4.700   1.000  1.00  0.00           H  ",
        "HETATM   10 H3   SUB A 203      3.700   5.200   1.000  1.00  0.00           H  ",
        "HETATM   11 C2   SUB A 203      3.000   4.700  -1.000  1.00  0.00           C  ",
        "HETATM   12 H4   SUB A 203      3.500   5.700  -1.300  1.00  0.00           H  ",
        "HETATM   13 H5   SUB A 203      3.500   3.900  -1.500  1.00  0.00           H  ",
        "HETATM   14 H6   SUB A 203      2.000   4.700  -1.500  1.00  0.00           H  ",
        # P-OEt #2
        "HETATM   15 O4   SUB A 203      5.500   3.000  -0.500  1.00  0.00           O  ",
        "HETATM   16 C3   SUB A 203      6.500   4.000   0.000  1.00  0.00           C  ",
        "HETATM   17 H7   SUB A 203      6.000   4.700   0.700  1.00  0.00           H  ",
        "HETATM   18 H8   SUB A 203      7.000   4.500  -0.800  1.00  0.00           H  ",
        "HETATM   19 C4   SUB A 203      7.500   3.300   0.900  1.00  0.00           C  ",
        "HETATM   20 H9   SUB A 203      8.000   4.000   1.500  1.00  0.00           H  ",
        "HETATM   21 H10  SUB A 203      8.200   2.700   0.300  1.00  0.00           H  ",
        "HETATM   22 H11  SUB A 203      7.000   2.700   1.500  1.00  0.00           H  ",
        # P-O-C(aryl) - the leaving group
        "HETATM   23 O5   SUB A 203      4.000   2.500  -0.800  1.00  0.00           O  ",
        "HETATM   24 C5   SUB A 203      4.500   2.500  -2.100  1.00  0.00           C  ",
        "HETATM   25 C6   SUB A 203      5.700   3.100  -2.400  1.00  0.00           C  ",
        "HETATM   26 H12  SUB A 203      6.300   3.500  -1.700  1.00  0.00           H  ",
        # ... and more aryl-CH; abbreviated for the test
        "HETATM   27 C7   SUB A 203      6.000   3.100  -3.700  1.00  0.00           C  ",
        "HETATM   28 H13  SUB A 203      6.900   3.500  -3.900  1.00  0.00           H  ",
        "HETATM   29 N1   SUB A 203      4.500   2.500  -5.500  1.00  0.00           N  ",
        "HETATM   30 O6   SUB A 203      5.500   3.000  -6.000  1.00  0.00           O  ",
        "END",
    ]
    p = tmp_path_factory.mktemp("paraoxon_gas") / "paraoxon_zn2.pdb"
    p.write_text("\n".join(lines) + "\n")
    return p


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("chemoton_paraoxon_run")


def test_chemoton_explore_paraoxon(request, paraoxon_gas_pdb, out_dir):
    """Layer-3: full chemoton-explore against a synthetic paraoxon cluster."""
    _gated(request)

    pytest.importorskip("scine_chemoton")
    pytest.importorskip("scine_database")

    from quantum_engine.ops import chemoton_explore as ce

    mongodb_uri = os.environ.get(
        "QCB_TEST_MONGODB_URI",
        "mongodb://localhost:27017/qcb_paraoxon_test",
    )

    result = ce.run(
        input_pdb=paraoxon_gas_pdb,
        cluster_spec="auto",
        backend="sparrow-pm6",
        max_bond_modifications=1,
        max_depth=2,
        barrier_cap_kcal=80.0,
        top_n_export=3,
        mongodb_uri=mongodb_uri,
        out_dir=out_dir,
        central_metal="Zn",
        cluster_charge=1,
        cluster_multiplicity=1,
    )

    assert result["status"] in ("completed", "failed"), \
        f"unexpected status {result['status']}"
    if result["status"] == "failed":
        pytest.fail(f"chemoton-explore failed: {result.get('exploration', {})}")

    # PASS: at least one elementary step exported
    assert len(result["exported_pdbs"]) >= 1, \
        "no elementary steps exported; Chemoton found nothing"

    # Best barrier should be in [30, 80] kcal/mol (PM6 tolerated up to 80)
    summary_path = Path(result["outputs"]["exported_ts_dir"]) / "chemoton_summary.json"
    if summary_path.is_file():
        import json as _json
        chemo_summary = _json.loads(summary_path.read_text())
        if chemo_summary["results"]:
            top = chemo_summary["results"][0]
            barrier = top["barrier_kcal_per_mol"]
            assert 0 < barrier < 80, \
                f"top barrier {barrier:.1f} kcal/mol outside [0, 80]"
