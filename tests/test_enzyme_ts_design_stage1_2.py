"""Dry-run integration test for Stages 1+2 of the enzyme TS-design pipeline.

Validates that:
  - ParseReaction (Stage 1) parses SMILES, runs RXNMapper, computes a
    bond-formed/broken diff, and stashes the data in ctx.metadata.
  - VacuumTSSearch (Stage 2) drives autodE with the vendored xTB binary
    and produces a TS guess on disk.

Marked @pytest.mark.slow because it shells out to xTB (and rxnmapper
loads a transformer model). Run with::

    pytest tests/test_enzyme_ts_design_stage1_2.py -m slow -s

The reaction picked is the textbook Diels-Alder cycloaddition between
butadiene and ethene — one of the cheapest non-trivial vacuum TS
searches autodE handles in <10 minutes on a CPU.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Auto-skip this whole module if any chem dep is missing (rather than
# erroring during collection).
chem_deps = pytest.importorskip  # alias for readability
_ = chem_deps("rdkit")
_ = chem_deps("rxnmapper")
_ = chem_deps("autode")


# ─────────────────────────────────────────────────────────────────────
# Stage 1 — fast unit test (RDKit + RXNMapper, no xTB)
# ─────────────────────────────────────────────────────────────────────

def test_parse_reaction_diels_alder(tmp_path):
    """Stage 1 alone — confirms bond_formed/broken diff is correct for a
    Diels-Alder cycloaddition (2 new C-C σ bonds, 0 broken σ bonds)."""
    from enz_qc_pipelines.enzyme_ts_design.orchestrator import ParseReaction
    from quantum_engine.pipelines import Context

    step = ParseReaction(
        reactant_smiles="C=CC=C.C=C",
        product_smiles="C1CC=CCC1",
    )
    ctx = Context(atoms=None, outdir=tmp_path)
    (tmp_path / step.name).mkdir()
    result = step.run(ctx)

    assert result.status == "completed"
    assert "bonds_formed" in ctx.metadata
    assert "bonds_broken" in ctx.metadata
    assert "atom_map" in ctx.metadata
    assert "net_charge" in ctx.metadata

    # Diels-Alder forms 2 new σ C-C bonds (between diene termini and ene
    # carbons). RXNMapper should give us mapped atoms for both.
    assert len(ctx.metadata["bonds_formed"]) >= 2, (
        f"expected ≥2 bonds formed in Diels-Alder, got "
        f"{ctx.metadata['bonds_formed']}"
    )
    # Net charge zero for a neutral cycloaddition
    assert ctx.metadata["net_charge"] == 0


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — slow integration test (autodE + xTB, real geometry opt)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_vacuum_ts_diels_alder(tmp_path):
    """Stage 1 + Stage 2 against a Diels-Alder cycloaddition.

    Skipped unless ``-m slow`` is passed. Requires the vendored xTB
    binary at ``deps/xtb/install/bin/xtb`` (or g-xTB at
    ``deps/g-xtb/install/xtb-6.7.1/bin/xtb``).
    """
    import time

    from enz_qc_pipelines.enzyme_ts_design.orchestrator import (
        ParseReaction,
        VacuumTSSearch,
    )
    from quantum_engine.pipelines import Context, Pipeline
    from quantum_engine.site import GXTB_BIN, XTB_BIN

    # Skip cleanly if neither xTB binary is built
    if not (os.path.isfile(XTB_BIN) or os.path.isfile(GXTB_BIN)):
        pytest.skip(
            f"no xTB binary built; tried {XTB_BIN} and {GXTB_BIN}. "
            "Run `bash deps/build_xtb.sh` first."
        )

    pipeline = Pipeline([
        ParseReaction(
            reactant_smiles="C=CC=C.C=C",
            product_smiles="C1CC=CCC1",
        ),
        VacuumTSSearch(tool="autode", qm_method="g-xtb"),
    ], write_summary=True)

    ctx = Context(atoms=None, outdir=tmp_path)

    t0 = time.perf_counter()
    pipeline.run(ctx)
    elapsed = time.perf_counter() - t0
    print(f"\n[test_vacuum_ts_diels_alder] elapsed = {elapsed:.1f}s")

    # Stage 1 assertions — metadata populated
    assert ctx.metadata.get("bonds_formed"), (
        "ParseReaction should have populated ctx.metadata['bonds_formed']"
    )
    assert "bonds_broken" in ctx.metadata
    assert ctx.metadata.get("net_charge") == 0

    # Stage 2 assertions — TS xyz exists
    vac_result = ctx.history["vacuum_ts"]
    assert vac_result.status == "completed", (
        f"VacuumTSSearch failed: {vac_result.extra}"
    )
    ts_xyz = Path(vac_result.outputs["vacuum_ts_xyz"])
    assert ts_xyz.is_file(), f"TS xyz missing at {ts_xyz}"

    # Sanity: file should have at least 6 atoms (Diels-Alder TS)
    lines = ts_xyz.read_text().strip().splitlines()
    n_atoms = int(lines[0])
    assert n_atoms >= 6, f"TS xyz has too few atoms: {n_atoms}"

    # ctx.atoms should now hold an ASE Atoms with ≥6 atoms
    assert ctx.atoms is not None, "VacuumTSSearch should advance ctx.atoms"
    assert len(ctx.atoms) == n_atoms
