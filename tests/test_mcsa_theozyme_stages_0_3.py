"""Integration test for mcsa_theozyme Stages 0–3.

Pulls M-CSA 159 (phosphotriesterase) from the API, resolves substrate
SMILES via ChEBI, downloads (or hits the PDB mirror for) 1hzy, crops
the active site, and verifies tier-2 expansion expands sensibly.

Marked ``@pytest.mark.network`` because it hits two remote APIs
(M-CSA + ChEBI) on first run; subsequent runs are offline-only thanks
to per-id JSON caching.

Run:
    pytest tests/test_mcsa_theozyme_stages_0_3.py -v -m network
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


pytestmark = pytest.mark.network


def test_stage0_fetch_mcsa_159():
    """Stage 0: pull M-CSA 159 (PTE), check basic shape.

    Heads-up: M-CSA's enzyme_name is the EC name 'aryldialkylphosphatase',
    not the colloquial 'phosphotriesterase' / 'PTE'. Match either."""
    from quantum_engine.data.mcsa import fetch_entry
    entry = fetch_entry(159)
    assert entry.mcsa_id == 159
    name = entry.enzyme_name.lower()
    assert any(k in name for k in (
        "phosphotri", "phosphate", "phosphatase", "amidohydrolase",
    )), f"unexpected enzyme name: {entry.enzyme_name!r}"
    assert entry.reference_pdb in {"1hzy", "1HZY", "1hxy", "1psc"}, \
        f"unexpected ref PDB: {entry.reference_pdb}"
    # PTE has Zn/Zn — at least 4 residues should be metal ligands
    assert len(entry.catalytic_residues) >= 4
    # PTE has carbamylated Lys169 — should be flagged as PTM. M-CSA
    # stores the canonical residue code (LYS) with a 'ptm' role flag;
    # the PDB itself uses KCX. We accept either.
    ptm_residues = entry.ptm_residues
    assert ptm_residues, \
        f"PTE entry 159 should flag at least one PTM residue; got none"
    ptm_codes = {r.code for r in ptm_residues}
    ptm_seqs = {r.auth_seq for r in ptm_residues}
    assert ("KCX" in ptm_codes) or ("LYS" in ptm_codes and 169 in ptm_seqs), \
        f"PTE entry 159 should flag the carbamylated Lys169 as PTM; got " \
        f"{[(r.code, r.auth_seq) for r in ptm_residues]}"


def test_stages_0_through_3_pte():
    """End-to-end: Stages 0-3 on PTE. Tier-1 crop should produce a
    small active-site cluster; tier-2 (mode='both') should expand it."""
    import tempfile
    from quantum_engine.pipelines import Context, Pipeline
    from enz_qc_pipelines.mcsa_theozyme.orchestrator import (
        FetchMCSAEntry, ResolveSubstrateSMILES,
        CropActiveSiteFromPDB, Tier2ResidueExpansion,
    )

    with tempfile.TemporaryDirectory() as td:
        ctx = Context(atoms=None, calc=None, outdir=Path(td),
                      metadata={"mcsa_id": 159})
        pipeline = Pipeline([
            FetchMCSAEntry(mcsa_id=159),
            ResolveSubstrateSMILES(),
            CropActiveSiteFromPDB(),
            Tier2ResidueExpansion(mode="both", radius_A=6.0),
        ], write_summary=False)
        pipeline.run(ctx)

        # Stage 0
        s0 = ctx.history["fetch_mcsa"]
        assert s0.outputs["n_catalytic_residues"] >= 4
        # PTE 159 stores the PTM as canonical 'LYS' with role='ptm'
        # (the PDB itself uses 'KCX'). Either signals the PTM correctly.
        assert s0.outputs["ptm_residues"], "no PTM residues flagged"
        assert any(c in {"KCX", "LYS"} for c in s0.outputs["ptm_residues"]), \
            f"PTE 159 PTM should be KCX or LYS-with-PTM-flag; got " \
            f"{s0.outputs['ptm_residues']}"

        # Stage 1 — SMILES resolved (or skipped if all compounds had
        # null ChEBI; in that case the test should skip with a clear hint)
        s1 = ctx.history["resolve_smiles"]
        if s1.outputs["unresolved_chebi_ids"]:
            pytest.skip(
                f"ChEBI returned no SMILES for {s1.outputs['unresolved_chebi_ids']} "
                "— pass --substrate / --product on the CLI in real runs."
            )
        assert ctx.metadata["reactant_smiles"], "Stage 1 produced no reactant"
        assert ctx.metadata["product_smiles"], "Stage 1 produced no product"

        # Stage 2 — cropped PDB exists, has atoms, includes Zn
        s2 = ctx.history["crop_active_site"]
        assert Path(s2.outputs["cropped_pdb"]).is_file()
        assert s2.outputs["n_atoms"] > 10
        assert "ZN" in s2.outputs["cofactors"], \
            f"PTE 1hzy should yield Zn cofactors; got {s2.outputs['cofactors']}"

        # Stage 3 — tier-2 should add residues
        s3 = ctx.history["tier2_expansion"]
        assert s3.outputs["n_residues_total"] > s2.outputs["n_residues"], \
            "Tier-2 expansion didn't add any residues"


def test_stages_0_through_3_pte_user_substrate_override():
    """If user passes a concrete substrate SMILES, Stage 1 honours it
    even when ChEBI lookup would have succeeded."""
    import tempfile
    from quantum_engine.pipelines import Context, Pipeline
    from enz_qc_pipelines.mcsa_theozyme.orchestrator import (
        FetchMCSAEntry, ResolveSubstrateSMILES,
    )

    paraoxon = "CCOP(=O)(OCC)Oc1ccc([N+](=O)[O-])cc1.O"
    diethyl_phosphate_plus_pnp = (
        "CCOP(=O)(OCC)O.Oc1ccc([N+](=O)[O-])cc1"
    )
    with tempfile.TemporaryDirectory() as td:
        ctx = Context(atoms=None, calc=None, outdir=Path(td))
        Pipeline([
            FetchMCSAEntry(mcsa_id=159),
            ResolveSubstrateSMILES(
                user_substrate=paraoxon,
                user_product=diethyl_phosphate_plus_pnp,
            ),
        ], write_summary=False).run(ctx)
        assert ctx.metadata["reactant_smiles"] == paraoxon
        assert ctx.metadata["product_smiles"] == diethyl_phosphate_plus_pnp
        s1 = ctx.history["resolve_smiles"]
        assert s1.outputs["n_user_overrides"] == 2


def test_stages_0_through_3_pte_with_paraoxon():
    """Realistic happy path: user provides concrete paraoxon SMILES,
    Stages 0-3 run end-to-end including PDB fetch + crop + tier-2."""
    import tempfile
    from quantum_engine.pipelines import Context, Pipeline
    from enz_qc_pipelines.mcsa_theozyme.orchestrator import (
        FetchMCSAEntry, ResolveSubstrateSMILES,
        CropActiveSiteFromPDB, Tier2ResidueExpansion,
    )

    paraoxon = "CCOP(=O)(OCC)Oc1ccc([N+](=O)[O-])cc1.O"
    diethyl_phosphate_plus_pnp = "CCOP(=O)(OCC)O.Oc1ccc([N+](=O)[O-])cc1"

    with tempfile.TemporaryDirectory() as td:
        ctx = Context(atoms=None, calc=None, outdir=Path(td))
        Pipeline([
            FetchMCSAEntry(mcsa_id=159),
            ResolveSubstrateSMILES(
                user_substrate=paraoxon,
                user_product=diethyl_phosphate_plus_pnp,
            ),
            CropActiveSiteFromPDB(),
            Tier2ResidueExpansion(mode="both", radius_A=6.0),
        ], write_summary=False).run(ctx)

        # Stage 1: SMILES resolved via override
        assert ctx.metadata["reactant_smiles"] == paraoxon

        # Stage 2: cropped PDB has Zn cofactors and reasonable atom count
        s2 = ctx.history["crop_active_site"]
        assert Path(s2.outputs["cropped_pdb"]).is_file(), \
            f"missing cropped PDB at {s2.outputs['cropped_pdb']}"
        assert s2.outputs["n_atoms"] > 20, \
            f"unexpectedly few atoms: {s2.outputs['n_atoms']}"
        assert "ZN" in s2.outputs["cofactors"], \
            f"PTE 1hzy should yield Zn cofactors; got {s2.outputs['cofactors']}"

        # Stage 3: tier-2 expansion adds residues
        s3 = ctx.history["tier2_expansion"]
        assert s3.outputs["n_residues_total"] > s2.outputs["n_residues"], \
            "Tier-2 expansion didn't add any residues"
        # Distance-based should find at least a few neighbours within 6 Å
        assert s3.outputs["n_added_distance"] >= 2, \
            f"Tier-2 distance mode found only " \
            f"{s3.outputs['n_added_distance']} neighbours within 6 Å"
