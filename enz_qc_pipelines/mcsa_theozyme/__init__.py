"""M-CSA-driven theozyme generation pipeline.

Inputs: M-CSA entry ID + concrete substrate SMILES (R-groups bound).
Output: per-mechanism-step theozyme — minimal residue set + substrate
at TS geometry — formatted for the AME benchmark
(github.com/RosettaCommons/RFdiffusion2, 41 active sites from M-CSA).

Why it's separate from :mod:`enz_qc_pipelines.enzyme_ts_design`: M-CSA
gives way more annotation than a generic input — catalytic residue
roles, cofactors, PTMs, mechanism arrow-pushing (Marvin XML),
reference PDB. This pipeline exploits ALL of it, with PTM topology
handling, role-aware protonation, per-step iteration, and
arrow-pushing → driving-coordinate translation for in-protein
single-ended GSM.

See :mod:`docs/plans/mcsa_theozyme.md` for the design doc.
"""
from enz_qc_pipelines.mcsa_theozyme.orchestrator import (
    CropActiveSiteFromPDB,
    FetchMCSAEntry,
    HighResTSPolish,
    InProteinPathRefindFromArrows,
    IterativeRefineWithPTMs,
    PerStepVacuumTS,
    ResolveSubstrateSMILES,
    Tier2ResidueExpansion,
    WriteTheozyme,
    build_mcsa_theozyme_pipeline,
)

__all__ = [
    "FetchMCSAEntry",
    "ResolveSubstrateSMILES",
    "CropActiveSiteFromPDB",
    "Tier2ResidueExpansion",
    "PerStepVacuumTS",
    "IterativeRefineWithPTMs",
    "InProteinPathRefindFromArrows",
    "HighResTSPolish",
    "WriteTheozyme",
    "build_mcsa_theozyme_pipeline",
]
