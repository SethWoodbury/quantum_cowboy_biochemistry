"""M-CSA-driven theozyme pipeline — 9 Step classes + builder.

Maximally exploits M-CSA annotation. Most stages are stubbed with the
exact wiring path documented; logic fills in once we close the M-CSA
159 (PTE) dry run.

Stage layout (different from the generic pipeline — leans on M-CSA):

    0.  FetchMCSAEntry          API pull + cache to /net/databases/mcsa_cache
    1.  ResolveSubstrateSMILES  user SMILES + ChEBI lookup → reactant + product
    2.  CropActiveSiteFromPDB   pull ref PDB from /net/databases/rcsb/, crop
    3.  Tier2ResidueExpansion   distance- / motif-based residue addition
    4.  PerStepVacuumTS         per mechanism step: vacuum TS via SCINE
    5.  IterativeRefineWithPTMs MACE-POLAR-1M; PTM-aware (KCX, SEP, …)
    6.  InProteinPathRefindFromArrows  SE-GSM driven by Marvin arrows
    7.  HighResTSPolish         Sella + MACE-POLAR-1M; freq + IRC checks
    8.  WriteTheozyme           AME-format JSON + .cif

Mechanism multiplicity: M-CSA entries can have multiple alternative
mechanisms (different proposed pathways). We run the pipeline once per
mechanism, then once per step within each mechanism. Tier-1 (catalytic
only) is run first; if it converges, Tier-2 (extended residues) is
run as refinement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantum_engine.pipelines import Context, Pipeline, Step, StepResult


# ─────────────────────────────────────────────────────────────────────
# Stage 0 — Fetch M-CSA entry (cached)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class FetchMCSAEntry:
    """Pull an M-CSA entry from the API; cache to
    /net/databases/mcsa_cache/. Sets ctx.metadata['mcsa_entry']."""
    name: str = "fetch_mcsa"
    mcsa_id: int = 0
    refresh: bool = False

    def run(self, ctx: Context) -> StepResult:
        from quantum_engine.data.mcsa import fetch_entry
        entry = fetch_entry(self.mcsa_id, refresh=self.refresh)
        ctx.metadata["mcsa_entry"] = entry
        ctx.metadata["mcsa_id"] = self.mcsa_id
        return StepResult(
            name=self.name,
            atoms=ctx.atoms,
            outputs={
                "enzyme_name": entry.enzyme_name,
                "ec": entry.ec,
                "reference_pdb": entry.reference_pdb,
                "n_catalytic_residues": len(entry.catalytic_residues),
                "n_mechanisms": len(entry.mechanisms),
                "ptm_residues": [r.code for r in entry.ptm_residues],
                "cofactors": entry.cofactors,
            },
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 1 — Resolve substrate SMILES (user + ChEBI)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ResolveSubstrateSMILES:
    """Resolve concrete reactant + product SMILES. Strategy:
      1. If ``user_substrate`` is provided, use it directly (R-groups
         bound by user choice).
      2. Else pull from M-CSA reaction.compounds[].chebi_id via ChEBI."""
    name: str = "resolve_smiles"
    user_substrate: str | None = None
    user_product: str | None = None

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError(
            "ResolveSubstrateSMILES: combine user inputs with "
            "quantum_engine.data.chebi.lookup_smiles for any unresolved "
            "compound. Should set ctx.metadata['reactant_smiles'] and "
            "ctx.metadata['product_smiles']."
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — Crop active site from reference PDB
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CropActiveSiteFromPDB:
    """Pull reference PDB from /net/databases/rcsb/pdb/ (or RCSB HTTP
    fallback) and crop around the catalytic residues. Tier-1 default
    is catalytic only; Tier-2 is added in the next stage."""
    name: str = "crop_active_site"
    add_waters_within_A: float = 4.0
    workdir: str | Path | None = None

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError(
            "CropActiveSiteFromPDB: use quantum_engine.io.pdb to read "
            "from PDB_MIRROR if available (fallback: HTTP). Slice by "
            "ctx.metadata['mcsa_entry'].catalytic_residues + cofactors + "
            "shell waters."
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 3 — Tier-2 residue expansion (distance + motif)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Tier2ResidueExpansion:
    """Add residues beyond the M-CSA catalytic set. Modes:
      * ``distance`` — all residues with any heavy atom within R Å of
        any catalytic residue.
      * ``motif`` — fill in motif segments (HExxH-style) around
        adjacent catalytic residues.
      * ``both`` — union of the above (default).
    """
    name: str = "tier2_expansion"
    mode: str = "both"               # "distance" | "motif" | "both" | "skip"
    radius_A: float = 6.0
    workdir: str | Path | None = None

    def run(self, ctx: Context) -> StepResult:
        if self.mode == "skip":
            return StepResult(name=self.name, atoms=ctx.atoms,
                              outputs={"added_residues": []})
        raise NotImplementedError(
            "Tier2ResidueExpansion: "
            "(distance) iterate ctx.atoms within radius of catalytic; "
            "(motif) detect HExxH / xx-H/E/D windows and fill gaps."
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 4 — Per mechanism step, vacuum TS
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PerStepVacuumTS:
    """Iterate over mechanism steps; build a vacuum TS per step.
    Stores results under ctx.metadata['mechanism_results'] indexed by
    (mechanism_idx, step_idx)."""
    name: str = "per_step_vacuum_ts"
    tool: str = "scine"              # "scine" | "autode" | "molecularGSM"
    qm_method: str = "g-xtb"
    workdir: str | Path | None = None

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError(
            "PerStepVacuumTS: for each Mechanism in entry.mechanisms, "
            "for each MechanismStep, parse marvin_xml for arrow-pushing, "
            "translate to driving coords, run vacuum TS via tool."
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 5 — Iterative refinement with PTM topology
# ─────────────────────────────────────────────────────────────────────

@dataclass
class IterativeRefineWithPTMs:
    """Sidechain + ligand co-optimisation, CA fixed. PTM-aware: KCX,
    SEP, TPO, PTR, MSE, CSO use bespoke topology + protonation rules.
    Per-mechanism-step refinement loop."""
    name: str = "iterative_refine"
    constraint_mode: str = "ca-only"
    mace_model: str = "mace-polar"
    n_iterations: int = 5

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError(
            "IterativeRefineWithPTMs: extend "
            "enz_qc_pipelines.enzyme_ts_design.IterativeRefine with a "
            "PTM-residue lookup table (init from "
            "quantum_engine.data.mcsa.PTM_CODES). Each PTM ships its "
            "own protonation defaults and OpenMM topology fragment."
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 6 — In-protein path re-find from M-CSA arrow-pushing
# ─────────────────────────────────────────────────────────────────────

@dataclass
class InProteinPathRefindFromArrows:
    """Single-ended GSM driven by the M-CSA mechanism's Marvin XML
    arrows. The arrow-pushing IS the driving-coord set."""
    name: str = "path_refind_from_arrows"
    tool: str = "pygsm-se"           # "pygsm-se" | "molecularGSM-ssm" | "scine-nt"
    n_max_nodes: int = 15

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError(
            "InProteinPathRefindFromArrows: "
            "(1) quantum_engine.data.mcsa.parse_marvin_xml to extract "
            "arrows; (2) quantum_engine.qm.pygsm.driving_coords_from_marvin "
            "to translate; (3) dispatch on tool."
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 7 — High-res TS polish
# ─────────────────────────────────────────────────────────────────────

@dataclass
class HighResTSPolish:
    """Sella + MACE-POLAR-1M polish; vibrational freq check; IRC."""
    name: str = "polish_ts"
    optimizer: str = "sella"
    fmax: float = 0.005
    mace_model: str = "mace-polar"

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError(
            "HighResTSPolish: same wiring as "
            "enz_qc_pipelines.enzyme_ts_design.HighResTSPolish."
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 8 — Write theozyme (AME-format JSON + .cif)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class WriteTheozyme:
    """Emit AME-format theozyme JSON + per-step .cif, for the AME
    benchmark feeder. Includes: minimal residue set, substrate at TS
    geometry, plausibility flags (frequency check passed, barrier in
    physical range, metal coordination preserved, qualitative match
    to M-CSA mechanism text)."""
    name: str = "write_theozyme"
    ame_format: bool = True

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError(
            "WriteTheozyme: emit per-step .cif via gemmi/biotite. AME "
            "format spec at github.com/RosettaCommons/RFdiffusion2 — "
            "needs a constraint file alongside the .cif. Plausibility "
            "judgments computed from ctx.history results."
        )


# ─────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────

def build_mcsa_theozyme_pipeline(
    *,
    mcsa_id: int,
    user_substrate: str | None = None,
    user_product: str | None = None,
    tier2_mode: str = "both",
    tier2_radius_A: float = 6.0,
    vacuum_ts_tool: str = "scine",
    path_refind_tool: str = "pygsm-se",
    polish_tool: str = "sella",
    mace_model: str = "mace-polar",
    constraint_mode: str = "ca-only",
) -> Pipeline:
    """Assemble the 9-stage M-CSA theozyme pipeline. Caller fills in
    ``ctx`` (often an empty Context — Stage 0 fetches everything)."""
    return Pipeline([
        FetchMCSAEntry(mcsa_id=mcsa_id),
        ResolveSubstrateSMILES(user_substrate=user_substrate,
                               user_product=user_product),
        CropActiveSiteFromPDB(),
        Tier2ResidueExpansion(mode=tier2_mode, radius_A=tier2_radius_A),
        PerStepVacuumTS(tool=vacuum_ts_tool),
        IterativeRefineWithPTMs(constraint_mode=constraint_mode,
                                mace_model=mace_model),
        InProteinPathRefindFromArrows(tool=path_refind_tool),
        HighResTSPolish(optimizer=polish_tool, mace_model=mace_model),
        WriteTheozyme(),
    ])
