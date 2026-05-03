"""Generic enzyme TS-design pipeline — 8 Step classes + builder.

All stages stubbed with NotImplementedError pointing at the QCB
subpackage that grows the missing primitive. The shape is fixed so
downstream code can target Step names; logic lands stage-by-stage.

Stage layout:

    1. ParseReaction       SMILES → atom-mapped reaction → bond-make/break
    2. VacuumTSSearch      vacuum TS (autodE | scine | molecularGSM | pygsm)
    3. ActiveSitePrep      protonate + charge cropped PDB
    4. TSConformerGen      RDKit ETKDGv3 + CREST conformer ensemble
    5. DockTSIntoActiveSite  constraint-based placement
    6. IterativeRefine     MACE-POLAR-1M + g-xTB; CA fixed
    7. InProteinPathRefind pyGSM SE | pysisyphus NEB | scine ReaDuct
    8. HighResTSPolish     Sella + MACE-POLAR-1M; freq check; IRC
    + WriteTSCif           multi-TS .cif output

Each stage has a ``tool`` field that picks the adapter; "auto" means
"first available in qcb-xtb at run time." See
:func:`build_enzyme_ts_design_pipeline` for the full assembly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantum_engine.pipelines import Context, Pipeline, Step, StepResult


# ─────────────────────────────────────────────────────────────────────
# Stage 1 — Reaction parsing (SMILES → atom-mapped + bond-change list)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ParseReaction:
    """Parse reactant + product SMILES → atom-mapped reaction graph,
    bond-make/break list, net charge.

    Tools (configurable): RDKit + Indigo cross-validate parsing;
    RXNMapper does atom mapping; CGRtools does bond-change diff.
    """
    name: str = "parse_reaction"
    reactant_smiles: str = ""
    product_smiles: str = ""
    mapper: str = "rxnmapper"        # "rxnmapper" | "indigo" | "cgrtools"
    workdir: str | Path | None = None

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError(
            "ParseReaction: wire to "
            "quantum_engine.io.reaction.parse_reaction_smiles "
            "(needs the [chem] optional deps: rdkit, epam.indigo, rxnmapper, "
            "CGRtools). Should set ctx.metadata['atom_map'], "
            "ctx.metadata['bonds_formed'], ctx.metadata['bonds_broken'], "
            "ctx.metadata['net_charge']."
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — Vacuum TS search
# ─────────────────────────────────────────────────────────────────────

@dataclass
class VacuumTSSearch:
    """Find a vacuum TS guess from the bond-change list. Tool-swappable."""
    name: str = "vacuum_ts"
    tool: str = "auto"               # "autode" | "scine" | "molecularGSM" | "pygsm" | "auto"
    qm_method: str = "g-xtb"         # default to g-xTB; MACE-POLAR-1M for polish later
    workdir: str | Path | None = None

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError(
            "VacuumTSSearch: dispatch on self.tool to "
            "quantum_engine.qm.{scine,pygsm,molecular_gsm}.* or "
            "autode (via [chem] optional deps). Output: ctx.atoms ← TS guess; "
            "outputs={'vacuum_ts_xyz': ..., 'vacuum_barrier_kcal': ...}."
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 3 — Active-site prep (protonate, charge, fix gaps)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ActiveSitePrep:
    """Protonate + charge a cropped active-site PDB. Net charge from
    Stage 1's bond-change accounting."""
    name: str = "active_site_prep"
    pH: float = 7.0
    methods: tuple[str, ...] = ("chimera", "propka", "pdbfixer", "rules")
    cofactor_charges: dict[str, int] | None = None  # {"ZN": 2}
    workdir: str | Path | None = None

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError(
            "ActiveSitePrep: reuse "
            "quantum_engine.prep.consensus_protonate (already implemented "
            "in active_site_ts.orchestrator.ProtonateActiveSite)."
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 4 — TS conformer generation
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TSConformerGen:
    """Generate a conformer ensemble of the vacuum TS. Combines RDKit
    ETKDGv3 (cheap distance-geometry) + CREST (xtb-driven, more
    realistic)."""
    name: str = "ts_conformers"
    n_conformers: int = 50
    use_crest: bool = True
    workdir: str | Path | None = None

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError(
            "TSConformerGen: wire to RDKit ETKDGv3 + "
            "quantum_engine.qm.crest.run_conformer_search."
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 5 — Dock TS-into-active-site
# ─────────────────────────────────────────────────────────────────────

@dataclass
class DockTSIntoActiveSite:
    """Constraint-based placement of TS conformers into the prepared
    active site. Constraints: preserve forming/breaking bonds, respect
    metal coordination, avoid clashes."""
    name: str = "dock_ts"
    constraint_distance_tol_A: float = 0.3
    metal_coordination_max_A: float = 2.5
    workdir: str | Path | None = None

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError(
            "DockTSIntoActiveSite: needs a constraint-based docker. "
            "Sketch: use AutoDock Vina + custom constraints, or RDKit "
            "+ ASE FixBondLengths + minimisation. Output: list of "
            "docked complex Atoms ranked by clash + constraint violation."
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 6 — Iterative refinement (MACE-POLAR-1M + g-xTB)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class IterativeRefine:
    """Sidechain + ligand co-optimisation with CA atoms fixed (or
    harmonic restraints in chopped-cluster mode). MACE-POLAR-1M as
    main calc; g-xTB for cheap pre-relax."""
    name: str = "iterative_refine"
    constraint_mode: str = "ca-only"    # "ca-only" | "backbone" | "ca-restrained" | "none"
    n_iterations: int = 5
    fmax: float = 0.05
    pre_relax_with_gxtb: bool = True
    mace_model: str = "mace-polar"      # alias for MACE-POLAR-1-M
    workdir: str | Path | None = None

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError(
            "IterativeRefine: chain "
            "quantum_engine.qm.xtb.run_xtb_opt (g-xTB pre-relax) → "
            "quantum_engine.calc MACE-POLAR-1M ASE optimiser. Use "
            "quantum_engine.qm.pysisyphus.harmonic_restraints when "
            "constraint_mode='ca-restrained'."
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 7 — In-protein path re-find
# ─────────────────────────────────────────────────────────────────────

@dataclass
class InProteinPathRefind:
    """Re-find the reaction path inside the enzyme. Tool-swappable:
    pyGSM single-ended (driving coords), pysisyphus NEB, ReaDuct."""
    name: str = "in_protein_path"
    tool: str = "pygsm-se"              # "pygsm-se" | "pysisyphus-neb" | "scine-bspline"
    n_images: int = 11
    workdir: str | Path | None = None

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError(
            "InProteinPathRefind: dispatch on self.tool. SE-GSM uses "
            "ctx.metadata['bonds_formed'/'bonds_broken'] from Stage 1 as "
            "driving coords. Output: ctx.atoms ← in-protein TS; "
            "outputs={'reactant_xyz', 'ts_xyz', 'product_xyz', 'barrier_kcal'}."
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 8 — High-res TS polish (Sella + MACE-POLAR-1M, freq, IRC)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class HighResTSPolish:
    """Polish the TS to tight convergence; verify with frequency
    analysis (exactly one imaginary mode); IRC connectivity check."""
    name: str = "polish_ts"
    optimizer: str = "sella"            # "sella" | "pysisyphus-rsirfo" | "scine-tsopt"
    fmax: float = 0.005
    mace_model: str = "mace-polar"
    workdir: str | Path | None = None

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError(
            "HighResTSPolish: Sella driver from quantum_engine.ops.saddle. "
            "Then quantum_engine.ops.freq for vib analysis. Then "
            "quantum_engine.qm.pysisyphus.irc for connectivity. Reject "
            "if # imaginary modes != 1 or IRC fails to reach reactant/product."
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 9 — Output (.cif per TS)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class WriteTSCif:
    """Emit a .cif per accepted TS, with proper space group P1, atom
    types, and a comment block listing the barrier + frequency check
    + level of theory."""
    name: str = "write_cif"
    workdir: str | Path | None = None

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError(
            "WriteTSCif: gemmi or biotite Cif writer; one .cif per "
            "ctx.history['polish_ts'] result entry."
        )


# ─────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────

def build_enzyme_ts_design_pipeline(
    *,
    reactant_smiles: str,
    product_smiles: str,
    constraint_mode: str = "ca-only",
    vacuum_ts_tool: str = "auto",
    path_refind_tool: str = "pygsm-se",
    polish_tool: str = "sella",
    mace_model: str = "mace-polar",
) -> Pipeline:
    """Assemble the 9-step pipeline. Caller fills in ``ctx`` with the
    cropped-active-site PDB path and runs ``Pipeline.run(ctx)``."""
    return Pipeline([
        ParseReaction(reactant_smiles=reactant_smiles,
                      product_smiles=product_smiles),
        VacuumTSSearch(tool=vacuum_ts_tool),
        ActiveSitePrep(),
        TSConformerGen(),
        DockTSIntoActiveSite(),
        IterativeRefine(constraint_mode=constraint_mode,
                        mace_model=mace_model),
        InProteinPathRefind(tool=path_refind_tool),
        HighResTSPolish(optimizer=polish_tool, mace_model=mace_model),
        WriteTSCif(),
    ])
