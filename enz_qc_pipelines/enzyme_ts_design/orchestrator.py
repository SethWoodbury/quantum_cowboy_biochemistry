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

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantum_engine.pipelines import Context, Pipeline, Step, StepResult

log = logging.getLogger("enz_qc_pipelines.enzyme_ts_design")


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
        try:
            from rdkit import Chem
        except ImportError as e:
            raise ImportError(
                "ParseReaction needs rdkit; install via `pip install rdkit` "
                "into the qcb-xtb env (also available via the [chem] extra)."
            ) from e

        if not self.reactant_smiles or not self.product_smiles:
            raise ValueError(
                "ParseReaction: both reactant_smiles and product_smiles must "
                "be non-empty SMILES strings."
            )

        # Sanity-check both endpoints parse with RDKit before we do anything
        # expensive (rxnmapper loads a transformer model).
        r_mol_check = Chem.MolFromSmiles(self.reactant_smiles)
        p_mol_check = Chem.MolFromSmiles(self.product_smiles)
        if r_mol_check is None:
            raise ValueError(f"RDKit could not parse reactant SMILES: {self.reactant_smiles!r}")
        if p_mol_check is None:
            raise ValueError(f"RDKit could not parse product SMILES: {self.product_smiles!r}")

        # ── 1. Atom-mapping via RXNMapper ────────────────────────────
        rxn_smiles = f"{self.reactant_smiles}>>{self.product_smiles}"
        atom_map: dict[int, int] = {}
        mapped_reactant = self.reactant_smiles
        mapped_product = self.product_smiles
        confidence: float | None = None

        if self.mapper in ("rxnmapper", "auto"):
            try:
                # Suppress the noisy pkg_resources DeprecationWarning the
                # rxnmapper package emits at import time.
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    from rxnmapper import RXNMapper
                rxn_mapper = RXNMapper()
                results = rxn_mapper.get_attention_guided_atom_maps([rxn_smiles])
                mapped_rxn = results[0]["mapped_rxn"]
                confidence = float(results[0].get("confidence", 0.0))
                mapped_reactant, mapped_product = mapped_rxn.split(">>")
                log.info(
                    f"  RXNMapper: confidence={confidence:.3f} "
                    f"({len(self.reactant_smiles)} → {len(self.product_smiles)} chars)"
                )
            except Exception as e:
                if self.mapper == "rxnmapper":
                    raise RuntimeError(
                        f"RXNMapper failed on {rxn_smiles!r}: {type(e).__name__}: {e}"
                    ) from e
                log.warning(f"  RXNMapper failed ({e}); falling back to RDKit-only.")

        # ── 2. Re-parse mapped sides with RDKit and collect bonds ────
        r_mol = Chem.MolFromSmiles(mapped_reactant)
        p_mol = Chem.MolFromSmiles(mapped_product)
        if r_mol is None or p_mol is None:
            raise ValueError(
                "RDKit could not re-parse mapped reactant/product after "
                f"atom-mapping (mapper={self.mapper!r})."
            )

        def _bonds_by_mapnum(mol) -> dict[tuple[int, int], float]:
            """Return {(min_mapnum, max_mapnum): bond_order_double} for every
            bond between two mapped atoms. Atoms with map num 0 are dropped
            so the diff focuses on the reacting fragment."""
            bonds: dict[tuple[int, int], float] = {}
            for b in mol.GetBonds():
                a1 = b.GetBeginAtom().GetAtomMapNum()
                a2 = b.GetEndAtom().GetAtomMapNum()
                if a1 == 0 or a2 == 0:
                    continue
                key = (min(a1, a2), max(a1, a2))
                bonds[key] = b.GetBondTypeAsDouble()
            return bonds

        r_bonds = _bonds_by_mapnum(r_mol)
        p_bonds = _bonds_by_mapnum(p_mol)

        # Bond-change diff (RDKit-based — equivalent to CGRtools' CGR delta
        # for our purposes since we only care about make/break, not order).
        all_keys = set(r_bonds) | set(p_bonds)
        bonds_formed: list[tuple[int, int]] = []
        bonds_broken: list[tuple[int, int]] = []
        bonds_order_changed: list[tuple[int, int, float, float]] = []
        for k in sorted(all_keys):
            in_r = k in r_bonds
            in_p = k in p_bonds
            if in_p and not in_r:
                bonds_formed.append(k)
            elif in_r and not in_p:
                bonds_broken.append(k)
            elif in_r and in_p and r_bonds[k] != p_bonds[k]:
                bonds_order_changed.append((k[0], k[1], r_bonds[k], p_bonds[k]))

        # ── 3. Net charge from explicit charges on reactant side ─────
        net_charge = sum(a.GetFormalCharge() for a in r_mol.GetAtoms())
        net_charge_p = sum(a.GetFormalCharge() for a in p_mol.GetAtoms())
        if net_charge != net_charge_p:
            log.warning(
                f"  net charge mismatch reactant={net_charge}, product={net_charge_p}; "
                f"using reactant value. Check input SMILES."
            )

        # ── 4. Atom-map dict {map_num → reactant_atom_idx} ──────────
        for atom in r_mol.GetAtoms():
            mn = atom.GetAtomMapNum()
            if mn:
                atom_map[mn] = atom.GetIdx()

        # ── 5. Stash everything in ctx.metadata for downstream stages ─
        ctx.metadata["atom_map"] = atom_map
        ctx.metadata["bonds_formed"] = bonds_formed
        ctx.metadata["bonds_broken"] = bonds_broken
        ctx.metadata["bonds_order_changed"] = bonds_order_changed
        ctx.metadata["net_charge"] = net_charge
        ctx.metadata["mapped_reactant_smiles"] = mapped_reactant
        ctx.metadata["mapped_product_smiles"] = mapped_product
        ctx.metadata["reactant_smiles"] = self.reactant_smiles
        ctx.metadata["product_smiles"] = self.product_smiles

        log.info(
            f"  ParseReaction: |formed|={len(bonds_formed)}, "
            f"|broken|={len(bonds_broken)}, "
            f"|order-changed|={len(bonds_order_changed)}, "
            f"net_charge={net_charge}"
        )

        # Optionally write a small JSON summary into the step dir for
        # debugging / re-runs.
        outputs: dict[str, str] = {}
        step_dir = ctx.outdir / self.name
        if step_dir.is_dir():
            import json
            summary_path = step_dir / "reaction.json"
            summary_path.write_text(json.dumps({
                "reactant_smiles": self.reactant_smiles,
                "product_smiles": self.product_smiles,
                "mapped_reactant_smiles": mapped_reactant,
                "mapped_product_smiles": mapped_product,
                "atom_map": atom_map,
                "bonds_formed": bonds_formed,
                "bonds_broken": bonds_broken,
                "bonds_order_changed": bonds_order_changed,
                "net_charge": net_charge,
                "mapper": self.mapper,
                "mapper_confidence": confidence,
            }, indent=2, default=str))
            outputs["reaction_json"] = str(summary_path)

        return StepResult(
            name=self.name,
            status="completed",
            outputs=outputs,
            extra={
                "n_bonds_formed": len(bonds_formed),
                "n_bonds_broken": len(bonds_broken),
                "n_bonds_order_changed": len(bonds_order_changed),
                "net_charge": net_charge,
                "mapper_confidence": confidence,
            },
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
        # Resolve which tool to use. Currently only autodE is implemented;
        # "auto" routes to autodE since it's the cleanest SMILES → TS path.
        tool = self.tool
        if tool == "auto":
            tool = "autode"

        if tool != "autode":
            raise NotImplementedError(
                f"VacuumTSSearch: tool={tool!r} not yet wired. Currently only "
                "tool='autode' (or 'auto') is implemented; scine/pygsm/"
                "molecularGSM live as stubs in quantum_engine.qm.*."
            )

        # ── 1. Locate inputs from Stage 1 metadata ────────────────────
        reactant_smiles = ctx.metadata.get("reactant_smiles")
        product_smiles = ctx.metadata.get("product_smiles")
        if not reactant_smiles or not product_smiles:
            raise RuntimeError(
                "VacuumTSSearch: no reactant/product SMILES in ctx.metadata; "
                "did ParseReaction (Stage 1) run first?"
            )
        net_charge = int(ctx.metadata.get("net_charge", 0))

        # ── 2. Delegate to the autodE adapter ─────────────────────────
        # All Config plumbing, monkey-patching, fragment splitting, and
        # disk-fallback logic now lives in quantum_engine.qm.autode.
        from quantum_engine.qm.autode import vacuum_ts_from_smiles

        workdir = (
            Path(self.workdir).resolve()
            if self.workdir is not None
            else (ctx.outdir / self.name).resolve()
        )

        log.info(
            f"  VacuumTSSearch(tool=autode, qm={self.qm_method}, charge={net_charge}) "
            f"→ {workdir}"
        )

        result = vacuum_ts_from_smiles(
            reactant_smiles=reactant_smiles,
            product_smiles=product_smiles,
            qm_method=self.qm_method,
            net_charge=net_charge,
            workdir=workdir,
        )

        ts_ase = result["ts_atoms"]
        ts_xyz_path = result["ts_xyz_path"]
        barrier_kcal = result["barrier_kcal"]
        energy_eV = result["energy_eV"]
        imag_freqs = result["imag_freqs"]
        ts_source = result["ts_source"]
        xtb_path = result["xtb_path"]

        # Advance ctx.atoms to the vacuum TS guess
        ctx.atoms = ts_ase

        log.info(
            f"  VacuumTSSearch: TS xyz → {ts_xyz_path}, "
            f"barrier={barrier_kcal!r} kcal/mol, "
            f"|imag freqs|={len(imag_freqs)}"
        )

        outputs = {"vacuum_ts_xyz": str(ts_xyz_path)}
        if barrier_kcal is not None:
            # StepResult.outputs is dict[str, str] per the contract; stash
            # the float in extra. Also keep it accessible via the
            # documented key as a stringified value.
            outputs["vacuum_barrier_kcal"] = f"{barrier_kcal:.4f}"

        return StepResult(
            name=self.name,
            status="completed",
            atoms=ts_ase,
            energy_eV=energy_eV,
            outputs=outputs,
            extra={
                "vacuum_barrier_kcal": barrier_kcal,
                "imag_frequencies_cm-1": imag_freqs,
                "tool": tool,
                "qm_method": self.qm_method,
                "xtb_path": xtb_path,
                "ts_source": ts_source,
            },
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
