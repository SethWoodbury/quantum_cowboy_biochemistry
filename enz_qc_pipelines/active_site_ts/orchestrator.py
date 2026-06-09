"""Active-site TS-search orchestrator (skeleton — TODO-marked).

Implements Seth's 9-stage pipeline:

  1.  Protonate the active site of a designed PDB
  2.  Build a small reactant-only model (ligand + a couple of polar
      neighbours) and find a NEB-derived TS *guess* in vacuum
  3.  Generate conformers of that TS guess with CREST
  4.  Rigid-dock each conformer back into the protonated active site
  5.  For each docked pose, run sidechain-torsion-only optimisation
      (chi-rotamers around the rigid TS conformer)
  6.  Repeat 4–5 until energies stabilise; pick the best (lowest E)
      sidechain ⊕ TS combination
  7.  Take the winning pose and refine the TS in the enzyme pocket
      via metadynamics (or a localised NEB)
  8.  IRC descend to confirm reactant/product connectivity
  9.  Compute the minimum-energy barrier and emit the report

This module is INTENTIONALLY stubbed: every stage that depends on an
external tool (CREST, a docking backend) raises ``NotImplementedError``
with a pointer to the tool we'd need to wire in. The aim is to give a
running shape — `Pipeline([...]).run(ctx)` — that we can flesh out
stage-by-stage as the engine grows the missing primitives.

When you pull this in, prefer fixing the `NotImplementedError` stubs
over rewriting the whole orchestrator: every stub points at the
specific quantum_engine subpackage that should grow the missing
function.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantum_engine.pipelines import Context, Pipeline, Step, StepResult
from quantum_engine.pipelines.steps import IRC, MTD, NEB, Opt


# ─────────────────────────────────────────────────────────────────────
# Stage 1 — Protonate
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ProtonateActiveSite:
    """Run deterministic staged protonation on the input PDB.

    Wraps the canonical protonation engine
    :mod:`quantum_engine.prep.protonator` (CLI: ``qcb protonate``). Writes a
    protonated PDB and swaps ``ctx.atoms`` for the protonated geometry.
    """
    name: str = "protonate"
    pH: float = 7.0
    set_overrides: dict[str, str] | None = None   # e.g. {"A:226": "HIP"}
    ptm: dict[str, str] | None = None             # e.g. {"A:169": "KCX"}

    def run(self, ctx: Context) -> StepResult:
        from quantum_engine.prep import protonator
        outdir = ctx.outdir / self.name
        outdir.mkdir(exist_ok=True)
        # Need a PDB on disk; if ctx has only ASE Atoms, dump first
        in_pdb = ctx.metadata.get("input_pdb")
        if in_pdb is None:
            from ase.io import write as ase_write
            in_pdb = outdir / "input.pdb"
            ase_write(str(in_pdb), ctx.atoms)
        out_pdb = outdir / "protonated.pdb"
        argv = ["--input-pdb", str(in_pdb), "--output-pdb", str(out_pdb),
                "--pH", str(self.pH)]
        for spec, state in (self.set_overrides or {}).items():
            argv += ["--set", f"{spec}={state}"]
        for spec, code in (self.ptm or {}).items():
            argv += ["--ptm", f"{spec}={code}"]
        rc = protonator.main(argv)
        if rc != 0:
            raise RuntimeError(f"protonator failed (rc={rc}) on {in_pdb}")
        # Swap atoms to protonated structure
        from ase.io import read as ase_read
        ctx.atoms = ase_read(str(out_pdb))
        ctx.atoms.calc = ctx.calc
        return StepResult(
            name=self.name, atoms=ctx.atoms,
            outputs={"protonated_pdb": str(out_pdb)},
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — Vacuum TS-guess from reactants only
# ─────────────────────────────────────────────────────────────────────

@dataclass
class VacuumNEBGuess:
    """Build a small "ligand only" model and run a fast NEB to get a
    TS guess in vacuum, before we worry about the protein pocket.

    Composes :class:`Opt` (twice — reactant and product) and
    :class:`NEB`. Stores the ts guess on ``ctx.metadata['ts_guess']``.
    """
    name: str = "vacuum_neb_guess"
    n_images: int = 11
    fmax: float = 0.05
    max_steps: int = 200

    def run(self, ctx: Context) -> StepResult:
        # We need reactant + product geometries for the ligand. The
        # `quantum_engine.qm.endpoints` (or mlff/endpoint_generation)
        # module knows how to build them from BOND_BREAKING_DEFS.
        raise NotImplementedError(
            f"{self.name}: needs a wrapper that calls "
            "quantum_engine.mlff.endpoint_generation.generate_endpoints "
            "to build ligand-only reactant/product, then runs Opt → "
            "NEB. Slot for a later sprint."
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 3 — CREST conformers of the TS guess
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CRESTConformers:
    """Generate conformers of the vacuum TS guess via CREST.

    CREST is *not* yet vendored in deps/. When this stub fires, the
    error message should point the user at adding deps/crest/ and
    a thin :mod:`quantum_engine.qm.crest` wrapper.
    """
    name: str = "crest_conformers"
    n_conformers: int = 20
    method: str = "gfn-ff"
    input_step: str = "vacuum_neb_guess"

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError(
            f"{self.name}: CREST is not yet in deps/. To wire this up: "
            "(a) add deps/crest/ as a submodule of grimme-lab/crest, "
            "(b) build per deps/build_crest.sh, "
            "(c) write quantum_engine/qm/crest.py with a "
            "`run_crest_conformer_search(atoms, **kw) -> list[Atoms]` "
            "function, then this Step becomes a 5-line wrapper."
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 4 — Rigid-dock TS conformers into the active site
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RigidDockTSConformers:
    """For each CREST conformer, rigid-dock into the protein active
    site (taking the protonated structure from Stage 1).

    Decouples docking-engine choice (Vina, Smina, Lightdock, even a
    simple grid-search) — the Step is a thin facade. We expect a
    function ``quantum_engine.docking.rigid_dock_into_pocket`` that
    takes (host=Atoms, guests=list[Atoms], pocket_residues=list[...])
    and returns a list of (pose_atoms, score) tuples.
    """
    name: str = "rigid_dock"
    pocket_residues: tuple[tuple[str, int], ...] = ()   # ("A", 41), ...
    backend: str = "smina"
    input_step: str = "crest_conformers"
    host_step: str = "protonate"

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError(
            f"{self.name}: docking backend not yet wired. Suggested: "
            "create quantum_engine/docking/ with a backend-agnostic "
            "rigid_dock_into_pocket() that delegates to Smina/Vina via "
            "CLI for the first cut. Use pocket_residues to define a "
            "binding-site sphere."
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 5 — Sidechain torsion optimisation around each pose
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SidechainTorsionOpt:
    """For each docked TS pose, optimise *only* sidechain chi angles
    of the pocket residues — backbone and ligand stay rigid.

    Implementation note: ASE doesn't have native chi-only freedom. We
    can either (a) run a constrained ASE relax with FixAtoms on
    backbone + ligand and let the underlying optimizer move sidechains
    only, or (b) use PyRosetta SidechainDihedralOptimizer (already used
    by woodbuse's alignment script — see
    ``/home/woodbuse/special_scripts/general_utils/align_prediction_to_ref_pdb_and_copy_lig.py``).
    Wrap whichever you pick as
    ``quantum_engine.prep.sidechain.optimise_sidechains`` and call from here.
    """
    name: str = "sidechain_torsion_opt"
    fmax: float = 0.05
    max_steps: int = 200
    input_step: str = "rigid_dock"

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError(
            f"{self.name}: needs sidechain-only optimiser. Two paths: "
            "(a) wrap PyRosetta's SidechainDihedralOptimizer in "
            "quantum_engine.prep.sidechain, (b) ASE FixAtoms(backbone "
            "+ ligand) + `quantum_engine.ops.opt.run`. Path (a) is "
            "faster but adds the PyRosetta dependency."
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 6 — Pick best (lowest energy) pose ⊕ sidechain combo
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PickBestPose:
    """Score the (TS-conformer × sidechain-rotamer) cartesian product
    by energy, return the best as ``ctx.atoms``.

    Pure post-processing on outputs from the previous step; no
    expensive computation. Implementation should live at
    ``quantum_engine.analysis.pose_picker.pick_lowest_energy_pose``.
    """
    name: str = "pick_best_pose"
    metric: str = "energy_eV"
    input_step: str = "sidechain_torsion_opt"

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError(
            f"{self.name}: trivial once stages 4 + 5 produce a list of "
            "(pose, score) tuples. Add "
            "quantum_engine/analysis/pose_picker.py with "
            "pick_lowest_energy_pose(history_step) -> Atoms."
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 7 — Refine TS in pocket via MTD (already implemented!)
# ─────────────────────────────────────────────────────────────────────
# Just use quantum_engine.pipelines.steps.MTD(...)


# ─────────────────────────────────────────────────────────────────────
# Stage 8 — IRC connectivity check (already implemented)
# ─────────────────────────────────────────────────────────────────────
# Just use quantum_engine.pipelines.steps.IRC(...)


# ─────────────────────────────────────────────────────────────────────
# Stage 9 — Barrier + report
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BarrierReport:
    """Compute fwd/rev barriers and emit a summary JSON+plot.

    Reads from ``ctx.history['mtd_pocket']`` (FES) and/or
    ``ctx.history['irc']`` (reactant/product/ts energies) — whichever
    is present. Wraps :func:`quantum_engine.analysis.barriers`.
    """
    name: str = "barrier_report"
    mtd_step: str = "mtd_pocket"
    irc_step: str = "irc"

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError(
            f"{self.name}: thin wrapper around "
            "quantum_engine.analysis.barriers + "
            "quantum_engine.analysis.fes.find_basins/barrier_2d. "
            "Roughly 30 LOC."
        )


# ─────────────────────────────────────────────────────────────────────
# Convenience: assemble the full pipeline
# ─────────────────────────────────────────────────────────────────────

def build_active_site_ts_pipeline(
    *,
    pocket_residues: list[tuple[str, int]] | None = None,
    cv_indices: tuple[int, int, int] | None = None,
    n_conformers: int = 20,
    n_images: int = 11,
    mtd_time_ps: float = 100.0,
) -> Pipeline:
    """Build the full 9-stage active-site TS pipeline.

    Args:
        pocket_residues: catalytic / nearby residues defining the
            binding pocket (chain, res_id) tuples.
        cv_indices: ``(P, nuc, lg)`` atom indices for the metadynamics
            CV at stage 7.
        n_conformers: how many CREST conformers to generate at stage 3.
        n_images: NEB image count at stage 2.
        mtd_time_ps: simulation time per walker at stage 7.

    Returns:
        :class:`Pipeline` of 9 :class:`Step` instances.
    """
    if cv_indices is None:
        center_idx = forming_idx = breaking_idx = None
    else:
        # cv_indices = (center, forming, breaking)  (legacy order: P, nuc, LG)
        center_idx, forming_idx, breaking_idx = cv_indices
    return Pipeline([
        ProtonateActiveSite(),
        VacuumNEBGuess(n_images=n_images),
        CRESTConformers(n_conformers=n_conformers),
        RigidDockTSConformers(pocket_residues=tuple(pocket_residues or ())),
        SidechainTorsionOpt(),
        PickBestPose(),
        MTD(name="mtd_pocket", center_idx=center_idx, forming_idx=forming_idx,
            breaking_idx=breaking_idx, total_time_ps=mtd_time_ps),
        IRC(name="irc", input_step="mtd_pocket"),
        BarrierReport(),
    ])


__all__ = [
    "ProtonateActiveSite",
    "VacuumNEBGuess",
    "CRESTConformers",
    "RigidDockTSConformers",
    "SidechainTorsionOpt",
    "PickBestPose",
    "BarrierReport",
    "build_active_site_ts_pipeline",
]
