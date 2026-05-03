"""Smoke tests — minimal coverage of the public surface.

These run on import + a tiny H2O system and don't need any MLFF model
or external binary. They exist to catch the kind of regression we hit
in the recent reshuffles (a missing import, a renamed module, a
duplicate registry getting out of sync, an empty pipeline call sig).

Run:  pytest tests/test_smoke.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────────────────────────────
# Test 1 — every public subpackage imports cleanly
# ─────────────────────────────────────────────────────────────────────

def test_subpackage_imports():
    """Each top-level quantum_engine subpackage must import without
    side effects or missing-import errors. Catches the kind of bug
    where renaming a module leaves a stale reference in __init__."""
    import quantum_engine
    assert quantum_engine.__version__

    # Each subpackage importable
    from quantum_engine import (        # noqa: F401
        analysis, calc, config, data, io,
        mlff, mm, ops, pipelines, prep, qm, select, slurm, units,
    )


def test_new_qm_adapters_import_clean():
    """Each lazy QM adapter (scine, pygsm, pysisyphus, molecular_gsm,
    yarp) must import without raising even when the underlying tool is
    absent — they expose <name>_available() bool checks instead."""
    from quantum_engine.qm import scine, pygsm, pysisyphus, molecular_gsm, yarp
    # Each module exposes an availability check that returns bool, not raise.
    assert isinstance(scine.scine_available(), bool)
    assert isinstance(pygsm.pygsm_available(), bool)
    assert isinstance(pysisyphus.pysisyphus_available(), bool)
    assert isinstance(molecular_gsm.molecular_gsm_available(), bool)
    assert yarp.yarp_available() is False  # stub: always False


def test_apptainer_exec_helper():
    """Apptainer exec helper builds the right --nv / --bind / sif sequence."""
    from quantum_engine.site import apptainer_exec, CONTAINERS
    assert "universal" in CONTAINERS
    assert "qcb" in CONTAINERS

    # Generic invocation
    cmd = apptainer_exec("universal", "python --version")
    assert "apptainer exec" in cmd
    assert "--bind /home:/home" in cmd
    assert "--bind /net:/net" in cmd
    assert "/net/software/containers/universal.sif" in cmd
    assert cmd.endswith(" python --version")
    assert "--nv" not in cmd  # gpu=False default

    # GPU + custom binds + env vars
    cmd2 = apptainer_exec("qcb", "python -c 'print(42)'",
                          gpu=True, binds=("/home", "/scratch"),
                          env={"QCB_TEST": "1"})
    assert "--nv" in cmd2
    assert "--bind /scratch:/scratch" in cmd2
    assert "--bind /net:/net" not in cmd2  # not in custom binds
    assert "--env QCB_TEST=1" in cmd2

    # Absolute path also accepted
    cmd3 = apptainer_exec("/some/custom.sif", "echo hi")
    assert "/some/custom.sif" in cmd3


def test_slurm_job_runner_imports_and_formats():
    """SLURM generic job runner — must build a sbatch script without
    actually submitting (no SLURM on the dev box). Verifies all directives
    land in the right format."""
    import tempfile
    from quantum_engine.slurm import (
        JobConfig, slurm_available,
    )
    from quantum_engine.slurm.job_runner import _format_sbatch

    assert isinstance(slurm_available(), bool)

    with tempfile.TemporaryDirectory() as td:
        cfg = JobConfig(
            command="python -c 'print(42)'",
            workdir=Path(td),
            job_name="qcb_smoke",
            gpu=True,
            n_gpus=2,
            cpus=8,
            mem="32G",
            time="01:30:00",
        )
        script = _format_sbatch(cfg)
        assert "#SBATCH -J qcb_smoke" in script
        assert "#SBATCH --gres=gpu:2" in script
        assert "#SBATCH -c 8" in script
        assert "#SBATCH --mem=32G" in script
        assert "#SBATCH -t 01:30:00" in script
        # Default GPU partition is 'gpu' (env override available).
        assert "#SBATCH -p gpu" in script
        assert "python -c 'print(42)'" in script


def test_mcsa_parser_offline():
    """MCSAEntry parser must build a valid object from a tiny stub
    JSON shaped like the API response — no network needed."""
    from quantum_engine.data.mcsa import _parse_entry, MCSAEntry
    raw = {
        "enzyme_name": "test enzyme",
        "all_ecs": ["1.1.1.1"],
        "reference_uniprot_id": "P12345",
        "residues": [
            {
                "function_location_abv": "general acid",
                "residue_chains": [{
                    "code": "HIS", "auth_resid": 7, "chain_name": "A",
                    "is_reference": True, "pdb_id": "1xyz",
                }],
                "residue_sequences": [{"resid": 7}],
                "roles_summary": ["proton donor"],
            },
            {
                "function_location_abv": "metal ligand",
                "residue_chains": [{
                    "code": "KCX", "auth_resid": 169, "chain_name": "A",
                    "is_reference": True, "pdb_id": "1xyz",
                }],
                "residue_sequences": [{"resid": 169}],
                "roles_summary": ["metal ligand"],
            },
        ],
        "reaction": {"compounds": [{"chebi_id": 25212}], "mechanisms": []},
    }
    entry = _parse_entry(raw, mcsa_id=999)
    assert isinstance(entry, MCSAEntry)
    assert entry.enzyme_name == "test enzyme"
    assert entry.ec == ["1.1.1.1"]
    assert entry.reference_pdb == "1xyz"
    assert len(entry.catalytic_residues) == 2
    assert entry.catalytic_residues[1].is_ptm is True   # KCX flagged
    assert entry.chebi_ids == [25212]


def test_new_pipeline_scaffolds_assemble():
    """Both new pipeline builders should construct without running.
    Stages will raise NotImplementedError at run-time; that's fine."""
    from enz_qc_pipelines.enzyme_ts_design import build_enzyme_ts_design_pipeline
    from enz_qc_pipelines.mcsa_theozyme import build_mcsa_theozyme_pipeline

    p1 = build_enzyme_ts_design_pipeline(
        reactant_smiles="CC", product_smiles="CC")
    assert len(p1.steps) == 9
    assert [s.name for s in p1.steps] == [
        "parse_reaction", "vacuum_ts", "active_site_prep",
        "ts_conformers", "dock_ts", "iterative_refine",
        "in_protein_path", "polish_ts", "write_cif",
    ]

    p2 = build_mcsa_theozyme_pipeline(mcsa_id=159)
    assert len(p2.steps) == 9
    assert [s.name for s in p2.steps] == [
        "fetch_mcsa", "resolve_smiles", "crop_active_site",
        "tier2_expansion", "per_step_vacuum_ts", "iterative_refine",
        "path_refind_from_arrows", "polish_ts", "write_theozyme",
    ]


def test_units_constants_consistent():
    """Sanity-check the EV_TO_KCAL value used in ~15 files."""
    from quantum_engine.units import EV_TO_KCAL, EV_TO_KJ, KJ_TO_KCAL
    # 1 eV = 23.06... kcal/mol, kJ/kcal = 4.184 exactly
    assert abs(EV_TO_KCAL - 23.060547) < 1e-3
    assert abs(EV_TO_KJ * KJ_TO_KCAL - EV_TO_KCAL) < 1e-6


# ─────────────────────────────────────────────────────────────────────
# Test 2 — Pipeline contract
# ─────────────────────────────────────────────────────────────────────

def test_pipeline_no_steps():
    """An empty pipeline should run cleanly and produce an empty
    history."""
    from quantum_engine.pipelines import Pipeline, Context
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        ctx = Context(atoms=None, calc=None, outdir=Path(td))
        Pipeline([], write_summary=False).run(ctx)
        assert ctx.history == {}


def test_pipeline_step_protocol():
    """A custom Step satisfying the protocol should plug in unchanged."""
    from quantum_engine.pipelines import Pipeline, Context, StepResult
    import tempfile

    class _Counter:
        name = "count"
        def __init__(self): self.n = 0
        def run(self, ctx):
            self.n += 1
            return StepResult(name=self.name, status="completed",
                              extra={"n": self.n})

    with tempfile.TemporaryDirectory() as td:
        ctx = Context(atoms=None, calc=None, outdir=Path(td))
        s = _Counter()
        Pipeline([s], write_summary=True).run(ctx)
        assert ctx.history["count"].extra["n"] == 1
        assert (Path(td) / "pipeline_summary.json").is_file()


def test_pipeline_duplicate_step_names_rejected():
    from quantum_engine.pipelines import Pipeline, StepResult

    class _S:
        def __init__(self, name): self.name = name
        def run(self, ctx): return StepResult(name=self.name)

    with pytest.raises(ValueError, match="duplicate step names"):
        Pipeline([_S("a"), _S("a")])


def test_orchestrator_skeleton_assembles():
    """The 9-stage active_site_ts orchestrator should at least
    construct (the stubs raise NotImplementedError when run, that's
    fine for the smoke test)."""
    from enz_qc_pipelines.active_site_ts import build_active_site_ts_pipeline
    p = build_active_site_ts_pipeline(cv_indices=(0, 1, 2))
    assert len(p.steps) == 9
    expected = [
        "protonate", "vacuum_neb_guess", "crest_conformers",
        "rigid_dock", "sidechain_torsion_opt", "pick_best_pose",
        "mtd_pocket", "irc", "barrier_report",
    ]
    assert [s.name for s in p.steps] == expected


# ─────────────────────────────────────────────────────────────────────
# Test 3 — calculator factory
# ─────────────────────────────────────────────────────────────────────

def test_list_models_non_empty():
    """list_models() should return the registry from
    quantum_engine.config.MACE_MODELS — single source."""
    from quantum_engine.calc import list_models
    from quantum_engine.site import MACE_MODELS
    models = list_models()
    assert models, "expected at least one model in registry"
    # Every key in the canonical config should appear.
    assert set(MACE_MODELS.keys()) <= set(models.keys())


def test_make_calc_unknown_raises():
    """A bogus model name should raise FileNotFoundError with a
    helpful 'Available registry keys' message."""
    from quantum_engine.calc import make_calc
    with pytest.raises(FileNotFoundError, match="Available registry keys"):
        make_calc(model="not_a_real_model_xyz", device="cpu")


# ─────────────────────────────────────────────────────────────────────
# Test 4 — selectors / constraints
# ─────────────────────────────────────────────────────────────────────

def test_select_module_imports():
    """`from quantum_engine.io import parse_constraints` (the
    backwards-compat path) and `from quantum_engine.select import
    parse_constraints` (the new canonical path) must both work."""
    from quantum_engine.io import parse_constraints as a
    from quantum_engine.select import parse_constraints as b
    assert a is b


def test_bond_breaking_defs_loaded():
    """The ligand bond-breaking dict (extracted from tools/run_neb_ts.py
    in Phase 2) should be loadable via the proper module path."""
    from quantum_engine.data import BOND_BREAKING_DEFS
    assert "YYE" in BOND_BREAKING_DEFS
    yye = BOND_BREAKING_DEFS["YYE"]
    assert isinstance(yye, list) and yye
    a, b, dist, direction = yye[0]
    assert isinstance(a, str) and isinstance(b, str)
    assert direction in ("attractive", "repulsive")


# ─────────────────────────────────────────────────────────────────────
# Test 5 — Context.constraint plumbed through Steps (codex regression)
# ─────────────────────────────────────────────────────────────────────

def test_context_has_constraint_field():
    """Context must declare a `constraint` field — Steps read it via
    direct attribute access, not getattr-with-default. If this regresses,
    every Pipeline silently drops user constraints (FixAtoms etc.)."""
    from quantum_engine.pipelines import Context
    ctx = Context(atoms=None, calc=None)
    assert hasattr(ctx, "constraint")
    assert ctx.constraint is None
    # Also: explicit assignment survives
    ctx2 = Context(atoms=None, calc=None, constraint="dummy_constraint")
    assert ctx2.constraint == "dummy_constraint"


def test_step_propagates_ctx_constraint():
    """Capture the constraint a Step would pass to its underlying op."""
    from quantum_engine.pipelines import Context, StepResult
    from quantum_engine.pipelines.steps import Sp
    captured = {}

    class _DummyOps:
        @staticmethod
        def run(atoms, calc, outdir, constraint=None, **kw):
            captured["constraint"] = constraint
            return {"status": "completed", "energy_eV": 0.0}

    # Patch quantum_engine.ops.sp.run so the Step's import sees our dummy.
    import quantum_engine.ops.sp as real_sp
    orig_run = real_sp.run
    real_sp.run = _DummyOps.run
    try:
        ctx = Context(atoms="dummy", calc=None,
                      constraint="MY_CONSTRAINT",
                      outdir=Path("/tmp"))
        Sp().run(ctx)
        assert captured["constraint"] == "MY_CONSTRAINT", \
            "Step did not propagate ctx.constraint to the op call"
    finally:
        real_sp.run = orig_run


# ─────────────────────────────────────────────────────────────────────
# Test 6 — no duplicate MACE_MODELS registries (codex regression)
# ─────────────────────────────────────────────────────────────────────

def test_no_duplicate_mace_registries():
    """quantum_engine.site.MACE_MODELS is the single source of truth.
    Walk every .py file under quantum_engine/, enz_qc_pipelines/, and
    tools/ (excluding the canonical site) and assert no other module
    has a top-level dict literal named MACE_MODELS or MODEL_PATHS.

    Imports of MACE_MODELS (e.g. ``from quantum_engine.site import
    MACE_MODELS``) are fine — what we ban is *redeclaration*.
    """
    import ast
    repo = Path(__file__).resolve().parent.parent
    canonical = (repo / "quantum_engine" / "site.py").resolve()

    def _is_dict_assign(node, names):
        if not isinstance(node, ast.Assign):
            return False
        if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            return False
        return node.targets[0].id in names and isinstance(node.value, ast.Dict)

    offenders: list[str] = []
    for d in (repo / "quantum_engine", repo / "enz_qc_pipelines",
              repo / "tools"):
        if not d.exists():
            continue
        for p in d.rglob("*.py"):
            if p.resolve() == canonical:
                continue
            try:
                tree = ast.parse(p.read_text())
            except SyntaxError:
                continue
            for node in tree.body:
                if _is_dict_assign(node, {"MACE_MODELS", "MODEL_PATHS"}):
                    offenders.append(f"{p.relative_to(repo)}:{node.lineno}")

    assert not offenders, (
        f"Duplicate MACE_MODELS / MODEL_PATHS dict declarations:\n  "
        + "\n  ".join(offenders)
        + "\n  Canonical source is quantum_engine/site.py — every "
          "other module should `from quantum_engine.site import "
          "MACE_MODELS` or use quantum_engine.calc.make_calc()."
    )


# ─────────────────────────────────────────────────────────────────────
# Test 7 — CLI entry point
# ─────────────────────────────────────────────────────────────────────

def test_cli_help_runs():
    """`qcb --help` should print the top-level usage without error."""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "quantum_engine.cli", "--help"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH":
             str(Path(__file__).resolve().parent.parent)},
    )
    # argparse exits 0 on --help
    assert result.returncode == 0, result.stderr
    assert "qcb" in result.stdout
    # Spot-check a few subcommands appear
    for sub in ("sp", "opt", "md", "neb", "mtd", "ts", "protonate"):
        assert sub in result.stdout, f"missing subcommand {sub!r}"


# ─────────────────────────────────────────────────────────────────────
# Test 8 — optional CGRtools install
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.optional
def test_cgrtools_optional_install():
    """CGRtools is the canonical reaction-graph (CGR) bond-diff backend.

    We pin to ``CGRtools==4.1.27`` in pyproject.toml — that's the last
    version that builds cleanly on Python 3.11 (later 4.1.33–4.1.35
    fail with a Cython compile error on ``containers/_unpack.pyx``).
    The pipeline keeps RDKit's atom-map-based diff as a fallback that
    produces equivalent formed/broken bond lists; this test verifies
    CGRtools-when-installed produces matching results on the textbook
    Diels-Alder reaction (2 σ-bonds formed, 0 broken).

    Marked ``optional`` — skipped if CGRtools isn't importable rather
    than failing the smoke suite.
    """
    pytest.importorskip("CGRtools")
    from CGRtools import ReactionContainer, smiles

    # Atom-mapped Diels-Alder: butadiene + ethene → cyclohexene.
    # CGRtools needs explicit atom maps to compose reactant ↔ product.
    r = smiles("[CH2:1]=[CH:2][CH:3]=[CH2:4].[CH2:5]=[CH2:6]")
    p = smiles("[CH2:1]1[CH:2]=[CH:3][CH2:4][CH2:5][CH2:6]1")
    rxn = ReactionContainer(reactants=[r], products=[p])
    cgr_graph = rxn.compose()

    bonds_formed = bonds_broken = 0
    for _n, _m, bond in cgr_graph.bonds():
        r_order = bond.order
        p_order = bond.p_order
        if r_order is None and p_order is not None:
            bonds_formed += 1
        elif r_order is not None and p_order is None:
            bonds_broken += 1

    assert bonds_formed == 2, (
        f"Diels-Alder should form 2 σ-bonds (CGRtools), got {bonds_formed}"
    )
    assert bonds_broken == 0, (
        f"Diels-Alder breaks no σ-bonds (CGRtools), got {bonds_broken}"
    )
