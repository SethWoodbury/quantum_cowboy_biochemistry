"""ts_pipeline_v2 — composable orchestrator for the extended TS pipeline.

Why this exists
---------------
The 2026-05-06 ChatGPT critique laid out a 9-stage TS workflow:

    1.  microstates (optional)
    2.  scan_1d (existing tools/scan_along_s)
    3.  scan_2d (NEW; tools/scan2d) — diagnostic
    4.  endpoint_release (NEW; tools/endpoint_release)
    5.  neb_ci (existing cowboy-qc neb)
    6.  dimer_polish (existing cowboy-qc saddle --backend dimer; uses NEB tangent)
    7.  validate_ts (NEW; quantum_engine.ops.expanded_hessian)
    8.  imag_mode_displace (NEW)
    9.  refine-ts wrap-up (existing)

Every stage is an INDIVIDUAL CLI tool / op. This orchestrator chains them
via a YAML config, gates each next stage on the previous one passing,
writes a unified ``pipeline_summary.json``, and supports
``--resume-from <stage>`` for restart.

Reaction-agnostic
-----------------
- ``reactive_atoms``, ``active_region``, charge, spin, all knobs come from
  the YAML config; no PTE-specific defaults.
- Each stage block has full passthrough to the underlying tool's CLI flags.
- Unknown / unimplemented stages skip with a warning instead of aborting,
  so the user can incrementally adopt features.

Example config
--------------
::

    input: MC.pdb
    charge_ledger: ledger.yaml
    reactive_atoms: ["P.SUB", "O3.OHX", "O7.SUB"]
    active_region: "site 5.0 SUB OHX KCX"
    boundary_fix_preset: ca-only
    model: mace-polar-m
    device: cuda

    stages:
      - microstates:
          generators: [his, water_shuffle]
          n_water_shuffle: 5
      - scan_1d:
          sum_target: auto
          n_points: 11
          drag_mode: pin-both
      - scan_2d:
          grid: 3x3
          delta_d: 0.20
      - endpoint_release:
          fmax: 0.02
      - neb_ci:
          n_images: 11
          optimizer: fire
          ts_tol_fmax: 0.02
      - dimer_polish:
          init_mode: neb_tangent
      - validate_ts:
          tier: b
          imag_mode_min_overlap: 0.5
      - imag_mode_displace:
          displacement_A: 0.20
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("quantum_engine.ops.ts_pipeline_v2")

KNOWN_STAGES = (
    "microstates",
    "protonation_ensemble",
    "scan_1d",
    "scan_2d",
    "endpoint_release",
    "neb_ci",
    "dimer_polish",
    "validate_ts",
    "imag_mode_displace",
    "refine_ts",
    "crest_funnel",
    "polish_ts",
)


# ----------------------------------------------------------------------------
# Free-residues / prune-residue-keep / prune-backbone-residues helpers
# ----------------------------------------------------------------------------
# These knobs may live at the TOP LEVEL of the YAML (so they propagate to
# EVERY stage uniformly) OR inside a single stage's args (overriding the
# global default). Callers obtain the resolved values via _resolve_prune_ctx.
def _csv_join_ints(values: Iterable[Any] | None) -> str:
    """Convert [131, 169] → '131,169'; None / empty → ''. Tolerates already-
    string inputs ('131,169') so the same helper works whether the YAML
    parsed the value as a list or a string."""
    if values is None:
        return ""
    if isinstance(values, str):
        return values.strip()
    if isinstance(values, (list, tuple)):
        return ",".join(str(int(x)) for x in values if str(x).strip())
    raise TypeError(f"_csv_join_ints: unsupported type {type(values).__name__}")


def _format_keep_specs(spec: dict[Any, Any] | None) -> list[str]:
    """Convert {131: [], 169: ['CD', 'CE', 'NZ']} into the
    repeatable RESID:ATOMS CLI tokens ['131:', '169:CD,CE,NZ'].
    Also tolerates the YAML-flat form (a list of strings already)
    AND the YAML scalar-string form ``{169: 'CD,CE,NZ'}`` (split on commas).
    Empty / None returns []."""
    if not spec:
        return []
    if isinstance(spec, list):
        return [str(s) for s in spec]
    out: list[str] = []
    for resid, atoms in spec.items():
        # YAML-side scalar strings: '{169: "CD,CE,NZ"}'.
        # Without this branch ",".join treated the str as iterable and produced
        # 'C,D,,,C,E,,,N,Z'.
        if isinstance(atoms, str):
            atoms_list = [a.strip() for a in atoms.split(",") if a.strip()]
        elif atoms is None:
            atoms_list = []
        else:
            atoms_list = list(atoms)
        atoms_str = ",".join(atoms_list)
        out.append(f"{int(resid)}:{atoms_str}")
    return out


def _resolve_prune_ctx(stage_args: dict, ctx: dict) -> dict[str, Any]:
    """Resolve the free / prune knobs for a stage. Stage-local args take
    priority; falls back to top-level YAML defaults stored in ``ctx``.

    Returns a dict with keys:
        free_residues: list[int] | None
        prune_residue_keep: dict[int, list[str]] | None
        prune_backbone_residues: list[int] | None
    """
    def pick(key):
        if key in stage_args and stage_args[key] is not None:
            return stage_args[key]
        return ctx.get(key)
    return {
        "free_residues": pick("free_residues"),
        "prune_residue_keep": pick("prune_residue_keep"),
        "prune_backbone_residues": pick("prune_backbone_residues"),
        "pre_crest_relax": pick("pre_crest_relax"),
        "pre_crest_fmax": pick("pre_crest_fmax"),
        "pre_crest_max_steps": pick("pre_crest_max_steps"),
        "cap_h_bond": pick("cap_h_bond"),
        "prune_xtb_relax": pick("prune_xtb_relax"),
        "prune_xtb_max_steps": pick("prune_xtb_max_steps"),
    }


@dataclass
class StageOutcome:
    name: str
    status: str
    outdir: Path | None
    extra: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    skipped: bool = False
    gate_pass: bool = True


@dataclass
class PipelineSummary:
    config_path: str | None
    base_outdir: str
    stages: list[StageOutcome] = field(default_factory=list)
    overall_pass: bool = True


# ----------------------------------------------------------------------------
# YAML loader
# ----------------------------------------------------------------------------
def _load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"ts-pipeline-v2 config not found: {p}")
    text = p.read_text()
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML required for ts-pipeline-v2 configs") from exc
    cfg = yaml.safe_load(text) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config {p} top-level must be a mapping")
    return cfg


def _flatten_stage_block(stage_block: Any) -> tuple[str, dict[str, Any]]:
    """Each list element is either {name: kwargs} or {'op': name, ...}."""
    if isinstance(stage_block, dict):
        if "op" in stage_block:
            name = str(stage_block["op"])
            kw = {k: v for k, v in stage_block.items() if k != "op"}
            return name, kw
        # single-key dict form: {scan_1d: {n_points: 11}}
        if len(stage_block) == 1:
            (name, kw), = stage_block.items()
            return str(name), dict(kw or {})
        raise ValueError(f"stage block must be {{name: kwargs}} or {{op: name, ...}}; got {stage_block}")
    raise ValueError(f"stage block must be a mapping; got {type(stage_block).__name__}")


# ----------------------------------------------------------------------------
# Stage runners (each emits a StageOutcome; uses subprocess to call the
# corresponding CLI to keep the orchestrator decoupled from in-process imports
# of every tool).
# ----------------------------------------------------------------------------
def _run_subprocess(cmd: list[str], stage_outdir: Path,
                    log_path: Path, env: dict | None = None) -> int:
    """Run a subprocess, tee stdout+stderr to ``log_path``."""
    stage_outdir.mkdir(parents=True, exist_ok=True)
    log.info("[ts_pipeline_v2] $ %s", " ".join(cmd))
    with open(log_path, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              env=env, check=False)
    return int(proc.returncode)


def _run_microstates(stage_args: dict, ctx: dict, stage_outdir: Path) -> StageOutcome:
    if not ctx.get("input"):
        return StageOutcome(name="microstates", status="failed",
                            outdir=stage_outdir,
                            error="ctx.input not set")
    cmd: list[str] = [
        sys.executable, str(Path(__file__).resolve().parents[2] / "tools" / "microstate_sampler.py"),
        "--input", str(ctx["input"]),
        "--out", str(stage_outdir),
        "--generators", ",".join(stage_args.get("generators", ["his"])),
    ]
    if stage_args.get("n_water_shuffle") is not None:
        cmd.extend(["--n-water-shuffle", str(stage_args["n_water_shuffle"])])
    if stage_args.get("n_water_translate") is not None:
        cmd.extend(["--n-water-translate", str(stage_args["n_water_translate"])])
    if stage_args.get("water_translate_delta") is not None:
        cmd.extend(["--water-translate-delta", str(stage_args["water_translate_delta"])])
    if ctx.get("charge_ledger"):
        cmd.extend(["--charge-ledger", str(ctx["charge_ledger"])])
    if stage_args.get("relax"):
        cmd.append("--relax")
    if ctx.get("model"):
        cmd.extend(["--model", ctx["model"]])
    if ctx.get("device"):
        cmd.extend(["--device", ctx["device"]])
    rc = _run_subprocess(cmd, stage_outdir, stage_outdir / "stage.log")
    return StageOutcome(
        name="microstates",
        status="completed" if rc == 0 else "failed",
        outdir=stage_outdir,
        extra={"rc": rc, "manifest": str(stage_outdir / "manifest.json")},
    )


def _run_scan_1d(stage_args: dict, ctx: dict, stage_outdir: Path) -> StageOutcome:
    cmd: list[str] = [
        sys.executable, str(Path(__file__).resolve().parents[2] / "tools" / "scan_along_s.py"),
        "--input", str(ctx["input"]),
        "--out", str(stage_outdir),
    ]
    for k in ("sum_target", "n_points", "s_min", "s_max", "fmax", "max_steps",
              "drag_mode", "model", "device"):
        v = stage_args.get(k, ctx.get(k))
        if v is not None:
            cmd.extend([f"--{k.replace('_','-')}", str(v)])
    if ctx.get("charge_ledger"):
        cmd.extend(["--charge-ledger", str(ctx["charge_ledger"])])
    # propagate --free-residues from top-level YAML
    pcx = _resolve_prune_ctx(stage_args, ctx)
    free_csv = _csv_join_ints(pcx["free_residues"])
    if free_csv:
        cmd.extend(["--free-residues", free_csv])
    rc = _run_subprocess(cmd, stage_outdir, stage_outdir / "stage.log")
    return StageOutcome(
        name="scan_1d",
        status="completed" if rc == 0 else "failed",
        outdir=stage_outdir,
        extra={"rc": rc},
    )


def _run_protonation_ensemble(stage_args: dict, ctx: dict, stage_outdir: Path) -> StageOutcome:
    """Generate a protonation-state ensemble of the input cluster.

    Stage args (all optional, all CLI-driven defaults from
    tools/microstate_sampler.py):

      mode: "auto" or "manual" (default "auto").
      rules_yaml: path to the rules YAML (when mode=="manual").
      max_microstates: cap (default 16).
      keep_top: number of variants to retain in ctx for downstream stages.
      protonation_families: e.g. "HIS,GLU".
      metal_cutoff_a, nh_bond_length, oh_bond_length, sh_bond_length: H
        placement knobs.

    Side effect: sets ``ctx['protonation_variants']`` to a list of variant
    PDB paths, and ``ctx['input']`` is REPLACED with the first variant
    (so a subsequent crest_funnel stage runs on it). Use ``crest_funnel``'s
    own ``--protonation-ensemble auto`` flag for a more integrated path
    that runs CREST per variant and ranks at the end.
    """
    if not ctx.get("input"):
        return StageOutcome(name="protonation_ensemble", status="failed",
                            outdir=stage_outdir,
                            error="ctx.input not set")
    mode = str(stage_args.get("mode", "auto"))
    cmd: list[str] = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "tools" / "microstate_sampler.py"),
        "--input", str(ctx["input"]),
        "--out", str(stage_outdir),
    ]
    if mode == "auto":
        cmd.append("--auto-protonation")
    elif mode == "manual":
        rules = stage_args.get("rules_yaml") or stage_args.get("rules")
        if not rules:
            return StageOutcome(
                name="protonation_ensemble", status="failed",
                outdir=stage_outdir,
                error="protonation_ensemble mode='manual' requires "
                      "stage_args.rules_yaml or stage_args.rules",
            )
        cmd.extend(["--protonation-rules", str(rules)])
    else:
        return StageOutcome(
            name="protonation_ensemble", status="failed",
            outdir=stage_outdir,
            error=f"unknown protonation_ensemble mode {mode!r}",
        )
    if stage_args.get("max_microstates") is not None:
        cmd.extend(["--max-microstates", str(stage_args["max_microstates"])])
    if stage_args.get("protonation_families"):
        cmd.extend(["--protonation-families", str(stage_args["protonation_families"])])
    if stage_args.get("metal_cutoff_a") is not None:
        cmd.extend(["--metal-cutoff-a", str(stage_args["metal_cutoff_a"])])
    if stage_args.get("nh_bond_length") is not None:
        cmd.extend(["--nh-bond-length", str(stage_args["nh_bond_length"])])
    if stage_args.get("oh_bond_length") is not None:
        cmd.extend(["--oh-bond-length", str(stage_args["oh_bond_length"])])
    if stage_args.get("sh_bond_length") is not None:
        cmd.extend(["--sh-bond-length", str(stage_args["sh_bond_length"])])
    if ctx.get("charge_ledger"):
        cmd.extend(["--charge-ledger", str(ctx["charge_ledger"])])
    rc = _run_subprocess(cmd, stage_outdir, stage_outdir / "stage.log")
    if rc != 0:
        return StageOutcome(
            name="protonation_ensemble", status="failed",
            outdir=stage_outdir, extra={"rc": rc},
        )
    # Read manifest, surface variant PDBs into ctx
    manifest_path = stage_outdir / "manifest.json"
    variants: list[str] = []
    if manifest_path.is_file():
        try:
            data = json.loads(manifest_path.read_text())
            variants = [m["pdb"] for m in data.get("microstates", [])]
        except Exception as exc:
            log.warning("could not read protonation manifest: %s", exc)
    keep_top = int(stage_args.get("keep_top", len(variants)))
    variants = variants[:keep_top]
    return StageOutcome(
        name="protonation_ensemble", status="completed",
        outdir=stage_outdir,
        extra={"rc": rc, "n_variants": len(variants), "variants": variants},
    )


def _run_crest_funnel(stage_args: dict, ctx: dict, stage_outdir: Path) -> StageOutcome:
    """Run tools/crest_funnel.py end-to-end with ``--free-residues`` /
    ``--prune-residue-keep`` / ``--prune-backbone-residues`` /
    ``--pre-crest-relax`` propagated from the YAML config or stage args.
    """
    if not ctx.get("input"):
        return StageOutcome(name="crest_funnel", status="failed",
                            outdir=stage_outdir,
                            error="ctx.input not set")
    cmd: list[str] = [
        sys.executable, str(Path(__file__).resolve().parents[2] / "tools" / "crest_funnel.py"),
        "--pdb", str(ctx["input"]),
        "--out", str(stage_outdir),
    ]
    # Pass-through scalar args (only when explicitly set so we don't override
    # crest_funnel's own argparse defaults). Codex flag 2026-05-07: a number
    # of valid CREST flags were dropped silently; expanded list here.
    for k in (
        "ncpu", "top_n", "top_keep", "solvent",
        "crest_preset", "crest_mdlen", "crest_rthr", "crest_mddump",
        "crest_opt_level", "crest_walltime", "stages", "charge",
        "xtb_preopt_timeout", "gxtb_sp_timeout", "per_job_timeout",
        "min_threads_per_job",
        "stage_c_opt_level", "stage_c_tiers", "stage_c_tier1_opt_level",
        "stage_c_tier1_timeout", "stage_c_tier2_opt_level",
        "stage_c_tier2_timeout", "stage_c_tier1_keep_top",
        "stage_c_energy_outlier_cutoff", "stage_c_sigterm_grace_s",
        "stage_c_salvage_gradnorm_max", "min_stage_c_survivors",
        "stage_c_progress_check_fraction", "stage_c_progress_grad_ratio",
        "stage_c_salvage_grad_ratio", "stage_c_max_extensions",
        "stage_c_extension_factor",
        "stage_e_opt_level", "stage_e_sigterm_grace_s",
        "stage_e_salvage_gradnorm_max", "min_stage_e_survivors",
        "stage_e_progress_check_fraction", "stage_e_progress_grad_ratio",
        "stage_e_salvage_grad_ratio", "stage_e_max_extensions",
        "stage_e_extension_factor",
        "post_crest_bond_cutoff", "post_crest_bad_bond_mode",
        "post_crest_min_survivors", "post_crest_repair_max_passes",
        "post_crest_source_shrink_tolerance",
        "post_crest_reactive_atoms", "post_crest_max_match_distance",
        # Protonation-ensemble pass-through
        "protonation_ensemble", "protonation_max_microstates",
        "protonation_keep_top", "protonation_metal_cutoff_a",
        "protonation_nh_bond", "protonation_oh_bond", "protonation_sh_bond",
    ):
        v = stage_args.get(k)
        if v is not None:
            cmd.extend([f"--{k.replace('_','-')}", str(v)])
    if stage_args.get("freeze_zn"):
        cmd.append("--freeze-zn")
    if stage_args.get("cleanup"):
        cmd.append("--cleanup")
    pcx = _resolve_prune_ctx(stage_args, ctx)
    free_csv = _csv_join_ints(pcx["free_residues"])
    if free_csv:
        cmd.extend(["--free-residues", free_csv])
    keep_specs = _format_keep_specs(pcx["prune_residue_keep"])
    if keep_specs:
        cmd.append("--prune-residue-keep")
        cmd.extend(keep_specs)
    bb_csv = _csv_join_ints(pcx["prune_backbone_residues"])
    if bb_csv:
        cmd.extend(["--prune-backbone-residues", bb_csv])
    if pcx["cap_h_bond"] is not None:
        cmd.extend(["--cap-h-bond", str(pcx["cap_h_bond"])])
    if pcx["prune_xtb_max_steps"] is not None:
        cmd.extend(["--prune-xtb-max-steps", str(pcx["prune_xtb_max_steps"])])
    if pcx["prune_xtb_relax"] is False:
        cmd.append("--no-prune-xtb-relax")
    if pcx["pre_crest_relax"]:
        cmd.extend(["--pre-crest-relax", str(pcx["pre_crest_relax"])])
    if pcx["pre_crest_fmax"] is not None:
        cmd.extend(["--pre-crest-fmax", str(pcx["pre_crest_fmax"])])
    if pcx["pre_crest_max_steps"] is not None:
        cmd.extend(["--pre-crest-max-steps", str(pcx["pre_crest_max_steps"])])
    # Propagate the global charge ledger to crest_funnel so per-variant runs
    # (--protonation-ensemble) can build per-variant ledgers from it. (codex
    # flag, 2026-05-07: previously dropped silently.)
    if ctx.get("charge_ledger"):
        cmd.extend(["--charge-ledger", str(ctx["charge_ledger"])])
    rc = _run_subprocess(cmd, stage_outdir, stage_outdir / "stage.log")
    return StageOutcome(
        name="crest_funnel",
        status="completed" if rc == 0 else "failed",
        outdir=stage_outdir,
        extra={"rc": rc, "results": str(stage_outdir / "results")},
    )


def _run_polish_ts(stage_args: dict, ctx: dict, stage_outdir: Path) -> StageOutcome:
    """Run tools/polish_ts_v3.py with --free-residues / --prune-* propagated."""
    if not ctx.get("input"):
        return StageOutcome(name="polish_ts", status="failed",
                            outdir=stage_outdir,
                            error="ctx.input not set")
    cmd: list[str] = [
        sys.executable, str(Path(__file__).resolve().parents[2] / "tools" / "polish_ts_v3.py"),
        "--input", str(ctx["input"]),
        "--out", str(stage_outdir),
    ]
    # Codex flag 2026-05-07: previously dropped reactive-override + constraint
    # + graft + control flags. Expanded list here.
    for k in (
        "model", "device", "charge", "fmax", "max_steps",
        "snapshot_stride", "out_basename",
        # Reactive overrides
        "p_name", "p_res", "nuc_name", "nuc_res", "lg_name", "lg_res",
        "target_d_p_onuc", "target_d_p_olg",
        # Graft-back + Mulliken + meta
        "graft_relax_max_steps", "metal_oxidation", "seed",
    ):
        v = stage_args.get(k, ctx.get(k))
        if v is not None:
            cmd.extend([f"--{k.replace('_','-')}", str(v)])
    # Boolean / list flags
    if stage_args.get("graft_back"):
        cmd.append("--graft-back")
    if stage_args.get("no_graft_relax"):
        cmd.append("--no-graft-relax")
    if stage_args.get("no_compute_mulliken"):
        cmd.append("--no-compute-mulliken")
    if stage_args.get("dry_run"):
        cmd.append("--dry-run")
    fix_atoms = stage_args.get("fix_atoms")
    if fix_atoms:
        cmd.extend(["--fix-atoms", _csv_join_ints(fix_atoms)])
    for spec_key in ("fix_distance", "fix_angle", "fix_dihedral"):
        specs = stage_args.get(spec_key)
        if specs:
            cmd.append(f"--{spec_key.replace('_','-')}")
            if isinstance(specs, str):
                cmd.append(specs)
            else:
                cmd.extend(str(s) for s in specs)
    pcx = _resolve_prune_ctx(stage_args, ctx)
    free_csv = _csv_join_ints(pcx["free_residues"])
    if free_csv:
        cmd.extend(["--free-residues", free_csv])
    keep_specs = _format_keep_specs(pcx["prune_residue_keep"])
    if keep_specs:
        cmd.append("--prune-residue-keep")
        cmd.extend(keep_specs)
    bb_csv = _csv_join_ints(pcx["prune_backbone_residues"])
    if bb_csv:
        cmd.extend(["--prune-backbone-residues", bb_csv])
    if pcx["cap_h_bond"] is not None:
        cmd.extend(["--cap-h-bond", str(pcx["cap_h_bond"])])
    if pcx["prune_xtb_max_steps"] is not None:
        cmd.extend(["--prune-xtb-max-steps", str(pcx["prune_xtb_max_steps"])])
    if pcx["prune_xtb_relax"] is False:
        cmd.append("--no-prune-xtb-relax")
    rc = _run_subprocess(cmd, stage_outdir, stage_outdir / "stage.log")
    return StageOutcome(
        name="polish_ts",
        status="completed" if rc == 0 else "failed",
        outdir=stage_outdir,
        extra={"rc": rc},
    )


def _run_scan_2d(stage_args: dict, ctx: dict, stage_outdir: Path) -> StageOutcome:
    ts_guess = stage_args.get("ts_guess") or ctx.get("scan_1d_ts_guess")
    if not ts_guess:
        return StageOutcome(name="scan_2d", status="skipped", outdir=stage_outdir,
                            skipped=True,
                            error="no ts_guess provided and no upstream scan_1d_ts_guess in ctx")
    cmd: list[str] = [
        sys.executable, str(Path(__file__).resolve().parents[2] / "tools" / "scan2d.py"),
        "--input", str(ctx["input"]),
        "--ts-guess", str(ts_guess),
        "--out", str(stage_outdir),
        "--bond-a", str(stage_args["bond_a"]),
        "--bond-b", str(stage_args["bond_b"]),
    ]
    if stage_args.get("grid"):
        cmd.extend(["--grid", str(stage_args["grid"])])
    if stage_args.get("delta_d") is not None:
        cmd.extend(["--delta-d", str(stage_args["delta_d"])])
    if ctx.get("charge_ledger"):
        cmd.extend(["--charge-ledger", str(ctx["charge_ledger"])])
    if ctx.get("model"):
        cmd.extend(["--model", ctx["model"]])
    if ctx.get("device"):
        cmd.extend(["--device", ctx["device"]])
    if ctx.get("boundary_fix_preset"):
        cmd.extend(["--boundary-fix-preset", ctx["boundary_fix_preset"]])
    rc = _run_subprocess(cmd, stage_outdir, stage_outdir / "stage.log")
    return StageOutcome(
        name="scan_2d",
        status="completed" if rc == 0 else "failed",
        outdir=stage_outdir,
        extra={"rc": rc},
    )


def _run_endpoint_release(stage_args: dict, ctx: dict, stage_outdir: Path) -> StageOutcome:
    """Run two endpoint-release calls (R-side and P-side) if R/P paths are in ctx."""
    r_path = stage_args.get("r_pdb") or ctx.get("scan_1d_r_endpoint")
    p_path = stage_args.get("p_pdb") or ctx.get("scan_1d_p_endpoint")
    if not r_path and not p_path:
        return StageOutcome(name="endpoint_release", status="skipped", outdir=stage_outdir,
                            skipped=True,
                            error="no R/P endpoint paths supplied (set r_pdb/p_pdb under stage args)")
    rc_total = 0
    extras: dict[str, Any] = {}
    for tag, src in [("R", r_path), ("P", p_path)]:
        if not src:
            continue
        out_pdb = stage_outdir / f"endpoint_{tag}_released.pdb"
        cmd: list[str] = [
            sys.executable, str(Path(__file__).resolve().parents[2] / "tools" / "endpoint_release.py"),
            str(src), "--out", str(out_pdb),
        ]
        if stage_args.get("fmax") is not None:
            cmd.extend(["--fmax", str(stage_args["fmax"])])
        if ctx.get("boundary_fix_preset"):
            cmd.extend(["--boundary-fix-preset", ctx["boundary_fix_preset"]])
        if ctx.get("charge_ledger"):
            cmd.extend(["--charge-ledger", str(ctx["charge_ledger"])])
        if ctx.get("model"):
            cmd.extend(["--model", ctx["model"]])
        if ctx.get("device"):
            cmd.extend(["--device", ctx["device"]])
        for spec in stage_args.get("release_bonds", []):
            cmd.extend(["--release-bond", str(spec)])
        rc = _run_subprocess(cmd, stage_outdir, stage_outdir / f"{tag}_stage.log")
        rc_total |= rc
        extras[f"{tag}_pdb"] = str(out_pdb)
        extras[f"{tag}_rc"] = rc
    return StageOutcome(
        name="endpoint_release",
        status="completed" if rc_total == 0 else "failed",
        outdir=stage_outdir,
        extra=extras,
    )


def _run_neb_ci(stage_args: dict, ctx: dict, stage_outdir: Path) -> StageOutcome:
    r_pdb = stage_args.get("r_pdb") or ctx.get("endpoint_R_pdb")
    p_pdb = stage_args.get("p_pdb") or ctx.get("endpoint_P_pdb")
    if not (r_pdb and p_pdb):
        return StageOutcome(name="neb_ci", status="skipped", outdir=stage_outdir,
                            skipped=True,
                            error="need both r_pdb and p_pdb (released endpoints)")
    cmd: list[str] = [
        "cowboy-qc", "neb", str(r_pdb), str(p_pdb),
        "--outdir", str(stage_outdir),
    ]
    for k in ("n_images", "optimizer", "interpolation", "k_spring",
              "fmax_climb", "fmax_noclimb", "ts_tol_fmax", "max_step"):
        v = stage_args.get(k)
        if v is not None:
            cmd.extend([f"--{k.replace('_','-')}", str(v)])
    if ctx.get("model"):
        cmd.extend(["--model", ctx["model"]])
    if ctx.get("device"):
        cmd.extend(["--device", ctx["device"]])
    if ctx.get("boundary_fix_preset"):
        cmd.extend(["--fix-preset", ctx["boundary_fix_preset"]])
    rc = _run_subprocess(cmd, stage_outdir, stage_outdir / "stage.log")
    return StageOutcome(
        name="neb_ci",
        status="completed" if rc == 0 else "failed",
        outdir=stage_outdir,
        extra={"rc": rc},
    )


def _run_dimer_polish(stage_args: dict, ctx: dict, stage_outdir: Path) -> StageOutcome:
    """Run cowboy-qc refine-ts --from-neb in dimer mode (uses NEB tangent automatically)."""
    neb_dir = stage_args.get("from_neb") or ctx.get("neb_outdir")
    if not neb_dir:
        return StageOutcome(name="dimer_polish", status="skipped", outdir=stage_outdir,
                            skipped=True,
                            error="need from_neb (path to neb run dir) in stage args or ctx")
    reactive_atoms = ctx.get("reactive_atoms") or stage_args.get("reactive_atoms")
    if not reactive_atoms:
        return StageOutcome(name="dimer_polish", status="failed", outdir=stage_outdir,
                            error="need reactive_atoms in ctx or stage args")
    cmd: list[str] = [
        "cowboy-qc", "refine-ts",
        "--outdir", str(stage_outdir),
        "--from-neb", str(neb_dir),
        "--backend", str(stage_args.get("backend", "dimer")),
        "--reactive-atoms", *(str(a) for a in reactive_atoms),
    ]
    if stage_args.get("saddle_fmax") is not None:
        cmd.extend(["--saddle-fmax", str(stage_args["saddle_fmax"])])
    if ctx.get("model"):
        cmd.extend(["--model", ctx["model"]])
    if ctx.get("device"):
        cmd.extend(["--device", ctx["device"]])
    if ctx.get("input"):
        cmd.extend(["--template-pdb", str(ctx["input"])])
    rc = _run_subprocess(cmd, stage_outdir, stage_outdir / "stage.log")
    return StageOutcome(
        name="dimer_polish",
        status="completed" if rc == 0 else "failed",
        outdir=stage_outdir,
        extra={"rc": rc},
    )


def _run_validate_ts(stage_args: dict, ctx: dict, stage_outdir: Path) -> StageOutcome:
    ts_pdb = stage_args.get("ts_pdb") or ctx.get("dimer_polish_ts_pdb")
    if not ts_pdb:
        return StageOutcome(name="validate_ts", status="skipped", outdir=stage_outdir,
                            skipped=True,
                            error="need ts_pdb (refined TS) in stage args or ctx")
    reactive_atoms = ctx.get("reactive_atoms") or stage_args.get("reactive_atoms")
    if not reactive_atoms:
        return StageOutcome(name="validate_ts", status="failed", outdir=stage_outdir,
                            error="need reactive_atoms")
    cmd: list[str] = [
        "cowboy-qc", "validate-ts", str(ts_pdb),
        "--outdir", str(stage_outdir),
        "--reactive-atoms", *(str(a) for a in reactive_atoms),
        "--tier", str(stage_args.get("tier", "b")),
    ]
    if ctx.get("active_region"):
        cmd.extend(["--active-region", str(ctx["active_region"])])
    if stage_args.get("imag_mode_min_overlap") is not None:
        cmd.extend(["--imag-mode-min-overlap", str(stage_args["imag_mode_min_overlap"])])
    if ctx.get("model"):
        cmd.extend(["--model", ctx["model"]])
    if ctx.get("device"):
        cmd.extend(["--device", ctx["device"]])
    if ctx.get("charge_ledger"):
        cmd.extend(["--charge-ledger", str(ctx["charge_ledger"])])
    rc = _run_subprocess(cmd, stage_outdir, stage_outdir / "stage.log")
    return StageOutcome(
        name="validate_ts",
        status="completed" if rc == 0 else "failed",
        outdir=stage_outdir,
        extra={"rc": rc},
    )


def _run_imag_mode_displace(stage_args: dict, ctx: dict, stage_outdir: Path) -> StageOutcome:
    ts_pdb = stage_args.get("ts_pdb") or ctx.get("dimer_polish_ts_pdb")
    mode_path = stage_args.get("imag_mode") or ctx.get("dimer_polish_modes")
    if not ts_pdb or not mode_path:
        return StageOutcome(name="imag_mode_displace", status="skipped",
                            outdir=stage_outdir, skipped=True,
                            error="need ts_pdb + imag_mode (npy or xyz)")
    cmd: list[str] = [
        "cowboy-qc", "verify-irc-like", str(ts_pdb),
        "--outdir", str(stage_outdir),
        "--imag-mode", str(mode_path),
        "--displacement", str(stage_args.get("displacement_A", 0.20)),
    ]
    if stage_args.get("fmax") is not None:
        cmd.extend(["--fmax", str(stage_args["fmax"])])
    if ctx.get("model"):
        cmd.extend(["--model", ctx["model"]])
    if ctx.get("device"):
        cmd.extend(["--device", ctx["device"]])
    rc = _run_subprocess(cmd, stage_outdir, stage_outdir / "stage.log")
    return StageOutcome(
        name="imag_mode_displace",
        status="completed" if rc == 0 else "failed",
        outdir=stage_outdir,
        extra={"rc": rc},
    )


_STAGE_RUNNERS = {
    "microstates": _run_microstates,
    "protonation_ensemble": _run_protonation_ensemble,
    "scan_1d": _run_scan_1d,
    "scan_2d": _run_scan_2d,
    "endpoint_release": _run_endpoint_release,
    "neb_ci": _run_neb_ci,
    "dimer_polish": _run_dimer_polish,
    "validate_ts": _run_validate_ts,
    "imag_mode_displace": _run_imag_mode_displace,
    "crest_funnel": _run_crest_funnel,
    "polish_ts": _run_polish_ts,
}


# ----------------------------------------------------------------------------
# Top-level driver
# ----------------------------------------------------------------------------
def run_pipeline(
    config_path: str | Path,
    *,
    base_outdir: str | Path | None = None,
    resume_from: str | None = None,
    dry_run: bool = False,
    only_stages: Iterable[str] | None = None,
) -> PipelineSummary:
    """Run the v2 TS pipeline as defined by ``config_path``.

    Args:
        config_path: YAML config with at least ``input`` and ``stages``.
        base_outdir: Override the config's ``outdir`` key (or default name).
        resume_from: stage name to start at (e.g. ``"endpoint_release"``);
            earlier stages are skipped (assumed already completed).
        dry_run: log the planned stage commands and exit.
        only_stages: if non-empty, restrict to this whitelist of stage names.
    """
    cfg = _load_config(config_path)
    base_outdir_p = Path(base_outdir or cfg.get("outdir") or "qcb-ts-pipeline-v2-out")
    base_outdir_p.mkdir(parents=True, exist_ok=True)

    stages_blocks = cfg.get("stages") or []
    if not stages_blocks:
        raise ValueError(f"config {config_path}: no 'stages:' list")

    ctx: dict[str, Any] = {
        "input": cfg.get("input"),
        "charge": cfg.get("charge"),         # codex flag 2026-05-07: was missing
        "spin": cfg.get("spin"),
        "charge_ledger": cfg.get("charge_ledger"),
        "model": cfg.get("model"),
        "device": cfg.get("device", "cuda"),
        "boundary_fix_preset": cfg.get("boundary_fix_preset"),
        "reactive_atoms": cfg.get("reactive_atoms"),
        "active_region": cfg.get("active_region"),
        # Free + prune propagate to ALL stages by default. Stage-local overrides
        # win (handled in _resolve_prune_ctx).
        "free_residues": cfg.get("free_residues"),
        "prune_residue_keep": cfg.get("prune_residue_keep"),
        "prune_backbone_residues": cfg.get("prune_backbone_residues"),
        "cap_h_bond": cfg.get("cap_h_bond"),
        "prune_xtb_relax": cfg.get("prune_xtb_relax"),
        "prune_xtb_max_steps": cfg.get("prune_xtb_max_steps"),
        "pre_crest_relax": cfg.get("pre_crest_relax"),
        "pre_crest_fmax": cfg.get("pre_crest_fmax"),
        "pre_crest_max_steps": cfg.get("pre_crest_max_steps"),
    }
    log.info("[ts_pipeline_v2] base_outdir=%s, input=%s, n_stages=%d",
             base_outdir_p, ctx.get("input"), len(stages_blocks))

    summary = PipelineSummary(
        config_path=str(config_path),
        base_outdir=str(base_outdir_p),
    )
    only_set = set(only_stages or [])
    skipping_until_resume = bool(resume_from)

    for i, block in enumerate(stages_blocks, start=1):
        stage_name, stage_args = _flatten_stage_block(block)
        stage_outdir = base_outdir_p / f"{i:02d}_{stage_name}"
        # Resume gating
        if skipping_until_resume:
            if stage_name == resume_from:
                skipping_until_resume = False
            else:
                outcome = StageOutcome(
                    name=stage_name, status="skipped_resume",
                    outdir=stage_outdir, skipped=True,
                )
                summary.stages.append(outcome)
                log.info("[ts_pipeline_v2] skipping %s (--resume-from %s)",
                         stage_name, resume_from)
                continue
        if only_set and stage_name not in only_set:
            outcome = StageOutcome(name=stage_name, status="skipped_only_stages",
                                   outdir=stage_outdir, skipped=True)
            summary.stages.append(outcome)
            continue
        if stage_name not in _STAGE_RUNNERS:
            log.warning("[ts_pipeline_v2] unknown stage %s; recording as skipped",
                        stage_name)
            outcome = StageOutcome(
                name=stage_name, status="unknown_stage",
                outdir=stage_outdir, skipped=True,
                error=f"stage {stage_name!r} not in {sorted(_STAGE_RUNNERS)}",
            )
            summary.stages.append(outcome)
            continue
        runner = _STAGE_RUNNERS[stage_name]

        if dry_run:
            log.info("[ts_pipeline_v2] DRY-RUN would execute %s in %s",
                     stage_name, stage_outdir)
            summary.stages.append(StageOutcome(
                name=stage_name, status="dry_run", outdir=stage_outdir,
                skipped=True,
            ))
            continue

        outcome = runner(stage_args, ctx, stage_outdir)
        summary.stages.append(outcome)
        # Update ctx with outputs from this stage so downstream stages can
        # consume them (best-effort heuristics).
        if stage_name == "scan_1d" and outcome.status == "completed":
            ctx["scan_1d_dir"] = str(stage_outdir)
            # Best-effort: pick last scan point as P, second as R
            scan_pdbs = sorted(stage_outdir.glob("point_*_s*.pdb"))
            if len(scan_pdbs) >= 2:
                ctx["scan_1d_r_endpoint"] = str(scan_pdbs[1])
                ctx["scan_1d_p_endpoint"] = str(scan_pdbs[-2])
                # ts_guess = energy max
                ctx["scan_1d_ts_guess"] = str(scan_pdbs[len(scan_pdbs) // 2])
        if stage_name == "endpoint_release" and outcome.status == "completed":
            ctx["endpoint_R_pdb"] = outcome.extra.get("R_pdb")
            ctx["endpoint_P_pdb"] = outcome.extra.get("P_pdb")
        if stage_name == "neb_ci" and outcome.status == "completed":
            ctx["neb_outdir"] = str(stage_outdir)
        if stage_name == "dimer_polish" and outcome.status == "completed":
            # refine-ts emits ts_refined.pdb under <outdir>/saddle/
            for cand in (stage_outdir / "ts_refined.pdb",
                         stage_outdir / "saddle" / "saddle.xyz"):
                if cand.is_file():
                    ctx["dimer_polish_ts_pdb"] = str(cand)
                    break
            for cand in (stage_outdir / "freq" / "modes.npy",):
                if cand.is_file():
                    ctx["dimer_polish_modes"] = str(cand)
                    break

        if outcome.status not in ("completed", "skipped", "skipped_resume",
                                  "skipped_only_stages", "dry_run"):
            log.error("[ts_pipeline_v2] stage %s failed (status=%s); halting",
                      stage_name, outcome.status)
            summary.overall_pass = False
            break

    summary_path = base_outdir_p / "pipeline_summary.json"
    summary_path.write_text(json.dumps({
        "config_path": summary.config_path,
        "base_outdir": summary.base_outdir,
        "overall_pass": summary.overall_pass,
        "stages": [
            {
                "name": s.name, "status": s.status,
                "outdir": str(s.outdir) if s.outdir else None,
                "extra": s.extra, "error": s.error, "skipped": s.skipped,
            }
            for s in summary.stages
        ],
    }, indent=2, default=str))
    log.info("[ts_pipeline_v2] summary written to %s", summary_path)
    return summary


# Convenience entry exposed via ``cowboy-qc ts-pipeline-v2``.
def run(config: str | Path, *, outdir: str | Path | None = None,
        resume_from: str | None = None, dry_run: bool = False,
        only_stages: Iterable[str] | None = None, **kwargs) -> dict:
    summary = run_pipeline(
        config, base_outdir=outdir, resume_from=resume_from,
        dry_run=dry_run, only_stages=only_stages,
    )
    return {
        "status": "completed" if summary.overall_pass else "failed",
        "overall_pass": summary.overall_pass,
        "n_stages": len(summary.stages),
        "outdir": summary.base_outdir,
        "outputs": {"summary": str(Path(summary.base_outdir) / "pipeline_summary.json")},
    }


_PIPELINE_EXAMPLE_YAML = """\
# Example config — reaction-agnostic. Replace placeholder atom tokens
# (NAME.RESNAME) with the bonds that change in YOUR reaction. The tokens
# below are illustrative SN2-style placeholders, not hardcoded PTE atoms.
#
# For an SN2-at-P enzyme: bonds are nucleophile-P and P-leaving group.
# For Diels-Alder: the two new C-C bonds.
# For hydride transfer: donor-H and H-acceptor.
input: MC.pdb
charge_ledger: ledger.yaml
# ALL reactive atoms (anything whose motion is part of the reaction
# coordinate). 1-based PDB serials, 0:idx, or NAME.RESNAME tokens.
reactive_atoms: ["NUC.SUB", "ELE.SUB", "LG.SUB"]
# Active-region grammar (cowboy-qc saddle): 'site R RES1 RES2 ...' or
# 'residue X Y' or 'chain A; residue Z'.
active_region: "site 5.0 SUB"
boundary_fix_preset: ca-only
model: mace-polar-m   # POLAR-1-M only for now; -L deferred per user
device: cuda
outdir: ts_pipeline_v2_out

# (optional) Top-level free-residues + prune knobs. These propagate to EVERY
# stage that supports them (crest_funnel, scan_1d, polish_ts, etc.). Per-
# stage overrides (in the stage block) win. Reaction-agnostic — set to your
# reactive sidechain residues; e.g. for a PTE GLU mutant: free TRP131 and
# GLU169 plus strip their backbones.
# free_residues: [131, 169]
# prune_residue_keep: {131: [], 169: [CG, CD, OE1, OE2]}  # 131 entirely; 169 sidechain
# prune_backbone_residues: [131, 169]
# pre_crest_relax: xtb-loose

stages:
  - microstates:
      generators: [his, water_shuffle]
      n_water_shuffle: 5
  - scan_1d:
      sum_target: auto
      n_points: 11
      drag_mode: pin-both
  - scan_2d:
      # bond tokens are the two reactive bonds; replace per reaction.
      bond_a: NUC.SUB,ELE.SUB
      bond_b: ELE.SUB,LG.SUB
      grid: 3x3
      delta_d: 0.20
  - endpoint_release:
      fmax: 0.02
      release_bonds:
        - NUC.SUB,ELE.SUB
        - ELE.SUB,LG.SUB
  - neb_ci:
      n_images: 11
      optimizer: fire
      ts_tol_fmax: 0.02
  - dimer_polish:
      backend: dimer
      saddle_fmax: 0.02
  - validate_ts:
      tier: b
      imag_mode_min_overlap: 0.5
  - imag_mode_displace:
      displacement_A: 0.20
"""
