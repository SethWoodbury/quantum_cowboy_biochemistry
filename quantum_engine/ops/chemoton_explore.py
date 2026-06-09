"""``cowboy-qc chemoton-explore`` — steered Chemoton reaction-network exploration.

This op orchestrates the full Chemoton-explore pipeline against a PDB input:

    pdb_to_scine_cluster   →   make_scine_model   →   make_pte_reactive_filter
                                       ↓
                                 SteeringWheel
                                       ↓
                            export_top_steps_as_pdb
                                       ↓
                       (optional) downstream MACE refinement

The actual SCINE plumbing lives in ``tools/scine_bridge.py``. This file is
glue: argparse → bridge calls → JSON-serializable result dict.

Note: requires a running MongoDB (mongod) reachable at the
``--mongodb-uri``. The companion script ``tools/scine_mongo.sh`` (built by
the container subagent) handles per-user mongod lifecycle on SLURM.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

log = logging.getLogger("quantum_engine.ops.chemoton_explore")


# ----- Helpers ------------------------------------------------------------- #

def _add_tools_to_path() -> None:
    """Ensure ``tools/`` is importable so ``scine_bridge`` can be imported."""
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    tools = repo_root / "tools"
    if tools.is_dir() and str(tools) not in sys.path:
        sys.path.insert(0, str(tools))


def _parse_mongodb_uri(uri: str) -> Tuple[str, int, str]:
    """Parse a ``mongodb://host:port/dbname`` URI → ``(host, port, dbname)``."""
    parsed = urlparse(uri)
    if parsed.scheme not in ("mongodb",):
        raise ValueError(
            f"--mongodb-uri must start with 'mongodb://', got {uri!r}"
        )
    host = parsed.hostname or "localhost"
    port = int(parsed.port) if parsed.port else 27017
    dbname = (parsed.path or "/").lstrip("/")
    if not dbname:
        # Generate a per-run name: ts-based + random suffix
        import datetime, uuid
        dbname = (
            "chemoton_run_"
            + datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            + "_"
            + uuid.uuid4().hex[:6]
        )
        log.info("no dbname in URI; using auto-generated %r", dbname)
    return host, port, dbname


def _build_credentials(host: str, port: int, dbname: str) -> Any:
    """Construct ``scine_database.Credentials`` for the given Mongo host."""
    import scine_database as db
    return db.Credentials(host, port, dbname)


def _check_mongo_connectable(creds: Any) -> Tuple[bool, str]:
    """Probe whether the configured Mongo is reachable. Best-effort."""
    try:
        import scine_database as db
        manager = db.Manager()
        manager.set_credentials(creds)
        manager.connect()
        ok = manager.is_connected()
        if ok:
            try:
                manager.disconnect()
            except Exception:
                pass
            return True, "connected"
        return False, "set_credentials succeeded but is_connected() returned False"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


# ----- Main run() ---------------------------------------------------------- #

def run(
    input_pdb: Path,
    *,
    cluster_spec: str = "auto",
    backend: str = "xtb-gfn2",
    max_bond_modifications: int = 2,
    max_depth: int = 2,
    barrier_cap_kcal: float = 60.0,
    top_n_export: int = 5,
    mongodb_uri: str = "mongodb://localhost:27017/",
    out_dir: Path = Path("runs/chemoton"),
    central_metal: str = "Zn",
    cluster_charge: int = 1,
    cluster_multiplicity: int = 1,
    restart_file: Optional[Path] = None,
    validate_with: Optional[str] = None,
    dry_run: bool = False,
    reactive_atoms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run the full ``cowboy-qc chemoton-explore`` pipeline.

    Parameters
    ----------
    input_pdb
        Path to the prepared PDB (active site + ligand + metals).
    cluster_spec
        Either ``"auto"`` (residues within 5 Å of HETATM core) or a
        space-separated atom-selection grammar (currently only ``"auto"``
        and explicit residue lists are supported; the grammar from the spec
        is treated as ``auto`` if anything else is supplied for now).
    backend
        Cheap-tier backend: ``'sparrow-pm6'`` (default), ``'sparrow-pm7'``,
        ``'xtb-gfn2'``.
    max_bond_modifications
        Per-step ``max_bond_associations`` and ``max_bond_dissociations``.
    max_depth
        Number of expansion rounds beyond the input.
    barrier_cap_kcal
        Step-3 barrier filter (gas-phase electronic, kcal/mol).
    top_n_export
        How many lowest-barrier steps to write back as PDB.
    mongodb_uri
        Mongo URI; if dbname is empty an auto-generated name is used.
    out_dir
        Destination for cluster XYZ, wheel state, exported PDBs, JSON summary.
    central_metal
        Metal symbol for CentralSiteFilter / CentralMetalSelection. Default
        ``'Zn'`` for PTE.
    cluster_charge, cluster_multiplicity
        Net charge and spin of the cluster (passed to FileInputSelection).
    restart_file
        Optional pickle from a previous wheel run.
    validate_with
        Reserved (e.g. ``'mace-polar-m'``) for downstream MACE refinement.
        Currently logged only — running mace polish is the user's next step
        via ``polish_ts_v3.py`` on the exported PDBs.
    dry_run
        If True, build the wheel + verify Mongo reachability but do not start
        the wheel. Useful for plumbing checks.
    reactive_atoms
        Reserved hint (e.g. ``["Zn1", "SUB:P1", "OHX:O3"]``). Currently logged
        only — the filter stack derives reactivity from element rules.

    Returns
    -------
    dict
        ``{"status", "outputs", "exploration", "exported_pdbs", ...}``
    """
    _add_tools_to_path()
    from scine_bridge import (
        export_top_steps_as_pdb,
        make_pte_reactive_filter,
        make_scine_model,
        pdb_to_scine_cluster,
        run_steered_exploration,
        scine_available,
    )

    if not scine_available():
        raise SystemExit(
            "SCINE not available in this environment. "
            "Run inside the cowboy-qc apptainer container "
            "(/net/software/containers/users/woodbuse/quantum_chem/)."
        )

    t_total_start = time.perf_counter()

    input_pdb = Path(input_pdb).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("=" * 70)
    log.info("cowboy-qc chemoton-explore")
    log.info("  input_pdb = %s", input_pdb)
    log.info("  out_dir = %s", out_dir)
    log.info("  backend = %s", backend)
    log.info("  max_bond_modifications = %d", max_bond_modifications)
    log.info("  max_depth = %d", max_depth)
    log.info("  barrier_cap_kcal = %.1f", barrier_cap_kcal)
    log.info("  top_n_export = %d", top_n_export)
    log.info("  mongodb_uri = %s", mongodb_uri)
    log.info("  cluster_spec = %s", cluster_spec)
    log.info("  central_metal = %s", central_metal)
    log.info("  charge / multiplicity = %d / %d", cluster_charge, cluster_multiplicity)
    if reactive_atoms:
        log.info("  reactive_atoms hint = %s", reactive_atoms)
    log.info("  dry_run = %s", dry_run)
    log.info("=" * 70)

    # ----- 1. Build cluster --------------------------------------------------
    cluster_residues_arg: Any = "auto"
    if cluster_spec and cluster_spec != "auto":
        # Keep "auto" as default; explicit grammar parsing is a v0.2 task.
        log.info(
            "cluster_spec=%s recognised but explicit grammar parsing is not yet "
            "implemented; falling back to auto-cluster.", cluster_spec,
        )
    cluster, atom_index_map = pdb_to_scine_cluster(
        input_pdb,
        cluster_residues=cluster_residues_arg,
    )
    log.info(
        "cluster: %d atoms, atom_index_map covers %d entries",
        cluster.size(), len(atom_index_map),
    )

    # ----- 2. SCINE model ----------------------------------------------------
    model = make_scine_model(backend=backend)

    # ----- 3. Reactive-site filter stack ------------------------------------
    site_filter = make_pte_reactive_filter(
        central_metal=central_metal,
        protein_backbone_xyz_lib=None,
        additional_charge=cluster_charge,
    )

    # ----- 4. Mongo credentials ---------------------------------------------
    host, port, dbname = _parse_mongodb_uri(mongodb_uri)
    log.info("mongo: host=%s port=%d dbname=%s", host, port, dbname)
    creds = _build_credentials(host, port, dbname)

    if not dry_run:
        connectable, mongo_msg = _check_mongo_connectable(creds)
        if not connectable:
            raise SystemExit(
                f"Cannot connect to MongoDB at {host}:{port}/{dbname}: {mongo_msg}\n"
                "Ensure mongod is running. Use `tools/scine_mongo.sh start` to "
                "launch a per-user mongod on the current node, or set --mongodb-uri "
                "to a known-good host."
            )
        log.info("mongo connectivity check: OK")

    # ----- 5. Run the SteeringWheel -----------------------------------------
    expl_result = run_steered_exploration(
        cluster=cluster,
        model=model,
        db_credentials=creds,
        site_filter=site_filter,
        max_bond_modifications=max_bond_modifications,
        barrier_cap_kcal=barrier_cap_kcal,
        max_depth=max_depth,
        central_metal=central_metal,
        cluster_charge=cluster_charge,
        cluster_multiplicity=cluster_multiplicity,
        workdir=out_dir / "wheel",
        restart_file=restart_file,
        dry_run=dry_run,
    )

    # ----- 6. Export top-N elementary steps as PDB --------------------------
    exported: List[Path] = []
    if not dry_run and expl_result.get("status") == "completed":
        exported = export_top_steps_as_pdb(
            db_credentials=creds,
            out_dir=out_dir / "exported_ts",
            top_n=top_n_export,
            template_pdb=input_pdb,
            atom_index_map=atom_index_map,
        )
    elif dry_run:
        log.info("dry_run=True: skipping export_top_steps_as_pdb")
    else:
        log.warning(
            "exploration status=%s; skipping export_top_steps_as_pdb",
            expl_result.get("status"),
        )

    # ----- 7. Optional MACE handoff hint ------------------------------------
    if validate_with:
        log.info(
            "validate_with=%s: rerun `cowboy-qc saddle` or `tools/polish_ts_v3.py` "
            "against each exported PDB next.",
            validate_with,
        )

    summary = {
        "status": "completed" if expl_result.get("status") == "completed" else expl_result.get("status"),
        "input_pdb": str(input_pdb),
        "out_dir": str(out_dir),
        "wall_seconds_total": time.perf_counter() - t_total_start,
        "backend": backend,
        "central_metal": central_metal,
        "cluster_charge": cluster_charge,
        "cluster_multiplicity": cluster_multiplicity,
        "cluster_n_atoms": int(cluster.size()),
        "max_bond_modifications": max_bond_modifications,
        "max_depth": max_depth,
        "barrier_cap_kcal": barrier_cap_kcal,
        "top_n_export": top_n_export,
        "mongodb": {"host": host, "port": port, "dbname": dbname},
        "exploration": expl_result,
        "exported_pdbs": [str(p) for p in exported],
        "validate_with": validate_with,
        "dry_run": dry_run,
    }
    summary_path = out_dir / "qcb_chemoton_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", summary_path)

    return {
        "status": summary["status"],
        "outputs": {
            "summary": str(summary_path),
            "wheel_workdir": str(out_dir / "wheel"),
            "exported_ts_dir": str(out_dir / "exported_ts"),
        },
        **summary,
    }
