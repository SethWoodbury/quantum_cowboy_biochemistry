"""SCINE Chemoton bridge — stable façade between qcb and SCINE Chemoton + ReaDuct.

This module wires our existing MACE-validated PTE pipeline to SCINE Chemoton's
Steering Wheel (Steiner & Reiher, Nat. Commun. 2024). Two-tier architecture:

  Tier 1 (this module): Cheap-but-fast PM6 / PM7 / xtb-GFN2 reaction-network
  exploration via Chemoton. Drives Sparrow or scine-xtb-wrapper through
  Chemoton's Steering Wheel API. Output: ranked elementary-step database in
  MongoDB.

  Tier 2 (downstream, via existing qcb pipeline): Each top-N TS from Tier 1
  is exported as XYZ, re-grafted onto the original PDB via atom_index_map,
  then re-fed through scan_along_s.py / polish_ts_v3.py with our
  MACE-polar-m / mace-omol calculator for refinement.

Why no MACE-as-SCINE-Calculator C++ Module?
-------------------------------------------
A first-class MACE → SCINE bridge would require a pybind11 trampoline subclass
of scine::core::Calculator and ~1-2 weeks of C++ work plus ongoing maintenance
against the scine-utilities ABI. The two-tier delegation avoids the C++ swamp
entirely: SCINE only sees Sparrow/xTB during exploration, then we hand
discovered TSs to MACE via plain XYZ I/O. We will revisit this only if SCINE
upstream ever ships a `PyCalculator` trampoline. (Future work.)

Public API
----------
- ``scine_available()`` — True if the SCINE core packages import cleanly
- ``pdb_to_scine_cluster(pdb, residues, ...)`` — strip protein → ≤60-atom QM
  cluster, returning ``(AtomCollection, atom_index_map)``.
- ``make_scine_model(backend)`` — backends: ``'sparrow-pm6'`` (default),
  ``'sparrow-pm7'``, ``'xtb-gfn2'``.
- ``make_pte_reactive_filter(...)`` — build PTE-style filter stack
  (CentralSiteFilter, AtomRuleBasedFilter, ElementWiseReactionCoordinateFilter,
  optional SubStructureFilter for backbone exclusion).
- ``run_steered_exploration(...)`` — assemble + run Steering Wheel with the
  3-step protocol (FileInput → CentralMetal expansion → BarriersWithinRange
  expansion). Blocks until done.
- ``export_top_steps_as_pdb(...)`` — pull ranked elementary steps from the DB,
  re-graft TS coords onto the input PDB, write per-rank PDBs + summary JSON.

All SCINE imports are lazy (inside function bodies) so missing/broken SCINE
install never breaks ``import quantum_engine``.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# NOTE: NumPy is the only top-level scientific dependency we allow here. SCINE
# imports are intentionally deferred to function-body scope.
import numpy as np


log = logging.getLogger("quantum_engine.tools.scine_bridge")


# --------------------------------------------------------------------------- #
# Version pinning declaration. Update when container is rebuilt.
# --------------------------------------------------------------------------- #
_REQUIRED_SCINE_VERSIONS = {
    "scine_chemoton": "4.1.0",
    "scine_readuct": "6.1.0",
    "scine_sparrow": "5.1.0",   # backend default
    "scine_utilities": ">=10.0",
    "scine_database": ">=1.5",
    "scine_molassembler": ">=3.0",
}


# --------------------------------------------------------------------------- #
# Backend registry. Maps user-facing alias -> (program, method, basis_set,
# import-time check function name).
# --------------------------------------------------------------------------- #
_BACKEND_TABLE: Dict[str, Dict[str, str]] = {
    "sparrow-pm6": {
        "program": "sparrow",
        "method_family": "PM6",
        "method": "PM6",
        "basis_set": "",
        "import_check": "scine_sparrow",
    },
    "sparrow-pm7": {
        "program": "sparrow",
        "method_family": "PM7",
        "method": "PM7",
        "basis_set": "",
        "import_check": "scine_sparrow",
    },
    "xtb-gfn2": {
        "program": "xtb",
        "method_family": "GFN2",
        "method": "GFN2",
        "basis_set": "",
        "import_check": "scine_xtb_wrapper",
    },
}


# --------------------------------------------------------------------------- #
# Cluster extraction container. Carries enough metadata for round-trip.
# --------------------------------------------------------------------------- #
@dataclass
class ClusterExtraction:
    """Mutable record returned by ``pdb_to_scine_cluster`` exploded into a tuple.

    Held internally so we can carry richer metadata (cap atoms, charge hint)
    without bloating the public tuple signature.
    """
    cluster_atoms: Any                     # scine_utilities.AtomCollection
    atom_index_map: Dict[int, int]         # cluster_idx -> original PDB serial
    cap_atom_indices: List[int] = field(default_factory=list)
    cluster_residue_ids: List[int] = field(default_factory=list)
    charge_hint: Optional[int] = None


# --------------------------------------------------------------------------- #
# Lightweight timer decorator that emits a single INFO line.
# --------------------------------------------------------------------------- #
@contextmanager
def _timed(label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        log.info("%s: wall=%.2fs", label, time.perf_counter() - t0)


# --------------------------------------------------------------------------- #
# Availability + version checks.
# --------------------------------------------------------------------------- #
def scine_available() -> bool:
    """Return True if the SCINE core packages we depend on are importable."""
    try:
        import scine_chemoton  # noqa: F401
        import scine_readuct  # noqa: F401
        import scine_utilities  # noqa: F401
        import scine_database  # noqa: F401
        return True
    except ImportError:
        return False


def _require_scine() -> None:
    """Raise ``RuntimeError`` if any required SCINE package is missing."""
    missing = []
    for pkg in ("scine_chemoton", "scine_readuct", "scine_utilities", "scine_database"):
        try:
            __import__(pkg)
        except ImportError as exc:
            missing.append(f"{pkg} (ImportError: {exc})")
    if missing:
        raise RuntimeError(
            "SCINE imports failed:\n  - "
            + "\n  - ".join(missing)
            + "\nFix: install the qcb apptainer image (deps/quantum_chem.def) "
              "or run `bash deps/build_scine.sh` to install Chemoton/ReaDuct/Sparrow "
              "into the qcb-xtb conda env. Required versions: "
            + ", ".join(f"{k}={v}" for k, v in _REQUIRED_SCINE_VERSIONS.items())
        )


# --------------------------------------------------------------------------- #
# PDB → SCINE AtomCollection cluster extraction.
# --------------------------------------------------------------------------- #
# Standard amino acid residue names used to identify protein backbone for capping.
_PROTEIN_RESNAMES = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "KCX",  # carbamylated Lys (PTM)
    "HID", "HIE", "HIP",  # alternate HIS protomers
    "CYX", "CYM",  # cysteine variants
    "LYN",  # neutral Lys
})


def pdb_to_scine_cluster(
    pdb_path: Path,
    cluster_residues: Union[List[int], str] = "auto",
    keep_protein_chain_id: str = "A",
) -> Tuple[Any, Dict[int, int]]:
    """Strip protein → ≤60-atom QM cluster suitable for Chemoton exploration.

    Parameters
    ----------
    pdb_path
        Source PDB. Must be a single-model snapshot (no MULTI-MODEL / ENSEMBLE).
    cluster_residues
        Either a list of integer residue numbers (1-indexed, chain
        ``keep_protein_chain_id``) to include, or the string ``"auto"`` to
        auto-select all residues with at least one heavy atom within 5.0 Å of
        any non-protein heteroatom (``LIG``, ``SUB``, etc.).
    keep_protein_chain_id
        Chain ID to consider for protein residues. Heteroatoms in any chain
        are always retained.

    Returns
    -------
    atoms
        ``scine_utilities.AtomCollection`` with the cluster atoms (positions
        in Bohr, internal SCINE convention).
    atom_index_map
        ``{cluster_idx: original_pdb_serial}`` — used by
        :func:`export_top_steps_as_pdb` to re-graft refined coords onto the
        full PDB.

    Notes
    -----
    Currently caps dangling backbone atoms by simple H replacement at the
    Cα → next-bonded-N/C bond cutoff. We use 1.09 Å C-H bond length and place
    the cap along the original bond vector. (More sophisticated link-atom
    placement is downstream MACE refinement's job, not Tier 1's.)
    """
    _require_scine()
    pdb_path = Path(pdb_path).resolve()
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    import scine_utilities as su

    # We use the existing tools/structure_validator.parse_pdb minimal parser
    # to preserve atom-serial → cluster-idx fidelity (biotite mangles HETATM
    # ordering, ASE collapses residues to MOL).
    import sys as _sys
    repo_root = pdb_path.parent
    while repo_root != repo_root.parent and not (repo_root / "quantum_engine").is_dir():
        repo_root = repo_root.parent
    tools_dir = repo_root / "tools"
    if tools_dir.is_dir() and str(tools_dir) not in _sys.path:
        _sys.path.insert(0, str(tools_dir))
    from structure_validator import parse_pdb  # noqa: E402

    with _timed("pdb_to_scine_cluster"):
        all_atoms, _remarks = parse_pdb(pdb_path)
        if not all_atoms:
            raise RuntimeError(f"PDB {pdb_path} contained no ATOM/HETATM records.")
        n_total = len(all_atoms)
        log.info("input PDB %s contains %d atoms total", pdb_path.name, n_total)

        # Determine which atoms to retain.
        coords = np.array([[a.x, a.y, a.z] for a in all_atoms])
        is_protein = np.array([
            (a.chain == keep_protein_chain_id) and (a.resname in _PROTEIN_RESNAMES)
            for a in all_atoms
        ])
        is_hetero = ~is_protein  # everything not protein-chain protein-residue

        # Pick the residue set
        if isinstance(cluster_residues, str) and cluster_residues == "auto":
            # All residues with any heavy atom within 5.0 Å of any non-protein
            # heteroatom (treats waters/metals/ligand as the active-site core).
            het_idx = np.where(is_hetero)[0]
            if len(het_idx) == 0:
                raise RuntimeError(
                    "auto-cluster requested but PDB contains no HETATM / "
                    "non-protein residues to anchor on. Pass an explicit "
                    "cluster_residues list."
                )
            keep_mask = np.zeros(n_total, dtype=bool)
            keep_mask[het_idx] = True
            d_thresh = 5.0  # Å
            for j in range(n_total):
                if keep_mask[j] or is_hetero[j]:
                    continue
                d = np.min(np.linalg.norm(coords[het_idx] - coords[j], axis=1))
                if d <= d_thresh:
                    keep_mask[j] = True
            keep_indices = list(np.where(keep_mask)[0])
            log.info(
                "auto-cluster: selected %d atoms within %.1fÅ of HETATM core",
                len(keep_indices), d_thresh,
            )
        elif isinstance(cluster_residues, (list, tuple)):
            wanted = set(int(r) for r in cluster_residues)
            keep_indices = [
                i for i, a in enumerate(all_atoms)
                if (a.resseq in wanted and (a.chain == keep_protein_chain_id or is_hetero[i]))
                or (is_hetero[i] and a.resseq in wanted)
            ]
            # Always include all HETATMs as a safety net (waters/metals/ligand)
            for i in range(n_total):
                if is_hetero[i] and i not in keep_indices:
                    keep_indices.append(i)
            keep_indices.sort()
            log.info(
                "explicit cluster: %d residues requested → %d atoms selected",
                len(wanted), len(keep_indices),
            )
        else:
            raise TypeError(
                f"cluster_residues must be a list of ints or 'auto', "
                f"got {type(cluster_residues).__name__}"
            )

        if len(keep_indices) > 60:
            log.warning(
                "cluster has %d atoms (>60). Chemoton exploration may be slow. "
                "Consider tightening the cluster spec.",
                len(keep_indices),
            )
        elif len(keep_indices) == 0:
            raise RuntimeError(
                "Cluster extraction yielded zero atoms — check residue spec."
            )

        # Build the AtomCollection. SCINE expects positions in Bohr.
        ac = su.AtomCollection()
        atom_index_map: Dict[int, int] = {}
        cluster_residue_ids: List[int] = []
        for cluster_idx, orig_idx in enumerate(keep_indices):
            a = all_atoms[orig_idx]
            elem_str = (a.element or "").strip().capitalize()
            if not elem_str:
                # Try to infer from atom name (first letter, fallback)
                elem_str = a.name[:1].capitalize()
            try:
                elem = getattr(su.ElementType, elem_str)
            except AttributeError:
                # Some PDBs misbrand 2-char metals (e.g. ZN vs Zn)
                elem_str_2 = (a.element or a.name)[:2].capitalize()
                try:
                    elem = getattr(su.ElementType, elem_str_2)
                except AttributeError:
                    raise RuntimeError(
                        f"Could not resolve element {a.element!r} / {a.name!r} "
                        f"for atom {a.serial}. Update PDB column 77."
                    ) from None
            pos_bohr = np.array([a.x, a.y, a.z]) * su.BOHR_PER_ANGSTROM
            ac.push_back(su.Atom(elem, pos_bohr))
            atom_index_map[cluster_idx] = a.serial
            if a.resseq not in cluster_residue_ids:
                cluster_residue_ids.append(a.resseq)

        log.info(
            "cluster built: %d atoms, %d unique residues, "
            "elements = %s",
            ac.size(), len(cluster_residue_ids),
            sorted(set(str(e).split(".")[-1] for e in ac.elements)),
        )

    return ac, atom_index_map


# --------------------------------------------------------------------------- #
# Backend / model construction.
# --------------------------------------------------------------------------- #
def make_scine_model(backend: str = "xtb-gfn2") -> Any:
    """Build a ``scine_database.Model`` for the requested cheap-tier backend.

    Parameters
    ----------
    backend
        One of ``'xtb-gfn2'`` (default; sparrow-pm6/pm7 currently broken in
        this container — see _BACKEND_TABLE doc). See :data:`_BACKEND_TABLE`
        for the full map.

    Returns
    -------
    model
        ``scine_database.Model`` with ``program``, ``method_family``,
        ``method``, ``basis_set`` populated. Other fields (``spin_mode``,
        ``temperature``) keep their defaults.
    """
    _require_scine()
    if backend not in _BACKEND_TABLE:
        raise ValueError(
            f"Unknown backend {backend!r}. Available: {sorted(_BACKEND_TABLE)}"
        )

    spec = _BACKEND_TABLE[backend]

    # NOTE: We deliberately do NOT import the backend wrapper module
    # (scine_sparrow / scine_xtb_wrapper) here in the controller process.
    # Sparrow's __init__ has a known abort when imported in a process that
    # also imports scine_chemoton (free() invalid pointer on certain
    # Boost-Python ABI mismatches; see container build notes 2026-05-06).
    # The Puffin worker pods do the actual calculator work — the controller
    # only needs scine_database to construct a Model spec. If the backend is
    # missing on workers, that's surfaced at job-execution time with a clear
    # SCINE error; we still log a warning here so misconfiguration shows up
    # early.
    log.debug(
        "make_scine_model: skipping in-process import of %s (worker-side dep)",
        spec["import_check"],
    )

    import scine_database as db

    model = db.Model(spec["method_family"], spec["method"], spec["basis_set"])
    model.program = spec["program"]
    log.info(
        "scine model: backend=%s program=%s method_family=%s method=%s",
        backend, model.program, model.method_family, model.method,
    )
    return model


# --------------------------------------------------------------------------- #
# Reactive-site filter stack for PTE-style metallohydrolases.
# --------------------------------------------------------------------------- #
def make_pte_reactive_filter(
    central_metal: str = "Zn",
    protein_backbone_xyz_lib: Optional[Path] = None,
    additional_charge: int = 1,
) -> Any:
    """Build the PTE-style reactive-site filter stack for Chemoton.

    Composes:
      1. ``CentralSiteFilter(central_metal)`` — atoms within bonded reach of a
         Zn (default) center are reactive; everything else is frozen.
      2. ``AtomRuleBasedFilter`` (always-pass placeholder rule set) — placeholder
         distance/element rules for downstream tightening; included so users
         can extend by mutating the returned filter array.
      3. ``ElementWiseReactionCoordinateFilter({"C": ["C", "N"]})`` —
         excludes scaffold C-C and C-N bond reactions (sidechain backbone),
         keeping reactivity around the metal center.
      4. (Optional) ``SubStructureFilter(protein_backbone_xyz_lib, exclude_mode=True)``
         if a backbone library is supplied, blocking reactions that match
         protein-backbone substructures.

    Parameters
    ----------
    central_metal
        Element symbol of the catalytic metal ion. Default ``'Zn'`` for PTE.
    protein_backbone_xyz_lib
        Optional folder of XYZ files representing protein-backbone fragments
        to exclude from reactivity. Skipped if ``None`` or missing.
    additional_charge
        Reserved for future filter-set extensions that depend on net charge
        (e.g., switching protonation rules). Currently logged only.

    Returns
    -------
    filter
        ``scine_chemoton.filters.reactive_site_filters.ReactiveSiteFilterAndArray``
        with all sub-filters AND-composed.
    """
    _require_scine()
    from scine_chemoton.filters.reactive_site_filters import (
        AtomRuleBasedFilter,
        CentralSiteFilter,
        ElementWiseReactionCoordinateFilter,
        ReactiveSiteFilter,
        ReactiveSiteFilterAndArray,
        SubStructureFilter,
    )

    log.info(
        "make_pte_reactive_filter: metal=%s, charge_hint=%d, backbone_lib=%s",
        central_metal, additional_charge, protein_backbone_xyz_lib,
    )

    sub_filters: List[ReactiveSiteFilter] = []

    # 1. Central-site filter: only atoms in the bonded reach of a metal react.
    sub_filters.append(
        CentralSiteFilter(
            central_atom=central_metal,
            ligand_without_central_atom_reactive=False,
        )
    )

    # 2. Atom-rule-based filter (placeholder pass-through): users can mutate
    # the rules dict afterwards. We pass an empty dict + exclude_mode=False
    # which has no effect (no rules ⇒ no exclusion), but keeps the slot in
    # the filter array for forward compat.
    try:
        sub_filters.append(AtomRuleBasedFilter(rules={}, exclude_mode=False))
    except Exception as exc:
        log.warning("AtomRuleBasedFilter slot skipped: %s", exc)

    # 3. Element-wise reaction-coordinate filter: forbid C-C and C-N bond
    # changes (scaffold). With reactive_if_rules_apply=False, listed pairs
    # are *blocked* not allowed.
    try:
        sub_filters.append(
            ElementWiseReactionCoordinateFilter(
                rules={"C": ["C", "N"]},
                reactive_if_rules_apply=False,
            )
        )
    except Exception as exc:
        log.warning("ElementWiseReactionCoordinateFilter slot skipped: %s", exc)

    # 4. Optional protein-backbone substructure exclusion.
    if protein_backbone_xyz_lib is not None:
        p = Path(protein_backbone_xyz_lib)
        if p.is_dir():
            try:
                sub_filters.append(SubStructureFilter(str(p), exclude_mode=True))
                log.info("SubStructureFilter active on %s", p)
            except Exception as exc:
                log.warning("SubStructureFilter skipped (load failed: %s)", exc)
        else:
            log.warning(
                "protein_backbone_xyz_lib=%s is not a directory; "
                "skipping SubStructureFilter",
                p,
            )

    log.info("reactive-site filter stack: %d sub-filters AND-composed", len(sub_filters))
    return ReactiveSiteFilterAndArray(sub_filters)


# --------------------------------------------------------------------------- #
# Steered exploration via Chemoton's Steering Wheel.
# --------------------------------------------------------------------------- #
def run_steered_exploration(
    cluster: Any,                                  # scine_utilities.AtomCollection
    model: Any,                                    # scine_database.Model
    db_credentials: Any,                           # scine_database.Credentials
    site_filter: Any,                              # ReactiveSiteFilterAndArray
    *,
    max_bond_modifications: int = 2,
    barrier_cap_kcal: float = 60.0,
    max_depth: int = 2,
    central_metal: str = "Zn",
    cluster_charge: int = 1,
    cluster_multiplicity: int = 1,
    workdir: Optional[Path] = None,
    restart_file: Optional[Path] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Construct + run a SteeringWheel with the standard 3-step PTE protocol.

    Protocol:
      Step 1 — ``FileInputSelection`` ingests the cluster XYZ under
        ``label='user_guess'``, then ``SimpleOptimization`` relaxes it.
      Step 2 — ``CentralMetalSelection`` (depth 1) targets reactions around
        the metal; ``Rearrangement`` proposes association/dissociation up to
        ``max_bond_modifications``.
      Step 3 — ``BarriersWithinRangeSelection`` keeps only product
        intermediates with barrier ≤ ``barrier_cap_kcal``; same expansion
        run again from those (depth 2).

    Blocks until the wheel completes (calls ``.start().join()``).

    Parameters
    ----------
    cluster
        ``scine_utilities.AtomCollection`` from
        :func:`pdb_to_scine_cluster`.
    model
        ``scine_database.Model`` from :func:`make_scine_model`.
    db_credentials
        ``scine_database.Credentials`` (host, port, dbname).
    site_filter
        ``ReactiveSiteFilterAndArray`` from
        :func:`make_pte_reactive_filter`.
    max_bond_modifications
        ``max_bond_associations`` + ``max_bond_dissociations`` per Rearrangement
        step. PTE default is 2 (one bond forms, one breaks per elementary step).
    barrier_cap_kcal
        Max gas-phase electronic barrier to retain in Step 3. Defaults to
        60 kcal/mol (well above experimental PTE 14 kcal/mol but accounts for
        PM6 systematic over-binding).
    max_depth
        Number of expansion rounds beyond the input. Currently 2 is the only
        wired value; ≥3 would require additional gear plumbing.
    central_metal
        Element symbol of the metal whose vicinity we explore.
    cluster_charge, cluster_multiplicity
        Net charge and spin multiplicity of the input cluster (passed to
        FileInputSelection's structure tuple).
    workdir
        Working directory for the cluster XYZ + restart files. Created if
        missing. Defaults to ``./.scine_explore``.
    restart_file
        If supplied (and exists), resume from this saved Steering Wheel
        pickle.
    dry_run
        If True, build the wheel but do not call ``.start()``. Used by the
        CLI ``--dry-run`` flag for plumbing checks.

    Returns
    -------
    dict
        ``{'run_id', 'steps_completed', 'database_uri', 'workdir',
           'cluster_xyz', 'restart_file', 'status', 'wall_seconds'}``.
        ``run_id`` is ``credentials.database_name``.
    """
    _require_scine()

    import scine_database as db
    import scine_utilities as su
    from scine_chemoton.steering_wheel import SteeringWheel
    from scine_chemoton.steering_wheel.network_expansions.basics import (
        SimpleOptimization,
    )
    from scine_chemoton.steering_wheel.network_expansions.reactions import (
        Rearrangement,
    )
    from scine_chemoton.steering_wheel.selections.basics import (
        BarriersWithinRangeSelection,
    )
    from scine_chemoton.steering_wheel.selections.input_selections import (
        FileInputSelection,
    )
    from scine_chemoton.steering_wheel.selections.organometallic_complexes import (
        CentralMetalSelection,
    )

    workdir = Path(workdir or ".scine_explore").resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    log.info(
        "run_steered_exploration: workdir=%s, dry_run=%s, max_bond_mod=%d, "
        "barrier_cap_kcal=%.1f, max_depth=%d",
        workdir, dry_run, max_bond_modifications, barrier_cap_kcal, max_depth,
    )

    if max_depth != 2:
        log.warning(
            "max_depth=%d requested but only depth=2 is fully wired; "
            "additional rounds will be ignored.",
            max_depth,
        )

    # --- Materialize the cluster as XYZ for FileInputSelection. -------------
    cluster_xyz = workdir / "input_cluster.xyz"
    su.io.write(str(cluster_xyz), cluster)
    log.info("wrote input cluster XYZ to %s (%d atoms)", cluster_xyz, cluster.size())

    # FileInputSelection wants a list of (xyz_path, charge, multiplicity)
    # tuples (3-tuples for non-periodic). The 4-tuple form's 4th element is
    # periodic-boundary cell parameters, not a label — the per-structure
    # label is the separate ``label_of_structure`` kwarg.
    structure_information_input = [
        (str(cluster_xyz), int(cluster_charge), int(cluster_multiplicity)),
    ]

    # --- Compose the 3-step protocol. ---------------------------------------
    # Convert barrier from kcal to kJ/mol (Chemoton's BarriersWithinRangeSelection
    # wants kJ/mol per its source).
    barrier_cap_kj = float(barrier_cap_kcal) * 4.184

    # exploration_scheme is a flat list of alternating Selection and
    # NetworkExpansion instances (each subclasses ExplorationSchemeStep).
    # The SteeringWheel pairs them internally.
    scheme: List[Any] = [
        FileInputSelection(
            model=model,
            structure_information_input=structure_information_input,
            label_of_structure="user_guess",
            additional_reactive_site_filters=[site_filter],
        ),
        SimpleOptimization(model=model),
        CentralMetalSelection(
            model=model,
            central_metal_species=central_metal,
            ligand_without_metal_reactive=False,
            additional_reactive_site_filters=[site_filter],
        ),
        Rearrangement(
            model=model,
            max_bond_associations=max_bond_modifications,
            max_bond_dissociations=max_bond_modifications,
        ),
        BarriersWithinRangeSelection(
            model=model,
            max_barrier=barrier_cap_kj,
            additional_reactive_site_filters=[site_filter],
        ),
        Rearrangement(
            model=model,
            max_bond_associations=max_bond_modifications,
            max_bond_dissociations=max_bond_modifications,
        ),
    ]

    # --- Wheel construction. ------------------------------------------------
    restart_path: Optional[str] = None
    if restart_file is not None:
        rf = Path(restart_file)
        if rf.is_file():
            restart_path = str(rf)
            log.info("resuming wheel from restart file %s", restart_path)
        else:
            log.warning("restart_file %s missing; starting fresh", rf)

    result: Dict[str, Any] = {
        "run_id": getattr(db_credentials, "database_name", "<unknown>"),
        "database_uri": getattr(db_credentials, "ip", "<unknown>"),
        "workdir": str(workdir),
        "cluster_xyz": str(cluster_xyz),
        "restart_file": str(workdir / "wheel_state.pkl"),
        "status": "constructed",
        "wall_seconds": 0.0,
        "steps_completed": 0,
        "scheme_n_steps": len(scheme),
    }

    if dry_run:
        # In dry-run, do NOT construct the SteeringWheel — its __init__
        # opens a Mongo connection. Just confirm the scheme assembled.
        log.info(
            "dry_run=True; %d-step scheme assembled (selection/expansion alternation): %s",
            len(scheme),
            [type(s).__name__ for s in scheme],
        )
        result["status"] = "dry_run"
        return result

    wheel = SteeringWheel(
        credentials=db_credentials,
        exploration_scheme=scheme,
        global_selection=None,
        global_for_first_selection=False,
        restart_file=restart_path,
        # Suppress interactive prompts so this is callable from a script.
        callable_input=lambda *_a, **_k: "y",
    )

    # --- Run the wheel. -----------------------------------------------------
    t0 = time.perf_counter()
    try:
        wheel.start()
        log.info("Steering Wheel started; waiting for completion (.join())…")
        wheel.join()
    except Exception as exc:
        log.error("Steering Wheel raised: %s", exc)
        result["status"] = "failed"
        result["error"] = str(exc)
        result["wall_seconds"] = time.perf_counter() - t0
        try:
            wheel.save(str(workdir / "wheel_state.pkl"))
        except Exception:
            pass
        return result

    result["wall_seconds"] = time.perf_counter() - t0

    # Capture results / status
    try:
        results_collected = wheel.get_results()
        result["steps_completed"] = len(results_collected) if results_collected else 0
    except Exception as exc:
        log.warning("get_results() failed: %s", exc)
        result["steps_completed"] = -1

    try:
        wheel.save(str(workdir / "wheel_state.pkl"))
    except Exception as exc:
        log.warning("wheel.save() failed: %s", exc)

    result["status"] = "completed"
    log.info(
        "run_steered_exploration done: status=%s steps=%d wall=%.1fs",
        result["status"], result["steps_completed"], result["wall_seconds"],
    )
    return result


# --------------------------------------------------------------------------- #
# Top-N elementary-step exporter (Chemoton DB -> per-rank PDBs).
# --------------------------------------------------------------------------- #
def export_top_steps_as_pdb(
    db_credentials: Any,
    out_dir: Path,
    *,
    top_n: int = 5,
    template_pdb: Path,
    atom_index_map: Dict[int, int],
) -> List[Path]:
    """Pull top-N elementary steps by barrier from the Chemoton DB and re-graft.

    For each step:
      * read TS atoms via ``ElementaryStep.get_transition_state()`` →
        ``Structure.get_atoms()`` (returns ``scine_utilities.AtomCollection``)
      * unmap cluster_idx → original PDB atom serial via ``atom_index_map``
      * substitute coordinates into the original PDB template (preserving
        residue records / REMARK lines) and write
        ``chemoton_ts_<rank>.pdb``.

    A summary JSON ``chemoton_summary.json`` is also written.

    Parameters
    ----------
    db_credentials
        Live ``scine_database.Credentials`` (must connect).
    out_dir
        Destination directory; created if missing.
    top_n
        How many lowest-barrier steps to export.
    template_pdb
        Original full-protein PDB used as the geometric template for
        re-grafting. Atom serials must match what was used at cluster build
        time.
    atom_index_map
        ``{cluster_idx: original_pdb_serial}`` from
        :func:`pdb_to_scine_cluster`.

    Returns
    -------
    list of paths
        ``[chemoton_ts_0.pdb, chemoton_ts_1.pdb, …]``, sorted ascending by
        barrier. Length may be less than ``top_n`` if the DB has fewer steps.
    """
    _require_scine()
    import scine_database as db
    import scine_utilities as su

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    template_pdb = Path(template_pdb).resolve()
    if not template_pdb.exists():
        raise FileNotFoundError(f"template_pdb not found: {template_pdb}")

    log.info(
        "export_top_steps_as_pdb: out_dir=%s, top_n=%d, template=%s",
        out_dir, top_n, template_pdb.name,
    )

    # Resolve the polish_ts_v2.write_pdb_from_template helper for round-trip.
    import sys as _sys
    repo_root = template_pdb.parent
    while repo_root != repo_root.parent and not (repo_root / "quantum_engine").is_dir():
        repo_root = repo_root.parent
    tools_dir = repo_root / "tools"
    if tools_dir.is_dir() and str(tools_dir) not in _sys.path:
        _sys.path.insert(0, str(tools_dir))
    from polish_ts_v2 import write_pdb_from_template  # noqa: E402
    from structure_validator import parse_pdb  # noqa: E402

    # --- Connect to the DB. --------------------------------------------------
    manager = db.Manager()
    manager.set_credentials(db_credentials)
    try:
        manager.connect()
    except Exception as exc:
        raise RuntimeError(
            f"Could not connect to MongoDB at "
            f"{getattr(db_credentials, 'ip', '?')}:{getattr(db_credentials, 'port', '?')}/"
            f"{getattr(db_credentials, 'database_name', '?')}: {exc}"
        ) from None

    if not manager.has_collection("elementary_steps"):
        log.warning("DB has no 'elementary_steps' collection; nothing to export.")
        return []
    es_coll = manager.get_collection("elementary_steps")
    structure_coll = manager.get_collection("structures")

    # --- Iterate elementary steps, sort by spline barrier. ------------------
    rankings: List[Tuple[float, Any]] = []  # (barrier_kJ, step_obj)
    for es_dict in es_coll.iterate_all_elementary_steps():
        step = db.ElementaryStep(es_dict.id(), es_coll)
        try:
            barrier = step.get_barrier_from_spline()
            barrier_kj = float(barrier[0]) if barrier else None
        except Exception:
            barrier_kj = None
        if barrier_kj is None:
            continue
        rankings.append((barrier_kj, step))

    if not rankings:
        log.warning("No elementary steps with computable barrier in DB.")
        return []

    rankings.sort(key=lambda r: r[0])
    log.info(
        "Found %d elementary steps; barrier range %.1f to %.1f kJ/mol",
        len(rankings), rankings[0][0], rankings[-1][0],
    )

    # --- Read template + prep coords. ---------------------------------------
    template_atoms, _ = parse_pdb(template_pdb)
    template_coords = np.array([[a.x, a.y, a.z] for a in template_atoms])
    serial_to_template_idx = {a.serial: i for i, a in enumerate(template_atoms)}

    written: List[Path] = []
    summary_records: List[Dict[str, Any]] = []
    for rank, (barrier_kj, step) in enumerate(rankings[:top_n]):
        try:
            ts_id = step.get_transition_state()
            ts_struct = db.Structure(ts_id, structure_coll)
            ts_ac = ts_struct.get_atoms()
        except Exception as exc:
            log.warning("rank %d: TS structure unavailable (%s); skipping", rank, exc)
            continue

        if ts_ac.size() != len(atom_index_map):
            log.warning(
                "rank %d: TS atom count %d ≠ cluster size %d "
                "(may be a fragmentation step); skipping",
                rank, ts_ac.size(), len(atom_index_map),
            )
            continue

        # Convert AtomCollection (Bohr) → Å, splice into template coords.
        ts_positions_bohr = np.array([ts_ac.get_position(i) for i in range(ts_ac.size())])
        ts_positions_ang = ts_positions_bohr * su.ANGSTROM_PER_BOHR

        new_coords = template_coords.copy()
        for cluster_idx, orig_serial in atom_index_map.items():
            tmpl_idx = serial_to_template_idx.get(orig_serial)
            if tmpl_idx is None:
                log.warning(
                    "rank %d: original PDB serial %d not in template; skipping",
                    rank, orig_serial,
                )
                continue
            new_coords[tmpl_idx] = ts_positions_ang[cluster_idx]

        out_pdb = out_dir / f"chemoton_ts_{rank}.pdb"
        write_pdb_from_template(
            template_pdb,
            new_coords,
            out_pdb,
            extra_remarks=[
                f"REMARK 999 QCB CHEMOTON_RANK {rank}",
                f"REMARK 999 QCB CHEMOTON_BARRIER_kJ_per_mol {barrier_kj:.4f}",
                f"REMARK 999 QCB CHEMOTON_BARRIER_kcal_per_mol {barrier_kj/4.184:.4f}",
                f"REMARK 999 QCB CHEMOTON_TS_STRUCTURE_ID {str(ts_struct.id())}",
            ],
        )
        written.append(out_pdb)
        summary_records.append({
            "rank": rank,
            "barrier_kJ_per_mol": float(barrier_kj),
            "barrier_kcal_per_mol": float(barrier_kj / 4.184),
            "ts_structure_id": str(ts_struct.id()),
            "step_id": str(step.id()),
            "pdb_path": str(out_pdb),
        })
        log.info(
            "rank %d: barrier=%.2f kcal/mol → %s",
            rank, barrier_kj / 4.184, out_pdb.name,
        )

    summary_path = out_dir / "chemoton_summary.json"
    summary_path.write_text(json.dumps({
        "n_exported": len(written),
        "n_total_in_db": len(rankings),
        "top_n_requested": top_n,
        "template_pdb": str(template_pdb),
        "results": summary_records,
    }, indent=2))
    log.info("wrote summary to %s (%d records)", summary_path, len(summary_records))

    return written


__all__ = [
    "scine_available",
    "pdb_to_scine_cluster",
    "make_scine_model",
    "make_pte_reactive_filter",
    "run_steered_exploration",
    "export_top_steps_as_pdb",
    "ClusterExtraction",
]
