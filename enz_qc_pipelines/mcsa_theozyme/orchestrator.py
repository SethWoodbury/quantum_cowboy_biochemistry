"""M-CSA-driven theozyme pipeline — 9 Step classes + builder.

Maximally exploits M-CSA annotation. Stages 0-3 implemented; Stages
4-8 still stubbed pending the M-CSA 159 (PTE) dry run.

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

import gzip
import io
import json
import logging
import re
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantum_engine.pipelines import Context, Pipeline, Step, StepResult

log = logging.getLogger("enz_qc_pipelines.mcsa_theozyme")


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
        from quantum_engine.data.chebi import lookup_smiles
        entry = ctx.metadata.get("mcsa_entry")
        if entry is None:
            raise RuntimeError(
                "ResolveSubstrateSMILES requires ctx.metadata['mcsa_entry'] — "
                "run FetchMCSAEntry first."
            )

        reactant_pieces: list[str] = []
        product_pieces: list[str] = []
        unresolved: list[int] = []

        for c in entry.compounds:
            ctype = (c.get("type") or "").lower()
            chebi_id = c.get("chebi_id")
            name = c.get("name") or ""
            smi = None
            if chebi_id is not None:
                try:
                    smi = lookup_smiles(int(chebi_id))
                except Exception as e:
                    log.warning(f"ChEBI lookup failed for {chebi_id}: {e}")
            if smi is None:
                unresolved.append(chebi_id)
                log.info(f"  unresolved compound: chebi={chebi_id} name={name!r}")
                continue
            # ChEBI represents schematic substrates with `*` wildcard atoms
            # (R-groups). These can't go straight into autodE / SCINE — the
            # user must supply a concrete substrate. Flag and skip.
            if "*" in smi:
                unresolved.append(chebi_id)
                log.info(f"  R-group SMILES from ChEBI:{chebi_id} ({name!r}): "
                         f"{smi} — needs user concrete substrate")
                continue
            # M-CSA labels reactants/substrates as 'reactant' (sometimes
            # 'substrate'); products as 'product'. Anything else (cofactors,
            # waters not modelled here) we skip.
            if "react" in ctype or "substr" in ctype:
                reactant_pieces.append(smi)
            elif "prod" in ctype:
                product_pieces.append(smi)
            else:
                log.debug(f"  skipping compound type {ctype!r}: {name}")

        # User overrides take precedence
        reactant_smiles = self.user_substrate or ".".join(reactant_pieces)
        product_smiles = self.user_product or ".".join(product_pieces)

        if not reactant_smiles or not product_smiles:
            raise RuntimeError(
                f"Could not resolve {('reactant' if not reactant_smiles else 'product')} "
                f"SMILES from M-CSA entry {entry.mcsa_id}. "
                f"Pass --substrate / --product on the CLI to override. "
                f"Unresolved ChEBI IDs: {unresolved}"
            )

        ctx.metadata["reactant_smiles"] = reactant_smiles
        ctx.metadata["product_smiles"] = product_smiles
        return StepResult(
            name=self.name,
            atoms=ctx.atoms,
            outputs={
                "reactant_smiles": reactant_smiles,
                "product_smiles": product_smiles,
                "unresolved_chebi_ids": unresolved,
                "n_user_overrides": int(bool(self.user_substrate))
                                     + int(bool(self.user_product)),
            },
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
        entry = ctx.metadata.get("mcsa_entry")
        if entry is None:
            raise RuntimeError("CropActiveSiteFromPDB: run FetchMCSAEntry first.")
        if not entry.reference_pdb:
            raise RuntimeError(
                f"M-CSA entry {entry.mcsa_id} has no reference PDB — cannot crop."
            )

        outdir = Path(self.workdir) if self.workdir else (ctx.outdir / self.name)
        outdir.mkdir(parents=True, exist_ok=True)

        full_pdb = _resolve_pdb(entry.reference_pdb, outdir)
        # Filter for the catalytic residue set; record the active set on
        # ctx.metadata so Stage 3 can extend it.
        catalytic_keys = {
            (r.chain, r.auth_seq) for r in entry.catalytic_residues
        }
        ctx.metadata["catalytic_keys"] = catalytic_keys
        ctx.metadata["active_residue_keys"] = set(catalytic_keys)
        ctx.metadata["full_pdb_path"] = str(full_pdb)

        cropped = outdir / f"active_site_{entry.reference_pdb}_tier1.pdb"
        n_atoms, cofactors_seen = _crop_pdb(
            full_pdb, cropped,
            keep_residue_keys=catalytic_keys,
            add_waters_within_A=self.add_waters_within_A,
            include_cofactor_metals=True,
        )

        # Load as ASE Atoms — the rest of the pipeline lives in ASE.
        from ase.io import read as ase_read
        ctx.atoms = ase_read(str(cropped))
        ctx.metadata["cropped_pdb_tier1"] = str(cropped)
        ctx.metadata["observed_cofactors"] = cofactors_seen

        return StepResult(
            name=self.name,
            atoms=ctx.atoms,
            outputs={
                "cropped_pdb": str(cropped),
                "n_atoms": n_atoms,
                "n_residues": len(catalytic_keys),
                "cofactors": cofactors_seen,
                "reference_pdb": entry.reference_pdb,
            },
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

        entry = ctx.metadata.get("mcsa_entry")
        full_pdb_path = ctx.metadata.get("full_pdb_path")
        catalytic_keys: set[tuple[str, int]] = ctx.metadata.get(
            "catalytic_keys", set())
        if entry is None or full_pdb_path is None or not catalytic_keys:
            raise RuntimeError(
                "Tier2ResidueExpansion: run FetchMCSAEntry + "
                "CropActiveSiteFromPDB first."
            )

        added_distance: set[tuple[str, int]] = set()
        added_motif: set[tuple[str, int]] = set()

        if self.mode in ("distance", "both"):
            added_distance = _residues_within_radius(
                Path(full_pdb_path), catalytic_keys, radius_A=self.radius_A,
            )
        if self.mode in ("motif", "both"):
            added_motif = _motif_fill(
                Path(full_pdb_path), catalytic_keys,
                max_gap=4,         # HExxH = 5-residue motif
            )

        added = (added_distance | added_motif) - catalytic_keys
        active = set(catalytic_keys) | added
        ctx.metadata["active_residue_keys"] = active

        # Re-crop with the expanded residue set.
        outdir = Path(self.workdir) if self.workdir else (ctx.outdir / self.name)
        outdir.mkdir(parents=True, exist_ok=True)
        cropped = outdir / (
            f"active_site_{entry.reference_pdb}_tier2_{self.mode}.pdb")
        n_atoms, cofactors_seen = _crop_pdb(
            Path(full_pdb_path), cropped,
            keep_residue_keys=active,
            add_waters_within_A=4.0,
            include_cofactor_metals=True,
        )
        from ase.io import read as ase_read
        ctx.atoms = ase_read(str(cropped))
        ctx.metadata["cropped_pdb_tier2"] = str(cropped)

        return StepResult(
            name=self.name,
            atoms=ctx.atoms,
            outputs={
                "cropped_pdb": str(cropped),
                "n_atoms": n_atoms,
                "n_residues_total": len(active),
                "added_residues": sorted(added),
                "n_added_distance": len(added_distance - catalytic_keys),
                "n_added_motif": len(added_motif - catalytic_keys),
            },
        )


# ─────────────────────────────────────────────────────────────────────
# PDB I/O helpers (used by Stages 2 and 3)
# ─────────────────────────────────────────────────────────────────────

def _resolve_pdb(pdb_id: str, cache_dir: Path) -> Path:
    """Resolve a PDB structure file. Tries, in order:
      1. site.PDB_MIRROR/<2-char hash>/pdb<id>.ent.gz (DIGS standard)
      2. site.PDB_MIRROR/<2-char hash>/<id>.pdb.gz
      3. RCSB HTTP fetch + cache to ``cache_dir``
    Returns a path to a plaintext .pdb file (decompresses .gz).
    """
    from quantum_engine.site import PDB_MIRROR
    pdb_id = pdb_id.lower()
    hash2 = pdb_id[1:3]                    # standard 2-char hash bucket
    candidates = []
    if PDB_MIRROR:
        candidates += [
            Path(PDB_MIRROR) / hash2 / f"pdb{pdb_id}.ent.gz",
            Path(PDB_MIRROR) / hash2 / f"{pdb_id}.pdb.gz",
            Path(PDB_MIRROR) / hash2 / f"{pdb_id}.pdb",
        ]
    for c in candidates:
        if c.is_file():
            log.debug(f"PDB mirror hit: {c}")
            return _decompress_to_plaintext(c, cache_dir)

    # Fallback: HTTP fetch from RCSB
    cache_pdb = cache_dir / f"{pdb_id}.pdb"
    if cache_pdb.is_file():
        return cache_pdb
    log.info(f"PDB mirror miss for {pdb_id}; fetching from RCSB…")
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"
    gz = cache_dir / f"{pdb_id}.pdb.gz"
    urllib.request.urlretrieve(url, gz)
    return _decompress_to_plaintext(gz, cache_dir)


def _decompress_to_plaintext(src: Path, cache_dir: Path) -> Path:
    """Decompress a .gz to <cache_dir>/<basename>.pdb if needed.
    Returns the plaintext path."""
    if src.suffix != ".gz":
        return src
    base = src.name
    base = base[len("pdb"):] if base.startswith("pdb") else base
    base = base.replace(".ent.gz", ".pdb").replace(".pdb.gz", ".pdb")
    dest = cache_dir / base
    if not dest.is_file():
        with gzip.open(src, "rb") as fin, dest.open("wb") as fout:
            shutil.copyfileobj(fin, fout)
    return dest


# ATOM record column layout (PDB v3.30):
# cols 1-6 record (ATOM/HETATM)
# cols 13-16 atom name; col 17 altLoc; cols 18-20 resName
# col 22 chainID; cols 23-26 resSeq; col 27 insertion code
# cols 31-38 x; 39-46 y; 47-54 z

def _parse_atom_record(line: str) -> tuple[str, str, str, int, float, float, float, str] | None:
    """Return (record, atom_name, res_name, resseq, x, y, z, chain) or None."""
    rec = line[:6].strip()
    if rec not in ("ATOM", "HETATM"):
        return None
    try:
        atom_name = line[12:16].strip()
        res_name = line[17:20].strip()
        chain = line[21:22].strip()
        resseq = int(line[22:26])
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        return rec, atom_name, res_name, resseq, x, y, z, chain
    except (ValueError, IndexError):
        return None


def _crop_pdb(
    src_pdb: Path,
    dst_pdb: Path,
    *,
    keep_residue_keys: set[tuple[str, int]],
    add_waters_within_A: float = 4.0,
    include_cofactor_metals: bool = True,
) -> tuple[int, list[str]]:
    """Write a subset of ATOM/HETATM records keeping only residues whose
    ``(chain, auth_seq)`` is in ``keep_residue_keys``. Adds:
      * cofactor metals (HETATM in METAL_CODES) anywhere — they're
        small and tend to be the catalytic centre.
      * waters whose oxygen is within ``add_waters_within_A`` Å of any
        kept atom.

    Returns ``(n_atoms_written, cofactor_codes_seen)``.
    """
    from quantum_engine.data.mcsa import METAL_CODES
    import numpy as np

    kept_lines: list[str] = []
    kept_coords: list[tuple[float, float, float]] = []
    cofactor_lines: list[tuple[str, str]] = []     # (line, code)
    water_lines: list[tuple[str, tuple[float, float, float]]] = []

    with src_pdb.open() as fh:
        for line in fh:
            rec = _parse_atom_record(line)
            if rec is None:
                continue
            kind, aname, resname, resseq, x, y, z, chain = rec
            key = (chain, resseq)
            if key in keep_residue_keys:
                kept_lines.append(line)
                kept_coords.append((x, y, z))
                continue
            if include_cofactor_metals and resname in METAL_CODES:
                cofactor_lines.append((line, resname))
                continue
            if resname in ("HOH", "WAT") and aname in ("O", "OW"):
                water_lines.append((line, (x, y, z)))

    cofactor_codes_seen = sorted({code for _, code in cofactor_lines})
    # Add cofactor metal atoms to kept coords so waters can be selected
    # relative to them too.
    for line, _ in cofactor_lines:
        rec = _parse_atom_record(line)
        if rec is not None:
            kept_coords.append(rec[4:7])

    # Filter waters by shell distance
    if water_lines and kept_coords:
        anchor = np.array(kept_coords)
        kept_water_lines: list[str] = []
        r2 = add_waters_within_A * add_waters_within_A
        for line, (x, y, z) in water_lines:
            d2 = ((anchor - np.array([x, y, z])) ** 2).sum(axis=1).min()
            if d2 <= r2:
                kept_water_lines.append(line)
    else:
        kept_water_lines = []

    n_atoms = len(kept_lines) + len(cofactor_lines) + len(kept_water_lines)
    with dst_pdb.open("w") as out:
        out.write("HEADER  cropped active site (qcb mcsa_theozyme)\n")
        for line in kept_lines:
            out.write(line)
        for line, _ in cofactor_lines:
            out.write(line)
        for line in kept_water_lines:
            out.write(line)
        out.write("END\n")

    log.info(
        f"Cropped {src_pdb.name} → {dst_pdb.name}: "
        f"{len(kept_lines)} catalytic atoms, {len(cofactor_lines)} cofactors "
        f"({cofactor_codes_seen}), {len(kept_water_lines)} shell waters."
    )
    return n_atoms, cofactor_codes_seen


def _residues_within_radius(
    src_pdb: Path,
    catalytic_keys: set[tuple[str, int]],
    *,
    radius_A: float,
) -> set[tuple[str, int]]:
    """Find all residues with any heavy atom within ``radius_A`` of any
    catalytic atom. Pure-PDB-text scan — no biotite dependency."""
    import numpy as np

    catalytic_coords: list[tuple[float, float, float]] = []
    other_atoms: list[tuple[tuple[str, int], tuple[float, float, float]]] = []

    with src_pdb.open() as fh:
        for line in fh:
            rec = _parse_atom_record(line)
            if rec is None:
                continue
            kind, aname, resname, resseq, x, y, z, chain = rec
            if aname.startswith("H"):
                continue                                # heavy-atom only
            key = (chain, resseq)
            if key in catalytic_keys:
                catalytic_coords.append((x, y, z))
            else:
                other_atoms.append((key, (x, y, z)))

    if not catalytic_coords:
        return set()
    anchor = np.array(catalytic_coords)
    r2 = radius_A * radius_A
    found: set[tuple[str, int]] = set()
    for key, xyz in other_atoms:
        d2 = ((anchor - np.array(xyz)) ** 2).sum(axis=1).min()
        if d2 <= r2:
            found.add(key)
    return found


def _motif_fill(
    src_pdb: Path,
    catalytic_keys: set[tuple[str, int]],
    *,
    max_gap: int = 4,
) -> set[tuple[str, int]]:
    """Fill gap residues between adjacent catalytic residues on the
    same chain — captures motifs like HExxH where flanking residues
    contribute to coordination geometry without being in M-CSA's
    minimal catalytic set.

    Strategy: group catalytic residues by chain. Within each chain,
    sort by seq num. For every adjacent pair within ``max_gap``
    residues, add the gap residues to the result.
    """
    by_chain: dict[str, list[int]] = {}
    for chain, seq in catalytic_keys:
        by_chain.setdefault(chain, []).append(seq)

    filled: set[tuple[str, int]] = set()
    for chain, seqs in by_chain.items():
        seqs = sorted(seqs)
        for a, b in zip(seqs, seqs[1:]):
            gap = b - a - 1
            if 1 <= gap <= max_gap:
                for s in range(a + 1, b):
                    filled.add((chain, s))
    return filled


# ─────────────────────────────────────────────────────────────────────
# Stage 4 — Per mechanism step, vacuum TS
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PerStepVacuumTS:
    """Vacuum TS for the overall reaction.

    Two regimes:
      * **overall=True** (default for now): build ONE vacuum TS for
        reactant_smiles → product_smiles using the swappable adapter
        from :mod:`enz_qc_pipelines.enzyme_ts_design.orchestrator`. This
        is the realistic flow when the user has already supplied
        concrete substrate SMILES and the M-CSA mechanism arrow-pushing
        is too generic to drive per-step coords (PTE 159 is exactly
        this case — Marvin XML is empty).
      * **overall=False** (TODO): iterate over
        ``entry.mechanisms[*].steps[*]``, parse marvin_xml for
        arrow-pushing → driving coords (via
        :func:`quantum_engine.data.mcsa.parse_marvin_xml`), and run
        SE-GSM per step (see :mod:`quantum_engine.qm.pygsm`). Stores
        per-step results under ctx.metadata['mechanism_results']
        keyed by (mechanism_idx, step_idx). Lands once we have a M-CSA
        entry with non-empty marvin_xml to test against (159 doesn't,
        641/900 do).
    """
    name: str = "per_step_vacuum_ts"
    tool: str = "autode"             # "autode" | "scine" | "molecularGSM"
    qm_method: str = "g-xtb"
    overall: bool = True
    workdir: str | Path | None = None

    def run(self, ctx: Context) -> StepResult:
        if not self.overall:
            raise NotImplementedError(
                "PerStepVacuumTS(overall=False): per-step iteration over "
                "entry.mechanisms[*].steps[*] needs the Marvin XML arrow "
                "parser to be wired (quantum_engine.data.mcsa.parse_marvin_xml "
                "is currently a stub). Use overall=True for now."
            )
        # Delegate to the generic vacuum-TS step from enzyme_ts_design.
        # That step handles tool dispatch (autode | scine | molecularGSM)
        # and the brittle autodE config / monkey-patching.
        from enz_qc_pipelines.enzyme_ts_design.orchestrator import (
            VacuumTSSearch,
        )
        reactant = ctx.metadata.get("reactant_smiles")
        product = ctx.metadata.get("product_smiles")
        if not reactant or not product:
            raise RuntimeError(
                "PerStepVacuumTS: ctx.metadata is missing reactant_smiles / "
                "product_smiles. Run ResolveSubstrateSMILES first."
            )

        sub_outdir = (Path(self.workdir) if self.workdir
                      else (ctx.outdir / self.name))
        sub_outdir.mkdir(parents=True, exist_ok=True)

        # VacuumTSSearch reads R/P from ctx.metadata — already populated
        # by Stage 1. The stage just needs ctx.outdir to point at our
        # subdir so its scratch files land cleanly. Save and restore.
        original_outdir = ctx.outdir
        ctx.outdir = sub_outdir
        try:
            sub_step = VacuumTSSearch(tool=self.tool, qm_method=self.qm_method,
                                      workdir=sub_outdir)
            sub_result = sub_step.run(ctx)
        finally:
            ctx.outdir = original_outdir

        # Persist the per-step result on metadata for downstream stages
        # (refine, path-refind, polish). Keyed under 'overall' since we
        # didn't iterate per mechanism step.
        ctx.metadata.setdefault("mechanism_results", {})["overall"] = {
            "ts_atoms": ctx.atoms,
            "outputs": dict(sub_result.outputs),
        }

        return StepResult(
            name=self.name,
            atoms=ctx.atoms,
            outputs={
                "mode": "overall",
                "delegated_tool": self.tool,
                **sub_result.outputs,
            },
        )


# ─────────────────────────────────────────────────────────────────────
# Stage 5 — Iterative refinement with PTM topology
# ─────────────────────────────────────────────────────────────────────

@dataclass
class IterativeRefineWithPTMs:
    """Sidechain + ligand co-optimisation, CA fixed.

    Two-step refinement of the cropped active-site cluster:
      1. **g-xTB pre-relax** (cheap; declashes the cropped structure).
         Always runs — g-xTB is in qcb-xtb / quantum_chem.sif by default.
      2. **MLFF relax** (MACE-POLAR-1M / OMOL fallback). Skipped with
         a warning if the MLFF can't be loaded in this Python env
         (MACE-POLAR needs gbg222's venv; OMOL needs mace-torch which
         is in the apptainer container but not the host conda env).

    CA atoms (alpha carbons of every protein residue) are fixed during
    both steps to preserve the fold of the chopped cluster. Ligand,
    cofactor metals, and waters stay flexible.

    PTM residues already carry their PTM atom set in the cropped PDB
    (we sliced by chain/seq, so KCX comes through as "KCX" with its
    carboxylate carbon attached). For now we don't apply any bespoke
    PTM force-field topology — the MLFF treats them as their atoms +
    bonds + charges, which is correct in vacuum. Classical-FF + bespoke
    KCX parameters become relevant only when we move to OpenMM-driven
    QM/MM in Stage 6.
    """
    name: str = "iterative_refine"
    constraint_mode: str = "ca-only"
    mace_model: str = "mace-polar"        # falls back to mace-omol if POLAR not loadable
    fmax_pre_relax: float = 0.1           # g-xTB target
    fmax_polish: float = 0.05             # MLFF target
    max_steps: int = 200
    # When True, refine the tier-1 (catalytic-only) cluster instead of
    # tier-2 (catalytic + neighbours). Tier-1 is ~50-150 atoms vs
    # tier-2 ~300-700 — xTB is intractably slow on tier-2 and segfaults
    # often (orphan-residue boundary issues). MLFF can handle tier-2;
    # when MACE is available, set prefer_tier1=False.
    prefer_tier1: bool = True
    # Hard cap on cluster size before we skip xTB pre-relax. xTB GFN2
    # is O(N^3) and segfaults intermittently above ~250 atoms with
    # cropped (broken peptide bond) systems.
    max_atoms_for_xtb: int = 200
    workdir: str | Path | None = None

    def run(self, ctx: Context) -> StepResult:
        outdir = Path(self.workdir) if self.workdir else (ctx.outdir / self.name)
        outdir.mkdir(parents=True, exist_ok=True)

        # Choose tier-1 vs tier-2 input. Tier-1 is much smaller and
        # avoids xTB blowing up on cropped boundaries.
        from ase.io import read as ase_read
        atoms = None
        if self.prefer_tier1 and ctx.metadata.get("cropped_pdb_tier1"):
            atoms = ase_read(str(ctx.metadata["cropped_pdb_tier1"]))
            log.info(f"  IterativeRefine: loading tier-1 cluster "
                     f"({ctx.metadata['cropped_pdb_tier1']})")
        elif ctx.atoms is not None:
            atoms = ctx.atoms.copy()
        else:
            raise RuntimeError(
                "IterativeRefineWithPTMs: ctx.atoms is None and no tier-1 PDB "
                "in ctx.metadata. Run Stage 2/3 first."
            )
        n_atoms = len(atoms)
        log.info(f"  IterativeRefine: {n_atoms} atoms, "
                 f"mace_model={self.mace_model!r}, "
                 f"max_atoms_for_xtb={self.max_atoms_for_xtb}")

        # ── Build CA-fix constraint ──────────────────────────────────
        # ASE Atoms.read('xxx.pdb') puts atom names into the .info dict
        # entries 'atomtypes' / 'arrays' depending on backend. The PDB
        # coordinate text has the atom name in cols 13-16; biotite-parsed
        # PDBs often surface it via atoms.arrays.get('atomtypes'). For
        # robustness we re-parse the source PDB and match by index.
        ca_indices = _ca_indices_from_pdb(ctx.metadata.get(
            "cropped_pdb_tier2", ctx.metadata.get("cropped_pdb_tier1", "")))
        from ase.constraints import FixAtoms
        if ca_indices and len(ca_indices) <= n_atoms:
            atoms.set_constraint(FixAtoms(indices=ca_indices))
            log.info(f"    CA-fix constraint: {len(ca_indices)} atoms held.")
        else:
            log.warning(
                f"    No CA atoms identified — refining unconstrained "
                f"(may drift). idx_count={len(ca_indices)} vs n_atoms={n_atoms}."
            )

        # Compute net charge from ligand SMILES if available — keeps
        # xtb / MACE charge-aware backends honest.
        net_charge = _infer_net_charge_from_smiles(
            ctx.metadata.get("reactant_smiles", ""))
        log.info(f"    inferred net_charge: {net_charge}")

        # ── 1. xTB GFN2 pre-relax ───────────────────────────────────
        max_force_pre: float | None = None
        max_force_post_xtb: float | None = None
        if n_atoms > self.max_atoms_for_xtb:
            log.warning(
                f"    skipping xTB pre-relax: {n_atoms} atoms > cap "
                f"{self.max_atoms_for_xtb} (xTB GFN2 is O(N^3) and "
                f"segfault-prone on cropped boundaries); going straight "
                f"to MLFF polish."
            )
        else:
            try:
                atoms, max_force_pre, max_force_post_xtb = _gxtb_relax(
                    atoms, outdir, fmax=self.fmax_pre_relax,
                    max_steps=self.max_steps, charge=net_charge,
                )
            except Exception as e:
                log.warning(
                    f"    xTB pre-relax failed ({type(e).__name__}: {e}); "
                    "skipping to MLFF polish on the un-relaxed cluster.")

        # ── 2. MLFF polish ────────────────────────────────────────────
        model_used: str = "(no MLFF available)"
        max_force_post_mlff: float | None = None
        try:
            atoms, model_used, max_force_post_mlff = _mlff_polish(
                atoms, outdir, model=self.mace_model,
                fmax=self.fmax_polish, max_steps=self.max_steps,
                charge=net_charge,
            )
        except Exception as e:
            log.warning(
                f"    MLFF polish failed ({type(e).__name__}: {e}); "
                "shipping g-xTB-relaxed cluster as 'refined'."
            )

        # ── Outputs ──────────────────────────────────────────────────
        refined_pdb = outdir / "refined.pdb"
        refined_xyz = outdir / "refined.xyz"
        from ase.io import write as ase_write
        try:
            ase_write(str(refined_pdb), atoms, format="proteindatabank")
        except Exception as e:
            log.warning(f"    PDB write failed: {e}")
            refined_pdb = None  # type: ignore[assignment]
        ase_write(str(refined_xyz), atoms, format="xyz")
        ctx.atoms = atoms
        ctx.metadata["refined_atoms"] = atoms

        return StepResult(
            name=self.name,
            atoms=atoms,
            outputs={
                "refined_pdb": str(refined_pdb) if refined_pdb else None,
                "refined_xyz": str(refined_xyz),
                "model_used": model_used,
                "constraint_mode": self.constraint_mode if ca_indices else "none",
                "n_ca_fixed": len(ca_indices),
                "max_force_pre": max_force_pre,
                "max_force_post_xtb": max_force_post_xtb,
                "max_force_post_mlff": max_force_post_mlff,
                "net_charge": net_charge,
            },
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
    """Emit AME-benchmark-compatible theozyme JSON + .cif.

    The output triplet for downstream consumers:
      * ``theozyme.cif`` — the cropped + refined active-site cluster as
        a CIF (PyMOL / ChimeraX ready). Atoms tagged with chain + seqnum
        carried over from the source PDB.
      * ``theozyme.json`` — metadata sidecar with M-CSA entry, EC,
        reference PDB, catalytic residues with role, cofactors observed,
        PTM flags, substrate / product SMILES, bond changes, barriers
        (when available), MLFF model used, plausibility flags.
      * ``review.pdb`` — alias for the refined PDB so PyMOL users can
        open the same coordinates without the CIF tooling.

    Plausibility flags are heuristic, not authoritative — they're meant
    to give a reviewer a quick "is this reasonable?" read before they
    decide whether to invest in a full QM/MM follow-up.
    """
    name: str = "write_theozyme"
    ame_format: bool = True

    def run(self, ctx: Context) -> StepResult:
        outdir = ctx.outdir / self.name
        outdir.mkdir(parents=True, exist_ok=True)

        # Pull every relevant piece off ctx.history + ctx.metadata
        entry = ctx.metadata.get("mcsa_entry")
        history = ctx.history

        cropped_pdb = (history.get("tier2_expansion", history.get("crop_active_site"))
                       and (
                           history["tier2_expansion"].outputs.get("cropped_pdb")
                           if "tier2_expansion" in history
                           else history["crop_active_site"].outputs.get("cropped_pdb")
                       ))
        refined_pdb = None
        if "iterative_refine" in history:
            refined_pdb = history["iterative_refine"].outputs.get("refined_pdb")

        review_pdb_src = refined_pdb or cropped_pdb
        review_pdb_dst: Path | None = None
        if review_pdb_src and Path(review_pdb_src).is_file():
            import shutil
            review_pdb_dst = outdir / "review.pdb"
            shutil.copyfile(review_pdb_src, review_pdb_dst)

        # CIF output — write from ctx.atoms via ASE if available,
        # otherwise via biotite from the PDB.
        theozyme_cif: Path | None = None
        try:
            from ase.io import write as ase_write
            theozyme_cif = outdir / "theozyme.cif"
            if ctx.atoms is not None:
                ase_write(str(theozyme_cif), ctx.atoms, format="cif")
            elif review_pdb_src and Path(review_pdb_src).is_file():
                from ase.io import read as ase_read
                a = ase_read(str(review_pdb_src))
                ase_write(str(theozyme_cif), a, format="cif")
            else:
                theozyme_cif = None
        except Exception as e:
            log.warning(f"    CIF write failed ({type(e).__name__}: {e})")
            theozyme_cif = None

        # ── Plausibility flags ───────────────────────────────────────
        plausibility = _compute_plausibility_flags(history, entry)

        # ── Build the AME-format JSON sidecar ────────────────────────
        catalytic_payload = []
        if entry is not None:
            for r in entry.catalytic_residues:
                catalytic_payload.append({
                    "chain": r.chain,
                    "auth_seq": r.auth_seq,
                    "code": r.code,
                    "uniprot_seq": r.uniprot_seq,
                    "role": r.role,
                    "is_ptm": r.is_ptm,
                })

        bonds_formed = []
        bonds_broken = []
        bond_order_changed = []
        net_charge: int | None = None
        if "per_step_vacuum_ts" in history:
            sub = history["per_step_vacuum_ts"]
            # Stage 4 carries delegated VacuumTSSearch outputs through
            extra = getattr(sub, "extra", None) or {}
            net_charge = extra.get("net_charge", net_charge)
            bonds_formed = extra.get("bonds_formed", bonds_formed) or bonds_formed
            bonds_broken = extra.get("bonds_broken", bonds_broken) or bonds_broken

        vacuum_barrier_kcal: float | None = None
        if "per_step_vacuum_ts" in history:
            vacuum_barrier_kcal = history["per_step_vacuum_ts"].outputs.get(
                "vacuum_barrier_kcal", None)

        in_protein_barrier_kcal: float | None = None  # populated by Stage 6/7 when implemented

        mlff_model_used: str | None = None
        if "iterative_refine" in history:
            mlff_model_used = history["iterative_refine"].outputs.get("model_used")

        cofactors_observed: list[str] = []
        if "crop_active_site" in history:
            cofactors_observed = history["crop_active_site"].outputs.get(
                "cofactors", []) or []

        ame_payload = {
            "schema_version": "qcb.theozyme.v1",
            "mcsa_id": entry.mcsa_id if entry else ctx.metadata.get("mcsa_id"),
            "enzyme_name": entry.enzyme_name if entry else None,
            "ec": entry.ec if entry else None,
            "reference_pdb": entry.reference_pdb if entry else None,
            "reference_uniprot": entry.reference_uniprot if entry else None,
            "catalytic_residues": catalytic_payload,
            "cofactors_observed": cofactors_observed,
            "ptm_residues": [r.code for r in entry.ptm_residues] if entry else [],
            "substrate_smiles": ctx.metadata.get("reactant_smiles"),
            "product_smiles": ctx.metadata.get("product_smiles"),
            "bonds_formed": list(bonds_formed),
            "bonds_broken": list(bonds_broken),
            "bonds_order_changed": list(bond_order_changed),
            "net_charge": net_charge,
            "vacuum_barrier_kcal_mol": vacuum_barrier_kcal,
            "in_protein_barrier_kcal_mol": in_protein_barrier_kcal,
            "mlff_model_used": mlff_model_used,
            "constraint_mode": (history.get("iterative_refine") and
                                history["iterative_refine"].outputs.get(
                                    "constraint_mode")) or "ca-only",
            "tier2_added_residues": (history.get("tier2_expansion") and
                                     history["tier2_expansion"].outputs.get(
                                         "added_residues")) or [],
            "n_atoms_total": (history.get("iterative_refine") and len(ctx.atoms))
                             or (history.get("tier2_expansion") and
                                 history["tier2_expansion"].outputs.get("n_atoms")),
            "plausibility_flags": plausibility,
            "ame_format_compat": "github.com/RosettaCommons/RFdiffusion2",
        }
        theozyme_json = outdir / "theozyme.json"
        theozyme_json.write_text(json.dumps(ame_payload, indent=2, default=str))

        return StepResult(
            name=self.name,
            atoms=ctx.atoms,
            outputs={
                "theozyme_cif": str(theozyme_cif) if theozyme_cif else None,
                "theozyme_json": str(theozyme_json),
                "review_pdb": str(review_pdb_dst) if review_pdb_dst else None,
                "plausibility_flags": plausibility,
                "n_atoms": ame_payload["n_atoms_total"],
            },
        )


# ─────────────────────────────────────────────────────────────────────
# Helpers for Stage 5 (refinement)
# ─────────────────────────────────────────────────────────────────────

def _ca_indices_from_pdb(pdb_path: str) -> list[int]:
    """Return zero-indexed CA atom indices in the order they appear in
    the PDB file. ASE's PDB reader keeps the same atom order as the
    file (after dropping non-coordinate lines), so the indices line up
    with the ASE Atoms object built from the same path.

    Returns an empty list if the file isn't readable or has no CA
    atoms (e.g. ligand-only crops)."""
    if not pdb_path or not Path(pdb_path).is_file():
        return []
    indices: list[int] = []
    seen = 0
    with Path(pdb_path).open() as fh:
        for line in fh:
            rec = _parse_atom_record(line)
            if rec is None:
                continue
            kind, aname, resname, resseq, x, y, z, chain = rec
            if aname == "CA" and resname not in ("HOH", "WAT"):
                indices.append(seen)
            seen += 1
    return indices


def _infer_net_charge_from_smiles(smiles: str) -> int:
    """Sum explicit formal charges across SMILES fragments. Returns 0
    for empty / unparseable SMILES (the safe default for vacuum runs)."""
    if not smiles:
        return 0
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return int(sum(a.GetFormalCharge() for a in mol.GetAtoms()))
    except Exception:
        return 0


def _gxtb_relax(atoms, outdir: Path, *, fmax: float, max_steps: int,
                charge: int) -> tuple[Any, float | None, float | None]:
    """xTB GFN2 BFGS-style optimisation via the vendored xTB binary.

    The vendored xtb binary is the GFN2 version (in deps/xtb); g-xTB is
    a separate binary not used here (the function name is historical).
    Path: run_xtb_opt(xyz, charge, ...) — file-based interface, returns
    the path to xtbopt.xyz which we read back into ASE.

    Returns (relaxed_atoms, None, None) — xTB prints final force to its
    log but parsing it isn't worth the round-trip; the relaxed atoms
    object carries forces if needed downstream.
    """
    try:
        from quantum_engine.qm.xtb import run_xtb_opt
    except Exception as e:
        raise RuntimeError(f"quantum_engine.qm.xtb not loadable: {e}")

    from ase.io import write as ase_write, read as ase_read
    sub = outdir / "xtb"
    sub.mkdir(parents=True, exist_ok=True)

    in_xyz = sub / "input.xyz"
    ase_write(str(in_xyz), atoms, format="xyz")

    log.info(f"    xTB GFN2 optimise (max_steps={max_steps}, charge={charge}) → {sub}")
    try:
        out_xyz = run_xtb_opt(
            xyz_path=in_xyz,
            charge=charge,
            method="gfn2",
            output_dir=sub,
        )
    except RuntimeError as e:
        # xTB SCF failures are expected for some substrates — surface
        # the message so the caller can fall through to the MLFF stage
        # without crashing the whole entry.
        raise RuntimeError(f"xTB optimisation failed: {e}")

    relaxed = ase_read(str(out_xyz))
    return relaxed, None, None


def _mlff_polish(atoms, outdir: Path, *, model: str, fmax: float,
                 max_steps: int, charge: int) -> tuple[Any, str, float]:
    """MACE polish via quantum_engine.calc.make_calc(model). Falls
    back to mace-omol if MACE-POLAR isn't loadable. Skips entirely
    (raises) if no MACE flavor works in this env."""
    try:
        from quantum_engine.calc import make_calc
    except Exception as e:
        raise RuntimeError(f"quantum_engine.calc.make_calc unavailable: {e}")

    fallback_chain = [model]
    if model == "mace-polar":
        fallback_chain.extend(["mace-polar-m", "mace-polar-l", "mace-omol"])
    elif "polar" in model:
        fallback_chain.append("mace-omol")

    last_err: Exception | None = None
    for candidate in fallback_chain:
        try:
            calc = make_calc(model=candidate, charge=charge,
                             device="cuda", default_dtype="float64")
        except Exception as e:
            last_err = e
            log.warning(f"    {candidate} not loadable: {type(e).__name__}: {e}")
            continue
        atoms.calc = calc
        try:
            from ase.optimize import BFGS
            traj_path = outdir / f"mlff_{candidate}_opt.traj"
            opt = BFGS(atoms, logfile=str(outdir / f"mlff_{candidate}_opt.log"),
                       trajectory=str(traj_path))
            opt.run(fmax=fmax, steps=max_steps)
            import numpy as _np
            max_f = float(_np.linalg.norm(atoms.get_forces(), axis=1).max())
            return atoms, candidate, max_f
        except Exception as e:
            last_err = e
            log.warning(f"    {candidate} optimisation failed: "
                        f"{type(e).__name__}: {e}")
            continue
    raise RuntimeError(
        f"All MLFF candidates failed for {model}: {fallback_chain}. "
        f"Last: {type(last_err).__name__ if last_err else '?'}: {last_err}"
    )


def _compute_plausibility_flags(history: dict, entry) -> dict:
    """Heuristic plausibility checks from accumulated stage history."""
    flags = {
        "frequency_check_passed": None,         # only Stage 7 can set this
        "barrier_in_physical_range": None,
        "metal_coordination_preserved": None,
        "qualitative_match_to_mcsa_text": None, # too soft; user-judged
        "tier2_motif_filled": False,
        "ptm_residues_present": False,
        "cofactor_metals_observed": False,
    }

    # tier-2 motif fill
    if "tier2_expansion" in history:
        n_motif = history["tier2_expansion"].outputs.get("n_added_motif", 0)
        flags["tier2_motif_filled"] = bool(n_motif)

    # PTMs
    if entry is not None and entry.ptm_residues:
        flags["ptm_residues_present"] = True

    # Cofactors observed in the cropped PDB
    if "crop_active_site" in history:
        cofs = history["crop_active_site"].outputs.get("cofactors", []) or []
        flags["cofactor_metals_observed"] = bool([c for c in cofs if c != "NA"])

    # Vacuum-TS barrier sanity
    if "per_step_vacuum_ts" in history:
        b = history["per_step_vacuum_ts"].outputs.get("vacuum_barrier_kcal")
        if isinstance(b, (int, float)):
            flags["barrier_in_physical_range"] = (0 < b < 60)

    # Frequency check (Stage 7) — only set when polish_ts ran
    if "polish_ts" in history:
        n_imag = history["polish_ts"].outputs.get("n_imaginary_modes")
        if n_imag is not None:
            flags["frequency_check_passed"] = (n_imag == 1)

    return flags


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
