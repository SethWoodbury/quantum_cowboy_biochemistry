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
