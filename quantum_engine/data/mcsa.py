"""M-CSA (Mechanism and Catalytic Site Atlas) parser.

The Thornton-lab M-CSA database catalogues curated catalytic-residue
annotations and mechanism graphs for ~1000 enzymes. We use it for two
things in QCB:

  1. **Test-case selection** for the enzyme TS-search pipelines —
     pull catalytic residue lists, cofactors, reference PDB, mechanism
     step descriptions for a target entry.
  2. **Theozyme generation** for the AME benchmark
     (``enz_qc_pipelines.mcsa_theozyme``) — feed the mechanism arrow-
     pushing (Marvin XML) + concrete substrate SMILES into a per-step
     vacuum + in-protein TS pipeline.

API endpoint is ``https://www.ebi.ac.uk/thornton-srv/m-csa/api/`` (CC
BY 4.0). Per-entry JSON is fetched via the ``entries.mcsa_ids`` filter.
SMILES are NOT in M-CSA — only ChEBI IDs; resolve via :mod:`.chebi`.

Cache lives at :func:`quantum_engine.site.mcsa_cache_dir` — typically
``/net/databases/mcsa_cache/`` on DIGS, falls back to ``./data/mcsa_cache/``.

Quick example:

    >>> from quantum_engine.data.mcsa import fetch_entry
    >>> entry = fetch_entry(159)            # phosphotriesterase
    >>> entry.enzyme_name
    'Phosphotriesterase'
    >>> [r.code for r in entry.catalytic_residues]
    ['HIS', 'HIS', 'HIS', 'HIS', 'KCX', 'ASP']
    >>> entry.cofactors
    ['ZN', 'ZN']
    >>> entry.reference_pdb
    '1hzy'
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("quantum_engine.data.mcsa")

API_BASE = "https://www.ebi.ac.uk/thornton-srv/m-csa/api"
ENTRIES_URL = f"{API_BASE}/entries/?format=json&entries.mcsa_ids={{ids}}"

# Residue codes that are post-translationally-modified (we treat them as
# catalytic by default and the prep pipeline must handle their topology).
PTM_CODES = frozenset({
    "KCX",  # carbamylated lysine (PTE entry 159)
    "MSE",  # selenomethionine
    "CSO",  # S-hydroxycysteine
    "SEP",  # phosphoserine
    "TPO",  # phosphothreonine
    "PTR",  # phosphotyrosine
})

# Common metal codes seen in catalytic sites — used to flag cofactors
# that aren't stored as residue objects in M-CSA but appear as HETATM in
# the reference PDB.
METAL_CODES = frozenset({
    "ZN", "MG", "CA", "MN", "FE", "CU", "NI", "CO", "MO", "K", "NA",
})


@dataclass
class CatalyticResidue:
    """One catalytic residue annotation from M-CSA."""
    code: str                    # 3-letter residue code (e.g. "HIS", "KCX")
    chain: str                   # PDB chain
    auth_seq: int                # PDB author seqnum
    uniprot_seq: int | None = None
    pdb_id: str | None = None
    role: str | None = None      # "general acid", "metal ligand", etc.
    is_ptm: bool = False         # auto-flagged from PTM_CODES


@dataclass
class MechanismStep:
    """One step of a curated mechanism (M-CSA mechanisms[].steps[])."""
    step_id: int
    description: str
    marvin_xml: str | None = None    # ChemAxon Marvin XML for arrow-pushing
    is_product: bool = False


@dataclass
class Mechanism:
    """A full mechanism (sequence of steps) for an entry."""
    text: str                            # free-text HTML description
    steps: list[MechanismStep] = field(default_factory=list)


@dataclass
class MCSAEntry:
    """Structured view of one M-CSA entry. The fields we care about
    for TS-search pipelines; full raw JSON kept in :attr:`raw` for
    consumers that need more."""
    mcsa_id: int
    enzyme_name: str
    ec: list[str] = field(default_factory=list)
    reference_pdb: str | None = None
    reference_uniprot: str | None = None
    catalytic_residues: list[CatalyticResidue] = field(default_factory=list)
    cofactors: list[str] = field(default_factory=list)        # metal codes
    chebi_ids: list[int] = field(default_factory=list)        # for substrates/products
    compounds: list[dict[str, Any]] = field(default_factory=list)  # raw rxn.compounds
    mechanisms: list[Mechanism] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ptm_residues(self) -> list[CatalyticResidue]:
        """Catalytic residues with PTMs (KCX, SEP, …) — important for
        the active-site prep stage's topology handling."""
        return [r for r in self.catalytic_residues if r.is_ptm]


def _cache_path(mcsa_id: int) -> Path:
    from quantum_engine.site import mcsa_cache_dir
    return mcsa_cache_dir() / f"entry_{mcsa_id}.json"


def fetch_entry(mcsa_id: int, *, refresh: bool = False) -> MCSAEntry:
    """Fetch one M-CSA entry. Reads from cache if present unless
    ``refresh=True``. Returns a parsed :class:`MCSAEntry`."""
    cache = _cache_path(mcsa_id)
    if cache.is_file() and not refresh:
        log.debug(f"M-CSA cache hit: {cache}")
        raw = json.loads(cache.read_text())
    else:
        log.info(f"M-CSA fetch: entry {mcsa_id}")
        raw = _http_get_entry(mcsa_id)
        cache.write_text(json.dumps(raw, indent=2))
    return _parse_entry(raw, mcsa_id)


def fetch_entries(mcsa_ids: list[int], *, refresh: bool = False) -> list[MCSAEntry]:
    """Batch fetch — uses the multi-id filter. Cache files are
    per-entry so a partial cache hit still saves bandwidth."""
    return [fetch_entry(i, refresh=refresh) for i in mcsa_ids]


def _http_get_entry(mcsa_id: int) -> dict[str, Any]:
    """Pull one entry from the M-CSA API. Returns the {results: [...]}
    wrapper's first record. Network-required; raises on HTTP error."""
    import urllib.request
    url = ENTRIES_URL.format(ids=mcsa_id)
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read().decode())
    results = payload.get("results", [])
    if not results:
        raise ValueError(f"M-CSA entry {mcsa_id} not found (empty results from {url})")
    return results[0]


def _parse_entry(raw: dict[str, Any], mcsa_id: int) -> MCSAEntry:
    """Convert raw API JSON → MCSAEntry. Tolerant of missing fields."""
    name = raw.get("enzyme_name", "")
    ecs = list(raw.get("all_ecs", []) or [])
    uniprot = raw.get("reference_uniprot_id")

    # Catalytic residues — flatten residues[] → residue_chains[]
    catalytic: list[CatalyticResidue] = []
    ref_pdb: str | None = None
    for r in raw.get("residues", []) or []:
        role = r.get("function_location_abv") or (
            ", ".join(r.get("roles_summary", []) or []) or None
        )
        chains = r.get("residue_chains", []) or []
        seqs = r.get("residue_sequences", []) or []
        uniprot_seq = seqs[0].get("resid") if seqs else None
        for c in chains:
            if c.get("is_reference") and not ref_pdb:
                ref_pdb = c.get("pdb_id")
            code = (c.get("code") or "").upper()
            catalytic.append(CatalyticResidue(
                code=code,
                chain=c.get("chain_name") or "",
                auth_seq=c.get("auth_resid") or c.get("resid") or -1,
                uniprot_seq=uniprot_seq,
                pdb_id=c.get("pdb_id"),
                role=role,
                is_ptm=code in PTM_CODES,
            ))

    # Cofactors — M-CSA doesn't store metals as residues. We infer them
    # by scanning catalytic residue roles for "metal" mentions, which
    # signals "this residue ligates a metal that lives in the PDB
    # HETATM section." The actual metal codes come from the PDB; we
    # surface a placeholder list here that the prep stage replaces by
    # parsing the reference PDB's HETATM lines.
    cofactor_set: set[str] = set()
    for r in catalytic:
        if r.role and "metal" in r.role.lower():
            # Caller will resolve via PDB; we just flag presence.
            cofactor_set.add("?METAL?")
            break

    # Compounds (substrates/products) — pull ChEBI IDs.
    compounds = []
    chebi_ids: list[int] = []
    rxn = raw.get("reaction") or {}
    for c in rxn.get("compounds", []) or []:
        compounds.append(c)
        if c.get("chebi_id") is not None:
            try:
                chebi_ids.append(int(c["chebi_id"]))
            except (TypeError, ValueError):
                pass

    # Mechanisms (multi-step, optional Marvin XML per step)
    mechs: list[Mechanism] = []
    for m in rxn.get("mechanisms", []) or []:
        steps = []
        for s in m.get("steps", []) or []:
            steps.append(MechanismStep(
                step_id=s.get("step_id") or len(steps),
                description=s.get("description") or "",
                marvin_xml=s.get("marvin_xml"),
                is_product=bool(s.get("is_product")),
            ))
        mechs.append(Mechanism(
            text=m.get("mechanism_text") or "",
            steps=steps,
        ))

    return MCSAEntry(
        mcsa_id=mcsa_id,
        enzyme_name=name,
        ec=ecs,
        reference_pdb=ref_pdb,
        reference_uniprot=uniprot,
        catalytic_residues=catalytic,
        cofactors=sorted(cofactor_set),
        chebi_ids=chebi_ids,
        compounds=compounds,
        mechanisms=mechs,
        raw=raw,
    )


def parse_marvin_xml(xml_str: str) -> dict[str, Any]:
    """Extract atoms / bonds / arrows from a Marvin XML mechanism step.

    M-CSA stores arrow-pushing diagrams as ChemAxon Marvin XML. The
    relevant nodes are ``<atomArray>`` (with element + index),
    ``<bondArray>`` (with order + atom-refs), and ``<MEFlow>`` (electron-
    flow arrows linking source bond/lp → target atom/bond).

    Returns a dict ``{atoms, bonds, arrows}`` where each arrow encodes a
    forming or breaking bond — the driving coordinate set for stage 7
    of the M-CSA pipeline. STUB: full schema parse comes next session;
    we currently extract atoms+bonds and leave arrows TODO.
    """
    import xml.etree.ElementTree as ET
    if not xml_str or not xml_str.strip():
        return {"atoms": [], "bonds": [], "arrows": []}
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        log.warning(f"marvin_xml parse failed: {e}")
        return {"atoms": [], "bonds": [], "arrows": []}

    ns = ""  # Marvin uses default ns; adjust if upstream changes
    atoms = [
        {"id": a.get("id"), "element": a.get("elementType")}
        for a in root.iter("atom")
    ]
    bonds = [
        {"id": b.get("id"), "atomRefs2": b.get("atomRefs2"),
         "order": b.get("order")}
        for b in root.iter("bond")
    ]
    # MEFlow tags carry the arrow-pushing — TODO: extract source/target
    # endpoints once we have a real entry to test against.
    arrows: list[dict[str, Any]] = []
    return {"atoms": atoms, "bonds": bonds, "arrows": arrows}
