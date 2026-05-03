"""ChEBI compound lookup — SMILES from a ChEBI ID.

M-CSA gives ChEBI IDs for substrates/products but the SMILES field is
null in practice; we resolve through ChEBI ourselves.

Two strategies, in order:
  1. Local flat-file dump at ``chebi_cache_dir()/chebi_compounds.tsv.gz``
     (~50 MB, downloaded once via :func:`download_chebi_dump`).
  2. ChEBI ontology lookup service (HTTP) as fallback if no dump.

Per-id results are cached as JSON in ``chebi_cache_dir()/by_id/<id>.json``.

Heads up: the ChEBI ontology has both "CHEBI:25212" and "25212" forms;
we accept either and normalise to the integer.
"""
from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("quantum_engine.data.chebi")

# Standard ChEBI download — flat-file with id, name, smiles, etc.
CHEBI_DUMP_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/chebi/Flat_file_tab_delimited/"
    "chemical_data_3star.tsv"
)

# OLS4 fallback (per-id JSON — slower but no bulk download needed)
OLS_URL = (
    "https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms?iri="
    "http://purl.obolibrary.org/obo/CHEBI_{id}"
)


def _normalise_id(chebi_id: int | str) -> int:
    """'CHEBI:25212' or '25212' or 25212 → 25212."""
    if isinstance(chebi_id, int):
        return chebi_id
    s = str(chebi_id).strip()
    if s.upper().startswith("CHEBI:"):
        s = s.split(":", 1)[1]
    return int(s)


def _per_id_cache(chebi_id: int) -> Path:
    from quantum_engine.site import chebi_cache_dir
    d = chebi_cache_dir() / "by_id"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{chebi_id}.json"


def lookup_smiles(chebi_id: int | str, *, refresh: bool = False) -> Optional[str]:
    """Return canonical SMILES for a ChEBI ID, or None if not found.

    Pulls from per-id cache first; falls back to local dump; falls back
    to OLS4 HTTP. Network-optional once the dump is present.
    """
    cid = _normalise_id(chebi_id)
    cache = _per_id_cache(cid)
    if cache.is_file() and not refresh:
        return json.loads(cache.read_text()).get("smiles")

    smiles = _lookup_in_dump(cid)
    if smiles is None:
        smiles = _lookup_via_ols(cid)
    cache.write_text(json.dumps({"chebi_id": cid, "smiles": smiles}))
    return smiles


def _dump_path() -> Path:
    from quantum_engine.site import chebi_cache_dir
    return chebi_cache_dir() / "chemical_data_3star.tsv"


def _lookup_in_dump(chebi_id: int) -> Optional[str]:
    """Scan the local ChEBI flat-file for a SMILES. The file is small
    enough (~few MB) to scan linearly; we don't bother indexing.
    Returns None if the dump isn't present or the id isn't found."""
    dump = _dump_path()
    if not dump.is_file():
        return None
    target = str(chebi_id)
    # File is "id\tcompound_id\tsource\ttype\tchemical_data" — SMILES
    # rows have type == "SMILES". Compound IDs are NOT the master ChEBI
    # IDs we usually want; rather, master IDs live in compounds.tsv.
    # For now we accept the simple case and revisit if it misses.
    with dump.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            if parts[3] == "SMILES" and parts[1] == target:
                return parts[4]
    return None


def _lookup_via_ols(chebi_id: int) -> Optional[str]:
    """Hit OLS4 for a single compound. SMILES lives under
    ``annotation['smiles_string']`` (a list of one) — verified against
    the live API. Older clients sometimes used 'SMILES' but OLS4
    canonicalises to lowercase + '_string' suffix."""
    import urllib.request
    url = OLS_URL.format(id=chebi_id)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as e:
        log.warning(f"OLS4 lookup failed for ChEBI:{chebi_id}: {e}")
        return None
    embedded = payload.get("_embedded", {}).get("terms", [])
    if not embedded:
        return None
    annot = embedded[0].get("annotation", {}) or {}
    # Try the canonical OLS4 key first, then fall back to legacy variants.
    for key in ("smiles_string", "SMILES", "smiles"):
        v = annot.get(key)
        if isinstance(v, list) and v:
            return v[0]
        if isinstance(v, str) and v:
            return v
    return None


def download_chebi_dump(*, force: bool = False) -> Path:
    """Pull the ChEBI flat-file dump to chebi_cache_dir(). One-shot
    setup; returns the path. ~few MB. Skip if already present unless
    ``force=True``."""
    import urllib.request
    dest = _dump_path()
    if dest.is_file() and not force:
        return dest
    log.info(f"Downloading ChEBI dump → {dest}…")
    urllib.request.urlretrieve(CHEBI_DUMP_URL, dest)
    log.info(f"ChEBI dump cached at {dest}")
    return dest
