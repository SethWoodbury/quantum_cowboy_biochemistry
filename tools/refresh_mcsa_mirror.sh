#!/usr/bin/env bash
# Refresh the lab-shared M-CSA mirror from EBI.
# Creates a new dated dir under /net/databases/lab/m_csa__embl_ebi/
# without overwriting prior dumps — the audit trail of what schema we
# worked against on a given date.
#
# Layout produced:
#   original_<YYYY-MM-DD>/
#     all_entries.json            (paginated; 1003 entries currently)
#     entries/entry_<id>.json     (one per entry, 1003 files)
#     flat_files/{curated_data,literature_pdb_residues,literature_pdb_residues_roles}.csv
#     homologues_residues.json    (~340 MB)
#
# Usage: bash tools/refresh_mcsa_mirror.sh
set -euo pipefail
ROOT=/net/databases/lab/m_csa__embl_ebi
DATESTAMP=$(date +%Y-%m-%d)
ORIG="$ROOT/original_${DATESTAMP}"

if [ -d "$ORIG" ]; then
    echo "ERROR: $ORIG already exists. Pick a new date or remove the dir first." >&2
    exit 1
fi

mkdir -p "$ORIG/entries" "$ORIG/flat_files"
chmod 2775 "$ORIG" "$ORIG/entries" "$ORIG/flat_files"
chgrp baker "$ORIG" "$ORIG/entries" "$ORIG/flat_files" 2>/dev/null || true

echo "==> Downloading flat CSV files…"
for f in curated_data.csv literature_pdb_residues.csv \
         literature_pdb_residues_roles.csv ; do
    curl -fsSL -o "$ORIG/flat_files/$f" \
        "https://www.ebi.ac.uk/thornton-srv/m-csa/media/flat_files/$f"
    echo "  ✓ $f"
done

echo "==> Downloading paginated entries…"
python3 <<PYEOF
import json, urllib.request, time
from pathlib import Path
ORIG = Path("$ORIG")
url = "https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/?format=json"
all_results = []
page = 1
while url:
    print(f"  page {page}: …")
    with urllib.request.urlopen(url, timeout=60) as r:
        d = json.loads(r.read().decode())
    all_results.extend(d["results"])
    url = d.get("next")
    page += 1
    time.sleep(0.5)
print(f"  total: {len(all_results)} entries")
(ORIG / "all_entries.json").write_text(
    json.dumps({"count": len(all_results), "results": all_results}, indent=2))
for e in all_results:
    mid = e.get("mcsa_id")
    if mid is not None:
        (ORIG / "entries" / f"entry_{mid}.json").write_text(json.dumps(e, indent=2))
PYEOF

echo "==> Downloading homologues_residues.json (~340 MB)…"
curl -fsSL -o "$ORIG/homologues_residues.json" \
    "https://www.ebi.ac.uk/thornton-srv/m-csa/api/homologues_residues.json"

echo "==> Group ownership + setgid…"
chgrp -R baker "$ORIG" 2>/dev/null || true

echo
echo "Done. Total size: $(du -sh "$ORIG" | cut -f1)"
echo "  Mirror root: $ROOT"
echo "  Newest dump: $ORIG"
