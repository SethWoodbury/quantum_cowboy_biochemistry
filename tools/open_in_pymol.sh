#!/usr/bin/env bash
# Open every entry's reviewable PDBs in PyMOL — one PyMOL session per
# entry, each loaded with tier-1, tier-2, refined, and theozyme.cif so
# you can flip between them with `enable / disable` in the right pane.
#
# Usage:
#     bash tools/open_in_pymol.sh runs/final_validation
#     bash tools/open_in_pymol.sh runs/quick_review --entry 159
#
# Requires `pymol` on PATH. Each entry opens in its own PyMOL window
# (so you can compare side-by-side).
set -euo pipefail

RUNDIR="${1:?usage: $0 <run-dir> [--entry <id>]}"
shift || true

if ! command -v pymol >/dev/null; then
    echo "ERROR: pymol not on PATH. Install via conda or load the cluster module." >&2
    exit 1
fi

WANT_ENTRY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --entry) WANT_ENTRY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

for entry_dir in "${RUNDIR}"/*_*/; do
    base="$(basename "${entry_dir}")"
    id="${base%%_*}"
    if [ -n "${WANT_ENTRY}" ] && [ "${id}" != "${WANT_ENTRY}" ]; then
        continue
    fi

    files=()
    for sub in crop_active_site tier2_expansion iterative_refine write_theozyme; do
        for ext in pdb cif; do
            for f in "${entry_dir}${sub}"/*."${ext}"; do
                if [ -f "${f}" ]; then files+=("${f}"); fi
            done
        done
    done

    if [ ${#files[@]} -eq 0 ]; then
        echo "No reviewable files in ${entry_dir} — skipping."
        continue
    fi

    echo "▶ M-CSA ${id} — opening ${#files[@]} structures in PyMOL"
    pymol -d "set retain_order, 1" "${files[@]}" &
done
wait
echo "All PyMOL sessions closed."
