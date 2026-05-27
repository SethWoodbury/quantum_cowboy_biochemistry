#!/bin/bash
# Codex review of scan profile.
# Usage: bash codex_review_scan.sh
set -euo pipefail
SCAN_DIR=/net/scratch/woodbuse/frankenstein_AChE_PdPTE_2026-05-07/06_scan
SUMMARY="$SCAN_DIR/summary.json"
OUTFILE="$SCAN_DIR/codex_review.txt"

if [ ! -f "$SUMMARY" ]; then
  echo "ERROR: $SUMMARY not found" >&2
  exit 2
fi

PROMPT="Review the relaxed-scan TS profile for an AChE-PdPTE Frankenstein \
chimera (paraoxon-like substrate; -2 net charge; 1060 atoms; CV s = d(P-O1) - d(P-O5)). \
The scan summary.json is at $SUMMARY. Tasks: \
(1) Read the JSON and summarize the energy vs s profile (interior max? endpoint? non-monotonic?). \
(2) Identify which scan point is the best TS guess (highest E, ideally interior). \
(3) Flag if convergence at any point failed (fmax > 0.05 or converged=false). \
(4) Note if the saddle is too close to an endpoint (suggests widening scan). \
(5) Check whether d(P-O1) and d(P-O5) cross or are still imbalanced at the maximum. \
Output 5-15 sentences, no bullets."

codex exec -c sandbox_workspace_write="$SCAN_DIR" "$PROMPT" 2>&1 | tee "$OUTFILE"
echo "WROTE: $OUTFILE"
