#!/bin/bash
set -euo pipefail
REF_DIR=/net/scratch/woodbuse/frankenstein_AChE_PdPTE_2026-05-07/08_refine_ts
SUMMARY="$REF_DIR/summary.json"
OUTFILE="$REF_DIR/codex_review.txt"

if [ ! -f "$SUMMARY" ]; then
  echo "ERROR: $SUMMARY not found" >&2
  exit 2
fi

PROMPT="Review the refine-ts result for the AChE-PdPTE Frankenstein TS \
(net charge -2, 1060 atoms, mace-polar-m). Read $SUMMARY. \
Output 5-12 sentences, no bullets, with a single explicit verdict: PASS, MARGINAL, or FAIL. \
Comment on: \
(1) overall_pass field; \
(2) n_imag (must be 1 for a true TS); \
(3) imag_freq_cm (should be < -25 cm^-1, more negative = stronger TS character); \
(4) imag_mode_overlap with reactive coordinate (should be > 0.5); \
(5) saddle fmax_final (should be < 0.05 eV/A); \
(6) any concerning warnings or partial-Hessian artefacts."

codex exec -c sandbox_workspace_write="$REF_DIR" "$PROMPT" 2>&1 | tee "$OUTFILE"
echo "WROTE: $OUTFILE"
