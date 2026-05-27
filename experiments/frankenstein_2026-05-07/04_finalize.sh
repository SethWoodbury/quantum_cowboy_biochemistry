#!/bin/bash
#SBATCH --job-name=frank_final_2026-05-07
#SBATCH --partition=cpu-bf
#SBATCH --mem=8g
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/net/scratch/woodbuse/PTE_slurm_logs/frank2_final-%j.out
#SBATCH --error=/net/scratch/woodbuse/PTE_slurm_logs/frank2_final-%j.err
set -euo pipefail

REPO=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
BASE=/net/scratch/woodbuse/frankenstein_AChE_PdPTE_2026-05-07
SCAN_DIR=$BASE/06_scan
NEB_DIR=$BASE/07_neb
REF_DIR=$BASE/08_refine_ts
OUT_DIR=/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/validated_TSs/frankenstein_AChE_PdPTE_KCX_set0__O1nuc_P1_O5lg__netCHG_minus_2
BASENAME=frankenstein_AChE_PdPTE_KCX_set0__O1nuc_P1_O5lg__netCHG_minus_2

mkdir -p "$OUT_DIR"

echo "==> [FRANK_FINAL] $(date -Iseconds)"
echo "==> SCAN_DIR  = $SCAN_DIR"
echo "==> NEB_DIR   = $NEB_DIR"
echo "==> REF_DIR   = $REF_DIR"
echo "==> OUT_DIR   = $OUT_DIR"

# Validate the refine-ts run
if [ ! -d "$REF_DIR" ]; then
  echo "ERROR: refine-ts dir missing"; exit 2
fi

# Find files refine-ts wrote
TS_PDB="$REF_DIR/ts_refined.pdb"
TS_CIF="$REF_DIR/ts_refined.cif"
SUMMARY="$REF_DIR/summary.json"
FREQLOG="$REF_DIR/refine_ts_run.log"

echo "==> TS_PDB   = $TS_PDB"
echo "==> TS_CIF   = $TS_CIF"
echo "==> SUMMARY  = $SUMMARY"
echo "==> FREQLOG  = $FREQLOG"

# Pull the most-recent NEB endpoints (NEB writes reactant.pdb / product.pdb)
REACT_PDB="$NEB_DIR/reactant.pdb"
PROD_PDB="$NEB_DIR/product.pdb"

# Copy artifacts with the standard naming pattern
cp -v "$BASE/05_minimize_mace_mp/relaxed.pdb" "$OUT_DIR/input.pdb" || true
[ -n "${TS_PDB:-}" ] && cp -v "$TS_PDB"  "$OUT_DIR/${BASENAME}_TS.pdb" || true
[ -n "${TS_CIF:-}" ] && cp -v "$TS_CIF"  "$OUT_DIR/${BASENAME}_TS.cif" || true
[ -n "${SUMMARY:-}" ] && cp -v "$SUMMARY" "$OUT_DIR/${BASENAME}_refine_summary.json" || true
[ -n "${FREQLOG:-}" ] && cp -v "$FREQLOG" "$OUT_DIR/${BASENAME}_freq.log" || true
[ -f "$REACT_PDB" ] && cp -v "$REACT_PDB" "$OUT_DIR/${BASENAME}_reactant.pdb" || true
[ -f "$PROD_PDB" ]  && cp -v "$PROD_PDB"  "$OUT_DIR/${BASENAME}_product.pdb"  || true

# Polish trajectory: prefer NEB's multi-MODEL PDB (already in PDB format).
# Fall back to the saddle trajectory if NEB doesn't have one.
NEB_TRAJ_PDB="$NEB_DIR/neb-path-trajectory.pdb"
TRAJ_OUT="$OUT_DIR/${BASENAME}_polish_trajectory.pdb"

if [ -f "$NEB_TRAJ_PDB" ]; then
  cp -v "$NEB_TRAJ_PDB" "$TRAJ_OUT"
else
  echo "==> WARNING: NEB trajectory PDB not found at $NEB_TRAJ_PDB"
  ls "$NEB_DIR"
fi

echo "==> Final dir contents:"
ls -la "$OUT_DIR"
echo "==> $(date -Iseconds) finalize done"
