#!/usr/bin/env bash
# Download the FairChem OMol25 eSEN + AllScAIP checkpoints into the lab MLFF
# weight store, in the flat `models--facebook--<name>/<file>.pt` convention the
# rest of site.MACE_MODELS uses.
#
# GATED: facebook/OMol25 needs FAIR Chemistry License approval. You must (1) have
# clicked "Agree and access repository" on https://huggingface.co/facebook/OMol25
# with your HF account, and (2) provide a token for THAT account:
#
#   HF_TOKEN=hf_xxxx bash deps/download_omol25_models.sh
#
# (or `huggingface-cli login` first, then run without HF_TOKEN.)
#
# Conserving vs direct: the *conserving* checkpoints have forces = -dE/dx and are
# the ones valid for TS / saddle / Hessian work; the *direct* ones are fast but
# non-conservative (single-points / MD only). All are downloaded here for
# completeness.
set -euo pipefail

BASE="${MLFF_BASE:-/net/databases/huggingface/mlFF_models}"
REPO="facebook/OMol25"

# subdir <- repo-filename  (flat destination name : source path in the repo)
declare -A MODELS=(
  ["esen-sm-conserving-all-omol/esen_sm_conserving_all.pt"]="checkpoints/esen_sm_conserving_all.pt"
  ["esen-sm-direct-all-omol/esen_sm_direct_all.pt"]="checkpoints/esen_sm_direct_all.pt"
  ["esen-md-direct-all-omol/esen_md_direct_all.pt"]="checkpoints/esen_md_direct_all.pt"
  ["allscaip-omol102m-md-cons/AllScAIP-OMol102M-md-cons.pt"]="checkpoints/AllScAIP/AllScAIP-OMol102M-md-cons.pt"
  ["allscaip-omol102m-md-d/AllScAIP-OMol102M-md-d.pt"]="checkpoints/AllScAIP/AllScAIP-OMol102M-md-d.pt"
)

for dest_rel in "${!MODELS[@]}"; do
  src="${MODELS[$dest_rel]}"
  dest="$BASE/models--facebook--$dest_rel"
  echo ">>> $src  ->  $dest"
  python - "$REPO" "$src" "$dest" <<'PY'
import os, shutil, sys
from huggingface_hub import hf_hub_download
repo, src, dest = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(os.path.dirname(dest), exist_ok=True)
p = hf_hub_download(repo_id=repo, filename=src, token=os.environ.get("HF_TOKEN"))
shutil.copy(p, dest)
print("  ok:", dest, os.path.getsize(dest), "bytes")
PY
done
echo "All OMol25 checkpoints downloaded under $BASE"
