#!/usr/bin/env bash
# Deploy a tagged release of quantum_cowboy_biochemistry.
#
# Model (see docs/deploy.md): dev lives in this checkout; a release is a git tag
# that is (1) synced to the shared install at /net/software/lab and (2) paired
# with a container rebuild, so dev tree / shared install / container never drift.
#
# Usage:
#   ./deploy.sh v0.3.0                 # tag, sync to /net/software/lab, print rebuild cmds
#   ./deploy.sh v0.3.0 --dry-run       # show what would happen, change nothing
#   DEST=/some/other/path ./deploy.sh v0.3.0
set -euo pipefail

VERSION="${1:-}"
DRY_RUN=0
[[ "${2:-}" == "--dry-run" ]] && DRY_RUN=1
DEST="${DEST:-/net/software/lab/quantum_cowboy_biochemistry}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF_DIR="/net/software/containers/users/${USER}/quantum_chem"

if [[ -z "$VERSION" ]]; then
  echo "usage: ./deploy.sh <version-tag> [--dry-run]" >&2; exit 2
fi

# 1. Require a clean tree so the tag is reproducible.
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "ERROR: working tree is dirty — commit or stash before deploying." >&2
  git -C "$REPO" status -s >&2; exit 1
fi

run() { if [[ $DRY_RUN -eq 1 ]]; then echo "  [dry-run] $*"; else eval "$*"; fi; }

echo "==> 1/3  Tagging release $VERSION"
run "git -C '$REPO' tag -a '$VERSION' -m 'release $VERSION'"

echo "==> 2/3  Syncing to shared install: $DEST"
# Sync tracked code only; exclude run artifacts, vendored build outputs, caches.
run "rsync -a --delete \
  --exclude '.git' --exclude 'outputs' --exclude 'runs' --exclude 'logs' \
  --exclude 'deps/*/install' --exclude 'deps/*/build' --exclude '__pycache__' \
  --exclude 'data/*_cache' --exclude '*.sif' \
  '$REPO/' '$DEST/'"

echo "==> 3/3  Container rebuild (run manually when deps change):"
cat <<EOF
  apptainer build --fakeroot \\
    ${SIF_DIR}/quantum_chem-\$(date +%Y%m%d).sif ${REPO}/deps/quantum_chem.def
  apptainer build --fakeroot \\
    ${SIF_DIR}/uma-\$(date +%Y%m%d).sif ${REPO}/deps/uma_sidecar.def

Then push the tag:  git push origin $VERSION
EOF
echo "Done${DRY_RUN:+ (dry-run)}."
