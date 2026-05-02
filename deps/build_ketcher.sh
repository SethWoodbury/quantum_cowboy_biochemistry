#!/usr/bin/env bash
# Build a standalone Ketcher bundle from the deps/ketcher submodule.
#
# Default: runs `npm install` + `npm run build:standalone` in the
# example/ app of the vendored repo, producing a static HTML+JS bundle
# at deps/ketcher/install/ that opens directly in a browser (no
# server, no backend).
#
# Build time: ~5-10 min on first run (pulls ~400 MB of node_modules
# under deps/ketcher/example/node_modules). Subsequent runs reuse
# the cache.
#
# Usage:
#   bash deps/build_ketcher.sh
#   xdg-open deps/ketcher/install/index.html   # then draw reactions
#
# Zero-install alternative: open tools/ketcher.html (CDN-loaded; needs
# internet on first load).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/deps/ketcher/example"
OUT="${ROOT}/deps/ketcher/install"

if ! command -v npm >/dev/null; then
    echo "ERROR: npm not found. Install Node.js first (>=18)." >&2
    exit 1
fi
if [ ! -d "${SRC}" ]; then
    echo "ERROR: ${SRC} missing — run 'git submodule update --init deps/ketcher'" >&2
    exit 1
fi

cd "${SRC}"
echo "==> npm install (≈400MB of node_modules)…"
npm install --silent

echo "==> Building standalone bundle…"
npm run build:standalone

# The Vite/React build lands in dist/standalone/ — copy to install/.
mkdir -p "${OUT}"
rm -rf "${OUT}/standalone"
cp -r dist/standalone "${OUT}/"
# Convenience: a top-level index.html that points at the standalone build.
cp "${OUT}/standalone/index.html" "${OUT}/index.html" 2>/dev/null || true

echo
echo "=========================================="
echo "Ketcher standalone build complete."
echo "  Open in browser:  ${OUT}/index.html"
echo "  Or:               file://${OUT}/standalone/index.html"
echo "=========================================="
