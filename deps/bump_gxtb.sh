#!/usr/bin/env bash
# Pull the latest g-xTB upstream and re-extract the linux x86_64 tarball
# into deps/g-xtb/install/. Idempotent: runs cleanly even if there's no
# new release.
#
# Why a separate script: g-xTB upstream is pre-release and the Grimme
# lab iterates on it frequently. .gitmodules pins the submodule to
# `branch = main`, so `git submodule update --remote deps/g-xtb` brings
# in whatever they pushed last. We then re-extract the binary tarball
# (its filename includes a build date — we glob for the *latest* linux
# x86_64 tarball in `binaries/`) and run a tiny H2O smoke test to make
# sure the new binary actually runs.
#
# Usage:
#   bash deps/bump_gxtb.sh                # default: bump and verify
#   bash deps/bump_gxtb.sh --no-verify    # skip the H2O test
#
# After running, commit the submodule pin update:
#   git -C ../.. add deps/g-xtb && git commit -m "bump g-xtb to <SHA>"

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GXTB_DIR="${SCRIPT_DIR}/g-xtb"
INSTALL_DIR="${GXTB_DIR}/install"
VERIFY=1
for a in "$@"; do
    case "$a" in
        --no-verify) VERIFY=0 ;;
        *) echo "unknown arg: $a" >&2; exit 2 ;;
    esac
done

if [ ! -d "${GXTB_DIR}" ]; then
    echo "ERROR: ${GXTB_DIR} missing — run: git submodule update --init deps/g-xtb"
    exit 1
fi

# 1. Bring submodule to latest upstream main
echo "==> Pulling g-xTB main → newest upstream commit…"
cd "${GXTB_DIR}"
old_sha=$(git rev-parse --short HEAD)
git fetch origin main
git checkout main
git pull --ff-only origin main
new_sha=$(git rev-parse --short HEAD)
if [ "${old_sha}" = "${new_sha}" ]; then
    echo "  no upstream changes (HEAD ${new_sha}); will still re-extract."
else
    echo "  bumped ${old_sha} → ${new_sha}"
fi

# 2. Find newest linux x86_64 tarball. The filename embeds a 6-digit
# date stamp like 210426 in `xtb-6.7.1-gxtb-210426-linux-x86_64.tar.xz`;
# `ls` sort matches that ordering.
shopt -s nullglob
TARBALLS=("${GXTB_DIR}/binaries/"xtb-*-gxtb-*-linux-x86_64.tar.xz)
shopt -u nullglob
if [ "${#TARBALLS[@]}" -eq 0 ]; then
    echo "ERROR: no linux-x86_64 tarball under ${GXTB_DIR}/binaries/"
    exit 1
fi
TARBALL=$(printf '%s\n' "${TARBALLS[@]}" | sort | tail -1)
echo "==> Re-extracting $(basename "${TARBALL}")"

# 3. Verify the published sha256 if available.
if [ -f "${TARBALL}.sha256" ]; then
    expected=$(awk '{print $1}' "${TARBALL}.sha256")
    actual=$(sha256sum "${TARBALL}" | awk '{print $1}')
    if [ "${expected}" != "${actual}" ]; then
        echo "ERROR: sha256 mismatch for ${TARBALL}"
        echo "  expected: ${expected}"
        echo "  actual:   ${actual}"
        exit 1
    fi
    echo "  sha256 ok (${actual:0:12}…)"
fi

# 4. Wipe the old install and extract fresh. Tarball top-level is e.g.
# `xtb-6.7.1/`; we keep that structure under install/.
rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"
tar xJf "${TARBALL}" -C "${INSTALL_DIR}"

# 5. Resolve the actual binary path (the top dir name includes the
# release tag and may change across versions).
BIN=$(find "${INSTALL_DIR}" -maxdepth 4 -name xtb -type f -executable | head -1)
if [ -z "${BIN}" ]; then
    echo "ERROR: no xtb binary found under ${INSTALL_DIR}"
    exit 1
fi
echo "  binary: ${BIN}"

if [ "${VERIFY}" -eq 1 ]; then
    echo "==> Smoke-testing g-xTB on H2O…"
    TMP=$(mktemp -d)
    cat >"${TMP}/h2o.xyz" <<'EOF'
3
H2O
O 0.000 0.000 0.000
H 0.957 0.000 0.000
H -0.239 0.927 0.000
EOF
    OMP_NUM_THREADS=1 "${BIN}" "${TMP}/h2o.xyz" --gxtb 2>&1 \
        | grep -E "TOTAL ENERGY|abnormal|ERROR" \
        | head -3 \
        || { echo "ERROR: g-xTB smoke test produced no recognisable output" >&2; exit 1; }
    rm -rf "${TMP}"
fi

echo
echo "=========================================="
echo "g-xTB updated to ${new_sha}"
echo "  binary  : ${BIN}"
echo "  qcb cfg : qcb.config.GXTB_BIN already points here automatically"
echo "Commit:  git -C $(cd "${SCRIPT_DIR}/.." && pwd) add deps/g-xtb && git commit -m 'bump g-xtb to ${new_sha}'"
echo "=========================================="
