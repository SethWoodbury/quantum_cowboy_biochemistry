#!/usr/bin/env bash
# SCINE Chemoton + ReaDuct install (Baker Lab QCB).
#
# Strategy: pip install pinned wheels into the qcb-xtb conda env. The
# vendored submodules (deps/scine_chemoton, deps/scine_readuct) are
# source-of-truth tracking only — we do NOT build from them; PyPI wheels
# are the official artefact and conda-forge is 3 years stale (2.2.0).
#
# Heads up:
#  * Linux-64 wheels only. Will fail on macOS dev boxes.
#  * Chemoton's reaction-network exploration uses MongoDB (port 27017).
#    Optional — mechanism-network runs need `mongod`; single-step
#    ReaDuct calls don't. Start mongo via the upstream
#    docker-compose.yml when needed.
#  * SCINE Puffin (the calculation engine) is a separate package; we
#    install Sparrow as the default backend below. xTB / ORCA / Gaussian
#    bridges live in scine-utilities and need their respective binaries.
#
# Usage:
#   bash deps/build_scine.sh                            # install into qcb-xtb
#   QCB_INSTALL_LAB_SHARED=1 bash deps/build_scine.sh   # also write a
#                                                       # /net/software/lab pin
set -euo pipefail

SCINE_CHEMOTON_VERSION="${SCINE_CHEMOTON_VERSION:-4.1.0}"
SCINE_READUCT_VERSION="${SCINE_READUCT_VERSION:-6.1.0}"
SCINE_SPARROW_VERSION="${SCINE_SPARROW_VERSION:-5.1.0}"
ENV_PREFIX="${ENV_PREFIX:-/home/woodbuse/conda/envs/qcb-xtb}"

if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
    echo "ERROR: ${ENV_PREFIX}/bin/python not found." >&2
    exit 1
fi

echo "==> Installing SCINE Chemoton ${SCINE_CHEMOTON_VERSION} + ReaDuct ${SCINE_READUCT_VERSION} + Sparrow ${SCINE_SPARROW_VERSION} into ${ENV_PREFIX}…"
"${ENV_PREFIX}/bin/python" -m pip install --upgrade \
    "scine-chemoton==${SCINE_CHEMOTON_VERSION}" \
    "scine-readuct==${SCINE_READUCT_VERSION}" \
    "scine-sparrow==${SCINE_SPARROW_VERSION}" \
    || {
        echo "WARN: pinned versions failed; retrying without pins (will pick latest compatible)…"
        "${ENV_PREFIX}/bin/python" -m pip install --upgrade \
            scine-chemoton scine-readuct scine-sparrow
    }

echo "==> Verifying imports…"
"${ENV_PREFIX}/bin/python" - <<'PY'
import scine_chemoton, scine_readuct
print("chemoton:", scine_chemoton.__version__)
print("readuct:", scine_readuct.__version__)
try:
    import scine_sparrow
    print("sparrow:", scine_sparrow.__version__)
except Exception as e:
    print("sparrow: not loadable ({}); xTB/ORCA backends still usable.".format(e))
PY

if [ "${QCB_INSTALL_LAB_SHARED:-0}" = "1" ]; then
    LAB_DIR="/net/software/lab/scine"
    mkdir -p "${LAB_DIR}"
    cat > "${LAB_DIR}/INSTALL.txt" <<EOF
SCINE Chemoton ${SCINE_CHEMOTON_VERSION} + ReaDuct ${SCINE_READUCT_VERSION}.
Lab-wide install: pip wheels live in each user's qcb-xtb env.
Pull-up command:
    pip install scine-chemoton==${SCINE_CHEMOTON_VERSION} scine-readuct==${SCINE_READUCT_VERSION} scine-sparrow==${SCINE_SPARROW_VERSION}
Source-of-truth git checkouts:
    deps/scine_chemoton (qcscine/chemoton)
    deps/scine_readuct  (qcscine/readuct)
EOF
    echo "Lab-shared note written → ${LAB_DIR}/INSTALL.txt"
fi

echo
echo "=========================================="
echo "SCINE install complete."
echo "Try:  ${ENV_PREFIX}/bin/python -m scine_chemoton --help"
echo "      ${ENV_PREFIX}/bin/python -m scine_readuct --help"
echo "=========================================="
