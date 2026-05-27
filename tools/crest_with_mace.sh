#!/bin/bash
# crest_with_mace.sh — run CREST 3.x with MACE forces via generic_sc + daemon.
#
# Architecture:
#   1. Start mace_engrad_daemon.py in the background; capture PID + socket.
#   2. Wait for the daemon to print "READY" to its log.
#   3. Generate (or use a user-supplied) TOML config that points CREST at
#      mace_engrad_client.py as a "method=generic" calculator.
#   4. Run CREST with that TOML.
#   5. After CREST exits, kill the daemon and clean up the socket.
#
# Usage:
#   crest_with_mace.sh INPUT.xyz [--model mace-mp] [--device cuda] \
#                      [--charge 0] [--spin 0] [--workdir DIR] \
#                      [--toml FILE] [--logfile FILE] -- [crest extra args]
#
# Anything after a literal '--' is forwarded directly to crest.
#
# Environment overrides:
#   QCB_PYTHON     python interpreter for daemon/client (default: same dir as crest's env)
#   QCB_CREST_BIN  path to the crest binary (default: deps/crest/install/bin/crest)
#
# This wrapper is ADDITIVE: existing xtb-CREST workflows are unaffected.
#
# Exit codes:
#   0  CREST finished cleanly
#   1  argument / setup error
#   2  daemon failed to start
#   3  CREST returned non-zero
set -euo pipefail

# ---------------------------------------------------------------------------
# Constants exposed as variables (avoid hidden magic numbers)
# ---------------------------------------------------------------------------
DAEMON_READY_TIMEOUT_S="${QCB_MACE_DAEMON_READY_TIMEOUT_S:-180}"   # max wait for daemon "READY"
DAEMON_SHUTDOWN_GRACE_S="${QCB_MACE_DAEMON_SHUTDOWN_GRACE_S:-5}"   # SIGTERM → SIGKILL grace
DAEMON_PROTOCOL_TIMEOUT_S="${QCB_MACE_PROTOCOL_TIMEOUT_S:-120}"     # per-call deadline

# ---------------------------------------------------------------------------
# Locate companions
# ---------------------------------------------------------------------------
TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TOOLS_DIR/.." && pwd)"
DAEMON_PY="$TOOLS_DIR/mace_engrad_daemon.py"
CLIENT_PY="$TOOLS_DIR/mace_engrad_client.py"

QCB_PYTHON="${QCB_PYTHON:-$(command -v python)}"
CREST_BIN="${QCB_CREST_BIN:-$REPO_ROOT/deps/crest/install/bin/crest}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
INPUT_XYZ=""
MODEL="mace-mp"
DEVICE="cuda"
DTYPE="float64"
HEAD=""
CHARGE=0
SPIN=0
WORKDIR=""
TOML_OUT=""
LOGFILE=""
CREST_EXTRA=()

print_usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
}

need_value() {
  # Helper: ensure that --flag has an argument; otherwise show usage and exit.
  if [[ $# -lt 2 ]]; then
    echo "ERROR: $1 requires an argument" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)        print_usage; exit 0 ;;
    --model)          need_value "$@"; MODEL="$2"; shift 2 ;;
    --device)         need_value "$@"; DEVICE="$2"; shift 2 ;;
    --dtype)          need_value "$@"; DTYPE="$2"; shift 2 ;;
    --head)           need_value "$@"; HEAD="$2"; shift 2 ;;
    --charge)         need_value "$@"; CHARGE="$2"; shift 2 ;;
    --spin)           need_value "$@"; SPIN="$2"; shift 2 ;;
    --workdir)        need_value "$@"; WORKDIR="$2"; shift 2 ;;
    --toml)           need_value "$@"; TOML_OUT="$2"; shift 2 ;;
    --logfile)        need_value "$@"; LOGFILE="$2"; shift 2 ;;
    --)               shift; CREST_EXTRA=("$@"); break ;;
    -*)               echo "Unknown flag: $1" >&2; exit 1 ;;
    *)
      if [[ -z "$INPUT_XYZ" ]]; then
        INPUT_XYZ="$1"
      else
        # Anything past the first positional (and before --) is treated as CREST extras
        CREST_EXTRA+=("$1")
      fi
      shift ;;
  esac
done

# Validate integer-typed flags before they leak into bash arithmetic / TOML / generated shell scripts.
if ! [[ "$CHARGE" =~ ^-?[0-9]+$ ]]; then
  echo "ERROR: --charge must be an integer (got: $CHARGE)" >&2; exit 1
fi
if ! [[ "$SPIN" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --spin must be a non-negative integer (got: $SPIN)" >&2; exit 1
fi

if [[ -z "$INPUT_XYZ" ]]; then
  echo "ERROR: input xyz required" >&2
  print_usage
  exit 1
fi
if [[ ! -f "$INPUT_XYZ" ]]; then
  echo "ERROR: input not found: $INPUT_XYZ" >&2
  exit 1
fi
INPUT_XYZ="$(readlink -f "$INPUT_XYZ")"

if [[ ! -x "$CREST_BIN" ]]; then
  echo "ERROR: crest binary not found at $CREST_BIN" >&2
  echo "  override with QCB_CREST_BIN=/path/to/crest" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Working directory (where CREST runs and the socket lives)
# ---------------------------------------------------------------------------
if [[ -z "$WORKDIR" ]]; then
  WORKDIR="$(mktemp -d -t crest_mace_XXXXXX)"
fi
mkdir -p "$WORKDIR"
WORKDIR="$(readlink -f "$WORKDIR")"
echo "[crest_with_mace] workdir: $WORKDIR" >&2

# ---------------------------------------------------------------------------
# Daemon socket + log
# ---------------------------------------------------------------------------
SOCKET="$WORKDIR/mace_engrad.$$.sock"
DAEMON_LOG="$WORKDIR/mace_daemon.log"
if [[ -n "$LOGFILE" ]]; then
  DAEMON_LOG="$LOGFILE"
fi

# ---------------------------------------------------------------------------
# Start daemon
# ---------------------------------------------------------------------------
DAEMON_CMD=(
  "$QCB_PYTHON" "$DAEMON_PY"
  --model "$MODEL"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --socket "$SOCKET"
  --protocol-timeout-s "$DAEMON_PROTOCOL_TIMEOUT_S"
  --logfile "$DAEMON_LOG"
)
if [[ -n "$HEAD" ]]; then
  DAEMON_CMD+=(--head "$HEAD")
fi

# Define the cleanup function FIRST, then start the daemon, so the trap is
# armed before we have a daemon PID to leak. The trap fires on EXIT (covers
# normal exit and most signal-induced exits) — ${DAEMON_PID:-} and
# ${CREST_PID:-} are empty until the relevant lines run, but by then the
# traps are already armed and skip the signaling for an empty PID.
cleanup_crest() {
  if [[ -n "${CREST_PID:-}" ]] && kill -0 "$CREST_PID" 2>/dev/null; then
    echo "[crest_with_mace] forwarding TERM to CREST pid=$CREST_PID" >&2
    kill -TERM "$CREST_PID" 2>/dev/null || true
    # Wait up to DAEMON_SHUTDOWN_GRACE_S for CREST to finish flushing.
    for ((i=0; i<DAEMON_SHUTDOWN_GRACE_S*10; i++)); do
      if ! kill -0 "$CREST_PID" 2>/dev/null; then break; fi
      sleep 0.1
    done
    if kill -0 "$CREST_PID" 2>/dev/null; then
      echo "[crest_with_mace] CREST didn't exit; SIGKILL" >&2
      kill -KILL "$CREST_PID" 2>/dev/null || true
    fi
  fi
}
cleanup_daemon() {
  if [[ -n "${DAEMON_PID:-}" ]] && kill -0 "$DAEMON_PID" 2>/dev/null; then
    echo "[crest_with_mace] stopping daemon pid=$DAEMON_PID" >&2
    kill -TERM "$DAEMON_PID" 2>/dev/null || true
    for ((i=0; i<DAEMON_SHUTDOWN_GRACE_S*10; i++)); do
      if ! kill -0 "$DAEMON_PID" 2>/dev/null; then break; fi
      sleep 0.1
    done
    if kill -0 "$DAEMON_PID" 2>/dev/null; then
      echo "[crest_with_mace] daemon didn't exit; SIGKILL" >&2
      kill -KILL "$DAEMON_PID" 2>/dev/null || true
    fi
  fi
  if [[ -n "${SOCKET:-}" ]]; then
    rm -f "$SOCKET"
  fi
}
cleanup() {
  local rc=$?
  cleanup_crest
  cleanup_daemon
  return "$rc"
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM HUP

echo "[crest_with_mace] starting daemon: ${DAEMON_CMD[*]}" >&2
"${DAEMON_CMD[@]}" > "$DAEMON_LOG.stdout" 2>"$DAEMON_LOG.stderr" &
DAEMON_PID=$!
echo "[crest_with_mace] daemon pid=$DAEMON_PID socket=$SOCKET" >&2

# ---------------------------------------------------------------------------
# Wait for daemon "READY" AND socket binding (both required)
# ---------------------------------------------------------------------------
echo "[crest_with_mace] waiting up to ${DAEMON_READY_TIMEOUT_S}s for daemon READY..." >&2
ready=0
for ((i=0; i<DAEMON_READY_TIMEOUT_S*10; i++)); do
  if grep -q "READY" "$DAEMON_LOG.stderr" 2>/dev/null && [[ -S "$SOCKET" ]]; then
    echo "[crest_with_mace] daemon ready after $((i/10)) s" >&2
    ready=1
    break
  fi
  if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
    echo "ERROR: daemon died before becoming ready. Tail of stderr:" >&2
    tail -30 "$DAEMON_LOG.stderr" >&2 || true
    exit 2
  fi
  sleep 0.1
done
if [[ "$ready" -ne 1 ]]; then
  echo "ERROR: daemon did not become ready within ${DAEMON_READY_TIMEOUT_S}s" >&2
  echo "       (need READY token in $DAEMON_LOG.stderr AND socket $SOCKET)" >&2
  tail -30 "$DAEMON_LOG.stderr" >&2 || true
  exit 2
fi

# ---------------------------------------------------------------------------
# Generate (or use) TOML config for CREST 3
# ---------------------------------------------------------------------------
# Wrapper script that CREST will exec per force eval; CREST passes the input
# xyz path as $1. We use `printf '%q'` to shell-escape every interpolated
# value so paths with spaces/quotes/backticks/$() are safe.
RUNNER_SH="$WORKDIR/run_mace.sh"
{
  printf '#!/bin/bash\n'
  printf '# Auto-generated by crest_with_mace.sh; called by CREST per force evaluation.\n'
  printf 'exec %q %q "$1" --socket %q --charge %q --spin %q --quiet\n' \
    "$QCB_PYTHON" "$CLIENT_PY" "$SOCKET" "$CHARGE" "$SPIN"
} > "$RUNNER_SH"
chmod +x "$RUNNER_SH"

if [[ -z "$TOML_OUT" ]]; then
  TOML_OUT="$WORKDIR/crest_mace.toml"
fi

# TOML basic-string escaping: backslash and double-quote must be escaped.
# Newlines, tabs, etc. would also need handling but file paths shouldn't
# contain those (we'd reject those at input). Refuse paths with control
# characters to keep the TOML well-formed.
toml_escape() {
  local s="$1"
  if [[ "$s" =~ [[:cntrl:]] ]]; then
    echo "ERROR: path contains a control character; refusing to write TOML: $s" >&2
    exit 1
  fi
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  printf '%s' "$s"
}

INPUT_XYZ_ESC="$(toml_escape "$INPUT_XYZ")"
WORKDIR_ESC="$(toml_escape "$WORKDIR")"
RUNNER_SH_ESC="$(toml_escape "$RUNNER_SH")"

cat > "$TOML_OUT" <<TOML
# CREST 3 input file generated by crest_with_mace.sh
input = "$INPUT_XYZ_ESC"
threads = 1

[calculation]
elog = "$WORKDIR_ESC/energies.log"

[[calculation.level]]
method = "generic"
binary = "$RUNNER_SH_ESC"
gradtype = "engrad"
chrg = $CHARGE
uhf = $SPIN
TOML

echo "[crest_with_mace] TOML config:" >&2
cat "$TOML_OUT" >&2
echo "---" >&2

# ---------------------------------------------------------------------------
# Run CREST
# ---------------------------------------------------------------------------
CREST_CMD=("$CREST_BIN" --input "$TOML_OUT")
if [[ ${#CREST_EXTRA[@]} -gt 0 ]]; then
  CREST_CMD+=("${CREST_EXTRA[@]}")
fi

echo "[crest_with_mace] running: ${CREST_CMD[*]}" >&2
echo "[crest_with_mace] cwd: $WORKDIR" >&2
T0=$(date +%s)

# Run CREST in the background so the wrapper shell can forward signals via
# the cleanup() trap installed before the daemon spawn. CREST_PID becomes
# visible to cleanup_crest() once we set it below.
set +e
( cd "$WORKDIR" && "${CREST_CMD[@]}" ) &
CREST_PID=$!

# `wait` is interrupted by signals; the trap fires, runs cleanup (which
# kills CREST + daemon), and exits 130. This handles INT/TERM/HUP. On
# normal exit, wait returns CREST's exit status which we then propagate.
wait "$CREST_PID"
CREST_RC=$?
set -e
T1=$(date +%s)
echo "[crest_with_mace] CREST exited with $CREST_RC after $((T1-T0))s" >&2

if [[ "$CREST_RC" -ne 0 ]]; then
  exit 3
fi
exit 0
