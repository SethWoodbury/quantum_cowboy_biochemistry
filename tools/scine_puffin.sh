#!/usr/bin/env bash
# scine_puffin.sh — start / stop / status the SCINE Puffin worker daemon.
#
# Puffin reads calculation jobs from MongoDB (deposited by Chemoton's
# SteeringWheel) and dispatches them to the configured calculator backend
# (xtb / sparrow / orca / etc.). Without a running Puffin, the SteeringWheel
# hangs forever waiting for results that never arrive.
#
# Usage:
#   scine_puffin.sh start <mongo_uri> [<n_workers>] [<sif_path>]
#   scine_puffin.sh stop <mongo_uri>
#   scine_puffin.sh status <mongo_uri>
#
# Examples:
#   scine_puffin.sh start mongodb://127.0.0.1:27017/pte_run01 4
#   scine_puffin.sh stop  mongodb://127.0.0.1:27017/pte_run01
#
# Implementation notes:
# - Puffin uses a YAML config file `puffin.yaml` to know which DB to poll
#   and which calculator backends are available.
# - When --apptainer-image is set, the puffin worker process runs INSIDE
#   apptainer (via apptainer instance start), so it survives the launching
#   shell exiting.
# - PID/lock files: $RUNDIR/puffin_${PORT}.pid, $RUNDIR/puffin_${PORT}.lock

set -euo pipefail
[ "${SCINE_PUFFIN_VERBOSE:-0}" = "1" ] && set -x

CMD="${1:?usage: scine_puffin.sh <start|stop|status> <mongo_uri> [args]}"
URI="${2:?mongo uri required, e.g. mongodb://127.0.0.1:27017/pte_run01}"

# Default container path if invoked without one
SIF_DEFAULT="/net/software/containers/users/woodbuse/quantum_chem/quantum_chem-20260506.sif"
SIF="${SCINE_PUFFIN_SIF:-${4:-$SIF_DEFAULT}}"
N_WORKERS="${3:-1}"
RUNDIR="${SCINE_PUFFIN_RUNDIR:-/net/scratch/$USER/scine_puffin}"
mkdir -p "$RUNDIR"

# Hash the URI to derive a stable instance name + pidfile per run
URI_HASH=$(echo -n "$URI" | sha256sum | head -c 8)
INSTANCE="puffin_${URI_HASH}"
PIDFILE="$RUNDIR/puffin_${URI_HASH}.pid"
LOGFILE="$RUNDIR/puffin_${URI_HASH}.log"
CONFIG="$RUNDIR/puffin_${URI_HASH}.yaml"

write_config() {
    cat > "$CONFIG" <<YAML
mongo:
  uri: "$URI"
runtime:
  worker_count: $N_WORKERS
  log_dir: "$RUNDIR/logs_${URI_HASH}"
  poll_interval_seconds: 2
calculator_settings:
  programs:
    - xtb
YAML
    echo "[puffin] config written → $CONFIG"
}

case "$CMD" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "[puffin] already running (pid=$(cat "$PIDFILE"))"
        exit 0
    fi
    write_config
    mkdir -p "$RUNDIR/logs_${URI_HASH}"
    if [ -f "$SIF" ]; then
        echo "[puffin] starting via apptainer instance ($SIF)"
        # Use apptainer instance so the daemon survives this shell exiting
        apptainer instance start --bind /home --bind /net "$SIF" "$INSTANCE" \
            >> "$LOGFILE" 2>&1
        # Run puffin runner inside the instance
        nohup apptainer exec instance://"$INSTANCE" \
            python -m scine_puffin --config "$CONFIG" \
            >> "$LOGFILE" 2>&1 &
        echo $! > "$PIDFILE"
        sleep 2
        echo "[puffin] launched pid=$(cat "$PIDFILE") instance=$INSTANCE"
        echo "[puffin] log → $LOGFILE"
    else
        echo "[puffin] no SIF found; trying host scine_puffin (may fail if not in PATH)"
        nohup python -m scine_puffin --config "$CONFIG" >> "$LOGFILE" 2>&1 &
        echo $! > "$PIDFILE"
        sleep 2
        echo "[puffin] launched pid=$(cat "$PIDFILE") (host mode)"
    fi
    # Trap signals: clean up on Ctrl-C if foregrounded, else require explicit stop
    trap "echo '[puffin] caught signal, stopping...'; bash $0 stop $URI; exit 130" INT TERM
    ;;
  stop)
    if [ ! -f "$PIDFILE" ]; then
        echo "[puffin] no pidfile; not running?"
        exit 0
    fi
    PID="$(cat "$PIDFILE")"
    echo "[puffin] stopping pid=$PID instance=$INSTANCE"
    kill "$PID" 2>/dev/null || true
    sleep 1
    apptainer instance list 2>/dev/null | grep -q "$INSTANCE" \
        && apptainer instance stop "$INSTANCE" || true
    rm -f "$PIDFILE"
    echo "[puffin] stopped"
    ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "[puffin] RUNNING pid=$(cat "$PIDFILE") instance=$INSTANCE log=$LOGFILE"
        tail -3 "$LOGFILE" 2>&1 | sed 's/^/[puffin] /'
        exit 0
    else
        echo "[puffin] NOT RUNNING"
        exit 1
    fi
    ;;
  *)
    echo "usage: $0 <start|stop|status> <mongo_uri> [n_workers] [sif_path]"
    exit 1
    ;;
esac
