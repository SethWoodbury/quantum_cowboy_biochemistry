#!/usr/bin/env bash
# scine_mongo.sh — MongoDB lifecycle helper for SCINE Chemoton + Puffin
# under SLURM. Per-user, per-node mongod with lockfile + verbose logging.
#
# Why this exists:
#   * Chemoton's reaction-network DB is mandatory MongoDB (no SQLite
#     fallback). Each compute node typically wants its own mongod —
#     shared head-node mongod risks write contention / network latency.
#   * SLURM nodes are ephemeral; we need a launcher we can call from
#     `srun` / sbatch prologs that survives node churn cleanly.
#   * Some SLURM partitions don't have mongod on the host PATH. Toggle
#     `--apptainer-image PATH` and we exec mongod inside a SIF instead.
#
# Verbs:
#   start <dbpath> <port>   Fork mongod, write log + pidfile.
#   stop  <port>            Kill mongod cleanly via the pidfile.
#   status <port>           Echo running/stopped + connection string.
#
# Lockfile pattern: $dbpath/mongod.pid + $dbpath/mongod.lock + $dbpath/mongod.log
#
# Examples:
#   tools/scine_mongo.sh start /scratch/$USER/scine_db 27201
#   tools/scine_mongo.sh status 27201
#   tools/scine_mongo.sh stop   27201
#
#   # Inside the qcb container (mongod baked in at /usr/local/bin/mongod):
#   tools/scine_mongo.sh \
#       --apptainer-image /net/software/containers/users/woodbuse/quantum_chem/quantum_chem-20260506.sif \
#       start /scratch/$USER/scine_db 27201
#
# Env knobs:
#   SCINE_MONGO_VERBOSE=1    set -x for very loud bash tracing
#   SCINE_MONGO_BIND_IP      override default bind_ip (default 127.0.0.1)

set -euo pipefail

# ── flag parse ────────────────────────────────────────────────────────────
APPTAINER_IMAGE=""
BIND_IP="${SCINE_MONGO_BIND_IP:-127.0.0.1}"
VERBOSE="${SCINE_MONGO_VERBOSE:-0}"

while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --apptainer-image)
            APPTAINER_IMAGE="$2"
            shift 2
            ;;
        --bind-ip)
            BIND_IP="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE=1
            shift
            ;;
        --help|-h)
            sed -n '1,40p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: unknown flag $1" >&2
            exit 2
            ;;
    esac
done

[[ "$VERBOSE" = "1" ]] && set -x

VERB="${1:-}"
shift || true

log() { printf "[scine_mongo %s] %s\n" "$(date +%H:%M:%S)" "$*" >&2; }

# Build the (host-side) name of the apptainer instance we use for the
# persistent mongod, derived from the port so multiple ports can coexist.
instance_name() { echo "scine_mongo_${1}"; }

# Wrap a mongod invocation so it runs either bare (host has mongod) or
# inside the apptainer image (host does not).
#
# Important: when running inside apptainer, we need the mongod process
# to OUTLIVE the apptainer exec call. `apptainer exec ... mongod --fork`
# does NOT achieve this — the squashfuse mount is torn down as soon as
# exec returns and the forked daemon is killed. The robust pattern is:
#   apptainer instance start IMG NAME    # persistent rootfs mount
#   apptainer exec instance://NAME mongod --fork ...
#   apptainer instance stop NAME         # tears it all down
# That's what we do below.
run_mongod() {
    local DBPATH="$1"
    shift
    if [[ -n "$APPTAINER_IMAGE" ]]; then
        local INST
        INST=$(instance_name "$PORT")
        # Start the instance if it isn't running yet. List existing
        # instances and grep for ours; tolerate missing `instance list`
        # output (apptainer >= 1.0 has this).
        if ! apptainer instance list 2>/dev/null | awk '{print $1}' | grep -qx "$INST"; then
            log "apptainer instance start: $INST"
            # Bind the dbpath AND its parent (so callers can drop their
            # own python driver next to it and exec it inside the same
            # instance — the smoke harness does this). Also bind /tmp
            # and /home so SLURM scratch / repo paths are visible.
            local DBPARENT
            DBPARENT=$(dirname "$DBPATH")
            apptainer instance start \
                --cleanenv --no-home \
                --bind "$DBPARENT:$DBPARENT" \
                --bind "/tmp:/tmp" \
                --bind "/home:/home" \
                "$APPTAINER_IMAGE" "$INST"
        fi
        # Now exec mongod inside the running instance. Because the
        # instance is the one that holds the squashfuse mount, mongod
        # survives this exec returning.
        apptainer exec "instance://$INST" mongod "$@"
    else
        mongod "$@"
    fi
}

# Same wrapper but for one-shot mongosh / mongo client commands. Uses
# the running instance if available (avoids re-mounting overhead).
run_mongosh() {
    if [[ -n "$APPTAINER_IMAGE" ]]; then
        local INST
        INST=$(instance_name "$PORT")
        if apptainer instance list 2>/dev/null | awk '{print $1}' | grep -qx "$INST"; then
            apptainer exec "instance://$INST" mongosh "$@"
        else
            apptainer exec --cleanenv --no-home "$APPTAINER_IMAGE" mongosh "$@"
        fi
    else
        mongosh "$@"
    fi
}

# Tear down the apptainer instance once mongod has exited. Tolerates
# the instance not existing (e.g. host-mode usage).
stop_instance() {
    if [[ -n "$APPTAINER_IMAGE" ]]; then
        local INST
        INST=$(instance_name "$PORT")
        if apptainer instance list 2>/dev/null | awk '{print $1}' | grep -qx "$INST"; then
            log "apptainer instance stop: $INST"
            apptainer instance stop "$INST" || true
        fi
    fi
}

case "$VERB" in
    # ── start ────────────────────────────────────────────────────────────
    start)
        DBPATH="${1:?missing dbpath}"
        PORT="${2:?missing port}"
        mkdir -p "$DBPATH"

        PIDFILE="$DBPATH/mongod.pid"
        LOGFILE="$DBPATH/mongod.log"

        if [[ -f "$PIDFILE" ]]; then
            existing=$(cat "$PIDFILE" 2>/dev/null || echo "")
            if [[ -n "$existing" ]] && kill -0 "$existing" 2>/dev/null; then
                log "mongod already running (pid=$existing) on $DBPATH; aborting start"
                exit 3
            else
                log "stale pidfile at $PIDFILE; removing"
                rm -f "$PIDFILE"
            fi
        fi

        log "starting mongod: dbpath=$DBPATH port=$PORT bind_ip=$BIND_IP"
        log "  log -> $LOGFILE"
        log "  pid -> $PIDFILE"
        if [[ -n "$APPTAINER_IMAGE" ]]; then
            log "  apptainer-image=$APPTAINER_IMAGE"
        fi

        # --fork + --logpath + --pidfilepath = standard "well-behaved
        # daemon" recipe; mongod will return immediately with the
        # pidfile populated. Use --bind_ip 127.0.0.1 by default for
        # security (per-node access only).
        run_mongod "$DBPATH" \
            --fork \
            --logpath "$LOGFILE" \
            --dbpath "$DBPATH" \
            --port "$PORT" \
            --bind_ip "$BIND_IP" \
            --pidfilepath "$PIDFILE"

        # Belt and suspenders: re-confirm pidfile exists.
        if [[ ! -f "$PIDFILE" ]]; then
            log "ERROR: mongod started but no pidfile at $PIDFILE; check $LOGFILE"
            exit 4
        fi

        log "mongod up; pid=$(cat "$PIDFILE"); URI=mongodb://$BIND_IP:$PORT"
        ;;

    # ── stop ─────────────────────────────────────────────────────────────
    stop)
        PORT="${1:?missing port}"
        # We need a dbpath to find the pidfile; users can pass it as the
        # second arg. If not given, try a heuristic: search nearby for
        # a pidfile referring to this port. For SLURM that's overkill —
        # require dbpath as second arg or fall back to `pgrep -f`.
        DBPATH="${2:-}"
        if [[ -n "$DBPATH" && -f "$DBPATH/mongod.pid" ]]; then
            pid=$(cat "$DBPATH/mongod.pid")
            log "stopping mongod (pid=$pid) via pidfile"
        else
            # Heuristic: pgrep for mongod on the requested port.
            pid=$(pgrep -f "mongod.* --port $PORT( |$)" || echo "")
            if [[ -z "$pid" ]]; then
                log "no mongod found on port $PORT; nothing to stop"
                exit 0
            fi
            log "stopping mongod (pid=$pid) via pgrep heuristic"
        fi

        kill -TERM "$pid" 2>/dev/null || true
        # Give it up to 30 s to flush + close.
        stopped=0
        for i in $(seq 1 30); do
            if ! kill -0 "$pid" 2>/dev/null; then
                log "mongod stopped cleanly"
                stopped=1
                break
            fi
            sleep 1
        done
        if [[ "$stopped" = "0" ]]; then
            log "mongod did not stop within 30s; sending SIGKILL"
            kill -9 "$pid" 2>/dev/null || true
        fi
        [[ -n "$DBPATH" ]] && rm -f "$DBPATH/mongod.pid" "$DBPATH/mongod.lock"
        # Tear down the apptainer instance (no-op if running host-side).
        stop_instance
        ;;

    # ── status ───────────────────────────────────────────────────────────
    status)
        PORT="${1:?missing port}"
        DBPATH="${2:-}"
        running=0
        if [[ -n "$DBPATH" && -f "$DBPATH/mongod.pid" ]]; then
            pid=$(cat "$DBPATH/mongod.pid")
            if kill -0 "$pid" 2>/dev/null; then
                running=1
                echo "running pid=$pid uri=mongodb://$BIND_IP:$PORT dbpath=$DBPATH"
            fi
        fi
        if [[ "$running" = "0" ]]; then
            pid=$(pgrep -f "mongod.* --port $PORT( |$)" || echo "")
            if [[ -n "$pid" ]]; then
                echo "running pid=$pid uri=mongodb://$BIND_IP:$PORT (no dbpath given)"
            else
                echo "stopped"
                exit 1
            fi
        fi
        ;;

    "")
        sed -n '1,40p' "$0"
        exit 2
        ;;
    *)
        echo "ERROR: unknown verb '$VERB' (expected start|stop|status)" >&2
        exit 2
        ;;
esac
