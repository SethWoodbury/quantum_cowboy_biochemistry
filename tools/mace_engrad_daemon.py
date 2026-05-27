#!/usr/bin/env python
"""mace_engrad_daemon.py — long-running MACE force/energy server for CREST.

Architecture
------------
CREST 3.x ``generic_sc`` calculator works by spawning a subprocess for every
energy/gradient call. Naively wrapping MACE in Python would re-load the
neural-net weights (~10-30 s for mace-mp/large) on every call, which makes
real CREST runs impossible.

This daemon loads a single MACE model once at startup, listens on a Unix
domain socket, and answers force/energy requests with a JSON line-protocol
roundtrip. Per-call wrapper (``mace_engrad_client.py``) does the socket
exchange and writes the ORCA ``.engrad`` file CREST expects.

Wire protocol
-------------
The protocol is **line-oriented JSON**: client writes one JSON object on a
single line terminated with ``\\n``, daemon writes one JSON object on one
line terminated with ``\\n``. This avoids any ambiguity about when a request
or response is complete and lets us reuse the connection for many calls.

Request fields:
    n_atoms : int
    coords  : list of [x, y, z] in Angstrom
    elements: list of element symbols (length == n_atoms)
    charge  : optional int (default 0); set on atoms.info["charge"]
    spin    : optional int (default 0); set on atoms.info["spin"]
    request_id : optional client-side identifier echoed in the response

Response fields:
    ok        : bool
    energy_eh : float (Hartree)
    gradient_eh_per_bohr : list of [gx, gy, gz]
    error     : null on ok, otherwise a string
    request_id: same as request, if provided
    elapsed_s : wall-time of the force eval inside the daemon

Lifecycle
---------
* SIGTERM / SIGINT → close socket, unlink the socket file, exit cleanly.
* The socket is created at ``--socket /tmp/mace_engrad.<PID>.sock`` (default).
  Each concurrent CREST run should use its own daemon (PID disambiguates).
* Prints ``READY`` to stderr once the socket is bound and the model is
  loaded; the wrapper bash script polls for this token (or just for the
  socket file to appear).

Concurrency
-----------
Force calls are serialized through a per-process lock (the model is held by
a single PyTorch process). CREST issues parallel calls in OMP threads; we
service them in the order they arrive. For typical CREST MTD walkers this
is fine because the dominant cost is the MACE forward pass.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Defer the heavy imports (numpy, torch, ase, mace) so that --help doesn't pay
# the load cost.
_calc_lock = threading.Lock()
_calc = None      # ASE calculator (loaded once at startup)
_log = logging.getLogger("mace_daemon")


# ---------------------------------------------------------------------------
# Constants — exposed as CLI flags to avoid hidden magic numbers (per
# project lessons-learned memory; see feedback_no_hardcoded_magic_numbers.md)
# ---------------------------------------------------------------------------
DEFAULT_PROTOCOL_TIMEOUT_S = 120.0   # per-call response deadline
DEFAULT_RECV_BUF = 1 << 20           # 1 MiB recv chunk size
DEFAULT_SOCKET_BACKLOG = 8


def _load_calculator(model: str, device: str, dtype: str, head: str | None) -> Any:
    """Resolve and instantiate a MACE calculator using the qcb factory."""
    # Import here so that --help is fast.
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from quantum_engine.calc.factory import make_calc
        return make_calc(model=model, head=head, device=device, default_dtype=dtype)
    except Exception as e:
        _log.warning("qcb factory failed (%s); falling back to mace_mp direct", e)
        # Fallback: use mace_mp directly so this daemon can run without the
        # repo on PYTHONPATH (useful for ad-hoc debugging or container runs).
        from mace.calculators import mace_mp  # type: ignore
        return mace_mp(model="medium", device=device, default_dtype=dtype)


def _build_atoms(coords, elements, charge: int, spin: int):
    """Construct a fresh Atoms object for a single force call.

    Each request gets a new Atoms instance; we never share state between
    requests, which keeps the threading model trivial (the only shared
    state is the calculator itself, protected by ``_calc_lock``).
    """
    from ase import Atoms
    import numpy as np

    coords_arr = np.asarray(coords, dtype=float)
    if coords_arr.shape != (len(elements), 3):
        raise ValueError(
            f"coords shape {coords_arr.shape} mismatches elements length {len(elements)}"
        )
    atoms = Atoms(symbols=elements, positions=coords_arr)
    atoms.info["charge"] = int(charge)
    atoms.info["spin"] = int(spin)
    return atoms


def _eval(coords, elements, charge: int, spin: int) -> dict:
    """Run a single MACE energy+gradient evaluation. Returns ORCA-engrad units."""
    import math

    import numpy as np

    t0 = time.perf_counter()
    atoms = _build_atoms(coords, elements, charge, spin)
    with _calc_lock:
        atoms.calc = _calc
        e_eV = float(atoms.get_potential_energy())
        f_eV_per_A = atoms.get_forces()  # shape (N, 3)

    # Convert ASE → ORCA engrad units.
    #   energy: eV → Hartree (1 Hartree = 27.211386245988 eV)
    #   gradient = -force, eV/Å → Eh/a0  (1 Eh/a0 = 27.2114 / 0.5291772 eV/Å)
    EV_PER_HARTREE = 27.211386245988
    BOHR_PER_A = 1.0 / 0.529177210903   # Å → Bohr conversion factor
    EV_PER_A_PER_EH_PER_BOHR = EV_PER_HARTREE * BOHR_PER_A  # convert force units

    energy_eh = e_eV / EV_PER_HARTREE
    gradient = -np.asarray(f_eV_per_A) / EV_PER_A_PER_EH_PER_BOHR

    # Validate finite — non-finite forces would corrupt CREST silently.
    if not math.isfinite(energy_eh):
        raise ValueError(f"non-finite energy: {energy_eh}")
    if not np.isfinite(gradient).all():
        raise ValueError("non-finite components in gradient")

    elapsed = time.perf_counter() - t0
    return {
        "ok": True,
        "energy_eh": energy_eh,
        "gradient_eh_per_bohr": gradient.tolist(),
        "error": None,
        "elapsed_s": elapsed,
    }


def _safe_dump(obj) -> bytes:
    """Encode response as strict JSON (no NaN/Infinity leakage onto the wire).

    If the payload contains data that strict JSON can't encode (e.g. a
    NaN echoed back from a hostile/buggy ``request_id``), drop down to a
    minimal hand-built error object so the client always gets exactly one
    newline-terminated JSON line.
    """
    try:
        return json.dumps(obj, allow_nan=False).encode("utf-8") + b"\n"
    except (ValueError, TypeError) as e:
        fallback = {"ok": False, "error": f"response-encode: {e}"}
        return json.dumps(fallback, allow_nan=False).encode("utf-8") + b"\n"


def _sanitize_rid(rid):
    """Return an echo-able request id: only str/int/float-finite values pass through."""
    import math
    if rid is None:
        return None
    if isinstance(rid, str):
        return rid
    if isinstance(rid, int):
        return rid
    if isinstance(rid, float) and math.isfinite(rid):
        return rid
    # Stringify everything else so we can echo it without breaking strict JSON.
    return str(rid)


def _handle_request(line: bytes) -> bytes:
    """Decode a request line, evaluate, encode response."""
    try:
        req = json.loads(line.decode("utf-8"))
    except Exception as e:
        return _safe_dump({"ok": False, "error": f"json-decode: {e}"})

    if not isinstance(req, dict):
        return _safe_dump(
            {"ok": False, "error": f"request must be a JSON object, got {type(req).__name__}"}
        )

    rid = _sanitize_rid(req.get("request_id"))
    try:
        n = int(req["n_atoms"])
        coords = req["coords"]
        elements = req["elements"]
        if len(coords) != n or len(elements) != n:
            raise ValueError(
                f"n_atoms={n} but coords({len(coords)})/elements({len(elements)}) differ"
            )
        charge = int(req.get("charge", 0))
        spin = int(req.get("spin", 0))
    except Exception as e:
        return _safe_dump(
            {"ok": False, "error": f"request-parse: {e}", "request_id": rid}
        )

    try:
        resp = _eval(coords, elements, charge, spin)
    except Exception as e:
        _log.exception("force evaluation failed")
        resp = {"ok": False, "error": f"eval: {e}"}

    if rid is not None:
        resp["request_id"] = rid
    return _safe_dump(resp)


class _LineHandler(socketserver.BaseRequestHandler):
    """Read line-delimited JSON requests until the client closes the socket."""

    def handle(self):
        # Disable Nagle for low-latency single-request roundtrips. Unix
        # sockets ignore TCP_NODELAY but we set timeout for safety.
        self.request.settimeout(self.server.protocol_timeout_s)  # type: ignore[attr-defined]
        buf = b""
        while True:
            try:
                chunk = self.request.recv(DEFAULT_RECV_BUF)
            except socket.timeout:
                _log.warning("client recv timeout")
                return
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                resp = _handle_request(line)
                try:
                    self.request.sendall(resp)
                except OSError:
                    return


class _UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr: str, handler, protocol_timeout_s: float):
        # Make sure stale socket file is not in the way
        try:
            os.unlink(addr)
        except FileNotFoundError:
            pass
        super().__init__(addr, handler)
        self.protocol_timeout_s = protocol_timeout_s


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="mace-mp",
                   help="MACE model alias from qcb factory or absolute path")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                   help="Torch device for MACE")
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"],
                   help="MACE default dtype")
    p.add_argument("--head", default=None,
                   help="Head selector for multi-head MACE (e.g. 'omol' for mace-mh)")
    p.add_argument("--socket", default=None,
                   help="Unix socket path. Default: /tmp/mace_engrad.<PID>.sock")
    p.add_argument("--protocol-timeout-s", type=float,
                   default=DEFAULT_PROTOCOL_TIMEOUT_S,
                   help=f"Per-call recv timeout in seconds (default {DEFAULT_PROTOCOL_TIMEOUT_S}s)")
    p.add_argument("--logfile", default=None,
                   help="Append daemon logs to this file (default: stderr only)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-request log lines")
    args = p.parse_args()

    handlers: list[logging.Handler] = [logging.StreamHandler(stream=sys.stderr)]
    if args.logfile:
        handlers.append(logging.FileHandler(args.logfile, mode="a"))
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="[mace-daemon %(asctime)s %(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )

    sock_path = args.socket or f"/tmp/mace_engrad.{os.getpid()}.sock"
    _log.info("starting; pid=%d socket=%s model=%s device=%s dtype=%s",
              os.getpid(), sock_path, args.model, args.device, args.dtype)

    # ---- early signal handler: clean up if we die before the socket is
    # bound (e.g., during model load). The handler is upgraded once the
    # server is alive so it can also call server.shutdown().
    def _early_shutdown(signo, _frame):
        _log.info("received signal %d during startup, exiting", signo)
        # No socket yet, nothing to unlink. Just exit.
        sys.exit(128 + signo)

    signal.signal(signal.SIGTERM, _early_shutdown)
    signal.signal(signal.SIGINT, _early_shutdown)

    # ---- model load ------------------------------------------------------
    global _calc
    t0 = time.perf_counter()
    _calc = _load_calculator(args.model, args.device, args.dtype, args.head)
    _log.info("MACE model loaded in %.1f s", time.perf_counter() - t0)

    # ---- bind socket -----------------------------------------------------
    server = _UnixServer(sock_path, _LineHandler, args.protocol_timeout_s)
    # Restrict access to the user (Unix sockets respect file-mode bits).
    try:
        os.chmod(sock_path, 0o600)
    except OSError:
        pass

    # ---- now that the server is alive, swap in the running-state handler.
    # NOTE: socketserver.BaseServer.shutdown() blocks until serve_forever()
    # returns and must NOT be called from the same thread that's running
    # serve_forever(). We dispatch the shutdown into a worker thread so the
    # signal handler returns immediately and the main loop can wake up.
    def _shutdown(signo, _frame):
        _log.info("received signal %d, shutting down", signo)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Print READY to stderr (line-flushed) so the wrapper script can detect
    # daemon up. The wrapper greps stderr for this token.
    print("READY", file=sys.stderr, flush=True)
    _log.info("listening on %s — READY", sock_path)

    try:
        server.serve_forever()
    finally:
        try:
            server.server_close()
        except Exception:
            pass
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass
        _log.info("daemon stopped cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
