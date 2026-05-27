#!/usr/bin/env python
"""mace_engrad_client.py — per-call wrapper invoked by CREST's generic_sc.

Invocation (CREST does this for us):
    python mace_engrad_client.py genericinp.xyz [--socket PATH] [--charge N] [--spin N]

Input  : ``genericinp.xyz`` — XMol .xyz file written by CREST (Angstroms).
Output : ``genericinp.engrad`` — ORCA-style energy + gradient file (Hartree, Eh/Bohr).

The wrapper does **no** ML work itself; it forwards a JSON request over a
Unix socket to a long-running MACE daemon (``mace_engrad_daemon.py``) and
writes the response into the format CREST expects.

Exit code: 0 on success, non-zero on any failure (so CREST sees an error).

Why we need this layer:
    The naive approach — having CREST shell out to a Python script that
    loads MACE and computes — re-loads the neural-net weights on every
    single force call. With mace-mp medium that's ~10-30 s overhead per
    call. CREST does thousands of calls, so the overhead is fatal. The
    daemon loads once; the wrapper is microseconds of socket overhead per
    call.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Constants — exposed where reasonable (no hidden magic numbers).
DEFAULT_CONNECT_TIMEOUT_S = 30.0    # initial connect deadline
DEFAULT_REQUEST_TIMEOUT_S = 120.0   # per-call deadline
RECV_BUF = 1 << 20                  # 1 MiB chunks


def parse_xyz(path: Path) -> Tuple[List[str], List[List[float]]]:
    """Parse an XMol .xyz file (Angstroms) → (elements, coords)."""
    with path.open() as fh:
        n = int(fh.readline().strip())
        _comment = fh.readline()  # discarded
        elements: List[str] = []
        coords: List[List[float]] = []
        for i in range(n):
            line = fh.readline()
            if not line:
                raise ValueError(f"unexpected EOF after {i}/{n} atoms in {path}")
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"bad xyz line {i+1} in {path!r}: {line!r}")
            elements.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(elements) != n:
        raise ValueError(f"declared n={n} but read {len(elements)}")
    return elements, coords


def write_engrad(path: Path, energy_eh: float, gradient_eh_per_bohr: List[List[float]]) -> None:
    """Write ORCA .engrad format that CREST's gradreader ``rd_grad_engrad`` parses.

    Format (line-by-line):
        #
        # Number of atoms
        #
            N
        #
        # The current total energy in Eh
        #
              E_HARTREE  (one float, %.15E)
        #
        # The current gradient in Eh/bohr
        #
              gx_atom1
              gy_atom1
              gz_atom1
              gx_atom2
              ...
    The Fortran reader only checks ``#`` for comments and reads N, E, then
    3N floats; the headers are decoration.
    """
    import math

    n = len(gradient_eh_per_bohr)
    if not math.isfinite(energy_eh):
        raise ValueError(f"refusing to write non-finite energy: {energy_eh!r}")
    flat = [v for row in gradient_eh_per_bohr for v in row]
    if any(not math.isfinite(v) for v in flat):
        raise ValueError("refusing to write non-finite gradient component")
    with path.open("w") as fh:
        fh.write("#\n# Number of atoms\n#\n")
        fh.write(f"     {n}\n")
        fh.write("#\n# The current total energy in Eh\n#\n")
        fh.write(f"  {energy_eh:.15E}\n")
        fh.write("#\n# The current gradient in Eh/bohr\n#\n")
        for atom_grad in gradient_eh_per_bohr:
            for component in atom_grad:
                fh.write(f"  {float(component):.15E}\n")


def request_engrad(
    sock_path: str,
    elements: List[str],
    coords: List[List[float]],
    charge: int,
    spin: int,
    *,
    connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
) -> dict:
    """One JSON-line roundtrip over a Unix socket; returns the parsed reply."""
    n = len(elements)
    req = {
        "n_atoms": n,
        "coords": coords,
        "elements": elements,
        "charge": int(charge),
        "spin": int(spin),
        "request_id": f"{os.getpid()}-{int(time.time()*1e6)}",
    }
    payload = (json.dumps(req) + "\n").encode("utf-8")

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(connect_timeout_s)
    try:
        s.connect(sock_path)
    except (FileNotFoundError, ConnectionRefusedError) as e:
        raise RuntimeError(
            f"could not connect to MACE daemon at {sock_path!r}. "
            f"Is the daemon running? ({e})"
        ) from e
    s.settimeout(request_timeout_s)
    try:
        s.sendall(payload)
        # Read until newline.
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(RECV_BUF)
            if not chunk:
                raise RuntimeError("daemon closed connection before response")
            buf += chunk
        line, _ = buf.split(b"\n", 1)
        return json.loads(line.decode("utf-8"))
    finally:
        s.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("input_xyz",
                   help="Path to the input xyz file (CREST writes 'genericinp.xyz')")
    p.add_argument("--socket", default=None,
                   help="Unix socket path of the MACE daemon. "
                        "Default: $MACE_DAEMON_SOCKET")
    p.add_argument("--charge", type=int, default=None,
                   help="System charge. Default: $QCB_CHARGE or 0")
    p.add_argument("--spin", type=int, default=None,
                   help="Number of unpaired electrons (atoms.info['spin']). "
                        "Default: $QCB_SPIN or 0")
    p.add_argument("--out", default=None,
                   help="Engrad output path. Default: <input>.engrad replacing .xyz suffix, "
                        "or 'genericinp.engrad' next to the input file (matching CREST's "
                        "expectation).")
    p.add_argument("--connect-timeout-s", type=float, default=DEFAULT_CONNECT_TIMEOUT_S)
    p.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    p.add_argument("--quiet", action="store_true", help="Suppress timing on stderr")
    args = p.parse_args()

    sock_path = args.socket or os.environ.get("MACE_DAEMON_SOCKET")
    if not sock_path:
        print("ERROR: --socket or $MACE_DAEMON_SOCKET required", file=sys.stderr)
        return 2

    charge = args.charge if args.charge is not None else int(os.environ.get("QCB_CHARGE", "0"))
    spin = args.spin if args.spin is not None else int(os.environ.get("QCB_SPIN", "0"))

    in_path = Path(args.input_xyz)
    if not in_path.is_file():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 3

    if args.out:
        out_path = Path(args.out)
    else:
        # CREST always reads back 'genericinp.engrad' from the same calcspace
        # as the input. Keep the convention: replace .xyz with .engrad.
        out_path = in_path.with_suffix(".engrad")

    t0 = time.perf_counter()
    try:
        elements, coords = parse_xyz(in_path)
        resp = request_engrad(
            sock_path, elements, coords, charge, spin,
            connect_timeout_s=args.connect_timeout_s,
            request_timeout_s=args.request_timeout_s,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 4

    # Validate response is a JSON object with the expected fields. A
    # malformed daemon (e.g. a buggy variant or version-skew) must not
    # silently feed garbage into CREST's optimizer.
    if not isinstance(resp, dict):
        print(
            f"ERROR: daemon response must be a JSON object, got {type(resp).__name__}",
            file=sys.stderr,
        )
        return 7
    if resp.get("ok") is not True:
        print(f"ERROR: daemon returned not-ok: {resp.get('error')}", file=sys.stderr)
        return 5

    grad = resp.get("gradient_eh_per_bohr")
    if not isinstance(grad, list) or len(grad) != len(elements):
        print(
            f"ERROR: malformed daemon response: gradient n_atoms mismatch "
            f"(expected {len(elements)}, got {len(grad) if isinstance(grad, list) else type(grad).__name__})",
            file=sys.stderr,
        )
        return 7
    for i, row in enumerate(grad):
        if not isinstance(row, list) or len(row) != 3:
            print(
                f"ERROR: malformed daemon response: gradient row {i} must be a 3-list",
                file=sys.stderr,
            )
            return 7
    energy_eh = resp.get("energy_eh")
    if not isinstance(energy_eh, (int, float)):
        print(
            f"ERROR: malformed daemon response: energy_eh must be a number, "
            f"got {type(energy_eh).__name__}",
            file=sys.stderr,
        )
        return 7

    try:
        write_engrad(out_path, float(energy_eh), grad)
    except Exception as e:
        print(f"ERROR: writing {out_path}: {e}", file=sys.stderr)
        return 6

    elapsed = time.perf_counter() - t0
    if not args.quiet:
        print(
            f"[mace-client] {in_path.name} -> {out_path.name} "
            f"E={resp['energy_eh']:.6f} Eh  daemon={resp.get('elapsed_s', 0):.3f}s "
            f"total={elapsed:.3f}s",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
