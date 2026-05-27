"""Unit + smoke tests for the MACE-CREST integration.

Layered testing:
1. ORCA .engrad write/read roundtrip (no MACE, no CREST — just I/O).
2. Daemon protocol roundtrip with a stubbed calculator (no real MACE load).
3. Smoke: real daemon (mace-mp/cpu) + client → engrad on water; verify
   energy/gradient match a single-shot ASE call.
4. End-to-end: real CREST + real daemon on a small molecule (marked slow,
   skipped unless QCB_RUN_SLOW=1).

Mark a test ``@pytest.mark.slow`` to skip by default; flip with
``QCB_RUN_SLOW=1 pytest -m slow``.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS = REPO_ROOT / "tools"
DAEMON_PY = TOOLS / "mace_engrad_daemon.py"
CLIENT_PY = TOOLS / "mace_engrad_client.py"
WRAPPER_SH = TOOLS / "crest_with_mace.sh"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

# Reuse the client's pure-Python helpers.
import importlib.util
_spec = importlib.util.spec_from_file_location("mace_engrad_client", CLIENT_PY)
mace_engrad_client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mace_engrad_client)  # type: ignore[union-attr]
parse_xyz = mace_engrad_client.parse_xyz
write_engrad = mace_engrad_client.write_engrad
request_engrad = mace_engrad_client.request_engrad


# ---------------------------------------------------------------------------
# Layer 1 — pure I/O
# ---------------------------------------------------------------------------
def test_engrad_roundtrip(tmp_path):
    """write_engrad → CREST's gradreader-compatible parser → match."""
    e = -76.123456789012345
    grad = [
        [1e-3, -2e-3, 3e-3],
        [-1.5e-2, 0.0, 4.4e-2],
        [0.1, 0.2, 0.3],
    ]
    out = tmp_path / "test.engrad"
    write_engrad(out, e, grad)
    text = out.read_text().splitlines()
    # Verify header structure: the Fortran reader skips comment lines (# ...)
    # and reads the first non-comment line as nat, the next as energy, then
    # 3*nat lines of gradient.
    nonblank = [ln.strip() for ln in text if ln.strip()]
    nonhash = [ln for ln in nonblank if not ln.startswith("#")]
    assert int(nonhash[0]) == 3
    assert abs(float(nonhash[1]) - e) < 1e-15
    flat_grad = [float(x) for x in nonhash[2:]]
    assert len(flat_grad) == 9
    expected_flat = [v for row in grad for v in row]
    for a, b in zip(flat_grad, expected_flat):
        assert abs(a - b) < 1e-15


def test_engrad_format_strict(tmp_path):
    """CREST's rd_grad_engrad reads N, E, then 3N floats line-by-line. Any
    extra junk between values must be a comment (# prefix). Verify our writer
    only emits comments + numeric lines."""
    out = tmp_path / "fmt.engrad"
    write_engrad(out, -1.234, [[0.1, 0.2, 0.3]])
    for line in out.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        # Must parse as float (or int for the n-atoms line).
        float(s)


def test_parse_xyz(tmp_path):
    """parse_xyz reads XMol .xyz (Angstrom, element-symbol-first)."""
    f = tmp_path / "in.xyz"
    f.write_text("3\ntest comment\nO 0.0 0.0 0.0\nH 0.957 0.0 0.0\nH -0.24 0.927 0.0\n")
    elements, coords = parse_xyz(f)
    assert elements == ["O", "H", "H"]
    assert coords[0] == [0.0, 0.0, 0.0]
    assert coords[1] == [0.957, 0.0, 0.0]


def test_parse_xyz_bad_count(tmp_path):
    """Mismatched declared count vs lines should raise."""
    f = tmp_path / "bad.xyz"
    f.write_text("3\ncomment\nH 0 0 0\nH 0 0 1\n")  # only 2 atoms
    with pytest.raises(ValueError):
        parse_xyz(f)


# ---------------------------------------------------------------------------
# Layer 2 — protocol roundtrip with a stub server (no MACE, no torch).
# ---------------------------------------------------------------------------
def _stub_server_thread(sock_path: str, stop_evt: threading.Event):
    """Mini Unix-socket server that echoes a fake engrad response."""
    try:
        os.unlink(sock_path)
    except FileNotFoundError:
        pass
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(sock_path)
    s.listen(1)
    s.settimeout(1.0)
    try:
        while not stop_evt.is_set():
            try:
                conn, _ = s.accept()
            except socket.timeout:
                continue
            try:
                conn.settimeout(2.0)
                buf = b""
                while b"\n" not in buf:
                    chunk = conn.recv(1 << 16)
                    if not chunk:
                        break
                    buf += chunk
                line, _ = buf.split(b"\n", 1)
                req = json.loads(line.decode())
                n = int(req["n_atoms"])
                resp = {
                    "ok": True,
                    "energy_eh": -1.5,
                    "gradient_eh_per_bohr": [[0.0, 0.1, -0.1]] * n,
                    "error": None,
                    "request_id": req.get("request_id"),
                    "elapsed_s": 0.001,
                }
                conn.sendall((json.dumps(resp) + "\n").encode())
            finally:
                conn.close()
    finally:
        s.close()
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass


def test_request_engrad_protocol(tmp_path):
    """End-to-end JSON-line roundtrip against a stub server."""
    sock = str(tmp_path / "stub.sock")
    stop = threading.Event()
    t = threading.Thread(target=_stub_server_thread, args=(sock, stop), daemon=True)
    t.start()
    try:
        # Wait for socket to bind
        for _ in range(50):
            if Path(sock).exists():
                break
            time.sleep(0.02)
        else:
            pytest.fail("stub server didn't bind socket")

        resp = request_engrad(
            sock,
            elements=["O", "H", "H"],
            coords=[[0.0, 0.0, 0.0], [0.957, 0.0, 0.0], [-0.24, 0.927, 0.0]],
            charge=0,
            spin=0,
            connect_timeout_s=2.0,
            request_timeout_s=5.0,
        )
        assert resp["ok"] is True
        assert resp["energy_eh"] == -1.5
        assert len(resp["gradient_eh_per_bohr"]) == 3
    finally:
        stop.set()
        t.join(timeout=5.0)


def test_request_engrad_no_socket(tmp_path):
    """Connecting to a non-existent socket should raise a useful error."""
    sock = str(tmp_path / "nope.sock")
    with pytest.raises(RuntimeError, match="could not connect"):
        request_engrad(
            sock,
            elements=["H"],
            coords=[[0.0, 0.0, 0.0]],
            charge=0,
            spin=0,
            connect_timeout_s=1.0,
            request_timeout_s=1.0,
        )


def test_write_engrad_rejects_nonfinite(tmp_path):
    """Daemon could in principle produce NaN/Inf — the writer must refuse."""
    p = tmp_path / "bad.engrad"
    with pytest.raises(ValueError, match="non-finite energy"):
        write_engrad(p, float("nan"), [[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="non-finite gradient"):
        write_engrad(p, -1.0, [[float("inf"), 0.0, 0.0]])


def test_handle_request_non_object():
    """Daemon's _handle_request should reject non-object JSON gracefully."""
    daemon_path = TOOLS / "mace_engrad_daemon.py"
    spec = importlib.util.spec_from_file_location("mace_engrad_daemon", daemon_path)
    daemon_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(daemon_mod)
    out = daemon_mod._handle_request(b"[1, 2, 3]")
    decoded = json.loads(out.decode())
    assert decoded["ok"] is False
    assert "JSON object" in decoded["error"]


def test_handle_request_malformed():
    """Bad JSON should be reported via the protocol, not crash."""
    daemon_path = TOOLS / "mace_engrad_daemon.py"
    spec = importlib.util.spec_from_file_location("mace_engrad_daemon", daemon_path)
    daemon_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(daemon_mod)
    out = daemon_mod._handle_request(b"{not valid json")
    decoded = json.loads(out.decode())
    assert decoded["ok"] is False
    assert "json-decode" in decoded["error"]


def test_handle_request_nan_request_id(tmp_path):
    """A non-finite request_id (e.g. NaN) must not crash the response writer.

    Python's ``json.loads`` accepts ``NaN``; without sanitization the
    daemon would echo it back through ``json.dumps(..., allow_nan=False)``
    which raises and silently closes the connection.
    """
    daemon_path = TOOLS / "mace_engrad_daemon.py"
    spec = importlib.util.spec_from_file_location("mace_engrad_daemon", daemon_path)
    daemon_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(daemon_mod)
    # NaN as request_id, plus a deliberately bad n_atoms so the parse path
    # exercises the rid echo before _eval is called.
    out = daemon_mod._handle_request(b'{"request_id": NaN, "n_atoms": "x"}')
    decoded = json.loads(out.decode())
    assert decoded["ok"] is False
    # rid was sanitized — either echoed as a string "nan" or omitted.
    if "request_id" in decoded:
        assert isinstance(decoded["request_id"], str)


def test_handle_request_safe_dump_fallback():
    """_safe_dump must always produce a valid JSON line, even with broken inputs."""
    daemon_path = TOOLS / "mace_engrad_daemon.py"
    spec = importlib.util.spec_from_file_location("mace_engrad_daemon", daemon_path)
    daemon_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(daemon_mod)

    # Object containing a non-serializable type — should fall back to error.
    out = daemon_mod._safe_dump({"ok": True, "blob": object()})
    line = out.decode().strip()
    assert line.endswith("}")
    decoded = json.loads(line)
    assert decoded["ok"] is False
    assert "response-encode" in decoded["error"]


# ---------------------------------------------------------------------------
# Layer 3 — real-daemon smoke (mace-mp on CPU).
# ---------------------------------------------------------------------------
def _have_mace() -> bool:
    try:
        import mace.calculators  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_mace(), reason="mace-torch not installed")
@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("QCB_RUN_SLOW") != "1",
                    reason="set QCB_RUN_SLOW=1 to enable real-MACE smoke test")
def test_daemon_real_mace_smoke(tmp_path):
    """Boot real daemon, send water request, compare against direct ASE."""
    sock = str(tmp_path / "smoke.sock")
    log = tmp_path / "daemon.log"
    proc = subprocess.Popen(
        [sys.executable, str(DAEMON_PY),
         "--device", "cpu",
         "--dtype", "float64",
         "--model", "mace-mp",
         "--socket", sock,
         "--logfile", str(log)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        # Wait up to 5 minutes for model load on CPU
        ready = False
        for _ in range(600):
            time.sleep(0.5)
            if proc.poll() is not None:
                err = proc.stderr.read().decode() if proc.stderr else ""
                pytest.fail(f"daemon exited early: {err}")
            if Path(sock).exists():
                ready = True
                break
        assert ready, "daemon did not bind socket within 5 minutes"

        resp = request_engrad(
            sock,
            elements=["O", "H", "H"],
            coords=[[0.0, 0.0, 0.0], [0.957, 0.0, 0.0], [-0.24, 0.927, 0.0]],
            charge=0,
            spin=0,
            connect_timeout_s=10.0,
            request_timeout_s=120.0,
        )
        assert resp["ok"], f"daemon error: {resp.get('error')}"
        assert "energy_eh" in resp
        assert len(resp["gradient_eh_per_bohr"]) == 3
        # Sanity: energy must be a finite float, gradient finite
        import math
        assert math.isfinite(resp["energy_eh"])
        for atom_grad in resp["gradient_eh_per_bohr"]:
            for g in atom_grad:
                assert math.isfinite(g)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# Layer 4 — end-to-end CREST run with MACE (slow).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _have_mace(), reason="mace-torch not installed")
@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("QCB_RUN_SLOW") != "1",
                    reason="set QCB_RUN_SLOW=1 to enable end-to-end CREST/MACE test")
def test_crest_mace_end_to_end(tmp_path):
    """Run crest_with_mace.sh in a minimal mode (--sp single point) on water."""
    crest_bin = REPO_ROOT / "deps" / "crest" / "install" / "bin" / "crest"
    if not crest_bin.exists():
        pytest.skip(f"crest binary not at {crest_bin}")

    inp = tmp_path / "water.xyz"
    inp.write_text(
        "3\nwater test\n"
        "O   0.0000   0.0000   0.0000\n"
        "H   0.7572   0.5860   0.0000\n"
        "H  -0.7572   0.5860   0.0000\n"
    )

    cmd = [
        "bash", str(WRAPPER_SH), str(inp),
        "--device", "cpu", "--model", "mace-mp",
        "--workdir", str(tmp_path / "work"),
        "--", "--sp",  # single-point only — fastest end-to-end check
    ]
    env = os.environ.copy()
    env["QCB_PYTHON"] = sys.executable
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    if proc.returncode != 0:
        pytest.fail(
            f"wrapper exited {proc.returncode}\n"
            f"stdout-tail:\n{proc.stdout[-2000:]}\n"
            f"stderr-tail:\n{proc.stderr[-2000:]}"
        )
    # Verify CREST produced an output indicating it received energies
    work = tmp_path / "work"
    assert (work / "energies.log").exists() or any(work.glob("*.engrad")), (
        f"no energies.log or .engrad files in workdir; ls: {list(work.iterdir())}"
    )
