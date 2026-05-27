# MACE × CREST 3 integration

`crest_with_mace.sh` runs CREST 3.x with MACE forces using CREST's built-in
`generic_sc` calculator and a long-running MACE daemon to avoid model-load
overhead.

## Why this exists

- CREST 3.x ships a `method = "generic"` calculator (see
  `deps/crest/src/calculator/generic_sc.f90`) that, on every force call, runs
  `<binary> genericinp.xyz [flags]` and reads back `genericinp.engrad` in
  ORCA `.engrad` format.
- A naive Python wrapper (load MACE inside the per-call script) would
  re-load the neural-net weights ~5–30 s **per force call** depending on
  model size and device. CREST issues thousands of calls per MTD walker;
  that overhead is fatal.
- Solution: load the model once in a daemon, listen on a Unix socket, and
  let the per-call wrapper do a microsecond JSON roundtrip.

## Files

| File | Role |
|---|---|
| `tools/mace_engrad_daemon.py` | Long-running MACE server. Loads model once, listens on a Unix socket. |
| `tools/mace_engrad_client.py` | Per-call wrapper invoked by CREST. Reads `genericinp.xyz`, JSON-roundtrips with the daemon, writes `genericinp.engrad`. |
| `tools/crest_with_mace.sh` | Orchestrator: starts daemon, generates TOML, runs CREST, cleans up. |
| `tools/MACE_CREST_README.md` | This document. |
| `tests/test_mace_engrad.py` | Unit + smoke tests. |

## Architecture

```
   ┌─────────────────────────┐
   │  crest_with_mace.sh     │  bash orchestrator
   │  • starts daemon (1×)   │
   │  • generates TOML       │
   │  • runs CREST           │
   │  • kills daemon on exit │
   └────────────┬────────────┘
                │
                │ (1) start in background
                ▼
   ┌──────────────────────────┐                  ┌──────────────────────────┐
   │ mace_engrad_daemon.py    │  ← Unix socket → │ mace_engrad_client.py    │
   │ • loads MACE once        │   (JSON line)    │ • reads genericinp.xyz   │
   │ • answers JSON requests  │                  │ • sends JSON request     │
   │ • holds GPU/CPU memory   │                  │ • writes genericinp.engrad
   └──────────────────────────┘                  └──────────────────────────┘
                                                         ▲       │
                                                         │       │ exec'd as a sub-
                                                         │       ▼ process per call
                                              ┌────────────────────────┐
                                              │ CREST 3.x              │
                                              │ method="generic"       │
                                              │ binary="run_mace.sh"   │
                                              └────────────────────────┘
```

## Usage

### Quick start

```bash
# From qcb-xtb env (or any env with mace-torch installed)
$ source ~/conda/etc/profile.d/conda.sh && conda activate qcb-xtb
$ tools/crest_with_mace.sh struc.xyz \
      --model mace-mp --device cuda --charge 0 --spin 0 \
      --workdir /tmp/run1 \
      -- -gfn2  # (anything after `--` is forwarded to crest)
```

### Forwarding CREST flags

Anything after the literal `--` is forwarded to CREST. Example: a quick
single-point run:

```bash
$ tools/crest_with_mace.sh struc.xyz --device cpu -- --sp
```

Conformer search (default iMTD-GC pipeline):

```bash
$ tools/crest_with_mace.sh struc.xyz --device cuda
```

### CLI options

| Flag | Default | Notes |
|---|---|---|
| `--model` | `mace-mp` | Any alias the qcb factory knows (`mace-mp`, `mace-omol`, `mace-polar-m`, ...) or an absolute `.model` path. |
| `--device` | `cuda` | `cuda` or `cpu`. Match daemon to the host's available GPU. |
| `--dtype` | `float64` | `float32` is faster and uses less memory but matches less well to xtb-validated forces. |
| `--head` | _(none)_ | For multi-head MACE models (e.g. `mace-mh --head omol`). |
| `--charge` | `0` | Forwarded to MACE via `atoms.info["charge"]` for charge-aware models, and into the TOML's `chrg = N`. |
| `--spin` | `0` | Forwarded to MACE via `atoms.info["spin"]`. |
| `--workdir` | _(tempdir)_ | Where CREST runs. Persisted on exit if you provide one explicitly. |
| `--toml` | _(autogen)_ | Provide your own TOML; otherwise we generate one in the workdir. |
| `--logfile` | _(workdir)_ | Daemon log path. |
| `--` | _(separator)_ | Anything after this is passed to crest. |

### Environment overrides

| Variable | Default | Effect |
|---|---|---|
| `QCB_PYTHON` | `which python` | Python interpreter for daemon + client. |
| `QCB_CREST_BIN` | `deps/crest/install/bin/crest` | CREST binary. |
| `QCB_MACE_DAEMON_READY_TIMEOUT_S` | `180` | Max wait for daemon to print `READY`. Increase for slow disks / huge models. |
| `QCB_MACE_DAEMON_SHUTDOWN_GRACE_S` | `5` | SIGTERM → SIGKILL grace. |
| `QCB_MACE_PROTOCOL_TIMEOUT_S` | `120` | Per-call recv deadline inside the daemon. |
| `MACE_DAEMON_SOCKET` | _(none)_ | Used by `mace_engrad_client.py` if `--socket` not given. |
| `QCB_CHARGE`, `QCB_SPIN` | _(0)_ | Used by client when `--charge` / `--spin` not given. |

## TOML schema (auto-generated)

The wrapper produces a TOML like this:

```toml
input = "/abs/path/to/struc.xyz"
threads = 1

[calculation]
elog = "/abs/path/to/workdir/energies.log"

[[calculation.level]]
method = "generic"
binary = "/abs/path/to/workdir/run_mace.sh"
gradtype = "engrad"
chrg = 0
uhf = 0
```

Where `run_mace.sh` is:

```bash
#!/bin/bash
exec "$QCB_PYTHON" mace_engrad_client.py "$1" \
     --socket "$SOCKET" --charge $CHARGE --spin $SPIN --quiet
```

CREST's parser (`deps/crest/src/parsing/parse_calcdata.f90`):
- `method = "generic"` ⇒ `jobtype%generic`, dispatched to `generic_engrad`.
- `binary` ⇒ the script CREST exec's per call.
- `gradtype = "engrad"` ⇒ CREST reads back ORCA `.engrad` format
  (energy in Eh, gradient in Eh/Bohr, 3N components).
- `chrg = N`, `uhf = N` are CREST-side bookkeeping.

If you need solvent or extra MACE arguments, add them to the runner script
or pass them through `--` to CREST.

## Concurrent runs

Each invocation of `crest_with_mace.sh` spins up its own daemon with its
own socket (`mace_engrad.<wrapper-pid>.sock` in the workdir). You can run
many concurrently as long as the GPU memory + per-daemon model RAM budget
allows.

## When to use this vs xtb-CREST

**Use MACE-CREST when:**
- You need a level of theory above xtb but cheaper than DFT (mace-omol /
  polar / off / mp foundation models are useful here).
- You're sampling a MACE-fit energy surface (e.g. evaluating a TS that was
  refined by `qcb refine-ts --model mace-omol`) and want consistent forces.
- The system contains atoms / charge states xtb doesn't handle reliably.

**Stick with xtb-CREST (the default `qcb crest` path) when:**
- You're doing a fast pre-screen for diversity (xtb-CREST is hard to beat
  on speed for typical organic chemistry).
- You don't have a GPU available and CPU MACE is too slow for the per-call
  budget you need (xtb on CPU is competitive with MACE on CPU for
  conformer searches).
- Your system is non-reactive vanilla organic — xtb is well-validated and
  the MACE foundation models offer little advantage on these.
- GPU contention with other work would make daemon performance erratic.

## Performance notes

- **Model load time**: dominated by the disk I/O + JIT trace. Ranges from
  ~5 s (mace-mp small / float32) to ~30 s (mace-omol XL / float64 / CPU).
  Pay this once per CREST run.
- **Per-call wall time**: dominated by the MACE forward pass. On a single
  H100, mace-mp on a 30-atom system is ~50–100 ms. Socket roundtrip adds
  <2 ms. Compare to ~20 ms for xtb on the same system.
- **MACE on CPU**: usable but ~10× slower than GPU. For real CREST
  conformer searches (10⁴–10⁵ force calls) you almost certainly want GPU.
- **Threading**: CREST issues parallel force calls in OMP threads but the
  daemon serializes them under a lock (one Torch context per process).
  Setting `threads = 1` in the TOML is fine; bump only if you confirmed
  speedup with your model.

## Limitations / gotchas

1. **Single batch per call.** CREST's `generic_sc` is one molecule per
   subprocess. We don't batch across multiple CREST walkers. For embarrassingly
   parallel workloads you'd want a daemon-per-walker (which the wrapper
   already does — start multiple wrappers).
2. **No WBO / dipole.** The generic calculator path does not pull bond
   orders or dipoles from MACE. CREST routines that depend on those (a
   small subset) will not work. Most conformer-search functionality does
   not need them.
3. **Generic + tblite mixing.** CREST 3 lets you stack multiple
   `[[calculation.level]]` blocks; mixing a MACE generic level with a
   tblite refinement level should work but isn't tested by us.
4. **MACE charge handling.** Only charge-aware models (mace-omol,
   mace-polar) consume `atoms.info["charge"]`. If you pass `--charge 1`
   to a charge-blind model (mace-mp, mace-off), the daemon silently
   ignores it. CREST itself uses `chrg` only for atom counting.
5. **Daemon orphan.** If the wrapper is killed with `SIGKILL` before its
   trap fires, the daemon will linger. Detect with
   `pgrep -f mace_engrad_daemon` and kill manually.
6. **GPU cleanup.** PyTorch holds CUDA memory until the process exits.
   This is fine for one-shot runs but if you reuse the same Python
   process for many sequential daemons, expect VRAM accumulation.
7. **TOML coordinates.** The wrapper resolves `--input` to an absolute
   path — relative paths break under CREST's `cd` semantics.

## Diagnostic recipes

### Daemon won't become READY

```bash
# Check the stderr log:
$ tail -50 /tmp/run1/mace_daemon.log.stderr

# Common causes:
#   - "ModuleNotFoundError: mace" → wrong env
#   - "torch.cuda.is_available() == False" with --device cuda
#   - "FileNotFoundError" loading model → bad path / model alias
```

### Per-call hang

```bash
# Check daemon is alive and socket exists
$ pgrep -af mace_engrad_daemon
$ ls -l /tmp/run1/mace_engrad.*.sock

# Hand-test a roundtrip:
$ python tools/mace_engrad_client.py /path/to/test.xyz \
      --socket /tmp/run1/mace_engrad.PID.sock --quiet
$ cat /path/to/test.engrad
```

### CREST not seeing the forces

```bash
# Re-run CREST manually with the generated TOML and check stdout:
$ /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/crest/install/bin/crest \
       --input /tmp/run1/crest_mace.toml --sp
# CREST prints "running engrad calc..." messages; if it stops there, the
# binary path or gradtype is wrong.
```

## qcb CLI integration

The capability is also exposed as a regular qcb subcommand:

```bash
$ qcb crest-mace input.xyz --model mace-mp --device cuda \
                 --crest-args -gfn2 --quick
```

This is a thin shell-out to `crest_with_mace.sh`; all flags work the same.

## Testing

```bash
$ pytest tests/test_mace_engrad.py            # 11 unit + protocol tests (~20s)
$ QCB_RUN_SLOW=1 pytest tests/test_mace_engrad.py  # +2 real-MACE end-to-end (~50s)
```

Smoke-test results (CPU, mace-mp medium float32):

| Phase | Wall time |
|---|---|
| Daemon model load (cold) | ~22 s |
| Daemon → wrapper roundtrip (warm Python) | ~4 ms total / ~2 ms forward pass |
| Daemon → wrapper roundtrip (cold Python interp) | ~400 ms (dominated by Python startup) |
| End-to-end CREST `--sp` on water | ~22 s (model load + 0.3 s CREST) |

For a real conformer search the model-load time amortizes to negligible —
the per-call overhead matters. On GPU expect 50–200 ms per call for
30-atom systems.

Verified energy/gradient match between daemon and direct ASE call
(identical model + dtype): **15-decimal agreement** for water in float64.

## References

- CREST 3 docs: <https://crest-lab.github.io/crest-docs/>
- generic_sc source: `deps/crest/src/calculator/generic_sc.f90`
- TOML parser: `deps/crest/src/parsing/parse_calcdata.f90`
- engrad reader (Fortran): `deps/crest/src/calculator/gradreader.f90` —
  `rd_grad_engrad` accepts `# ...` comments, then N (atoms), then E (Eh),
  then 3N gradient components (Eh/Bohr).
- Project rule on hidden magic numbers:
  `~/.claude/projects/.../feedback_no_hardcoded_magic_numbers.md` —
  every wait/timeout/buffer here is exposed as an env var or CLI flag.
