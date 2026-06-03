# Deploy & centralization model

This repo has **three** places the code can live. They must not drift:

| Location | Role |
|----------|------|
| `~/codebase_projects/quantum_cowboy_biochemistry` (this checkout) | **Development.** Edit + test here. |
| `/net/software/lab/quantum_cowboy_biochemistry` | **Shared install.** What other people / cron / notebooks import when not bind-mounting a dev tree. |
| `quantum_chem-*.sif` (+ `uma-*.sif`) | **Runtime.** The container where `qcb` actually executes. |

## The rule

A **release is a git tag**, and cutting one updates the shared install **and** the
container together. Never hand-copy code into `/net/software/lab` or `pip install`
into the container ad hoc — that is exactly how the three drift apart (the
pre-cleanup state had `/net/software/lab` six days behind this checkout).

## Cutting a release

```bash
./deploy.sh v0.3.0            # tag + rsync to /net/software/lab + print rebuild cmds
./deploy.sh v0.3.0 --dry-run  # preview, change nothing
```

`deploy.sh` refuses to run on a dirty tree (so the tag is reproducible), tags the
release, rsyncs tracked code to `/net/software/lab` (excluding run artifacts,
vendored build outputs, and caches), and prints the two `apptainer build`
commands. Rebuild the container **only when `deps/` changed** (a new package, a
pinned-version bump); pure-Python changes are picked up from the shared install
or a bind-mounted dev tree without a rebuild.

## How notebooks should invoke the code

Notebooks call the `qcb` CLI inside the container — never a raw `.py` path:

```bash
SIF=/net/software/containers/users/$USER/quantum_chem/quantum_chem-20260506.sif
apptainer exec --nv --bind /home --bind /net $SIF qcb <op> ...
```

To test **uncommitted** dev changes, bind-mount this checkout over the installed
package and prepend it to `PYTHONPATH`:

```bash
apptainer exec --nv --bind /home --bind /net \
  --env PYTHONPATH=$HOME/codebase_projects/quantum_cowboy_biochemistry \
  $SIF qcb <op> ...
```

## Building / rebuilding containers

Definitions live in `deps/`:
- `deps/quantum_chem.def` — main container (Python 3.11, torch 2.11, MACE, SCINE, PLUMED, MongoDB).
- `deps/uma_sidecar.def` — FairChem/UMA sidecar (numpy 2.0, torch 2.8) — isolated so it
  doesn't perturb the main container's numpy 1.26 pin.

Build logs are archived under `deps/container_build_logs/`. See `deps/README.md`
for the per-dependency build scripts.
