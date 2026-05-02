# deps/

Vendored third-party dependencies built from source. Adding a new dep here means we own the build, the version is pinned in `.gitmodules`, and nothing in `qcb/` depends on a binary in someone else's `$HOME`.

## Contents

| Submodule | Purpose | Build script |
|---|---|---|
| `atomworks/` | Baker-lab structure / sequence utilities | (pip install) |
| `pysisyphus/` | TS search engines (FSM, GSM, NEB) | (pip install) |
| `plumed2/` | Enhanced sampling library | `plumed2_build.sh` |
| `xtb/` | Grimme's extended-tight-binding semiempirical QM | `build_xtb.sh` |

Loose source trees: `graph_longrange_src/`, `mace_polar_src/`, `pyGSM/` — checked-in copies, not submodules.

## Building xTB

Builds the binary into `deps/xtb/install/bin/xtb` and the shared library into `deps/xtb/install/lib/libxtb.so`. The script uses a dedicated conda env (`qcb-xtb`) that ships a Fortran toolchain — never the system compilers.

### One-time conda env setup

```bash
/home/woodbuse/conda/bin/conda create -n qcb-xtb -c conda-forge \
    python=3.11 \
    gfortran_linux-64 gxx_linux-64 gcc_linux-64 \
    meson ninja cmake pkg-config \
    openblas liblapack \
    numpy ase \
    -y
```

### Build

```bash
git submodule update --init deps/xtb
NJOBS=16 deps/build_xtb.sh
```

Re-run the script anytime you `git submodule update` to pull a newer xtb commit. The script wipes `build/` first so it's always reproducible.

### Verifying

```bash
deps/xtb/install/bin/xtb --version
```

`qcb/config.py` already points `XTB_BIN` at `deps/xtb/install/bin/xtb`, so any qcb tool that calls xTB will pick this up automatically (no additional configuration).

## g-xTB (general extended tight-binding)

Vendored at `deps/g-xtb/` (submodule of `grimme-lab/g-xtb`). The upstream repo ships static prebuilt binaries in `binaries/` — extracted into `deps/g-xtb/install/xtb-<ver>/bin/xtb`. **No build needed.**

The g-xTB binary is a modified xtb 6.7.1 (Grimme group's `thfroitzheim/xtb gxtb` branch). It accepts everything regular xtb does, plus `--gxtb` to enable g-xTB (a wB97M-V/def2-TZVPPD-approximating semiempirical method, all elements Z=1–103).

`qcb/config.py` exposes this as `GXTB_BIN`, separate from `XTB_BIN` (the regular xtb at `deps/xtb/install/bin/xtb`). Use `XTB_BIN` for GFN-FF/GFN1/GFN2; use `GXTB_BIN` only when you specifically want the g-xTB method.

### Tracking the latest release

`.gitmodules` pins `deps/g-xtb` to `branch = main` (not a fixed commit) because upstream is pre-release and iterates frequently — we want their newest binaries by default. To bump to whatever they pushed last:

```bash
bash deps/bump_gxtb.sh
```

The script:
1. `git fetch && git pull` on the submodule's `main` branch
2. picks the newest `xtb-*-gxtb-*-linux-x86_64.tar.xz` tarball under `binaries/` (filename embeds a build date so `sort | tail -1` wins)
3. verifies the published `.sha256` if present
4. wipes and re-extracts into `install/`
5. runs a tiny H2O smoke test through `--gxtb` to confirm the new binary actually executes

Then commit the new submodule pin:

```bash
git add deps/g-xtb && git commit -m "bump g-xtb to <SHA>"
```

Status: pre-release. Treat geometric agreement on the YYE/Zn₂ benchmark (`enzyme_design_applications/active_site_refine/geom_score.py`) as the regression test before relying on a new version for production work.

## Why submodules and not just `pip install`?

xTB is a Fortran binary, not a Python package — `pip` can't install it. The conda-forge `xtb` package would work, but vendoring the source means:

1. We pin the exact commit (reproducibility for paper figures).
2. We can patch upstream bugs locally without forking on GitHub.
3. Cluster admins don't need to install anything system-wide.
4. The build is reproducible from a clean clone with `git submodule update --init && deps/build_xtb.sh`.
