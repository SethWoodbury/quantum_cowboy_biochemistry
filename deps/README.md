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

Vendored at `deps/g-xtb/` (submodule of `grimme-lab/g-xtb`). The upstream repo ships static prebuilt binaries in `binaries/` — extracted into `deps/g-xtb/install/xtb-6.7.1/bin/xtb`. **No build needed.**

```bash
# After git submodule update:
mkdir -p deps/g-xtb/install
tar xJf deps/g-xtb/binaries/xtb-6.7.1-gxtb-210426-linux-x86_64.tar.xz \
    -C deps/g-xtb/install
```

The g-xtb binary is a modified xtb 6.7.1 (Grimme group's `thfroitzheim/xtb gxtb` branch). It accepts everything regular xtb does, plus `--gxtb` to enable g-xTB (a wB97M-V/def2-TZVPPD-approximating semiempirical method, all elements Z=1–103).

```bash
deps/g-xtb/install/xtb-6.7.1/bin/xtb file.xyz --gxtb --opt
```

`qcb/config.py` exposes this as `GXTB_BIN`, separate from `XTB_BIN` (the regular xtb at `deps/xtb/install/bin/xtb`). Use `XTB_BIN` for GFN-FF/GFN1/GFN2; use `GXTB_BIN` only when you specifically want the g-xTB method. Status: pre-release — track upstream tags before relying on it for paper figures.

## Why submodules and not just `pip install`?

xTB is a Fortran binary, not a Python package — `pip` can't install it. The conda-forge `xtb` package would work, but vendoring the source means:

1. We pin the exact commit (reproducibility for paper figures).
2. We can patch upstream bugs locally without forking on GitHub.
3. Cluster admins don't need to install anything system-wide.
4. The build is reproducible from a clean clone with `git submodule update --init && deps/build_xtb.sh`.
