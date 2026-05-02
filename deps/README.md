# deps/

Vendored third-party dependencies built from source. Adding a new dep here means we own the build, the version is pinned in `.gitmodules`, and nothing in `qcb/` depends on a binary in someone else's `$HOME`.

## Contents

| Submodule | Purpose | Build / install script |
|---|---|---|
| `atomworks/` | Baker-lab structure / sequence utilities | (pip install) |
| `pysisyphus/` | TS search engines (FSM, GSM, NEB) | (pip install) |
| `plumed2/` | Enhanced sampling library | `plumed2_build.sh` |
| `xtb/` | Grimme extended-tight-binding semiempirical QM | `build_xtb.sh` |
| `g-xtb/` | Grimme g-xTB (wB97M-V approx, all elements) | `bump_gxtb.sh` |
| `crest/` | Grimme conformer-rotamer ensemble sampler | `build_crest.sh` |

Loose source trees: `graph_longrange_src/`, `mace_polar_src/`, `pyGSM/` — checked-in copies, not submodules.

External tools that don't fit a submodule build (commercial / restricted / web-only):

| Tool | How qcb finds it | Notes |
|---|---|---|
| Gaussian g16 | `quantum_engine.site.GAUSSIAN_BIN` → `/net/software/gaussian/g16/g16` | Cluster-wide install, license-controlled |
| ORCA 4.1.1 | `quantum_engine.site.ORCA_BIN` → `/net/software/orca/orca_4_1_1_linux_x86-64_openmpi313/orca` | Multiple cluster versions in `ORCA_VERSIONS` |
| ChemShell | `$QCB_CHEMSHELL` → `deps/chemshell/install/bin/chemsh` → `/net/software/lab/chemshell/bin/chemsh` | Manual install — see "ChemShell" section below |
| Ketcher | (web-only — not vendor-able) | Use https://lifescience.opensource.epam.com/ketcher/ for 2D reaction drawing; export to MOL/RXN, feed to qcb |

Pip-installable chemistry tools (declared in `pyproject.toml` under `[project.optional-dependencies] chem`):

| Package | Stage of the workflow |
|---|---|
| `rdkit` | format conversion · 2D→3D (ETKDGv3) · bond-diff · ensemble enumeration |
| `epam.indigo` | secondary stereo / format backend; cross-validate against RDKit for tricky cases |
| `rxnmapper` | atom-mapping (transformer-based; validate manually for proton-transfer chemistry) |
| `CGRtools` | bond-change / reaction-graph diff |
| `autode` | SMILES → reaction profile (defaults to xTB / ORCA backends, both vendored above) |

Install with `pip install -e .[chem]` from the repo root.

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

Status: pre-release. Treat geometric agreement on the YYE/Zn₂ benchmark (`enz_qc_pipelines/active_site_refine/geom_score.py`) as the regression test before relying on a new version for production work.

## Why submodules and not just `pip install`?

xTB is a Fortran binary, not a Python package — `pip` can't install it. The conda-forge `xtb` package would work, but vendoring the source means:

1. We pin the exact commit (reproducibility for paper figures).
2. We can patch upstream bugs locally without forking on GitHub.
3. Cluster admins don't need to install anything system-wide.
4. The build is reproducible from a clean clone with `git submodule update --init && deps/build_xtb.sh`.

## CREST (conformer-rotamer ensemble)

Submodule at `deps/crest/` (pinned at v3.0.2 of `crest-lab/crest`). The build script defaults to a fast conda-forge install into the `qcb-xtb` env (~2 min, identical binary to what we'd compile from source); pass `BUILD_FROM_SOURCE=1` to rebuild from the submodule's source via meson + the qcb-xtb env's gfortran toolchain.

```bash
bash deps/build_crest.sh                  # conda path (default, recommended)
BUILD_FROM_SOURCE=1 bash deps/build_crest.sh
```

After the script, `quantum_engine.site.CREST_BIN` resolves automatically. Use:

```python
from quantum_engine.qm.crest import run_conformer_search
result = run_conformer_search(atoms, charge=0, method="gfn2")
result["best"]                     # ASE Atoms — lowest-energy conformer
result["conformers"]                # list — full ensemble
```

CLI: `qcb crest <input.xyz>` (TODO — add a thin CLI wrapper).

## ChemShell (QM/MM, opt-in)

Distribution friction warning: ChemShell needs registration at https://www.chemshell.org/ — not on PyPI / conda-forge / public git. We don't vendor it as a submodule because the upstream source isn't publicly clone-able. To install:

  1. Register on chemshell.org and download the latest tarball.
  2. Extract into `deps/chemshell/install/` so the binary lives at `deps/chemshell/install/bin/chemsh`. (Or extract into `/net/software/lab/chemshell/` for a lab-wide install — `quantum_engine.site.CHEMSHELL_BIN` checks both.)
  3. `quantum_engine.qm.chemshell.chemshell_available()` should return `True`.

Override the resolver with `QCB_CHEMSHELL=/path/to/chemsh` if you have it elsewhere. If you don't have ChemShell, qcb's QM/MM workflows fall back to `quantum_engine.qm.openmm_xtb` (OpenMM + xtb-via-ASE; conda-installable, no registration).

`quantum_engine.qm.chemshell` is currently scaffolded — the wrapper resolves the binary and signals when missing, but the input-file generator + output parser are TODO. Wire those once you have a binary on the cluster to test against.

## Bumping submodules

```bash
git submodule update --remote deps/g-xtb         # pre-release; bump frequently
bash deps/bump_gxtb.sh                            # one-shot: pull + extract + smoke-test

git submodule update --remote deps/xtb           # rare; pin manually after release
bash deps/build_xtb.sh                            # rebuild

bash deps/build_crest.sh                          # pulls latest conda-forge crest
```
