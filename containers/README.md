# Generative-TS sidecar containers

The `.sif` images here are **git-ignored** (multi-GB). Build them from the
definition files in `deps/` and they land in this directory.

| Sidecar  | Def                        | Role                         | Built/validated |
|----------|----------------------------|------------------------------|-----------------|
| React-OT | `deps/reactot_sidecar.def` | TS **proposer** (R,P→guess)  | 2026-06-05 ✓    |
| AEFM     | `deps/aefm_sidecar.def`    | TS **refiner** (guess→guess) | 2026-06-05 ✓    |

## Build

These hosts aren't in `/etc/subuid`, so apptainer builds in a root-mapped
namespace where the `fakeroot` wrapper is broken/absent — pass
`--ignore-fakeroot-command`:

```bash
export APPTAINER_CACHEDIR=/tmp/apptainer_cache APPTAINER_TMPDIR=/tmp/apptainer_tmp
apptainer build --fakeroot --ignore-fakeroot-command \
  containers/reactot-$(date +%Y%m%d).sif deps/reactot_sidecar.def
apptainer build --fakeroot --ignore-fakeroot-command \
  containers/aefm-$(date +%Y%m%d).sif deps/aefm_sidecar.def
```

## Validated end-to-end (2026-06-05, CPU)

Each adapter was exercised inside its sidecar (model load → forward pass → result):

- React-OT: `quantum_engine.qm.reactot.run` on an HCN→HNC pair → a 3-atom TS guess
  (`sb-pretrained.ckpt`, Zenodo 13131875).
- AEFM: `quantum_engine.qm.aefm.run` on a 14-atom CHON guess → refined in 6
  fixed-point steps, RMSD 0.39 Å (`aefm_xtb_ci_neb.pt`, Zenodo 16414436).

## Usage + a deployment note

Run the generative step inside its sidecar:

```bash
apptainer exec --nv --bind /home --bind /net containers/reactot-<date>.sif \
  python -m quantum_engine.cli ts-entry --entry reactant-product --proposer react-ot ...
apptainer exec --nv --bind /home --bind /net containers/aefm-<date>.sif \
  python -m quantum_engine.cli ts-entry --entry ts-guess --refiner aefm ...
```

**Energy-backend caveat + the two-step handoff.** These sidecars carry the
*generative* model only, not an MLFF/xTB energy backend. The proposer/refiner
produces a GUESS; the downstream saddle-refine → Hessian → IRC gate needs an energy
calculator. So run it in two steps: the generative step in its sidecar (emits a
guess), then the validated `ts-entry --entry ts-guess` in the main container (which
has the MLFF). The dedicated `ts-propose` / `ts-refine` subcommands do step 1:

```bash
# step 1a — React-OT proposes a TS guess (in its sidecar)
apptainer exec --nv --bind /home --bind /net containers/reactot-<date>.sif \
  env PYTHONPATH=$PWD python -m quantum_engine.cli ts-propose \
    --method react-ot --reactant R.xyz --product P.xyz --out guess.xyz

# step 1b (optional) — AEFM refines a guess (in its sidecar)
apptainer exec --nv --bind /home --bind /net containers/aefm-<date>.sif \
  env PYTHONPATH=$PWD python -m quantum_engine.cli ts-refine \
    --method aefm --ts-guess guess.xyz --out refined.xyz

# step 2 — refine + Hessian + IRC gate in the MAIN container (has the MLFF)
apptainer exec --nv --bind /home --bind /net <main>.sif \
  python -m quantum_engine.cli ts-entry --entry ts-guess \
    --ts-guess refined.xyz --reaction-spec rxn.yaml --model mace-omol ...
```

Both adapters are CHNO-/gas-phase guarded (out of domain for the di-Zn theozyme).
