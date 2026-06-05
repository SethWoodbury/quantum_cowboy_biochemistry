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

**Energy-backend caveat:** these sidecars carry the *generative* model only, not
an MLFF/xTB energy backend. The proposer/refiner produces a GUESS; the downstream
saddle-refine → Hessian → IRC gate needs an energy calculator. So either (a) add an
energy backend (e.g. xtb) to the sidecar for one-shot `ts-entry`, or (b) run the
proposer/refiner to get the guess, then feed it to the main container's
`ts-entry --entry ts-guess` (which has the MLFF). Both adapters are CHNO-/gas-phase
guarded (out of domain for the di-Zn theozyme).
