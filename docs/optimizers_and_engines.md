# Optimizers & engines

Two orthogonal plug-and-play axes: the **energy function** (what computes
energy/forces) and the **optimizer / TS engine** (what drives the geometry).

## The unifying principle: optimizer ⊥ energy function

Every optimizer consumes any ASE `Calculator`, so any energy function pairs with
any optimizer "for free". Two exceptions:

- **`torch-sim-*` minimizers are MLFF/torch-only** (a GPU fast path; they need a
  torch model, not a generic ASE calc). `list_backends(available_only=True)`
  hides them when `torch_sim` isn't installed.
- **Analytic Hessians come only from xTB / DFT.** MLFFs use a finite-difference
  (partial) Hessian for the saddle/validation gate.

## Energy functions (`make_calc`, registry `ENERGY_FAMILIES`)

| Family   | Aliases (examples)                         | Charge-aware | Notes |
|----------|--------------------------------------------|--------------|-------|
| MACE     | `mace-omol`, `mace-mh-1 --head omol`, `mace-polar-m`, `mace-off-small` | omol/mh-omol/polar | default; POLAR needs the baked-in fork |
| UMA      | `uma-s-1p1`, `uma-m-1p1` (FairChem)        | yes (calc)   | sidecar container (numpy2/torch2.8) |
| ORB      | `orb-mol` (conservative)                   | yes (SystemConfig) | |
| AIMNet2  | `aimnet2-rxn`                              | yes (construct) | organic only |
| qc (xTB) | `gfn2-xtb`, `gfn-ff`, `g-xtb`              | GFN1 charge-aware; **GFN2 not charge-aware on metals** | semiempirical; analytic Hessian |

Use a **charge-aware MLFF** (mace-omol / mace-polar / uma) for metal / charged
TSs; xTB is for organic substrates and cheap sanity checks. `make_calc` warns (or
forbids) GFN2 on metals.

## Minimizers (`make_optimizer`, registry `OPTIMIZERS`)

| Backend            | Aliases        | ASE-agnostic | When |
|--------------------|----------------|--------------|------|
| `ase-lbfgs`        | `lbfgs`        | yes          | default |
| `ase-fire`         | `fire`         | yes          | rough/forgiving |
| `ase-bfgs`         | `bfgs`         | yes          | small systems |
| `ase-precon-lbfgs` | `precon`       | yes          | large (>100-atom) systems |
| `ase-fire2`        | `fire2`        | yes          | improved FIRE schedule |
| `torch-sim-fire`   | —              | **MLFF-only** | GPU-batched |
| `torch-sim-lbfgs`  | —              | **MLFF-only** | GPU-batched |

## Saddle optimizers (`saddle.run`, registry `SADDLE_OPTIMIZERS`)

| Backend             | Modern? | Notes |
|---------------------|---------|-------|
| `sella`             | ✓ | Hessian-eigvec following (Cartesian) |
| `sella-internal`    | ✓ | TRIC internals — robust fallback past ~200 free atoms |
| `dimer`             | ✓ | ASE dimer; gradient-only; NEB tangent is a natural seed |
| `pysisyphus-rsprfo` | ✓ | RS-P-RFO; small systems with an affordable Hessian |
| `pysisyphus-dimer`  | ✓ | second opinion when ASE dimer stalls |
| `auto`              |   | cascade sella → sella-internal → dimer on LinAlgError |

## Path methods (`path_search.run`, registry `PATH_METHODS`)

| Method   | Aliases  | Type | When |
|----------|----------|------|------|
| `neb`    | `ci-neb` | double-ended | **default**; geodesic CI-NEB, robust on MLFFs |
| `fsm`    | —        | double-ended | shallow/concerted/floppy where CI-NEB struggles |
| `gsm-de` | `gsm`    | double-ended | smoother path at slightly higher cost |

SE-GSM (reactant-only) and AutoNEB are planned (drop in via `register_path`).
Path methods only PROPOSE a TS — the saddle+Hessian+overlap+IRC-like gate is the
acceptance authority.

## QM-native engines (`ts_entry` via `ctx.engine`, registry `ENGINES`)

| Engine | Modes | Notes |
|--------|-------|-------|
| `orca` | NEB-TS (R+P), OptTS+Freq (guess) | **default for DFT**; ORCA runs host-side (not in the container). `--no-execute` prepares an input to `sbatch`. |

ORCA is also available as an **ASE calculator** (`qm.calc.make_qm_calc`) so the
factories above can drive it per-step (mode A) — use that for small systems /
constraints / method experiments; prefer the native engine for routine DFT TS work.

## Picking a combination

- **Cheap survey / organic:** `--model gfn2-xtb` + `--saddle-backend dimer`.
- **Metal / charged TS (the OPAA test case):** `--model mace-polar-m`
  (or `mace-mh-1 --head omol`) + `--rigor standard` (CI-NEB → auto saddle →
  Hessian+overlap+IRC-like).
- **DFT reference:** `--engine orca --model "wB97X-D3/def2-TZVP"` (native NEB-TS).
- **Large system minimization:** `--optimizer ase-precon-lbfgs`.

See [extending.md](extending.md) to add any new method, and
[ts_workflow.md](ts_workflow.md) for the entry-point decision tree.
