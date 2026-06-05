# TS workflow (`ts_entry`)

`ops/ts_entry.py` turns a TS/reactant/product guess for **any** reaction into a
validated TS. It is reaction-agnostic: the reaction is a user-owned
`ReactionSpec`, the run state a `RunContext` — nothing is hardcoded.

## The canonical core

All three entry points funnel into one core:

```
clean basins (R, P)
   → pluggable path search          (default geodesic CI-NEB)
   → refine the peak to a saddle     (Sella/dimer/… cascade)
   → active-region partial Hessian   (exactly one imaginary mode, below cutoff,
                                       with ≥ overlap on the reactive atoms)
   → IRC-like validation             (displace ±mode → relax → two distinct basins)
```

Path/saddle methods only **propose** a TS; the saddle + Hessian + overlap +
IRC-like gate is the acceptance authority.

## Entry-point decision tree

```
Do you have a TS guess?           → entry="ts-guess"
   (skip path search; saddle-refine → validate)
Do you have reactant AND product? → entry="reactant-product"
   (clean basins → path search → peak → saddle-refine → validate)
Only the reactant (+ a CV)?       → entry="reactant-only"
   (drive the CV to a product basin → reactant-product)
   ✗ no CV/driving coords → error (no silent fake endpoints)
```

## Rigor tiers (`--rigor`)

| Tier         | n_images | saddle | validate | imag cutoff | overlap |
|--------------|----------|--------|----------|-------------|---------|
| `draft`      | 7        | dimer  | no       | −30 cm⁻¹    | 0.30    |
| `standard`   | 11       | auto   | yes      | −50 cm⁻¹    | 0.50    |
| `publication`| 17       | auto   | yes      | −50 cm⁻¹    | 0.50    |

Override any of `--path-method` / `--saddle-backend` / `--n-images` /
`--validate/--no-validate` per run.

## Gates & logging

Every stage runs inside a `logging_utils.Step` (human log + machine
`<stage>.json`) and contributes `Gate`s to one aggregated `GateReport`
(`gates.json`). A **FAIL with no fallback** (n_imag ≠ 1, imag ≥ cutoff,
overlap < min) is a *critical fail* → `status="failed"` and validation is
skipped. Path search is WARN-with-fallback, so a rough path still feeds
saddle-refine.

## ReactionSpec (YAML)

```yaml
reaction:
  forming_bonds:  [["SUB:P1", "OHX:O3"]]     # tokens: int (1-based PDB serial),
  breaking_bonds: [["SUB:P1", "SUB:O5"]]     #   "0:idx" (0-based), or "RES:ID:NAME"
  reactive_atoms: ["SUB:P1", "OHX:O3", "SUB:O5"]
  cv: {kind: bond_difference, atoms: ["SUB:P1", "OHX:O3", "SUB:O5"]}  # (P, nuc, lg)
  atom_map: {0: 0, 1: 1, ...}                # optional R→P map for double-ended
```

Validate/resolve it first: `qcb reaction-spec spec.yaml --structure site.pdb`.

## CLI

```bash
# reactant + product, charge-aware MLFF, standard rigor
qcb ts-entry --entry reactant-product \
    --reaction-spec spec.yaml --reactant R.pdb --product P.pdb \
    --model mace-polar-m --charge 2 --spin 1 --rigor standard --outdir ts_out

# a TS guess, refine + validate only
qcb ts-entry --entry ts-guess --reaction-spec spec.yaml --ts-guess guess.pdb \
    --model mace-mh-1 --head omol --charge 2

# DFT via ORCA native NEB-TS — prepare an input to sbatch (host-side ORCA)
qcb ts-entry --entry reactant-product --reaction-spec spec.yaml \
    --reactant R.xyz --product P.xyz --engine orca --model "wB97X-D3/def2-TZVP" \
    --no-execute --outdir orca_job
```

The result dict carries `status`, `ts`/`reactant`/`product`, energies +
barriers (kcal/mol), `imag_freq_cm`, `n_imag`, and `gates`; outputs include
`gates.json`. See [optimizers_and_engines.md](optimizers_and_engines.md) for
method choices and [extending.md](extending.md) to add new ones.
