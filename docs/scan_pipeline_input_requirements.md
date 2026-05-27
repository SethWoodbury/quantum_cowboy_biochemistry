# Input requirements for scan_along_s + polish_ts_v3

**Status: v2 (auto-detect).** Captures what the pipeline needs to succeed on
arbitrary single-bond-forming/single-bond-breaking reactions — both the
things the code enforces and the implicit assumptions baked into the
defaults. For multi-bond reactions (Diels-Alder, Cope, sigmatropic), see
`docs/multi_bond_design.md`.

## TL;DR

```
input.pdb  ⇒  scan_along_s.py  ⇒  TS

  In v2 (current): NO required CLI flags beyond --input/--out. The script
  auto-detects the reactive triplet from a ligand-residue
  (resname ∈ {SUB,LIG,SBT,UNL,UNK,MOL,DRG,INH}) and auto-computes
  sum_target from the input geometry.

  Optional overrides for non-default cases:
    --p-name P  --p-res SUB
    --nuc-name O --nuc-res LIG
    --lg-name O  --lg-res LIG
    --sum-target 4.25     # default 'auto' = d(P-Onuc)+d(P-Olg) from input
    --charge N             # default 1 (PTE), set per system!
    --model mace-polar-m   # default fits charged systems
```

## Required input PDB content

1. **Active-site cluster already assembled** — substrate must be docked into
   the catalytic pose. The pipeline does NOT do docking. Input should
   represent a "TS-like pose" within ±1.5 Å in `s = d(P–Lg) − d(P–Nuc)` of the
   actual saddle, OR widen `--s-min/--s-max` to cover the gap.
2. **Identifiable reactive triplet:** electrophilic center (P), nucleophile
   (Nuc), leaving group (LG). v2 auto-detects from a ligand residue with a
   single P/C/S/Si/B/Al center and 3-5 covalent O/N/halide neighbors.
   Override any subset via the CLI flags if auto-detect picks the wrong
   atoms. Auto-detect raises a clear error if no candidate is found.
3. **CA atoms in chain A** if you want the rigid-protein-scaffold mode to
   work. If your protein chain ID is different, currently you'd need to
   rename or use `--free-residues` to liberate everything.
4. **Net charge** — supplied via `--charge`. The tool writes formal charges
   from PDB cols 79-80 to the output CIF; pass `--metal-oxidation Zn:+2 Fe:+3`
   if your PDB doesn't have charge columns set on metals.
5. **Element column** (PDB cols 77-78) — populated with element symbols
   recommended; v2 falls back to a full-periodic-table guess from the atom
   name when missing (so `PD1` → Pd, `AG2` → Ag, not `P`/`A`).

## What the input does NOT need

- A perfect TS. The scan walks ±1.5 Å in `s` from the input — input can be a
  reactant-side pose, product-side pose, or anywhere in between.
- Hydrogen atoms placed correctly. xTB SCF is robust to small H positioning
  errors. Use `--prune-residue-keep` + `--prune-xtb-relax` if you need to
  trim+re-cap.
- REMARK 666 entries — these are auto-added for chain-A protein residues if
  missing.

## Defaults and which are PTE-specific (v2)

| default | PTE-specific? | what to do for non-PTE |
|---|---|---|
| reactive triplet | NO (auto-detect) | override individual atoms via --p-name etc. |
| `--sum-target auto` | NO (auto-compute from input geometry) | override with float for literature TS |
| `--target-d-p-onuc None` (polish_ts_v3) | NO (default = input geometry, no shift) | override to push toward TS pose |
| `--target-d-p-olg  None` (polish_ts_v3) | NO (default = input geometry, no shift) | override to push toward TS pose |
| `--s-min/-max ±1.5` | NO | OK for any associative TS |
| `--n-points 11` | NO | bump to 21 if you want Δs=0.15 resolution |
| `--charge 1` | YES (system-dependent) | **ALWAYS override per system** |
| `--model mace-polar-m` | sort of | swap to mace-omol for non-charged organics |
| ligand resname set `{SUB,LIG,SBT,UNL,UNK,MOL,DRG,INH}` | NO (covers most PDB inputs) | rename your residue, or extend the set in `tools/reactive_autodetect.py` |

## Reaction classes the pipeline handles well

- **Bimolecular nucleophilic substitution at P/C/S** with associative TS
  (single bond breaking + single bond forming). The 1D `s` CV captures the
  reaction coordinate.
- **Phosphoryl transfer / hydrolysis** (PTE, kinases, phosphatases) — the
  archetypal use case.
- **SN2-at-C** in metalloenzymes or organic systems (e.g. methylation,
  haloperoxidases) — with appropriate sum_target.
- **Hydride transfer** — but tighten `--s-min/-max` to ±0.8 (shorter bonds).

## Reaction classes the pipeline does NOT handle today

- **Concerted multi-bond reactions** where 2+ bonds break and 2+ form
  simultaneously (e.g. Diels-Alder, sigmatropic). Single 1D `s` CV is
  inadequate; use NEB-CI or freezing string instead.
- **Stepwise mechanisms requiring >1 saddle** (e.g. acyl-enzyme intermediates
  in serine proteases). Need to apply the pipeline twice with different
  reactive-atom assignments.
- **Reactions where reactant and product geometries are radically different**
  (large conformational change in addition to bond rearrangement). The
  CA-rigid scaffold prevents that motion; relax it via `--free-residues all`
  or accept sequential polish steps.

## How to apply to a new reaction (5-step recipe)

1. **Identify your electrophile, nucleophile, leaving group atom names** by
   visual inspection in PyMOL/ChimeraX.
2. **Read the input geometry's d(E–Nuc) and d(E–Lg)** at your TS guess.
   Compute `sum = d_E_Nuc + d_E_Lg`. Pass `--sum-target $sum` (or use v2
   auto-compute when available).
3. **Pick `--charge`** for the cluster. Sum of formal charges per PDB cols
   79-80, plus any neutral substrate.
4. **Pick `--model`** — `mace-polar-m` for charged systems, `mace-omol` for
   neutral organics, `mace-mh` for diverse coverage.
5. **Run scan_along_s** with `--out my_run/`. Inspect
   `my_run/summary.json:scan_results` energy profile. The energy max is
   your TS; the lowest energy point on the reactant side is your true
   Michaelis complex.

## Pipeline timing

Single scan_along_s run on a 229-atom system (mace-polar-m, L40 GPU):
**~18 minutes wall** for 11 scan points × ~100 LBFGS steps each. Scales
linearly with `--n-points` and approximately linearly with system size up to
the GPU's memory limit (a 1000-atom system would be ~3-4× slower).

For other GPUs:
- H200: ~8 min
- A6000: ~30 min
- A100: ~15 min
- L40: ~18 min (used for the campaign)

## Validation gates baked in

- `structure_validator.py` checks atom-count, atom-identity preservation,
  rigid-body anchor preservation (CA scaffold), Zn coordination, reactive
  distance drift, no-decoupling.
- `benchmark_synth.py` flags suspiciously-low barriers (< 5 kcal/mol),
  out-of-range distances, unconverged energies.
- `qcb freq` partial Hessian on reactive core confirms n_imaginary == 1.

## v2 changes (already shipped)

- Auto-detection of reactive triplet from ligand residue: implemented in
  `tools/reactive_autodetect.py`.
- Auto-compute of `sum_target` from input geometry: enabled by
  `--sum-target auto` (default).
- Auto-default of `--target-d-p-onuc / --target-d-p-olg` in polish_ts_v3 to
  the input geometry distances.
- Argmax-endpoint guard: warns if the energy max sits at the boundary of
  the scan window (signal that input pose is outside the saddle's basin in
  s).
- Periodic-table extension of `_guess_element` (Pd/Pt/Ru/Mo etc. correctly
  identified instead of mis-elemented).
- `summary.json` now contains `reactive_triplet` and `autodetect_log`.
- Legacy v1 frozen at `tools/scan_along_s_v1_legacy.py` (sum=4.25, hardcoded
  PTE atom names) for reproducibility of the published 22.7 kcal/mol PTE
  result.

## Known unknowns / future work

- **Multi-bond CV** beyond 1D `s` — scan_along_s.py is structurally limited
  to single-bond-forming/single-bond-breaking topologies. For Diels-Alder,
  Cope, sigmatropic, electrocyclic, or any K+L > 2 reaction, see
  `docs/multi_bond_design.md` for the proposed `qcb scan-multi` + `qcb
  auto-ts` policy layer that orchestrates scan / NEB-CI / Sella based on
  bond count.
- **R-P connectivity diff** to auto-detect forming/breaking bonds when both
  endpoints are provided (RDKit + covalent radii).
- **OA-ReactDiff seeding** for NEB-CI (3-5× iteration savings, opt-in flag).
- Cross-method automated validation (mace-omol + g-xTB SP cross-check;
  scaffolded in `benchmark_synth` but not yet auto-triggered).
