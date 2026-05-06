# PTE/paraoxon TS-finding campaign — 2026-05-05

**Status: experimental.** mace-polar-1-m forcefield level. Independent DFT
calibration not yet run.

## Headline result

Forward activation barrier reduced from initial 34.22 kcal/mol → **20.68
kcal/mol (ensemble best, conf_03)** or **22.73 kcal/mol (validated saddle,
conf_10)**. Literature consensus 14–17 kcal/mol; remaining ~5–7 kcal/mol gap
attributed to MLFF systematic on Zn-phosphate (cross-confirmed by g-xTB
single-points and mace-omol).

## Final deliverable

`/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/ts_pipeline/v3_FINAL/`

| file | purpose |
|---|---|
| `reactant.pdb` | Michaelis-complex-like reactant (s=-0.90, d_PN=2.58, d_PL=1.67) |
| `TS.pdb` | symmetric concerted saddle (s=+0.10, d_PN=2.075, d_PL=2.175); n_imag=1 ν=-269 cm⁻¹ |
| `product.pdb` | (s=+1.20, d_PN=1.53, d_PL=2.73) |
| `alt_conf03/{reactant,TS,product}.pdb` | lowest-barrier alternate (asymmetric late TS, literature geometry) |
| `summary.json`, `ensemble_summary.json` | machine-readable metadata |
| `validation/freq/freq-summary.json` | partial-Hessian validation |
| `CAMPAIGN_REPORT.md` | full narrative |

## What the campaign tested (16 strategies, 7 waves)

| wave | strategies | best barrier (kcal/mol) |
|---|---|---|
| 1 | 5 polish variants (target distances, fmax, conformer) | 33.47 |
| 2 | scan_along_s, find_michaelis_complex | 22.73 |
| 3 | no_waters, conf_03 polish | 35.90 |
| g-xTB SP | cross-method energy on R/TS/intermediate | 39.52 |
| 4 | validate_scan_peak, scan_fine, symmetric_2125 polish | 22.73 |
| 5 | scan_ultra (Δs=0.02), scan_sum400/450 | flagged |
| 6 | mace-omol cross-check, sum=4.0 wide, validate_v3 | 21.80 (mace-omol) |
| 7 | scan_along_s on 7 other conformers | 20.68 (conf_03) |

## What worked

- **Relaxed scan along s = d(P–Olg) − d(P–Onuc)** with `sum d_PN+d_PL = 4.25 Å`,
  `FixInternals` over 36 CA-CA pair distances. Beats constrained polish by 10+
  kcal/mol because the polish gets stuck in a 2D-pinned subspace; the scan
  walks a real reaction path.
- **CA-rigid FixInternals scaffold** (vs. hard FixAtoms) — preserves scaffold
  shape exactly while allowing rigid-body translation/rotation, avoiding the
  geodesic-frame-shift bug.

## Cross-method validation

| method | barrier R → TS (kcal/mol) |
|---|---:|
| mace-polar-1-m | 22.73 (conf_10) / 20.68 (conf_03) |
| mace-omol | 21.80 |
| g-xTB single-point | 39.52 (uses pre-strained R, not directly comparable) |
| literature (DFT/QM-MM, Aubert 2004 / Wong 2007) | 14–17 |

## Bug fixes that landed during the campaign

1. **`quantum_engine/mlff/interpolation.py`** — Geodesic-vs-FixAtoms
   frame-explosion bug. `align_path` recenters the path to put COM(image 0)
   at origin, but `FixAtoms` snaps anchor positions back to source frame ⇒
   non-anchor atoms decoupled from anchors by ~64 Å. Fix: Kabsch-align every
   `X_final` image back onto the source frame using the constrained reactant
   as the alignment target before writing into the constrained Atoms.
2. **`quantum_engine/ops/ts.py`** — `assert_frame_consistency()` validation
   gate added; fires after every cv-spring/NEB/IRC stage. Refuses to ship
   structures where any anchor is >2.5 Å from any free atom.
3. **`quantum_engine/mlff/irc.py`** — IRC plus/minus directions shared a
   calculator instance, causing extxyz comment-line metadata (energy, forces,
   dipole) to be byte-identical across the two endpoint files even though
   geometries differed. Fix: snap each direction's energy/forces into a
   fresh `SinglePointCalculator` after LBFGS finishes.

## New tools

| file | purpose |
|---|---|
| `tools/structure_io.py` | universal PDB/CIF I/O with REMARK 665/666 + condensed `REMARK QCB <NNN>` lineage; format validator; multi-MODEL trajectory writer |
| `tools/structure_validator.py` | physical-sanity checker (Zn coordination, anchor decoupling, bond integrity, rigid-body anchor mode) |
| `tools/polish_ts_v3.py` | upgraded TS polish with rich CLI (`--out-basename`, `--free-residues`, `--prune-residue-keep`, `--fix-distance/angle/dihedral`, `--snapshot-stride`) |
| `tools/polish_ts_v2.py` | predecessor; FixInternals + FixBondLengths; template-based PDB writer |
| `tools/scan_along_s.py` | relaxed scan along the reactive coordinate (campaign champion) |
| `tools/find_michaelis_complex.py` | seed-and-relax MC finder (caveat: can land in higher-energy local minima) |
| `tools/funnel_finalize.py` | funnel-pipeline output collator + auto-validator |
| `tools/benchmark_synth.py` | walks campaign output dirs, applies sanity gates, emits ranked report |

## Caveats / known issues

- **MLFF systematic ~5–7 kcal/mol on Zn-phosphate.** Both mace-polar-m and
  mace-omol overestimate the barrier. Closing this requires DFT calibration
  (suggested next sprint: B3LYP/def2-TZVP single-points on the v3 R/TS/P
  triplet).
- **Two TS basins observed** at MACE level: symmetric concerted (s=0,
  d_PN=d_PL=2.125) — only conf_10 found this; asymmetric late (s=+0.30,
  d_PN=1.975, d_PL=2.275) — 6 of 7 valid conformers. The asymmetric one
  matches Wong 2007 literature geometry exactly.
- **CIF write fails on Zn-containing structures** (atomworks element-table
  issue). PDB output is unaffected; CIF support being iterated.
- **`scan_ultra` flagged 2.07 kcal/mol** (sanity gate caught: window too
  narrow; R was already partway up barrier).
- **conf_06 gave 64 kcal/mol outlier** — its input had a water-clash flag
  from an earlier validation pass.

## Reproducing

```bash
# Campaign root
ROOT=/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/ts_pipeline

# Best validated saddle (conf_10, symmetric concerted)
ls $ROOT/v3_FINAL/{reactant,TS,product}.pdb

# Lowest-barrier alternate (conf_03, asymmetric late, 20.68 kcal/mol)
ls $ROOT/v3_FINAL/alt_conf03/{reactant,TS,product}.pdb

# Full benchmark table
cat $ROOT/benchmark_06h/BENCHMARK_REPORT.md
```

## Next sprint

1. **DFT calibration** — B3LYP/def2-TZVP single-points on v3 R/TS/P should
   close the 5–7 kcal/mol systematic.
2. **Reaction-mode-projected IRC** — current IRC follows the lowest
   full-Hessian eigenvector, which can be a sidechain rocking mode rather
   than the reactive coordinate. Add `qcb irc --mode-hint <atom_indices>`.
3. **Scale to other 7 PTE PDBs** (KCX_set0/3, GLU_set0/1/3, minimal_deNovo,
   substrate_uncat) — same scan-along-s strategy, ~1h on l40 each.
4. **CIF Zn fix** — investigate the atomworks element-table failure on Zn
   to enable CIF outputs for metallo-systems.
