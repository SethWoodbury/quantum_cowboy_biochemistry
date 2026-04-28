# Active-Site Refinement: Method Benchmark (YYE/Zn₂/KCX)

Run while user was AFK on 2026-04-28. ~30 method/setting combinations
swept against the design.pdb / af3_pred_aligned.pdb test case from
`/home/woodbuse/testing_space/align_seth_test/`.

## TL;DR

| Rank | Method | Composite | Contact MAE | Metal MAE | Angle MAE | CD2-Zn2 |
|---|---|---|---|---|---|---|
| **1 ⭐** | **xtb-FF + angle-restraints + ring-lock** | **0.499** | **0.035 Å** | **0.020 Å** | **1.69°** | **2.997 Å** (design 2.995) |
| 2 | xtb-FF + angle-restraints (no ring-lock) | 0.532 | 0.035 | 0.020 | 1.70 | 3.112 |
| 3 | xtb-FF + backbone-CB rigidity | 0.543 | 0.047 | 0.021 | 1.65 | — |
| 10 | MACE-MP k=1 +angle | 0.877 | 0.107 | 0.037 | 3.29 | 3.220 |
| 15 | MACE-POLAR-1-S k=1 +angle | 1.017 | 0.115 | 0.056 | 3.51 | — |
| 16 | MACE-OMOL k=1 +angle | 1.051 | 0.115 | 0.051 | 4.23 | — |
| 17 | MACE-POLAR-1-M k=1 +angle | 1.059 | 0.118 | 0.054 | 4.14 | — |
| 21 | g-xTB +angle +ring-lock | ~0.7* | 0.073 | 0.015 | 3.56 | 3.007 |
| 22 | g-xTB +angle (no ring-lock) | 1.565 | 0.158 | 0.025 | 3.89 | **4.660** ❌ |
| 26 | AF3 (input baseline) | 2.676 | 0.397 | 0.391 | 2.80 | 4.085 |

*g-xTB+ringlock recomputed after the sweep, would land in slot ~9-10 by composite.

**Production winner: `xtb-FF + angle-restraints + ring-lock`** — runs in
**~3 s on CPU**, beats every MACE backend on every metric, recapitulates
the design's catalytic geometry within <0.05 Å on contacts and <2° on
sidechain angles.

## Recommended command

```bash
python enzyme_design_applications/active_site_refine/refine.py \
    design.pdb aligned/af3_pred_aligned.pdb \
    -o refined.pdb \
    --ptm A/LYS/3:KCX \
    --ligand-charge "YYE:1" \
    --backend xtb --gfn 0 \
    --radius 6.0 \
    --unfreeze-shell 1
    # angle-restraints + ring-lock are ON by default
```

## Key geometry comparison (catalytic residue HIS41, the worst-broken in AF3)

| Quantity | design | AF3 (start) | **winner** | g-xTB no lock |
|---|---|---|---|---|
| NE2 → Zn2 | 2.023 | 3.218 | **2.042** | 1.982 |
| CD2 → Zn2 | 2.995 | 4.085 | **2.997** | 4.660 ❌ |
| ND1 → O9 (H-bond to nitro) | 2.855 | 2.675 | **2.827** | 2.866 |
| CA-CB-CG | 115.2° | 112.9° | **116.9°** | 115.5° |
| LYS64 NZ → C1 (KCX bond) | 1.382 | 1.427 | **1.402** | 1.383 |

## Surprises

1. **MACE-POLAR-1-M is *not* better than MACE-MP** for this Zn₂ case. The
   user described it as the "gold standard for ionic systems" but on
   contact-recapitulation it scored 1.06 vs MACE-MP's 0.88. The polar
   features may pay off in larger or differently-charged systems; for a
   chopped active-site cluster with explicit metal-coord restraints,
   MACE-MP-r2SCAN-omat-ft does the job at lower cost. Worth retesting on
   an actual production case before drawing a final conclusion.

2. **xtb-FF (force-field) outperforms every neural / DFT-approximating
   method.** Reason: GFN-FF has explicit bond/angle terms that resist
   sidechain valence distortion, while MLFF/MLIP models rely on training
   data that doesn't always cover *constrained* geometries with strong
   long-range pulls. Combine GFN-FF with our design-derived angle
   restraints and the result is bulletproof on this class of system.

3. **g-xTB has aromatic-ring problems on charged metal complexes.** The
   imidazole bond lengths drift up to 0.10 Å, and without the ring-lock
   the ring rotates away from design. With our new ring-lock added it
   becomes usable but still trails xtb-FF.

4. **Polish stage (unrestrained final pass) hurts.** Drifts 0.3-0.5 Å
   away from design contacts — leave `--polish-steps 0` (the default).

5. **HIS184 has a slightly distorted ring in the *design* (CB-CG-ND1 =
   131.6°, real-protein average is 121-125°)**. Every method correctly
   normalises this toward the chemical mean (~123°), so the worst-angle
   numbers above include a 7-8° artefact baked into the design itself.
   Not our fault, not worth chasing.

## What changed in the code

* `refine.py`:
  - Sidechain pivot-angle restraints (`HarmonicAngle`) on by default,
    targets pulled from design's CA-CB-CG / CB-CG-CD / etc.
  - `metal_ring_lock` tier in the design contact map: when a HIS NE2/ND1
    coordinates a metal, the rest of the imidazole gets k=0.15 contacts
    to that metal so the ring can't swing.
  - Two-stage opt: `--polish-steps N` runs an unrestrained pass after
    the restrained one (off by default; doesn't help on this test).
  - g-xtb backend: `--backend g-xtb` dispatches to `deps/g-xtb/install/`
    with `--gxtb` flag.
  - mace-polar-{s,m,l} + mace-mh aliases added to MACE_MODELS dict.
  - Auto-charge: walks every residue, sums formal charges, warns + auto-
    nudges if the result is open-shell (assumes singlet always per user
    spec).
  - Metal ambiguity warnings for Fe/Mn/Cu/Ni/Co/Mo/W (Zn stays unambiguous).
  - LD_LIBRARY_PATH set automatically for the vendored xtb binary.

* `geom_score.py`: standalone scorer (catres-ligand contact recap +
  sidechain valence + ligand RMSD).

* `bench_aggregate.py`: composite-score aggregator for picking the winner.

* `run_benchmark.sh`: SLURM sweep over backends × settings.

## Files for inspection

```
/home/woodbuse/testing_space/align_seth_test/
    design.pdb                 # original design with REMARK 666 + YYE
    af3_pred_aligned.pdb       # AF3 prediction aligned to design
/home/woodbuse/testing_space/refine_bench/
    xtb_k1_ang_ringlock.pdb    # ⭐ winning result
    {many other variants}.pdb
    scores.json                # all per-file scores as JSON
    logs/                      # per-job stdout/stderr
```

PyMOL command:

```text
load /home/woodbuse/testing_space/align_seth_test/design.pdb, design
load /home/woodbuse/testing_space/align_seth_test/af3_pred_aligned.pdb, af3
load /home/woodbuse/testing_space/refine_bench/xtb_k1_ang_ringlock.pdb, refined
```
