# active_site_refine

Refine an AlphaFold3 prediction of a designed enzyme to recapitulate the
catalytic geometry encoded by its design template — without ever moving the
ligand transition state.

## Why

A theozyme design PDB contains a DFT-quality TS geometry (the ligand HETATM
complex) plus catalytic residues placed *exactly* where they need to be to
stabilise that TS. AlphaFold3 then predicts the *protein only*, and the
catalytic residues land somewhere close but not perfect. Naive minimisation
of the AF3 model with the ligand inserted loses the TS bond lengths and
drifts away from the design intent.

This application:

1. Treats the entire ligand HETATM complex as a **rigid body** (`$fix` in xTB).
2. Builds a "design contact map" of catres-atom → ligand-atom distances
   from the design PDB, then applies them as tiered harmonic distance
   restraints (`$constrain`) on the AF3 model — pulling AF3's active site
   toward the design pose.
3. Handles post-translational modifications (currently KCX = carbamylated
   lysine) by stripping the right number of protons and adjusting net charge.
4. Allows the catalytic residues' backbone (±N residues) to flex during
   relaxation, so the protein can adapt to fit the ligand.

## Required inputs

- **`design.pdb`** — the design model, must include `REMARK 666` lines
  (RFdiffusion / Rosetta theozyme convention) and the ligand as `HETATM`.
- **`<af3>_aligned.pdb`** — the AF3 prediction *already aligned* to the
  design and with the ligand transferred. Produced by:

  ```bash
  /net/software/containers/universal.sif \
      /home/woodbuse/special_scripts/general_utils/align_prediction_to_ref_pdb_and_copy_lig.py \
      --ref_pdb design.pdb --pdb_for_alignment af3_pred.pdb \
      --output_dir aligned/ --catres_subset 1,2,3,4,5,6 \
      --outlier_threshold 0.5 --convergence_threshold 0.001 \
      --max_iterations 50 --min_residues 10 \
      --sidechain_cycles 3 --sidechain_fine_grid 5.0 \
      --winner_threshold 0.1 --verbose
  ```

## Recommended command (from the YYE/Zn₂ benchmark)

```bash
python enzyme_design_applications/active_site_refine/refine.py \
    design.pdb aligned/af3_pred_aligned.pdb \
    -o refined.pdb \
    --ptm A/LYS/3:KCX \
    --ligand-charge "YYE:1" \
    --backend xtb --gfn 0 \
    --radius 6.0 \
    --unfreeze-shell 1
    # angle-restraints are ON by default; --no-angle-restraints to disable
    # k-scale = 1.0 default
```

This combination — **xtb-FF (GFN-FF) + design-contact distance restraints +
sidechain pivot-angle restraints + closed-shell auto-charge** — gave the
best composite score on the test case (YYE/Zn₂/KCX):

  contact_mae 0.035 Å · metal_mae 0.020 Å · angle_mae 1.7° · ligand_rmsd 0 Å

Catalytic geometry recapitulation:
  HIS41 NE2-Zn2: design 2.023, AF3 3.218, **refined 2.067** Å
  HIS41 CA-CB-CG: design 115.2°, AF3 112.9°, **refined 116.3°**
  LYS64 NZ-C1 (KCX): design 1.382, AF3 1.427, **refined 1.402** Å

`--ptm CHAIN/RES/CAT_IDX:NCAA` follows the same syntax as the alignment
script: `A/LYS/3:KCX` means "the catalytic residue in REMARK 666 slot 3
(`A LYS 64`) is post-translationally modified to KCX". Multiple `--ptm`
flags allowed.

`--ligand-charge RESNAME:N` declares HETATM net charges. Defaults: ZN +2,
MG +2, etc. Ambiguous metals (Fe/Mn/Cu/Ni/Co/Mo/W) emit a warning and use
their most-common state — pass an explicit value to be safe.

## Backend choice

| backend | best for | speed (278 atoms / CPU) |
|---|---|---|
| `xtb --gfn 0` (GFN-FF) | **default** — best angle preservation, very fast | ~3 s |
| `xtb --gfn 2` (GFN2-xTB) | better electronic structure (charges, polarisation) | ~5–10 min |
| `g-xtb` | Grimme's wB97M-V approximator (pre-release) | similar to GFN2 |
| `mace-mp` | r2SCAN-trained MLFF, all elements | ~3–4 min |
| `mace-omol` | charge-aware, TS-trained | ~5–8 min |
| `mace-polar-m` | polarisable (Seth's gold standard for ionic systems) | ~5 min |

xtb-FF wins on this Zn₂ test because force-field bond/angle terms hold
sidechain valence in place better than MLFFs without explicit angle terms;
combine that with our angle restraints from design and you preserve both
the design's Zn-coordination AND realistic sp3 geometry.

For systems where electronic effects dominate (e.g., redox cofactors, very
ionic complexes) try `--backend mace-polar-m` or `--backend xtb --gfn 2`.

## Outputs

- `refined.pdb` — full structure with refined active site stitched back
- `refined_cluster/input.pdb` — what was sent to xTB (cluster + caps)
- `refined_cluster/refined.pdb` — cluster post-relax

A deviation table is printed to the log: design-distance, AF3 starting
distance, final distance, and per-contact deltas — useful for spotting
which catres got pulled into place vs. which fought the restraint.

## PyMOL inspection

```text
load design.pdb
load af3_pred_aligned.pdb
load refined.pdb
load refined_cluster/refined.pdb, cluster_relaxed
```

## What lives in `qcb` vs. here

The `qcb` library provides the building blocks (PDB I/O conventions, xTB
runners, protonation utilities). This application provides the
*opinionated workflow* on top — design contact map, KCX-aware protonation
strip, restraint scheduling. If a piece of logic stops being design-specific
and becomes reusable, promote it back into `qcb`.
