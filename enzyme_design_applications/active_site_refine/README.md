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

## Run

```bash
python enzyme_design_applications/active_site_refine/refine.py \
    design.pdb aligned/af3_pred_aligned.pdb \
    -o refined.pdb \
    --ptm A/LYS/3:KCX \
    --gfn 0 \
    --radius 6.0 \
    --unfreeze-shell 1
```

`--ptm CHAIN/RES/CAT_IDX:NCAA` follows the same syntax as the alignment
script: `A/LYS/3:KCX` means "the catalytic residue in REMARK 666 slot 3
(`A LYS 64`) is post-translationally modified to KCX". Multiple `--ptm`
flags allowed.

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
