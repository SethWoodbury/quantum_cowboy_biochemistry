# enzyme_design_applications

End-to-end workflows that combine **protein-design tools** (RFdiffusion outputs,
AlphaFold3 predictions, TMalign-based structural alignment, sidechain dihedral
optimisation) with **quantum-chemistry tools** from the `qcb` library
(xTB, MACE-MLFF, NEB-TS, …).

Each subdirectory is a self-contained application: its own README, its own
CLI entry point, its own example inputs. The `qcb` package itself stays free
of design-specific assumptions — these directories are where opinionated,
project-specific compositions live.

## Contents

| Directory | What it does |
|---|---|
| `active_site_refine/` | Take a design PDB + its AF3 prediction, build a design "contact map" of catalytic-residue ↔ ligand distances, and run an xTB constrained optimisation that drives the AF3 active site toward the design intent (rigid-ligand restraint + KCX protonation handling). |

## Adding a new application

```text
enzyme_design_applications/
└── my_new_app/
    ├── README.md           # what this does, when to use it, example commands
    ├── my_new_app.py       # CLI entry point
    └── (optional) batch.py # parallel runner for high-throughput use
```

Keep the bar high: applications belong here only if they (a) compose multiple
toolchains, (b) have a reproducible CLI, and (c) are meant for end users — not
for internal helper code, which lives in `scripts/`.
