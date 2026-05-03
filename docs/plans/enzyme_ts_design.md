# Enzyme TS Design — Generic Pipeline

Cross-links: [`00_index.md`](00_index.md) · [`mcsa_theozyme.md`](mcsa_theozyme.md)

## Goal

Given:

- SMILES of reactant
- SMILES of product
- Cropped enzyme active site PDB

produce:

- One or more docked TS complexes (TS embedded in active-site cluster)
- High-resolution TS-in-protein geometry (frequency- and IRC-verified)
- Multi-TS `.cif` output with barriers and a residue-movement report

The pipeline is enzyme-agnostic and assumes only that the user has already
identified an active-site crop. It is the substrate path used when no
M-CSA entry is available; for M-CSA-driven runs, see
[`mcsa_theozyme.md`](mcsa_theozyme.md).

## Inputs

- `--smiles-r SMILES` reactant
- `--smiles-p SMILES` product
- `--active-site PDB` cropped active site (post-extraction)
- `--cofactor PDB` optional cofactor (separate file, kept rigid by default)
- `--metals JSON` optional metal-coord spec `[{"resid":..., "ligands":[...]}]`
- `--catalytic-residues LIST` optional explicit catalytic residue list
- `--shell-radius R` tier-2 expansion radius (Å, default 6.0)

## Outputs

- `out/ts_<n>.cif` per discovered TS
- `out/barriers.json` `{ts_id: {dE_forward, dE_reverse, freq_imag}}`
- `out/residue_movement.json` per-residue RMSD (CA, sidechain) before/after
- `out/conformers/` Stage 4 ensemble (kept for re-runs)

## Stage flow

Each stage is a `quantum_engine.pipelines.Step` adapter; alternate tools
are selectable via `--stage<N>-tool=<name>`. Default tool listed first.

### Stage 1 — Reaction parsing (CPU, local)

- Tools: `rdkit` + `indigo` + `rxnmapper` + `cgrtools`
- Input: SMILES R, SMILES P
- Output: atom-mapped reaction; bond-make/break list; net charge of
  reacting fragment; driving-coordinate vector for Stage 7
- Cross-validate atom map between RDKit and Indigo. RXNMapper supplies the
  mapping; CGRtools computes the bond delta graph.
- Net charge propagates to Stage 3 (system charge) and Stage 8 (Sella).

### Stage 2 — Vacuum TS guess (GPU optional, local or SLURM)

- Default: `autode`
- Alts: `scine_chemoton` (Chemoton + ReaDuct), `molecular_gsm` (single-ended GSM)
- Adapter contract: emits `ts_guess.xyz`, `path.xyz` (image series),
  `irc_check_passed: bool`
- autodE: SMILES → reaction profile via xTB or ORCA backend
- Chemoton/ReaDuct: arrow-pushing parseable mechanisms only; preferred
  when input is M-CSA-derived
- GSM: single-ended driving coords from Stage 1 bond-deltas; no product
  geometry needed

### Stage 3 — Active-site prep (CPU, local)

- Tools: `pdbfixer` + `openmm` + `propka` + (optionally) `reduce`
- Protonate, fix gaps, cap fragment termini, assign formal charges
- System charge = Stage 1 reactant charge + active-site residue charges
- Metal-coord respected from `--metals`; coordination geometry frozen if
  requested
- Output: `prepped_site.pdb`, `system_charge.json`

### Stage 4 — TS conformer generation (CPU, local)

- Tools: `rdkit` ETKDGv3 + `crest`
- Generate up to N conformers of the Stage 2 TS geometry constrained to
  the bond-make/break distances from Stage 1
- ETKDGv3 supplies a 3D embedding ensemble; CREST refines with GFN2-xTB
  and filters by energy + RMSD
- Output: `conformers/ts_conf_*.xyz`

### Stage 5 — Dock TS into active site (CPU, local)

- Tool: in-house constraint-based placement
  (`quantum_engine.prep.dock_ts`)
- Anchors: catalytic residue side-chain heavy atoms + metal centers
- Constraints: preserve formed/breaking bonds (Stage 1), respect metal
  coordination (`--metals`)
- Stage emits N candidate poses ranked by clash + interaction score
- Output: `poses/pose_<i>.pdb` ranked

### Stage 6 — Iterative refinement (GPU, SLURM L40)

- Default calc: MACE-POLAR-1M
- Cheap calc: g-xTB (`GXTB_BIN`) for fast pre-screen
- Constraint regime: CA atoms fixed by default (or harmonic restraint via
  `pysisyphus` `SpringConstraint`). Backbone heavy atoms breathe;
  side-chains fully relaxed.
- Adapter exposes `--constraint-mode` from `scripts/run_neb_ts.py`:
  `ca-only` (default), `backbone-harmonic`, `none`, custom
- Iterate: relax → check geometry → re-dock if drift > threshold

### Stage 7 — In-protein path re-find (GPU, SLURM L40)

- Default: `pyGSM` single-ended with driving coords from Stage 1
- Alts: `pysisyphus_neb` double-ended NEB, `readuct_bspline`
  double-ended b-spline
- Adapter: emits `ts_in_pocket.xyz`, `path.xyz`, `barrier_estimate`
- Single-ended GSM is preferred when no in-protein product geometry
  exists; double-ended NEB needs both endpoints relaxed in-pocket.

### Stage 8 — High-res TS polish (GPU, SLURM L40 — or CPU output)

- Tool: `sella` saddle-point with MACE-POLAR-1M
- Frequency check (one imaginary mode, magnitude > 50 cm⁻¹)
- IRC verification: forward + reverse trajectories should connect to the
  reactant and product geometries from Stage 7
- Output written from CPU node:
  - `ts_<n>.cif` per polished TS
  - `barriers.json`
  - `residue_movement.json`

## Two-tier residue selection

Residue selection feeds Stage 3 (active-site crop) and is exposed at run
time so a single PDB input can be re-cropped quickly.

- **Tier 1 — catalytic only**: residues from `--catalytic-residues` (or
  Stage 1's reacting-fragment contact list if the flag is omitted).
- **Tier 2 — distance-expanded**: tier 1 + every residue with any heavy
  atom within `--shell-radius R` Å of tier 1 (default R = 6).
- **Tier 2-motif**: tier 1 + motif-aware fill-in. Motifs encoded:
  - `HExxH` zinc-metalloprotease
  - `Ser-His-Asp/Glu` triad
  - `Cys-His` / `Cys-Asp-His` dyad/triad
  - `Asp-Asp` carboxylate diad
- Motif library lives at `quantum_engine.prep.motifs`. Adding a motif =
  adding a `Motif` dataclass + regex matcher.

## CPU vs GPU split

| Stage | Locality | Notes |
|---|---|---|
| 1 | CPU local | RDKit / Indigo / RXNMapper / CGRtools |
| 2 | CPU or GPU | autodE on CPU; Chemoton/ReaDuct on GPU SLURM if MACE backend |
| 3 | CPU local | PDBFixer + OpenMM minimisation |
| 4 | CPU local | RDKit + CREST (CREST uses xTB only) |
| 5 | CPU local | constraint solver |
| 6 | GPU SLURM (L40) | MACE-POLAR-1M; g-xTB for pre-screen |
| 7 | GPU SLURM (L40) | pyGSM / NEB / b-spline |
| 8 | GPU SLURM (L40) | Sella + MACE; CPU writes final `.cif` |

Orchestrator polls SLURM for Stage 2/6/7/8 jobs. **Use L40, never A4000**
(see `~/.claude/.../feedback_gpu_queues.md` — A4000 queue is 13 days vs
hours on L40).

## Constraints contract

All optimisation stages share a constraint specification grammar
implemented in `quantum_engine.io.constraints`. Stage adapters consume the
same grammar so users can override at any point.

```bash
# Default: CAs fixed in MD, backbone harmonic in opt, none in TS polish
qcb-enz-ts run config.yaml \
    --stage6-tool=mace-polar \
    --stage6-constraint-mode=ca-only \
    --stage8-constraint-mode=none
```

`--constraint-mode` flag is honored by `scripts/run_neb_ts.py` already and
will be re-used verbatim by the orchestrator.

## DFT policy

- **No DFT in default loop.**
- High-res = MACE-POLAR-1M.
- Fast eval = g-xTB.
- Gaussian / ORCA only as offline validators run on demand against the
  final `ts_<n>.cif`. Hooks live in `quantum_engine.qm.gaussian` and
  `quantum_engine.qm.orca`.

## Swappability

Every stage tool is a `Step` adapter implementing
`quantum_engine.pipelines.Step`. CLI flag pattern:

```bash
qcb-enz-ts run \
    --stage2-tool=scine_chemoton \
    --stage7-tool=readuct_bspline \
    --stage8-tool=sella
```

Adapter discovery via entry-points in `pyproject.toml`:

```toml
[project.entry-points."qcb.enz_ts.stages"]
"stage2.autode" = "quantum_engine.adapters.autode:Autode"
"stage2.scine_chemoton" = "quantum_engine.adapters.scine:Chemoton"
"stage2.molecular_gsm" = "quantum_engine.adapters.gsm:MolecularGSM"
"stage7.pygsm" = "quantum_engine.adapters.pygsm:PyGSM"
"stage7.pysisyphus_neb" = "quantum_engine.adapters.pysis:NEB"
"stage7.readuct_bspline" = "quantum_engine.adapters.readuct:BSpline"
```

## CLI sketch

```bash
qcb-enz-ts run \
    --smiles-r "CC(=O)OCC" \
    --smiles-p "CC(=O)O.CCO" \
    --active-site site.pdb \
    --catalytic-residues "HIS:55,ASP:102,SER:195" \
    --shell-radius 6.0 \
    --stage6-constraint-mode ca-only \
    --outdir runs/esterase_001
```

Same DAG as `qcb run config.yaml`, just with reaction-aware stages
(see `quantum_engine.pipelines.Pipeline` contract).

## Open items

- Stage 5 dock scorer needs a reaction-aware term (currently a
  steric/clash-only solver).
- Stage 7 b-spline adapter for ReaDuct still TODO.
- Need a reaction-class router so Stage 2 default flips to Chemoton when
  arrow-pushing XML is supplied (M-CSA path inherits this).
