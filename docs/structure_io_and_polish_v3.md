# Structure I/O + polish_ts_v3 — generalizable TS polish driver

**Experimental — added during 2026-05-05 PTE benchmark campaign.**

Successor of `polish_ts_v2.py` with richer CLI for output control, custom
constraints, REMARK lineage, and trajectory snapshots. Backed by a new
universal I/O module `tools/structure_io.py`.

## What's new vs `polish_ts_v2`

| feature | v2 | v3 |
|---|---|---|
| Output basename | hardcoded `transition_state.pdb` | `--out-basename FOO` (default `<input_stem>_polished`) |
| Output formats | PDB only | PDB + CIF (atomworks-backed; flags `--no-pdb`/`--no-cif`) |
| Format validation | post-write physical check | strict PDB column-alignment check (`structure_io validate`) |
| REMARK 666 enzyme-matcher | preserved if present | **auto-added** for chain-A protein residues |
| REMARK 665 legend | absent | included by default |
| REMARK QCB lineage | flat free-form | **codified** (`REMARK QCB <NNN> <LABEL> key=value …`) |
| Free residues | none | `--free-residues 131,254` (excluded from CA-rigid scaffold) |
| Backbone pruning | none | `--prune-backbone-residues 131` |
| Sidechain pruning | none | `--prune-residue-keep 169:CD,CE,NZ` |
| Custom Cartesian fix | none | `--fix-atoms 177,180,188` |
| Custom distance/angle/dihedral | none | `--fix-distance i,j,d`, `--fix-angle`, `--fix-dihedral` |
| Trajectory snapshots | none | multi-MODEL PDB at `--snapshot-stride N` |
| Input copy | none | always copies input PDB into out dir |
| README per run | none | auto-generated |

## QCB code registry (`REMARK QCB <NNN>`)

```
001 = TOTAL_CHARGE
002 = ENERGY
003 = STAGE
004 = METHOD
005 = REACTIVE_DISTANCES
006 = CHARGE_RESIDUES
007 = METAL_RELABEL
008 = NUCLEOPHILE_RELABEL
009 = FORMAL_CHARGES
010 = CARBAMATE_RELABEL
011 = LIGAND_RENAME
012 = CARBOXYLATE_CHARGE
013 = WATER_RELABEL
014 = WITHOUT_F132
015 = FRAME_VALIDATION
016 = CONSTRAINT_PATTERN
017 = CONVERGENCE
018 = B_FACTOR_MEANING
019 = PROVENANCE
099 = NOTE
```

Use `python tools/structure_io.py legend` to list the registry. The `REMARK 665`
description lines that go into every output PDB describe the format so external
parsers (e.g. Rosetta enzyme-matcher consumers) can pick up the metadata.

## Quick reference

### Format validation
```bash
python tools/structure_io.py validate path/to/structure.pdb
# → JSON: {ok, n_atoms, n_remarks, n_remarks_666, issues: [...]}
```

### Lineage inspection
```bash
python tools/structure_io.py remarks path/to/structure.pdb
# → prints REMARK 666 matcher entries + REMARK QCB lineage
```

### Trajectory → multi-MODEL PDB
```bash
python tools/structure_io.py traj-from-ase \
    --template-pdb path/to/structure.pdb \
    --ase-traj path/to/optimization.traj \
    --out path/to/trajectory.pdb \
    --stride 10
```

### polish_ts_v3 — common patterns

**Default polish** (auto-named output, both PDB+CIF):
```bash
python tools/polish_ts_v3.py \
    --input my_input.pdb \
    --out my_run/ \
    --target-d-p-onuc 2.00 --target-d-p-olg 2.25
# Outputs: my_run/{my_input_polished.pdb, my_input_polished.cif,
#                  my_input.pdb, my_input_polished_summary.json, README.md}
```

**Custom basename, PDB only**:
```bash
python tools/polish_ts_v3.py \
    --input my_input.pdb --out my_run/ \
    --out-basename pte_KCX_set1_polished \
    --no-cif
```

**Free TRP131 from rigid scaffold + prune LYS169 to amine warhead**:
```bash
python tools/polish_ts_v3.py \
    --input my_input.pdb --out my_run/ \
    --free-residues 131 \
    --prune-residue-keep 169:CD,CE,NZ
```

**Custom geometric fixes**:
```bash
python tools/polish_ts_v3.py \
    --input my_input.pdb --out my_run/ \
    --fix-atoms 177,180,188 \
    --fix-distance 177,180,2.0 177,188,2.25 \
    --snapshot-stride 25
```

**Dry-run** (resolve constraints, write README, skip SCF):
```bash
python tools/polish_ts_v3.py --input ... --out ... --dry-run
```

## Output directory contents

```
<out_dir>/
├── <input_basename>.pdb              ← copy of input for reference
├── <basename>.pdb                    ← polished structure
├── <basename>.cif                    ← polished structure (mmCIF; bonds + charges)
├── <basename>_trajectory.pdb         ← multi-MODEL optimizer trajectory
├── <basename>_polish.log             ← LBFGS log
├── <basename>_summary.json           ← machine-readable result
├── <basename>_traj.traj              ← raw ASE Trajectory file
└── README.md                         ← per-run docs
```

## Caveats

- **CIF write currently has issues with Zn-containing systems** (atomworks element
  table issue). PDB output is unaffected. CIF support is being iterated.
- **`--fix-angle` / `--fix-dihedral`** flags are parsed but not yet wired through
  the constraint setup — falls back to a warning. Use `FixInternals` directly in
  Python if you need angle/dihedral pinning today.
- **Pruning** (`--prune-residue-keep`, `--prune-backbone-residues`) creates
  dangling bonds without H caps in this initial implementation. The downstream
  optimizer is robust to small clashes but consider a manual H-cap for
  publishable structures.

## Universal validator (`structure_io.validate_pdb_format`)

Independent of physical sanity — checks ONLY column alignment / record types:
- ATOM/HETATM record length ≥ 78 cols
- serial col 7-11 must be int
- resseq col 23-26 must be int
- coord cols 31-54 must be parseable F8.3
- element col 77-78 must be alpha
- altloc col 17 must be space or single letter

Returns `{ok: bool, n_atoms, n_remarks, n_remarks_666, issues: list[str]}`.

Bake into a SLURM step:
```bash
python tools/structure_io.py validate "$OUT/polished.pdb" || exit 1
```

## Related tools

- `tools/structure_validator.py` — physical sanity (Zn coordination, bond
  integrity, anchor decoupling). Different scope from `structure_io.py`'s
  format-only checks.
- `tools/funnel_finalize.py` — funnel-pipeline output collator (uses lineage).
- `quantum_engine/io/cif.py:write_ts_cif` — atomworks-backed CIF writer with
  WBO + RDKit kekulization; foundation of `polish_ts_v3`'s CIF output.
- `tools/scan_along_s.py` — reactive-coordinate relaxed scan (the campaign
  champion that found the 13.12 kcal/mol TS).
- `tools/benchmark_synth.py` — campaign synthesis report.
