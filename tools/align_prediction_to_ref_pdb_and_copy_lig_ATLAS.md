# Align Prediction to Ref PDB and Copy Lig: Function-by-Function Atlas

**File:** `tools/align_prediction_to_ref_pdb_and_copy_lig.py` (2917 lines)
**Purpose:** Advanced PDB alignment for AlphaFold3 predictions with designed enzyme structures, featuring multiple alignment strategies, iterative outlier removal, sidechain dihedral optimization, and comprehensive catalytic-residue metrics.

---

## Table of Contents

1. [Data Structures](#data-structures)
2. [NCAA Processing & PTM Handling](#ncaa-processing--ptm-handling)
3. [Symmetric Atom Handling](#symmetric-atom-handling)
4. [PDB Utilities & REMARK Parsing](#pdb-utilities--remark-parsing)
5. [RMSD & Metrics Calculations](#rmsd--metrics-calculations)
6. [Alignment Strategies](#alignment-strategies)
7. [Sidechain Optimization](#sidechain-optimization)
8. [Multi-Strategy Orchestration](#multi-strategy-orchestration)
9. [Main Entry Point](#main-entry-point)
10. [Special-Attention Areas](#special-attention-areas)

---

## Data Structures

### `AlignmentMetrics` (line 327)
Stores metrics for a single alignment strategy. Fields: `strategy_name`, `converged_rmsd`, `n_iterations`, `n_atoms_final`, `all_backbone_rmsd`, `ca_rmsd`, `catres_backbone_rmsd`, `catres_ca_rmsd`, `catres_subset_backbone_rmsd`, `catres_subset_ca_rmsd`, `catres_subset_all_atom_rmsd_before_opt`, `catres_subset_all_atom_rmsd_after_opt` (PRIMARY winner metric), `n_sidechain_opt_iterations`, `sidechain_opt_improvement`, `catres_subset_lddt`, `tm_score`.

### `CatalyticResidue` (line 349)
Metadata from REMARK 666. Fields: `chain: str`, `resname: str`, `resnum: int`, `catres_index: int` (1-indexed).

### `PTMSpec` (line 358)
Specification for a post-translational modification. Fields: `chain`, `canonical_resname` (e.g., 'LYS'), `catres_index`, `ncaa_resname` (e.g., 'KCX'), `atoms_to_cut: List[str]`.

---

## NCAA Processing & PTM Handling

### Global `NCAA_ATOMS_TO_CUT` (lines 85–101)
Maps NCAA names → atoms that distinguish them from canonical parent. **KCX**: `['CX', 'OQ1', 'OQ2']` → cuts to `LYS`. Also handles MLY, ALY, MLZ, M3L, HYP, SEP, TPO, PTR, CSO, CSD, OCS, MSE.

### `parse_ptm_spec(spec_str)` (line 371)
Parses `"CHAIN/RESNAME/CATRES_IDX:NCAA[-ATOMS]"`. Example: `"A/LYS/6:KCX"`.

### `parse_ptm_specs(spec_list)` (line 398)
Multiple specs via list comprehension over `parse_ptm_spec`.

### `process_prediction_pdb(pdb_path, output_path, ptm_specs, catalytic_residues, verbose)` (line 407)
**Critical preprocessing step.** Writes processed PDB.
1. Strip ALL HETATM from prediction (ligands re-added later from reference).
2. For each NCAA: cut specified atoms, convert HETATM → ATOM, rename NCAA → canonical (KCX → LYS).
Maps PTM specs by `(chain, resnum)` for fast lookup. Output is "clean" protein.

---

## Symmetric Atom Handling

Global symmetric atom pairs (lines 114–122): PHE/TYR (CD1↔CD2, CE1↔CE2), ASP/GLU (OD1↔OD2, OE1↔OE2), ARG (NH1↔NH2), LEU/VAL (CD1↔CD2 or CG1↔CG2).

### `_get_symmetric_atom_names(resname)` (line 508)
Returns set of atom names involved in symmetric pairs (empty if none).

### `_apply_swap_to_name(name, resname)` (line 519)
Reverse lookup for symmetric pair.

### `_determine_best_swap_for_residue(ref_res, mob_res, verbose)` (line 531)
Per-residue Kabsch-based decision. Aligns non-symmetric atoms, then compares RMSD with vs. without swap. Returns True if swap improves.

### `resolve_symmetric_atoms(ref_structure, mobile_structure, catalytic_residues, verbose)` (line 621)
Called ONCE after alignment + sidechain opt (line 2568). Returns `Dict[(chain, resnum), should_swap]`. All subsequent metrics (RMSD, lDDT) use this map.

---

## PDB Utilities & REMARK Parsing

### `separate_protein_and_hetatm(pdb_content)` (line 694)
Splits ATOM/TER from HETATM.

### `renumber_pdb_atoms(pdb_lines, start_number=1)` (line 706)
Sequential atom numbering.

### `build_his_tautomer_map_from_raw_pdb(pdb_path, debug=False)` (line 720)
Scans PDB for HIS tautomer state. Returns `Dict[(chain, resnum), "HIS"|"HIS_D"]`. Logic: HD1 present + HE2 absent → HIS_D (delta-protonated), else HIS (epsilon).

### `extract_all_remark_lines(pdb_path)` (line 757)
All HEADER, REMARK, HETNAM, LINK, CONECT lines (deduplicated).

### `parse_remark666_lines(pdb_path)` (line 770)
Returns `(remark_lines, catalytic_residues_list)`. Format expected: `REMARK 666 MATCH TEMPLATE ... MOTIF chain resname resnum catres_index ...`.

### `filter_catres_by_subset(catalytic_residues, subset_indices)` (line 803)
Selects subset of catres by 1-indexed list (or all if None).

### `extract_hetatm_lines(pdb_path)` (line 819)
All HETATM lines from reference (ligands, cofactors, water).

---

## RMSD & Metrics Calculations

### Atom selectors
- `get_backbone_atoms(residue)` line 833 → N, CA, C, O
- `get_ca_atom(residue)` line 842
- `get_all_heavy_atoms(residue)` line 847

### `calculate_rmsd(atoms1, atoms2)` (line 852)
Raw RMSD; returns `np.inf` on mismatch.

### `calculate_rmsd_with_residues(residues1, residues2, atom_selector='ca')` (line 864)
RMSD using `'ca' | 'backbone' | 'all_heavy'`. Returns `(rmsd, atoms1, atoms2)`.

### `calculate_rmsd_with_symmetry(residues1, residues2, catalytic_residues, swap_map, atom_selector='all_heavy')` (line 902)
RMSD WITH symmetric atom handling via swap_map.

### `calculate_catres_subset_lddt(ref, mobile, catres_subset, swap_map, thresholds=(0.5,1.0,2.0,4.0), inclusion_radius=15.0)` (line 971)
**Inter-residue** distance preservation only — atom pairs from DIFFERENT residues within subset (line 1057–1065). Score 0.0–1.0.

---

## Residue Matching

### `match_residues_by_chain_and_number(structure1, structure2)` (line 1101)
Match by `(chain_id, resnum)`.

### `get_catalytic_residues_from_structure(structure, catalytic_residues)` (line 1125)
Extract specified catres from a structure.

---

## Alignment Strategies

### IterativeAligner (Kabsch with outlier removal) — class at line 1298

**Constructor params:** `ref_structure`, `mobile_structure`, `atom_selector='ca'`, `rmsd_threshold=0.5`, `max_iterations=50`, `convergence_threshold=0.001`, `min_atoms=10`, `verbose=True`.

**`align_with_outlier_removal(ref_residues, mobile_residues)` (line 1322)**
1. For up to `max_iterations`:
   - Atoms from `atom_selector`
   - Kabsch superposition (BioPython Superimposer)
   - Apply rotation/translation to mobile structure in-place
   - Identify outliers (per-residue deviation > `rmsd_threshold`)
   - Drop worst outlier
2. Convergence on RMSD diff < `convergence_threshold` or no outliers.
3. Returns `(final_rmsd, n_iterations, kept_residue_indices)`.

### TM-Align via biotite (function at line 1152)

**`tmalign_superimpose(ref_structure, mobile_structure, residue_subset=None, verbose=False)`**
Uses biotite's `superimpose_structural_homologs`. TM-score = (1/L) × Σ(1 / (1 + (d_i/d0)²)) with d0 = 1.24·∛(L−15) − 1.8.
Writes temp PDBs, then loads with biotite, then applies rotation/translation to all atoms in mobile structure. Returns `(tm_score, n_aligned_residues)`. Returns `(0.0, 0)` if biotite unavailable.

---

## Sidechain Optimization

### `ROTAMER_LIBRARY` (lines 1405–1468)
Pre-computed Dunbrack rotamer library, chi angles in degrees. Example: ARG has 12 rotamers.

### `SP3_CARBONS` (lines 1472–1491)
Per-residue SP3 carbons eligible for bond-angle flexibility.

### Class `SidechainDihedralOptimizer` (line 1494)

**Constructor params:** `ref_structure`, `mobile_structure`, `catalytic_residues`, `verbose=True`, `sp3_angle_flexibility=False`, `sp3_angle_tolerance=3.0`.

### `optimize_single_residue(ref_res, mob_res, convergence_threshold=0.001, max_passes=10)` (line 1817)

**Five-phase optimization:**

| Phase | Lines | What | Evaluations |
|---|---|---|---|
| 1 | 1857–1866 | Apply reference chi angles directly | 1 |
| 2 | 1868–1882 | Rotamer library (Dunbrack) — 5–12 rotamers | ~5–12 per residue |
| 3 | 1887–1904 | Per-chi full 360° search at 30° steps | ~12 per chi |
| 4 | 1906–1945 | Iterative local refinement: grids 10°→5°→2°→1° | many; convergence-bounded |
| 5 | 1950–1958 | SP3 bond-angle adjust ±3° if enabled | ~5 × #sp3 |

Returns `(initial_rmsd, final_rmsd, total_evaluations, final_chi_angles)`.

### `optimize_all_residues(n_cycles=3, fine_grid_degrees=5.0)` (line 1969)

Multi-cycle global pass. Early-stops if cycle improvement < 0.01 Å (line 2047–2050). Returns `(initial_total_rmsd, final_total_rmsd, total_evaluations)`.

---

## Multi-Strategy Orchestration

### `select_winner(df, tiebreaker_threshold=0.1, verbose=True)` (line 2080)

**Hierarchical priority:**
1. Primary: `catres_subset_all_atom_rmsd_after_opt` (lowest)
2. Tiebreaker 1: `catres_subset_ca_rmsd`
3. Tiebreaker 2: `all_backbone_rmsd`

Sort by primary, find all within `tiebreaker_threshold` of best, apply tiebreakers.

### Class `MultiStrategyAligner` (line 2163)

**Constructor (lines 2166–2256):**
- File paths: `ref_pdb_path`, `mobile_pdb_path`, `output_dir`
- Optimization: `catres_subset`, `ptm_specs`, `outlier_rmsd_threshold=0.5`, `convergence_threshold=0.001`, `max_iterations=50`, `min_residues=10`
- Sidechain: `sidechain_cycles=3`, `sidechain_fine_grid=5.0`, `enable_sidechain_opt=True`, `sp3_angle_flexibility=False`, `sp3_angle_tolerance=3.0`
- Output: `keep_all_outputs=False`, `save_csv=False`, `winner_threshold=0.1`, `verbose=True`

**Side effects on construction (lines 2214–2255):** creates output dir, parses REMARK 666, processes prediction PDB, extracts reference HETATM, builds HIS map.

### `run_all_strategies()` (line 2264)

**Strategies (lines 2267–2280):**

| Strategy | Atoms | Residues | Notes |
|---|---|---|---|
| `all_backbone_rmsd` | backbone | all | global backbone Kabsch |
| `ca_rmsd` | CA | all | global CA Kabsch |
| `catres_backbone_rmsd` | backbone | catres | catres-only backbone |
| `catres_ca_rmsd` | CA | catres | catres-only CA |
| `catres_subset_backbone_rmsd` | backbone | subset | optimized-subset backbone |
| `catres_subset_ca_rmsd` | CA | subset | optimized-subset CA |
| `global_TMalign` | CA | all | biotite TM-align (if available) |

All non-TM-align use IterativeAligner.

### `run_single_strategy(strategy_name, atom_selector, residue_set)` (line 2425)

**Workflow:**
1. Determine residues (lines 2434–2450)
2. Alignment (lines 2464–2502): TM-align or IterativeAligner
3. Metrics before opt (lines 2506–2533): backbone + CA RMSD over all matched / catres / subset
4. Sidechain optimization (lines 2536–2560): `optimize_all_residues()`
5. Resolve symmetric atoms once (lines 2562–2590) and recalculate all-atom RMSD → PRIMARY winner metric
6. lDDT (lines 2592–2607) using resolved swap_map
7. Returns `AlignmentMetrics`

### `save_aligned_structure(structure, output_path, strategy_name)` (line 2628)

Writes the aligned PDB:
1. Save aligned structure to temp file
2. Separate protein and HETATM (line 2645)
3. PyRosetta H-add (optional, lines 2648–2678)
4. Fix HIS tautomers (lines 2655–2669)
5. Copy ALL REMARK lines from reference (line 2682)
6. Copy ALL HETATM from reference (line 2685) — ligands/cofactors/water aligned with sidechain-optimized protein
7. Renumber atoms (line 2688)

---

## Main Entry Point — `main()` (line 2776)

**Required CLI args:** `--ref_pdb`, `--pdb_for_alignment`, `--output_dir`.

**Optional CLI args:** `--catres_subset` (e.g., "1,5,11"), `--ptm_from_remark666` (e.g., "A/LYS/6:KCX", nargs='+'), `--outlier_threshold` (0.5), `--convergence_threshold` (0.001), `--max_iterations` (50), `--min_residues` (10), `--no_sidechain_opt`, `--sidechain_cycles` (3), `--sidechain_fine_grid` (5.0), `--sp3_angle_flexibility`, `--sp3_angle_tolerance` (3.0), `--winner_threshold` (0.1), `--keep_all`, `--save_csv`, `--verbose`.

**Workflow (lines 2841–2912):** parse CLI → init PyRosetta → create `MultiStrategyAligner` → `run_all_strategies()` → report elapsed time.

---

## Special-Attention Areas

### The 5 sidechain-search phases (refactor target for Phase 2)

The Phase 2 work — ligand-aware torsion search — should hook into `optimize_single_residue()` (line 1817). Currently:
- Phases 1, 2, 3 try chi angles and apply them
- Phase 4 does iterative refinement with progressively finer grids
- The cost function is RMSD-to-reference ONLY — the ligand is NOT present yet

**Phase 2 refactor strategy:**
- Move HETATM paste-in earlier so ligand is present during sidechain opt
- Inject a clash term into the cost function (clash_penalty against ligand + other catres)
- Replace Phases 2–4 exhaustive grids with FASPR seed + basin-hopping + local fine grid

### PTM handling (relevant for Phase 3 relax tool)

The PTM transformation in `process_prediction_pdb()`:
- Strips NCAA-specific atoms (e.g., KCX → cut CX/OQ1/OQ2)
- Renames to canonical (KCX → LYS)
- Output has canonical LYS atom layout (3 NZ Hs after Rosetta H-add)

The OUTPUT does NOT have the KCX-specific atoms. They remain in the YYE substrate HETATM (where the CO2 group and Zn coordination live). For the energy-relax step, the protein-side KCX must be transformed back: strip 2 of 3 NZ Hs → leave 1 H. This is a NEW step the relax tool will perform.

### HETATM stripping + paste-back

Input prediction → strip ALL HETATM (line 407–501). After alignment + sidechain opt, paste back HETATM from REFERENCE (line 2685). So the final aligned PDB has ALL ligands/cofactors/Zn/water from the reference structure, in coordinates that have been aligned with the protein.

### Symmetric atom resolution

Critical for fair RMSD comparison. Applied ONCE after alignment + sidechain opt (line 2568). All metrics use this resolved swap_map.

---

**END OF ATLAS**
