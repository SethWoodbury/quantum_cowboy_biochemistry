# M-CSA Theozyme Pipeline

Cross-links: [`00_index.md`](00_index.md) · [`enzyme_ts_design.md`](enzyme_ts_design.md)

## Goal

Given:

- An M-CSA entry ID
- Concrete substrate SMILES (binds R-groups in M-CSA mechanism diagrams)

produce an automated theozyme suitable for the AME structure-generation
benchmark (https://pmc.ncbi.nlm.nih.gov/articles/PMC12791007/). The AME
benchmark needs theozyme reference data; this pipeline supplies it.

## Why a separate pipeline

M-CSA entries carry far more annotation than a generic SMILES + PDB
input. Throwing that annotation away (i.e. routing through
[`enzyme_ts_design.md`](enzyme_ts_design.md) directly) wastes information
that should drive Stage 4 / Stage 7 choices. M-CSA gives:

- Catalytic residues, **per residue role** (general acid, general base,
  nucleophile, electrostatic stabiliser, metal ligand, ...)
- Mechanism steps with arrow-pushing in Marvin XML
- Cofactors and PTMs (e.g. carbamylated Lys = KCX in PTE entry 159)
- Reference PDB(s) with the catalytic site
- ChEBI IDs for substrate / product / intermediates

Use all of it.

## R-group resolution

M-CSA mechanism diagrams typically contain symbolic R-groups
(e.g. `R-O-P=O` for a phosphoester; `R'-COO-R''` for an ester).
Resolution algorithm:

1. Parse Marvin XML mechanism file → atom-mapped reaction skeleton with
   R-labels.
2. User supplies concrete substrate SMILES.
3. Walk both graphs, matching the M-CSA core moiety to a substructure of
   the concrete SMILES (RDKit substructure match with stereo-aware
   mapping).
4. Bind each R-label to the corresponding fragment of the concrete
   SMILES.
5. Re-apply the Marvin arrow-pushing on the now-concrete atom-mapped
   reaction to derive bond-deltas for Stage 4 and driving coords for
   Stage 7.

Optional `--r-group-dict` JSON `{"R": "OCH3", "R'": "C2H5"}` for
ambiguity. If multiple substructure matches exist, the pipeline emits all
and chains them as alternative TS searches.

Algorithm code: `quantum_engine.prep.mcsa_rgroup`.

## SMILES from M-CSA

`reaction.compounds[]` in M-CSA contains ChEBI IDs. The SMILES field is
`null` in practice, so:

- Fetch SMILES via ChEBI REST API
  - https://www.ebi.ac.uk/chebi/webServices.do
  - https://www.ebi.ac.uk/chebi/searchId.do?chebiId=<id>
- Cache to `/net/databases/chebi_cache/<chebi_id>.json`
- Local fallback cache: `~/.cache/qcb/chebi/`

Resolver lives at `quantum_engine.prep.chebi`. Cache TTL is unbounded
(ChEBI IDs are stable).

## PTM handling

M-CSA flags PTMs explicitly. Build a PTM library `quantum_engine.prep.ptms`
seeded with at least:

| PTM | Three-letter | Notes |
|---|---|---|
| Carbamylated Lys | KCX | PTE entry 159; bridges Zn pair |
| Selenomethionine | MSE | Treat as MET unless heavy-atom QM |
| S-hydroxycysteine | CSO | Active in some peroxidases |
| Phospho-Ser/Thr/Tyr | SEP/TPO/PTR | Standard phosphorylation |

Each PTM ships:

- Topology (atom names, connectivity)
- AMBER parameter mapping for OpenMM (Stage 3 protonation step)
- Default protonation state by role annotation:
  - "general acid" → protonated
  - "general base" → deprotonated
- PROPKA fall-back when the role tag is missing

## Mechanism step iteration

For multi-step mechanisms (PTE has 7 steps), the pipeline runs each
step's TS search separately AND chains them: intermediate of step N =
reactant of step N+1.

```
M-CSA step 1: R + Zn-OH-Zn → tetrahedral intermediate
M-CSA step 2: tetrahedral intermediate → P + Zn-O-Zn
                ^^^                      ^^^
                reactant of step 2 = product of step 1
```

Implementation: orchestrator emits one Stage 4–8 sub-pipeline per step,
with intermediate `.cif`s feeding the next step's Stage 4.

## Stage flow

### Stage 0 — M-CSA fetch + cache

- Source: M-CSA REST API
- Cache: `/net/databases/mcsa_cache/<entry_id>/`
  - `entry.json`
  - `mechanism.xml` (Marvin)
  - `reference.pdb`
  - `compounds/<chebi_id>.json`
- Local fallback: `~/.cache/qcb/mcsa/`

### Stage 1 — Resolve substrate

- Inputs: user SMILES + ChEBI lookups (Stage 0 cache)
- Apply R-group resolution above
- Output: per-step concrete reactant + product SMILES, atom-mapped

### Stage 2 — Active-site cropping

- From M-CSA reference PDB
- Take catalytic residues + cofactors + waters within shell
- Handle PTMs (apply topology overrides from PTM library)
- Output: `cropped_site.pdb`, `system_charge.json`

### Stage 3 — Two-tier residue expansion

- Tier 1: M-CSA catalytic residues only
- Tier 2-distance: + residues within R Å of tier 1 (default R = 6)
- Tier 2-motif: HExxH, Ser-His-Asp triad, Cys-His dyad, Asp-Asp diad
- Identical motif library to [`enzyme_ts_design.md`](enzyme_ts_design.md)

### Stage 4 — Per mechanism step, vacuum TS guess

- Preferred: SCINE Chemoton/ReaDuct when arrow-pushing parseable from
  Marvin XML (M-CSA always has it, so this is the default path)
- Fallback: autodE
- One TS per mechanism step

### Stage 5 — Dock TS into active-site cluster

- Same constraint-based placement as the generic pipeline
- M-CSA residue role annotation feeds the dock scorer
  (e.g. nucleophile must be within 3.5 Å of electrophile)

### Stage 6 — Iterative refinement

- MACE-POLAR-1M default; g-xTB pre-screen
- CA-only fixed; Stage 6 contract identical to generic pipeline

### Stage 7 — In-protein path re-find

- pyGSM single-ended with driving coords from **Marvin XML
  arrow-pushing** (not just bond-delta heuristics)
- Driving coord weighting: bond-form/break > bond-rotate > bond-stretch

### Stage 8 — High-res TS polish

- Sella + MACE-POLAR-1M
- Frequency check (one imaginary mode)
- IRC verification: forward → reactant of this step; reverse → product
  of previous step's IRC end-point (chain consistency)

### Stage 9 — Theozyme output

- Minimal residue set: M-CSA catalytic residues + any tier-2 residue
  whose RMSD vs. starting active-site exceeds 0.5 Å during refinement
  (= "actively participating")
- Substrate at TS geometry
- AME-benchmark-compatible JSON
- `.cif` of the theozyme-only complex

## Inputs / outputs

### Inputs

```bash
qcb-mcsa-theozyme run \
    --mcsa-id 159 \
    --substrate "CC(=O)OCC1=CC=CC=C1[N+](=O)[O-]" \
    --r-group-dict r_groups.json   # optional
```

### Outputs

- `out/step_<n>/ts.cif` per mechanism step
- `out/theozyme.json` AME-format
- `out/theozyme.cif` minimal residue + substrate at TS
- `out/barriers.json` per step
- `out/judgment.json` plausibility flags

## AME benchmark integration

End of pipeline emits AME-format theozyme JSON. Schema TBD when AME
schema is published; until then, follow paper figures:
https://pmc.ncbi.nlm.nih.gov/articles/PMC12791007/

Schema sketch (placeholder):

```json
{
  "mcsa_entry": 159,
  "substrate_smiles": "...",
  "step": 1,
  "residues": [{"resid": "KCX:169", "atoms": [...], "coords": [...]}],
  "substrate_at_ts": {"atoms": [...], "coords": [...]},
  "barrier_kcal": 17.2,
  "imag_freq_cm": -442
}
```

## Test cases (priority order)

| M-CSA | Enzyme | Why |
|---|---|---|
| 159 | PTE (phosphotriesterase) | **Primary** — lab gold standard, PTM (KCX), Zn₂ |
| 641 | Anthrax LF | HExxH motif test |
| 922 | AChE | Ser-His-Glu triad test |
| 376 | ADA (adenosine deaminase) | Simple mononuclear Zn |
| 900 | PNB esterase | Ester hydrolysis baseline |

## Validation philosophy

M-CSA has no TS coordinates — use as a **realistic use-case test**, not
a numerical benchmark. Plausibility judged on:

- TS geometry passes frequency check (exactly one imaginary mode)
- Barrier in physical range (≤ 50 kcal/mol; flag > 35)
- Residue movement realistic (no CA flying away; sidechain RMSD < 2 Å)
- Metal coordination preserved across the TS
- Qualitative match to mechanism described in M-CSA prose
  (nucleophile attacks correct electrophile; right bond breaks)

The pipeline emits these as `judgment.json` flags; user reviews. Hard
numerical comparison-to-reference is not appropriate here because there
is no reference geometry.

## Open items

- Marvin XML parser (`quantum_engine.prep.marvin`) is scaffolded but not
  written. Needed for Stage 4 default and Stage 7 driving coords.
- AME JSON schema awaiting upstream publication.
- ChEBI cache directory needs creation on `/net/databases/`.
- PTM library has KCX wired up (PTE-driven); other entries TODO.
