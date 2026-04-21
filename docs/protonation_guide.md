# Protonation Pipeline Guide

Complete guide to the `protonate_active_site.py` script for preparing protein active-site clusters for QM/MLFF calculations.

---

## Overview

The pipeline assigns protonation states to a protein active-site cluster (PDB format) using a multi-tool strategy that handles metals, non-standard residues, fragment termini, and cross-residue covalent bonds.

```bash
python scripts/protonate_active_site.py input.pdb \
    -o protonated.pdb --ligand-charge 0 --pH 7.0 --relax-h
```

---

## Pipeline Steps (in order)

### Step 1: Backbone Rebuilding (if needed)

Some theozyme residues (from RFdiffusion) only have sidechain + CA (no N, C, O backbone atoms). The script detects partial residues and rebuilds missing backbone using standard L-amino acid geometry from CA/CB.

**When it triggers:** Any residue missing N, C, or O backbone atoms (but has CA).

**Geometry used:**
- CA-N bond: 1.47 A
- CA-C bond: 1.52 A
- C=O bond: 1.23 A
- Tetrahedral angles from CB direction (proper L-chirality)

Rebuilt atoms are marked as FREE during the subsequent xTB relaxation (not frozen with the other heavy atoms).

### Step 2: Reduce -- Metal Coordination Detection

`reduce` (Richardson lab tool) is run NOT to add hydrogens, but to detect which residues coordinate metals. It identifies "NoAdj-H" clashes -- atoms where hydrogen placement would clash with a nearby metal.

```
USER  MOD NoAdj-H: A   1 HIS HE2 : A   1 HIS NE2 : Z   9 YYE ZN1 :(H bumps)
```

This tells us HIS A:1 NE2 coordinates ZN1, so that nitrogen must NOT be protonated.

**Why reduce and not pdbfixer for this?** reduce has clash-aware logic that explicitly reports metal coordination. pdbfixer simply adds all H according to templates without metal awareness.

**Output:** A set of residues with metal-coordination annotations, used to override propka decisions later.

### Step 3: PDBFixer -- Full Hydrogen Addition

PDBFixer (OpenMM) adds ALL missing hydrogens with proper PDB naming (HA, HB2, HB3, HD1, etc.). It uses full amino acid templates.

**Important:** Only ATOM records are protonated. HETATM (ligand) lines are preserved untouched from the original input.

```python
from pdbfixer import PDBFixer
fixer = PDBFixer(filename=str(input_pdb))
fixer.addMissingHydrogens(pH)
```

### Step 4: Post-processing

Fixes common issues with pdbfixer's hydrogen placement:

#### 4a. Fragment N-terminal handling (NH2)

Active-site fragments have non-standard termini. The pipeline:
- Detects fragment termini (gaps in residue numbering)
- Removes the 3rd H on N-terminal residues: NH3+ (charged) -> NH2 (neutral)
- Adds a 2nd H if only one was placed: NH (peptide) -> NH2 (neutral)

This produces neutral termini appropriate for fragment QM calculations.

#### 4b. C-terminal aldehyde H

For C-terminal residues in fragments, the bare C=O gets an aldehyde hydrogen added:
- C=O -> C(=O)H
- Placed at sp2 geometry (120 degrees from CA and O)
- Bond length 1.10 A

#### 4c. Cross-residue covalent bonds (carbamylated LYS)

Detects ATOM nitrogen atoms within 1.65 A of HETATM carbon atoms (covalent bond to ligand). When found:
- Removes excess H on the bonded nitrogen (e.g., NZ of LYS: NH3+ -> NH)
- Records bond for charge correction

Example: Carbamylated lysine (KCX) has NZ bonded to ligand carbon. This NZ should have 1 H (NH), not 3 H (NH3+).

### Step 5: PROPKA -- pKa Prediction

Predicts pKa of all ionizable residues at the specified pH. Residues with predicted pKa above the target pH are protonated (neutral), below are deprotonated (charged).

**Overrides applied:**
- Metal-coordinating residues (from Step 2) are forced neutral regardless of propka prediction
- Residues with cross-residue covalent bonds get charge corrected
- Pre-existing H on carboxylates (ASP/GLU) overrides propka (see below)

### Step 6: HIS Clash Detection for Double Protonation

Histidine can be:
- HID: ND1 protonated (neutral, 0)
- HIE: NE2 protonated (neutral, 0)
- HIP: Both protonated (charged, +1)

The pipeline uses reduce's flip optimization to determine the best tautomer:
- If reduce reports NoAdj-H on NE2 (metal coordination): force HID (only ND1 protonated)
- If reduce reports NoAdj-H on ND1 (metal coordination): force HIE (only NE2 protonated)
- If propka predicts pKa > pH for both: assign HIP (+1 charge)
- If hydrogen bond analysis shows both N-H are satisfied: assign HIP

### Step 7: xTB Hydrogen Relaxation (optional, `--relax-h`)

Optimizes hydrogen positions using GFN2-xTB while freezing all heavy atoms. Fixes sp2/sp3 geometry issues from template-based placement.

```bash
# Runs in ~seconds for 100-atom systems
xtb input.xyz --opt tight --input fix.inp --gfn 2 --chrg CHARGE
```

**What gets relaxed:**
- All hydrogen atoms (free)
- Rebuilt backbone atoms from Step 1 (free)
- All other heavy atoms (frozen)

### Step 8: PDB Validation

Final checks:
- Serial number continuity
- No severe steric clashes (H-H < 0.7 A warns)
- Hydrogen coverage (every heavy atom that should have H does)
- Backbone completeness

---

## How Pre-existing H on Carboxylates Affects Charge

If the input PDB already has hydrogen atoms on a carboxylate (ASP OD1/OD2, GLU OE1/OE2), this overrides propka:

- **No H on carboxylate** -> deprotonated -> charge = -1 (default at pH 7)
- **H present on one O** -> protonated -> charge = 0 (user intended neutral)

This allows manual control: if you know ASP should be neutral (e.g., in a hydrogen bond network), just add the H in the input PDB.

---

## Net Charge Calculation

The total system charge is computed as:

```
Net charge = sum(protein residue charges) + ligand_charge
```

Where each protein residue contributes:
- ARG: +1, LYS: +1 (unless cross-residue bond -> 0)
- ASP: -1, GLU: -1 (unless protonated -> 0)
- HIP: +1, HID/HIE: 0
- KCX (carbamylated lysine): 0 (NH +1 and COO -1 cancel)
- All others: 0
- Fragment termini: 0 (neutral NH2 and C(=O)H)

```bash
# Ligand charge must be specified by the user
python scripts/protonate_active_site.py input.pdb \
    --ligand-charge 3  # e.g., Zn2+ + substrate(-1) = net +1... user specifies
```

The calculated charge is written to the output filename: `..._netCHG_plus3.pdb` or `..._netCHG_minus2.pdb`.

---

## The `--strip-h` Flag

Removes ALL existing hydrogen atoms before re-protonating from scratch:

```bash
python scripts/protonate_active_site.py input.pdb -o out.pdb --strip-h
```

**When to use:**
- Input has incorrect/inconsistent protonation from another tool
- Input has partial H (some residues protonated, others not)
- You want a completely fresh protonation assignment

**When NOT to use:**
- Input has carefully placed H on carboxylates (manual override, see above)
- Input has H from a previous QM optimization you want to preserve

---

## Common Edge Cases

### 1. Zinc coordination (PTE active sites)

Typical PTE theozymes have 2 Zn ions coordinated by HIS, ASP, and carbamylated LYS (KCX).

**What happens:**
- reduce detects all HIS-Zn coordination (NoAdj-H)
- Metal-coordinating HIS forced to HID or HIE (never HIP)
- KCX handled as net-neutral (NH+ and COO- cancel)
- Bridging hydroxide/water between Zn ions: charge depends on pH

### 2. Disulfide bonds (CYS-CYS)

If two CYS SG atoms are within 2.5 A, they form a disulfide:
- No H added to either SG
- Both CYS are neutral (charge = 0)

### 3. Residues with unusual connectivity

Proline (PRO) is handled correctly by pdbfixer (ring nitrogen, no HN).
Non-standard residues (KCX, selenomethionine, etc.) are handled via HETATM preservation or template matching.

### 4. Water molecules

- HOH/WAT in ATOM records: H added by pdbfixer
- HOH/WAT in HETATM records: preserved untouched

### 5. Multiple chains

The pipeline handles multi-chain inputs correctly:
- Each chain is analyzed independently for termini
- Cross-chain contacts (metal coordination) are detected
- Charge is summed across all chains

### 6. Missing residues in the fragment

Theozyme clusters often have non-consecutive residue numbers (e.g., 55, 56, 131, 132, 169). The pipeline:
- Detects gaps in numbering as fragment boundaries
- Assigns neutral termini at each boundary
- Does NOT attempt to fill missing residues

### 7. Hydrogen already on input

If the input already has some H atoms:
- Use `--strip-h` for a fresh start (recommended)
- Without `--strip-h`, existing H are preserved and pdbfixer only adds missing ones
- This can lead to inconsistencies -- prefer `--strip-h` unless you have a specific reason

---

## CLI Reference

```
python scripts/protonate_active_site.py INPUT.pdb [OPTIONS]

Required:
  INPUT.pdb                     Input PDB (active site cluster)

Output:
  -o, --output FILE             Output PDB path (default: input_protonated.pdb)

Protonation:
  --pH FLOAT                    Target pH for propka (default: 7.0)
  --ligand-charge INT           Formal charge of the ligand/HETATM atoms
  --strip-h                     Remove all existing H before re-protonating

Relaxation:
  --relax-h                     Relax H positions with xTB (GFN2-xTB)
  --no-relax-h                  Skip xTB relaxation

Overrides:
  --force-hip CHAIN:RESNUM      Force HIS to be doubly protonated (HIP, +1)
  --force-neutral CHAIN:RESNUM  Force a residue to be neutral (override propka)
  --force-charged CHAIN:RESNUM  Force a residue to be charged (override propka)

Output:
  --write-charge-json           Write charge breakdown to JSON file
  --verbose                     Extra logging
```

---

## Example: Full Protonation of PTE Theozyme

```bash
# Input: designed PTE active site from RFdiffusion, 890 atoms, ligand YYE
# Ligand has 2 Zn2+ (+4), substrate (-2), bridging OH- (-1) = net +1... 
# but user may package differently. Know your ligand charge!

apptainer exec --nv --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python scripts/protonate_active_site.py \
        data/examples/zapp_p1d1_bestHIT/input_no_h.pdb \
        -o data/examples/zapp_p1d1_bestHIT/protonated.pdb \
        --ligand-charge 0 \
        --pH 7.0 \
        --relax-h \
        --strip-h \
        --write-charge-json

# Output:
#   protonated.pdb                    (full PDB with H)
#   protonated_charges.json           (charge breakdown per residue)
#   Filename encoded: ..._netCHG_0.pdb
```

**Expected log output:**
```
12:00:01 [INFO] Loading input.pdb (890 atoms)
12:00:01 [INFO] Backbone check: all residues complete
12:00:02 [INFO] reduce: detected 4 metal-coordinating residues
12:00:02 [INFO]   HIS A:55 NE2 -> ZN (NoAdj-H)
12:00:02 [INFO]   HIS A:57 NE2 -> ZN (NoAdj-H)
12:00:02 [INFO]   HIS A:131 ND1 -> ZN (NoAdj-H)
12:00:02 [INFO]   HIS A:169 NE2 -> ZN (NoAdj-H)
12:00:03 [INFO] pdbfixer: 450 -> 890 ATOM lines (440 H added)
12:00:03 [INFO] Fragment termini: 6 N-term, 6 C-term
12:00:03 [INFO]   Terminal fix: removed H3 on A:55 (NH3+ -> NH2)
12:00:03 [INFO]   Terminal fix: added aldehyde H (HXT) on C-terminus
12:00:03 [INFO] Cross-residue bond: LYS A:169 NZ -> YYE Z:1 C8 (1.45 A)
12:00:04 [INFO] Net charge: protein=+1, ligand=0, total=+1
12:00:05 [INFO] xTB H relaxation: 890 atoms, 450 frozen
12:00:08 [INFO] xTB H relaxation done (sp2/sp3 geometry fixed)
12:00:08 [INFO] Written: protonated_netCHG_plus1.pdb
```
