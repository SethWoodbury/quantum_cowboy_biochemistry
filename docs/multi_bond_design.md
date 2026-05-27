# Multi-bond TS finding: design notes for v3

**Status: design proposal.** Not implemented yet. Captures the path from the
current 1D-CV scan_along_s (single bond forming + single bond breaking) to a
generalized multi-bond pipeline that handles Diels-Alder, Cope rearrangement,
sigmatropic shifts, electrocyclic, ene reactions, and any K+L bond-changing
topology.

This doc is the synthesis of an external code review (2026-05-06) by a
research subagent surveying the comp-chem TS-finding methods catalog.

---

## Why the 1D-scan paradigm has a hard ceiling

`tools/scan_along_s.py` works by parameterizing the reaction with a single
collective variable

```
s = d(P-OL) − d(P-ON)        (pin: d(P-OL) + d(P-ON) = sum_target)
```

It works because **SN2-at-P is a perfectly asymmetric stretch**: one bond
shortens by exactly the amount the other lengthens, both at the same atom.
That's a special case of a general reaction parameterized by

```
{forming bonds B_form_1, ..., B_form_K} + {breaking bonds B_break_1, ..., B_break_L}
```

When K = L = 1 the 1D CV captures the saddle. When K + L ≥ 3, no single CV
faithfully projects the K+L-dimensional bond manifold and the scan finds a
projected maximum that **isn't the actual saddle**, producing barriers off
by 5-15 kcal/mol with no way to tell from the scan profile alone.

**Failure modes confirmed in the literature:**

| Reaction | K | L | 1D scan verdict |
|---|---|---|---|
| SN2-at-P / -C / -S (PTE, kinases) | 1 | 1 | works |
| Hydride / proton transfer | 1 | 1 | works (tighten s window) |
| Diels-Alder | 2 | 0 | partial (synchronous saddle ~ok, asynchronous skewed) |
| 1,3-dipolar cycloaddition | 2 | 0 | partial |
| Cope rearrangement | 3 | 3 | fails — saddle in 6D bond manifold |
| Sigmatropic [1,5]-H shift | 1 | 1 + dihedral | fails (CV missing dihedral) |
| Electrocyclic ring closure | 1 | 0 + dihedrals | fails (no σ stretch CV exists) |
| Stepwise mechanisms | varies | varies | profile becomes monotonic, false TS |
| Ene reactions (H + C-C concerted) | 1 | 1 + 1 | partial — H much faster than C |

Plus structurally unfixable cases inside the 1D paradigm: bifurcating PESs,
post-TS branching, cis-trans isomerizations (no σ bond changes, only
dihedral).

---

## Methods catalog (what comp chemists actually use)

### Chain-of-states (need R + P)

- **NEB / NEB-CI (climbing image, Henkelman 2000)** — the workhorse.
  Embarrassingly parallel across images, no CV choice needed. ASE built-in
  (`ase.mep.neb.NEB`). Cost: ~10-20 images × ~50 force calls. Failure:
  poor initial path (linear interpolation of cyclopentadiene + maleic
  anhydride collapses through atomic overlaps). Fix: image-dependent
  constraints or geodesic interpolation.
- **GSM / FSM (Zimmerman; Voorhis)** — grows path adaptively, focuses on
  saddle region. `pyGSM` on Github. Best for organic gas phase, less
  validated in active sites.
- **String method / zero-temperature string** — discretized constrained
  path. Mathematically clean; rarely better than NEB in practice.

### Local saddle (need TS guess only)

- **Sella (Hermes 2022)** — saddle optimizer using internal coordinates +
  quasi-Newton + eigenvector following. Pip-installable, ASE-native.
  **Always run this AFTER NEB-CI or AFIR** — never standalone unless guess
  is excellent.
- **Dimer (Henkelman/Jónsson)** — local saddle from 1 point + initial
  direction. Cheap (~2 force evals/step). ASE built-in. Surface-science
  workhorse, less robust for soft modes in solution.
- **P-RFO** — Gaussian-style eigenvector follower; subsumed by Sella.

### Driving / scan (need R only)

- **Multi-D relaxed scan** — generalization of scan_along_s. Practical to
  2D (~121 points × ~5-15 min/point on MACE). Past 2D intractable.
  Naturally catches asynchronous mechanisms.
- **AFIR (Maeda)** — Artificial Force Induced Reaction. Adds fragment-pulling
  potential; max along driven path is TS guess. Needs only R + push-list.
  Finds *unknown* products. GRRM software.
- **Bond-order CV scan** — multi-D scan with Pauling bond order as CV.

### Sampling (need R, expensive)

- **Metadynamics / WT-MTD** — drop bias along CVs to escape reactant well.
  Expensive (~ns × MLFF cost) but explores spontaneously. PLUMED + ASE.
  Use when hand-picked CVs aren't trustworthy.
- **Reaction Path Hamiltonian / instanton** — quantum tunneling
  corrections; orthogonal.

### ML-based TS guess generators ("chemotron"-territory)

- **OA-ReactDiff** (Duan et al. 2023, *Nature Comp Sci*) — equivariant
  diffusion model trained on Transition1x (~10k QM TS structures).
  Input: aligned R + P 3D structures. Output: TS samples + MEP guess.
  ~70-85% DFT-validated success on Transition1x test, ~5% on
  out-of-distribution organometallic. Small organic only, no protein
  context. **Use as NEB seed, not final TS.**
- **TSDiff** — diffusion variant, similar performance.
- **TS-EGNN / TS-GEN** — direct-prediction equivariant nets. Faster, less
  accurate.
- **Chemoton (Reiher, ETH)** — *not* a TS predictor; automated reaction
  network exploration framework calling AFIR + Newton-trajectory. SCINE
  backend, not ML-FF native.
- **ChemTraYzer** — reactive-MD reaction discovery.
- **NeuralPlexer / RoseTTAFold-AA / Boltz** — protein-ligand structure
  predictors, NOT TS. Give Michaelis complex; still need TS method on top.

---

## Generic CV definition

Three CV families for K forming + L breaking bonds:

### (a) Average distance progress (default)

```
λ ∈ [0, 1]
d_form_i(λ) = (1 − λ) * d_form_i^R   + λ * d_form_i^TS_guess
d_break_j(λ) = (1 − λ) * d_break_j^R  + λ * d_break_j^P
```

Pin all bonds simultaneously via FixBondLengths, sweep λ.
**Best when bonds are chemically similar** (Diels-Alder: two ~equivalent C-C).
Reduces to scan_along_s when K = L = 1.

### (b) Pauling bond order (most physical)

```
BO_i = exp(−(d_i − d_i^eq) / 0.3 Å)
CV = sum(BO_form) − sum(BO_break)
```

Saturates near 1 for full bonds, decays smoothly to 0. Bounded,
dimensionless. **Best when bond types differ** (C-H breaking + C-C forming,
ene reaction). Used by metadynamics-on-reactions papers (Ensing/Laio).
d_eq from covalent-radii sum (RDKit `Chem.GetPeriodicTable`).

### (c) Per-bond explicit (when reaction is asynchronous)

Don't aggregate; treat each bond as independent CV. 2D scan when K + L = 2;
NEB when K + L > 2. Asynchronicity tracker:

```
α = (BO_form_1 - BO_form_2) / (BO_form_1 + BO_form_2)
```

α ≈ 0 → synchronous; α ≈ 0.3-0.6 → asynchronous; α > 0.7 → stepwise (abandon
scan, re-cast as two sequential SN2 saddles).

---

## v3 design proposal

Don't write `scan_generic.py` as a monolith. Split into composable scripts
matching the existing `qcb` CLI:

### `qcb scan-multi` (new)

```
qcb scan-multi --input R.pdb \
    --product P.pdb \                         # optional; enables auto bond detection
    --forming-bond C1,C6 --forming-bond C2,C5 \
    --breaking-bond C2,C3 \                   # repeatable
    --cv-mode {avg,bond-order} \
    --n-points 11 --lambda-min 0 --lambda-max 1 \
    --backbone-fix \                          # FixInternals on Cα-Cα as today
    --out scan_dir/ \
    --fallback-neb-on-monotonic               # auto-trigger NEB on monotonic profile
```

When `--product` is given, auto-detect forming/breaking bonds by diffing
connectivity (RDKit + covalent radii). Same backbone-pinning constraint
pattern as scan_along_s. Same FixBondLengths machinery, just with N pins
instead of 2.

### `qcb neb-multi` (extend existing `qcb neb`)

- `--seed-from oa-reactdiff` — interpolate via diffusion model when both
  endpoints provided. Sanity check: any pairwise distance < 0.7 Å → fall
  back to linear/geodesic.
- `--batched-images` — evaluate all images in single MACE forward pass
  (~5× speedup vs serial).

### `qcb auto-ts` (the policy layer — what users actually call)

```
qcb auto-ts --input R.pdb --product P.pdb [--max-bonds-changed 4] --out ts_dir/
```

Decision tree:

1. Diff connectivity. If K + L = 2 → call `qcb scan-multi` with avg CV.
2. If 3 ≤ K + L ≤ 6 → call `qcb neb-multi` with 12 images.
3. If scan profile monotonic → re-dispatch as NEB-CI.
4. Always finish with `qcb saddle` (Sella) on best guess.
5. Always validate with `qcb freq` (single imaginary mode along reaction
   coordinate).

This matches how the field actually works: **scan/NEB/AFIR are guess
generators; Sella + freq is the validator.** Don't conflate them.

### What NOT to build

- Don't reimplement NEB, Sella, Dimer, AFIR — ASE, sella, pyGSM cover those.
- Don't build a metadynamics flow yet — `qcb mtd` exists.
- Don't rewrite OA-ReactDiff — it's a 1500-line PyTorch repo; just shell
  out and parse outputs.

---

## Diels-Alder concrete walkthrough

**Cyclopentadiene (Cp) + maleic anhydride (MA)**, gas-phase reference:
ΔG‡ ≈ 16-18 kcal/mol, two new C-C bonds form synchronously, no σ bonds break.

Working CLI today (uses the generalized `scan_along_s.py` shipped 2026-05-06,
not the future `qcb auto-ts` policy layer):

```
python tools/scan_along_s.py \
    --input cp_ma_complex.pdb \
    --out cp_ma_scan/ \
    --model mace-off-m \
    --device cuda \
    --charge 0 \
    --drag-mode pin-cv-custom \
    --bond Cp_C1.CPD,MA_C1.MAL \
    --bond-role Cp_C1.CPD,MA_C1.MAL:forming \
    --bond Cp_C4.CPD,MA_C2.MAL \
    --bond-role Cp_C4.CPD,MA_C2.MAL:forming \
    --cv-formula 'mean(d_Cp_C1_CPD_MA_C1_MAL, d_Cp_C4_CPD_MA_C2_MAL)' \
    --s-grid='3.7,3.5,3.0,2.7,2.5,2.3,2.2,2.1,2.0,1.8,1.55' \
    --fmax 0.05 --max-steps 200
```

The CV value `mean(d_C1, d_C2)` decreases from ~3.7 Å (vdW contact) through
~2.25 Å (synchronous TS) to ~1.55 Å (cyclohexene product). Pass the s-grid
explicitly so it concentrates points near the saddle. After the scan, polish
the energy-max frame with `qcb saddle --backend dimer` (or sella) and run
freq to confirm exactly one imaginary mode.

Future CLI invocation (when the `qcb auto-ts` policy layer ships):

```
qcb auto-ts --input cp_ma_complex.pdb --product cp_ma_endo.pdb \
    --forming-bond Cp_C1,MA_C1 --forming-bond Cp_C4,MA_C2 \
    --cv-mode avg --n-points 11 \
    --lambda-min 0 --lambda-max 1 \
    --out cp_ma_ts/
```

**Expected geometry:**
- Reactant: d(Cp_C1, MA_C1) ≈ d(Cp_C4, MA_C2) ≈ 3.5-4.0 Å (vdW contact)
- TS: both ≈ 2.20-2.30 Å (synchronous)
- Product: both ≈ 1.55 Å
- CV `s = -(d_form_1 + d_form_2)/2` ranges -3.7 → -1.55; TS at s ≈ -2.25.

**Pitfalls:**

1. **Synchronous is an idealization.** Even textbook Diels-Alder forms the
   two C-C ~0.05 Å apart at the TS. Pinning both equal drives a saddle
   slightly above the true asynchronous one. Mitigation: post-scan Sella
   without constraints relaxes to true saddle.
2. **Endo vs exo**: two TSs ~1-2 kcal/mol apart. Geometry decides which
   you find. Run both endo and exo product PDBs in parallel.
3. **Linear interpolation collapses**: LERP gives intermediate frames
   with ring strain artifacts. NEB-CI fixes via image-dependent
   constraints; raw linear is a footgun.
4. **MACE-MP-0 vs MACE-OFF23**: gas-phase neutral organics are MACE-OFF
   territory. Don't use polar/electrolyte model.
5. **Rotational drift of MA**: it can re-orient during scan. Add CoM
   distance restraint or accept weird s_min point and trust TS region.

---

## Recommendation summary

**Build:** `qcb auto-ts` (policy decision tree) + `qcb scan-multi`
(generalized scan with `--forming-bond/--breaking-bond` and CV modes).

**Reuse:** ASE NEB, sella, pyGSM, RDKit covalent-radii diff, OA-ReactDiff
shell-out.

**Fall back to NEB-CI:** when K + L > 2, scan profile monotonic, or
argmax sits at endpoint (existing scan_along_s warnings already detect
the latter two).

**Always validate:** Sella optimization + frequency calc with single
imaginary mode along reaction coordinate.

**`scan_along_s.py` is now the generalized 1-D-CV workhorse.** As of 2026-05-06
it accepts arbitrary `--bond` lists and a safe `--cv-formula` parser. It
remains the cheapest and most reliable choice when K + L ≤ 2 and the saddle
is well-described by a single CV; for K + L ≥ 3 use NEB-CI instead.

---

## References (cited briefly, not exhaustive)

- Henkelman, Uberuaga, Jónsson 2000 — NEB-CI (J. Chem. Phys.)
- Hermes 2022 — Sella saddle optimizer (J. Chem. Theory Comput.)
- Maeda et al. — AFIR / GRRM (Phys. Chem. Chem. Phys.)
- Duan, Du, Gomes-Tavares, Hsu, Gomez-Bombarelli 2023 — OA-ReactDiff
  (Nature Comp Sci)
- Zimmerman — pyGSM growing-string
- Reiher group — Chemoton / SCINE
- Ensing, Laio — bond-order CVs in metadynamics
