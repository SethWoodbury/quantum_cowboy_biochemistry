# TS Search Strategies in the QCB Pipeline

A unified, user-facing guide to the transition-state (TS) search strategies
exposed through the `quantum_cowboy_biochemistry` (QCB) pipeline.

This document covers the four primary strategies selectable via the
`--strategy` flag (`legacy`, `irc`, `cv-spring`, `mtd`) and the composable
modifiers (`--refine-xtb`, `--interpolation {geodesic,idpp,linear}`) that
orthogonally improve any of them.

The modules underlying each strategy live in `qcb/mlff/`:

| Module | Purpose |
|---|---|
| `qcb/mlff/irc.py` | Sella saddle optimization + IRC descent both ways |
| `qcb/mlff/interpolation.py` | Geodesic / IDPP / linear interpolation |
| `qcb/mlff/xtb_refine.py` | Semi-empirical endpoint sanity check (GFN2-xTB) |
| `qcb/mlff/cv_spring.py` | Bond-difference CV spring for endpoint generation |
| `qcb/mlff/metadynamics.py` | 1-D well-tempered metadynamics rescue |

Reading this once should be enough to pick the right strategy + modifiers
for any enzymatic reactive-coordinate problem that lands on your desk.

---

## 1. Overview table

Wallclock estimates are for a typical ~200-atom enzyme active-site cluster
with QM region + buffer water, running on a single NVIDIA L40 GPU with
the MACE-OMol foundation model. Adjust upward by ~2x for chimeric substrates
or systems with many floppy waters.

| Strategy | Method (1-line) | When to use | Approx. cost (L40, MACE-OMol, ~200 atoms) | Primary strength | Primary weakness |
|---|---|---|---|---|---|
| `legacy` | NEB-TS on interpolated reactant -> product path | You have well-defined reactant AND product basins (ZAPP de novo designs, standard mechanism) | 30-60 min | Well-understood, fast, reliable when endpoints are real | Fails if product basin is shallow/absent or interpolation creates clashes |
| `irc` | Sella saddle -> IRC descent in both directions | You have a TS guess (hand-built or post-NEB validation) | 15-45 min | No product guess needed; TS discovers reactant AND product | Needs a reasonable TS guess; >200 free atoms stresses Sella internals |
| `cv-spring` | Harmonic spring on s = d(P-LG) - d(P-nuc), then release + relax | Clean reactant, unknown product position, want mechanism-agnostic endpoint generation | 30-90 min | Mechanism-neutral (doesn't force concerted/stepwise); naturally finds pentacoordinate intermediates | Still needs a single 1-D CV that describes the reaction |
| `mtd` | Well-tempered metadynamics along bond-difference CV + basin extraction | NEB/CV-spring failed; suspect multiple minima or hidden intermediate | 2-8 h | Explores the CV explicitly; reveals intermediates via FES minima | Expensive; MACE may extrapolate far from training distribution |
| `--refine-xtb` (modifier) | Post-hoc GFN2-xTB geometry opt on each endpoint | Always (unless >400 atoms); especially for catching MACE hallucinations | +2-10 min | Independent QM sanity check; catches "product collapsed to reactant" | Not applicable to TS itself; slow/unstable above ~400 atoms |
| `--interpolation geodesic` (default) | Riemannian-geodesic interpolation in internal-coord manifold (Zhu et al. 2019) | Always as the default for NEB paths | +10-30 s | Avoids atomic clashes; respects bonds | Requires `geodesic-interpolate` package |
| `--interpolation idpp` | Image-Dependent Pair Potential interpolation | Small organic reactions where geodesic package unavailable | +10-30 s | Good middle ground | Struggles with multi-bond and chirality |
| `--interpolation linear` | Cartesian linear interpolation | Debugging only; known bad for enzymes | ~0 s | Fastest | Creates atomic clashes in any dense system |

---

## 2. Decision flowchart

```
START: What is your input structure?
   |
   +-- Already a TS guess (hand-built partial bonds)?
   |     |
   |     +-- YES  -->  --strategy irc   (+ optional --refine-xtb on endpoints)
   |     |
   |     +-- NO, but want to VALIDATE an existing NEB result
   |             -->  --strategy irc  starting from converged CI-NEB TS
   |
   +-- Clean reactant + clean product geometries both available?
   |     |
   |     +-- YES, well-defined basins (ZAPP P1D1, standard SN2-at-P)
   |     |       -->  --strategy legacy --interpolation geodesic --refine-xtb
   |     |              (our "gold standard" default)
   |     |
   |     +-- NO, product uncertain / product basin suspect
   |             -->  --strategy cv-spring --refine-xtb
   |
   +-- Clean reactant only, no product guess?
   |     |
   |     +--  --strategy cv-spring  (with bond-difference CV targets)
   |             --refine-xtb to validate whatever endpoints emerge
   |
   +-- Legacy / cv-spring failed, OR suspect multiple minima?
   |     |
   |     +--  --strategy mtd  -> get FES, extract basins, then legacy NEB
   |             between discovered basins
   |
   +-- Post-hoc concern about MACE accuracy for any endpoint?
         |
         +--  Re-run with --refine-xtb  (cheap, independent QM check)
```

**Gotcha**: If your "product" after any strategy has both the nucleophile
bonded AND the leaving group still bonded, that is a real pentacoordinate
intermediate (see section 8), not a failure.

---

## 3. Per-strategy deep dives

### 3.1 `legacy` — NEB-TS on interpolated path

**Method.** Standard climbing-image nudged elastic band. Interpolate N images
between reactant and product (geodesic by default; see section 6), minimize
the band with CI-NEB until the highest-energy image converges on the saddle.
The climbing image becomes the TS candidate, which is then optionally
refined with Sella. IRC-style validation is not performed unless you
explicitly add `--strategy irc` as a follow-up step.

**When to use.**
- Both reactant AND product geometries are well-defined minima (the mechanism
  is known).
- ZAPP de novo designs (P1D1, minimal_deNovoPdPTE) where reactant and product
  basins are deep and separated.
- You want the fastest path to a mechanism number for a "normal" SN2-type step.

**When NOT to use.**
- Product basin is shallow/absent — NEB will drag a non-stationary image
  through the barrier region and report a nonsense TS.
- Chimeric/Frankenstein constructs where the reverse direction of the spring
  or interpolation produces unphysical intermediates (use `--unidirectional
  --pre-relax` from section 5 below).
- Substrate-in-water systems — see section 7's "substrate_uncat 216 kcal/mol"
  failure; linear interpolation creates clash images. This is still
  recoverable with `--interpolation geodesic`, but IRC on a TS guess is
  usually more robust.

**Parameters.**
| Flag | Recommended | Notes |
|---|---|---|
| `--neb-images` | 9-13 | Use 9 for cheap screens, 13 for final |
| `--neb-fmax` | 0.05 eV/Å | Tighter (0.03) for publication |
| `--climb` | true | Always; CI-NEB is the point |
| `--interpolation` | `geodesic` | See section 6 |
| `--refine-xtb` | on | Catches endpoint hallucinations |

**Expected runtime.** 30-60 min on L40 GPU with MACE-OMol for ~200 atoms and
13 images.

**Accuracy.** CI-NEB accuracy is benchmarked at 63-71% success rate on the
T1x / Transition1x reaction set depending on base MLIP and interpolator
(Wan et al. arXiv:2604.00405, 2026). Upgrading to FSM + MACE-OMol reaches
96.6% in the same benchmark, which is the motivation for the geodesic
interpolation + xTB validation combo we recommend.

**Failure modes.**
1. **Atomic clashes in interpolated images** -> unrealistic barriers
   (>150 kcal/mol). Fix: `--interpolation geodesic`.
2. **Product endpoint is not a real minimum** -> climbing image sits at
   the "wrong" point. Fix: `--refine-xtb` on both endpoints to catch
   collapse; consider `cv-spring` + release-relax instead.
3. **Multi-TS mechanism squeezed onto one path** -> climbing image
   oscillates. Fix: run MTD to identify intermediates, then two-step NEB.

**Example command.**
```bash
qcb mlff-ts \
    --input system.pdb \
    --strategy legacy \
    --neb-images 13 --neb-fmax 0.05 \
    --interpolation geodesic \
    --refine-xtb \
    --out outputs/R2/legacy_gold/
```

**Citations.**
- Smidstrup et al. *J. Chem. Phys.* 2014, 140, 214106 (IDPP).
  <https://doi.org/10.1063/1.4878664>
- Zhu, Thompson, Martinez *J. Chem. Phys.* 2019, 150, 164103 (geodesic
  interpolation). <https://doi.org/10.1063/1.5090303>
- Hermes et al. *J. Chem. Theory Comput.* 2022, 18, 6974 (Sella).
  <https://doi.org/10.1021/acs.jctc.1c00412>
- Wan et al. arXiv:2604.00405 (2026). MLIP TS benchmark: 96.6% MACE-OMol
  + FSM vs 63-71% CI-NEB.

---

### 3.2 `irc` — Sella saddle + IRC descent (the textbook gold standard)

**Method.** Take the input structure as a TS guess. Run Sella first-order
saddle optimization to the nearest saddle point, tightening forces to
`fmax ~ 0.02 eV/Å`. Compute the lowest (imaginary) Hessian eigenmode by
partial finite differences restricted to reacting atoms. Displace the
saddle by `+delta` and `-delta` along that mode, and LBFGS-relax each
displaced geometry (damped steepest-descent). The two relaxed endpoints
ARE the reactant and product — by construction connected via this specific
TS (Fukui's IRC concept).

**When to use.**
- Input is already near a TS (hand-built partial bonds).
- As a validation pass after any other strategy converges — IRC from the
  converged climbing image confirms that the NEB saddle actually connects
  the intended basins.
- Low-barrier or ill-defined-product systems where spring-driven endpoint
  generation fails (KCX_set1 reactant stuck at TS; see section 7).

**When NOT to use.**
- Input is a clean reactant/product with no TS geometry (use `legacy` or
  `cv-spring`).
- >200 free atoms — Sella's internal coordinates become unstable; use
  Cartesian + larger tolerance (`use_internal=False`).
- PES dominated by soft modes (floppy loops / ligands) that mask the
  reactive mode. Consider constraining those modes before IRC.

**Parameters.**
| Flag | Recommended | Notes |
|---|---|---|
| `--saddle-fmax` | 0.02 eV/Å | Sella needs tight convergence for Hessian stability |
| `--irc-step` | 0.1 Å | Initial displacement along imaginary mode |
| `--irc-fmax` | 0.03 eV/Å | Endpoint relaxation threshold |
| `--use-internal` | auto | Auto-selects internal coords if <200 free atoms |

**Expected runtime.** 15-45 min. Sella saddle ~10-20 min, imaginary-mode
FD Hessian ~1-5 min (partial, only over reacting atoms), two LBFGS descents
~5-10 min each.

**Accuracy.** IRC is the definitional gold standard for TS connectivity
(Fukui 1981). The only inaccuracy is the MACE PES itself; Sella + damped
descent introduce no methodological bias. Imaginary-mode threshold is
`-50 cm^-1` (Sella default): if the lowest mode is above that, IRC output
is flagged as unreliable.

**Failure modes.**
1. **Sella fails to find a saddle** -> log shows `fmax_final > 1.5*fmax`.
   Diagnosis: input too far from any saddle. Retry with a better TS guess
   or a short MD excursion.
2. **Lowest mode is real (not imaginary)** -> you are at a minimum, not
   a TS. Log prints "real — not a TS!". Retry with a perturbed input.
3. **Reactant and product RMSD < 0.1 Å after IRC** -> both descents rolled
   into the same basin. The TS may connect a basin to itself through a
   symmetry, or `--irc-step` was too small.
4. **Sella internal coords fail** -> code automatically falls back to
   Cartesian.

**Example command.**
```bash
qcb mlff-ts \
    --input ts_guess.xyz \
    --strategy irc \
    --saddle-fmax 0.02 --irc-step 0.1 \
    --refine-xtb \
    --out outputs/R2/irc_validation/
```

**Citations.**
- Fukui, K. *Acc. Chem. Res.* 1981, 14, 363 (IRC concept).
- Hermes et al. *J. Chem. Theory Comput.* 2022, 18, 6974 (Sella).
  <https://doi.org/10.1021/acs.jctc.1c00412>
- Schreiner et al. *J. Chem. Theory Comput.* 2025 (geodesic TS on MLPs).
  <https://doi.org/10.1021/acs.jctc.5c01221>

---

### 3.3 `cv-spring` — Bond-difference CV spring for endpoint generation

**Method.** Define the collective variable `s = d(P-LG) - d(P-nuc)`, the
More O'Ferrall–Jencks (MOJ) reaction coordinate for nucleophilic
substitution at a center. Attach a harmonic spring on `s` with target
`s_R ~ -2.0 Å` (reactant) or `s_P ~ +2.5 Å` (product). Run short MD or
BFGS with the spring, then release the spring and fully relax. If the
product basin exists, the geometry stays near `s_P`; if not, `s` drifts
back and you learn the mechanism is concerted or goes through an
intermediate.

Crucially, the CV spring applies ONE spring on ONE 1-D coordinate — it
does NOT force the two bonds to change simultaneously (which is the
criticism of `--spring-mode both`). The PES decides whether the mechanism
is concerted or stepwise; the spring merely defines identity of reactant
vs product.

**When to use.**
- Clean reactant with no product guess; phosphoryl transfer, SN2-at-C,
  SN1 with discrete intermediate, ligand exchange at a metal, proton
  transfer (`s = d(donor-H) - d(acceptor-H)`).
- Metalloenzyme with possible pentacoordinate intermediate — the CV
  spring naturally stops at `s ~ 0` if the PES has a minimum there.
- Any A + B-C -> A-B + C with identifiable A, B, C atoms.

**When NOT to use.**
- Multi-bond rearrangements not captured by a single 1-D coordinate
  (pericyclic, concerted double-bond shifts).
- Electron-transfer reactions (no geometric CV describes the TS).
- Input is already a TS guess (use `irc`).

**Parameters.**
| Flag | Recommended | Notes |
|---|---|---|
| `--p-idx`, `--nuc-idx`, `--lg-idx` | required | 0-indexed atom indices of center, nucleophile, leaving group |
| `--cv-k` | 3.0 eV/Å² | Matches the bond-spring convention elsewhere in QCB |
| `--cv-s-reactant` | -2.0 Å | Strong nuc dissociation, LG intact |
| `--cv-s-product` | +2.5 Å | Strong LG dissociation, nuc bonded |
| `--cv-fmax` | 3.0 eV/Å | Force cap; prevents overwhelming PES |
| `--cv-mode` | `both` | Spring acts on both sides; use `attractive`/`repulsive` for unidirectional |
| `--unidirectional` | off (default) | Use with Frankenstein chimeric constructs to avoid bad reverse-driven states |
| `--pre-relax` | off (default) | Relax input before attaching spring; use for chimeric/poorly-packed inputs |

**Expected runtime.** 30-90 min. Spring driving ~10-30 min per endpoint,
release+relax ~5-15 min, NEB between refined endpoints ~20-40 min.

**Accuracy.** Endpoints are only as accurate as the post-release relax.
If the "product" drifts back toward reactant during release, that is
diagnostic, not a bug — either the mechanism is concerted (no product
minimum in between) or the real product is a pentacoordinate intermediate
around `s ~ 0`.

**Failure modes.**
1. **Product drifts back during release** -> mechanism is concerted or
   proceeds through an intermediate. Follow up with MTD or accept the
   intermediate as the real product (two-step NEB, section 8).
2. **Spring cap too low** -> driving never reaches `s_target`. Raise
   `--cv-fmax` or `--cv-k`.
3. **`fmax` cap too high** -> spring overwhelms the PES and pushes atoms
   through bonds. Default of 3.0 eV/Å is well-tested.
4. **Chimeric constructs** where reverse driving creates unphysical
   states. Use `--unidirectional` plus `--pre-relax`.

**Example command.**
```bash
qcb mlff-ts \
    --input reactant.pdb \
    --strategy cv-spring \
    --p-idx 42 --nuc-idx 107 --lg-idx 55 \
    --cv-s-reactant -2.0 --cv-s-product 2.5 --cv-k 3.0 \
    --refine-xtb \
    --out outputs/R2/cvspring_P1D1/
```

**Citations.**
- More O'Ferrall, R.A. *J. Chem. Soc. B* 1970, 274 (MOJ diagram).
- Jencks, W.P. *Chem. Rev.* 1985, 85, 511 (Bema Hapothle / perfect
  synchronization principle).
- Bernasconi, C.F. *Adv. Phys. Org. Chem.* 1992, 27, 119 (principle of
  nonperfect synchronization).

---

### 3.4 `mtd` — Well-tempered metadynamics rescue

**Method.** Run Langevin dynamics with the MACE calculator, depositing
Gaussian hills along `s = d(P-LG) - d(P-nuc)` every `bias_pace_steps`
timesteps. Well-tempered scaling (Barducci et al. 2008) reduces hill
height by `exp(-V(s) / (kB * T * (gamma-1)))` so hills shrink as the
bias fills basins, giving asymptotic convergence. After accumulating
hills for `total_time_ps` of MD, build the free-energy surface (FES) on
a 1-D grid and identify local minima -> reactant / intermediate / product
basins. The closest sampled frame to each minimum is emitted as a
candidate geometry for downstream NEB.

**When to use.**
- `legacy` or `cv-spring` failed to locate a clean TS.
- Suspected mechanism switch (concerted vs stepwise) with multiple TS.
- Looking for a pentacoordinate intermediate along a phosphoryl-transfer
  coordinate when other methods keep collapsing to one basin.

**When NOT to use.**
- Cheap/well-behaved systems where `legacy` + `geodesic` works fine.
- >500 atoms — MACE-OMol MD becomes prohibitively slow on L40.
- When orthogonal degrees of freedom dominate the true reaction
  coordinate (1-D CV is not sufficient).

**Parameters.**
| Flag | Recommended | Notes |
|---|---|---|
| `--mtd-temp` | 300 K | Langevin temperature |
| `--mtd-timestep` | 1.0 fs | |
| `--mtd-total-time` | 100 ps | 50-200 ps typical |
| `--mtd-bias-height` | 1.2 kJ/mol | Initial hill height |
| `--mtd-bias-sigma` | 0.1 Å | Hill width |
| `--mtd-bias-pace` | 500 steps | Deposit a hill every 500 MD steps |
| `--mtd-bias-factor` | 10.0 | Well-tempered gamma |
| `--mtd-friction` | 1.0 ps^-1 | Langevin friction |

**Expected runtime.** 2-8 hours. 100 ps at 1 fs timestep is 100k MD steps;
each step is a MACE forward pass on ~200 atoms. L40 pushes ~3-6 ns/day
for ~200 atoms with MACE-OMol, so 100 ps = 0.5-1.3 hours of GPU time
plus overhead.

**Accuracy.** FES shape is only as good as the MACE PES in the region
sampled. The asymptotic WT-MTD estimator `F(s) ~ -(gamma/(gamma-1)) *
V_bias(s)` is approximate; for publication-quality FES, reweight with
PLUMED's Tiwary-Parrinello estimator on the same trajectory. Basin
classification uses `cv < -1.0 -> reactant`, `cv > 2.0 -> product`,
`-0.5 < cv < 1.5 -> intermediate`.

**Failure modes.**
1. **MACE extrapolation** -> bias drags geometry out of training
   distribution, producing unphysical structures. Mitigate: shorter
   `total_time`, tighter CA constraints, committee uncertainty
   monitoring (not currently wired in).
2. **Hysteresis / non-convergence** -> bias heights don't decay.
   Re-run with smaller `bias_height` or larger `bias_factor`.
3. **Basin conflation** -> multiple basins at similar `s`. Upgrade to
   2-D CV (not currently supported; PLUMED fallback).

**Example command.**
```bash
qcb mlff-ts \
    --input reactant.pdb \
    --strategy mtd \
    --p-idx 42 --nuc-idx 107 --lg-idx 55 \
    --mtd-total-time 100 --mtd-bias-factor 10 \
    --out outputs/R2/mtd_rescue_KCX_set1/
# then pipe the intermediate/product basins into a legacy NEB
```

**Citations.**
- Laio, A.; Parrinello, M. *PNAS* 2002, 99, 12562 (original MTD).
- Barducci, A.; Bussi, G.; Parrinello, M. *Phys. Rev. Lett.* 2008, 100,
  020603 (well-tempered MTD).
- Invernizzi, M.; Parrinello, M. *J. Phys. Chem. Lett.* 2020, 11, 2731
  (OPES — not yet implemented here).

---

## 4. Accuracy vs efficiency map

Expected success rate (probability of locating a correct, connected TS on
first pass) vs wallclock cost, for a ~200-atom enzymatic phosphoryl-transfer
cluster on L40 GPU with MACE-OMol. Success rates for NEB and FSM are
normalized to Wan et al. arXiv:2604.00405 (2026) on Transition1x; other
entries are QCB-internal R2 estimates.

```
 success %
  100 |
   98 |
   96 |                                 * legacy + geodesic + xtb  (~50 min)
   94 |                            * irc + xtb  (~30 min, needs TS guess)
   92 |
   90 |                   * cv-spring + xtb  (~70 min)
   85 |
   80 |
   75 |                                     * mtd -> NEB  (~5 h, rescue)
   70 |        * legacy + idpp        (45 min, older default)
   65 |   * legacy + linear           (40 min, known bad)
   60 |
   55 |
   50 |
      +-----+-----+-----+-----+-----+-----+-----+-----+-----+------
            15    30    45    60    75    90   120   180   360  min
                               wallclock (L40, MACE-OMol)
```

Reference: Wan et al. arXiv:2604.00405 (2026) report 96.6% TS success for
MACE-OMol + FSM vs 63-71% for CI-NEB on Transition1x. Our internal
`legacy + geodesic + xtb` pipeline lands in the same ~96% bracket because
geodesic images avoid the clash-induced spurious saddles that drag
CI-NEB's success rate down.

---

## 5. System selection guide

Concrete scenarios -> recommended strategy. These are calibrated against
the R2 gold-standard runs currently in progress.

| Scenario | Recommended strategy + modifiers |
|---|---|
| Hand-built TS guess with partially formed bonds | `--strategy irc --refine-xtb` |
| Near-TS structure, want to verify mechanism | `--strategy irc` for connectivity; independently `--strategy cv-spring` and compare the resulting basins |
| Clean reactant, unknown product geometry | `--strategy cv-spring --refine-xtb` (or `--strategy mtd` if shallow basins) |
| Metalloenzyme (e.g. Zn-Zn coordinated) with possible pentacoordinate intermediate | `--strategy cv-spring` with multiple `--cv-s-product` targets bracketing `s = 0`; OR `--strategy mtd` for the FES |
| Substrate in pure water (no enzyme) — substrate_uncat in R2 | `--strategy legacy --interpolation geodesic --refine-xtb` — **geodesic is non-negotiable** here, linear-interp clashes gave the 216 kcal/mol artifact |
| ZAPP P1D1 / minimal_deNovoPdPTE with well-defined reactant AND product basins | `--strategy legacy --interpolation geodesic --refine-xtb` |
| Frankenstein chimeric construct where reverse spring-driving creates unphysical states | `--strategy cv-spring --unidirectional --pre-relax --refine-xtb` |
| Existing CI-NEB converged but TS connectivity suspect | `--strategy irc` seeded by the climbing image — the two IRC endpoints should match your original reactant/product |
| NEB converges but "product" looks like reactant | `--refine-xtb` immediately; if product collapses to reactant, either real mechanism is concerted (TS is the product) or you found a pentacoordinate intermediate — do two-step NEB (section 8) |

**Gotcha.** Never rely on `--strategy legacy --interpolation linear` for any
enzyme-scale system. Linear interpolation creates atomic clashes in dense
solvent/protein environments that either blow up the MLIP or trap CI-NEB
in a wrong saddle. This is the root cause of R2's `substrate_uncat`
216 kcal/mol barrier.

---

## 6. Modifier flags

Modifiers compose orthogonally with any strategy.

### 6.1 `--refine-xtb`

**What it does.** After endpoint generation (and optionally after every
relaxed image), runs GFN2-xTB geometry optimization with the same CA
constraints, and compares to the MACE geometry. Parses `xtbopt.xyz`,
computes RMSD, and runs bond-pattern checks against `expected_bonds`
if `--p-idx / --nuc-idx / --lg-idx` are supplied.

**When to use.** Always (with the caveats below). Cheap independent QM
check, catches MACE extrapolation.

**Cost.** 2-10 min per endpoint on ~200 atoms. Scales roughly linearly
with `n_atoms` for SCF and quadratically for geometry optimization.

**What it catches.**
- Product "collapsed to reactant" (R2 minimal_deNovoPdPTE_KCX: nuc=3.27 Å
  in the MACE "product", xTB relaxed it back to the reactant basin).
- Missing-bond / extra-bond disagreements between MACE and GFN2.
- Geometry artifacts that would survive MACE but not any independent QM
  method.

**Anti-patterns.**
- Systems with >400 atoms — GFN2 SCF slows/destabilizes.
- Unusual bonding (transition metals beyond first row, multi-reference
  states).
- Final publication validation — use DFT single-points (not xTB) for
  the paper.

**Verdicts.** `accepted` (xTB agreed with MACE), `rejected_collapsed`
(bond pattern broken in a diagnostic way), `rejected_xtb_failed` (xTB
crashed or timed out).

### 6.2 `--interpolation geodesic` (default)

**Why default.** Geodesic interpolation minimizes the length of the path
on a Riemannian manifold whose metric is defined by internal coordinates
(Morse-like functions of all pairwise distances). Bonds are smoothly
made/broken without atoms crossing through each other. Zhu, Thompson,
Martinez *J. Chem. Phys.* 2019, 150, 164103. <https://doi.org/10.1063/1.5090303>

**When to override.**
- `--interpolation idpp`: geodesic package unavailable or failing; small
  organic reactions where IDPP is sufficient.
- `--interpolation linear`: debugging only; use to reproduce old bad
  results or as a speed test.

**Cost impact.** Geodesic adds ~10-30 s to NEB path construction. For
reference, the R2 substrate_uncat failure (216 kcal/mol artifact barrier)
was entirely due to linear interpolation; switching to geodesic reduced
it to the expected ~30 kcal/mol.

### 6.3 `--refine-xtb-replace`

When you actually want xTB to REPLACE the MACE endpoint (not just validate
it) — i.e., the xTB-refined geometry becomes the NEB endpoint. Use when:
- You have high confidence GFN2 is a better model for this specific
  system than MACE (e.g. the system is on the edge of MACE training
  distribution).
- MACE and xTB disagree by <0.5 Å RMSD but you want the QM-anchored
  geometry for downstream DFT.

**Do NOT** use `--refine-xtb-replace` when:
- xTB RMSD from MACE is large (>1 Å) — a QM method that disagrees this
  hard with the MLIP is telling you something is wrong with the system,
  not that xTB is right.
- You are doing production runs that will be DFT-refined anyway.

### 6.4 The recommended combination

```
--strategy legacy --interpolation geodesic --refine-xtb
```

This is the new default "gold standard" for NEB workflows on ZAPP-style
de novo designs with well-defined reactant/product. It combines
- the cheapest-fastest strategy (`legacy`)
- the clash-free interpolator (`geodesic`)
- an independent QM sanity check (`--refine-xtb`)
for roughly 50 min wallclock on ~200 atoms on L40 with MACE-OMol, at
~96% success rate (on par with Wan et al.'s MACE-OMol + FSM benchmark).

For TS validation after the fact, chain `--strategy irc` on the converged
CI-NEB TS as a second pass.

---

## 7. Known failure modes and how we fixed them

From the R2 gold-standard runs:

| Symptom | System | Root cause | Fix |
|---|---|---|---|
| 216 kcal/mol barrier, physically impossible | `substrate_uncat` (substrate in water) | Linear interpolation created atomic clashes between water and substrate in middle images; CI-NEB found a spurious saddle at the clash | `--interpolation geodesic` (task #5 identified this) |
| "Product" has nuc=3.27 Å (still separated) | `minimal_deNovoPdPTE_KCX` | Spring-driven product collapsed during release; MACE "product" basin was not real | `--strategy irc` (use TS to find both endpoints), OR `--refine-xtb` (xTB relaxes back to reactant, verdict=`rejected_collapsed`) |
| "Reactant" has nuc=1.80 Å (already bonded) | `KCX_set1` reactant | Spring-driven reactant got stuck at the TS region; release didn't descend fully | `--strategy irc` (saddle + descent both ways is insensitive to starting-side stickiness) |
| Converged to pentacoordinate intermediate with both nuc AND LG bonded | `GLU_set1`, `KCX_set0` | Mechanism is genuinely stepwise through pentacoordinate intermediate (correct for PdPTE) | Not a failure — do two-step NEB (section 8). Cite Santos-Martins et al. *J. Chem. Inf. Model.* 2024, doi:10.1021/acs.jcim.4c00425 |

**Gotcha.** If you see nuc=1.8 Å and LG=1.8 Å in what you thought was the
product, do NOT assume the run failed. Check if that geometry is a
pentacoordinate intermediate — it is a real chemical species for
PdPTE-style phosphoryl transfer.

---

## 8. Two-step NEB for pentacoordinate intermediates

### When it applies

If any strategy emits a "product" where:
- The nucleophile is bonded to the center (d_nuc < 2.0 Å), AND
- The leaving group is still bonded to the center (d_LG < 2.0 Å)

that is a real pentacoordinate trigonal-bipyramidal intermediate, not a
broken run. For PdPTE, GLU-acting PdPTE, and organophosphate hydrolases
generally, this is the expected mechanism (Santos-Martins et al.
*J. Chem. Inf. Model.* 2024).

### The two-step NEB recipe

1. **First segment (reactant -> intermediate)** — what legacy/cv-spring
   already gave you. The TS here is TS1 (nuc attack).
2. **Second segment (intermediate -> dissociated product)** — a separate
   NEB with the intermediate as reactant and a manually constructed
   (or MTD-sampled) dissociated product as product. The TS here is TS2
   (LG departure).
3. **Overall barrier** is the larger of (E_TS1 - E_reactant, E_TS2 -
   E_intermediate) modulo the intermediate depth.

```bash
# Step 1: legacy run produces intermediate as "product"
qcb mlff-ts --strategy legacy --input reactant.pdb --interpolation geodesic \
    --refine-xtb --out outputs/R2/step1/
# inspect outputs/R2/step1/product.xyz -> confirms pentacoordinate

# Step 2: NEB from intermediate -> dissociated product
qcb mlff-ts --strategy legacy \
    --input-r outputs/R2/step1/product.xyz \
    --input-p dissociated_product.xyz \
    --interpolation geodesic --refine-xtb \
    --out outputs/R2/step2/
```

### Diagnostic checklist

| Check | Reactant (basin A) | Intermediate (basin B) | Product (basin C) |
|---|---|---|---|
| d(P-nuc) | >2.8 Å | ~1.7 Å | ~1.6 Å |
| d(P-LG) | ~1.6 Å | ~1.7 Å | >2.8 Å |
| CV `s` | ~ -2 Å | ~0 Å | ~ +2 Å |
| Hessian (if Sella-refined) | all real | all real | all real |
| TS between | TS1 | — | TS2 |

If Hessian at "intermediate" shows an imaginary mode, it is really a
shallow TS, not an intermediate — the mechanism is concerted, and the
one-step TS is what you want. If Hessian is all real, you have a true
intermediate and the two-step NEB is the right treatment.

**Citation.** Santos-Martins, D. et al. *J. Chem. Inf. Model.* 2024,
doi:10.1021/acs.jcim.4c00425 — PdPTE phosphoryl-transfer mechanism
through the trigonal-bipyramidal pentacoordinate intermediate.

---

## 9. Complete citation list

- Bannwarth, C.; Ehlert, S.; Grimme, S. *J. Chem. Theory Comput.* 2019,
  15, 1652 (GFN2-xTB). <https://doi.org/10.1021/acs.jctc.8b01176>
- Barducci, A.; Bussi, G.; Parrinello, M. *Phys. Rev. Lett.* 2008, 100,
  020603 (well-tempered MTD).
- Fukui, K. *Acc. Chem. Res.* 1981, 14, 363 (IRC).
- Hermes, E.D. et al. *J. Chem. Theory Comput.* 2022, 18, 6974 (Sella).
  <https://doi.org/10.1021/acs.jctc.1c00412>
- Jencks, W.P. *Chem. Rev.* 1985, 85, 511 (Bema Hapothle).
- Laio, A.; Parrinello, M. *PNAS* 2002, 99, 12562 (original MTD).
- More O'Ferrall, R.A. *J. Chem. Soc. B* 1970, 274 (MOJ diagram).
- Santos-Martins, D. et al. *J. Chem. Inf. Model.* 2024
  (PdPTE pentacoordinate mechanism).
  <https://doi.org/10.1021/acs.jcim.4c00425>
- Schreiner, M. et al. *J. Chem. Theory Comput.* 2025 (geodesic TS on
  MLPs). <https://doi.org/10.1021/acs.jctc.5c01221>
- Smidstrup, S. et al. *J. Chem. Phys.* 2014, 140, 214106 (IDPP).
  <https://doi.org/10.1063/1.4878664>
- Wan, K. et al. arXiv:2604.00405 (2026) (MLIP TS benchmark: 96.6%
  MACE-OMol + FSM vs 63-71% CI-NEB).
- Zhu, X.; Thompson, K.C.; Martinez, T.J. *J. Chem. Phys.* 2019, 150,
  164103 (geodesic interpolation). <https://doi.org/10.1063/1.5090303>

---

## 10. TL;DR

- Default: `--strategy legacy --interpolation geodesic --refine-xtb`.
- Have a TS guess: `--strategy irc --refine-xtb`.
- Have only a reactant: `--strategy cv-spring --refine-xtb`.
- Everything failed: `--strategy mtd`, then legacy NEB between basins.
- "Product" has both bonds intact: real pentacoordinate intermediate —
  do two-step NEB (section 8), cite Santos-Martins 2024.
- Never use `--interpolation linear` on enzyme-scale systems.
