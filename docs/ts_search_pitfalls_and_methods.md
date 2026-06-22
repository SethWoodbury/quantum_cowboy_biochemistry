# TS search: method selection, pitfalls, and automatic sanity guards

Hard-won notes from validating the di-Zn OPAA/PTE theozyme (`mace-polar-m`). These apply to any
reaction-path → saddle → Hessian workflow in `cowboy-qc`.

## 1. Path-search method selection

| method | best for | why |
|---|---|---|
| **CI-NEB** (`neb`, geodesic) | **large, flexible** clusters; **dissociative** steps | geodesic interpolation routes the path *around* repulsive walls / clashes |
| **GSM / FSM** (`gsm --method gsm\|fsm`) | smaller / gas-phase / **concerted** reactions | grows nodes by pysisyphus internal-coord interpolation (no geodesic) |

**Failure we hit:** on the OPAA cluster, GSM and CI-NEB used *identical* endpoints, but GSM reported
a spurious **+17.84 kcal/mol** "barrier" — a node parked on the repulsive dissociative wall (leaving
group half-out, P–O ≈ 2.8 Å). CI-NEB's geodesic interpolation found the real **+8.12 kcal/mol**
pentacoordinate TS. **For large flexible enzyme TSs, prefer CI-NEB.**

GSM/FSM *do* honor `--fix-preset` now (pysisyphus `freeze_atoms`; grown nodes inherit it via
`Geometry.copy()`), so the backbone freeze is not the issue — **node placement** is.

## 2. Saddle refinement pitfall — the Sella-Cartesian collapse

`refine-ts --backend auto` runs the cascade `sella → sella-internal → dimer`. **Sella-Cartesian runs
first, and from a guess that sits inside a basin (no negative curvature in the reactive subspace) it
simply MINIMISES and "succeeds"** — so the climbing methods never run. The post-hoc frequency check
then reports `n_imag=0` and the TS is rejected, *but the saddle is already lost.*

Symptom: a refine that **converges but drops far below the NEB barrier** (it rolled ~14 kcal/mol
downhill into a side-basin). That collapse-basin is **not** a real intermediate — don't over-interpret it.

**Fix — use a backend that CLIMBS:**
- `--backend dimer` seeded by the NEB tangent (pass `--from-neb <neb_dir>`), or
- `--backend sella-internal` (internal coordinates represent the bond-stretch reaction mode directly).

A valid TS **holds near the NEB barrier with exactly one imaginary mode** overlapping the reactive atoms.

**...but the climbing optimizers can themselves fail on a large, flat, flexible enzyme surface.** On
the OPAA cluster we saw Sella-Cartesian **collapse** (minimise into a basin) *and* the dimer **diverge**:
ASE's dimer logged "could not figure out which atoms to displace → displace all atoms", then translated
uphill for hundreds of steps and **tore the active site apart** (the hydroxide flew 13 Å, E +325 kcal/mol —
non-physical, though *not* a crash: no NaN, the calculator ran fine). So the dimer is **not** a safe
default here.

**When every saddle search collapses or diverges, stop guessing and validate the NEB climbing image
DIRECTLY:** a partial Hessian on it (`freq --indices <reactive atoms>`) showing one strong imaginary
mode confirms a first-order saddle at NEB+MLFF accuracy (a well-converged CI-NEB climbing image *is* the
TS to that accuracy). A **CV-constrained optimisation** — freeze the reaction coordinate at the
climbing-image value, minimise the rest, then a single Hessian — is the other robust fallback.

## 3. Automatic sanity guards (`quantum_engine.analysis.ts_sanity`)

To make these failures loud instead of silent, two cheap checks (geometry + energies only, no extra
force calls) now run automatically and log banner-style warnings into the result `warnings` field:

- **`ts_endpoint_similarity`** (wired into GSM/FSM results) — warns when a "TS" is geometrically
  ~identical to the reactant or product (small RMSD, or vastly closer to one endpoint than the other),
  i.e. it's a basin, not a saddle. *This is the check that would have caught the GSM-≈-product failure immediately.*
- **`flag_spurious_path_peak`** — warns when a string/NEB peak is a lone "spike" above both neighbours
  (a misplaced interpolation node), is an endpoint, or has an outlier perpendicular force.
- **`refine-ts`** now emits a loud warning when it converges with `n_imag < expected` (the collapse).

## TL;DR for an enzyme TS
1. Minimise reactant + product (`--fix-preset ca-only`).
2. **CI-NEB (geodesic)** between them.
3. Refine the climbing image with **`--backend dimer --from-neb`** (or `sella-internal`) — never trust an
   `auto` refine that collapses far below the barrier.
4. Require **one imaginary mode** with reactive-atom overlap (the partial-Hessian gate in `refine-ts`).
