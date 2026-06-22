# Models Guide (Cowboy Quantum Chemistry)

Every ML force field / quantum-chemistry energy function wired into the pipeline, how to
select one, which container it loads in, and when to use each.

**Single source of truth:**
- `quantum_engine/site.py` — `MACE_MODELS`: alias → on-disk weight path (edit here to add/move weights).
- `quantum_engine/calc/factory.py` — `ENERGY_FAMILIES`: the family registry (`register_energy`).

Adding a brand-new model or family is a one-liner + a small adapter — see
[`docs/extending.md`](extending.md). This file is the *catalogue*; that file is the *how-to-extend*.

---

## How a model is selected

Every op (opt / scan / neb / refine-ts / freq / …) calls `make_calc(model=<alias>, charge=, spin=, device=)`
and gets an ASE-compatible `Calculator` back. Dispatch is by **family predicate** over the alias
(first match wins); `mace` is the implicit fallback (also catches absolute `.model` paths). Aliases are
case-insensitive.

```bash
python -m quantum_engine.cli <op> ... --model mace-omol        # charge-aware MACE
python -m quantum_engine.cli <op> ... --model so3lr-m          # SO3LR (JAX sidecar)
```

---

## Model summary table

| Family | Aliases | Lineage / DFT level | Metals | Charge/Spin | Container | Notes |
|--------|---------|---------------------|:------:|:-----------:|-----------|-------|
| **MACE-OMol** | `mace-omol` (=`mace-omol25`) | OMol25 · ωB97M-V/def2-TZVPPD | ✓ | ✓ | main | Best barrier accuracy; XL (1024ch), needs A6000+ |
| **MACE-MH** | `mace-mh`, `mace-mh-1` | 7 heads (see below) | ✓ | ✓ (omol head) | main | Multi-level; `--head omol` = OMol quality on A4000 |
| **MACE-MP** | `mace-mp` | r2SCAN (MatPES+OMAT) | ✓ | ✗ | main | Fast general-purpose; good geometries |
| **MACE-OFF** | `mace-off-small/medium`, `mace-off`, `mace-off-24` | ωB97M-D3BJ (OFF23/24) | ✗ | ✗ | main | Organic only (H,C,N,O,F,S) — **no P, no metals** |
| **MACE-POLAR** | `mace-polar-s/m/l`, `mace-polar` | Polarizable + long-range electrostatics | ✓ | ✓ | main | Beta/early-access; baked into the main container |
| **UMA** (FairChem) | `uma-sm`, `uma-s-1p1`, `uma-s-1p2`, `uma-m-1p1`(=`uma-m`) | OMol25-lineage, multi-task | ✓ | ✓ | **UMA sidecar** | Gated weights (FAIR license) |
| **eSEN** (FairChem) | `esen-s`(=`esen-sm-conserving`), `esen-sm-direct`, `esen-md-direct` | OMol25 · ωB97M-V | ✓ | ✓ | **UMA sidecar** | **conserving** only for TS (see below) |
| **AllScAIP** (FairChem) | `allscaip-md-conserving`, `allscaip-md-direct` | OMol25 | ✓ | ✓ | **UMA sidecar** | **conserving** only for TS |
| **ORB-Mol** | `orb-mol` (=`orb-mol-conservative`) | OMol25, Orbital Materials | ✓ | ✓ | main | Conservative (true gradients); Apache-2.0 |
| **AIMNet2-rxn** | `aimnet2-rxn` | Isayev lab, organic TS-tuned | ✗ | ✓ | main | H/C/N/O only; 4-member ensemble |
| **SO3LR** | `so3lr`, `so3lr-s`, `so3lr-m`, `so3lr-l` | **PBE0+MBD** (general-molecular-simulations) | ✓ | ✓ | **SO3LR sidecar** | **Independent** of OMol25; see below |
| **xTB / DFT** | `gfn2-xtb`, `gfn1-xtb`, `gfn-ff`, `g-xtb`, `xtb` | semiempirical (GFN) / g-xTB | ✓* | ✓ | main | *GFN2+metal warns/forbids — use a charge-aware MLFF |

> **Lineage matters.** `mace-omol`, `mace-polar`, `uma`, `esen`, `allscaip`, `orb-mol` are **all
> OMol25-lineage** — they share training-data bias, so cross-agreement among them is *not* independent
> validation. **SO3LR (PBE0+MBD) is the one independent ML cross-check** in the catalogue.

---

## Which container? (the 3-image story)

Dependency stacks conflict, so the catalogue spans three apptainer images. `make_calc` builds the
calculator **in-process**, assuming you're already inside the right image; if the backend isn't
importable it raises an `ImportError` naming the sidecar + the `apptainer exec` line to use.

| Image | Path | Hosts |
|-------|------|-------|
| **main** `quantum_chem-*.sif` | `/net/software/containers/users/woodbuse/quantum_chem/` | MACE (incl. POLAR), ORB, AIMNet2, xTB/g-xTB, SCINE, propka |
| **UMA sidecar** `uma-*.sif` | same dir | fairchem-core → **UMA + eSEN + AllScAIP** (numpy≥2 / torch~2.8) — `deps/uma_sidecar.def` |
| **SO3LR sidecar** `so3lr-*.sif` | same dir | **SO3LR** (JAX/orbax, Python ≥3.12) — `deps/so3lr_sidecar.def` |

POLAR is **baked into the main container** (the `graph_longrange` + `PolarMACE` fork are installed in
`deps/quantum_chem.def`), so `mace-polar-*` loads in-process — no separate venv.

---

## conserving vs direct (eSEN / AllScAIP)

FairChem ships two flavours. Only **conserving** checkpoints have forces = −dE/dx (true energy
gradients), which the saddle + partial-Hessian + IRC gate **requires**. **direct** checkpoints predict
forces with a separate head (faster, non-conservative) — fine for single-points / MD, **invalid for
TS/saddle/Hessian work**. Use `esen-sm-conserving` / `allscaip-md-conserving` for transition states.

---

## SO3LR — the independent cross-check

[`so3lr-v2-beta`](https://github.com/general-molecular-simulations/so3lr) — SO3krates (equivariant
transformer) + universal pairwise **L**ong-**R**ange (electrostatics / dispersion / ZBL repulsion), from
**general-molecular-simulations** (Unke / Müller, TU Berlin / BIFOLD). JACS 2025, 147(37), 33723.

- **Why it matters:** trained on **PBE0+MBD** data, a different lineage from the OMol25 models — so it's
  an independent adjudicator. On the OPAA di-Zn theozyme it gives a **validated** 14.4 kcal/mol forward
  barrier (one clean imaginary mode), vs the OMol25 models' ~6 kcal/mol *flat, saddle-less ridge*. That
  disagreement is the main argument for DFT/QM-MM adjudication.
- **Charge-aware** like MACE: set the net charge on `atoms.info['charge']` (the caller stamps it); the
  builder uses `lr_cutoff=100 Å` + `float64` to match the validated reference run.
- **Model = a directory.** Each size is a workdir (`params.pkl` + checkpoints), not a single file;
  `site.py` registers the `params.pkl` and `_make_so3lr` derives the workdir. Bundled sizes:
  `so3lr-{s,m,l}` + the original `so3lr`.
- **Weights:** `/net/databases/huggingface/mlFF_models/models--general-molecular-simulations--so3lr-v2-beta/`
  (`weights/` = checkpoints; `package/` = the JAX source + standalone `find_ts.py` + the OPAA reference TS).
- **Run it:** build the sidecar once (`apptainer build --fakeroot deps/so3lr_sidecar.def`), then
  `--model so3lr-m` inside it.

---

## MACE-MH heads

Seven DFT heads in one model; select with `--head`:

| Head | DFT level | Training set | Notes |
|------|-----------|--------------|-------|
| `omol` | ωB97M-V/def2-TZVPPD | OMol25 | OMol quality; **charge-aware; default** |
| `rgd1_b3lyp` | B3LYP-D3/def2-SVP | RGD1 | TS-optimized |
| `matpes_r2scan` | r2SCAN | MatPES | same as MACE-MP |
| `spice_wB97M` | ωB97M-D3BJ | SPICE | biomolecular |
| `ani2x` | ωB97X/6-31G* | ANI-2x | small molecules, fast |
| `tmqm` | TPSSh/def2-SVP | tmQM | transition-metal complexes |
| `dipeptides` | ωB97M-D3BJ | dipeptides | protein backbone |

```bash
python -m quantum_engine.cli neb input.pdb --model mace-mh-1 --head rgd1_b3lyp
```

---

## Paths on DIGS

Centralised HF-cache layout at `/net/databases/huggingface/mlFF_models/` (group `baker`, setgid 2775 —
anyone in the lab can add), naming `models--<org>--<name>/`. Per-alias resolution order in `site.py`:

1. `QCB_MACE_<ALIAS>` env override (e.g. `QCB_MACE_SO3LR_M=/path/to/workdir/params.pkl`)
2. central HF location (above)
3. legacy lab paths (`/mnt/projects/...`, `/home/gbg222/...`) — kept only as a fallback for in-flight runs

Gated weights (UMA, eSEN, AllScAIP, OMol25 extras) are populated by
`/net/databases/huggingface/mlFF_models/download_uma_models.sh` /
`deps/download_omol25_models.sh` with an approved `HF_TOKEN`. See the **authoritative** dict in
`quantum_engine/site.py` rather than hard-coding paths here.

---

## GPU memory + SLURM

Measured on ZAPP P1D1 (890 atoms) / Frankenstein (1060 atoms):

| Model | ~890 atoms | ~1060 atoms | ~475 atoms | Min GPU |
|-------|-----------|------------|-----------|---------|
| `mace-mp` | ~8 GB | ~10 GB | ~5 GB | **A4000 (16 GB)** |
| `mace-mh` | ~10 GB | ~12 GB | ~6 GB | **A4000 (16 GB)** |
| `mace-off` | ~12 GB | ~14 GB | ~7 GB | **A4000 (16 GB)** |
| `mace-omol` | ~20 GB | ~28 GB | ~12 GB | **A6000 (48 GB)** / B4000 |
| `mace-polar-m` | ~14 GB | ~18 GB | ~8 GB | **A4000** (tight) |
| `mace-polar-l` | ~22 GB | ~30 GB | ~13 GB | **B4000 (32 GB)** |
| `so3lr-m` | JAX, modest | — | — | A4000+; ~6 min/TS on one A100 |

| GPU | VRAM | Partition | Notes |
|-----|------|-----------|-------|
| A4000 | 16 GB | `gpu` | Most available. mace-mp/mh, so3lr. |
| B4000 | 32 GB | `gpu-b4000` | mace-omol <600 atoms. |
| A6000 / L40 | 48 GB | `gpu` / `gpu-bf` | mace-omol ~1000 atoms. |
| H200 | 80 GB | `gpu-bf` | Fits everything; fastest. |

```bash
#SBATCH --partition=gpu      # mace-mp / mace-mh / so3lr
#SBATCH --gres=gpu:a4000:1
#SBATCH --mem=32g
# mace-omol → --partition=gpu-bf --gres=gpu:h200:1
```

---

## When to use which

| Use case | Model | Why |
|----------|-------|-----|
| Quick screening (many designs) | `mace-mp` | Fast, A4000, good for ranking |
| Production barriers (publication) | `mace-omol` | Best accuracy; charge-aware; TS data |
| Budget production | `mace-mh-1 --head omol` | OMol quality on A4000 |
| Organic reactions (no metals/P) | `mace-off` / `aimnet2-rxn` | Organic TS-tuned |
| Charged metal pocket | `mace-polar-m` | Explicit polarization / long-range |
| **Independent cross-check** | `so3lr-m` | **Different lineage** (PBE0+MBD) — adjudicates OMol25 |
| Multi-level comparison | `mace-mh` (heads) | B3LYP vs ωB97M in one run |

**A robust theozyme workflow:** screen with `mace-mp`/`mace-mh --head omol`, confirm barriers with
`mace-omol`, then **cross-check the winner with `so3lr-m`**. If OMol25 and SO3LR disagree by more than a
couple kcal/mol (as on OPAA), escalate to DFT/QM-MM before trusting the number.

---

## Dual-model strategy (`--model-relax`)

Split a cheap model (relaxation/MD) from an expensive one (NEB/TS energy surface):

```bash
python -m quantum_engine.cli neb input.pdb --model mace-omol --model-relax mace-mp
```

Endpoint driving/MD/polish use the cheap model; NEB path + CI-NEB climbing image + saddle refinement use
the expensive one. Works because mace-mp and mace-omol agree on geometries and differ mainly on
energetics. Don't mix very different models (e.g. mace-off relax + mace-omol NEB), and remember endpoints
optimized on one PES may not be exact minima on the other.

---

## Historical validated results (MACE, ZAPP P1D1, 890 atoms, exp. ΔG‡ = 21.5 kcal/mol)

| Model | Barrier | Wall time | GPU | vs exp |
|-------|---------|----------|-----|--------|
| mace-mp (r2SCAN) | 19.5 | ~129 min | A4000 | −2.0 |
| mace-omol (ωB97M-V) | 21.3 | ~240 min | A6000 | −0.2 |
| mace-mh --head omol | ~20 | ~145 min | A4000 | ~−1.5 |

mace-mp is already within 2 kcal/mol; mace-omol is best but ~2× slower on a bigger GPU. (These are a
historical MACE-only run; for new systems, cross-check with SO3LR as above.)
