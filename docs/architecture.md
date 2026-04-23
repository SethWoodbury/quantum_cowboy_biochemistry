# QCB Architecture

## Package layout

```
qcb/
  calc/
    factory.py         # make_calc() — MACE calculator instantiation
  io/
    structure.py       # load_structure(), write_pdb() — PDB/XYZ/CIF I/O with biotite
    constraints.py     # parse_constraints(), preset_to_specs() — freeze/unfreeze grammar
  ops/                 # Gaussian-style atomic operations
    sp.py              # single-point energy
    opt.py             # energy minimization (LBFGS / BFGS / FIRE)
    md.py              # molecular dynamics (Langevin NVT / Verlet NVE)
    freq.py            # vibrational frequencies via finite-difference Hessian
    scan.py            # 1D bond/angle/dihedral scan (like opt=modredundant)
    saddle.py          # Sella saddle-point search
    irc.py             # intrinsic reaction coordinate from a TS
    neb.py             # nudged elastic band (R + P given)
    mtd.py             # well-tempered metadynamics
    ts.py              # full TS pipeline (native composition of saddle/irc/neb/mtd);
                       #   legacy subprocess to scripts/run_neb_ts.py available via --legacy-subprocess
  mlff/                # lower-level ML-FF internals
    irc.py             # Sella saddle + IRC primitives
    interpolation.py   # geodesic / IDPP / linear NEB interpolation
    xtb_refine.py      # xTB GFN2 endpoint refinement
    cv_spring.py       # bond-difference CV spring constraint
    metadynamics.py    # pure-Python WT-MTD implementation
    cif_writer.py      # atomworks-based CIF output
    endpoint_generation.py  # legacy spring-driven endpoint methods
    models.py          # MACE model registry (legacy; use qcb.calc.factory)
    dimer.py           # ASE dimer method wrapper
  prep/                # structure preparation (protonation, extraction)
  qm/                  # QM job generation (Gaussian, ORCA, xTB)
  analysis/            # post-processing (barriers, bonds, CIF)
  config.py            # cluster-specific paths

scripts/
  qcb                  # NEW: top-level CLI entry point (qcb <op> <input> ...)
  run_neb_ts.py        # legacy full-pipeline entry (works; redirects users to `qcb ts`)
  protonate_active_site.py
  refine_ts_guess.py
  generate_cifs.py
  analyze_neb_results.py
```

## Design principles

1. **One operation per module.** Each `qcb/ops/*.py` file does one thing
   (single-point, minimization, MD, frequencies, scan, etc.) and returns a
   standardized result dict.

2. **Shared infrastructure.** Calculator creation (`qcb.calc`), structure I/O
   (`qcb.io`), and constraint parsing (`qcb.io.constraints`) are used by every
   op. No duplicated code.

3. **Consistent interface.** Every op's `run()` accepts:
   - `atoms`: ASE `Atoms` object (with `.calc` optionally set)
   - `calculator`: ASE calculator (assigned if `atoms.calc is None`)
   - `outdir`: output directory (created if missing)
   - `constraint`: ASE constraint, list, or None
   - `**kwargs`: operation-specific parameters

4. **Standardized output.** Every op returns a dict with at minimum:
   - `status`: "converged" | "completed" | "not_converged" | "failed"
   - `atoms`: final Atoms object (may be same as input for `sp`)
   - `energy_eV`: final energy
   - `outputs`: dict of paths to trajectories, logs, plots, etc.
   Plus operation-specific keys (frequencies, barriers, basin depths, ...).

5. **Composable.** Higher-level ops (`ts`) are implemented by composing lower
   ones (`saddle` + `irc` + `neb`). Users can compose their own pipelines.

## Usage model

```bash
# Python (from a notebook or script)
from qcb.calc import make_calc
from qcb.io import load_structure, parse_constraints, build_fix_atoms
from qcb.ops import opt, md, freq

atoms, template, charge = load_structure("enzyme.pdb")
atoms.calc = make_calc("mace-omol", charge=charge)
atoms.info["charge"] = charge

# Freeze all alpha carbons
mask = parse_constraints(atoms, template, ["atoms CA"])
c = build_fix_atoms(mask)

# Minimize
r_opt = opt.run(atoms, outdir="out/opt", constraint=c, fmax=0.01)

# Run MD on the minimized structure
r_md = md.run(r_opt["atoms"], outdir="out/md", constraint=c,
              total_time_ps=5.0, temperature_K=300)

# Frequencies at the optimized geometry
r_freq = freq.run(r_opt["atoms"], outdir="out/freq", constraint=c)
```

```bash
# CLI
qcb opt enzyme.pdb --fmax 0.01 --fix-preset ca-only
qcb md  enzyme.pdb --time 5 --temp 300 --fix-preset ca-only
qcb freq enzyme.pdb --fix-preset ca-only
qcb scan enzyme.pdb --coord bond --indices 133 120 --start 1.8 --end 3.2 --n-steps 20
qcb ts  enzyme.pdb --strategy irc
```

## Constraint grammar

The `--fix` and `--free` flags accept spec strings. Multiple specs union into
a mask:

| Spec | Meaning |
|------|---------|
| `residue HIS ASP` | atoms in HIS or ASP residues |
| `resid 100 101 102` | atoms in residue IDs 100-102 |
| `chain A B` | atoms in chain A or B |
| `atoms CA CB` | atoms named CA or CB (anywhere) |
| `element C H` | atoms of carbon or hydrogen |
| `range 0 50` | atom indices 0 through 50 |
| `all` | every atom |
| `none` | no atoms |

Presets (expand to specs + exclusions):
- `ca-only` → fix CA atoms of protein residues (ligand/water/metals free)
- `backbone` → fix CA, C, N, O of protein residues
- `backbone-water` → backbone + water oxygens
- `none` → no constraints

Example:
```bash
qcb md input.pdb --fix "chain B" "atoms CA" --free "residue YYE"
# Fix all chain-B atoms AND any CA anywhere, but unfix YYE ligand atoms.
```

## Backward compatibility

The legacy `scripts/run_neb_ts.py` continues to work unchanged. It supports
the same `--strategy` flag and all 5 TS-search strategies (legacy, irc,
cv-spring, mtd). The new `qcb ts` subcommand is a thin wrapper around it
for now; a future refactor will re-implement the TS pipeline as a
composition of `qcb.ops` without subprocessing.

All existing SLURM scripts and notebooks continue to work. Migration is
opt-in: new scripts should prefer `qcb <op>` since it's faster for
non-TS operations (no subprocess overhead) and has a cleaner API.
