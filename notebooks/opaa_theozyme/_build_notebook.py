#!/usr/bin/env python3
"""Generate the complete OPAA di-Zn theozyme TS-pipeline notebook.

Each pipeline STEP is a markdown header + a driver code cell laid out
INPUTS / SETTINGS / CONSTANTS / BUILD COMMAND(S) / SUBMIT, mirroring the user's
house style (theozyme_opaa_i1_260521.ipynb). Driver cells build the qcb command(s),
write them to a CMDS_DIR commands file, and emit an sbatch array script via
nb.submit_array_job — preview by default, submit when SUBMIT=True.
"""
import json
from pathlib import Path

cells = []
def md(text):  cells.append(("markdown", text))
def code(text): cells.append(("code", text))

# ════════════════════════════════════════════════════════════════════════════
md(r"""# OPAA di-Zn phosphotriesterase theozyme — full TS-optimization pipeline
### command / sbatch generator (Cowboy Quantum Chemistry, `qcb`)

A reaction-agnostic, plug-and-play notebook that turns the protonated theozyme into
a **validated transition state + barrier**. The OPAA di-Zn site (ligand `PXN`,
carboxy-lysine `KCX`, bridging hydroxide) is only the *test case* — every atom index,
charge, and bond comes from your edits, nothing is hardcoded.

**How to use.** Run the **INIT** cell once. Then each step cell below builds its
`qcb` command(s), writes them to `CMDS_DIR/<step>.cmds`, and writes an sbatch array
script to `SUBMIT_DIR/<step>.sh`. By default it only *previews* (prints the command +
the `sbatch ...` line). Set `SUBMIT = True` in a cell (or call with `submit=True`) to
actually queue it. Lightweight steps (protonate / monitor / reaction-spec) just print.

**The pipeline (pick the path that fits your reaction):**

```
 0  protonate ........ qcb protonate            → protonated theozyme PDB (+ protomers)
 1  monitor .......... qcb monitor              → bond/metal coordination sanity report
 2  reaction-spec .... write rxn.yaml           → forming/breaking bonds, reactive atoms, cv
 3  relax ............ qcb opt   (CA-frozen)    → relaxed active site
 ── get a TS GUESS by ONE of: ───────────────────────────────────────────────────
 4  scan ............. qcb scan + extract       → 1D relaxed bond scan → max-E frame (TS guess)
 5  path search ...... qcb neb / gsm            → CI-NEB / GSM peak (TS guess) from R + P
 6  ts-entry ......... qcb ts-entry             → ONE call: path→refine→Hessian→IRC gate
 7  React-OT ......... qcb ts-propose (sidecar) → generative TS guess from R + P  (CHNO only)
 8  AEFM ............. qcb ts-refine  (sidecar) → ML-refine ANY guess              (CHNO only)
 ── then VALIDATE the guess: ─────────────────────────────────────────────────────
 9  refine-ts ........ qcb refine-ts            → saddle + partial-Hessian (1 imag mode)
10  validate-ts ...... qcb validate-ts          → tiered Hessian validation
11  verify-irc-like .. qcb verify-irc-like      → imag-mode displacement → R/P basins
12  DFT (optional) ... qcb ts-entry --engine orca → native ORCA NEB-TS for a DFT reference
```

Steps 4/5/6/7+8 are **alternative ways to get a TS guess** — use whichever suits the
reaction; 6 (`ts-entry`) wraps guess→refine→gate in one call. Steps 9–11 are the
acceptance authority and run regardless of how the guess was made.""")

# ════════════════════════════════════════════════════════════════════════════
md(r"""## INIT — project, paths, containers, helpers  *(run once per session)*""")

code(r'''# ══════════════════════════════════════════════════════════════════════════════
#  NOTEBOOK INITIALIZATION — run this cell at the start of every session
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_NAME = 'opaa_theozyme'

# > USER CONFIGURATION
HOME_DIR     = '/home/woodbuse/'
THEOZYME_DIR = f'{HOME_DIR}for/antonia/opaa_theozyme/'

# >> manual overrides (set to None to use defaults: notebook dir / WORKING_DIR/output)
_WORKING_DIR_OVERRIDE = THEOZYME_DIR
_OUTPUT_DIR_OVERRIDE  = THEOZYME_DIR

# >> subdirectories to create (each becomes an UPPERCASE <NAME>_DIR global)
WORKING_SUBDIRS = ['cmds', 'submit', 'logs', 'FINAL_THEOZYMES']
OUTPUT_SUBDIRS  = ['protomers', 'monitor', 'reaction_spec', 'relax_minimize',
                   'scan', 'path_search', 'ts_entry', 'generative',
                   'refine_ts', 'ts_validation', 'dft']

# > IMPORTS
import json, os, shlex, subprocess, sys, textwrap
from pathlib import Path
import numpy as np, pandas as pd

# >> custom notebook helpers (vendored copy ships in the repo; falls back to your
#    special_scripts copy). Provides resolve_*/setup_directories/print_initialization
#    and the SLURM helpers submit_array_job / submit_cpu.
QUANTUM_COWBOY_DIR = '/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/'
for _p in (f'{QUANTUM_COWBOY_DIR}notebooks/lib',
           f'{HOME_DIR}special_scripts/notebook_functions'):
    if Path(_p).is_dir() and _p not in sys.path:
        sys.path.insert(0, _p)
import notebook_core as nb

# > PATHS
WORKING_DIR = nb.resolve_working_dir(override=_WORKING_DIR_OVERRIDE, strip_mnt=True)
OUTPUT_DIR  = nb.resolve_output_dir(WORKING_DIR, override=_OUTPUT_DIR_OVERRIDE, strip_mnt=True)
for p in (WORKING_DIR, OUTPUT_DIR):
    Path(p).mkdir(parents=True, exist_ok=True)
nb.setup_directories(WORKING_DIR, WORKING_SUBDIRS, export_globals=True, globals_dict=globals())
nb.setup_directories(OUTPUT_DIR,  OUTPUT_SUBDIRS,  export_globals=True, globals_dict=globals())
nb.set_pandas_display(all_on=True)

# ── CONTAINERS ────────────────────────────────────────────────────────────────
# EDIT to your deployed sif paths. The MAIN sif has the MLFFs (MACE/POLAR), xtb,
# and the `qcb` CLI on PATH. The generative MODELS live in their own sidecars
# (built from deps/*.def → quantum_cowboy_biochemistry/containers/), which carry the
# generative model only (no MLFF) — see the two-step handoff in steps 7–8.
MAIN_SIF    = '/net/software/containers/users/woodbuse/quantum_chem/quantum_chem-20260604.sif'
UMA_SIF     = '/net/software/containers/users/woodbuse/quantum_chem/uma-20260527.sif'
REACTOT_SIF = f'{QUANTUM_COWBOY_DIR}containers/reactot-20260605.sif'   # React-OT proposer
AEFM_SIF    = f'{QUANTUM_COWBOY_DIR}containers/aefm-20260605.sif'      # AEFM refiner

def container_for(model):
    """Route an energy-model alias to the sif that can load it."""
    m = (model or '').lower()
    if m.startswith(('uma', 'esen', 'allscaip')):
        return UMA_SIF
    return MAIN_SIF        # mace* / mace-polar* / mace-omol* / xtb / qc

def apptainer(sif, gpu=True):
    """apptainer exec prefix. --nv on GPU; binds /home + /net."""
    pre = ['apptainer', 'exec'] + (['--nv'] if gpu else [])
    return pre + ['--bind', '/home', '--bind', '/net', sif]

def qcb(sif, *args, gpu=True):
    """A `qcb <args>` invocation inside the MAIN sif (qcb is on PATH there)."""
    return [*apptainer(sif, gpu=gpu), 'qcb', *map(str, args)]

def qcb_sidecar(sif, *args, gpu=True):
    """A `python -m quantum_engine.cli <args>` invocation inside a generative
    sidecar (qcb isn't installed there; the repo is bind-mounted onto PYTHONPATH)."""
    return [*apptainer(sif, gpu=gpu), 'env', f'PYTHONPATH={QUANTUM_COWBOY_DIR}',
            'python', '-m', 'quantum_engine.cli', *map(str, args)]

# ── STAGE → cmds-file + sbatch-script helper ──────────────────────────────────
def stage(name, cmds, *, gpu=True, time='12:00:00', cpus=4, mem='48g',
          gpu_class='small', constraint=None, cmds_per_job=1, submit=False):
    """Write `cmds` to CMDS_DIR/<name>.cmds, write an sbatch array script to
    SUBMIT_DIR/<name>.sh via nb.submit_array_job, print both, and (if submit=True)
    queue it. `cmds` is a list of full command strings (one array task each)."""
    cmds = [c if isinstance(c, str) else ' '.join(map(str, c)) for c in cmds]
    cmds_file   = f'{CMDS_DIR}{name}.cmds'
    submit_file = f'{SUBMIT_DIR}{name}.sh'
    Path(cmds_file).write_text('\n'.join(cmds) + '\n')
    nb.submit_array_job(
        cmds_file, time, cpus, f'opaa_{name}', mem, submit_file, LOGS_DIR,
        num_jobs=len(cmds), cmds_per_job=cmds_per_job,
        queue=('gpu' if gpu else 'cpu'),
        gpu_class=(gpu_class if gpu else None), constraint=constraint)
    print(f'### {name}: {len(cmds)} command(s) → {cmds_file}')
    for c in cmds:
        print('\n' + c)
    print(f'\n# sbatch script written → {submit_file}')
    if submit:
        r = subprocess.run(['sbatch', submit_file], capture_output=True, text=True)
        print('# ' + (r.stdout or r.stderr).strip())
    else:
        print(f'# to queue it:   sbatch {submit_file}')
    return cmds_file, submit_file

# > INITIALIZE
os.chdir(WORKING_DIR)
nb.print_initialization(WORKING_DIR, OUTPUT_DIR, project_name=PROJECT_NAME,
                        globals_dict=globals(), preview=False)
print('\nCONTAINERS:')
for _n, _s in [('MAIN', MAIN_SIF), ('UMA', UMA_SIF), ('REACTOT', REACTOT_SIF), ('AEFM', AEFM_SIF)]:
    print(f'  {_n:8} {"OK " if Path(_s).exists() else "MISSING"} {_s}')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r"""## Step 0 — Protonate the theozyme  (`qcb protonate`)

Deterministic, staged protonation of the protein (termini caps → PTM checks →
pH-aware canonical → His tautomers → propka → optional protomers), with an optional
CPU MLFF relax of *only* the new hydrogens. HETATM (ligand/metal/water) is assumed
already protonated. Lightweight → this cell prints the command (run it directly, or
flip `SUBMIT` to queue a single CPU job).""")

code(r'''# ── Step 0: PROTONATE ─────────────────────────────────────────────────────────
SUBMIT = False

# INPUTS
input_pdb  = f'{THEOZYME_DIR}opaa_3l7g_optimal_maximal_theozyme_pxn_unprotonated.pdb'  # <-- edit
output_pdb = f'{PROTOMERS_DIR}opaa_3l7g_optimal_maximal_theozyme_pxn.pdb'              # base name if protomers>1

# CORE SETTINGS
pH        = 7.5
protomers = 1                       # N>1 → write the N most likely protomers
n_cap, c_cap = 'nh2', 'cho'         # N: nh2|nh3+|nme|nfo|none   C: cho|coo-|cooh|conh2|conhme|none

# overrides (always win). "CHAIN:RESID": state/cap/code
set_overrides   = {}                # HID HIE HIP ASP ASH GLU GLH LYS LYN CYS CYM CYX TYR TYM ARG
n_cap_overrides = {}
c_cap_overrides = {}
ptm             = {}                # e.g. {"A:169": "KCX"} — carboxy-lysine bridging the di-Zn
ptm_charge      = {}                # e.g. {"A:169": -1}
ligand_charges  = {}                # e.g. {"PXN": -1} — for total-charge reporting

# optional MLFF relax of ONLY the new H (CPU, charge-free)
relax_h        = True
relax_h_model  = 'mace-off-small'   # auto-falls back to mace-mp if metals present
relax_h_fmax, relax_h_steps, relax_h_device = 0.05, 200, 'cpu'

skip_propka, keep_input_hydrogens = False, False
log_level = 'DEBUG'

# BUILD
cmd  = qcb(MAIN_SIF, 'protonate', '--input-pdb', input_pdb, '--output-pdb', output_pdb,
           gpu=False)
cmd += ['--pH', pH, '--protomers', protomers, '--n-cap', n_cap, '--c-cap', c_cap]
for spec, t in n_cap_overrides.items(): cmd += ['--n-cap-override', f'{spec}={t}']
for spec, t in c_cap_overrides.items(): cmd += ['--c-cap-override', f'{spec}={t}']
for spec, s in set_overrides.items():   cmd += ['--set', f'{spec}={s}']
for spec, c in ptm.items():             cmd += ['--ptm', f'{spec}={c}']
for spec, q in ptm_charge.items():      cmd += ['--ptm-charge', f'{spec}={q}']
for rn, q in ligand_charges.items():    cmd += ['--ligand-charge', f'{rn}={q}']
if relax_h:
    cmd += ['--relax-h', '--relax-h-model', relax_h_model, '--relax-h-fmax', relax_h_fmax,
            '--relax-h-steps', relax_h_steps, '--relax-h-device', relax_h_device]
if skip_propka:           cmd += ['--skip-propka']
if keep_input_hydrogens:  cmd += ['--keep-input-hydrogens']
cmd += ['--log-level', log_level]
cmd = [str(x) for x in cmd]

# (protonation is one short CPU job — print it; or queue a single CPU job)
print('### PROTONATE (copy-paste, or set SUBMIT=True) ###\n')
print(' '.join(cmd))
print(f'\n# Output: {output_pdb}' + ('  (+ _protomer1..N.pdb)' if protomers > 1 else ''))
if SUBMIT:
    nb.submit_cpu(' '.join(cmd), '02:00:00', 8, 'opaa_protonate', '32g',
                  f'{SUBMIT_DIR}protonate.sh')
    print(f'# sbatch {SUBMIT_DIR}protonate.sh')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r"""## Step 1 — Monitor the active site  (`qcb monitor`)

Non-constraining sanity report: measured key bonds + auto-detected metal
coordination shells. Use it to confirm the protonated geometry before you commit GPU
time. Pure CPU, instant — just prints + writes a JSON.""")

code(r'''# ── Step 1: MONITOR (bond / metal coordination report) ────────────────────────
input_pdb = f'{PROTOMERS_DIR}opaa_3l7g_optimal_maximal_theozyme_pxn.pdb'   # <-- edit

# 0-based atom index pairs to measure (e.g. forming/breaking bonds); [] for none.
monitor_bonds = []          # e.g. [(1849, 1871)]
report_metals = True        # auto-detect metals + their coordination shells

cmd = qcb(MAIN_SIF, 'monitor', input_pdb, '--outdir', MONITOR_DIR, gpu=False)
for i, j in monitor_bonds: cmd += ['--bond', f'{i},{j}']
if report_metals: cmd += ['--metals']
cmd = [str(x) for x in cmd]

print('### MONITOR (copy-paste; instant CPU) ###\n')
print(' '.join(cmd))
print(f'\n# JSON report → {MONITOR_DIR}')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r"""## Step 2 — Reaction spec  (the chemistry; nothing hardcoded)

The `ReactionSpec` YAML declares what reacts: `forming_bonds`, `breaking_bonds`,
`reactive_atoms`, an optional collective variable (`cv`), and `charge`/`spin`. Atom
tokens may be **0-based ASE indices**, **1-based PDB serials** (`serial:N`), or
`CHAIN:RESID:NAME`. This single file is threaded through `ts-entry` / `refine-ts` /
`validate-ts`. The cell writes it and validates it with `qcb reaction-spec`.""")

code(r'''# ── Step 2: REACTION SPEC ─────────────────────────────────────────────────────
spec_path  = f'{REACTION_SPEC_DIR}opaa_sn2_at_P.yaml'
struct_pdb = f'{PROTOMERS_DIR}opaa_3l7g_optimal_maximal_theozyme_pxn.pdb'  # to resolve tokens

# di-Zn phosphotriesterase: SN2-at-P — bridging hydroxide O attacks substrate P,
# leaving-group O departs. EDIT the atom tokens for YOUR structure
# (`qcb monitor`/PyMOL give serials; ReactionSpec accepts serial:N or CHAIN:RESID:NAME).
spec_yaml = textwrap.dedent("""\
    # OPAA di-Zn phosphotriesterase — SN2 at phosphorus
    charge: 0            # EDIT: di-Zn(II) + bridging OH- + substrate phosphate (~ -2 to -3)
    spin: 1              # closed-shell di-Zn(II) singlet
    forming_bonds:       # nucleophile O (hydroxide) -> P
      - [serial:1872, serial:1850]
    breaking_bonds:      # P -> leaving-group O
      - [serial:1850, serial:1860]
    reactive_atoms:      # atoms on the imaginary mode (Onuc, P, Olg)
      - serial:1872
      - serial:1850
      - serial:1860
    cv:                  # optional 1-D collective variable for scans / reactant-only
      kind: bond_difference
      atoms: [serial:1872, serial:1850, serial:1850, serial:1860]
""")

Path(spec_path).write_text(spec_yaml)
print(f'# wrote {spec_path}\n')
print(spec_yaml)

cmd = qcb(MAIN_SIF, 'reaction-spec', spec_path, '--structure', struct_pdb, gpu=False)
print('### VALIDATE (copy-paste; instant CPU) ###\n' + ' '.join(map(str, cmd)))
''')

# ════════════════════════════════════════════════════════════════════════════
md(r"""## Step 3 — Relax the active site  (`qcb opt`, CA-frozen)

Backbone-CA-frozen relax: scaffold the protein, let the chemistry breathe
(ligand/metals/waters/sidechains move). Charge-aware MLFF. `--fix-bond` can hard-pin
the forming/breaking bond *during* the relax if you want to hold the reactant geometry.""")

code(r'''# ── Step 3: RELAX (qcb opt) ───────────────────────────────────────────────────
SUBMIT = False

input_pdb  = f'{PROTOMERS_DIR}opaa_3l7g_optimal_maximal_theozyme_pxn.pdb'   # <-- edit
out_dir    = f'{RELAX_MINIMIZE_DIR}relax/'
relaxed_pdb = f'{out_dir}relaxed.pdb'

# charge / spin of the FULL active-site cluster (match the reaction spec)
charge, spin = 0, 1

# MLFF model (+ optional multi-head). mace-polar-m = charge-aware, polarizable, ideal
# for a charged Zn pocket; fall back to mace-mh-1 --head omol if POLAR isn't in the sif.
model, head, device = 'mace-polar-m', None, 'cuda'

fix_preset = 'ca-only'          # ca-only | backbone | backbone-water | none
extra_fix, extra_free = [], []  # extra select specs, e.g. ['residue HOH'], ['resid 169']
optimizer, fmax, max_steps = 'lbfgs', 0.05, 500
# optionally hold the forming/breaking bond during relax (0-based ASE idx [+ R0]):
fix_bonds = []                  # e.g. [[1849, 1871]]  or  [[1849, 1871, 1.8]]

cmd  = qcb(container_for(model), 'opt', input_pdb)
cmd += ['--model', model, '--charge', charge, '--spin', spin, '--device', device]
if head: cmd += ['--head', head]
cmd += ['--fix-preset', fix_preset]
for s in extra_fix:  cmd += ['--fix', s]
for s in extra_free: cmd += ['--free', s]
for b in fix_bonds:  cmd += ['--fix-bond', *map(str, b)]
cmd += ['--optimizer', optimizer, '--fmax', fmax, '--max-steps', max_steps,
        '--outdir', out_dir, '--output-pdb', relaxed_pdb]

stage('relax', [' '.join(map(str, cmd))], gpu=True, time='12:00:00',
      cpus=8, mem='64g', gpu_class='small', submit=SUBMIT)
print(f'\n# Output: {relaxed_pdb}')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r"""## Step 4 — 1D relaxed scan → TS guess  (`qcb scan`)

Slide the forming/breaking bond and relax everything else at each point
(`FixBondLength` on the scanned pair + the CA-only preset). The highest-energy frame
is your TS guess. **Indices are 0-based ASE indices** (`= PDB serial − 1`).""")

code(r'''# ── Step 4: SCAN + extract TS guess (qcb scan) ────────────────────────────────
SUBMIT = False

relaxed_pdb = f'{RELAX_MINIMIZE_DIR}relax/relaxed.pdb'                 # <-- from Step 3
out_dir      = f'{SCAN_DIR}scan/'
ts_guess_pdb = f'{out_dir}ts_guess.pdb'

charge, spin = 0, 1
model, head, device = 'mace-polar-m', None, 'cuda'
fix_preset = 'ca-only'

# scan the FORMING bond (0-based ASE indices = serial-1). Onuc..P here:
scan_indices = [1871, 1849]          # <-- edit (0-based)
scan_coord   = 'bond'                # bond | angle | dihedral
scan_start, scan_end, scan_n = 1.6, 3.0, 16
scan_fmax = 0.05

cmd  = qcb(container_for(model), 'scan', relaxed_pdb)
cmd += ['--model', model, '--charge', charge, '--spin', spin, '--device', device]
if head: cmd += ['--head', head]
cmd += ['--fix-preset', fix_preset, '--coord', scan_coord,
        '--indices', *scan_indices, '--start', scan_start, '--end', scan_end,
        '--n-steps', scan_n, '--fmax', scan_fmax, '--outdir', out_dir]

# Stage 4b — extract the max-energy frame as a TS-guess PDB (tiny helper script so the
# printed command stays copy-pasteable; runs in the same sif).
extract_py = f'{out_dir}extract_max_e.py'
extract_body = textwrap.dedent(f"""\
    import json; from pathlib import Path; import ase.io as io
    from quantum_engine.io import load_structure, write_pdb
    frames = io.read(r"{out_dir}scan-trajectory.xyz", index=":")
    e = lambda a: a.info.get("energy_eV", a.get_potential_energy())
    i = max(range(len(frames)), key=lambda k: e(frames[k]))
    _, bt, _ = load_structure(r"{relaxed_pdb}")
    write_pdb(frames[i], bt, r"{ts_guess_pdb}", total_charge={charge})
    print(f"TS guess = frame {{i}}/{{len(frames)}} -> {ts_guess_pdb}")
""")
Path(out_dir).mkdir(parents=True, exist_ok=True)
Path(extract_py).write_text(extract_body)
cmd_extract = [*apptainer(container_for(model), gpu=True), 'python', extract_py]

stage('scan', [' '.join(map(str, cmd)), ' '.join(map(str, cmd_extract))],
      gpu=True, time='24:00:00', cpus=8, mem='64g', cmds_per_job=2, submit=SUBMIT)
print(f'\n# Outputs: {out_dir}scan-trajectory.xyz, scan-summary.json, scan.png')
print(f'#          {ts_guess_pdb}  (TS guess → feed Step 9 refine-ts)')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r"""## Step 5 — Path search: CI-NEB / GSM / FSM  (`qcb neb` / `qcb gsm`)

If you have both a **reactant and a product** geometry (same atom order), a
double-ended path search gives a TS guess at the climbing image / string peak.
CI-NEB is robust on MLFFs; GSM/FSM are cheaper. The peak feeds `refine-ts`.""")

code(r'''# ── Step 5: PATH SEARCH (qcb neb | gsm) ───────────────────────────────────────
SUBMIT = False

reactant_pdb = f'{RELAX_MINIMIZE_DIR}relax/relaxed.pdb'      # <-- reactant basin
product_pdb  = f'{PATH_SEARCH_DIR}product.pdb'               # <-- product basin (same atom order!)
out_dir      = f'{PATH_SEARCH_DIR}neb/'

method = 'neb'                       # 'neb' (CI-NEB) | 'gsm' (use qcb gsm, see below)
charge, spin = 0, 1
model, head, device = 'mace-polar-m', None, 'cuda'

if method == 'neb':
    n_images, interpolation, optimizer = 11, 'geodesic', 'fire'
    cmd  = qcb(container_for(model), 'neb', reactant_pdb, product_pdb)
    cmd += ['--model', model, '--charge', charge, '--spin', spin, '--device', device]
    if head: cmd += ['--head', head]
    cmd += ['--fix-preset', 'ca-only', '--n-images', n_images,
            '--interpolation', interpolation, '--optimizer', optimizer,
            '--outdir', out_dir]
    # → highest-energy inner image; refine-ts reads it via --from-neb {out_dir}
else:  # gsm (fsm/gsm); note: gsm has no --spin/--fix-preset
    gsm_method, n_images = 'gsm', 15   # 'fsm' | 'gsm'
    cmd  = qcb(container_for(model), 'gsm', reactant_pdb, product_pdb)
    cmd += ['--method', gsm_method, '--model', model, '--charge', charge,
            '--device', device, '--n-images', n_images, '--outdir', out_dir]
    if head: cmd += ['--head', head]

stage('path_search', [' '.join(map(str, cmd))], gpu=True, time='24:00:00',
      cpus=8, mem='64g', submit=SUBMIT)
print(f'\n# Output: {out_dir}  → feed Step 9 with  qcb refine-ts --from-neb {out_dir}')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r"""## Step 6 — `ts-entry` orchestrator  (one call: path → refine → Hessian → IRC)

The modern all-in-one. Pick an `--entry`: `reactant-product` (path search → saddle →
gate), `ts-guess` (you already have a guess), or `reactant-only` (drive R along the
`cv` to a product basin, then proceed). `--rigor` scales images/backend/thresholds.
Swap in a generative guess with `--proposer` and/or an ML `--refiner` (steps 7–8).""")

code(r'''# ── Step 6: TS-ENTRY (orchestrator) ───────────────────────────────────────────
SUBMIT = False

entry      = 'reactant-product'      # reactant-product | ts-guess | reactant-only
spec_path  = f'{REACTION_SPEC_DIR}opaa_sn2_at_P.yaml'
out_dir    = f'{TS_ENTRY_DIR}{entry}/'

reactant_pdb = f'{RELAX_MINIMIZE_DIR}relax/relaxed.pdb'
product_pdb  = f'{PATH_SEARCH_DIR}product.pdb'      # for reactant-product
ts_guess_pdb = f'{SCAN_DIR}scan/ts_guess.pdb'       # for ts-guess

charge, spin = 0, 1
model, head, device = 'mace-polar-m', None, 'cuda'
rigor = 'standard'                   # draft | standard | publication

# optional knobs (None = use the rigor preset):
path_method   = None                 # neb | fsm | gsm-de | autoneb | pygsm-de ...
saddle_backend = None                # dimer | sella | pysisyphus-rsprfo | auto
proposer      = None                 # 'react-ot' (needs the sidecar — see Step 7) | 'midpoint'
refiner       = None                 # 'aefm' (needs the sidecar — see Step 8) | 'identity'
n_images      = None

cmd  = qcb(container_for(model), 'ts-entry', '--entry', entry,
           '--reaction-spec', spec_path, '--outdir', out_dir)
cmd += ['--model', model, '--charge', charge, '--spin', spin, '--device', device,
        '--rigor', rigor]
if head:           cmd += ['--head', head]
if entry in ('reactant-product', 'reactant-only'): cmd += ['--reactant', reactant_pdb]
if entry == 'reactant-product':                    cmd += ['--product', product_pdb]
if entry == 'ts-guess':                            cmd += ['--ts-guess', ts_guess_pdb]
if path_method:    cmd += ['--path-method', path_method]
if saddle_backend: cmd += ['--saddle-backend', saddle_backend]
if proposer:       cmd += ['--proposer', proposer]
if refiner:        cmd += ['--refiner', refiner]
if n_images:       cmd += ['--n-images', n_images]

stage('ts_entry', [' '.join(map(str, cmd))], gpu=True, time='48:00:00',
      cpus=8, mem='80g', gpu_class='large', submit=SUBMIT)
print(f'\n# Outputs: {out_dir}ts_entry.json (status/barrier/n_imag), gates.json, ts.* ')
print('# NOTE: --proposer react-ot / --refiner aefm need those models IN this sif.')
print('#       The prebuilt sidecars are model-only, so use the two-step handoff')
print('#       (Steps 7-8) → then this cell with --entry ts-guess.')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r"""## Step 7 — Generative TS proposer: **React-OT**  (sidecar, two-step handoff)

React-OT generates a TS guess directly from R + P in ~one shot (deterministic OT).
It runs in its **own sidecar** (model only, no MLFF), so this is step 1 of the
handoff: emit a guess here → feed it to `ts-entry --entry ts-guess` (Step 6) or
`refine-ts` (Step 9) in the MAIN sif. **Domain: H/C/N/O only** (the released model has
no Zn/P embedding) — use it for organic substrate reactions, not the metal cluster.""")

code(r'''# ── Step 7: REACT-OT proposer (generative; sidecar) ───────────────────────────
SUBMIT = False

reactant_xyz = f'{GENERATIVE_DIR}reactant.xyz'      # <-- CHNO R (same atom order as P)
product_xyz  = f'{GENERATIVE_DIR}product.xyz'       # <-- CHNO P
guess_out    = f'{GENERATIVE_DIR}reactot_guess.xyz'
charge, spin = 0, 1                                  # React-OT ignores charge (neutral CHNO)

cmd  = qcb_sidecar(REACTOT_SIF, 'ts-propose', '--method', 'react-ot',
                   '--reactant', reactant_xyz, '--product', product_xyz,
                   '--charge', charge, '--spin', spin, '--out', guess_out,
                   '--outdir', f'{GENERATIVE_DIR}reactot/')

stage('reactot_propose', [' '.join(map(str, cmd))], gpu=True, time='02:00:00',
      cpus=4, mem='32g', gpu_class='small', submit=SUBMIT)
print(f'\n# Output: {guess_out}  → Step 6 (--entry ts-guess --ts-guess ...) or Step 9.')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r"""## Step 8 — Generative TS refiner: **AEFM**  (sidecar, two-step handoff)

AEFM *refines* a low-fidelity TS guess (from a scan peak, NEB peak, or React-OT) into
a better one via learned equilibrium flow matching. Its **own sidecar** (model only).
Chain it after Step 7 for `React-OT → AEFM`, or feed it any guess. **Domain: H/C/N/O**.
It's a structure prior, not a saddle finder — the QM gate (Steps 9–11) still rules.""")

code(r'''# ── Step 8: AEFM refiner (generative; sidecar) ────────────────────────────────
SUBMIT = False

guess_xyz   = f'{GENERATIVE_DIR}reactot_guess.xyz'   # <-- any CHNO TS guess (e.g. from Step 7)
refined_out = f'{GENERATIVE_DIR}aefm_refined.xyz'
charge, spin = 0, 1

cmd  = qcb_sidecar(AEFM_SIF, 'ts-refine', '--method', 'aefm',
                   '--ts-guess', guess_xyz, '--charge', charge, '--spin', spin,
                   '--out', refined_out, '--outdir', f'{GENERATIVE_DIR}aefm/')

stage('aefm_refine', [' '.join(map(str, cmd))], gpu=True, time='02:00:00',
      cpus=4, mem='32g', gpu_class='small', submit=SUBMIT)
print(f'\n# Output: {refined_out}  → Step 6 (--entry ts-guess) or Step 9 refine-ts.')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r"""## Step 9 — Refine the TS  (`qcb refine-ts`: saddle + partial Hessian)

The acceptance core: dimer/Sella/pysisyphus saddle search → partial Hessian on the
reactive atoms → require exactly one imaginary mode (< cutoff) that overlaps the
reaction coordinate → write `ts_refined.pdb`. Accepts a TS-guess PDB *or* `--from-neb`
a path-search dir. `--reactive-atoms` are **1-based PDB serials**.""")

code(r'''# ── Step 9: REFINE-TS (saddle + partial Hessian) ──────────────────────────────
SUBMIT = False

# EITHER a TS-guess structure (scan/generative) OR a path-search dir via --from-neb:
ts_guess_pdb = f'{SCAN_DIR}scan/ts_guess.pdb'        # <-- edit (or set from_neb)
from_neb     = None                                  # e.g. f'{PATH_SEARCH_DIR}neb/'
template_pdb = f'{PROTOMERS_DIR}opaa_3l7g_optimal_maximal_theozyme_pxn.pdb'
out_dir      = f'{REFINE_TS_DIR}refine/'

charge, spin = 0, 1
model, head, device = 'mace-polar-m', None, 'cuda'
fix_preset = 'ca-only'

reactive_atoms = [1872, 1850, 1860]   # 1-based PDB serials (Onuc, P, Olg)
backend = 'dimer'                     # dimer | sella | sella-internal | pysisyphus-rsprfo | auto
saddle_fmax, saddle_max_steps = 0.02, 500
imag_cm_cutoff, imag_overlap, n_imag = -50.0, 0.5, 1

cmd  = qcb(container_for(model), 'refine-ts')
if from_neb: cmd += ['--from-neb', from_neb, '--template-pdb', template_pdb]
else:        cmd += [ts_guess_pdb]
cmd += ['--model', model, '--charge', charge, '--spin', spin, '--device', device,
        '--fix-preset', fix_preset, '--reactive-atoms', *reactive_atoms,
        '--backend', backend, '--saddle-fmax', saddle_fmax,
        '--saddle-max-steps', saddle_max_steps, '--imag-cm-cutoff', imag_cm_cutoff,
        '--imag-mode-overlap', imag_overlap, '--n-imag-expected', n_imag,
        '--outdir', out_dir]
if head: cmd += ['--head', head]

stage('refine_ts', [' '.join(map(str, cmd))], gpu=True, time='48:00:00',
      cpus=8, mem='80g', gpu_class='large', submit=SUBMIT)
print(f'\n# Outputs: {out_dir}ts_refined.pdb (validated TS), summary.json (PASS/FAIL)')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r"""## Step 10 — Validate the TS  (`qcb validate-ts`: tiered Hessian)

Independent, tiered Hessian validation of a refined TS: Tier A (reactive-atom partial
Hessian), Tier B (active-region Hessian), Tier C (iterative full-mode check). Confirms
exactly one imaginary mode on the reaction coordinate.""")

code(r'''# ── Step 10: VALIDATE-TS (tiered Hessian) ─────────────────────────────────────
SUBMIT = False

ts_pdb  = f'{REFINE_TS_DIR}refine/ts_refined.pdb'    # <-- from Step 9
out_dir = f'{TS_VALIDATION_DIR}validate/'

charge, spin = 0, 1
model, head, device = 'mace-polar-m', None, 'cuda'
reactive_atoms = [1872, 1850, 1860]   # 1-based PDB serials
tier = 'b'                            # a | b | c | all | comma list
active_region = None                  # select-grammar spec for Tier B, e.g. 'sphere 6.0 around resid 169'
imag_cm_cutoff, imag_overlap, n_imag = -50.0, 0.5, 1

cmd  = qcb(container_for(model), 'validate-ts', ts_pdb, '--outdir', out_dir)
cmd += ['--model', model, '--charge', charge, '--spin', spin, '--device', device,
        '--reactive-atoms', *reactive_atoms, '--tier', tier,
        '--imag-cm-cutoff', imag_cm_cutoff, '--imag-mode-min-overlap', imag_overlap,
        '--n-imag-expected', n_imag]
if head:          cmd += ['--head', head]
if active_region: cmd += ['--active-region', active_region]

stage('validate_ts', [' '.join(map(str, cmd))], gpu=True, time='24:00:00',
      cpus=8, mem='80g', gpu_class='large', submit=SUBMIT)
print(f'\n# Output: {out_dir}  (per-tier PASS/FAIL + frequencies)')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r"""## Step 11 — Verify IRC-like  (`qcb verify-irc-like`)

Displace along the imaginary mode in both directions and relax → confirm the TS
connects the intended reactant and product basins (energy drops on both sides).
Needs the imag-mode vector emitted by Step 9/10 (`--imag-mode <.npy|.xyz>`).""")

code(r'''# ── Step 11: VERIFY IRC-LIKE ──────────────────────────────────────────────────
SUBMIT = False

ts_pdb    = f'{REFINE_TS_DIR}refine/ts_refined.pdb'           # <-- from Step 9
imag_mode = f'{REFINE_TS_DIR}refine/imag_mode.npy'           # <-- emitted by refine-ts/validate-ts
out_dir   = f'{TS_VALIDATION_DIR}irc_like/'

charge, spin = 0, 1
model, head, device = 'mace-polar-m', None, 'cuda'
displacement, fmax, max_steps, optimizer = 0.20, 0.05, 200, 'lbfgs'

cmd  = qcb(container_for(model), 'verify-irc-like', ts_pdb, '--imag-mode', imag_mode,
           '--outdir', out_dir)
cmd += ['--model', model, '--charge', charge, '--spin', spin, '--device', device,
        '--displacement', displacement, '--fmax', fmax, '--max-steps', max_steps,
        '--optimizer', optimizer]
if head: cmd += ['--head', head]

stage('verify_irc_like', [' '.join(map(str, cmd))], gpu=True, time='24:00:00',
      cpus=8, mem='80g', gpu_class='large', submit=SUBMIT)
print(f'\n# Output: {out_dir}  (forward/back basins + Δenergies)')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r"""## Step 12 — *(optional)* DFT reference via ORCA native NEB-TS

For a publication-grade reference, route the whole TS step to ORCA's native NEB-TS /
OptTS through the QM-engine gateway (`ts-entry --engine orca`). `--no-execute` writes
the ORCA input + an sbatch wrapper *without* running, so you can inspect/queue it.""")

code(r'''# ── Step 12: DFT via ORCA (optional) ──────────────────────────────────────────
SUBMIT = False
EXECUTE = False                       # False → write ORCA input + wrapper, don't run

entry     = 'reactant-product'
spec_path = f'{REACTION_SPEC_DIR}opaa_sn2_at_P.yaml'
out_dir   = f'{DFT_DIR}orca_nebts/'
reactant_pdb = f'{RELAX_MINIMIZE_DIR}relax/relaxed.pdb'
product_pdb  = f'{PATH_SEARCH_DIR}product.pdb'
charge, spin = 0, 1
engine_method = 'wB97X-D3/def2-TZVP'  # passed through to the ORCA engine config

cmd  = qcb(MAIN_SIF, 'ts-entry', '--entry', entry, '--reaction-spec', spec_path,
           '--engine', 'orca', '--reactant', reactant_pdb, '--product', product_pdb,
           '--charge', charge, '--spin', spin, '--outdir', out_dir)
cmd += (['--execute'] if EXECUTE else ['--no-execute'])

stage('dft_orca', [' '.join(map(str, cmd))], gpu=False, time='72:00:00',
      cpus=16, mem='120g', submit=SUBMIT)
print(f'\n# Outputs: {out_dir}  (ORCA NEB-TS input/job; method = {engine_method})')
print('# Set the ORCA functional/basis + resources in the engine config; this cell')
print('# prepares the job — inspect it, then sbatch (or set EXECUTE=True).')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r"""---
### Done — what you have

A driver cell per pipeline step, each emitting a `qcb` command → `CMDS_DIR/<step>.cmds`
→ an sbatch array script in `SUBMIT_DIR/<step>.sh`. Preview by default; `SUBMIT=True`
(or `sbatch SUBMIT_DIR/<step>.sh`) to queue. Everything is reaction-agnostic — edit the
atom tokens, charge/spin, and model per system. The acceptance authority is always the
QM saddle + partial-Hessian + IRC-like gate (Steps 9–11), regardless of how the guess
was produced (scan / NEB / GSM / React-OT / AEFM).""")

# ── assemble + write ─────────────────────────────────────────────────────────
def _src(t):
    lines = t.split('\n')
    return [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

nb_cells = []
for kind, text in cells:
    c = {'cell_type': kind, 'metadata': {}, 'source': _src(text)}
    if kind == 'code':
        c['outputs'] = []; c['execution_count'] = None
    nb_cells.append(c)

notebook = {
    'cells': nb_cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    },
    'nbformat': 4, 'nbformat_minor': 5,
}
out = Path('/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/'
           'notebooks/opaa_theozyme/opaa_ts_pipeline.ipynb')
out.write_text(json.dumps(notebook, indent=1))
print(f'wrote {out}  ({len(nb_cells)} cells)')
