#!/usr/bin/env python3
"""Generate two TS-pipeline notebooks in the user's exact house style:

  notebooks/general_ts_pipeline/ts_pipeline_GENERAL.ipynb   — every method, a menu,
        entry-agnostic (start from reactant / product / ts-guess), all constraints.
  notebooks/opaa_theozyme/opaa_ts_pipeline.ipynb            — the in-order OPAA protocol
        (a subset of the general cells) with OPAA inputs pre-filled.

Driver-cell format (matches the user's PROTEIN_CHISEL template):
  banner ### TITLE ### / INPUTS / OUTPUTS / CONSTANTS / <TOOL> PARAMETERS /
  COMMAND-SUBMIT-FILE-NAMES / SANITY CHECKS / QUICK LOGIC / GENERATE COMMANDS /
  the verbatim [SETUP BATCH JOBS] block. Markdown headers: '# **STEP N: Title**'.
Nothing is hardcoded to OPAA in the general notebook; the OPAA notebook only fills inputs.
"""
import json
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════════
#  The verbatim [SETUP BATCH JOBS] block (the user's fixed template). Only the
#  value lines (@@...@@) change per cell; every comment is preserved.
# ════════════════════════════════════════════════════════════════════════════
SBATCH = """
##################################### ---- [SETUP BATCH JOBS] ---- #####################################
# ── core knobs (always set these) ────────────────────────
qtime         = '@@QTIME@@'   # NOTE: cluster MinTime is 15min — anything shorter gets bumped
cmds_per_job  = @@CPJ@@
cpus_per_task = '@@CPUS@@'          # bump for dataloaders / parallel images, etc. | DEFAULT SHOULD ALWAYS BE 1
memory        = '@@MEM@@'         # g = Gigabytes
queue         = '@@QUEUE@@'        # 'cpu' | 'gpu' | 'gpu-bf' | 'gpu-train'
job_name      = os.path.basename(commands_file_path)
submit_file   = f'{SUBMIT_DIR}{job_name}.sh'
num_jobs      = math.ceil(sum(1 for _ in open(commands_file_path, "r")) / cmds_per_job)
# ── GPU targeting (optional — only relevant if queue is a GPU partition) ──
gpu_class     = @@GPU@@        # 'small' (A4000/A5000/B4000/4000Ada) | 'large' (A6000/L40S/A100) | 'h200'
constraint    = None        # e.g. 'A4000', 'B4000|A5000', 'Blackwell', 'UW'  (None = any in class) | gpu model/gen, cpu model, location; '&'/'|' ok
exclude_nodes = None        # e.g. ['g2702']  if a node is misbehaving
# ── requeue + resilience (defaults are sensible; leave them) ───────
requeue              = True
max_restarts         = 2    # up to 3 attempts total per array task
pre_timeout_seconds  = 45   # USR1 fires 45s before walltime → graceful kill + requeue
# ── multi-GPU / MPI (leave defaults for typical single-GPU work) ──
ntasks               = 1    # MPI ranks per array task; >1 auto-adds --nodes=1
gpus_per_task        = None # None = 1 on GPU partitions, 0 on cpu | only specify to get more gpus if needed
# ── escape hatch + force-redo ─────────────────────────
extra_sbatch = None         # list[str] of raw '#SBATCH ...' lines for things this API misses # e.g. ['#SBATCH --mail-type=FAIL']
force_redo   = False        # True = wipe {logs_dir}/progress/{job_name}_* first (markers only, NOT cmd outputs)
# ── submit ──────────────────────────────────
nb.submit_array_job(commands_file_path, qtime, cpus_per_task, job_name, memory, submit_file, LOGS_DIR, num_jobs, cmds_per_job, queue,
    gpu_class=gpu_class, constraint=constraint, exclude_nodes=exclude_nodes, ntasks=ntasks, gpus_per_task=gpus_per_task,                   # GPU targeting & multi-GPU / MPI
    requeue=requeue, max_restarts=max_restarts, pre_timeout_seconds=pre_timeout_seconds, extra_sbatch=extra_sbatch, force_redo=force_redo,) # requeue / resilience & escape hatch
""".rstrip("\n")

def sbatch(qtime="12:00:00", cpj=1, cpus="8", mem="64g", queue="gpu", gpu="'small'"):
    return (SBATCH.replace("@@QTIME@@", qtime).replace("@@CPJ@@", str(cpj))
            .replace("@@CPUS@@", str(cpus)).replace("@@MEM@@", mem)
            .replace("@@QUEUE@@", queue).replace("@@GPU@@", gpu))

# ════════════════════════════════════════════════════════════════════════════
#  INIT cell — the user's exact init (verbatim imports) + container vars/helpers.
# ════════════════════════════════════════════════════════════════════════════
def init_cell(profile):
    if profile == "opaa":
        proj = "opaa_theozyme"
        sysblock = ("HOME_DIR      = '/home/woodbuse/'\n"
                    "THEOZYME_DIR  = f'{HOME_DIR}for/antonia/opaa_theozyme/'\n"
                    "SYSTEM_DIR    = THEOZYME_DIR        # the active-site / structure working dir\n")
        wd = "f'{HOME_DIR}for/antonia/opaa_theozyme'"
    else:
        proj = "my_reaction"          # EDIT
        sysblock = ("HOME_DIR      = '/home/woodbuse/'\n"
                    "SYSTEM_DIR    = f'{HOME_DIR}my_reaction/'   # EDIT: your structure / active-site working dir\n")
        wd = "SYSTEM_DIR"
    return f'''\
# ═══════════════════════════════════════════════════════════════════════════════
#  NOTEBOOK INITIALIZATION — run this cell at the start of every session
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_NAME = '{proj}'

# > USER CONFIGURATION
# >> project paths
{sysblock}
# >> manual overrides (set to None to use defaults)
_WORKING_DIR_OVERRIDE = {wd}   # default: notebook directory
_OUTPUT_DIR_OVERRIDE  = {wd}   # default: WORKING_DIR/output/

# >> subdirectories to create (each becomes an UPPERCASE <NAME>_DIR global)
WORKING_SUBDIRS = ['cmds', 'submit', 'logs', 'FINAL']
OUTPUT_SUBDIRS  = ['protomers', 'monitor', 'reaction_spec', 'relax_minimize',
                   'scan', 'path_search', 'ts_search', 'generative',
                   'refine_ts', 'ts_validation', 'dft']

# > IMPORTS
# >> standard library
import concurrent.futures, copy, glob, itertools, json, math, multiprocessing, os, operator
import random, re, shlex, shutil, statistics, string, subprocess, sys, textwrap, time, warnings
from collections import Counter, defaultdict, OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, FIRST_COMPLETED, wait
from datetime import datetime
from itertools import permutations, product, islice
from math import log10, floor
from pathlib import Path
from pprint import pprint
# >> third-party: data & math
import numpy as np, pandas as pd
from scipy.stats import gaussian_kde
# >> third-party: plotting
import matplotlib, matplotlib.pyplot as plt, seaborn as sns
from matplotlib.colors import to_rgba, ListedColormap, LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator
# >> third-party: structural biology
import pyrosetta, pyrosetta.distributed.tasks.rosetta_scripts as rosetta_scripts
from Bio import PDB
from Bio.PDB import PDBParser, PPBuilder
from Bio.SeqUtils import seq1
# >> third-party: notebook & misc
from difflib import SequenceMatcher
from IPython.display import display, HTML
# >> custom modules  (notebook_core re-exports the SLURM helpers from slurm_submission)
_NB_FUNCS_PATH = f'{{HOME_DIR}}special_scripts/notebook_functions'
if _NB_FUNCS_PATH not in sys.path:
    sys.path.insert(0, _NB_FUNCS_PATH)
import notebook_core as nb  # colors, setup_directories, print_initialization, submit_array_job, submit_cpu

# > PATHS
# >> working & output
WORKING_DIR = nb.resolve_working_dir(override=_WORKING_DIR_OVERRIDE, strip_mnt=True)
OUTPUT_DIR  = nb.resolve_output_dir(WORKING_DIR, override=_OUTPUT_DIR_OVERRIDE, strip_mnt=True)

# >> params / ligand files (OK if these don't exist yet)
PARAMS_DIR = f'{{SYSTEM_DIR}}params/'
CST_DIR    = f'{{SYSTEM_DIR}}cst_files/'

# >> tools & software
QUANTUM_COWBOY_DIR  = '/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/'
SPECIAL_SCRIPTS_DIR = f'{{HOME_DIR}}special_scripts/'
GIT_DIR             = f'{{HOME_DIR}}git/'
OBABEL_PATH         = f'{{HOME_DIR}}conda_envs/openbabel_env/bin/obabel'

# >> CONTAINERS (Cowboy Quantum Chemistry) — EDIT to your deployed sif paths.
#    MAIN_SIF has the MLFFs (MACE / MACE-OMol / ORB / AIMNet2) + xTB + the `qcb` CLI.
#    UMA_SIF holds the FairChem models (UMA / eSEN / AllScAIP). The generative MODELS
#    (React-OT proposer, AEFM refiner) each live in their own sidecar.
MAIN_SIF    = '/net/software/containers/users/woodbuse/quantum_chem/quantum_chem-20260604.sif'
UMA_SIF     = '/net/software/containers/users/woodbuse/quantum_chem/uma-20260527.sif'
REACTOT_SIF = f'{{QUANTUM_COWBOY_DIR}}containers/reactot-20260605.sif'
AEFM_SIF    = f'{{QUANTUM_COWBOY_DIR}}containers/aefm-20260605.sif'

def container_for(model):
    """Route an energy-model alias to the sif that can load it (plug-and-play models)."""
    m = (model or '').lower()
    return UMA_SIF if m.startswith(('uma', 'esen', 'allscaip')) else MAIN_SIF

def APPTAINER(sif, gpu=True):
    """apptainer exec prefix (list). --nv for GPU; binds /home + /net."""
    return ['apptainer', 'exec'] + (['--nv'] if gpu else []) + ['--bind', '/home', '--bind', '/net', sif]

def qcb_cmd(model, *args, gpu=True):
    """Full `qcb <args>` command (list) in the sif that can load `model`."""
    return [*APPTAINER(container_for(model), gpu=gpu), 'qcb', *map(str, args)]

def sidecar_cmd(sif, *args, gpu=True):
    """A generative-sidecar command: qcb isn't installed there, so run the CLI module
    with the bind-mounted repo on PYTHONPATH."""
    return [*APPTAINER(sif, gpu=gpu), 'env', f'PYTHONPATH={{QUANTUM_COWBOY_DIR}}',
            'python', '-m', 'quantum_engine.cli', *map(str, args)]

# > SETUP
for p in [WORKING_DIR, OUTPUT_DIR]:
    Path(p).mkdir(parents=True, exist_ok=True)
nb.setup_directories(WORKING_DIR, WORKING_SUBDIRS, export_globals=True, globals_dict=globals())
nb.setup_directories(OUTPUT_DIR,  OUTPUT_SUBDIRS,  export_globals=True, globals_dict=globals())
nb.set_pandas_display(all_on=True)

# > INITIALIZE
os.chdir(WORKING_DIR)
nb.print_initialization(WORKING_DIR, OUTPUT_DIR, project_name=PROJECT_NAME, obabel_path=OBABEL_PATH, globals_dict=globals(), preview=True)
for _n, _s in [('MAIN', MAIN_SIF), ('UMA', UMA_SIF), ('REACTOT', REACTOT_SIF), ('AEFM', AEFM_SIF)]:
    print(f'  CONTAINER {{_n:8}} {{"OK " if Path(_s).exists() else "MISSING"}} {{_s}}')
'''

# bundle one (markdown, code) pair builder
def cells_init(profile):
    return [("markdown", "# **NOTEBOOK INITIALIZATION**"), ("code", init_cell(profile))]


# ════════════════════════════════════════════════════════════════════════════
#  Profile-specific input defaults
# ════════════════════════════════════════════════════════════════════════════
def defaults(profile):
    if profile == "opaa":
        return dict(
            unprot='f"{THEOZYME_DIR}opaa_3l7g_optimal_maximal_theozyme_pxn_unprotonated.pdb"',
            prot='f"{PROTOMERS_DIR}opaa_3l7g_optimal_maximal_theozyme_pxn.pdb"',
            model="'mace-polar-m'", head="None", charge="0", spin="1",
            # SN2-at-P: O_nuc(hydroxide)..P forming, P..O_lg breaking. EDIT serials per structure.
            f_nuc="'serial:1872'", f_elec="'serial:1850'", b_p="'serial:1850'", b_lg="'serial:1860'",
            scan_nuc="1871", scan_p="1849",   # 0-based = serial-1 (for qcb scan / fix-bond)
            react_serials="1872, 1850, 1860",   # 1-based serials (for refine-ts/validate-ts)
            ptm='{}',              # OPAA construct: NO post-translational modification (no KCX)
            ligand_charges='{}',   # net charge of the protonated system = 0 (set on --charge)
        )
    return dict(
        unprot='f"{SYSTEM_DIR}structure_unprotonated.pdb"   # EDIT',
        prot='f"{PROTOMERS_DIR}structure.pdb"               # EDIT',
        model="'mace-polar-m'", head="None", charge="0", spin="1",
        f_nuc="'serial:NUC'", f_elec="'serial:ELEC'", b_p="'serial:ELEC'", b_lg="'serial:LG'",
        scan_nuc="0", scan_p="1",
        react_serials="1, 2, 3",
        ptm='{}', ligand_charges='{}',
    )


# Shared ENERGY MODEL guidance block (the plethora of plug-and-play models).
MODEL_MENU = '''\
### ENERGY MODEL (plug-and-play: change `model`/`head`; container_for() picks the sif) ###
# Charged METAL active sites (e.g. di-Zn) — use a CHARGE-AWARE model:
#   mace-polar-m : polarizable + long-range electrostatics — IDEAL for a charged metal
#                  pocket (DEFAULT; baked into MAIN_SIF, loads in-process). sizes -s|-m|-l
#   mace-omol    : wB97M-V/OMol25, charge-aware, highest accuracy (large GPU: A6000/H200)
#   mace-mh-1    : multi-head foundation model -> set head='omol' for the OMol25 head
#   orb-mol-conservative : Orbital-Materials, charge/spin-aware, Zn-capable (true gradients)
#   uma-m-1p1 / esen-sm-conserving / allscaip-md-conserving : FairChem (route to UMA_SIF)
# Organic-only (NO metals): mace-off-* (wB97M organic) ; aimnet2-rxn (CHON, TS-tuned).
# AVOID GFN2-xTB on metals (not charge-aware there). --head applies to MACE multi-head only.'''


# ════════════════════════════════════════════════════════════════════════════
#  STEP cells (each returns (markdown, code)). Profile only changes input values.
# ════════════════════════════════════════════════════════════════════════════

def cell_protonate(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: Protonate / Generate Protomers**"
    code = f'''\
##################################################################
###          PROTONATE STRUCTURE   (qcb protonator v2)         ###
##################################################################
# Deterministic, staged protonation of the protein in a PDB/CIF.
#   Stage 1  cap open backbone N/C termini (geometry-detected)
#   Stage 2  PTM / covalent safety checks (warnings only)
#   Stage 3  pH-aware canonical protonation
#   Stage 4  histidine tautomer (metal / clash / H-bond geometry)
#   Stage 5  propka refinement of ambiguous states
#   Stage 6  optional multi-protomer output
#   + optional cheap CPU MLFF relax of ONLY the new hydrogens
# HETATM (ligand/metal/water) is assumed already protonated and left alone.
# This cell only PRINTS the command — copy-paste it into your terminal (no sbatch).

print_commands = True

### INPUTS ###
input_pdb = {D['unprot']}

### OUTPUTS ###
# With protomers > 1 this is the BASE name -> <base>_protomer1..N.pdb
output_pdb = {D['prot']}

### CONSTANTS ###
CONTAINER  = MAIN_SIF       # the qcb / protonator container (defined in INIT)
PROTONATOR = f"{{QUANTUM_COWBOY_DIR}}quantum_engine/prep/protonator.py"
# (`qcb protonate <same args>` is equivalent to running this file directly.)

### PROTONATOR PARAMETERS ###
# --- core ----------------------------------------------------------
pH        = 7.5                # target pH for canonical assignment + propka
protomers = 1                  # 1 = single best state; N>1 = N most likely protomers
# --- terminus capping ----------------------------------------------
# N-cap: nh2 (default) | nh3+ | nme | nfo | none     C-cap: cho | coo- | cooh | conh2 | conhme | none
n_cap = "nh2"
c_cap = "cho"
n_cap_overrides = {{}}            # "CHAIN:RESID": captype
c_cap_overrides = {{}}
# --- hard protonation overrides (always win over canonical + propka) ---
# "CHAIN:RESID": STATE  (HID HIE HIP ASP ASH GLU GLH LYS LYN CYS CYM CYX TYR TYM ARG)
set_overrides = {{}}
# --- PTMs (you MUST declare them; residue is then frozen) ----------
ptm        = {D['ptm']}            # "CHAIN:RESID": CODE (KCX SEP TPO PTR ...)
ptm_charge = {{}}                  # "CHAIN:RESID": int (overrides default charge)
# --- ligand ---------------------------------------------------------
protonate_ligands = False      # stub (HETATM assumed already protonated)
ligand_charges    = {D['ligand_charges']}   # "RESNAME": charge (for total-charge reporting)
# --- protomer tuning (only when protomers > 1) ---------------------
protomer_min_prob     = 0.15
protomer_max_variable = 12
couples = []                   # [("CHAIN:RESID","CHAIN:RESID"), ...] vary in lockstep
# --- optional MLFF relax of ONLY the new H (CPU, charge-free) -------
relax_h        = True
relax_h_model  = "mace-off-small"   # auto-falls back to mace-mp if metals present
relax_h_fmax, relax_h_steps, relax_h_device = 0.05, 200, "cpu"
# --- misc -----------------------------------------------------------
skip_propka          = False
keep_input_hydrogens = False
output_info_file     = None    # e.g. f"{{PROTOMERS_DIR}}protonation_info.json"
log_level            = "DEBUG"
# --- geometry cutoffs (advanced; defaults are sensible) ------------
bond_cutoff, metal_coord_cutoff, hbond_cutoff = 1.8, 2.8, 3.5
clash_cutoff, h_clash_cutoff, disulfide_cutoff = 2.0, 1.5, 2.5

### SANITY CHECKS ###
if not Path(input_pdb).is_file():
    raise FileNotFoundError(f"input_pdb not found: {{input_pdb}}")
Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)

### BUILD THE COMMAND ###
cmd  = [*APPTAINER(CONTAINER, gpu=False), "python", PROTONATOR, "--input-pdb", input_pdb]
if output_pdb:                 cmd += ["--output-pdb", output_pdb]
cmd += ["--pH", str(pH), "--protomers", str(protomers), "--n-cap", n_cap, "--c-cap", c_cap]
for k, v in n_cap_overrides.items(): cmd += ["--n-cap-override", f"{{k}}={{v}}"]
for k, v in c_cap_overrides.items(): cmd += ["--c-cap-override", f"{{k}}={{v}}"]
for k, v in set_overrides.items():   cmd += ["--set", f"{{k}}={{v}}"]
for k, v in ptm.items():             cmd += ["--ptm", f"{{k}}={{v}}"]
for k, v in ptm_charge.items():      cmd += ["--ptm-charge", f"{{k}}={{v}}"]
for k, v in ligand_charges.items():  cmd += ["--ligand-charge", f"{{k}}={{v}}"]
if protonate_ligands: cmd += ["--protonate-ligands"]
if protomers > 1:
    cmd += ["--protomer-min-prob", str(protomer_min_prob), "--protomer-max-variable", str(protomer_max_variable)]
    for a, b in couples: cmd += ["--couple", f"{{a}},{{b}}"]
if relax_h:
    cmd += ["--relax-h", "--relax-h-model", relax_h_model, "--relax-h-fmax", str(relax_h_fmax),
            "--relax-h-steps", str(relax_h_steps), "--relax-h-device", relax_h_device]
if skip_propka:          cmd += ["--skip-propka"]
if keep_input_hydrogens: cmd += ["--keep-input-hydrogens"]
if output_info_file:     cmd += ["--output-info-file", output_info_file]
cmd += ["--bond-cutoff", str(bond_cutoff), "--metal-coord-cutoff", str(metal_coord_cutoff),
        "--hbond-cutoff", str(hbond_cutoff), "--clash-cutoff", str(clash_cutoff),
        "--h-clash-cutoff", str(h_clash_cutoff), "--disulfide-cutoff", str(disulfide_cutoff)]
cmd += ["--log-level", log_level]

### PRINT COMMAND ###
if print_commands:
    print("### CONSTRUCTED COMMAND (copy-paste into terminal) ###")
    print(" ".join(str(x) for x in cmd))
    _out = output_pdb if output_pdb else input_pdb.rsplit(".", 1)[0] + "_protonated.pdb"
    if protomers > 1:
        print(f"\\n# Output: {{_out.rsplit('.pdb',1)[0]}}_protomer1..{{protomers}}.pdb")
    else:
        print(f"\\n# Output: {{_out}}")
'''
    return [("markdown", md), ("code", code)]


def cell_monitor(profile, step):
    md = f"# **STEP {step}: Monitor Active Site (bond / metal coordination)**"
    code = f'''\
##################################################################
###          MONITOR ACTIVE SITE   (qcb monitor)              ###
##################################################################
# Non-constraining sanity report: measured key bonds + auto-detected metal
# coordination shells. Confirm the protonated geometry before spending GPU time.
# Instant CPU — this cell PRINTS the command (copy-paste; no sbatch).

print_commands = True

### INPUTS ###
input_pdb = {defaults(profile)['prot']}

### OUTPUTS ###
out_dir = MONITOR_DIR

### MONITOR PARAMETERS ###
# 0-based atom-index pairs to measure (e.g. forming/breaking bonds); [] for none.
monitor_bond_pairs = []        # e.g. [(1849, 1871)]   (0-based ASE indices)
report_metals      = True      # auto-detect metals + their coordination shells

### CONSTANTS ###
CONTAINER = MAIN_SIF

### BUILD THE COMMAND ###
cmd = [*APPTAINER(CONTAINER, gpu=False), "qcb", "monitor", input_pdb, "--outdir", out_dir]
for i, j in monitor_bond_pairs: cmd += ["--bond", f"{{i}},{{j}}"]
if report_metals: cmd += ["--metals"]

### PRINT COMMAND ###
if print_commands:
    print("### CONSTRUCTED COMMAND (copy-paste into terminal) ###")
    print(" ".join(str(x) for x in cmd))
    print(f"\\n# JSON report → {{out_dir}}")
'''
    return [("markdown", md), ("code", code)]


def cell_reaction_spec(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: Reaction Spec (clean variables → auto-generated YAML)**"
    code = f'''\
##################################################################
###     REACTION SPEC   (define the chemistry, autogen YAML)   ###
##################################################################
# You set clean Python variables here; the cell BUILDS the ReactionSpec YAML for you
# (there is no hardcoded YAML blob to edit). The spec declares WHAT reacts:
# forming/breaking bonds, the reactive atoms (imag-mode overlap set), and an optional
# 1-D collective variable (cv) used by scans + the reactant-only entry.
#
# ATOM TOKENS: "serial:N" (1-based PDB serial, RECOMMENDED), "CHAIN:RESID:NAME"
#              (e.g. "A:169:NZ"), or a bare 0-based ASE index. Use one style consistently.
# IMPORTANT: charge & multiplicity are NOT read from this YAML — always pass --charge/--multiplicity on
#            every qcb command (the YAML keys would be silently ignored).

print_commands = True

### INPUTS ###
struct_pdb = {D['prot']}        # used only to resolve atom tokens during validation

### OUTPUTS ###
spec_path = f"{{REACTION_SPEC_DIR}}reaction_spec.yaml"

### REACTION DEFINITION ###
# (nucleophile, electrophile) pairs that FORM; (atomA, atomB) pairs that BREAK.
forming_bonds  = [({D['f_nuc']}, {D['f_elec']})]     # e.g. O_nuc -> P
breaking_bonds = [({D['b_p']}, {D['b_lg']})]     # e.g. P -> O_leaving
reactive_atoms = [{D['f_nuc']}, {D['f_elec']}, {D['b_lg']}]   # atoms on the imaginary mode
# Optional 1-D collective variable (bond_difference: s = d(a,b) - d(a,c)); set cv_kind=None to omit.
cv_kind  = "bond_difference"
cv_atoms = [{D['f_nuc']}, {D['f_elec']}, {D['b_p']}, {D['b_lg']}]   # [a, b, a, c]
# Optional reactant->product atom map for double-ended methods ({{}} = identical ordering).
atom_map = {{}}

### CONSTANTS ###
CONTAINER = MAIN_SIF

### GENERATE YAML ###
def _tok(t):  return str(t)
def _bond(b): return f"  - [{{_tok(b[0])}}, {{_tok(b[1])}}]"
_lines = ["# auto-generated ReactionSpec (charge/spin live on the CLI, not here)"]
_lines += ["forming_bonds:"]  + [_bond(b) for b in forming_bonds]
_lines += ["breaking_bonds:"] + [_bond(b) for b in breaking_bonds]
_lines += ["reactive_atoms:"] + [f"  - {{_tok(a)}}" for a in reactive_atoms]
if cv_kind:
    _lines += ["cv:", f"  kind: {{cv_kind}}", f"  atoms: [{{', '.join(_tok(a) for a in cv_atoms)}}]"]
if atom_map:
    _lines += ["atom_map:"] + [f"  {{k}}: {{v}}" for k, v in atom_map.items()]
spec_yaml = "\\n".join(_lines) + "\\n"
Path(spec_path).parent.mkdir(parents=True, exist_ok=True)
Path(spec_path).write_text(spec_yaml)
print(f"# wrote {{spec_path}}\\n"); print(spec_yaml)

### BUILD + PRINT VALIDATION COMMAND ###
cmd = [*APPTAINER(CONTAINER, gpu=False), "qcb", "reaction-spec", spec_path, "--structure", struct_pdb]
if print_commands:
    print("### VALIDATE (copy-paste into terminal) ###")
    print(" ".join(str(x) for x in cmd))
'''
    return [("markdown", md), ("code", code)]


def cell_relax(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: Minimize / Relax a Geometry (`qcb opt`)**"
    code = f'''\
##################################################################
###     MINIMIZE / RELAX   (qcb opt; constrained or not)       ###
##################################################################
# Relax ANY input geometry: a reactant, a product, a TS-region pose, or the whole
# protonated cluster. The constraint regime is the key knob:
#   * unconstrained   (fix_preset='none')      -> a true minimum (final R / P endpoints)
#   * CA-frozen        (fix_preset='ca-only')  -> scaffold the backbone, let chemistry breathe
#   * bond-pinned      (fix_bonds=[...])        -> hold the forming/breaking distance(s) while
#                                                  everything else relaxes (constrained TS-region min)
# fix_bond / restrain_bond use 0-based ASE indices (= PDB serial - 1).

### INPUTS ###
input_pdb = {D['prot']}        # the geometry to relax (reactant / product / ts-region / cluster)

### OUTPUTS ###
out_dir     = f"{{RELAX_MINIMIZE_DIR}}relax/"
relaxed_pdb = f"{{out_dir}}relaxed.pdb"

{MODEL_MENU}
model  = {D['model']}
head   = {D['head']}            # MACE multi-head only (e.g. 'omol' for mace-mh-1); None for polar/omol
device = "cuda"
charge       = {D['charge']}        # FULL-cluster net charge (CLI-only; the spec YAML ignores charge/multiplicity)
multiplicity = {D['spin']}        # spin MULTIPLICITY M=2S+1 (1=singlet/no radicals, 2=doublet, 3=triplet)
                                    # NOTE: the CLI flag is --multiplicity. (--spin takes S and converts to 2S+1.)

### CONSTRAINTS ###
fix_preset = "ca-only"          # 'none' | 'ca-only' | 'backbone' | 'backbone-water'
extra_fix  = []                 # extra select specs, e.g. ['residue HOH', 'chain B', 'resid 169']
extra_free = []                 # subtract from the preset, e.g. ['atoms ZN1 ZN2']
fix_bonds      = []             # hard-pin: [[i, j]] or [[i, j, R0]]  (0-based ASE idx)  e.g. [[{D['scan_p']}, {D['scan_nuc']}]]
restrain_bonds = []             # harmonic: [[i, j, K, R0]]  (0-based; K in eV/A^2)

### OPT PARAMETERS ###
optimizer = "lbfgs"             # lbfgs | bfgs | fire
fmax      = 0.05                # eV/A (0.05 is fine for MLFFs)
max_steps = 500

### COMMAND / SUBMIT FILE NAMES ###
commands_name      = f"{{PROJECT_NAME}}_relax"
commands_file_path = os.path.join(CMDS_DIR, commands_name)

### SANITY CHECKS ###
if not Path(input_pdb).is_file():
    raise FileNotFoundError(f"input_pdb not found: {{input_pdb}}")
Path(out_dir).mkdir(parents=True, exist_ok=True)

### GENERATE COMMANDS ###
commands = []
cmd  = qcb_cmd(model, "opt", input_pdb, "--model", model, "--charge", charge, "--multiplicity", multiplicity, "--device", device)
if head: cmd += ["--head", head]
cmd += ["--fix-preset", fix_preset]
for s in extra_fix:  cmd += ["--fix", s]
for s in extra_free: cmd += ["--free", s]
for b in fix_bonds:      cmd += ["--fix-bond", *map(str, b)]
for b in restrain_bonds: cmd += ["--restrain-bond", *map(str, b)]
cmd += ["--optimizer", optimizer, "--fmax", fmax, "--max-steps", max_steps,
        "--outdir", out_dir, "--output-pdb", relaxed_pdb]
commands.append(" ".join(str(x) for x in cmd))
with open(commands_file_path, "w") as f:
    f.write("\\n".join(commands) + "\\n")
print(f"# {{len(commands)}} command(s) → {{commands_file_path}}")
for c in commands: print("\\n" + c)
print(f"\\n# Output: {{relaxed_pdb}}")
{sbatch(qtime="12:00:00", cpus="8", mem="64g", queue="gpu", gpu="'small'")}
'''
    return [("markdown", md), ("code", code)]


def cell_scan(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: 1-D Relaxed Scan → TS Guess + Endpoints (`qcb scan`)**"
    code = f'''\
##################################################################
###     1-D RELAXED SCAN   (qcb scan; bond / angle / dihedral) ###
##################################################################
# Slide one internal coordinate (a forming/breaking BOND for SN2-like steps) and
# relax everything else at each step. The scan gives you, in one shot:
#   * an approximate reactant basin (one end of the scan)
#   * an approximate product basin  (other end)
#   * an approximate TS guess        (highest-energy frame)  -> feeds Step refine-ts
# qcb scan INDICES are 0-based ASE indices (= PDB serial - 1).

### INPUTS ###
relaxed_pdb = f"{{RELAX_MINIMIZE_DIR}}relax/relaxed.pdb"   # usually the CA-frozen relaxed cluster

### OUTPUTS ###
# A 1-D relaxed scan IS a (single-ended, 1-coordinate) PATH SEARCH: it yields a TS
# GUESS (highest-E frame) AND approximate reactant/product endpoints (the two ends).
out_dir           = f"{{SCAN_DIR}}scan/"
ts_guess_pdb      = f"{{out_dir}}ts_guess.pdb"        # max-energy frame -> Step refine-ts
reactant_scan_pdb = f"{{out_dir}}reactant_scan.pdb"  # first frame (approx reactant) -> minimize next
product_scan_pdb  = f"{{out_dir}}product_scan.pdb"   # last frame  (approx product)  -> minimize next

### ENERGY MODEL ###
model, head, device = {D['model']}, {D['head']}, "cuda"   # default mace-polar-m (see relax cell's menu)
charge, multiplicity = {D['charge']}, {D['spin']}                 # net charge (CLI-only); multiplicity M=2S+1 (1=singlet)

### CONSTRAINTS ###
fix_preset = "ca-only"          # the scanned bond is auto-pinned ON TOP of this preset

### SCAN PARAMETERS ###
scan_indices = [{D['scan_nuc']}, {D['scan_p']}]    # 0-based ASE indices of the scanned coordinate (e.g. O_nuc..P)
scan_coord   = "bond"           # bond | angle | dihedral
scan_start   = 1.6              # A  (TS-like end of a forming bond)
scan_end     = 3.0              # A  (well past bond-broken)
scan_n_steps = 16
scan_fmax    = 0.05

### CONSTANTS ###
template_pdb = relaxed_pdb      # residue-annotation template for the extracted PDB

### COMMAND / SUBMIT FILE NAMES ###
commands_name      = f"{{PROJECT_NAME}}_scan"
commands_file_path = os.path.join(CMDS_DIR, commands_name)

### SANITY CHECKS ###
if not Path(relaxed_pdb).is_file():
    raise FileNotFoundError(f"relaxed_pdb not found: {{relaxed_pdb}}  (run the relax step first)")
Path(out_dir).mkdir(parents=True, exist_ok=True)

### GENERATE COMMANDS ###
# 1) the scan; 2) a tiny helper that extracts the max-energy frame as a TS-guess PDB.
commands = []
cmd_scan  = qcb_cmd(model, "scan", relaxed_pdb, "--model", model, "--charge", charge,
                    "--multiplicity", multiplicity, "--device", device, "--fix-preset", fix_preset,
                    "--coord", scan_coord, "--indices", *scan_indices,
                    "--start", scan_start, "--end", scan_end, "--n-steps", scan_n_steps,
                    "--fmax", scan_fmax, "--outdir", out_dir)
extract_py = f"{{out_dir}}extract_frames.py"
Path(extract_py).write_text(textwrap.dedent(f"""\\
    import ase.io as io
    from quantum_engine.io import load_structure, write_pdb
    frames = io.read(r'{{out_dir}}scan-trajectory.xyz', index=':')
    e = lambda a: a.info.get('energy_eV', a.get_potential_energy())
    i = max(range(len(frames)), key=lambda k: e(frames[k]))
    _, bt, _ = load_structure(r'{{template_pdb}}')
    write_pdb(frames[0],  bt, r'{{reactant_scan_pdb}}', total_charge={{charge}})
    write_pdb(frames[-1], bt, r'{{product_scan_pdb}}',  total_charge={{charge}})
    write_pdb(frames[i],  bt, r'{{ts_guess_pdb}}',      total_charge={{charge}})
    print(f'TS guess = frame {{{{i}}}}/{{{{len(frames)}}}} ; reactant=frame 0 ; product=frame {{{{len(frames)-1}}}}')
"""))
cmd_extract = [*APPTAINER(container_for(model)), "python", extract_py]
commands.append(" ".join(str(x) for x in cmd_scan))
commands.append(" ".join(str(x) for x in cmd_extract))
with open(commands_file_path, "w") as f:
    f.write("\\n".join(commands) + "\\n")
print(f"# {{len(commands)}} command(s) → {{commands_file_path}}")
for c in commands: print("\\n" + c)
print(f"\\n# Outputs: {{out_dir}}scan-trajectory.xyz, scan-summary.json, scan.png")
print(f"#          TS guess: {{ts_guess_pdb}} ; endpoints: {{reactant_scan_pdb}}, {{product_scan_pdb}}")
{sbatch(qtime="24:00:00", cpj=2, cpus="8", mem="64g", queue="gpu", gpu="'small'")}
'''
    return [("markdown", md), ("code", code)]


def cell_min_endpoints(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: Minimize the Reactant & Product Endpoints (`qcb opt`)**"
    code = f'''\
##################################################################
###  MINIMIZE ENDPOINTS  (scan ends -> TRUE reactant / product) ###
##################################################################
# WHY: the barrier is E(TS) - E(reactant_min); you must compare two TRUE stationary
# points. The scan's first/last frames are constraint-biased approximations, so relax
# each to a real minimum. Use the SAME CA-frozen scaffold + model/charge/spin as the TS
# so the energies are comparable. Reactive bonds are FREE here (no --fix-bond) — these are
# basins, not the TS. These two minima are also what verify-irc-like should reproduce.

### INPUTS ###
reactant_scan_pdb = f"{{SCAN_DIR}}scan/reactant_scan.pdb"   # first scan frame
product_scan_pdb  = f"{{SCAN_DIR}}scan/product_scan.pdb"    # last  scan frame

### OUTPUTS ###
out_dir = f"{{RELAX_MINIMIZE_DIR}}endpoints/"
R_min   = f"{{out_dir}}reactant_min.pdb"
P_min   = f"{{out_dir}}product_min.pdb"

### ENERGY MODEL ###
model, head, device = {D['model']}, {D['head']}, "cuda"   # default mace-polar-m (see relax cell's menu)
charge, multiplicity = {D['charge']}, {D['spin']}                 # net charge (CLI-only); multiplicity M=2S+1 (1=singlet)

### CONSTRAINTS ###
fix_preset = "ca-only"          # same scaffold as the TS; reactive bonds FREE (no pin)

### OPT PARAMETERS ###
optimizer, fmax, max_steps = "lbfgs", 0.03, 500   # tighter fmax for clean basins

### COMMAND / SUBMIT FILE NAMES ###
commands_name      = f"{{PROJECT_NAME}}_min_endpoints"
commands_file_path = os.path.join(CMDS_DIR, commands_name)

### SANITY CHECKS ###
for p in (reactant_scan_pdb, product_scan_pdb):
    if not Path(p).is_file():
        raise FileNotFoundError(f"scan endpoint not found: {{p}}  (run the scan step first)")
Path(out_dir).mkdir(parents=True, exist_ok=True)

### GENERATE COMMANDS ###
commands = []
for _src, _dst in [(reactant_scan_pdb, R_min), (product_scan_pdb, P_min)]:
    cmd = qcb_cmd(model, "opt", _src, "--model", model, "--charge", charge, "--multiplicity", multiplicity,
                  "--device", device, "--fix-preset", fix_preset, "--optimizer", optimizer,
                  "--fmax", fmax, "--max-steps", max_steps, "--outdir", out_dir, "--output-pdb", _dst)
    if head: cmd += ["--head", head]
    commands.append(" ".join(str(x) for x in cmd))
with open(commands_file_path, "w") as f:
    f.write("\\n".join(commands) + "\\n")
print(f"# {{len(commands)}} command(s) → {{commands_file_path}}")
for c in commands: print("\\n" + c)
print(f"\\n# Outputs: {{R_min}} , {{P_min}}  (barrier = E(TS) - E(reactant_min))")
{sbatch(qtime="12:00:00", cpj=2, cpus="8", mem="64g", queue="gpu", gpu="'small'")}
'''
    return [("markdown", md), ("code", code)]


def cell_neb_opaa(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: *(optional, more rigorous)* CI-NEB between the minimized endpoints (`qcb neb`)**"
    code = f'''\
##################################################################
###  OPTIONAL: DOUBLE-ENDED CI-NEB  (more rigorous than 1-D scan) ###
##################################################################
# OPTIONAL alternative path search. A 1-D scan can slice BESIDE the true saddle for an
# ASYNCHRONOUS SN2-at-P (P-O_nuc and P-O_lg not changing in lockstep). A double-ended
# geodesic CI-NEB between the two MINIMIZED endpoints relaxes all orthogonal DOFs at every
# image, so its climbing image is a better guess. Feed it to refine-ts via --from-neb.
# (If your scan peak already refines to a clean 1-imag-mode saddle, you can skip this.)

### INPUTS ###
reactant_min = f"{{RELAX_MINIMIZE_DIR}}endpoints/reactant_min.pdb"
product_min  = f"{{RELAX_MINIMIZE_DIR}}endpoints/product_min.pdb"

### OUTPUTS ###
out_dir = f"{{PATH_SEARCH_DIR}}neb/"

### ENERGY MODEL ###
model, head, device = {D['model']}, {D['head']}, "cuda"
charge, multiplicity = {D['charge']}, {D['spin']}

### NEB PARAMETERS ###
fix_preset    = "ca-only"
n_images      = 17              # publication-tier; 11 for a quicker pass
interpolation = "geodesic"      # REQUIRED for dense/charged sites (never 'linear')
optimizer     = "fire"

### COMMAND / SUBMIT FILE NAMES ###
commands_name      = f"{{PROJECT_NAME}}_neb"
commands_file_path = os.path.join(CMDS_DIR, commands_name)

### SANITY CHECKS ###
for p in (reactant_min, product_min):
    if not Path(p).is_file():
        raise FileNotFoundError(f"minimized endpoint not found: {{p}}  (run min-endpoints first)")
Path(out_dir).mkdir(parents=True, exist_ok=True)

### GENERATE COMMANDS ###
commands = []
cmd = qcb_cmd(model, "neb", reactant_min, product_min, "--model", model, "--charge", charge,
              "--multiplicity", multiplicity, "--device", device, "--fix-preset", fix_preset, "--n-images", n_images,
              "--interpolation", interpolation, "--optimizer", optimizer, "--outdir", out_dir)
if head: cmd += ["--head", head]
commands.append(" ".join(str(x) for x in cmd))
with open(commands_file_path, "w") as f:
    f.write("\\n".join(commands) + "\\n")
print(f"# {{len(commands)}} command(s) → {{commands_file_path}}")
for c in commands: print("\\n" + c)
print(f"\\n# Output dir: {{out_dir}}  → in refine-ts set  from_neb = '{{out_dir}}'")
{sbatch(qtime="24:00:00", cpus="8", mem="80g", queue="gpu", gpu="'large'")}
'''
    return [("markdown", md), ("code", code)]


def cell_neb(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: Double-Ended Path Search — CI-NEB / AutoNEB (`qcb neb`)**"
    code = f'''\
##################################################################
###     DOUBLE-ENDED PATH SEARCH   (qcb neb; needs R AND P)    ###
##################################################################
# Climbing-image NEB between a reactant and product (same atom ordering, or supply an
# atom_map in the spec). Geodesic interpolation is the default and STRONGLY recommended
# for dense/charged sites (linear interpolation can blow energies up). The climbing
# image is your TS guess; refine-ts reads it via  --from-neb {{out_dir}}.

### INPUTS ###
reactant_pdb = f"{{RELAX_MINIMIZE_DIR}}relax/reactant.pdb"   # EDIT  (relaxed reactant basin)
product_pdb  = f"{{RELAX_MINIMIZE_DIR}}relax/product.pdb"    # EDIT  (relaxed product basin)

### OUTPUTS ###
out_dir = f"{{PATH_SEARCH_DIR}}neb/"

### ENERGY MODEL ###
model, head, device = {D['model']}, {D['head']}, "cuda"
charge, multiplicity = {D['charge']}, {D['spin']}

### CONSTRAINTS ###
fix_preset = "ca-only"

### NEB PARAMETERS ###
n_images      = 11              # 7-9 small, 13-17 multi-bond, 17-25 floppy/metal
interpolation = "geodesic"      # geodesic (recommended) | idpp | linear
optimizer     = "fire"          # fire | lbfgs | mdmin | bfgs | ode
key_bonds     = []              # [("serial:A","serial:B"), ...] bond-aware kink detection (optional)

### CONSTANTS ###
CONTAINER = container_for(model)

### COMMAND / SUBMIT FILE NAMES ###
commands_name      = f"{{PROJECT_NAME}}_neb"
commands_file_path = os.path.join(CMDS_DIR, commands_name)

### SANITY CHECKS ###
for p in (reactant_pdb, product_pdb):
    if not Path(p).is_file():
        raise FileNotFoundError(f"endpoint not found: {{p}}")
Path(out_dir).mkdir(parents=True, exist_ok=True)

### GENERATE COMMANDS ###
commands = []
cmd  = qcb_cmd(model, "neb", reactant_pdb, product_pdb, "--model", model, "--charge", charge,
               "--multiplicity", multiplicity, "--device", device, "--fix-preset", fix_preset,
               "--n-images", n_images, "--interpolation", interpolation, "--optimizer", optimizer,
               "--outdir", out_dir)
if head: cmd += ["--head", head]
for a, b in key_bonds: cmd += ["--key-bond", f"{{a}},{{b}}"]
commands.append(" ".join(str(x) for x in cmd))
with open(commands_file_path, "w") as f:
    f.write("\\n".join(commands) + "\\n")
print(f"# {{len(commands)}} command(s) → {{commands_file_path}}")
for c in commands: print("\\n" + c)
print(f"\\n# Output dir: {{out_dir}}  → feed refine-ts with  --from-neb {{out_dir}}")
{sbatch(qtime="24:00:00", cpus="8", mem="64g", queue="gpu", gpu="'large'")}
'''
    return [("markdown", md), ("code", code)]


def cell_gsm(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: Double-Ended String — GSM / FSM (`qcb gsm`)**"
    code = f'''\
##################################################################
###     GROWING / FREEZING STRING   (qcb gsm; needs R AND P)   ###
##################################################################
# String methods: cheaper than NEB, good TS guesses. GSM = Growing String,
# FSM = Freezing String. NOTE: qcb gsm has NO --multiplicity/--spin and NO --fix-preset (string
# methods don't take ASE constraints). For single-ended (reactant + driving coords)
# use ts-entry --path-method gsm-se instead (see the single-ended step).

### INPUTS ###
reactant_pdb = f"{{RELAX_MINIMIZE_DIR}}relax/reactant.pdb"   # EDIT
product_pdb  = f"{{RELAX_MINIMIZE_DIR}}relax/product.pdb"    # EDIT

### OUTPUTS ###
out_dir = f"{{PATH_SEARCH_DIR}}gsm/"

### ENERGY MODEL ###
model, head, device = {D['model']}, {D['head']}, "cuda"
charge = {D['charge']}

### GSM PARAMETERS ###
gsm_method = "gsm"              # 'gsm' (Growing String) | 'fsm' (Freezing String)
n_images   = 15
fmax       = 0.05

### COMMAND / SUBMIT FILE NAMES ###
commands_name      = f"{{PROJECT_NAME}}_gsm"
commands_file_path = os.path.join(CMDS_DIR, commands_name)

### SANITY CHECKS ###
for p in (reactant_pdb, product_pdb):
    if not Path(p).is_file():
        raise FileNotFoundError(f"endpoint not found: {{p}}")
Path(out_dir).mkdir(parents=True, exist_ok=True)

### GENERATE COMMANDS ###
commands = []
cmd  = qcb_cmd(model, "gsm", reactant_pdb, product_pdb, "--method", gsm_method, "--model", model,
               "--charge", charge, "--device", device, "--n-images", n_images, "--fmax", fmax,
               "--outdir", out_dir)
if head: cmd += ["--head", head]
commands.append(" ".join(str(x) for x in cmd))
with open(commands_file_path, "w") as f:
    f.write("\\n".join(commands) + "\\n")
print(f"# {{len(commands)}} command(s) → {{commands_file_path}}")
for c in commands: print("\\n" + c)
print(f"\\n# Output dir: {{out_dir}}")
{sbatch(qtime="24:00:00", cpus="8", mem="64g", queue="gpu", gpu="'large'")}
'''
    return [("markdown", md), ("code", code)]


def cell_single_ended(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: Single-Ended Search — reactant-only / SE-GSM (`qcb ts-entry`)**"
    code = f'''\
##################################################################
###  SINGLE-ENDED / ONE-DIRECTIONAL SEARCH (no product needed) ###
##################################################################
# When you have ONLY a reactant (no product geometry):
#   * mode='reactant-only' : ts-entry drives the reactant along the spec's `cv`
#                            bond-difference to synthesize a product, then runs the
#                            usual double-ended core. REQUIRES a cv in the reaction spec.
#   * mode='gsm-se'        : SE-GSM grows a string from the reactant along driving
#                            coordinates (set in the spec's driving_coords). Finicky;
#                            fails cleanly with a status.
# This runs the WHOLE pipeline (guess -> refine -> Hessian -> IRC gate).

### INPUTS ###
reactant_pdb = f"{{RELAX_MINIMIZE_DIR}}relax/relaxed.pdb"   # EDIT  (relaxed reactant)
spec_path    = f"{{REACTION_SPEC_DIR}}reaction_spec.yaml"

### OUTPUTS ###
out_dir = f"{{TS_SEARCH_DIR}}single_ended/"

### ENERGY MODEL ###
model, head, device = {D['model']}, {D['head']}, "cuda"
charge, multiplicity = {D['charge']}, {D['spin']}

### SINGLE-ENDED PARAMETERS ###
mode         = "reactant-only"  # 'reactant-only' (needs cv) | 'gsm-se' (needs driving_coords)
rigor        = "standard"       # draft | standard | publication
cv_product_s = None             # reactant-only: explicit product-side CV target s (A); None = auto

### COMMAND / SUBMIT FILE NAMES ###
commands_name      = f"{{PROJECT_NAME}}_single_ended"
commands_file_path = os.path.join(CMDS_DIR, commands_name)

### SANITY CHECKS ###
if not Path(reactant_pdb).is_file():
    raise FileNotFoundError(f"reactant_pdb not found: {{reactant_pdb}}")
if not Path(spec_path).is_file():
    raise FileNotFoundError(f"reaction spec not found: {{spec_path}}  (run the reaction-spec step)")
Path(out_dir).mkdir(parents=True, exist_ok=True)

### GENERATE COMMANDS ###
commands = []
cmd = qcb_cmd(model, "ts-entry", "--entry", "reactant-only", "--reaction-spec", spec_path,
              "--reactant", reactant_pdb, "--model", model, "--charge", charge, "--multiplicity", multiplicity,
              "--device", device, "--rigor", rigor, "--outdir", out_dir)
if head: cmd += ["--head", head]
if mode == "gsm-se":      cmd += ["--path-method", "gsm-se"]
if cv_product_s is not None: cmd += ["--cv-product-s", cv_product_s]
commands.append(" ".join(str(x) for x in cmd))
with open(commands_file_path, "w") as f:
    f.write("\\n".join(commands) + "\\n")
print(f"# {{len(commands)}} command(s) → {{commands_file_path}}")
for c in commands: print("\\n" + c)
print(f"\\n# Outputs: {{out_dir}}ts_entry.json (status/barrier/n_imag), gates.json, ts.*")
{sbatch(qtime="48:00:00", cpus="8", mem="80g", queue="gpu", gpu="'large'")}
'''
    return [("markdown", md), ("code", code)]


def cell_ts_entry(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: One-Call Orchestrator (`qcb ts-entry`)**"
    code = f'''\
##################################################################
###  TS-ENTRY ORCHESTRATOR  (guess -> refine -> Hessian -> IRC) ###
##################################################################
# The all-in-one. Pick --entry by what you HAVE:
#   'reactant-product' : R + P  (path search -> saddle -> gate)        [default path=neb]
#   'ts-guess'         : a TS-like geometry already in hand
#   'reactant-only'    : R + a cv in the spec (drives to a product, then double-ended)
# --rigor scales images/backend/thresholds: draft(7,dimer,no-validate) |
#   standard(11,auto,validate,-50cm,0.5) | publication(17,auto,validate).
# Optional plug-ins: --proposer (e.g. react-ot, sidecar; CHNO only) makes the guess;
#   --refiner (e.g. aefm, sidecar; CHNO only) polishes it (non-critical, falls back).

### INPUTS ###
entry        = "reactant-product"    # reactant-product | ts-guess | reactant-only
spec_path    = f"{{REACTION_SPEC_DIR}}reaction_spec.yaml"
reactant_pdb = f"{{RELAX_MINIMIZE_DIR}}relax/reactant.pdb"   # for reactant-product / reactant-only
product_pdb  = f"{{RELAX_MINIMIZE_DIR}}relax/product.pdb"    # for reactant-product
ts_guess_pdb = f"{{SCAN_DIR}}scan/ts_guess.pdb"              # for ts-guess

### OUTPUTS ###
out_dir = f"{{TS_SEARCH_DIR}}ts_entry/"

### ENERGY MODEL ###
model, head, device = {D['model']}, {D['head']}, "cuda"
charge, multiplicity = {D['charge']}, {D['spin']}

### ORCHESTRATOR PARAMETERS ###
rigor          = "standard"     # draft | standard | publication
path_method    = None           # None=preset(neb) | neb | autoneb | fsm | gsm-de | pygsm-de | gsm-se
saddle_backend = None           # None=preset | dimer | sella | sella-internal | pysisyphus-rsprfo | auto
n_images       = None           # None = preset
proposer       = None           # None | 'midpoint' | 'react-ot'(sidecar, CHNO)
refiner        = None           # None | 'identity' | 'aefm'(sidecar, CHNO)

### COMMAND / SUBMIT FILE NAMES ###
commands_name      = f"{{PROJECT_NAME}}_ts_entry"
commands_file_path = os.path.join(CMDS_DIR, commands_name)

### SANITY CHECKS ###
if not Path(spec_path).is_file():
    raise FileNotFoundError(f"reaction spec not found: {{spec_path}}")
Path(out_dir).mkdir(parents=True, exist_ok=True)

### GENERATE COMMANDS ###
commands = []
cmd = qcb_cmd(model, "ts-entry", "--entry", entry, "--reaction-spec", spec_path,
              "--model", model, "--charge", charge, "--multiplicity", multiplicity, "--device", device,
              "--rigor", rigor, "--outdir", out_dir)
if head: cmd += ["--head", head]
if entry in ("reactant-product", "reactant-only"): cmd += ["--reactant", reactant_pdb]
if entry == "reactant-product":                    cmd += ["--product", product_pdb]
if entry == "ts-guess":                            cmd += ["--ts-guess", ts_guess_pdb]
if path_method:    cmd += ["--path-method", path_method]
if saddle_backend: cmd += ["--saddle-backend", saddle_backend]
if n_images:       cmd += ["--n-images", n_images]
if proposer:       cmd += ["--proposer", proposer]
if refiner:        cmd += ["--refiner", refiner]
commands.append(" ".join(str(x) for x in cmd))
with open(commands_file_path, "w") as f:
    f.write("\\n".join(commands) + "\\n")
print(f"# {{len(commands)}} command(s) → {{commands_file_path}}")
for c in commands: print("\\n" + c)
print(f"\\n# Outputs: {{out_dir}}ts_entry.json, gates.json, ts.*")
print("# NOTE: --proposer react-ot / --refiner aefm need those MODELS in this sif; the prebuilt")
print("#       sidecars are model-only -> use the two-step handoff cells, then --entry ts-guess.")
{sbatch(qtime="48:00:00", cpus="8", mem="80g", queue="gpu", gpu="'large'")}
'''
    return [("markdown", md), ("code", code)]


def cell_reactot(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: Generative Proposer — React-OT (sidecar; CHNO only)**"
    code = f'''\
##################################################################
###  REACT-OT PROPOSER  (generative TS guess from R+P; sidecar) ###
##################################################################
# React-OT generates a TS guess directly from a reactant + product (~one shot, OT).
# Runs in its OWN sidecar (REACTOT_SIF; model only, no MLFF). This is step 1 of the
# two-step handoff: emit a guess here -> feed it to ts-entry --entry ts-guess (which has
# the MLFF for refine+gate). DOMAIN: H/C/N/O only, gas-phase neutral organics — NOT for
# metal / charged sites. Use for organic substrate reactions.

### INPUTS ###
reactant_xyz = f"{{GENERATIVE_DIR}}reactant.xyz"     # CHNO reactant (same atom order as product)
product_xyz  = f"{{GENERATIVE_DIR}}product.xyz"      # CHNO product

### OUTPUTS ###
guess_out = f"{{GENERATIVE_DIR}}reactot_guess.xyz"
out_dir   = f"{{GENERATIVE_DIR}}reactot/"

### MODEL ###
charge, multiplicity = {D['charge']}, {D['spin']}            # React-OT ignores charge (trained neutral)

### COMMAND / SUBMIT FILE NAMES ###
commands_name      = f"{{PROJECT_NAME}}_reactot"
commands_file_path = os.path.join(CMDS_DIR, commands_name)

### SANITY CHECKS ###
for p in (reactant_xyz, product_xyz):
    if not Path(p).is_file():
        raise FileNotFoundError(f"input not found: {{p}}")
Path(out_dir).mkdir(parents=True, exist_ok=True)

### GENERATE COMMANDS ###
commands = []
cmd = sidecar_cmd(REACTOT_SIF, "ts-propose", "--method", "react-ot",
                  "--reactant", reactant_xyz, "--product", product_xyz,
                  "--charge", charge, "--multiplicity", multiplicity, "--out", guess_out, "--outdir", out_dir)
commands.append(" ".join(str(x) for x in cmd))
with open(commands_file_path, "w") as f:
    f.write("\\n".join(commands) + "\\n")
print(f"# {{len(commands)}} command(s) → {{commands_file_path}}")
for c in commands: print("\\n" + c)
print(f"\\n# Output: {{guess_out}}  → ts-entry --entry ts-guess  (or refine-ts)")
{sbatch(qtime="02:00:00", cpus="4", mem="32g", queue="gpu", gpu="'small'")}
'''
    return [("markdown", md), ("code", code)]


def cell_aefm(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: Generative Refiner — AEFM (sidecar; CHNO only)**"
    code = f'''\
##################################################################
###  AEFM REFINER  (ML-refine a low-fidelity TS guess; sidecar) ###
##################################################################
# AEFM polishes an existing TS guess (a scan/NEB peak, or a React-OT output) via learned
# equilibrium flow matching. Its OWN sidecar (AEFM_SIF; model only). Chain after React-OT
# for React-OT -> AEFM. DOMAIN: H/C/N/O gas-phase only. A refiner is a structure prior,
# not a saddle finder — the QM gate (refine-ts/validate-ts) still rules.

### INPUTS ###
guess_xyz = f"{{GENERATIVE_DIR}}reactot_guess.xyz"   # any CHNO TS guess to refine

### OUTPUTS ###
refined_out = f"{{GENERATIVE_DIR}}aefm_refined.xyz"
out_dir     = f"{{GENERATIVE_DIR}}aefm/"

### MODEL ###
charge, multiplicity = {D['charge']}, {D['spin']}

### COMMAND / SUBMIT FILE NAMES ###
commands_name      = f"{{PROJECT_NAME}}_aefm"
commands_file_path = os.path.join(CMDS_DIR, commands_name)

### SANITY CHECKS ###
if not Path(guess_xyz).is_file():
    raise FileNotFoundError(f"guess_xyz not found: {{guess_xyz}}")
Path(out_dir).mkdir(parents=True, exist_ok=True)

### GENERATE COMMANDS ###
commands = []
cmd = sidecar_cmd(AEFM_SIF, "ts-refine", "--method", "aefm", "--ts-guess", guess_xyz,
                  "--charge", charge, "--multiplicity", multiplicity, "--out", refined_out, "--outdir", out_dir)
commands.append(" ".join(str(x) for x in cmd))
with open(commands_file_path, "w") as f:
    f.write("\\n".join(commands) + "\\n")
print(f"# {{len(commands)}} command(s) → {{commands_file_path}}")
for c in commands: print("\\n" + c)
print(f"\\n# Output: {{refined_out}}  → ts-entry --entry ts-guess  (or refine-ts)")
{sbatch(qtime="02:00:00", cpus="4", mem="32g", queue="gpu", gpu="'small'")}
'''
    return [("markdown", md), ("code", code)]


def cell_refine_ts(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: Refine to a Saddle (`qcb refine-ts`)**"
    code = f'''\
##################################################################
###  REFINE-TS  (saddle search + partial-Hessian acceptance)   ###
##################################################################
# The acceptance core: dimer/Sella/pysisyphus saddle search -> partial Hessian on the
# reactive atoms -> require exactly ONE imaginary mode (< cutoff) overlapping the reaction
# coordinate -> ts_refined.pdb. Input is EITHER a TS-guess PDB or a path-search dir
# (--from-neb). --reactive-atoms are 1-BASED PDB serials (note: scan/opt used 0-based!).

### INPUTS ###
ts_guess_pdb = f"{{SCAN_DIR}}scan/ts_guess.pdb"   # EITHER a guess PDB ...
from_neb     = None                               # ... OR a path-search dir, e.g. f"{{PATH_SEARCH_DIR}}neb/"
template_pdb = {D['prot']}                         # residue-annotation template (used with --from-neb)

### OUTPUTS ###
out_dir = f"{{REFINE_TS_DIR}}refine/"

### ENERGY MODEL ###
model, head, device = {D['model']}, {D['head']}, "cuda"
charge, multiplicity = {D['charge']}, {D['spin']}

### CONSTRAINTS ###
fix_preset = "ca-only"          # kept during saddle + freq

### REFINE-TS PARAMETERS ###
reactive_atoms   = [{D['react_serials']}]   # 1-based PDB serials (O_nuc, P, O_lg)
backend          = "auto"       # auto (sella->sella-internal->dimer) | dimer | sella | sella-internal | pysisyphus-rsprfo
saddle_fmax      = 0.02
saddle_max_steps = 500
imag_cm_cutoff   = -50.0        # imag mode must be MORE negative than this
imag_overlap     = 0.5          # >= this fraction of the mode on the reactive atoms
n_imag_expected  = 1            # first-order saddle

### COMMAND / SUBMIT FILE NAMES ###
commands_name      = f"{{PROJECT_NAME}}_refine_ts"
commands_file_path = os.path.join(CMDS_DIR, commands_name)

### SANITY CHECKS ###
if from_neb is None and not Path(ts_guess_pdb).is_file():
    raise FileNotFoundError(f"ts_guess_pdb not found: {{ts_guess_pdb}}  (or set from_neb=)")
Path(out_dir).mkdir(parents=True, exist_ok=True)

### GENERATE COMMANDS ###
commands = []
cmd = qcb_cmd(model, "refine-ts")
if from_neb: cmd += ["--from-neb", from_neb, "--template-pdb", template_pdb]
else:        cmd += [ts_guess_pdb]
cmd += ["--model", model, "--charge", charge, "--multiplicity", multiplicity, "--device", device,
        "--fix-preset", fix_preset, "--reactive-atoms", *map(str, reactive_atoms),
        "--backend", backend, "--saddle-fmax", saddle_fmax, "--saddle-max-steps", saddle_max_steps,
        "--imag-cm-cutoff", imag_cm_cutoff, "--imag-mode-overlap", imag_overlap,
        "--n-imag-expected", n_imag_expected, "--outdir", out_dir]
if head: cmd += ["--head", head]
commands.append(" ".join(str(x) for x in cmd))
with open(commands_file_path, "w") as f:
    f.write("\\n".join(commands) + "\\n")
print(f"# {{len(commands)}} command(s) → {{commands_file_path}}")
for c in commands: print("\\n" + c)
print(f"\\n# Outputs: {{out_dir}}ts_refined.pdb (validated TS), imag_mode.npy, summary.json (PASS/FAIL)")
{sbatch(qtime="48:00:00", cpus="8", mem="80g", queue="gpu", gpu="'large'")}
'''
    return [("markdown", md), ("code", code)]


def cell_validate(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: Validate the TS — tiered Hessian (`qcb validate-ts`)**"
    code = f'''\
##################################################################
###  VALIDATE-TS  (independent tiered Hessian validation)      ###
##################################################################
# Independent of refine-ts. Tier A = reactive-atom partial Hessian; Tier B = active-region
# Hessian (catches a hidden 2nd imag mode in the metal/water shell); Tier C = Lanczos full
# check. Confirms exactly one imaginary mode on the reaction coordinate.
# --reactive-atoms are 1-BASED PDB serials.

### INPUTS ###
ts_pdb = f"{{REFINE_TS_DIR}}refine/ts_refined.pdb"   # the refined TS

### OUTPUTS ###
out_dir = f"{{TS_VALIDATION_DIR}}validate/"

### ENERGY MODEL ###
model, head, device = {D['model']}, {D['head']}, "cuda"
charge, multiplicity = {D['charge']}, {D['spin']}

### VALIDATE PARAMETERS ###
reactive_atoms   = [{D['react_serials']}]   # 1-based PDB serials
tier             = "b"          # 'a' | 'b' | 'c' | 'all' | comma list
active_region    = None         # Tier B select spec, e.g. 'sphere 6.0 around resid 169' (None=auto by reactive atoms)
imag_cm_cutoff   = -50.0
imag_overlap     = 0.5
n_imag_expected  = 1

### COMMAND / SUBMIT FILE NAMES ###
commands_name      = f"{{PROJECT_NAME}}_validate_ts"
commands_file_path = os.path.join(CMDS_DIR, commands_name)

### SANITY CHECKS ###
if not Path(ts_pdb).is_file():
    raise FileNotFoundError(f"ts_pdb not found: {{ts_pdb}}  (run refine-ts first)")
Path(out_dir).mkdir(parents=True, exist_ok=True)

### GENERATE COMMANDS ###
commands = []
cmd = qcb_cmd(model, "validate-ts", ts_pdb, "--outdir", out_dir, "--model", model,
              "--charge", charge, "--multiplicity", multiplicity, "--device", device,
              "--reactive-atoms", *map(str, reactive_atoms), "--tier", tier,
              "--imag-cm-cutoff", imag_cm_cutoff, "--imag-mode-min-overlap", imag_overlap,
              "--n-imag-expected", n_imag_expected)
if head:          cmd += ["--head", head]
if active_region: cmd += ["--active-region", active_region]
commands.append(" ".join(str(x) for x in cmd))
with open(commands_file_path, "w") as f:
    f.write("\\n".join(commands) + "\\n")
print(f"# {{len(commands)}} command(s) → {{commands_file_path}}")
for c in commands: print("\\n" + c)
print(f"\\n# Output dir: {{out_dir}}  (per-tier PASS/FAIL + frequencies)")
{sbatch(qtime="24:00:00", cpus="8", mem="80g", queue="gpu", gpu="'large'")}
'''
    return [("markdown", md), ("code", code)]


def cell_irc(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: Verify IRC-like (`qcb verify-irc-like`)**"
    code = f'''\
##################################################################
###  VERIFY-IRC-LIKE  (TS connects reactant & product basins)  ###
##################################################################
# Displace +/- along the imaginary mode and relax both branches -> confirm the TS connects
# the intended reactant and product (two distinct lower basins). Needs the imag-mode vector
# from refine-ts/validate-ts (imag_mode.npy). For organophosphate hydrolysis, watch for a
# PENTACOORDINATE intermediate (s ~ 0, both bonds ~1.7 A) — that's a real stepwise mechanism,
# not a failure (then do two-step NEB: R->intermediate, intermediate->P).

### INPUTS ###
ts_pdb    = f"{{REFINE_TS_DIR}}refine/ts_refined.pdb"
imag_mode = f"{{REFINE_TS_DIR}}refine/imag_mode.npy"   # emitted by refine-ts / validate-ts

### OUTPUTS ###
out_dir = f"{{TS_VALIDATION_DIR}}irc_like/"

### ENERGY MODEL ###
model, head, device = {D['model']}, {D['head']}, "cuda"
charge, multiplicity = {D['charge']}, {D['spin']}

### IRC-LIKE PARAMETERS ###
displacement = 0.20             # A along the imag mode
fmax         = 0.05
max_steps    = 200
optimizer    = "lbfgs"          # lbfgs | bfgs | fire

### COMMAND / SUBMIT FILE NAMES ###
commands_name      = f"{{PROJECT_NAME}}_verify_irc"
commands_file_path = os.path.join(CMDS_DIR, commands_name)

### SANITY CHECKS ###
for p in (ts_pdb, imag_mode):
    if not Path(p).is_file():
        raise FileNotFoundError(f"not found: {{p}}  (run refine-ts/validate-ts first)")
Path(out_dir).mkdir(parents=True, exist_ok=True)

### GENERATE COMMANDS ###
commands = []
cmd = qcb_cmd(model, "verify-irc-like", ts_pdb, "--imag-mode", imag_mode, "--outdir", out_dir,
              "--model", model, "--charge", charge, "--multiplicity", multiplicity, "--device", device,
              "--displacement", displacement, "--fmax", fmax, "--max-steps", max_steps,
              "--optimizer", optimizer)
if head: cmd += ["--head", head]
commands.append(" ".join(str(x) for x in cmd))
with open(commands_file_path, "w") as f:
    f.write("\\n".join(commands) + "\\n")
print(f"# {{len(commands)}} command(s) → {{commands_file_path}}")
for c in commands: print("\\n" + c)
print(f"\\n# Output dir: {{out_dir}}  (forward/back basins + delta-energies)")
{sbatch(qtime="24:00:00", cpus="8", mem="80g", queue="gpu", gpu="'large'")}
'''
    return [("markdown", md), ("code", code)]


def cell_dft(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: (optional) DFT Reference — ORCA native NEB-TS**"
    code = f'''\
##################################################################
###  DFT REFERENCE  (qcb ts-entry --engine orca; native NEB-TS) ###
##################################################################
# For a publication-grade barrier, route the whole TS step to ORCA's native NEB-TS / OptTS
# via the QM-engine gateway. --no-execute writes the ORCA input + an sbatch wrapper WITHOUT
# running, so you can inspect/queue it. Set the functional/basis + resources in the engine
# config. CPU partition (ORCA is CPU/MPI).

### INPUTS ###
entry        = "reactant-product"
spec_path    = f"{{REACTION_SPEC_DIR}}reaction_spec.yaml"
reactant_pdb = f"{{RELAX_MINIMIZE_DIR}}relax/reactant.pdb"   # EDIT
product_pdb  = f"{{RELAX_MINIMIZE_DIR}}relax/product.pdb"    # EDIT

### OUTPUTS ###
out_dir = f"{{DFT_DIR}}orca_nebts/"

### DFT PARAMETERS ###
charge, multiplicity  = {D['charge']}, {D['spin']}
engine_method = "wB97X-D3/def2-TZVP"   # set in the ORCA engine config; shown here for reference
EXECUTE       = False           # False -> write ORCA input + wrapper, don't run

### COMMAND / SUBMIT FILE NAMES ###
commands_name      = f"{{PROJECT_NAME}}_dft_orca"
commands_file_path = os.path.join(CMDS_DIR, commands_name)

### SANITY CHECKS ###
if not Path(spec_path).is_file():
    raise FileNotFoundError(f"reaction spec not found: {{spec_path}}")
Path(out_dir).mkdir(parents=True, exist_ok=True)

### GENERATE COMMANDS ###
commands = []
cmd = [*APPTAINER(MAIN_SIF, gpu=False), "qcb", "ts-entry", "--entry", entry, "--reaction-spec", spec_path,
       "--engine", "orca", "--reactant", reactant_pdb, "--product", product_pdb,
       "--charge", str(charge), "--multiplicity", str(multiplicity), "--outdir", out_dir,
       ("--execute" if EXECUTE else "--no-execute")]
commands.append(" ".join(str(x) for x in cmd))
with open(commands_file_path, "w") as f:
    f.write("\\n".join(commands) + "\\n")
print(f"# {{len(commands)}} command(s) → {{commands_file_path}}")
for c in commands: print("\\n" + c)
print(f"\\n# Outputs: {{out_dir}}  (ORCA NEB-TS job; method = {{engine_method}})")
{sbatch(qtime="72:00:00", cpus="16", mem="120g", queue="cpu", gpu="None")}
'''
    return [("markdown", md), ("code", code)]


# ════════════════════════════════════════════════════════════════════════════
#  Assemble the two notebooks
# ════════════════════════════════════════════════════════════════════════════
def overview_general():
    return ("markdown", r"""# Generalized TS-search notebook — a menu of every method

Reaction-agnostic, plug-and-play. Run **INIT** once, then pick the cells that fit your
inputs. Each driver cell builds a `qcb` command into `CMDS_DIR/<name>` and the
`[SETUP BATCH JOBS]` block writes/queues the sbatch array (preview the printed command;
`nb.submit_array_job` writes `SUBMIT_DIR/<name>.sh`). Protonate / monitor / reaction-spec
print a command to run in the terminal (no sbatch).

**Which "get-a-TS-guess" method do I use?** (steps are a MENU, not a strict order)

| You have... | Use |
|---|---|
| reactant **+** product | 1-D scan, **CI-NEB** (`neb`), GSM/FSM, or `ts-entry --entry reactant-product` |
| **only a reactant** (+ a `cv`) | single-ended: `ts-entry --entry reactant-only` or `--path-method gsm-se` |
| **only a TS guess** | `ts-entry --entry ts-guess`, or go straight to `refine-ts` |
| CHNO organic R+P | generative **React-OT** proposer (+ **AEFM** refiner), then `ts-entry --entry ts-guess` |

Every route ends at the **acceptance gate**: `refine-ts` (saddle + 1 imaginary mode) →
`validate-ts` (tiered Hessian) → `verify-irc-like`. Constraints (per cell): `--fix-preset`
{ca-only,backbone,backbone-water,none}, `--fix/--free` select specs, `--fix-bond` (hard pin),
`--restrain-bond` (harmonic). Models are plug-and-play via the `model` variable +
`container_for()` routing (MACE/OMol/ORB/AIMNet2 → MAIN_SIF; UMA/eSEN/AllScAIP → UMA_SIF).""")


def overview_opaa():
    return ("markdown", r"""# OPAA di-Zn phosphotriesterase — TS pipeline (ordered protocol)

The recommended in-order protocol for the OPAA di-Zn theozyme (SN2 at phosphorus), a
charged metal active site, synthesized from a committee review (3 independent agents +
codex) of the docs + code. A focused subset of the generalized notebook with OPAA inputs
pre-filled — **edit the atom serials and input PDB for your structure** (only example
values are filled; nothing is hardcoded).

**What "path search" vs "refine to a saddle" means** (a question worth nailing): a
**path search** *proposes* a TS GUESS along the reaction coordinate — the **1-D relaxed
scan** here IS a (single-ended) path search, and it hands you a TS guess **plus**
approximate reactant/product endpoints. **`refine-ts` is a SEPARATE step**: it optimizes
that guess to an *exact* first-order saddle and runs the Hessian gate — it does NOT search
a path. So you need a guess (scan or NEB) *before* refine-ts. And yes — you then
**energy-minimize the endpoints** to true basins, because the barrier is
`E(TS) − E(reactant_min)`.

**Protocol:** `0` protonate (no PTM) → `1` monitor (--metals) → `2` reaction-spec
(O_nuc→P forming, P→O_lg breaking) → `3` CA-frozen relax of the cluster (reactant basin) →
`4` 1-D relaxed scan → TS guess **+ R/P endpoint frames** → `5` **minimize the R & P
endpoints** (true basins) → `6` *(optional)* CI-NEB between the minima (more rigorous guess
for an asynchronous step) → `7` refine-ts (`--backend auto`, 1 imaginary mode) →
`8` validate-ts (tier b, Zn-shell active region) → `9` verify-irc-like → `10` optional ORCA DFT.

**Model:** default **`mace-polar-m`** (MACE-POLAR-1-M) — polarizable + long-range
electrostatics, ideal for the charged di-Zn pocket, and **baked into MAIN_SIF so it loads
in-process**. `mace-omol` is the higher-accuracy second pass (large GPU); `mace-mh-1 --head
omol` and `orb-mol-conservative` are charge-aware alternatives; UMA/eSEN route to UMA_SIF.
**Never** GFN2-xTB on the metals. **Don't** use React-OT/AEFM here — they are CHNO/gas-phase
only and HARD-FAIL on Zn/P.

**Charge & multiplicity (pre-filled):** net charge of the protonated system = **0** (no PTM);
no radicals → spin quantum number **S = 0** → spin **multiplicity M = 2S+1 = 1** (singlet), so
pass **`--multiplicity 1`**. (The codebase uses *multiplicity* (2S+1) everywhere; the optional
`--spin` flag instead takes **S** and converts it to 2S+1, so `--spin 0` ⇒ `--multiplicity 1`.)
Charge & multiplicity are **CLI-only** — the reaction-spec YAML ignores them.

**Watch for a pentacoordinate intermediate.** Organophosphate hydrolysis at P is often
*stepwise* through a trigonal-bipyramidal phosphorane (both P–O bonds ~1.7 Å, CV s≈0). If
`verify-irc-like` lands in such a basin and an all-real Hessian confirms it's a minimum, it's
a real intermediate — split into two TS searches (R→intermediate, intermediate→P).""")


def build(profile):
    cells = []
    if profile == "general":
        cells.append(overview_general())
    else:
        cells.append(overview_opaa())
    cells += cells_init(profile)
    if profile == "opaa":
        seq = [cell_protonate, cell_monitor, cell_reaction_spec, cell_relax, cell_scan,
               cell_min_endpoints, cell_neb_opaa, cell_refine_ts, cell_validate, cell_irc, cell_dft]
    else:
        seq = [cell_protonate, cell_monitor, cell_reaction_spec, cell_relax, cell_scan,
               cell_neb, cell_gsm, cell_single_ended, cell_ts_entry, cell_reactot, cell_aefm,
               cell_refine_ts, cell_validate, cell_irc, cell_dft]
    for i, fn in enumerate(seq):
        cells += fn(profile, i)
    return cells


def _src(t):
    lines = t.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

def write_nb(cells, path):
    nb_cells = []
    for kind, text in cells:
        c = {"cell_type": kind, "metadata": {}, "source": _src(text)}
        if kind == "code":
            c["outputs"] = []; c["execution_count"] = None
        nb_cells.append(c)
    notebook = {"cells": nb_cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(notebook, indent=1))
    print(f"wrote {path}  ({len(nb_cells)} cells)")


ROOT = "/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/notebooks"
write_nb(build("general"), f"{ROOT}/general_ts_pipeline/ts_pipeline_GENERAL.ipynb")
write_nb(build("opaa"),    f"{ROOT}/opaa_theozyme/opaa_ts_pipeline.ipynb")
