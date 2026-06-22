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

def sbatch(qtime="04:00:00", cpj=1, cpus="1", mem="6g", queue="gpu", gpu="'small'"):
    return (SBATCH.replace("@@QTIME@@", qtime).replace("@@CPJ@@", str(cpj))
            .replace("@@CPUS@@", str(cpus)).replace("@@MEM@@", mem)
            .replace("@@QUEUE@@", queue).replace("@@GPU@@", gpu))

# ════════════════════════════════════════════════════════════════════════════
#  INIT cell — the user's exact init (verbatim imports) + container vars/helpers.
# ════════════════════════════════════════════════════════════════════════════
def init_cell(profile):
    if profile == "opaa":
        proj = "opaa_theozyme"
        _demo = "f'{HOME_DIR}codebase_projects/quantum_cowboy_biochemistry/notebooks/opaa_theozyme"
        sysblock = ("HOME_DIR      = '/home/woodbuse/'\n"
                    f"THEOZYME_DIR  = {_demo}/theozyme/'\n"
                    "SYSTEM_DIR    = THEOZYME_DIR        # the active-site / structure working dir\n")
        wd = f"{_demo}'"
        od = f"{_demo}/output/'"
    else:
        proj = "my_reaction"          # EDIT
        sysblock = ("HOME_DIR      = '/home/woodbuse/'\n"
                    "SYSTEM_DIR    = f'{HOME_DIR}my_reaction/'   # EDIT: your structure / active-site working dir\n")
        wd = "SYSTEM_DIR"
        od = "SYSTEM_DIR"
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
_OUTPUT_DIR_OVERRIDE  = {od}   # default: WORKING_DIR/output/

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
#    MAIN_SIF has the MLFFs (MACE / MACE-OMol / ORB / AIMNet2) + xTB + the `cowboy-qc` CLI.
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

# The `cowboy-qc` console-script is NOT baked into the sifs — invoke the CLI as a
# module against the bind-mounted repo (PYTHONPATH) instead. One definition, reused
# everywhere (qcb_cmd / sidecar_cmd / monitor / reaction-spec).
CLI = ['env', f'PYTHONPATH={{QUANTUM_COWBOY_DIR}}', 'python', '-m', 'quantum_engine.cli']

def qcb_cmd(model, *args, gpu=True):
    """Full cowboy-qc CLI command (list) in the sif that can load `model`."""
    return [*APPTAINER(container_for(model), gpu=gpu), *CLI, *map(str, args)]

def sidecar_cmd(sif, *args, gpu=True):
    """A generative-sidecar command (React-OT / AEFM) in its own sif."""
    return [*APPTAINER(sif, gpu=gpu), *CLI, *map(str, args)]

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
            unprot='f"{THEOZYME_DIR}input_theozyme/opaa_3l7g_optimal_maximal_theozyme_pxn_unprotonated.pdb"',
            prot='f"{PROTOMERS_DIR}opaa_3l7g_optimal_maximal_theozyme_pxn.pdb"',
            model="'mace-polar-m'", head="None", charge="0", spin="1",
            # SN2-at-P by atom DESCRIPTOR: nucleophile OHX-O3 attacks P (SUB-P1);
            # the P-O7 (SUB-O7) leaving-group bond breaks. (Works for any reaction;
            # see the ATOM TOKENS note. Also accepts serial:N / CHAIN:RESID:ATOM / index.)
            f_nuc="'OHX-O3'", f_elec="'SUB-P1'", b_p="'SUB-P1'", b_lg="'SUB-O7'",
            scan_nuc="'OHX-O3'", scan_p="'SUB-P1'",   # scanned forming bond O_nuc..P
            react_serials="'OHX-O3', 'SUB-P1', 'SUB-O7'",   # reactive atoms (nuc, P, lg)
            ptm='{}',              # OPAA construct: NO post-translational modification (no KCX)
            # HETATM (non-protein) charge — NEVER assumed: 2x Zn(II) +2, bridging OHX -1, SUB 0 = +3
            ligand_charges='{"ZN": 2, "OHX": -1, "SUB": 0}',
            nonprotein_charge='None',
            # monitor / CV atoms by descriptor (forming OHX-O3..SUB-P1, breaking SUB-P1..SUB-O7)
            monitor_bonds=('[("OHX-O3", "SUB-P1"),    # forming: hydroxide O -> phosphorus\n'
                           '                      ("SUB-P1", "SUB-O7")]    # breaking: phosphorus -> leaving-group O'),
        )
    return dict(
        unprot='f"{SYSTEM_DIR}structure_unprotonated.pdb"   # EDIT',
        prot='f"{PROTOMERS_DIR}structure.pdb"               # EDIT',
        model="'mace-polar-m'", head="None", charge="0", spin="1",
        f_nuc="'RES-NUC'", f_elec="'RES-CENTER'", b_p="'RES-CENTER'", b_lg="'RES-LG'",   # EDIT: e.g. OHX-O3, A519-ZN, serial:1872, 0:1848
        scan_nuc="'RES-NUC'", scan_p="'RES-CENTER'",   # EDIT
        react_serials="'RES-NUC', 'RES-CENTER', 'RES-LG'",   # EDIT
        ptm='{}',
        ligand_charges='{}',   # "RESNAME": charge for each non-water HETATM (none -> not assumed)
        nonprotein_charge='None',
        monitor_bonds='[]',
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
    nonprot_note = ('# OPAA di-Zn site: 2x Zn(II) +2 + bridging OHX -1 + SUB 0  ->  +3'
                    if profile == "opaa" else
                    '# leave empty to report ONLY the protein charge (non-protein never assumed)')
    md = f"# **STEP {step}: Protonate / Generate Protomers**"
    code = f'''\
##################################################################
###          PROTONATE STRUCTURE   (cowboy-qc protonator v2)         ###
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
CONTAINER  = MAIN_SIF       # the cowboy-qc / protonator container (defined in INIT)
PROTONATOR = f"{{QUANTUM_COWBOY_DIR}}quantum_engine/prep/protonator.py" # (`cowboy-qc protonate <same args>` is equivalent to running this file directly.)

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
# --- non-protein (HETATM) charge — NEVER assumed; declare it yourself --------------
# Per-residue formal charges (summed over INSTANCES, so two ZN at +2 -> +4), OR a
# single total via nonprotein_charge. Reported as NET_THEOZYME_NONPROTEIN_CHARGE.
{nonprot_note}
protonate_ligands = False      # stub (HETATM assumed already protonated)
ligand_charges    = {D['ligand_charges']}   # "RESNAME": charge per non-water HETATM
nonprotein_charge = {D['nonprotein_charge']}        # int OVERRIDES the dict with one total; None = use dict
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
if nonprotein_charge is not None:    cmd += ["--nonprotein-charge", str(nonprotein_charge)]
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
    D = defaults(profile)
    md = f"# **STEP {step}: Monitor Active Site (bond / metal coordination)**"
    code = f'''\
##################################################################
###          MONITOR ACTIVE SITE   (cowboy-qc monitor)              ###
##################################################################
# Non-constraining sanity report: measured key bonds + auto-detected metal
# coordination shells. Confirm the protonated geometry before spending GPU time.
# Instant CPU — this cell PRINTS the command (copy-paste; no sbatch).

print_commands = True

### INPUTS ###
input_pdb = {D['prot']}

### OUTPUTS ###
out_dir = MONITOR_DIR

### MONITOR PARAMETERS ###
# Atom pairs to measure. Each atom is a descriptor — RESNAME-ATOM (OHX-O3),
# <Chain><ResNo>-ATOM (A519-ZN), <RESNAME><ResNo>-ATOM (ZN519-ZN), CHAIN:RESID:ATOM
# (A:519:ZN) — or a 0-based index. (`cowboy-qc monitor --bond` resolves these.)
monitor_bond_pairs = {D['monitor_bonds']}
report_metals      = True      # auto-detect metals + their coordination shells

### CONSTANTS ###
CONTAINER = MAIN_SIF

### BUILD THE COMMAND ###
cmd = [*APPTAINER(CONTAINER, gpu=False), *CLI, "monitor", input_pdb, "--outdir", out_dir]
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
# ATOM TOKENS (every cowboy-qc command that takes a PDB resolves these): RESNAME-ATOM
#              (OHX-O3, SUB-P1), <Chain><ResNo>-ATOM (A519-ZN), <RESNAME><ResNo>-ATOM
#              (ZN519-ZN), CHAIN:RESID:ATOM (A:169:NZ), serial:N (1-based), 0:N, or a
#              0-based index. A token must be UNIQUE — ambiguous ones error with candidates.
# IMPORTANT: charge & multiplicity are NOT read from this YAML — always pass --charge/--multiplicity on
#            every cowboy-qc command (the YAML keys would be silently ignored).

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
# Optional 1-D collective variable (bond_difference: s = d(a,b) - d(a,c) over EXACTLY 3
# atoms [a=center, b=breaking-partner, c=forming-partner]); set cv_kind=None to omit.
cv_kind  = "bond_difference"
cv_atoms = [{D['b_p']}, {D['b_lg']}, {D['f_nuc']}]   # [center, breaking, forming]
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
cmd = [*APPTAINER(CONTAINER, gpu=False), *CLI, "reaction-spec", spec_path, "--structure", struct_pdb]
if print_commands:
    print("### VALIDATE (copy-paste into terminal) ###")
    print(" ".join(str(x) for x in cmd))
'''
    return [("markdown", md), ("code", code)]


def cell_relax(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: Minimize / Relax a Geometry (`cowboy-qc opt`)**"
    code = f'''\
##################################################################
###     MINIMIZE / RELAX   (cowboy-qc opt; constrained or not)       ###
##################################################################
# Relax ANY input geometry: a reactant, a product, a TS-region pose, or the whole
# protonated cluster. The constraint regime is the key knob:
#   * unconstrained    (fix_preset='none')      -> a true minimum (final R / P endpoints)
#   * CA-frozen        (fix_preset='ca-only')   -> scaffold the backbone, let chemistry breathe
#   * bond-pinned      (fix_bonds=[...])        -> hold the forming/breaking distance(s) while everything else relaxes (constrained TS-region min)
# fix_bond / restrain_bond atoms accept any token: descriptor (OHX-O3, A519-ZN), serial:N, CHAIN:RESID:ATOM, or a 0-based index. (Trailing R0/K stays numeric.)

### INPUTS ###
input_pdb = {D['prot']}        # the geometry to relax (reactant / product / ts-region / cluster)

### OUTPUTS ###
out_dir     = f"{{RELAX_MINIMIZE_DIR}}relax/"
relaxed_pdb = f"{{out_dir}}relaxed.pdb"

### ENERGY MODEL (plug-and-play: change `model`/`head`; container_for() picks the sif) ###
# Charged METAL active sites (e.g. di-Zn) — use a CHARGE-AWARE model:
#   mace-polar-m : polarizable + long-range electrostatics — IDEAL for a charged metal pocket (DEFAULT; baked into MAIN_SIF, loads in-process). sizes -s|-m|-l
#   mace-omol    : wB97M-V/OMol25, charge-aware, highest accuracy (large GPU: A6000/H200)
#   mace-mh-1    : multi-head foundation model -> set head='omol' for the OMol25 head
#   orb-mol-conservative : Orbital-Materials, charge/spin-aware, Zn-capable (true gradients)
#   uma-m-1p1 / esen-sm-conserving / allscaip-md-conserving : FairChem (route to UMA_SIF)
# Organic-only (NO metals): mace-off-* (wB97M organic) ; aimnet2-rxn (CHON, TS-tuned).
# AVOID GFN2-xTB on metals (not charge-aware there). --head applies to MACE multi-head only.
model        = 'mace-polar-m'
head         = None            # MACE multi-head only (e.g. 'omol' for mace-mh-1); None for polar/omol
device       = "cuda"
charge       = 0               # FULL-cluster net charge (CLI-only; the spec YAML ignores charge/multiplicity)
multiplicity = 1               # spin MULTIPLICITY M=2S+1 (1=singlet/no radicals, 2=doublet, 3=triplet) # NOTE: the CLI flag is --multiplicity. (--spin takes S and converts to 2S+1.)

### CONSTRAINTS ###
fix_preset     = "ca-only"      # 'none' | 'ca-only' | 'backbone' | 'backbone-water'
extra_fix      = []             # extra select specs, e.g. ['residue HOH', 'chain B', 'resid 169']
extra_free     = []             # subtract from the preset, e.g. ['atoms ZN1 ZN2']
fix_bonds      = []             # hard-pin: [[A, B]] or [[A, B, R0]] (atom tokens)  e.g. [[{D['scan_p']}, {D['scan_nuc']}]]
restrain_bonds = []             # harmonic: [[i, j, K, R0]]  (0-based; K in eV/A^2)

### TS-GUESS AUTO-PIN (pull forming/breaking bonds from the STEP-2 reaction spec) ###
# A TS-region pose is a SADDLE: a plain minimizer rolls downhill OFF it and collapses to the
# reactant or product. Set is_ts_guess=True to auto-pin the reaction-spec's forming+breaking
# bonds (at their current lengths) so only the scaffold/H-bonds/waters relax. Pins in `fix_bonds`
# above are ADDED on top; set is_ts_guess=False for a reactant/product/cluster (a real minimum).
is_ts_guess        = True
reaction_spec_yaml = f"{{REACTION_SPEC_DIR}}reaction_spec.yaml"   # STEP 2 output (forming/breaking bonds)

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
# is_ts_guess: pin the STEP-2 reaction spec's forming+breaking bonds (held at current length)
auto_fix_bonds = []
if is_ts_guess:
    import yaml
    if not Path(reaction_spec_yaml).is_file():
        raise FileNotFoundError(f"is_ts_guess=True needs the STEP-2 reaction spec: {{reaction_spec_yaml}}")
    _rs = yaml.safe_load(Path(reaction_spec_yaml).read_text()) or {{}}
    auto_fix_bonds = [list(b) for b in (_rs.get("forming_bonds") or [])] + \\
                     [list(b) for b in (_rs.get("breaking_bonds") or [])]
    print(f"# is_ts_guess=True -> auto-pinned {{len(auto_fix_bonds)}} reaction bond(s): {{auto_fix_bonds}}")
for b in (auto_fix_bonds + list(fix_bonds)): cmd += ["--fix-bond", *map(str, b)]
for b in restrain_bonds: cmd += ["--restrain-bond", *map(str, b)]
cmd += ["--optimizer", optimizer, "--fmax", fmax, "--max-steps", max_steps,
        "--outdir", out_dir, "--output-pdb", relaxed_pdb]
commands.append(" ".join(str(x) for x in cmd))
with open(commands_file_path, "w") as f:
    f.write("\\n".join(commands) + "\\n")
print(f"# {{len(commands)}} command(s) → {{commands_file_path}}")
for c in commands: print("\\n" + c)
print(f"\\n# Output: {{relaxed_pdb}}")
{sbatch(qtime="02:00:00", cpus="1", mem="6g", queue="gpu", gpu="'small'")}
'''
    return [("markdown", md), ("code", code)]


def cell_scan(profile, step):
    D = defaults(profile)
    md = f"""# **STEP {step}: 1-D Relaxed Scan → Reactant, Product & TS Guess (`cowboy-qc scan`)**

A 1-D *relaxed* scan slides one reaction coordinate and re-minimizes everything else at each
step. In one shot it yields the **reactant** (one end), the **product** (other end), and a
**TS guess** (highest-energy frame) that feeds `refine-ts`.

**Two coordinate choices — set `scan_kind`:**

| `scan_kind` | coordinate | when to use |
|---|---|---|
| `"bond"` | one forming/breaking bond distance `d(i,j)` | a single bond dominates the step |
| `"bond-difference"` | the More-O'Ferrall–Jencks CV `s = d(center,breaking) − d(center,forming)` | **SN2-like** steps — drives both bonds antisymmetrically and does **not** pre-bias concerted vs stepwise *(recommended here)* |

**Direction matters** — the scan always runs **reactant → product**, so `frame[0]` is the reactant
and `frame[-1]` the product:
- `"bond"` on the *forming* bond: **long** (nucleophile far ⇒ reactant) → **short** (bonded ⇒ product).
- `"bond-difference"`: **s ≪ 0** (reactant) → **s ≫ 0** (product).

> ⚠️ **Your monitor showed the active site already near the TS** (O3–P ≈ 1.97 Å bonded, P–O7
> stretched). So push the **reactant** end far enough (O3–P ≳ 3.2 Å, or `s ≈ −2.5`) to reach a
> clean reactant basin, and don't be surprised if the TS guess lands near `frame[0]`. If a
> **pentacoordinate phosphorane intermediate** is real you'll see a dip mid-scan — the optional
> **2-D scan below** resolves concerted vs stepwise.

**Troubleshooting**
- *Product never forms / reactant never separates* → widen `scan_start`/`scan_end`.
- *Jagged profile / hysteresis* → more `scan_n_steps`, lower `scan_fmax`, or use `"bond-difference"`.
- *A bond snaps or the cluster distorts* → tighten `fix_preset` (e.g. `"backbone"`) so only the coordinate moves.
- *Endpoints look swapped* → re-check the direction note above (`frame[0]` must be the reactant)."""
    code = f'''\
##################################################################
###     1-D RELAXED SCAN   (cowboy-qc scan; reactant -> product) ###
##################################################################
# Slides the reaction coordinate and relaxes everything else at each step ->
# reactant (frame 0), product (last frame), TS guess (highest-E frame -> refine-ts).
# Atom tokens are descriptors (OHX-O3), serial:N, CHAIN:RESID:ATOM, or 0-based indices.

### INPUTS ###
relaxed_pdb = f"{{RELAX_MINIMIZE_DIR}}relax/relaxed.pdb"   # usually the CA-frozen relaxed cluster

### OUTPUTS ###
out_dir           = f"{{SCAN_DIR}}scan/"
ts_guess_pdb      = f"{{out_dir}}ts_guess.pdb"        # max-energy frame -> Step refine-ts
reactant_scan_pdb = f"{{out_dir}}reactant_scan.pdb"  # frame 0   (approx reactant) -> minimize next
product_scan_pdb  = f"{{out_dir}}product_scan.pdb"   # last frame (approx product) -> minimize next

### ENERGY MODEL ###
model, head, device = {D['model']}, {D['head']}, "cuda"   # default mace-polar-m (see relax cell's menu)
charge, multiplicity = {D['charge']}, {D['spin']}                 # net charge (CLI-only); multiplicity M=2S+1 (1=singlet)

### CONSTRAINTS ###
fix_preset = "ca-only"          # the scanned coordinate is auto-pinned ON TOP of this preset

### REACTION-COORDINATE ATOMS (descriptors) ###
center   = {D['b_p']}     # shared atom = the electrophilic centre (P)
forming  = {D['f_nuc']}   # FORMING-bond partner (nucleophile -> centre)
breaking = {D['b_lg']}    # BREAKING-bond partner (centre -> leaving group)

### SCAN PARAMETERS ###
scan_kind    = "bond-difference"   # "bond" (one forming bond) | "bond-difference" (MOJ CV; recommended for SN2)
scan_n_steps = 16
scan_fmax    = 0.05
# Direction is ALWAYS reactant -> product, so frame[0]=reactant and frame[-1]=product:
if scan_kind == "bond":
    scan_coord, scan_indices = "bond", [forming, center]   # the forming bond, e.g. O_nuc..P
    scan_start, scan_end = 3.2, 1.6        # Angstrom: reactant (nuc far) -> product (nuc bonded)
    traj_name = "scan-trajectory.xyz"
elif scan_kind == "bond-difference":
    scan_coord, scan_indices = "bond-difference", [center, breaking, forming]
    scan_start, scan_end = -2.5, 2.5       # CV s (Angstrom): reactant (s<0) -> product (s>0)
    traj_name = "scan-bonddiff-trajectory.xyz"
else:
    raise ValueError("scan_kind must be 'bond' or 'bond-difference'")

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
# 1) the scan; 2) a helper that pulls reactant/product/TS-guess frames out as PDBs.
commands = []
cmd_scan  = qcb_cmd(model, "scan", relaxed_pdb, "--model", model, "--charge", charge,
                    "--multiplicity", multiplicity, "--device", device, "--fix-preset", fix_preset,
                    "--coord", scan_coord, "--indices", *scan_indices,
                    "--start", scan_start, "--end", scan_end, "--n-steps", scan_n_steps,
                    "--fmax", scan_fmax, "--outdir", out_dir)
extract_py = f"{{out_dir}}extract_frames.py"
Path(extract_py).write_text(textwrap.dedent(f"""\\
    import ase.io as io, json, glob
    from quantum_engine.io import load_structure, write_pdb
    OUT      = r'{{out_dir}}'
    TRAJ     = r'{{out_dir}}{{traj_name}}'
    TEMPLATE = r'{{template_pdb}}'
    CHARGE   = {{charge}}
    frames = io.read(TRAJ, index=':')   # frame 0 = reactant, last = product
    energy = lambda a: a.info['energy_eV'] if 'energy_eV' in a.info else a.get_potential_energy()
    # TS-guess frame = highest INTERIOR maximum (the scan summary picks it; endpoints are basins).
    sums = sorted(glob.glob(OUT + '*-summary.json'))
    if sums:
        s = json.loads(open(sums[0]).read()); i = int(s['ts_guess_idx']); barrierless = s.get('barrierless', False)
    else:
        es = [energy(a) for a in frames]
        interior = [k for k in range(1, len(es) - 1) if es[k] >= es[k - 1] and es[k] >= es[k + 1]]
        i = max(interior, key=lambda k: es[k]) if interior else max(range(len(es)), key=lambda k: es[k])
        barrierless = not interior
    _, bt, _ = load_structure(TEMPLATE)
    write_pdb(frames[0],  bt, r'{{reactant_scan_pdb}}', total_charge=CHARGE)
    write_pdb(frames[-1], bt, r'{{product_scan_pdb}}',  total_charge=CHARGE)
    write_pdb(frames[i],  bt, r'{{ts_guess_pdb}}',      total_charge=CHARGE)
    print('TS guess = frame %d/%d ; reactant=frame 0 ; product=frame %d' % (i, len(frames) - 1, len(frames) - 1))
    if barrierless:
        print('# WARNING: no interior barrier (monotonic) -> TS guess is an endpoint; widen/shift the scan range.')
"""))
cmd_extract = [*APPTAINER(container_for(model)), "python", extract_py]
commands.append(" ".join(str(x) for x in cmd_scan))
commands.append(" ".join(str(x) for x in cmd_extract))
with open(commands_file_path, "w") as f:
    f.write("\\n".join(commands) + "\\n")
print(f"# scan_kind={{scan_kind}} ; {{len(commands)}} command(s) → {{commands_file_path}}")
for c in commands: print("\\n" + c)
print(f"\\n# Outputs: {{out_dir}}{{traj_name}}, scan*-summary.json, scan*.png")
print(f"#          TS guess: {{ts_guess_pdb}} ; endpoints: {{reactant_scan_pdb}} (reactant), {{product_scan_pdb}} (product)")
{sbatch(qtime="04:00:00", cpj=2, cpus="1", mem="6g", queue="gpu", gpu="'small'")}
'''
    md2 = """## *(optional)* 2-D Relaxed Scan — concerted vs stepwise diagnostic (`cowboy-qc scan2d`)

Drives the **forming** (center–forming) and **breaking** (center–breaking) bonds *independently*
on a grid around a TS guess, relaxing everything else. Reading the 2-D energy map:
- a single **diagonal ridge** ⇒ **concerted** (one TS; both bonds change together);
- a distinct **off-diagonal basin** ⇒ **stepwise** via a **pentacoordinate phosphorane** intermediate.

Diagnostic only (no endpoints) and ~grid² relaxations, so it's pricier — run it when you suspect a
stepwise mechanism (as the near-TS geometry here hints); skip it for a routine concerted step."""
    code2 = f'''\
##################################################################
###  (OPTIONAL) 2-D RELAXED SCAN   (cowboy-qc scan2d; diagnostic) ###
##################################################################
# Grid scan of the FORMING (center-forming) x BREAKING (center-breaking) bonds around a
# TS guess. Off-diagonal basin => stepwise (pentacoordinate intermediate); diagonal ridge
# => concerted. PRINTS the command (copy-paste / sbatch). ~grid^2 relaxations.

print_commands = True

### INPUTS ###
ts_guess_pdb = f"{{SCAN_DIR}}scan/ts_guess.pdb"   # from the 1-D scan above (or any TS-like geometry)

### OUTPUTS ###
out_dir_2d = f"{{SCAN_DIR}}scan2d/"

### COORDINATE BONDS (descriptors) + ENERGY MODEL ###
center, forming, breaking = {D['b_p']}, {D['f_nuc']}, {D['b_lg']}
model, head, device  = {D['model']}, {D['head']}, "cuda"
charge, multiplicity = {D['charge']}, {D['spin']}

### SCAN2D PARAMETERS ###
grid    = "5x5"     # n_a x n_b relaxed grid points (cost ~ n_a*n_b)
delta_d = 0.20      # Angstrom step per bond, out from the TS guess

### SANITY + COMMAND ###
Path(out_dir_2d).mkdir(parents=True, exist_ok=True)
cmd_2d = qcb_cmd(model, "scan2d", "--input", ts_guess_pdb, "--ts-guess", ts_guess_pdb,
                 "--bond-a", f"{{forming}},{{center}}", "--bond-b", f"{{center}},{{breaking}}",
                 "--grid", grid, "--delta-d", delta_d, "--charge", charge,
                 "--multiplicity", multiplicity, "--device", device, "--outdir", out_dir_2d)
if head: cmd_2d += ["--head", head]
if print_commands:
    print("### (optional) 2-D scan command (copy-paste / sbatch) ###")
    print(" ".join(str(x) for x in cmd_2d))
    print(f"\\n# Output: {{out_dir_2d}}  (energy grid + plot; off-diagonal basin = stepwise/pentacoordinate)")
{sbatch(qtime="04:00:00", cpj=1, cpus="1", mem="6g", queue="gpu", gpu="'small'")}
'''
    return [("markdown", md), ("code", code), ("markdown", md2), ("code", code2)]


def cell_min_endpoints(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: Minimize the Reactant & Product Endpoints (`cowboy-qc opt`)**"
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
# Each endpoint gets its OWN subdir so their opt-summary.json energies don't collide
# (the barrier-analysis cell reads both: barrier = E(TS) - E(reactant_min)).
R_min   = f"{{out_dir}}reactant/reactant_min.pdb"
P_min   = f"{{out_dir}}product/product_min.pdb"

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
    _sub = str(Path(_dst).parent)
    cmd = qcb_cmd(model, "opt", _src, "--model", model, "--charge", charge, "--multiplicity", multiplicity,
                  "--device", device, "--fix-preset", fix_preset, "--optimizer", optimizer,
                  "--fmax", fmax, "--max-steps", max_steps, "--outdir", _sub, "--output-pdb", _dst)
    if head: cmd += ["--head", head]
    commands.append(" ".join(str(x) for x in cmd))
with open(commands_file_path, "w") as f:
    f.write("\\n".join(commands) + "\\n")
print(f"# {{len(commands)}} command(s) → {{commands_file_path}}")
for c in commands: print("\\n" + c)
print(f"\\n# Outputs: {{R_min}} , {{P_min}}  (barrier = E(TS) - E(reactant_min))")
{sbatch(qtime="03:00:00", cpj=2, cpus="1", mem="6g", queue="gpu", gpu="'small'")}
'''
    return [("markdown", md), ("code", code)]


def cell_path_search(profile, step, optional=False):
    """ONE modular path-search cell: CI-NEB / GSM / FSM via a `path_method` selector."""
    D = defaults(profile)
    opt_tag = "*(optional, more rigorous than the 1-D scan)* " if optional else ""
    md = f"# **STEP {step}: {opt_tag}Path search — CI-NEB / GSM / FSM (one cell, pick `path_method`)**"
    code = f'''\
##################################################################
###  DOUBLE-ENDED PATH SEARCH  (CI-NEB | GSM | FSM — selectable)  ###
##################################################################
# ONE cell, plug-and-play method. Between a reactant and product (the MINIMIZED
# endpoints), find a TS guess. Set `path_method`:
#   'ci-neb' : climbing-image NEB (cowboy-qc neb) — robust, geodesic interp, honours --fix-preset
#   'gsm'    : Growing String (cowboy-qc gsm --method gsm) — cheaper; honours --fix-preset (pysisyphus freeze_atoms)
#   'fsm'    : Freezing String (cowboy-qc gsm --method fsm) — cheapest; honours --fix-preset (pysisyphus freeze_atoms)
# RECOMMENDED for LARGE, flexible clusters with a DISSOCIATIVE step: CI-NEB — its geodesic interpolation routes the
# path AROUND repulsive walls. GSM/FSM grow nodes by pysisyphus internal-coord interpolation (no geodesic), which can
# park a node on a high-energy wall and report a spurious 'barrier'; they suit smaller / gas-phase / concerted reactions.
# (AutoNEB / pyGSM / single-ended SE-GSM are also available via
#  `cowboy-qc ts-entry --path-method {{autoneb|pygsm-de|gsm-se}}`.)
# A 1-D scan can slice BESIDE the true saddle for an ASYNCHRONOUS step; a double-ended
# search relaxes all orthogonal DOFs, so it's a better guess there. Feed CI-NEB to
# refine-ts via --from-neb; for GSM/FSM use the emitted TS-guess structure.

### INPUTS ###
reactant_min = f"{{RELAX_MINIMIZE_DIR}}endpoints/reactant/reactant_min.pdb"   # or any reactant basin
product_min  = f"{{RELAX_MINIMIZE_DIR}}endpoints/product/product_min.pdb"     # same atom order!

### OUTPUTS ###
path_method = "ci-neb"          # 'ci-neb' | 'gsm' | 'fsm'   <-- pick the method
out_dir     = f"{{PATH_SEARCH_DIR}}{{path_method}}/"

### ENERGY MODEL ###
model, head, device = {D['model']}, {D['head']}, "cuda"
charge, multiplicity = {D['charge']}, {D['spin']}

### PATH PARAMETERS ###
n_images      = 17              # CI-NEB publication-tier (11 quicker); GSM/FSM ~15
interpolation = "geodesic"      # CI-NEB only — REQUIRED for dense/charged sites (never 'linear')
optimizer     = "fire"          # CI-NEB only
fix_preset    = "ca-only"       # honoured by CI-NEB AND GSM/FSM (GSM/FSM via pysisyphus freeze_atoms)
fmax          = 0.05

### COMMAND / SUBMIT FILE NAMES ###
commands_name      = f"{{PROJECT_NAME}}_path_{{path_method}}"
commands_file_path = os.path.join(CMDS_DIR, commands_name)

### SANITY CHECKS ###
for p in (reactant_min, product_min):
    if not Path(p).is_file():
        raise FileNotFoundError(f"endpoint not found: {{p}}  (run min-endpoints first)")
Path(out_dir).mkdir(parents=True, exist_ok=True)

### GENERATE COMMANDS ###
commands = []
if path_method == "ci-neb":
    cmd = qcb_cmd(model, "neb", reactant_min, product_min, "--model", model, "--charge", charge,
                  "--multiplicity", multiplicity, "--device", device, "--fix-preset", fix_preset,
                  "--n-images", n_images, "--interpolation", interpolation, "--optimizer", optimizer,
                  "--outdir", out_dir)
    _feed = f"refine-ts: set from_neb = '{{out_dir}}'"
elif path_method in ("gsm", "fsm"):
    # GSM/FSM now honour --fix-preset via pysisyphus freeze_atoms (grown nodes inherit it via Geometry.copy()).
    cmd = qcb_cmd(model, "gsm", reactant_min, product_min, "--method", path_method, "--model", model,
                  "--charge", charge, "--multiplicity", multiplicity, "--device", device,
                  "--fix-preset", fix_preset, "--n-images", n_images, "--fmax", fmax,
                  "--outdir", out_dir)
    _feed = f"refine-ts: feed the TS-guess structure written under {{out_dir}}"
else:
    raise ValueError(f"path_method must be ci-neb|gsm|fsm, got {{path_method!r}}")
if head: cmd += ["--head", head]
commands.append(" ".join(str(x) for x in cmd))
with open(commands_file_path, "w") as f:
    f.write("\\n".join(commands) + "\\n")
print(f"# {{len(commands)}} command(s) [{{path_method}}] → {{commands_file_path}}")
for c in commands: print("\\n" + c)
print(f"\\n# Output dir: {{out_dir}}  → {{_feed}}")
{sbatch(qtime="04:00:00", cpus="1", mem="6g", queue="gpu", gpu="'small'")}
'''
    return [("markdown", md), ("code", code)]


def _analysis_cell(step, title, body):
    """An ANALYSIS cell (no sbatch): loads a step's outputs + reports/plots."""
    md = f"# **STEP {step}: {title}**"
    return [("markdown", md), ("code", body)]


def cell_scan_analysis(profile, step):
    body = '''\
##################################################################
###  ANALYSIS: 1-D SCAN PROFILE + TS GUESS  (run after the scan) ###
##################################################################
# Reads the scan summary + trajectory, plots the profile, and reports the forward
# barrier + TS-guess frame. The TS guess is the highest INTERIOR maximum (the scan
# engine picks it — endpoints are basins in a relaxed scan). Auto-finds the outputs
# by glob, so it works for ANY scan_kind (scan-* or scan-bonddiff-*).

### INPUTS ###
scan_dir = f"{SCAN_DIR}scan/"

### ANALYSIS ###
import json, glob
from ase.io import read as _read
_trajs = sorted(glob.glob(str(Path(scan_dir) / "*-trajectory.xyz")))
_sums  = sorted(glob.glob(str(Path(scan_dir) / "*-summary.json")))
if not _trajs:
    print(f"# no *-trajectory.xyz yet at {scan_dir} — run the scan step first.")
else:
    frames = _read(_trajs[0], index=":")
    e   = [a.info["energy_eV"] if "energy_eV" in a.info else 0.0 for a in frames]
    crd = [a.info.get("scan_value", a.info.get("s_target", i)) for i, a in enumerate(frames)]
    e0  = min(e); ek = [(x - e0) * 23.0605 for x in e]      # eV -> kcal/mol, relative
    s   = json.loads(Path(_sums[0]).read_text()) if _sums else {}
    ts  = int(s.get("ts_guess_idx", max(range(len(e)), key=lambda k: e[k])))   # interior max (engine)
    print(f"scan trajectory         : {Path(_trajs[0]).name}  ({len(frames)} frames)")
    print(f"forward barrier         : {s.get('barrier_kcal', ek[ts]):.2f} kcal/mol")
    print(f"TS guess                : frame {ts} of {len(frames) - 1}  (coord {crd[ts]:+.3f})")
    print(f"reactant=frame 0 (coord {crd[0]:+.3f})  |  product=frame {len(frames) - 1} (coord {crd[-1]:+.3f})")
    if s.get("barrierless"):
        print("# WARNING: no interior barrier (monotonic) — TS guess is an endpoint; widen/shift the range.")
    try:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(crd, ek, "o-", color=nb.good_teal)
        ax.axvline(crd[ts], ls="--", color=nb.good_red, label="TS guess")
        ax.set_xlabel("scan coordinate"); ax.set_ylabel("rel. energy (kcal/mol)")
        ax.set_title("1-D relaxed scan profile"); ax.legend(); plt.tight_layout(); plt.show()
    except Exception as _ex:
        print(f"# (plot skipped: {_ex})")
'''
    return _analysis_cell(step, "Analysis — scan profile + TS guess", body)


def cell_refine_analysis(profile, step):
    body = '''\
##################################################################
###  ANALYSIS: REFINED-TS VERDICT  (run after refine-ts)        ###
##################################################################
# Reads refine-ts summary.json: did we get a genuine first-order saddle?
# (exactly one imaginary mode below the cutoff, with enough reactive-atom overlap).

### INPUTS ###
refine_summary = f"{REFINE_TS_DIR}refine/summary.json"

### ANALYSIS ###
import json
_p = Path(refine_summary)
if not _p.is_file():
    print(f"# no summary.json yet at {refine_summary} — run refine-ts first.")
else:
    r = json.loads(_p.read_text())
    verdict = "PASS ✓" if r.get("overall_pass") else "FAIL ✗"
    print(f"refine-ts verdict        : {verdict}")
    print(f"n_imag (significant)     : {r.get('n_imag')}   (want exactly 1)")
    print(f"imaginary frequency      : {r.get('imag_freq_cm')} cm^-1   (want < -50)")
    print(f"imag-mode reactive overlap: {r.get('imag_mode_overlap')}   (want >= 0.5; 0.7-0.8 for SN2)")
    print(f"E(TS)                    : {r.get('energy_eV')} eV ({r.get('energy_kcal_mol'):.2f} kcal/mol)")
    print(f"saddle backend used      : {r.get('backend_used')}")
    if not r.get("overall_pass"):
        print("\\n# FAIL → inspect: wrong/insufficient imaginary mode, or a 2nd imag mode."
              " Try a better guess (CI-NEB), --backend auto, or check for a pentacoordinate intermediate.")
'''
    return _analysis_cell(step, "Analysis — refined-TS verdict (1 imaginary mode?)", body)


def cell_barrier_summary(profile, step):
    body = '''\
##################################################################
###  ANALYSIS: ENERGY BARRIER + REACTION ENERGY + RATE          ###
##################################################################
# The headline number. Barrier height = E(TS) - E(reactant_min); reaction energy =
# E(product_min) - E(reactant_min). Reads the three opt/refine summary.json energies.
# Then an Eyring estimate of the rate constant k(T) from the barrier.

### INPUTS ###
ts_summary       = f"{REFINE_TS_DIR}refine/summary.json"
reactant_summary = f"{RELAX_MINIMIZE_DIR}endpoints/reactant/opt-summary.json"
product_summary  = f"{RELAX_MINIMIZE_DIR}endpoints/product/opt-summary.json"

### PARAMETERS ###
T = 298.15                       # K, for the Eyring rate

### ANALYSIS ###
import json, math
def _E(path):   # eV from a summary.json
    p = Path(path)
    return json.loads(p.read_text()).get("energy_eV") if p.is_file() else None
E_ts, E_r, E_p = _E(ts_summary), _E(reactant_summary), _E(product_summary)
EV2KCAL = 23.0605
if None in (E_ts, E_r):
    print("# need E(TS) and E(reactant_min). Missing:",
          [n for n, v in [("E_TS", E_ts), ("E_reactant", E_r)] if v is None],
          "— run refine-ts + min-endpoints first.")
else:
    dE_fwd = (E_ts - E_r) * EV2KCAL   # ELECTRONIC barrier ΔE‡ (no ZPE/thermal/entropy)
    print(f"forward barrier  ΔE‡(fwd) = {dE_fwd:8.2f} kcal/mol")
    if E_p is not None:
        dE_rxn = (E_p - E_r) * EV2KCAL
        dE_rev = (E_ts - E_p) * EV2KCAL
        print(f"reverse barrier  ΔE‡(rev) = {dE_rev:8.2f} kcal/mol")
        print(f"reaction energy  ΔE(rxn)  = {dE_rxn:8.2f} kcal/mol")
    # Eyring: k = (kB T / h) exp(-ΔG‡ / RT); here we use the ELECTRONIC ΔE‡ as a ΔG‡ proxy.
    kB, h, R = 1.380649e-23, 6.62607015e-34, 1.987204e-3   # J/K, J·s, kcal/mol/K
    k = (kB * T / h) * math.exp(-dE_fwd / (R * T))
    print(f"\\nEyring k({T:.0f} K)         = {k:.3e} s^-1   (ΔE‡ used as ΔG‡ proxy; add ZPE/entropy for rigor)")
    print("# NOTE: an MLFF electronic barrier; for a publication number re-evaluate at DFT (Step: ORCA).")
'''
    return _analysis_cell(step, "Analysis — energy barrier + reaction energy + Eyring rate", body)


def cell_aefm_opaa(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: *(optional, EXPERIMENTAL)* AEFM refine the TS guess (out-of-domain on metals)**"
    code = f'''\
##################################################################
###  OPTIONAL/EXPERIMENTAL: AEFM refine on the di-Zn guess       ###
##################################################################
# ⚠ EXPERIMENTAL + OUT OF DOMAIN. AEFM is trained on CHNO gas-phase organics; a di-Zn/P
# active site is OUTSIDE its training, so its weights for Zn/P are UNTRAINED and the
# refinement is UNVALIDATED. AEFM does NOT crash on metals (LEFTNet embeds Z<100) — unlike
# React-OT — so you CAN try it with --allow-out-of-domain to see if it helps the guess.
# The QM saddle+Hessian+IRC gate (refine-ts/validate-ts) remains the sole authority; treat
# any AEFM output as a guess to be re-refined, never as a result. Runs in the AEFM sidecar.

### INPUTS ###
guess_pdb = f"{{SCAN_DIR}}scan/ts_guess.pdb"     # the scan/refine TS guess (pdb)

### OUTPUTS ###
out_dir     = f"{{GENERATIVE_DIR}}aefm_opaa/"
guess_xyz   = f"{{out_dir}}ts_guess.xyz"          # AEFM reads xyz — converted below
refined_out = f"{{out_dir}}aefm_opaa_refined.xyz"

### MODEL ###
charge, multiplicity = {D['charge']}, {D['spin']}

### COMMAND / SUBMIT FILE NAMES ###
commands_name      = f"{{PROJECT_NAME}}_aefm_experimental"
commands_file_path = os.path.join(CMDS_DIR, commands_name)

### SANITY CHECKS ###
if not Path(guess_pdb).is_file():
    print(f"# TS-guess pdb not found yet: {{guess_pdb}}  (run scan/refine-ts first)")
Path(out_dir).mkdir(parents=True, exist_ok=True)

### GENERATE COMMANDS ###
# (1) convert the pdb guess -> xyz (AEFM reads xyz); (2) ts-refine with --allow-out-of-domain
# (REQUIRED — Zn/P are out of AEFM's CHNO training). Two commands, run in order.
commands = []
conv_py = f"{{out_dir}}pdb2xyz.py"
Path(conv_py).write_text(
    f"from ase.io import read, write; write(r'{{guess_xyz}}', read(r'{{guess_pdb}}'))\\n")
cmd_conv = [*APPTAINER(AEFM_SIF), "python", conv_py]
cmd_ref = sidecar_cmd(AEFM_SIF, "ts-refine", "--method", "aefm", "--ts-guess", guess_xyz,
                      "--charge", charge, "--multiplicity", multiplicity, "--allow-out-of-domain",
                      "--out", refined_out, "--outdir", out_dir)
commands.append(" ".join(str(x) for x in cmd_conv))
commands.append(" ".join(str(x) for x in cmd_ref))
with open(commands_file_path, "w") as f:
    f.write("\\n".join(commands) + "\\n")
print(f"# {{len(commands)}} command(s) → {{commands_file_path}}")
for c in commands: print("\\n" + c)
print(f"\\n# Output: {{refined_out}}  (EXPERIMENTAL — re-refine + validate with the QM gate!)")
{sbatch(qtime="02:00:00", cpj=2, cpus="1", mem="6g", queue="gpu", gpu="'small'")}
'''
    return [("markdown", md), ("code", code)]


def cell_single_ended(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: Single-Ended Search — reactant-only / SE-GSM (`cowboy-qc ts-entry`)**"
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
{sbatch(qtime="04:00:00", cpus="1", mem="6g", queue="gpu", gpu="'small'")}
'''
    return [("markdown", md), ("code", code)]


def cell_ts_entry(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: One-Call Orchestrator (`cowboy-qc ts-entry`)**"
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
{sbatch(qtime="04:00:00", cpus="1", mem="6g", queue="gpu", gpu="'small'")}
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
{sbatch(qtime="02:00:00", cpus="1", mem="6g", queue="gpu", gpu="'small'")}
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
{sbatch(qtime="02:00:00", cpus="1", mem="6g", queue="gpu", gpu="'small'")}
'''
    return [("markdown", md), ("code", code)]


def cell_refine_ts(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: Refine to a Saddle (`cowboy-qc refine-ts`)**"
    code = f'''\
##################################################################
###  REFINE-TS  (saddle search + partial-Hessian acceptance)   ###
##################################################################
# The acceptance core: dimer/Sella/pysisyphus saddle search -> partial Hessian on the
# reactive atoms -> require exactly ONE imaginary mode (< cutoff) overlapping the reaction
# coordinate -> ts_refined.pdb. Input is EITHER a TS-guess PDB or a path-search dir
# (--from-neb). --reactive-atoms accept atom tokens: descriptor (OHX-O3), serial:N, index.

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
reactive_atoms   = [{D['react_serials']}]   # atom tokens (nucleophile, center P, leaving group)
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
{sbatch(qtime="04:00:00", cpus="1", mem="6g", queue="gpu", gpu="'small'")}
'''
    return [("markdown", md), ("code", code)]


def cell_validate(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: Validate the TS — tiered Hessian (`cowboy-qc validate-ts`)**"
    code = f'''\
##################################################################
###  VALIDATE-TS  (independent tiered Hessian validation)      ###
##################################################################
# Independent of refine-ts. Tier A = reactive-atom partial Hessian; Tier B = active-region
# Hessian (catches a hidden 2nd imag mode in the metal/water shell); Tier C = Lanczos full
# check. Confirms exactly one imaginary mode on the reaction coordinate.
# --reactive-atoms accept atom tokens: descriptor (OHX-O3), serial:N, or a 0-based index.

### INPUTS ###
ts_pdb = f"{{REFINE_TS_DIR}}refine/ts_refined.pdb"   # the refined TS

### OUTPUTS ###
out_dir = f"{{TS_VALIDATION_DIR}}validate/"

### ENERGY MODEL ###
model, head, device = {D['model']}, {D['head']}, "cuda"
charge, multiplicity = {D['charge']}, {D['spin']}

### VALIDATE PARAMETERS ###
reactive_atoms   = [{D['react_serials']}]   # atom tokens (nucleophile, center P, leaving group)
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
{sbatch(qtime="04:00:00", cpus="1", mem="6g", queue="gpu", gpu="'small'")}
'''
    return [("markdown", md), ("code", code)]


def cell_irc(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: Verify IRC-like (`cowboy-qc verify-irc-like`)**"
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
{sbatch(qtime="04:00:00", cpus="1", mem="6g", queue="gpu", gpu="'small'")}
'''
    return [("markdown", md), ("code", code)]


def cell_dft(profile, step):
    D = defaults(profile)
    md = f"# **STEP {step}: (optional) DFT Reference — ORCA native NEB-TS**"
    code = f'''\
##################################################################
###  DFT REFERENCE  (cowboy-qc ts-entry --engine orca; native NEB-TS) ###
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
cmd = [*APPTAINER(MAIN_SIF, gpu=False), *CLI, "ts-entry", "--entry", entry, "--reaction-spec", spec_path,
       "--engine", "orca", "--reactant", reactant_pdb, "--product", product_pdb,
       "--charge", str(charge), "--multiplicity", str(multiplicity), "--outdir", out_dir,
       ("--execute" if EXECUTE else "--no-execute")]
commands.append(" ".join(str(x) for x in cmd))
with open(commands_file_path, "w") as f:
    f.write("\\n".join(commands) + "\\n")
print(f"# {{len(commands)}} command(s) → {{commands_file_path}}")
for c in commands: print("\\n" + c)
print(f"\\n# Outputs: {{out_dir}}  (ORCA NEB-TS job; method = {{engine_method}})")
{sbatch(qtime="24:00:00", cpus="4", mem="32g", queue="cpu", gpu="None")}
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
    _path_opt = lambda p, s: cell_path_search(p, s, optional=True)   # OPAA: optional/alt path search
    if profile == "opaa":
        seq = [cell_protonate, cell_monitor, cell_reaction_spec, cell_relax, cell_scan,
               cell_scan_analysis, cell_min_endpoints, _path_opt,
               cell_refine_ts, cell_refine_analysis, cell_validate, cell_irc,
               cell_barrier_summary, cell_aefm_opaa, cell_dft]
    else:
        seq = [cell_protonate, cell_monitor, cell_reaction_spec, cell_relax, cell_scan,
               cell_scan_analysis, cell_min_endpoints, cell_path_search, cell_single_ended,
               cell_ts_entry, cell_reactot, cell_aefm, cell_refine_ts, cell_refine_analysis,
               cell_validate, cell_irc, cell_barrier_summary, cell_dft]
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
    # ensure_ascii=False keeps unicode (→, —) literal, matching how Jupyter saves.
    Path(path).write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
    print(f"wrote {path}  ({len(nb_cells)} cells)")


if __name__ == "__main__":
    ROOT = "/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/notebooks"
    write_nb(build("general"), f"{ROOT}/general_ts_pipeline/ts_pipeline_GENERAL.ipynb")
    write_nb(build("opaa"),    f"{ROOT}/opaa_theozyme/opaa_ts_pipeline.ipynb")
