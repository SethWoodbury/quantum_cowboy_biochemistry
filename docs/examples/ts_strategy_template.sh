#!/bin/bash
#SBATCH --job-name=tsbench_PLACEHOLDER
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=48g
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --output=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/tsbench_PLACEHOLDER_%j.stdout
#SBATCH --error=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/tsbench_PLACEHOLDER_%j.stderr
# Generic TS-strategy benchmark job — polish + freq + irc + summary.
# Set environment variables: STRATEGY, INPUT, TARGET_PN, TARGET_PL,
#                             FMAX (default 0.05), MAX_STEPS (default 500)
set -eu
cd /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
mkdir -p logs

: ${STRATEGY:?must set STRATEGY env var}
: ${INPUT:?must set INPUT env var}
: ${TARGET_PN:=2.00}
: ${TARGET_PL:=2.25}
: ${FMAX:=0.05}
: ${MAX_STEPS:=500}
: ${RUN_FREQ:=1}
: ${RUN_IRC:=1}
: ${MODEL:=mace-polar-m}

OUT=/home/woodbuse/for/rflatent/original_theozymes/pte/mlff_outputs/ts_pipeline/benchmark_06h/$STRATEGY
mkdir -p "$OUT"

echo "=== STRATEGY=$STRATEGY  $(date) ==="
echo "host: $(hostname)"
echo "input=$INPUT"
echo "target d(P-Onuc)=$TARGET_PN  d(P-Olg)=$TARGET_PL  fmax=$FMAX  max_steps=$MAX_STEPS  model=$MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

T0=$(date +%s)

# ---- Stage 1: polish ----
# polish_ts_v2 may exit nonzero on a post-write validation hiccup; we tolerate
# that and check explicitly for the TS PDB.
set +e
apptainer exec --nv \
    --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python tools/polish_ts_v2.py \
        --input "$INPUT" \
        --out "$OUT/polish" \
        --model "$MODEL" \
        --device cuda \
        --charge 1 \
        --target-d-p-onuc "$TARGET_PN" \
        --target-d-p-olg  "$TARGET_PL" \
        --ca-rigid \
        --fmax "$FMAX" \
        --max-steps "$MAX_STEPS"
POLISH_RC=$?
set -e
echo "  polish exit_code=$POLISH_RC"
T1=$(date +%s)

TS_PDB="$OUT/polish/transition_state.pdb"
if [ ! -f "$TS_PDB" ]; then
    echo "FAIL: no TS produced; exit"
    echo "{\"strategy\": \"$STRATEGY\", \"status\": \"polish_failed\"}" > "$OUT/summary.json"
    exit 1
fi

# ---- Stage 2: freq (partial Hessian on reactive core) ----
if [ "$RUN_FREQ" = "1" ]; then
apptainer exec --nv \
    --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python tools/cowboy-qc freq "$TS_PDB" \
        --outdir "$OUT/freq" \
        --model "$MODEL" --device cuda --charge 1 \
        --indices 177 178 179 180 181 182 183 184 185 186 187 188 189 \
        --delta 0.02 --method central || echo "freq failed (non-fatal)"
fi
T2=$(date +%s)

# ---- Stage 3: IRC descent ----
if [ "$RUN_IRC" = "1" ]; then
apptainer exec --nv \
    --bind /home:/home --bind /mnt:/mnt --bind /net:/net \
    --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python tools/cowboy-qc irc "$TS_PDB" \
        --outdir "$OUT/irc" \
        --model "$MODEL" --device cuda --charge 1 \
        --no-refine-ts --step 0.1 --fmax 0.05 || echo "IRC failed (non-fatal)"
fi
T3=$(date +%s)

# ---- Stage 4: compile summary.json with common schema ----
apptainer exec \
    --bind /home:/home --bind /net:/net \
    --env "PYTHONPATH=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/deps/.local_pkgs" \
    /net/software/containers/universal.sif \
    python << PYEOF
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry')

OUT = Path("$OUT")
TS = "$TS_PDB"
EV = 23.0605

# load freq if present
freq_path = OUT / "freq" / "freq-summary.json"
freq_data = json.loads(freq_path.read_text()) if freq_path.exists() else {}

# IRC endpoints if present
from ase.io import read
irc_dir = OUT / "irc"
e_r = e_ts = e_p = None
ts_atoms = read(TS); ts_atoms.info['charge']=1
from quantum_engine.calc import make_calc
calc = make_calc("$MODEL", device="cpu", charge=1)
ts_atoms.calc = calc
e_ts = float(ts_atoms.get_potential_energy())

if (irc_dir / "reactant.xyz").exists():
    r = read(irc_dir / "reactant.xyz"); r.info['charge']=1; r.calc = calc
    e_r = float(r.get_potential_energy())
if (irc_dir / "product.xyz").exists():
    p = read(irc_dir / "product.xyz"); p.info['charge']=1; p.calc = calc
    e_p = float(p.get_potential_energy())

# Compute distances on TS
P, ON, OL = 177, 180, 188
pos = ts_atoms.positions
d_pn = float(np.linalg.norm(pos[P]-pos[ON]))
d_pl = float(np.linalg.norm(pos[P]-pos[OL]))

summary = {
    "strategy": "$STRATEGY",
    "input": "$INPUT",
    "target_d_P_Onuc": $TARGET_PN,
    "target_d_P_Olg": $TARGET_PL,
    "fmax_setting": $FMAX,
    "model": "$MODEL",
    "ts_pdb": str(TS),
    "reactant_xyz": str(irc_dir / "reactant.xyz") if (irc_dir / "reactant.xyz").exists() else None,
    "product_xyz": str(irc_dir / "product.xyz") if (irc_dir / "product.xyz").exists() else None,
    "ts_geometry": {
        "d_P_Onuc": d_pn,
        "d_P_Olg": d_pl,
        "s": d_pl - d_pn,
    },
    "energies_eV": {"R": e_r, "TS": e_ts, "P": e_p},
    "barriers_kcal": {
        "R_to_TS": (e_ts - e_r) * EV if e_r is not None else None,
        "P_to_TS": (e_ts - e_p) * EV if e_p is not None else None,
        "dE_rxn": (e_p - e_r) * EV if (e_r is not None and e_p is not None) else None,
    },
    "n_imaginary": freq_data.get("n_imaginary"),
    "imag_freq_cm": freq_data.get("frequencies_cm", [None])[0] if freq_data.get("frequencies_cm") else None,
    "walltime_min": ($T3 - $T0) / 60,
    "stage_walltime_s": {"polish": $T1 - $T0, "freq": $T2 - $T1, "irc": $T3 - $T2},
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"summary written: {OUT / 'summary.json'}")
print(f"barrier R→TS = {summary['barriers_kcal']['R_to_TS']} kcal/mol")
PYEOF
echo "=== done $(date) ==="
