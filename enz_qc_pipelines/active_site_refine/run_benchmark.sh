#!/usr/bin/env bash
# Benchmark sweep for active-site refinement methods on the YYE/Zn₂ test.
# Submits one sbatch job per (backend × settings) combo. Each job writes a
# refined PDB into ${BENCH_DIR}/<tag>.pdb.
#
# Use:  bash run_benchmark.sh
#       (then: squeue -u $USER ; geom_score.py once jobs done)

set -e
QCB=/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry
BENCH=/home/woodbuse/testing_space/refine_bench
mkdir -p "$BENCH/logs"

DESIGN=/home/woodbuse/testing_space/align_seth_test/design.pdb
ALIGNED=/home/woodbuse/testing_space/align_seth_test/af3_pred_aligned.pdb

# Python interpreters per backend
PY_SMATHIS=/home/smathis/miniconda3/bin/python      # has ase + mace
PY_GBG=/home/gbg222/projects/enz-ts/.venv/bin/python  # has ase + mace + graph_electrostatics
PY_RFD3=/home/woodbuse/.conda/envs/rfd3/bin/python   # has biotite, will work for xtb-only runs

submit_job() {
    local tag=$1; shift
    local py=$1; shift
    local extra="$@"
    local script="$BENCH/logs/run_${tag}.sh"
    cat > "$script" <<EOF
#!/usr/bin/env bash
#SBATCH -J refine_${tag}
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 1:00:00
#SBATCH -o ${BENCH}/logs/${tag}.out
#SBATCH -e ${BENCH}/logs/${tag}.err

set -e
cd ${QCB}
${py} enzyme_design_applications/active_site_refine/refine.py \\
    ${DESIGN} ${ALIGNED} \\
    -o ${BENCH}/${tag}.pdb \\
    --ptm A/LYS/3:KCX --ligand-charge "YYE:1" \\
    --radius 4.0 --unfreeze-shell 1 \\
    ${extra}
EOF
    chmod +x "$script"
    sbatch "$script" | tail -1
}

echo "=== Submitting refinement benchmark ==="

# k_scale sweep with default angle restraints on
for ks in 0.3 1.0; do
  for backend in xtb mace-mp; do
    py=$PY_SMATHIS
    submit_job "${backend}_k${ks}_ang" "$py" \
        --backend "$backend" --gfn 0 --k-scale "$ks"
    submit_job "${backend}_k${ks}_noang" "$py" \
        --backend "$backend" --gfn 0 --k-scale "$ks" --no-angle-restraints
  done
done

# GFN-1 + GFN-2 + g-xtb (xtb-family methods, no MACE so smathis env works for parsing)
for method in "xtb --gfn 1" "xtb --gfn 2" "g-xtb --gfn 0"; do
  read -r backend args <<<"$method"
  tag=$(echo "$method" | tr ' ' '_' | tr -d '-')
  submit_job "${tag}_k1_ang" "$PY_SMATHIS" --backend "$backend" $args --k-scale 1.0
done

# MACE-OMOL (charge-aware, TS-trained — slower)
submit_job "mace_omol_k1_ang"   "$PY_SMATHIS" --backend mace-omol --k-scale 1.0
submit_job "mace_omol_k03_ang"  "$PY_SMATHIS" --backend mace-omol --k-scale 0.3

# MACE-POLAR-1-M (the user's gold standard for ionic systems)
submit_job "mace_polar_m_k1_ang"  "$PY_GBG" --backend mace-polar-m --k-scale 1.0
submit_job "mace_polar_m_k03_ang" "$PY_GBG" --backend mace-polar-m --k-scale 0.3

# Rigidity variant: fix CB so the chi pivot is mechanically locked
submit_job "xtb_k1_bb_cb" "$PY_SMATHIS" \
    --backend xtb --gfn 0 --k-scale 1.0 --rigidity backbone-cb
submit_job "macemp_k1_bb_cb" "$PY_SMATHIS" \
    --backend mace-mp --k-scale 1.0 --rigidity backbone-cb

# Polish-stage variants (small)
submit_job "xtb_k1_polish10" "$PY_SMATHIS" \
    --backend xtb --gfn 0 --k-scale 1.0 --polish-steps 10
submit_job "xtb_k1_polish30" "$PY_SMATHIS" \
    --backend xtb --gfn 0 --k-scale 1.0 --polish-steps 30

echo
echo "=== submitted; watch with: squeue -u woodbuse ==="
echo "=== outputs land in $BENCH ==="
