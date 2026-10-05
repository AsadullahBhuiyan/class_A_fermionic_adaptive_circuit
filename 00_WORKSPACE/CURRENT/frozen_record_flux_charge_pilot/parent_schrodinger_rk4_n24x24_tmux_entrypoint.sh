#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot"
PRIMARY_SESSION="parent_schrodinger_rk4_N20x24_s50_tau1e4_v1"
REFINEMENT_SESSION="parent_schrodinger_rk4_dt_half_followup"
OUTPUT_ROOT="$PROJECT_ROOT/results/N24x24_parent_schrodinger_rk4_s50_tau1e4_v1"
LOG_ROOT="$OUTPUT_ROOT/logs"
LOG_PATH="$LOG_ROOT/tmux_parent_schrodinger_rk4_n24x24_s50.log"

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[queue] N24x24 width comparison queued at $(date --iso-8601=seconds)"
echo "[queue] waiting for $PRIMARY_SESSION and $REFINEMENT_SESSION"
while tmux has-session -t "$PRIMARY_SESSION" 2>/dev/null || tmux has-session -t "$REFINEMENT_SESSION" 2>/dev/null; do
    sleep 60
done

echo "[queue] dependencies exited; runner will independently verify 100 primary paths and the step-halving gate"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

taskset -c 28-55 python -u "$PROJECT_ROOT/run_parent_schrodinger_rk4_n24x24_s50.py" \
    run \
    --config "$PROJECT_ROOT/campaign_config.parent_schrodinger_rk4_n24x24_s50_tau1e4_v1.json" \
    --output-root "$OUTPUT_ROOT" \
    --workers 28 \
    --resume \
    --analyze

echo "[queue] N24x24 width comparison completed at $(date --iso-8601=seconds)"
