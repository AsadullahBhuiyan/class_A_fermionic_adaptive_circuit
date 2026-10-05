#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT"
RUNNER="$REPO_ROOT/tangent_cocycle_flux_snapshot/run_flux_cocycle_snapshot.py"
CONFIG="$REPO_ROOT/tangent_cocycle_flux_snapshot/reference_config.json"
OUTPUT_ROOT="$REPO_ROOT/tangent_cocycle_flux_snapshot/outputs"
CPU_LIST="${CPU_LIST:-0-55}"
THREAD_COUNT="${THREAD_COUNT:-56}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION="flux_cocycle_N20x24_B48_C48_P21_${STAMP}"
LOG_DIR="$OUTPUT_ROOT/tmux_logs"
LOG_PATH="$LOG_DIR/${SESSION}.log"

mkdir -p "$LOG_DIR"
if [[ "${ALLOW_BUSY:-0}" != "1" ]] && pgrep -f \
  '[r]un_cpu_cft_sweep.py|topological_frustration_diagnostics/[r]un_cpu.py' \
  >/dev/null; then
  echo "Refusing to launch while a known CPU-heavy campaign is active." >&2
  echo "Wait for it to finish, or set ALLOW_BUSY=1 to override deliberately." >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

COMMAND="cd '$REPO_ROOT' && export OMP_NUM_THREADS='$THREAD_COUNT' OPENBLAS_NUM_THREADS='$THREAD_COUNT' MKL_NUM_THREADS='$THREAD_COUNT' NUMEXPR_NUM_THREADS='$THREAD_COUNT'; taskset -c '$CPU_LIST' python '$RUNNER' --config '$CONFIG' --output-root '$OUTPUT_ROOT' 2>&1 | tee '$LOG_PATH'"
tmux new-session -d -s "$SESSION" "$COMMAND"
echo "Started tmux session: $SESSION"
echo "Log: $LOG_PATH"
