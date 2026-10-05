#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PACKAGE_ROOT/../../.." && pwd)"
RUNNER="$PACKAGE_ROOT/run_nx_wall_purification_convergence_cpu.py"
ANALYZER="$PACKAGE_ROOT/analyze_nx_wall_purification_convergence.py"
RESULTS_ROOT="${RESULTS_ROOT:-$PACKAGE_ROOT/results}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$PACKAGE_ROOT/analysis_outputs}"
PROFILE="${PROFILE:-pilot}"
MAX_WORKERS="${MAX_WORKERS:-48}"
WORKERS="${WORKERS:-auto}"
CPU_LIST="${CPU_LIST:-}"
CPU_SELECTION="${CPU_SELECTION:-free}"
TARGET_CYCLES="${TARGET_CYCLES:-100}"
TARGET_SAMPLES="${TARGET_SAMPLES:-}"
CHECKPOINT_STRIDE="${CHECKPOINT_STRIDE:-5}"
BOOTSTRAP_COUNT="${BOOTSTRAP_COUNT:-2000}"
STAMP="$(date +%Y%m%d_%H%M%S)"

case "$PROFILE" in
  smoke)
    CAMPAIGN_ID="${CAMPAIGN_ID:-Nx20-24-28_Ny20_nsh1_dwtrunc1_init-maxmix_S25_C100_smoke}"
    TARGET_CYCLES=3
    TARGET_SAMPLES=1
    RUN_PROFILE_ARGS=(--smoke)
    RUN_ANALYSIS=0
    RUN_SMOKE_TESTS=1
    ;;
  pilot)
    CAMPAIGN_ID="${CAMPAIGN_ID:-Nx20-24-28_Ny20_nsh1_dwtrunc1_init-maxmix_S25_C100}"
    TARGET_SAMPLES="${TARGET_SAMPLES:-25}"
    RUN_PROFILE_ARGS=()
    RUN_ANALYSIS=1
    RUN_SMOKE_TESTS=0
    ;;
  topup)
    CAMPAIGN_ID="${CAMPAIGN_ID:-Nx20-24-28_Ny20_nsh1_dwtrunc1_init-maxmix_S25_C100}"
    TARGET_SAMPLES="${TARGET_SAMPLES:-50}"
    RUN_PROFILE_ARGS=()
    RUN_ANALYSIS=1
    RUN_SMOKE_TESTS=0
    ;;
  *)
    echo "PROFILE must be smoke, pilot, or topup; got: $PROFILE" >&2
    exit 2
    ;;
esac

if [[ ! "$MAX_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_WORKERS must be a positive integer." >&2
  exit 2
fi
if [[ "$WORKERS" != "auto" && ! "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS must be 'auto' or a positive integer." >&2
  exit 2
fi
if [[ -n "$CPU_LIST" && ! "$CPU_LIST" =~ ^[0-9,-]+$ ]]; then
  echo "CPU_LIST must use taskset syntax such as 0-47 or 0-23,56-79." >&2
  exit 2
fi

if [[ "${ALLOW_BUSY:-0}" != "1" ]] && pgrep -f \
  '[r]un_nx_wall_purification_convergence_cpu.py|[r]un_purification_charge_sharpening_alpha_sweep_cpu.py|[r]un_cpu_cft_sweep.py|topological_frustration_diagnostics/[r]un_cpu.py|mean_channel_lindblad_cpu_campaign/.+[r]un' \
  >/dev/null; then
  echo "Refusing to launch while a known CPU-heavy campaign is active." >&2
  echo "Wait for it to finish, or set ALLOW_BUSY=1 to override deliberately." >&2
  exit 1
fi

SESSION="nxwall_${PROFILE}_Ny20_Nx20-24-28_${STAMP}"
LOG_DIR="$RESULTS_ROOT/campaigns/$CAMPAIGN_ID/tmux_logs"
LOG_PATH="$LOG_DIR/${SESSION}.log"
MANIFEST_PATH="$RESULTS_ROOT/campaigns/$CAMPAIGN_ID/campaign_manifest.json"
mkdir -p "$LOG_DIR"

RUN_ARGS=(
  python "$RUNNER"
  --campaign-id "$CAMPAIGN_ID"
  --output-root "$RESULTS_ROOT"
  --nx-values 20 24 28
  --ny 20
  --cycles "$TARGET_CYCLES"
  --samples "$TARGET_SAMPLES"
  --checkpoint-stride "$CHECKPOINT_STRIDE"
  --workers "$WORKERS"
  --max-workers "$MAX_WORKERS"
  --cpu-selection "$CPU_SELECTION"
  "${RUN_PROFILE_ARGS[@]}"
)
if [[ -n "$CPU_LIST" ]]; then
  RUN_ARGS=(taskset -c "$CPU_LIST" "${RUN_ARGS[@]}")
fi
printf -v RUN_COMMAND '%q ' "${RUN_ARGS[@]}"

if [[ "$RUN_ANALYSIS" == "1" ]]; then
  ANALYSIS_ARGS=(
    python "$ANALYZER"
    --campaign-id "$CAMPAIGN_ID"
    --results-root "$RESULTS_ROOT"
    --output-root "$ANALYSIS_ROOT"
    --bootstrap-count "$BOOTSTRAP_COUNT"
  )
  printf -v ANALYSIS_COMMAND '%q ' "${ANALYSIS_ARGS[@]}"
  PIPELINE="$RUN_COMMAND && $ANALYSIS_COMMAND"
elif [[ "$RUN_SMOKE_TESTS" == "1" ]]; then
  SMOKE_TEST_ARGS=(
    python -m pytest -q
    "$PACKAGE_ROOT/tests/test_nx_wall_purification_convergence.py"
    -k "standard_domain_wall_geometry or cycle_zero_maxmix_observables_and_contour_sum or interrupted_resume_matches_uninterrupted_run"
  )
  printf -v SMOKE_TEST_COMMAND '%q ' "${SMOKE_TEST_ARGS[@]}"
  PIPELINE="$RUN_COMMAND && $SMOKE_TEST_COMMAND"
else
  PIPELINE="$RUN_COMMAND"
fi

printf -v QUOTED_REPO '%q' "$REPO_ROOT"
printf -v QUOTED_LOG '%q' "$LOG_PATH"
COMMAND="cd $QUOTED_REPO && export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; ( $PIPELINE ) 2>&1 | tee $QUOTED_LOG"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" "$COMMAND"

echo "Started tmux session: $SESSION"
echo "Profile: $PROFILE"
echo "Campaign: $CAMPAIGN_ID"
echo "Log: $LOG_PATH"
echo
echo "Attach:  tmux attach -t '$SESSION'"
echo "Follow:  tail -f '$LOG_PATH'"
echo "Status:  jq '{status, results, failures}' '$MANIFEST_PATH'"
echo "Resume:  PROFILE='$PROFILE' CAMPAIGN_ID='$CAMPAIGN_ID' TARGET_CYCLES='$TARGET_CYCLES' TARGET_SAMPLES='$TARGET_SAMPLES' '$PACKAGE_ROOT/launch_tmux.sh'"
