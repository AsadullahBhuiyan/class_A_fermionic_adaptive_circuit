#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PACKAGE_ROOT/../../.." && pwd)"
CONFIG="${CONFIG:-$PACKAGE_ROOT/campaign_config.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PACKAGE_ROOT/outputs/maxmix_manybody_lyapunov_nx16_ny20to30_hard-soft_s100_4ny_cpu_v1}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$PACKAGE_ROOT/analysis_outputs/maxmix_manybody_lyapunov_nx16_ny20to30_hard-soft_s100_4ny_cpu_v1}"
WORKERS="${WORKERS:-auto}"
MAX_WORKERS="${MAX_WORKERS:-16}"
MAX_NEW_TASKS="${MAX_NEW_TASKS:-}"
RUN_ANALYSIS="${RUN_ANALYSIS:-0}"
CPU_LIST="${CPU_LIST:-}"
NICE_LEVEL="${NICE_LEVEL:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION="${SESSION:-maxmix_lyapunov_cpu_${STAMP}}"
LOG_DIR="$OUTPUT_ROOT/tmux_logs"
LOG_PATH="$LOG_DIR/${SESSION}.log"
mkdir -p "$LOG_DIR"

if [[ "$WORKERS" != "auto" && ! "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS must be auto or a positive integer." >&2
  exit 2
fi
if [[ ! "$MAX_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_WORKERS must be a positive integer." >&2
  exit 2
fi
if [[ -n "$CPU_LIST" && ! "$CPU_LIST" =~ ^[0-9,-]+$ ]]; then
  echo "CPU_LIST must use taskset syntax, for example 0-15 or 0-7,16-23." >&2
  exit 2
fi
if [[ ! "$NICE_LEVEL" =~ ^[0-9]+$ ]] || (( NICE_LEVEL > 19 )); then
  echo "NICE_LEVEL must be an integer from 0 through 19." >&2
  exit 2
fi

RUN=(
  python -u "$PACKAGE_ROOT/run_campaign.py"
  --config "$CONFIG"
  --output-root "$OUTPUT_ROOT"
  --workers "$WORKERS"
  --max-workers "$MAX_WORKERS"
)
if [[ -n "$MAX_NEW_TASKS" ]]; then
  RUN+=(--max-new-tasks "$MAX_NEW_TASKS")
fi
if [[ -n "$CPU_LIST" ]]; then
  RUN=(taskset -c "$CPU_LIST" "${RUN[@]}")
fi
if (( NICE_LEVEL > 0 )); then
  RUN=(nice -n "$NICE_LEVEL" "${RUN[@]}")
fi
printf -v RUN_COMMAND '%q ' "${RUN[@]}"

PIPELINE="$RUN_COMMAND"
if [[ "$RUN_ANALYSIS" == "1" ]]; then
  ANALYSIS=(
    python -u "$PACKAGE_ROOT/analyze_campaign.py"
    --config "$CONFIG"
    --results-root "$OUTPUT_ROOT"
    --output-root "$ANALYSIS_ROOT"
  )
  printf -v ANALYSIS_COMMAND '%q ' "${ANALYSIS[@]}"
  PIPELINE="$PIPELINE && $ANALYSIS_COMMAND"
fi

printf -v QUOTED_REPO '%q' "$REPO_ROOT"
printf -v QUOTED_LOG '%q' "$LOG_PATH"
COMMAND="cd $QUOTED_REPO && export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; ( $PIPELINE ) 2>&1 | tee $QUOTED_LOG"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" "$COMMAND"

echo "Started local CPU campaign in tmux: $SESSION"
echo "Output: $OUTPUT_ROOT"
echo "Log:    $LOG_PATH"
echo "Attach: tmux attach -t '$SESSION'"
echo "Follow: tail -f '$LOG_PATH'"
echo "Report: python '$PACKAGE_ROOT/run_campaign.py' --config '$CONFIG' --output-root '$OUTPUT_ROOT' --report-only"
echo "Resume:  '$PACKAGE_ROOT/launch_tmux.sh'"
