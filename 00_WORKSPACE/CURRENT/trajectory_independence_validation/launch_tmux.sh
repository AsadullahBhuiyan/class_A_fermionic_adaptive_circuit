#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DRY_RUN=0
PREFLIGHT_ONLY=0
RESUME_ID=""

usage() {
  echo "usage: $0 [--dry-run] [--preflight-only] [--resume CAMPAIGN_ID]"
  echo "environment overrides: CPU_LIST=0,2,... ALLOW_BUSY=1 TMUX_SESSION=name"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    --resume)
      [[ $# -ge 2 ]] || { echo "--resume requires a campaign ID" >&2; exit 2; }
      RESUME_ID=$2
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -n "$RESUME_ID" ]]; then
  CAMPAIGN_ID=$RESUME_ID
  [[ -f "$HERE/results/$CAMPAIGN_ID/manifest.json" ]] || {
    echo "resume manifest not found: $HERE/results/$CAMPAIGN_ID/manifest.json" >&2
    exit 2
  }
else
  CAMPAIGN_ID="N16_$(date -u +%Y%m%dT%H%M%SZ)"
fi

STAGE=all
if [[ $PREFLIGHT_ONLY -eq 1 ]]; then
  STAGE=preflight
fi

CPU_LIST_VALUE=${CPU_LIST:-}
if [[ "$STAGE" == "all" && -z "$CPU_LIST_VALUE" ]]; then
  while true; do
    SELECT_ARGS=(--select-idle-cpus --limit 10)
    if [[ ${ALLOW_BUSY:-0} == 1 ]]; then SELECT_ARGS+=(--allow-busy); fi
    CPU_LIST_VALUE=$(python "$HERE/run_campaign.py" "${SELECT_ARGS[@]}")
    CPU_COUNT=$(awk -F, '{print NF}' <<< "$CPU_LIST_VALUE")
    if [[ -n "$CPU_LIST_VALUE" && $CPU_COUNT -eq 10 ]]; then break; fi
    if [[ $DRY_RUN -eq 1 ]]; then
      CPU_LIST_VALUE="<wait-for-10-idle-same-NUMA-physical-cores>"
      break
    fi
    echo "Waiting for ten idle physical cores on one NUMA node; retrying in 60 s."
    sleep 60
  done
fi

SESSION=${TMUX_SESSION:-trajectory_independence_${CAMPAIGN_ID}}
RESULT_ROOT="$HERE/results/$CAMPAIGN_ID"
LOG_PATH="$RESULT_ROOT/logs/tmux_campaign.log"
STATUS_PATH="$RESULT_ROOT/status/tmux_exit.json"

echo "tmux_session=$SESSION"
echo "campaign_log=$LOG_PATH"
echo "campaign_output=$RESULT_ROOT"
echo "geometry=16x16"
echo "cpu_list=$CPU_LIST_VALUE"

COMMAND=("$HERE/tmux_entrypoint.sh" "$CAMPAIGN_ID" "$CPU_LIST_VALUE" "$STAGE" "$LOG_PATH" "$STATUS_PATH")
if [[ $DRY_RUN -eq 1 ]]; then
  printf 'dry_run_command='
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

command -v tmux >/dev/null 2>&1 || { echo "tmux is required" >&2; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/status"
tmux new-session -d -s "$SESSION" "$(printf '%q ' "${COMMAND[@]}")"
echo "launched=1"
echo "attach_command=tmux attach -t $SESSION"
