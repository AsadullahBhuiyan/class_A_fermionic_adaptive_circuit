#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DRY_RUN=0
PREFLIGHT_ONLY=0
RESUME_ID=""
SIZE=16
SIZE_SET=0

usage() {
  echo "usage: $0 [--size EVEN_N] [--dry-run] [--preflight-only] [--resume CAMPAIGN_ID]"
  echo "environment overrides: CPU_LIST=0,2,... ALLOW_BUSY=1 TMUX_SESSION=name"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --preflight-only)
      PREFLIGHT_ONLY=1
      shift
      ;;
    --resume)
      if [[ $# -lt 2 ]]; then
        echo "--resume requires a campaign ID" >&2
        exit 2
      fi
      RESUME_ID=$2
      shift 2
      ;;
    --size)
      if [[ $# -lt 2 ]]; then
        echo "--size requires an even integer" >&2
        exit 2
      fi
      SIZE=$2
      SIZE_SET=1
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$SIZE" =~ ^[0-9]+$ ]] || (( SIZE < 4 || SIZE % 2 != 0 )); then
  echo "--size must be an even integer at least 4" >&2
  exit 2
fi

if [[ -n "$RESUME_ID" ]]; then
  CAMPAIGN_ID=$RESUME_ID
  if [[ ! -f "$HERE/results/$CAMPAIGN_ID/manifest.json" ]]; then
    echo "resume manifest not found: $HERE/results/$CAMPAIGN_ID/manifest.json" >&2
    exit 2
  fi
  RESUME_SIZE=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["geometry"]["Nx"])' \
    "$HERE/results/$CAMPAIGN_ID/campaign_config.v1.json")
  if [[ $SIZE_SET -eq 1 && "$SIZE" -ne "$RESUME_SIZE" ]]; then
    echo "resume size mismatch: requested $SIZE but campaign uses $RESUME_SIZE" >&2
    exit 2
  fi
  SIZE=$RESUME_SIZE
else
  CAMPAIGN_ID="N${SIZE}_$(date -u +%Y%m%dT%H%M%SZ)"
fi

STAGE=all
if [[ $PREFLIGHT_ONLY -eq 1 ]]; then
  STAGE=preflight
fi

CPU_LIST_VALUE=${CPU_LIST:-}
if [[ "$STAGE" == "all" && -z "$CPU_LIST_VALUE" ]]; then
  while true; do
    SELECT_ARGS=(--select-idle-cpus --limit 10 --size "$SIZE")
    if [[ ${ALLOW_BUSY:-0} == 1 ]]; then
      SELECT_ARGS+=(--allow-busy)
    fi
    CPU_LIST_VALUE=$(python "$HERE/run_campaign.py" "${SELECT_ARGS[@]}")
    CPU_COUNT=$(awk -F, '{print NF}' <<< "$CPU_LIST_VALUE")
    if [[ -n "$CPU_LIST_VALUE" && $CPU_COUNT -eq 10 ]]; then
      break
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
      CPU_LIST_VALUE="<wait-for-10-idle-same-NUMA-physical-cores>"
      break
    fi
    echo "Waiting for ten idle physical cores on one NUMA node; retrying in 60 s."
    sleep 60
  done
fi
if [[ "$STAGE" == "preflight" && -z "$CPU_LIST_VALUE" ]]; then
  CPU_LIST_VALUE=""
fi

SESSION=${TMUX_SESSION:-occupied_frame_${CAMPAIGN_ID}}
RESULT_ROOT="$HERE/results/$CAMPAIGN_ID"
LOG_PATH="$RESULT_ROOT/logs/tmux_campaign.log"
STATUS_PATH="$RESULT_ROOT/status/tmux_exit.json"

echo "tmux_session=$SESSION"
echo "campaign_log=$LOG_PATH"
echo "campaign_output=$RESULT_ROOT"
echo "geometry=${SIZE}x${SIZE}"
echo "cpu_list=$CPU_LIST_VALUE"

COMMAND=("$HERE/tmux_entrypoint.sh" "$CAMPAIGN_ID" "$SIZE" "$CPU_LIST_VALUE" "$STAGE" "$LOG_PATH" "$STATUS_PATH")
if [[ $DRY_RUN -eq 1 ]]; then
  printf 'dry_run_command='
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but was not found" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/status"
tmux new-session -d -s "$SESSION" "$(printf '%q ' "${COMMAND[@]}")"
echo "launched=1"
echo "attach_command=tmux attach -t $SESSION"
