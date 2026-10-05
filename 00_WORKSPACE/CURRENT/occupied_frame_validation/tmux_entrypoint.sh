#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 CAMPAIGN_ID SIZE CPU_LIST STAGE LOG_PATH STATUS_PATH" >&2
  exit 2
fi

CAMPAIGN_ID=$1
SIZE=$2
CPU_LIST_VALUE=$3
STAGE=$4
LOG_PATH=$5
STATUS_PATH=$6
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$HERE/../../.." && pwd)

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$HERE/results/$CAMPAIGN_ID/.mplconfig"
mkdir -p "$(dirname "$LOG_PATH")" "$MPLCONFIGDIR"

exec > >(tee -a "$LOG_PATH") 2>&1
echo "campaign_id=$CAMPAIGN_ID"
echo "size=${SIZE}x${SIZE}"
echo "stage=$STAGE"
echo "cpu_list=$CPU_LIST_VALUE"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

set +e
if [[ "$STAGE" == "preflight" ]]; then
  python "$HERE/run_campaign.py" init --campaign-id "$CAMPAIGN_ID" --size "$SIZE"
  RETURN_CODE=$?
  if [[ $RETURN_CODE -eq 0 ]]; then
    python "$HERE/run_campaign.py" preflight --campaign-id "$CAMPAIGN_ID" --cpu-list "$CPU_LIST_VALUE" --size "$SIZE"
    RETURN_CODE=$?
  fi
else
  taskset -c "$CPU_LIST_VALUE" \
    python "$HERE/run_campaign.py" all \
      --campaign-id "$CAMPAIGN_ID" \
      --size "$SIZE" \
      --cpu-list "$CPU_LIST_VALUE" \
      --max-workers 10
  RETURN_CODE=$?
fi
set -e

printf '{"returncode":%d,"finished_utc":"%s"}\n' \
  "$RETURN_CODE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_PATH.tmp"
mv "$STATUS_PATH.tmp" "$STATUS_PATH"
echo "finished_returncode=$RETURN_CODE"
exit "$RETURN_CODE"
