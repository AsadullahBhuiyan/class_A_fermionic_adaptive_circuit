#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CAMPAIGN_ID="${CAMPAIGN_ID:-N16_T16_${STAMP}}"
SESSION="${SESSION:-twist_torus_${STAMP}}"
LOG_DIR="${HERE}/results/${CAMPAIGN_ID}/logs"
LOG_PATH="${LOG_DIR}/tmux.log"

mkdir -p "${LOG_DIR}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

EXTRA=()
for argument in "$@"; do
  case "${argument}" in
    --dry-run|--preflight-only|--resume) EXTRA+=("${argument}") ;;
    *) echo "Unsupported launcher argument: ${argument}" >&2; exit 2 ;;
  esac
done

if [[ " ${EXTRA[*]} " == *" --dry-run "* ]]; then
  cd "${REPO_ROOT}"
  exec python "${HERE}/run_campaign.py" all --campaign-id "${CAMPAIGN_ID}" "${EXTRA[@]}"
fi

COMMAND="cd $(printf '%q' "${REPO_ROOT}") && exec bash $(printf '%q' "${HERE}/tmux_entrypoint.sh") $(printf '%q' "${CAMPAIGN_ID}") $(printf '%q' "${LOG_PATH}")"
for argument in "${EXTRA[@]}"; do
  COMMAND+=" $(printf '%q' "${argument}")"
done

tmux new-session -d -s "${SESSION}" "${COMMAND}"
echo "session=${SESSION}"
echo "log=${LOG_PATH}"
echo "output=${HERE}/results/${CAMPAIGN_ID}"
echo "attach: tmux attach -t ${SESSION}"
