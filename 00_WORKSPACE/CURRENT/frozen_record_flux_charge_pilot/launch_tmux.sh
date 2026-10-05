#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../.." && pwd)"
CONFIG="${CONFIG:-${HERE}/campaign_config.v1.json}"
CAMPAIGN_ID="${CAMPAIGN_ID:-N16x20_frozen_flux_charge_v1}"
SESSION="${SESSION:-frozen_flux_charge_${CAMPAIGN_ID}}"
CPU_LIST="${CPU_LIST:-0-55}"
WORKERS="${WORKERS:-56}"
BLAS_THREADS="${BLAS_THREADS:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HERE}/results}"
LOG_DIR="${OUTPUT_ROOT}/${CAMPAIGN_ID}/logs"
LOG_PATH="${LOG_DIR}/tmux.log"

EXTRA=()
for argument in "$@"; do
  case "${argument}" in
    --dry-run|--resume) EXTRA+=("${argument}") ;;
    *) echo "Unsupported launcher argument: ${argument}" >&2; exit 2 ;;
  esac
done

if [[ " ${EXTRA[*]} " == *" --dry-run "* ]]; then
  cd "${REPO_ROOT}"
  exec python -u "${HERE}/run_campaign.py" all \
    --config "${CONFIG}" \
    --output-root "${OUTPUT_ROOT}" \
    --campaign-id "${CAMPAIGN_ID}" \
    --cpu-list "${CPU_LIST}" \
    --workers "${WORKERS}" \
    --blas-threads "${BLAS_THREADS}" \
    --dry-run
fi

mkdir -p "${LOG_DIR}"
COMMAND="cd $(printf '%q' "${REPO_ROOT}") && exec bash $(printf '%q' "${HERE}/tmux_entrypoint.sh")"
COMMAND+=" $(printf '%q' "${CONFIG}") $(printf '%q' "${OUTPUT_ROOT}")"
COMMAND+=" $(printf '%q' "${CAMPAIGN_ID}") $(printf '%q' "${CPU_LIST}")"
COMMAND+=" $(printf '%q' "${WORKERS}") $(printf '%q' "${BLAS_THREADS}")"
COMMAND+=" $(printf '%q' "${LOG_PATH}")"
for argument in "${EXTRA[@]}"; do
  COMMAND+=" $(printf '%q' "${argument}")"
done

tmux new-session -d -s "${SESSION}" "${COMMAND}"
echo "session=${SESSION}"
echo "log=${LOG_PATH}"
echo "output=${OUTPUT_ROOT}/${CAMPAIGN_ID}"
echo "attach: tmux attach -t ${SESSION}"
echo "follow: tail -f ${LOG_PATH}"
