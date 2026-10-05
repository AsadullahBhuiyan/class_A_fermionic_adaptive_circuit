#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="wall_diabatic_spectral_pump_s100_v1"
OUTPUT="${SCRIPT_DIR}/results/N20_24_28_32x24_wall_diabatic_spectral_pump_s100_v1"
LOG="${OUTPUT}/tmux.log"
ENTRYPOINT="${SCRIPT_DIR}/wall_diabatic_local_tmux_entrypoint.sh"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
if [[ ! -x "${ENTRYPOINT}" ]]; then
  echo "entrypoint is missing or not executable: ${ENTRYPOINT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT}"
tmux new-session -d -s "${SESSION}" "bash '${ENTRYPOINT}' 2>&1 | tee -a '${LOG}'"
echo "launched ${SESSION}; log=${LOG}"
