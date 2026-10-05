#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${SCRIPT_DIR}/results/N20_24_28_32x24_wall_diabatic_spectral_pump_s100_v1"
CONTROL_SESSION="wall_diabatic_controls_s100_v1"
SMOKE_SESSION="wall_diabatic_smoke_s100_v1"

echo "[gate queue] waiting for exact controls and the 16-cell smoke"
while tmux has-session -t "${CONTROL_SESSION}" 2>/dev/null \
   || tmux has-session -t "${SMOKE_SESSION}" 2>/dev/null; do
  sleep 30
done

if ! grep -Fq "[complete] requested control/sensitivity stage finished" \
    "${OUTPUT}/controls_tmux.log"; then
  echo "[gate queue] exact-control process did not report successful completion" >&2
  exit 2
fi
if ! grep -Fq "[complete] all selected wall-diabatized endpoint tasks finished" \
    "${OUTPUT}/smoke_tmux.log"; then
  echo "[gate queue] smoke process did not report successful completion" >&2
  exit 2
fi

smoke_pairs="$({
  find "${OUTPUT}/pump" -type f -name 'sample_000.completion.json' -print 2>/dev/null || true
} | wc -l)"
if [[ "${smoke_pairs}" -ne 16 ]]; then
  echo "[gate queue] expected 16 durable sample-0 smoke pairs, found ${smoke_pairs}" >&2
  exit 2
fi

python -u "${SCRIPT_DIR}/validate_wall_diabatic_controls.py" \
  --output-root "${OUTPUT}"
echo "[gate queue] controls and smoke passed; starting the resumable full pipeline"
exec bash "${SCRIPT_DIR}/wall_diabatic_local_tmux_entrypoint.sh"
