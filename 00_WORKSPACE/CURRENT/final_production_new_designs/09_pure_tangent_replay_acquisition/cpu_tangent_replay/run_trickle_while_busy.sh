#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  printf 'usage: %s CURRENT_SESSION START SAMPLE_STRIDE CPU_GROUP THREADS\n' "$0" >&2
  exit 2
fi

CURRENT_SESSION=$1
START=$2
STRIDE=$3
CPU_GROUP=$4
THREADS=$5
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNNER="$HERE/run_cpu_tangent_replay.py"

# Let the already-running first sample retain its CPU group.  If it failed,
# the first loop iteration below deterministically retries the same sample.
while tmux has-session -t "$CURRENT_SESSION" 2>/dev/null; do
  sleep 15
done

for ((sample=START; sample<100; sample+=STRIDE)); do
  # Trickle work exists only to exploit sibling threads while the wall-pump
  # campaign owns the physical cores.  The main production launcher takes
  # over after the current restart unit finishes once that campaign exits.
  if ! pgrep -f '[r]un_wall_diabatic_spectral_pump_s100.py' >/dev/null; then
    printf '[trickle] physical cores are free; handing off before sample %d: %s\n' \
      "$sample" "$(date --iso-8601=seconds)"
    exit 0
  fi
  printf '[trickle] sample=%d cpu_group=%s threads=%s: %s\n' \
    "$sample" "$CPU_GROUP" "$THREADS" "$(date --iso-8601=seconds)"
  nice -n 19 python -u "$RUNNER" \
    --construction hard \
    --ny 24 \
    --alpha-1 1 \
    --sample "$sample" \
    --max-new-tasks 1 \
    --workers 1 \
    --threads-per-worker "$THREADS" \
    --cpu-groups "$CPU_GROUP" \
    --skip-input-checksums
done
