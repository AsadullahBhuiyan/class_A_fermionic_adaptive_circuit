#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNNER="$HERE/run_cpu_tangent_replay.py"

# The first three samples were launched separately.  Do not expose any of
# them to the general pending-task scheduler until their completion pairs
# have either committed or the corresponding processes have exited.
while pgrep -f '[r]un_cpu_tangent_replay.py.*--cpu-groups (56-63|64-73|74-83)' >/dev/null; do
  sleep 15
done

while pgrep -f '[r]un_wall_diabatic_spectral_pump_s100.py' >/dev/null; do
  printf '[trickle-7] launching seven pending samples on sibling CPUs: %s\n' \
    "$(date --iso-8601=seconds)"
  nice -n 19 python -u "$RUNNER" \
    --max-new-tasks 7 \
    --workers 7 \
    --threads-per-worker 4 \
    --cpu-groups '56-59;60-63;64-67;68-71;72-75;76-79;80-83' \
    --skip-input-checksums
done

printf '[trickle-7] physical cores are free; handing off: %s\n' \
  "$(date --iso-8601=seconds)"
