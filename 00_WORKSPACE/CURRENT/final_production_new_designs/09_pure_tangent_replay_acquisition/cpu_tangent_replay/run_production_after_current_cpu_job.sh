#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNNER="$HERE/run_cpu_tangent_replay.py"
OUTPUT="$HERE/../cpu_data/pure_tangent_cpu_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v3"

mkdir -p "$OUTPUT"

while pgrep -f '[r]un_wall_diabatic_spectral_pump_s100.py' >/dev/null; do
  printf '[wait] existing wall-diabatic CPU campaign still owns the 56 physical cores: %s\n' "$(date --iso-8601=seconds)"
  sleep 60
done

while pgrep -f '[r]un_cpu_tangent_replay.py.*--skip-input-checksums' >/dev/null; do
  printf '[wait] low-priority tangent pilots are still committing disjoint samples: %s\n' "$(date --iso-8601=seconds)"
  sleep 30
done

printf '[pilot] validate one full hard-wall Ny=24 trajectory on 28 physical cores: %s\n' "$(date --iso-8601=seconds)"
python -u "$RUNNER" \
  --construction hard \
  --ny 24 \
  --alpha-1 1 \
  --sample 0 \
  --max-new-tasks 1 \
  --workers 1 \
  --threads-per-worker 28 \
  --cpu-groups '0-27' \
  --skip-input-checksums

printf '[launch] four tangent trajectories in parallel, 14 physical cores each: %s\n' "$(date --iso-8601=seconds)"
exec python -u "$RUNNER" \
  --workers 4 \
  --threads-per-worker 14 \
  --cpu-groups '0-13;14-27;28-41;42-55'
