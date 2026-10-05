#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNNER="$HERE/run_cpu_tangent_replay.py"

pending_count() {
  python -u "$RUNNER" --report-only --skip-input-checksums 2>/dev/null |
    python -c 'import json,sys; print(int(json.load(sys.stdin)["pending"]))'
}

printf '[audit] checksum-verifying all acquisition batches before production: %s\n' \
  "$(date --iso-8601=seconds)"
python -u "$RUNNER" --report-only

pending=$(pending_count)
if [[ "$pending" -gt 0 ]]; then
  if pgrep -f '[r]un_campaign.py.*maxmix_manybody_lyapunov' >/dev/null; then
    workers=6
    groups='0-13;14-27;28-41;42-55;56-69;70-83'
    printf '[queue] pending=%d; continuously refilling six tangent workers alongside the existing 28-thread campaign: %s\n' \
      "$pending" "$(date --iso-8601=seconds)"
  else
    workers=8
    groups='0-13;14-27;28-41;42-55;56-69;70-83;84-97;98-111'
    printf '[queue] pending=%d; continuously refilling eight tangent workers across all logical CPUs: %s\n' \
      "$pending" "$(date --iso-8601=seconds)"
  fi

  python -u "$RUNNER" \
    --workers "$workers" \
    --threads-per-worker 14 \
    --cpu-groups "$groups" \
    --skip-input-checksums
fi

printf '[audit] all tasks present; performing final full checksum audit: %s\n' \
  "$(date --iso-8601=seconds)"
python -u "$RUNNER" --report-only
printf '[complete] verified all 1,200 CPU tangent replay outputs: %s\n' \
  "$(date --iso-8601=seconds)"
