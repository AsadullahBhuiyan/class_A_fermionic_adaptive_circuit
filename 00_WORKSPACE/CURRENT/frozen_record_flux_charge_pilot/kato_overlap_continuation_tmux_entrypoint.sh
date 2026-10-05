#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPU_LIST="${CPU_LIST:-28-55}"
WORKERS="${WORKERS:-28}"
SMOKE_WORKERS="${SMOKE_WORKERS:-8}"
CAMPAIGN_ROOT="${PROJECT_DIR}/results/N20x24_N24x24_kato_overlap_continuation_s25_v1"
SMOKE_ROOT="${PROJECT_DIR}/results/N20x24_N24x24_kato_overlap_continuation_s25_v1_smoke_tol1e8"
TIGHT_ROOT="${PROJECT_DIR}/results/N20x24_N24x24_kato_overlap_continuation_s25_v1_smoke_tol2p5e9"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TASK_ARGS=()
for size in N20x24 N24x24; do
  for wall in soft hard; do
    for direction in ccw cw; do
      TASK_ARGS+=(--task-id "kato_${size}_${wall}_${direction}_sample_000")
    done
  done
done

echo "[stage] sample-0 smoke at adaptive tolerance 1e-8"
taskset -c "${CPU_LIST}" python -u "${PROJECT_DIR}/run_kato_overlap_continuation_s25.py" \
  --output-root "${SMOKE_ROOT}" --workers "${SMOKE_WORKERS}" "${TASK_ARGS[@]}"

echo "[stage] sample-0 smoke at tightened adaptive tolerance 2.5e-9"
taskset -c "${CPU_LIST}" python -u "${PROJECT_DIR}/run_kato_overlap_continuation_s25.py" \
  --output-root "${TIGHT_ROOT}" --workers "${SMOKE_WORKERS}" \
  --adaptive-tolerance 2.5e-9 "${TASK_ARGS[@]}"

echo "[stage] compare sample-0 smoke paths"
taskset -c "${CPU_LIST}" python -u "${PROJECT_DIR}/validate_kato_overlap_smoke.py" \
  --ordinary-root "${SMOKE_ROOT}" --tight-root "${TIGHT_ROOT}"

echo "[stage] production 200-path adaptive Kato campaign"
taskset -c "${CPU_LIST}" python -u "${PROJECT_DIR}/run_kato_overlap_continuation_s25.py" \
  --output-root "${CAMPAIGN_ROOT}" --workers "${WORKERS}"

echo "[stage] verified analysis"
taskset -c "${CPU_LIST}" python -u "${PROJECT_DIR}/analyze_kato_overlap_continuation_s25.py" \
  --output-root "${CAMPAIGN_ROOT}"

echo "[complete] adaptive Kato campaign and analysis finished"
