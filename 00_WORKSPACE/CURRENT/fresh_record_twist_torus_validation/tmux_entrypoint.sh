#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_ID="$1"
LOG_PATH="$2"
shift 2

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python "${HERE}/run_campaign.py" all --campaign-id "${CAMPAIGN_ID}" "$@" 2>&1 | tee -a "${LOG_PATH}"
