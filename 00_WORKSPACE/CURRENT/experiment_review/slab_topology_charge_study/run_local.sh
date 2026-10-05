#!/usr/bin/env bash
set -euo pipefail
STUDY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$STUDY_DIR"
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
python -u run_study.py --phase run 2>&1 | tee results/v1/run.log
python -u run_study.py --phase report 2>&1 | tee -a results/v1/run.log
