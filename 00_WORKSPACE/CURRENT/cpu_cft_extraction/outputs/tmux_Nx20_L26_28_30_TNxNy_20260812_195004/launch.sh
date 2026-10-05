#!/usr/bin/env bash
set -euo pipefail

cd /home/abhuiyan/class_A_fermionic_adaptive_circuit

RUN_ROOT="$1"
SESSION_NAME="$2"

echo "[config] session: $SESSION_NAME"
echo "[config] run root: $RUN_ROOT"
echo "[config] production: Nx=20 Ny=[26,28,30] samples=10 cycles_factor=20 cycles=[520,560,600]"
echo "[config] sample_workers=30 blas_threads=1 cpu_list=0-55 alpha=1 fock_rank=1"
echo "[step] CPU CFT L=26,28,30 T=Nx*Ny production campaign"

python cpu_cft_extraction/run_cpu_cft_sweep.py \
  --nx 20 \
  --ny 26 28 30 \
  --samples 10 \
  --cycles-factor 20 \
  --alpha 1 \
  --fock-rank 1 \
  --sample-workers 30 \
  --blas-threads 1 \
  --cpu-list 0-55 \
  --output-root "$RUN_ROOT/production" \
  --progress \
  2>&1 | tee "$RUN_ROOT/logs/production.log"

CAMPAIGN_DIR="$(find "$RUN_ROOT/production" -type f -name scalars_by_size.csv -printf '%h\n' | sort | tail -n 1)"
if [[ -z "$CAMPAIGN_DIR" ]]; then
  echo "[error] production completed without scalars_by_size.csv" >&2
  exit 1
fi

echo "$CAMPAIGN_DIR" | tee "$RUN_ROOT/logs/analysis_campaign.txt"
echo "[result] analysis campaign: $CAMPAIGN_DIR"
echo "[step] execute analysis notebook"

CAMPAIGN_DIR="$CAMPAIGN_DIR" jupyter nbconvert \
  --to notebook \
  --execute cpu_cft_extraction/analyze_cpu_cft_extraction.ipynb \
  --output-dir "$RUN_ROOT/notebook" \
  --output analyze_cpu_cft_extraction.executed.ipynb \
  --ExecutePreprocessor.timeout=900 \
  2>&1 | tee "$RUN_ROOT/logs/notebook.log"

echo "[done] L=26,28,30 T=Nx*Ny production+notebook complete: $RUN_ROOT"
