#!/usr/bin/env bash
set -euo pipefail
cd /home/abhuiyan/class_A_fermionic_adaptive_circuit
RUN_ROOT="$1"
echo "[config] session: $2"
echo "[config] run root: $RUN_ROOT"
echo "[config] production: Nx=20 Ny=[30,40] samples=10 cycles_factor=5 cycles=[150,200]"
echo "[config] sample_workers=20 blas_threads=1 alpha=1 fock_rank=1"
echo "[step] CPU CFT L=30,40 T=5L production campaign"
python cpu_cft_extraction/run_cpu_cft_sweep.py \
  --nx 20 \
  --ny 30 40 \
  --samples 10 \
  --cycles-factor 5 \
  --alpha 1 \
  --fock-rank 1 \
  --sample-workers 20 \
  --blas-threads 1 \
  --output-root "$RUN_ROOT/production" \
  --progress \
  2>&1 | tee "$RUN_ROOT/logs/production.log"
CAMPAIGN_DIR="$(find "$RUN_ROOT/production" -type f -name scalars_by_size.csv -printf '%h\n' | sort | tail -n 1)"
echo "$CAMPAIGN_DIR" | tee "$RUN_ROOT/logs/analysis_campaign.txt"
echo "[result] analysis campaign: $CAMPAIGN_DIR"
echo "[step] execute analysis notebook"
mkdir -p "$RUN_ROOT/notebook"
CAMPAIGN_DIR="$CAMPAIGN_DIR" jupyter nbconvert \
  --to notebook \
  --execute cpu_cft_extraction/analyze_cpu_cft_extraction.ipynb \
  --output-dir "$RUN_ROOT/notebook" \
  --output analyze_cpu_cft_extraction.executed.ipynb \
  --ExecutePreprocessor.timeout=900 \
  2>&1 | tee "$RUN_ROOT/logs/notebook.log"
echo "[done] L=30,40 T=5L production+notebook complete: $RUN_ROOT"
