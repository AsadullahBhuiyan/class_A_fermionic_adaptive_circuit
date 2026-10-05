#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${OUTPUT_ROOT:-$ROOT/cpu_cft_extraction/outputs/tmux_cft_${STAMP}}"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

NX="${NX:-20}"
PROD_NY="${PROD_NY:-20 30 40}"
PROD_SAMPLES="${PROD_SAMPLES:-10}"
CYCLES_FACTOR="${CYCLES_FACTOR:-2}"
ALPHA="${ALPHA:-1}"
FOCK_RANK="${FOCK_RANK:-1}"
BLAS_THREADS="${BLAS_THREADS:-1}"
CPU_LIST="${CPU_LIST:-}"
SMOKE_WORKERS="${SMOKE_WORKERS:-2}"
BENCHMARK_WORKERS="${BENCHMARK_WORKERS:-8 14 20 28 40 56}"
BENCHMARK_SAMPLES="${BENCHMARK_SAMPLES:-30}"
BENCHMARK_NY="${BENCHMARK_NY:-20}"
BENCHMARK_CYCLES="${BENCHMARK_CYCLES:-1}"
RUN_PRODUCTION="${RUN_PRODUCTION:-1}"
RUN_NOTEBOOK="${RUN_NOTEBOOK:-1}"

export OMP_NUM_THREADS="$BLAS_THREADS"
export MKL_NUM_THREADS="$BLAS_THREADS"
export OPENBLAS_NUM_THREADS="$BLAS_THREADS"
export NUMEXPR_NUM_THREADS="$BLAS_THREADS"

echo "[config] run root: $RUN_ROOT"
echo "[config] production: Nx=$NX Ny=[$PROD_NY] samples=$PROD_SAMPLES cycles_factor=$CYCLES_FACTOR"
echo "[config] benchmark: Ny=$BENCHMARK_NY samples=$BENCHMARK_SAMPLES cycles=$BENCHMARK_CYCLES workers=[$BENCHMARK_WORKERS]"
echo "[config] BLAS threads per worker: $BLAS_THREADS"
if [[ -n "$CPU_LIST" ]]; then
  echo "[config] CPU affinity list: $CPU_LIST"
fi

echo "[step] Python syntax check"
python -m py_compile \
  cpu_cft_extraction/run_cpu_cft_sweep.py \
  cpu_cft_extraction/cft_analysis.py \
  cpu_cft_extraction/test_cft_analysis.py \
  2>&1 | tee "$LOG_DIR/py_compile.log"

echo "[step] unit tests"
pytest -q cpu_cft_extraction/test_cft_analysis.py \
  2>&1 | tee "$LOG_DIR/tests.log"

echo "[step] parallel smoke campaign"
smoke_args=(
  cpu_cft_extraction/run_cpu_cft_sweep.py
  --smoke
  --sample-workers "$SMOKE_WORKERS"
  --blas-threads "$BLAS_THREADS"
  --output-root "$RUN_ROOT/smoke"
  --progress
)
if [[ -n "$CPU_LIST" ]]; then
  smoke_args+=(--cpu-list "$CPU_LIST")
fi
python "${smoke_args[@]}" 2>&1 | tee "$LOG_DIR/smoke.log"

echo "[step] benchmark core tuning"
benchmark_args=(
  cpu_cft_extraction/run_cpu_cft_sweep.py
  --nx "$NX"
  --benchmark-workers $BENCHMARK_WORKERS
  --benchmark-samples "$BENCHMARK_SAMPLES"
  --benchmark-ny "$BENCHMARK_NY"
  --benchmark-cycles "$BENCHMARK_CYCLES"
  --blas-threads "$BLAS_THREADS"
  --output-root "$RUN_ROOT/benchmark"
)
if [[ -n "$CPU_LIST" ]]; then
  benchmark_args+=(--cpu-list "$CPU_LIST")
fi
python "${benchmark_args[@]}" 2>&1 | tee "$LOG_DIR/benchmark.log"

BENCH_DIR="$(find "$RUN_ROOT/benchmark" -maxdepth 1 -type d -name 'benchmark_*' | sort | tail -n 1)"
BENCH_SUMMARY="$BENCH_DIR/benchmark_summary.json"
CHOSEN_WORKERS="$(
python - "$BENCH_SUMMARY" <<'PY2'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    summary = json.load(fh)
recommended = summary.get("recommended") or {}
workers = recommended.get("effective_workers") or recommended.get("requested_workers")
if not workers:
    raise SystemExit("benchmark summary did not contain a recommended worker count")
print(int(workers))
PY2
)"
echo "[result] chosen sample workers: $CHOSEN_WORKERS" | tee "$LOG_DIR/chosen_workers.txt"

ANALYSIS_CAMPAIGN=""
if [[ "$RUN_PRODUCTION" == "1" || "$RUN_PRODUCTION" == "true" || "$RUN_PRODUCTION" == "yes" ]]; then
  echo "[step] production campaign"
  production_args=(
    cpu_cft_extraction/run_cpu_cft_sweep.py
    --nx "$NX"
    --ny $PROD_NY
    --samples "$PROD_SAMPLES"
    --cycles-factor "$CYCLES_FACTOR"
    --alpha "$ALPHA"
    --fock-rank "$FOCK_RANK"
    --sample-workers "$CHOSEN_WORKERS"
    --blas-threads "$BLAS_THREADS"
    --output-root "$RUN_ROOT/production"
    --progress
  )
  if [[ -n "$CPU_LIST" ]]; then
    production_args+=(--cpu-list "$CPU_LIST")
  fi
  python "${production_args[@]}" 2>&1 | tee "$LOG_DIR/production.log"
  ANALYSIS_CAMPAIGN="$(find "$RUN_ROOT/production" -type f -name scalars_by_size.csv -printf '%h\n' | sort | tail -n 1)"
else
  echo "[skip] production campaign disabled by RUN_PRODUCTION=$RUN_PRODUCTION"
  ANALYSIS_CAMPAIGN="$(find "$RUN_ROOT/smoke" -type f -name scalars_by_size.csv -printf '%h\n' | sort | tail -n 1)"
fi

echo "[result] analysis campaign: $ANALYSIS_CAMPAIGN" | tee "$LOG_DIR/analysis_campaign.txt"

if [[ "$RUN_NOTEBOOK" == "1" || "$RUN_NOTEBOOK" == "true" || "$RUN_NOTEBOOK" == "yes" ]]; then
  echo "[step] execute analysis notebook"
  mkdir -p "$RUN_ROOT/notebook"
  CAMPAIGN_DIR="$ANALYSIS_CAMPAIGN" jupyter nbconvert \
    --to notebook \
    --execute cpu_cft_extraction/analyze_cpu_cft_extraction.ipynb \
    --output-dir "$RUN_ROOT/notebook" \
    --output analyze_cpu_cft_extraction.executed.ipynb \
    --ExecutePreprocessor.timeout=600 \
    2>&1 | tee "$LOG_DIR/notebook.log"
else
  echo "[skip] notebook execution disabled by RUN_NOTEBOOK=$RUN_NOTEBOOK"
fi

echo "[done] tmux CFT run complete: $RUN_ROOT"
