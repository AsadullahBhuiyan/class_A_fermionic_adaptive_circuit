#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runner="$repo_root/validation_campaigns/v0_v1/run.py"

if [[ "${1:-}" == "--inside" ]]; then
    output_dir="$2"
    log_path="$3"
    mode="$4"
    v1_nx="$5"
    v1_ny="$6"
    v1_samples="$7"
    no_auto_escalation="$8"
    v0_summary="$9"
    wait_session="${10}"
    workers="${11}"
    cpu_list="${12}"
    cd "$repo_root"
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    {
        echo "[config] output_dir=$output_dir"
        echo "[config] mode=$mode Nx=$v1_nx Ny=$v1_ny samples_per_schedule=$v1_samples"
        echo "[config] cpu_list=$cpu_list workers=$workers threads_per_worker=1"
        echo "[config] auto_escalation=$([[ "$no_auto_escalation" == 1 ]] && echo false || echo true)"
        if [[ -n "$wait_session" ]]; then
            while tmux has-session -t "$wait_session" 2>/dev/null; do
                echo "$(date --iso-8601=seconds) waiting for tmux session '$wait_session'"
                sleep 60
            done
            echo "$(date --iso-8601=seconds) prerequisite tmux gate cleared"
        fi
        taskset -c "$cpu_list" pytest -q tests
        taskset -c "$cpu_list" python "$runner" all \
            --output-dir "$output_dir/smoke" \
            --cpu-list "$cpu_list" \
            --workers 2 \
            --threads-per-worker 1 \
            --bootstrap-samples 200 \
            --seed 20260816 \
            --resume \
            --smoke
        common_args=(
            --output-dir "$output_dir"
            --cpu-list "$cpu_list"
            --workers "$workers"
            --threads-per-worker 1
            --bootstrap-samples 2000
            --seed 20260816
            --v1-nx "$v1_nx"
            --v1-ny "$v1_ny"
            --v1-samples "$v1_samples"
            --resume
        )
        if [[ -n "$v0_summary" ]]; then
            common_args+=(--v0-summary "$v0_summary")
        fi
        if [[ "$no_auto_escalation" == 1 ]]; then
            common_args+=(--no-v1-auto-escalation)
        fi
        if [[ "$mode" == v1 ]]; then
            taskset -c "$cpu_list" python "$runner" benchmark "${common_args[@]}"
        fi
        taskset -c "$cpu_list" python "$runner" "$mode" "${common_args[@]}"
    } 2>&1 | tee -a "$log_path"
    exit "${PIPESTATUS[0]}"
fi

mode=all
v1_nx=20
v1_ny=48
v1_samples=100
no_auto_escalation=0
v0_summary=""
wait_session=""
workers=28
cpu_list=56-111
output_dir=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) mode="$2"; shift 2 ;;
        --v1-nx) v1_nx="$2"; shift 2 ;;
        --v1-ny) v1_ny="$2"; shift 2 ;;
        --v1-samples) v1_samples="$2"; shift 2 ;;
        --no-v1-auto-escalation) no_auto_escalation=1; shift ;;
        --v0-summary) v0_summary="$2"; shift 2 ;;
        --wait-for-session) wait_session="$2"; shift 2 ;;
        --workers) workers="$2"; shift 2 ;;
        --cpu-list) cpu_list="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *)
            if [[ -n "$output_dir" ]]; then
                echo "Unexpected positional argument: $1" >&2
                exit 2
            fi
            output_dir="$1"
            shift
            ;;
    esac
done

if [[ "$mode" != all && "$mode" != v1 ]]; then
    echo "Launcher mode must be 'all' or 'v1'." >&2
    exit 2
fi
if [[ "$mode" == v1 && -z "$v0_summary" ]]; then
    echo "V1-only launch requires --v0-summary." >&2
    exit 2
fi

stamp="$(date +%Y%m%d_%H%M%S)"
if [[ "$mode" == v1 ]]; then
    session="v1_cpu_N${v1_nx}x${v1_ny}_S${v1_samples}_${stamp}"
else
    session="v0_v1_cpu_${stamp}"
fi
output_dir="${output_dir:-$repo_root/validation_campaigns/results/$session}"
mkdir -p "$output_dir"
log_path="$output_dir/session.log"
tmux new-session -d -s "$session" \
    "$repo_root/validation_campaigns/v0_v1/launch_tmux.sh" \
    --inside "$output_dir" "$log_path" "$mode" "$v1_nx" "$v1_ny" \
    "$v1_samples" "$no_auto_escalation" "$v0_summary" "$wait_session" \
    "$workers" "$cpu_list"
printf 'session=%s\noutput_dir=%s\nlog=%s\n' "$session" "$output_dir" "$log_path"
