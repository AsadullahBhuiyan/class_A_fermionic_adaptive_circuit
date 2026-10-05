#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_b1_campaign.py"

campaign_id="$1"
max_workers="$2"
blas_threads="$3"
allow_busy="$4"
cpu_override="$5"
preflight_only="$6"

cd "${REPO_ROOT}"
export OMP_NUM_THREADS="${blas_threads}"
export OPENBLAS_NUM_THREADS="${blas_threads}"
export MKL_NUM_THREADS="${blas_threads}"
export NUMEXPR_NUM_THREADS="${blas_threads}"

select_stage_cpus() {
    if [[ -n "${cpu_override}" ]]; then
        printf '%s\n' "${cpu_override}"
        return
    fi
    local args=(--select-idle-cpus --limit "${max_workers}")
    [[ "${allow_busy}" == "1" ]] && args+=(--allow-busy)
    python "${RUNNER}" "${args[@]}"
}

wait_for_cpus() {
    local selected=""
    while [[ -z "${selected}" ]]; do
        selected="$(select_stage_cpus)"
        if [[ -z "${selected}" ]]; then
            echo "$(date --iso-8601=seconds) waiting for ${max_workers} idle physical cores" >&2
            sleep 60
        fi
    done
    printf '%s\n' "${selected}"
}

run_stage() {
    local stage="$1"
    local cpus
    cpus="$(wait_for_cpus)"
    echo "$(date --iso-8601=seconds) stage=${stage} cpu_list=${cpus}"
    taskset -c "${cpus}" nice -n 10 python "${RUNNER}" "${stage}" \
        --campaign-id "${campaign_id}" --cpu-list "${cpus}" \
        --max-workers "${max_workers}"
}

python "${RUNNER}" init --campaign-id "${campaign_id}"
run_stage preflight
if [[ "${preflight_only}" == 0 ]]; then
    run_stage static
    run_stage trajectory
    run_stage analyze
    run_stage report
fi
