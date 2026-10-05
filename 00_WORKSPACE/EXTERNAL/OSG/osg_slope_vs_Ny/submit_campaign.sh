#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

REQ_MEM_MB=32768
MAX_IDLE=10
GENERATOR_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mem-mb)
            REQ_MEM_MB="$2"
            shift 2
            ;;
        --max-idle)
            MAX_IDLE="$2"
            shift 2
            ;;
        *)
            GENERATOR_ARGS+=("$1")
            shift
            ;;
    esac
done

mkdir -p logs

generator_output="$(python3 generate_params.py "${GENERATOR_ARGS[@]}")"
printf '%s\n' "$generator_output"

job_count="$(printf '%s\n' "$generator_output" | awk -F= '/^JOB_COUNT=/{count=$2} END{print count}')"
if [[ -z "${job_count}" ]]; then
    echo "Failed to extract JOB_COUNT from generate_params.py output." >&2
    exit 1
fi

if [[ ! "${REQ_MEM_MB}" =~ ^[0-9]+$ || ! "${MAX_IDLE}" =~ ^[0-9]+$ || ! "${job_count}" =~ ^[0-9]+$ ]]; then
    echo "REQ_MEM_MB, MAX_IDLE, and JOB_COUNT must be positive integers." >&2
    exit 1
fi

export REQ_MEM_MB MAX_IDLE
export N_JOBS="${job_count}"

echo "Submitting ${N_JOBS} job(s) with REQ_MEM_MB=${REQ_MEM_MB} and MAX_IDLE=${MAX_IDLE}"
condor_submit all_system_sizes.submit
