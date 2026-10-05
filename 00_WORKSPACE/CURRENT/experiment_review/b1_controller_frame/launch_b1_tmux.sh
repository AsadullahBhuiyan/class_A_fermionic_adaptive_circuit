#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_b1_campaign.py"

dry_run=0
preflight_only=0
resume_id=""
while (($#)); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --preflight-only) preflight_only=1; shift ;;
        --resume)
            (($# >= 2)) || { echo "error: --resume requires a campaign_id" >&2; exit 2; }
            resume_id="$2"; shift 2 ;;
        -h|--help) sed -n '1,100p' "${SCRIPT_DIR}/README.md"; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; exit 2 ;;
    esac
done

MAX_WORKERS="${MAX_WORKERS:-4}"
BLAS_THREADS="${BLAS_THREADS:-1}"
ALLOW_BUSY="${ALLOW_BUSY:-0}"
[[ "${MAX_WORKERS}" =~ ^[1-4]$ ]] || { echo "error: MAX_WORKERS must be 1 through 4" >&2; exit 2; }
[[ "${BLAS_THREADS}" =~ ^[1-9][0-9]*$ ]] || { echo "error: BLAS_THREADS must be positive" >&2; exit 2; }

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -n "${resume_id}" ]]; then
    campaign_id="${resume_id}"
    campaign_dir="${SCRIPT_DIR}/results/${campaign_id}"
    [[ -f "${campaign_dir}/manifest.json" ]] || { echo "error: missing campaign ${campaign_id}" >&2; exit 2; }
    session="b1_frame_resume_${timestamp}"
else
    campaign_id="${timestamp}"
    campaign_dir="${SCRIPT_DIR}/results/${campaign_id}"
    session="b1_frame_${timestamp}"
fi

select_cpus() {
    if [[ -n "${CPU_LIST:-}" ]]; then
        printf '%s\n' "${CPU_LIST}"
        return
    fi
    local args=(--select-idle-cpus --limit "${MAX_WORKERS}")
    [[ "${ALLOW_BUSY}" == "1" ]] && args+=(--allow-busy)
    python "${RUNNER}" "${args[@]}"
}

initial_cpu_list="$(select_cpus)"
if [[ -z "${initial_cpu_list}" ]]; then
    initial_cpu_list="waiting_for_${MAX_WORKERS}_idle_physical_cores"
fi

printf 'campaign_id: %s\n' "${campaign_id}"
printf 'tmux_session: %s\n' "${session}"
printf 'initial_cpu_selection: %s\n' "${initial_cpu_list}"
printf 'max_workers: %s\n' "${MAX_WORKERS}"
printf 'output: %s\n' "${campaign_dir}"

if ((dry_run)); then
    echo "dry_run: no tmux session or output directory created"
    exit 0
fi

mkdir -p "${campaign_dir}/logs"
log_path="${campaign_dir}/logs/tmux.log"
printf -v inside_command '%q ' "${SCRIPT_DIR}/inside_b1_tmux.sh" \
    "${campaign_id}" "${MAX_WORKERS}" "${BLAS_THREADS}" "${ALLOW_BUSY}" \
    "${CPU_LIST:-}" "${preflight_only}"
inside_command+="2>&1 | tee -a $(printf '%q' "${log_path}")"
tmux new-session -d -s "${session}" -c "${REPO_ROOT}" \
    "bash -lc $(printf '%q' "${inside_command}")"
echo "launched: tmux attach -t ${session}"
echo "log: ${log_path}"
