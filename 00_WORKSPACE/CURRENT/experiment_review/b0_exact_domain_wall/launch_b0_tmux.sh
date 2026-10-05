#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_b0_campaign.py"

dry_run=0
preflight_only=0
resume_id=""
while (($#)); do
    case "$1" in
        --dry-run)
            dry_run=1
            shift
            ;;
        --preflight-only)
            preflight_only=1
            shift
            ;;
        --resume)
            if (($# < 2)); then
                echo "error: --resume requires a campaign_id" >&2
                exit 2
            fi
            resume_id="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '1,80p' "${SCRIPT_DIR}/README.md"
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

MAX_WORKERS="${MAX_WORKERS:-24}"
BLAS_THREADS="${BLAS_THREADS:-1}"
ALLOW_BUSY="${ALLOW_BUSY:-0}"
if ! [[ "${MAX_WORKERS}" =~ ^[1-9][0-9]*$ ]] || ((MAX_WORKERS > 24)); then
    echo "error: MAX_WORKERS must be an integer from 1 through 24" >&2
    exit 2
fi
if ! [[ "${BLAS_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: BLAS_THREADS must be a positive integer" >&2
    exit 2
fi

if [[ "${ALLOW_BUSY}" != "1" ]]; then
    heavy="$({ ps -eo pid=,args= || true; } | awk -v self="$$" '
        $1 != self && $0 !~ /awk -v self=/ && $0 ~ /(run_markov_circuit|run_campaign\.py|run_.*campaign|size_sweep|transfer_matrix|choi.*campaign)/ {print}
    ')"
    if [[ -n "${heavy}" ]]; then
        echo "error: known CPU-heavy campaign processes are active:" >&2
        echo "${heavy}" >&2
        echo "Set ALLOW_BUSY=1 only after checking the allocation." >&2
        exit 3
    fi
fi

if [[ -n "${CPU_LIST:-}" ]]; then
    cpu_list="${CPU_LIST}"
else
    cpu_list="$(python "${RUNNER}" --select-idle-cpus --limit "${MAX_WORKERS}")"
fi
if [[ -z "${cpu_list}" ]]; then
    echo "error: no idle physical cores found" >&2
    exit 3
fi
if ! taskset -c "${cpu_list}" true >/dev/null 2>&1; then
    echo "error: invalid or unavailable CPU_LIST=${cpu_list}" >&2
    exit 2
fi
cpu_count="$(python - "${cpu_list}" <<'PY'
import sys
values = set()
for token in sys.argv[1].split(','):
    token = token.strip()
    if '-' in token:
        first, last = (int(value) for value in token.split('-', 1))
        values.update(range(first, last + 1))
    elif token:
        values.add(int(token))
print(len(values))
PY
)"
if ((cpu_count < MAX_WORKERS)); then
    echo "error: found ${cpu_count} eligible physical cores but MAX_WORKERS=${MAX_WORKERS}" >&2
    echo "Wait for idle cores, provide CPU_LIST, or explicitly lower MAX_WORKERS." >&2
    exit 3
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -n "${resume_id}" ]]; then
    campaign_id="${resume_id}"
    campaign_dir="${SCRIPT_DIR}/results/${campaign_id}"
    if [[ ! -f "${campaign_dir}/manifest.json" ]]; then
        echo "error: cannot resume missing campaign ${campaign_id}" >&2
        exit 2
    fi
    session="b0_exact_dw_resume_${timestamp}"
    runner_args=(--resume "${campaign_id}")
else
    campaign_id="${timestamp}"
    campaign_dir="${SCRIPT_DIR}/results/${campaign_id}"
    session="b0_exact_dw_${timestamp}"
    runner_args=(--campaign-id "${campaign_id}")
fi
if ((preflight_only)); then
    runner_args+=(--preflight-only)
fi
runner_args+=(--max-workers "${MAX_WORKERS}" --cpu-list "${cpu_list}")

printf 'campaign_id: %s\n' "${campaign_id}"
printf 'tmux_session: %s\n' "${session}"
printf 'cpu_list: %s\n' "${cpu_list}"
printf 'max_workers: %s\n' "${MAX_WORKERS}"
printf 'blas_threads: %s\n' "${BLAS_THREADS}"
printf 'output: %s\n' "${campaign_dir}"

if ((dry_run)); then
    printf 'dry_run: no tmux session or output directory created\n'
    printf 'command:'
    printf ' %q' taskset -c "${cpu_list}" env \
        "OMP_NUM_THREADS=${BLAS_THREADS}" \
        "OPENBLAS_NUM_THREADS=${BLAS_THREADS}" \
        "MKL_NUM_THREADS=${BLAS_THREADS}" \
        "NUMEXPR_NUM_THREADS=${BLAS_THREADS}" \
        python "${RUNNER}" "${runner_args[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "${campaign_dir}/logs"
log_path="${campaign_dir}/logs/tmux.log"
printf -v command '%q ' taskset -c "${cpu_list}" env \
    "OMP_NUM_THREADS=${BLAS_THREADS}" \
    "OPENBLAS_NUM_THREADS=${BLAS_THREADS}" \
    "MKL_NUM_THREADS=${BLAS_THREADS}" \
    "NUMEXPR_NUM_THREADS=${BLAS_THREADS}" \
    "MAX_WORKERS=${MAX_WORKERS}" \
    "CPU_LIST=${cpu_list}" \
    "BLAS_THREADS=${BLAS_THREADS}" \
    python "${RUNNER}" "${runner_args[@]}"
printf -v quoted_log '%q' "${log_path}"
command+="2>&1 | tee -a ${quoted_log}"

tmux new-session -d -s "${session}" -c "${REPO_ROOT}" "bash -lc $(printf '%q' "${command}")"
echo "launched: tmux attach -t ${session}"
echo "log: ${log_path}"
