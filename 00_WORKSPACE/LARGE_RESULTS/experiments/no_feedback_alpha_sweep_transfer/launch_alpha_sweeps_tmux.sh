#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_alpha_sweep.py"
MAX_CPUS_PER_RUN=12
RANDOM_NY_VALUES=(20 26 32)
MAXMIX_NY_VALUES=(20)

DRY_RUN=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

CPU_LIST="$(python "${RUNNER}" --list-free-cpus)"
if [[ -z "${CPU_LIST}" ]]; then
  echo "No CPUs discovered." >&2
  exit 1
fi

IFS=',' read -r -a CPUS <<< "${CPU_LIST}"
WINDOW_COUNT=$(( ${#RANDOM_NY_VALUES[@]} + ${#MAXMIX_NY_VALUES[@]} ))
if [[ "${#CPUS[@]}" -lt "${WINDOW_COUNT}" ]]; then
  echo "Need at least ${WINDOW_COUNT} CPUs to partition across tmux windows; got ${#CPUS[@]}." >&2
  exit 1
fi

partition_cpus() {
  local offset="$1"
  local modulo="$2"
  local out=()
  local idx
  for idx in "${!CPUS[@]}"; do
    if (( idx % modulo == offset )); then
      out+=("${CPUS[$idx]}")
      if (( ${#out[@]} >= MAX_CPUS_PER_RUN )); then
        break
      fi
    fi
  done
  local IFS=,
  echo "${out[*]}"
}

quote_extra_args() {
  local args=("$@")
  local extra=""
  if (( ${#args[@]} > 0 )); then
    printf -v extra ' %q' "${args[@]}"
  fi
  echo "${extra}"
}

build_window_cmd() {
  local init_mode="$1"
  local ny="$2"
  local cpuset="$3"
  local track_entropy="$4"
  shift 4
  local extra
  extra="$(quote_extra_args "$@")"
  local entropy_arg=""
  if [[ "${track_entropy}" == "1" ]]; then
    entropy_arg=" --track-global-entropy"
  fi
  local cmd
  printf -v cmd 'cd %q && python %q --init-mode %q --ny %q --dwtrunc 0 --cpu-list %q --resume%s%s && python %q --init-mode %q --ny %q --dwtrunc 1 --cpu-list %q --resume%s%s' \
    "${SCRIPT_DIR}" "${RUNNER}" "${init_mode}" "${ny}" "${cpuset}" "${entropy_arg}" "${extra}" \
    "${RUNNER}" "${init_mode}" "${ny}" "${cpuset}" "${entropy_arg}" "${extra}"
  echo "${cmd}"
}

launch_window() {
  local session="$1"
  local window="$2"
  local cmd="$3"
  local first_window="$4"
  echo "session=${session} window=${window}"
  echo "command=${cmd}"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    if [[ "${first_window}" == "1" ]]; then
      tmux new-session -d -s "${session}" -n "${window}" "${cmd}"
    else
      tmux new-window -t "${session}" -n "${window}" "${cmd}"
    fi
  fi
}

if [[ "${DRY_RUN}" -eq 0 ]]; then
  for session in nofb_alpha_sweep_random nofb_alpha_sweep_maxmix; do
    if tmux has-session -t "${session}" 2>/dev/null; then
      echo "tmux session ${session} already exists; refusing to overwrite." >&2
      exit 1
    fi
  done
fi

pool_index=0
first_random=1
for ny in "${RANDOM_NY_VALUES[@]}"; do
  cpuset="$(partition_cpus "${pool_index}" "${WINDOW_COUNT}")"
  echo "random Ny=${ny} cpus=${cpuset}"
  cmd="$(build_window_cmd random "${ny}" "${cpuset}" 0 "${EXTRA_ARGS[@]}")"
  launch_window nofb_alpha_sweep_random "Ny${ny}" "${cmd}" "${first_random}"
  first_random=0
  pool_index=$((pool_index + 1))
done

first_maxmix=1
for ny in "${MAXMIX_NY_VALUES[@]}"; do
  cpuset="$(partition_cpus "${pool_index}" "${WINDOW_COUNT}")"
  echo "maxmix Ny=${ny} cpus=${cpuset}"
  cmd="$(build_window_cmd maxmix "${ny}" "${cpuset}" 1 "${EXTRA_ARGS[@]}")"
  launch_window nofb_alpha_sweep_maxmix "Ny${ny}" "${cmd}" "${first_maxmix}"
  first_maxmix=0
  pool_index=$((pool_index + 1))
done

if [[ "${DRY_RUN}" -eq 0 ]]; then
  echo "Launched sessions. Attach with:"
  echo "  tmux attach -t nofb_alpha_sweep_random"
  echo "  tmux attach -t nofb_alpha_sweep_maxmix"
  echo "Random windows: Ny20, Ny26, Ny32"
  echo "Maxmix windows: Ny20"
else
  echo "Dry run only; no tmux sessions launched."
fi
