#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_campaign.py"
MAX_CPUS_PER_RUN=12

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
if [[ "${#CPUS[@]}" -lt 3 ]]; then
  echo "Need at least 3 CPUs to partition across campaigns; got ${#CPUS[@]}." >&2
  exit 1
fi

partition_cpus() {
  local offset="$1"
  local out=()
  local idx
  for idx in "${!CPUS[@]}"; do
    if (( idx % 3 == offset )); then
      out+=("${CPUS[$idx]}")
      if (( ${#out[@]} >= MAX_CPUS_PER_RUN )); then
        break
      fi
    fi
  done
  local IFS=,
  echo "${out[*]}"
}

launch_campaign() {
  local session="$1"
  local campaign="$2"
  local cpuset="$3"
  shift 3
  local args=("$@")
  local cmd
  printf -v cmd 'cd %q && python %q --campaign %q --init-mode random --cpu-list %q --resume %s && python %q --campaign %q --init-mode maxmix --cpu-list %q --resume %s' \
    "${SCRIPT_DIR}" "${RUNNER}" "${campaign}" "${cpuset}" "${args[*]}" \
    "${RUNNER}" "${campaign}" "${cpuset}" "${args[*]}"

  echo "session=${session}"
  echo "campaign=${campaign}"
  echo "cpus=${cpuset}"
  echo "command=${cmd}"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    if tmux has-session -t "${session}" 2>/dev/null; then
      echo "tmux session ${session} already exists; refusing to overwrite." >&2
      exit 1
    fi
    tmux new-session -d -s "${session}" "${cmd}"
  fi
}

POOL0="$(partition_cpus 0)"
POOL1="$(partition_cpus 1)"
POOL2="$(partition_cpus 2)"

launch_campaign "nofb_uniform_alpha1" "uniform_alpha1_no_dw" "${POOL0}" "${EXTRA_ARGS[@]}"
launch_campaign "nofb_dw_untrunc_alpha1_alpha30" "domain_wall_untruncated_alpha1_alpha30" "${POOL1}" "${EXTRA_ARGS[@]}"
launch_campaign "nofb_dw_trunc_alpha1_alpha30" "domain_wall_truncated_alpha1_alpha30" "${POOL2}" "${EXTRA_ARGS[@]}"

if [[ "${DRY_RUN}" -eq 0 ]]; then
  echo "Launched sessions. Attach with:"
  echo "  tmux attach -t nofb_uniform_alpha1"
  echo "  tmux attach -t nofb_dw_untrunc_alpha1_alpha30"
  echo "  tmux attach -t nofb_dw_trunc_alpha1_alpha30"
else
  echo "Dry run only; no tmux sessions launched."
fi
