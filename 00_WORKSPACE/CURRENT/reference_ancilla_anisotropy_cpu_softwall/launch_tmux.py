#!/usr/bin/env python3
"""Select idle physical CPUs and launch the L=24 soft-wall campaign in tmux."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RUNNER = HERE / "run_l24_campaign.py"
CONFIG = HERE / "campaign_config.l24.v3.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_cpu_times() -> dict[int, tuple[int, int]]:
    rows: dict[int, tuple[int, int]] = {}
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("cpu") or not fields[0][3:].isdigit():
            continue
        values = [int(value) for value in fields[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        rows[int(fields[0][3:])] = (sum(values), idle)
    return rows


def sample_cpu_utilization(seconds: float = 5.0) -> dict[int, float]:
    before = read_cpu_times()
    time.sleep(float(seconds))
    after = read_cpu_times()
    result = {}
    for cpu in sorted(set(before) & set(after)):
        total = after[cpu][0] - before[cpu][0]
        idle = after[cpu][1] - before[cpu][1]
        result[cpu] = 0.0 if total <= 0 else 100.0 * (total - idle) / total
    return result


def cpu_topology(cpu: int) -> tuple[int, int, int]:
    root = Path(f"/sys/devices/system/cpu/cpu{cpu}")
    topology = root / "topology"
    package = int((topology / "physical_package_id").read_text())
    core = int((topology / "core_id").read_text())
    nodes = list(root.glob("node[0-9]*"))
    node = int(nodes[0].name[4:]) if nodes else package
    return node, package, core


def physical_cpus() -> list[int]:
    available = os.sched_getaffinity(0)
    representatives: dict[tuple[int, int], int] = {}
    for cpu in sorted(available):
        _, package, core = cpu_topology(cpu)
        representatives.setdefault((package, core), cpu)
    return sorted(representatives.values())


def choose_cpus(utilization: dict[int, float], required: int, maximum_utilization: float) -> tuple[list[int], dict[str, Any]]:
    by_node: dict[int, list[int]] = {}
    for cpu in physical_cpus():
        node, _, _ = cpu_topology(cpu)
        if utilization.get(cpu, 100.0) <= float(maximum_utilization):
            by_node.setdefault(node, []).append(cpu)
    candidates = []
    for node, cpus in by_node.items():
        if len(cpus) >= required:
            ordered = sorted(cpus, key=lambda cpu: (utilization[cpu], cpu))[:required]
            candidates.append((sum(utilization[cpu] for cpu in ordered) / required, node, ordered))
    if not candidates:
        counts = {node: len(cpus) for node, cpus in by_node.items()}
        raise RuntimeError(f"No NUMA node has {required} distinct physical CPUs below {maximum_utilization:.1f}% utilization; eligible counts={counts}.")
    mean_utilization, node, selected = min(candidates)
    metadata = {
        "selected_numa_node": node,
        "selected_physical_cpus": selected,
        "selected_mean_utilization_percent": mean_utilization,
        "maximum_allowed_utilization_percent": maximum_utilization,
        "utilization_percent": {str(cpu): utilization[cpu] for cpu in selected},
    }
    return selected, metadata


def build_tmux_command(session: str, cpu_list: str, shell_script: str) -> list[str]:
    return ["tmux", "new-session", "-d", "-s", session, "taskset", "-c", cpu_list, "bash", "-lc", shell_script]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument("--utilization-sample-seconds", type=float, default=5.0)
    parser.add_argument("--wait-for-idle-minutes", type=float, default=30.0)
    parser.add_argument("--retry-seconds", type=float, default=30.0)
    parser.add_argument(
        "--maximum-utilization-percent",
        type=float,
        default=None,
        help="Explicit launch-time override for the configured per-core utilization ceiling.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = json.loads(CONFIG.read_text())
    required = int(config["parallel"]["required_physical_cpus"])
    configured_maximum = float(config["parallel"]["maximum_cpu_utilization_percent"])
    maximum = configured_maximum if args.maximum_utilization_percent is None else float(args.maximum_utilization_percent)
    if not 0.0 <= maximum <= 100.0:
        raise ValueError("--maximum-utilization-percent must lie in [0, 100].")
    deadline = time.monotonic() + max(0.0, float(args.wait_for_idle_minutes)) * 60.0
    while True:
        utilization = sample_cpu_utilization(args.utilization_sample_seconds)
        try:
            selected, preflight = choose_cpus(utilization, required, maximum)
            break
        except RuntimeError as exc:
            if time.monotonic() >= deadline:
                raise
            remaining = max(0.0, deadline - time.monotonic())
            delay = min(max(1.0, float(args.retry_seconds)), remaining)
            print(f"[wait-for-idle] {exc} Retrying in {delay:.0f}s.", flush=True)
            time.sleep(delay)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = f"ref_aniso_soft_l24_v1_{timestamp}"
    run_root = (args.output_root / f"l24_softwall_aniso_v1_{timestamp}").resolve()
    (run_root / "logs").mkdir(parents=True, exist_ok=False)
    cpu_list = ",".join(str(cpu) for cpu in selected)
    log_path = run_root / "logs" / "campaign.log"
    command = [
        sys.executable,
        str(RUNNER),
        "all",
        "--run-root",
        str(run_root),
        "--cpu-list",
        cpu_list,
        "--workers",
        "auto",
        "--resume",
    ]
    shell_lines = [
        "set -euo pipefail",
        "export OMP_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export NUMEXPR_NUM_THREADS=1",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"{shlex.join(command)} 2>&1 | tee {shlex.quote(str(log_path))}",
    ]
    shell_script = "\n".join(line for line in shell_lines if line)
    tmux_command = build_tmux_command(session, cpu_list, shell_script)
    launch_record = {
        "created_utc": utc_now(),
        "session": session,
        "run_root": str(run_root),
        "cpu_list": cpu_list,
        "preflight": preflight,
        "configured_maximum_utilization_percent": configured_maximum,
        "launch_maximum_utilization_percent": maximum,
        "utilization_threshold_overridden": bool(maximum != configured_maximum),
        "runner_command": command,
        "tmux_command": tmux_command,
        "dry_run": bool(args.dry_run),
    }
    (run_root / "cpu_preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    (run_root / "tmux_launch.json").write_text(json.dumps(launch_record, indent=2, sort_keys=True) + "\n")
    if not args.dry_run:
        subprocess.run(tmux_command, check=True)
    print(f"run_root={run_root}")
    print(f"session={session}")
    print(f"attach: tmux attach -t {session}")
    print(f"log: tail -f {shlex.quote(str(log_path))}")
    print(f"resume: {shlex.join(command)}")
    if args.dry_run:
        print("dry-run: tmux was not started")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
