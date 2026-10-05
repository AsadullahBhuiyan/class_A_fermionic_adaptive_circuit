#!/usr/bin/env python3
"""Select idle physical cores on one NUMA node and launch the campaign in tmux."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path

import psutil

from run_campaign import scientific_config_hash, source_hashes


BUNDLE_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = BUNDLE_ROOT / "campaign_config.v1.json"


def physical_cpus_by_node() -> dict[int, list[int]]:
    raw = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET,NODE,ONLINE"], text=True)
    rows = csv.reader(StringIO("\n".join(line for line in raw.splitlines() if not line.startswith("#"))))
    representatives: dict[tuple[int, int], tuple[int, int]] = {}
    for cpu_text, core_text, socket_text, node_text, online_text in rows:
        if online_text.strip().upper() != "Y":
            continue
        cpu, core, socket_id, node = map(int, (cpu_text, core_text, socket_text, node_text))
        key = (socket_id, core)
        previous = representatives.get(key)
        if previous is None or cpu < previous[1]:
            representatives[key] = (node, cpu)
    grouped: dict[int, list[int]] = {}
    for node, cpu in representatives.values():
        grouped.setdefault(node, []).append(cpu)
    return {node: sorted(cpus) for node, cpus in grouped.items()}


def select_idle_cores(count: int) -> tuple[int, list[int], dict[int, float]]:
    grouped = physical_cpus_by_node()
    usage = psutil.cpu_percent(interval=1.0, percpu=True)
    eligible = {node: cpus for node, cpus in grouped.items() if len(cpus) >= int(count)}
    if not eligible:
        raise RuntimeError(f"No NUMA node has {count} physical cores: {grouped}")
    scored = []
    for node, cpus in eligible.items():
        selected = sorted(cpus, key=lambda cpu: (usage[cpu], cpu))[: int(count)]
        scored.append((sum(usage[cpu] for cpu in selected) / len(selected), node, selected))
    _mean, node, selected = min(scored)
    return node, sorted(selected), {cpu: float(usage[cpu]) for cpu in selected}


def build_launch(config: dict, node: int, cpus: list[int], session: str) -> tuple[list[str], str]:
    workers = len(cpus)
    output = BUNDLE_ROOT / config["execution"]["output_subdirectory"]
    command = [
        "numactl", f"--cpunodebind={node}", f"--membind={node}",
        "taskset", "-c", ",".join(map(str, cpus)),
        sys.executable, "-u", str(BUNDLE_ROOT / "run_campaign.py"), "all", "--resume",
        "--workers", str(workers), "--config", str(CONFIG_PATH), "--output-root", str(output),
    ]
    env = " ".join(f"{name}=1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ))
    log = output / "logs" / "tmux_campaign.log"
    shell_command = f"{env} {' '.join(shlex.quote(part) for part in command)} 2>&1 | tee -a {shlex.quote(str(log))}"
    tmux = ["tmux", "new-session", "-d", "-s", session, "bash", "-lc", shell_command]
    return tmux, shell_command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--session", default=None)
    args = parser.parse_args(argv)
    for executable in ("tmux", "numactl", "taskset", "lscpu"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Required executable is missing: {executable}")
    config = json.loads(CONFIG_PATH.read_text())
    count = int(args.cores or config["execution"]["workers"])
    session = str(args.session or config["execution"]["tmux_session"])
    node, cpus, utilization = select_idle_cores(count)
    tmux, shell_command = build_launch(config, node, cpus, session)
    output = BUNDLE_ROOT / config["execution"]["output_subdirectory"]
    launch_record = {
        "schema": "modular_handedness_tmux_launch_v1",
        "created_unix": time.time(), "host": socket.gethostname(), "session": session,
        "numa_node": node, "physical_cpus": cpus, "cpu_utilization_percent_at_selection": utilization,
        "workers": count, "command": shell_command, "output_root": str(output.resolve()),
        "expected_wall_time_hours": [8, 10],
        "simulation_config_hash": scientific_config_hash(config, "simulation"),
        "analysis_config_hash": scientific_config_hash(config, "analysis"),
        "simulation_source_hashes": source_hashes("simulation"),
        "analysis_source_hashes": source_hashes("analysis"),
    }
    print(json.dumps(launch_record, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        print("[dry-run] tmux was not started.")
        return 0
    if subprocess.run(["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError(f"tmux session already exists: {session}")
    (output / "logs").mkdir(parents=True, exist_ok=True)
    record_path = output / "launch.json"
    temp = record_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(launch_record, indent=2, sort_keys=True) + "\n")
    os.replace(temp, record_path)
    subprocess.run(tmux, check=True)
    print(f"[launched] session={session}")
    print(f"[attach] tmux attach -t {session}")
    print(f"[status] {sys.executable} {BUNDLE_ROOT / 'run_campaign.py'} status")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
