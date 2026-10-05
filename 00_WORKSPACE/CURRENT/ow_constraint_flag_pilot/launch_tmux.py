#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT.parents[2]


def cpu_times() -> dict[int, tuple[int, int]]:
    records: dict[int, tuple[int, int]] = {}
    for line in Path("/proc/stat").read_text().splitlines():
        name, *raw = line.split()
        if not name.startswith("cpu") or not name[3:].isdigit():
            continue
        values = [int(value) for value in raw]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        records[int(name[3:])] = (sum(values), idle)
    return records


def utilization(window: float) -> dict[int, float]:
    before = cpu_times()
    time.sleep(window)
    after = cpu_times()
    result = {}
    for cpu, (total_after, idle_after) in after.items():
        total_before, idle_before = before[cpu]
        delta = total_after - total_before
        result[cpu] = 1.0 - (idle_after - idle_before) / delta if delta else 1.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight and launch the OW flag pilot in tmux")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-utilization", type=float, default=0.35)
    parser.add_argument("--sample-seconds", type=float, default=3.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    if not 1 <= args.workers <= 56:
        raise SystemExit("--workers must lie in 1..56")
    observed = utilization(args.sample_seconds)
    eligible = sorted(range(56), key=lambda cpu: (observed.get(cpu, 1.0), cpu))
    sufficiently_idle = [cpu for cpu in eligible if observed.get(cpu, 1.0) <= args.max_utilization]
    if len(sufficiently_idle) < args.workers:
        raise SystemExit(
            f"Refusing launch: only {len(sufficiently_idle)} physical CPU IDs in 0..55 "
            f"were <= {100*args.max_utilization:.1f}% utilized"
        )
    chosen = sufficiently_idle[: args.workers]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = f"ow_flag_N16x20_S10_{stamp}"
    run_root = EXPERIMENT / "outputs" / "runs" / session
    logs = EXPERIMENT / "outputs" / "tmux_logs"
    logs.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=False)
    preflight = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_seconds": args.sample_seconds,
        "maximum_accepted_utilization": args.max_utilization,
        "physical_cpu_domain": list(range(56)),
        "excluded_sibling_hyperthreads": list(range(56, 112)),
        "utilization_by_cpu": {str(cpu): observed.get(cpu) for cpu in range(56)},
        "selected_cpu_ids": chosen,
        "workers": args.workers,
        "blas_threads_per_worker": 1,
    }
    preflight_path = run_root / "cpu_preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    cpu_list = ",".join(map(str, chosen))
    log_path = logs / f"{session}.log"
    runner = EXPERIMENT / "run_pilot.py"
    tests = EXPERIMENT / "tests"
    smoke_root = run_root / "validation_smoke"
    commands = [
        "set -euo pipefail",
        f"exec > >(tee -a {str(log_path)!r}) 2>&1",
        f"cd {str(REPO_ROOT)!r}",
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        f"export OW_FLAG_PREFLIGHT_JSON={str(preflight_path)!r}",
        f"python -m pytest -q {str(tests)!r}",
        f"python {str(runner)!r} smoke --resume --workers 2 --cpu-list {cpu_list!r} --output-root {str(smoke_root)!r} --bootstrap-samples 100",
        f"python {str(runner)!r} all --resume --workers {args.workers} --cpu-list {cpu_list!r} --output-root {str(run_root)!r} --bootstrap-samples {args.bootstrap_samples}",
    ]
    shell_command = "\n".join(commands)
    launch = ["tmux", "new-session", "-d", "-s", session, "taskset", "-c", cpu_list, "bash", "-lc", shell_command]
    subprocess.run(launch, cwd=REPO_ROOT, check=True)
    payload = {"session": session, "cpu_list": cpu_list, "run_root": str(run_root), "log": str(log_path)}
    (run_root / "tmux_launch.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
