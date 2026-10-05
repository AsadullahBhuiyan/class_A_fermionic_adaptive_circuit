#!/usr/bin/env python3
"""Run the isolated half-step convergence check for the long RK4 ramp."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any

import numpy as np
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_parent_schrodinger_rk4_s50 as primary  # noqa: E402


SCHEMA = "parent_schrodinger_rk4_refinement_v1"
DEFAULT_CONFIG = PROJECT_ROOT / "campaign_config.parent_schrodinger_rk4_n20x24_s2_tau1e4_dt_half_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "N20x24_parent_schrodinger_rk4_s2_tau1e4_dt_half_v1"


def load_refinement(path: Path) -> dict[str, Any]:
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA!r}")
    if row.get("campaign_id") != "N20x24_parent_schrodinger_rk4_s2_tau1e4_dt_half_v1":
        raise ValueError("unexpected refinement identity")
    if row.get("sample_ids") != [0] or row.get("walls") != ["soft", "hard"]:
        raise ValueError("refinement must use sample 0 for both wall constructions")
    if row.get("directions") != ["ccw", "cw"] or int(row.get("steps_per_interval", 0)) != 640:
        raise ValueError("refinement must halve the primary RK4 step in both directions")
    return row


def refined_config(spec: dict[str, Any]) -> dict[str, Any]:
    base_path = (PROJECT_ROOT / spec["primary_config"]).resolve()
    config = copy.deepcopy(primary.load_config(base_path))
    primary.validate_config(config)
    config["schema"] = SCHEMA
    config["campaign_id"] = spec["campaign_id"]
    config["evolution"]["steps_per_interval"] = int(spec["steps_per_interval"])
    config["execution"]["workers"] = int(spec["workers"])
    config["execution"]["checkpoint_every_intervals"] = int(spec["checkpoint_every_intervals"])
    config["ensemble"] = {
        "walls": list(spec["walls"]),
        "endpoint_states_total": 2,
        "paths_total": 4,
        "independent_sampling_unit": "saved monitored endpoint trajectory",
    }
    return config


def selected_tasks(config: dict[str, Any], spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "task_id": f"rk4_{wall}_{direction}_sample_{sample_id:03d}",
            "wall": wall,
            "direction": direction,
            "sigma": int(config["evolution"]["directions"][direction]),
            "sample_id": int(sample_id),
            "source_task_id": f"burnin_{wall}_sample_{sample_id:03d}",
        }
        for wall in spec["walls"]
        for sample_id in spec["sample_ids"]
        for direction in spec["directions"]
    ]
    if len(rows) != 4 or len({row["task_id"] for row in rows}) != 4:
        raise RuntimeError("expected four unique half-step paths")
    return rows


def identities(config: dict[str, Any]) -> tuple[dict[str, str], str]:
    hashes = primary.source_hashes()
    hashes["refinement_runner"] = primary.sha256_path(Path(__file__).resolve())
    return hashes, primary.scientific_config_hash(config)


def inventory(
    config: dict[str, Any], spec: dict[str, Any], output_root: Path,
    context: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str], str]:
    hashes, config_hash = identities(config)
    complete, pending = [], []
    for task in selected_tasks(config, spec):
        source_row = context["rows"][task["source_task_id"]]
        ok, _, _ = primary.verify_result(
            output_root, task, config_hash=config_hash, hashes=hashes,
            source_row=source_row, config=config,
        )
        (complete if ok else pending).append(task)
    return complete, pending, hashes, config_hash


def compare_primary(spec: dict[str, Any], output_root: Path, config: dict[str, Any]) -> dict[str, Any] | None:
    primary_root = (PROJECT_ROOT / spec["primary_output_root"]).resolve()
    rows = []
    for task in selected_tasks(config, spec):
        fine_path, _ = primary.result_paths(output_root, task)
        coarse_path, _ = primary.result_paths(primary_root, task)
        if not fine_path.is_file() or not coarse_path.is_file():
            return None
        with np.load(fine_path, allow_pickle=False) as fine, np.load(coarse_path, allow_pickle=False) as coarse:
            delta = np.asarray(fine["q_x"]) - np.asarray(coarse["q_x"])
            rows.append({
                "task_id": task["task_id"],
                "coarse_dt": float(np.asarray(coarse["dt"]).item()),
                "fine_dt": float(np.asarray(fine["dt"]).item()),
                "coarse_endpoint_q_x": float(np.asarray(coarse["q_x"])[-1]),
                "fine_endpoint_q_x": float(np.asarray(fine["q_x"])[-1]),
                "endpoint_absolute_difference": float(abs(delta[-1])),
                "maximum_path_absolute_difference": float(np.max(np.abs(delta))),
            })
    summary = {
        "schema": "parent_schrodinger_rk4_step_halving_summary_v1",
        "rows": rows,
        "maximum_endpoint_absolute_difference": max(row["endpoint_absolute_difference"] for row in rows),
        "maximum_path_absolute_difference": max(row["maximum_path_absolute_difference"] for row in rows),
    }
    primary._atomic_json(output_root / "step_halving_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("report", "run"), nargs="?", default="report")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    spec = load_refinement(args.config.resolve())
    config = refined_config(spec)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    context = primary.source_context(primary.load_config((PROJECT_ROOT / spec["primary_config"]).resolve()))
    complete, pending, hashes, config_hash = inventory(config, spec, output_root, context)
    total_steps = int(config["evolution"]["flux_intervals"]) * int(config["evolution"]["steps_per_interval"])
    print(f"[refinement] {spec['campaign_id']}")
    print(f"[workload] 4 paths x {total_steps} steps; verified={len(complete)}/4 pending={len(pending)}")
    print(f"[identity] config_sha256={config_hash}")
    print(f"[output] {output_root}")
    if args.command == "report":
        return 0
    workers = int(args.workers or spec["workers"])
    failures = []
    if pending:
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(workers, len(pending)), mp_context=mp_context) as pool:
            futures = {
                pool.submit(
                    primary._worker,
                    (task, config, str(output_root), config_hash, hashes,
                     context["rows"][task["source_task_id"]], None),
                ): task
                for task in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="half-step paths", unit="path"):
                row = future.result()
                if not row["ok"]:
                    failures.append(row)
    if failures:
        raise RuntimeError(f"half-step refinement failures: {failures}")
    complete, pending, _, _ = inventory(config, spec, output_root, context)
    if pending:
        raise RuntimeError("half-step refinement ended without four verified paths")
    summary = compare_primary(spec, output_root, config)
    if summary is None:
        print("[pending comparison] primary sample-0 paths are not all complete yet")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
