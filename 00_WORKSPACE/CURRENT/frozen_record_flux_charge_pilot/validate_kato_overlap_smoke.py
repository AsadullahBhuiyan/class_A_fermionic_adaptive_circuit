#!/usr/bin/env python3
"""Verify the eight sample-0 Kato paths and their tightened-tolerance repeat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_kato_overlap_continuation_s25 as campaign


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinary-root", type=Path, required=True)
    parser.add_argument("--tight-root", type=Path, required=True)
    args = parser.parse_args()
    config = campaign.load_config(campaign.DEFAULT_CONFIG)
    campaign.validate_config(config)
    context = campaign.source_context(config)
    hashes, config_hash = campaign.source_hashes(), campaign.scientific_config_hash(config)
    selected = [task for task in campaign.tasks(config) if task["sample_id"] == 0]
    rows = []
    for task in selected:
        source_row = context[task["size"]]["rows"][task["source_task_id"]]
        paths = []
        for root, tolerance in ((args.ordinary_root, 1e-8), (args.tight_root, 2.5e-9)):
            ok, reason, _ = campaign.verify_result(
                root, task, config_hash=config_hash, hashes=hashes, source_row=source_row,
                config=config, adaptive_tolerance=tolerance,
            )
            if not ok:
                raise RuntimeError(f"smoke result is not verified: {task['task_id']}: {reason}")
            path, _ = campaign.result_paths(root, task)
            with np.load(path, allow_pickle=False) as saved:
                paths.append(np.array(saved["q_x"], copy=True))
        maximum = float(np.max(np.abs(paths[0] - paths[1])))
        endpoint = float(abs(paths[0][-1] - paths[1][-1]))
        if maximum > 1e-4 or endpoint > 1e-4:
            raise FloatingPointError(
                f"tight-tolerance smoke mismatch for {task['task_id']}: "
                f"path={maximum:.3e}, endpoint={endpoint:.3e}"
            )
        rows.append({"task_id": task["task_id"], "maximum_path_q_x_difference": maximum, "endpoint_q_x_difference": endpoint})
    report = {
        "schema": "kato_overlap_smoke_validation_v1",
        "paths": len(rows),
        "ordinary_tolerance": 1e-8,
        "tight_tolerance": 2.5e-9,
        "required_maximum_q_x_difference": 1e-4,
        "maximum_path_q_x_difference": max(row["maximum_path_q_x_difference"] for row in rows),
        "maximum_endpoint_q_x_difference": max(row["endpoint_q_x_difference"] for row in rows),
        "rows": rows,
    }
    campaign._atomic_json(args.ordinary_root / "smoke_validation.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
