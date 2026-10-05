#!/usr/bin/env python3
"""Continue low-gap particle-hole tangent modes around the signed flux loops."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

import run_pure_tangent_flux as campaign


ANALYSIS_SCHEMA = "frozen_record_flux_pure_tangent_analysis_v1"


def _load_task(output_root: Path, task: dict[str, Any]) -> dict[str, np.ndarray]:
    if not campaign.verify_task(output_root, task):
        raise RuntimeError(f"task is not complete: {task['task_id']}")
    path, _ = campaign.task_paths(output_root, task)
    with np.load(path, allow_pickle=False) as saved:
        return {key: np.array(saved[key], copy=True) for key in saved.files}


def _continue_direction(rows: list[dict[str, np.ndarray]], tracked: int) -> dict[str, np.ndarray]:
    points = len(rows)
    if int(rows[0]["candidate_pair_rates"].size) < tracked:
        raise RuntimeError("initial flux point has too few finite candidates")
    selected = np.empty((points, tracked), dtype=np.int32)
    overlap = np.ones((points, tracked), dtype=np.float64)
    selected[0] = np.arange(tracked, dtype=np.int32)
    for point in range(1, points):
        previous_indices = selected[point - 1]
        previous_occ = rows[point - 1]["candidate_output_occupied"][:, previous_indices]
        previous_emp = rows[point - 1]["candidate_output_empty"][:, previous_indices]
        current_occ = rows[point]["candidate_output_occupied"]
        current_emp = rows[point]["candidate_output_empty"]
        mode_overlap = np.abs(previous_occ.conj().T @ current_occ) * np.abs(
            previous_emp.conj().T @ current_emp
        )
        previous_gap = rows[point - 1]["candidate_effective_gaps_per_cycle"][previous_indices]
        current_gap = rows[point]["candidate_effective_gaps_per_cycle"]
        gap_scale = max(float(np.median(np.abs(current_gap))), 1e-12)
        cost = -mode_overlap + 1e-6 * np.abs(
            previous_gap[:, None] - current_gap[None, :]
        ) / gap_scale
        row_index, column_index = linear_sum_assignment(cost)
        if len(row_index) != tracked or len(np.unique(column_index)) != tracked:
            raise RuntimeError("particle-hole branch assignment is incomplete")
        chosen = np.empty(tracked, dtype=np.int32)
        chosen[row_index] = column_index.astype(np.int32)
        selected[point] = chosen
        overlap[point] = mode_overlap[np.arange(tracked), chosen]
        if np.any(chosen < 0) or np.any(
            chosen >= int(rows[point]["candidate_pair_rates"].size)
        ):
            raise RuntimeError("branch assignment left the saved candidate pool")

    def gather(key: str) -> np.ndarray:
        return np.asarray([rows[p][key][selected[p]] for p in range(points)])

    first_occ = rows[0]["candidate_output_occupied"][:, selected[0]]
    first_emp = rows[0]["candidate_output_empty"][:, selected[0]]
    last_occ = rows[-1]["candidate_output_occupied"][:, selected[-1]]
    last_emp = rows[-1]["candidate_output_empty"][:, selected[-1]]
    closure_overlap = np.abs(first_occ.conj().T @ last_occ).diagonal() * np.abs(
        first_emp.conj().T @ last_emp
    ).diagonal()
    return {
        "candidate_index": selected,
        "step_overlap": overlap,
        "closure_overlap": closure_overlap,
        "pair_indices": gather("candidate_pair_indices"),
        "pair_rates": gather("candidate_pair_rates"),
        "effective_gaps_per_cycle": gather("candidate_effective_gaps_per_cycle"),
        "excitation_charge": gather("candidate_excitation_charge"),
        "occupied_left_weight": gather("candidate_occupied_left_weight"),
        "occupied_right_weight": gather("candidate_occupied_right_weight"),
        "empty_left_weight": gather("candidate_empty_left_weight"),
        "empty_right_weight": gather("candidate_empty_right_weight"),
        "svd_residual": gather("candidate_svd_residual"),
        "occupied_physical_occupation": gather(
            "candidate_output_occupied_physical_occupation"
        ),
        "empty_physical_occupation": gather(
            "candidate_output_empty_physical_occupation"
        ),
        "occupied_projector_residual": gather(
            "candidate_output_occupied_projector_residual"
        ),
        "empty_projector_residual": gather(
            "candidate_output_empty_projector_residual"
        ),
        "x_density": gather("candidate_x_density"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=campaign.DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=campaign.DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    config = campaign.load_json(config_path)
    campaign.validate_config(config)
    sources = campaign.source_hashes(config_path)
    config_hash = campaign.canonical_hash(
        {"schema": campaign.SCHEMA, "config": config, "source_hashes": sources}
    )
    tasks = campaign.expand_tasks(config, config_hash, sources)
    missing = [row["task_id"] for row in tasks if not campaign.verify_task(output_root, row)]
    if missing:
        raise RuntimeError(f"analysis requires all 68 verified tasks; missing={missing}")

    tracked = int(config["tangent"]["tracked_mode_count"])
    arms = [str(row["name"]) for row in config["arms"]]
    directions = [str(row["name"]) for row in config["twist"]["directions"]]
    task_map = {
        (row["arm"], row["direction"], int(row["twist_index"])): row for row in tasks
    }
    analysis: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    raw_rows: dict[tuple[str, str], list[dict[str, np.ndarray]]] = {}
    points = int(config["twist"]["points_per_direction"])
    for arm in arms:
        for direction in directions:
            rows = [
                _load_task(output_root, task_map[(arm, direction, index)])
                for index in range(points)
            ]
            raw_rows[(arm, direction)] = rows
            analysis[(arm, direction)] = _continue_direction(rows, tracked)

    out_dir = output_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "schema": np.asarray(ANALYSIS_SCHEMA),
        "arms": np.asarray(arms),
        "directions": np.asarray(directions),
        "twist_index": np.arange(points, dtype=np.int32),
        "tracked_mode_count": np.asarray(tracked, dtype=np.int32),
    }
    for arm in arms:
        for direction in directions:
            prefix = f"{arm}_{direction}"
            rows = raw_rows[(arm, direction)]
            arrays[f"{prefix}_phi"] = np.asarray([float(row["phi"]) for row in rows])
            arrays[f"{prefix}_N_left"] = np.asarray([float(row["N_left"]) for row in rows])
            arrays[f"{prefix}_N_right"] = np.asarray([float(row["N_right"]) for row in rows])
            arrays[f"{prefix}_N_total"] = np.asarray([float(row["N_total"]) for row in rows])
            for key, value in analysis[(arm, direction)].items():
                arrays[f"{prefix}_{key}"] = value
    campaign.atomic_npz(out_dir / "tracked_branches.npz", **arrays)

    csv_path = out_dir / "tracked_branches.csv"
    fieldnames = [
        "arm", "direction", "twist_index", "phi", "branch", "candidate_index",
        "occupied_index", "empty_index", "pair_rate", "effective_gap_per_cycle",
        "excitation_charge", "step_overlap", "svd_residual",
        "occupied_physical_occupation", "empty_physical_occupation",
        "occupied_projector_residual", "empty_projector_residual",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for arm in arms:
            for direction in directions:
                continued = analysis[(arm, direction)]
                phi = arrays[f"{arm}_{direction}_phi"]
                for point in range(points):
                    for branch in range(tracked):
                        pair = continued["pair_indices"][point, branch]
                        writer.writerow(
                            {
                                "arm": arm,
                                "direction": direction,
                                "twist_index": point,
                                "phi": f"{phi[point]:.17g}",
                                "branch": branch,
                                "candidate_index": int(continued["candidate_index"][point, branch]),
                                "occupied_index": int(pair[0]),
                                "empty_index": int(pair[1]),
                                "pair_rate": f"{continued['pair_rates'][point, branch]:.17g}",
                                "effective_gap_per_cycle": f"{continued['effective_gaps_per_cycle'][point, branch]:.17g}",
                                "excitation_charge": f"{continued['excitation_charge'][point, branch]:.17g}",
                                "step_overlap": f"{continued['step_overlap'][point, branch]:.17g}",
                                "svd_residual": f"{continued['svd_residual'][point, branch]:.17g}",
                                "occupied_physical_occupation": f"{continued['occupied_physical_occupation'][point, branch]:.17g}",
                                "empty_physical_occupation": f"{continued['empty_physical_occupation'][point, branch]:.17g}",
                                "occupied_projector_residual": f"{continued['occupied_projector_residual'][point, branch]:.17g}",
                                "empty_projector_residual": f"{continued['empty_projector_residual'][point, branch]:.17g}",
                            }
                        )
    summary = {
        "schema": ANALYSIS_SCHEMA,
        "created_unix": time.time(),
        "config_hash": config_hash,
        "task_count": 68,
        "tracked_mode_count": tracked,
        "claim_boundary": config["claim_boundary"],
        "directions": {
            f"{arm}_{direction}": {
                "minimum_step_overlap": float(np.min(analysis[(arm, direction)]["step_overlap"][1:])),
                "median_step_overlap": float(np.median(analysis[(arm, direction)]["step_overlap"][1:])),
                "minimum_closure_overlap": float(np.min(analysis[(arm, direction)]["closure_overlap"])),
                "maximum_svd_residual": float(np.max(analysis[(arm, direction)]["svd_residual"])),
                "minimum_saved_candidate_count": int(
                    min(int(row["candidate_mode_count"]) for row in raw_rows[(arm, direction)])
                ),
                "minimum_finite_pair_count": int(
                    min(int(row["finite_pair_mode_count"]) for row in raw_rows[(arm, direction)])
                ),
                "occupied_physical_occupation_range": [
                    float(np.min(analysis[(arm, direction)]["occupied_physical_occupation"])),
                    float(np.max(analysis[(arm, direction)]["occupied_physical_occupation"])),
                ],
                "empty_physical_occupation_range": [
                    float(np.min(analysis[(arm, direction)]["empty_physical_occupation"])),
                    float(np.max(analysis[(arm, direction)]["empty_physical_occupation"])),
                ],
                "maximum_occupied_projector_residual": float(
                    np.max(analysis[(arm, direction)]["occupied_projector_residual"])
                ),
                "maximum_empty_projector_residual": float(
                    np.max(analysis[(arm, direction)]["empty_projector_residual"])
                ),
            }
            for arm in arms
            for direction in directions
        },
        "products": {
            "npz": campaign.file_record(out_dir / "tracked_branches.npz"),
            "csv": campaign.file_record(csv_path),
        },
    }
    campaign.atomic_json(out_dir / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
