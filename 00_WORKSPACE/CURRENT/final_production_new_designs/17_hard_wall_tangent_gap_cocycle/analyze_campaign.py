#!/usr/bin/env python3
"""Assemble trajectory-first quenched gap statistics after slot 17 completes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

import run_campaign as campaign


ANALYSIS_SCHEMA = "hard_wall_tangent_gap_quenched_analysis_v1"


def _case_seed(root_seed: int, ny: int, alpha_1: float) -> int:
    raw = f"{root_seed}|Ny={ny}|alpha1={alpha_1:.12g}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def bootstrap_mean_interval(
    values: np.ndarray, *, draws: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = values.shape[0]
    estimates = np.empty((draws, values.shape[1]), dtype=np.float64)
    chunk = 500
    for start in range(0, draws, chunk):
        stop = min(draws, start + chunk)
        indices = rng.integers(0, samples, size=(stop - start, samples))
        estimates[start:stop] = values[indices].mean(axis=1)
    return (
        np.percentile(estimates, 2.5, axis=0),
        np.percentile(estimates, 97.5, axis=0),
    )


def collect(output_root: Path, v1_root: Path) -> tuple[list[dict[str, object]], str]:
    config = campaign.load_config()
    config_sha = campaign.canonical_hash(config)
    hashes = campaign.source_hashes()
    groups: dict[tuple[int, float], list[tuple[campaign.Task, Path, Path]]] = {}
    missing = []
    for task in campaign.expand_tasks(config):
        complete, reason = campaign.verified_complete(
            output_root, task, config_sha256=config_sha, hashes=hashes, v1_root=v1_root
        )
        if not complete:
            missing.append(f"{task.task_id}: {reason}")
            continue
        result_path, completion_path = campaign.result_paths(
            v1_root if task.import_v1 else output_root, task
        )
        groups.setdefault((task.ny, task.alpha_1), []).append(
            (task, result_path, completion_path)
        )
    if missing:
        raise RuntimeError(
            f"campaign is incomplete ({len(missing)} of {campaign.EXPECTED_TASKS} tasks pending); "
            f"first={missing[0]}"
        )

    rows: list[dict[str, object]] = []
    digest = hashlib.sha256()
    for (ny, alpha_1), entries in sorted(groups.items()):
        gaps = np.full(
            (campaign.SAMPLES_PER_CASE, campaign.SLOW_GAP_COUNT),
            np.nan,
            dtype=np.float64,
        )
        for task, result_path, completion_path in entries:
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            digest.update(task.task_id.encode("utf-8"))
            digest.update(str(completion["result_sha256"]).encode("ascii"))
            with np.load(result_path, allow_pickle=False) as archive:
                gaps[np.asarray(task.case_sample_indices)] = np.asarray(
                    archive["slow_effective_gaps_per_cycle"], dtype=np.float64
                )
        if not np.all(np.isfinite(gaps)):
            raise FloatingPointError(f"incomplete/nonfinite case gaps at Ny={ny}, alpha1={alpha_1}")
        mean = gaps.mean(axis=0)
        sem = gaps.std(axis=0, ddof=1) / np.sqrt(gaps.shape[0])
        low, high = bootstrap_mean_interval(
            gaps,
            draws=int(config["bootstrap_draws"]),
            seed=_case_seed(int(config["bootstrap_seed"]), ny, alpha_1),
        )
        rows.append(
            {
                "Ny": ny,
                "alpha_1": alpha_1,
                "samples": gaps.shape[0],
                "gaps": gaps,
                "mean": mean,
                "sem": sem,
                "ci_low": low,
                "ci_high": high,
            }
        )
    if len(rows) != campaign.EXPECTED_CASES:
        raise RuntimeError("analysis did not assemble all 67 cases")
    return rows, digest.hexdigest()


def write_outputs(output_root: Path, scratch_root: Path, v1_root: Path) -> None:
    rows, result_set_sha = collect(output_root, v1_root)
    scratch_root.mkdir(parents=True, exist_ok=True)
    analysis_root = output_root / "analysis"
    case_ny = np.asarray([row["Ny"] for row in rows], dtype=np.int64)
    case_alpha = np.asarray([row["alpha_1"] for row in rows], dtype=np.float64)
    arrays = {
        "schema": np.asarray(ANALYSIS_SCHEMA),
        "sampling_revision": np.asarray(campaign.REVISION),
        "estimator": np.asarray("trajectory_first_quenched_finite_time_endpoint"),
        "uncertainty": np.asarray("sample_SEM_and_10000_draw_whole_trajectory_bootstrap_95pct"),
        "case_Ny": case_ny,
        "case_alpha_1": case_alpha,
        "raw_gaps": np.stack([row["gaps"] for row in rows]),
        "mean_gaps": np.stack([row["mean"] for row in rows]),
        "sem_gaps": np.stack([row["sem"] for row in rows]),
        "bootstrap_ci_low": np.stack([row["ci_low"] for row in rows]),
        "bootstrap_ci_high": np.stack([row["ci_high"] for row in rows]),
        "result_set_sha256": np.asarray(result_set_sha),
        "imported_v1_revision": np.asarray(campaign.V1_REVISION),
        "imported_v1_result_sha256": np.asarray(campaign.V1_RESULT_SHA256),
        "imported_v1_case_sample_indices": np.arange(25, dtype=np.int64),
    }
    local_npz = scratch_root / "quenched_gap_summary.npz"
    campaign._atomic_npz(local_npz, arrays)
    local_csv = scratch_root / "quenched_gap_summary.csv"
    with local_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["Ny", "alpha_1", "gap_rank", "samples", "mean", "sem", "ci95_low", "ci95_high"]
        )
        for row in rows:
            for rank in range(campaign.SLOW_GAP_COUNT):
                writer.writerow(
                    [
                        row["Ny"],
                        f"{float(row['alpha_1']):.12g}",
                        rank + 1,
                        row["samples"],
                        f"{row['mean'][rank]:.17g}",
                        f"{row['sem'][rank]:.17g}",
                        f"{row['ci_low'][rank]:.17g}",
                        f"{row['ci_high'][rank]:.17g}",
                    ]
                )
    published = {}
    for local in (local_npz, local_csv):
        published[local.name] = campaign.publish_file(local, analysis_root / local.name)
    manifest = {
        "schema": ANALYSIS_SCHEMA,
        "sampling_revision": campaign.REVISION,
        "estimator_order": "extract five gaps per trajectory, then average over trajectories",
        "result_set_sha256": result_set_sha,
        "cases": len(rows),
        "samples_per_case": campaign.SAMPLES_PER_CASE,
        "bootstrap_draws": campaign.expected_config()["bootstrap_draws"],
        "pinned_v1_import": {
            "revision": campaign.V1_REVISION,
            "task_id": "Ny040_a1-1_batch-000_samples-000-024",
            "result_sha256": campaign.V1_RESULT_SHA256,
            "case_sample_indices": list(range(25)),
        },
        "files": published,
    }
    local_manifest = scratch_root / "analysis_manifest.json"
    campaign._atomic_json(local_manifest, manifest)
    campaign.publish_file(local_manifest, analysis_root / local_manifest.name)
    print(
        f"[analysis complete] cases={len(rows)}, output={analysis_root}, "
        f"result_set_sha256={result_set_sha[:16]}...",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=campaign.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--v1-root", type=Path, default=campaign.DEFAULT_V1_ROOT)
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    scratch = args.scratch_root or Path(tempfile.mkdtemp(prefix="slot17_analysis_"))
    write_outputs(args.output_root.resolve(), scratch.resolve(), args.v1_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
