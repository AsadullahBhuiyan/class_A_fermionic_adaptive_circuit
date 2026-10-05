#!/usr/bin/env python3
"""Read-only integrity and scientific-progress monitor for the S50 RK4 campaign.

The monitor never imports the runner's checkpoint loader because that loader is
allowed to discard an invalid pair before a deterministic rerun.  Monitoring
must be observational: it verifies what is present and reports problems without
changing campaign state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import run_parent_schrodinger_rk4_s50 as campaign


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(saved: Any, key: str) -> Any:
    return np.asarray(saved[key]).item()


def verify_checkpoint(
    output_root: Path,
    task: dict[str, Any],
    *,
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    checkpoint, receipt_path = campaign.checkpoint_paths(output_root, task)
    if not checkpoint.exists() and not receipt_path.exists():
        return {"status": "absent"}
    if not checkpoint.is_file() or not receipt_path.is_file():
        return {"status": "invalid", "reason": "incomplete checkpoint pair"}
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema") != campaign.CHECKPOINT_SCHEMA:
            raise RuntimeError("checkpoint receipt schema mismatch")
        if receipt.get("metadata") != metadata:
            raise RuntimeError("checkpoint receipt identity mismatch")
        record = receipt["checkpoint"]
        if record.get("name") != checkpoint.name:
            raise RuntimeError("checkpoint filename mismatch")
        if int(record.get("bytes", -1)) != checkpoint.stat().st_size:
            raise RuntimeError("checkpoint byte-count mismatch")
        if record.get("sha256") != sha256_path(checkpoint):
            raise RuntimeError("checkpoint checksum mismatch")

        intervals = int(config["evolution"]["flux_intervals"])
        steps_per_interval = int(config["evolution"]["steps_per_interval"])
        count = intervals + 1
        nx = int(config["geometry"]["Nx"])
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(_scalar(saved, "schema")) != campaign.CHECKPOINT_SCHEMA:
                raise RuntimeError("checkpoint payload schema mismatch")
            if json.loads(str(_scalar(saved, "metadata_json"))) != metadata:
                raise RuntimeError("checkpoint payload identity mismatch")
            completed_interval = int(_scalar(saved, "completed_interval"))
            completed_step = int(_scalar(saved, "completed_step"))
            if completed_step != completed_interval * steps_per_interval:
                raise RuntimeError("checkpoint step/interval mismatch")
            if not 0 <= completed_interval < intervals:
                raise RuntimeError("checkpoint interval is out of bounds")
            if int(receipt.get("completed_interval", -1)) != completed_interval:
                raise RuntimeError("receipt/payload interval mismatch")
            if int(receipt.get("completed_step", -1)) != completed_step:
                raise RuntimeError("receipt/payload step mismatch")

            frame = np.asarray(saved["frame"])
            if frame.dtype != np.complex128 or frame.ndim != 2 or frame.shape[0] != 2 * nx * int(config["geometry"]["Ny"]):
                raise RuntimeError("checkpoint frame shape/dtype mismatch")
            if not np.all(np.isfinite(frame)):
                raise RuntimeError("checkpoint frame is nonfinite")
            one_d = ("phi", "time", "N_left", "N_right", "current_left", "energy")
            arrays = {key: np.asarray(saved[key]) for key in one_d}
            density_x = np.asarray(saved["density_x"])
            if any(value.shape != (count,) for value in arrays.values()) or density_x.shape != (count, nx):
                raise RuntimeError("checkpoint observable shape mismatch")
            prefix = slice(0, completed_interval + 1)
            future = slice(completed_interval + 1, None)
            if any(not np.all(np.isfinite(value[prefix])) for value in arrays.values()):
                raise RuntimeError("checkpoint observable prefix is nonfinite")
            if not np.all(np.isfinite(density_x[prefix])):
                raise RuntimeError("checkpoint density prefix is nonfinite")
            if any(not np.all(np.isnan(value[future])) for value in arrays.values()):
                raise RuntimeError("checkpoint future observable slots are populated")
            if not np.all(np.isnan(density_x[future])):
                raise RuntimeError("checkpoint future density slots are populated")

            total = arrays["N_left"][prefix] + arrays["N_right"][prefix]
            charge_residual = float(np.max(np.abs(total - total[0])))
            post_gram = float(_scalar(saved, "maximum_post_qr_gram"))
            pre_gram = float(_scalar(saved, "maximum_pre_qr_gram"))
            if charge_residual > float(config["acceptance"]["charge_conservation_tolerance"]):
                raise RuntimeError("checkpoint violates charge conservation")
            if post_gram > float(config["acceptance"]["post_qr_gram_tolerance"]):
                raise RuntimeError("checkpoint violates post-QR tolerance")
            q_x = 0.5 * (
                (arrays["N_right"][: completed_interval + 1] - arrays["N_right"][0])
                - (arrays["N_left"][: completed_interval + 1] - arrays["N_left"][0])
            )
            return {
                "status": "checkpoint",
                "completed_interval": completed_interval,
                "completed_step": completed_step,
                "phi": arrays["phi"][: completed_interval + 1].tolist(),
                "q_x": q_x.tolist(),
                "maximum_charge_residual": charge_residual,
                "maximum_pre_qr_gram_residual": pre_gram,
                "maximum_post_qr_gram_residual": post_gram,
            }
    except Exception as exc:
        return {"status": "invalid", "reason": f"{type(exc).__name__}: {exc}"}


def inspect_campaign(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = campaign.load_config(config_path)
    campaign.validate_config(config)
    context = campaign.source_context(config)
    hashes = campaign.source_hashes()
    config_hash = campaign.scientific_config_hash(config)
    tasks = campaign.tasks(config)
    rows: list[dict[str, Any]] = []
    observations: dict[tuple[str, str, int], dict[str, Any]] = {}

    for task in tasks:
        source_row = context["rows"][task["source_task_id"]]
        metadata = campaign._metadata(task, config_hash, hashes, source_row)
        ok, reason, completion = campaign.verify_result(
            output_root,
            task,
            config_hash=config_hash,
            hashes=hashes,
            source_row=source_row,
            config=config,
        )
        if ok:
            result, _ = campaign.result_paths(output_root, task)
            with np.load(result, allow_pickle=False) as saved:
                obs = {
                    "status": "complete",
                    "completed_interval": int(config["evolution"]["flux_intervals"]),
                    "phi": np.asarray(saved["phi"]).tolist(),
                    "q_x": np.asarray(saved["q_x"]).tolist(),
                    "maximum_charge_residual": float(np.max(np.abs(np.asarray(saved["delta_N_total"])))),
                    "maximum_pre_qr_gram_residual": float(_scalar(saved, "maximum_pre_qr_gram_residual")),
                    "maximum_post_qr_gram_residual": float(_scalar(saved, "maximum_post_qr_gram_residual")),
                }
            obs["elapsed_seconds"] = float(completion["elapsed_seconds"])
        else:
            obs = verify_checkpoint(output_root, task, metadata=metadata, config=config)
            if obs["status"] == "absent":
                result, completion_path = campaign.result_paths(output_root, task)
                if result.exists() or completion_path.exists():
                    obs = {"status": "invalid", "reason": reason}
                else:
                    failure = campaign.failure_path(output_root, task)
                    if failure.is_file():
                        obs = {"status": "failed", "reason": failure.read_text(encoding="utf-8")}
                    else:
                        obs = {"status": "pending"}
        row = {**task, **obs}
        rows.append(row)
        if "q_x" in obs:
            observations[(task["wall"], task["direction"], int(task["sample_id"]))] = obs

    status_counts = Counter(row["status"] for row in rows)
    interval_counts = Counter(
        int(row["completed_interval"])
        for row in rows
        if row["status"] in {"checkpoint", "complete"}
    )
    numerical_rows = [row for row in rows if "maximum_charge_residual" in row]
    numerical = {
        "maximum_charge_residual": max((row["maximum_charge_residual"] for row in numerical_rows), default=None),
        "maximum_pre_qr_gram_residual": max((row["maximum_pre_qr_gram_residual"] for row in numerical_rows), default=None),
        "maximum_post_qr_gram_residual": max((row["maximum_post_qr_gram_residual"] for row in numerical_rows), default=None),
    }

    paired: dict[str, Any] = {}
    for wall in config["ensemble"]["walls"]:
        wall_pairs = []
        for sample_id in config["source_campaign"]["sample_ids"]:
            ccw = observations.get((wall, "ccw", int(sample_id)))
            cw = observations.get((wall, "cw", int(sample_id)))
            if ccw is None or cw is None:
                continue
            common = min(int(ccw["completed_interval"]), int(cw["completed_interval"]))
            q_ccw = float(ccw["q_x"][common])
            q_cw = float(cw["q_x"][common])
            wall_pairs.append({
                "sample_id": int(sample_id),
                "common_interval": common,
                "q_x_ccw": q_ccw,
                "q_x_cw": q_cw,
                "q_x_odd": 0.5 * (q_ccw - q_cw),
                "q_x_even": 0.5 * (q_ccw + q_cw),
            })
        by_interval: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in wall_pairs:
            by_interval[int(row["common_interval"])].append(row)
        summaries = []
        for interval, group in sorted(by_interval.items()):
            odd = np.asarray([row["q_x_odd"] for row in group])
            even = np.asarray([row["q_x_even"] for row in group])
            summaries.append({
                "common_interval": interval,
                "abs_phi": float(2.0 * np.pi * interval / int(config["evolution"]["flux_intervals"])),
                "paired_samples": len(group),
                "q_x_odd_mean": float(np.mean(odd)),
                "q_x_odd_sd": float(np.std(odd, ddof=1)) if len(odd) > 1 else None,
                "maximum_abs_q_x_even": float(np.max(np.abs(even))),
            })
        latest_shared = None
        if wall_pairs:
            interval = min(int(row["common_interval"]) for row in wall_pairs)
            q_ccw = np.asarray([
                observations[(wall, "ccw", int(row["sample_id"]))]["q_x"][interval]
                for row in wall_pairs
            ], dtype=float)
            q_cw = np.asarray([
                observations[(wall, "cw", int(row["sample_id"]))]["q_x"][interval]
                for row in wall_pairs
            ], dtype=float)
            odd = 0.5 * (q_ccw - q_cw)
            even = 0.5 * (q_ccw + q_cw)
            latest_shared = {
                "common_interval": interval,
                "abs_phi": float(2.0 * np.pi * interval / int(config["evolution"]["flux_intervals"])),
                "paired_samples": len(wall_pairs),
                "q_x_ccw_mean": float(np.mean(q_ccw)),
                "q_x_ccw_sd": float(np.std(q_ccw, ddof=1)) if len(q_ccw) > 1 else None,
                "q_x_cw_mean": float(np.mean(q_cw)),
                "q_x_cw_sd": float(np.std(q_cw, ddof=1)) if len(q_cw) > 1 else None,
                "q_x_odd_mean": float(np.mean(odd)),
                "q_x_odd_sd": float(np.std(odd, ddof=1)) if len(odd) > 1 else None,
                "q_x_odd_minimum": float(np.min(odd)),
                "q_x_odd_maximum": float(np.max(odd)),
                "maximum_abs_q_x_even": float(np.max(np.abs(even))),
            }
        paired[wall] = {
            "pairs": wall_pairs,
            "latest_boundary_shared_by_all_available_pairs": latest_shared,
            "summaries_by_common_progress": summaries,
        }

    invalid = [
        {"task_id": row["task_id"], "status": row["status"], "reason": row.get("reason")}
        for row in rows
        if row["status"] in {"invalid", "failed"}
    ]
    return {
        "schema": "parent_schrodinger_rk4_monitor_v1",
        "campaign_id": config["campaign_id"],
        "config_sha256": config_hash,
        "tasks_total": len(tasks),
        "status_counts": dict(sorted(status_counts.items())),
        "checkpoint_interval_histogram": {str(k): v for k, v in sorted(interval_counts.items())},
        "numerical_health": numerical,
        "invalid_or_failed": invalid,
        "paired_progress": paired,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    report = inspect_campaign(args.config.resolve(), args.output_root.resolve())
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 1 if report["invalid_or_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
