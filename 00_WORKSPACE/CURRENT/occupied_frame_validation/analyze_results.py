#!/usr/bin/env python3
"""Analyze campaign shards and render the RevTeX validation note."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent

TIMING_METRICS = (
    "trajectory_total",
    "cycle_total",
    "schedule_or_replay_decode",
    "site_total",
    "orbital_fetch",
    "channel_total",
    "occupation_probability",
    "branch_sample_or_validate",
    "measurement_total",
    "feedback_total",
    "branch_record_bookkeeping",
    "gain_overlap",
    "gain_residual_projection",
    "gain_normalize_append",
    "loss_overlap",
    "loss_householder_build",
    "loss_householder_apply",
    "loss_column_delete",
    "unitary_or_even_transport",
    "gram_residual_check",
    "covariance_occupation_matvec",
    "rank1_resolvent_action",
    "regularized_dense_fallback",
    "measurement_outer_update",
    "reset_outer_update",
    "hermitian_symmetrization",
    "initial_state_generation",
    "state_copy_for_replay",
    "pure_frame_factorization",
    "maxmix_frame_allocation",
    "physical_covariance_reconstruction",
    "choi_projector_reconstruction",
    "global_entropy_eigh_or_svd",
    "regional_entropy_eigh_or_svd",
    "real_space_chern",
)

OBSERVER_METRICS = (
    "charge_observer",
    "global_entropy_eigh_or_svd",
    "regional_entropy_eigh_or_svd",
    "physical_covariance_reconstruction",
    "real_space_chern",
    "checkpoint_copy",
)

CORRECTNESS_ARRAYS = (
    "covariance_relative_error",
    "covariance_max_error",
    "charge_covariance",
    "charge_frame",
    "charge_error",
    "global_entropy_covariance",
    "global_entropy_frame",
    "global_entropy_error",
    "regional_entropy_covariance",
    "regional_entropy_frame",
    "regional_entropy_error",
    "real_space_chern_covariance",
    "real_space_chern_frame",
    "real_space_chern_error",
    "frame_gram_residual",
    "frame_rank",
    "frame_rank_expected_from_words",
    "frame_rank_word_residual",
    "frame_rank_charge_residual",
    "branch_probability_error",
    "log_weight_covariance",
    "log_weight_frame",
    "log_weight_error",
    "minimum_selected_probability_covariance",
    "minimum_selected_probability_frame",
    "choi_relative_error",
    "event_sketch_error_by_cycle",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows([{key: row.get(key) for key in fields} for row in rows])
    temporary.replace(path)


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_bootstrap_ratio(
    numerator: np.ndarray, denominator: np.ndarray, *, repetitions: int, seed: int
) -> tuple[float, float]:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if numerator.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(repetitions), dtype=np.float64)
    for index in range(int(repetitions)):
        selected = rng.integers(0, numerator.size, numerator.size)
        estimates[index] = np.mean(numerator[selected]) / np.mean(denominator[selected])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def classification(interval: tuple[float, float], band: float) -> str:
    low, high = interval
    if low > 1.0 + band:
        return "faster"
    if high < 1.0 - band:
        return "slower"
    return "comparable"


def descriptive(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "standard_deviation": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _plot_path(root: Path, stem: str) -> tuple[Path, Path]:
    return root / "figures" / f"{stem}.png", root / "figures" / f"{stem}.pdf"


def _save_figure(root: Path, stem: str) -> list[Path]:
    png, pdf = _plot_path(root, stem)
    for path in (png, pdf):
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
    plt.close()
    return [png, pdf]


def load_correctness(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples = int(config["geometry"]["samples"])
    aggregate: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    for spec in config["initializations"]:
        label = str(spec["label"])
        arrays: dict[str, list[np.ndarray]] = {name: [] for name in CORRECTNESS_ARRAYS}
        for sample in range(samples):
            directory = root / "raw/correctness" / label / f"sample_{sample:02d}"
            summary_path = directory / "summary.json"
            data_path = directory / "cycle_data.npz"
            if not summary_path.exists() or not data_path.exists():
                continue
            summary = read_json(summary_path)
            summaries.append(summary)
            with np.load(data_path) as data:
                aggregate[f"{label}__cycles"] = np.asarray(data["cycles"])
                for name in CORRECTNESS_ARRAYS:
                    if name in data:
                        arrays[name].append(np.asarray(data[name]))
        for name, values in arrays.items():
            if values:
                aggregate[f"{label}__{name}"] = np.stack(values, axis=0)
    return aggregate, summaries


def correctness_tables(
    root: Path, config: dict[str, Any], aggregate: dict[str, Any], summaries: list[dict[str, Any]]
) -> list[Path]:
    sample_rows: list[dict[str, Any]] = []
    for summary in summaries:
        row: dict[str, Any] = {
            "initialization": summary["label"],
            "sample": summary["sample"],
            "seed": summary["seed"],
            "cpu": summary["cpu"],
            "passed": summary["gate_passed"],
            "branch_disagreements": summary["branch_disagreement_count"],
            "dense_fallbacks": summary["regularized_dense_fallback_count"],
            "dense_choi_audit": bool(summary.get("dense_choi_audit", False)),
            "peak_rss_kib_process": summary["peak_rss_kib_process"],
        }
        for key, value in summary["checks"].items():
            row[f"max_{key}_error"] = value
        for backend in ("covariance", "frame"):
            payload = summary["backends"][backend]
            row[f"{backend}_wall_ns"] = payload["wall_ns"]
            row[f"{backend}_cpu_ns"] = payload["cpu_ns"]
            row[f"{backend}_native_state_bytes"] = payload["native_state_bytes"]
        sample_rows.append(row)
    sample_path = root / "processed/tables/correctness_per_sample.csv"
    write_csv_atomic(sample_path, sample_rows)

    cycle_rows: list[dict[str, Any]] = []
    for spec in config["initializations"]:
        label = str(spec["label"])
        cycles = aggregate.get(f"{label}__cycles")
        if cycles is None:
            continue
        for position, cycle in enumerate(cycles):
            row = {"initialization": label, "cycle": int(cycle)}
            for name in (
                "covariance_relative_error",
                "covariance_max_error",
                "branch_probability_error",
                "log_weight_error",
                "global_entropy_error",
                "regional_entropy_error",
                "real_space_chern_error",
                "frame_gram_residual",
                "choi_relative_error",
            ):
                key = f"{label}__{name}"
                if key not in aggregate:
                    continue
                values = np.asarray(aggregate[key])[:, position]
                finite = values[np.isfinite(values)]
                row[f"{name}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
                row[f"{name}_max"] = float(np.max(finite)) if finite.size else float("nan")
            cycle_rows.append(row)
    cycle_path = root / "processed/tables/correctness_per_cycle.csv"
    write_csv_atomic(cycle_path, cycle_rows)
    return [sample_path, cycle_path]


def detailed_timing_table(root: Path, summaries: list[dict[str, Any]]) -> Path:
    rows: list[dict[str, Any]] = []
    labels = sorted({str(summary["label"]) for summary in summaries})
    for label in labels:
        selected = [summary for summary in summaries if summary["label"] == label]
        for backend in ("covariance", "frame"):
            for metric in TIMING_METRICS:
                total = 0
                calls = 0
                for summary in selected:
                    timing = summary["backends"][backend].get("timing") or {}
                    total += int(timing.get("total_ns", {}).get(metric, 0))
                    calls += int(timing.get("counts", {}).get(f"{metric}_calls", 0))
                rows.append(
                    {
                        "initialization": label,
                        "backend": backend,
                        "metric": metric,
                        "raw_total_ns": total,
                        "calls": calls,
                        "time_per_call_ns": float(total / calls) if calls else float("nan"),
                        "time_per_sample_ns": float(total / len(selected)) if selected else float("nan"),
                    }
                )
            for metric in OBSERVER_METRICS:
                total = sum(
                    int(summary["backends"][backend].get("observer_timing_ns", {}).get(metric, 0))
                    for summary in selected
                )
                rows.append(
                    {
                        "initialization": label,
                        "backend": backend,
                        "metric": f"observer.{metric}",
                        "raw_total_ns": total,
                        "calls": len(selected),
                        "time_per_call_ns": float(total / len(selected)) if selected else float("nan"),
                        "time_per_sample_ns": float(total / len(selected)) if selected else float("nan"),
                    }
                )
            sketch_total = sum(
                int(summary["backends"][backend].get("event_sketch_observer_ns", 0))
                for summary in selected
            )
            rows.append(
                {
                    "initialization": label,
                    "backend": backend,
                    "metric": "observer.event_sketch",
                    "raw_total_ns": sketch_total,
                    "calls": len(selected),
                    "time_per_call_ns": float(sketch_total / len(selected)) if selected else float("nan"),
                    "time_per_sample_ns": float(sketch_total / len(selected)) if selected else float("nan"),
                }
            )
    path = root / "processed/tables/detailed_timing_breakdown.csv"
    write_csv_atomic(path, rows)
    return path


def operation_count_table(root: Path, summaries: list[dict[str, Any]]) -> Path:
    rows: list[dict[str, Any]] = []
    labels = sorted({str(summary["label"]) for summary in summaries})
    for label in labels:
        selected = [summary for summary in summaries if summary["label"] == label]
        for backend in ("covariance", "frame"):
            names = sorted(
                {
                    name
                    for summary in selected
                    for name in (
                        (summary["backends"][backend].get("timing") or {}).get("counts", {})
                    )
                }
            )
            for name in names:
                values = np.asarray(
                    [
                        int(
                            (summary["backends"][backend].get("timing") or {})
                            .get("counts", {})
                            .get(name, 0)
                        )
                        for summary in selected
                    ],
                    dtype=np.int64,
                )
                rows.append(
                    {
                        "initialization": label,
                        "backend": backend,
                        "counter": name,
                        "total": int(np.sum(values)),
                        "mean_per_sample": float(np.mean(values)),
                        "minimum_per_sample": int(np.min(values)),
                        "maximum_per_sample": int(np.max(values)),
                    }
                )
    path = root / "processed/tables/operation_counts.csv"
    write_csv_atomic(path, rows)
    return path


def benchmark_analysis(
    root: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    samples = int(config["geometry"]["samples"])
    repetitions = int(config["analysis"]["paired_bootstrap_repetitions"])
    band = float(config["analysis"]["practical_equivalence_fraction"])
    result: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for family_index, spec in enumerate(config["initializations"]):
        label = str(spec["label"])
        summaries = []
        for sample in range(samples):
            path = root / "raw/benchmark" / label / f"sample_{sample:02d}/summary.json"
            if path.exists():
                summaries.append(read_json(path))
        if not summaries:
            continue
        wall = {backend: np.asarray([row["backends"][backend]["wall_ns"] for row in summaries]) for backend in ("covariance", "frame")}
        update = {backend: np.asarray([row["backends"][backend]["update_ns"] for row in summaries]) for backend in ("covariance", "frame")}
        cpu = {backend: np.asarray([row["backends"][backend]["cpu_ns"] for row in summaries]) for backend in ("covariance", "frame")}
        native = {backend: np.asarray([row["backends"][backend]["native_state_bytes"] for row in summaries]) for backend in ("covariance", "frame")}
        update_ratio = update["covariance"] / update["frame"]
        wall_ratio = wall["covariance"] / wall["frame"]
        update_interval = paired_bootstrap_ratio(
            update["covariance"], update["frame"], repetitions=repetitions, seed=1201 + family_index
        )
        wall_interval = paired_bootstrap_ratio(
            wall["covariance"], wall["frame"], repetitions=repetitions, seed=2201 + family_index
        )
        family = {
            "paired_samples": len(summaries),
            "update_speedup": {
                **descriptive(update_ratio),
                "ratio_of_means": float(np.mean(update["covariance"]) / np.mean(update["frame"])),
                "bootstrap_95_interval": list(update_interval),
                "classification": classification(update_interval, band),
            },
            "end_to_end_speedup": {
                **descriptive(wall_ratio),
                "ratio_of_means": float(np.mean(wall["covariance"]) / np.mean(wall["frame"])),
                "bootstrap_95_interval": list(wall_interval),
                "classification": classification(wall_interval, band),
            },
            "mean_wall_seconds": {backend: float(np.mean(values) / 1e9) for backend, values in wall.items()},
            "mean_cpu_seconds": {backend: float(np.mean(values) / 1e9) for backend, values in cpu.items()},
            "mean_native_state_bytes": {backend: float(np.mean(values)) for backend, values in native.items()},
        }
        benchmark_status = root / "status/benchmark.json"
        if benchmark_status.exists():
            stage = read_json(benchmark_status).get("parallel_timing_ns", {}).get(label, {})
            elapsed = int(stage.get("stage_total", 0))
            if elapsed > 0:
                family["ten_way_throughput_trajectories_per_second"] = float(
                    2 * len(summaries) * 1e9 / elapsed
                )
        result[label] = family
        arrays[f"{label}__update_speedup"] = update_ratio
        arrays[f"{label}__wall_speedup"] = wall_ratio
        for backend in ("covariance", "frame"):
            cycle_rows = []
            for summary in summaries:
                per_cycle = (summary["backends"][backend].get("timing") or {}).get(
                    "per_cycle_total_ns", {}
                )
                cycle_rows.append(
                    [int(per_cycle.get(str(cycle), {}).get("cycle_total", 0)) for cycle in range(1, int(config["geometry"]["cycles"]) + 1)]
                )
            arrays[f"{label}__{backend}__cycle_total_ns"] = np.asarray(cycle_rows, dtype=np.int64)
        for index, summary in enumerate(summaries):
            rows.append(
                {
                    "initialization": label,
                    "sample": summary["sample"],
                    "seed": summary["seed"],
                    "cpu": summary["cpu"],
                    "backend_order": "-then-".join(summary["backend_order"]),
                    "covariance_update_ns": int(update["covariance"][index]),
                    "frame_update_ns": int(update["frame"][index]),
                    "update_speedup": float(update_ratio[index]),
                    "covariance_wall_ns": int(wall["covariance"][index]),
                    "frame_wall_ns": int(wall["frame"][index]),
                    "end_to_end_speedup": float(wall_ratio[index]),
                    "covariance_cpu_ns": int(cpu["covariance"][index]),
                    "frame_cpu_ns": int(cpu["frame"][index]),
                    "covariance_native_state_bytes": int(native["covariance"][index]),
                    "frame_native_state_bytes": int(native["frame"][index]),
                    "peak_rss_kib_process": summary["peak_rss_kib_process"],
                }
            )
    return result, rows, arrays


def render_figures(
    root: Path,
    config: dict[str, Any],
    aggregate: dict[str, Any],
    benchmark: dict[str, Any],
    benchmark_arrays: dict[str, np.ndarray],
    timing_rows: list[dict[str, Any]],
) -> list[Path]:
    plt.rcParams.update({"font.size": 9, "figure.dpi": int(config["analysis"]["dpi"])})
    width = float(config["analysis"]["figure_width_inches"])
    outputs: list[Path] = []
    colors = {"random_pure": "#2b6cb0", "maxmix": "#c05621"}

    fig, axes = plt.subplots(1, 2, figsize=(width, 2.8))
    for spec in config["initializations"]:
        label = str(spec["label"])
        cycles = aggregate.get(f"{label}__cycles")
        if cycles is None:
            continue
        covariance = aggregate[f"{label}__covariance_relative_error"]
        gram = aggregate[f"{label}__frame_gram_residual"]
        axes[0].semilogy(cycles, np.maximum(np.max(covariance, axis=0), 1e-18), label=label, color=colors[label])
        axes[1].semilogy(cycles, np.maximum(np.max(gram, axis=0), 1e-18), label=label, color=colors[label])
    axes[0].set(xlabel="cycle", ylabel="max relative covariance error")
    axes[1].set(xlabel="cycle", ylabel="max normalized Gram residual")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    outputs += _save_figure(root, "correctness_errors")

    fig, axes = plt.subplots(1, 2, figsize=(width, 2.8))
    for spec in config["initializations"]:
        label = str(spec["label"])
        cycles = aggregate.get(f"{label}__cycles")
        if cycles is None:
            continue
        for backend, style in (("covariance", "-"), ("frame", "--")):
            global_values = aggregate[f"{label}__global_entropy_{backend}"]
            regional_values = aggregate[f"{label}__regional_entropy_{backend}"]
            axes[0].plot(cycles, np.mean(global_values, axis=0), style, color=colors[label], label=f"{label}: {backend}")
            axes[1].plot(cycles, np.mean(regional_values, axis=0), style, color=colors[label], label=f"{label}: {backend}")
    axes[0].set(xlabel="cycle", ylabel="global entropy")
    axes[1].set(xlabel="cycle", ylabel="half-system entropy")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)
    outputs += _save_figure(root, "entropy_trajectories")

    fig, ax = plt.subplots(figsize=(width, 3.0))
    for spec in config["initializations"]:
        label = str(spec["label"])
        cycles = aggregate.get(f"{label}__cycles")
        if cycles is None:
            continue
        for backend, style in (("covariance", "-"), ("frame", "--")):
            values = aggregate[f"{label}__real_space_chern_{backend}"]
            mean = np.mean(values, axis=0)
            stderr = (
                np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
                if values.shape[0] > 1
                else np.zeros(values.shape[1], dtype=np.float64)
            )
            ax.plot(
                cycles,
                mean,
                style,
                color=colors[label],
                label=f"{label}: {backend}",
            )
            ax.fill_between(
                cycles,
                mean - stderr,
                mean + stderr,
                color=colors[label],
                alpha=0.10,
            )
    ax.axhline(1.0, color="0.25", linewidth=0.8, linestyle=":", label="C=+1")
    ax.set(xlabel="cycle", ylabel="disk real-space Chern estimator")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=7)
    outputs += _save_figure(root, "chern_convergence_four_lanes")

    if benchmark:
        fig, axes = plt.subplots(1, 2, figsize=(width, 2.8))
        labels = list(benchmark)
        positions = np.arange(len(labels))
        for axis, kind, title in (
            (axes[0], "update_speedup", "update-only"),
            (axes[1], "end_to_end_speedup", "end-to-end"),
        ):
            for position, label in enumerate(labels):
                values = benchmark_arrays[f"{label}__{'update' if kind == 'update_speedup' else 'wall'}_speedup"]
                summary = benchmark[label][kind]
                low, high = summary["bootstrap_95_interval"]
                center = summary["ratio_of_means"]
                axis.scatter(np.full(values.size, position), values, color=colors[label], alpha=0.65, s=18)
                axis.errorbar(position, center, yerr=[[center - low], [high - center]], fmt="o", color="black", capsize=3)
            axis.axhspan(0.95, 1.05, color="0.8", alpha=0.4)
            axis.axhline(1.0, color="0.35", linewidth=0.8)
            axis.set(xticks=positions, xticklabels=labels, ylabel="covariance / frame speed ratio", title=title)
            axis.grid(axis="y", alpha=0.25)
        outputs += _save_figure(root, "paired_speedup")

        fig, ax = plt.subplots(figsize=(width, 3.0))
        for spec in config["initializations"]:
            label = str(spec["label"])
            for backend, style in (("covariance", "-"), ("frame", "--")):
                key = f"{label}__{backend}__cycle_total_ns"
                if key in benchmark_arrays:
                    values = benchmark_arrays[key] / 1e6
                    ax.plot(np.arange(1, values.shape[1] + 1), np.mean(values, axis=0), style, color=colors[label], label=f"{label}: {backend}")
        ax.set(xlabel="cycle", ylabel="mean cycle time [ms]")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=2, fontsize=7)
        outputs += _save_figure(root, "cycle_timing")

        fig, axes = plt.subplots(1, 2, figsize=(width, 2.8))
        labels = list(benchmark)
        positions = np.arange(len(labels))
        width_bar = 0.34
        for backend_index, backend in enumerate(("covariance", "frame")):
            native_mib = [
                benchmark[label]["mean_native_state_bytes"][backend] / (1024.0**2)
                for label in labels
            ]
            axes[0].bar(
                positions + (backend_index - 0.5) * width_bar,
                native_mib,
                width_bar,
                label=backend,
            )
        axes[0].set(
            xticks=positions,
            xticklabels=labels,
            ylabel="native state [MiB]",
            title="native-state storage",
        )
        axes[0].legend(frameon=False)
        throughput = [
            benchmark[label].get("ten_way_throughput_trajectories_per_second", np.nan)
            for label in labels
        ]
        axes[1].bar(positions, throughput, color=[colors[label] for label in labels])
        axes[1].set(
            xticks=positions,
            xticklabels=labels,
            ylabel="backend trajectories / s",
            title="ten-way campaign throughput",
        )
        for axis in axes:
            axis.grid(axis="y", alpha=0.25)
        outputs += _save_figure(root, "throughput_and_memory")

    nonzero = [row for row in timing_rows if row["raw_total_ns"] and row["metric"] not in ("trajectory_total", "cycle_total", "site_total", "channel_total")]
    if nonzero:
        totals: dict[tuple[str, str], list[tuple[str, float]]] = {}
        for row in nonzero:
            totals.setdefault((row["initialization"], row["backend"]), []).append((row["metric"], row["time_per_sample_ns"] / 1e9))
        fig, axes = plt.subplots(len(totals), 1, figsize=(width, max(3.0, 1.8 * len(totals))), squeeze=False)
        for axis, ((label, backend), values) in zip(axes[:, 0], totals.items()):
            values = sorted(values, key=lambda item: item[1], reverse=True)[:10][::-1]
            axis.barh([item[0] for item in values], [item[1] for item in values], color=colors[label], alpha=0.8)
            axis.set_title(f"{label}: {backend}", fontsize=9)
            axis.set_xlabel("mean raw time per sample [s]")
        plt.tight_layout()
        outputs += _save_figure(root, "detailed_timing_breakdown")
    return outputs


def analyze(root: Path) -> dict[str, Any]:
    config = read_json(root / "campaign_config.v1.json")
    aggregate, correctness_summaries = load_correctness(root, config)
    if not correctness_summaries:
        raise RuntimeError("No correctness shards are available to analyze.")
    array_path = root / "processed/arrays/correctness_aggregate.npz"
    save_npz_atomic(array_path, **aggregate)
    outputs = [array_path]
    outputs += correctness_tables(root, config, aggregate, correctness_summaries)
    timing_path = detailed_timing_table(root, correctness_summaries)
    outputs.append(timing_path)
    outputs.append(operation_count_table(root, correctness_summaries))
    with timing_path.open(newline="") as handle:
        timing_rows = list(csv.DictReader(handle))
    for row in timing_rows:
        row["raw_total_ns"] = int(row["raw_total_ns"])
        row["time_per_sample_ns"] = float(row["time_per_sample_ns"])

    benchmark, benchmark_rows, benchmark_arrays = benchmark_analysis(root, config)
    if benchmark_rows:
        benchmark_table = root / "processed/tables/benchmark_per_sample.csv"
        benchmark_array_path = root / "processed/arrays/benchmark_aggregate.npz"
        write_csv_atomic(benchmark_table, benchmark_rows)
        save_npz_atomic(benchmark_array_path, **benchmark_arrays)
        outputs += [benchmark_table, benchmark_array_path]

    figures = render_figures(root, config, aggregate, benchmark, benchmark_arrays, timing_rows)
    outputs += figures
    numerical_gate_passed = all(
        bool(summary["gate_passed"]) for summary in correctness_summaries
    )
    gate_status = (
        read_json(root / "status/correctness_gate.json")
        if (root / "status/correctness_gate.json").exists()
        else {}
    )
    physics_gate = gate_status.get("physics_gate")
    physics_gate_passed = (
        bool(physics_gate.get("passed")) if isinstance(physics_gate, dict) else True
    )
    gate_passed = numerical_gate_passed and physics_gate_passed
    error_maxima = {
        label: {
            name: float(np.nanmax(values))
            for name in (
                "covariance_relative_error",
                "covariance_max_error",
                "branch_probability_error",
                "log_weight_error",
                "global_entropy_error",
                "regional_entropy_error",
                "real_space_chern_error",
                "frame_gram_residual",
                "choi_relative_error",
            )
            if (values := aggregate.get(f"{label}__{name}")) is not None
            and np.any(np.isfinite(values))
        }
        for label in (str(spec["label"]) for spec in config["initializations"])
    }
    summary = {
        "schema_version": 1,
        "correctness_gate_passed": gate_passed,
        "numerical_correctness_gate_passed": numerical_gate_passed,
        "uniform_insulator_physics_gate_passed": physics_gate_passed,
        "correctness_samples": len(correctness_summaries),
        "correctness_error_maxima": error_maxima,
        "benchmark": benchmark,
        "uniform_insulator_physics_gate": physics_gate,
        "performance_is_descriptive": True,
        "practical_equivalence_band": [0.95, 1.05],
        "products": {str(path.relative_to(root)): sha256_file(path) for path in outputs},
    }
    summary_path = root / "processed/analysis_summary.json"
    write_json_atomic(summary_path, summary)
    return summary


def latex_escape(value: Any) -> str:
    text = str(value)
    for source, replacement in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        text = text.replace(source, replacement)
    return text


def render_report(root: Path) -> Path:
    summary_path = root / "processed/analysis_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else analyze(root)
    config = read_json(root / "campaign_config.v1.json")
    nx = int(config["geometry"]["Nx"])
    ny = int(config["geometry"]["Ny"])
    terminal_cycle = int(config["geometry"]["cycles"])
    correctness_word = "passed" if summary["correctness_gate_passed"] else "failed"
    performance_paragraphs = []
    for label, payload in summary.get("benchmark", {}).items():
        update = payload["update_speedup"]
        end = payload["end_to_end_speedup"]
        performance_paragraphs.append(
            f"For {latex_escape(label)}, the update-only covariance/frame ratio was "
            f"{update['ratio_of_means']:.3f} with paired-bootstrap 95\\% interval "
            f"[{update['bootstrap_95_interval'][0]:.3f},{update['bootstrap_95_interval'][1]:.3f}] "
            f"({latex_escape(update['classification'])}); the end-to-end ratio was "
            f"{end['ratio_of_means']:.3f} [{end['bootstrap_95_interval'][0]:.3f},"
            f"{end['bootstrap_95_interval'][1]:.3f}] ({latex_escape(end['classification'])})."
        )
    if not performance_paragraphs:
        performance_paragraphs.append(
            "No benchmark result is reported because the correctness gate did not pass or the benchmark stage is incomplete."
        )
    maxima_lines = []
    for label, values in summary["correctness_error_maxima"].items():
        maxima_lines.append(
            latex_escape(label)
            + ": "
            + ", ".join(f"{latex_escape(name)}={value:.3e}" for name, value in values.items())
            + r".\\"
        )
    physics = summary.get("uniform_insulator_physics_gate") or {}
    physics_lines = []
    for lane, values in physics.get("lanes", {}).items():
        physics_lines.append(
            f"{latex_escape(lane)}: $C_0={values['initial_chern_mean']:.4f}$, "
            f"$C_{{T}}={values['terminal_chern_mean']:.4f}$, "
            f"$S_{{\rm global}}(T)/N={values['terminal_global_entropy_density_mean']:.3e}$ "
            f"with $T={terminal_cycle}$.\\\\"
        )
    if not physics_lines:
        physics_lines.append("Uniform-insulator convergence products are unavailable.\\\\")
    report = rf"""\documentclass[aps,prb,onecolumn,superscriptaddress]{{revtex4-2}}
\usepackage{{amsmath,amssymb,graphicx,booktabs}}
\begin{{document}}
\title{{Occupied-Frame Validation for a U(1)-Symmetric Gaussian Transfer Circuit}}
\author{{Automated validation campaign}}
\date{{\today}}
\begin{{abstract}}
We validate a frame-native implementation of the canonical CPU Markov circuit against its rank-one covariance implementation at $N_x={nx}$, $N_y={ny}$ for random pure and maximally mixed initial states. The first physics gate is a uniform $\alpha=1$, domain-wall-free Chern insulator. The combined numerical and physics gate {correctness_word}. Performance data, when present, are descriptive and are not part of implementation validity.
\end{{abstract}}
\maketitle

\section{{Representation and update rules}}
For a pure physical Gaussian state, the occupied frame $V\in\mathbb C^{{N\times r}}$ obeys $V^\dagger V=I$ and $C=VV^\dagger$. For a mixed physical state we use a doubled pure frame $F=(A;B)$ with $C=BB^\dagger$; maximal mixing starts from $F=2^{{-1/2}}(I;I)$. Given a normalized orbital $e$, gain uses $z=F^\dagger e$, $h=e-Fz$, and appends $h/\lVert h\rVert$. Loss builds a stable complex Householder reflector taking $z$ to the last coefficient coordinate, applies its rank-one right action, and deletes the final column. Correct occupied and empty projections are chronological loss--gain and gain--loss words. Under perfect correction, mismatches reduce to a bare loss or gain.

The evolution kernel evaluates overlaps from the local $n_{{\rm shell}}=1$ support. It performs no covariance construction, pseudoinverse, or global QR. Covariance and doubled-projector construction occur only in the instrumented correctness observers and declared checkpoints; those costs are excluded from update-only benchmark timing.

\section{{Campaign design}}
The locked campaign uses $N={2*int(config['geometry']['Nx'])*int(config['geometry']['Ny'])}$ physical orbitals, {int(config['geometry']['cycles'])} cycles, complex128 arithmetic, $\alpha_1=\alpha_2=1$, no domain wall, and ten paired records per initialization. Ten single-sample worker processes are pinned to distinct physical cores on one NUMA node. Backend order is counterbalanced five/five, and workers synchronize before each timed backend phase.

\section{{Correctness}}
The gate {correctness_word}. Maximum observed errors were:\\
{chr(10).join(maxima_lines)}
Any branch disagreement, nonfinite value, dense covariance fallback, missing shard, or rank inconsistency is a hard failure. Full covariance and doubled-Choi comparisons are diagnostic-only operations.

\begin{{figure}}[t]
\includegraphics[width=\linewidth]{{../figures/correctness_errors.pdf}}
\caption{{Cycle-resolved covariance and Gram errors.}}
\end{{figure}}

\section{{Uniform-insulator convergence}}
The finite-size target projector gives $C_{{\rm target}}={float(physics.get('target_chern_finite_size', float('nan'))):.6f}$. Cycle-resolved disk-partition real-space Chern estimators are retained independently for all four initialization/backend lanes. Half-system entropy is retained for every lane, while global entropy density supplies the purification diagnostic for the maximally mixed initialization.\\
{chr(10).join(physics_lines)}

\begin{{figure}}[t]
\includegraphics[width=\linewidth]{{../figures/chern_convergence_four_lanes.pdf}}
\caption{{Real-space Chern convergence for the covariance and frame replays of both initial-state families.}}
\end{{figure}}

\section{{Performance}}
{' '.join(performance_paragraphs)} Ratios greater than one favor the frame backend. Classification uses a predeclared $\pm5\%$ practical-equivalence band and the paired-bootstrap interval, with each record as the independent unit.

\section{{Limitations}}
This study covers a CPU-only ${nx}\times{ny}$, $n_{{\rm shell}}=1$ uniform controller and {terminal_cycle} cycles. The disk Chern estimator is a finite-size convergence diagnostic; the calibrated target is therefore used instead of assuming an exactly quantized finite-lattice value. Purification frames retain doubled-space costs and may show no speed improvement. Observer Chern contractions, eigendecompositions/SVDs, and diagnostic reconstructions are reported separately from dynamics.

\end{{document}}
"""
    tex_path = root / "reports/occupied_frame_validation.tex"
    write_text_atomic(tex_path, report)
    compile_log = root / "logs/report_compile.log"
    if shutil.which("latexmk"):
        process = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            text=True,
            capture_output=True,
            check=False,
        )
        write_text_atomic(compile_log, process.stdout + "\n" + process.stderr)
    else:
        write_text_atomic(compile_log, "latexmk is unavailable; the RevTeX source was generated but not compiled.\n")
    report_manifest = {
        "tex": {"path": str(tex_path.relative_to(root)), "sha256": sha256_file(tex_path)},
        "pdf": None,
        "compile_log": str(compile_log.relative_to(root)),
    }
    pdf_path = tex_path.with_suffix(".pdf")
    if pdf_path.exists():
        report_manifest["pdf"] = {
            "path": str(pdf_path.relative_to(root)),
            "sha256": sha256_file(pdf_path),
        }
    write_json_atomic(root / "reports/report_manifest.json", report_manifest)
    return tex_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("analyze", "report"))
    parser.add_argument("--campaign-root", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.campaign_root.resolve()
    if args.mode == "analyze":
        summary = analyze(root)
        print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
    else:
        path = render_report(root)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
