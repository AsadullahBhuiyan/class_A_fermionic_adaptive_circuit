#!/usr/bin/env python3
"""Paired trajectory-bootstrap analysis for the S10 online flux-ramp pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_online_flux_ramp as campaign  # noqa: E402


ANALYSIS_SCHEMA = "online_flux_ramp_analysis_v1"
WALLS = ("soft", "hard")
DIRECTIONS = ("ccw", "cw")


def _bootstrap_mean(
    values: np.ndarray, *, draws: int, seed: int, confidence: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[0] < 1:
        raise ValueError("bootstrap input requires a nonempty trajectory axis")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.shape[0], size=(int(draws), array.shape[0]))
    samples = array[indices].mean(axis=1)
    alpha = 0.5 * (1.0 - float(confidence))
    return (
        array.mean(axis=0),
        np.quantile(samples, alpha, axis=0),
        np.quantile(samples, 1.0 - alpha, axis=0),
    )


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "Computer Modern Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.dpi": 300,
        }
    )


def _save_figure(fig: plt.Figure, root: Path, stem: str) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    pdf = root / f"{stem}.pdf"
    png = root / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}


def _panel_letters(axes: Any) -> None:
    for label, axis in zip(("(a)", "(b)", "(c)", "(d)"), np.asarray(axes).flat):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes, va="bottom", ha="left", fontsize=9)


def _mean_ci_cube(
    values: np.ndarray, config: dict[str, Any], *, seed_offset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Input ordering is wall, direction, sample, then observable axes.
    shape = values.shape[:2] + values.shape[3:]
    mean = np.empty(shape)
    low = np.empty(shape)
    high = np.empty(shape)
    ensemble = config["ensemble"]
    for wall_index in range(values.shape[0]):
        for direction_index in range(values.shape[1]):
            result = _bootstrap_mean(
                values[wall_index, direction_index],
                draws=int(ensemble["bootstrap_draws"]),
                seed=int(ensemble["bootstrap_seed"]) + seed_offset + 10 * wall_index + direction_index,
                confidence=float(ensemble["confidence_level"]),
            )
            mean[wall_index, direction_index], low[wall_index, direction_index], high[wall_index, direction_index] = result
    return mean, low, high


def _mean_and_sample_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the trajectory mean and one-sample standard deviation."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[0] < 1:
        raise ValueError("mean/std input requires a nonempty trajectory axis")
    ddof = 1 if array.shape[0] > 1 else 0
    return array.mean(axis=0), array.std(axis=0, ddof=ddof)


def _load(config: dict[str, Any], output_root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    hashes = campaign.source_hashes()
    config_hash = campaign.scientific_config_hash(config)
    burnins = {task["task_id"]: task for task in campaign.expand_burnin_tasks(config)}
    ramp_tasks = campaign.expand_ramp_tasks(config)
    count = int(config["ensemble"]["samples_per_wall"])
    fields = (
        "phi",
        "N_left",
        "N_right",
        "N_total",
        "delta_N_left",
        "delta_N_right",
        "delta_N_total",
        "q_x_raw",
        "source_A_left",
        "source_A_right",
        "net_injected_charge",
        "injection_count",
        "q_left_corrected",
        "q_right_corrected",
        "q_x_corrected",
        "charge_continuity_residual",
        "corrected_balance_residual",
        "rank",
    )
    arrays = {field: np.empty((2, 2, count, 17), dtype=np.float64) for field in fields}
    density_fields = ("density_x", "delta_density_x", "source_A_x", "corrected_delta_density_x")
    arrays.update(
        {field: np.empty((2, 2, count, 17, 16), dtype=np.float64) for field in density_fields}
    )
    scalar_partition = np.empty((2, 2, count), dtype=np.float64)
    burnin_hashes: dict[str, str] = {}
    for parent in burnins.values():
        ok, reason, completion = campaign.verify_pair(output_root, parent, config_hash, hashes)
        if not ok:
            raise RuntimeError(f"unverified burn-in {parent['task_id']}: {reason}")
        burnin_hashes[parent["task_id"]] = str(completion["result"]["sha256"])
    for task in ramp_tasks:
        burnin_sha = burnin_hashes[task["burnin_task_id"]]
        ok, reason, _ = campaign.verify_pair(
            output_root, task, config_hash, hashes, burnin_sha256=burnin_sha
        )
        if not ok:
            raise RuntimeError(f"unverified ramp {task['task_id']}: {reason}")
        wall_index = WALLS.index(task["wall"])
        direction_index = DIRECTIONS.index(task["direction"])
        sample = int(task["sample_id"])
        result_path, _ = campaign.result_paths(output_root, task)
        with np.load(result_path, allow_pickle=False) as saved:
            for field in fields + density_fields:
                arrays[field][wall_index, direction_index, sample] = np.asarray(saved[field])
            scalar_partition[wall_index, direction_index, sample] = float(
                saved["maximum_source_partition_residual"]
            )
    arrays["maximum_source_partition_residual"] = scalar_partition
    metadata = {
        "schema": ANALYSIS_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_hash": config_hash,
        "source_hashes": hashes,
        "walls": list(WALLS),
        "directions": list(DIRECTIONS),
        "samples_per_wall": count,
        "paired_resampling_unit": config["ensemble"]["paired_resampling_unit"],
        "uncertainty": f"{100*float(config['ensemble']['confidence_level']):g}% trajectory bootstrap interval",
    }
    return arrays, metadata


def _line_with_band(ax: plt.Axes, x: np.ndarray, mean: np.ndarray, low: np.ndarray, high: np.ndarray, **kwargs: Any) -> None:
    color = kwargs.get("color")
    ax.plot(x, mean, **kwargs)
    ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)


def _write_long_csv(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "wall", "direction", "sample_id", "cycle", "phi", "delta_N_left", "delta_N_right",
        "source_A_left", "source_A_right", "q_left_corrected", "q_right_corrected",
        "q_x_raw", "q_x_corrected", "net_injected_charge", "charge_continuity_residual",
        "corrected_balance_residual",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for wi, wall in enumerate(WALLS):
            for di, direction in enumerate(DIRECTIONS):
                for sample in range(arrays["phi"].shape[2]):
                    for cycle in range(17):
                        row = {"wall": wall, "direction": direction, "sample_id": sample, "cycle": cycle}
                        for field in fields[4:]:
                            row[field] = float(arrays[field][wi, di, sample, cycle])
                        writer.writerow(row)


def _write_qx_mean_std_csv(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["wall", "direction", "ramp_index", "phi", "mean_q_x", "std_q_x", "samples"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for wi, wall in enumerate(WALLS):
            for di, direction in enumerate(DIRECTIONS):
                phi = arrays["phi"][wi, di, 0]
                if not np.array_equal(arrays["phi"][wi, di], np.broadcast_to(phi, arrays["phi"][wi, di].shape)):
                    raise RuntimeError(f"nonidentical phase schedule across {wall}/{direction} trajectories")
                mean, std = _mean_and_sample_std(arrays["q_x_raw"][wi, di])
                for index, (phase, center, width) in enumerate(zip(phi, mean, std)):
                    writer.writerow(
                        {
                            "wall": wall,
                            "direction": direction,
                            "ramp_index": index,
                            "phi": float(phase),
                            "mean_q_x": float(center),
                            "std_q_x": float(width),
                            "samples": int(arrays["q_x_raw"].shape[2]),
                        }
                    )


def analyze(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    arrays, metadata = _load(config, output_root)
    analysis_root = output_root / "analysis"
    figures_root = analysis_root / "figures"
    analysis_root.mkdir(parents=True, exist_ok=True)
    estimates: dict[str, np.ndarray] = {}
    for offset, field in enumerate(
        (
            "delta_N_left",
            "delta_N_right",
            "q_x_raw",
            "q_left_corrected",
            "q_right_corrected",
            "q_x_corrected",
            "delta_density_x",
            "corrected_delta_density_x",
        )
    ):
        mean, low, high = _mean_ci_cube(arrays[field], config, seed_offset=100 * offset)
        estimates[f"{field}_mean"] = mean
        estimates[f"{field}_ci_low"] = low
        estimates[f"{field}_ci_high"] = high

    raw_pair = arrays["q_x_raw"][:, 0] - arrays["q_x_raw"][:, 1]
    raw_sum = arrays["q_x_raw"][:, 0] + arrays["q_x_raw"][:, 1]
    corrected_pair = arrays["q_x_corrected"][:, 0] - arrays["q_x_corrected"][:, 1]
    corrected_sum = arrays["q_x_corrected"][:, 0] + arrays["q_x_corrected"][:, 1]
    paired_fields = {
        "q_x_raw_direction_odd": 0.5 * raw_pair,
        "q_x_raw_direction_even": 0.5 * raw_sum,
        "q_x_corrected_direction_odd": 0.5 * corrected_pair,
        "q_x_corrected_direction_even": 0.5 * corrected_sum,
    }
    ensemble = config["ensemble"]
    for offset, (field, values) in enumerate(paired_fields.items(), start=20):
        means, lows, highs = [], [], []
        for wi in range(2):
            mean, low, high = _bootstrap_mean(
                values[wi],
                draws=int(ensemble["bootstrap_draws"]),
                seed=int(ensemble["bootstrap_seed"]) + 100 * offset + wi,
                confidence=float(ensemble["confidence_level"]),
            )
            means.append(mean); lows.append(low); highs.append(high)
        estimates[f"{field}_mean"] = np.asarray(means)
        estimates[f"{field}_ci_low"] = np.asarray(lows)
        estimates[f"{field}_ci_high"] = np.asarray(highs)

    np.savez_compressed(
        analysis_root / "online_flux_ramp_aggregate.npz",
        metadata_json=np.asarray(campaign.canonical_json(metadata)),
        **arrays,
        **estimates,
    )
    _write_long_csv(analysis_root / "online_flux_ramp_trajectories.csv", arrays)
    qx_mean_std_csv = analysis_root / "online_flux_ramp_qx_mean_std.csv"
    _write_qx_mean_std_csv(qx_mean_std_csv, arrays)

    _configure_style()
    colors = {"ccw": "#1f77b4", "cw": "#d62728"}
    styles = {"ccw": ("o", "-"), "cw": ("s", "--")}
    theta = np.linspace(0.0, 2.0 * np.pi, 17)

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), sharex=True, sharey=True)
    for wi, wall in enumerate(WALLS):
        ax = axes[wi]
        for di, direction in enumerate(DIRECTIONS):
            phi = arrays["phi"][wi, di, 0]
            if not np.array_equal(arrays["phi"][wi, di], np.broadcast_to(phi, arrays["phi"][wi, di].shape)):
                raise RuntimeError(f"nonidentical phase schedule across {wall}/{direction} trajectories")
            mean, std = _mean_and_sample_std(arrays["q_x_raw"][wi, di])
            marker, linestyle = styles[direction]
            _line_with_band(
                ax,
                phi,
                mean,
                mean - std,
                mean + std,
                color=colors[direction],
                marker=marker,
                linestyle=linestyle,
                markerfacecolor="white",
                markersize=3,
                linewidth=1.0,
                label=(r"CCW: $0\to+2\pi$" if direction == "ccw" else r"CW: $0\to-2\pi$"),
            )
        ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.8)
        ax.axvline(0.0, color="0.65", linestyle=":", linewidth=0.7)
        ax.set_title(f"{wall.capitalize()} wall")
        ax.set_xlabel(r"signed flux $\phi$")
        ax.set_xlim(-2.0 * np.pi, 2.0 * np.pi)
        ax.set_xticks(
            [-2.0 * np.pi, -np.pi, 0.0, np.pi, 2.0 * np.pi],
            [r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"],
        )
        ax.legend(frameon=False, loc="best")
    axes[0].set_ylabel(r"raw response $q_x=(\Delta N_R-\Delta N_L)/2$")
    for label, axis in zip(("(a)", "(b)"), axes):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes, va="bottom", ha="left", fontsize=9)
    fig.tight_layout(pad=0.7, w_pad=0.9)
    qx_mean_std_figures = _save_figure(fig, figures_root, "online_flux_ramp_qx_mean_std")

    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.8), sharex=True)
    for wi, wall in enumerate(WALLS):
        for row, field in enumerate(("q_x_raw", "q_x_corrected")):
            ax = axes[row, wi]
            for di, direction in enumerate(DIRECTIONS):
                marker, linestyle = styles[direction]
                _line_with_band(
                    ax,
                    theta,
                    estimates[f"{field}_mean"][wi, di],
                    estimates[f"{field}_ci_low"][wi, di],
                    estimates[f"{field}_ci_high"][wi, di],
                    color=colors[direction], marker=marker, linestyle=linestyle,
                    markerfacecolor="white", markersize=3, linewidth=1.0,
                    label=direction.upper(),
                )
            ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.8)
            ax.set_title(f"{wall.capitalize()} wall: {'raw' if row == 0 else 'source-corrected'}")
            ax.set_ylabel(r"$q_x$")
            if row == 1:
                ax.set_xlabel(r"ramp coordinate $|\phi|$")
            ax.set_xticks([0, np.pi, 2 * np.pi], [r"$0$", r"$\pi$", r"$2\pi$"])
            ax.legend(frameon=False, ncol=2)
    _panel_letters(axes)
    fig.tight_layout(pad=0.7, w_pad=0.9, h_pad=0.9)
    response_figures = _save_figure(fig, figures_root, "online_flux_ramp_response")

    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.9))
    for wi, wall in enumerate(WALLS):
        ax = axes[0, wi]
        for di, direction in enumerate(DIRECTIONS):
            marker, linestyle = styles[direction]
            for field, color, label in (
                ("q_left_corrected", "#2ca02c", r"$q_L$"),
                ("q_right_corrected", "#9467bd", r"$q_R$"),
            ):
                _line_with_band(
                    ax, theta, estimates[f"{field}_mean"][wi, di],
                    estimates[f"{field}_ci_low"][wi, di], estimates[f"{field}_ci_high"][wi, di],
                    color=color, marker=marker, linestyle=linestyle, markerfacecolor="white",
                    markersize=2.5, linewidth=0.9, label=f"{label}, {direction.upper()}",
                )
        ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.8)
        ax.set_title(f"{wall.capitalize()} corrected regional charge")
        ax.set_xlabel(r"$|\phi|$"); ax.set_ylabel("charge")
        ax.set_xticks([0, np.pi, 2 * np.pi], [r"$0$", r"$\pi$", r"$2\pi$"])
        ax.legend(frameon=False, ncol=2)
    ax = axes[1, 0]
    wall_colors = {"soft": "#2ca02c", "hard": "#1f77b4"}
    for wi, wall in enumerate(WALLS):
        for parity, linestyle in (("odd", "-"), ("even", "--")):
            field = f"q_x_corrected_direction_{parity}"
            _line_with_band(
                ax, theta, estimates[f"{field}_mean"][wi], estimates[f"{field}_ci_low"][wi],
                estimates[f"{field}_ci_high"][wi], color=wall_colors[wall], linestyle=linestyle,
                linewidth=1.0, label=f"{wall}, {parity}",
            )
    ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.8)
    ax.set_title("Direction parity, corrected")
    ax.set_xlabel(r"$|\phi|$"); ax.set_ylabel(r"$q_x^{\rm odd/even}$")
    ax.set_xticks([0, np.pi, 2 * np.pi], [r"$0$", r"$\pi$", r"$2\pi$"])
    ax.legend(frameon=False, ncol=2)
    ax = axes[1, 1]
    positions, labels, values = [], [], []
    position = 1
    for wi, wall in enumerate(WALLS):
        for di, direction in enumerate(DIRECTIONS):
            positions.append(position); labels.append(f"{wall[0].upper()}\n{direction.upper()}")
            values.append(arrays["q_x_corrected"][wi, di, :, -1]); position += 1
        position += 0.5
    ax.boxplot(values, positions=positions, widths=0.55, showfliers=False)
    for pos, vals in zip(positions, values):
        ax.scatter(np.full_like(vals, pos, dtype=float), vals, s=12, facecolors="white", edgecolors="0.25", linewidths=0.7)
    ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.8)
    ax.set_xticks(positions, labels); ax.set_ylabel(r"endpoint $q_x^{\rm corr}$")
    ax.set_title("Trajectory endpoints")
    _panel_letters(axes)
    fig.tight_layout(pad=0.7, w_pad=0.9, h_pad=0.9)
    diagnostic_figures = _save_figure(fig, figures_root, "online_flux_ramp_diagnostics")

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), sharey=True)
    x = np.arange(16)
    for wi, wall in enumerate(WALLS):
        ax = axes[wi]
        for di, direction in enumerate(DIRECTIONS):
            marker, linestyle = styles[direction]
            _line_with_band(
                ax, x, estimates["corrected_delta_density_x_mean"][wi, di, -1],
                estimates["corrected_delta_density_x_ci_low"][wi, di, -1],
                estimates["corrected_delta_density_x_ci_high"][wi, di, -1],
                color=colors[direction], marker=marker, linestyle=linestyle,
                markerfacecolor="white", markersize=3, linewidth=1.0, label=direction.upper(),
            )
        ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.8)
        ax.axvline(7.5, color="0.55", linestyle=":", linewidth=0.7)
        ax.set_title(f"{wall.capitalize()} endpoint")
        ax.set_xlabel(r"$x$"); ax.legend(frameon=False)
    axes[0].set_ylabel(r"$\Delta n_x-A_x$")
    for label, axis in zip(("(a)", "(b)"), axes):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes, va="bottom", ha="left", fontsize=9)
    fig.tight_layout(pad=0.7, w_pad=0.9)
    density_figures = _save_figure(fig, figures_root, "online_flux_ramp_density")

    summary = {
        **metadata,
        "maximum_absolute_charge_continuity_residual": float(np.max(np.abs(arrays["charge_continuity_residual"]))),
        "maximum_absolute_corrected_balance_residual": float(np.max(np.abs(arrays["corrected_balance_residual"]))),
        "maximum_source_partition_residual": float(np.max(arrays["maximum_source_partition_residual"])),
        "endpoint_corrected_q_x_mean": {
            wall: {
                direction: float(estimates["q_x_corrected_mean"][wi, di, -1])
                for di, direction in enumerate(DIRECTIONS)
            }
            for wi, wall in enumerate(WALLS)
        },
        "figures": {
            "raw_q_x_mean_sample_std": qx_mean_std_figures,
            "response": response_figures,
            "diagnostics": diagnostic_figures,
            "density": density_figures,
        },
        "raw_q_x_mean_sample_std_csv": str(qx_mean_std_csv),
        "claim_boundary": "Online monitored ramp; quantization is not an acceptance gate.",
    }
    campaign._atomic_json(analysis_root / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=campaign.DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=campaign.DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = campaign.load_config(args.config.resolve())
    campaign.validate_config(config)
    analyze(config, args.output_root.resolve())


if __name__ == "__main__":
    main()
