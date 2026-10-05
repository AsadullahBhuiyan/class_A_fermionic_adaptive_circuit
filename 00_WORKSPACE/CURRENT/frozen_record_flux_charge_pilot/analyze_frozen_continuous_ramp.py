#!/usr/bin/env python3
"""Analyze continuously replayed frozen-record flux ramps."""

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

import run_frozen_continuous_ramp as campaign  # noqa: E402


ANALYSIS_SCHEMA = "frozen_record_continuous_ramp_analysis_v1"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "Computer Modern Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.dpi": 300,
        }
    )


def _save(fig: plt.Figure, root: Path, stem: str) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    pdf, png = root / f"{stem}.pdf", root / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}


def _mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def _line_band(
    ax: plt.Axes, x: np.ndarray, values: np.ndarray, *, color: str,
    marker: str, linestyle: str, label: str,
) -> None:
    mean, std = _mean_std(values)
    ax.plot(
        x, mean, color=color, marker=marker, linestyle=linestyle,
        markerfacecolor="white", markersize=2.8, linewidth=1.0, label=label,
    )
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.17, linewidth=0)


def _panel_letters(axes: Any) -> None:
    for index, axis in enumerate(np.asarray(axes).flat):
        axis.text(
            -0.14, 1.04, f"({chr(ord('a') + index)})",
            transform=axis.transAxes, va="bottom", ha="left", fontsize=9,
        )


def _load(config: dict[str, Any], output_root: Path) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    burnins = campaign.verify_burnins(config)
    hashes = campaign.source_hashes()
    config_hash = campaign.scientific_config_hash(config)
    references = campaign.verified_references(config, output_root, burnins, config_hash, hashes)
    if len(references) != 20:
        raise RuntimeError(f"analysis requires 20 verified references, found {len(references)}")
    sample_count = int(config["ensemble"]["samples_per_wall"])
    fields = (
        "phi", "N_left", "N_right", "N_total", "delta_N_left",
        "delta_N_right", "delta_N_total", "q_x", "net_injected_charge",
        "injection_count", "charge_continuity_residual", "rank",
        "minimum_selected_probability_by_cycle", "branch_log_probability_by_cycle",
    )
    data: dict[int, dict[str, np.ndarray]] = {}
    for cycles in config["dynamics"]["ramp_cycles"]:
        length = int(cycles) + 1
        values = {
            field: np.empty((2, 2, sample_count, length), dtype=np.float64)
            for field in fields
        }
        values["density_x"] = np.empty((2, 2, sample_count, length, 16), dtype=np.float64)
        values["record_prefix_sha256"] = np.empty((2, 2, sample_count), dtype="U64")
        data[int(cycles)] = values
    for task in campaign.replay_tasks(config):
        dependencies = campaign.dependencies_for(task, burnins, references)
        ok, reason, _ = campaign.verify_pair(output_root, task, config_hash, hashes, dependencies)
        if not ok:
            raise RuntimeError(f"unverified replay {task['task_id']}: {reason}")
        wi = campaign.WALLS.index(task["wall"])
        di = campaign.DIRECTIONS.index(task["direction"])
        si = int(task["sample_id"])
        values = data[int(task["cycles"])]
        result, _ = campaign.result_paths(output_root, task)
        with np.load(result, allow_pickle=False) as saved:
            for field in fields:
                values[field][wi, di, si] = np.asarray(saved[field])
            values["density_x"][wi, di, si] = np.asarray(saved["density_x"])
            values["record_prefix_sha256"][wi, di, si] = str(saved["record_prefix_sha256"])
    metadata = {
        "schema": ANALYSIS_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_hash": config_hash,
        "source_hashes": hashes,
        "samples_per_wall": sample_count,
        "walls": list(campaign.WALLS),
        "directions": list(campaign.DIRECTIONS),
        "ramp_cycles": list(config["dynamics"]["ramp_cycles"]),
        "uncertainty": "one sample standard deviation across frozen-record parents",
    }
    return data, metadata


def _validate_paired(data: dict[int, dict[str, np.ndarray]]) -> None:
    for cycles, values in data.items():
        if not np.array_equal(values["rank"][:, 0], values["rank"][:, 1]):
            raise RuntimeError(f"CW/CCW rank histories differ at M={cycles}")
        if not np.array_equal(
            values["net_injected_charge"][:, 0], values["net_injected_charge"][:, 1]
        ):
            raise RuntimeError(f"CW/CCW injected-charge histories differ at M={cycles}")
        if not np.array_equal(values["injection_count"][:, 0], values["injection_count"][:, 1]):
            raise RuntimeError(f"CW/CCW injection-count histories differ at M={cycles}")
        if not np.array_equal(
            values["record_prefix_sha256"][:, 0], values["record_prefix_sha256"][:, 1]
        ):
            raise RuntimeError(f"CW/CCW record prefixes differ at M={cycles}")
        if not np.array_equal(values["phi"][:, 0], -values["phi"][:, 1]):
            raise RuntimeError(f"CW/CCW schedules do not reverse at M={cycles}")


def _write_csv(path: Path, data: dict[int, dict[str, np.ndarray]]) -> None:
    fields = [
        "wall", "direction", "sample_id", "M", "cycle", "phi",
        "delta_N_left", "delta_N_right", "delta_N_total", "q_x",
        "net_injected_charge", "injection_count", "rank",
        "minimum_selected_probability", "branch_log_probability",
        "charge_continuity_residual",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cycles, values in sorted(data.items()):
            for wi, wall in enumerate(campaign.WALLS):
                for di, direction in enumerate(campaign.DIRECTIONS):
                    for sample in range(values["phi"].shape[2]):
                        for index in range(cycles + 1):
                            writer.writerow(
                                {
                                    "wall": wall,
                                    "direction": direction,
                                    "sample_id": sample,
                                    "M": cycles,
                                    "cycle": index,
                                    "phi": values["phi"][wi, di, sample, index],
                                    "delta_N_left": values["delta_N_left"][wi, di, sample, index],
                                    "delta_N_right": values["delta_N_right"][wi, di, sample, index],
                                    "delta_N_total": values["delta_N_total"][wi, di, sample, index],
                                    "q_x": values["q_x"][wi, di, sample, index],
                                    "net_injected_charge": values["net_injected_charge"][wi, di, sample, index],
                                    "injection_count": values["injection_count"][wi, di, sample, index],
                                    "rank": values["rank"][wi, di, sample, index],
                                    "minimum_selected_probability": values["minimum_selected_probability_by_cycle"][wi, di, sample, index],
                                    "branch_log_probability": values["branch_log_probability_by_cycle"][wi, di, sample, index],
                                    "charge_continuity_residual": values["charge_continuity_residual"][wi, di, sample, index],
                                }
                            )


def analyze(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    data, metadata = _load(config, output_root)
    _validate_paired(data)
    analysis_root = output_root / "analysis"
    figures_root = analysis_root / "figures"
    analysis_root.mkdir(parents=True, exist_ok=True)
    aggregate: dict[str, Any] = {"metadata_json": np.asarray(campaign.canonical_json(metadata))}
    for cycles, values in sorted(data.items()):
        for field, value in values.items():
            aggregate[f"M{cycles:03d}_{field}"] = value
    np.savez_compressed(analysis_root / "frozen_continuous_ramp_aggregate.npz", **aggregate)
    csv_path = analysis_root / "frozen_continuous_ramp_trajectories.csv"
    _write_csv(csv_path, data)

    _style()
    colors = {"ccw": "#1f77b4", "cw": "#d62728"}
    markers = {"ccw": "o", "cw": "^"}
    linestyles = {"ccw": "-", "cw": ":"}
    cycles_list = [int(value) for value in config["dynamics"]["ramp_cycles"]]

    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.35), sharex=True, sharey=True)
    for wi, wall in enumerate(campaign.WALLS):
        for column, cycles in enumerate(cycles_list):
            ax = axes[wi, column]
            values = data[cycles]
            theta = np.linspace(0.0, 2.0 * np.pi, cycles + 1)
            for di, direction in enumerate(campaign.DIRECTIONS):
                _line_band(
                    ax, theta, values["q_x"][wi, di], color=colors[direction],
                    marker=markers[direction], linestyle=linestyles[direction],
                    label=(r"CCW: $+2\pi$" if direction == "ccw" else r"CW: $-2\pi$"),
                )
            ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.75)
            ax.set_title(f"{wall.capitalize()} wall, $M={cycles}$")
            ax.set_xticks([0.0, np.pi, 2.0 * np.pi], [r"$0$", r"$\pi$", r"$2\pi$"])
            if wi == 1:
                ax.set_xlabel(r"flux-path coordinate $|\phi|$")
            if column == 0:
                ax.set_ylabel(r"$q_x=(\Delta N_R-\Delta N_L)/2$")
            if wi == 0 and column == 0:
                ax.legend(frameon=False)
    _panel_letters(axes)
    fig.tight_layout(pad=0.7, w_pad=0.7, h_pad=0.9)
    qx_figures = _save(fig, figures_root, "frozen_continuous_ramp_qx")

    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.35), sharex=True, sharey=True)
    for wi, wall in enumerate(campaign.WALLS):
        for column, cycles in enumerate(cycles_list):
            ax = axes[wi, column]
            values = data[cycles]
            theta = np.linspace(0.0, 2.0 * np.pi, cycles + 1)
            for di, direction in enumerate(campaign.DIRECTIONS):
                _line_band(
                    ax, theta, values["delta_N_left"][wi, di],
                    color=colors[direction], marker=markers[direction],
                    linestyle=linestyles[direction],
                    label=rf"$\Delta N_L$, {direction.upper()}",
                )
                _line_band(
                    ax, theta, values["delta_N_right"][wi, di],
                    color=colors[direction], marker=None,
                    linestyle="--" if direction == "ccw" else "-.",
                    label=rf"$\Delta N_R$, {direction.upper()}",
                )
            ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.75)
            ax.set_title(f"{wall.capitalize()} wall, $M={cycles}$")
            ax.set_xticks([0.0, np.pi, 2.0 * np.pi], [r"$0$", r"$\pi$", r"$2\pi$"])
            if wi == 1:
                ax.set_xlabel(r"flux-path coordinate $|\phi|$")
            if column == 0:
                ax.set_ylabel("mean regional charge change")
            if wi == 0 and column == 0:
                ax.legend(frameon=False, ncol=2)
    _panel_letters(axes)
    fig.tight_layout(pad=0.7, w_pad=0.7, h_pad=0.9)
    regional_figures = _save(fig, figures_root, "frozen_continuous_ramp_regional_charge")

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.7), sharey=True)
    endpoints: dict[str, Any] = {}
    for wi, wall in enumerate(campaign.WALLS):
        ax = axes[wi]
        endpoints[wall] = {}
        for di, direction in enumerate(campaign.DIRECTIONS):
            samples = np.asarray([data[cycles]["q_x"][wi, di, :, -1] for cycles in cycles_list])
            means = samples.mean(axis=1)
            stds = samples.std(axis=1, ddof=1)
            ax.errorbar(
                cycles_list, means, yerr=stds, color=colors[direction],
                marker=markers[direction], linestyle=linestyles[direction],
                markerfacecolor="white", capsize=2, linewidth=1.0,
                label=direction.upper(),
            )
            endpoints[wall][direction] = {
                str(cycles): {"mean": float(means[index]), "sample_std": float(stds[index])}
                for index, cycles in enumerate(cycles_list)
            }
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.75)
        ax.set_title(f"{wall.capitalize()} wall")
        ax.set_xlabel(r"ramp cycles $M$")
        ax.set_xticks(cycles_list)
        ax.legend(frameon=False)
    axes[0].set_ylabel(r"endpoint $q_x(\sigma2\pi)$")
    _panel_letters(axes)
    fig.tight_layout(pad=0.7, w_pad=0.9)
    endpoint_figures = _save(fig, figures_root, "frozen_continuous_ramp_endpoint_convergence")

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.85), sharey=True)
    offsets = {"ccw": -0.13, "cw": 0.13}
    for wi, wall in enumerate(campaign.WALLS):
        ax = axes[wi]
        for di, direction in enumerate(campaign.DIRECTIONS):
            for mi, cycles in enumerate(cycles_list):
                sample_values = data[cycles]["q_x"][wi, di, :, -1]
                positions = np.full(sample_values.shape, mi + offsets[direction], dtype=float)
                ax.scatter(
                    positions, sample_values, s=16, facecolors="white",
                    edgecolors=colors[direction], marker=markers[direction],
                    linewidths=0.8, alpha=0.9,
                    label=direction.upper() if mi == 0 else None,
                )
                ax.hlines(
                    float(np.mean(sample_values)), mi + offsets[direction] - 0.09,
                    mi + offsets[direction] + 0.09, color=colors[direction], linewidth=1.5,
                )
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.75)
        ax.set_title(f"{wall.capitalize()} wall")
        ax.set_xlabel(r"ramp cycles $M$")
        ax.set_xticks(range(len(cycles_list)), [str(value) for value in cycles_list])
        ax.legend(frameon=False)
    axes[0].set_ylabel(r"trajectory endpoint $q_x(\sigma2\pi)$")
    _panel_letters(axes)
    fig.tight_layout(pad=0.7, w_pad=0.9)
    endpoint_distribution_figures = _save(
        fig, figures_root, "frozen_continuous_ramp_endpoint_distributions"
    )

    max_continuity = max(
        float(np.max(np.abs(values["charge_continuity_residual"])))
        for values in data.values()
    )
    minimum_probability = min(
        float(np.min(values["minimum_selected_probability_by_cycle"][:, :, :, 1:]))
        for values in data.values()
    )
    direction_odd = {
        wall: {
            str(cycles): {
                "mean": float(
                    np.mean(0.5 * (data[cycles]["q_x"][wi, 0, :, -1] - data[cycles]["q_x"][wi, 1, :, -1]))
                ),
                "sample_std": float(
                    np.std(
                        0.5 * (data[cycles]["q_x"][wi, 0, :, -1] - data[cycles]["q_x"][wi, 1, :, -1]),
                        ddof=1,
                    )
                ),
            }
            for cycles in cycles_list
        }
        for wi, wall in enumerate(campaign.WALLS)
    }
    summary = {
        **metadata,
        "endpoint_q_x": endpoints,
        "endpoint_direction_odd_q_x": direction_odd,
        "maximum_absolute_charge_continuity_residual": max_continuity,
        "minimum_selected_replay_probability": minimum_probability,
        "figures": {
            "q_x": qx_figures,
            "regional_charge": regional_figures,
            "endpoint_convergence": endpoint_figures,
            "endpoint_distributions": endpoint_distribution_figures,
        },
        "trajectory_csv": str(csv_path),
        "claim_boundary": "Continuously evolved conditional frozen-record ramps; quantization is not an acceptance gate.",
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
