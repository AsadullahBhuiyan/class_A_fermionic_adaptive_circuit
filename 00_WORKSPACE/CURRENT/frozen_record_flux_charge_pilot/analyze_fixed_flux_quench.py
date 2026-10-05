#!/usr/bin/env python3
"""Analyze the S10 fixed-flux frozen-record quench campaign."""

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

import run_fixed_flux_quench as campaign  # noqa: E402


ANALYSIS_SCHEMA = "fixed_flux_quench_analysis_v1"


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
    ax: plt.Axes,
    x: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    marker: str | None,
    linestyle: str,
    label: str,
) -> None:
    mean, std = _mean_std(values)
    ax.plot(
        x, mean, color=color, marker=marker, linestyle=linestyle,
        markerfacecolor="white", markersize=2.6, markevery=8,
        linewidth=1.0, label=label,
    )
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)


def _panel_letters(axes: Any) -> None:
    for index, axis in enumerate(np.asarray(axes).flat):
        axis.text(
            -0.13, 1.04, f"({chr(ord('a') + index)})",
            transform=axis.transAxes, va="bottom", ha="left", fontsize=9,
        )


def _load(
    config: dict[str, Any], source: dict[str, Any], output_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    hashes = campaign.source_hashes()
    config_hash = campaign.scientific_config_hash(config)
    sample_count = int(config["ensemble"]["samples_per_wall"])
    twist_count = len(config["fixed_twists"])
    cycles = int(config["dynamics"]["replay_cycles"])
    length = cycles + 1
    fields = (
        "phi", "N_left", "N_right", "N_total", "raw_delta_N_left",
        "raw_delta_N_right", "raw_delta_N_total", "raw_q_x",
        "reference_N_left", "reference_N_right", "reference_N_total",
        "response_delta_N_left", "response_delta_N_right",
        "response_delta_N_total", "response_q_x", "net_injected_charge",
        "injection_count", "charge_continuity_residual", "rank",
        "minimum_selected_probability_by_cycle", "branch_log_probability_by_cycle",
    )
    values = {
        field: np.empty((2, twist_count, 2, sample_count, length), dtype=np.float64)
        for field in fields
    }
    for field in ("density_x", "reference_density_x", "response_density_x"):
        values[field] = np.empty(
            (2, twist_count, 2, sample_count, length, 16), dtype=np.float64
        )
    values["record_sha256"] = np.empty((2, twist_count, 2, sample_count), dtype="U64")
    for task in campaign.tasks(config):
        dependencies = campaign.dependencies_for(task, source)
        ok, reason, _ = campaign.verify_pair(
            output_root, task, config_hash, hashes, dependencies
        )
        if not ok:
            raise RuntimeError(f"unverified fixed-flux task {task['task_id']}: {reason}")
        wi = campaign.WALLS.index(task["wall"])
        ti = int(task["twist_index"])
        di = campaign.DIRECTIONS.index(task["direction"])
        si = int(task["sample_id"])
        result, _ = campaign.result_paths(output_root, task)
        with np.load(result, allow_pickle=False) as saved:
            for field in fields:
                values[field][wi, ti, di, si] = np.asarray(saved[field])
            for field in ("density_x", "reference_density_x", "response_density_x"):
                values[field][wi, ti, di, si] = np.asarray(saved[field])
            values["record_sha256"][wi, ti, di, si] = str(saved["record_sha256"])
    metadata = {
        "schema": ANALYSIS_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_hash": config_hash,
        "source_hashes": hashes,
        "samples_per_wall": sample_count,
        "cycles": cycles,
        "walls": list(campaign.WALLS),
        "directions": list(campaign.DIRECTIONS),
        "fixed_twists": config["fixed_twists"],
        "uncertainty": "one sample standard deviation across paired frozen-record parents",
    }
    return values, metadata


def _validate_paired(values: dict[str, np.ndarray]) -> None:
    for field in ("rank", "net_injected_charge", "injection_count"):
        if not np.array_equal(values[field][:, :, 0], values[field][:, :, 1]):
            raise RuntimeError(f"CW/CCW paired {field} histories differ")
    if not np.array_equal(values["record_sha256"][:, :, 0], values["record_sha256"][:, :, 1]):
        raise RuntimeError("CW/CCW frozen-record hashes differ")
    if not np.array_equal(values["phi"][:, :, 0], -values["phi"][:, :, 1]):
        raise RuntimeError("CW/CCW fixed twists do not reverse")


def _write_csv(path: Path, config: dict[str, Any], values: dict[str, np.ndarray]) -> None:
    fields = [
        "wall", "twist_name", "direction", "sample_id", "cycle", "phi",
        "response_delta_N_left", "response_delta_N_right", "response_delta_N_total",
        "response_q_x", "raw_q_x", "net_injected_charge", "injection_count",
        "rank", "minimum_selected_probability", "charge_continuity_residual",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for wi, wall in enumerate(campaign.WALLS):
            for ti, twist in enumerate(config["fixed_twists"]):
                for di, direction in enumerate(campaign.DIRECTIONS):
                    for sample_id in range(values["phi"].shape[3]):
                        for cycle in range(values["phi"].shape[-1]):
                            writer.writerow(
                                {
                                    "wall": wall,
                                    "twist_name": twist["name"],
                                    "direction": direction,
                                    "sample_id": sample_id,
                                    "cycle": cycle,
                                    "phi": values["phi"][wi, ti, di, sample_id, cycle],
                                    "response_delta_N_left": values["response_delta_N_left"][wi, ti, di, sample_id, cycle],
                                    "response_delta_N_right": values["response_delta_N_right"][wi, ti, di, sample_id, cycle],
                                    "response_delta_N_total": values["response_delta_N_total"][wi, ti, di, sample_id, cycle],
                                    "response_q_x": values["response_q_x"][wi, ti, di, sample_id, cycle],
                                    "raw_q_x": values["raw_q_x"][wi, ti, di, sample_id, cycle],
                                    "net_injected_charge": values["net_injected_charge"][wi, ti, di, sample_id, cycle],
                                    "injection_count": values["injection_count"][wi, ti, di, sample_id, cycle],
                                    "rank": values["rank"][wi, ti, di, sample_id, cycle],
                                    "minimum_selected_probability": values["minimum_selected_probability_by_cycle"][wi, ti, di, sample_id, cycle],
                                    "charge_continuity_residual": values["charge_continuity_residual"][wi, ti, di, sample_id, cycle],
                                }
                            )


def analyze(config: dict[str, Any], source: dict[str, Any], output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    values, metadata = _load(config, source, output_root)
    _validate_paired(values)
    analysis_root = output_root / "analysis"
    figures_root = analysis_root / "figures"
    analysis_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        analysis_root / "fixed_flux_quench_aggregate.npz",
        metadata_json=np.asarray(campaign.canonical_json(metadata)),
        **values,
    )
    csv_path = analysis_root / "fixed_flux_quench_trajectories.csv"
    _write_csv(csv_path, config, values)

    _style()
    colors = {"ccw": "#1f77b4", "cw": "#d62728"}
    markers = {"ccw": "o", "cw": "^"}
    linestyles = {"ccw": "-", "cw": ":"}
    twist_titles = [r"$|\phi|=\pi/2$", r"$|\phi|=\pi$", r"$|\phi|=3\pi/2$", r"$|\phi|=2\pi-10^{-7}$"]
    cycles = np.arange(int(config["dynamics"]["replay_cycles"]) + 1)

    fig, axes = plt.subplots(2, 4, figsize=(7.05, 4.2), sharex=True, sharey=True)
    for wi, wall in enumerate(campaign.WALLS):
        for ti, title in enumerate(twist_titles):
            ax = axes[wi, ti]
            for di, direction in enumerate(campaign.DIRECTIONS):
                _line_band(
                    ax, cycles, values["response_q_x"][wi, ti, di],
                    color=colors[direction], marker=markers[direction],
                    linestyle=linestyles[direction], label=direction.upper(),
                )
            ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.75)
            ax.set_title(f"{wall.capitalize()}, {title}")
            if wi == 1:
                ax.set_xlabel("replay cycle")
            if ti == 0:
                ax.set_ylabel(r"paired $q_x$ response")
            if wi == 0 and ti == 0:
                ax.legend(frameon=False)
    _panel_letters(axes)
    fig.tight_layout(pad=0.65, w_pad=0.6, h_pad=0.85)
    response_figures = _save(fig, figures_root, "fixed_flux_quench_qx_response")

    direction_odd = 0.5 * (
        values["response_q_x"][:, :, 0] - values["response_q_x"][:, :, 1]
    )
    direction_even = 0.5 * (
        values["response_q_x"][:, :, 0] + values["response_q_x"][:, :, 1]
    )
    fig, axes = plt.subplots(2, 4, figsize=(7.05, 4.2), sharex=True, sharey=True)
    for wi, wall in enumerate(campaign.WALLS):
        for ti, title in enumerate(twist_titles):
            ax = axes[wi, ti]
            _line_band(
                ax, cycles, direction_odd[wi, ti], color="#2ca02c", marker="s",
                linestyle="--", label="direction odd",
            )
            ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.75)
            ax.set_title(f"{wall.capitalize()}, {title}")
            if wi == 1:
                ax.set_xlabel("replay cycle")
            if ti == 0:
                ax.set_ylabel(r"$[q_x(+|\phi|)-q_x(-|\phi|)]/2$")
            if wi == 0 and ti == 0:
                ax.legend(frameon=False)
    _panel_letters(axes)
    fig.tight_layout(pad=0.65, w_pad=0.6, h_pad=0.85)
    odd_figures = _save(fig, figures_root, "fixed_flux_quench_direction_odd")

    fig, axes = plt.subplots(2, 4, figsize=(7.05, 4.2), sharex=True, sharey=True)
    for wi, wall in enumerate(campaign.WALLS):
        for ti, title in enumerate(twist_titles):
            ax = axes[wi, ti]
            for di, direction in enumerate(campaign.DIRECTIONS):
                _line_band(
                    ax, cycles, values["response_delta_N_left"][wi, ti, di],
                    color=colors[direction], marker=markers[direction],
                    linestyle=linestyles[direction], label=rf"$\Delta N_L$, {direction.upper()}",
                )
                _line_band(
                    ax, cycles, values["response_delta_N_right"][wi, ti, di],
                    color=colors[direction], marker=None,
                    linestyle="--" if direction == "ccw" else "-.",
                    label=rf"$\Delta N_R$, {direction.upper()}",
                )
            ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.75)
            ax.set_title(f"{wall.capitalize()}, {title}")
            if wi == 1:
                ax.set_xlabel("replay cycle")
            if ti == 0:
                ax.set_ylabel("paired regional response")
            if wi == 0 and ti == 0:
                ax.legend(frameon=False, ncol=2)
    _panel_letters(axes)
    fig.tight_layout(pad=0.65, w_pad=0.6, h_pad=0.85)
    regional_figures = _save(fig, figures_root, "fixed_flux_quench_regional_response")

    magnitudes = np.asarray([row["radians"] for row in config["fixed_twists"]]) / np.pi
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75), sharey=True)
    endpoint_summary: dict[str, Any] = {}
    odd_summary: dict[str, Any] = {}
    for wi, wall in enumerate(campaign.WALLS):
        ax = axes[wi]
        endpoint_summary[wall] = {}
        odd_summary[wall] = {}
        for di, direction in enumerate(campaign.DIRECTIONS):
            endpoints = values["response_q_x"][wi, :, di, :, -1]
            means = endpoints.mean(axis=1)
            stds = endpoints.std(axis=1, ddof=1)
            ax.errorbar(
                magnitudes, means, yerr=stds, color=colors[direction],
                marker=markers[direction], linestyle=linestyles[direction],
                markerfacecolor="white", capsize=2, linewidth=1.0,
                label=direction.upper(),
            )
            endpoint_summary[wall][direction] = {
                row["name"]: {
                    "mean": float(means[index]),
                    "sample_std": float(stds[index]),
                }
                for index, row in enumerate(config["fixed_twists"])
            }
        odd_endpoints = direction_odd[wi, :, :, -1]
        odd_summary[wall] = {
            row["name"]: {
                "mean": float(odd_endpoints[index].mean()),
                "sample_std": float(odd_endpoints[index].std(ddof=1)),
                "maximum_absolute_mean_over_time": float(
                    np.max(np.abs(direction_odd[wi, index].mean(axis=0)))
                ),
            }
            for index, row in enumerate(config["fixed_twists"])
        }
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.75)
        ax.set_title(f"{wall.capitalize()} wall")
        ax.set_xlabel(r"fixed $|\phi|/\pi$")
        ax.set_xticks([0.5, 1.0, 1.5, 2.0], [r"$1/2$", r"$1$", r"$3/2$", r"$2-10^{-7}/\pi$"])
        ax.legend(frameon=False)
    axes[0].set_ylabel(r"endpoint paired $q_x$ response")
    _panel_letters(axes)
    fig.tight_layout(pad=0.7, w_pad=0.9)
    endpoint_figures = _save(fig, figures_root, "fixed_flux_quench_endpoints")

    max_continuity = float(np.max(np.abs(values["charge_continuity_residual"])))
    max_total_response = float(np.max(np.abs(values["response_delta_N_total"])))
    minimum_probability = float(
        np.min(values["minimum_selected_probability_by_cycle"][..., 1:])
    )
    summary = {
        **metadata,
        "endpoint_response_q_x": endpoint_summary,
        "direction_odd_response_q_x": odd_summary,
        "maximum_absolute_charge_continuity_residual": max_continuity,
        "maximum_absolute_paired_total_charge_response": max_total_response,
        "minimum_selected_replay_probability": minimum_probability,
        "figures": {
            "q_x_response": response_figures,
            "direction_odd": odd_figures,
            "regional_response": regional_figures,
            "endpoints": endpoint_figures,
        },
        "trajectory_csv": str(csv_path),
        "claim_boundary": (
            "Conditioned fixed-flux response to a sudden projector quench; "
            "not an adiabatic or quantized pump."
        ),
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
    source = campaign.source_context(config)
    analyze(config, source, args.output_root.resolve())


if __name__ == "__main__":
    main()
