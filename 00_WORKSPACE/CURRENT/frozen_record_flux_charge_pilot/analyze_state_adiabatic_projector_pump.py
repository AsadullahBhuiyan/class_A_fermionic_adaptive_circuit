#!/usr/bin/env python3
"""Analyze the S10 state-derived adiabatic projector pump."""

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

import run_state_adiabatic_projector_pump as campaign  # noqa: E402


ANALYSIS_SCHEMA = "state_adiabatic_projector_pump_analysis_v1"
WALLS = ("soft", "hard")
DIRECTIONS = ("ccw", "cw")


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


def _save(fig: plt.Figure, root: Path, stem: str) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    pdf, png = root / f"{stem}.pdf", root / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}


def _load(
    config: dict[str, Any], output_root: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = campaign.source_context(config)
    state = campaign.inventory(config, output_root, source)
    config_hash, hashes = state["config_hash"], state["hashes"]
    sample_count = int(config["ensemble"]["samples_per_wall"])
    point_count = int(config["continuation"]["grid_intervals"]) + 1
    curve_fields = (
        "phi",
        "path_fraction",
        "continued_delta_N_left",
        "continued_delta_N_right",
        "continued_delta_N_total",
        "continued_q_x",
        "instantaneous_delta_N_left",
        "instantaneous_delta_N_right",
        "instantaneous_delta_N_total",
        "instantaneous_q_x",
        "instantaneous_rank_gap",
        "principal_overlap",
        "selected_weight_floor",
    )
    density_fields = ("continued_density_x", "instantaneous_density_x", "source_density_x")
    arrays = {
        field: np.empty((2, 2, sample_count, point_count), dtype=np.float64)
        for field in curve_fields
    }
    arrays.update(
        {
            field: np.empty((2, 2, sample_count, point_count, 16), dtype=np.float64)
            for field in density_fields[:2]
        }
    )
    arrays["source_density_x"] = np.empty((2, 2, sample_count, 16), dtype=np.float64)
    scalars = (
        "rank",
        "input_frame_gram_residual",
        "input_projector_residual",
        "flattened_parent_involution_residual",
        "large_gauge_parent_error",
    )
    arrays.update({field: np.empty((2, 2, sample_count), dtype=np.float64) for field in scalars})

    for task in campaign.tasks(config):
        burnin = source["burnins"][task["burnin_task_id"]]
        ok, reason, _ = campaign.verify_pair(
            output_root, task, config_hash, hashes, burnin, config
        )
        if not ok:
            raise RuntimeError(f"unverified state-pump task {task['task_id']}: {reason}")
        wi, di, si = WALLS.index(task["wall"]), DIRECTIONS.index(task["direction"]), int(task["sample_id"])
        result, _ = campaign.result_paths(output_root, task)
        with np.load(result, allow_pickle=False) as saved:
            for field in curve_fields + density_fields + scalars:
                arrays[field][wi, di, si] = np.asarray(saved[field])
    metadata = {
        "schema": ANALYSIS_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_hash": config_hash,
        "source_hashes": hashes,
        "walls": list(WALLS),
        "directions": list(DIRECTIONS),
        "samples_per_wall": sample_count,
        "independent_sampling_unit": config["ensemble"]["independent_sampling_unit"],
        "uncertainty": "trajectory sample standard deviation",
    }
    return arrays, metadata


def _mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(values, axis=0), np.std(values, axis=0, ddof=1)


def _write_curve_csv(path: Path, arrays: dict[str, np.ndarray]) -> None:
    fields = [
        "wall",
        "direction",
        "point",
        "phi",
        "path_fraction",
        "continued_q_x_mean",
        "continued_q_x_std",
        "instantaneous_q_x_mean",
        "instantaneous_q_x_std",
        "continued_delta_N_left_mean",
        "continued_delta_N_left_std",
        "continued_delta_N_right_mean",
        "continued_delta_N_right_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for wi, wall in enumerate(WALLS):
            for di, direction in enumerate(DIRECTIONS):
                phase = arrays["phi"][wi, di, 0]
                fraction = arrays["path_fraction"][wi, di, 0]
                estimates = {
                    field: _mean_std(arrays[field][wi, di])
                    for field in (
                        "continued_q_x",
                        "instantaneous_q_x",
                        "continued_delta_N_left",
                        "continued_delta_N_right",
                    )
                }
                for point in range(len(phase)):
                    writer.writerow(
                        {
                            "wall": wall,
                            "direction": direction,
                            "point": point,
                            "phi": float(phase[point]),
                            "path_fraction": float(fraction[point]),
                            "continued_q_x_mean": float(estimates["continued_q_x"][0][point]),
                            "continued_q_x_std": float(estimates["continued_q_x"][1][point]),
                            "instantaneous_q_x_mean": float(estimates["instantaneous_q_x"][0][point]),
                            "instantaneous_q_x_std": float(estimates["instantaneous_q_x"][1][point]),
                            "continued_delta_N_left_mean": float(estimates["continued_delta_N_left"][0][point]),
                            "continued_delta_N_left_std": float(estimates["continued_delta_N_left"][1][point]),
                            "continued_delta_N_right_mean": float(estimates["continued_delta_N_right"][0][point]),
                            "continued_delta_N_right_std": float(estimates["continued_delta_N_right"][1][point]),
                        }
                    )


def analyze(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    arrays, metadata = _load(config, output_root)
    root, figures = output_root / "analysis", output_root / "analysis" / "figures"
    root.mkdir(parents=True, exist_ok=True)
    aggregate = root / "state_adiabatic_projector_pump_aggregate.npz"
    np.savez_compressed(
        aggregate,
        metadata_json=np.asarray(campaign.canonical_json(metadata)),
        **arrays,
    )
    curve_csv = root / "state_adiabatic_projector_pump_curves.csv"
    _write_curve_csv(curve_csv, arrays)

    _configure_style()
    colors = {"ccw": "#1f77b4", "cw": "#d62728"}
    markers = {"ccw": "o", "cw": "s"}
    figure_paths: dict[str, dict[str, str]] = {}

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), sharex=True, sharey=True)
    for wi, (wall, ax) in enumerate(zip(WALLS, axes)):
        for di, direction in enumerate(DIRECTIONS):
            x = arrays["path_fraction"][wi, di, 0]
            mean, std = _mean_std(arrays["continued_q_x"][wi, di])
            ax.plot(x, mean, color=colors[direction], marker=markers[direction], markevery=8,
                    linewidth=1.2, markersize=3, label=f"{direction.upper()} continued")
            ax.fill_between(x, mean - std, mean + std, color=colors[direction], alpha=0.18, linewidth=0)
            control, _ = _mean_std(arrays["instantaneous_q_x"][wi, di])
            ax.plot(x, control, color=colors[direction], linestyle=":", linewidth=0.9,
                    label=f"{direction.upper()} instantaneous")
        ax.axhline(0.0, color="0.45", linestyle="--", linewidth=0.8)
        ax.set_title(f"{wall.capitalize()} wall, $S=10$")
        ax.set_xlabel(r"flux-path fraction $|\phi|/(2\pi)$")
        ax.text(-0.12, 1.04, f"({chr(ord('a') + wi)})", transform=ax.transAxes, fontsize=9)
    axes[0].set_ylabel(r"$q_x=(\Delta N_R-\Delta N_L)/2$")
    axes[1].legend(frameon=False, ncol=1, loc="best")
    fig.tight_layout()
    figure_paths["qx"] = _save(fig, figures, "state_adiabatic_projector_qx")

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), sharex=True, sharey=True)
    for wi, (wall, ax) in enumerate(zip(WALLS, axes)):
        for di, direction in enumerate(DIRECTIONS):
            x = arrays["path_fraction"][wi, di, 0]
            for field, label, linestyle in (
                ("continued_delta_N_left", r"$\Delta N_L$", "--"),
                ("continued_delta_N_right", r"$\Delta N_R$", "-"),
            ):
                mean, std = _mean_std(arrays[field][wi, di])
                color = colors[direction]
                ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=1.2,
                        label=f"{direction.upper()} {label}")
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        ax.axhline(0.0, color="0.45", linestyle=":", linewidth=0.8)
        ax.set_title(f"{wall.capitalize()} wall, $S=10$")
        ax.set_xlabel(r"flux-path fraction $|\phi|/(2\pi)$")
        ax.text(-0.12, 1.04, f"({chr(ord('a') + wi)})", transform=ax.transAxes, fontsize=9)
    axes[0].set_ylabel("continued subsystem charge change")
    axes[1].legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    figure_paths["regional_charge"] = _save(fig, figures, "state_adiabatic_projector_regional_charge")

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), sharey=True)
    endpoint_rows: list[dict[str, Any]] = []
    positions = np.arange(2)
    for wi, (wall, ax) in enumerate(zip(WALLS, axes)):
        continued_values = []
        instantaneous_values = []
        for di, direction in enumerate(DIRECTIONS):
            values = arrays["continued_q_x"][wi, di, :, -1]
            control = arrays["instantaneous_q_x"][wi, di, :, -1]
            continued_values.append(values)
            instantaneous_values.append(control)
            endpoint_rows.append(
                {
                    "wall": wall,
                    "direction": direction,
                    "continued_q_x_mean": float(values.mean()),
                    "continued_q_x_std": float(values.std(ddof=1)),
                    "instantaneous_q_x_mean": float(control.mean()),
                    "instantaneous_q_x_std": float(control.std(ddof=1)),
                    "samples": len(values),
                }
            )
        ax.boxplot(continued_values, positions=positions - 0.14, widths=0.22, patch_artist=True,
                   boxprops={"facecolor": "#8fbce6"}, medianprops={"color": "black"})
        ax.boxplot(instantaneous_values, positions=positions + 0.14, widths=0.22, patch_artist=True,
                   boxprops={"facecolor": "#d0d0d0"}, medianprops={"color": "black"})
        ax.axhline(0.0, color="0.45", linestyle="--", linewidth=0.8)
        ax.set_xticks(positions, ("CCW", "CW"))
        ax.set_title(f"{wall.capitalize()} wall, $S=10$")
        ax.text(-0.12, 1.04, f"({chr(ord('a') + wi)})", transform=ax.transAxes, fontsize=9)
    axes[0].set_ylabel(r"endpoint $q_x$")
    fig.tight_layout()
    figure_paths["endpoints"] = _save(fig, figures, "state_adiabatic_projector_endpoints")

    summary = {
        **metadata,
        "aggregate": str(aggregate),
        "curve_csv": str(curve_csv),
        "figures": figure_paths,
        "endpoint_statistics": endpoint_rows,
        "maximum_abs_charge_residual": float(np.max(np.abs(arrays["continued_delta_N_total"]))),
        "maximum_large_gauge_error": float(np.max(arrays["large_gauge_parent_error"])),
        "minimum_principal_overlap": float(np.min(arrays["principal_overlap"])),
    }
    summary_path = root / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=campaign.DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=campaign.DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = campaign.load_config(args.config.resolve())
    campaign.validate_config(config)
    analyze(config, args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
