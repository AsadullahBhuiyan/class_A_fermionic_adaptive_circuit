#!/usr/bin/env python3
"""Plot hard-wall purification and its x-resolved entropy contour."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import analyze_completed_campaign as campaign


BUNDLE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = BUNDLE_ROOT / "analysis_outputs" / "hard_wall_purification_spatial_v1"
HEATMAP_NY = 40
EXPONENTIAL_FIT_MIN = 2.0
EXPONENTIAL_FIT_MAX = 4.0
HEATMAP_LOG_VMIN = 1.0e-8
SELECTED_X = (5, 6, 7, 10, 13, 14, 15)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_hard_wall_data(
    *, verify_hashes: bool
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    spec = next(item for item in campaign.CAMPAIGNS if item.construction == "hard")
    inventory = campaign._inventory_campaigns(campaign.DEFAULT_REMOTE_INVENTORY)
    record = inventory.get(spec.key)
    if record is None or int(record["result_completion_pairs"]) != 60:
        raise RuntimeError("hard-wall remote inventory is incomplete")

    grouped: dict[int, dict[str, list[np.ndarray]]] = {
        ny: {"sample_indices": [], "total_entropy": [], "entropy_x": []}
        for ny in campaign.NY_VALUES
    }
    expected_paths: set[Path] = set()
    maximum_closure_error = 0.0

    for remote in record["records"]:
        result_path = spec.data_root / remote["relative_result_path"]
        completion_path = spec.data_root / remote["relative_completion_path"]
        expected_paths.update((result_path, completion_path))
        for path, bytes_key, hash_key in (
            (result_path, "result_bytes", "result_sha256"),
            (completion_path, "completion_bytes", "completion_sha256"),
        ):
            if not path.is_file() or path.stat().st_size != int(remote[bytes_key]):
                raise RuntimeError(f"missing or wrong-sized campaign file: {path}")
            if verify_hashes and campaign.sha256_file(path) != remote[hash_key]:
                raise RuntimeError(f"checksum mismatch: {path}")

        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        campaign._validate_completion(
            completion,
            spec,
            result_path,
            expected_result_bytes=int(remote["result_bytes"]),
            expected_result_sha256=remote["result_sha256"],
        )
        ny = int(completion["Ny"])
        with np.load(result_path, allow_pickle=False) as data:
            campaign._validate_product(
                data, spec=spec, completion=completion, result_path=result_path
            )
            sample_indices = np.asarray(data["sample_indices"], dtype=np.int64)
            total_entropy = np.asarray(data["total_entropy"], dtype=np.float64)
            entropy_x = np.asarray(data["entropy_contour"], dtype=np.float64).sum(axis=-1)
            closure = float(np.max(np.abs(entropy_x.sum(axis=-1) - total_entropy)))
            maximum_closure_error = max(maximum_closure_error, closure)
            if closure > 5.0e-10:
                raise RuntimeError(f"{result_path.name}: x-contour closure failure")
            grouped[ny]["sample_indices"].append(sample_indices)
            grouped[ny]["total_entropy"].append(total_entropy)
            grouped[ny]["entropy_x"].append(entropy_x)

    discovered = set(spec.data_root.rglob("*.npz")) | set(
        spec.data_root.rglob("*.complete.json")
    )
    if discovered != expected_paths:
        raise RuntimeError("hard-wall result tree differs from the verified inventory")

    arrays: dict[int, dict[str, np.ndarray]] = {}
    for ny in campaign.NY_VALUES:
        combined = {
            key: np.concatenate(parts, axis=0) for key, parts in grouped[ny].items()
        }
        order = np.argsort(combined["sample_indices"])
        combined = {key: value[order] for key, value in combined.items()}
        if not np.array_equal(combined["sample_indices"], np.arange(100)):
            raise RuntimeError(f"Ny={ny}: expected samples 0,...,99")
        if combined["total_entropy"].shape != (100, 4 * ny + 1):
            raise RuntimeError(f"Ny={ny}: total-entropy shape mismatch")
        if combined["entropy_x"].shape != (100, 4 * ny + 1, 20):
            raise RuntimeError(f"Ny={ny}: x-contour shape mismatch")
        arrays[ny] = combined

    provenance = {
        "campaign": spec.revision,
        "construction": "hard",
        "configuration_hash": spec.configuration_hash,
        "source_hashes": spec.source_hashes,
        "verified_result_completion_pairs": 60,
        "verified_trajectories": 300,
        "verify_hashes": verify_hashes,
        "remote_inventory": str(campaign.DEFAULT_REMOTE_INVENTORY),
        "remote_inventory_sha256": campaign.sha256_file(campaign.DEFAULT_REMOTE_INVENTORY),
        "maximum_x_contour_closure_error": maximum_closure_error,
    }
    return arrays, provenance


def summarize(
    arrays: dict[int, dict[str, np.ndarray]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    curve_rows: list[dict[str, Any]] = []
    for ny in campaign.NY_VALUES:
        entropy_density = arrays[ny]["total_entropy"] / ny
        for cycle in range(4 * ny + 1):
            values = entropy_density[:, cycle]
            curve_rows.append(
                {
                    "Ny": ny,
                    "samples": 100,
                    "cycle": cycle,
                    "normalized_cycle": cycle / ny,
                    "mean_entropy_over_Ny": float(values.mean()),
                    "sem_entropy_over_Ny": float(values.std(ddof=1) / math.sqrt(100)),
                }
            )

    heat_samples = arrays[HEATMAP_NY]["entropy_x"] / HEATMAP_NY
    expected_shape = (100, 4 * HEATMAP_NY + 1, 20)
    if heat_samples.shape != expected_shape:
        raise RuntimeError("Ny=40 hard-wall contour shape mismatch")
    heat_cycles = np.arange(4 * HEATMAP_NY + 1, dtype=np.float64) / HEATMAP_NY
    heat_mean = heat_samples.mean(axis=0)
    heat_sem = heat_samples.std(axis=0, ddof=1) / math.sqrt(heat_samples.shape[0])
    heat_rows = [
        {
            "Ny": HEATMAP_NY,
            "cycle": time_index,
            "normalized_cycle": float(heat_cycles[time_index]),
            "x": x,
            "independent_trajectories": 100,
            "mean_entropy_x_over_Ny": float(heat_mean[time_index, x]),
            "sem_entropy_x_over_Ny": float(heat_sem[time_index, x]),
        }
        for time_index in range(heat_cycles.size)
        for x in range(20)
    ]
    return curve_rows, heat_rows, heat_mean


def fit_pooled_exponential(
    arrays: dict[int, dict[str, np.ndarray]],
) -> dict[str, float | int | str]:
    """Fit pooled late-time ensemble means to S/Ny = A exp[-k t/Ny]."""
    normalized_cycles: list[np.ndarray] = []
    entropy_densities: list[np.ndarray] = []
    for ny in campaign.NY_VALUES:
        all_x = np.arange(1, 4 * ny + 1, dtype=np.float64) / ny
        all_y = (arrays[ny]["total_entropy"] / ny).mean(axis=0)[1:]
        selected = (all_x >= EXPONENTIAL_FIT_MIN) & (all_x <= EXPONENTIAL_FIT_MAX)
        x = all_x[selected]
        y = all_y[selected]
        if np.any(~np.isfinite(y)) or np.any(y <= 0.0):
            raise RuntimeError(f"Ny={ny}: exponential fit requires finite positive means")
        normalized_cycles.append(x)
        entropy_densities.append(y)

    x = np.concatenate(normalized_cycles)
    y = np.concatenate(entropy_densities)
    slope, intercept = np.polyfit(x, np.log(y), deg=1)
    fitted_log_y = intercept + slope * x
    residual_sum = float(np.sum((np.log(y) - fitted_log_y) ** 2))
    total_sum = float(np.sum((np.log(y) - np.log(y).mean()) ** 2))
    return {
        "model": "S/Ny = amplitude * exp[-rate * cycle/Ny]",
        "fit_basis": "pooled ensemble means for Ny=20,30,40; 2 <= cycle/Ny <= 4",
        "amplitude": float(np.exp(intercept)),
        "rate": float(-slope),
        "characteristic_normalized_time": float(-1.0 / slope),
        "r_squared_log_space": float(1.0 - residual_sum / total_sum),
        "normalized_cycle_min": float(x.min()),
        "normalized_cycle_max": float(x.max()),
        "pooled_points": int(x.size),
    }


def configure_plotting() -> None:
    available = {font.name for font in mpl.font_manager.fontManager.ttflist}
    sans = "CMU Sans Serif" if "CMU Sans Serif" in available else "DejaVu Sans"
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [sans],
            "mathtext.fontset": "cm",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(axis: Any, label: str) -> None:
    axis.text(
        -0.14,
        1.03,
        label,
        transform=axis.transAxes,
        fontweight="bold",
        va="bottom",
    )


def make_figure(
    arrays: dict[int, dict[str, np.ndarray]],
    heat_mean: np.ndarray,
    exponential_fit: dict[str, float | int | str],
) -> None:
    configure_plotting()
    colors = {20: "#d62728", 30: "#2ca02c", 40: "#1f77b4"}
    markers = {20: "^", 30: "s", 40: "o"}
    linestyles = {20: ":", 30: "--", 40: "-"}
    figure, (curve_axis, heat_axis) = plt.subplots(
        1, 2, figsize=(7.05, 2.75), constrained_layout=True
    )

    for ny in campaign.NY_VALUES:
        normalized_cycle = np.arange(4 * ny + 1, dtype=np.float64) / ny
        values = arrays[ny]["total_entropy"] / ny
        mean = values.mean(axis=0)
        error = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
        lower = np.maximum(mean - error, np.finfo(np.float64).tiny)
        upper = mean + error
        curve_axis.fill_between(
            normalized_cycle, lower, upper, color=colors[ny], alpha=0.13, linewidth=0
        )
        curve_axis.plot(
            normalized_cycle,
            mean,
            color=colors[ny],
            marker=markers[ny],
            markevery=max(1, ny // 4),
            markersize=3.1,
            markerfacecolor="white",
            markeredgewidth=0.8,
            linestyle=linestyles[ny],
            linewidth=1.15,
            label=rf"$N_y={ny}$",
        )
    fit_x = np.linspace(
        float(exponential_fit["normalized_cycle_min"]),
        float(exponential_fit["normalized_cycle_max"]),
        300,
    )
    fit_y = float(exponential_fit["amplitude"]) * np.exp(
        -float(exponential_fit["rate"]) * fit_x
    )
    curve_axis.plot(
        fit_x,
        fit_y,
        color="black",
        linestyle=(0, (4, 2)),
        linewidth=1.15,
        label=rf"late exp.: $\kappa={float(exponential_fit['rate']):.2f}$",
        zorder=5,
    )
    curve_axis.set_yscale("log")
    curve_axis.set_xlim(0.0, 4.0)
    curve_axis.set_xlabel(r"cycle$/N_y$")
    curve_axis.set_ylabel(r"$S/N_y$")
    curve_axis.legend(frameon=False, loc="lower left", handlelength=2.5)
    panel_label(curve_axis, "(a)")

    image = heat_axis.pcolormesh(
        np.arange(20),
        np.arange(4 * HEATMAP_NY + 1, dtype=np.float64) / HEATMAP_NY,
        heat_mean,
        shading="nearest",
        cmap="magma",
        norm=LogNorm(
            vmin=HEATMAP_LOG_VMIN,
            vmax=float(np.max(heat_mean)),
            clip=True,
        ),
        rasterized=True,
    )
    for wall_x in (5, 15):
        heat_axis.axvline(wall_x, color="white", linestyle="--", linewidth=0.8, alpha=0.9)
    heat_axis.set_xlim(-0.5, 19.5)
    heat_axis.set_ylim(0.0, 4.0)
    heat_axis.set_xticks((0, 5, 10, 15, 19))
    heat_axis.set_xlabel(r"$x$")
    heat_axis.set_ylabel(r"cycle$/N_y$")
    panel_label(heat_axis, "(b)")
    colorbar = figure.colorbar(image, ax=heat_axis, pad=0.025)
    colorbar.set_label(r"$\langle s_x/N_y\rangle$")
    colorbar.ax.tick_params(direction="in", labelsize=7.0)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_ROOT / "hard_wall_purification_entropy_and_x_contour.pdf")
    figure.savefig(
        OUTPUT_ROOT / "hard_wall_purification_entropy_and_x_contour.png", dpi=300
    )
    plt.close(figure)


def make_selected_x_figure(arrays: dict[int, dict[str, np.ndarray]]) -> None:
    """Plot selected Ny=40 x-resolved entropy-density traces."""
    configure_plotting()
    normalized_cycle = np.arange(4 * HEATMAP_NY + 1, dtype=np.float64) / HEATMAP_NY
    values = arrays[HEATMAP_NY]["entropy_x"] / HEATMAP_NY
    mean = values.mean(axis=0)
    styles = {
        5: ("#d62728", "-"),
        6: ("#ff7f0e", "-"),
        7: ("#2ca02c", "-"),
        10: ("#1f77b4", "-"),
        13: ("#2ca02c", "--"),
        14: ("#ff7f0e", "--"),
        15: ("#d62728", "--"),
    }
    figure, axis = plt.subplots(figsize=(3.45, 2.8), constrained_layout=True)
    for x in SELECTED_X:
        color, linestyle = styles[x]
        axis.plot(
            normalized_cycle,
            mean[:, x],
            color=color,
            linestyle=linestyle,
            linewidth=1.25,
            label=rf"$x={x}$",
        )
    axis.set_yscale("log")
    axis.set_xlim(0.0, 4.0)
    axis.set_xlabel(r"cycle$/N_y$")
    axis.set_ylabel(r"$\langle s_x\rangle/N_y$")
    axis.legend(
        frameon=False,
        ncol=2,
        loc="upper right",
        columnspacing=0.8,
        handlelength=2.2,
    )
    axis.text(
        0.03,
        0.05,
        rf"hard wall, $N_y={HEATMAP_NY}$, $S=100$",
        transform=axis.transAxes,
        fontsize=7.2,
    )
    figure.savefig(OUTPUT_ROOT / "hard_wall_Ny40_selected_x_entropy_traces.pdf")
    figure.savefig(
        OUTPUT_ROOT / "hard_wall_Ny40_selected_x_entropy_traces.png", dpi=300
    )
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Skip repeated SHA-256 reads while retaining schema and identity checks.",
    )
    args = parser.parse_args()
    arrays, provenance = load_hard_wall_data(verify_hashes=not args.skip_hashes)
    curve_rows, heat_rows, heat_mean = summarize(arrays)
    exponential_fit = fit_pooled_exponential(arrays)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_ROOT / "purification_entropy_curves.csv", curve_rows)
    write_csv(OUTPUT_ROOT / "x_resolved_entropy_heatmap.csv", heat_rows)
    summary = {
        "schema": "hard_wall_purification_spatial_analysis_v1",
        "construction": "hard",
        "Nx": 20,
        "Ny": list(campaign.NY_VALUES),
        "samples_per_size": 100,
        "cycles": "0,...,4Ny",
        "entropy_log_base": "natural",
        "curve_uncertainty": "sample SEM; std(ddof=1)/sqrt(100)",
        "curve_axes": "linear normalized-cycle axis; logarithmic entropy-density axis",
        "pooled_late_time_exponential_fit": exponential_fit,
        "x_contour_definition": "s_x(t)=sum_y s(x,y,t)",
        "heatmap_Ny": HEATMAP_NY,
        "heatmap_definition": "mean of s_x/Ny over the 100 Ny=40 hard-wall trajectories",
        "heatmap_color_normalization": (
            "LogNorm with vmin=1e-8 and vmax equal to the data maximum"
        ),
        "selected_x_trace_Ny": HEATMAP_NY,
        "selected_x_trace_positions": list(SELECTED_X),
        "selected_x_trace_axes": (
            "linear normalized-cycle axis; logarithmic mean x-resolved entropy-density axis"
        ),
        "provenance": provenance,
    }
    (OUTPUT_ROOT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_figure(arrays, heat_mean, exponential_fit)
    make_selected_x_figure(arrays)
    print(json.dumps({"output_root": str(OUTPUT_ROOT), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
