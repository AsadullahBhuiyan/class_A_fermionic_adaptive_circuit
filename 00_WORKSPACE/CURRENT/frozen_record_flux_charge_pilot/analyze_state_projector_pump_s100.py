#!/usr/bin/env python3
"""Trajectory-level analysis for the N20x24 S100 state-projector pump."""

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
from scipy.stats import beta


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_state_projector_pump_s100 as campaign  # noqa: E402


ANALYSIS_SCHEMA = "state_projector_pump_s100_analysis_v1"
WALLS = ("soft", "hard")
DIRECTIONS = ("ccw", "cw")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "Computer Modern Sans Serif", "DejaVu Sans"],
            "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
            "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
            "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
            "xtick.top": True, "ytick.right": True, "savefig.dpi": 300,
        }
    )


def _save(fig: plt.Figure, root: Path, stem: str) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    pdf, png = root / f"{stem}.pdf", root / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}


def _bootstrap_mean(
    values: np.ndarray, *, draws: int, seed: int, confidence: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.shape[0], size=(draws, array.shape[0]))
    sampled = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return array.mean(axis=0), np.quantile(sampled, alpha, axis=0), np.quantile(sampled, 1 - alpha, axis=0)


def _bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    draws: int,
    seed: int,
    confidence: float,
) -> tuple[float, float, float]:
    """Independent-sampling bootstrap interval for a difference of means."""
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    lhs_indices = rng.integers(0, lhs.size, size=(draws, lhs.size))
    rhs_indices = rng.integers(0, rhs.size, size=(draws, rhs.size))
    sampled = lhs[lhs_indices].mean(axis=1) - rhs[rhs_indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(lhs.mean() - rhs.mean()),
        float(np.quantile(sampled, alpha)),
        float(np.quantile(sampled, 1.0 - alpha)),
    )


def _clopper_pearson(successes: int, total: int, confidence: float) -> tuple[float, float]:
    alpha = 1.0 - confidence
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    high = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return low, high


def load_verified(config: dict[str, Any], output_root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    status = campaign.inventory(config, output_root)
    if not all(row[0] for row in status["burnins"].values()):
        raise RuntimeError("analysis requires 200 verified burn-ins")
    if not all(row[0] for row in status["pumps"].values()):
        raise RuntimeError("analysis requires 400 verified pump paths")
    samples = int(config["ensemble"]["samples_per_wall"])
    points = int(config["projector_pump"]["grid_intervals"]) + 1
    curve_fields = (
        "phi", "path_fraction", "continued_delta_N_left", "continued_delta_N_right",
        "continued_delta_N_total", "continued_q_x", "instantaneous_delta_N_left",
        "instantaneous_delta_N_right", "instantaneous_delta_N_total", "instantaneous_q_x",
        "instantaneous_rank_gap", "principal_overlap", "selected_weight_floor",
    )
    arrays = {field: np.empty((2, 2, samples, points)) for field in curve_fields}
    arrays["continued_density_x"] = np.empty((2, 2, samples, points, 20))
    arrays["source_density_x"] = np.empty((2, 2, samples, 20))
    scalar_fields = (
        "rank", "input_frame_gram_residual", "input_projector_residual",
        "flattened_parent_involution_residual", "large_gauge_parent_error",
    )
    arrays.update({field: np.empty((2, 2, samples)) for field in scalar_fields})
    for task in campaign.pump_tasks(config):
        wi, di, si = WALLS.index(task["wall"]), DIRECTIONS.index(task["direction"]), int(task["sample_id"])
        path, _ = campaign.result_paths(output_root, task)
        with np.load(path, allow_pickle=False) as saved:
            for field in curve_fields + ("continued_density_x", "source_density_x") + scalar_fields:
                arrays[field][wi, di, si] = np.asarray(saved[field])
    metadata = {
        "schema": ANALYSIS_SCHEMA, "campaign_id": config["campaign_id"],
        "config_hash": status["config_hash"], "source_hashes": status["source_hashes"],
        "analysis_source_sha256": campaign.sha256_path(Path(__file__).resolve()),
        "samples_per_wall": samples, "walls": list(WALLS), "directions": list(DIRECTIONS),
        "independent_sampling_unit": config["ensemble"]["independent_sampling_unit"],
    }
    return arrays, metadata


def _write_endpoint_csv(path: Path, arrays: dict[str, np.ndarray], threshold: float) -> None:
    fields = [
        "wall", "sample_id", "ccw_q_x", "cw_q_x", "direction_odd_q_x",
        "direction_even_q_x", "pump_event", "rank",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for wi, wall in enumerate(WALLS):
            ccw = arrays["continued_q_x"][wi, 0, :, -1]
            cw = arrays["continued_q_x"][wi, 1, :, -1]
            odd, even = 0.5 * (ccw - cw), 0.5 * (ccw + cw)
            for sample in range(len(odd)):
                writer.writerow(
                    {
                        "wall": wall, "sample_id": sample,
                        "ccw_q_x": float(ccw[sample]), "cw_q_x": float(cw[sample]),
                        "direction_odd_q_x": float(odd[sample]),
                        "direction_even_q_x": float(even[sample]),
                        "pump_event": int(abs(odd[sample]) > threshold),
                        "rank": int(arrays["rank"][wi, 0, sample]),
                    }
                )


def analyze(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    arrays, metadata = load_verified(config, output_root)
    root, figures = output_root / "analysis", output_root / "analysis" / "figures"
    root.mkdir(parents=True, exist_ok=True)
    threshold = float(config["acceptance"]["pump_event_threshold"])
    endpoint_csv = root / "state_projector_pump_endpoints.csv"
    _write_endpoint_csv(endpoint_csv, arrays, threshold)
    aggregate = root / "state_projector_pump_aggregate.npz"
    np.savez_compressed(aggregate, metadata_json=np.asarray(campaign.canonical_json(metadata)), **arrays)

    ensemble = config["ensemble"]
    draws, seed, confidence = int(ensemble["bootstrap_draws"]), int(ensemble["bootstrap_seed"]), float(ensemble["confidence_level"])
    endpoint_rows = []
    endpoint_odd: list[np.ndarray] = []
    endpoint_events: list[np.ndarray] = []
    for wi, wall in enumerate(WALLS):
        ccw = arrays["continued_q_x"][wi, 0, :, -1]
        cw = arrays["continued_q_x"][wi, 1, :, -1]
        odd, even = 0.5 * (ccw - cw), 0.5 * (ccw + cw)
        mean, low, high = _bootstrap_mean(odd, draws=draws, seed=seed + wi, confidence=confidence)
        event_mask = np.abs(odd) > threshold
        events = int(np.sum(event_mask))
        event_low, event_high = _clopper_pearson(events, len(odd), confidence)
        endpoint_odd.append(odd)
        endpoint_events.append(event_mask.astype(np.float64))
        endpoint_rows.append(
            {
                "wall": wall, "samples": len(odd),
                "ccw_mean": float(ccw.mean()), "ccw_std": float(ccw.std(ddof=1)),
                "cw_mean": float(cw.mean()), "cw_std": float(cw.std(ddof=1)),
                "direction_odd_mean": float(mean), "direction_odd_ci_low": float(low),
                "direction_odd_ci_high": float(high),
                "maximum_abs_direction_even": float(np.max(np.abs(even))),
                "pump_event_threshold": threshold, "pump_events": events,
                "pump_event_fraction": events / len(odd),
                "pump_event_cp95_low": event_low, "pump_event_cp95_high": event_high,
                "event_endpoint_mean": float(odd[event_mask].mean()),
                "event_endpoint_minimum": float(odd[event_mask].min()),
                "non_event_maximum_abs_endpoint": float(np.max(np.abs(odd[~event_mask]))),
            }
        )

    odd_difference = _bootstrap_difference(
        endpoint_odd[0], endpoint_odd[1],
        draws=draws, seed=seed + 100, confidence=confidence,
    )
    event_fraction_difference = _bootstrap_difference(
        endpoint_events[0], endpoint_events[1],
        draws=draws, seed=seed + 101, confidence=confidence,
    )
    wall_contrast = {
        "contrast": "soft_minus_hard",
        "direction_odd_mean_difference": odd_difference[0],
        "direction_odd_mean_difference_ci_low": odd_difference[1],
        "direction_odd_mean_difference_ci_high": odd_difference[2],
        "pump_event_fraction_difference": event_fraction_difference[0],
        "pump_event_fraction_difference_ci_low": event_fraction_difference[1],
        "pump_event_fraction_difference_ci_high": event_fraction_difference[2],
        "resampling": "independent wall-specific trajectories",
    }

    _style()
    colors = {"ccw": "#1f77b4", "cw": "#d62728"}
    markers = {"ccw": "o", "cw": "s"}
    figure_paths: dict[str, dict[str, str]] = {}

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), sharex=True, sharey=True)
    for wi, (wall, ax) in enumerate(zip(WALLS, axes)):
        for di, direction in enumerate(DIRECTIONS):
            values = arrays["continued_q_x"][wi, di]
            mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
            x = arrays["path_fraction"][wi, di, 0]
            ax.plot(x, mean, color=colors[direction], marker=markers[direction], markevery=8,
                    linewidth=1.2, markersize=3, label=direction.upper())
            ax.fill_between(x, mean - std, mean + std, color=colors[direction], alpha=0.18, linewidth=0)
            control = arrays["instantaneous_q_x"][wi, di].mean(axis=0)
            ax.plot(x, control, color=colors[direction], linestyle=":", linewidth=0.8)
        ax.axhline(0, color="0.45", linestyle="--", linewidth=0.8)
        ax.set_title(f"{wall.capitalize()} wall, $S=100$")
        ax.set_xlabel(r"flux-path fraction $|\phi|/(2\pi)$")
        ax.text(-0.13, 1.04, f"({chr(97 + wi)})", transform=ax.transAxes, fontsize=9)
    axes[0].set_ylabel(r"$q_x=(\Delta N_R-\Delta N_L)/2$")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    figure_paths["mean_qx"] = _save(fig, figures, "state_projector_pump_mean_qx")

    representative_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), sharex=True, sharey=True)
    for wi, (wall, ax) in enumerate(zip(WALLS, axes)):
        odd = 0.5 * (
            arrays["continued_q_x"][wi, 0, :, -1]
            - arrays["continued_q_x"][wi, 1, :, -1]
        )
        event_indices = np.flatnonzero(np.abs(odd) > threshold)
        pool = event_indices if event_indices.size else np.arange(odd.size)
        selected = int(pool[np.argmin(np.abs(odd[pool] - 1.0))])
        representative_rows.append(
            {
                "wall": wall,
                "sample_id": selected,
                "direction_odd_endpoint": float(odd[selected]),
                "ccw_endpoint": float(arrays["continued_q_x"][wi, 0, selected, -1]),
                "cw_endpoint": float(arrays["continued_q_x"][wi, 1, selected, -1]),
                "selection": "closest direction-odd endpoint to +1 among classified events",
            }
        )
        for di, direction in enumerate(DIRECTIONS):
            signed_flux = arrays["phi"][wi, di, selected] / (2.0 * np.pi)
            ax.plot(
                signed_flux,
                arrays["continued_q_x"][wi, di, selected],
                color=colors[direction], marker=markers[direction], markevery=8,
                linewidth=1.2, markersize=3, label=direction.upper(),
            )
        ax.axhline(0, color="0.45", linestyle="--", linewidth=0.8)
        ax.axhline(1, color="0.72", linestyle=":", linewidth=0.7)
        ax.axhline(-1, color="0.72", linestyle=":", linewidth=0.7)
        ax.set_title(f"{wall.capitalize()} wall, sample {selected}")
        ax.set_xlabel(r"signed flux $\phi/(2\pi)$")
        ax.text(-0.13, 1.04, f"({chr(97 + wi)})", transform=ax.transAxes, fontsize=9)
    axes[0].set_ylabel(r"$q_x=(\Delta N_R-\Delta N_L)/2$")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    figure_paths["representative_quantized_paths"] = _save(
        fig, figures, "state_projector_pump_representative_quantized_paths"
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), sharex=True, sharey=True)
    for wi, (wall, ax) in enumerate(zip(WALLS, axes)):
        fraction = arrays["path_fraction"][wi, 0, 0]
        sign_aligned = 0.5 * (
            arrays["continued_q_x"][wi, 0] - arrays["continued_q_x"][wi, 1]
        )
        for row in sign_aligned:
            ax.plot(fraction, row, color="#1f77b4", alpha=0.16, linewidth=0.65)
        ax.plot(fraction, sign_aligned.mean(axis=0), color="black", linewidth=1.3, label="trajectory mean")
        ax.axhline(0, color="0.45", linestyle="--", linewidth=0.8)
        ax.set_title(f"{wall.capitalize()} wall")
        ax.set_xlabel(r"flux-path fraction $|\phi|/(2\pi)$")
        ax.text(-0.13, 1.04, f"({chr(97 + wi)})", transform=ax.transAxes, fontsize=9)
    axes[0].set_ylabel(r"direction-odd $q_x$")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    figure_paths["individual_paths"] = _save(fig, figures, "state_projector_pump_individual_paths")

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65))
    bins = np.linspace(-0.1, 1.1, 25)
    for wi, (wall, ax) in enumerate(zip(WALLS, axes)):
        odd = 0.5 * (
            arrays["continued_q_x"][wi, 0, :, -1]
            - arrays["continued_q_x"][wi, 1, :, -1]
        )
        ax.hist(odd, bins=bins, color="#1f77b4", edgecolor="white", linewidth=0.5)
        ax.axvline(threshold, color="0.3", linestyle="--", linewidth=0.8, label="event threshold")
        ax.set_title(f"{wall.capitalize()} wall")
        ax.set_xlabel(r"endpoint direction-odd $q_x$")
        ax.set_ylabel("trajectories")
        ax.text(-0.13, 1.04, f"({chr(97 + wi)})", transform=ax.transAxes, fontsize=9)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    figure_paths["endpoint_histogram"] = _save(fig, figures, "state_projector_pump_endpoint_histogram")

    fig, ax = plt.subplots(figsize=(3.375, 2.65))
    rates = np.asarray([row["pump_event_fraction"] for row in endpoint_rows])
    lows = np.asarray([row["pump_event_cp95_low"] for row in endpoint_rows])
    highs = np.asarray([row["pump_event_cp95_high"] for row in endpoint_rows])
    ax.errorbar(np.arange(2), rates, yerr=np.vstack((rates - lows, highs - rates)),
                fmt="o", color="#1f77b4", capsize=3)
    ax.set_xticks(np.arange(2), ("Soft", "Hard"))
    ax.set_ylabel(r"fraction with $|q_x^{\rm odd}|>0.5$")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    figure_paths["event_fraction"] = _save(fig, figures, "state_projector_pump_event_fraction")

    summary = {
        **metadata, "aggregate": str(aggregate), "endpoint_csv": str(endpoint_csv),
        "figures": figure_paths, "endpoint_statistics": endpoint_rows,
        "representative_quantized_paths": representative_rows,
        "wall_contrast": wall_contrast,
        "maximum_abs_continued_charge_residual": float(np.max(np.abs(arrays["continued_delta_N_total"]))),
        "maximum_abs_instantaneous_endpoint": float(np.max(np.abs(arrays["instantaneous_q_x"][:, :, :, -1]))),
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
