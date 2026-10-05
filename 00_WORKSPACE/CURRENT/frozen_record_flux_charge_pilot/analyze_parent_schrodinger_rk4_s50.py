#!/usr/bin/env python3
"""Analysis for the Ny=24 S50 finite-difference parent-Hamiltonian ramp."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_parent_schrodinger_rk4_s50 as campaign  # noqa: E402
import run_state_projector_pump_s100 as static_campaign  # noqa: E402


DEFAULT_CONFIG = campaign.DEFAULT_CONFIG
DEFAULT_OUTPUT = campaign.DEFAULT_OUTPUT
ANALYSIS_SCHEMA = "parent_schrodinger_rk4_analysis_v3"
WALLS = ("soft", "hard")
DIRECTIONS = ("ccw", "cw")


def _crossing_slope(phi: np.ndarray, gap: np.ndarray) -> tuple[float, float]:
    coordinate = np.abs(np.asarray(phi, dtype=float))
    values = np.asarray(gap, dtype=float)
    center = int(np.argmin(values))
    lower, upper = max(0, center - 2), min(values.size, center + 3)
    x = coordinate[lower:upper] - coordinate[center]
    minimum = float(values[center])

    def residual(parameter: np.ndarray) -> np.ndarray:
        return np.sqrt(minimum * minimum + (float(parameter[0]) * x) ** 2) - values[lower:upper]

    fit = least_squares(residual, x0=np.asarray([0.743]), bounds=(1e-6, 10.0))
    return minimum, float(fit.x[0])


def _bootstrap_mean(values: np.ndarray, *, seed: int, draws: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    distribution = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return float(values.mean()), float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))


def _load(config: dict[str, Any], output_root: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, np.ndarray]]]:
    context = campaign.source_context(config)
    status = campaign.inventory(config, output_root, context)
    if not all(row[0] for row in status["paths"].values()):
        missing = [key for key, row in status["paths"].items() if not row[0]]
        raise RuntimeError(f"analysis requires 100 verified paths; missing {len(missing)}")
    source_config = static_campaign.load_config(
        (PROJECT_ROOT / config["source_campaign"]["config"]).resolve()
    )
    static_campaign.validate_config(source_config)
    source_root = (PROJECT_ROOT / config["source_campaign"]["output_root"]).resolve()
    source_status = static_campaign.inventory(source_config, source_root)
    rows: list[dict[str, Any]] = []
    histories: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {
        (wall, direction): {"phi": [], "q_x": [], "delta_N_left": [], "delta_N_right": []}
        for wall in WALLS for direction in DIRECTIONS
    }
    task_lookup = {task["task_id"]: task for task in campaign.tasks(config)}
    for wall in WALLS:
        for sample_id in config["source_campaign"]["sample_ids"]:
            dynamic: dict[str, dict[str, Any]] = {}
            static: dict[str, dict[str, float]] = {}
            for direction in DIRECTIONS:
                task = task_lookup[f"rk4_{wall}_{direction}_sample_{int(sample_id):03d}"]
                result_path, _ = campaign.result_paths(output_root, task)
                with np.load(result_path, allow_pickle=False) as saved:
                    q_history = np.asarray(saved["q_x"], dtype=float)
                    peak_index = int(np.argmax(np.abs(q_history)))
                    dynamic[direction] = {
                        "q_history": q_history,
                        "q_pi": float(q_history[q_history.size // 2]),
                        "q_endpoint": float(q_history[-1]),
                        "q_peak": float(q_history[peak_index]),
                        "peak_progress": float(abs(np.asarray(saved["phi"])[peak_index]) / (2.0 * np.pi)),
                        "excitation": float(saved["instantaneous_excitation_number"]),
                        "gauge_overlap": float(saved["large_gauge_mean_principal_overlap"]),
                        "gauge_overlap_min": float(saved["large_gauge_minimum_principal_overlap"]),
                        "pre_qr_gram": float(saved["maximum_pre_qr_gram_residual"]),
                        "post_qr_gram": float(saved["maximum_post_qr_gram_residual"]),
                        "charge_residual": float(np.max(np.abs(saved["delta_N_total"]))),
                    }
                    for key in histories[(wall, direction)]:
                        histories[(wall, direction)][key].append(np.asarray(saved[key], dtype=float))
                source_task = {
                    "stage": "pump",
                    "task_id": f"pump_{wall}_{direction}_sample_{int(sample_id):03d}",
                    "wall": wall,
                    "direction": direction,
                    "sigma": int(source_config["projector_pump"]["directions"][direction]),
                    "sample_id": int(sample_id),
                    "burnin_task_id": f"burnin_{wall}_sample_{int(sample_id):03d}",
                }
                source_ok = source_status["pumps"][source_task["task_id"]]
                if not source_ok[0]:
                    raise RuntimeError(f"static comparison path is invalid: {source_task['task_id']}: {source_ok[1]}")
                source_path, _ = static_campaign.result_paths(source_root, source_task)
                with np.load(source_path, allow_pickle=False) as saved:
                    minimum_gap, crossing_slope = _crossing_slope(
                        saved["phi"], saved["instantaneous_rank_gap"]
                    )
                    static[direction] = {
                        "q_endpoint": float(saved["continued_q_x"][-1]),
                        "minimum_gap": minimum_gap,
                        "crossing_slope": crossing_slope,
                    }
            dynamic_odd = 0.5 * (dynamic["ccw"]["q_endpoint"] - dynamic["cw"]["q_endpoint"])
            dynamic_odd_history = 0.5 * (
                dynamic["ccw"]["q_history"] - dynamic["cw"]["q_history"]
            )
            dynamic_even_history = 0.5 * (
                dynamic["ccw"]["q_history"] + dynamic["cw"]["q_history"]
            )
            odd_peak_index = int(np.argmax(np.abs(dynamic_odd_history)))
            static_odd = 0.5 * (static["ccw"]["q_endpoint"] - static["cw"]["q_endpoint"])
            minimum_gap = min(static[direction]["minimum_gap"] for direction in DIRECTIONS)
            crossing_slope = float(np.mean([static[direction]["crossing_slope"] for direction in DIRECTIONS]))
            predicted_diabatic_probability = float(
                np.exp(-float(config["evolution"]["ramp_time"]) * minimum_gap**2 / (4.0 * crossing_slope))
            )
            ramp_time_for_pd_0p1 = float(4.0 * crossing_slope * np.log(10.0) / minimum_gap**2)
            ramp_time_for_pd_0p01 = float(4.0 * crossing_slope * np.log(100.0) / minimum_gap**2)
            rows.append(
                {
                    "wall": wall,
                    "sample_id": int(sample_id),
                    "static_q_x_odd": static_odd,
                    "static_event": int(abs(static_odd) > float(config["analysis"]["static_event_threshold"])),
                    "static_minimum_gap": minimum_gap,
                    "local_crossing_slope": crossing_slope,
                    "predicted_diabatic_probability": predicted_diabatic_probability,
                    "predicted_rk4_q_x_odd": predicted_diabatic_probability * static_odd,
                    "ramp_time_for_predicted_diabatic_probability_0p1": ramp_time_for_pd_0p1,
                    "ramp_time_for_predicted_diabatic_probability_0p01": ramp_time_for_pd_0p01,
                    "rk4_ccw_q_x": dynamic["ccw"]["q_endpoint"],
                    "rk4_cw_q_x": dynamic["cw"]["q_endpoint"],
                    "rk4_ccw_q_x_pi": dynamic["ccw"]["q_pi"],
                    "rk4_cw_q_x_pi": dynamic["cw"]["q_pi"],
                    "rk4_q_x_odd_pi": float(dynamic_odd_history[dynamic_odd_history.size // 2]),
                    "rk4_q_x_odd": dynamic_odd,
                    "rk4_q_x_even": 0.5 * (dynamic["ccw"]["q_endpoint"] + dynamic["cw"]["q_endpoint"]),
                    "rk4_max_abs_q_x_odd": float(np.max(np.abs(dynamic_odd_history))),
                    "rk4_peak_q_x_odd": float(dynamic_odd_history[odd_peak_index]),
                    "rk4_peak_progress": float(odd_peak_index / (dynamic_odd_history.size - 1)),
                    "rk4_max_abs_q_x_even_over_path": float(np.max(np.abs(dynamic_even_history))),
                    "rk4_pump_like": int(abs(dynamic_odd) > 0.5),
                    "ccw_excitation_number": dynamic["ccw"]["excitation"],
                    "cw_excitation_number": dynamic["cw"]["excitation"],
                    "ccw_large_gauge_mean_overlap": dynamic["ccw"]["gauge_overlap"],
                    "cw_large_gauge_mean_overlap": dynamic["cw"]["gauge_overlap"],
                    "minimum_large_gauge_principal_overlap": min(
                        dynamic["ccw"]["gauge_overlap_min"], dynamic["cw"]["gauge_overlap_min"]
                    ),
                    "maximum_pre_qr_gram_residual": max(dynamic["ccw"]["pre_qr_gram"], dynamic["cw"]["pre_qr_gram"]),
                    "maximum_post_qr_gram_residual": max(dynamic["ccw"]["post_qr_gram"], dynamic["cw"]["post_qr_gram"]),
                    "maximum_charge_residual": max(dynamic["ccw"]["charge_residual"], dynamic["cw"]["charge_residual"]),
                }
            )
    stacked = {
        key: {field: np.stack(values) for field, values in payload.items()}
        for key, payload in histories.items()
    }
    return rows, stacked


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def _save(fig: plt.Figure, root: Path, stem: str) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    pdf, png = root / f"{stem}.pdf", root / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}


def _plot(rows: list[dict[str, Any]], histories: dict[tuple[str, str], dict[str, np.ndarray]], root: Path) -> dict[str, dict[str, str]]:
    _style()
    colors = {"ccw": "#0072B2", "cw": "#D55E00"}
    linestyles = {"ccw": "-", "cw": "--"}
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.3), constrained_layout=True)
    for wall_index, wall in enumerate(WALLS):
        axis = axes[wall_index, 0]
        for direction in DIRECTIONS:
            payload = histories[(wall, direction)]
            progress = np.abs(payload["phi"][0]) / (2.0 * np.pi)
            values = payload["q_x"]
            mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
            axis.plot(progress, mean, color=colors[direction], ls=linestyles[direction], label=direction.upper())
            axis.fill_between(progress, mean - std, mean + std, color=colors[direction], alpha=0.18, linewidth=0)
        axis.axhline(0.0, color="0.35", ls=":", lw=0.8)
        axis.set_xlabel(r"ramp progress $|\phi|/(2\pi)$")
        axis.set_ylabel(r"$q_x$")
        axis.set_title(f"{wall.capitalize()} wall, S=25")
        axis.legend(frameon=False, ncol=2)
    for wall, marker, color in (("soft", "o", "#0072B2"), ("hard", "s", "#D55E00")):
        selected = [row for row in rows if row["wall"] == wall]
        axes[0, 1].scatter(
            [row["static_q_x_odd"] for row in selected],
            [row["rk4_q_x_odd"] for row in selected],
            facecolors="none", edgecolors=color, marker=marker, label=wall.capitalize(), s=25,
        )
        event = np.asarray([row["static_event"] for row in selected], dtype=bool)
        gaps = np.asarray([row["static_minimum_gap"] for row in selected])
        dynamic = np.asarray([abs(row["rk4_q_x_odd"]) for row in selected])
        axes[1, 1].scatter(
            gaps[~event], dynamic[~event], facecolors="none", edgecolors=color,
            marker=marker, alpha=0.8, s=25,
        )
        axes[1, 1].scatter(
            gaps[event], dynamic[event], facecolors=color, edgecolors=color,
            marker=marker, alpha=0.8, s=25, label=f"{wall.capitalize()} static event",
        )
    axes[0, 1].axline((0, 0), slope=1, color="0.35", ls=":", lw=0.8)
    axes[0, 1].axhline(0, color="0.6", lw=0.6)
    axes[0, 1].axvline(0, color="0.6", lw=0.6)
    axes[0, 1].set_xlabel(r"static continued $q_x^{\rm odd}(2\pi)$")
    axes[0, 1].set_ylabel(r"RK4 $q_x^{\rm odd}(2\pi)$")
    axes[0, 1].legend(frameon=False)
    axes[1, 1].axhline(0.5, color="0.35", ls="--", lw=0.8)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel(r"minimum static occupied--empty gap")
    axes[1, 1].set_ylabel(r"$|q_x^{\rm odd}(2\pi)|$")
    axes[1, 1].legend(frameon=False)
    for label, axis in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        axis.text(-0.14, 1.04, label, transform=axis.transAxes, fontweight="bold", va="bottom")
    paths = {"overview": _save(fig, root, "parent_schrodinger_rk4_s50_overview")}

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.8), constrained_layout=True)
    bins = np.linspace(-1.1, 1.1, 23)
    for axis, wall in zip(axes, WALLS):
        selected = [row for row in rows if row["wall"] == wall]
        static = np.asarray([row["static_q_x_odd"] for row in selected])
        dynamic = np.asarray([row["rk4_q_x_odd"] for row in selected])
        axis.hist(static, bins=bins, histtype="step", color="0.25", lw=1.2, label="static continuation")
        axis.hist(dynamic, bins=bins, color="#56B4E9", alpha=0.55, label=r"RK4, $T=10^4$")
        axis.axvline(0.0, color="0.35", ls=":", lw=0.8)
        axis.set_title(f"{wall.capitalize()} wall")
        axis.set_xlabel(r"endpoint $q_x^{\rm odd}$")
        axis.set_ylabel("trajectories")
        axis.legend(frameon=False)
    axes[0].text(-0.13, 1.04, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.13, 1.04, "(b)", transform=axes[1].transAxes, fontweight="bold")
    paths["histogram"] = _save(fig, root, "parent_schrodinger_rk4_s50_endpoint_histogram")

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.8), constrained_layout=True)
    for wall, marker, color in (("soft", "o", "#0072B2"), ("hard", "s", "#D55E00")):
        selected = [row for row in rows if row["wall"] == wall]
        gap = np.asarray([row["static_minimum_gap"] for row in selected])
        probability = np.asarray([row["predicted_diabatic_probability"] for row in selected])
        midpoint = np.abs(np.asarray([row["rk4_q_x_odd_pi"] for row in selected]))
        axes[0].scatter(gap, midpoint, facecolors="none", edgecolors=color, marker=marker, s=25, label=wall.capitalize())
        axes[1].scatter(probability, midpoint, facecolors="none", edgecolors=color, marker=marker, s=25, label=wall.capitalize())
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"minimum static occupied--empty gap")
    axes[0].set_ylabel(r"$|q_x^{\rm odd}(\pi)|$")
    axes[1].set_xlabel(r"two-level $P_{\rm diabatic}$")
    axes[1].set_ylabel(r"$|q_x^{\rm odd}(\pi)|$")
    for axis in axes:
        axis.axhline(0.5, color="0.35", ls=":", lw=0.8)
        axis.legend(frameon=False)
    axes[0].text(-0.13, 1.04, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.13, 1.04, "(b)", transform=axes[1].transAxes, fontweight="bold")
    paths["crossing"] = _save(fig, root, "parent_schrodinger_rk4_s50_crossing_diagnostics")
    return paths


def analyze(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    rows, histories = _load(config, output_root)
    analysis_root = Path(output_root) / "analysis"
    _write_csv(analysis_root / "parent_schrodinger_rk4_samplewise.csv", rows)
    figures = _plot(rows, histories, analysis_root / "figures")
    statistics = []
    draws = int(config["analysis"]["bootstrap_draws"])
    for wall_index, wall in enumerate(WALLS):
        selected = [row for row in rows if row["wall"] == wall]
        dynamic = np.asarray([row["rk4_q_x_odd"] for row in selected])
        dynamic_pi = np.asarray([row["rk4_q_x_odd_pi"] for row in selected])
        dynamic_peak = np.asarray([row["rk4_max_abs_q_x_odd"] for row in selected])
        static = np.asarray([row["static_q_x_odd"] for row in selected])
        gaps = np.asarray([row["static_minimum_gap"] for row in selected])
        mean, low, high = _bootstrap_mean(
            dynamic, seed=int(config["analysis"]["bootstrap_seed"]) + wall_index, draws=draws
        )
        pi_mean, pi_low, pi_high = _bootstrap_mean(
            dynamic_pi, seed=int(config["analysis"]["bootstrap_seed"]) + 10 + wall_index, draws=draws
        )
        peak_mean, peak_low, peak_high = _bootstrap_mean(
            dynamic_peak, seed=int(config["analysis"]["bootstrap_seed"]) + 20 + wall_index, draws=draws
        )
        correlation = spearmanr(gaps, np.abs(dynamic))
        midpoint_correlation = spearmanr(gaps, np.abs(dynamic_pi))
        predicted_probability = np.asarray(
            [row["predicted_diabatic_probability"] for row in selected]
        )
        predicted_q = np.asarray([row["predicted_rk4_q_x_odd"] for row in selected])
        excitation_numbers = np.asarray([
            0.5 * (row["ccw_excitation_number"] + row["cw_excitation_number"])
            for row in selected
        ])
        ramp_time_pd_0p1 = np.asarray(
            [row["ramp_time_for_predicted_diabatic_probability_0p1"] for row in selected]
        )
        ramp_time_pd_0p01 = np.asarray(
            [row["ramp_time_for_predicted_diabatic_probability_0p01"] for row in selected]
        )
        prediction_correlation = spearmanr(predicted_probability, np.abs(dynamic))
        midpoint_prediction_correlation = spearmanr(predicted_probability, np.abs(dynamic_pi))
        excitation_spearman = spearmanr(excitation_numbers, np.abs(dynamic))
        excitation_pearson = pearsonr(excitation_numbers, np.abs(dynamic))
        statistics.append(
            {
                "wall": wall,
                "endpoint_states": len(selected),
                "static_pump_like": int(np.sum(np.abs(static) > 0.5)),
                "rk4_pump_like": int(np.sum(np.abs(dynamic) > 0.5)),
                "rk4_mean_q_x_odd": mean,
                "rk4_mean_q_x_odd_bootstrap95": [low, high],
                "rk4_mean_q_x_odd_at_pi": pi_mean,
                "rk4_mean_q_x_odd_at_pi_bootstrap95": [pi_low, pi_high],
                "rk4_mean_max_abs_q_x_odd": peak_mean,
                "rk4_mean_max_abs_q_x_odd_bootstrap95": [peak_low, peak_high],
                "rk4_std_q_x_odd": float(np.std(dynamic, ddof=1)),
                "rk4_median_abs_q_x_odd": float(np.median(np.abs(dynamic))),
                "rk4_endpoint_closure_count_abs_below_0p25": int(np.sum(np.abs(dynamic) < 0.25)),
                "rk4_endpoint_order_one_count_abs_above_0p75": int(np.sum(np.abs(dynamic) > 0.75)),
                "rk4_endpoint_intermediate_count": int(np.sum((np.abs(dynamic) >= 0.25) & (np.abs(dynamic) <= 0.75))),
                "rk4_median_peak_progress": float(np.median([row["rk4_peak_progress"] for row in selected])),
                "rk4_max_abs_q_x_even": float(np.max(np.abs([row["rk4_q_x_even"] for row in selected]))),
                "rk4_max_abs_q_x_even_over_path": float(max(row["rk4_max_abs_q_x_even_over_path"] for row in selected)),
                "spearman_static_gap_vs_abs_rk4_q_x": float(correlation.statistic),
                "spearman_pvalue": float(correlation.pvalue),
                "spearman_static_gap_vs_abs_rk4_q_x_at_pi": float(midpoint_correlation.statistic),
                "spearman_at_pi_pvalue": float(midpoint_correlation.pvalue),
                "spearman_predicted_probability_vs_abs_rk4_q_x": float(prediction_correlation.statistic),
                "spearman_prediction_pvalue": float(prediction_correlation.pvalue),
                "spearman_predicted_probability_vs_abs_rk4_q_x_at_pi": float(midpoint_prediction_correlation.statistic),
                "spearman_at_pi_prediction_pvalue": float(midpoint_prediction_correlation.pvalue),
                "landau_zener_q_x_mean_absolute_error": float(np.mean(np.abs(dynamic - predicted_q))),
                "landau_zener_predicted_pump_like": int(np.sum(predicted_probability > 0.5)),
                "predicted_diabatic_probability_median_at_ramp_time": float(
                    np.median(predicted_probability)
                ),
                "predicted_diabatic_probability_below_0p1_count_at_ramp_time": int(
                    np.sum(predicted_probability < 0.1)
                ),
                "predicted_diabatic_probability_below_0p01_count_at_ramp_time": int(
                    np.sum(predicted_probability < 0.01)
                ),
                "ramp_time_for_predicted_diabatic_probability_0p1_quantiles": {
                    key: float(value)
                    for key, value in zip(
                        ("minimum", "q25", "median", "q75", "maximum"),
                        np.quantile(ramp_time_pd_0p1, [0.0, 0.25, 0.5, 0.75, 1.0]),
                        strict=True,
                    )
                },
                "ramp_time_for_predicted_diabatic_probability_0p01_quantiles": {
                    key: float(value)
                    for key, value in zip(
                        ("minimum", "q25", "median", "q75", "maximum"),
                        np.quantile(ramp_time_pd_0p01, [0.0, 0.25, 0.5, 0.75, 1.0]),
                        strict=True,
                    )
                },
                "mean_excitation_number": float(np.mean(excitation_numbers)),
                "mean_abs_q_x_odd_minus_excitation_number": float(
                    np.mean(np.abs(np.abs(dynamic) - excitation_numbers))
                ),
                "spearman_excitation_number_vs_abs_rk4_q_x": float(
                    excitation_spearman.statistic
                ),
                "spearman_excitation_number_pvalue": float(excitation_spearman.pvalue),
                "pearson_excitation_number_vs_abs_rk4_q_x": float(
                    excitation_pearson.statistic
                ),
                "pearson_excitation_number_pvalue": float(excitation_pearson.pvalue),
            }
        )
    summary = {
        "schema": ANALYSIS_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_hash": campaign.scientific_config_hash(config),
        "endpoint_states": 50,
        "paths": 100,
        "ramp_time": float(config["evolution"]["ramp_time"]),
        "dt": float(config["evolution"]["ramp_time"]) / (
            int(config["evolution"]["flux_intervals"]) * int(config["evolution"]["steps_per_interval"])
        ),
        "statistics": statistics,
        "maximum_charge_residual": float(max(row["maximum_charge_residual"] for row in rows)),
        "maximum_post_qr_gram_residual": float(max(row["maximum_post_qr_gram_residual"] for row in rows)),
        "figures": figures,
        "interpretation": (
            "This is unitary finite-rate evolution under the state-derived parent Hamiltonian. "
            "It is neither spectral reprojection nor monitored-circuit evolution."
        ),
    }
    analysis_root.mkdir(parents=True, exist_ok=True)
    (analysis_root / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    config = campaign.load_config(DEFAULT_CONFIG)
    campaign.validate_config(config)
    analyze(config, DEFAULT_OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
