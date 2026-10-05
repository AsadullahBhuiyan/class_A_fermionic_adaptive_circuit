#!/usr/bin/env python3
"""Compare N20x24 and N24x24 parent-Hamiltonian RK4 ensembles."""

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
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import analyze_parent_schrodinger_rk4_s50 as base_analysis  # noqa: E402
import run_parent_schrodinger_rk4_s50 as n20_campaign  # noqa: E402
import run_parent_schrodinger_rk4_n24x24_s50 as n24_campaign  # noqa: E402
import run_state_projector_pump_s100 as n20_static  # noqa: E402
import run_state_projector_pump_variants as n24_static  # noqa: E402


ANALYSIS_SCHEMA = "parent_schrodinger_rk4_width_comparison_v1"
WALLS = ("soft", "hard")
DIRECTIONS = ("ccw", "cw")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _dynamic_context(module: Any, config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    context = module.source_context(config)
    status = module.inventory(config, output_root, context)
    missing = [key for key, row in status["paths"].items() if not row[0]]
    if missing:
        raise RuntimeError(f"{config['campaign_id']} has {len(missing)} invalid or missing paths")
    return context


def _static_context(module: Any, config: dict[str, Any]) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    source_config = module.load_config((PROJECT_ROOT / config["source_campaign"]["config"]).resolve())
    module.validate_config(source_config)
    source_root = (PROJECT_ROOT / config["source_campaign"]["output_root"]).resolve()
    status = module.inventory(source_config, source_root)
    return source_config, source_root, status


def _load_size(
    *,
    nx: int,
    config: dict[str, Any],
    output_root: Path,
    dynamic_module: Any,
    static_module: Any,
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], np.ndarray], np.ndarray]:
    _dynamic_context(dynamic_module, config, output_root)
    source_config, source_root, source_status = _static_context(static_module, config)
    task_lookup = {row["task_id"]: row for row in dynamic_module.tasks(config)}
    rows: list[dict[str, Any]] = []
    histories: dict[tuple[int, str], list[np.ndarray]] = {(nx, wall): [] for wall in WALLS}
    phi_reference: np.ndarray | None = None
    for wall in WALLS:
        for sample_id in config["source_campaign"]["sample_ids"]:
            dynamic: dict[str, dict[str, Any]] = {}
            static: dict[str, dict[str, float]] = {}
            for direction in DIRECTIONS:
                task = task_lookup[f"rk4_{wall}_{direction}_sample_{int(sample_id):03d}"]
                path, _ = n20_campaign.result_paths(output_root, task)
                with np.load(path, allow_pickle=False) as saved:
                    q = np.asarray(saved["q_x"], dtype=float)
                    phi = np.abs(np.asarray(saved["phi"], dtype=float))
                    if phi_reference is None:
                        phi_reference = phi
                    elif not np.array_equal(phi_reference, phi):
                        raise RuntimeError("RK4 flux grids differ")
                    dynamic[direction] = {
                        "q": q,
                        "endpoint": float(q[-1]),
                        "pi": float(q[q.size // 2]),
                        "excitation": float(saved["instantaneous_excitation_number"]),
                        "charge_residual": float(np.max(np.abs(saved["delta_N_total"]))),
                        "post_qr": float(saved["maximum_post_qr_gram_residual"]),
                    }
                static_task = {
                    "stage": "pump",
                    "task_id": f"pump_{wall}_{direction}_sample_{int(sample_id):03d}",
                    "wall": wall,
                    "direction": direction,
                    "sigma": int(source_config["projector_pump"]["directions"][direction]),
                    "sample_id": int(sample_id),
                    "burnin_task_id": f"burnin_{wall}_sample_{int(sample_id):03d}",
                }
                valid = source_status["pumps"][static_task["task_id"]]
                if not valid[0]:
                    raise RuntimeError(f"invalid static path {static_task['task_id']}: {valid[1]}")
                static_path, _ = static_module.result_paths(source_root, static_task)
                with np.load(static_path, allow_pickle=False) as saved:
                    gap, slope = base_analysis._crossing_slope(
                        saved["phi"], saved["instantaneous_rank_gap"]
                    )
                    static[direction] = {
                        "q": float(saved["continued_q_x"][-1]),
                        "gap": gap,
                        "slope": slope,
                    }
            q_odd_history = 0.5 * (dynamic["ccw"]["q"] - dynamic["cw"]["q"])
            q_even_history = 0.5 * (dynamic["ccw"]["q"] + dynamic["cw"]["q"])
            histories[(nx, wall)].append(q_odd_history)
            gap = min(static[d]["gap"] for d in DIRECTIONS)
            slope = float(np.mean([static[d]["slope"] for d in DIRECTIONS]))
            q_static = 0.5 * (static["ccw"]["q"] - static["cw"]["q"])
            q_endpoint = float(q_odd_history[-1])
            rows.append(
                {
                    "Nx": nx,
                    "Ny": 24,
                    "wall_separation": nx // 2,
                    "wall": wall,
                    "sample_id": int(sample_id),
                    "static_q_x_odd": q_static,
                    "static_minimum_gap": gap,
                    "local_crossing_slope": slope,
                    "delta_squared_times_T": gap * gap * float(config["evolution"]["ramp_time"]),
                    "predicted_diabatic_probability": float(
                        np.exp(-float(config["evolution"]["ramp_time"]) * gap * gap / (4.0 * slope))
                    ),
                    "rk4_q_x_odd_pi": float(q_odd_history[q_odd_history.size // 2]),
                    "rk4_q_x_odd": q_endpoint,
                    "rk4_max_abs_q_x_odd": float(np.max(np.abs(q_odd_history))),
                    "rk4_max_abs_q_x_even": float(np.max(np.abs(q_even_history))),
                    "mean_instantaneous_excitation_number": float(
                        0.5 * (dynamic["ccw"]["excitation"] + dynamic["cw"]["excitation"])
                    ),
                    "maximum_charge_residual": max(dynamic[d]["charge_residual"] for d in DIRECTIONS),
                    "maximum_post_qr_gram_residual": max(dynamic[d]["post_qr"] for d in DIRECTIONS),
                }
            )
    assert phi_reference is not None
    stacked = {key: np.stack(value) for key, value in histories.items()}
    return rows, stacked, phi_reference


def _bootstrap_difference(
    left: np.ndarray, right: np.ndarray, *, seed: int, draws: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = (
        right[rng.integers(0, right.size, size=(draws, right.size))].mean(axis=1)
        - left[rng.integers(0, left.size, size=(draws, left.size))].mean(axis=1)
    )
    return [float(right.mean() - left.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["CMU Sans Serif", "Computer Modern Sans Serif", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
    })


def _save(fig: plt.Figure, root: Path, stem: str) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    pdf, png = root / f"{stem}.pdf", root / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}


def _plots(
    rows: list[dict[str, Any]], histories: dict[tuple[int, str], np.ndarray],
    phi: np.ndarray, root: Path,
) -> dict[str, dict[str, str]]:
    _style()
    colors = {20: "#0072B2", 24: "#D55E00"}
    marks = {20: "o", 24: "s"}
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.8), constrained_layout=True)
    for axis, wall in zip(axes, WALLS):
        for nx in (20, 24):
            values = histories[(nx, wall)]
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            axis.plot(phi / (2 * np.pi), mean, color=colors[nx], label=rf"$N_x={nx}$")
            axis.fill_between(phi / (2 * np.pi), mean - sd, mean + sd, color=colors[nx], alpha=0.16, linewidth=0)
        axis.axhline(0, color="0.35", ls=":", lw=0.8)
        axis.set_title(f"{wall.capitalize()} wall")
        axis.set_xlabel(r"ramp progress $|\phi|/(2\pi)$")
        axis.set_ylabel(r"$q_x^{\rm odd}$")
        axis.legend(frameon=False)
    for label, axis in zip(("(a)", "(b)"), axes):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes, fontweight="bold")
    paths = {"mean_paths": _save(fig, root, "parent_rk4_width_mean_paths")}

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.8), constrained_layout=True)
    bins = np.linspace(-1.1, 1.1, 23)
    for axis, wall in zip(axes, WALLS):
        for nx in (20, 24):
            values = [row["rk4_q_x_odd"] for row in rows if row["Nx"] == nx and row["wall"] == wall]
            axis.hist(values, bins=bins, histtype="step", lw=1.3, color=colors[nx], label=rf"$N_x={nx}$")
        axis.axvline(0, color="0.35", ls=":", lw=0.8)
        axis.set_title(f"{wall.capitalize()} wall")
        axis.set_xlabel(r"endpoint $q_x^{\rm odd}(2\pi)$")
        axis.set_ylabel("endpoint states")
        axis.legend(frameon=False)
    for label, axis in zip(("(a)", "(b)"), axes):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes, fontweight="bold")
    paths["endpoint_histograms"] = _save(fig, root, "parent_rk4_width_endpoint_histograms")

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.8), constrained_layout=True)
    for axis, wall in zip(axes, WALLS):
        for nx in (20, 24):
            chosen = [row for row in rows if row["Nx"] == nx and row["wall"] == wall]
            axis.scatter(
                [row["static_minimum_gap"] for row in chosen],
                [abs(row["rk4_q_x_odd"]) for row in chosen],
                facecolors="none", edgecolors=colors[nx], marker=marks[nx], s=24,
                label=rf"$N_x={nx}$",
            )
        axis.set_xscale("log")
        axis.axhline(0.5, color="0.35", ls="--", lw=0.8)
        axis.set_title(f"{wall.capitalize()} wall")
        axis.set_xlabel("minimum occupied--empty gap")
        axis.set_ylabel(r"$|q_x^{\rm odd}(2\pi)|$")
        axis.legend(frameon=False)
    for label, axis in zip(("(a)", "(b)"), axes):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes, fontweight="bold")
    paths["gap_scatter"] = _save(fig, root, "parent_rk4_width_gap_scatter")
    return paths


def analyze(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    n24_campaign.validate_config(config)
    n20_config = n20_campaign.load_config((PROJECT_ROOT / config["dependency_gate"]["primary_config"]).resolve())
    n20_campaign.validate_config(n20_config)
    n20_root = (PROJECT_ROOT / config["dependency_gate"]["primary_output_root"]).resolve()
    rows20, histories20, phi20 = _load_size(
        nx=20, config=n20_config, output_root=n20_root,
        dynamic_module=n20_campaign, static_module=n20_static,
    )
    rows24, histories24, phi24 = _load_size(
        nx=24, config=config, output_root=output_root,
        dynamic_module=n24_campaign, static_module=n24_static,
    )
    if not np.array_equal(phi20, phi24):
        raise RuntimeError("size-comparison flux grids differ")
    rows = rows20 + rows24
    histories = histories20 | histories24
    analysis_root = Path(output_root) / "analysis" / "width_comparison"
    _write_csv(analysis_root / "parent_rk4_width_samplewise.csv", rows)
    figures = _plots(rows, histories, phi20, analysis_root / "figures")
    draws, seed = int(config["analysis"]["bootstrap_draws"]), int(config["analysis"]["bootstrap_seed"])
    thresholds = (float(config["analysis"]["static_event_threshold"]), float(config["analysis"]["strong_transport_threshold"]))
    statistics: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    for wall_index, wall in enumerate(WALLS):
        by_size = {}
        for nx in (20, 24):
            chosen = [row for row in rows if row["Nx"] == nx and row["wall"] == wall]
            q = np.abs(np.asarray([row["rk4_q_x_odd"] for row in chosen]))
            gap = np.asarray([row["static_minimum_gap"] for row in chosen])
            excitation = np.asarray([row["mean_instantaneous_excitation_number"] for row in chosen])
            by_size[nx] = q
            correlation = spearmanr(gap, q)
            statistics.append({
                "Nx": nx, "Ny": 24, "wall": wall, "states": len(chosen),
                "mean_abs_q_x_odd": float(q.mean()), "standard_deviation_abs_q_x_odd": float(q.std(ddof=1)),
                "median_abs_q_x_odd": float(np.median(q)), "median_gap": float(np.median(gap)),
                "event_count_abs_above_0p5": int(np.sum(q > thresholds[0])),
                "event_count_abs_above_0p75": int(np.sum(q > thresholds[1])),
                "spearman_gap_vs_abs_q": float(correlation.statistic),
                "spearman_gap_vs_abs_q_pvalue": float(correlation.pvalue),
                "mean_excitation_number": float(excitation.mean()),
                "maximum_charge_residual": float(max(row["maximum_charge_residual"] for row in chosen)),
                "maximum_post_qr_gram_residual": float(max(row["maximum_post_qr_gram_residual"] for row in chosen)),
            })
        contrast = {
            "wall": wall,
            "mean_abs_q_x_difference_N24_minus_N20_bootstrap95": _bootstrap_difference(
                by_size[20], by_size[24], seed=seed + wall_index, draws=draws
            ),
        }
        for threshold_index, threshold in enumerate(thresholds):
            contrast[f"event_fraction_difference_N24_minus_N20_above_{threshold}"] = _bootstrap_difference(
                (by_size[20] > threshold).astype(float),
                (by_size[24] > threshold).astype(float),
                seed=seed + 10 + 10 * wall_index + threshold_index,
                draws=draws,
            )
        contrasts.append(contrast)
    summary = {
        "schema": ANALYSIS_SCHEMA,
        "campaign_id": config["campaign_id"],
        "comparison_campaign_id": n20_config["campaign_id"],
        "between_size_sampling": "independent ensembles; unpaired bootstrap",
        "within_state_direction_sampling": "paired CW/CCW direction-odd response",
        "states_per_size": 50, "paths_per_size": 100,
        "statistics": statistics, "contrasts": contrasts, "figures": figures,
        "quantization_is_acceptance_gate": False,
        "increased_event_fraction_is_acceptance_gate": False,
    }
    analysis_root.mkdir(parents=True, exist_ok=True)
    (analysis_root / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    config = n24_campaign.load_config(n24_campaign.DEFAULT_CONFIG)
    analyze(config, n24_campaign.DEFAULT_OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
