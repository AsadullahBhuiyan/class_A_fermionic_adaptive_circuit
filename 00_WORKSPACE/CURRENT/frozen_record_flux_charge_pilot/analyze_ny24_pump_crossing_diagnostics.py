#!/usr/bin/env python3
"""Test whether Ny=24 pump sectors are controlled by the phi=pi wall crossing."""

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
from scipy.stats import mannwhitneyu, rankdata, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_state_projector_pump_s100 as campaign  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "campaign_config.state_projector_pump_n20x24_s100_v1.json"
INPUT_ROOT = PROJECT_ROOT / "results" / "N20x24_state_projector_pump_s100_v1"
OUTPUT_ROOT = INPUT_ROOT / "analysis" / "crossing_diagnostics_v1"
WALLS = ("soft", "hard")
DIRECTIONS = ("ccw", "cw")


def _bootstrap_median_difference(
    first: np.ndarray, second: np.ndarray, *, seed: int, draws: int = 20000
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    lhs = np.asarray(first, dtype=float)
    rhs = np.asarray(second, dtype=float)
    difference = np.median(
        lhs[rng.integers(0, lhs.size, size=(draws, lhs.size))], axis=1
    ) - np.median(
        rhs[rng.integers(0, rhs.size, size=(draws, rhs.size))], axis=1
    )
    return (
        float(np.median(lhs) - np.median(rhs)),
        float(np.quantile(difference, 0.025)),
        float(np.quantile(difference, 0.975)),
    )


def _small_gap_auc(gap: np.ndarray, event: np.ndarray) -> float:
    scores = rankdata(-np.asarray(gap, dtype=float))
    positive = np.asarray(event, dtype=bool)
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    return float(
        (scores[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def _best_gap_threshold(gap: np.ndarray, event: np.ndarray) -> dict[str, float]:
    best: tuple[float, float, float, float, float] | None = None
    for threshold in np.unique(gap):
        prediction = gap <= threshold
        sensitivity = float(np.mean(prediction[event]))
        specificity = float(np.mean(~prediction[~event]))
        youden = sensitivity + specificity - 1.0
        accuracy = float(np.mean(prediction == event))
        candidate = (youden, float(threshold), sensitivity, specificity, accuracy)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return {
        "descriptive_threshold": best[1],
        "sensitivity": best[2],
        "specificity": best[3],
        "in_sample_accuracy": best[4],
        "threshold_is_preregistered": False,
    }


def _load() -> list[dict[str, Any]]:
    config = campaign.load_config(CONFIG_PATH)
    campaign.validate_config(config)
    status = campaign.inventory(config, INPUT_ROOT)
    if not all(row[0] for row in status["pumps"].values()):
        raise RuntimeError("crossing analysis requires all 400 verified Ny=24 pump paths")
    rows = []
    for wall in WALLS:
        for sample_id in range(100):
            direction_rows = {}
            for direction in DIRECTIONS:
                task = {
                    "stage": "pump",
                    "task_id": f"pump_{wall}_{direction}_sample_{sample_id:03d}",
                    "wall": wall,
                    "direction": direction,
                    "sigma": int(config["projector_pump"]["directions"][direction]),
                    "sample_id": sample_id,
                    "burnin_task_id": f"burnin_{wall}_sample_{sample_id:03d}",
                }
                path, _ = campaign.result_paths(INPUT_ROOT, task)
                with np.load(path, allow_pickle=False) as saved:
                    phi = np.asarray(saved["phi"], dtype=float)
                    gap = np.asarray(saved["instantaneous_rank_gap"], dtype=float)
                    overlap = np.asarray(saved["principal_overlap"], dtype=float)
                    weight = np.asarray(saved["selected_weight_floor"], dtype=float)
                    q_endpoint = float(saved["continued_q_x"][-1])
                gap_index = int(np.argmin(gap))
                direction_rows[direction] = {
                    "q_endpoint": q_endpoint,
                    "minimum_gap": float(gap[gap_index]),
                    "minimum_gap_phi_over_pi": float(abs(phi[gap_index]) / np.pi),
                    "minimum_principal_overlap": float(np.min(overlap)),
                    "minimum_selected_weight": float(np.min(weight)),
                }
            q_odd = 0.5 * (
                direction_rows["ccw"]["q_endpoint"]
                - direction_rows["cw"]["q_endpoint"]
            )
            q_even = 0.5 * (
                direction_rows["ccw"]["q_endpoint"]
                + direction_rows["cw"]["q_endpoint"]
            )
            rows.append(
                {
                    "wall": wall,
                    "sample_id": sample_id,
                    "ccw_q_x": direction_rows["ccw"]["q_endpoint"],
                    "cw_q_x": direction_rows["cw"]["q_endpoint"],
                    "direction_odd_q_x": q_odd,
                    "direction_even_q_x": q_even,
                    "pump_event": int(abs(q_odd) > 0.5),
                    "minimum_instantaneous_rank_gap": min(
                        direction_rows[direction]["minimum_gap"] for direction in DIRECTIONS
                    ),
                    "mean_minimum_gap_phi_over_pi": float(
                        np.mean(
                            [
                                direction_rows[direction]["minimum_gap_phi_over_pi"]
                                for direction in DIRECTIONS
                            ]
                        )
                    ),
                    "minimum_principal_overlap": min(
                        direction_rows[direction]["minimum_principal_overlap"]
                        for direction in DIRECTIONS
                    ),
                    "minimum_selected_weight": min(
                        direction_rows[direction]["minimum_selected_weight"]
                        for direction in DIRECTIONS
                    ),
                }
            )
    return rows


def _statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for wall_index, wall in enumerate(WALLS):
        selected = [row for row in rows if row["wall"] == wall]
        event = np.asarray([row["pump_event"] for row in selected], dtype=bool)
        gap = np.asarray(
            [row["minimum_instantaneous_rank_gap"] for row in selected], dtype=float
        )
        response = np.abs(
            np.asarray([row["direction_odd_q_x"] for row in selected], dtype=float)
        )
        pumped, closing = gap[event], gap[~event]
        difference, low, high = _bootstrap_median_difference(
            pumped, closing, seed=2026090505 + wall_index
        )
        test = mannwhitneyu(pumped, closing, alternative="two-sided")
        locations = np.asarray(
            [row["mean_minimum_gap_phi_over_pi"] for row in selected], dtype=float
        )
        output.append(
            {
                "wall": wall,
                "samples": len(selected),
                "pump_events": int(event.sum()),
                "closing_events": int((~event).sum()),
                "pumped_gap_median": float(np.median(pumped)),
                "pumped_gap_interquartile": [
                    float(np.quantile(pumped, 0.25)), float(np.quantile(pumped, 0.75))
                ],
                "closing_gap_median": float(np.median(closing)),
                "closing_gap_interquartile": [
                    float(np.quantile(closing, 0.25)), float(np.quantile(closing, 0.75))
                ],
                "pumped_minus_closing_median_gap": difference,
                "pumped_minus_closing_median_gap_bootstrap95": [low, high],
                "mann_whitney_u": float(test.statistic),
                "mann_whitney_p": float(test.pvalue),
                "small_gap_predicts_pump_auc": _small_gap_auc(gap, event),
                "spearman_gap_vs_abs_q_x": float(spearmanr(gap, response).statistic),
                "minimum_gap_phi_over_pi_range": [
                    float(np.min(locations)), float(np.max(locations))
                ],
                "maximum_abs_direction_even_q_x": float(
                    np.max(np.abs([row["direction_even_q_x"] for row in selected]))
                ),
                **_best_gap_threshold(gap, event),
            }
        )
    return output


def _plot(rows: list[dict[str, Any]]) -> dict[str, str]:
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
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.0), constrained_layout=True)
    colors = {"soft": "#1675b9", "hard": "#c4473a"}
    for column, wall in enumerate(WALLS):
        selected = [row for row in rows if row["wall"] == wall]
        event = np.asarray([row["pump_event"] for row in selected], dtype=bool)
        gap = np.asarray(
            [row["minimum_instantaneous_rank_gap"] for row in selected], dtype=float
        )
        response = np.abs(
            np.asarray([row["direction_odd_q_x"] for row in selected], dtype=float)
        )
        ax = axes[0, column]
        for mask, label, linestyle in (
            (event, "pumped", "-"), (~event, "closing", "--")
        ):
            values = np.sort(gap[mask])
            ax.step(
                values,
                np.arange(1, values.size + 1) / values.size,
                where="post",
                color=colors[wall],
                linestyle=linestyle,
                linewidth=1.2,
                label=label,
            )
        ax.set_xscale("log")
        ax.set_xlabel(r"Minimum instantaneous rank gap")
        ax.set_title(f"{wall.capitalize()} wall")
        ax.legend(frameon=False)
        axes[1, column].scatter(
            gap,
            response,
            s=18,
            facecolors="none",
            edgecolors=colors[wall],
            linewidths=0.8,
            alpha=0.8,
        )
        axes[1, column].set_xscale("log")
        axes[1, column].axhline(0.5, color="0.25", linestyle="--", linewidth=0.9)
        axes[1, column].set_xlabel(r"Minimum instantaneous rank gap")
    axes[0, 0].set_ylabel("Empirical cumulative fraction")
    axes[1, 0].set_ylabel(r"Pump-response magnitude $|q_x^{\rm odd}|$")
    for index, axis in enumerate(axes.flat):
        axis.text(-0.13, 1.04, f"({chr(ord('a') + index)})", transform=axis.transAxes)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pdf = OUTPUT_ROOT / "ny24_pump_sector_vs_pi_crossing_gap.pdf"
    png = OUTPUT_ROOT / "ny24_pump_sector_vs_pi_crossing_gap.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}


def main() -> int:
    rows = _load()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "ny24_samplewise_crossing_diagnostics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "ny24_state_projector_pump_crossing_diagnostics_v1",
        "hypothesis": "pump/closure sector is controlled by the near-degenerate wall crossing at phi=pi",
        "samples_per_wall": 100,
        "flux_grid_intervals": 64,
        "samplewise_csv": str(csv_path),
        "figure": _plot(rows),
        "statistics": _statistics(rows),
        "interpretation_guard": (
            "Association with the pi gap supports a crossing mechanism but does not by itself "
            "distinguish physical avoided-crossing structure from finite-grid branch matching."
        ),
    }
    (OUTPUT_ROOT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
