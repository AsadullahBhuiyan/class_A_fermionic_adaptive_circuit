#!/usr/bin/env python3
"""Compare raw endpoint and time-normalized bundle-13 gap definitions."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import plot_endpoint_lyapunov_gap as original


BUNDLE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = BUNDLE_ROOT / "analysis_outputs" / "endpoint_gap_definitions_v1"
NY_VALUES = original.NY_VALUES


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sem(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(values.size))


def derive_gap_definitions(
    source_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_rows: list[dict[str, Any]] = []
    for row in source_rows:
        ny = int(row["Ny"])
        total_time = int(row["endpoint_cycle"])
        if total_time != 4 * ny:
            raise RuntimeError(f"Ny={ny}: endpoint is not T=4Ny")
        lyapunov_gap = float(row["endpoint_lyapunov_half_gap"])
        raw_logit_gap = 2.0 * total_time * lyapunov_gap
        centered_gap = math.tanh(0.5 * raw_logit_gap)
        sample_rows.append(
            {
                "Ny": ny,
                "sample_index": int(row["sample_index"]),
                "endpoint_cycle_T": total_time,
                "centered_occupation_gap_g_a": centered_gap,
                "raw_logit_gap_g_epsilon": raw_logit_gap,
                "lyapunov_gap_Delta_lambda": lyapunov_gap,
                "Ny_times_lyapunov_gap": ny * lyapunov_gap,
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for ny in NY_VALUES:
        selected = [row for row in sample_rows if int(row["Ny"]) == ny]
        if len(selected) != 100:
            raise RuntimeError(f"Ny={ny}: expected 100 samples")
        sample_ids = sorted(int(row["sample_index"]) for row in selected)
        if sample_ids != list(range(100)):
            raise RuntimeError(f"Ny={ny}: sample coverage mismatch")
        centered = np.asarray(
            [row["centered_occupation_gap_g_a"] for row in selected], dtype=np.float64
        )
        raw = np.asarray(
            [row["raw_logit_gap_g_epsilon"] for row in selected], dtype=np.float64
        )
        lyapunov = np.asarray(
            [row["lyapunov_gap_Delta_lambda"] for row in selected], dtype=np.float64
        )
        rescaled = np.asarray(
            [row["Ny_times_lyapunov_gap"] for row in selected], dtype=np.float64
        )
        centered_mean, centered_sem = mean_sem(centered)
        raw_mean, raw_sem = mean_sem(raw)
        lyapunov_mean, lyapunov_sem = mean_sem(lyapunov)
        rescaled_mean, rescaled_sem = mean_sem(rescaled)
        summary_rows.append(
            {
                "Ny": ny,
                "samples": 100,
                "endpoint_cycle_T": 4 * ny,
                "mean_centered_occupation_gap_g_a": centered_mean,
                "sem_centered_occupation_gap_g_a": centered_sem,
                "mean_raw_logit_gap_g_epsilon": raw_mean,
                "sem_raw_logit_gap_g_epsilon": raw_sem,
                "mean_lyapunov_gap_Delta_lambda": lyapunov_mean,
                "sem_lyapunov_gap_Delta_lambda": lyapunov_sem,
                "mean_Ny_times_lyapunov_gap": rescaled_mean,
                "sem_Ny_times_lyapunov_gap": rescaled_sem,
            }
        )
    return sample_rows, summary_rows


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
            "legend.fontsize": 6.7,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
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
    axis.text(-0.18, 1.03, label, transform=axis.transAxes, fontweight="bold", va="bottom")


def errorbar(axis: Any, x: np.ndarray, y: np.ndarray, error: np.ndarray, **kwargs: Any) -> None:
    marker = kwargs.pop("marker", "o")
    axis.errorbar(
        x,
        y,
        yerr=error,
        marker=marker,
        markersize=3.8,
        markerfacecolor="white",
        markeredgewidth=0.9,
        linewidth=1.05,
        capsize=2.0,
        **kwargs,
    )


def make_figure(summary_rows: list[dict[str, Any]]) -> None:
    configure_plotting()
    ny = np.asarray([row["Ny"] for row in summary_rows], dtype=np.float64)
    figure, axes = plt.subplots(1, 3, figsize=(7.05, 2.55), constrained_layout=True)

    centered = np.asarray([row["mean_centered_occupation_gap_g_a"] for row in summary_rows])
    centered_sem = np.asarray([row["sem_centered_occupation_gap_g_a"] for row in summary_rows])
    errorbar(axes[0], ny, centered, centered_sem, color="#2b6cb0")
    axes[0].set_ylabel(r"$g_a=\min_j|2\nu_j-1|$")
    axes[0].set_ylim(0.9, 1.005)
    panel_label(axes[0], "(a)")

    raw = np.asarray([row["mean_raw_logit_gap_g_epsilon"] for row in summary_rows])
    raw_sem = np.asarray([row["sem_raw_logit_gap_g_epsilon"] for row in summary_rows])
    errorbar(axes[1], ny, raw, raw_sem, color="#2f855a")
    axes[1].set_ylabel(r"$g_\epsilon=\min_j|\log[(1-\nu_j)/\nu_j]|$")
    panel_label(axes[1], "(b)")

    lyapunov = np.asarray([row["mean_lyapunov_gap_Delta_lambda"] for row in summary_rows])
    lyapunov_sem = np.asarray([row["sem_lyapunov_gap_Delta_lambda"] for row in summary_rows])
    rescaled = np.asarray([row["mean_Ny_times_lyapunov_gap"] for row in summary_rows])
    rescaled_sem = np.asarray([row["sem_Ny_times_lyapunov_gap"] for row in summary_rows])
    errorbar(
        axes[2], ny, lyapunov, lyapunov_sem,
        color="#805ad5", label=r"$\Delta_\lambda$",
    )
    axes[2].set_ylabel(r"$\Delta_\lambda=g_\epsilon/(2T)$", color="#805ad5")
    axes[2].tick_params(axis="y", colors="#805ad5")
    twin = axes[2].twinx()
    errorbar(
        twin, ny, rescaled, rescaled_sem,
        color="#dd6b20", marker="s", label=r"$N_y\Delta_\lambda$",
    )
    twin.set_ylabel(r"$N_y\Delta_\lambda$", color="#dd6b20")
    twin.tick_params(axis="y", colors="#dd6b20")
    axes[2].set_ylim(bottom=0.0)
    twin.set_ylim(0.9, 1.25)
    handles = [
        Line2D([0], [0], color="#805ad5", marker="o", markerfacecolor="white", label=r"$\Delta_\lambda$"),
        Line2D([0], [0], color="#dd6b20", marker="s", markerfacecolor="white", label=r"$N_y\Delta_\lambda$"),
    ]
    axes[2].legend(handles=handles, frameon=False, loc="center right")
    panel_label(axes[2], "(c)")

    for axis in axes:
        axis.set_xlabel(r"circumference $N_y$")
        axis.set_xticks(ny.astype(int))
        axis.tick_params(axis="x", labelrotation=45)
        axis.set_xlim(17.5, 62.5)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_ROOT / "endpoint_gap_definitions_vs_Ny.pdf")
    figure.savefig(OUTPUT_ROOT / "endpoint_gap_definitions_vs_Ny.png", dpi=300)
    plt.close(figure)


def main() -> int:
    source_rows, _, diagnostics = original.load_endpoint_gaps()
    sample_rows, summary_rows = derive_gap_definitions(source_rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_ROOT / "endpoint_gap_definitions_samples.csv", sample_rows)
    write_csv(OUTPUT_ROOT / "endpoint_gap_definitions_summary.csv", summary_rows)
    summary = {
        "schema": "bundle13_endpoint_gap_definitions_v1",
        "campaign": original.analyze_campaign.REVISION,
        "sizes": list(NY_VALUES),
        "samples_per_size": 100,
        "trajectory_count": 700,
        "endpoint": "T=4Ny",
        "definitions": {
            "g_a": "min_j abs(2*nu_j-1)",
            "g_epsilon": "min_j abs(log((1-nu_j)/nu_j)) = 2*atanh(g_a)",
            "Delta_lambda": "g_epsilon/(2*T)",
            "Ny_Delta_lambda": "Ny*Delta_lambda = g_epsilon/8 because T=4Ny",
        },
        "uncertainty": "sample-wise SEM; std(ddof=1)/sqrt(100)",
        "summary_rows": summary_rows,
        "source_diagnostics": diagnostics,
    }
    (OUTPUT_ROOT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_figure(summary_rows)
    print(json.dumps({"output_root": str(OUTPUT_ROOT), "summary_rows": summary_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
