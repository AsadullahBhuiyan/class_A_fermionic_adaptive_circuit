#!/usr/bin/env python3
"""Combine verified Ny=24,28,30,32,34,36 S100 projector-pump analyses."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
NY_VALUES = (24, 28, 30, 32, 34, 36)
SERIES_ROOT = PROJECT_ROOT / "results" / "N20_state_projector_pump_Ny24_36_s100_series_v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _analysis_root(ny: int) -> Path:
    name = (
        "N20x24_state_projector_pump_s100_v1"
        if ny == 24
        else f"N20x{ny}_state_projector_pump_s100_v3"
    )
    return PROJECT_ROOT / "results" / name / "analysis"


def load_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for ny in NY_VALUES:
        root = _analysis_root(ny)
        summary_path = root / "analysis_summary.json"
        aggregate_path = root / "state_projector_pump_aggregate.npz"
        if not summary_path.is_file() or not aggregate_path.is_file():
            raise RuntimeError(f"Ny={ny} analysis is incomplete")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_campaign = (
            "N20x24_state_projector_pump_s100_v1"
            if ny == 24
            else f"N20x{ny}_state_projector_pump_s100_v3"
        )
        if summary.get("campaign_id") != expected_campaign:
            raise RuntimeError(f"Ny={ny} campaign identity mismatch")
        if int(summary.get("samples_per_wall", -1)) != 100:
            raise RuntimeError(f"Ny={ny} sample count mismatch")
        if len(summary.get("endpoint_statistics", [])) != 2:
            raise RuntimeError(f"Ny={ny} summary lacks both walls")
        rows.extend({"Ny": ny, **item} for item in summary["endpoint_statistics"])
        inputs.append(
            {
                "Ny": ny,
                "summary": str(summary_path),
                "summary_sha256": _sha256(summary_path),
                "aggregate": str(aggregate_path),
                "aggregate_sha256": _sha256(aggregate_path),
                "config_hash": summary["config_hash"],
                "source_hashes": summary["source_hashes"],
            }
        )
    return rows, inputs


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
            "savefig.dpi": 300,
        }
    )


def analyze() -> dict[str, Any]:
    rows, inputs = load_rows()
    analysis_root = SERIES_ROOT / "analysis"
    figure_root = analysis_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_root / "state_projector_pump_size_series.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), sharex=True)
    styles = {
        "soft": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
        "hard": {"color": "#d62728", "marker": "s", "linestyle": "--"},
    }
    for wall in ("soft", "hard"):
        selected = sorted((row for row in rows if row["wall"] == wall), key=lambda row: row["Ny"])
        x = np.asarray([row["Ny"] for row in selected])
        mean = np.asarray([row["direction_odd_mean"] for row in selected])
        mean_low = np.asarray([row["direction_odd_ci_low"] for row in selected])
        mean_high = np.asarray([row["direction_odd_ci_high"] for row in selected])
        rate = np.asarray([row["pump_event_fraction"] for row in selected])
        rate_low = np.asarray([row["pump_event_cp95_low"] for row in selected])
        rate_high = np.asarray([row["pump_event_cp95_high"] for row in selected])
        style = styles[wall]
        axes[0].errorbar(
            x,
            mean,
            yerr=np.vstack((mean - mean_low, mean_high - mean)),
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.1,
            markersize=3.5,
            capsize=2,
            label=wall.capitalize(),
        )
        axes[1].errorbar(
            x,
            rate,
            yerr=np.vstack((rate - rate_low, rate_high - rate)),
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.1,
            markersize=3.5,
            capsize=2,
            label=wall.capitalize(),
        )
    axes[0].set_ylabel(r"endpoint $\langle q_x^{\rm odd}\rangle$")
    axes[1].set_ylabel(r"fraction with $|q_x^{\rm odd}|>0.5$")
    for index, ax in enumerate(axes):
        ax.set_xlabel(r"circumference $N_y$")
        ax.set_xticks(NY_VALUES)
        ax.set_ylim(0, 1)
        ax.axhline(0, color="0.55", linestyle=":", linewidth=0.7)
        ax.text(-0.14, 1.04, f"({chr(97 + index)})", transform=ax.transAxes, fontsize=9)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    pdf = figure_root / "state_projector_pump_size_series.pdf"
    png = figure_root / "state_projector_pump_size_series.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "schema": "state_projector_pump_size_series_analysis_v2",
        "Nx": 20,
        "Ny_values": list(NY_VALUES),
        "samples_per_wall_per_size": 100,
        "independent_sampling_unit": "wall-specific monitored trajectory",
        "input_analyses": inputs,
        "rows": rows,
        "csv": str(csv_path),
        "figures": {"pdf": str(pdf), "png": str(png)},
        "analysis_source_sha256": _sha256(Path(__file__).resolve()),
    }
    summary_path = analysis_root / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    analyze()
