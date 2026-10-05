#!/usr/bin/env python3
"""Create the compact diagnostic figure for the frozen-record flux pilot."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_rows(path: Path) -> list[dict[str, Any]]:
    numeric = {
        "sigma": int,
        "twist_index": int,
        "phi": float,
        "sweep_fraction": float,
        "N_left": float,
        "N_right": float,
        "N_total": float,
        "N_initial_total": float,
        "net_injected_charge": int,
        "delta_N_left": float,
        "delta_N_right": float,
        "q_wall": float,
        "charge_continuity_residual": float,
        "regional_balance_residual": float,
        "branch_log_probability": float,
        "minimum_selected_probability": float,
        "elapsed_seconds": float,
    }
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for key, caster in numeric.items():
                row[key] = caster(row[key])
            rows.append(row)
    return rows


def select(rows: list[dict[str, Any]], arm: str, direction: str) -> list[dict[str, Any]]:
    return sorted(
        (row for row in rows if row["arm"] == arm and row["direction"] == direction),
        key=lambda row: row["twist_index"],
    )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
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
            "savefig.transparent": False,
        }
    )


def direction_style(direction: str) -> dict[str, Any]:
    if direction == "ccw":
        return {"linestyle": "-", "marker": "o", "label": "CCW / increasing"}
    return {"linestyle": "--", "marker": "s", "label": "CW / decreasing"}


def plot_region_response(ax: plt.Axes, rows: list[dict[str, Any]], arm: str) -> None:
    colors = {"delta_N_left": "#d62728", "delta_N_right": "#1f77b4"}
    labels = {"delta_N_left": r"$\Delta N_L$", "delta_N_right": r"$\Delta N_R$"}
    for direction in ("ccw", "cw"):
        subset = select(rows, arm, direction)
        x = np.asarray([row["sweep_fraction"] for row in subset])
        for field in ("delta_N_left", "delta_N_right"):
            style = direction_style(direction)
            ax.plot(
                x,
                [row[field] for row in subset],
                color=colors[field],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=3,
                linewidth=1.0,
                markerfacecolor="white",
                label=f"{labels[field]}, {direction.upper()}",
            )
    ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"signed sweep $(\phi-\phi_0)/(2\pi)$")
    ax.set_ylabel("endpoint charge change")
    ax.set_title(f"{arm.capitalize()} wall")
    ax.legend(frameon=False, ncol=2, handlelength=2.1, columnspacing=0.8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    args = parser.parse_args()
    campaign_dir = args.campaign_dir.resolve()
    manifest_path = campaign_dir / "manifest.json"
    csv_path = campaign_dir / "charge_vs_phi.csv"
    if not manifest_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError("completed manifest and charge_vs_phi.csv are required")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "complete":
        raise RuntimeError("campaign manifest is not complete")
    rows = load_rows(csv_path)
    expected = int(manifest["task_count"])
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} aggregate rows, found {len(rows)}")

    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.1))
    plot_region_response(axes[0, 0], rows, "soft")
    plot_region_response(axes[0, 1], rows, "hard")

    ax = axes[1, 0]
    arm_color = {"soft": "#2ca02c", "hard": "#1f77b4"}
    for arm in ("soft", "hard"):
        for direction in ("ccw", "cw"):
            subset = select(rows, arm, direction)
            style = direction_style(direction)
            ax.plot(
                [row["sweep_fraction"] for row in subset],
                [row["q_wall"] for row in subset],
                color=arm_color[arm],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=3,
                linewidth=1.0,
                markerfacecolor="white",
                label=f"{arm}, {direction.upper()}",
            )
    ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"signed sweep $(\phi-\phi_0)/(2\pi)$")
    ax.set_ylabel(r"$q_{\rm wall}=(\Delta N_R-\Delta N_L)/2$")
    ax.set_title("Conditional wall-to-wall response")
    ax.legend(frameon=False, ncol=2, handlelength=2.1, columnspacing=0.8)

    ax = axes[1, 1]
    for arm in ("soft", "hard"):
        for direction in ("ccw", "cw"):
            subset = select(rows, arm, direction)
            style = direction_style(direction)
            residual = np.maximum(
                np.abs([row["regional_balance_residual"] for row in subset]), 1e-18
            )
            ax.semilogy(
                [row["sweep_fraction"] for row in subset],
                residual,
                color=arm_color[arm],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=3,
                linewidth=1.0,
                markerfacecolor="white",
                label=f"{arm}, {direction.upper()}",
            )
    tolerance = float(manifest["configuration"]["acceptance"]["regional_balance_tolerance"])
    ax.axhline(tolerance, color="0.2", linestyle=":", linewidth=0.9, label="tolerance")
    ax.set_xlabel(r"signed sweep $(\phi-\phi_0)/(2\pi)$")
    ax.set_ylabel(r"$|\Delta N_L+\Delta N_R|$")
    ax.set_title("Fixed-injection balance")
    ax.legend(frameon=False, ncol=2, handlelength=2.1, columnspacing=0.8)

    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=9, va="bottom", ha="left")
    fig.tight_layout(pad=0.7, w_pad=1.0, h_pad=1.0)
    figure_dir = campaign_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figure_dir / "frozen_record_flux_charge.pdf"
    png_path = figure_dir / "frozen_record_flux_charge.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "schema": "frozen_record_flux_charge_analysis_v1",
        "campaign_id": manifest["campaign_id"],
        "rows": len(rows),
        "maximum_absolute_q_wall": {
            arm: max(abs(row["q_wall"]) for row in rows if row["arm"] == arm)
            for arm in ("soft", "hard")
        },
        "maximum_charge_continuity_residual": max(
            abs(row["charge_continuity_residual"]) for row in rows
        ),
        "maximum_regional_balance_residual": max(
            abs(row["regional_balance_residual"]) for row in rows
        ),
        "large_gauge_closure_frobenius_per_dimension": manifest[
            "large_gauge_closure_frobenius_per_dimension"
        ],
        "minimum_selected_probability": min(
            row["minimum_selected_probability"] for row in rows
        ),
        "figures": {"pdf": str(pdf_path), "png": str(png_path)},
        "interpretation": (
            "Static fixed-record response. The closed 2*pi endpoint is not a quantized "
            "online monitored pump."
        ),
    }
    atomic_json(campaign_dir / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
