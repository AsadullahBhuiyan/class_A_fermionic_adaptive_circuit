#!/usr/bin/env python3
"""Plot raw sample-resolved endpoint q_x for each wall and ramp direction."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
ANALYSIS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "N20x24_parent_schrodinger_rk4_s50_tau1e4_v1"
    / "analysis"
)
SOURCE_CSV = ANALYSIS_ROOT / "parent_schrodinger_rk4_samplewise.csv"
OUTPUT_CSV = ANALYSIS_ROOT / "parent_schrodinger_rk4_raw_endpoint_qx_samplewise.csv"
OUTPUT_PDF = ANALYSIS_ROOT / "figures" / "parent_schrodinger_rk4_raw_endpoint_qx_distribution.pdf"
OUTPUT_PNG = ANALYSIS_ROOT / "figures" / "parent_schrodinger_rk4_raw_endpoint_qx_distribution.png"
OUTPUT_JSON = ANALYSIS_ROOT / "raw_endpoint_qx_distribution_summary.json"


def load_rows() -> list[dict[str, object]]:
    with SOURCE_CSV.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    for row in source:
        for direction, key in (("ccw", "rk4_ccw_q_x"), ("cw", "rk4_cw_q_x")):
            rows.append(
                {
                    "wall": row["wall"],
                    "direction": direction,
                    "sample_id": int(row["sample_id"]),
                    "q_x": float(row[key]),
                }
            )
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 raw endpoint observations, found {len(rows)}")
    return rows


def write_rows(rows: list[dict[str, object]]) -> None:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("wall", "direction", "sample_id", "q_x"))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    groups = []
    for wall in ("soft", "hard"):
        for direction in ("ccw", "cw"):
            values = np.asarray(
                [row["q_x"] for row in rows if row["wall"] == wall and row["direction"] == direction],
                dtype=float,
            )
            if values.size != 25:
                raise RuntimeError(f"expected 25 values for {wall}/{direction}, found {values.size}")
            groups.append(
                {
                    "wall": wall,
                    "direction": direction,
                    "samples": int(values.size),
                    "mean": float(np.mean(values)),
                    "standard_deviation": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "count_abs_above_0p5": int(np.count_nonzero(np.abs(values) > 0.5)),
                    "count_abs_above_0p75": int(np.count_nonzero(np.abs(values) > 0.75)),
                }
            )
    return {
        "schema": "parent_schrodinger_rk4_raw_endpoint_qx_distribution_v1",
        "campaign_id": "N20x24_parent_schrodinger_rk4_s50_tau1e4_v1",
        "observable": "raw q_x(2 pi) for each direction; no CW/CCW antisymmetrization",
        "independent_sampling_unit": "saved monitored endpoint trajectory",
        "groups": groups,
    }


def plot(rows: list[dict[str, object]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.6), sharex=True, sharey=True)
    bins = np.linspace(-1.05, 1.05, 22)
    colors = {"soft": "#0072B2", "hard": "#D55E00"}
    letters = ("(a)", "(b)", "(c)", "(d)")
    combinations = (("soft", "ccw"), ("soft", "cw"), ("hard", "ccw"), ("hard", "cw"))
    for axis, letter, (wall, direction) in zip(axes.flat, letters, combinations, strict=True):
        selected = [row for row in rows if row["wall"] == wall and row["direction"] == direction]
        selected.sort(key=lambda row: int(row["sample_id"]))
        values = np.asarray([row["q_x"] for row in selected], dtype=float)
        axis.hist(values, bins=bins, color=colors[wall], alpha=0.72, edgecolor="black", linewidth=0.55)
        axis.scatter(values, np.full(values.size, 0.12), marker="|", s=42, linewidths=0.9, color="black", zorder=3)
        axis.axvline(0.0, color="0.35", linestyle="--", linewidth=0.8)
        axis.axvline(float(np.mean(values)), color=colors[wall], linestyle="-", linewidth=1.4)
        axis.set_title(f"{wall.capitalize()} wall, {direction.upper()}; $S=25$")
        axis.text(-0.12, 1.05, letter, transform=axis.transAxes, fontsize=9, va="bottom")
        axis.tick_params(direction="in", top=True, right=True)
        axis.set_xlim(-1.05, 1.05)
        axis.set_ylim(0, 19)
    for axis in axes[:, 0]:
        axis.set_ylabel("trajectory count")
    for axis in axes[-1, :]:
        axis.set_xlabel(r"raw endpoint $q_x(2\pi)$")
    fig.tight_layout(pad=0.7, w_pad=0.8, h_pad=0.9)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    rows = load_rows()
    write_rows(rows)
    summary = summarize(rows)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot(rows)
    print(json.dumps(summary, indent=2))
    print(f"[saved] {OUTPUT_CSV}")
    print(f"[saved] {OUTPUT_PDF}")
    print(f"[saved] {OUTPUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
