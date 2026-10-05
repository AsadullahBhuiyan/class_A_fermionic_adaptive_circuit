#!/usr/bin/env python3
"""Plot sample-wise CCW/CW endpoint wall response for Ny=24 and 28."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUTS = {
    24: ROOT / "results/N20x24_state_projector_pump_s100_v1/analysis/state_projector_pump_endpoints.csv",
    28: ROOT / "results/N20x28_state_projector_pump_s100_v3/analysis/state_projector_pump_endpoints.csv",
}
OUTPUT = ROOT / "results/N20_state_projector_pump_Ny24_36_s100_series_v3/analysis"
THRESHOLD = 0.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, np.ndarray]] = {}
    for wall in ("soft", "hard"):
        selected = sorted(
            (row for row in rows if row["wall"] == wall),
            key=lambda row: int(row["sample_id"]),
        )
        if [int(row["sample_id"]) for row in selected] != list(range(100)):
            raise RuntimeError(f"{path} does not contain exactly samples 0,...,99 for {wall}")
        ccw = np.asarray([float(row["ccw_q_x"]) for row in selected])
        cw = np.asarray([float(row["cw_q_x"]) for row in selected])
        values = np.asarray([float(row["direction_odd_q_x"]) for row in selected])
        if not np.allclose(values, 0.5 * (ccw - cw), atol=1e-13, rtol=0):
            raise RuntimeError(f"{path} has inconsistent direction-odd values for {wall}")
        events = np.asarray([bool(int(row["pump_event"])) for row in selected])
        if not np.array_equal(events, np.abs(values) > THRESHOLD):
            raise RuntimeError(f"{path} has inconsistent pump-event labels for {wall}")
        result[wall] = {"ccw": ccw, "cw": cw, "odd": values}
    if len(rows) != 200:
        raise RuntimeError(f"{path} contains {len(rows)} rows rather than 200")
    return result


def main() -> int:
    data = {ny: _load(path) for ny, path in INPUTS.items()}
    OUTPUT.mkdir(parents=True, exist_ok=True)
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
    bins = {
        "ccw": np.linspace(-0.05, 1.05, 23),
        "cw": np.linspace(-1.05, 0.05, 23),
    }
    colors = {24: "#1f77b4", 28: "#d95f02"}
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.8), sharey=True)
    summary = []
    panel = 0
    for row, wall in enumerate(("soft", "hard")):
        for column, direction in enumerate(("ccw", "cw")):
            ax = axes[row, column]
            for ny in (24, 28):
                values = data[ny][wall][direction]
                events = int(np.sum(np.abs(values) > THRESHOLD))
                ax.hist(
                    values,
                    bins=bins[direction],
                    histtype="stepfilled",
                    alpha=0.18,
                    color=colors[ny],
                )
                ax.hist(
                    values,
                    bins=bins[direction],
                    histtype="step",
                    linewidth=1.35,
                    color=colors[ny],
                    label=rf"$N_y={ny}$ ({events}/100 unit)",
                )
                summary.append(
                    {
                        "Ny": ny,
                        "wall": wall,
                        "direction": direction,
                        "samples": int(values.size),
                        "mean_q_x": float(values.mean()),
                        "sample_standard_deviation": float(values.std(ddof=1)),
                        "unit_response_count": events,
                        "unit_response_fraction": events / values.size,
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                    }
                )
            threshold = THRESHOLD if direction == "ccw" else -THRESHOLD
            ax.axvline(threshold, color="0.35", linestyle="--", linewidth=0.8)
            ax.set_title(f"{wall.capitalize()} wall, {direction.upper()}")
            ax.set_xlabel(r"sample-wise endpoint $q_x$")
            ax.set_xlim((-0.05, 1.05) if direction == "ccw" else (-1.05, 0.05))
            ax.text(-0.14, 1.04, f"({chr(97 + panel)})", transform=ax.transAxes, fontsize=9)
            ax.legend(frameon=False, loc="upper center")
            panel += 1
    axes[0, 0].set_ylabel("trajectory count")
    axes[1, 0].set_ylabel("trajectory count")
    fig.tight_layout()
    pdf = OUTPUT / "ny24_ny28_ccw_cw_wall_qx_histograms.pdf"
    png = OUTPUT / "ny24_ny28_ccw_cw_wall_qx_histograms.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "schema": "ny24_ny28_samplewise_ccw_cw_wall_qx_histograms_v1",
        "definition": "raw q_x endpoint shown separately for CCW and CW paths",
        "pump_event_threshold": THRESHOLD,
        "inputs": [
            {"Ny": ny, "path": str(path), "sha256": _sha256(path)}
            for ny, path in INPUTS.items()
        ],
        "statistics": summary,
        "figure_pdf": str(pdf),
        "figure_png": str(png),
        "script_sha256": _sha256(Path(__file__).resolve()),
    }
    summary_path = OUTPUT / "ny24_ny28_ccw_cw_wall_qx_histograms.json"
    summary_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
