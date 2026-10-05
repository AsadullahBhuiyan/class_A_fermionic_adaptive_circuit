"""Original loading/style helpers with bundled CSV input."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import math, csv
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
SOURCE_CSV=Path(__file__).resolve().parents[1]/"data/central_charge/ceff_vs_cycle_Nx20_Ny30_40_50_S100.csv"
NY_VALUES = (30, 40, 50)

STYLES = {
    30: {"color": "#D92725", "marker": "^", "linestyle": ":"},
    40: {"color": "#2CA02C", "marker": "s", "linestyle": "--"},
    50: {"color": "#1F77B4", "marker": "o", "linestyle": "-"},
}

def load_rows() -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with SOURCE_CSV.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            ny = int(raw["Ny"])
            if ny not in NY_VALUES:
                continue
            ceff = float(raw["c_eff"])
            rows.append(
                {
                    "Nx": int(raw["Nx"]),
                    "Ny": ny,
                    "samples": int(raw["samples"]),
                    "cycle": int(raw["cycle"]),
                    "normalized_cycle": float(raw["normalized_cycle"]),
                    "c_eff": ceff,
                    "abs_c_eff_minus_one": abs(ceff - 1.0),
                    "c_eff_fit_error": float(raw["c_eff_error"]),
                }
            )
    for ny in NY_VALUES:
        selected = [row for row in rows if row["Ny"] == ny]
        cycles = np.asarray([row["cycle"] for row in selected])
        if not np.array_equal(cycles, np.arange(1, 2 * ny + 1)):
            raise ValueError(f"Ny={ny}: incomplete cycle sequence")
        ordinate = np.asarray([row["abs_c_eff_minus_one"] for row in selected])
        if not np.isfinite(ordinate).all() or np.any(ordinate <= 0):
            raise ValueError(f"Ny={ny}: invalid log-scale ordinate")
    return rows

def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "mathtext.fontset": "cm",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
