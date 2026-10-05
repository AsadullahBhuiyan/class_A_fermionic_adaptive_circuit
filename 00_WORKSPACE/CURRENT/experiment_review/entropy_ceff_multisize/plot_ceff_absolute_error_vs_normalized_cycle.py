#!/usr/bin/env python3
"""Plot log-scale convergence of the mean-entropy central-charge estimator."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE_CSV = HERE / "data/ceff_vs_cycle_Nx20_Ny30_40_50_S100.csv"
OUTPUT_CSV = (
    HERE / "data/abs_ceff_minus_one_vs_normalized_cycle_Nx20_Ny30_40_50_S100.csv"
)
PDF_PATH = (
    HERE / "figures/abs_ceff_minus_one_vs_normalized_cycle_Nx20_Ny30_40_50_S100.pdf"
)
PNG_PATH = PDF_PATH.with_suffix(".png")
MANIFEST_PATH = HERE / "abs_ceff_minus_one_manifest.json"
NY_VALUES = (30, 40, 50)
STYLES = {
    30: {"color": "#D92725", "marker": "^", "linestyle": ":"},
    40: {"color": "#2CA02C", "marker": "s", "linestyle": "--"},
    50: {"color": "#1F77B4", "marker": "o", "linestyle": "-"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def main() -> None:
    rows = load_rows()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(3.375, 2.65))
    for ny in NY_VALUES:
        selected = [row for row in rows if row["Ny"] == ny]
        style = STYLES[ny]
        ax.plot(
            [row["normalized_cycle"] for row in selected],
            [row["abs_c_eff_minus_one"] for row in selected],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor="white",
            markeredgewidth=0.7,
            markevery=5,
            label=rf"$N_y={ny}$",
        )
    ax.set_yscale("log")
    ax.set_xlim(0.0, 2.02)
    ax.set_xlabel(r"normalized cycle $t/N_y$")
    ax.set_ylabel(r"$|c_{\mathrm{eff}}(t)-1|$")
    ax.set_title(r"$N_x=20$, $S=100$")
    ax.legend(loc="upper right", handletextpad=0.35)
    ax.tick_params(which="both", top=True, right=True)
    fig.subplots_adjust(left=0.205, right=0.98, bottom=0.18, top=0.89)
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_PATH)
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)

    manifest = {
        "schema": "mean_entropy_ceff_absolute_error_figure_v1",
        "quantity": "abs(c_eff(t)-1)",
        "axes": {"x": "cycle/Ny", "y": "abs(c_eff-1), logarithmic"},
        "estimator_order": "average entropy over trajectories, then fit",
        "fit_window": "Ay=8..Ny/2 inclusive",
        "source": {
            "path": str(SOURCE_CSV.relative_to(ROOT)),
            "bytes": SOURCE_CSV.stat().st_size,
            "sha256": sha256(SOURCE_CSV),
        },
        "outputs": {
            "csv": str(OUTPUT_CSV.relative_to(ROOT)),
            "pdf": str(PDF_PATH.relative_to(ROOT)),
            "png": str(PNG_PATH.relative_to(ROOT)),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for ny in NY_VALUES:
        selected = [row for row in rows if row["Ny"] == ny]
        minimum = min(selected, key=lambda row: row["abs_c_eff_minus_one"])
        print(
            f"Ny={ny}: minimum={minimum['abs_c_eff_minus_one']:.6g} "
            f"at t/Ny={minimum['normalized_cycle']:.3f}; "
            f"endpoint={selected[-1]['abs_c_eff_minus_one']:.6g}"
        )


if __name__ == "__main__":
    main()
