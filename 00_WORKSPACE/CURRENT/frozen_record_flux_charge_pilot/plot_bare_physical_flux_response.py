#!/usr/bin/env python3
"""Plot the bare physical wall response from the completed frozen-record pilot."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_CAMPAIGN = HERE / "results/N16x20_frozen_flux_charge_v1"


def load_flux_paths(
    csv_path: Path,
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    rows: dict[tuple[str, str], list[tuple[int, float, float, float]]] = {
        (arm, direction): []
        for arm in ("soft", "hard")
        for direction in ("ccw", "cw")
    }
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["arm"]), str(row["direction"]))
            if key in rows:
                rows[key].append(
                    (
                        int(row["twist_index"]),
                        float(row["q_wall"]),
                        float(row["delta_N_left"]),
                        float(row["delta_N_right"]),
                    )
                )
    result: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for key, values in rows.items():
        values.sort()
        if [value[0] for value in values] != list(range(17)):
            raise RuntimeError(f"{key} does not contain the complete 17-point flux path")
        result[key] = {
            "phi": 2.0 * np.pi * np.asarray([value[0] for value in values]) / 16.0,
            "q_wall": np.asarray([value[1] for value in values], dtype=np.float64),
            "delta_N_left": np.asarray([value[2] for value in values], dtype=np.float64),
            "delta_N_right": np.asarray([value[3] for value in values], dtype=np.float64),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    campaign = args.campaign_dir.resolve()
    output = (args.output_dir or campaign / "figures").resolve()
    data = load_flux_paths(campaign / "charge_vs_phi.csv")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.85), sharex=True, sharey=True)
    styles = {
        "ccw": {
            "color": "#1f77b4",
            "marker": "o",
            "linestyle": "-",
            "label": "counterclockwise",
        },
        "cw": {
            "color": "#d62728",
            "marker": "s",
            "linestyle": "--",
            "label": "clockwise",
        },
    }
    for ax, arm in zip(axes, ("soft", "hard"), strict=True):
        for direction in ("ccw", "cw"):
            path = data[(arm, direction)]
            ax.plot(
                path["phi"],
                path["q_wall"],
                linewidth=1.25,
                markersize=3.5,
                markerfacecolor="white",
                markeredgewidth=1.0,
                **styles[direction],
            )
        ax.axhline(0.0, color="0.25", linestyle=":", linewidth=0.8, zorder=0)
        ax.set_xlim(0.0, 2.0 * np.pi)
        ax.set_ylim(-0.70, 0.50)
        ax.set_xticks([0.0, np.pi, 2.0 * np.pi])
        ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
        ax.set_xlabel(r"flux path $\phi$")
        ax.set_title(f"{arm.capitalize()} wall")
        ax.tick_params(which="both", direction="in", top=True, right=True)
    axes[0].set_ylabel(r"wall response $q_x$")
    axes[1].legend(frameon=False, loc="lower right", handlelength=2.4)
    fig.tight_layout(pad=0.35)

    output.mkdir(parents=True, exist_ok=True)
    pdf = output / "bare_physical_wall_response_phi.pdf"
    png = output / "bare_physical_wall_response_phi.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.55), sharex=True, sharey=True)
    region_styles = {
        "delta_N_left": {
            "color": "#9467bd",
            "marker": "o",
            "region": r"$\Delta N_L$",
        },
        "delta_N_right": {
            "color": "#ff7f0e",
            "marker": "s",
            "region": r"$\Delta N_R$",
        },
    }
    direction_styles = {
        "ccw": {"linestyle": "-", "direction": "counterclockwise"},
        "cw": {"linestyle": "--", "direction": "clockwise"},
    }
    for ax, arm in zip(axes, ("soft", "hard"), strict=True):
        for direction in ("ccw", "cw"):
            path = data[(arm, direction)]
            for field in ("delta_N_left", "delta_N_right"):
                region = region_styles[field]
                sense = direction_styles[direction]
                ax.plot(
                    path["phi"],
                    path[field],
                    color=region["color"],
                    marker=region["marker"],
                    linestyle=sense["linestyle"],
                    linewidth=1.2,
                    markersize=3.2,
                    markerfacecolor="white",
                    markeredgewidth=0.9,
                    label=f'{region["region"]}, {sense["direction"]}',
                )
        ax.axhline(0.0, color="0.25", linestyle=":", linewidth=0.8, zorder=0)
        ax.set_xlim(0.0, 2.0 * np.pi)
        ax.set_ylim(-0.70, 0.70)
        ax.set_xticks([0.0, np.pi, 2.0 * np.pi])
        ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
        ax.set_xlabel(r"flux path $\phi$")
        ax.set_title(f"{arm.capitalize()} wall")
        ax.tick_params(which="both", direction="in", top=True, right=True)
    axes[0].set_ylabel(r"regional charge change")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        ncol=4,
        handlelength=2.2,
        fontsize=6.8,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0), pad=0.35)

    regional_pdf = output / "bare_physical_regional_charge_phi.pdf"
    regional_png = output / "bare_physical_regional_charge_phi.png"
    fig.savefig(regional_pdf)
    fig.savefig(regional_png, dpi=300)
    plt.close(fig)
    print(pdf)
    print(png)
    print(regional_pdf)
    print(regional_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
