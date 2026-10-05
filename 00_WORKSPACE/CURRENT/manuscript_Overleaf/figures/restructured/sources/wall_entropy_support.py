"""Original drawing helpers with destination redirected to the bundle."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import math, csv
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
FIGURE_DIR=Path(__file__).resolve().parents[1]
NX = 20

NY_VALUES = (30, 35, 40, 45, 55)

SIZE_STYLES = {
    30: {"color": "#D92725", "marker": "^"},
    35: {"color": "#F08050", "marker": "<"},
    40: {"color": "#8FC1E3", "marker": "v"},
    45: {"color": "#2CA02C", "marker": "s"},
    55: {"color": "#000000", "marker": "P"},
}

def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times", "Nimbus Roman", "Times New Roman", "Liberation Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "lines.linewidth": 0.9,
            "lines.markersize": 3.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

def make_figure(
    fits: dict[str, tuple[dict[int, dict[str, Any]], dict[str, float]]],
    figure_stem: Path,
    panel_names: dict[str, tuple[str, str, str, str]],
    wall_window_cells: dict[str, np.ndarray] | None = None,
    inset_right_wall_at_cell_edge: bool = False,
    plot_min_ay: int = 1,
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(3.375, 4.85), sharex=True)
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    for panel, (axis, component) in enumerate(zip(axes, ("left", "right"))):
        by_size, summary = fits[component]
        fit_min = min(float(item["x_fit"].min()) for item in by_size.values())
        axis.axvspan(fit_min, 0.0, color="0.5", alpha=0.15, linewidth=0, zorder=0)
        for ny in NY_VALUES:
            item = by_size[ny]
            style = SIZE_STYLES[ny]
            shown = item['ay_all'] >= plot_min_ay
            empirical = axis.errorbar(
                item["x_all"][shown],
                item["mean_all"][shown],
                yerr=item["sem_all"][shown],
                color=style["color"],
                marker=style["marker"],
                linestyle="none",
                markerfacecolor="white",
                markeredgewidth=0.8,
                markersize=3.4,
                elinewidth=0.45,
                capsize=0.0,
                zorder=3,
            )
            if panel == 0:
                legend_handles.append(empirical[0])
                legend_labels.append(rf"$N_y={ny}$")
        x_line = np.linspace(
            min(float(item["x_all"][item['ay_all'] >= plot_min_ay].min())
                for item in by_size.values()), 0.0, 300
        )
        axis.plot(
            x_line,
            summary["slope"] * x_line,
            color="black",
            linestyle="--",
            linewidth=0.9,
            zorder=2,
        )
        wall_name, x_range, slope_symbol, entropy_symbol = panel_names[component]
        axis.text(
            0.025,
            0.93,
            (
                rf"{wall_name}  ({x_range})" "\n"
                rf"${slope_symbol}={summary['slope']:.5f}\pm"
                rf"{summary['slope_covariance_sem']:.5f}$, "
                rf"$R_0^2={summary['R0_squared']:.6f}$"
            ),
            transform=axis.transAxes,
            fontsize=6.2,
            va="top",
        )
        if wall_window_cells is not None:
            selected = np.asarray(wall_window_cells[component], dtype=np.int64)
            if selected.ndim != 1 or selected.size == 0:
                raise ValueError(f"invalid wall-window cells for {component}")
            inset_axis = axis.inset_axes([0.075, 0.33, 0.17, 0.43])
            inset_axis.add_patch(
                Rectangle(
                    (float(selected.min()), 0.0),
                    float(selected.max() - selected.min() + 1),
                    30.0,
                    facecolor="#4C78A8",
                    edgecolor="none",
                    alpha=0.40,
                    zorder=0,
                )
            )
            for coordinate in range(NX + 1):
                inset_axis.axvline(
                    coordinate,
                    color="0.45",
                    alpha=0.34,
                    linewidth=0.16,
                    zorder=1,
                )
            for coordinate in range(31):
                inset_axis.axhline(
                    coordinate,
                    color="0.45",
                    alpha=0.34,
                    linewidth=0.16,
                    zorder=1,
                )
            for wall_x, wall_label in ((5, r"$x_{\rm L}$"), (15, r"$x_{\rm R}$")):
                if inset_right_wall_at_cell_edge and component == "right" and wall_x == 15:
                    wall_x = 16  # Right edge of cell x=15 in the inset grid.
                inset_axis.axvline(
                    wall_x,
                    color="#A51C30",
                    linestyle="--",
                    linewidth=0.75,
                    zorder=3,
                )
                inset_axis.text(
                    wall_x,
                    30.7,
                    wall_label,
                    color="#A51C30",
                    fontsize=4.8,
                    ha="center",
                    va="bottom",
                    clip_on=False,
                    zorder=4,
                )
            inset_axis.set_xlim(0, NX)
            inset_axis.set_ylim(0, 30)
            inset_axis.set_aspect("equal")
            inset_axis.set_xticks(())
            inset_axis.set_yticks(())
            inset_axis.set_axis_off()
        axis.text(
            -0.18,
            1.04,
            f"({chr(ord('a') + panel)})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
        )
        axis.set_xlim(float(x_line.min()) - .06 if plot_min_ay > 1 else -3.05, 0.05)
        axis.set_ylabel(rf"$\Delta\langle {entropy_symbol}\rangle_\xi$")
    axes[0].legend(
        legend_handles,
        legend_labels,
        ncol=2,
        loc="lower right",
        columnspacing=0.7,
        handletextpad=0.3,
        borderaxespad=0.4,
    )
    axes[1].set_xlabel(
        r"$\log[\sin(\pi A_y/N_y)/\sin(\pi A_y^\star/N_y)]$"
    )
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.105,
                        top=0.95 if plot_min_ay > 1 else 0.965, hspace=0.16)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_stem.with_suffix(".pdf"))
    fig.savefig(figure_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
