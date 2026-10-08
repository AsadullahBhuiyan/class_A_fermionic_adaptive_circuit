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
from manuscript_typography import configure_style as manuscript_style, prepare_figure, record_typography
FIGURE_DIR=Path(__file__).resolve().parents[1]
NX = 20

from endpoint_even_support import SIZES as NY_VALUES, STYLES
SIZE_STYLES = {ny: {'color':color, 'marker':marker} for ny,(color,marker) in STYLES.items()}

def configure_matplotlib() -> None:
    manuscript_style({'figure.dpi': 140, 'savefig.dpi': 300, 'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.linewidth': 0.7, 'lines.linewidth': 0.9, 'lines.markersize': 3.5, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.width': 0.65, 'ytick.major.width': 0.65, 'xtick.major.size': 2.8, 'ytick.major.size': 2.8, 'axes.spines.top': True, 'axes.spines.right': True, 'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})

def make_figure(
    fits: dict[str, tuple[dict[int, dict[str, Any]], dict[str, float]]],
    figure_stem: Path,
    panel_names: dict[str, tuple[str, str, str, str]],
    wall_window_cells: dict[str, np.ndarray] | None = None,
    inset_right_wall_at_cell_edge: bool = False,
    plot_min_ay: int = 1,
    contour_maps: dict | None = None,
) -> None:
    configure_matplotlib()
    horizontal = contour_maps is not None
    height = 3.25 if horizontal else 5.05
    if horizontal:
        fig = plt.figure(figsize=(7.05, height))
        axes = [fig.add_axes([.425, .18, .24, .70]),
                fig.add_axes([.755, .18, .235, .70])]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(3.375, height), sharex=False)
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
                legend_labels.append(rf"${ny}$")
        x_line = np.linspace(
            min(float(item["x_all"][item['ay_all'] >= plot_min_ay].min())
                for item in by_size.values()), 0.0, 300
        )
        fit_line, = axis.plot(
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
            0.96,
            (
                rf"{wall_name}  ({x_range})" "\n"
                rf"${slope_symbol}={6*summary['slope']:.4f}\pm"
                rf"{6*summary['slope_covariance_sem']:.4f}$"
            ),
            transform=axis.transAxes,
            fontsize=8,
            va="top",
        )
        # Preserve the R-squared baseline while lifting the wall/slope block.
        axis.text(
            0.025,
            0.7644923113,
            rf"$R^2={summary['R0_squared']:.6f}$",
            transform=axis.transAxes,
            fontsize=8,
            va="baseline",
        )
        if wall_window_cells is not None:
            selected = np.asarray(wall_window_cells[component], dtype=np.int64)
            if selected.ndim != 1 or selected.size == 0:
                raise ValueError(f"invalid wall-window cells for {component}")
            inset_axis = axis.inset_axes([0.075, 0.37, 0.21, 0.30] if horizontal else [0.075, 0.30, 0.17, 0.37])
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
                    fontsize=8,
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
            f"({chr(ord('a') + panel + (1 if contour_maps is not None else 0))})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
        )
        axis.set_xlim(float(x_line.min()) - .06 if plot_min_ay > 1 else -3.05, 0.05)
        axis.set_xticks([-2, -1.5, -1, -.5, 0])
        axis.set_ylabel(rf"$\Delta {entropy_symbol}$")
        axis.set_xlabel(r"$\log[\sin(\pi A_y/N_y)]$")
        fit_legend = axis.legend(
            handles=[fit_line],
            labels=[rf"$\frac{{{slope_symbol}}}{{6}}\log[\sin(\pi A_y/N_y)]$"],
            loc="upper right", bbox_to_anchor=(0.98, 0.14 if horizontal else 0.48),
            handlelength=1.3, handletextpad=0.4, borderaxespad=0,
        )
        if panel == 0:
            axis.add_artist(fit_legend)
    if horizontal:
        fig.legend(legend_handles, legend_labels, title=r"$N_y$", ncol=3,
                   loc="upper center", bbox_to_anchor=(.20, .19),
                   frameon=False, handlelength=.6, columnspacing=.5,
                   handletextpad=.15, labelspacing=.2, borderaxespad=0)
    else:
        axes[0].legend(legend_handles, legend_labels, title=r"$N_y$", ncol=3,
                       loc="lower right", frameon=False, handlelength=.6,
                       columnspacing=.5, handletextpad=.15,
                       labelspacing=.2, borderaxespad=.4)
    if contour_maps is not None:
        from matplotlib.colors import PowerNorm
        norm = PowerNorm(gamma=.5, vmin=0, vmax=max(v.max() for v in contour_maps.values()))
        for index, alpha in enumerate((1, 3)):
            cmap_axis = fig.add_axes([.055 + index*.16, .42, .125, .125*7.05/height*16/20])
            image = cmap_axis.imshow(contour_maps[alpha], origin="lower", cmap="Blues", norm=norm,
                                     interpolation="none", aspect="equal")
            cmap_axis.set_xticks([0, 5, 15, 19]); cmap_axis.set_yticks([0, 5, 10, 15])
            cmap_axis.set_xlabel("$x$", labelpad=1)
            if index == 0:
                cmap_axis.set_ylabel(r"$\delta y$", labelpad=1)
            else:
                cmap_axis.tick_params(labelleft=False)
            cmap_axis.set_title(rf"$\alpha_1={alpha}$", pad=3)
            cmap_axis.tick_params(direction="in", pad=1, length=2)
        color_axis = fig.add_axes([.075, .30, .255, .018])
        colorbar = fig.colorbar(image, cax=color_axis, orientation="horizontal")
        colorbar.set_ticks([0, .1, .3, .44])
        colorbar.set_label(r"$\overline{s}(x,\delta y;A)$", labelpad=1)
        colorbar.ax.tick_params(pad=1, length=2)
        fig.text(.012, .943, "(a)", va="top")
    prepare_figure(fig, "Figure_09_wall_entropy")
    if not horizontal:
        fig.subplots_adjust(left=.20, right=.985, bottom=.105,
                            top=.95 if plot_min_ay > 1 else .965, hspace=.42)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    record_typography(fig, "Figure_09_wall_entropy")
    fig.savefig(figure_stem.with_suffix(".pdf"))
    fig.savefig(figure_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
