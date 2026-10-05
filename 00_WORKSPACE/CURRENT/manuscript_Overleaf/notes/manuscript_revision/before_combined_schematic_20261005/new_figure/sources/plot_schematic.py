#!/usr/bin/env python3
"""Build a double-column schematic of the 2+1D domain-wall adaptive circuit."""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from manuscript_typography import configure_style as manuscript_style, prepare_figure, record_typography


HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE.parent
PDF_PATH = FIGURE_DIR / "Figure_01_schematic.pdf"
PNG_PATH = FIGURE_DIR / "Figure_01_schematic.png"

INK = "#20252B"
MID_GRAY = "#707780"
LIGHT_GRAY = "#ECECF1"
PALE_GRAY = "#F7F7F8"
TOPOLOGICAL = "#9BD0EA"
LEFT_WALL = "#8B1E2D"
RIGHT_WALL = "#164A7B"
INTERFACE = RIGHT_WALL
ANCILLA = "#E5B45B"


def _configure_style() -> None:
    manuscript_style({'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.linewidth': 0.7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': None})


def _arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], **kwargs: object) -> None:
    properties: dict[str, object] = {
        "arrowstyle": "-|>",
        "color": INK,
        "lw": 0.9,
        "mutation_scale": 8,
        "shrinkA": 0,
        "shrinkB": 0,
    }
    properties.update(kwargs)
    axis.annotate("", xy=end, xytext=start, arrowprops=properties)


def _box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "white",
    edgecolor: str = INK,
    linewidth: float = 0.9,
    fontsize: float = 7.5,
    rounding: float = 0.025,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=3,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        zorder=4,
    )
    return patch


def _draw_spacetime_geometry(axis: plt.Axes) -> None:
    """Stack plan-view slabs to emphasize two spatial directions plus circuit time."""

    layer_width = 1.64
    layer_height = 0.58
    left_fraction = 0.29
    right_fraction = 0.71
    layers = [(0.18, 0.15), (0.29, 0.49), (0.40, 0.83)]
    site_x_fractions = [0.08 + index * 0.084 for index in range(11)]
    site_y_fractions = [0.14 + index * 0.18 for index in range(5)]

    for index, (x0, y0) in enumerate(layers):
        alpha = 0.68 if index < 2 else 1.0
        axis.add_patch(
            Rectangle(
                (x0, y0),
                layer_width,
                layer_height,
                facecolor=LIGHT_GRAY,
                edgecolor=MID_GRAY,
                linewidth=1.15,
                alpha=alpha,
                zorder=1 + 3 * index,
            )
        )
        x_left = x0 + left_fraction * layer_width
        x_right = x0 + right_fraction * layer_width
        axis.add_patch(
            Rectangle(
                (x_left, y0),
                x_right - x_left,
                layer_height,
                facecolor=TOPOLOGICAL,
                edgecolor="none",
                alpha=alpha,
                zorder=2 + 3 * index,
            )
        )
        for x_fraction in site_x_fractions:
            for y_fraction in site_y_fractions:
                axis.add_patch(
                    Circle(
                        (
                            x0 + x_fraction * layer_width,
                            y0 + y_fraction * layer_height,
                        ),
                        0.009,
                        facecolor=INK,
                        edgecolor="none",
                        alpha=0.72 * alpha,
                        zorder=2.5 + 3 * index,
                    )
                )
        axis.plot(
            [x_left, x_left],
            [y0, y0 + layer_height],
            color=INTERFACE,
            linewidth=1.8,
            alpha=alpha,
            zorder=3 + 3 * index,
        )
        axis.plot(
            [x_right, x_right],
            [y0, y0 + layer_height],
            color=INTERFACE,
            linewidth=1.8,
            alpha=alpha,
            zorder=3 + 3 * index,
        )

    top_x, top_y = layers[-1]
    support_center_x = top_x + site_x_fractions[4] * layer_width
    # A lightly indicated 3 x 3 unit-cell neighborhood around the selected mode.
    support_size = 0.31
    center_y = top_y + site_y_fractions[1] * layer_height
    axis.add_patch(
        Rectangle(
            (support_center_x - support_size / 2, center_y - support_size / 2),
            support_size,
            support_size,
            facecolor=(1, 1, 1, 0.25),
            edgecolor=(0, 0, 0, 0.45),
            linewidth=0.45,
            zorder=13,
        )
    )
    axis.add_patch(Circle((support_center_x, center_y), 0.025, facecolor=LEFT_WALL, edgecolor="none", zorder=14))
    axis.text(
        support_center_x,
        center_y + support_size / 2 + 0.012,
        r"$\hat{\boldsymbol{\mathcal{N}}}_{\boldsymbol{r},\nu,\sigma}$",
        ha="center",
        va="bottom",
        fontsize=8,
        zorder=20,
        bbox={"facecolor": TOPOLOGICAL, "edgecolor": "none", "pad": 0.35},
    )

    phase_label_y = top_y + 0.80 * layer_height
    axis.text(top_x + 0.50 * layer_width, phase_label_y, r"$\boldsymbol{\alpha}_1$", ha="center", va="center", fontsize=8, zorder=20)
    axis.text(top_x + 0.145 * layer_width, phase_label_y, r"$\boldsymbol{\alpha}_2$", ha="center", va="center", fontsize=8, zorder=20)
    axis.text(top_x + 0.855 * layer_width, phase_label_y, r"$\boldsymbol{\alpha}_2$", ha="center", va="center", fontsize=8, zorder=20)
    interface_label_y = top_y + layer_height + 0.025
    axis.text(top_x + left_fraction * layer_width, interface_label_y, r"$\boldsymbol{x_L}$", color=INTERFACE, ha="center", va="bottom", fontsize=8, zorder=20)
    axis.text(top_x + right_fraction * layer_width, interface_label_y, r"$\boldsymbol{x_R}$", color=INTERFACE, ha="center", va="bottom", fontsize=8, zorder=20)
    _arrow(axis, (0.08, 0.18), (0.34, 1.43), color=MID_GRAY, lw=1.0)
    axis.text(0.08, 0.83, "circuit time", color=MID_GRAY, rotation=78, ha="center", va="center")
    axis.set_xlim(-0.02, 2.13)
    axis.set_ylim(0.04, 1.55)
    axis.set_aspect("equal")
    axis.axis("off")


def _draw_feedback_tile(axis: plt.Axes) -> None:
    """Show one Born measurement followed by conditional ideal feedback."""

    wire_y = 1.02
    _box(axis, (0.58, 0.80), 0.72, 0.44, r"measure" + "\n" + r"$\hat{\mathcal{N}}_{\boldsymbol{r},\nu,\sigma}$", facecolor=PALE_GRAY)
    _arrow(axis, (1.30, wire_y), (1.47, wire_y))
    diamond_center = (1.76, wire_y)
    diamond_w, diamond_h = 0.58, 0.50
    axis.add_patch(
        Polygon(
            [
                (diamond_center[0], diamond_center[1] + diamond_h / 2),
                (diamond_center[0] + diamond_w / 2, diamond_center[1]),
                (diamond_center[0], diamond_center[1] - diamond_h / 2),
                (diamond_center[0] - diamond_w / 2, diamond_center[1]),
            ],
            closed=True,
            facecolor="white",
            edgecolor=INK,
            linewidth=0.9,
            zorder=3,
        )
    )
    axis.text(*diamond_center, r"$\mathrm{m}=s_\sigma$?", ha="center", va="center", fontsize=8, zorder=4)

    # Matching branch: proceed directly to the next OW-mode measurement.
    _arrow(axis, (2.05, wire_y), (3.30, wire_y))
    axis.text((2.05 + 3.10) / 2, 1.13, "yes", ha="center", va="bottom", fontsize=8)

    # Mismatch branch: prepare the target ancilla and fSWAP.
    axis.plot([1.76, 1.76], [0.77, 0.49], color=INK, linewidth=0.9)
    _arrow(axis, (1.76, 0.49), (2.23, 0.49))
    axis.text(1.85, 0.57, "no", ha="left", va="bottom", fontsize=8)
    _box(axis, (2.23, 0.30), 0.64, 0.38, r"fSWAP", facecolor="white", fontsize=8)
    _arrow(axis, (2.87, 0.49), (3.10, 0.49))

    f_swap_center_x = 2.55
    axis.add_patch(Circle((f_swap_center_x, 0.03), 0.082, facecolor=ANCILLA, edgecolor=INK, linewidth=0.85, zorder=4))
    axis.text(f_swap_center_x, 0.03, r"$s_\sigma$", ha="center", va="center", fontsize=8, zorder=5)
    _arrow(axis, (f_swap_center_x, 0.12), (f_swap_center_x, 0.30))
    axis.text(2.69, 0.03, "fresh\nancilla", ha="left", va="center", fontsize=8, linespacing=1.0)

    # Merge both branches at a target-occupation output.
    axis.plot([3.10, 3.10], [0.49, wire_y], color=INK, linewidth=0.9)
    axis.add_patch(Circle((3.10, wire_y), 0.022, facecolor=INK, edgecolor="none", zorder=5))
    _box(axis, (3.30, 0.82), 0.76, 0.40, "measure" + "\n" + "next OW" + "\n" + "mode", facecolor=PALE_GRAY, fontsize=8)

    axis.text(
        1.25,
        0.03,
        "$s_-=1$ (fill)\n$s_+=0$ (empty)",
        ha="left",
        va="center",
        linespacing=1.25,
    )
    axis.set_xlim(0.35, 4.17)
    axis.set_ylim(-0.12, 1.52)
    axis.axis("off")


def build() -> None:
    _configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(7.05, 2.72))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.45, 1.75], wspace=0.02)
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
    ]

    _draw_spacetime_geometry(axes[0])
    _draw_feedback_tile(axes[1])

    figure.text(0.022, 0.975, "(a)", ha="left", va="top", fontsize=9)
    figure.text(0.465, 0.975, "(b)", ha="left", va="top", fontsize=9)

    prepare_figure(figure, "Figure_01_schematic")
    figure.subplots_adjust(left=0.025, right=0.992, bottom=0.025, top=0.99)
    record_typography(figure, "Figure_01_schematic")
    figure.savefig(PDF_PATH)
    plt.close(figure)
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-r",
            "300",
            "-singlefile",
            str(PDF_PATH),
            str(PNG_PATH.with_suffix("")),
        ],
        check=True,
    )


if __name__ == "__main__":
    build()
