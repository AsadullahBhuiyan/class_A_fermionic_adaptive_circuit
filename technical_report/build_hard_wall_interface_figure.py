#!/usr/bin/env python3
"""Build a single-interface schematic of hard-wall OW-support truncation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Rectangle


HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE / "figures"
PDF_PATH = FIGURE_DIR / "hard_wall_interface_schematic.pdf"
PNG_PATH = FIGURE_DIR / "hard_wall_interface_schematic.png"

INK = "#20252B"
MID_GRAY = "#707780"
LIGHT_GRAY = "#ECECF1"
TOPOLOGICAL = "#9BD0EA"
INTERFACE = "#164A7B"
MODE_CENTER = "#8B1E2D"
REMOVED = "#D97A58"


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.linewidth": 0.7,
            "hatch.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": None,
        }
    )


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    **kwargs: object,
) -> None:
    zorder = float(kwargs.pop("zorder", 10))
    properties: dict[str, object] = {
        "arrowstyle": "-|>",
        "color": INK,
        "lw": 0.8,
        "mutation_scale": 7,
        "shrinkA": 0,
        "shrinkB": 0,
    }
    properties.update(kwargs)
    axis.annotate("", xy=end, xytext=start, arrowprops=properties, zorder=zorder)


def _draw_layer(
    axis: plt.Axes,
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    interface_offset: float,
    alpha: float,
    zorder: float,
) -> float:
    """Draw one cropped lattice layer and return the interface coordinate."""

    x_wall = x0 + interface_offset
    axis.add_patch(
        Rectangle(
            (x0, y0),
            x_wall - x0,
            height,
            facecolor=TOPOLOGICAL,
            edgecolor="none",
            alpha=alpha,
            zorder=zorder,
        )
    )
    axis.add_patch(
        Rectangle(
            (x_wall, y0),
            x0 + width - x_wall,
            height,
            facecolor=LIGHT_GRAY,
            edgecolor="none",
            alpha=alpha,
            zorder=zorder,
        )
    )
    axis.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            facecolor="none",
            edgecolor=MID_GRAY,
            linewidth=1.05,
            alpha=alpha,
            zorder=zorder + 0.3,
        )
    )

    x_offsets = [0.12 + 0.20 * index for index in range(9)]
    y_offsets = [0.10 + 0.13 * index for index in range(5)]
    for x_offset in x_offsets:
        for y_offset in y_offsets:
            axis.add_patch(
                Circle(
                    (x0 + x_offset, y0 + y_offset),
                    0.010,
                    facecolor=INK,
                    edgecolor="none",
                    alpha=0.70 * alpha,
                    zorder=zorder + 0.4,
                )
            )

    axis.plot(
        [x_wall, x_wall],
        [y0, y0 + height],
        color=INTERFACE,
        linewidth=2.0,
        alpha=alpha,
        zorder=zorder + 0.8,
    )
    return x_wall


def build() -> None:
    _configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(3.375, 2.35))

    layer_width = 1.82
    layer_height = 0.68
    interface_offset = 1.02
    layers = [(0.12, 0.16), (0.25, 0.50), (0.38, 0.84)]

    wall_positions: list[float] = []
    for index, (x0, y0) in enumerate(layers):
        wall_positions.append(
            _draw_layer(
                axis,
                x0=x0,
                y0=y0,
                width=layer_width,
                height=layer_height,
                interface_offset=interface_offset,
                alpha=(0.48, 0.68, 1.0)[index],
                zorder=1.0 + 3.0 * index,
            )
        )

    top_x, top_y = layers[-1]
    x_wall = wall_positions[-1]

    # The selected OW mode sits on the lattice site immediately to the left of
    # x_R.  Its nominal 3 x 3 support crosses the interface by one site column.
    center_x = top_x + 0.92
    center_y = top_y + 0.36
    support_x0 = center_x - 0.30
    support_y0 = center_y - 0.205
    support_x1 = center_x + 0.30
    support_height = 0.41

    # Nominal support before truncation.
    axis.add_patch(
        Rectangle(
            (support_x0, support_y0),
            support_x1 - support_x0,
            support_height,
            facecolor="none",
            edgecolor=MID_GRAY,
            linewidth=0.75,
            linestyle=(0, (3, 2)),
            zorder=13,
        )
    )

    # Retained, renormalized hard-wall support.
    axis.add_patch(
        Rectangle(
            (support_x0, support_y0),
            x_wall - support_x0,
            support_height,
            facecolor=to_rgba("white", 0.40),
            edgecolor=INTERFACE,
            linewidth=1.05,
            zorder=13.2,
        )
    )

    # The part that would cross the wall is explicitly marked as removed.
    axis.add_patch(
        Rectangle(
            (x_wall, support_y0),
            support_x1 - x_wall,
            support_height,
            facecolor=to_rgba(REMOVED, 0.18),
            edgecolor=REMOVED,
            linewidth=0.7,
            hatch="////",
            zorder=13.1,
        )
    )
    axis.plot(
        [x_wall + 0.035, support_x1 - 0.035],
        [support_y0 + 0.04, support_y0 + support_height - 0.04],
        color=MODE_CENTER,
        linewidth=0.8,
        zorder=14,
    )
    axis.plot(
        [x_wall + 0.035, support_x1 - 0.035],
        [support_y0 + support_height - 0.04, support_y0 + 0.04],
        color=MODE_CENTER,
        linewidth=0.8,
        zorder=14,
    )

    # Redraw the wall on top of the support patches and mark the OW center.
    axis.plot(
        [x_wall, x_wall],
        [top_y, top_y + layer_height],
        color=INTERFACE,
        linewidth=2.2,
        zorder=15,
    )
    axis.add_patch(
        Circle(
            (center_x, center_y),
            0.029,
            facecolor=MODE_CENTER,
            edgecolor="white",
            linewidth=0.45,
            zorder=16,
        )
    )

    axis.text(
        0.5 * (top_x + x_wall),
        top_y + 0.60,
        r"topological $\alpha_1$",
        ha="center",
        va="center",
        zorder=17,
    )
    axis.text(
        0.5 * (x_wall + top_x + layer_width),
        top_y + 0.60,
        r"trivial $\alpha_2$",
        ha="center",
        va="center",
        zorder=17,
    )
    axis.text(
        x_wall,
        top_y + layer_height + 0.045,
        r"$x_R$",
        color=INTERFACE,
        ha="center",
        va="bottom",
        fontsize=8.5,
        zorder=17,
    )
    axis.text(
        support_x0 - 0.035,
        center_y,
        r"$\hat{N}^{\mathrm{hard}}_{\boldsymbol{r},\nu,\sigma}$",
        ha="right",
        va="center",
        fontsize=7.8,
        zorder=17,
    )

    removed_center = (0.5 * (x_wall + support_x1), center_y)
    axis.text(
        2.24,
        1.18,
        "discarded\nsupport",
        color=MODE_CENTER,
        ha="left",
        va="center",
        fontsize=7.2,
        linespacing=1.05,
    )
    _arrow(
        axis,
        (2.18, 1.18),
        (removed_center[0] + 0.02, removed_center[1]),
        color=MODE_CENTER,
        lw=0.75,
        zorder=18,
    )

    _arrow(axis, (0.04, 0.21), (0.28, 1.47), color=MID_GRAY, lw=0.9)
    axis.text(
        0.01,
        0.85,
        "circuit time",
        color=MID_GRAY,
        rotation=79,
        ha="center",
        va="center",
    )

    axis.set_xlim(-0.05, 2.93)
    axis.set_ylim(0.04, 1.68)
    axis.set_aspect("equal")
    axis.axis("off")
    figure.subplots_adjust(left=0.015, right=0.995, bottom=0.02, top=0.985)
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
