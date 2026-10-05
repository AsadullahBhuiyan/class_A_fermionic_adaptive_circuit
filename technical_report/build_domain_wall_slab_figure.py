#!/usr/bin/env python3
"""Build a single-column schematic of the periodic domain-wall slab geometry."""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE / "figures"
PDF_PATH = FIGURE_DIR / "domain_wall_slab_geometry.pdf"
PNG_PATH = FIGURE_DIR / "domain_wall_slab_geometry.png"

OUTER_BORDER = "#707780"
DIMENSION_COLOR = "#30343A"
TRIVIAL_FILL = "#ECECF1"
TOPOLOGICAL_FILL = "#9BD0EA"
LEFT_WALL = "#8B1E2D"
RIGHT_WALL = "#164A7B"


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": None,
        }
    )


def build() -> None:
    _configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(3.375, 2.15))

    x0, y0 = 0.22, 0.25
    width, height = 1.36, 0.88
    x_left = x0 + 0.27 * width
    x_right = x0 + 0.73 * width

    axis.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            facecolor=TRIVIAL_FILL,
            edgecolor="none",
        )
    )
    axis.add_patch(
        Rectangle(
            (x_left, y0),
            x_right - x_left,
            height,
            facecolor=TOPOLOGICAL_FILL,
            edgecolor="none",
        )
    )
    axis.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            facecolor="none",
            edgecolor=OUTER_BORDER,
            linewidth=1.6,
        )
    )

    axis.plot([x_left, x_left], [y0, y0 + height], color=LEFT_WALL, linewidth=2.0)
    axis.plot([x_right, x_right], [y0, y0 + height], color=RIGHT_WALL, linewidth=2.0)

    axis.text(
        0.5 * (x_left + x_right),
        y0 + 0.53 * height,
        "topological\n" + r"$\alpha_1$",
        ha="center",
        va="center",
    )
    axis.text(
        0.5 * (x0 + x_left),
        y0 + 0.53 * height,
        "trivial\n" + r"$\alpha_2$",
        ha="center",
        va="center",
    )
    axis.text(
        0.5 * (x_right + x0 + width),
        y0 + 0.53 * height,
        "trivial\n" + r"$\alpha_2$",
        ha="center",
        va="center",
    )
    axis.text(x_left, y0 + height + 0.055, "$x_L$", ha="center", va="bottom")
    axis.text(x_right, y0 + height + 0.055, "$x_R$", ha="center", va="bottom")

    arrow = {"arrowstyle": "<->", "color": DIMENSION_COLOR, "lw": 0.75}
    axis.annotate(
        "",
        xy=(x0, y0 - 0.11),
        xytext=(x0 + width, y0 - 0.11),
        arrowprops=arrow,
    )
    axis.text(x0 + 0.5 * width, y0 - 0.17, "$N_x$", ha="center", va="top")
    axis.annotate(
        "",
        xy=(x0 - 0.10, y0),
        xytext=(x0 - 0.10, y0 + height),
        arrowprops=arrow,
    )
    axis.text(
        x0 - 0.15,
        y0 + 0.5 * height,
        "$N_y$",
        ha="right",
        va="center",
        rotation=90,
    )

    axis.set_xlim(0.0, 1.78)
    axis.set_ylim(0.0, 1.42)
    axis.set_aspect("equal")
    axis.axis("off")
    figure.subplots_adjust(left=0.01, right=0.995, bottom=0.02, top=0.98)
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
