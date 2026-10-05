#!/usr/bin/env python3
"""Build a wide single-panel schematic of hard-wall OW-support truncation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Rectangle


HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE / "figures"
PDF_PATH = FIGURE_DIR / "ow_overlap_truncation_schematic.pdf"
PNG_PATH = FIGURE_DIR / "ow_overlap_truncation_schematic.png"

INK = "#173746"
OUTER_BORDER = "#707780"
LEFT_BULK = "#ECECF1"
RIGHT_BULK = "#9BD0EA"
LEFT_MODE = "#D97A58"
RIGHT_MODE = "#2386A8"
LEFT_DOT = "#8B1E2D"
RIGHT_DOT = "#164A7B"


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": None,
        }
    )


def _base_panel(axis: plt.Axes) -> None:
    width = 1.70
    interface = width / 2
    axis.add_patch(Rectangle((0, 0), interface, 1, facecolor=LEFT_BULK, edgecolor="none"))
    axis.add_patch(
        Rectangle((interface, 0), width - interface, 1, facecolor=RIGHT_BULK, edgecolor="none")
    )
    axis.add_patch(
        Rectangle(
            (0, 0),
            width,
            1,
            facecolor="none",
            edgecolor=OUTER_BORDER,
            linewidth=1.5,
        )
    )

    # The interface lies between two columns of lattice sites.
    site_x = [0.15 + 0.20 * index for index in range(8)]
    site_y = [0.20, 0.40, 0.60, 0.80]
    for x in site_x:
        for y in site_y:
            axis.add_patch(
                Circle(
                    (x, y),
                    0.013,
                    facecolor=INK,
                    edgecolor="none",
                    alpha=0.62,
                    zorder=2,
                )
            )

    axis.plot(
        [interface, interface],
        [0, 1],
        color=INK,
        linewidth=0.75,
        linestyle=(0, (2, 2)),
        zorder=3,
    )
    axis.text(0.42, 0.89, r"$\boldsymbol{\alpha}_2$", ha="center", va="center", fontsize=9, zorder=6)
    axis.text(1.28, 0.89, r"$\boldsymbol{\alpha}_1$", ha="center", va="center", fontsize=9, zorder=6)
    axis.set_xlim(-0.05, 1.75)
    axis.set_ylim(-0.25, 1.08)
    axis.set_aspect("equal")
    axis.axis("off")


def _mode_window(
    axis: plt.Axes,
    *,
    x0: float,
    width: float,
    color: str,
    center_x: float,
    center_y: float,
    dot_color: str,
) -> None:
    axis.add_patch(
        Rectangle(
            (x0, 0.12),
            width,
            0.56,
            facecolor=to_rgba(color, 0.52),
            edgecolor="black",
            linewidth=1.35,
        )
    )
    axis.plot(
        center_x,
        center_y,
        "o",
        color=dot_color,
        markeredgecolor="white",
        markeredgewidth=0.35,
        markersize=3.8,
        zorder=5,
    )


def build() -> None:
    _configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(3.375, 1.72))

    _base_panel(axis)
    interface = 0.85

    # Each nominal three-column support would cross the interface.  At the
    # hard wall only the part on the mode center's own side is retained.
    _mode_window(
        axis,
        x0=0.45,
        width=interface - 0.45,
        color=LEFT_MODE,
        center_x=0.75,
        center_y=0.40,
        dot_color=LEFT_DOT,
    )
    _mode_window(
        axis,
        x0=interface,
        width=1.25 - interface,
        color=RIGHT_MODE,
        center_x=0.95,
        center_y=0.40,
        dot_color=RIGHT_DOT,
    )
    axis.text(
        interface,
        -0.12,
        "truncated OW support (hard wall)",
        ha="center",
        va="top",
    )

    figure.subplots_adjust(left=0.025, right=0.995, bottom=0.035, top=0.99)
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
