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


def _draw_geometry(axis: plt.Axes) -> None:
    """One periodic cell; shaded OW supports are clipped to their center's region."""
    width, height = 16, 24
    left, right = 4, 12
    axis.add_patch(Rectangle((0, 0), width, height, facecolor=LIGHT_GRAY,
                             edgecolor="none", zorder=0))
    axis.add_patch(Rectangle((left, 0), right-left, height,
                             facecolor=TOPOLOGICAL, edgecolor="none", zorder=0))
    for x in (i + 0.5 for i in range(width)):
        for y in (i + 0.5 for i in range(height)):
            axis.add_patch(Circle((x, y), 0.055, facecolor=INK,
                                  edgecolor="none", alpha=0.58, zorder=2))

    # The centers lie on unit cells, and the walls lie between columns.
    # Each nominal w=1 support contains exactly 3 x 3 cells before wall clipping.
    operations = [
        (1.5, 20.5, 0, left, "#D97A58", LEFT_WALL, (4.6, 20.5), "left"),
        (4.5, 14.5, left, right, "#2386A8", RIGHT_WALL, (6.5, 14.5), "left"),
        (8.5, 8.5, left, right, "#2386A8", RIGHT_WALL, (8.5, 11.0), "center"),
        (12.5, 2.5, right, width, "#D97A58", LEFT_WALL, (11.5, 5.0), "right"),
    ]
    for x, y, lo, hi, fill, dot, label_xy, align in operations:
        x0, x1 = max(x-1.5, lo), min(x+1.5, hi)
        axis.add_patch(Rectangle((x0, y-1.5), x1-x0, 3,
                                 facecolor=fill, alpha=0.52, edgecolor="none", zorder=1))
        axis.add_patch(Rectangle((x0, y-1.5), x1-x0, 3,
                                 facecolor="none", edgecolor="black", linewidth=1.0, zorder=3))
        axis.plot(x, y, "o", color=dot, markeredgecolor="white",
                  markeredgewidth=0.4, markersize=4.5, zorder=5)
        axis.text(*label_xy,
                  r"$\hat{\mathcal{N}}_{\boldsymbol{r},\nu,\sigma}(\alpha_{\boldsymbol{r}})$",
                  ha=align, va="center", zorder=6,
                  bbox=dict(facecolor=TOPOLOGICAL,
                            edgecolor="none", pad=0.6))
    for x in (left, right):
        axis.plot([x, x], [0, height], color=INTERFACE, linewidth=1.5, zorder=4)
    axis.add_patch(Rectangle((0, 0), width, height, facecolor="none",
                             edgecolor=MID_GRAY, linewidth=1.1, zorder=4))
    for x, text in ((2, r"$\alpha_2$"), (8, r"$\alpha_1$"), (14, r"$\alpha_2$")):
        axis.text(x, height+0.65, text, ha="center", va="bottom")
    _arrow(axis, (0, -1), (width, -1), arrowstyle="<->", lw=0.8)
    axis.text(width/2, -1.3, r"$N_x$", ha="center", va="top")
    _arrow(axis, (-1, 0), (-1, height), arrowstyle="<->", lw=0.8)
    axis.text(-1.4, height/2, r"$N_y$", ha="right", va="center", rotation=90)
    axis.set_xlim(-2.5, 16.4)
    axis.set_ylim(-2.5, 26)
    axis.set_aspect("equal")
    axis.axis("off")


def _draw_feedback_tile(axis: plt.Axes) -> None:
    """Show one Born measurement followed by conditional ideal feedback."""

    wire_y = 1.15
    _box(axis, (0.35, 0.86), 0.95, 0.58, r"measure" + "\n" + r"$\hat{\mathcal{N}}_{\boldsymbol{r},\nu,\sigma}(\alpha_{\boldsymbol{r}})$", facecolor=PALE_GRAY)
    _arrow(axis, (1.30, wire_y), (1.45, wire_y))
    diamond_center = (1.85, wire_y)
    diamond_w, diamond_h = 0.80, 0.80
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
    _arrow(axis, (2.25, wire_y), (3.30, wire_y))
    axis.text((2.05 + 3.10) / 2, 1.26, "yes", ha="center", va="bottom", fontsize=8)

    # Mismatch branch: prepare the target ancilla and fSWAP.
    axis.plot([1.85, 1.85], [0.75, 0.49], color=INK, linewidth=0.9)
    _arrow(axis, (1.85, 0.49), (2.23, 0.49))
    axis.text(1.93, 0.60, "no", ha="left", va="bottom", fontsize=8)
    _box(axis, (2.23, 0.30), 0.64, 0.38, r"fSWAP", facecolor="white", fontsize=8)
    _arrow(axis, (2.87, 0.49), (3.10, 0.49))

    f_swap_center_x = 2.55
    axis.add_patch(Circle((f_swap_center_x, -0.17), 0.082, facecolor=ANCILLA, edgecolor=INK, linewidth=0.85, zorder=4))
    axis.text(f_swap_center_x, -0.17, r"$s_\sigma$", ha="center", va="center", fontsize=8, zorder=5)
    _arrow(axis, (f_swap_center_x, -0.08), (f_swap_center_x, 0.30))
    axis.text(2.69, -0.17, "fresh\nancilla", ha="left", va="center", fontsize=8, linespacing=1.0)

    # Merge both branches at a target-occupation output.
    axis.plot([3.10, 3.10], [0.49, wire_y], color=INK, linewidth=0.9)
    axis.add_patch(Circle((3.10, wire_y), 0.022, facecolor=INK, edgecolor="none", zorder=5))
    _box(axis, (3.30, 0.86), 0.76, 0.58, "measure" + "\n" + "next OW" + "\n" + "mode", facecolor=PALE_GRAY, fontsize=8)

    axis.text(
        1.25,
        -0.17,
        "$s_-=1$ (fill)\n$s_+=0$ (empty)",
        ha="left",
        va="center",
        linespacing=1.25,
    )
    axis.set_xlim(0.20, 4.17)
    axis.set_ylim(-0.4, 1.68)
    axis.set_aspect("equal")
    axis.axis("off")


def build() -> None:
    _configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(7.05, 4.8))
    geometry = figure.add_axes([0.025, 0.025, 0.46, 0.94])
    feedback = figure.add_axes([0.50, 0.25, 0.49, 0.49])
    _draw_geometry(geometry)
    _draw_feedback_tile(feedback)
    figure.text(0.022, 0.975, "(a)", ha="left", va="top")
    feedback.text(0.0, 1.01, "(b)", transform=feedback.transAxes, ha="left", va="bottom")
    prepare_figure(figure, "Figure_01_schematic")
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
