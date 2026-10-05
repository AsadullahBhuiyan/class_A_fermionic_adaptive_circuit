#!/usr/bin/env python3
"""Build separate single-column geometry and adaptive-circuit schematics."""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path as DrawingPath
from manuscript_typography import configure_style as manuscript_style, prepare_figure, record_typography


HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE.parent

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


def _elbow(axis, points):
    """A continuous orthogonal connector, with one terminal arrowhead."""
    path = DrawingPath(points, [DrawingPath.MOVETO] + [DrawingPath.LINETO]*(len(points)-1))
    axis.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=8,
                                  linewidth=0.9, color=INK, joinstyle="miter", zorder=2))


def _draw_feedback_tile(axis: plt.Axes) -> None:
    """Top-down measurement and correction, followed by the next mode."""
    _box(axis, (0.64, 2.92), 1.42, 0.56,
         "measure\n" + r"$\hat{\mathcal{N}}_{\boldsymbol{r},\nu,\sigma}(\alpha_{\boldsymbol{r}})$",
         facecolor=PALE_GRAY)
    _arrow(axis, (1.35, 2.908), (1.35, 2.65))
    axis.add_patch(Polygon([(1.35, 2.65), (1.85, 2.28), (1.35, 1.91), (0.85, 2.28)],
                           closed=True, facecolor="white", edgecolor=INK,
                           linewidth=0.9, zorder=3))
    axis.text(1.35, 2.28, r"$\mathrm{m}=s_\sigma$?", ha="center", va="center", zorder=4)

    # A correct occupation bypasses feedback; a mismatch enters the correction.
    _elbow(axis, [(0.85, 2.28), (0.25, 2.28), (0.25, 1.12), (1.35, 1.12), (1.35, 0.93)])
    axis.text(0.53, 2.39, "yes", ha="center", va="bottom")
    _elbow(axis, [(1.85, 2.28), (2.19, 2.28), (2.19, 1.862)])
    axis.text(2.30, 2.11, "no", ha="left", va="center")
    _box(axis, (1.76, 1.40), 0.86, 0.45, "fSWAP")

    # The ancilla enters laterally, keeping its wire separate from the output.
    axis.add_patch(Circle((3.08, 1.625), 0.13, facecolor=ANCILLA,
                          edgecolor=INK, linewidth=0.9, zorder=3))
    axis.text(3.08, 1.625, r"$s_\sigma$", ha="center", va="center", zorder=4)
    _arrow(axis, (2.95, 1.625), (2.632, 1.625))
    axis.text(3.08, 1.86, "fresh\nancilla", ha="center", va="bottom", linespacing=1.0)
    axis.text(1.02, 1.50, "$s_-=1$ (fill)\n$s_+=0$ (empty)",
              ha="center", va="center", linespacing=1.35)

    _elbow(axis, [(2.19, 1.388), (2.19, 0.93), (1.35, 0.93)])
    axis.add_patch(Circle((1.35, 0.93), 0.025, facecolor=INK, edgecolor="none", zorder=4))
    _arrow(axis, (1.35, 0.905), (1.35, 0.702))
    _box(axis, (0.72, 0.14), 1.26, 0.55, "measure next\nOW mode", facecolor=PALE_GRAY)
    _arrow(axis, (1.992, 0.415), (2.80, 0.415))
    axis.text(2.92, 0.415, r"$\cdots$", ha="left", va="center")
    axis.set_xlim(0.05, 3.39)
    axis.set_ylim(0.01, 3.61)
    axis.set_aspect("equal")
    axis.axis("off")


def _save(figure, stem):
    prepare_figure(figure, stem)
    record_typography(figure, stem)
    pdf = FIGURE_DIR / (stem + ".pdf")
    figure.savefig(pdf)
    plt.close(figure)
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile",
                    str(pdf), str(pdf.with_suffix(""))], check=True)


def build() -> None:
    _configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(3.375, 4.8))
    _draw_geometry(figure.add_axes([0.025, 0.025, 0.96, 0.96]))
    _save(figure, "Figure_01_schematic")
    figure = plt.figure(figsize=(3.375, 3.65))
    _draw_feedback_tile(figure.add_axes([0.015, 0.015, 0.97, 0.97]))
    _save(figure, "Figure_02_adaptive_circuit")


if __name__ == "__main__":
    build()
