#!/usr/bin/env python3
"""Build separate single-column geometry and adaptive-circuit schematics."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
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


def _draw_feedback_tile(axis: plt.Axes) -> None:
    """Shared centerline, equal measurement boxes, and a plain bypass junction."""
    # All positions derive from a small set of common alignment anchors.
    spine = 1.30
    box_width, box_height = 1.42, 0.56
    gap, stroke, padding = 0.25, 0.9, 0.012
    final_bottom = 0.14
    final_top = final_bottom + box_height
    junction_y = final_top + gap
    swap_bottom = junction_y + gap
    swap_width, swap_height = 0.96, 0.46
    swap_top = swap_bottom + swap_height
    swap_y = (swap_bottom + swap_top)/2
    diamond_bottom = swap_top + gap
    diamond_height, diamond_width = 0.74, 1.00
    diamond_top = diamond_bottom + diamond_height
    decision_y = (diamond_bottom + diamond_top)/2
    measure_bottom = diamond_top + gap
    box_left = spine - box_width/2
    bypass_x = 0.20
    ancilla_x, ancilla_radius = 2.78, 0.13
    label_gap = 0.11

    _box(axis, (box_left, measure_bottom), box_width, box_height,
         "measure\n" + r"$\hat{\mathcal{N}}_{\boldsymbol{r},\nu,\sigma}(\alpha_{\boldsymbol{r}})$",
         facecolor=PALE_GRAY, linewidth=stroke)
    _arrow(axis, (spine, measure_bottom-padding), (spine, diamond_top))
    axis.add_patch(Polygon([(spine, diamond_top), (spine+diamond_width/2, decision_y),
                           (spine, diamond_bottom), (spine-diamond_width/2, decision_y)],
                          closed=True, facecolor="white", edgecolor=INK,
                          linewidth=stroke, zorder=3))
    axis.text(spine, decision_y, r"$\mathrm{m}=s_\sigma$?", ha="center", va="center", zorder=4)

    # Matching outcomes bypass the reset. No arrowhead or dot crowds the T-junction.
    axis.plot([spine-diamond_width/2, bypass_x, bypass_x, spine],
              [decision_y, decision_y, junction_y, junction_y],
              color=INK, linewidth=stroke, solid_joinstyle="miter", solid_capstyle="butt")
    axis.text((spine-diamond_width/2+bypass_x)/2, decision_y+label_gap,
              "yes", ha="center", va="bottom")
    _arrow(axis, (spine, diamond_bottom), (spine, swap_top+padding))
    axis.text(spine+label_gap, (diamond_bottom+swap_top)/2,
              "no", ha="left", va="center")
    _box(axis, (spine-swap_width/2, swap_bottom), swap_width, swap_height,
         "fSWAP", linewidth=stroke)

    # The ancilla, its labels, and the incoming wire form one aligned group.
    axis.add_patch(Circle((ancilla_x, swap_y), ancilla_radius, facecolor=ANCILLA,
                          edgecolor=INK, linewidth=stroke, zorder=3))
    axis.text(ancilla_x, swap_y, r"$s_\sigma$", ha="center", va="center", zorder=4)
    _arrow(axis, (ancilla_x-ancilla_radius, swap_y),
           (spine+swap_width/2+padding, swap_y))
    axis.text(ancilla_x, swap_y+ancilla_radius+label_gap, "fresh\nancilla",
              ha="center", va="bottom", linespacing=1.0)
    axis.text(ancilla_x, swap_y-ancilla_radius-label_gap,
              "$s_-=1$ (fill)\n$s_+=0$ (empty)", ha="center", va="top", linespacing=1.35)

    # A single arrow passes through the merge and enters the next measurement.
    _arrow(axis, (spine, swap_bottom-padding), (spine, final_top+padding))
    _box(axis, (box_left, final_bottom), box_width, box_height,
         "measure next\nOW mode", facecolor=PALE_GRAY, linewidth=stroke)
    final_y = final_bottom + box_height/2
    continuation_tip = 2.91
    _arrow(axis, (spine+box_width/2+padding, final_y), (continuation_tip, final_y))
    axis.text(continuation_tip+label_gap, final_y, r"$\cdots$", ha="left", va="center")
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


def build(only=None) -> None:
    _configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if only in (None, "geometry"):
        figure = plt.figure(figsize=(3.375, 4.8))
        _draw_geometry(figure.add_axes([0.025, 0.025, 0.96, 0.96]))
        _save(figure, "Figure_01_schematic")
    if only in (None, "circuit"):
        figure = plt.figure(figsize=(3.375, 3.65))
        _draw_feedback_tile(figure.add_axes([0.015, 0.015, 0.97, 0.97]))
        _save(figure, "Figure_02_adaptive_circuit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("geometry", "circuit"))
    build(parser.parse_args().only)
