from typing import Any
HARD_WALL_X_LEFT=5
HARD_WALL_X_RIGHT=15
INK="#20252B"
MID_GRAY="#707780"
LIGHT_GRAY="#ECECF1"
TOPOLOGICAL="#9BD0EA"
STRIP_A="#d62728"
STRIP_B="#1f77b4"
def draw_compact_geometry_axis(axis: Any) -> None:
    """Draw the BPJ-style opposite-strip geometry without auxiliary formulas."""

    from matplotlib.patches import Rectangle

    nx = 20.0
    x_left = float(HARD_WALL_X_LEFT)
    x_right = float(HARD_WALL_X_RIGHT)
    y0 = 0.08
    width = 0.25
    opposite_start = y0 + 0.5
    axis.add_patch(
        Rectangle((0, 0), nx, 1, facecolor=LIGHT_GRAY, edgecolor=MID_GRAY, lw=0.65)
    )
    axis.add_patch(
        Rectangle(
            (x_left, 0),
            x_right - x_left,
            1,
            facecolor=TOPOLOGICAL,
            edgecolor="none",
        )
    )
    axis.add_patch(
        Rectangle(
            (0, y0),
            nx,
            width,
            facecolor=STRIP_A,
            edgecolor=STRIP_A,
            alpha=0.36,
            lw=0.8,
        )
    )
    axis.add_patch(
        Rectangle(
            (0, opposite_start),
            nx,
            width,
            facecolor=STRIP_B,
            edgecolor=STRIP_B,
            alpha=0.36,
            lw=0.8,
        )
    )
    for wall_x in (x_left, x_right):
        axis.plot(
            (wall_x, wall_x),
            (0, 1),
            color="#173f5f",
            linewidth=1.0,
            solid_capstyle="butt",
            zorder=5,
        )
    axis.text(0.65, y0 + width / 2, r"$a$", color="#8c1515", fontsize=8, va="center")
    axis.text(
        0.65,
        opposite_start + width / 2,
        r"$b$",
        color="#174f83",
        fontsize=8,
        va="center",
    )
    axis.text(
        x_left / 2,
        0.455,
        "triv.",
        color=INK,
        fontsize=8,
        ha="center",
        va="center",
    )
    axis.text(
        (x_left + x_right) / 2,
        0.455,
        "top.",
        color=INK,
        fontsize=8,
        ha="center",
        va="center",
    )
    axis.text(
        (x_right + nx) / 2,
        0.455,
        "triv.",
        color=INK,
        fontsize=8,
        ha="center",
        va="center",
    )
    axis.set_xlim(-0.15, nx + 0.15)
    axis.set_ylim(-0.04, 1.10)
    axis.axis("off")
