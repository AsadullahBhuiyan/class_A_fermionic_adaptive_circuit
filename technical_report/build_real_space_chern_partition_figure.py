#!/usr/bin/env python3
"""Build the three-sector real-space Chern partition schematic."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge


ROOT = Path(__file__).resolve().parent
PDF = ROOT / "figures" / "real_space_chern_partition.pdf"
PNG = ROOT / "figures" / "real_space_chern_partition.png"


def main() -> int:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(3.375, 2.20))
    radius = 1.0
    sectors = (
        (0, 120, "#DCEAF7", "..", r"$A$", (0.43, 0.56)),
        (120, 240, "#F8E1D5", "..", r"$B$", (-0.62, 0.0)),
        (240, 360, "#DDEFE5", "..", r"$C$", (0.43, -0.56)),
    )
    for theta1, theta2, color, hatch, label, label_xy in sectors:
        ax.add_patch(
            Wedge(
                (0, 0),
                radius,
                theta1,
                theta2,
                facecolor=color,
                edgecolor="0.25",
                linewidth=0.65,
                hatch=hatch,
            )
        )
        ax.text(*label_xy, label, ha="center", va="center", fontsize=12)

    # Keep the common trijunction visible above the hatch edges.
    ax.add_patch(Circle((0, 0), 0.035, facecolor="black", edgecolor="none", zorder=6))

    ax.set_aspect("equal")
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.08, 1.08)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF)
    fig.savefig(PNG, dpi=300)
    plt.close(fig)
    print(PDF)
    print(PNG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
