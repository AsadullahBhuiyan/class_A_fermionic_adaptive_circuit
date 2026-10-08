#!/usr/bin/env python3
"""Tightly cropped, ten-column-slab version of the approved four-slice geometry."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
MANUSCRIPT = OUT.parents[1]
sys.path.insert(0, str(MANUSCRIPT / "figures/new_figure/sources"))
import manuscript_typography as typography
import plot_schematic as geometry

STEM = "domain_wall_four_step_small_lattice"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protected = {str(p.relative_to(MANUSCRIPT)): sha(p) for p in [
        MANUSCRIPT / "manuscript.tex", MANUSCRIPT / "manuscript.pdf",
        MANUSCRIPT / "figures/new_figure/manifest.json"]}
    # Preserve all sizes and lattice spacing from the approved standalone;
    # trim the canvas to the geometry and its labels.
    spacing = 7.5 * (.98 - .035) / 52.2
    width = float(np.diff(geometry.SLICE_XLIM)[0] * spacing)
    height = float(np.diff(geometry.SLICE_YLIM)[0] * spacing)
    typography.ROOT = OUT
    original_width = typography.inclusion_width
    typography.inclusion_width = lambda stem: width if stem == STEM else original_width(stem)

    def role(fig, artist, stem):
        prominent = (r"\alpha" in artist.get_text() or r"\mathcal" in artist.get_text()
                     or artist.get_text() == "time")
        return "schematic", 19 if prominent else 17

    typography.text_role = role
    geometry._configure_style()
    fig = plt.figure(figsize=(width, height))
    label, details = geometry._draw_four_slice_geometry(fig.add_axes([0, 0, 1, 1]))
    typography.prepare_figure(fig, STEM)
    geometry._center_operator_ink(fig, label)
    typography.record_typography(fig, STEM)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=300)
    plt.close(fig)
    review = typography.verify_typography(STEM)
    assert all(sha(MANUSCRIPT / p) == value for p, value in protected.items())
    assert details["slab_columns"] == 10
    assert details["exterior_columns"] == [5, 5]
    assert [s["retained_cells"] for s in details["stages_oldest_to_latest"]] == [9, 6, 6, 9]
    details.update(figure_inches=[width, height], outer_whitespace_trimmed=True,
                   plane_aspect_ratio=geometry.SLICE_WIDTH/(geometry.SLICE_TOP-geometry.SLICE_BOTTOM),
                   approved_plane_aspect_ratio=18/20,
                   previous_version="previous_eight_column_slab/",
                   typography=review, manuscript_unchanged_by_standalone_renderer=protected,
                   outputs={f"{STEM}.{ext}": sha(OUT / f"{STEM}.{ext}") for ext in ("pdf", "png")})
    (OUT / "validation.json").write_text(json.dumps(details, indent=2) + "\n")
    print(OUT / f"{STEM}.pdf")


if __name__ == "__main__":
    main()
