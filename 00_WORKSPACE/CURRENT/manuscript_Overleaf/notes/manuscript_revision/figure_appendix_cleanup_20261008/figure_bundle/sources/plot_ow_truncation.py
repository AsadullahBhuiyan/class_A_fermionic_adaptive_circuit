#!/usr/bin/env python3
"""Render A1 with consistent typography from the exact native plot arrays.

No analytical recalculation, optimization, fitting, or dynamics occurs here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from log_ticks import add_log_minor_ticks
from manuscript_typography import configure_style as manuscript_style, prepare_figure, record_typography

OUT = Path(__file__).resolve().parents[1]
DATA = OUT / "data" / "ow_truncation"
STEM = "Figure_A01_ow_truncation"


def digest(path):
    return {"bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    provenance = json.loads((DATA / "provenance.json").read_text())
    for name, expected in provenance["bundle_inputs"].items():
        assert digest(DATA / name) == expected, name
    metadata = json.loads((DATA / "native_artists.json").read_text())
    with np.load(DATA / "plot_data.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    manuscript_style({'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black'})
    figure, axes = plt.subplots(2, 1, figsize=(3.375, 3.85))
    selected = [metadata[2], metadata[3]]
    rendered_lines = 0
    for axis, native in zip(axes, selected):
        axis.set_xscale(native["xscale"])
        axis.set_yscale(native["yscale"])
        for item in native["lines"]:
            x, y = arrays[item["prefix"] + "_x"], arrays[item["prefix"] + "_y"]
            style = {key: item[key] for key in (
                "label", "color", "marker", "linewidth", "markersize",
                "markerfacecolor", "markeredgecolor", "markeredgewidth", "markevery",
                "clip_on", "alpha", "zorder",
            )}
            offset, dash = item["dash_pattern"]
            style["linestyle"] = (offset, tuple(dash)) if dash else item["linestyle"]
            transform = {"data": axis.transData, "xaxis": axis.get_xaxis_transform(),
                         "yaxis": axis.get_yaxis_transform()}[item["transform"]]
            line, = axis.plot(x, y, transform=transform, **style)
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_array_equal(line.get_ydata(), y)
            rendered_lines += 1
        axis.set(xlim=native["xlim"], ylim=native["ylim"],
                 xlabel=native["xlabel"], ylabel=native["ylabel"],
                 xticks=[v for v in native["xticks"] if native["xlim"][0] <= v <= native["xlim"][1]],
                 yticks=[v for v in native["yticks"] if native["ylim"][0] <= v <= native["ylim"][1]])
        # Tick placement must not enlarge the original logarithmic panel limits.
        axis.set_xlim(native["xlim"])
        axis.set_ylim(native["ylim"])
        axis.tick_params(pad=1.5)
        axis.xaxis.labelpad = 2
        axis.yaxis.labelpad = 2
    assert rendered_lines == 4
    for axis in axes:
        axis.set_xticks([1, 4, 8, 12])
    axes[0].grid(alpha=.16, linewidth=.5)
    axes[1].set_ylabel("retained weight (\\%)")
    axes[0].legend(loc="upper right", frameon=False, handlelength=1.4)
    retained = float(arrays["axis3_line0_y"][0])
    axes[1].annotate("$w=1$"+"\n"+rf"${retained:.4f}\%$", xy=(1, retained),
                     xytext=(.96, .40), textcoords="axes fraction", ha="right",
                     fontsize=8, color="black",
                     arrowprops=dict(arrowstyle="-", color="#202020", linewidth=.5))
    for axis, letter in zip(axes, "ab"):
        axis.text(-.17, 1.045, f"({letter})", transform=axis.transAxes,
                  fontsize=9, ha="left", va="bottom", clip_on=False)
    add_log_minor_ticks(figure)
    prepare_figure(figure, STEM)
    figure.subplots_adjust(left=.19, right=.965, bottom=.12, top=.95, hspace=.55)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    clipped = []
    for artist in figure.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > figure.bbox.x1+1 or box.y1 > figure.bbox.y1+1:
                clipped.append(artist.get_text())
    assert not clipped, clipped
    record_typography(figure, STEM)
    for extension in ("pdf", "png"):
        figure.savefig(OUT / f"{STEM}.{extension}", dpi=300)
    plt.close(figure)
    validation = dict(
        status="passed", figure_size_inches=[3.375, 3.85], layout="2x1; native panels c,d", native_axis_indices=[2,3],
        scientific_line_arrays_equal_to_native=True, preserved_line_artists=rendered_lines,
        scientific_limits_styles_and_fits_unchanged=True, clipped_text=clipped,
        font_family="Computer Modern / AMS", math_fontset="LaTeX", panel_letters="plain 9 pt",
        axis_labels_and_ticks="9 pt axes, 8 pt ticks at manuscript width", compact_legend_sizes_pt=[8, 8],
        typography_spacing="Width ticks 1,4,8,12; two retained panels stacked",
        typography_only=False, optimization_refitting_and_dynamics=False,
        native_raster_comparison=provenance["native_raster_comparison"],
        outputs={extension: digest(OUT / f"{STEM}.{extension}") for extension in ("pdf", "png")},
    )
    (DATA / "typography_validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    print(OUT / f"{STEM}.png")


if __name__ == "__main__":
    main()
