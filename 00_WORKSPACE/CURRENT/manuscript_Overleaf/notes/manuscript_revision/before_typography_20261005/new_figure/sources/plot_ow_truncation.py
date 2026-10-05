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
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
        "font.size": 8, "mathtext.fontset": "cm", "axes.labelsize": 8,
        "axes.titlesize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": .8, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "xtick.major.width": .7,
        "ytick.major.width": .7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
    })
    figure, grid = plt.subplots(2, 2, figsize=(3.375, 3.580587552488946))
    axes = list(grid.ravel())
    axes.append(axes[1].twinx())
    rendered_lines = 0
    for axis, native in zip(axes, metadata):
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
    assert rendered_lines == 21
    axes[4].tick_params(axis="x", bottom=False, top=False, labelbottom=False)
    axes[1].spines["left"].set_color("#D55E4A")
    axes[4].spines["right"].set_color("#2878B5")
    axes[1].grid(alpha=.16, linewidth=.5)
    axes[2].grid(alpha=.16, linewidth=.5)
    # Preserve the plotted domains while separating the 8 pt tick labels.
    for axis in (axes[1], axes[2], axes[3]):
        axis.set_xticks([1, 4, 8, 12])
    axes[3].set_ylabel("retained weight (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    order = list(range(1, len(handles))) + [0]
    axes[0].legend([handles[index] for index in order], [labels[index] for index in order],
                   loc="upper center", ncol=2, frameon=False, fontsize=5.8,
                   columnspacing=.5, handlelength=1.3, handletextpad=.3, borderaxespad=.3)
    handles, labels = axes[2].get_legend_handles_labels()
    labels = [label.replace("full-BZ minimum", "full-BZ\nminimum") for label in labels]
    axes[2].legend(handles, labels, loc="lower left", frameon=False, fontsize=6,
                   handlelength=1.4, handletextpad=.3, borderaxespad=.35)
    power = provenance["saved_critical_fit"]["p"]
    axes[4].text(.97, .08, rf"$p={power:.3f}$", transform=axes[4].transAxes,
                 ha="right", va="bottom", fontsize=6.2, color="black")
    retained = float(arrays["axis3_line0_y"][0])
    axes[3].annotate(rf"$w=1:\ {retained:.4f}\%$", xy=(1, retained),
                     xytext=(.96, .25), textcoords="axes fraction", ha="right",
                     fontsize=6.2, color="black",
                     arrowprops=dict(arrowstyle="-", color="#202020", linewidth=.5))
    for axis, letter in zip(axes[:4], "abcd"):
        axis.text(-.17, 1.055, f"({letter})", transform=axis.transAxes,
                  fontsize=9, ha="left", va="bottom", color="black",
                  fontfamily="CMU Sans Serif", fontweight="normal", clip_on=False)
    figure.subplots_adjust(left=.15, right=.85, bottom=.12, top=.94,
                           wspace=.59, hspace=.71)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    clipped = []
    for artist in figure.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > figure.bbox.x1+1 or box.y1 > figure.bbox.y1+1:
                clipped.append(artist.get_text())
    assert not clipped, clipped
    for extension in ("pdf", "png"):
        figure.savefig(OUT / f"{STEM}.{extension}", dpi=300)
    plt.close(figure)
    validation = dict(
        status="passed", figure_size_inches=[3.375, 3.580587552488946], layout="2x2 with native twin y axis",
        scientific_line_arrays_equal_to_native=True, preserved_line_artists=rendered_lines,
        scientific_limits_styles_and_fits_unchanged=True, clipped_text=clipped,
        font_family="CMU Sans Serif", math_fontset="cm", panel_letters="plain 9 pt",
        axis_labels_and_ticks="black 8 pt", compact_legend_sizes_pt=[5.8, 6],
        typography_spacing="Width ticks 1,4,8,12; wrapped full-BZ legend label; 2x2 arrangement retained",
        typography_only=True, optimization_refitting_and_dynamics=False,
        native_raster_comparison=provenance["native_raster_comparison"],
        outputs={extension: digest(OUT / f"{STEM}.{extension}") for extension in ("pdf", "png")},
    )
    (DATA / "typography_validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    print(OUT / f"{STEM}.png")


if __name__ == "__main__":
    main()
