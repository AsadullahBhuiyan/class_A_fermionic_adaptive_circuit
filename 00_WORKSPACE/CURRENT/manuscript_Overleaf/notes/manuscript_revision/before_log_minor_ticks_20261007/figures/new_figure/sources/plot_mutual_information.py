#!/usr/bin/env python3
"""Figure A2: saved all-origin entropy contours and the unchanged MI sweep.

No state construction, diagonalization, circuit evolution, or refitting occurs.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.ticker import FixedLocator
import numpy as np

from mi_geometry import draw_compact_geometry_axis
from manuscript_typography import configure_style as manuscript_style, prepare_figure, record_typography

OUT = Path(__file__).resolve().parents[1]
DATA = OUT / "data/mutual_information"


def main():
    manifest = json.loads((DATA / "analysis_manifest_y0avg.json").read_text())
    contour_path = DATA / "contours_y0avg.npz"
    assert hashlib.sha256(contour_path.read_bytes()).hexdigest() == manifest["data_sha256"]
    assert manifest["origin_average"] and manifest["origin_count"] == 32
    assert manifest["samples_per_alpha"] == 100 and manifest["endpoint_cycle"] == 64
    maps, totals = {}, {}
    with np.load(contour_path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["y0_values"], np.arange(32))
        np.testing.assert_array_equal(z["sample_ids"], np.arange(100))
        for alpha in (1, 3):
            maps[alpha] = z[f"alpha1_{alpha}_mean"]
            samples = z[f"alpha1_{alpha}_per_sample"]
            assert samples.shape == (100, 16, 20)
            np.testing.assert_allclose(maps[alpha], samples.mean(0), atol=1e-15, rtol=1e-14)
            np.testing.assert_allclose(samples.sum((1, 2)), z[f"alpha1_{alpha}_entropy"], atol=1e-12)
            totals[alpha] = float(maps[alpha].sum())
    with (DATA / "hard_wall_nshell1_sample_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 63
    assert all(int(r["samples"]) == 100 and int(r["cycles"]) == 2 * int(r["Ny"]) for r in rows)

    manuscript_style({'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.linewidth': 0.8, 'pdf.fonttype': 42, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})
    fig = plt.figure(figsize=(3.375, 4.8))
    map_axes = [fig.add_axes([.15, .704, .365, .235]),
                fig.add_axes([.60, .704, .365, .235])]
    norm = PowerNorm(gamma=.5, vmin=0, vmax=.4416719467194624)
    for ax, alpha in zip(map_axes, (1, 3)):
        # pcolormesh keeps every cell vector-valued in the PDF.
        mesh = ax.pcolormesh(np.arange(21)-.5, np.arange(17)-.5, maps[alpha],
                            cmap="Blues", norm=norm, shading="flat",
                            edgecolors=(.65, .65, .65, .35), linewidth=.12,
                            rasterized=False)
        ax.set(xlim=(-.5,19.5), ylim=(-.5,15.5), aspect="equal", xlabel="$x$",
               xticks=[0,5,15,19], yticks=[0,5,10,15])
        ax.set_title(rf"$\alpha_1={alpha}$", pad=4)
        ax.tick_params(length=2.5, pad=2)
    map_axes[0].set_ylabel(r"$\delta y$", labelpad=1)
    map_axes[1].tick_params(labelleft=False)
    cax = fig.add_axes([.245, .632, .625, .018])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[0,.1,.3,.4416719467194624])
    cb.ax.set_xticklabels(["0.00", "0.10", "0.30", "0.44"])
    cb.ax.xaxis.set_major_locator(FixedLocator([0,.1,.3,.4416719467194624]))
    cb.set_label(r"$\overline{s_1}(x,\delta y)$", labelpad=1)
    cb.ax.tick_params(length=2, pad=1)
    fig.text(.025, .955, "(a)", fontsize=9, ha="left", va="top")

    ax = fig.add_axes([.15, .095, .815, .425])
    styles = [(20,"#d62728","^",":"),(24,"#2ca02c","s","--"),(28,"#1f77b4","o","-")]
    plotted = []
    for ny, color, marker, style in styles:
        subset = sorted([r for r in rows if int(r["Ny"]) == ny], key=lambda r:float(r["alpha_1"]))
        assert len(subset) == 21 and all(int(r["width"]) == ny//4 for r in subset)
        x, means, sems = [np.array([float(r[key]) for r in subset]) for key in
                          ("alpha_1", "mutual_information_mean", "mutual_information_sem")]
        curve = ax.errorbar(x, means, yerr=sems, color=color, marker=marker, ls=style,
                           lw=1.1, ms=3.3, mew=.55, elinewidth=.65, capsize=1.4,
                           label=rf"$N_y={ny},\ \ell={ny//4}$")
        np.testing.assert_array_equal(curve.lines[0].get_xdata(), x)
        np.testing.assert_array_equal(curve.lines[0].get_ydata(), means)
        plotted.extend(subset)
    ax.axhline(np.log(2)/3, color="#7F6F91", ls=(0,(4,2.5)), lw=1,
               label=r"$(\log 2)/3$", zorder=0)
    ax.set(xlim=(.98,3.02), ylim=(-.015,.675), xlabel=r"$\alpha_1$",
           ylabel=r"$\overline{I}_{a,b}$", xticks=[1,1.5,2,2.5,3], yticks=[0,.2,.4,.6])
    ax.legend(frameon=False, loc="upper right", handlelength=2.1, labelspacing=.3,
              borderaxespad=.4, handletextpad=.5)
    ax.text(-.15,1.06,"(b)",transform=ax.transAxes,fontsize=9,ha="left",va="bottom")
    inset = ax.inset_axes([.055,.035,.34,.30], zorder=6)
    inset.set_facecolor("white")
    draw_compact_geometry_axis(inset)

    prepare_figure(fig, "Figure_A02_mutual_information")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box=artist.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1+1 or box.y1 > fig.bbox.y1+1:
                outside.append(artist.get_text())
    assert not outside, outside
    stem = "Figure_A02_mutual_information"
    record_typography(fig, "Figure_A02_mutual_information")
    for ext in ("pdf","png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300)
    plt.close(fig)
    validation = dict(status="passed", figure_size_inches=[3.375,4.8],
        contour_estimator=manifest["estimator"], contour_origins=32,
        contour_trajectories_per_alpha=100, contour_endpoint=64,
        contour_shape=[16,20], contour_sum=totals, gamma=.5,
        vmin=0, vmax=.4416719467194624, mi_points=len(plotted), mi_data_changed=False,
        mi_independent_trajectories_per_point=100, mi_uncertainty="trajectory SEM",
        clipped_text=outside, display_changes="Two-row composition; strip width relabeled ell; no refits")
    (DATA / "composition_validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    print(OUT / f"{stem}.png")


if __name__ == "__main__":
    main()
