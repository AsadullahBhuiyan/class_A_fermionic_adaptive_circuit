#!/usr/bin/env python3
"""Render Figure 03 from a portable, previously verified analysis cache.

This script performs no dynamics. Paths resolve relative to this bundle.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Rectangle, Wedge
import numpy as np
from log_ticks import add_log_minor_ticks
from manuscript_typography import configure_style as manuscript_style, prepare_figure, record_typography


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bulk_topology"
STEM = "Figure_03_bulk_topology"


def digest(path):
    return {"bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def absolute_interval(mean, sem):
    """Image of [mean-sem, mean+sem] under x -> abs(x-1)."""
    delta = np.abs(mean - 1)
    lower = np.where(delta <= sem, 0,
                     np.minimum(abs(mean-sem-1), abs(mean+sem-1)))
    upper = np.maximum(abs(mean-sem-1), abs(mean+sem-1))
    return delta, lower, upper


def load_and_validate():
    provenance = json.loads((DATA / "provenance.json").read_text())
    verified = []
    for source in provenance["sources"]:
        if "bundle_path" not in source:
            continue
        path = ROOT / source["bundle_path"]
        assert digest(path) == {k: source[k] for k in ("bytes", "sha256")}, path
        verified.append(source["bundle_path"])
    with np.load(DATA / "plot_data.npz", allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    np.testing.assert_array_equal(arrays["cycles"], np.arange(41))
    np.testing.assert_array_equal(arrays["sizes"], [20, 30, 40])
    np.testing.assert_array_equal(arrays["marker_sample_ids"], np.arange(100))
    assert arrays["means"].shape == arrays["sems"].shape == (3, 41)
    assert arrays["marker_samples"].shape == (100, 30, 30)
    assert all(np.isfinite(v).all() for v in arrays.values())
    np.testing.assert_allclose(arrays["marker"], arrays["marker_samples"].mean(axis=0),
                               atol=1e-14, rtol=0)
    np.testing.assert_allclose(arrays["marker_sem"],
                               arrays["marker_samples"].std(axis=0, ddof=1)/10,
                               atol=1e-14, rtol=0)
    with (DATA / "cycles.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 123
    for row in rows:
        index = list(arrays["sizes"]).index(int(row["L"]))
        t = int(row["cycle"])
        assert int(row["samples"]) == 100
        np.testing.assert_allclose([float(row["mean"]), float(row["sem"])],
                                   [arrays["means"][index,t], arrays["sems"][index,t]],
                                   atol=1e-15, rtol=0)
    with (DATA / "endpoint_marker.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 900
    for row in rows:
        x, y = int(row["x"]), int(row["y"])
        np.testing.assert_allclose(float(row["local_chern_marker_mean"]),
                                   arrays["marker"][x,y], atol=1e-15, rtol=0)
    source_summary = json.loads((DATA / "source_summary.json").read_text())
    assert [row["samples"] for row in source_summary["results"]] == [100]*3
    assert source_summary["marker"]["samples"] == 100
    assert source_summary["config"]["cycles"] == 40
    np.testing.assert_allclose(arrays["means"][:,-1],
                               [row["endpoint_mean"] for row in source_summary["results"]],
                               atol=0, rtol=0)
    return arrays, verified, source_summary


def draw_geometry(ax):
    """Actual integer coordinates; the center's y coordinate is illustrative."""
    length, left, right = 30, 8, 22
    center, radius = (15, 15), 6
    assert left < center[0]-radius and center[0]+radius < right
    ax.add_patch(Rectangle((-.5, -.5), length, length,
                           facecolor="#ECECF1", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((left, -.5), right-left, length,
                           facecolor="#9BD0EA", edgecolor="none", zorder=0))
    for begin, end, color in ((0,120,"#DCEAF7"), (120,240,"#F8E1D5"),
                               (240,360,"#DDEFE5")):
        ax.add_patch(Wedge(center, radius, begin, end, facecolor=color,
                           edgecolor=".25", linewidth=.65, zorder=1))
    xx, yy = np.meshgrid(np.arange(length), np.arange(length))
    ax.scatter(xx.ravel(), yy.ravel(), s=.8, color="#404448", alpha=.65,
               linewidths=0, zorder=2)
    for wall in (left, right):
        ax.plot([wall,wall], [-.5,29.5], color="#164A7B", lw=1.15, zorder=3)
    ax.add_patch(Rectangle((-.5,-.5), length, length, facecolor="none",
                           edgecolor="#707780", linewidth=.9, zorder=3))
    # Matching marks identify the two periodic edge pairs of the torus.
    # Offset the top mark from alpha_1 so both stay readable at column width.
    for edge_y in (-.5,29.5):
        ax.plot([17.4,19.6], [edge_y-.9,edge_y+.9], color="black",
                linewidth=1.5, clip_on=False, zorder=7, solid_capstyle="butt")
    for edge_x in (-.5,29.5):
        for edge_y in (13.7,15.3):
            ax.plot([edge_x-.8,edge_x+.8], [edge_y-1.1,edge_y+1.1],
                    color="black", linewidth=1.5, clip_on=False,
                    zorder=7, solid_capstyle="butt")
    ax.add_patch(Circle(center, .23, facecolor="black", edgecolor="none", zorder=5))
    for label, x, y in (("A",17.1,18.3),("B",11.6,15),("C",17.1,11.7)):
        ax.text(x,y,rf"${label}$",ha="center",va="center",fontsize=9,
                zorder=6,bbox=dict(facecolor="white",edgecolor="none",alpha=.72,pad=.15))
    for x, label in ((3.75,r"$\alpha_2$"),(15,r"$\alpha_1$"),(25.75,r"$\alpha_2$")):
        ax.text(x,31.4,label,ha="center",va="center",fontsize=9)
    ax.set(xlim=(-.5,29.5),ylim=(-.5,29.5),aspect="equal")
    ax.axis("off")


def main():
    values, verified, source_summary = load_and_validate()
    manuscript_style({'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.linewidth': 0.75, 'xtick.direction': 'in', 'ytick.direction': 'in', 'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    width, height = 3.95, 4.2
    fig = plt.figure(figsize=(width, height))
    # Physical-inch anchors keep the two square panels aligned and reserve
    # room for the vertical colorbar without reducing the typography.
    def axes_inches(left, bottom, panel_width, panel_height):
        return fig.add_axes([left/width, bottom/height,
                             panel_width/width, panel_height/height])
    top_bottom, square = 2.80, 1.08
    convergence_left, convergence_width = .15*3.375 + .03, .82*3.375
    center = convergence_left + convergence_width/2
    upper_gap = .46
    upper_left = center - (2*square + upper_gap)/2
    map_left = upper_left + square + upper_gap
    geometry_left = convergence_left
    geometry = axes_inches(geometry_left, top_bottom, square, square)
    spatial = axes_inches(map_left, top_bottom, square, square)
    ax = axes_inches(convergence_left, .50, convergence_width, 1.75)
    draw_geometry(geometry)
    floor = 1e-6
    cycles, sizes = values["cycles"], values["sizes"]
    inset = ax.inset_axes([.48,.78,.48,.20])
    interval_zero_count = 0
    for index,(length,color,mark,ls) in enumerate(zip(
            sizes,("#d94738","#249b57","#1675bd"),("^","s","o"),(":","--","-"))):
        mean,sem = values["means"][index],values["sems"][index]
        delta,lower,upper = absolute_interval(mean,sem)
        interval_zero_count += int((lower==0).sum())
        ax.plot(cycles,delta,color=color,marker=mark,ls=ls,lw=.9,ms=2.1,
                mfc="white",mew=.55,label=rf"$L={length}$")
        ax.fill_between(cycles,np.maximum(lower,floor),upper,color=color,alpha=.15,lw=0)
        inset.plot(cycles,mean,color=color,ls=ls,lw=.85)
        inset.fill_between(cycles,mean-sem,mean+sem,color=color,alpha=.15,lw=0)
    ax.set(xlim=(0,40),ylim=(floor,1.5),yscale="log",xlabel="cycle",
           ylabel=r"$|\overline{\mathcal{C}_G}-1|$",xticks=[0,20,40])
    ax.set_yticks([1e0,1e-2,1e-4,1e-6])
    ax.legend(loc="upper left",fontsize=8,handlelength=1.4,labelspacing=.17,
               borderaxespad=.3,handletextpad=.35)
    inset.axhline(1,color=".4",ls="--",lw=.6)
    inset.set(xlim=(0,40),ylim=(-.04,1.08),xticks=[0,40],yticks=[0,1])
    inset.tick_params(labelsize=8,pad=1,top=True,right=True,length=2)
    inset.set_ylabel(r"$\overline{\mathcal{C}_G}$", labelpad=1)
    inset.set_xlabel("cycle", labelpad=0)
    marker = values["marker"]
    mmin,mmax = float(marker.min()),float(marker.max())
    image = spatial.imshow(np.tanh(marker.T),origin="lower",extent=(-.5,29.5,-.5,29.5),
                            cmap="RdBu_r",norm=Normalize(vmin=-1,vmax=1),
                            interpolation="nearest",aspect="equal")
    for wall in (8,22):
        spatial.axvline(wall,color="k",ls="--",lw=.75)
    spatial.set(xlabel=r"$x$",ylabel=r"$y$",xticks=[0,15,29],yticks=[0,15,29])
    colorbar_ax = axes_inches(map_left + square + .08, top_bottom, .055, square)
    cb = fig.colorbar(image,cax=colorbar_ax,orientation="vertical",ticks=[-1,0,1])
    cb.ax.set_yticklabels(["$-1$","$0$","$1$"])
    cb.ax.tick_params(labelsize=8,pad=2,length=2)
    cb.set_label(r"$\tanh[\overline{C}(\boldsymbol{r})]$",fontsize=9,labelpad=3)
    for axis in (ax,spatial):
        axis.tick_params(top=True,right=True,pad=2)
    for x,y,label in ((geometry_left-.12,4.15,"a"),(map_left-.26,4.15,"b"),(.05,2.43,"c")):
        fig.text(x/width,y/height,f"({label})",fontsize=9,va="top")
    add_log_minor_ticks(fig)
    prepare_figure(fig, STEM)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for axis in (ax, spatial, colorbar_ax, inset):
        bounds = axis.get_tightbbox(renderer)
        assert bounds.x0 >= 0 and bounds.y0 >= 0, (axis.get_label(), bounds)
        assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
    assert spatial.get_tightbbox(renderer).x1 < colorbar_ax.bbox.x0, "Map labels overlap colorbar"
    np.testing.assert_allclose(geometry.bbox.extents[[1,3]], spatial.bbox.extents[[1,3]], atol=1e-8)
    upper_center = (geometry.bbox.x0 + spatial.bbox.x1)/2
    convergence_center = (ax.bbox.x0 + ax.bbox.x1)/2
    np.testing.assert_allclose(geometry.bbox.x0, ax.bbox.x0, atol=1e-8)
    panel_gap = min(spatial.get_tightbbox(renderer).y0, colorbar_ax.get_tightbbox(renderer).y0) - ax.get_tightbbox(renderer).y1
    assert panel_gap >= 6, "Upper panels and convergence plot are too close"
    inset_bounds = inset.get_tightbbox(renderer)
    inset_clearances = []
    for mean, sem in zip(values["means"], values["sems"]):
        _, _, upper = absolute_interval(mean, sem)
        displayed_upper = ax.transData.transform(np.column_stack([cycles, upper]))
        under_inset = ((displayed_upper[:, 0] >= inset_bounds.x0) &
                       (displayed_upper[:, 0] <= inset_bounds.x1))
        inset_clearances.append(float(inset_bounds.y0 - displayed_upper[under_inset, 1].max()))
    assert min(inset_clearances) >= 6, ("Inset touches the convergence uncertainty band", inset_clearances, inset_bounds)
    np.testing.assert_allclose(colorbar_ax.bbox.extents[[1,3]], spatial.bbox.extents[[1,3]], atol=1e-8)
    layout_metrics = {
        "convergence_axes_inches": [float(ax.bbox.width / fig.dpi), float(ax.bbox.height / fig.dpi)],
        "marker_map_inches": [float(spatial.bbox.width / fig.dpi), float(spatial.bbox.height / fig.dpi)],
        "colorbar_inches": [float(colorbar_ax.bbox.width / fig.dpi), float(colorbar_ax.bbox.height / fig.dpi)],
        "inset_to_uncertainty_clearance_inches": [gap / fig.dpi for gap in inset_clearances],
        "upper_row_to_convergence_clearance_inches": float(panel_gap / fig.dpi),
        "text_within_canvas": True,
        "upper_panel_edges_aligned": True,
        "geometry_to_convergence_left_offset_inches": float((geometry.bbox.x0-ax.bbox.x0)/fig.dpi),
        "colorbar_orientation": "vertical",
    }
    # Verify rendering keeps the exact complete-S100 scientific arrays.
    np.testing.assert_array_equal(image.get_array(),np.tanh(marker.T))
    assert image.norm.vmin == -1 and image.norm.vmax == 1
    for line,mean in zip(ax.lines,values["means"]):
        np.testing.assert_array_equal(line.get_ydata(),np.abs(mean-1))
    record_typography(fig, STEM)
    for ext in ("pdf","png"):
        fig.savefig(ROOT/f"{STEM}.{ext}",dpi=300)
    plt.close(fig)
    validation = {
        "figure":STEM,"checks_passed":True,
        "checks":["Every copied input matches its original SHA-256 receipt",
                  "123 cycle mean/SEM pairs equal original CSV values",
                  "900 local marker values equal original CSV values",
                  "Marker mean and SEM recomputed from all 100 per-trajectory maps",
                  "All three size ensembles contain 100 trajectories",
                  "Plotted lines equal source values; displayed map equals tanh of the source trajectory mean",
                  "Illustrative R=6 disk lies strictly inside walls x=8 and x=22",
                  "Inset clears convergence curves and uncertainty bands",
                  "Panel labels fit canvas; square upper panels align; vertical colorbar clears map; compact gap above full-width convergence"],
        "verified_inputs":verified,
        "figure_inches":[width,height],"layout":[2,1],"top_row":["geometry","marker"],"bottom_row":"convergence","png_dpi":300,
        "layout_metrics":layout_metrics,
        "periodic_edge_marks":{"top_bottom":"single slash","left_right":"double slash"},
        "samples_per_size":100,"sizes":[20,30,40],"cycles":[0,40],
        "endpoint_means":values["means"][:,-1].tolist(),
        "endpoint_sems":values["sems"][:,-1].tolist(),
        "marker_range":[mmin,mmax],"marker_color_norm":"tanh of trajectory mean, linear color scale [-1,1]",
        "displayed_marker_range":[float(np.tanh(marker).min()),float(np.tanh(marker).max())],
        "uncertainty":"One sample SEM after averaging ten centers within each trajectory",
        "log_display_floor":floor,"zero_lower_uncertainty_endpoints":interval_zero_count,
        "source_canonical_marker_max_error":source_summary["marker"]["checks"][0]["canonical_max_error"],
        "source_disk_estimator_sample0_max_error":source_summary["marker"]["disk_estimator_sample0_max_error"],
        "renderer":digest(Path(__file__)),
        "outputs":{f"{STEM}.{ext}":digest(ROOT/f"{STEM}.{ext}") for ext in ("pdf","png")},
    }
    (DATA/"validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    print(json.dumps({"outputs":list(validation["outputs"]),"checks_passed":True},indent=2))


if __name__ == "__main__":
    main()
