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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle, Rectangle, Wedge
import numpy as np


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
    ax.add_patch(Circle(center, .23, facecolor="black", edgecolor="none", zorder=5))
    for label, x, y in (("A",17.1,18.3),("B",11.6,15),("C",17.1,11.7)):
        ax.text(x,y,rf"${label}$",ha="center",va="center",fontsize=9,
                zorder=6,bbox=dict(facecolor="white",edgecolor="none",alpha=.72,pad=.15))
    for x, label in ((3.75,r"$\alpha_2$"),(15,r"$\alpha_1$"),(25.75,r"$\alpha_2$")):
        ax.text(x,31.4,label,ha="center",va="center",fontsize=9)
    ax.text(8,-2.2,r"$x_L=8$",ha="center",va="top",fontsize=8)
    ax.text(22,-2.2,r"$x_R=22$",ha="center",va="top",fontsize=8)
    ax.set(xlim=(-1,30),ylim=(-.75,30.25),aspect="equal")
    ax.axis("off")


def main():
    values, verified, source_summary = load_and_validate()
    plt.rcParams.update({"font.family":"CMU Sans Serif", "font.size":8,
                         "text.usetex":False,"font.weight":"normal","text.color":"black",
                         "axes.labelweight":"normal","axes.labelcolor":"black",
                         "xtick.color":"black","ytick.color":"black",
                         "mathtext.fontset":"cm", "axes.linewidth":.75,
                         "xtick.direction":"in", "ytick.direction":"in",
                         "xtick.labelsize":8,"ytick.labelsize":8,
                         "legend.frameon":False,"pdf.fonttype":42,"ps.fonttype":42})
    fig = plt.figure(figsize=(3.375,7.0))
    geometry = fig.add_axes([.17,.773,.78,.207])
    ax = fig.add_axes([.21,.452,.745,.287])
    spatial = fig.add_axes([.21,.120,.745,.250])
    draw_geometry(geometry)
    floor = 1e-6
    cycles, sizes = values["cycles"], values["sizes"]
    inset = ax.inset_axes([.48,.69,.48,.27])
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
    inset.text(.49,.49,r"$\overline{\mathcal{C}_G}$",transform=inset.transAxes,
                fontsize=9,ha="center")
    inset.text(.60,.12,"cycle",transform=inset.transAxes,fontsize=8,ha="center")
    marker = values["marker"]
    mmin,mmax = float(marker.min()),float(marker.max())
    image = spatial.imshow(marker.T,origin="lower",extent=(-.5,29.5,-.5,29.5),
                            cmap="RdBu_r",norm=TwoSlopeNorm(vmin=mmin,vcenter=0,vmax=mmax),
                            interpolation="nearest",aspect="equal")
    for wall in (8,22):
        spatial.axvline(wall,color="k",ls="--",lw=.75)
    spatial.set(xlabel=r"$x$",ylabel=r"$y$",xticks=[0,15,29],yticks=[0,15,29])
    spatial.set_title(r"$30\times30$, cycle 40",fontsize=8,pad=5)
    colorbar_ax = fig.add_axes([.3525,.055,.46,.010])
    cb = fig.colorbar(image,cax=colorbar_ax,orientation="horizontal",ticks=[mmin,0,mmax])
    cb.ax.set_xticklabels([f"{mmin:.2f}","0",f"{mmax:.2f}"])
    cb.ax.tick_params(labelsize=8,pad=2,length=2)
    cb.set_label(r"$\overline{C(\boldsymbol{r})}$",fontsize=9,labelpad=0)
    for axis in (ax,spatial):
        axis.tick_params(top=True,right=True,pad=2)
    for y,label in ((.995,"a"),(.764,"b"),(.405,"c")):
        fig.text(.015,y,f"({label})",fontsize=9,va="top")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for axis in (ax, spatial, colorbar_ax, inset):
        bounds = axis.get_tightbbox(renderer)
        assert bounds.x0 >= 0 and bounds.y0 >= 0, (axis.get_label(), bounds)
        assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
    assert spatial.get_tightbbox(renderer).y0 > colorbar_ax.get_tightbbox(renderer).y1, "Map labels overlap colorbar"
    panel_gap = ax.get_tightbbox(renderer).y0 - spatial.get_tightbbox(renderer).y1
    assert panel_gap >= 6, "Cycle label and marker-map title are too close"
    inset_bounds = inset.get_tightbbox(renderer)
    inset_clearances = []
    for mean, sem in zip(values["means"], values["sems"]):
        _, _, upper = absolute_interval(mean, sem)
        displayed_upper = ax.transData.transform(np.column_stack([cycles, upper]))
        under_inset = ((displayed_upper[:, 0] >= inset_bounds.x0) &
                       (displayed_upper[:, 0] <= inset_bounds.x1))
        inset_clearances.append(float(inset_bounds.y0 - displayed_upper[under_inset, 1].max()))
    assert min(inset_clearances) >= 6, ("Inset touches the convergence uncertainty band", inset_clearances, inset_bounds)
    assert colorbar_ax.bbox.width < spatial.bbox.width, "Colorbar exceeds map width"
    layout_metrics = {
        "panel_b_axes_inches": [float(ax.bbox.width / fig.dpi), float(ax.bbox.height / fig.dpi)],
        "marker_map_inches": [float(spatial.bbox.width / fig.dpi), float(spatial.bbox.height / fig.dpi)],
        "colorbar_inches": [float(colorbar_ax.bbox.width / fig.dpi), float(colorbar_ax.bbox.height / fig.dpi)],
        "inset_to_uncertainty_clearance_inches": [gap / fig.dpi for gap in inset_clearances],
        "panel_b_to_c_text_clearance_inches": float(panel_gap / fig.dpi),
        "text_within_canvas": True,
    }
    # Verify rendering keeps the exact complete-S100 scientific arrays.
    np.testing.assert_array_equal(image.get_array(),marker.T)
    for line,mean in zip(ax.lines,values["means"]):
        np.testing.assert_array_equal(line.get_ydata(),np.abs(mean-1))
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
                  "Plotted lines and image array equal source values exactly",
                  "Illustrative R=6 disk lies strictly inside walls x=8 and x=22",
                  "Inset clears convergence curves and uncertainty bands",
                  "Panel labels fit canvas; map title clears cycle label; colorbar narrower than map"],
        "verified_inputs":verified,
        "figure_inches":[3.375,7.0],"layout":[3,1],"png_dpi":300,
        "layout_metrics":layout_metrics,
        "samples_per_size":100,"sizes":[20,30,40],"cycles":[0,40],
        "endpoint_means":values["means"][:,-1].tolist(),
        "endpoint_sems":values["sems"][:,-1].tolist(),
        "marker_range":[mmin,mmax],"marker_color_norm":"TwoSlopeNorm centered at zero; separate linear ranges; unchanged",
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
