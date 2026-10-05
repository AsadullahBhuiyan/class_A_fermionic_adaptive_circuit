#!/usr/bin/env python3
"""Replot verified v2 observables without recomputing or modifying analysis caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_chirality_figure as reduction
import analyze_endpoint_packets as base
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np

WALL_COLORS = ("#D55E00", "#0072B2")
WALL_STYLES = ("-", "--")
MAIN = "modular_handedness_n20x32_y8_v2"
SNAPSHOT = "modular_packet_snapshots_n20x32_y8_v2"


def style():
    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["CMU Sans Serif","DejaVu Sans"],
        "mathtext.fontset":"cm", "font.size":8, "axes.titlesize":9, "axes.labelsize":8,
        "legend.fontsize":8, "xtick.labelsize":8, "ytick.labelsize":8,
        "xtick.direction":"in", "ytick.direction":"in", "xtick.top":True,
        "ytick.right":True, "legend.frameon":False, "pdf.fonttype":42})


def save(fig, output, stem):
    for suffix in ("pdf", "png"):
        fig.savefig(output/f"{stem}.{suffix}", dpi=300)
    plt.close(fig)


def band(ax, times, mean, sem, *, color, ls="-", label=None, lw=1.15):
    ax.plot(times, mean, color=color, ls=ls, lw=lw, label=label)
    ax.fill_between(times, mean-sem, mean+sem, color=color, alpha=.18, lw=0)


def main_figure(data, output, samples):
    times = data["times"]
    fig, axes = plt.subplots(2, 2, figsize=(7.05,5.6), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=.06, h_pad=.07, wspace=.06, hspace=.06)
    time_edges = np.r_[times[0]-.005, (times[1:]+times[:-1])/2, times[-1]+.005]
    y_edges = np.arange(17)-8.5
    for p, (wall, ax) in enumerate(zip(base.WALLS, axes[0])):
        probability = data["alpha1_longitudinal_probability_mean"][p].T
        mesh = ax.pcolormesh(time_edges, y_edges, probability, cmap="viridis",
                             norm=Normalize(0,1), shading="flat", rasterized=False)
        ax.axhline(0, color="white", ls="--", lw=.75)
        line, = ax.plot(times, data["alpha1_dy_window_mean"][p], color=WALL_COLORS[p],
                        ls=WALL_STYLES[p], lw=1.25)
        line.set_path_effects([pe.Stroke(linewidth=2.3, foreground="white"), pe.Normal()])
        ax.set(xlim=(0,1), ylim=(-8,7), xlabel=r"Modular time $t_{\rm mod}$",
               ylabel="Position relative to injection\n"+r"$y_{\rm rel}-8$",
               title=rf"{'Left' if p==0 else 'Right'} wall $x={wall}$; $\alpha_1=1$")
        ax.set_yticks([-8,-4,0,4,7])
        ax.set_xticks([0,.25,.5,.75,1])
        ax.text(.97,.05,"Line: mean displacement",ha="right",transform=ax.transAxes,
                color="white",fontsize=8)
    cbar=fig.colorbar(mesh, ax=list(axes[0]), location="bottom", fraction=.09, pad=.13, aspect=45)
    cbar.set_label("Wall-window conditional probability per row")
    cbar.set_ticks([0,.25,.5,.75,1])
    ax=axes[1,0]
    for alpha in (1,3):
        for p,wall in enumerate(base.WALLS):
            band(ax,times,data[f"alpha{alpha}_dy_window_mean"][p],data[f"alpha{alpha}_dy_window_sem"][p],
                 color=WALL_COLORS[p] if alpha==1 else (".55",".05")[p], ls=WALL_STYLES[p],
                 label=rf"$\alpha_1={alpha},\ x={wall}$")
    ax.set(xlim=(0,1),ylim=(-.95,.95),xlabel=r"Modular time $t_{\rm mod}$",
           ylabel="Mean displacement from injection\n"+r"$\langle\Delta y\rangle$ (lattice spacings)",
           title="Wall displacements; mean ± SEM")
    ax.set_yticks([-.8,-.4,0,.4,.8])
    ax.legend(loc="center right",ncol=2, handlelength=1.5,columnspacing=.8)
    ax=axes[1,1]
    for alpha,color,ls in ((1,"#7B3294","-"),(3,".15","--")):
        band(ax,times,data[f"alpha{alpha}_contrast_mean"],data[f"alpha{alpha}_contrast_sem"],
             color=color,ls=ls,label=rf"$\alpha_1={alpha}$"+(" (control)" if alpha==3 else ""))
    ax.set(xlim=(0,1),ylim=(-.05,.95),xlabel=r"Modular time $t_{\rm mod}$",
           ylabel=r"Wall contrast $D_\chi$ (lattice spacings)", title=r"$D_\chi=(\langle\Delta y_5\rangle-\langle\Delta y_{15}\rangle)/2$")
    ax.legend(loc="center right")
    for ax in axes[1]:
        ax.axhline(0,color=".7",ls=":",lw=.6,zorder=0)
        ax.set_xticks([0,.25,.5,.75,1])
    for letter,ax in zip("abcd",axes.flat):
        ax.text(-.14,1.04,f"({letter})",transform=ax.transAxes,weight="bold")
    fig.suptitle(rf"$20\times32$ hard walls; $S={samples}$ per $\alpha_1$; $\epsilon=10^{{-10}}$",fontsize=9)
    save(fig, output, MAIN)


def snapshot_figure(data, output):
    snapshots = data["snapshot_times"]
    fig, axes = plt.subplots(2,5,figsize=(7.05,3.8),sharex=True,sharey=True)
    xx, yy = np.meshgrid(np.arange(base.NX),np.arange(base.LENGTH))
    for row,alpha in enumerate((3,1)):
        for ti,t in enumerate(snapshots):
            ax=axes[row,ti]
            for p,wall in enumerate(base.WALLS):
                values=data[f"alpha{alpha}_density_mean"][p,ti]
                mask=values>1e-4
                ax.scatter(xx[mask],yy[mask],s=50*values[mask]/2,
                           color=WALL_COLORS[p],edgecolors="none")
                index=int(np.argmin(abs(data["times"]-t)))
                center=8+data[f"alpha{alpha}_dy_window_mean"][p,index]
                ax.plot(wall,center,marker="_",markersize=8,color="black",mew=1.1,ls="",zorder=5)
                ax.scatter([wall],[8],s=35,facecolors="none",edgecolors=".4",linewidths=.7,zorder=4)
                ax.axvline(wall,color=".6",ls=":",lw=.5,zorder=0)
            ax.axhline(8,color=".5",ls="--",lw=.6,zorder=0)
            ax.set(xlim=(-.5,19.5),ylim=(-.5,15.5),aspect="equal")
            ax.set_xticks([0,5,15,19]); ax.set_yticks([0,8,15])
            if row==0: ax.set_title(rf"$t_{{\rm mod}}={t:g}$")
            if row==1: ax.set_xlabel("$x$")
            if ti==0: ax.set_ylabel(rf"$\alpha_1={alpha}$"+(" (control)" if alpha==3 else "")+"\n"+r"$y_{\rm rel}$")
    fig.suptitle("Position within subsystem; packets start at $y_{\\rm rel}=8$",fontsize=9,y=.99)
    handles=[Line2D([],[],color=c,marker="o",ls="",markersize=4,label=f"Packet at x={w}")
             for w,c in zip(base.WALLS,WALL_COLORS)]
    handles += [Line2D([],[],color="black",marker="_",ls="",markersize=8,label="Mean wall-window center"),
                Line2D([],[],color=".5",ls="--",label="Injection row")]
    fig.legend(handles=handles,loc="lower center",ncol=2,bbox_to_anchor=(.5,0),columnspacing=1.5)
    fig.subplots_adjust(left=.09,right=.985,bottom=.23,top=.86,wspace=.14,hspace=.20)
    save(fig,output,SNAPSHOT)


def supporting_figures(data, old, output):
    times=data["times"]
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.65),layout="constrained")
    for alpha,ax in zip((1,3),axes):
        for ei,(eps,color) in enumerate(zip(old["epsilons"],("#D55E00","#009E73","#0072B2"))):
            for p,ls in enumerate(WALL_STYLES):
                band(ax,times,old[f"alpha{alpha}_dy_window_mean"][ei,p],old[f"alpha{alpha}_dy_window_sem"][ei,p],
                     color=color,ls=ls,label=rf"$\epsilon={eps:.0e}$" if p==0 else None)
        ax.axhline(0,color=".65",lw=.6)
        ax.set(xlim=(0,1),xlabel=r"Modular time $t_{\rm mod}$",ylabel=r"$\langle\Delta y\rangle$ from injection",
               title=rf"$\alpha_1={alpha}$"+(" (control; expanded scale)" if alpha==3 else ""))
        ax.legend(loc="best")
    fig.suptitle("Cutoff comparison: solid x=5; dashed x=15. Panel scales differ.",fontsize=8)
    save(fig,output,"modular_handedness_cutoff_comparison_v2")
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.65),layout="constrained")
    for alpha,ax in zip((1,3),axes):
        for p,wall in enumerate(base.WALLS):
            band(ax,times,data[f"alpha{alpha}_window_charge_mean"][p]/2,
                 data[f"alpha{alpha}_window_charge_sem"][p]/2,
                 color=WALL_COLORS[p],ls=WALL_STYLES[p],label=rf"$x={wall}$")
        ax.set(xlim=(0,1),ylim=(0,1.02),xlabel=r"Modular time $t_{\rm mod}$",
               ylabel="Retained wall-window charge / 2",title=rf"$\alpha_1={alpha}$")
        ax.legend(loc="lower left")
    save(fig,output,"modular_packet_retention_v2")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=reduction.OUTPUT)
    args=parser.parse_args()
    source=args.output/"averaged_observables.npz"
    before=base.digest(source)
    summary=json.loads((args.output/"analysis_summary.json").read_text())
    if before != summary["data_sha256"]:
        raise ValueError("Figure input checksum mismatch")
    reference=reduction.REFERENCE/"averaged_observables.npz"
    if base.digest(reference) != summary["reference_sha256"]:
        raise ValueError("Cutoff reference checksum mismatch")
    with np.load(source,allow_pickle=False) as saved:
        data={k:saved[k] for k in saved.files}
    with np.load(reference,allow_pickle=False) as saved:
        old={k:saved[k] for k in saved.files}
    for alpha in (1,3):
        reduction.validate_probability(data[f"alpha{alpha}_longitudinal_probability_mean"],
                                       data[f"alpha{alpha}_dy_window_mean"])
    style()
    main_figure(data,args.output,summary["samples_per_alpha"])
    snapshot_figure(data,args.output)
    supporting_figures(data,old,args.output)
    if base.digest(source) != before:
        raise RuntimeError("Plotting modified its input")
    metadata=dict(data_sha256=before,plot_script_sha256=base.digest(__file__),
                  baseline_epsilon=1e-10,heatmap_color_limits=[0,1],heatmap_scale="linear",
                  heatmap_y_range=[-8,7],curve_time_range=[0,1],raw_curve_y_range=[-.95,.95],
                  snapshot_times=data["snapshot_times"].tolist(),snapshot_area="50 * density/2",
                  smoothing=False,uncertainty="trajectory SEM; paired walls in contrast",
                  supporting_reference_sha256=summary["reference_sha256"])
    (args.output/"figure_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(f"[done] Main, snapshots, cutoff comparison, and retention figures: {args.output}")


if __name__=="__main__":
    main()
