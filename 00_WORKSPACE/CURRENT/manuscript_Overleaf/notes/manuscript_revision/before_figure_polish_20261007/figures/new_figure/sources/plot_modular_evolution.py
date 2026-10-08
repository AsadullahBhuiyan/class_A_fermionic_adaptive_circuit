#!/usr/bin/env python3
"""Presentation-only replot of verified early-time ensemble observables."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from manuscript_typography import configure_style as manuscript_style, prepare_figure, record_typography

OUTPUT = Path(__file__).resolve().parents[1]
DATA = OUTPUT / "data/modular_evolution"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--five-times", action="store_true",
                        help="Separate preview with 0,0.1,0.2,0.5,1 snapshots from v2 data")
    args = parser.parse_args()
    output = (OUTPUT.parent / "hard_n20x32_a1-1-3_y8_chirality_figure_v2"
              if args.five_times else OUTPUT)
    stem = "Figure_10_modular_evolution"
    if args.five_times: raise ValueError("This bundle retains only snapshots 0, 0.1, 0.2")
    source = DATA / "averaged_observables.npz"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source, allow_pickle=False) as saved:
        ei = None if args.five_times else int(np.flatnonzero(saved["epsilons"] == 1e-10)[0])
        snapshots = saved["snapshot_times"]
        times = saved["times"]
        densities = {a: saved[f"alpha{a}_density_mean"] if args.five_times
                     else saved[f"alpha{a}_density_mean"][ei] for a in (1,3)}
        mean, sem = [saved[f"alpha1_dy_window_{key}"] if args.five_times
                     else saved[f"alpha1_dy_window_{key}"][ei] for key in ("mean","sem")]
    np.testing.assert_allclose(snapshots, [0,.1,.2,.5,1] if args.five_times else [0,.1,.2], rtol=0, atol=1e-12)
    if args.five_times:
        summary = json.loads((output / "analysis_summary.json").read_text())
        if summary["data_sha256"] != before:
            raise ValueError("Figure source checksum mismatch")
    manuscript_style({'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'legend.frameon': False, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(3.95, 3.9))
    # Equal-aspect snapshots share a row; the displacement spans their full width.
    axes = [fig.add_axes([.16, .60, .35, .284]),
            fig.add_axes([.60, .60, .35, .284]),
            fig.add_axes([.16, .13, .79, .31])]
    colors = ["#332288", "#E69F00", "#009E73"]
    if args.five_times:
        colors += ["#CC79A7", "#56B4E9"]
    xx, yy = np.meshgrid(np.arange(20),np.arange(16))
    handles = [Line2D([],[],marker="o",ls="",color=c,markersize=4,label=f"{t:g}",
                     markerfacecolor="none" if args.five_times and t==0 else c)
               for t,c in zip(snapshots,colors)]
    # Requested right rotation of old panels: C -> A, A -> B, B -> C.
    for ax, alpha, letter in zip(axes[:2], (3,1), ("a","b")):
        for packet in range(2):
            for ti,color in enumerate(colors):
                values = densities[alpha][packet,ti]
                mask = values > 1e-4
                ax.scatter(xx[mask],yy[mask],s=85*np.sqrt(values[mask]/2),
                           facecolors="none" if args.five_times and ti==0 else color,
                           edgecolors=color,linewidths=1 if args.five_times and ti==0 else .2,
                           zorder=10 if args.five_times and ti==0 else 5-ti)
        for wall in (5,15):
            ax.axvline(wall,color=".5",ls=":",lw=.7,zorder=0)
        ax.set(xlim=(-.5,19.5),ylim=(-.5,15.5),xlabel="$x$",ylabel=r"$y-y_0$")
        if args.five_times:
            ax.set_ylabel("Position within subsystem\n"+r"$y_{\rm rel}$")
            ax.axhline(8,color=".65",ls="--",lw=.6,zorder=0)
            ax.text(.99,1.035,"Injection: $y_{\\rm rel}=8$",transform=ax.transAxes,
                    ha="right",va="bottom",fontsize=8,
                    bbox={"facecolor":"white","edgecolor":"none","pad":1.5})
        ax.set_aspect("equal")
        ax.set_xticks([0,5,15,19]); ax.set_yticks([0,4,8,12,15])
        ax.text(.5,.97,rf"$\alpha_1={alpha}$",transform=ax.transAxes,
                ha="center",va="top",bbox={"facecolor":"white","edgecolor":"none","pad":1.5},zorder=10)
        if alpha == 1:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)
    legend_title = Line2D([], [], ls='', label=r'$t_{\rm mod}:$')
    fig.legend(handles=[legend_title]+handles, loc='center', bbox_to_anchor=(.55,.493),
               ncol=4, handlelength=.6, handletextpad=.4, columnspacing=.8, borderpad=0)
    ax = axes[2]
    for packet,(wall,color,ls) in enumerate(zip((5,15),("#D55E00","#0072B2"),("-","--"))):
        ax.plot(times,mean[packet],color=color,ls=ls,lw=1,label=rf"$({wall},8)$")
        ax.fill_between(times,mean[packet]-sem[packet],mean[packet]+sem[packet],color=color,alpha=.18,lw=0)
    ax.axhline(0,color=".7",lw=.5,zorder=0)
    ax.set(xlim=(0,1),ylim=(-.95,.95),xlabel=r"modular time $t_{\rm mod}$",ylabel=r"$\overline{\Delta y}$")
    if args.five_times:
        ax.set_ylabel("Mean displacement from injection\n"+r"$\overline{\Delta y}$")
    ax.set_xticks([0,.25,.5,.75,1]); ax.set_yticks([-.5,0,.5])
    ax.legend(loc="center",bbox_to_anchor=(.5,.61),ncol=2,
              fontsize=8,columnspacing=1,handlelength=1.8,borderaxespad=.2)
    ax.text(.5,.40,r'$\alpha_1=1$',transform=ax.transAxes,ha='center',va='center')
    for panel, letter in zip(axes, 'abc'):
        box = panel.get_position()
        fig.text(box.x0-.065, box.y1+.025, f'({letter})',ha='left',va='bottom',fontsize=9)
    prepare_figure(fig, stem)
    fig.canvas.draw()
    # The lower panel aligns with the outside borders of the two snapshots.
    np.testing.assert_allclose(axes[2].get_position().x0, axes[0].get_position().x0, atol=1e-3)
    np.testing.assert_allclose(axes[2].get_position().x1, axes[1].get_position().x1, atol=1e-3)
    for packet, line in enumerate(axes[2].lines[:2]):
        np.testing.assert_array_equal(line.get_xdata(), times)
        np.testing.assert_array_equal(line.get_ydata(), mean[packet])
    record_typography(fig, stem)
    for suffix in ("pdf","png"):
        fig.savefig(output/f"{stem}.{suffix}",dpi=300)
    plt.close(fig)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    metadata = {"data_sha256":before,"plot_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "panel_order":["alpha1=3 density","alpha1=1 density","alpha1=1 displacement"],
                "layout":"two snapshots across top, full-width displacement below", "figure_size_inches":[3.95,3.9], "parameter_label_printed_pt":11, "epsilon":1e-10,"snapshot_times":snapshots.tolist(),
                "curve_time_range":[0,1],"data_changed":False}
    if args.five_times:
        metadata.update(initial_packet="outline",marker_area="85*sqrt(mean_density/2)",
                        curve_changed=False,curve_ensemble="alpha1=1, trajectory mean +/- SEM",
                        note="Original overlay scaling retained; no smoothing or analysis changes")
    metadata_name = f"{stem}_metadata.json" if args.five_times else "stacked_figure_metadata.json"
    (DATA/metadata_name).write_text(json.dumps(metadata,indent=2)+"\n")
    print(output/f"{stem}.png")


if __name__ == "__main__":
    main()
