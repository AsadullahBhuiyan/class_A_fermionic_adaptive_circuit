#!/usr/bin/env python3
"""Three stacked hard-wall purification/contour/Lyapunov-gap panels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import analyze_campaign as campaign13
from plot_endpoint_lyapunov_gap import write_csv, configure_plotting, weighted_power_law

ROOT = Path(__file__).resolve().parent
BUNDLE07 = ROOT.parent / "07_maxmix_hard_soft_purification"
sys.path.insert(0, str(BUNDLE07))
from plot_hard_wall_purification_spatial import load_hard_wall_data

OUT = ROOT / "analysis_outputs" / "purification_contour_gap_3x1_v3_single_column"
WINDOWS = ("endpoint",)


def mean_sem(values):
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))


def finite_time_gaps(occupations, caps, times):
    """Take min|lambda(t)| per sample/checkpoint before any averaging."""
    if np.any(times <= 0):
        raise ValueError("Positive elapsed cycle count required")
    costs = np.full_like(occupations, np.inf)
    finite = ~caps
    nu = occupations[finite]
    if np.any(nu <= 0) or np.any(nu >= 1):
        raise ValueError("Uncapped occupation outside (0,1)")
    costs[finite] = np.abs(np.log(nu) - np.log1p(-nu))
    gaps = costs.min(axis=-1) / (2 * times[None, :])
    if not np.all(np.isfinite(gaps)):
        raise ValueError("Sample has no resolved finite gap")
    return gaps


def gap_data():
    rows = []
    max_difference = 0.0
    for path in sorted((campaign13.DATA_ROOT / "results").rglob("*.npz")):
        with np.load(path, allow_pickle=False) as d:
            ny = int(d["Ny"])
            times = np.array([4 * ny])
            all_times = d["spectrum_cycles"]
            idx = np.searchsorted(all_times, times)
            if not np.array_equal(all_times[idx], times):
                raise ValueError(f"Missing late checkpoints: {path}")
            g = finite_time_gaps(d["occupations"][:, idx], d["cap_mask"][:, idx], times)
            stored = d["soft_mode_flip_costs"][:, idx].min(axis=-1) / (2 * times)
            max_difference = max(max_difference, float(np.abs(g - stored).max()))
            for j, sample in enumerate(d["sample_indices"]):
                rows.append(dict(Ny=ny, sample_index=int(sample),
                                 T=int(times[0]), endpoint=float(g[j, 0])))
    if max_difference > 5e-14:
        raise ValueError("Stored soft-mode cost and direct gap disagree")
    summaries = []
    for ny in campaign13.NY_VALUES:
        local = sorted((r for r in rows if r["Ny"] == ny), key=lambda r:r["sample_index"])
        if [r["sample_index"] for r in local] != list(range(100)):
            raise ValueError("Sample coverage mismatch")
        for name in WINDOWS:
            values = np.array([r[name] for r in local])
            m, e = mean_sem(values)
            summaries.append(dict(Ny=ny, window=name, samples=100,
                                  mean_gap=float(m), sample_sem=float(e)))
    return rows, summaries, max_difference


def make_figure(entropy_rows, spatial_rows, gap_rows, fit, *, density=None, density_vmax=None,
                output=None, alpha_comparison=False, smallest_mode_density=False,
                normalized_stop=4, gap_time_multiple=4, density_time_multiple=None,
                loglog_abc=False):
    configure_plotting()
    plt.rcParams.update({"font.size":8, "axes.labelsize":8,
                         "legend.fontsize":8, "xtick.labelsize":8,
                         "ytick.labelsize":8})
    four_panel = density is not None
    fig, axes = plt.subplots(4 if four_panel else 3, 1,
                             figsize=(3.375, 6.8 if four_panel else 6.5), layout="constrained")
    if four_panel:
        fig.get_layout_engine().set(h_pad=.025, w_pad=.025, hspace=.02)
    a, b, c = axes[:3]
    grouping = "alpha_1" if alpha_comparison else "Ny"
    sizes = sorted({r[grouping] for r in entropy_rows})
    colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#e377c2",
              "#ff7f0e", "#008b8b", "#555555"]
    markers = ["^", "s", "o", "D", "v", "P", "X", "h"]
    styles = [":", "--", "-", "-.", ":", "--", "-", "-."]
    if alpha_comparison:
        colors, markers, styles = ["#d62728", "#1f77b4"], ["^", "o"], [":", "-"]
    for ny, color, marker, style in zip(sizes, colors, markers, styles):
        rows = [r for r in entropy_rows if r[grouping] == ny and (not loglog_abc or r['normalized_cycle']>0)]
        u = np.array([r["normalized_cycle"] for r in rows])
        m = np.array([r["mean_entropy_over_Ny"] for r in rows])
        e = np.array([r["sem"] for r in rows])
        a.fill_between(u, np.maximum(m-e, 1e-12), m+e, color=color, alpha=.1, lw=0)
        a.plot(u, m, color=color, marker=marker, ls=style, lw=1.3,
               markevery=(np.unique(np.rint(np.geomspace(1,len(u),10)).astype(int)-1).tolist() if loglog_abc else max(1,len(u)//12)), ms=3.5, mfc="white", mew=.8,
               label=rf"$\alpha_1={ny}$" if alpha_comparison else rf"$N_y={ny}$")
    a.set(yscale="log", xlim=(0,normalized_stop), xlabel=r"cycle $t/N_y$",
          ylabel=r"$\langle S(t)\rangle/N_y$")
    a.legend(loc="upper right", frameon=False)
    if alpha_comparison:
        a.text(.98,.5,r"$N_y=40$",transform=a.transAxes,ha="right",fontsize=8)

    colors07={20:"#d62728",30:"#2ca02c",40:"#1f77b4"}
    marks07={20:"^",30:"s",40:"o"}
    pos_styles={5:"-",15:"--",10:":"}
    for ny in (20,30,40):
        for x in (5,15,10):
            rows=[r for r in spatial_rows if r["Ny"]==ny and r["x"]==x and (not loglog_abc or r['normalized_cycle']>0)]
            u=np.array([r["normalized_cycle"] for r in rows])
            m=np.array([r["mean_entropy_x_over_Ny"] for r in rows])
            e=np.array([r["sem"] for r in rows])
            b.fill_between(u,np.maximum(m-e,1e-12),m+e,color=colors07[ny],alpha=.08,lw=0)
            b.plot(u,m,color=colors07[ny],ls=pos_styles[x],lw=1.3,
                   marker=marks07[ny],markevery=(np.unique(np.rint(np.geomspace(1,len(u),10)).astype(int)-1).tolist() if loglog_abc else ny//2),ms=3.5,mfc="white",mew=.8,
                   alpha=.7 if x==10 else 1)
    b.set(yscale="log",xlim=(0,normalized_stop),ylim=(1e-6,10 if normalized_stop==2 else 2),xlabel=r"cycle $t/N_y$",
          ylabel=r"$\langle s_x(t)\rangle/N_y$")
    legend=b.legend(handles=[Line2D([],[],color=colors07[n],marker=marks07[n],mfc="white",
                           ms=4,label=rf"$N_y={n}$") for n in (20,30,40)],
                    loc="upper right",frameon=False,
                    **({"labelspacing":.12,"handlelength":1.4,"handletextpad":.5} if four_panel else {}))
    b.add_artist(legend)
    b.legend(handles=[Line2D([],[],color=".2",ls=pos_styles[x],
                     label=rf"$x={x}$")
                     for x in (5,15,10)],loc="upper center",bbox_to_anchor=(.43 if four_panel else .53,1),
             frameon=False, **({"labelspacing":.12,"handlelength":1.4,"handletextpad":.5} if four_panel else {}))

    definitions=[("endpoint",rf"$T={gap_time_multiple}N_y$: mean $\pm$ SEM", "#1f77b4","o","none")]
    for name,label,color,marker,style in definitions:
        rows=[r for r in gap_rows if r["window"]==name]
        c.errorbar([r["Ny"] for r in rows],[r["mean_gap"] for r in rows],
                   yerr=[r["sample_sem"] for r in rows],color=color,marker=marker,
                   ls=style,lw=1.2,ms=4,mfc="white",mew=1,capsize=2,label=label)
    dense_ny = np.linspace(min(campaign13.NY_VALUES), max(campaign13.NY_VALUES), 300)
    c.plot(dense_ny, fit["amplitude"] * dense_ny**(-fit["exponent"]),
           color=".25", ls="--", lw=1,
           label=rf"$A N_y^{{-z}},\ z={fit['exponent']:.2f}\pm{fit['exponent_sem']:.2f}$")
    gap_upper=max(.065,1.15*max(r['mean_gap']+r['sample_sem'] for r in gap_rows)) if gap_time_multiple!=4 else .065
    c.set(xlabel=r"circumference $N_y$",ylabel=r"$\Delta$",ylim=(0,gap_upper))
    c.set_xticks(campaign13.NY_VALUES)
    c.legend(loc="upper right",frameon=False)
    if loglog_abc:
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
        a.get_legend().set_loc('lower left')
        b.get_legend().set_bbox_to_anchor(None)
        b.get_legend().set_loc('lower left')
        for ax,rows in ((a,entropy_rows),(b,spatial_rows)):
            lower=min(r['normalized_cycle'] for r in rows if r['normalized_cycle']>0)
            ax.set_xscale('log')
            ax.set_xlim(lower,normalized_stop)
            ax.xaxis.set_major_locator(FixedLocator([v for v in (.03,.1,.3,1,2,4) if lower<=v<=normalized_stop]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
        lows=[r['mean_gap']-r['sample_sem'] for r in gap_rows]
        assert min(lows)>0
        c.set(xscale='log',yscale='log',xlim=(19,63),ylim=(min(lows)*.85,max(r['mean_gap']+r['sample_sem'] for r in gap_rows)*1.6))
        c.xaxis.set_major_locator(FixedLocator(campaign13.NY_VALUES))
        c.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
        c.xaxis.set_minor_formatter(NullFormatter())
        c.yaxis.set_major_locator(FixedLocator([.015,.02,.03,.04,.06]))
        c.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
        c.yaxis.set_minor_formatter(NullFormatter())
    if four_panel:
        from matplotlib.ticker import LogLocator, NullFormatter
        # Sparse log tick labels keep the shorter panels readable at 8 pt.
        a.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        b.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        for ax in (a,b):
            ax.yaxis.set_minor_formatter(NullFormatter())
        d = axes[3]
        im = d.imshow(density, origin="lower", interpolation="nearest", aspect="auto",
                      extent=(-.5,19.5,-.5,29.5), cmap="magma", vmin=0, vmax=density_vmax)
        d.set(xlabel="$x$", ylabel="$y$", xticks=[0,5,10,15,19], yticks=[0,10,20,29])
        density_label = r"$N_y=30$" if density_time_multiple is None else rf"$N_y=30,\ T={density_time_multiple}N_y$"
        d.text(.5,.97,density_label,color="white",ha="center",va="top",
               transform=d.transAxes,fontsize=8)
        # Inset colorbar keeps the four main axes at identical widths.
        cax = d.inset_axes([1.025,0,.035,1])
        cb = fig.colorbar(im,cax=cax,**({} if smallest_mode_density else {"ticks":[0,.01,.02,.03]}))
        if smallest_mode_density:
            from matplotlib.ticker import MaxNLocator
            cb.locator = MaxNLocator(nbins=3)
            cb.update_ticks()
        cb.set_label(r"$\overline{p_{\min}(x,y)}$" if smallest_mode_density else r"$\overline{p(x,y)}$",labelpad=2)
        cb.ax.tick_params(labelsize=8,pad=2)
    for ax,label in zip(axes,["(a)","(b)","(c)","(d)"]):
        ax.text(-.18,1.035,label,transform=ax.transAxes,fontsize=9)
    stem=(output or OUT)/("hard_wall_purification_contours_gap_density_4x1" if four_panel
                         else "hard_wall_purification_contours_gap_3x1")
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"),dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    print("Verifying bundle 13: 140 shards, 700 trajectories",flush=True)
    _,_,prov13=campaign13.load_and_verify(verify_hashes=True)
    print("Verifying bundle 07 contours: 60 shards, 300 trajectories",flush=True)
    data07,prov07=load_hard_wall_data(verify_hashes=True)
    entropy_rows=[]
    for ny in (20,30,40):
        bundle=7
        times=np.arange(4*ny+1)
        v=data07[ny]["total_entropy"]/ny
        m,e=mean_sem(v)
        for t,mean,error in zip(times,m,e):
            entropy_rows.append(dict(bundle=bundle,Ny=ny,cycle=int(t),normalized_cycle=t/ny,
                                     samples=100,mean_entropy_over_Ny=float(mean),sem=float(error)))
    spatial_rows=[]
    for ny in (20,30,40):
        for x in (5,15,10):
            m,e=mean_sem(data07[ny]["entropy_x"][:,:,x]/ny)
            for t,(mean,error) in enumerate(zip(m,e)):
                spatial_rows.append(dict(Ny=ny,x=x,cycle=t,normalized_cycle=t/ny,
                                         samples=100,mean_entropy_x_over_Ny=float(mean),sem=float(error)))
    samples,gaps,diff=gap_data()
    write_csv(OUT/"total_entropy_curves.csv",entropy_rows)
    write_csv(OUT/"spatial_entropy_curves.csv",spatial_rows)
    write_csv(OUT/"gap_samples.csv",samples)
    write_csv(OUT/"gap_summary.csv",gaps)
    fit = weighted_power_law(np.array([r["Ny"] for r in gaps]),
                             np.array([r["mean_gap"] for r in gaps]),
                             np.array([r["sample_sem"] for r in gaps]))
    make_figure(entropy_rows,spatial_rows,gaps,fit)
    summary=dict(bundle13_provenance=prov13,bundle07_provenance=prov07,
                 entropy_sizes=sorted({r["Ny"] for r in entropy_rows}),
                 total_entropy_source="bundle 07 for all three sizes, identical ensemble to spatial panel",
                 gap_definition="min_j abs(log(nu_j)-log(1-nu_j))/(2*t)",
                 gap_time="endpoint T=4Ny only",
                 weighted_log_space_power_law_fit=fit,
                 fit_method="ln(mean gap)=ln(A)-z ln(Ny), sigma_log=sample SEM/mean; absolute covariance, no bootstrap",
                 figure_size_inches=[3.375,6.5],
                 uncertainty="sample SEM across 100 independent trajectories; no bootstrap; time points not independent samples",
                 maximum_gap_soft_cost_difference=diff)
    (OUT/"analysis_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(gaps,indent=2),flush=True)
    print(OUT/"hard_wall_purification_contours_gap_3x1.png",flush=True)


if __name__=="__main__":
    main()
