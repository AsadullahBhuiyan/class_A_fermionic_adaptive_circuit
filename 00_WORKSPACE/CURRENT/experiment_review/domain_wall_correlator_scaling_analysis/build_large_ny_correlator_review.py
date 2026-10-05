#!/usr/bin/env python3
"""Build the independent larger-Ny correlator review; never change old outputs."""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, LogLocator, LogFormatterSciNotation, NullLocator
import numpy as np
from threadpoolctl import threadpool_limits
from tqdm import tqdm

import large_ny_correlator_data as data
from large_ny_correlator_data import SIZES, Endpoint, distribution, fit, observables, typical, windows

OUTPUT = data.PROJECT / "outputs/large_ny_correlator_review_v1"
LABELS = {"xavg": "$x$ average", "left": "$x_L$", "right": "$x_R$", "walls": "two walls",
          "pair_left": "left pair", "pair_right": "right pair", "pairs": "two pairs"}
COLORS = dict(zip(SIZES, ("#882255", "#AA4499", "#44AA99", "#117733", "#DDCC77", "#0072B2")))
OBS_COLORS = dict(zip(LABELS, ("#333333", "#0072B2", "#D55E00", "#009E73", "#56B4E9", "#CC79A7", "#887700")))
STYLES = ("-", "--", "-.", ":", "--", "-")
SLAB = (5, 6, 7, 9, 11, 13, 14, 15)


def write_json(path, payload):
    def default(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(payload, indent=2, default=default, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Review:
    def __init__(self, output):
        self.output = output
        self.pages = []
        self.rows = []
        self.current = []
        self.pdf = PdfPages(output / "large_ny_correlator_review_atlas.pdf")

    def line(self, ax, x, y, label, metadata=None, **kw):
        x, y = np.asarray(x), np.asarray(y)
        for xi, yi in zip(x.ravel(), y.ravel()):
            self.current.append({"panel": getattr(ax, "review_panel", "a"), "series": label,
                                 "x": float(xi), "y": float(yi) if np.isfinite(yi) else "",
                                 "displayed": bool(np.isfinite(yi)), **(metadata or {})})
        return ax.plot(x, y, label=label, **kw)

    def error(self, ax, x, means, q25, q75, label, **kw):
        # IQR is an empirical interval, not necessarily centered on the mean.
        self.line(ax, x, means, label, linestyle="none", **kw)
        color = kw.get("color", "black")
        ax.vlines(x, q25, q75, colors=color, linewidth=.8)
        for xi, mean, low, high in zip(x, means, q25, q75):
            self.current.append({"panel": getattr(ax, "review_panel", "b"), "series": label+" IQR",
                                 "x": xi, "y": mean, "q25": low, "q75": high, "displayed": True})

    def save(self, fig, stem, caption):
        for i, ax in enumerate(fig.axes):
            if getattr(ax, "is_colorbar", False): continue
            if len(fig.axes) > 1:
                ax.text(-.15, 1.03, f"({chr(97+i)})", transform=ax.transAxes, fontsize=10)
            if ax.get_xscale() == "linear" and not getattr(ax, "review_categorical", False):
                ax.xaxis.set_major_locator(MaxNLocator(5))
            if ax.get_yscale() == "log":
                ymin,ymax=ax.get_ylim()
                narrow=ymin>0 and ymax/ymin<100
                ax.yaxis.set_major_locator(LogLocator(subs=(1,2,5) if narrow else (1,),numticks=5))
                if narrow:
                    ax.yaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False,minor_thresholds=(np.inf,np.inf)))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_locator(NullLocator())
        for suffix in ("pdf", "png"):
            fig.savefig(self.output / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
        self.pdf.savefig(fig, bbox_inches="tight")
        self.pages.append({"page": len(self.pages)+1, "stem": stem, "caption": caption})
        for row in self.current: row["figure"] = stem
        self.rows.extend(self.current)
        write_csv(self.output / f"{stem}_plotted.csv", self.current)
        self.current = []
        plt.close(fig)


def axes_grid(rows=1, columns=1, width=3.375, height=3.):
    fig, axes = plt.subplots(rows, columns, figsize=(width, height), squeeze=False, layout="constrained")
    for i, ax in enumerate(axes.ravel()): ax.review_panel = chr(97+i)
    return fig, axes.ravel()


def primary_fit_band(ax, sizes, *, log_coordinates=False):
    """Shade the envelope of size-specific primary fit domains, never an error band."""
    bounds = np.array([data.prior.chord(ny, np.array([2, ny//4])) for ny in sizes])
    if log_coordinates:
        bounds = np.log(bounds)
    low, high = float(bounds[:, 0].min()), float(bounds[:, 1].max())
    ax.axvspan(low, high, color="0.5", alpha=.15, linewidth=0, zorder=0)
    return low, high


def mean_curve_fit_overlay(review, ax, curve, ny, label, *, log_coordinates=False):
    """Fit the displayed ensemble-mean curve, not the trajectory-first exponent mean."""
    model = fit(curve, ny, 2, ny//4)
    if not model['valid']:
        return model
    # Extend the fitted line across the independent domain, like the entropy
    # figures. The gray band identifies the fitted points; outside is extrapolation.
    chord = data.prior.chord(ny, np.array([1, ny//2]))
    log_d = np.linspace(*np.log(chord), 100)
    log_c = model['log_amplitude'] - model['beta']*log_d
    review.line(ax, log_d if log_coordinates else np.exp(log_d),
                log_c if log_coordinates else np.exp(log_c), f"_fit {label}",
                color="black", ls="--", lw=.9, zorder=2,
                metadata=dict(role="mean_curve_fit", Ny=ny, fit_min=2, fit_max=ny//4,
                              estimator="OLS log of trajectory-mean squared correlator",
                              **model))
    return model


def fit_legend_key(ax):
    ax.plot([], [], color="black", ls="--", lw=.9, label="fit")


def summarize_fits(endpoints):
    rows, summaries = [], {}
    for key, endpoint in endpoints.items():
        ny = endpoint.ny
        summaries[key] = {}
        for observable, curves in observables(endpoint).items():
            summaries[key][observable] = {}
            for window, (lo, hi) in windows(ny).items():
                for cutoff in (0., 1e-8):
                    condition = window + ("_cut1e-8" if cutoff else "")
                    fits = [fit(curve, ny, lo, hi, cutoff) for curve in curves]
                    summaries[key][observable][condition] = {
                        "window": [lo, hi], "cutoff": cutoff,
                        "trajectory_beta": distribution([f["beta"] for f in fits]),
                        "fit_to_mean_curve": fit(curves.mean(axis=0), ny, lo, hi, cutoff),
                        "excluded_points": sum(f["excluded_points"] for f in fits),
                        "invalid_fits": sum(not f["valid"] for f in fits),
                    }
                    for sample, f in zip(endpoint.ids, fits):
                        rows.append(dict(cohort=endpoint.cohort, Ny=ny, sample_index=int(sample),
                                         observable=observable, window=window, fit_min=lo, fit_max=hi,
                                         cutoff=cutoff, **f))
    return rows, summaries


def regressions(summaries):
    rows = []
    for observable in LABELS:
        for subset, sizes in (("all_six", SIZES), ("Ny_ge_32", (32, 40, 50, 60))):
            y = np.array([summaries[f"primary_{n}"][observable]["primary"]["trajectory_beta"]["mean"] for n in sizes])
            for power in (1, 2):
                x = 1. / np.asarray(sizes, dtype=float)**power
                slope, intercept = np.polyfit(x, y, 1)
                rows.append({"observable": observable, "subset": subset, "power": power,
                             "intercept": float(intercept), "slope": float(slope),
                             "r_squared": float(1 - np.sum((y-slope*x-intercept)**2)/np.sum((y-y.mean())**2)),
                             "interpretation": "descriptive fixed-Nx circumference extrapolation; no CI"})
    return rows


def old_regression(endpoint, summaries):
    old = data.PROJECT / "outputs"
    expected = json.loads((old / "typical_hard_wall_x_resolved_correlator_summary.json").read_text())["selection"]
    selected = typical(endpoint)
    assert selected["sample_id"] == expected["selected_sample_index"]
    np.testing.assert_allclose(selected["selected_beta"], expected["selected_beta"], atol=2e-13, rtol=0)
    np.testing.assert_allclose(selected["median_beta"], expected["ensemble_median_beta"], atol=2e-13, rtol=0)
    obs = observables(endpoint)
    checks = 0
    with (old / "hard_wall_xresolved_finite_size_power_law_trajectory_fits.csv").open() as stream:
        for row in csv.DictReader(stream):
            if int(row["Ny"]) != 32: continue
            sample = int(row["sample_index"])
            for name, column in (("left", "left_wall"), ("right", "right_wall"), ("walls", "two_wall_average")):
                observed = fit(obs[name][sample], 32, 2, 8)["beta"]
                np.testing.assert_allclose(observed, float(row[f"beta_{column}"]), atol=3e-13, rtol=0)
                checks += 1
    if checks != 300: raise ValueError("Missing prior Ny32 regression rows")
    for c in endpoint.xavg:
        np.testing.assert_allclose(fit(c, 32, 2, 8)["beta"], data.prior.fit_curve(c,32,2,8)[0], atol=3e-13, rtol=0)
    return {"typical_selection": selected, "saved_wall_fits_reproduced": checks,
            "prior_xavg_fits_reproduced": 100, "absolute_tolerance": 3e-13}


def representative_plots(review, endpoint):
    ny = endpoint.ny
    selected = typical(endpoint)
    xdata = endpoint.xresolved[selected["position"]]
    r = np.arange(1, ny//2+1)
    fig, axes = axes_grid(2, 1, height=4.1)
    for ax, sites in zip(axes, ((5, 6, 10), (2, 4))):
        for index, x in enumerate(sites):
            review.line(ax,r,xdata[x,1:],f"$x={x}$",marker=("o","s","^")[index],ms=2.5,
                        linestyle=STYLES[index],lw=1)
        ax.set(yscale="log",xlabel="$r_y$",ylabel="$C_G(x,r_y)$")
        ax.legend(frameon=False,ncol=len(sites),fontsize=8)
    axes[0].set_title(rf"$20\times{ny}$; sample {selected['sample_id']}")
    axes[1].set_title("Exterior: numerical-floor diagnostic",fontsize=8)
    review.save(fig,f"anatomy_Ny{ny:03}",f"Typical pure trajectory {selected['sample_id']}, Ny={ny}, endpoint 2Ny. "
                "Selection: median raw-r x-average exponent on 2..8. No averaging over trajectories. Exterior values are numerical-floor diagnostics.")

    for short in (False, True):
        fig, (ax,) = axes_grid(height=3.3)
        rr = np.arange(1,8) if short else np.arange(1,ny+1)
        colors = plt.get_cmap("viridis")(np.linspace(.03,.95,len(SLAB)))
        for j, x in enumerate(SLAB):
            values = xdata[x,1:8] if short else np.r_[xdata[x,1:],xdata[x,-2::-1]]
            shown = np.where(values > 1e-8, values, np.nan) if short else values
            review.line(ax,rr,shown,f"$x={x}$",color=colors[j],ls=STYLES[j%6],marker=("o","s","^")[j%3],ms=2,lw=1)
        ax.set(yscale="log",xlabel="$r_y$",ylabel="$C_G(x,r_y)$",title=rf"$20\times{ny}$; sample {selected['sample_id']}")
        ax.legend(frameon=False,ncol=4,fontsize=8,loc="upper center",bbox_to_anchor=(.5,1.32))
        if short:
            ax.axhline(1e-8,color="gray",ls=":",lw=.8)
            ax.set_ylim(bottom=7e-9)
            ax.set_xticks([1,3,5,7]); ax.review_categorical=True
        else:
            ax.axvspan(ny/2,ny,color="gray",alpha=.07)
            ax.text(.61,.08,"periodic reflection",transform=ax.transAxes,fontsize=8)
        review.save(fig,f"slab_{'short' if short else 'extended'}_Ny{ny:03}",
                    f"Selected trajectory {selected['sample_id']}, endpoint 2Ny, x=5,6,7,9,11,13,14,15. "
                    +( "Display cutoff 1e-8 on 1..7; unmasked values remain in endpoint_curves.npz."
                     if short else "Only r<=Ny/2 are independent; the shaded reflected half and r=Ny contact recurrence are not fit points."))


def logchord_plot(review, endpoint):
    ny = endpoint.ny
    fig, axes = axes_grid(2,1,height=4.5)
    r = np.arange(1,ny//2+1)
    xx = np.log(data.prior.chord(ny,r))
    groups = ({f"$x={x}$":endpoint.xresolved[:,x] for x in (5,6,14,15)},
              {LABELS[k]:v for k,v in observables(endpoint).items() if k in ("pair_left","pair_right")})
    for ax, curves in zip(axes,groups):
        primary_fit_band(ax, (ny,), log_coordinates=True)
        for j,(label,values) in enumerate(curves.items()):
            mean_curve = values.mean(axis=0)
            review.line(ax,xx,np.log(mean_curve[1:]),label,ls="none",marker=("o","s","^","v")[j],
                        ms=3.4,mfc="white",mew=.8,zorder=3)
            mean_curve_fit_overlay(review,ax,mean_curve,ny,label,log_coordinates=True)
        fit_legend_key(ax)
        mean = next(iter(curves.values()))[:,1:].mean(axis=0)
        anchor=2; guide=np.array([xx[anchor],xx[-1]])
        review.line(ax,guide,np.log(mean[anchor])-2*(guide-xx[anchor]),"slope $-2$",color="0.4",ls=":",lw=.8)
        ax.set(xlabel=r"$\log d_{N_y}(r_y)$",ylabel=r"$\log\overline{C_G}$")
        ax.legend(frameon=False,fontsize=8,ncol=2)
    axes[0].set_title(rf"$20\times{ny}$; $S=100$, $t=2N_y$")
    review.save(fig,f"log_chord_Ny{ny:03}","Natural log of the trajectory-mean squared correlator, not mean log or squared mean covariance. "
                "Top: individual wall/neighbor columns. Bottom: within-trajectory two-site averages. "
                f"Open markers are data; gray marks the primary fit window r=2..{ny//4}. "
                "Black dashed lines are separate unconstrained OLS fits to each displayed mean curve; outside the gray band they are extrapolations. "
                "The dotted slope -2 line is only a critical guide. These visual mean-curve fits do not replace the trajectory-first exponent statistics.")


def scaling_plots(review, endpoints, summaries, regression):
    for stem, names in (("xavg_scaling",("xavg",)),("wall_scaling",("left","right","walls")),
                        ("paired_wall_scaling",("pair_left","pair_right","pairs"))):
        fig, axes = axes_grid(2,1,height=4.4)
        ax,bx = axes
        curve_name=names[-1]
        primary_fit_band(ax,SIZES)
        for j,ny in enumerate(SIZES):
            r=np.arange(1,ny//2+1); c=observables(endpoints[f"primary_{ny}"])[curve_name].mean(axis=0)
            review.line(ax,data.prior.chord(ny,r),c[1:],f"$N_y={ny}$",color=COLORS[ny],ls="none",
                        marker=("^","s","D","v","P","o")[j],ms=3.4,mfc="white",mew=.8,zorder=3)
            mean_curve_fit_overlay(review,ax,c,ny,f"$N_y={ny}$")
        fit_legend_key(ax)
        end=observables(endpoints['primary_60'])[curve_name][:,1:].mean(axis=0)
        guide=np.array([2.,12.]); amp=end[2]*data.prior.chord(60,np.array([3]))[0]**2
        review.line(ax,guide,amp*guide**-2,"$d^{-2}$",color="0.4",ls=":",lw=.9)
        ax.set(xscale="log",yscale="log",xlabel="$d_{N_y}(r_y)$",ylabel=r"$\overline{C_G}$")
        ax.legend(frameon=False,fontsize=8,ncol=2)
        inv=1/np.array(SIZES,dtype=float)
        for j,name in enumerate(names):
            ss=[summaries[f'primary_{ny}'][name]['primary']['trajectory_beta'] for ny in SIZES]
            review.error(bx,inv,[s['mean'] for s in ss],[s['q25'] for s in ss],[s['q75'] for s in ss],
                         LABELS[name],color=OBS_COLORS[name],marker=("o","s","^")[j],ms=3)
            for power in (1,2):
                model=next(v for v in regression if v['observable']==name and v['subset']=='all_six' and v['power']==power)
                xx=np.linspace(0,inv.max(),100)
                review.line(bx,xx,model['intercept']+model['slope']*xx**power,
                            (f"$1/N_y{'^2' if power==2 else ''}$" if len(names)==1 else f"{LABELS[name]} extrap. {power}"),
                            color=OBS_COLORS[name],ls="--" if power==1 else ":",lw=.8,alpha=.75)
        bx.axhline(2,color="gray",ls=":",lw=.8)
        bx.set(xlim=(-.001,.043),xlabel="$1/N_y$",ylabel=r"$\langle\beta\rangle$")
        handles,labels=bx.get_legend_handles_labels()
        if len(names)>1:
            chosen=[i for i,l in enumerate(labels) if "extrap." not in l]
            bx.legend([handles[i] for i in chosen],[labels[i] for i in chosen],frameon=False,fontsize=8)
        else: bx.legend(frameon=False,fontsize=8)
        review.save(fig,stem,"Endpoint 2Ny; 100 independent trajectories per size. Top: mean squared correlators. "
                    "Open markers are data and black dashed curves are separate unconstrained fits of log(mean C) against log chord for each size. "
                    "The gray band is the envelope of the size-specific primary windows, not a common window or uncertainty band: "
                    "each fit uses only r=2..floor(Ny/4) for its own size. Dashed lines extend outside their fitted ranges as extrapolations. "
                    "The dotted d^-2 curve is a guide, not a fit. "
                    "Bottom: mean trajectory-first chord exponent, with trajectory IQR (not SEM). Primary window 2..floor(Ny/4). "
                    "Dashed/dotted descriptive extrapolations use 1/Ny and 1/Ny^2 at fixed Nx=20; beta=2 is not enforced.")


def exponent_distributions(review, endpoints):
    for ny in (32,60):
        obs=observables(endpoints[f'primary_{ny}'])
        values={name:np.array([fit(c,ny,2,ny//4)['beta'] for c in curves]) for name,curves in obs.items()}
        lo=min(v.min() for v in values.values()); hi=max(v.max() for v in values.values())
        edges=np.linspace(lo,hi,31)
        fig,axes=axes_grid(2,1,height=4.4)
        for ax,names in zip(axes,(("xavg","left","right","walls"),("pair_left","pair_right","pairs"))):
            for j,name in enumerate(names):
                count,_=np.histogram(values[name],edges)
                density=count/(len(values[name])*np.diff(edges))
                review.line(ax,(edges[1:]+edges[:-1])/2,density,LABELS[name],color=OBS_COLORS[name],ls=STYLES[j],drawstyle="steps-mid")
            ax.set(xlabel=r"$\beta$",ylabel="Trajectory density")
            ax.legend(frameon=False,fontsize=8,ncol=2)
        axes[0].set_title(rf"$N_y={ny}$; 100 trajectory-wise fits")
        review.save(fig,f"exponent_distributions_Ny{ny:03}","Normalized histograms of primary trajectory-wise fitted exponents, common 30-bin edges across observables at this size. "
                    "These are distributions of samples, not confidence distributions for a mean.")


def window_plots(review, summaries):
    fig,axes=axes_grid(2,2,width=7.05,height=5.1)
    selected=("primary","fixed_short","full_half","legacy_tail","outer_tail")
    labels=("quarter","2..8","full half","5..half","outer quarter")
    for ax,name in zip(axes,("xavg","left","right","pairs")):
        for j,(window,label) in enumerate(zip(selected,labels)):
            ys=[summaries[f'primary_{ny}'][name][window]['trajectory_beta']['mean'] for ny in SIZES]
            review.line(ax,SIZES,ys,label,ls=STYLES[j],marker=("o","s","^","v","D")[j],ms=3)
        ax.axhline(2,color='gray',ls=':',lw=.8)
        ax.set(xlabel="$N_y$",ylabel=r"$\langle\beta\rangle$",title=LABELS[name])
    axes[0].legend(frameon=False,fontsize=8,ncol=2)
    review.save(fig,"window_sensitivity","Trajectory-first means under all five prespecified windows, no amplitude cutoff. Differences measure window sensitivity, not uncertainty on a mean.")
    fig,axes=axes_grid(2,2,width=7.05,height=5.2)
    heatmap_cases=((32,'xavg'),(60,'xavg'),(32,'pairs'),(60,'pairs'))
    matrices=[np.array([[summaries[f'primary_{ny}'][name][f'grid_r{lo}_N{den}']['trajectory_beta']['mean']
                         for den in (4,3,2)] for lo in (2,3,4,6)],dtype=float) for ny,name in heatmap_cases]
    vmin=float(np.nanmin(matrices)); vmax=float(np.nanmax(matrices))
    for ax,(ny,name),matrix in zip(axes,heatmap_cases,matrices):
        ax.imshow(matrix,aspect='auto',origin='upper',cmap='viridis',vmin=vmin,vmax=vmax)
        for i in range(4):
            for j in range(3):
                value=matrix[i,j]
                ax.text(j,i,f"{value:.3f}" if np.isfinite(value) else "invalid",ha='center',va='center',fontsize=8,color='white',bbox=dict(facecolor='black',alpha=.35,pad=1,edgecolor='none'))
                review.current.append(dict(panel=ax.review_panel,series=f"Ny={ny} {name}",x=j,y=i,mean_beta=value if np.isfinite(value) else '',displayed=np.isfinite(value)))
        ax.set_xticks(range(3),['$N_y/4$','$N_y/3$','$N_y/2$']);ax.set_yticks(range(4),['2','3','4','6'])
        ax.review_categorical = True
        ax.set(xlabel='Upper separation',ylabel='Lower separation',title=f"$N_y={ny}$, {LABELS[name]}")
    review.save(fig,"fit_window_grid","Mean sample exponent for the fixed window grid; upper bounds are floored. Invalid means have fewer than four usable points per fit. "
                "All four panels share one color scale. All six sizes and both cutoff variants are retained in fit_tables; these panels compare Ny32 and Ny60.")
    fig,axes=axes_grid(2,1,height=4.4)
    for ax,name in zip(axes,('xavg','pairs')):
        for j,window in enumerate(('primary','outer_tail')):
            for cutoff in ('','_cut1e-8'):
                vals=[summaries[f'primary_{ny}'][name][window+cutoff]['trajectory_beta']['mean'] for ny in SIZES]
                review.line(ax,SIZES,[np.nan if v is None else v for v in vals],window+(' >$10^{-8}$' if cutoff else ' uncut'),
                            color=('C0','C1')[j],ls='--' if cutoff else '-',marker='s' if cutoff else 'o',ms=3)
        ax.set(xlabel='$N_y$',ylabel=r'$\langle\beta\rangle$',title=LABELS[name]);ax.legend(frameon=False,fontsize=8)
    review.save(fig,"amplitude_cutoff_sensitivity","Primary fits remain uncut. Dashed curves separately require C>1e-8, an amplitude sensitivity threshold rather than a rigorous error bound. Invalid samples and removed-point counts are in fit_summary.json.")


def cohort_plot(review,endpoints,summaries):
    fig,axes=axes_grid(2,1,height=4.4)
    for ny in (30,40,50):
        for cohort in ('legacy','primary'):
            key=f'{cohort}_{ny}'
            if key not in endpoints:continue
            e=endpoints[key];r=np.arange(1,ny//2+1)
            review.line(axes[0],data.prior.chord(ny,r),e.xavg[:,1:].mean(axis=0),f'{ny} {cohort}',
                        color={30:'#999999',40:COLORS[40],50:COLORS[50]}[ny],ls='--' if cohort=='legacy' else '-',lw=1)
            ss=summaries[key]['xavg']['primary']['trajectory_beta']
            review.error(axes[1],np.array([ny+(-.4 if cohort=='legacy' else .4)]),[ss['mean']],[ss['q25']],[ss['q75']],f'{ny} {cohort}',
                         color={30:'#999999',40:COLORS[40],50:COLORS[50]}[ny],marker='s' if cohort=='legacy' else 'o',ms=4)
    axes[0].set(xscale='log',yscale='log',xlabel='$d_{N_y}$',ylabel=r'$\overline{C_G^{x\mathrm{av}}}$');axes[0].legend(frameon=False,ncol=2,fontsize=8)
    axes[1].set(xlabel='$N_y$ (small offsets separate cohorts)',ylabel=r'$\langle\beta\rangle$')
    review.save(fig,'legacy_cohort_crosscheck','Legacy streaming and newly downloaded ensembles remain separate. Ny30 is legacy-only, never inserted into the primary x-resolved size series. Bottom bars: trajectory IQR; primary quarter-window fits.')


def compute_benchmarks(output,threads):
    # The CPU source is canonical. No stochastic dynamics or GPU methods run.
    sys.path.insert(0,str(data.ROOT/'src/fgtn'))
    from classA_U1FGTN import classA_U1FGTN
    import plot_hard_wall_flattened_benchmark as benchmark
    provenance=[data.file_record(data.ROOT/'src/fgtn/classA_U1FGTN.py','static_constructor'),
                data.file_record(Path(benchmark.__file__),'regulated_projector')]
    result={}; diagnostic_rows=[]
    with threadpool_limits(limits=threads):
        for ny in tqdm((40,50,60),desc='CPU ground-state references',unit='size'):
            for shell in (1,None):
                model=classA_U1FGTN(Nx=20,Ny=ny,DW=True,nshell=shell,alpha_1=1,alpha_2=30,
                                   trial_orbitals='X',dw_truncation=True,filling_frac=.5)
                model.construct_OW_projectors(nshell=shell,DW=True,trial_orbitals='X',dw_truncation=True)
                assert model.DW_loc==[5,15] and model.dw_truncation
                delta,diag=benchmark.regulated_flattened_momentum_projector(model)
                del model;gc.collect()
                if diag['projector_idempotency_max_abs']>1e-11 or diag['projector_block_hermiticity_max_abs']>1e-11:
                    raise ValueError('Ground-state projector diagnostic failed')
                curve=correlator_from_delta(delta)
                tag=f'Ny{ny}_nsh{shell if shell else "dense"}'
                result[tag]=curve
                diagnostic_rows.append(dict(Nx=20,Ny=ny,nshell=shell,alpha_1=1,alpha_2=30,
                                            DW=True,dw_truncation=True,wall_locations=[5,15],**diag))
    np.savez_compressed(output/'ground_state_correlators.npz',**result)
    write_json(output/'ground_state_diagnostics.json',dict(cases=diagnostic_rows,sources=provenance,
               definition='Signed sum of normalized hard/support-truncated OW rank-one projectors; lowest half filled',
               regulator='Existing B0 occupation-grid phi=1e-7, periodic reconstruction; not a physical flux experiment'))
    return result,diagnostic_rows,provenance


def correlator_from_delta(delta):
    ny,dimension,_=delta.shape
    return np.array([[.5*np.square(np.abs(delta[-r%ny,2*x:2*x+2,2*x:2*x+2])).sum()
                      for r in range(ny//2+1)] for x in range(dimension//2)])


def benchmark_plots(review,endpoints,benchmarks):
    rows=[]
    for ny in (40,50,60):
        rr=np.arange(1,ny//2+1);dd=data.prior.chord(ny,rr)
        a=benchmarks[f'Ny{ny}_nsh1'];b=benchmarks[f'Ny{ny}_nshdense']
        dynamic=endpoints[f'primary_{ny}'].xresolved.mean(axis=0)
        fig,axes=axes_grid(1,2,width=7.05,height=3.5)
        colors=plt.get_cmap('viridis')(np.linspace(.03,.95,8))
        for ax,values,label in zip(axes,(a,b),(r'$n_{\rm shell}=1$',r'$n_{\rm shell}=\infty$')):
            for j,x in enumerate(SLAB):
                review.line(ax,dd,values[x,1:],f'$x={x}$',color=colors[j],ls=STYLES[j%6],marker=('o','s','^')[j%3],ms=2,lw=1)
            ax.set(xscale='log',yscale='log',xlabel='$d_{N_y}$',ylabel='$C_G(x,r_y)$',title=label)
        axes[0].legend(frameon=False,fontsize=8,ncol=2)
        fig.suptitle(rf'Hard-wall ground states, $20\times{ny}$',fontsize=10)
        review.save(fig,f'flattened_shell_comparison_Ny{ny:03}','Static hard-wall ground-state comparison at matched geometry, alpha1=1, alpha2=30. Dense support is a distinct parent construction, not the n_shell=1 dynamical target. Independent separations only.')
        fig,axes=axes_grid(2,1,height=4.5)
        for j,x in enumerate((5,10,15)):
            col=('C0','C2','C1')[j]
            review.line(axes[0],dd,dynamic[x,1:],f'$x={x}$ dyn.',color=col,marker=('o','s','^')[j],ms=2,lw=1)
            review.line(axes[0],dd,a[x,1:],f'$x={x}$ GS',color=col,ls='--',lw=1)
            ratio=np.divide(dynamic[x,1:],a[x,1:],out=np.full_like(dd,np.nan),where=a[x,1:]>1e-14)
            review.line(axes[1],dd,ratio,f'$x={x}$',color=col,marker=('o','s','^')[j],ms=2,lw=1)
        axes[0].set(xscale='log',yscale='log',xlabel='$d_{N_y}$',ylabel='$C_G$');axes[0].legend(frameon=False,fontsize=8,ncol=2)
        axes[0].set_title(rf'$20\times{ny}$; $n_{{\rm shell}}=1$')
        axes[1].axhline(1,color='gray',ls=':',lw=.8)
        axes[1].set(xscale='log',yscale='log',xlabel='$d_{N_y}$',ylabel='Dynamics / ground state');axes[1].legend(frameon=False,fontsize=8)
        review.save(fig,f'dynamic_ground_state_Ny{ny:03}',f'Ny={ny}; mean of 100 trajectory-resolved correlators versus the static n_shell=1 parent. Ratios mask reference values <=1e-14 only to avoid numerical denominator artifacts; raw reference and dynamical curves remain saved.')
        for name,values in [('xavg',dynamic.mean(axis=0)),('left',dynamic[5]),('right',dynamic[15])]:
            ref=a.mean(axis=0) if name=='xavg' else a[5 if name=='left' else 15]
            dense=b.mean(axis=0) if name=='xavg' else b[5 if name=='left' else 15]
            for lo,hi,label in ((2,ny//4,'primary'),((ny+3)//4,ny//2,'outer_tail')):
                rows.append(dict(Ny=ny,observable=name,window=label,dynamic_mean_curve_beta=fit(values,ny,lo,hi)['beta'],
                                 shell1_ground_state_beta=fit(ref,ny,lo,hi)['beta'],dense_ground_state_beta=fit(dense,ny,lo,hi)['beta']))
    return rows


def write_findings(out,endpoints,summaries,regression,benchmark_rows,pages):
    def mean(n,name,window='primary',cohort='primary'):
        return summaries[f'{cohort}_{n}'][name][window]['trajectory_beta']['mean']
    table=['| Ny | x average | left wall | right wall | two walls | two pairs |',
           '|---:|---:|---:|---:|---:|---:|']
    size_rows=[]
    for n in SIZES:
        table.append('| '+str(n)+' | '+' | '.join(f'{mean(n,k):.4f}' for k in ('xavg','left','right','walls','pairs'))+' |')
        for name in LABELS:
            record=summaries[f'primary_{n}'][name]['primary']
            size_rows.append(dict(Ny=n,observable=name,fit_min=2,fit_max=n//4,
                                  mean_curve_beta=record['fit_to_mean_curve']['beta'],
                                  **record['trajectory_beta']))
    write_csv(out/'size_summary.csv',size_rows)
    comparisons=[]
    for n in SIZES:
        obs=observables(endpoints[f'primary_{n}'])
        left=np.array([fit(v,n,2,n//4)['beta'] for v in obs['left']])
        right=np.array([fit(v,n,2,n//4)['beta'] for v in obs['right']])
        comparisons.append(dict(Ny=n,quantity='within_trajectory_beta_right_minus_left',**distribution(right-left)))
    write_csv(out/'left_right_difference_summary.csv',comparisons)
    removed=sum(summaries[f'primary_{n}'][name]['outer_tail_cut1e-8']['excluded_points'] for n in SIZES for name in LABELS)
    texts=[
        '# Larger-system hard-wall correlator review',
        '## What changed',
        'The extra sizes resolve longer spatial separations and reveal a downward drift in the primary fitted exponent. '
        'They do not yet establish a window-independent exponent of exactly two. This is a static endpoint analysis of saved trajectories, not a new dynamics run.',
        'The primary ensemble contains 600 trajectories: 100 each at Ny=24,28,32,40,50,60. '
        'All have Nx=20, alpha1=1, alpha2=30, n_shell=1, hard/support-terminated walls at x=5,15, '
        'raster-y order, pure half-filled initialization, perfect correction, complex128 and endpoint 2Ny. '
        'The independent separation range is 1..Ny/2, reaching 30; no longer transverse wall separation was created.',
        '## Primary sample-wise exponents',
        'Fit each trajectory first on 2 <= ry <= floor(Ny/4), using '
        '`log C = log A - beta log[(Ny/pi) sin(pi ry/Ny)]`, then take the sample mean. '
        'The critical squared-correlator guide is beta=2; it is not imposed.',
        '\n'.join(table),
        f'The x-average exponent falls from {mean(32,"xavg"):.4f} at Ny32 to {mean(60,"xavg"):.4f} at Ny60. '
        f'The two-wall value falls from {mean(32,"walls"):.4f} to {mean(60,"walls"):.4f}. '
        'This extends the earlier observation of a wall-dominated algebraic sector, while making the finite-circumference drift visible in the wall-resolved data too.',
        '## Wall averaging and left-right structure',
        f'At Ny60 the separate walls give {mean(60,"left"):.4f} and {mean(60,"right"):.4f}; '
        f'their difference is {mean(60,"right")-mean(60,"left"):.4f}, compared with '
        f'{mean(32,"right")-mean(32,"left"):.4f} at Ny32. It remains visible. '
        'This is a measured endpoint asymmetry; these data alone do not identify its cause.',
        f'Averaging the neighboring topological-side sites gives {mean(60,"pairs"):.4f}, versus '
        f'{mean(60,"walls"):.4f} for the two exact wall columns. It therefore does not improve agreement with two in the primary window. '
        f'The sample SDs at Ny60 are {summaries["primary_60"]["pairs"]["primary"]["trajectory_beta"]["sd"]:.4f} and '
        f'{summaries["primary_60"]["walls"]["primary"]["trajectory_beta"]["sd"]:.4f}, respectively: essentially the same empirical spread. '
        'The averaging is performed on curves within a trajectory, not on already-fitted exponents.',
        '## Does the longer tail settle the exponent?',
        f'No plateau at exactly two is demonstrated by this review. For Ny60 the x-average means are '
        f'{mean(60,"xavg","fixed_short"):.4f} on 2..8, {mean(60,"xavg"):.4f} on 2..15, '
        f'{mean(60,"xavg","full_half"):.4f} on 2..30, {mean(60,"xavg","legacy_tail"):.4f} on 5..30, '
        f'and {mean(60,"xavg","outer_tail"):.4f} on 15..30. '
        'The outer-tail window is more sensitive to sample variation and the compressed range of log chord near the antipode; '
        'extending a window is not automatically a cleaner asymptotic fit.',
        f'The C>1e-8 variant removes {removed} points across the primary ensembles and seven observables in the outer-tail tests. '
        'Thus that amplitude cutoff does not explain the observed exponent drift. The separate short-distance bulk plots still mask values <=1e-8; '
        'the cutoff is not a measured floating-point error bound.',
        'Nonlinear estimator order matters in the far tail. At Ny60, the x-average outer-tail exponent is '
        f'{mean(60,"xavg","outer_tail"):.4f} when fitting each sample first, but '
        f'{summaries["primary_60"]["xavg"]["outer_tail"]["fit_to_mean_curve"]["beta"]:.4f} when fitting the mean curve. '
        'Those are different questions, not interchangeable estimates.',
        '## Static benchmark and interpretation',
        'The hard-wall ground states were rebuilt with the canonical CPU OW constructor and the existing phi=1e-7 occupation-grid convention. '
        'They are static half-filled references; no flux-threading experiment is involved. Dense support remains a different Hamiltonian from n_shell=1.',
    ]
    for name in ('left','right'):
        b=next(r for r in benchmark_rows if r['Ny']==60 and r['observable']==name and r['window']=='outer_tail')
        texts.append(f'Ny60 {name} wall, outer-tail fit: static n_shell=1 beta={b["shell1_ground_state_beta"]:.4f}, '
                     f'static dense beta={b["dense_ground_state_beta"]:.4f}, '
                     f'dynamical mean-curve beta={b["dynamic_mean_curve_beta"]:.4f}.')
    texts.extend([
        'The static reference approaches the exponent-two tail much more closely than the dynamical endpoint. '
        'The dynamical bulk also retains much larger small long-distance correlations than the ground-state bulk. '
        'The latter comparison becomes sensitive to very small denominators, so the ratio figure masks static values <=1e-14. '
        'This does not diagnose roundoff, incomplete relaxation, or a different stationary ensemble by itself. '
        'Finite-duration and finite-width explanations remain possibilities, not measured conclusions.',
        '## Cohort cross-check and extrapolation',
        'The old Ny32 representative selection and 400 numerical fits were reproduced before interpretation. '
        'Legacy streaming data are independent ensembles, not additional rows silently merged into the new datasets.',
    ])
    for n in (40,50):
        texts.append(f'At Ny{n}, the new x-average primary exponent is {mean(n,"xavg"):.4f}, versus '
                     f'{mean(n,"xavg",cohort="legacy"):.4f} in the legacy ensemble. '
                     'The empirical trajectory IQRs overlap; this is descriptive agreement, not an SEM-based hypothesis test.')
    texts.append('Descriptive extrapolated x-average intercepts are:')
    for r in regression:
        if r['observable']=='xavg':
            texts.append(f'- {r["subset"]}, linear in 1/Ny^{r["power"]}: {r["intercept"]:.4f}.')
    texts.extend([
        'Their model/subset dependence is material. These are fixed-Nx=20 circumference extrapolations, not a two-dimensional thermodynamic limit. '
        'They do not justify replacing the measured exponents with two. The earlier qualitative claim of an algebraic wall contribution survives; '
        'the interpretation remains consistency in functional form with the critical prediction, not a precision confirmation of its exponent. '
        'The static/dynamical difference is explicitly a protocol/state comparison, not evidence of a plotting regression.',
        '## What to put forward for review',
        'Start with `xavg_scaling.pdf`, `wall_scaling.pdf`, `log_chord_Ny060.pdf`, and `window_sensitivity.pdf`. '
        'Use `dynamic_ground_state_Ny060.pdf` to expose the remaining benchmark differences. '
        'The paired-site plot is a useful stability check, not an improvement that should replace the wall estimator automatically.',
        f'`large_ny_correlator_review_atlas.pdf` contains {len(pages)} pages; `FIGURES.md` maps pages to files and captions. '
        '`size_summary.csv` gives primary means, SDs, quantiles and mean-curve fits. '
        '`trajectory_fits.csv` contains every sample/window/cutoff record, including invalid status and exclusions. '
        '`endpoint_curves.npz` preserves compact unmasked curves; no endpoint frame or covariance was loaded for these fits. '
        'Each figure has a plotted-series CSV; static arrays and diagnostics have their own files. '
        'All old figures, raw datasets and the manuscript are untouched.',
        'All spread summaries describe trajectories themselves: SD, IQR and empirical percentiles. No bootstrap confidence intervals or SEM are used.',
    ])
    (out/'FINDINGS.md').write_text('\n\n'.join(texts)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    data.prior.configure_matplotlib()
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
                         'legend.fontsize':8,'legend.frameon':False})
    print('[validate] resolving checksum-bound preparation and verifying all result pairs',flush=True)
    identity=data.preparation_identity()
    endpoints={f'primary_{ny}':data.load_endpoint(ny,identity) for ny in SIZES}
    for ny in (30,40,50):
        avg,ids,provenance=data.prior.load_legacy_size(ny)
        endpoints[f'legacy_{ny}']=Endpoint(ny,'legacy_streaming',ids,avg,None,provenance)
    compact={}
    for key,e in endpoints.items():
        compact[f'{key}_xavg']=e.xavg;compact[f'{key}_sample_ids']=e.ids
        if e.xresolved is not None:compact[f'{key}_xresolved']=e.xresolved
    np.savez_compressed(out/'endpoint_curves.npz',**compact)
    provenance=identity['evidence']+[p for e in endpoints.values() for p in e.provenance]
    write_json(out/'input_provenance.json',dict(preparation=identity,files=provenance,
               primary_sizes=list(SIZES),samples_per_size=100,legacy_sizes=[30,40,50],
               estimator='sum_y,mu,nu |C[x,y,mu;x,y+r,nu]|^2/(2Ny); xavg=mean_x',
               endpoint='2Ny',independent_separations='0..Ny/2',cohorts='Separate, never pooled'))
    print('[fit] trajectory-first windows, cutoff diagnostics, separate legacy cohorts',flush=True)
    rows,summaries=summarize_fits(endpoints)
    write_csv(out/'trajectory_fits.csv',rows);write_json(out/'fit_summary.json',summaries)
    regression=regressions(summaries);write_csv(out/'finite_size_regressions.csv',regression)
    validation=old_regression(endpoints['primary_32'],summaries)
    write_json(out/'old_result_regression.json',validation)
    write_json(out/'typical_trajectories.json',{str(n):typical(endpoints[f'primary_{n}']) for n in (32,40,50,60)})
    print('[validated] Ny32 selection and 400 historical fits reproduced',flush=True)
    benchmarks,diagnostics,static_sources=compute_benchmarks(out,args.threads)
    print('[plot] generating review figures and atlas',flush=True)
    review=Review(out)
    try:
        for ny in tqdm((32,40,50,60),desc='trajectory and chord figures'):
            representative_plots(review,endpoints[f'primary_{ny}']);logchord_plot(review,endpoints[f'primary_{ny}'])
        exponent_distributions(review,endpoints)
        scaling_plots(review,endpoints,summaries,regression)
        window_plots(review,summaries)
        cohort_plot(review,endpoints,summaries)
        benchmark_rows=benchmark_plots(review,endpoints,benchmarks)
    finally:review.pdf.close()
    write_csv(out/'benchmark_comparison.csv',benchmark_rows)
    write_csv(out/'all_plotted_series.csv',review.rows)
    write_json(out/'figure_index.json',review.pages)
    captions='# Figure index and captions\n\n'+''.join(f"## {p['page']}. {p['stem']}\n\n{p['caption']}\n\n" for p in review.pages)
    (out/'FIGURES.md').write_text(captions)
    write_findings(out,endpoints,summaries,regression,benchmark_rows,review.pages)
    write_json(out/'analysis_manifest.json',dict(schema='large_ny_correlator_review_v1',
               scripts=[data.file_record(Path(__file__),'review_runner'),data.file_record(Path(data.__file__),'endpoint_loader')],
               reused_analysis_sources=[data.file_record(Path(data.prior.__file__),'legacy_loader_and_plot_style')],
               static_sources=static_sources,figures=len(review.pages),trajectory_fit_rows=len(rows),
               no_stochastic_simulation=True,no_manuscript_changes=True))
    print(f'[done] {len(review.pages)} figures; {len(rows)} trajectory fit rows; {out}',flush=True)


if __name__=='__main__':main()
