#!/usr/bin/env python3
"""Single-column endpoint correlator summary and exact spatial decomposition."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator, FixedLocator, ScalarFormatter, LogFormatterMathtext
import numpy as np
import plot_alpha_comparison_ny60 as matched

data = matched.data
OUT = matched.PROJECT / 'outputs/correlator_summary_3x1_v2_r8_half'
REPORT = data.ROOT / 'technical_report/figures'
SIZES = (24, 28, 32, 40, 50, 60)
SITES = (5, 6, 10, 14, 15)
BOUNDARY = (5, 6, 14, 15)
CUTOFF = 1e-8


def partition_curves(x):
    """Return Nx-normalized sums, not unweighted regional averages."""
    other = sorted(set(range(x.shape[-2])) - set(BOUNDARY))
    boundary = x[..., BOUNDARY, :].sum(axis=-2) / x.shape[-2]
    rest = x[..., other, :].sum(axis=-2) / x.shape[-2]
    total = x.mean(axis=-2)
    np.testing.assert_allclose(boundary + rest, total, rtol=2e-14, atol=1e-18)
    return boundary, rest, total


def anchored_fit(means, lower=2, upper='quarter'):
    """Equal total weight per size, free slope through measured half-chord anchor."""
    denominators = {'quarter': 4, 'third': 3, 'half': 2}
    if upper not in denominators:
        raise ValueError('Unknown upper-window rule')
    curves = {}
    num = den = 0.
    for ny, c in means.items():
        r = np.arange(1, ny//2+1)
        c = np.asarray(c)
        if c.shape != (ny//2+1,) or not np.isfinite(c).all() or np.any(c <= 0):
            raise ValueError('Invalid mean curve for logarithmic anchor')
        x = np.log(data.prior.chord(ny, r) / data.prior.chord(ny, ny//2))
        y = np.log(c[1:] / c[-1])
        hi = ny//denominators[upper]
        keep = (r >= lower) & (r <= hi)
        if keep.sum() < 4:
            raise ValueError('At least four points required per size')
        num += np.mean(x[keep] * y[keep])
        den += np.mean(x[keep] ** 2)
        curves[ny] = (r, x, y, keep)
    slope = num / den
    rss = sum(np.mean((y[k]-slope*x[k])**2) for r,x,y,k in curves.values())
    tss = sum(np.mean(y[k]**2) for r,x,y,k in curves.values())
    return dict(slope=float(slope), beta=float(-slope), R0_squared=float(1-rss/tss) if tss > 0 else None,
                lower=lower, upper=upper, weighting='each size has total weight one'), curves


def shade_fit_window(ax, curves):
    """Light union and darker intersection of the size-specific fit ranges."""
    spans = {ny: [float(x[k].min()), float(x[k].max())]
             for ny,(r,x,y,k) in curves.items()}
    lo, hi = min(v[0] for v in spans.values()), max(v[1] for v in spans.values())
    common_lo, common_hi = max(v[0] for v in spans.values()), min(v[1] for v in spans.values())
    ax.axvspan(lo, hi, color='0.5', alpha=.12, lw=0, zorder=0)
    if common_lo < common_hi:
        ax.axvspan(common_lo, common_hi, color='0.5', alpha=.12, lw=0, zorder=0)
    return dict(per_size=spans, union=[lo,hi], intersection=[common_lo,common_hi])


def window_scan(means):
    rows=[]
    for lower in (1,2,3,4,5,6,8):
        for upper,denom in (('quarter',4),('third',3),('half',2)):
            counts={n:max(0,n//denom-lower+1) for n in means}
            row=dict(lower=lower,upper=upper,points_per_size=counts,
                     valid=min(counts.values())>=4,beta=None)
            if row['valid']:
                row.update(anchored_fit(means,lower,upper)[0])
            else:
                row['reason']='fewer than four points for at least one size; no sizes dropped'
            rows.append(row)
    return rows


def style(ax, letter):
    ax.tick_params(direction='in', top=True, right=True)
    ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.text(-.17, 1.035, f'({letter})', transform=ax.transAxes, fontsize=9)


def logcurve(ax, mean, ny, label, color, marker, ls='-', rows=None, panel='', raw_separation=False):
    r = np.arange(1, ny//2+1)
    xx = r if raw_separation else np.log(data.prior.chord(ny, r))
    keep = mean[1:] > CUTOFF
    positive = np.where(mean[1:] > 0, mean[1:], np.nan)
    yy = positive if raw_separation else np.log(positive)
    ax.plot(xx, np.where(keep, yy, np.nan), color=color, marker=marker, ls=ls,
            ms=3, mfc='white', mew=.65, lw=.75, label=label)
    if rows is not None:
        for ri, xi, ci, yi, shown in zip(r, xx, mean[1:], yy, keep):
            rows.append(dict(panel=panel, series=label, Ny=ny, ry=int(ri),
                             x=float(xi), y=float(yi), unmasked_correlator=float(ci),
                             displayed=bool(shown), in_fit=False))


def save(fig, stem, report=False):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT/f'{stem}.{ext}', dpi=300)
        if report:
            fig.savefig(REPORT/f'{stem}.{ext}', dpi=300)
    plt.close(fig)


def main(raw_separation=False):
    global OUT
    saved_output = matched.PROJECT / 'outputs/correlator_summary_3x1_v2_r8_half'
    if raw_separation:
        OUT = matched.PROJECT / 'outputs/correlator_summary_3x1_v3_ry_loglog'
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    if raw_separation:
        # Axis-only remake from the preserved, previously verified figure export.
        # Do not reinterpret historical data using today's mutable engine copies.
        from types import SimpleNamespace
        cache = saved_output / 'compact_curves.npz'
        with np.load(cache, allow_pickle=False) as z:
            endpoints = {}
            for ny in SIZES:
                x = z[f'alpha1_Ny{ny}_xresolved']
                assert x.shape == (100,20,ny//2+1)
                endpoints[ny] = SimpleNamespace(xresolved=x, xavg=x.mean(1), provenance=[])
            x = z['alpha3_Ny60_xresolved']
            assert x.shape == (100,20,31)
            alpha3 = SimpleNamespace(xresolved=x, xavg=x.mean(1), provenance=[])
        # Check the compact means against the independently saved plotted table.
        with (saved_output/'plotted_data.csv').open() as f:
            for row in csv.DictReader(f):
                n, r = int(row['Ny']), int(row['ry'])
                if row['panel'] == 'a':
                    c = alpha3.xavg.mean(0) if '3' in row['series'] else endpoints[60].xavg.mean(0)
                elif row['panel'] == 'b':
                    site = int(row['series'].split('=')[1].rstrip('$'))
                    c = endpoints[60].xresolved[:,site].mean(0)
                elif row['panel'] == 'c':
                    c = endpoints[n].xavg.mean(0)
                else:
                    continue
                np.testing.assert_allclose(c[r], float(row['unmasked_correlator']), rtol=2e-14, atol=0)
        identity = {'evidence':[data.file_record(saved_output/name, 'preserved_figure_export')
                    for name in ('compact_curves.npz','plotted_data.csv','summary.json')]}
    else:
        identity = data.preparation_identity()
        endpoints = {ny: data.load_endpoint(ny, identity) for ny in SIZES}
        alpha3, hashes3 = matched.load_alpha3()
        for name in ('src/classA_U1FGTN_gpu.py', 'src/occupied_frame_gpu.py'):
            assert identity['source_hashes'][name] == hashes3[name]
    means = {}
    for ny, e in endpoints.items():
        np.testing.assert_allclose(e.xresolved.mean(axis=1), e.xavg, rtol=2e-14, atol=1e-15)
        means[ny] = e.xresolved.mean(axis=1).mean(axis=0)
    np.testing.assert_allclose(alpha3.xresolved.mean(axis=1), alpha3.xavg, rtol=2e-14, atol=1e-15)
    fit, anchored = anchored_fit(means, lower=8, upper='half')
    sensitivity = window_scan(means)
    per_size_sensitivity = [dict(Ny=ny,upper=upper,
        anchored_beta=anchored_fit({ny:c},upper=upper)[0]['beta'],
        free_intercept_beta=data.fit(c,ny,2,ny//denom)['beta'])
        for ny,c in means.items() for upper,denom in (('quarter',4),('half',2))]
    x60 = endpoints[60].xresolved.mean(axis=0)
    boundary_s, rest_s, total_s = partition_curves(endpoints[60].xresolved)
    boundary, rest, total = (v.mean(axis=0) for v in (boundary_s, rest_s, total_s))
    data.prior.configure_matplotlib()
    plt.rcParams.update({'font.size':8, 'axes.labelsize':8, 'axes.titlesize':8,
                         'xtick.labelsize':8, 'ytick.labelsize':8, 'legend.fontsize':8})
    fig, axes = plt.subplots(3, 1, figsize=(3.375, 6.0))
    rows = []
    logcurve(axes[0], means[60], 60, r'$\alpha_1=1$', '#0072B2', 'o', rows=rows, panel='a', raw_separation=raw_separation)
    logcurve(axes[0], alpha3.xavg.mean(axis=0), 60, r'$\alpha_1=3$', '#D55E00', '^', ':', rows, 'a', raw_separation=raw_separation)
    axes[0].set_ylabel(r'$\log\overline{C_G^{\mathrm{av}}(r_y)}$')
    axes[0].legend(loc='lower left', frameon=False, handletextpad=.4)
    for site, color, marker, ls in zip(SITES,
            ('#0072B2','#E69F00','#555555','#009E73','#D55E00'),
            ('o','s','D','^','v'), ('-','--',':','-.',':')):
        logcurve(axes[1], x60[site], 60, rf'$x={site}$', color, marker, ls, rows, 'b', raw_separation=raw_separation)
    axes[1].set_ylabel(r'$\log\overline{C_G(x,r_y)}$')
    axes[1].text(.97, .95, r'DWs at $x=5,15$', transform=axes[1].transAxes,
                 ha='right', va='top', fontsize=8)
    axes[1].legend(loc='lower left', frameon=False, ncol=2, columnspacing=.7,
                   handlelength=1.3, handletextpad=.3, labelspacing=.25)
    for ax in axes[:2]:
        ax.set_xlim(-.06,3.03); ax.set_xticks([0,1,2,3])
        ax.set_ylim(np.log(CUTOFF)-.3,-2.5)
        ax.set_xlabel(r'$\log d_{N_y}(r_y)$', labelpad=2)
    fit_shading = shade_fit_window(axes[2], anchored)
    for ny,color,marker in zip(SIZES,
            ('#D55E00','#009E73','#0072B2','#CC79A7','#E69F00','#333333'),
            ('^','s','o','v','D','>')):
        r, x, y, keep = anchored[ny]
        axes[2].plot(x, y, marker=marker, ls='none', color=color, ms=3,
                     mfc='white', mew=.65, label=rf'${ny}$')
        for ri,xi,yi,k in zip(r,x,y,keep):
            rows.append(dict(panel='c', series=str(ny), Ny=ny, ry=int(ri), x=float(xi),
                             y=float(yi), unmasked_correlator=float(means[ny][ri]),
                             displayed=True, in_fit=bool(k)))
    grid = np.linspace(-3.02, 0, 200)
    axes[2].plot(grid, fit['slope']*grid, 'k--', lw=.9, zorder=5)
    axes[2].text(.04,.09,rf'$\beta={fit["beta"]:.2f}$', transform=axes[2].transAxes, fontsize=8)
    axes[2].legend(title=r'$N_y$', loc='upper right', ncol=3, frameon=False,
                   handlelength=.6, columnspacing=.5, handletextpad=.15, labelspacing=.2,
                   title_fontsize=8)
    axes[2].set_xlabel(r'$\log[d_{N_y}(r_y)/d_{N_y}(N_y/2)]$', labelpad=2)
    axes[2].set_ylabel(r'$\log[\overline{C_G^{\mathrm{av}}(r_y)}/\overline{C_G^{\mathrm{av}}(N_y/2)}]$')
    axes[2].set_xlim(-3.04,.06); axes[2].set_xticks([-3,-2,-1,0])
    axes[2].set_ylim(-.3,9.4)
    for ax, letter in zip(axes, 'abc'): style(ax, letter)
    if raw_separation:
        axes[0].set_ylabel(r'$\overline{C_G^{\mathrm{av}}(r_y)}$')
        axes[1].set_ylabel(r'$\overline{C_G(x,r_y)}$')
        for ax in axes[:2]:
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlim(.94,32); ax.set_ylim(CUTOFF*np.exp(-.3),np.exp(-2.5))
            ax.xaxis.set_major_locator(FixedLocator([1,2,5,10,20,30]))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_locator(FixedLocator([1e-8,1e-6,1e-4,1e-2]))
            ax.yaxis.set_major_formatter(LogFormatterMathtext())
            ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
            ax.set_xlabel(r'$r_y$',labelpad=2)
    fig.subplots_adjust(left=.205,right=.975,bottom=.075,top=.975,hspace=.53)
    save(fig, 'hard_wall_correlator_summary_3x1', report=not raw_separation)

    fig, axs = plt.subplots(1,2,figsize=(7.05,2.95),sharex=True,sharey=True)
    comparison_shading={}
    for ax,upper,title in zip(axs,('quarter','half'),
            (r'$2\leq r_y\leq\lfloor N_y/4\rfloor$',r'$2\leq r_y\leq N_y/2$')):
        result,curves=anchored_fit(means,upper=upper)
        comparison_shading[upper]=shade_fit_window(ax,curves)
        for ny,color,marker in zip(SIZES,
                ('#D55E00','#009E73','#0072B2','#CC79A7','#E69F00','#333333'),
                ('^','s','o','v','D','>')):
            r,x,y,k=curves[ny]
            ax.plot(x[k],y[k],ls='none',marker=marker,color=color,mfc='white',ms=3,mew=.65,label=str(ny))
            ax.plot(x[~k],y[~k],'x',color='0.6',ms=2.3,mew=.55,zorder=1)
        xx=np.linspace(-3.02,0,200)
        ax.plot(xx,result['slope']*xx,'k--',lw=.9)
        ax.text(.05,.09,rf'$\beta={result["beta"]:.4f}$',transform=ax.transAxes)
        ax.set_title(title);ax.set_xlim(-3.04,.06);ax.set_ylim(-.3,9.4)
        ax.set_xticks([-3,-2,-1,0]);ax.set_xlabel(r'$\log[d_{N_y}(r_y)/d_{N_y}(N_y/2)]$')
    axs[0].set_ylabel(r'$\log[\overline{C}(r_y)/\overline{C}(N_y/2)]$')
    axs[1].legend(title=r'$N_y$',ncol=3,frameon=False,loc='upper right',handlelength=.7,columnspacing=.5)
    for ax,letter in zip(axs,'ab'):style(ax,letter)
    fig.subplots_adjust(left=.08,right=.985,bottom=.21,top=.85,wspace=.12)
    save(fig,'quarter_vs_half_fit_windows')

    fig,ax=plt.subplots(figsize=(3.375,2.65))
    for upper,color,marker,label in (
            ('quarter','#0072B2','o',r'$N_y/4$'),
            ('third','#009E73','s',r'$N_y/3$'),
            ('half','#D55E00','^',r'$N_y/2$')):
        selected=[q for q in sensitivity if q['upper']==upper]
        ax.plot([q['lower'] for q in selected],
                [q['beta'] if q['valid'] else np.nan for q in selected],
                color=color,marker=marker,mfc='white',ms=3.5,lw=.8,label=label)
    ax.axhline(2,color='0.4',ls=':',lw=.8)
    ax.set_xlabel(r'$r_{\min}$');ax.set_ylabel(r'$\beta$')
    ax.set_xticks([1,2,4,6,8]);ax.set_ylim(1.98,2.72)
    ax.legend(title=r'$r_{\max}$',frameon=False,loc='upper right',fontsize=8)
    ax.tick_params(direction='in',top=True,right=True)
    ax.xaxis.set_minor_locator(NullLocator());ax.yaxis.set_minor_locator(NullLocator())
    fig.subplots_adjust(left=.19,right=.97,bottom=.2,top=.97)
    save(fig,'fit_window_sensitivity')

    # Separate supporting diagnostic, not a fourth panel in the main figure.
    fig, axs = plt.subplots(1,2,figsize=(7.05,2.75))
    for curve,label,color,marker,ls in (
            (total,r'total','#333333','o','-'),
            (boundary,r'$B$ contribution','#0072B2','^','--'),
            (rest,r'$U$ contribution','#D55E00','s',':')):
        logcurve(axs[0],curve,60,label,color,marker,ls,rows,'decomposition')
    axs[0].set_xlabel(r'$\log d_{60}(r_y)$')
    axs[0].set_ylabel(r'$\log C$'); axs[0].set_xticks([0,1,2,3])
    axs[0].set_ylim(np.log(CUTOFF)-.3,-3)
    axs[0].legend(frameon=False,loc='lower left',handlelength=1.6)
    r=np.arange(1,31); bulk=x60[10,1:]; keep=(r>=2)&(r<=6)&(bulk>CUTOFF)
    assert keep.sum()==5
    slope,intercept=np.polyfit(r[keep],np.log(bulk[keep]),1)
    predicted=intercept+slope*r[keep]
    bulk_fit=dict(window=[2,6],slope=float(slope),xi_C=float(-1/slope),
                  R_squared=float(1-np.sum((np.log(bulk[keep])-predicted)**2)/
                                    np.sum((np.log(bulk[keep])-np.log(bulk[keep]).mean())**2)))
    axs[1].plot(r,np.where(bulk>CUTOFF,np.log(bulk),np.nan),'D',color='#555555',
                mfc='white',ms=3.3,label=r'$x=10$')
    grid=np.linspace(2,6,100)
    axs[1].plot(grid,intercept+slope*grid,'k--',lw=.9,label='exponential fit')
    axs[1].axvspan(2,6,color='0.5',alpha=.15,lw=0,zorder=0)
    axs[1].set_xlabel(r'$r_y$'); axs[1].set_ylabel(r'$\log\overline{C_G(10,r_y)}$')
    axs[1].set_xlim(.8,9.2); axs[1].set_ylim(np.log(CUTOFF)-.3,-2.5)
    axs[1].set_xticks([1,3,5,7,9]); axs[1].legend(frameon=False,loc='upper right')
    for ax,letter in zip(axs,'ab'):style(ax,letter)
    fig.subplots_adjust(left=.085,right=.985,bottom=.21,top=.92,wspace=.27)
    save(fig,'boundary_bulk_addition_diagnostic')
    with (OUT/'plotted_data.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    scan_rows=[dict(lower=q['lower'],upper=q['upper'],valid=q['valid'],beta=q['beta'],
                    points_per_size=json.dumps(q['points_per_size']),reason=q.get('reason',''))
               for q in sensitivity]
    with (OUT/'fit_window_sensitivity.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(scan_rows[0]));writer.writeheader();writer.writerows(scan_rows)
    # Include every r, including hidden data, in a compact reproducible export.
    np.savez_compressed(OUT/'compact_curves.npz',**{f'alpha1_Ny{n}_xresolved':e.xresolved for n,e in endpoints.items()},
                        alpha3_Ny60_xresolved=alpha3.xresolved,boundary=boundary,rest=rest,total=total)
    summary=dict(primary_fit=fit,sensitivity=sensitivity,bulk_fit=bulk_fit,
                 panels_ab_axes='raw separation and correlator on log-log axes' if raw_separation else 'log chord and log correlator on linear axes',
                 fit_shading=fit_shading,comparison_shading=comparison_shading,
                 per_size_sensitivity=per_size_sensitivity,
                 boundary_columns=BOUNDARY,remaining_columns=sorted(set(range(20))-set(BOUNDARY)),
                 boundary_fraction={str(r):float(boundary[r]/total[r]) for r in (1,2,5,15,30)},
                 display_cutoff=CUTOFF,sizes=SIZES,samples_per_size=100,
                 estimator='arithmetic mean of trajectory-wise squared correlators, then log',
                 anchor='ratio of ensemble means at r and Ny/2; not mean of trajectory ratios',
                 inputs=sum([e.provenance for e in endpoints.values()],[])+alpha3.provenance+identity['evidence'],
                 sources=[data.file_record(Path(__file__),'plot_script'),
                          data.file_record(Path(data.__file__),'loader'),
                          data.file_record(Path(matched.__file__),'alpha3_loader')])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('primary_fit','sensitivity','bulk_fit','boundary_fraction')},indent=2))
    print('[done]',OUT)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw-separation',action='store_true',
                        help='Separate copy with panels a,b on log-log axes versus ry; report unchanged.')
    main(raw_separation=parser.parse_args().raw_separation)
