#!/usr/bin/env python3
"""Wall-only tail fits, deliberately separate from the technical report."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import build_correlator_summary_3x1 as summary

data = summary.data
OUT = summary.matched.PROJECT / 'outputs/wall_only_tail_windows_v1'
LOWERS = (2, 5, 8, 10, 12, 15)
WALLS = (5, 15)


def write_csv(name, rows):
    with (OUT / name).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, name):
    for extension in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{extension}', dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Reload verified scientific inputs, not an unchecked plotting cache.
    identity = data.preparation_identity()
    endpoints = {ny: data.load_endpoint(ny, identity) for ny in summary.SIZES}
    mean_rows, sample_rows, distribution_rows, plotted_rows = [], [], [], []
    for ny, endpoint in endpoints.items():
        for wall in WALLS:
            curves = endpoint.xresolved[:, wall, :]
            for lower in LOWERS:
                key = dict(Ny=ny, x=wall, lower=lower, upper=ny//2)
                mean_rows.append(dict(key, **data.fit(curves.mean(0), ny, lower, ny//2)))
                fits = [data.fit(c, ny, lower, ny//2) for c in curves]
                sample_rows.extend(dict(key, sample_id=int(sid), **fit)
                                   for sid, fit in zip(endpoint.ids, fits))
                distribution_rows.append(dict(key, **data.distribution([f['beta'] for f in fits]),
                                              invalid_samples=sum(not f['valid'] for f in fits)))
            mean = curves.mean(0)
            for r in range(1, ny//2+1):
                plotted_rows.append(dict(Ny=ny, x=wall, ry=r, mean_correlator=float(mean[r]),
                    log_chord=float(np.log(data.prior.chord(ny, r))),
                    log_mean_correlator=float(np.log(mean[r])), in_primary_fit=r >= 8))

    data.prior.configure_matplotlib()
    plt.rcParams.update({'font.size':8, 'axes.labelsize':8, 'legend.fontsize':8,
                         'xtick.labelsize':8, 'ytick.labelsize':8})
    colors = ('#0072B2', '#D55E00')
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65))
    r = np.arange(1, 31)
    x = np.log(data.prior.chord(60, r))
    keep = r >= 8
    for ax, wall, color, letter in zip(axes, WALLS, colors, 'ab'):
        mean = endpoints[60].xresolved[:, wall, :].mean(0)
        fit = data.fit(mean, 60, 8, 30)
        ax.axvspan(x[keep].min(), x[keep].max(), color='0.5', alpha=.18, lw=0)
        ax.plot(x[~keep], np.log(mean[1:])[~keep], 'x', color='0.65', ms=3)
        ax.plot(x[keep], np.log(mean[1:])[keep], 'o', color=color, mfc='white', ms=3)
        ax.plot(x[keep], fit['log_amplitude']-fit['beta']*x[keep], 'k--', lw=.9)
        ax.text(.04, .07, rf'$x={wall}$'+'\n'+rf'$\beta={fit["beta"]:.3f}$', transform=ax.transAxes)
        ax.set_xlabel(r'$\log d_{60}(r_y)$')
        ax.set_ylabel(r'$\log\overline{C_G(x,r_y)}$')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        summary.style(ax, letter)
    fig.subplots_adjust(left=.08, right=.985, bottom=.2, top=.9, wspace=.32)
    save(fig, 'ny60_wall_tail_fits')

    fig, ax = plt.subplots(figsize=(3.375, 2.7))
    for wall, color, marker in zip(WALLS, colors, ('o','s')):
        rows = [q for q in mean_rows if q['Ny']==60 and q['x']==wall]
        ax.plot([q['lower'] for q in rows], [q['beta'] for q in rows],
                marker=marker, color=color, mfc='white', ms=3, lw=.9, label=rf'$x={wall}$')
    ax.axhline(2, color='0.35', ls='--', lw=.8)
    ax.axvline(8, color='0.65', ls=':', lw=.8)
    ax.set_xticks([2,5,8,12,15])
    ax.set_xlabel(r'$r_{\min}\quad(r_{\max}=30)$')
    ax.set_ylabel(r'$\beta$')
    ax.legend(frameon=False, loc='lower right')
    summary.style(ax, 'a')
    fig.subplots_adjust(left=.18, right=.97, bottom=.2, top=.9)
    save(fig, 'ny60_wall_window_sensitivity')

    write_csv('ensemble_mean_fits.csv', mean_rows)
    write_csv('sample_fits.csv', sample_rows)
    write_csv('sample_exponent_distributions.csv', distribution_rows)
    write_csv('plotted_curves.csv', plotted_rows)
    provenance = dict(
        alpha1=1, sizes=summary.SIZES, walls=WALLS, lower_bounds=LOWERS, upper='Ny/2',
        primary_window='8..Ny/2', fit='free-intercept OLS of log C versus log chord',
        averaging='ensemble-mean fit and sample-wise fits saved separately',
        cutoff='strictly finite positive points only; at least four; no display threshold in fits',
        invalid_fits='retained with reason and point counts; no slope or R-squared rejection',
        report_inclusion=False, inputs=sum([e.provenance for e in endpoints.values()], [])+identity['evidence'],
        sources=[data.file_record(Path(__file__), 'analysis_script'),
                 data.file_record(Path(data.__file__), 'loader_and_fitter')])
    (OUT/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps([q for q in mean_rows if q['Ny']==60], indent=2))
    print('[done]', OUT)


if __name__ == '__main__':
    main()
