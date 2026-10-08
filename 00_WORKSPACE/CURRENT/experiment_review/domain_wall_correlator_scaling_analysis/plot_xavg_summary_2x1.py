"""Standalone x-averaged comparison and preserved multi-size power-law collapse."""
import csv
import hashlib
import importlib.util
import json

from plot_xavg_cutoff20 import PROJECT, ROOT, SOURCE, CUTOFF
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter, LogFormatterMathtext
import numpy as np

SIZES = (24, 28, 32, 40, 50, 60)
OUT = PROJECT / 'outputs/xavg_summary_2x1_loglog_v2_ry_ge2'


def collapse_fit(means):
    curves = {}
    numerator = denominator = 0.
    for ny, c in means.items():
        r = np.arange(2, ny // 2 + 1)
        d_ratio = np.sin(np.pi * r / ny)
        c_ratio = c[r] / c[ny // 2]
        assert np.all(np.isfinite(c_ratio)) and np.all(c_ratio > 0)
        keep = (r >= 8) & (r <= ny // 2)
        assert keep.sum() >= 4
        x, y = np.log(d_ratio), np.log(c_ratio)
        numerator += np.mean(x[keep] * y[keep])
        denominator += np.mean(x[keep] ** 2)
        curves[ny] = (r, d_ratio, c_ratio, keep)
    return -numerator / denominator, curves


def main():
    style_path = ROOT / '00_WORKSPACE/CURRENT/manuscript_Overleaf/figures/new_figure/sources/manuscript_typography.py'
    spec = importlib.util.spec_from_file_location('typography', style_path)
    typography = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(typography)
    typography.configure_style()
    with np.load(SOURCE / 'compact_curves.npz', allow_pickle=False) as z:
        means = {}
        for ny in SIZES:
            x = z[f'alpha1_Ny{ny}_xresolved']
            assert x.shape == (100, 20, ny // 2 + 1)
            means[ny] = x.mean(axis=1).mean(axis=0)
        x3 = z['alpha3_Ny60_xresolved']
        assert x3.shape == (100, 20, 31)
        mean3 = x3.mean(axis=1).mean(axis=0)
    with (SOURCE / 'plotted_data.csv').open() as f:
        for row in csv.DictReader(f):
            if row['panel'] not in ('a', 'c'):
                continue
            c = mean3 if row['panel'] == 'a' and row['series'] == r'$\alpha_1=3$' else means[int(row['Ny'])]
            np.testing.assert_allclose(c[int(row['ry'])], float(row['unmasked_correlator']), rtol=2e-14, atol=0)
    beta, curves = collapse_fit(means)
    old_fit = json.loads((SOURCE / 'summary.json').read_text())['primary_fit']
    np.testing.assert_allclose(beta, old_fit['beta'], rtol=2e-14)
    synthetic = {ny: np.r_[1., np.sin(np.pi * np.arange(1, ny//2+1)/ny)**(-2.3)] for ny in SIZES}
    np.testing.assert_allclose(collapse_fit(synthetic)[0], 2.3, rtol=1e-14)
    fig, axes = plt.subplots(2, 1, figsize=(3.375, 4.5))
    rows = []
    r = np.arange(2, 31)
    for alpha, c, color, marker, ls in [(1, means[60], '#0072B2', 'o', '-'),
                                      (3, mean3, '#D55E00', '^', ':')]:
        shown = np.isfinite(c[r]) & (c[r] > CUTOFF)
        axes[0].plot(r, np.where(shown, c[r], np.nan), color=color, marker=marker,
                     ls=ls, lw=.85, ms=3, mfc='white', mew=.7, label=rf'$\alpha_1={alpha}$')
        rows.extend(dict(panel='a', alpha1=alpha, Ny=60, ry=int(ri), x=float(ri),
                         y=float(ci), correlator=float(ci), displayed=bool(k), in_fit=False)
                    for ri, ci, k in zip(r, c[r], shown))
    axes[0].set(xscale='log', yscale='log', xlim=(1.88, 32), ylim=(CUTOFF/2, .001),
                xlabel=r'$r_y$', ylabel=r'$C(r_y)$')
    axes[0].xaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 30]))
    axes[0].xaxis.set_major_formatter(ScalarFormatter())
    axes[0].yaxis.set_major_locator(FixedLocator([1e-20, 1e-16, 1e-12, 1e-8, 1e-4]))
    axes[0].legend(loc='lower left', frameon=False, handlelength=1.8, handletextpad=.4)
    starts = [float(d[k].min()) for r, d, c, k in curves.values()]
    axes[1].axvspan(min(starts), 1, color='0.5', alpha=.12, lw=0)
    axes[1].axvspan(max(starts), 1, color='0.5', alpha=.12, lw=0)
    for ny, color, marker in zip(SIZES, ('#D55E00','#009E73','#0072B2','#CC79A7','#E69F00','#333333'),
                                 ('^','s','o','v','D','>')):
        rr, d, c, keep = curves[ny]
        axes[1].plot(d, c, marker=marker, ls='none', color=color, ms=3,
                     mfc='white', mew=.65, label=rf'${ny}$')
        rows.extend(dict(panel='b', alpha1=1, Ny=ny, ry=int(ri), x=float(di), y=float(ci),
                         correlator=float(means[ny][ri]), displayed=True, in_fit=bool(k))
                    for ri, di, ci, k in zip(rr, d, c, keep))
    grid = np.geomspace(.1, 1, 200)
    axes[1].plot(grid, grid**(-beta), 'k--', lw=.9)
    axes[1].text(.04, .09, rf'$\beta={beta:.2f}$', transform=axes[1].transAxes, fontsize=8)
    axes[1].set(xscale='log', yscale='log', xlim=(.098, 1.06), ylim=(.75, 200),
                xlabel=r'$d_{N_y}(r_y)/d_{N_y}(N_y/2)$', ylabel=r'$C(r_y)/C(N_y/2)$')
    axes[1].xaxis.set_major_locator(FixedLocator([.1, .2, .5, 1]))
    axes[1].xaxis.set_major_formatter(ScalarFormatter())
    axes[1].yaxis.set_major_locator(FixedLocator([1, 10, 100]))
    axes[1].legend(title=r'$N_y$', loc='upper right', ncol=3, frameon=False,
                   handlelength=.6, columnspacing=.5, handletextpad=.15, labelspacing=.2)
    for ax, letter in zip(axes, 'ab'):
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.minorticks_off()
        ax.tick_params(direction='in', top=True, right=True)
        ax.text(-.19, 1.045, f'({letter})', transform=ax.transAxes, fontsize=9)
        assert ax.get_xscale() == ax.get_yscale() == 'log'
    fig.subplots_adjust(left=.19, right=.975, bottom=.105, top=.955, hspace=.48)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(plt.matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1, artist.get_text()
    assert all(row['ry'] >= 2 for row in rows)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / f'{OUT.name}.{ext}', dpi=300)
    plt.close(fig)
    with (OUT / 'plotted_data.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    provenance = dict(beta=float(beta), fit_window='8 <= ry <= Ny/2', sizes=SIZES,
                      fit='log(C(r)/C(Ny/2)) = -beta log(d(r)/d(Ny/2)); equal total weight per size',
                      estimator='C(r) = trajectory mean of (1/Nx) sum_x C_G(x,r); not C of the averaged state',
                      samples_per_ensemble=100, Nx=20, cutoff_panel_a=CUTOFF, excluded_ry=[0, 1],
                      caption='(a) X- and trajectory-averaged squared correlator at Ny=60 for alpha1=1,3. (b) Alpha1=1 normalized chord collapse for Ny=24,28,32,40,50,60, with the preserved power-law fit. Gray shading marks union/intersection of size-specific fit windows. All ensembles use 100 independent pure-initial-state perfect-correction hard-wall trajectories, Nx=20, alpha2=30, nshell=1, raster-y, evaluated at 2Ny cycles. No uncertainty bars. C denotes the spatial and trajectory average, performed after evaluating the squared correlator.',
                      validation='Cached curves match preserved table, beta reproduces prior fit, synthetic beta recovered. No fresh raw-campaign validation.',
                      inputs={str(SOURCE / name): hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                              for name in ['compact_curves.npz','plotted_data.csv','summary.json']},
                      manuscript_changed=False)
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(f'beta={beta:.12f}; all checks passed')
    print(OUT / f'{OUT.name}.png')


if __name__ == '__main__':
    main()
