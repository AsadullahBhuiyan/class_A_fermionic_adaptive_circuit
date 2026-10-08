"""Standalone wall/bulk correlators: linear separation, logarithmic C axis."""
import argparse
import csv
import hashlib
import importlib.util
import json

from plot_xavg_cutoff20 import PROJECT, ROOT, SOURCE, CUTOFF
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, ScalarFormatter
import numpy as np


def main(log_x=False, trivial_bulk=False):
    style_path = ROOT / '00_WORKSPACE/CURRENT/manuscript_Overleaf/figures/new_figure/sources/manuscript_typography.py'
    spec = importlib.util.spec_from_file_location('typography', style_path)
    typography = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(typography)
    typography.configure_style()
    out = PROJECT / 'outputs/xresolved_Ny60_linear_ry_logC'
    if log_x:
        out = out.with_name('xresolved_Ny60_ry_loglog')
    if trivial_bulk:
        log_x = True
        out = out.with_name('xresolved_Ny60_trivial_bulk_unmasked_loglog')
    out.mkdir(parents=True, exist_ok=True)
    with (SOURCE / 'plotted_data.csv').open() as f:
        previous = list(csv.DictReader(f))
    with np.load(SOURCE / 'compact_curves.npz', allow_pickle=False) as z:
        samples = z['alpha1_Ny60_xresolved']
        assert samples.shape == (100, 20, 31)
        mean = samples.mean(axis=0)
    r = np.arange(1, 31)
    fig, ax = plt.subplots(figsize=(3.375, 2.30))
    rows = []
    series = [(5, '#0072B2', 'o', '-'), (10, '#555555', 'D', ':'),
              (15, '#D55E00', 'v', ':')]
    if trivial_bulk:
        series = [(2, '#009E73', 's', '--'), (19, '#CC79A7', '^', ':')]
    for x, color, marker, ls in series:
        label = rf'$x={x}$'
        curve = mean[x, r]
        old = [float(p['unmasked_correlator']) for p in previous
               if p['panel'] == 'b' and p['series'] == label]
        if trivial_bulk:
            np.testing.assert_allclose(curve, samples[:, x, 1:].sum(axis=0)/100,
                                       rtol=2e-14, atol=0)
        else:
            np.testing.assert_allclose(curve, old, rtol=2e-14, atol=0)
        keep = np.isfinite(curve) & (curve > (0 if trivial_bulk else CUTOFF))
        ax.plot(r, np.where(keep, curve, np.nan), color=color,
                marker=marker, ls=ls, lw=.85, ms=3.1, mfc='white',
                mew=.7, label=label)
        rows.extend(dict(x=x, ry=int(ri), correlator=float(ci), displayed=bool(k))
                    for ri, ci, k in zip(r, curve, keep))
    ax.set(yscale='log', xlim=(.4, 30.6), ylim=(1e-10, .2),
           xlabel=r'$r_y$', ylabel=r'$\overline{C_G(x,r_y)}$')
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    if log_x:
        ax.set(xscale='log', xlim=(.94, 32))
        ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 30]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_locator(FixedLocator([1e-2, 1e-4, 1e-6, 1e-8, 1e-10]))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    if trivial_bulk:
        ax.set_ylim(1e-32, 1e-29)
        ax.yaxis.set_major_locator(FixedLocator([1e-32, 1e-31, 1e-30, 1e-29]))
    ax.minorticks_off()
    ax.tick_params(direction='in', top=True, right=True)
    annotation = (r'Unmasked: below $10^{-20}$ cutoff' if trivial_bulk
                  else r'DWs at $x=5,15$')
    ax.text(.97, .96, annotation, transform=ax.transAxes,
            ha='right', va='top', fontsize=8)
    ax.legend(loc='lower left', frameon=False, handlelength=1.6,
              handletextpad=.4, labelspacing=.25)
    fig.text(.025, .95, '(a)', fontsize=9, va='top')
    fig.subplots_adjust(left=.18, right=.975, bottom=.19, top=.88)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(plt.matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1, artist.get_text()
    assert ax.get_xscale() == ('log' if log_x else 'linear') and ax.get_yscale() == 'log'
    for line in ax.lines:
        assert np.array_equal(line.get_xdata(), r)
    for ext in ['pdf', 'png']:
        fig.savefig(out / f'{out.name}.{ext}', dpi=300)
    plt.close(fig)
    with (out / 'plotted_data.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = dict(Nx=20, Ny=60, alpha1=1, samples=100, x=[s[0] for s in series],
                      excluded_ry=[0], cutoff=None if trivial_bulk else CUTOFF,
                      original_display_cutoff=CUTOFF,
                      diagnostic_only=trivial_bulk,
                      axes=('Logarithmic' if log_x else 'Linear') + ' raw ry; logarithmic axis displaying C, not log(C)',
                      validation=('Direct trajectory mean from cached compact export; original panel b has no x=2,19.' if trivial_bulk else
                                  'Cached compact export agrees with original panel b table; no new raw-campaign validation.'),
                      caution='Display cutoff only, not a numerical-error bound.',
                      inputs={str(SOURCE / name): hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                              for name in ['compact_curves.npz', 'plotted_data.csv', 'summary.json']},
                      manuscript_changed=False)
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(out / f'{out.name}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-x', action='store_true')
    parser.add_argument('--trivial-bulk', action='store_true',
                        help='Separate unmasked x=2,19 numerical-scale diagnostic; does not change earlier cutoff figures.')
    args = parser.parse_args()
    main(args.log_x, args.trivial_bulk)
