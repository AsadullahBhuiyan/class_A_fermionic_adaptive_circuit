"""Standalone display-only remake of summary panel (a); no manuscript edits."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter, LogFormatterMathtext
import numpy as np

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[3]
SOURCE = PROJECT / 'outputs/correlator_summary_3x1_v2_r8_half'
OUT = PROJECT / 'outputs/xavg_alpha_comparison_Ny60_cutoff1e-20'
CUTOFF = 1e-20


def main(raw_separation=False):
    out = OUT.with_name(OUT.name + '_ry_loglog') if raw_separation else OUT
    # Use the shared Computer Modern/LaTeX style for this standalone preview.
    # Do not register it as a manuscript inclusion or write manuscript artifacts.
    style_path = ROOT / '00_WORKSPACE/CURRENT/manuscript_Overleaf/figures/new_figure/sources/manuscript_typography.py'
    spec = importlib.util.spec_from_file_location('typography', style_path)
    typography = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(typography)
    typography.configure_style()
    out.mkdir(parents=True, exist_ok=True)
    with (SOURCE / 'plotted_data.csv').open() as f:
        previous = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(3.375, 2.30))
    rows, counts = [], {}
    r = np.arange(1, 31)  # Contact term r_y=0 is excluded before taking logs.
    logd = np.log(60 / np.pi * np.sin(np.pi * r / 60))
    with np.load(SOURCE / 'compact_curves.npz', allow_pickle=False) as z:
        for alpha, color, marker, ls in [(1, '#0072B2', 'o', '-'), (3, '#D55E00', '^', ':')]:
            samples = z[f'alpha{alpha}_Ny60_xresolved']
            assert samples.shape == (100, 20, 31)
            mean = samples.mean(axis=1).mean(axis=0)[r]
            label = rf'$\alpha_1={alpha}$'
            old = [float(p['unmasked_correlator']) for p in previous
                   if p['panel'] == 'a' and p['series'] == label]
            np.testing.assert_allclose(mean, old, rtol=2e-14, atol=0)
            keep = np.isfinite(mean) & (mean > CUTOFF)
            # Mask rather than clamp; no artificial plateau or connection over gaps.
            y = np.full(mean.shape, np.nan)
            y[keep] = mean[keep] if raw_separation else np.log(mean[keep])
            ax.plot(r if raw_separation else logd, y, color=color, marker=marker, ls=ls, lw=.85,
                    ms=3.1, mfc='white', mew=.7, label=label)
            counts[str(alpha)] = {'shown': int(keep.sum()),
                                  'last_ry_shown': int(r[keep][-1])}
            for ri, xi, ci, shown in zip(r, logd, mean, keep):
                rows.append(dict(alpha1=alpha, ry=int(ri), log_chord=float(xi),
                                 correlator=float(ci), displayed=bool(shown)))
    if raw_separation:
        ax.set(xscale='log', yscale='log', xlim=(.94, 32),
               ylim=(CUTOFF / 2, .05), xlabel=r'$r_y$',
               ylabel=r'$\overline{C_G^{\mathrm{av}}(r_y)}$')
        ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 30]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_locator(FixedLocator([1e-20, 1e-16, 1e-12, 1e-8, 1e-4]))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
    else:
        ax.set(xlim=(-.06, 3.03), ylim=(np.log(CUTOFF) - .7, -2),
               xlabel=r'$\log d_{N_y}(r_y)$',
               ylabel=r'$\log\overline{C_G^{\mathrm{av}}(r_y)}$')
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks([-40, -30, -20, -10])
    ax.tick_params(direction='in', top=True, right=True)
    ax.minorticks_off()
    ax.legend(loc='lower left', frameon=False, handlelength=1.8, handletextpad=.4)
    fig.text(.025, .95, '(a)', fontsize=9, va='top')
    fig.subplots_adjust(left=.18, right=.975, bottom=.19, top=.88)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1, artist.get_text()
    stem = out.name
    for ext in ['pdf', 'png']:
        fig.savefig(out / f'{stem}.{ext}', dpi=300)
    plt.close(fig)
    with (out / 'plotted_data.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = dict(Nx=20, Ny=60, samples_per_alpha=100, alpha1=[1, 3],
                      excluded_ry=[0], cutoff=CUTOFF, counts=counts,
                      cutoff_rule='Ensemble/x-averaged C > 1e-20; no clamping.',
                      axes='Raw ry and C on logarithmic axes' if raw_separation else 'Natural log chord and natural log C',
                      caution='Display threshold, not a certified numerical error bound.',
                      validation='Cached compact export agrees with prior plotted-data table; raw campaign not revalidated.',
                      inputs={str(SOURCE / name): hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                              for name in ['compact_curves.npz', 'plotted_data.csv', 'summary.json']},
                      typography_source=str(style_path), manuscript_changed=False)
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(counts))
    print(out / f'{stem}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-separation', action='store_true')
    main(parser.parse_args().raw_separation)
