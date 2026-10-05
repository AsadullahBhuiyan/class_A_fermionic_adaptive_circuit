"""Ascending saved endpoint occupation spectra for the two hard-wall controls."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from plot_hard_wall_postselected_alpha_comparison import ROOT, REVISIONS, load_curve

OUTPUT = ROOT / 'analysis_outputs/hard_wall_postselected_endpoint_occupations_v1'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['CMU Sans Serif' if 'CMU Sans Serif' in available else 'DejaVu Sans'],
        'mathtext.fontset': 'cm', 'font.size': 8, 'axes.labelsize': 8.5,
        'legend.fontsize': 8, 'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True, 'axes.linewidth': .8, 'pdf.fonttype': 42,
    })
    fig, ax = plt.subplots(figsize=(3.375, 2.7), layout='constrained')
    rows, records = [], []
    for alpha, color, marker, style in [(3, '#1f77b4', 'o', '-'), (1, '#d62728', '^', ':')]:
        _, _, provenance = load_curve(alpha)  # Reverify pinned manifest/config/result pair.
        path = ROOT / 'gpu_data' / REVISIONS[alpha] / 'results/hard/postselected_trajectory.npz'
        with np.load(path, allow_pickle=False) as d:
            raw = d['endpoint_occupations'].copy()
            assert np.allclose(raw, (d['endpoint_centered_spectrum'] + 1) / 2,
                               atol=1e-15, rtol=0)
        order = np.argsort(raw, kind='stable')
        occupations = raw[order]
        assert occupations.shape == (880,) and np.all(np.isfinite(occupations))
        assert occupations.min() >= 0 and occupations.max() <= 1
        ranks = np.arange(1, 881)
        # Draw every saved mode; sparse markers keep coincident caps legible.
        marks = sorted(set(range(0, 880, 80)) | {438, 439, 440, 441, 879})
        ax.plot(ranks, occupations, color=color, marker=marker, ls=style, lw=1.2,
                markevery=marks, ms=3.5, mfc='white', mew=.8, label=rf'$\alpha_1={alpha}$')
        rows.extend({'alpha_1': alpha, 'ascending_rank': int(i),
                     'original_mode_index': int(j), 'occupation': float(n)}
                    for i, j, n in zip(ranks, order, occupations))
        records.append({**provenance, 'mode_count': 880,
                        'central_occupations_ranks_440_441': occupations[439:441].tolist()})
    ax.axhline(.5, color='.65', ls='--', lw=.7, zorder=0)
    ax.annotate(r'two modes near $1/2$', xy=(440, .5), xytext=(55, .67),
                fontsize=8, arrowprops={'arrowstyle': '->', 'color': '.35', 'lw': .7})
    ax.set(xlim=(1, 880), ylim=(-.035, 1.035),
           xlabel=r'mode rank $j$ (ascending)', ylabel=r'occupation $\nu_j$')
    ax.set_xticks([1, 220, 440, 660, 880])
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_title(r'Hard wall, postselected; $N_y=40$, $T=160$', fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='lower right', frameon=False)
    stem = OUTPUT / 'hard_wall_endpoint_occupations_alpha1_alpha3'
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.png'), dpi=300)
    plt.close(fig)
    with (OUTPUT / 'sorted_occupations.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r['alpha_1'], r['ascending_rank'])))
    (OUTPUT / 'summary.json').write_text(json.dumps({
        'sources': records, 'sorting': 'each trajectory independently, ascending occupation',
        'uncertainty': 'none; one fully postselected trajectory per alpha',
        'clipping_or_averaging': 'none; saved exact caps retained',
    }, indent=2) + '\n')
    (OUTPUT / 'README.md').write_text(
        '# Ascending hard-wall endpoint occupations\n\n'
        'Reproduce with `python plot_hard_wall_endpoint_occupations.py` in bundle 19.\n\n'
        'Caption: All 880 active-mode endpoint occupations, sorted independently in '
        'ascending order for alpha1=1 and 3. Nx=20, Ny=40, alpha2=30, nshell=1, '
        'hard support-truncated walls, raster-y active-slab measurements, maxmix '
        'active-slab initialization after exterior preparation, full postselection, '
        'perfect_correction=False, T=4Ny=160. One trajectory per alpha, no error bars, '
        'fitting or sample averaging. The saved occupations are not clipped or '
        'regularized by this plot; exact 0/1 caps are retained. All modes are joined '
        'in rank order with sparse markers for readability. The two alpha1=1 modes '
        'near half occupation have ranks 440 and 441. The spectra otherwise largely '
        'overlap at 0 and 1. The dashed gray line marks half occupation.\n')
    print(json.dumps(records, indent=2))
    print(stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
