#!/usr/bin/env python3
"""Pool individual endpoint occupations; never histogram the mean spectrum."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'analysis_outputs/mean_endpoint_occupations_alpha1_alpha3_v1'
OUT = ROOT / 'analysis_outputs/pooled_endpoint_occupations_alpha1_alpha3_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = SOURCE / 'sample_sorted_occupations.npz'
    bins = np.linspace(0, 1, 51)
    rows, cases = [], {}
    plt.rcParams.update({'font.family': 'CMU Sans Serif', 'font.size': 8,
                         'mathtext.fontset': 'cm', 'axes.linewidth': .8})
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(3.375, 4.25))
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['sample_indices'], np.arange(100))
        for ax, alpha, color, letter in zip(axes, (1, 3), ('#e41a1c', '#1879bb'), 'ab'):
            values = z[f'alpha1_{alpha}']
            assert values.shape == (100, 880) and np.isfinite(values).all()
            assert values.min() > -1e-9 and values.max() < 1 + 1e-9
            # Include tiny floating-point excursions in the endpoint bins.
            pooled = np.clip(values, 0, 1).ravel()
            counts, edges = np.histogram(pooled, bins=bins)
            assert counts.sum() == 88000
            probability = counts / pooled.size
            np.testing.assert_allclose(probability.sum(), 1, atol=1e-15)
            # Nonempty bins only: zero bins stay absent; no pseudocounts.
            mask = counts > 0
            ax.bar(edges[:-1][mask], probability[mask], width=np.diff(edges)[mask],
                   align='edge', color=color, edgecolor=color, alpha=.75, linewidth=.5)
            ax.set_yscale('log')
            ax.set_ylim(.5 / pooled.size, 1)
            ax.set_xlim(0, 1)
            ax.set_ylabel('Probability per bin')
            ax.tick_params(which='both', direction='in', top=True, right=True)
            ax.text(-.19, 1.03, f'({letter})', transform=ax.transAxes, fontsize=9)
            ax.text(.5, .93, rf'$\alpha_1={alpha}$', ha='center', va='top',
                    transform=ax.transAxes, fontsize=10)
            ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
            for lo, hi, n, p in zip(edges[:-1], edges[1:], counts, probability):
                rows.append(dict(alpha_1=alpha, left_edge=float(lo), right_edge=float(hi),
                                 count=int(n), probability=float(p)))
            cases[alpha] = dict(samples=100, modes_per_sample=880, pooled_modes=int(pooled.size),
                                probability_sum=float(probability.sum()),
                                interior_bin_count=int(counts[1:-1].sum()),
                                interior_bin_probability=float(probability[1:-1].sum()))
    axes[0].set_title(r'Hard wall: $N_y=40$, $T=4N_y$, 100 samples', fontsize=8)
    axes[1].set_xlabel(r'Endpoint occupation $\nu$')
    axes[1].set_xticks(np.linspace(0, 1, 6))
    fig.tight_layout(pad=.6, h_pad=.8)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'pooled_endpoint_histograms.{ext}', dpi=300)
    plt.close(fig)
    with (OUT / 'histogram_bins.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    metadata = dict(source=str(path), source_sha256=sha(path),
        source_provenance=str(SOURCE/'summary.json'), source_provenance_sha256=sha(SOURCE/'summary.json'),
        script_sha256=sha(Path(__file__)), bin_width=.02, cases=cases,
        normalization='Count per bin / 88000; sum of bin probabilities is one for each alpha.',
        estimator='Pool all 880 active-slab occupations of each of 100 trajectories, without averaging spectra first.',
        numerical_handling='Only roundoff outside [0,1] within 1e-9 is clipped; retain all caps. No pseudocounts.',
        display='Log probability axis to reveal rare interior occupations; zero bins omitted, not altered.')
    (OUT/'summary.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (OUT/'README.md').write_text('# Pooled endpoint occupation histograms\n\n'
        'Hard-wall Nx=20, Ny=40, T=160, alpha1=1 (top, bundle 07) and 3 (bottom, bundle 20), '
        'alpha2=30, nshell=1, maximally mixed initialization, Born sampling and perfect correction. '
        'Each panel pools 100 independent trajectories with 880 active-slab modes each; '
        'deterministic exterior product sectors are excluded. Fifty equal-width bins span [0,1]. '
        'Heights are counts/88000, so they sum to one; these are bin probabilities, not probability densities. '
        'The vertical axis is logarithmic to expose rare interior occupations. Empty bins are absent, '
        'without pseudocounts. No errors or fits are shown; modes are not treated as independent trajectories. '
        'Only numerical excursions within 1e-9 outside [0,1] are clipped to the endpoints. '
        'The underlying sample spectra, provenance hashes, and bin counts are retained.\n')
    print(json.dumps(cases, indent=2))
    print(OUT)


if __name__ == '__main__':
    main()
