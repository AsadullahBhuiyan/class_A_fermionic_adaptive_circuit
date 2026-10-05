#!/usr/bin/env python3
"""Average saved x-resolved correlators over all x, then over trajectories."""
from pathlib import Path
import argparse
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
import numpy as np
import plot_alpha_comparison_ny60 as matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overlay', action='store_true', help='Draw both masses on one single-column axis.')
    args = parser.parse_args()
    data = matched.data
    identity = data.preparation_identity()
    a1 = data.load_endpoint(60, identity)
    a3, hashes = matched.load_alpha3()
    for name in ('src/classA_U1FGTN_gpu.py', 'src/occupied_frame_gpu.py'):
        assert identity['source_hashes'][name] == hashes[name]
    out = matched.OUT
    out.mkdir(parents=True, exist_ok=True)
    stem = 'hard_wall_xaveraged_alpha1_1_vs_3'
    if args.overlay:
        stem += '_overlay'
    means, diagnostics = [], {}
    for alpha, endpoint in zip((1, 3), (a1, a3)):
        # Do not threshold columns or trajectories before either arithmetic mean.
        per_sample = endpoint.xresolved.mean(axis=1)
        np.testing.assert_allclose(per_sample, endpoint.xavg, rtol=2e-14, atol=1e-15)
        means.append(per_sample.mean(axis=0))
        diagnostics[str(alpha)] = dict(samples=len(endpoint.ids), x_columns=list(range(20)),
            max_saved_xavg_difference=float(np.max(np.abs(per_sample-endpoint.xavg))))
    data.prior.configure_matplotlib()
    plt.rcParams.update({'font.size':8, 'axes.labelsize':9, 'axes.titlesize':10,
                         'xtick.labelsize':8, 'ytick.labelsize':8})
    if args.overlay:
        fig, ax = plt.subplots(figsize=(3.375, 3.15))
        axes = [ax, ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.2), sharex=True, sharey=True)
    r = np.arange(1, 31)
    logd = np.log(data.prior.chord(60, r))
    ymax = max(float(np.log(m[1:]).max()) for m in means) + .4
    rows = []
    for panel, (alpha, ax, mean, color, marker) in enumerate(zip(
            (1, 3), axes, means, ('#0072B2', '#D55E00'), ('o', '^'))):
        shown = mean[1:] > matched.CUTOFF
        ax.plot(logd, np.where(shown, np.log(mean[1:]), np.nan), color=color,
                marker=marker, mfc='white', mew=.8, ms=3.8, lw=.9,
                ls='-' if alpha == 1 else ':', label=rf'$\alpha_1={alpha}$')
        if not args.overlay:
            ax.set_title(rf'$\alpha_1={alpha}$')
        ax.set_xlabel(r'$\log d_{N_y}(r_y)$')
        ax.set_xlim(-.06, 3.03); ax.set_xticks([0, 1, 2, 3])
        ax.set_ylim(np.log(matched.CUTOFF)-.3, ymax)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(direction='in', top=True, right=True)
        if not args.overlay:
            ax.text(-.12, 1.04, f'({chr(97+panel)})', transform=ax.transAxes, fontsize=10)
        diagnostics[str(alpha)]['displayed_ry'] = r[shown].tolist()
        for ry, ld, value, show in zip(r, logd, mean[1:], shown):
            rows.append(dict(alpha_1=alpha, ry=int(ry), log_chord=float(ld),
                             mean_xaveraged_correlator=float(value), displayed=bool(show)))
    axes[0].set_ylabel(r'$\log\overline{C_G^{\mathrm{av}}(r_y)}$')
    if args.overlay:
        axes[0].legend(loc='lower left', frameon=False, fontsize=8)
        axes[0].set_title(r'$20\times60$, $n_{\rm shell}=1$, $S=100$'+'\n'+
                          r'Hard walls; $t=120$', fontsize=9)
        fig.text(.57, .02, r'All $x$; display: $\overline{C_G^{\mathrm{av}}}>10^{-8}$',
                 ha='center', fontsize=8)
        fig.subplots_adjust(left=.20, right=.97, bottom=.22, top=.82)
    else:
        fig.suptitle(r'$20\times60$, $n_{\rm shell}=1$, $S=100$, $t=120$; hard walls',
                     fontsize=9, y=.985)
        fig.text(.53, .025, r'All 20 columns averaged; display: $\overline{C_G^{\mathrm{av}}}>10^{-8}$.',
                 ha='center', fontsize=8)
        fig.subplots_adjust(left=.09, right=.985, bottom=.22, top=.82, wspace=.12)
    for ext in ('png', 'pdf'):
        fig.savefig(out/f'{stem}.{ext}', dpi=300)
    plt.close(fig)
    with (out/f'{stem}_curves.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    caption = (
        'Hard/support-terminated Nx=20, Ny=60, alpha_2=30, nshell=1; alpha_1=1 '
        '(left) and 3 (right). S=100 independent pure half-filled initialized trajectories '
        'per ensemble, Born-conditioned exterior preparation, raster-y perfect correction, '
        'complex128, endpoint cycle 120=2Ny. Within each trajectory C_G^av(r)=(1/20) '
        'sum_{x=0}^{19} C_G(x,r); markers show log of its arithmetic trajectory mean. '
        'The saved squared correlators are averaged, not the state matrices or logarithms. '
        'The per-trajectory x averages reproduce the saved x-average arrays. '
        'Natural logarithms versus chord d=Ny/pi*sin(pi*r/Ny), independent r=1..30. '
        'No uncertainty bands or fits; lines connect points. Threshold 1e-8 is a display '
        'cutoff applied after both averages, not a certified numerical-error floor. '
        'Unmasked means are saved in the CSV. Raw data and previous figures are unchanged.'
    )
    if args.overlay:
        caption = caption.replace('(left) and 3 (right)', '(blue circles) and 3 (orange triangles)')
    (out/f'{stem}_caption.md').write_text(caption+'\n')
    (out/f'{stem}_summary.json').write_text(json.dumps(dict(
        diagnostics=diagnostics, caption=caption, inputs=a1.provenance+a3.provenance+identity['evidence'],
        source=data.file_record(Path(__file__), 'plot_script'),
        loader=data.file_record(Path(matched.__file__), 'alpha3_loader')), indent=2)+'\n')
    print(json.dumps(diagnostics, indent=2))
    print(out/f'{stem}.png')


if __name__ == '__main__':
    main()
