#!/usr/bin/env python3
"""Matched Ny60 alpha1=1/3 x-resolved means; compact data only, no dynamics."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
import numpy as np

import large_ny_correlator_data as data

PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / 'outputs/alpha1_comparison_Ny060_nshell1'
ALPHA3 = data.ROOT / '00_WORKSPACE/CURRENT/final_production_new_designs/15_hard_wall_alpha3_xresolved_correlator'
REV3 = 'hard_wall_xresolved_nx20_ny60_a1-3_nsh1_s100_2ny_raster_endpoint_v1'
SITES = (5, 6, 14, 15)
CUTOFF = 1e-8
STEM = 'hard_wall_xresolved_alpha1_1_vs_3'


def load_alpha3():
    directory = ALPHA3 / 'gpu_data' / REV3 / 'results/Ny060'
    paths = sorted(directory.glob('*.npz'))
    assert len(paths) == len(list(directory.glob('*.complete.json'))) == 20
    pieces, provenance = [], []
    configs = set()
    for path in paths:
        receipt_path = path.with_suffix('.complete.json')
        receipt = json.loads(receipt_path.read_text())
        data.verify_pair(path, receipt)
        assert receipt['sampling_revision'] == REV3
        for name, digest in receipt['source_hashes'].items():
            assert data.sha256(ALPHA3 / name) == digest
        configs.add(receipt['configuration_sha256'])
        with np.load(path, allow_pickle=False) as z:
            expected = dict(Nx=20, Ny=60, alpha_1=3., alpha_2=30., nshell=1,
                            DW=True, dw_truncation=True, meas_slab_only=True,
                            perfect_correction=True, sequence='raster_y', dtype='complex128',
                            construction='hard', init_mode='default', filling_frac=.5,
                            trial_orbitals='X', state_representation='physical_frame',
                            exterior_preparation='born_conditioned_onsite_before_cycle_0',
                            canonical_entry_point='classA_U1FGTN_gpu.run_markov_circuit')
            for name, value in expected.items():
                assert z[name].item() == value, name
            assert z['configuration_sha256'].item() == receipt['configuration_sha256']
            assert z['sampling_revision'].item() == REV3
            np.testing.assert_array_equal(z['cycles'], [120])
            np.testing.assert_array_equal(z['wall_locations'], [5, 15])
            np.testing.assert_array_equal(z['x_values'], np.arange(20))
            np.testing.assert_array_equal(z['ry_values'], np.arange(31))
            np.testing.assert_array_equal(z['global_sample_indices'], receipt['global_sample_indices'])
            x = z['x_resolved_square_correlator'][:, 0]
            avg = z['xavg_square_correlator_vs_ry'][:, 0]
            assert x.shape == (5, 20, 31)
            assert np.isfinite(x).all() and (x >= 0).all()
            np.testing.assert_allclose(x.mean(axis=1), avg, rtol=2e-14, atol=1e-15)
            pieces.append((z['global_sample_indices'].copy(), avg.copy(), x.copy()))
        provenance.extend([data.file_record(path, 'verified_endpoint'),
                           data.file_record(receipt_path, 'completion')])
    assert len(configs) == 1
    return data.assemble(60, 'alpha3_endpoint', pieces, provenance), receipt['source_hashes']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    identity = data.preparation_identity()
    a1 = data.load_endpoint(60, identity)
    a3, hashes3 = load_alpha3()
    for name in ('src/classA_U1FGTN_gpu.py', 'src/occupied_frame_gpu.py'):
        assert identity['source_hashes'][name] == hashes3[name], name
    data.prior.configure_matplotlib()
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 10,
                         'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.45), sharex=True, sharey=True)
    ry = np.arange(1, 31)
    logd = np.log(data.prior.chord(60, ry))
    ymax = max(float(np.log(e.xresolved[:, SITES, 1:].mean(axis=0)).max())
               for e in (a1, a3)) + .4
    rows = []
    for panel, (ax, alpha, endpoint) in enumerate(zip(axes, (1, 3), (a1, a3))):
        for x, color, marker, ls in zip(SITES, ('#0072B2', '#E69F00', '#009E73', '#D55E00'),
                                         ('o', 's', '^', 'v'), ('-', '--', '-.', ':')):
            mean = endpoint.xresolved[:, x, 1:].mean(axis=0)
            shown = np.isfinite(mean) & (mean > CUTOFF)
            logmean = np.log(np.where(mean > 0, mean, np.nan))
            ax.plot(logd, np.where(shown, logmean, np.nan), color=color, ls=ls,
                    marker=marker, mfc='white', mew=.75, ms=3.3, lw=.8, label=rf'$x={x}$')
            for r, d, c, lc, show in zip(ry, logd, mean, logmean, shown):
                rows.append(dict(alpha_1=alpha, Ny=60, x=x, ry=int(r), log_chord=float(d),
                                 mean_correlator=float(c), log_mean_correlator=float(lc),
                                 displayed=bool(show)))
        ax.set_title(rf'$\alpha_1={alpha}$')
        ax.set_xlabel(r'$\log d_{N_y}(r_y)$')
        ax.set_xlim(-.06, 3.03)
        ax.set_ylim(np.log(CUTOFF)-.3, ymax)
        ax.set_xticks([0, 1, 2, 3])
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(direction='in', top=True, right=True)
        ax.text(-.12, 1.04, f'({chr(97+panel)})', transform=ax.transAxes, fontsize=10)
    axes[0].set_ylabel(r'$\log\overline{C_G(x,r_y)}$')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(.53, .035), handletextpad=.4, columnspacing=1.7)
    fig.suptitle(r'$20\times60$, $n_{\rm shell}=1$, $S=100$, $t=120$; hard walls',
                 fontsize=9, y=.985)
    fig.text(.53, .007, r'Display: $\overline{C_G}>10^{-8}$; connecting lines are not fits.',
             ha='center', fontsize=8)
    fig.subplots_adjust(left=.09, right=.985, bottom=.25, top=.83, wspace=.12)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{STEM}.{ext}', dpi=300)
    plt.close(fig)
    with (OUT / f'{STEM}_curves.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    caption = (
        'Nx=20, Ny=60 hard/support-terminated wall comparison at alpha_1=1 (left) and 3 (right), '
        'alpha_2=30, nshell=1, walls [5,15]. Each independent ensemble has S=100 pure '
        'half-filled initialized trajectories, Born-conditioned exterior preparation, raster-y '
        'perfect correction, complex128, and endpoint t=120=2Ny. Markers show log of the '
        'arithmetic trajectory mean of the x-resolved squared correlator, not a correlator '
        'of the averaged state or a mean of logarithms. No uncertainty bands, SEM, or fits. '
        'Lines only connect points. Both axes use natural logarithms labeled log; '
        'd_Ny(r)=Ny/pi*sin(pi*r/Ny). Only independent separations r=1..30 are eligible. '
        'The historical display cutoff mean C_G>1e-8 is applied equally to both panels; '
        'it is not a certified numerical-error bound. The CSV retains all unmasked means. '
        'The independent ensembles are not paired by sample index. No raw data or prior figures changed.'
    )
    (OUT / 'caption.md').write_text(caption+'\n')
    summary = dict(Nx=20, Ny=60, samples_per_alpha=100, x_columns=SITES,
                   cutoff=CUTOFF, estimator='log(arithmetic trajectory mean C_G)',
                   shared_engine_hash=hashes3['src/classA_U1FGTN_gpu.py'],
                   inputs=a1.provenance+a3.provenance+identity['evidence'],
                   source=data.file_record(Path(__file__), 'plot_script'), caption=caption)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'[done] {OUT / STEM}.png')


if __name__ == '__main__':
    main()
