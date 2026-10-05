"""Verified single-trajectory entropy comparison, hard walls, alpha1=1 and 3."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'analysis_outputs/hard_wall_postselected_alpha1_alpha3_v1'
REVISIONS = {
    1: 'postselected_maxmix_hard_soft_nx20_ny40_s1_4ny_gpu_v1',
    3: 'postselected_maxmix_hard_soft_nx20_ny40_alpha3_s1_4ny_gpu_v1',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_curve(alpha):
    root = ROOT / 'gpu_data' / REVISIONS[alpha]
    manifest_path = root / 'DOWNLOAD_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    config = manifest['resolved_configuration']
    assert config['alpha_1'] == alpha
    assert config['Nx'] == 20 and config['Ny'] == 40 and config['cycles'] == 160
    assert config['postselect'] is True and config['postselect_probability'] == 1
    assert config['perfect_correction'] is False and config['samples'] == 1
    assert config['constructions']['hard']['meas_slab_only'] is True
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert config_hash == manifest['configuration_hash']
    for row in manifest['files']:
        if row['construction'] != 'hard':
            continue
        path = root / row['relative_path']
        assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256']
    path = root / 'results/hard/postselected_trajectory.npz'
    receipt = json.loads(path.with_name('completion.json').read_text())
    assert digest(path) == receipt['result_sha256']
    assert receipt['source_hashes'] == manifest['executed_source_hashes']
    with np.load(path, allow_pickle=False) as d:
        assert d['construction'].item() == 'hard'
        assert d['configuration_hash'].item() == config_hash == receipt['configuration_hash']
        cycles, entropy = d['cycles'].copy(), d['total_entropy_nats'].copy()
    assert np.array_equal(cycles, np.arange(161))
    assert entropy.shape == (161,) and np.all(np.isfinite(entropy))
    assert np.all(entropy > 0), 'Log axis requires positive saved values; no artificial floor applied'
    assert np.isclose(entropy[0], 880 * np.log(2), rtol=1e-12)
    return cycles / 40, entropy / 40, {
        'alpha_1': alpha, 'result': str(path.relative_to(ROOT)),
        'result_sha256': digest(path), 'manifest_sha256': digest(manifest_path),
        'endpoint_entropy_nats': float(entropy[-1]),
        'endpoint_entropy_over_Ny': float(entropy[-1] / 40),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['CMU Sans Serif' if 'CMU Sans Serif' in available else 'DejaVu Sans'],
        'mathtext.fontset': 'cm', 'font.size': 8, 'axes.labelsize': 8.5,
        'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True, 'axes.linewidth': .8,
        'pdf.fonttype': 42,
    })
    fig, ax = plt.subplots(figsize=(3.375, 2.85), layout='constrained')
    rows, provenance = [], []
    for alpha, color, marker, style in [(1, '#d62728', '^', ':'), (3, '#1f77b4', 'o', '-')]:
        t, s, record = load_curve(alpha)
        provenance.append(record)
        ax.plot(t, s, color=color, marker=marker, ls=style, lw=1.3,
                markersize=3.3, markerfacecolor='white', markeredgewidth=.8,
                markevery=16, label=rf'$\alpha_1={alpha}$', zorder=3)
        rows.extend({'alpha_1': alpha, 'cycle': int(round(u * 40)),
                     'cycle_over_Ny': float(u), 'entropy_over_Ny': float(v)}
                    for u, v in zip(t, s))
    reference = 2 * np.log(2) / 40
    ax.axhline(reference, color='.45', ls='--', lw=.8,
               label=r'$2\ln 2/N_y$', zorder=1)
    ax.set(xlim=(0, 4), yscale='log', ylim=(1e-14, 40),
           xlabel=r'cycle $t/N_y$', ylabel=r'$S(t)/N_y$')
    ax.set_yticks([1e1, 1e-2, 1e-5, 1e-8, 1e-11, 1e-14])
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_title(r'Hard wall, postselected; $N_x=20$, $N_y=40$', fontsize=8)
    ax.legend(frameon=False, loc='center right', bbox_to_anchor=(1, .59))
    ax.text(.98, .10, 'roundoff-limited tail', transform=ax.transAxes,
            ha='right', fontsize=7.5, color='.35')
    stem = OUTPUT / 'hard_wall_postselected_entropy_alpha1_alpha3'
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.png'), dpi=300)
    plt.close(fig)
    with (OUTPUT / 'entropy_curves.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'samples_per_alpha': 1, 'uncertainty': 'none: one postselected trajectory per alpha',
               'x_scale': 'linear', 'y_scale': 'logarithmic', 'figure_inches': [3.375, 2.85],
               'entropy_reference_over_Ny': reference, 'sources': provenance,
               'note': 'No fitting, smoothing, clipping or sample averaging; tiny alpha3 tail is roundoff-limited.'}
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (OUTPUT / 'README.md').write_text(
        '# Hard-wall postselected purification comparison\n\n'
        'Reproduce with `python plot_hard_wall_postselected_alpha_comparison.py` in bundle 19.\n\n'
        'Caption: Entropy S(t)/Ny versus cycle t/Ny for alpha1=1 and 3, at Nx=20, Ny=40, '
        'alpha2=30, nshell=1, hard support-truncated walls, raster-y slab measurements, '
        'and full postselection (perfect_correction=False). One trajectory per alpha '
        'starts from the maximally mixed active slab after exterior preparation and runs '
        'through T=4Ny=160. Entropy is in nats and is computed on each trajectory, not '
        'an averaged covariance. No averaging, error bars, bootstrap or fitting is applied. '
        'All 161 saved cycles are plotted with linear time and logarithmic entropy axes. '
        'The dashed reference is 2 ln(2)/Ny. The alpha1=3 residual at roughly 10^-13 in S/Ny '
        'is roundoff-limited; its small fluctuations should not be interpreted physically. '
        'Raw data and earlier figures are unchanged. The vector PDF is 3.375 inches wide '
        'for a single RevTeX column.\n')
    print(json.dumps(summary, indent=2))
    print(stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
