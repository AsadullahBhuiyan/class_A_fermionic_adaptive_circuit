"""Sort occupations within each trajectory, then average at fixed rank and cycle.

Uses the full-system Ny=30 purification runs behind manuscript entropy panel (a).
No simulations, clipping, or modifications of source data or manuscript figures.
"""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
CURRENT = HERE.parents[2]
sys.path.insert(0, str(CURRENT/'manuscript_Overleaf/figures/new_figure/sources'))
from manuscript_typography import configure_style, FONT_PATTERN

CYCLES = (0, 10, 20, 40, 60)
ROOTS = {
    1: CURRENT/'final_production_new_designs/21_hard_wall_full_measurement_purification/gpu_data/hard_wall_full_measurement_nx20_ny30_alpha1-3_s100_2ny_v1',
    3: CURRENT/'final_production_new_designs/22_hard_wall_full_measurement_clipped/gpu_data/hard_wall_full_measurement_nx20_ny30_alpha3_s100_2ny_clipped_v2',
}
STEM = 'sorted_occupation_spectra_cycles'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load(alpha):
    paths = sorted((ROOTS[alpha]/f'results/hard/alpha1_{alpha}/Ny030').glob('*.npz'))
    assert len(paths) == 20
    ids, arrays, records = [], [], []
    for path in tqdm(paths, desc=f'Check and sort alpha_1={alpha}', unit='shard'):
        receipt = json.loads(path.with_suffix('.complete.json').read_text())
        sha = digest(path)
        assert sha == receipt['result_sha256']
        assert path.stat().st_size == receipt['result_bytes']
        with np.load(path, allow_pickle=False) as saved:
            config = json.loads(str(saved['configuration_json']))
            assert int(saved['Nx']) == 20 and int(saved['Ny']) == 30
            assert float(saved['alpha_1']) == alpha
            assert config['init_mode'] == 'maxmix' and config['perfect_correction']
            assert not bool(saved['meas_slab_only'])
            np.testing.assert_array_equal(saved['cycles'], np.arange(61))
            np.testing.assert_array_equal(saved['sample_indices'], receipt['sample_indices'])
            nu = saved['occupation_spectrum'][:, CYCLES, :]
            assert nu.shape == (5, 5, 1200) and np.isfinite(nu).all()
            assert nu.min() >= -1e-9 and nu.max() <= 1+1e-9
            # Rank labels are reassigned independently at each cycle and sample.
            ranked = np.sort(nu, axis=-1)
            assert np.all(np.diff(ranked, axis=-1) >= 0)
            np.testing.assert_allclose(ranked[:, 0], .5, atol=1e-12, rtol=0)
            np.testing.assert_allclose(ranked.sum(-1), nu.sum(-1), atol=1e-10, rtol=0)
            arrays.append(ranked)
            ids.extend(saved['sample_indices'].tolist())
        records.append({'path': str(path), 'sha256': sha})
    assert sorted(ids) == list(range(100))
    ranked = np.concatenate(arrays, axis=0)
    mean, sem = ranked.mean(axis=0), ranked.std(axis=0, ddof=1)/10
    assert mean.shape == (5, 1200) and np.all(np.diff(mean, axis=-1) >= 0)
    return mean, sem, records


def main():
    summaries, inputs = {}, []
    for alpha in (1, 3):
        mean, sem, records = load(alpha)
        summaries[alpha] = (mean, sem)
        inputs.extend(records)
    with (HERE/'ranked_occupation_means.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['alpha_1', 'cycle', 'rank', 'mean_occupation', 'trajectory_sem', 'samples'])
        for alpha, (mean, sem) in summaries.items():
            for i, cycle in enumerate(CYCLES):
                for rank in range(1200):
                    writer.writerow([alpha, cycle, rank+1, mean[i, rank], sem[i, rank], 100])

    configure_style({'axes.linewidth': .8, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True})
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.85), sharex=True, sharey=True)
    styles = [('0.45', '--'), ('#0072B2', '-'), ('#D55E00', ':'),
              ('#009E73', '-.'), ('#CC79A7', '-')]
    rank = np.arange(1, 1201)
    for ax, alpha, letter in zip(axes, (1, 3), 'ab'):
        mean, sem = summaries[alpha]
        for i, (cycle, (color, linestyle)) in enumerate(zip(CYCLES, styles)):
            ax.plot(rank, mean[i], color=color, ls=linestyle, lw=1,
                    label=rf'${cycle}$', zorder=10-i)
        ax.set(xlim=(1, 1200), ylim=(-.035, 1.035), xticks=[1, 300, 600, 900, 1200],
               yticks=[0, .25, .5, .75, 1], xlabel=r'ordered eigenvalue index $j$')
        ax.text(.96, .10, rf'$\alpha_1={alpha}$', transform=ax.transAxes,
                ha='right', fontsize=10)
        ax.text(-.13, 1.04, f'({letter})', transform=ax.transAxes, fontsize=9)
        ax.legend(title=r'cycle $t$', loc='upper left', frameon=False, fontsize=8,
                  title_fontsize=8, ncol=1, handlelength=2.2, labelspacing=.25)
        inset = ax.inset_axes([.115, .12, .32, .29])
        for i, (color, linestyle) in enumerate(styles):
            inset.plot(rank, mean[i], color=color, ls=linestyle, lw=.9, zorder=10-i)
        inset.set(xlim=(580, 620), ylim=(-.035, 1.035), xticks=[580,600,620], yticks=[0,1])
        inset.tick_params(labelsize=8, length=2, pad=1)
        inset.text(.5, 1.06, 'central ranks', ha='center', va='bottom',
                   transform=inset.transAxes, fontsize=8)
    axes[0].set_ylabel(r'$\overline{\nu_{(j)}(t)}$', fontsize=9)
    fig.subplots_adjust(left=.08, right=.97, bottom=.20, top=.91, wspace=.14)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        box = ax.get_tightbbox(renderer)
        assert box.x0 >= 0 and box.y0 >= 0
        assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
    for ext in ('pdf', 'png'):
        fig.savefig(HERE/f'{STEM}.{ext}', dpi=300)
    plt.close(fig)
    fonts = subprocess.check_output(['pdffonts', str(HERE/f'{STEM}.pdf')], text=True)
    names = [line.split()[0].split('+')[-1] for line in fonts.splitlines()[2:] if line.strip()]
    assert names and all(FONT_PATTERN.fullmatch(name) for name in names)
    assert 'Type 3' not in fonts
    (HERE/'provenance.json').write_text(json.dumps({
        'estimator': 'Sort eigenvalues ascending independently within each sample and cycle; average each rank over 100 samples.',
        'shape_per_alpha': [100, 5, 1200], 'cycles': CYCLES, 'Nx': 20, 'Ny': 30,
        'alpha_1': [1, 3], 'initialization': 'full system maximally mixed',
        'rank_is_not_a_tracked_mode': True, 'inset_rank_limits': [580, 620],
        'alpha3_caveat': 'Unclipped prefix 0..30; numerical covariance spectral clipping at handoff and each subsequent cycle end.',
        'uncertainty': 'Means plotted; ordinary trajectory SEM recorded in CSV, not displayed.',
        'inputs': inputs, 'fonts': names, 'dpi': 300, 'script_sha256': digest(Path(__file__)),
        'checks': 'Input receipt hashes, sample IDs 0..99, physical bounds, initial occupations 1/2, trace preservation under sorting, and ascending means verified.',
    }, indent=2)+'\n')
    print(HERE/f'{STEM}.pdf')


if __name__ == '__main__':
    main()
