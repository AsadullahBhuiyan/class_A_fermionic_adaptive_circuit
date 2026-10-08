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

CYCLES = (60,)
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
            all_nu = saved['occupation_spectrum']
            np.testing.assert_allclose(all_nu[:, 0], .5, atol=1e-12, rtol=0)
            nu = all_nu[:, CYCLES, :]
            assert nu.shape == (5, len(CYCLES), 1200) and np.isfinite(nu).all()
            assert nu.min() >= -1e-9 and nu.max() <= 1+1e-9
            # Rank labels are reassigned independently at each cycle and sample.
            ranked = np.sort(nu, axis=-1)
            assert np.all(np.diff(ranked, axis=-1) >= 0)
            np.testing.assert_allclose(ranked.sum(-1), nu.sum(-1), atol=1e-10, rtol=0)
            arrays.append(ranked)
            ids.extend(saved['sample_indices'].tolist())
        records.append({'path': str(path), 'sha256': sha})
    assert sorted(ids) == list(range(100))
    ranked = np.concatenate(arrays, axis=0)
    mean, sem = ranked.mean(axis=0), ranked.std(axis=0, ddof=1)/10
    assert mean.shape == (len(CYCLES), 1200) and np.all(np.diff(mean, axis=-1) >= 0)
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
    fig, ax = plt.subplots(figsize=(3.375, 2.85))
    axes = [ax]
    rank = np.arange(1, 1201)
    inset = ax.inset_axes([.10, .15, .34, .32], zorder=20)
    for alpha, color, marker in ((1, '#0072B2', 'o'), (3, '#D55E00', '^')):
        mean, sem = summaries[alpha]
        ax.plot(rank, mean[0], color=color, ls='none', marker=marker,
                ms=1.8, mfc='none', mew=.45, label=rf'$\alpha_1={alpha}$')
        inset.plot(rank, mean[0], color=color, ls='none', marker=marker,
                   ms=3, mfc='none', mew=.6)
    ax.set(xlim=(1, 1200), ylim=(-.035, 1.035), xticks=[1, 300, 600, 900, 1200],
           yticks=[0, .25, .5, .75, 1], xlabel=r'ordered eigenvalue index $j$',
           ylabel=r'$\overline{\nu_{(j)}(t)}$')
    ax.legend(title=r'cycle $t=60$', loc='upper left', frameon=False, fontsize=8,
              title_fontsize=8, markerscale=2, handlelength=1.5, labelspacing=.25)
    inset.set(xlim=(580, 620), ylim=(-.035, 1.035), xticks=[580, 600, 620], yticks=[0, 1])
    inset.tick_params(labelsize=8, length=2, pad=1)
    fig.subplots_adjust(left=.19, right=.965, bottom=.20, top=.96)
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
        'layout': [1, 1], 'inset_titles': False,
        'shape_per_alpha': [100, len(CYCLES), 1200], 'cycles': CYCLES, 'Nx': 20, 'Ny': 30,
        'alpha_1': [1, 3], 'initialization': 'full system maximally mixed',
        'rank_is_not_a_tracked_mode': True, 'inset_rank_limits': [580, 620],
        'alpha3_caveat': 'Cycles 0..30 are from the unclipped prefix; cycle 60 is from the continuation with numerical covariance spectral clipping.',
        'display': 'All 1200 ranks plotted as markers only; no connecting lines.',
        'uncertainty': 'Means plotted; ordinary trajectory SEM recorded in CSV, not displayed.',
        'inputs': inputs, 'fonts': names, 'dpi': 300, 'script_sha256': digest(Path(__file__)),
        'checks': 'Input receipt hashes, sample IDs 0..99, physical bounds, initial occupations 1/2, trace preservation under sorting, and ascending means verified.',
    }, indent=2)+'\n')
    print(HERE/f'{STEM}.pdf')


if __name__ == '__main__':
    main()
