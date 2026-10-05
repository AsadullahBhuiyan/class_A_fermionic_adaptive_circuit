"""Local-only descriptive relaxation fits for the verified 220-trajectory snapshot.

Does not modify the production bundle, raw results, or previous analysis.
Run with: python analyze_relaxation.py
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / 'analysis_outputs/snapshot_L20S100_L30S100_L40S20'
OUT = ROOT / 'analysis_outputs/relaxation_L20S100_L30S100_L40S20'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return dict(bytes=path.stat().st_size, sha256=h.hexdigest())


def stats(values):
    return values.mean(0), values.std(0, ddof=1) / np.sqrt(len(values))


def fit(t, gap, lo, hi):
    selected = (t >= lo) & (t <= hi)
    assert np.all(gap[selected] > 0)
    slope, intercept = np.polyfit(t[selected], np.log(gap[selected]), 1)
    assert slope < 0
    return dict(first_cycle=lo, last_cycle=hi, rate_per_cycle=float(-slope),
                e_folding_cycles=float(-1 / slope), log_amplitude=float(intercept),
                log_residual_rms=float(np.sqrt(np.mean(
                    (np.log(gap[selected]) - (intercept + slope * t[selected]))**2))))


def main():
    provenance = json.loads((SNAPSHOT / 'summary.json').read_text())
    data = ROOT / 'data' / provenance['config']['revision'] if 'revision' in provenance['config'] else None
    if data is None or not data.is_dir():
        candidates = list((ROOT / 'data').glob('square_hard_wall_random_center_chern_*'))
        assert len(candidates) == 1
        data = candidates[0]
    groups = {20: [], 30: [], 40: []}
    sources = []
    for entry in provenance['source_files']:
        path = data / entry['name']
        actual = digest(path)
        assert actual == {k: entry[k] for k in ('bytes', 'sha256')}, path
        receipt_path = path.with_suffix('.json')
        receipt = json.loads(receipt_path.read_text())
        assert receipt['file'] == actual and receipt['identity'] == provenance['identity']
        with np.load(path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['cycles'], np.arange(41))
            c, raw = z['center_average'], z['real_space_chern']
            np.testing.assert_allclose(c, raw.mean(2), rtol=1e-12, atol=1e-14)
            groups[receipt['task']['nx']].append((z['sample_ids'], c))
        sources.append(dict(name=path.name, **actual, receipt=digest(receipt_path)))
        print('Verified', path.name, flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'CMU Sans Serif', 'font.size': 9,
                         'mathtext.fontset': 'cm', 'xtick.direction': 'in',
                         'ytick.direction': 'in', 'legend.frameon': False})
    fig, axes = plt.subplots(2, 1, figsize=(3.375, 5.1), layout='constrained')
    early, ax = plt.subplots(figsize=(3.375, 3.2), layout='constrained')
    results, rows, fits = [], [], []
    t = np.arange(41)
    windows = [(0, 3), (1, 4), (1, 5), (2, 4), (2, 5)]
    for L, color, marker, line in zip(groups, ('#d94738', '#249b57', '#1675bd'),
                                      ('^', 's', 'o'), (':', '--', '-')):
        ids = np.concatenate([p[0] for p in groups[L]])
        c = np.concatenate([p[1] for p in groups[L]])
        S = len(c)
        np.testing.assert_array_equal(np.sort(ids), np.arange(S))
        assert S == {20: 100, 30: 100, 40: 20}[L] and c.shape == (S, 41)
        mean, sem = stats(c)
        plateau_samples = c[:, 21:41].mean(1)
        plateau = plateau_samples.mean()
        gap, gap_sem = stats(plateau_samples[:, None] - c)
        fitted = fit(t, gap, 2, 5)
        sensitivity = [fit(t, gap, *window) for window in windows]
        label = rf'$L={L},\ S={S}$' + (' (partial)' if S < 100 else '')
        style = dict(color=color, marker=marker, ls=line, ms=2.7,
                     mfc='white', mew=.8, lw=1, label=label)
        axes[0].plot(t, mean, **style)
        axes[0].fill_between(t, mean-sem, mean+sem, color=color, alpha=.15)
        deviation = abs(mean-1)
        low = np.maximum(deviation-sem, 0)
        high = deviation+sem
        axes[1].plot(t, deviation, **style)
        axes[1].fill_between(t, np.maximum(low, 1e-6), high, color=color, alpha=.15)
        # Paired trajectory statistics include uncertainty of the late-time baseline.
        ax.errorbar(t[1:6], gap[1:6], yerr=gap_sem[1:6], capsize=2,
                    color=color, marker=marker, mfc='white', ms=4, ls='none', label=label)
        tf = np.linspace(2, 5, 100)
        ax.plot(tf, np.exp(fitted['log_amplitude']-fitted['rate_per_cycle']*tf),
                color=color, ls=line, lw=1.2)
        results.append(dict(L=L, samples=S, partial=S < 100,
                            late_mean=float(plateau), late_sem=float(stats(plateau_samples)[1]),
                            fit=fitted, fit_window_sensitivity=sensitivity))
        for item in sensitivity:
            fits.append(dict(L=L, samples=S, **item))
        for cycle in t:
            rows.append(dict(L=L, samples=S, cycle=int(cycle), chern_mean=mean[cycle],
                             chern_sem=sem[cycle], plateau_subtracted_gap=gap[cycle],
                             plateau_subtracted_gap_sem=gap_sem[cycle]))
    axes[0].axhline(1, color='.5', lw=.7, ls='--')
    axes[0].set(ylabel=r'$\overline{C_G}$', ylim=(-.04, 1.06))
    axes[0].legend(loc='lower right', fontsize=8, handlelength=3)
    axes[1].set(ylabel=r'$|\overline{C_G}-1|$', yscale='log', ylim=(1e-6, 1.3))
    for letter, a in zip('ab', axes):
        a.set(xlabel='cycle', xlim=(0, 40))
        a.tick_params(top=True, right=True)
        a.text(-.19, 1.025, f'({letter})', transform=a.transAxes)
    ax.set(xlabel='cycle', ylabel=r'$C_\infty-\overline{C_G}(t)$',
           yscale='log', xlim=(.7, 5.3), ylim=(2e-4, .4), xticks=np.arange(1, 6))
    ax.tick_params(top=True, right=True)
    ax.legend(loc='upper right', fontsize=8)
    for name, figure in [('all_sizes_all_cycles', fig), ('early_relaxation', early)]:
        for ext in ('pdf', 'png'):
            figure.savefig(OUT / f'{name}.{ext}', dpi=300)
        plt.close(figure)
    for name, table in [('cycles.csv', rows), ('rates.csv', fits)]:
        with (OUT / name).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    summary = dict(source_snapshot=str(SNAPSHOT.relative_to(ROOT)), source_files=sources,
                   identity=provenance['identity'], results=results,
                   method='Descriptive unweighted log-linear fit C_inf-mean(C_G)=A exp(-k t), cycles 2..5. '
                   'C_inf is the trajectory-first mean over cycles 21..40. All SEMs use independent '
                   'trajectories and ddof=1. Early-gap SEMs include paired baseline uncertainty. '
                   'No independent-cycle assumption, bootstrap, or formal fitted-rate errors. '
                   'Window sensitivity is not a confidence interval; rates are not asymptotic gaps.',
                   script=digest(Path(__file__)), products={p.name: digest(p) for p in OUT.iterdir()
                                                          if p.suffix in ('.csv', '.png', '.pdf')})
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
