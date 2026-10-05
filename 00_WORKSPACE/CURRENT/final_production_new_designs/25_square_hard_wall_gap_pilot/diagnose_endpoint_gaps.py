"""Fixed-time sampling/numerical diagnostics; no dynamics and no exponent fit."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import run_campaign as runner
from endpoint_spectrum import spectral_products

HERE = Path(__file__).resolve().parent
DATA = HERE / 'gpu_data' / runner.default_config()['sampling_revision']
DEST = HERE / 'analysis_outputs' / 'diagnostics'


def main():
    config = runner.default_config()
    rows, cases, inputs = [], [], []
    groups = {}
    for task in reversed(runner.tasks(config)):
        sample_rows = []
        for start in (0, 5):
            identity = runner.identity(task, config)
            assert runner.result_verified(DATA, task, start, identity)
            path, receipt = runner.result_paths(DATA, task, start)
            inputs.extend(dict(path=str(p), sha256=runner.sha(p)) for p in (path, receipt))
            with np.load(path, allow_pickle=False) as z:
                raw = z['occupation_spectrum_raw']
                rebuilt = spectral_products(raw, 10)
                for key, value in rebuilt.items():
                    np.testing.assert_allclose(z[key], value, rtol=1e-13, atol=1e-13)
                for i, sample in enumerate(z['sample_indices']):
                    nu = raw[i]
                    nearest = float(nu[np.abs(nu-.5).argmin()])
                    row = dict(L=task.L, sample_index=int(sample), T=10,
                               modular_gap=float(z['modular_gap'][i]),
                               lyapunov_gap=float(z['lyapunov_gap'][i]),
                               nearest_half_occupation=nearest,
                               uncapped_modes=int((~z['cap_mask'][i]).sum()),
                               occupation_bound_excess=float(z['occupation_bound_excess'][i]),
                               hermiticity_residual=float(z['hermiticity_residual'][i]))
                    assert 1e-9 < nearest < 1-1e-9
                    sample_rows.append(row)
        assert [r['sample_index'] for r in sample_rows] == list(range(10))
        rows.extend(sample_rows)
        gaps = np.array([r['lyapunov_gap'] for r in sample_rows])
        groups[task.L] = gaps
        cases.append(dict(L=task.L, mean=float(gaps.mean()),
                          sem=float(gaps.std(ddof=1)/np.sqrt(10)),
                          median=float(np.median(gaps)), minimum=float(gaps.min()),
                          maximum=float(gaps.max()),
                          mean_uncapped_modes=float(np.mean([r['uncapped_modes'] for r in sample_rows]))))
    small, large = cases[0], cases[-1]
    difference = small['mean']-large['mean']
    difference_sem = float(np.hypot(small['sem'], large['sem']))
    report = dict(cases=cases,
                  L20_minus_L44=dict(difference=difference, sampling_sem=difference_sem,
                                    difference_over_sem=difference/difference_sem,
                                    fractional_decrease=difference/small['mean']),
                  max_occupation_bound_excess=max(r['occupation_bound_excess'] for r in rows),
                  max_hermiticity_residual=max(r['hermiticity_residual'] for r in rows),
                  minimum_nearest_half_occupation=min(r['nearest_half_occupation'] for r in rows),
                  maximum_nearest_half_occupation=max(r['nearest_half_occupation'] for r in rows),
                  uncertainty='Ordinary sample SEM; endpoint contrast combines independent size SEMs in quadrature',
                  fitted_exponent=None, inputs=inputs)
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST/'sample_spectral_diagnostics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['CMU Sans Serif','DejaVu Sans'],
                         'font.size':8, 'mathtext.fontset':'cm', 'pdf.fonttype':42,
                         'xtick.direction':'in', 'ytick.direction':'in', 'xtick.top':True, 'ytick.right':True})
    fig, ax = plt.subplots(figsize=(3.375, 2.7), layout='constrained')
    for L, gaps in groups.items():
        # Deterministic horizontal offsets expose samples without changing their values.
        ax.scatter(L+np.linspace(-.7,.7,10), gaps, s=9, color='0.65', alpha=.7,
                   linewidths=0, label='individual trajectories' if L==20 else None)
    ax.errorbar([r['L'] for r in cases], [r['mean'] for r in cases],
                yerr=[r['sem'] for r in cases], color='#1f77b4', marker='o', mfc='white',
                ls='--', lw=1, ms=4, capsize=2, label=r'$T=10$: mean $\pm$ SEM')
    ax.set(xlabel=r'square size $L=N_x=N_y$', ylabel=r'$\Delta=g_{\mathrm{mod}}/20$',
           xticks=config['sizes'], ylim=(0, .069))
    ax.legend(frameon=False, fontsize=7, loc='upper right')
    for ext in ('pdf','png'):
        fig.savefig(DEST/f'gap_with_individual_samples.{ext}', dpi=300)
    plt.close(fig)
    report['outputs'] = [dict(path=p.name, sha256=runner.sha(p)) for p in sorted(DEST.iterdir())
                         if p.suffix in ('.csv','.pdf','.png')]
    (DEST/'spectral_diagnostics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','outputs')}, indent=2))


if __name__ == '__main__':
    main()
