"""Plot the verified, untwirled steady-state two-point purity gap only."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'results/20261005T184136Z'
OUT = ROOT / 'stationary_gap_plot'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    rows, inputs = [], {}
    for alpha in (1, 3):
        for ny in (20, 40, 60, 80, 100):
            folder = ROOT / f'alpha{alpha}_Ny{ny:03d}'
            receipt = folder / 'completion.json'
            rec = json.loads(receipt.read_text())
            result = folder / rec['result_filename']
            assert rec['status'] == 'complete' and rec['diagnostics']['converged']
            assert result.stat().st_size == rec['result_bytes']
            assert sha(result) == rec['result_sha256']
            cfg = rec['config']
            assert cfg['alpha_1'] == alpha and cfg['Nx'] == 20 and cfg['Ny'] == ny
            assert cfg['all_slabs_active'] and cfg['dw_truncation']
            with np.load(result, allow_pickle=False) as data:
                assert data['observation_cycles'][-1] == 61
                gap = float(np.min(abs(1 - 2 * data['occupations_full'][-1])))
                assert abs(gap - rec['diagnostics']['purity_gap_full']) < 1e-13
                assert max(data['successive_frobenius_change'][-5:]) < 1e-10
                assert abs(data['purity_gap_full'][-1] - data['purity_gap_full'][-2]) < 2e-10
            rows.append(dict(alpha_1=alpha, Nx=20, Ny=ny, purity_gap=gap,
                             half_filling_distance=gap / 2))
            inputs[str(receipt.relative_to(ROOT))] = sha(receipt)
            inputs[str(result.relative_to(ROOT))] = rec['result_sha256']

    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'CMU Sans Serif', 'font.size': 8,
                         'axes.labelsize': 9, 'mathtext.fontset': 'cm',
                         'xtick.direction': 'in', 'ytick.direction': 'in',
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(3.375, 2.7))
    for alpha, color, marker, ls in ((1, '#2468ad', 'o', '-'), (3, '#c0392b', '^', ':')):
        selected = [row for row in rows if row['alpha_1'] == alpha]
        ax.semilogy([r['Ny'] for r in selected], [r['purity_gap'] for r in selected],
                    color=color, marker=marker, ls=ls, ms=4.5, mfc='white',
                    lw=1.1, label=rf'$\alpha_1={alpha}$')
    ax.set(xlabel=r'$N_y$', ylabel=r'$\Delta_{\mathrm{pur}}=\min_a|1-2n_a|$',
           xlim=(15, 105), ylim=(0.0025, 1.5), xticks=[20, 40, 60, 80, 100])
    ax.tick_params(which='both', top=True, right=True)
    ax.legend(frameon=False, loc='center right')
    ax.text(.04, .84, r'Hard wall, $N_x=20$', transform=ax.transAxes, va='top')
    fig.tight_layout(pad=.7)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'stationary_purity_gap_vs_Ny.{ext}', dpi=300)
    plt.close(fig)
    with (OUT / 'data.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'caption.txt').write_text(
        'Steady-state two-point purity gap versus Ny, fixed Nx=20. '
        'Delta_pur=min_a|1-2n_a|=2 min_a|n_a-1/2|, using the full untwirled '
        'endpoint occupation spectrum, including both hard-wall sectors. '
        'Exact outcome-averaged canonical CPU channel; no trajectory sampling '
        'or statistical error bars. Maximally mixed initialization; 60 cycles '
        'plus a 61st stationarity check; all 10 cases verified. '
        'alpha_1=1,3; alpha_2=30; nshell=1; inclusive walls x=5,15; '
        'hard-wall support truncation; all slabs active; periodic boundaries; '
        'zero twist; X trial orbitals; complex128; perfect correction and '
        'measurement dephasing; raster-y Ap,Am,Bp,Bm. Logarithmic vertical axis; '
        'lines guide the eye, not fits. These finite-size data do not establish '
        'a thermodynamic closing. This is not the channel relaxation gap.\n')
    manifest = dict(inputs=inputs, script_sha256=sha(Path(__file__)),
                    estimator='min(abs(1-2*occupations_full[-1]))',
                    verified_cases=len(rows), outputs={p.name: sha(p) for p in OUT.iterdir()
                                                       if p.name != 'manifest.json'})
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Verified {len(rows)} cases; saved {OUT}')


if __name__ == '__main__':
    main()
