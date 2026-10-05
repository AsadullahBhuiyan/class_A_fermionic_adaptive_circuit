"""Compare verified twirled gaps and inspect the discrete momentum grid."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_stationary_gap import ROOT, sha

OUT = ROOT / 'twirled_gap_plot'


def main():
    rows, spectra, inputs = [], {}, {}
    for alpha in (1, 3):
        for ny in (20, 40, 60, 80, 100):
            folder = ROOT / f'alpha{alpha}_Ny{ny:03d}'
            receipt = folder / 'completion.json'
            rec = json.loads(receipt.read_text())
            result = folder / rec['result_filename']
            assert rec['status'] == 'complete' and rec['diagnostics']['converged']
            assert result.stat().st_size == rec['result_bytes']
            assert sha(result) == rec['result_sha256']
            with np.load(result, allow_pickle=False) as data:
                s = data['occupations_ky_twirl'][-1]
                ky = data['ky'] / np.pi
                assert s.shape == (ny, 40)
                assert data['observation_cycles'][-1] == 61
                assert max(data['successive_frobenius_change'][-5:]) < 1e-10
                assert abs(data['purity_gap_twirl'][-1] - data['purity_gap_twirl'][-2]) < 2e-10
                k, band = np.unravel_index(np.argmin(abs(1 - 2*s)), s.shape)
                gap = float(abs(1 - 2*s[k, band]))
                actual = float(np.min(abs(1 - 2*data['occupations_full'][-1])))
                np.testing.assert_allclose(gap, rec['diagnostics']['purity_gap_twirl'], atol=1e-13, rtol=0)
                np.testing.assert_allclose(actual, rec['diagnostics']['purity_gap_full'], atol=1e-13, rtol=0)
                spectra[(alpha, ny)] = (ky.copy(), s.copy())
            rows.append(dict(alpha_1=alpha, Nx=20, Ny=ny, purity_gap_full=actual,
                             purity_gap_twirl=gap, closest_ky_over_pi=float(ky[k]),
                             closest_band_zero_based=int(band), closest_occupation=float(s[k, band])))
            inputs[str(receipt.relative_to(ROOT))] = sha(receipt)
            inputs[str(result.relative_to(ROOT))] = rec['result_sha256']
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'CMU Sans Serif', 'font.size': 8,
                         'axes.labelsize': 9, 'mathtext.fontset': 'cm',
                         'xtick.direction': 'in', 'ytick.direction': 'in',
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.05, 2.85))
    for alpha, color, marker in ((1, '#2468ad', 'o'), (3, '#c0392b', '^')):
        selected = [r for r in rows if r['alpha_1'] == alpha]
        x = [r['Ny'] for r in selected]
        ax.semilogy(x, [r['purity_gap_twirl'] for r in selected],
                    color=color, marker=marker, ms=4, mfc='white', lw=1,
                    label=rf'Twirled, $\alpha_1={alpha}$')
        ax.semilogy(x, [r['purity_gap_full'] for r in selected], color='.45',
                    ls='--', lw=.8, label='Untwirled' if alpha == 1 else None, zorder=1)
    ax.set(xlabel=r'$N_y$', ylabel=r'$\Delta_{\mathrm{pur}}=\min|1-2n|$',
           xlim=(15, 105), ylim=(.0025, 1.5), xticks=[20, 40, 60, 80, 100])
    ax.legend(frameon=False, loc='center right')
    ax.text(.04, .84, r'Hard wall, $N_x=20$', transform=ax.transAxes, va='top')

    styles = [(20, '#7570b3', 'v'), (40, '#1b9e77', 's'), (60, '#d95f02', '^'),
              (80, '#b04882', 'D'), (100, '#2468ad', 'o')]
    for ny, color, marker in styles:
        k, s = spectra[(1, ny)]
        order = np.argsort(k)
        mask = abs(k[order]) <= .15
        # Ordered eigenvalue 20 of 40: the branch controlling all five minima.
        assert next(r for r in rows if r['alpha_1'] == 1 and r['Ny'] == ny)['closest_band_zero_based'] == 19
        bx.plot(k[order][mask], s[order, 19][mask], color=color, marker=marker,
                ms=4, mfc='white', lw=.65, alpha=.9, label=rf'$N_y={ny}$')
    bx.axhline(.5, color='.35', ls='--', lw=.8)
    bx.set(xlabel=r'$k_y/\pi$', ylabel=r'$n_{20}(k_y)$',
           xlim=(-.115, .115), ylim=(.435, .575), xticks=[-.1, -.05, 0, .05, .1])
    bx.text(.5, .96, r'Twirled, $\alpha_1=1$', transform=bx.transAxes, ha='center', va='top')
    bx.legend(frameon=False, loc='lower center', ncol=3, columnspacing=.7, handlelength=1.4)
    for axis, panel in zip((ax, bx), ('a', 'b')):
        axis.tick_params(which='both', top=True, right=True)
        axis.text(-.13, 1.035, f'({panel})', transform=axis.transAxes, fontsize=9)
    fig.tight_layout(pad=.8, w_pad=1.8)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'twirled_gap_and_momentum_sampling.{ext}', dpi=300)
    plt.close(fig)
    with (OUT / 'data.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'caption.txt').write_text(
        'Translation-twirled endpoint spectra: (a) purity gap versus Ny for alpha_1=1,3; '
        'gray dashed curves are the untwirled gaps. (b) ordered eigenvalue 20 of 40 '
        'in each ky block, alpha_1=1, near half filling; it controls all five sampled '
        'minima. Points are actual discrete momenta; connecting lines only guide the '
        'eye and do not assert an interpolated spectrum. Fixed Nx=20, hard-wall '
        'support truncation at x=5,15 inclusive; all slabs active; alpha_2=30; nshell=1; '
        'periodic; X trials; zero twist; complex128; perfect correction and measurement '
        'dephasing; raster-y Ap,Am,Bp,Bm. Exact canonical outcome-averaged channel '
        'from a maximally mixed initial state; 60 cycles plus a 61st stationarity '
        'check. No sampled trajectories, statistical errors, fits, or averages over '
        'history. Twirling means averaging the endpoint correlation matrix over '
        'discrete y translations before diagonalization, not averaging eigenvalues; '
        'the twirled state is not assumed fixed by the raster channel. Branch values '
        'straddle 1/2 on the sampled grid; a continuous-momentum crossing and a '
        'thermodynamic gap closing remain interpretations, not established limits.\n')
    (OUT / 'manifest.json').write_text(json.dumps(dict(inputs=inputs,
        script_sha256=sha(Path(__file__)), verified_cases=len(rows),
        outputs={p.name: sha(p) for p in OUT.iterdir() if p.name != 'manifest.json'}), indent=2)+'\n')
    print(OUT)


if __name__ == '__main__':
    main()
