"""Plot verified old and extended square spectra; never extrapolate missing sizes."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run_large import PROJECT, SIZES, verified, folder, sha, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    old = PROJECT/'square_2Ny_v1/results/20260929T213855Z'
    records = []
    inputs = {}
    for alpha in (1,3):
        for n in (20,24,28,32,36,40,44,50,60,80,100):
            new = n in SIZES
            path = folder(root,alpha,n) if new else old/f'alpha{alpha}_L{n:03d}/spectral'
            if new and not verified(root,alpha,n):
                raise RuntimeError(f'Incomplete new spectrum: {path}')
            receipt = json.loads((path/'completion.json').read_text())
            archive = path/'spectrum.npz'
            assert sha(archive) == receipt['result_sha256']
            assert archive.stat().st_size == receipt['result_bytes']
            cfg = receipt['config']
            assert cfg['Nx'] == cfg['Ny'] == n and cfg['alpha_1'] == alpha
            assert cfg['alpha_2'] == 30 and cfg['walls'] == [n//4,3*n//4]
            assert cfg['dw_truncation'] and cfg['all_slabs_active']
            assert cfg['nshell'] == 1 and cfg['dtype'] == 'complex128'
            assert cfg['sequence'] == 'raster_y' and cfg['channel_order'] == ['Ap','Am','Bp','Bm']
            assert cfg['perfect_correction'] and cfg['dephasing']
            with np.load(archive, allow_pickle=False) as data:
                assert data['eigenvalues'].size == 2*n*n
                radius = float(np.max(abs(data['eigenvalues'])))
            gap = -2*np.log(radius)
            np.testing.assert_allclose(gap, receipt['diagnostics']['covariance_gap_raw'], atol=1e-12)
            records.append(dict(alpha_1=alpha, Nx=n, Ny=n, gap=gap, rho=radius,
                                status=receipt['diagnostics']['gap_status']))
            inputs[str(archive)] = sha(archive)
            inputs[str(path/'completion.json')] = sha(path/'completion.json')
    out = root/'analysis'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':9,
                         'xtick.direction':'in','ytick.direction':'in'})
    means = {}
    for centered in (False,True):
        fig, axes = plt.subplots(1,2,figsize=(7.05,3.05),sharey=centered)
        for ax,alpha,color,marker,style,panel in zip(axes,(3,1),('#c0392b','#2468ad'),('^','o'),(':','-'),('a','b')):
            selected = [r for r in records if r['alpha_1'] == alpha]
            values = np.array([r['gap'] for r in selected])
            means[str(alpha)] = float(values.mean())
            if centered:
                values -= values.mean()
                ax.set_yscale('symlog',linthresh=1e-3)
                ax.axhline(0,color='.5',ls='--',lw=.8)
            ax.plot([r['Ny'] for r in selected], values,color=color,marker=marker,ls=style,mfc='white',lw=1,ms=4)
            ax.set_xlabel(r'$N_x=N_y$')
            ax.set_xticks([20,40,60,80,100])
            ax.tick_params(top=True,right=True)
            ax.text(-.12,1.03,f'({panel})',transform=ax.transAxes)
            ax.text(.96,.95,rf'$\alpha_1={alpha}$',transform=ax.transAxes,ha='right',va='top')
        axes[0].set_ylabel((r'$\Delta_C-\langle\Delta_C\rangle_{N_y}$' if centered else r'Exact $\Delta_C$')+r' (cycle$^{-1}$)')
        fig.tight_layout(pad=.8)
        name = 'centered_square_gaps' if centered else 'exact_square_gaps'
        for ext in ('png','pdf'):
            fig.savefig(out/f'{name}.{ext}',dpi=300)
        plt.close(fig)
    with (out/'gaps.csv').open('w') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    (out/'caption.txt').write_text(
        'Exact square-system covariance-channel spectra, alpha_1=3 (left), 1 (right). '
        'Nx=Ny=20,24,28,32,36,40,44,50,60,80,100. Hard-wall support truncation, '
        'inclusive walls floor(L/4),floor(3L/4), alpha_2=30, nshell=1, periodic, '
        'X orbitals, zero twist, all slabs active, complex128, perfect correction '
        'and measurement dephasing. Raster-y Ap/Am/Bp/Bm. Complete spectra; no trajectories, '
        'time horizon, fit or sampling uncertainty. Larger cases diagonalize both exact '
        'hard-wall sectors separately, preserving the full spectrum. Lines guide the eye. '
        'Centered plot subtracts the arithmetic mean of ALL ELEVEN sizes separately for each '
        'alpha (different reference from the prior eight-size plot), using symlog, linear '
        'within +/-0.001 cycle^-1. Centering does not estimate an asymptotic gap.\n')
    atomic_json(out/'manifest.json',dict(inputs=inputs,means=means,source_sha256=sha(__file__),
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}))
    print('[analysis complete]',out,flush=True)


if __name__ == '__main__':
    main()
