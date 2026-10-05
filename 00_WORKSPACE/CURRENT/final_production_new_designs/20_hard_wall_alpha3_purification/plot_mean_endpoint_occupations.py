#!/usr/bin/env python3
"""Compare rank-wise mean endpoint occupations, never eigenvalues of mean state."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvalsh
from threadpoolctl import threadpool_limits
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'gpu_data/hard_wall_alpha3_maxmix_nx20_ny40_s100_4ny_v1'
MODES = ROOT.parents[1] / 'experiment_review/purification_slow_mode_profiles'
OUT = ROOT / 'analysis_outputs/mean_endpoint_occupations_alpha1_alpha3_v1'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_spectra():
    provenance = []
    manifest = json.loads((MODES / 'analysis_manifest.json').read_text())
    expected = {r['result_file']: r['result_sha256'] for r in manifest['outputs']}
    one = []
    for i in tqdm(range(100), desc='Verify alpha1=1', unit='sample'):
        path = MODES / f'endpoint_modes/hard/Ny040/sample_{i:03d}.npz'
        digest = sha(path)
        assert digest == expected[str(path.relative_to(MODES))]
        with np.load(path, allow_pickle=False) as z:
            assert int(z['sample_index']) == i and int(z['cycles']) == 160
            assert int(z['Ny']) == 40 and float(z['alpha_1']) == 1
            assert str(z['construction']) == 'hard'
            one.append(np.sort(z['occupations']))
        provenance.append(dict(path=str(path), sha256=digest))
    manifest = json.loads((DATA / 'DOWNLOAD_MANIFEST.json').read_text())
    config = manifest['resolved_configuration']
    assert config['perfect_correction'] and not config['postselect']
    assert config['alpha_1'] == 3 and config['Ny_values'] == [40]
    files = {r['path']: r for r in manifest['files']}
    for name, row in files.items():
        path = DATA / name
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        provenance.append(dict(path=str(path), sha256=row['sha256']))
    active = np.array([i for i in range(1600) if 5 <= (i // 2) % 20 <= 15])
    exterior = np.setdiff1d(np.arange(1600), active)
    assert len(active) == 880
    three, ids, diagnostics = [], [], []
    for name in tqdm(sorted(n for n in files if n.endswith('.npz')), desc='Alpha1=3 slab spectra', unit='shard'):
        path = DATA / name
        receipt = json.loads(path.with_suffix('.complete.json').read_text())
        assert receipt['result_sha256'] == files[name]['sha256']
        with np.load(path, allow_pickle=False) as z:
            assert str(z['configuration_hash']) == receipt['configuration_hash'] == manifest['configuration_hash']
            assert int(z['Ny']) == 40 and float(z['alpha_1']) == 3
            np.testing.assert_array_equal(z['cycles'], np.arange(161))
            np.testing.assert_array_equal(z['sample_indices'], receipt['sample_indices'])
            states = z['G_final']
            assert states.dtype == np.complex128
            for i, state in zip(z['sample_indices'], states):
                g = state[np.ix_(active, active)]
                herm = float(np.max(abs(g - g.conj().T)))
                coupling = float(np.max(abs(state[np.ix_(active, exterior)])))
                assert herm < 1e-8 and coupling < 1e-9
                c = (g + g.conj().T) / 4 + np.eye(880) / 2
                with threadpool_limits(limits=2):
                    nu = eigvalsh(c, driver='evd')
                np.testing.assert_allclose(nu.sum(), np.trace(c).real, atol=1e-8)
                three.append(nu); ids.append(int(i))
                diagnostics.append(dict(sample_index=int(i), hermiticity_error=herm, exterior_coupling=coupling))
    order = np.argsort(ids)
    np.testing.assert_array_equal(np.asarray(ids)[order], np.arange(100))
    spectra = {1: np.asarray(one), 3: np.asarray(three)[order]}
    for a in spectra.values():
        assert a.shape == (100, 880) and np.isfinite(a).all()
        assert a.min() > -1e-9 and a.max() < 1 + 1e-9
        assert np.all(np.diff(a, axis=1) >= 0)
    return spectra, provenance, diagnostics


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spectra, inputs, diagnostics = load_spectra()
    np.savez_compressed(OUT / 'sample_sorted_occupations.npz',
                        alpha1_1=spectra[1], alpha1_3=spectra[3], sample_indices=np.arange(100))
    rows, summary = [], {}
    plt.rcParams.update({'font.family': 'CMU Sans Serif', 'font.size': 8,
                         'mathtext.fontset': 'cm', 'axes.linewidth': .8})
    fig, axes = plt.subplots(2, 1, figsize=(3.375, 4.5))
    ranks = np.arange(1, 881)
    for alpha, color, marker, style in [(1, '#e41a1c', '^', ':'), (3, '#1879bb', 'o', '-')]:
        raw = spectra[alpha]
        # Only remove floating-point excursions outside the physical interval.
        a = np.clip(raw, 0, 1)
        mean, sem = a.mean(axis=0), a.std(axis=0, ddof=1) / np.sqrt(len(a))
        for ax in axes:
            ax.plot(ranks, mean, color=color, marker=marker, ls=style, lw=1.2,
                    ms=3, mfc='white', markevery=1 if ax is axes[1] else 40,
                    label=rf'$\alpha_1={alpha}$')
            ax.fill_between(ranks, mean-sem, mean+sem, color=color, alpha=.18, linewidth=0)
        for j, (m, s) in enumerate(zip(mean, sem), 1):
            rows.append(dict(alpha_1=alpha, rank=j, mean_occupation=float(m), sem=float(s), samples=100))
        finite = ((a > 1e-9) & (a < 1-1e-9)).sum(axis=1)
        summary[alpha] = dict(minimum_raw_occupation=float(raw.min()), maximum_raw_occupation=float(raw.max()),
            mean_finite_modes=float(finite.mean()), min_finite_modes=int(finite.min()), max_finite_modes=int(finite.max()),
            mean_active_charge=float(a.sum(axis=1).mean()), min_active_charge=float(a.sum(axis=1).min()),
            max_active_charge=float(a.sum(axis=1).max()),
            closest_to_half_min=float(np.min(abs(a-.5))), closest_to_half_mean=float(np.min(abs(a-.5),axis=1).mean()))
    means = [np.clip(a,0,1).mean(axis=0) for a in spectra.values()]
    transition = np.flatnonzero(np.any([(m > .001) & (m < .999) for m in means],axis=0)) + 1
    axes[0].set_xlim(1,880)
    axes[1].set_xlim(int(transition.min())-3, int(transition.max())+3)
    axes[0].set_title(r'Hard wall: $N_x=20$, $N_y=40$, $T=160$', fontsize=8)
    axes[0].legend(frameon=False, loc='lower right')
    axes[1].text(.04,.95,'Transition detail; mean $\\pm$ SEM',transform=axes[1].transAxes,va='top',fontsize=8)
    for letter,ax in zip('ab',axes):
        ax.set_ylim(-.035,1.035)
        ax.set_ylabel(r'$\overline{\nu_{(j)}(T)}$')
        ax.set_xlabel('Ascending mode rank $j$')
        ax.tick_params(which='both',direction='in',top=True,right=True)
        ax.text(-.18,1.02,f'({letter})',transform=ax.transAxes,fontsize=9)
    fig.tight_layout(pad=.5,h_pad=1.3)
    for ext in ['pdf','png']:
        fig.savefig(OUT / f'mean_endpoint_occupations.{ext}',dpi=300)
    plt.close(fig)
    with (OUT/'mean_endpoint_occupations.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report=dict(protocol='Born-sampled perfect correction, maximally mixed initialization; no postselection',
        Nx=20,Ny=40,T=160,samples_per_alpha=100,active_modes=880,
        estimator='Sort each sample independently, then mean and ddof=1 SEM at fixed rank; not spectrum of mean covariance.',
        warning='Sample-dependent charge can broaden the averaged rank transition even when each sample is nearly pure.',
        cases=summary,alpha3_diagnostics=diagnostics,inputs=inputs,script_sha256=sha(Path(__file__)))
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'README.md').write_text('# Mean endpoint occupation spectra\n\n'
        'Hard wall, Nx=20, Ny=40, T=160, alpha2=30, nshell=1; 100 independent Born trajectories '
        'per alpha1, maximally mixed initialization and perfect correction. Alpha1=1 is bundle 07; '
        'alpha1=3 is bundle 20. Restrict to x=5,...,15 (880 active modes), diagonalize each sample, '
        'sort ascending, then average at fixed rank. Shading is sample-wise SEM, not bootstrap. '
        'Panel (b) magnifies the same rank transition. No fitting or averaged covariance is used. '
        'Input hashes and numerical checks are in summary.json; unrounded sample spectra are preserved in the NPZ. '
        'Only sub-1e-9 roundoff outside [0,1] is clipped for plotting.\n\n'
        'A broadened rank-averaged step can reflect sample-dependent charge, and does not by itself '
        'establish mixed modes or gaplessness in individual trajectories.\n')
    print(json.dumps(summary,indent=2),flush=True)
    print(OUT,flush=True)


if __name__ == '__main__':
    main()
