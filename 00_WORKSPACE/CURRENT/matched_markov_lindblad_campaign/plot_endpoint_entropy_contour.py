"""Full-system entropy contours of the saved, untwirled channel endpoints."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from src.fgtn.classA_U1FGTN import classA_U1FGTN

ROOT = HERE / 'results/raster_y_channel_endpoints_v1_20260915T025739Z'
OUT = HERE / 'analysis_outputs/raster_y_endpoint_entropy_contour_v1'


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def contour(c, nx, ny):
    c = np.asarray(c, dtype=np.complex128)
    if c.shape != (2*nx*ny, 2*nx*ny) or not np.isfinite(c).all():
        raise ValueError('Expected finite, two-dimensional occupation matrix')
    np.testing.assert_allclose(c, c.conj().T, atol=1e-13, rtol=0)
    # The canonical helper accepts engine covariance Gamma=2C-I, NOT C.
    # It needs no initialized engine instance or OW construction.
    return classA_U1FGTN.entanglement_contour(None, 2*c-np.eye(len(c)), nx, ny)


def validate_small():
    np.testing.assert_allclose(contour(.5*np.eye(12), 2, 3), 2*np.log(2), atol=1e-13)
    assert np.max(contour(np.diag([0., 1.]*6), 2, 3)) < 1e-9
    n = np.linspace(.1, .9, 12)
    expected = (-n*np.log(n)-(1-n)*np.log1p(-n)).reshape(2, 2, 3, order='F').sum(axis=0)
    np.testing.assert_allclose(contour(np.diag(n), 2, 3), expected, atol=1e-13)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', type=int, default=2)
    args = parser.parse_args()
    if args.cpu not in os.sched_getaffinity(0):
        parser.error('Requested CPU is unavailable')
    os.sched_setaffinity(0, {args.cpu})
    validate_small()
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, diagnostics = {}, {}
    for alpha in tqdm((1, 3), desc='endpoint contours', unit='matrix'):
        receipt_path = ROOT / f'alpha{alpha}_hard/completion.json'
        receipt = json.loads(receipt_path.read_text())
        path = receipt_path.parent / receipt['result_filename']
        assert receipt['status'] == 'complete'
        assert path.stat().st_size == receipt['result_bytes'] and sha(path) == receipt['result_sha256']
        cfg = receipt['config']
        assert (cfg['Nx'], cfg['Ny'], cfg['cycles'], cfg['alpha_1'], cfg['nshell']) == (20, 64, 128, alpha, 1)
        assert cfg['family'] == 'markov_channel' and cfg['evolution_domain'] == 'full_system'
        assert cfg['wall'] == 'hard' and cfg['site_schedule'] == 'raster_y'
        with np.load(path, allow_pickle=False) as saved:
            assert json.loads(str(saved['config_json'])) == cfg
            c = saved['G_final'][0]  # already the occupation matrix, not Gamma
            occupations = saved['active_occupations'][-1]
        assert occupations.shape == (2560,)
        assert occupations.min() >= -1e-10 and occupations.max() <= 1+1e-10
        print(f'[diagonalize] alpha_1={alpha}, dimension={len(c)}', flush=True)
        s = contour(c, 20, 64)
        n = np.clip(occupations, 1e-12, 1-1e-12)
        spectral_entropy = float(np.sum(-n*np.log(n)-(1-n)*np.log1p(-n)))
        np.testing.assert_allclose(s.sum(), spectral_entropy, atol=1e-8, rtol=1e-10)
        assert np.min(s) >= 0 and np.max(s) <= 2*np.log(2)+1e-12
        arrays[f'alpha{alpha}_contour'] = s
        arrays[f'alpha{alpha}_ymean'] = s.mean(axis=1)
        diagnostics[str(alpha)] = dict(source=str(path), source_sha256=receipt['result_sha256'],
            receipt_sha256=sha(receipt_path), config=cfg, total_entropy_nats=float(s.sum()),
            saved_spectrum_entropy_nats=spectral_entropy,
            sum_rule_error=float(abs(s.sum()-spectral_entropy)),
            minimum_nats_per_cell=float(s.min()), maximum_nats_per_cell=float(s.max()),
            ymean_by_x=s.mean(axis=1).tolist(),
            y_variation_max=float(np.max(np.ptp(s, axis=1))))
        print('[verified]', alpha, diagnostics[str(alpha)]['total_entropy_nats'], 'nats', flush=True)
    np.savez_compressed(OUT / 'contours.npz', **arrays)
    with (OUT / 'contours.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['alpha_1', 'x', 'y', 'entropy_nats_per_cell'])
        for alpha in (1, 3):
            s = arrays[f'alpha{alpha}_contour']
            writer.writerows((alpha, x, y, s[x, y]) for x in range(20) for y in range(64))
    plt.rcParams.update({'font.family': 'CMU Sans Serif', 'font.size': 8, 'text.usetex': True,
                         'xtick.direction': 'in', 'ytick.direction': 'in'})
    fig, axes = plt.subplots(1, 2, figsize=(3.375, 4.9), sharey=True, layout='constrained')
    vmax = max(float(arrays[f'alpha{a}_contour'].max()) for a in (1, 3))
    for panel, (ax, alpha) in enumerate(zip(axes, (1, 3))):
        im = ax.imshow(arrays[f'alpha{alpha}_contour'].T, origin='lower', cmap='magma',
                       interpolation='nearest', vmin=0, vmax=vmax, aspect='auto',
                       extent=(-.5, 19.5, -.5, 63.5))
        for wall in (4.5, 15.5):
            ax.axvline(wall, color='0.6', ls='--', lw=.6)
        ax.set(xlabel=r'$x$', xticks=[0, 5, 10, 15, 19], yticks=[0, 16, 32, 48, 63])
        ax.set_title(rf'$\alpha_1={alpha}$', fontsize=11, pad=8)
        ax.text(-.14, 1.03, f'({chr(97+panel)})', transform=ax.transAxes)
        ax.tick_params(top=True, right=True)
    axes[0].set_ylabel(r'$y$')
    cbar = fig.colorbar(im, ax=axes, location='bottom', shrink=.95, pad=.025, aspect=28)
    cbar.set_label(r'$s(x,y)$ (nats per unit cell)')
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'endpoint_entropy_contour.{ext}', dpi=300)
    plt.close(fig)
    caption = ('Full-system endpoint entropy contours s(x,y) = sum_mu [h(C)]_(x,y,mu),(x,y,mu), '
        'h(n)=-n ln n-(1-n)ln(1-n), using untwirled G_final[0]=C at cycle 128. '
        'Nx=20, Ny=64, hard wall, alpha_1=1,3, alpha_2=30, nshell=1, X trial orbitals, '
        'all slabs active, perfect correction and measurement dephasing, maxmix initialization, '
        'fixed raster-y order. Outcomes averaged analytically: no sampled-trajectory count, '
        'SEM, temporal average, spatial twirl, fit, or new dynamics. Sum both orbitals at each '
        'site; natural logarithms; common linear color scale. Dashed lines separate the '
        'inclusive x=5,...,15 slab from the exterior. This is the entropy contour of the '
        'Gaussian state associated with the averaged covariance, not the average of '
        'trajectory entropy contours, and not a subsystem entanglement contour. '
        'Canonical helper input is Gamma=2C-I; its eigenvalue clipping is [1e-12,1-1e-12].\n')
    (OUT / 'caption.txt').write_text(caption)
    manifest = dict(cases=diagnostics, small_checks='maxmix, pure, diagonal/site-order passed',
        estimator='canonical classA_U1FGTN.entanglement_contour(None, 2*C-I, Nx, Ny)',
        cpu_affinity=sorted(os.sched_getaffinity(0)), source_sha256={
            str(Path(__file__)): sha(Path(__file__)),
            str(REPO/'src/fgtn/classA_U1FGTN.py'): sha(REPO/'src/fgtn/classA_U1FGTN.py')},
        output_sha256={p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'manifest.json'})
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('[saved]', OUT, flush=True)


if __name__ == '__main__':
    main()
