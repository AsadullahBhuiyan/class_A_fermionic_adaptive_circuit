#!/usr/bin/env python3
"""Recompute the legacy packet figure with both sources at y_rel=10."""
from __future__ import annotations

import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from tqdm import tqdm

from build_modular_chirality_figure import MODULAR_SOURCE, FIGURES, _packet_density, _style
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ('00_WORKSPACE/COLAB/colab_small_system_testing/gpu_data/'
                 'pure_state_covariance_snapshots/runs/N16x40_nsh1_perfect_correction/'
                 'run_3301a140f933/batch_00000_snapshots.npy')
STEM = 'modular_charge_chirality_cycle50_nsh1_y10'
TIMES = np.linspace(0, 32, 3201)
SNAPSHOTS = (0., .5, 1.)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def reduce_cut(job):
    sample, y0 = job
    raw = np.load(SOURCE, mmap_mode='r')
    assert raw.shape == (10, 4, 1280, 1280)
    idx = np.asarray([o + 2*x + 32*((y0+y) % 40)
                      for y in range(20) for x in range(16) for o in range(2)])
    g = raw[sample, 3][np.ix_(idx, idx)]
    vals, vecs = np.linalg.eigh((g + g.conj().T)/2)
    hvals = -2*np.arctanh(np.clip(vals, -1+1e-10, 1-1e-10))
    # Batch time/source columns through BLAS without storing a full density history.
    occupied = [o + 2*x + 32*10 for x in (5, 11) for o in (0, 1)]
    coefficients = vecs[occupied].conj().T
    full_paths = np.empty((2, len(TIMES)))
    window_paths = np.empty_like(full_paths)
    drift = np.zeros(2)
    for start in range(0, len(TIMES), 32):
        stop = min(start+32, len(TIMES))
        phase = np.exp(-1j*hvals[:, None]*TIMES[None, start:stop])
        amplitudes = vecs @ (phase[:, :, None]*coefficients[:, None, :]).reshape(640, -1)
        probability = abs(amplitudes.reshape(20, 16, 2, stop-start, 2, 2))**2
        cells = probability.sum(axis=(2, 5))  # y, x, time, packet
        for p, x in enumerate((5, 11)):
            density = cells[:, :, :, p]
            charge = density.sum(axis=(0, 1))
            full_paths[p, start:stop] = np.einsum('yxt,y->t', density, np.arange(20))/charge
            xx = np.arange(16)
            mask = np.minimum(abs(xx-x), 16-abs(xx-x)) <= 2
            window = density[:, mask]
            window_paths[p, start:stop] = np.einsum('yxt,y->t', window, np.arange(20))/window.sum(axis=(0, 1))
            drift[p] = max(drift[p], float(np.max(abs(charge-2))))
    if drift.max() > 1e-10:
        raise RuntimeError(f'Packet charge conservation failed: {drift}')
    full_paths -= full_paths[:, [0]]
    window_paths -= window_paths[:, [0]]
    return sample, y0, full_paths, window_paths, drift


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    FIGURES.mkdir(exist_ok=True)
    cache = FIGURES / f'{STEM}_data.npz'
    identity = dict(source_sha256=digest(SOURCE), generator_sha256=digest(MODULAR_SOURCE),
                    schema='midpoint_packet_v1', source_y_rel=10, curve_stop=32,
                    eps=1e-10, Nx=16, Ny=40, cycle=50)
    if cache.exists() and not args.recompute:
        with np.load(cache) as f:
            if json.loads(str(f['identity_json'])) != identity:
                raise RuntimeError('Cached data identity mismatch; use --recompute.')
            paths, window_paths = f['paths'], f['window_paths']
            drift = f['drift']
    else:
        paths = np.empty((10, 40, 2, len(TIMES)))
        window_paths = np.empty_like(paths)
        drift = np.empty((10, 40, 2))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [(s, y) for s in range(10) for y in range(40)]
            for s, y, p, wp, d in tqdm(pool.map(reduce_cut, jobs), total=400,
                                       desc='Recompute y=10 packets', unit='cut'):
                paths[s, y], window_paths[s, y], drift[s, y] = p, wp, d
        np.savez_compressed(cache, paths=paths, window_paths=window_paths,
                            drift=drift, times=TIMES, identity_json=json.dumps(identity))
    # Preserve the original spatial-panel estimator (propagate the saved mean K).
    with np.load(MODULAR_SOURCE) as f:
        ci = int(np.flatnonzero(f['snapshot_cycles'] == 50)[0])
        eig, vec = f['avg_h_vals'][ci], f['avg_h_vecs'][ci]
    densities = np.asarray([_packet_density(eig, vec, nx=16, ny_sub=20,
                            x=x, y=10, times=SNAPSHOTS) for x in (5, 11)])
    # Direct reproduction of a saved corner curve confirms radius-2 wall COM;
    # the legacy metadata's full-subsystem label does not match its raw arrays.
    trajectories = window_paths.mean(axis=1)
    mean = trajectories.mean(axis=0)
    sem = trajectories.std(axis=0, ddof=1)/np.sqrt(10)
    _style()
    plt.rcParams.update({'font.family': 'sans-serif',
                        'font.sans-serif': ['CMU Sans Serif', 'DejaVu Sans'],
                        'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8})
    fig = plt.figure(figsize=(7.05, 2.8))
    grid = fig.add_gridspec(1, 2, width_ratios=(.92, 1.55), wspace=.36)
    spatial, curve = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    xx, yy = np.meshgrid(np.arange(16), np.arange(20), indexing='ij')
    colors = ('#332288', '#E69F00', '#009E73')
    for packet in range(2):
        for ti, color in enumerate(colors):
            density = densities[packet, ti]
            mask = density > 1e-4
            spatial.scatter(xx[mask], yy[mask], s=105*np.sqrt(density[mask]/2),
                            facecolors=color, edgecolors=color, linewidths=.25, zorder=5-ti)
    for x in (5, 11):
        spatial.axvline(x, color='.35', ls=':', lw=.85, zorder=1)
    spatial.set(xlim=(-.5, 15.5), ylim=(-.5, 19.5), xlabel='$x$', ylabel='$y-y_0$')
    spatial.set_aspect('equal')
    spatial.set_xticks([0, 5, 11, 15]); spatial.set_yticks([0, 5, 10, 15, 19])
    handles = [Line2D([], [], marker='o', ls='', color=c, label=f'{t:g}', markersize=4)
               for t, c in zip(SNAPSHOTS, colors)]
    spatial.legend(handles=handles, title=r'$t_{\rm mod}$', loc='upper left',
                   frameon=False, handletextpad=.3, handlelength=.7)
    for i, (x, color, ls) in enumerate(((5, '#D55E00', '-'), (11, '#0072B2', '--'))):
        curve.plot(TIMES, mean[i], color=color, ls=ls, lw=1.2, label=f'$({x},10)$')
        curve.fill_between(TIMES, mean[i]-sem[i], mean[i]+sem[i], color=color, alpha=.15, lw=0)
    curve.axhline(0, color='.35', lw=.7)
    curve.set(xlim=(0, 32), xlabel=r'modular time $t_{\rm mod}$', ylabel=r'$\langle\Delta y\rangle$')
    curve.set_xticks((0, 8, 16, 24, 32))
    curve.legend(frameon=False, ncol=2)
    for ax, label in ((spatial, '(a)'), (curve, '(b)')):
        ax.text(-.17, 1.04, label, transform=ax.transAxes, fontweight='bold')
    fig.subplots_adjust(left=.07, right=.985, bottom=.19, top=.91)
    for suffix in ('pdf', 'png'):
        fig.savefig(FIGURES/f'{STEM}.{suffix}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    summary = dict(identity, source=str(SOURCE.relative_to(ROOT)),
                   snapshots=list(SNAPSHOTS), source_positions=[[5,10],[11,10]],
                   samples=10, translated_cuts=40, max_charge_error=float(drift.max()),
                   final_mean=mean[:,-1].tolist(), final_sem=sem[:,-1].tolist(),
                   spatial_estimator='propagate saved sample/cut-averaged modular generator',
                   curve_estimator='radius-2 wall-window COM; mean cuts within trajectory, then mean +/- SEM over trajectories',
                   legacy_estimator_check='radius-2 reproduces the saved corner curve; legacy full-subsystem metadata is inaccurate',
                   script_sha256=digest(__file__))
    (FIGURES/f'{STEM}_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    caption = ('Modular charge propagation with both packets initialized at y-y0=10. '
               'Cycle-50 hard-wall 16 x 40 ensemble, nshell=1, alpha1=1, alpha2=30, '
               'S=10 independent pure half-filled trajectories, raster-y, perfect correction. '
               'Panel (a): independent two-orbital packets at (5,10) and (11,10), '
               'propagated with the saved mean modular generator at tmod=0,0.5,1 '
               '(purple, orange, green); marker area proportional to sqrt(local density). '
               'Panel (b): recomputed radius-2 wall-window COM displacement, averaging 40 '
               'translated cuts within each trajectory before ensemble mean +/- SEM. '
               'Spectral clipping eps=1e-10; no velocity fit. Modular time is not circuit time.\n')
    (FIGURES/f'{STEM}_caption.txt').write_text(caption)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
