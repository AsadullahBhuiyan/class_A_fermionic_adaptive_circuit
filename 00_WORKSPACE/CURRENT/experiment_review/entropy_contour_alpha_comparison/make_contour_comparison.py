#!/usr/bin/env python3
"""Matched half-strip entanglement contours from Campaign 09 pure endpoints."""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
from scipy.linalg import eigh, svdvals
from scipy.special import xlogy
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REVISION = 'pure_tangent_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1'
SOURCE = ROOT / '00_WORKSPACE/CURRENT/final_production_new_designs/09_pure_tangent_replay_acquisition/gpu_data' / REVISION
STEM = HERE / 'figures/entropy_contours_hard_n20x32_alpha1_1_3'
DATA = HERE / 'contours.npz'
MANIFEST = HERE / 'analysis_manifest.json'
NX, NY, AY = 20, 32, 16


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def entropy_weights(nu):
    if not np.isfinite(nu).all() or nu.min() < -1e-8 or nu.max() > 1 + 1e-8:
        raise ValueError('Nonphysical restricted occupations')
    nu = np.clip(nu, 0, 1)  # Roundoff only, after checking; no entropy floor.
    return -xlogy(nu, nu) - xlogy(1 - nu, 1 - nu)


def contour(frame, rank, nx, ny, check_svd=False, y0=0, check_gram=True):
    occupied = np.asarray(frame[:, :rank], dtype=np.complex128)
    gram_error = float(np.max(abs(occupied.conj().T @ occupied - np.eye(rank)))) if check_gram else 0.
    assert gram_error < 1e-8
    # Canonical indexing is (y,x,orbital). Keep rows ordered by relative dy,
    # including cuts that wrap through the periodic boundary.
    assert ny % 2 == 0 and 0 <= y0 < ny
    rows = (((y0 + np.arange(ny//2)) % ny)[:, None] * (2*nx)
            + np.arange(2*nx)[None, :]).reshape(-1)
    restricted = occupied[rows]
    correlation = restricted @ restricted.conj().T
    nu, vectors = eigh((correlation + correlation.conj().T) / 2,
                       driver='evd', check_finite=True)
    weights = entropy_weights(nu)
    values = ((abs(vectors)**2) @ weights).reshape(ny//2, nx, 2).sum(axis=2)
    entropy = float(weights.sum())
    closure = abs(float(values.sum()) - entropy)
    assert closure < 1e-10 and np.isfinite(values).all() and values.min() >= 0
    svd_error = None
    if check_svd:
        independent = entropy_weights(svdvals(restricted)**2).sum()
        svd_error = abs(float(independent) - entropy)
        assert svd_error < 1e-8
    return values, entropy, dict(gram_error=gram_error, closure=closure,
        occupation_min=float(nu.min()), occupation_max=float(nu.max()), svd_error=svd_error)


def origin_averaged_contour(frame, rank, nx, ny, check_svd=False):
    total = np.zeros((ny//2, nx), dtype=np.float64)
    entropies, diagnostics = [], []
    for y0 in range(ny):
        values, entropy, diag = contour(frame, rank, nx, ny,
            check_svd=check_svd and y0 in (0, ny-1), y0=y0, check_gram=(y0==0))
        total += values
        entropies.append(entropy)
        diagnostics.append(diag)
    total /= ny
    entropy = float(np.mean(entropies))
    assert abs(total.sum()-entropy) < 1e-10
    diag = dict(gram_error=diagnostics[0]['gram_error'],
        closure=max(d['closure'] for d in diagnostics),
        origin_mean_closure=float(abs(total.sum()-entropy)),
        occupation_min=min(d['occupation_min'] for d in diagnostics),
        occupation_max=max(d['occupation_max'] for d in diagnostics),
        origin_count=ny, svd_error=max((d['svd_error'] for d in diagnostics
                                      if d['svd_error'] is not None), default=None))
    return total, entropy, diag


def compute(average_origins=False, workers=8):
    maps, entropies, sources, diagnostics = {}, {}, [], []
    for alpha in (1, 3):
        folder = SOURCE / f'hard/Ny032/alpha1_{alpha}'
        files = sorted(folder.glob('*.npz'))
        assert len(files) == 4
        records, totals = {}, {}
        for path in tqdm(files, desc=f'alpha1={alpha}: verify/read batches', unit='batch'):
            receipt_path = path.with_suffix('.complete.json')
            meta = json.loads(receipt_path.read_text())
            for key, value in dict(Nx=NX, Ny=NY, alpha_1=float(alpha), alpha_2=30.,
                                   cycles=64, nshell=1, construction='hard', status='complete',
                                   sampling_revision=REVISION).items():
                assert meta[key] == value, (path, key)
            assert meta['result_filename'] == path.name
            assert path.stat().st_size == meta['result_bytes']
            assert digest(path) == meta['result_sha256']
            sources.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                sha256=meta['result_sha256'], receipt_sha256=digest(receipt_path),
                source_hashes=meta['source_hashes']))
            with np.load(path, allow_pickle=False) as z:
                frames, ranks = z['final_frame'], z['final_ranks']
                ids = z['case_sample_indices']
                np.testing.assert_array_equal(ids, meta['case_sample_indices'])
                assert frames.dtype == np.complex128 and frames.shape[1] == 2*NX*NY
                assert int(z['cycles_total']) == 64
                assert str(z['sequence']) == 'raster_y'
                assert str(z['configuration_sha256']) == meta['configuration_sha256']
                assert json.loads(str(z['source_hashes_json'])) == meta['source_hashes']
                if average_origins:
                    with ProcessPoolExecutor(max_workers=workers) as pool:
                        futures = {pool.submit(origin_averaged_contour, frames[i], int(ranks[i]),
                            NX, NY, check_svd=(int(sid)==0)): int(sid) for i, sid in enumerate(ids)}
                        for future in tqdm(as_completed(futures), total=len(ids),
                                desc='32 cuts per trajectory', leave=False, unit='sample'):
                            sid = futures[future]
                            assert sid not in records
                            values, total, diag = future.result()
                            records[sid], totals[sid] = values, total
                            diagnostics.append(dict(alpha_1=alpha, sample_id=sid, **diag))
                else:
                    for i, sid in enumerate(tqdm(ids, desc='Half-strip contours', leave=False, unit='sample')):
                        sid = int(sid)
                        assert sid not in records
                        values, total, diag = contour(frames[i], int(ranks[i]), NX, NY,
                                                       check_svd=(sid == 0))
                        records[sid], totals[sid] = values, total
                        diagnostics.append(dict(alpha_1=alpha, sample_id=sid, **diag))
            del frames
        assert sorted(records) == list(range(100))
        maps[f'alpha1_{alpha}'] = np.stack([records[i] for i in range(100)])
        entropies[f'alpha1_{alpha}'] = np.array([totals[i] for i in range(100)])
    arrays, summary = {}, {}
    for key, values in maps.items():
        arrays[key+'_per_sample'] = values
        arrays[key+'_mean'] = values.mean(axis=0)
        arrays[key+'_sem'] = values.std(axis=0, ddof=1) / 10
        arrays[key+'_entropy'] = entropies[key]
        summary[key] = dict(entropy_mean=float(entropies[key].mean()),
            entropy_sem=float(entropies[key].std(ddof=1)/10),
            peak_mean_contour=float(values.mean(axis=0).max()))
    np.savez_compressed(DATA, **arrays, x=np.arange(NX), y=np.arange(AY),
        y0_values=np.arange(NY) if average_origins else np.array([0]), sample_ids=np.arange(100))
    metadata = dict(schema='matched_alpha_entanglement_contours_v1', campaign=REVISION,
        Nx=NX, Ny=NY, alpha_1=[1,3], alpha_2=30, nshell=1, construction='hard',
        endpoint_cycle=64, samples_per_alpha=100, sequence='raster_y',
        initial_state='pure half-filled random, with Born-conditioned hard-wall exterior',
        subsystem='all x, Ay=16 consecutive periodic y rows; both orbitals',
        origin_average=average_origins, origin_count=NY if average_origins else 1,
        y_coordinate='relative dy=(y-y0) mod Ny' if average_origins else 'absolute y, y0=0',
        estimator='individual restricted-projector von Neumann contour, then origin mean within trajectory, then trajectory mean',
        entropy_log_base='natural', entropy_endpoint_floor=None,
        sem='SD across 100 independent trajectories / sqrt(100); no bootstrap',
        sources=sources, diagnostics=sorted(diagnostics,key=lambda d:(d['alpha_1'],d['sample_id'])),
        summary=summary, data_sha256=digest(DATA))
    MANIFEST.write_text(json.dumps(metadata, indent=2)+'\n')
    return arrays, metadata


def plot(arrays, metadata):
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['CMU Sans Serif','DejaVu Sans'],
        'mathtext.fontset':'cm', 'font.size':8, 'axes.labelsize':8,
        'xtick.labelsize':7, 'ytick.labelsize':7, 'xtick.direction':'in', 'ytick.direction':'in',
        'axes.linewidth':.7, 'pdf.fonttype':42})
    fig, axes = plt.subplots(1, 2, figsize=(3.375, 2.25), sharey=True)
    fig.subplots_adjust(left=.115, right=.98, bottom=.32, top=.86, wspace=.22)
    vmax = max(arrays[f'alpha1_{a}_mean'].max() for a in (1,3))
    for i, (ax, alpha) in enumerate(zip(axes, (1,3))):
        im = ax.imshow(arrays[f'alpha1_{alpha}_mean'], origin='lower', interpolation='nearest',
            extent=(-.5,NX-.5,-.5,AY-.5), aspect='equal', cmap='Blues', norm=PowerNorm(.5, vmin=0, vmax=vmax))
        ax.set(xlabel=r'$x$', title=rf'$\alpha_1={alpha}$', xlim=(-.5,NX-.5), ylim=(-.5,AY-.5))
        ax.set_xticks([0,5,10,15]); ax.set_yticks([0,5,10,15])
        for x in np.arange(-.5,NX,1): ax.axvline(x,color='.5',alpha=.28,linewidth=.22)
        for y in np.arange(-.5,AY,1): ax.axhline(y,color='.5',alpha=.28,linewidth=.22)
        ax.tick_params(top=True,right=True,length=2.5,pad=2)
        ax.text(-.16,1.08,f'({chr(97+i)})',transform=ax.transAxes)
    axes[0].set_ylabel(r'$\delta y$' if metadata['origin_average'] else r'$y$')
    cax=fig.add_axes([.23,.17,.65,.035])
    cb=fig.colorbar(im,cax=cax,orientation='horizontal')
    cb.set_label(r'$\langle\overline{s}_1(x,\delta y)\rangle_\xi$' if metadata['origin_average']
                 else r'$\langle s_1(x,y)\rangle_\xi$',labelpad=2)
    cb.set_ticks([0,.1,.3,round(float(vmax),2)])
    cb.ax.tick_params(labelsize=6,length=2,pad=1)
    fig.canvas.draw()
    for ax in (*axes,cax):
        b=ax.get_tightbbox(fig.canvas.get_renderer())
        assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height, (b,fig.bbox)
    STEM.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.pdf','.png'): fig.savefig(STEM.with_suffix(suffix),dpi=300)
    plt.close(fig)
    metadata['figure']=dict(size_inches=[3.375,2.25],colormap='Blues',gamma=.5,
        shared_color_scale=True,vmin=0,vmax=float(vmax),unit_cell_grid=True)
    metadata['outputs']={p.name:digest(p) for p in (DATA,STEM.with_suffix('.pdf'),STEM.with_suffix('.png'))}
    metadata['script_sha256']=digest(Path(__file__))
    MANIFEST.write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata['summary'],indent=2),flush=True)
    print(STEM.with_suffix('.pdf'),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plot-only',action='store_true')
    parser.add_argument('--average-origins',action='store_true')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.average_origins:
        STEM = HERE / 'figures/entropy_contours_hard_n20x32_alpha1_1_3_y0avg'
        DATA = HERE / 'contours_y0avg.npz'
        MANIFEST = HERE / 'analysis_manifest_y0avg.json'
    if args.plot_only:
        metadata=json.loads(MANIFEST.read_text())
        assert digest(DATA)==metadata['data_sha256']
        with np.load(DATA,allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
    else:
        arrays,metadata=compute(average_origins=args.average_origins,workers=args.workers)
    plot(arrays,metadata)
