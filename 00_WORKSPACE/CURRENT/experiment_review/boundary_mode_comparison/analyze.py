#!/usr/bin/env python3
"""Trajectory-first boundary profiles: mixed endpoints and pure-state reductions."""
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OLD = HERE.parent / 'purification_slow_mode_profiles'
RAW = REPO / ('00_WORKSPACE/CURRENT/final_production_new_designs/'
              '09_pure_tangent_replay_acquisition/gpu_data/'
              'pure_tangent_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1')
CAP = 1e-9
WALL_X = np.array([4, 5, 6, 14, 15, 16])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def density(v, indices, ny, nx=20):
    full = np.zeros((2 * nx * ny, v.shape[1]))
    full[indices] = abs(v)**2
    return full.reshape(ny, nx, 2, -1).sum(axis=2).transpose(2, 0, 1)


def nearest_mask(energy, count):
    if not len(energy): return np.zeros(0, dtype=bool)
    edge = np.sort(abs(energy))[min(count, len(energy)) - 1]
    # Include ties instead of imposing an arbitrary vector basis at a boundary.
    return abs(energy) <= edge + 1e-8


def extract(frame, ny, nx=20):
    """Reduce to all x and first half y; no full covariance is constructed."""
    if frame.dtype != np.complex128 or frame.shape[0] != 2 * nx * ny:
        raise ValueError('unexpected frame shape/dtype')
    gram = float(np.max(abs(frame.conj().T @ frame - np.eye(frame.shape[1]))))
    if gram > 1e-8: raise ValueError(f'nonorthonormal occupied frame: {gram}')
    indices = np.arange(nx * ny)
    fa = frame[indices]
    g = fa @ fa.conj().T
    nu, u = eigh(g, driver='evd')
    if nu.min() < -1e-8 or nu.max() > 1 + 1e-8: raise ValueError('nonphysical spectrum')
    finite = (nu > CAP) & (nu < 1 - CAP)
    eps = np.full(nu.size, np.nan)
    eps[finite] = np.log1p(-nu[finite]) - np.log(nu[finite])
    ids = np.flatnonzero(finite)
    ids = ids[np.argsort(abs(eps[ids]), kind='stable')]
    keep = ids[nearest_mask(eps[ids], 16)]
    low = nearest_mask(eps[keep], 4)
    v = u[:, keep].copy()
    if len(keep):
        phases = v[np.argmax(abs(v), axis=0), np.arange(len(keep))]
        v *= (phases.conj() / abs(phases))[None, :]
    residual = np.linalg.norm(g @ v - v * nu[keep], axis=0)
    if residual.max(initial=0) > 1e-9: raise ValueError('eigenvector residual')
    xy = density(v, indices, ny, nx)
    np.testing.assert_allclose(xy.sum(axis=(1, 2)), 1, atol=1e-9)
    # Full-spectrum boundary diagnostics do not require persisting all vectors.
    full_xy = density(u, indices, ny, nx)
    cut_y = np.array([0, 1, ny//2 - 2, ny//2 - 1])
    clusters = np.r_[0, np.cumsum(np.diff(nu) > 1e-10)]
    sizes = np.bincount(clusters)
    return dict(occupations=nu, modular_energies=eps, finite_mask=finite,
                selected_indices=keep, selected_energies=eps[keep], selected_vectors=v,
                selected_xy=xy, low_mask=low, subsystem_indices=indices,
                selected_cluster_sizes=sizes[clusters[keep]],
                selected_individually_separated=sizes[clusters[keep]] == 1,
                all_mode_wall_weights=full_xy[:, :, WALL_X].sum(axis=(1, 2)),
                all_mode_cut_weights=full_xy[:, cut_y, :].sum(axis=(1, 2)),
                frame_gram_error=np.array(gram), eigen_residuals=residual,
                cap=np.array(CAP), Nx=np.array(nx), Ny=np.array(ny))


def row_from(result, dataset, wall, ny, alpha, sample, filename, source_sha):
    xy = result['selected_xy'][result['low_mask']]
    count = len(xy)
    p = xy.mean(axis=0) if count else np.full((ny, 20), np.nan)
    px = p.sum(axis=0)
    cut = [0, 1, ny//2 - 2, ny//2 - 1] if dataset == 'slot09' else []
    return dict(dataset=dataset, wall=wall, Ny=ny, alpha1=alpha, sample=int(sample),
                finite_count=int(result['finite_mask'].sum()), selected_count=count,
                individually_separated=bool(np.all(result['selected_individually_separated'][result['low_mask']])),
                wall_weight=float(px[WALL_X].sum()),
                cut_weight=float(p[cut].sum()) if cut else None,
                min_abs_energy=float(abs(result['selected_energies']).min(initial=np.inf)),
                profile=px.tolist(), file=str(filename.relative_to(HERE)),
                source_sha256=source_sha, output_sha256=sha(filename))


def pure_batch(path):
    path = Path(path)
    receipt = json.loads(path.with_suffix('.complete.json').read_text())
    if (receipt['status'] != 'complete' or receipt['result_filename'] != path.name
            or receipt['result_bytes'] != path.stat().st_size or sha(path) != receipt['result_sha256']):
        raise ValueError(f'invalid acquisition pair: {path}')
    rows = []
    with np.load(path, allow_pickle=False) as z, threadpool_limits(limits=2):
        ny, wall, alpha = int(z['Ny']), str(z['construction']), float(z['alpha_1'])
        for key in ['Ny', 'Nx', 'alpha_1', 'alpha_2', 'nshell', 'construction', 'task_id', 'configuration_sha256']:
            if z[key].item() != receipt[key]: raise ValueError(f'identity mismatch {key}')
        if int(z['cycles_total']) != 2*ny or str(z['sequence']) != 'raster_y':
            raise ValueError('unexpected dynamics contract')
        np.testing.assert_array_equal(z['case_sample_indices'], receipt['case_sample_indices'])
        frames, ranks, samples = z['final_frame'], z['final_ranks'], z['case_sample_indices']
        for i, sample in enumerate(samples):
            r = extract(frames[i, :, :int(ranks[i])], ny)
            r.update(alpha1=np.array(alpha), wall=np.array(wall), sample=np.array(sample),
                     final_rank=ranks[i], cycles=np.array(2*ny),
                     source_file=np.array(str(path.relative_to(REPO))),
                     source_sha256=np.array(receipt['result_sha256']),
                     source_receipt_json=np.array(json.dumps(receipt, sort_keys=True)))
            out = HERE / 'modes' / 'slot09' / wall / f'Ny{ny:03d}' / f'alpha1_{alpha:g}' / f'sample_{sample:03d}.npz'
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix('.partial.npz')
            np.savez_compressed(tmp, **r); os.replace(tmp, out)
            rows.append(row_from(r, 'slot09', wall, ny, alpha, sample, out, receipt['result_sha256']))
    return rows


def mixed_rows():
    manifest = json.loads((OLD / 'analysis_manifest.json').read_text())
    rows = []
    for entry in tqdm(manifest['outputs'], desc='Verify slot-07 eigenmodes', unit='sample'):
        path = OLD / entry['result_file']
        if sha(path) != entry['result_sha256']: raise ValueError('slot07 extraction checksum')
        with np.load(path, allow_pickle=False) as z:
            ny, wall, sample = int(z['Ny']), str(z['construction']), int(z['sample_index'])
            v, inds = z['finite_eigenvectors'], z['active_indices']
            # Slot07 has at most four finite modes: use entire resolved subspace.
            r = dict(selected_xy=density(v, inds, ny), low_mask=np.ones(v.shape[1], bool),
                     finite_mask=z['finite_mask'], selected_energies=z['finite_signed_rates']*2*int(z['cycles']),
                     selected_individually_separated=z['finite_mode_individually_separated'],
                     selected_vectors=v, subsystem_indices=inds, Ny=np.array(ny),
                     source_file=np.array(str(path.relative_to(REPO))))
            out = HERE / 'modes' / 'slot07' / wall / f'Ny{ny:03d}' / f'sample_{sample:03d}.npz'
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, **r)
            rows.append(row_from(r, 'slot07', wall, ny, 1., sample, out, entry['result_sha256']))
    return rows


def summarize(rows):
    summary = []
    for key in sorted({(r['dataset'], r['wall'], r['Ny'], r['alpha1']) for r in rows}):
        group = [r for r in rows if (r['dataset'], r['wall'], r['Ny'], r['alpha1']) == key]
        if sorted(r['sample'] for r in group) != list(range(100)): raise ValueError(f'coverage {key}')
        valid = [r for r in group if r['selected_count']]
        weights = np.array([r['wall_weight'] for r in valid])
        profiles = np.array([r['profile'] for r in valid])
        s = dict(zip(['dataset', 'wall', 'Ny', 'alpha1'], key))
        s.update(samples=100, profile_samples=len(valid),
                 finite_count_mean=float(np.mean([r['finite_count'] for r in group])),
                 wall_weight_mean=float(weights.mean()), wall_weight_sem=float(weights.std(ddof=1)/np.sqrt(len(valid))),
                 wall_weight_min=float(weights.min()), wall_weight_max=float(weights.max()),
                 samples_wall_weight_over_0p8=int((weights > .8).sum()),
                 profile_mean=profiles.mean(axis=0).tolist(),
                 profile_sem=(profiles.std(axis=0, ddof=1)/np.sqrt(len(valid))).tolist(),
                 cut_weight_mean=float(np.mean([r['cut_weight'] for r in valid])) if key[0] == 'slot09' else None)
        # Deterministic representative: closest to median wall weight, not most localized.
        rep = min(valid, key=lambda r: (abs(r['wall_weight'] - np.median(weights)), r['sample']))
        s['representative_file'], s['representative_sample'] = rep['file'], rep['sample']
        summary.append(s)
    return summary


def figures(summary):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    os.environ['TEXINPUTS'] = str(HERE.parent / 'hard_wall_tangent_gap_analysis/latex_support') + os.pathsep + os.environ.get('TEXINPUTS', '')
    mpl.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['CMU Sans Serif'], 'font.size':8,
        'text.usetex':True, 'text.latex.preamble':r'\usepackage{amsmath}\renewcommand{\familydefault}{\sfdefault}',
        'xtick.direction':'in', 'ytick.direction':'in', 'xtick.top':True, 'ytick.right':True, 'legend.frameon':False})
    out = HERE / 'figures'; out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.5), layout='constrained')
    heat, hx = plt.subplots(2, 3, figsize=(7.05, 4.7), layout='constrained')
    for row, wall in enumerate(['hard', 'soft']):
        for col, (dataset, alpha) in enumerate([('slot07', 1.), ('slot09', 1.), ('slot09', 3.)]):
            ax = axes[row, col]
            cases = [s for s in summary if (s['dataset'], s['wall'], s['alpha1']) == (dataset, wall, alpha)]
            for s, color, marker, ls in zip(cases, ['#d62728','#2ca02c','#1f77b4'], ['^','s','o'], [':','--','-']):
                ax.errorbar(range(20), s['profile_mean'], yerr=s['profile_sem'], color=color, marker=marker,
                    linestyle=ls, linewidth=.8, markersize=2, capsize=1,
                    label=rf"${s['Ny']}$, $S={s['profile_samples']}$")
            title = f"{dataset.replace('slot','Slot ')} {wall}" + (rf", $\alpha_1={alpha:g}$" if col else '')
            ax.set(title=title, xlabel='$x$', ylabel='$p(x)$' if col == 0 else '', xlim=(0,19), ylim=(0,.65), xticks=[0,5,10,15,19])
            ax.legend(fontsize=8, loc='upper center', title='$N_y$, contributing samples', title_fontsize=8)
            for x in [5,15]: ax.axvline(x, color='.6', ls='--', lw=.6)
            ax.text(-.1,1.05,f'({chr(97+row*3+col)})',transform=ax.transAxes)
            s = cases[-1]
            with np.load(HERE / s['representative_file']) as z:
                # Show the slowest single mode, only if individually separated.
                if not bool(z['selected_individually_separated'][0]):
                    raise ValueError('representative single mode is degenerate')
                im = z['selected_xy'][0]
                ny = s['Ny'] if dataset == 'slot07' else s['Ny']//2
                artist = hx[row,col].imshow(im[:ny], origin='lower', aspect='auto', cmap='magma', vmin=0,
                    extent=(-.5,19.5,-.5,ny-.5))
                hx[row,col].set(title=title+rf"\newline $N_y={s['Ny']}$, sample {s['representative_sample']}",
                    xlabel='$x$', ylabel='$y$' if col == 0 else '', xticks=[0,5,10,15,19])
                for x in [5,15]: hx[row,col].axvline(x,color='cyan',ls='--',lw=.5)
                heat.colorbar(artist, ax=hx[row,col], label=r'$\sum_\mu |u(x,y,\mu)|^2$', shrink=.85)
                hx[row,col].text(-.1,1.05,f'({chr(97+row*3+col)})',transform=hx[row,col].transAxes)
    for f, name in [(fig,'boundary_profiles'),(heat,'representative_boundary_modes')]:
        pdf = out / f'{name}.pdf'
        f.savefig(pdf); plt.close(f)
        subprocess.run(['pdftoppm','-png','-r','300','-singlefile',str(pdf),str(pdf.with_suffix(''))],check=True)


def main():
    paths = sorted(RAW.rglob('*.npz'))
    if len(paths) != 32: raise ValueError('expected 32 acquisition batches')
    print('Analyzing 1200 pure endpoints and 600 saved purification endpoints; no dynamics.',flush=True)
    rows = mixed_rows()
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(pure_batch, str(p)) for p in paths]
        for f in tqdm(as_completed(futures), total=len(futures), desc='Pure endpoint half-system modes', unit='batch'):
            rows.extend(f.result())
    rows.sort(key=lambda r: (r['dataset'],r['wall'],r['Ny'],r['alpha1'],r['sample']))
    summary = summarize(rows)
    (HERE / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (HERE / 'manifest.json').write_text(json.dumps(dict(script_sha256=sha(__file__), cap=CAP,
        half_system='all x, y=0,...,Ny/2-1, both orbitals',
        selection='slot07 all finite; slot09 four closest-to-zero modular energies including ties; retain at least sixteen closest vectors',
        samples=rows),indent=2)+'\n')
    with (HERE / 'summary.csv').open('w') as f:
        clean = [{k:v for k,v in s.items() if not k.startswith('profile_') or k=='profile_samples'} for s in summary]
        writer=csv.DictWriter(f,fieldnames=list(clean[0]));writer.writeheader();writer.writerows(clean)
    figures(summary)
    print(json.dumps([{k:v for k,v in s.items() if k not in ['profile_mean','profile_sem']} for s in summary],indent=2),flush=True)


if __name__ == '__main__': main()
