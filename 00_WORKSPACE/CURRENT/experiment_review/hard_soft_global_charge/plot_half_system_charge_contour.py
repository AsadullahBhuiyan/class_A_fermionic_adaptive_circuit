"""Intrinsic half-strip charge-variance contours from pure endpoint frames."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from analyze_pure_endpoints import HERE, PUMP
from analyze_charge_origin import verify


def contour(frame, rank, nx, ny):
    """Return sum_orbital diag(G_A-G_A^2), without averaging states."""
    u = np.asarray(frame[:, :rank], dtype=np.complex128)
    gram_error = float(np.max(abs(u.conj().T @ u - np.eye(rank))))
    if gram_error > 1e-8:
        raise ValueError(f'Endpoint is not an orthonormal occupied frame: {gram_error}')
    # Canonical mode ordering is (y, x, orbital); A includes y=0,...,Ny/2-1.
    cut = nx * ny
    a, b = u[:cut], u[cut:]
    cross = a @ b.conj().T
    local = np.sum(abs(cross)**2, axis=1)
    # For a pure projector this equals the conventional restricted-covariance
    # expression. Check it independently for every trajectory.
    ga = a @ a.conj().T
    direct = ga.diagonal().real - np.sum(abs(ga)**2, axis=1)
    error = float(np.max(abs(direct-local)))
    if error > 1e-8:
        raise ValueError(f'Charge-contour identity failed: {error}')
    return local.reshape(ny//2, nx, 2).sum(axis=2), gram_error, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shell', choices=('dense', 'nsh1'), default='dense')
    args = parser.parse_args()
    nx, ny = 32, 24
    out = HERE / 'outputs' / f'half_system_charge_contour_n32x24_{args.shell}'
    out.mkdir(parents=True, exist_ok=True)
    sources, maps, diagnostics = [], {'hard': {}, 'soft': {}}, []
    receipts = []
    for receipt in sorted((PUMP/'endpoints').rglob('*.completion.json')):
        meta = json.loads(receipt.read_text())
        if meta['Nx'] == nx and meta['Ny'] == ny and meta['protocol'] == args.shell:
            receipts.append(receipt)
    assert len(receipts) == 40
    for receipt in tqdm(receipts, desc=f'Endpoint contours ({args.shell})', unit='shard'):
        path, meta = verify(receipt)
        assert meta['status'] == 'complete'
        sources.append({'path': str(path), 'sha256': meta['result_sha256']})
        with np.load(path, allow_pickle=False) as z:
            assert int(z['cycles_total']) == 48
            assert float(z['alpha_1']) == 1 and float(z['alpha_2']) == 30
            assert np.array_equal(z['wall_locations'], [8,24])
            wall = str(z['wall'])
            frames = z['frames']
            for index, sid in enumerate(z['sample_ids']):
                sid = int(sid)
                assert sid not in maps[wall]
                values, gram, error = contour(frames[index], int(z['ranks'][index]), nx, ny)
                maps[wall][sid] = values
                diagnostics.append({'wall': wall, 'sample_id': sid,
                    'gram_max_error': gram, 'contour_identity_max_error': error})
    arrays, summary = {}, {}
    for wall, records in maps.items():
        assert sorted(records) == list(range(100))
        values = np.stack([records[i] for i in range(100)])
        arrays[wall+'_per_trajectory'] = values
        arrays[wall+'_mean'] = values.mean(axis=0)
        arrays[wall+'_sem'] = values.std(axis=0, ddof=1)/10
        total = values.sum(axis=(1,2))
        summary[wall] = {'samples': 100, 'half_system_variance_mean': float(total.mean()),
                         'half_system_variance_sem': float(total.std(ddof=1)/10)}
    np.savez_compressed(out/'charge_contours.npz', **arrays,
                        x=np.arange(nx), y=np.arange(ny//2), sample_ids=np.arange(100))
    plt.rcParams.update({'font.family': 'CMU Sans Serif', 'font.size': 9,
                         'xtick.direction': 'in', 'ytick.direction': 'in'})
    fig, axes = plt.subplots(1,2,figsize=(7.05,2.8), layout='constrained', sharey=True)
    vmax = max(arrays[wall+'_mean'].max() for wall in maps)
    for i, (ax, wall) in enumerate(zip(axes, ('soft','hard'))):
        im=ax.imshow(arrays[wall+'_mean'], origin='lower', interpolation='nearest',
                     extent=(-.5,nx-.5,-.5,ny//2-.5), vmin=0, vmax=vmax,
                     cmap='magma', aspect='equal')
        for x in (8,24): ax.axvline(x, color='cyan', linestyle='--', linewidth=.65)
        ax.set(xlabel=r'$x$', title=wall.capitalize()+' wall')
        ax.set_xticks([0,8,16,24,31]); ax.set_yticks([0,4,8,11])
        ax.tick_params(top=True,right=True)
        ax.text(-.12,1.08,f'({chr(97+i)})',transform=ax.transAxes)
    axes[0].set_ylabel(r'$y$ within half-system $A$')
    fig.colorbar(im, ax=axes, shrink=.85, label=r'$\overline{f_A(x,y)}$')
    for ext in ('pdf','png'): fig.savefig(out/f'charge_variance_heatmaps.{ext}',dpi=300)
    caption = (
        f'Trajectory-averaged intrinsic half-system charge-variance contour, Nx=32, Ny=24, '
        f'OW support={args.shell}, alpha_1=1, alpha_2=30, endpoint cycle 48. '
        'A contains all x and y=0,...,11; each pixel sums both orbitals. '
        'S=100 independent pure-state trajectories per wall protocol; default half-filled '
        'initialization, raster_y, perfect correction, complex128. '
        'Hard walls include Born-conditioned exterior preparation and slab-only dynamics; '
        'soft walls update everywhere with untruncated support. Dashed lines mark x=8,24. '
        'For each trajectory f_A(i)=[G_A-G_A^2]_{ii}=sum_{j outside A}|G_ij|^2. '
        'Compute the nonlinear contour before averaging trajectories; its spatial sum is '
        'the intrinsic quantum Var(N_A), not the across-trajectory variance of mean charge. '
        'The panels share one linear color scale; no smoothing or origin averaging is applied. '
        'Pointwise trajectory SEMs and all individual maps are saved in the NPZ; scalar '
        'uncertainties are trajectory SEMs. No fit is performed.\n')
    (out/'caption.txt').write_text(caption)
    (out/'summary.json').write_text(json.dumps({'parameters': {'Nx':nx,'Ny':ny,'shell':args.shell,
        'alpha_1':1,'alpha_2':30,'cycles':48,'subsystem':'all x, y=0..11'},
        'results':summary,'sources':sources,'diagnostics':diagnostics,
        'estimator_order':'individual pure-state contour, then trajectory mean'},indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
