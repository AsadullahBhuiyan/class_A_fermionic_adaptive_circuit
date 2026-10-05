"""Offline, complete-S100 convergence and trajectory-averaged endpoint marker.

No dynamics or production/deployment files are changed. The lower panel is the
canonical position-commutator marker, not a map of finite-radius disk estimators.
"""
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from threadpoolctl import threadpool_limits
from tqdm import tqdm

import run_campaign as R
from random_center_observer import ensemble_statistics, sector_table

ROOT = Path(__file__).resolve().parent
REPO = next(p for p in ROOT.parents if (p / 'PROJECT_ADMIN/REPO_POLICY.md').exists())
DATA = ROOT / 'data' / R.REVISION
OUT = ROOT / 'analysis_outputs/convergence_and_endpoint_marker_S100'
sys.path.insert(0, str(REPO / 'src/fgtn'))
from classA_U1FGTN import classA_U1FGTN


def write_csv(name, rows):
    with (OUT / name).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def absolute_interval(mean, sem):
    delta = np.abs(mean - 1)
    lower = np.where(delta <= sem, 0, np.minimum(abs(mean-sem-1), abs(mean+sem-1)))
    upper = np.maximum(abs(mean-sem-1), abs(mean+sem-1))
    return delta, lower, upper


def endpoint_marker(frame, length, *, explicit_check=False):
    """2pi i diag(P X P Y P - P Y P X P), summed over two orbitals.

    U=V* so P=UU^dagger=(VV^dagger)^T. Projected coordinates allow the
    diagonal to be computed in occupied space, without dense triple products.
    Ordinary sawtooth coordinates are used; periodic-cut effects are retained.
    """
    u = frame.conj()
    np.testing.assert_allclose(u.conj().T @ u, np.eye(u.shape[1]), atol=2e-10, rtol=0)
    index = np.arange(2*length*length)
    x, y = (index // 2) % length, index // (2*length)
    a = u.conj().T @ (x[:, None]*u)
    b = u.conj().T @ (y[:, None]*u)
    diag = np.einsum('ij,ij->i', (u @ a) @ b, u.conj())
    marker = (-4*np.pi*diag.imag).reshape(length, length, 2).sum(axis=2).T
    error = None
    if explicit_check:
        # Independent evaluation through the canonical CPU implementation; no tanh.
        covariance = 2*(frame @ frame.conj().T) - np.eye(len(frame))
        model = SimpleNamespace(Nx=length, Ny=length, Ntot=len(frame))
        explicit = classA_U1FGTN.local_chern_marker_flat(model, covariance, apply_tanh=False)
        np.testing.assert_allclose(marker, explicit, atol=2e-9, rtol=2e-9)
        error = float(abs(marker-explicit).max())
    assert abs(marker.sum()) < 1e-7  # Finite commutator trace, not bulk Chern zero.
    return marker, dict(canonical_max_error=error,
                        orthonormality_max_error=float(abs(u.conj().T@u-np.eye(u.shape[1])).max()),
                        sum=float(marker.sum()), minimum=float(marker.min()), maximum=float(marker.max()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = json.loads((DATA / 'execution_plan.json').read_text())
    R.validate_plan(plan, R.default_config(), R.identity(R.default_config()))
    groups = {l: [] for l in (20, 30, 40)}
    sources = []
    for task in tqdm(plan['tasks'], desc='Verify all 25 batches', unit='batch'):
        if not R.verified_result(DATA, task, plan['identity'], plan['config']):
            raise ValueError('Incomplete or invalid batch: '+task['id'])
        path = DATA / (task['id'] + '.npz')
        receipt = json.loads(path.with_suffix('.json').read_text())
        with np.load(path, allow_pickle=False) as z:
            groups[task['nx']].append((z['sample_ids'], z['center_average']))
        sources.append(dict(path=str(path), **receipt['file'], receipt=R.digest(path.with_suffix('.json'))))
    assert len(sources) == 25
    curves, cycle_rows, summary_rows = {}, [], []
    for l, parts in groups.items():
        ids = np.concatenate([a[0] for a in parts])
        order = np.argsort(ids)
        np.testing.assert_array_equal(ids[order], np.arange(100))
        values = np.concatenate([a[1] for a in parts])[order]
        assert values.shape == (100, 41)
        mean, sem = ensemble_statistics(values)
        np.testing.assert_allclose(sem, np.sqrt(((values-mean)**2).sum(axis=0)/(100*99)), atol=1e-15)
        curves[l] = mean, sem
        delta, lower, upper = absolute_interval(mean, sem)
        for t in range(41):
            cycle_rows.append(dict(L=l, samples=100, cycle=t, mean=mean[t], sem=sem[t],
                                   absolute_deviation=delta[t], deviation_lower=lower[t], deviation_upper=upper[t]))
        late = values[:, 21:41].mean(axis=1)
        summary_rows.append(dict(L=l, samples=100, endpoint_mean=float(mean[-1]), endpoint_sem=float(sem[-1]),
                                 late_mean=float(late.mean()), late_sem=float(late.std(ddof=1)/10)))
    print('All 300 trajectories verified. Computing all 100 L=30 endpoint markers.', flush=True)
    maps, checks, marker_ids = [], [], []
    chosen = None
    with tqdm(total=100, desc='L=30 endpoint markers', unit='sample') as progress:
        for task in plan['tasks']:
            if task['nx'] != 30:
                continue
            with np.load(DATA/(task['id']+'.npz'), allow_pickle=False) as z:
                frames, ranks = z['final_frame'], z['final_ranks']
                for j, sample in enumerate(task['sample_ids']):
                    frame = frames[j, :, :int(ranks[j])]
                    sample_map, check = endpoint_marker(frame, 30, explicit_check=sample == 0)
                    maps.append(sample_map)
                    checks.append(dict(sample_id=sample, **check))
                    marker_ids.append(sample)
                    if sample == 0:
                        chosen = dict(frame=frame.copy(), centers=z['centers_y'][j,-1], chern=z['real_space_chern'][j,-1])
                    progress.update()
                del frames
    order = np.argsort(marker_ids)
    np.testing.assert_array_equal(np.asarray(marker_ids)[order], np.arange(100))
    maps = np.stack(maps)[order]
    marker, marker_sem = maps.mean(axis=0), maps.std(axis=0, ddof=1)/10
    np.testing.assert_allclose(marker_sem, np.sqrt(((maps-marker)**2).sum(axis=0)/(100*99)), atol=1e-15)
    # Match the saved three-sector sign/convention on exactly the plotted sample.
    p = (chosen['frame'] @ chosen['frame'].conj().T).T
    tables = sector_table(30, 30, 6.)
    check_chern = []
    for y0 in chosen['centers']:
        a,b,c = [table[y0] for table in tables]
        check_chern.append(-24*np.pi*np.trace(p[np.ix_(c,a)] @ p[np.ix_(a,b)] @ p[np.ix_(b,c)]).imag)
    np.testing.assert_allclose(check_chern, chosen['chern'], rtol=2e-10, atol=2e-10)
    disk_error = float(np.max(abs(np.asarray(check_chern)-chosen['chern'])))
    write_csv('cycles.csv', cycle_rows)
    write_csv('endpoint_marker.csv', [dict(x=x, y=y, samples=100, local_chern_marker_mean=marker[x,y], sem=marker_sem[x,y]) for x in range(30) for y in range(30)])
    np.savez_compressed(OUT/'plot_data.npz', cycles=np.arange(41), sizes=[20,30,40],
                        means=np.stack([curves[l][0] for l in curves]),
                        sems=np.stack([curves[l][1] for l in curves]), marker=marker,
                        marker_sem=marker_sem, marker_samples=maps, marker_sample_ids=np.arange(100))
    plt.rcParams.update({'font.family': 'CMU Sans Serif', 'font.size': 9,
                         'mathtext.fontset': 'cm', 'axes.linewidth': .8,
                         'xtick.direction': 'in', 'ytick.direction': 'in',
                         'legend.frameon': False, 'pdf.fonttype': 42})
    fig, (ax, spatial) = plt.subplots(2, 1, figsize=(3.375, 6.3),
                                      gridspec_kw={'height_ratios': [1, 1.12]})
    fig.subplots_adjust(left=.19, right=.96, bottom=.16, top=.965, hspace=.40)
    inset = ax.inset_axes([.43, .65, .54, .30])
    floor = 1e-6
    for l, color, mark, ls in zip((20,30,40), ('#d94738','#249b57','#1675bd'), ('^','s','o'), (':','--','-')):
        mean, sem = curves[l]
        delta, lower, upper = absolute_interval(mean, sem)
        style = dict(color=color, marker=mark, ls=ls, lw=1, ms=2.5, mfc='white', mew=.6)
        ax.plot(np.arange(41), delta, label=rf'${l}\times{l}$', **style)
        ax.fill_between(np.arange(41), np.maximum(lower, floor), upper, color=color, alpha=.15, lw=0)
        inset.plot(np.arange(41), mean, color=color, ls=ls, lw=.85)
        inset.fill_between(np.arange(41), mean-sem, mean+sem, color=color, alpha=.15, lw=0)
    ax.set(xlim=(0,40), ylim=(floor,1.5), yscale='log', xlabel='cycle', ylabel=r'$|\overline{C_G}-1|$')
    ax.legend(loc='upper left', fontsize=8, handlelength=2.2, labelspacing=.22, borderaxespad=.35)
    inset.axhline(1, color='.4', ls='--', lw=.6)
    inset.set(xlim=(0,40), ylim=(-.04,1.08), xticks=[0,20,40], yticks=[0,.5,1])
    inset.tick_params(labelsize=8, pad=1, top=True, right=True)
    inset.text(.47,.45,r'$\overline{C_G}$',transform=inset.transAxes,fontsize=9)
    inset.text(.73,.12,'cycle',transform=inset.transAxes,fontsize=8)
    image = spatial.imshow(marker.T, origin='lower', extent=(-.5,29.5,-.5,29.5),
                           cmap='RdBu_r', norm=TwoSlopeNorm(vmin=float(marker.min()),vcenter=0,vmax=float(marker.max())),
                           interpolation='nearest', aspect='equal')
    for wall in R.interfaces(30):
        spatial.axvline(wall, color='k', ls='--', lw=.8)
    spatial.set(xlabel=r'$x$', ylabel=r'$y$', xticks=[0,10,20,29], yticks=[0,10,20,29])
    spatial.set_title(r'$30\times30$, cycle 40, $S=100$', fontsize=9, pad=6)
    # Horizontal colorbar keeps the map and convergence panel at the same width.
    cax = spatial.inset_axes([0, -.27, 1, .05])
    cb = fig.colorbar(image, cax=cax, orientation='horizontal', ticks=[marker.min(),0,marker.max()])
    cb.ax.set_xticklabels([f'{marker.min():.2f}', '0', f'{marker.max():.2f}'])
    cb.ax.tick_params(labelsize=8, pad=2)
    cb.set_label(r'mean local Chern marker $\overline{C(\boldsymbol{r})}$', fontsize=9, labelpad=1)
    fig.canvas.draw()
    for label, a in zip('ab', (ax,spatial)):
        a.tick_params(top=True, right=True)
        fig.text(.045,a.get_position().y1+.01,f'({label})',fontsize=10)
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'slab_chern_convergence_and_marker.{ext}',dpi=300)
    plt.close(fig)
    caption = ('Hard-wall bulk topology. (a) Absolute deviation of the sample-averaged disk Chern number from unity; '
               'inset: the same means on a linear scale. S=100 independent, initially half-filled pure trajectories per size, '
               '40 cycles, perfect correction, raster-y updates. Ten random transverse centers are averaged within each trajectory, '
               'with x0=L/2 and R=0.2L, before taking the ensemble mean. Shading is one sample SEM (ddof=1); '
               'the absolute-value-transformed interval is used in the main panel, with zero lower endpoints clipped only for log display. '
               '(b) Local Chern marker at L=30 and cycle 40, evaluated for each of the 100 endpoint projectors, summed over the two orbitals, then averaged over trajectories. '
               'Dashed lines mark x=8 and x=22. All runs use nshell=1, alpha1=1, alpha2=30 and slab-only measurements. '
               'The position-commutator marker uses ordinary coordinates on a periodic sample and retains periodic-seam artifacts; '
               'it is not the finite-radius three-sector estimator in (a). No tanh, smoothing, or value clipping is applied to the map. '
               'The diverging color scale has separate linear ranges below/above zero; endpoints show the full data range.\n')
    (OUT/'caption.txt').write_text(caption)
    summary = dict(revision=R.REVISION, config=plan['config'], identity=plan['identity'], sources=sources,
                   results=summary_rows, marker=dict(samples=100, L=30, cycle=40,
                   formula='Mean over 100 per-trajectory Re(2*pi*i*diag(P X P Y P - P Y P X P)), orbitals summed; P=(V V^dagger)^T',
                   checks=checks, disk_estimator_sample0_max_error=disk_error), log_floor=floor,
                   script=R.digest(Path(__file__)), canonical_cpu_source=R.digest(REPO/'src/fgtn/classA_U1FGTN.py'),
                   outputs={p.name:R.digest(p) for p in OUT.iterdir() if p.suffix in ('.pdf','.png','.csv','.npz','.txt')})
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(dict(results=summary_rows, marker_mean_range=[float(marker.min()),float(marker.max())],
                         disk_check=disk_error, output=str(OUT)),indent=2),flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
