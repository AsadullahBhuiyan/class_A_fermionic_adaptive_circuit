"""Offline scalar analysis of verified bundle-23 results; no dynamics or frame loads.

Run with OPENBLAS_NUM_THREADS=1 python analyze_results.py. All uncertainties
use independent trajectories after center/time averaging, never pooled centers.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
REVISION = 'hard_wall_random_center_chern_nx20_ny20-30-40_s100_r0p2_v1'
DATA = ROOT / 'data' / REVISION
OUT = ROOT / 'analysis_outputs' / 'completed_campaign_v1'
REFERENCE = ROOT.parent / '06_domain_wall_flattened_ground_state_reference/results/random_center_chern_hard_nx20_ny20-30-40_nsh1_r4_v1/summary.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return dict(bytes=path.stat().st_size, sha256=h.hexdigest())


def stats(x):
    x = np.asarray(x)
    mean, sem = float(x.mean()), float(x.std(ddof=1)/np.sqrt(len(x)))
    assert np.isclose(sem, np.sqrt(np.sum((x-mean)**2)/(len(x)*(len(x)-1))), rtol=1e-12, atol=1e-15)
    return dict(mean=mean, sem=sem)


def csv_save(name, rows):
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    verification = json.loads((DATA / 'VERIFICATION.json').read_text())
    assert verification['status'] == 'complete' and verification['trajectories'] == 300
    plan = json.loads((DATA / 'execution_plan.json').read_text())
    reference = json.loads(REFERENCE.read_text())
    assert reference['source_sha256'][str(ROOT.relative_to(next(p for p in ROOT.parents if (p/'PROJECT_ADMIN').is_dir())) / 'random_center_observer.py')] == plan['identity']['sources']['random_center_observer.py']
    refs = {r['ny']: r for r in reference['results']}
    grouped = {ny: [] for ny in (20, 30, 40)}
    sources = []
    for task in tqdm(plan['tasks'], desc='Verify and read scalar products', unit='batch'):
        path = DATA / (task['id'] + '.npz')
        receipt = json.loads(path.with_suffix('.json').read_text())
        actual = digest(path)
        assert actual == receipt['file']
        assert receipt['task'] == task and receipt['identity'] == plan['identity']
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata_json']))
            assert meta['task'] == task and meta['identity'] == plan['identity'] and meta['config'] == plan['config']
            a = {k: z[k] for k in ('cycles', 'sample_ids', 'centers_y', 'real_space_chern', 'center_average', 'global_charge')}
        assert np.array_equal(a['cycles'], np.arange(2*task['ny']+1))
        assert np.array_equal(a['sample_ids'], task['sample_ids'])
        assert np.array_equal(a['center_average'], a['real_space_chern'].mean(2))
        assert np.isfinite(a['real_space_chern']).all()
        grouped[task['ny']].append(a)
        sources.append(dict(path=str(path.relative_to(ROOT)), **actual,
                            receipt=digest(path.with_suffix('.json'))))
    OUT.mkdir(parents=True, exist_ok=True)
    cycle_rows, trajectory_rows, summaries, curves = [], [], [], {}
    for ny, parts in grouped.items():
        ids = np.concatenate([a['sample_ids'] for a in parts])
        assert np.array_equal(np.sort(ids), np.arange(100))
        order = np.argsort(ids)
        c, v, q = [np.concatenate([a[k] for a in parts])[order] for k in
                   ('center_average', 'real_space_chern', 'global_charge')]
        dq = q - 20*ny
        t = np.arange(c.shape[1])
        mean, sem = c.mean(0), c.std(0, ddof=1)/10
        spatial_var = v.var(2, ddof=1)
        # Paired trajectory differences retain temporal correlations in their SEM.
        drift = c[:, -10:].mean(1) - c[:, -20:-10].mean(1)
        common_drift = c[:, 31:41].mean(1) - c[:, 21:31].mean(1)
        late = c[:, -20:].mean(1)
        common = c[:, 21:41].mean(1)
        summary = dict(ny=ny, samples=100, final_cycle=int(t[-1]),
                       initial=stats(c[:, 0]), endpoint=stats(c[:, -1]),
                       last20_window=[int(t[-20]), int(t[-1])], last20=stats(late),
                       common_cycles21_40=stats(common), last20_paired_half_drift=stats(drift),
                       common_paired_half_drift=stats(common_drift),
                       endpoint_center_variance=stats(spatial_var[:, -1]),
                       last20_center_variance=stats(spatial_var[:, -20:].mean(1)),
                       endpoint_center_rms_sd=float(np.sqrt(spatial_var[:, -1].mean())),
                       endpoint_trajectory_sd=float(c[:, -1].std(ddof=1)),
                       endpoint_individual_center_quantiles=np.quantile(v[:, -1], [.01, .05, .5, .95, .99]).tolist(),
                       endpoint_fraction_centers_below_099=float(np.mean(v[:, -1] < .99)),
                       endpoint_fraction_centers_above_one=float(np.mean(v[:, -1] > 1)),
                       endpoint_charge_offset=stats(dq[:, -1]),
                       endpoint_charge_sd=float(dq[:, -1].std(ddof=1)),
                       endpoint_charge_range=[int(dq[:, -1].min()), int(dq[:, -1].max())],
                       endpoint_abs_relative_charge_percent=stats(100*np.abs(dq[:, -1])/(20*ny)),
                       equilibrium_chern=refs[ny]['center_mean'],
                       equilibrium_center_variance=float(np.var(refs[ny]['chern_values'], ddof=1)))
        summaries.append(summary)
        for i in t:
            cycle_rows.append(dict(ny=ny, cycle=int(i), chern_mean=mean[i], chern_sem=sem[i],
                                   abs_mean_minus_one=abs(mean[i]-1),
                                   center_variance_mean=spatial_var[:, i].mean(),
                                   center_variance_sem=spatial_var[:, i].std(ddof=1)/10,
                                   charge_offset_mean=dq[:, i].mean(), charge_offset_sem=dq[:, i].std(ddof=1)/10,
                                   abs_relative_charge_percent_mean=(100*np.abs(dq[:, i])/(20*ny)).mean()))
        for s in range(100):
            trajectory_rows.append(dict(ny=ny, sample_id=s, endpoint_chern=c[s, -1],
                                        last20_chern=late[s], common_cycles21_40_chern=common[s],
                                        last20_paired_half_drift=drift[s],
                                        endpoint_center_variance=spatial_var[s, -1],
                                        endpoint_charge_offset=int(dq[s, -1])))
        curves[ny] = (t, mean, sem, spatial_var, dq)
    csv_save('cycle_statistics.csv', cycle_rows)
    csv_save('trajectory_statistics.csv', trajectory_rows)
    plt.rcParams.update({'font.family': 'CMU Sans Serif', 'font.size': 9,
                         'mathtext.fontset': 'cm', 'axes.labelsize': 10,
                         'xtick.direction': 'in', 'ytick.direction': 'in',
                         'legend.frameon': False, 'savefig.dpi': 300})
    fig, ax = plt.subplots(2, 2, figsize=(7.05, 5.2), layout='constrained')
    for ny, color, marker, ls in zip((20,30,40), ('#d94738','#249b57','#1675bd'), ('^','s','o'), (':','--','-')):
        t, mean, sem, variance, dq = curves[ny]
        style = dict(color=color, marker=marker, linestyle=ls, markersize=3,
                     markerfacecolor='white', linewidth=1.1, label=rf'$N_y={ny}$')
        ax[0,0].plot(t, mean, **style)
        ax[0,0].fill_between(t, mean-sem, mean+sem, color=color, alpha=.15)
        delta = abs(mean-1)
        lower = np.where(abs(mean-1) <= sem, 0, np.minimum(abs(mean-sem-1),abs(mean+sem-1)))
        upper = np.maximum(abs(mean-sem-1), abs(mean+sem-1))
        ax[0,1].plot(t, delta, **style)
        ax[0,1].fill_between(t, np.maximum(lower, 1e-5), upper, color=color, alpha=.15)
        var_mean, var_sem = variance.mean(0), variance.std(0,ddof=1)/10
        ax[1,0].plot(t, var_mean, **style)
        ax[1,0].fill_between(t, np.maximum(var_mean-var_sem,1e-8), var_mean+var_sem, color=color, alpha=.15)
        absq = 100*np.abs(dq)/(20*ny)
        qm, qe = absq.mean(0), absq.std(0,ddof=1)/10
        ax[1,1].plot(t, qm, **style)
        ax[1,1].fill_between(t, qm-qe, qm+qe, color=color, alpha=.15)
    ax[0,0].axhline(1,color='.4',ls='--',lw=.8)
    ax[0,0].set(xlim=(0,10), ylim=(-.03,1.04), ylabel=r'$\overline{C_G}$')
    ax[0,0].legend(loc='lower right', fontsize=9)
    ax[0,1].set(yscale='log', ylim=(1e-5,1.3), ylabel=r'$|\overline{C_G}-1|$')
    ax[0,1].axhline(abs(refs[20]['center_mean']-1),color='.4',ls='--',lw=.8)
    ax[0,1].text(27,6e-5,'ground-state reference',color='.35',fontsize=8)
    ax[1,0].set(yscale='log', ylabel=r'Mean center variance of $C_G$')
    ax[1,1].set(ylabel=r'$100\,\overline{|Q-N_xN_y|}/(N_xN_y)$')
    for label, a in zip('abcd', ax.flat):
        a.set_xlabel('cycle')
        a.tick_params(top=True, right=True)
        a.text(-.17,1.035,f'({label})',transform=a.transAxes)
    for ext in ('pdf','png'):
        fig.savefig(OUT / f'campaign_summary.{ext}')
    plt.close(fig)
    result = dict(revision=REVISION, config=plan['config'], identity=plan['identity'],
                  inputs=sources, prior_full_validation=digest(DATA/'VERIFICATION.json'),
                  reference=dict(path=str(REFERENCE), **digest(REFERENCE)),
                  script=digest(Path(__file__)),
                  statistics='Center mean within each trajectory; sample SEM ddof=1/sqrt(100). Late windows averaged within each trajectory. Center variance uses ddof=1 across ten centers; descriptive, not an independent-trajectory uncertainty.',
                  drift='Paired difference of last ten versus preceding ten cycles, computed within each trajectory; exploratory stationarity diagnostic, not proof of convergence.',
                  results=summaries,
                  products={p.name: digest(p) for p in OUT.iterdir() if p.suffix in ('.csv','.pdf','.png')})
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))
    print('Saved:',OUT)


if __name__ == '__main__':
    main()
