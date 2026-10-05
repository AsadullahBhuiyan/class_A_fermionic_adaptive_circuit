"""Global particle-number spread across saved pure-state trajectories."""
from pathlib import Path
import csv
import hashlib
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
REPLAY = REPO / '00_WORKSPACE/CURRENT/final_production_new_designs/09_pure_tangent_replay_acquisition/gpu_data/pure_tangent_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1'
PUMP = REPO / '00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot/imported_endpoints/wall_pump_width_endpoints_s100_v1'


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    out = HERE / 'outputs'
    out.mkdir(parents=True, exist_ok=True)
    trajectories, sources = [], []
    receipts = [('pure_replay', p) for p in sorted(REPLAY.rglob('*.complete.json'))]
    receipts += [('wall_pump_primary', p) for p in sorted((PUMP / 'endpoints').rglob('*.completion.json'))]
    for campaign, receipt in tqdm(receipts, desc='Verify endpoint batches', unit='file'):
        m = json.loads(receipt.read_text())
        assert m['status'] == 'complete'
        p = receipt.parent / m['result_filename']
        assert p.stat().st_size == m['result_bytes']
        with p.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        assert digest == m['result_sha256'], p
        with np.load(p, allow_pickle=False) as z:
            if campaign == 'pure_replay':
                q, q0 = z['final_ranks'], z['initial_ranks']
                ids, wall, alpha, shell = z['case_sample_indices'], str(z['construction']), float(z['alpha_1']), '1'
            else:
                q, q0 = z['ranks'], z['initial_total_charge']
                ids, wall, alpha, shell = z['sample_ids'], str(z['wall']), float(z['alpha_1']), str(z['nshell_label'])
                assert np.array_equal(q, z['final_total_charge'])
                assert np.array_equal(q, z['global_charge'][:, -1])
            nx, ny = int(z['Nx']), int(z['Ny'])
            assert q.dtype.kind in 'iu' and len(ids) == len(q)
            for sid, initial, charge in zip(ids, q0, q):
                trajectories.append(dict(campaign=campaign, wall=wall, Nx=nx, Ny=ny,
                    alpha_1=alpha, shell=shell, sample_id=int(sid), initial_charge=int(initial),
                    final_charge=int(charge), half_filling_offset=int(charge)-nx*ny))
        sources.append(dict(path=str(p), sha256=digest, bytes=p.stat().st_size))
    groups = defaultdict(list)
    for row in trajectories:
        key = tuple(row[k] for k in ('campaign', 'wall', 'Nx', 'Ny', 'alpha_1', 'shell'))
        groups[key].append(row)
    rng = np.random.default_rng(2026091401)
    summaries = []
    for key, rows in sorted(groups.items()):
        assert sorted(r['sample_id'] for r in rows) == list(range(100)), key
        q = np.array([r['final_charge'] for r in rows], dtype=float)
        volume = key[2] * key[3]
        variance = q.var(ddof=1)
        boot = q[rng.integers(0, len(q), size=(5000, len(q)))].var(axis=1, ddof=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        s = dict(zip(('campaign', 'wall', 'Nx', 'Ny', 'alpha_1', 'shell'), key))
        s.update(samples=len(q), mean_charge=float(q.mean()), mean_offset=float(q.mean()-volume),
            variance=float(variance), variance_ci_low=float(lo), variance_ci_high=float(hi),
            std_charge=float(np.sqrt(variance)), relative_std_percent=float(100*np.sqrt(variance)/volume),
            relative_std_ci_low=float(100*np.sqrt(lo)/volume), relative_std_ci_high=float(100*np.sqrt(hi)/volume),
            initial_variance=float(np.var([r['initial_charge'] for r in rows], ddof=1)))
        summaries.append(s)
    write_csv(out/'trajectory_charges.csv', trajectories)
    write_csv(out/'charge_summary.csv', summaries)
    plt.rcParams.update({'font.family':'CMU Sans Serif', 'font.size':9,
        'xtick.direction':'in', 'ytick.direction':'in', 'savefig.dpi':300})
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.3))
    for col, (campaign, coord, title) in enumerate([
        ('pure_replay', 'Ny', r'$N_x=20$, $n_{\rm shell}=1$, $T=2N_y$'),
        ('wall_pump_primary', 'Nx', r'$N_y=24$, dense OW support, $T=48$')]):
        for wall, color, marker, ls in [('hard','#c0392b','^',':'), ('soft','#29804a','s','--')]:
            selected = [s for s in summaries if s['campaign']==campaign and s['wall']==wall
                        and s['alpha_1']==1 and (campaign=='pure_replay' or s['shell']=='dense')]
            selected.sort(key=lambda s:s[coord])
            assert len(selected)==3, selected
            x = np.array([s[coord] for s in selected])
            for ax, key, low, high in [(axes[0,col],'variance','variance_ci_low','variance_ci_high'),
                                      (axes[1,col],'relative_std_percent','relative_std_ci_low','relative_std_ci_high')]:
                y = np.array([s[key] for s in selected])
                err = np.array([[s[key]-s[low] for s in selected], [s[high]-s[key] for s in selected]])
                ax.errorbar(x,y,yerr=err,color=color,marker=marker,ls=ls,ms=5,capsize=3,
                            mfc='white',label=wall.capitalize())
                ax.set_xticks(x)
                ax.tick_params(top=True,right=True)
                ax.set_xlabel(r'$N_y$' if coord=='Ny' else r'$N_x$')
        axes[0,col].set_title(title,fontsize=10)
        axes[0,col].legend(frameon=False)
    axes[0,0].set_ylabel(r'Across-trajectory $\mathrm{Var}(Q)$')
    axes[1,0].set_ylabel(r'$100\,\sigma_Q/(N_xN_y)$ [%]')
    for i,ax in enumerate(axes.flat):
        ax.text(-0.17,1.04,f'({chr(97+i)})',transform=ax.transAxes)
        ax.set_ylim(bottom=0)
    fig.suptitle('Pure-state endpoint global-charge fluctuations; 100 trajectories per point',fontsize=10)
    fig.tight_layout()
    for ext in ('pdf','png'):
        fig.savefig(out/f'hard_soft_global_charge.{ext}',bbox_inches='tight')
    caption = ('Across-trajectory global particle-number variance and standard deviation relative to half filling. '
        'Each point uses 100 independent pure-state trajectories with perfect correction, raster-y ordering, '
        'alpha_1=1, alpha_2=30. Left: independent pure replay acquisition, Nx=20, nshell=1, endpoint T=2Ny. '
        'Right: independent wall-pump endpoint acquisition, Ny=24, dense support, endpoint T=48. '
        'Campaigns are not pooled. Error bars are percentile 95% confidence intervals from 5,000 trajectory '
        'bootstrap replicates. Dashed/dotted lines guide the eye; no size fit. '
        'Hard walls include support truncation, Born-conditioned exterior preparation, and slab-only subsequent '
        'updates; soft walls retain cross-interface support and updates in both regions. '
        'This compares the full saved protocols, not a support-mask-only intervention. '
        'For each pure number-conserving Slater determinant, Q is its occupied-frame rank and intrinsic '
        'global quantum variance is zero. The plotted variance is the spread of Q across outcomes. '
        'Endpoints are finite-time results; steady-state convergence is not asserted.\n')
    (out/'caption.txt').write_text(caption)
    (out/'source_manifest.json').write_text(json.dumps(dict(sources=sources, bootstrap_seed=2026091401,
        bootstrap_replicates=5000, caption=caption),indent=2)+'\n')
    for s in summaries:
        if s['alpha_1']==1:
            print(json.dumps(s),flush=True)


if __name__ == '__main__':
    main()
