#!/usr/bin/env python3
"""Figure-15-style views of recovered H1-v3 data; no simulations or raw edits."""
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[3]
INPUT = ROOT / '00_WORKSPACE/COLAB/final_production_drive_recovery_2026-09-01/outputs/production_25sample_h1_endpoint_packet_v3/08_h1_endpoint_packet'
OBSERVER = ROOT / '00_WORKSPACE/LEGACY/final_production_new_designs_quarantine_2026-09-01/08_h1_endpoint_packet/src/h1_packet_observables.py'
OUT = PROJECT / 'outputs/figure15_hard_alpha1_alpha3_v1'
EXPECTED = {1: list(range(20)), 3: list(range(25))}
PACKETS = ((0, 0), (1, 1))


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def member(archive, suffix):
    matches = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f'Expected unique archive member: {suffix}')
    return archive.extractfile(matches[0]).read()


def verify_bytes(raw, row):
    if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
        raise ValueError('Product byte count or checksum mismatch')


def reduce_profiles(centers, profiles):
    # Inputs: (sample,cut,wall,endpoint,time), (sample,wall,endpoint,time,y).
    if centers.shape[1:] != (40,2,2,161) or profiles.shape != (len(centers),2,2,161,20):
        raise ValueError('Unexpected profile axes')
    if not np.isfinite(centers).all() or not np.isfinite(profiles).all() or np.any(profiles < -1e-14):
        raise ValueError('Invalid profile values')
    np.testing.assert_allclose(profiles.sum(-1), 1, atol=1e-12)
    # Linear center calculation commutes with cut averaging after normalization.
    np.testing.assert_allclose(profiles @ np.arange(20), centers.mean(1), atol=2e-12)
    dy = centers - centers[..., :1]
    return np.stack([dy[:,:,w,e].mean(1) for w,e in PACKETS], axis=1)


def load():
    groups = {a: [] for a in EXPECTED}
    records = []
    engine_hashes = set()
    for path in tqdm(sorted(INPUT.glob('*.tar.gz')), desc='verify recovered H1', unit='archive'):
        with tarfile.open(path) as archive:
            manifest = json.loads(member(archive, '/manifest.json'))
            if manifest['protocol'] != 'hard':
                continue
            alpha = int(manifest['alpha_1'])
            if alpha not in groups or manifest['alpha_1'] != alpha:
                raise ValueError('Unexpected hard-wall alpha')
            receipt = json.loads(Path(str(path)+'.receipt.json').read_text())
            if receipt['archive'] != path.name or receipt['archive_bytes'] != path.stat().st_size or receipt['archive_sha256'] != digest(path):
                raise ValueError('Archive receipt mismatch')
            if manifest['sampling_revision'] != 'production_25sample_h1_endpoint_packet_v3' or manifest['numerical_status'] != 'pass':
                raise ValueError('Incompatible revision or numerical status')
            case = manifest['run_config']['case']
            model, run, obs = case['model'], case['run'], case['observer']
            expected_model = dict(Nx=20, Ny=40, DW=True, dw_truncation=True, alpha_1=alpha,
                                  alpha_2=30, nshell=1, dtype='complex128', filling_frac=.5, init_mode='default')
            if any(model[k] != v for k,v in expected_model.items()):
                raise ValueError('Scientific model mismatch')
            if any(run[k] != v for k,v in dict(cycles=80, sequence='random', perfect_correction=True,
                                               postselect=False, meas_slab_only=True).items()):
                raise ValueError('Dynamics contract mismatch')
            if obs['primary_source_width_columns'] != 3 or obs['primary_retention_width_columns'] != 3 or obs['primary_spectral_clip_eps'] != 1e-10:
                raise ValueError('Primary packet convention mismatch')
            if manifest['source_hashes']['h1_packet_observables.py'] != digest(OBSERVER):
                raise ValueError('Observer interpretation does not match archived source')
            engine_hashes.add(manifest['canonical_engine_sha256'])
            declared = {p['path']: p for p in manifest['products']['h1_endpoint_packet']['files']}
            loaded = {}
            for name in ('common.npz','packet_drift.npz','primary_profiles.npz'):
                raw = member(archive, '/h1_endpoint_packet/'+name)
                verify_bytes(raw, declared[name])
                with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                    if name == 'common.npz':
                        for k,expected in [('checkpoints',[40,48,56,64,72,80]),('cut_origins',np.arange(40)),('wall_x',[5,15])]:
                            np.testing.assert_array_equal(z[k],expected)
                        np.testing.assert_allclose(z['modular_times'],np.arange(161)*.05,atol=1e-14)
                        np.testing.assert_array_equal(z['global_sample_ids'],manifest['global_sample_indices'])
                        ids = z['global_sample_ids'].copy()
                        if json.loads(str(z['config_json'])) != manifest['run_config']:
                            raise ValueError('Embedded configuration mismatch')
                    elif name == 'primary_profiles.npz':
                        for k in ('primary_endpoint_center','primary_endpoint_retention','primary_cut_mean_conditional_profile'):
                            loaded[k] = z[k][:,-1].copy()
            c,p,q = (loaded[k] for k in ('primary_endpoint_center','primary_cut_mean_conditional_profile','primary_endpoint_retention'))
            dy = reduce_profiles(c,p)
            if not np.isfinite(q).all() or np.any(q <= 0) or np.any(q > 1+1e-12):
                raise ValueError('Invalid retained probabilities')
            paired = .5*(c[:,:,:,0]+c[:,:,:,1]-19)
            paired = (paired-paired[...,:1]).mean(1)
            groups[alpha].append(dict(ids=ids, dy=dy, profiles=p,
                                     retention=np.stack([q[:,:,w,e].mean(1) for w,e in PACKETS],axis=1),
                                     paired=paired))
            records.append(dict(path=str(path.relative_to(ROOT)),sha256=receipt['archive_sha256'],
                                manifest=manifest))
    if len(engine_hashes) != 1:
        raise ValueError('Mixed engine identities')
    result = {}
    for alpha, rows in groups.items():
        arrays = {k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
        order = np.argsort(arrays['ids'])
        arrays = {k:v[order] for k,v in arrays.items()}
        np.testing.assert_array_equal(arrays['ids'],EXPECTED[alpha])
        result[alpha] = arrays
    return result, records


def style():
    plt.rcParams.update({'font.family':'CMU Sans Serif','mathtext.fontset':'stix','font.size':8,
                         'axes.labelsize':8,'legend.fontsize':7,'axes.linewidth':.8,
                         'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})


def draw_row(fig, grid, arrays, alpha, letters):
    left, right = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    times = np.arange(161)*.05
    colors = ('#332288','#E69F00','#009E73')
    snapshots = (0,.5,1)
    p = arrays['profiles'].mean(0)
    for w,e in PACKETS:
        for time,color in zip(snapshots,colors):
            values = p[w,e,int(round(time/.05))]
            keep = values > 1e-4
            # x is the strip label only: no transverse density is reconstructed.
            left.scatter(np.full(keep.sum(),(5,15)[w]), np.arange(20)[keep],
                         s=105*np.sqrt(values[keep]), color=color, alpha=.7,
                         linewidths=.3, zorder=5-int(time*2))
    for wall in (5,15):left.axvline(wall,color='.35',ls=':',lw=.85)
    left.set(xlim=(-.5,19.5),ylim=(-1,20),xticks=(0,5,15,19),yticks=(0,5,10,15,19),
             xlabel=r'wall center $x$ (projected)',ylabel=r'$y-y_0$')
    left.set_aspect('equal')
    handles=[Line2D([],[],ls='none',marker='o',color=c,ms=4,label=f'{t:g}') for t,c in zip(snapshots,colors)]
    left.legend(handles=handles,title=r'$t_{\rm mod}$',loc='center',frameon=False,
                handlelength=.8,handletextpad=.3)
    mean = arrays['dy'].mean(0)
    sem = arrays['dy'].std(0,ddof=1)/np.sqrt(len(arrays['ids']))
    for k,(color,ls,label) in enumerate(zip(('#D55E00','#0072B2'),('-','--'),(r'$(5,0)$',r'$(15,19)$'))):
        right.plot(times,mean[k],color=color,ls=ls,lw=1.15,label=label)
        right.fill_between(times,mean[k]-sem[k],mean[k]+sem[k],color=color,alpha=.12,lw=0)
    right.axhline(0,color='.4',lw=.7)
    right.set(xlim=(0,8),ylim=(-19.5,19.5),xticks=(0,2,4,6,8),yticks=(-15,0,15),
              xlabel=r'modular time $t_{\rm mod}$',ylabel=r'$\overline{\Delta y}_{\rm strip}$')
    right.legend(frameon=False,loc='upper right',ncol=2)
    right.text(.03,.94,rf'$\alpha_1={alpha},\ S={len(arrays["ids"])}$',transform=right.transAxes,va='top')
    for ax,letter in zip((left,right),letters):ax.text(-.16,1.04,f'({letter})',transform=ax.transAxes)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    groups, records = load()
    style()
    fig = plt.figure(figsize=(7.05,5.35))
    grid = fig.add_gridspec(2,2,width_ratios=(.95,1.55),wspace=.4,hspace=.45)
    for row,(alpha,arrays) in enumerate(groups.items()):
        draw_row(fig,(grid[row,0],grid[row,1]),arrays,alpha,('ab','cd')[row])
    fig.subplots_adjust(left=.07,right=.985,bottom=.09,top=.94)
    for ext in ('pdf','png'):fig.savefig(OUT/f'figure15_recovered_hard_comparison.{ext}',dpi=300)
    plt.close(fig)
    for alpha,arrays in groups.items():
        fig=plt.figure(figsize=(7.05,2.65))
        grid=fig.add_gridspec(1,2,width_ratios=(.95,1.55),wspace=.4)
        draw_row(fig,(grid[0,0],grid[0,1]),arrays,alpha,'ab')
        fig.subplots_adjust(left=.07,right=.985,bottom=.2,top=.91)
        for ext in ('pdf','png'):fig.savefig(OUT/f'figure15_recovered_hard_alpha{alpha}.{ext}',dpi=300)
        plt.close(fig)
    np.savez_compressed(OUT/'plotted_data.npz',times=np.arange(161)*.05,
                        **{f'alpha{a}_{k}':v for a,arrays in groups.items() for k,v in arrays.items()})
    rows=[]
    for alpha,arrays in groups.items():
        for k,(w,e) in enumerate(PACKETS):
            mean=arrays['dy'][:,k].mean(0);sem=arrays['dy'][:,k].std(0,ddof=1)/np.sqrt(len(arrays['ids']))
            for j,t in enumerate(np.arange(161)*.05):
                rows.append(dict(alpha1=alpha,S=len(arrays['ids']),wall=(5,15)[w],endpoint=(0,19)[e],
                                 modular_time=t,mean_dy=mean[j],sem_dy=sem[j],
                                 mean_retention=arrays['retention'][:,k,j].mean()))
    with (OUT/'displacement_curves.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    report=dict(cycle=80, Nx=20,Ny=40, packets=[[5,0],[15,19]],
                uncertainty='ddof=1 SEM over trajectories, matching original Figure 15; no bootstrap',
                estimator='strip-normalized center separately per cut; subtract initial center; average 40 cuts within trajectory; then ensemble mean',
                spatial='saved conditional y profiles, integrated over three-column strips; plotted at wall center, NOT full xy density',
                sources=records,plot_script_sha256=digest(__file__),observer_sha256=digest(OBSERVER),
                samples={str(a):len(v['ids']) for a,v in groups.items()},
                diagnostics={str(a):dict(mean_displacements_t8=v['dy'].mean(0)[:,-1].tolist(),
                    mean_paired_wall_drift_t2=v['paired'].mean(0)[:,40].tolist(),
                    minimum_saved_endpoint_retention_mean_over_cuts=float(v['retention'].min())) for a,v in groups.items()})
    (OUT/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['diagnostics'],indent=2));print('[done]',OUT)


if __name__=='__main__':main()
