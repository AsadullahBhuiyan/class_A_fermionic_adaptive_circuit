"""Import browser ZIPs without changing originals; analyze completed batches only.

Local-only helper, not a production dependency. Centers and late-time cycles
are averaged within trajectories before computing sample SEM (ddof=1).
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import run_campaign as R

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data'/R.REVISION


def import_archives(archives):
    DATA.mkdir(parents=True, exist_ok=True)
    members, handles = {}, []
    try:
        for path in archives:
            z = zipfile.ZipFile(path)
            handles.append(z)
            for info in z.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name
                if name in members:
                    raise ValueError('Duplicate ZIP entry: '+name)
                members[name] = (z, info, path)
        z, info, _ = members['execution_plan.json']
        plan = json.loads(z.read(info))
        R.validate_plan(plan, R.default_config(), R.identity(R.default_config()))
        tasks = {t['id']: t for t in plan['tasks']}
        allowed = {'execution_plan.json'} | {t+ext for t in tasks for ext in ('.npz','.json')}
        if not set(members) <= allowed:
            raise ValueError('Unexpected files in campaign ZIPs')
        receipts = {}
        for task_id, task in tasks.items():
            name = task_id+'.json'
            if name not in members:
                continue
            z, info, _ = members[name]
            receipt = json.loads(z.read(info))
            assert receipt['task'] == task and receipt['identity'] == plan['identity']
            assert receipt['result'] == task_id+'.npz'
            receipts[task_id+'.npz'] = receipt
        products = []
        for name, (z, info, archive) in tqdm(members.items(), desc='Import verified snapshot', unit='file'):
            expected = receipts[name]['file'] if name.endswith('.npz') else None
            if expected:
                assert info.file_size == expected['bytes']
            destination = DATA/name
            if destination.exists():
                actual = R.digest(destination)
                if expected:
                    assert actual == expected, name
                else:
                    assert destination.read_bytes() == z.read(info), name
            else:
                fd, tmp = tempfile.mkstemp(dir=DATA, prefix='.'+name+'.')
                try:
                    with os.fdopen(fd, 'wb') as stream, z.open(info) as source:
                        shutil.copyfileobj(source, stream, 8*1024**2)
                    actual = R.digest(tmp)
                    assert actual['bytes'] == info.file_size
                    if expected:
                        assert actual == expected, name
                    os.replace(tmp, destination)
                finally:
                    Path(tmp).unlink(missing_ok=True)
            products.append(dict(name=name, archive=str(archive.resolve()), member=info.filename, **actual))
        manifest = dict(revision=R.REVISION, method='browser Drive ZIP download',
                        archives=[dict(path=str(p.resolve()), **R.digest(p)) for p in archives],
                        products=products)
        (DATA/'IMPORT_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    finally:
        for z in handles:
            z.close()


def statistics(x):
    x = np.asarray(x)
    mean = x.mean(axis=0)
    sem = x.std(axis=0, ddof=1)/np.sqrt(len(x))
    np.testing.assert_allclose(sem, np.sqrt(((x-mean)**2).sum(axis=0)/(len(x)*(len(x)-1))), rtol=1e-12, atol=1e-15)
    return mean, sem


def pair(x):
    mean, sem = statistics(x)
    return dict(mean=float(mean), sem=float(sem))


def analyze():
    plan = json.loads((DATA/'execution_plan.json').read_text())
    R.validate_plan(plan, R.default_config(), R.identity(R.default_config()))
    groups = {l:[] for l in (20,30,40)}
    sources, pending = [], []
    for task in tqdm(plan['tasks'], desc='Verify scientific results', unit='batch'):
        path = DATA/(task['id']+'.npz')
        if not path.exists() or not path.with_suffix('.json').exists():
            pending.append(task['id'])
            continue
        # Includes whole-file checksum, metadata, full frame dtype/rank/norm,
        # center determinism, cycle coverage, integer charge and exact means.
        if not R.verified_result(DATA, task, plan['identity'], plan['config']):
            raise ValueError('Result failed validation: '+task['id'])
        with np.load(path, allow_pickle=False) as z:
            groups[task['nx']].append({k:z[k] for k in ('sample_ids','cycles','real_space_chern','center_average','global_charge')})
        sources.append(dict(name=path.name, **R.digest(path)))
    counts = {l:sum(len(a['sample_ids']) for a in rows) for l,rows in groups.items()}
    out = ROOT/'analysis_outputs'/('snapshot_'+'_'.join(f'L{l}S{n}' for l,n in counts.items()))
    out.mkdir(parents=True, exist_ok=True)
    summaries, cycle_rows, sample_rows, curves = [], [], [], {}
    for l, parts in groups.items():
        if not parts:
            continue
        ids = np.concatenate([a['sample_ids'] for a in parts])
        assert len(set(ids)) == len(ids) and np.all((ids>=0)&(ids<100))
        order = np.argsort(ids)
        c,v,q = [np.concatenate([a[k] for a in parts])[order] for k in
                 ('center_average','real_space_chern','global_charge')]
        n, nt = c.shape
        assert nt==41
        mean, sem = statistics(c)
        spatial_var = v.var(axis=2,ddof=1)
        late = c[:,21:41].mean(axis=1)
        drift = c[:,31:41].mean(axis=1)-c[:,21:31].mean(axis=1)
        relative_charge = 100*np.abs(q-l*l)/(l*l)
        summary = dict(L=l, samples=n, complete=n==100, radius=.2*l,
                       endpoint=pair(c[:,-1]), cycles21_40=pair(late), paired_late_drift=pair(drift),
                       endpoint_center_variance=pair(spatial_var[:,-1]),
                       cycles21_40_center_variance=pair(spatial_var[:,21:41].mean(1)),
                       endpoint_center_quantiles=np.quantile(v[:,-1],[.01,.05,.5,.95,.99]).tolist(),
                       endpoint_mean_absolute_charge_percent=pair(relative_charge[:,-1]),
                       cycle_means={str(t):float(mean[t]) for t in (0,1,2,3,5,10,20,30,40)})
        summaries.append(summary)
        for t in range(41):
            cycle_rows.append(dict(L=l,samples=n,cycle=t,chern_mean=mean[t],chern_sem=sem[t],
                                   center_variance_mean=spatial_var[:,t].mean(),
                                   center_variance_sem=spatial_var[:,t].std(ddof=1)/np.sqrt(n)))
        for j,sample in enumerate(ids[order]):
            sample_rows.append(dict(L=l,sample_id=int(sample),endpoint_chern=c[j,-1],
                                    cycles21_40_chern=late[j],paired_late_drift=drift[j],
                                    endpoint_center_variance=spatial_var[j,-1],endpoint_charge=int(q[j,-1])))
        curves[l]=(mean,sem)
    for name,rows in [('cycles.csv',cycle_rows),('trajectories.csv',sample_rows)]:
        with (out/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':9,'mathtext.fontset':'cm',
                         'xtick.direction':'in','ytick.direction':'in','legend.frameon':False})
    fig,axes=plt.subplots(2,1,figsize=(3.375,4.8),layout='constrained')
    t=np.arange(41)
    for l,color,marker,ls in zip((20,30,40),('#d94738','#249b57','#1675bd'),('^','s','o'),(':','--','-')):
        if l not in curves: continue
        mean,sem=curves[l]
        label=rf'$L={l},\ S={counts[l]}$'+(' (partial)' if counts[l]<100 else '')
        style=dict(color=color,marker=marker,ls=ls,ms=2.6,mfc='white',lw=1,label=label)
        axes[0].plot(t,mean,**style);axes[0].fill_between(t,mean-sem,mean+sem,color=color,alpha=.15)
        delta=np.abs(mean-1)
        low=np.where(delta<=sem,0,np.minimum(abs(mean-sem-1),abs(mean+sem-1)))
        high=np.maximum(abs(mean-sem-1),abs(mean+sem-1))
        axes[1].plot(t,delta,**style)
        axes[1].fill_between(t,np.maximum(low,1e-6),high,color=color,alpha=.15)
    axes[0].axhline(1,color='.5',ls='--',lw=.8)
    axes[0].set(ylabel=r'$\overline{C_G}$',xlim=(0,10));axes[0].legend(loc='lower right',fontsize=8)
    axes[1].set(ylabel=r'$|\overline{C_G}-1|$',yscale='log',ylim=(1e-6,1.2),xlim=(0,40))
    for letter,a in zip('ab',axes):
        a.set_xlabel('cycle');a.tick_params(top=True,right=True)
        a.text(-.17,1.03,f'({letter})',transform=a.transAxes)
    for ext in ('pdf','png'): fig.savefig(out/f'square_chern_convergence.{ext}',dpi=300)
    plt.close(fig)
    summary=dict(config=plan['config'],identity=plan['identity'],source_files=sources,pending=pending,
                 statistics='Ten centers averaged within each trajectory; cycles21..40 also averaged within each trajectory; SEM=std(ddof=1)/sqrt(actual S). Partial sizes labeled separately.',
                 results=summaries,script=R.digest(Path(__file__)),
                 products={p.name:R.digest(p) for p in out.iterdir() if p.suffix in ('.csv','.pdf','.png')})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summaries,indent=2),flush=True)
    print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,action='append',default=[])
    args=parser.parse_args()
    if args.archive: import_archives(args.archive)
    analyze()
