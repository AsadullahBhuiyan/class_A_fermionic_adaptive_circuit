"""Mean-first growth fits at each fixed size; finite-window diagnostics only."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import analyze as source

HERE=Path(__file__).resolve().parent
OUT=HERE/'cycle_growth_fits_v1'


def r2(y,prediction):
    denominator=np.sum((y-y.mean())**2)
    return float(1-np.sum((y-prediction)**2)/denominator) if denominator else 1.


def mean_first_fit(cycles,samples,start,stop,model):
    """OLS of the mean; uncertainty retains paired trajectory time covariance."""
    t=np.asarray(cycles,dtype=float)
    values=np.asarray(samples,dtype=float)
    mask=(t>=start)&(t<=stop)
    t=t[mask];values=values[:,mask]
    if values.ndim!=2 or len(values)<2 or len(t)<3 or np.any(t<=0):
        raise ValueError('Need >=2 trajectories and >=3 positive saved times')
    if not np.isfinite(values).all() or np.any(values<=0):
        raise ValueError('Positive finite modular gaps required')
    mean=values.mean(0)
    x=np.log(t) if model=='power' else t
    if model not in ('power','affine'):raise ValueError(model)
    design=np.column_stack((x,np.ones(len(x))))
    linear_map=np.linalg.pinv(design)
    y=np.log(mean) if model=='power' else mean
    params=linear_map@y
    # Delta method for log(mean); exact covariance propagation for raw-mean OLS.
    deviations=(values-mean)/mean if model=='power' else values-mean
    influence=deviations@linear_map.T
    covariance=np.cov(influence,rowvar=False,ddof=1)/len(values)
    errors=np.sqrt(np.maximum(np.diag(covariance),0))
    prediction=np.exp(design@params) if model=='power' else design@params
    row=dict(model=model,requested_start=float(start),requested_stop=float(stop),
        first_cycle=int(t[0]),last_cycle=int(t[-1]),points=len(t),samples=len(values),
        coefficient=float(params[0]),coefficient_sem=float(errors[0]),
        intercept=float(params[1]),intercept_sem=float(errors[1]),
        coefficient_intercept_covariance=float(covariance[0,1]),
        r2_raw=r2(mean,prediction),r2_fit_space=r2(y,design@params),
        raw_rmse=float(np.sqrt(np.mean((mean-prediction)**2))))
    if model=='power':
        row.update(amplitude=float(np.exp(params[1])),
            amplitude_sem=float(np.exp(params[1])*errors[1]),
            rate_exponent=float(params[0]-1),rate_exponent_sem=float(errors[0]))
        # The same log-OLS fit to mean g/(2t) has exponent p-1 exactly.
        rate_params=linear_map@np.log(mean/(2*t))
        np.testing.assert_allclose(rate_params[0],params[0]-1,atol=1e-12)
    else:
        row.update(local_rate_from_slope=float(params[0]/2),
            local_rate_from_slope_sem=float(errors[0]/2))
        np.testing.assert_allclose(params,(values@linear_map.T).mean(0),atol=1e-12)
    return row,influence


def windows(protocol,ny,end):
    if protocol=='slab_only':
        return [('all_positive',4,end),('common_20_60',20,60),
                ('Ny_to_end',ny,end),('primary',2*ny,end),('last_quarter',3*ny,end)]
    return [('all_positive',1,end),('common_20_60',20,60),
            ('primary',30,end),('last_quarter',45,end)]


def save_csv(name,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with (OUT/name).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def fit_all(protocol,groups):
    fits=[];changes=[];curves=[]
    for ny,data in sorted(groups.items()):
        t=data['cycles'];raw=data['raw'];mean,sem=source.mean_sem(raw)
        for i,cycle in enumerate(t):
            curves.append(dict(protocol=protocol,Ny=ny,cycle=int(cycle),mean_raw=float(mean[i]),
                raw_sem=float(sem[i]),mean_rate=float(mean[i]/(2*cycle)),rate_sem=float(sem[i]/(2*cycle))))
        cache={}
        for name,start,stop in windows(protocol,ny,int(t[-1])):
            for model in ('power','affine'):
                row,influence=mean_first_fit(t,raw,start,stop,model)
                row.update(protocol=protocol,Ny=ny,window=name)
                fits.append(row);cache[name,model]=(row,influence)
        for model in ('power','affine'):
            first,fi=cache['primary',model];last,li=cache['last_quarter',model]
            change=last['coefficient']-first['coefficient']
            sem_change=float((li[:,0]-fi[:,0]).std(ddof=1)/np.sqrt(len(raw)))
            changes.append(dict(protocol=protocol,Ny=ny,model=model,
                last_quarter_minus_primary=change,paired_sem=sem_change))
    return fits,changes,curves


def plot(groups,fits):
    source.style()
    fig,axes=plt.subplots(4,2,figsize=(7.05,8.0),layout='constrained')
    for index,(ny,data) in enumerate(sorted(groups.items())):
        ax=axes.flat[index];t=data['cycles'];mean,sem=source.mean_sem(data['raw'])
        rows={r['model']:r for r in fits if r['protocol']=='slab_only' and r['Ny']==ny and r['window']=='primary'}
        linear,power=rows['affine'],rows['power']
        ax.fill_between(t,mean-sem,mean+sem,color='#1f77b4',alpha=.15,lw=0)
        ax.plot(t,mean,'o',color='#1f77b4',mfc='white',ms=2.7,mew=.7)
        xx=np.linspace(linear['first_cycle'],linear['last_cycle'],200)
        ax.plot(xx,linear['coefficient']*xx+linear['intercept'],'--',color='.2',lw=1.1)
        ax.plot(xx,power['amplitude']*xx**power['coefficient'],':',color='#d62728',lw=1.1)
        ax.axvline(2*ny,color='.7',ls=':',lw=.7)
        ax.text(.04,.95,rf'$N_y={ny}$',transform=ax.transAxes,va='top')
        ax.text(.96,.05,rf"$p={power['coefficient']:.2f}\pm{power['coefficient_sem']:.2f}$",
            transform=ax.transAxes,ha='right',va='bottom',fontsize=8)
        ax.set(xlabel=r'cycle $t$',ylabel=r'$\langle g_{\rm mod}(t)\rangle$',ylim=(0,None),xlim=(0,t[-1]*1.025))
        ax.text(-.16,1.02,'('+chr(97+index)+')',transform=ax.transAxes)
    ax=axes.flat[7];ax.axis('off')
    ax.text(.03,.90,'Fixed-size growth fits\n\n'
        r'Blue: mean $\pm$ trajectory SEM'+'\n'
        r'Dashed: $at+b$'+'\n'+r'Dotted: $At^p$'+'\n\n'
        r'Fits restricted to $2N_y\leq t\leq4N_y$.'+'\n'
        '100 trajectories per size; slab-only.\n'
        'Finite-window fits, not asymptotic laws.',transform=ax.transAxes,va='top',fontsize=8)
    for ext in ('pdf','png'):fig.savefig(OUT/f'raw_gap_growth_by_size.{ext}',dpi=300)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.8),layout='constrained')
    colors=['#d62728','#2ca02c','#1f77b4','#ff7f0e','#9467bd','#222222','#17becf']
    markers=['^','s','o','v','D','P','>']
    for idx,(ny,data) in enumerate(sorted(groups.items())):
        t=data['cycles'];mean,sem=source.mean_sem(data['rate'])
        fit=next(r for r in fits if r['protocol']=='slab_only' and r['Ny']==ny and r['window']=='primary' and r['model']=='affine')
        for ax,scale in zip(axes,(1,ny)):
            ax.fill_between(t/scale,mean-sem,mean+sem,color=colors[idx],alpha=.10,lw=0)
            ax.plot(t/scale,mean,color=colors[idx],marker=markers[idx],ms=3,mfc='white',
                lw=.7,markevery=max(1,len(t)//8),label=rf'$N_y={ny}$')
            xx=np.linspace(fit['first_cycle'],fit['last_cycle'],150)
            ax.plot(xx/scale,fit['coefficient']/2+fit['intercept']/(2*xx),color=colors[idx],ls='--',lw=1.3)
    axes[0].set(xlabel=r'cycle $t$',ylabel=r'$\langle\Delta(t)\rangle$',ylim=(0,None))
    axes[1].set(xlabel=r'$t/N_y$',ylabel=r'$\langle\Delta(t)\rangle$',ylim=(0,None))
    axes[0].legend(frameon=False,ncol=2,fontsize=7,loc='upper right')
    for ax,label in zip(axes,'ab'):ax.text(-.15,1.02,f'({label})',transform=ax.transAxes)
    for ext in ('pdf','png'):fig.savefig(OUT/f'rate_with_affine_growth_fits.{ext}',dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    slab,slab_inputs,slab_sources=source.load_slab()
    full,full_inputs,full_sources=source.load_full()
    fits=[];changes=[];curves=[]
    for protocol,groups in [('slab_only',slab),('full_measurement',full)]:
        ff,cc,rr=fit_all(protocol,groups);fits+=ff;changes+=cc;curves+=rr
    save_csv('fits.csv',fits);save_csv('paired_window_changes.csv',changes);save_csv('mean_curves.csv',curves)
    plot(slab,fits)
    primary=[r for r in fits if r['protocol']=='slab_only' and r['window']=='primary']
    lines=[]
    for ny in sorted(slab):
        p=next(r for r in primary if r['Ny']==ny and r['model']=='power')
        a=next(r for r in primary if r['Ny']==ny and r['model']=='affine')
        lines.append(f"| {ny} | {p['first_cycle']}–{p['last_cycle']} | {p['coefficient']:.3f} ± {p['coefficient_sem']:.3f} | {a['coefficient']:.5f} ± {a['coefficient_sem']:.5f} | {a['intercept']:.3f} ± {a['intercept_sem']:.3f} | {a['r2_raw']:.5f} |")
    (OUT/'README.md').write_text('''# Growth with cycle at each fixed system size

No dynamics or manuscript edits. Campaign 13: Nx20, Ny20,24,30,36,44,56,60,
100 independent Born trajectories each, hard walls, alpha1=1, alpha2=30,
nshell1, raster-y, complex128, perfect correction, slab-only measurements
with Born-conditioned exterior. All 140 pairs validated. The separate
full-measurement Ny30 Campaign 21 (20 pairs) is included in CSVs only and
never pooled with slab-only data.

Take minimum absolute modular energy inside each sample at each cycle,
excluding capped occupations, then average the 100 resulting gap histories.
Fits use the ensemble-mean curve, not an averaged occupation spectrum.
All saved times in the chosen interval have equal OLS weight.

Two descriptive models: mean g(t)=A*t**p (OLS on log of the mean), and
mean g(t)=a*t+b (OLS on the raw mean, with free intercept). Ordinary
sampling SEMs propagate the full trajectory covariance across time:
exactly for affine fits, first-order delta method for log-mean fits.
No bootstrap, independent-time assumption, or residual-based parameter error.
R² is descriptive; high R² alone does not establish either asymptotic model.

Primary window: last half of each slab-only history, 2Ny..4Ny. Sensitivity
windows: all positive saved times, common physical cycles20..60, Ny..4Ny,
and 3Ny..4Ny. Separate full-measurement Ny30 primary window is30..60.
Paired changes in fit coefficients retain the same trajectories across windows.

| Ny | Primary cycles | Power p | Affine a | Affine b | Affine raw R² |
|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(lines)+'''

The corresponding finite-time rate is Delta=g/(2t). A power fit implies
Delta ~ t**(p-1) only within that fitted interval. An affine raw-gap fit
implies Delta=a/2+b/(2t): a negative b allows a rising rate even if the
raw growth slope is constant. a/2 is a LOCAL slope-derived rate, not a
validated infinite-time extrapolation. Fits are displayed only within the
observed fit windows, not extrapolated. Positive fitted p>1 does not establish
permanent superlinear growth: a negative-intercept affine law can mimic it.

raw_gap_growth_by_size.pdf/png: each size separately, mean±SEM, both fits.
rate_with_affine_growth_fits.pdf/png: measured rates with the affine fits
divided by 2t, shown against both physical and scaled cycle. All times and
uncertainties retain the original sampling units. Previous figures unchanged.
Reproduce: python fit_cycle_growth.py.
''')
    manifest=dict(schema='fixed_size_cycle_growth_fits_v1',
        primary_window='2Ny..4Ny, slab-only; 30..60, separate full-measurement Ny30',
        estimator='fit ensemble mean of per-trajectory modular minima',
        errors='full paired temporal covariance, ordinary sampling SEM; no bootstrap',
        models=['A*t**p','a*t+b'],fits=fits,window_changes=changes,
        inputs=slab_inputs+full_inputs,slab_sources=slab_sources,full_sources=full_sources,
        source_hashes={str(p):source.sha(p) for p in [Path(__file__),HERE/'analyze.py']},
        outputs={p.name:source.sha(p) for p in OUT.iterdir() if p.name!='analysis_manifest.json'})
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print('\n'.join(lines));print(OUT)


if __name__=='__main__':main()
