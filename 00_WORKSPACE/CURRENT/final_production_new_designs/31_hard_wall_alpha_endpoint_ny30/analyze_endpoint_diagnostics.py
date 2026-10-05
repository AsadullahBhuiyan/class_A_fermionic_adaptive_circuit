"""Downloaded campaign diagnostics; no simulations and no averaging sentinels."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_campaign import analyze, write_csv
from run_campaign import default_config, tasks, result_paths


def mean_sem(values):
    a=np.asarray(values,dtype=float)
    return (float(a.mean()) if len(a) else None,
            float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None)


def diagnostics(output,destination):
    output,destination=Path(output),Path(destination)
    base=analyze(output,destination)  # verifies all pairs, identities and spectra first
    samples=[];profiles=[];summary=[];timings=[]
    for task in sorted(tasks(default_config()),key=lambda t:t.alpha_1):
        group=[];group_profiles=[]
        for start in (0,5):
            path,_=result_paths(output,task,start)
            with np.load(path,allow_pickle=False) as z:
                # Half-gap consistency independently reconstructed from uncapped raw a.
                a=z['centered_spectrum_raw'];caps=z['cap_mask']
                costs=np.full(a.shape,np.inf)
                costs[~caps]=abs(np.arctanh(a[~caps]))/60
                np.testing.assert_allclose(costs.min(1),z['gap_raw'],rtol=8*np.finfo(float).eps,atol=0)
                # Only remove eigensolver roundoff outside the physical interval.
                nu=np.clip(z['occupation_spectrum_raw'],0,1)
                s=np.zeros_like(nu); interior=(nu>0)&(nu<1)
                s[interior]=-nu[interior]*np.log(nu[interior])-(1-nu[interior])*np.log1p(-nu[interior])
                x=z['active_coordinates_x_y_orbital'][:,0]
                for j,index in enumerate(z['sample_indices']):
                    valid=bool(z['gap_mode_valid'][j]);v=z['gap_mode_vector'][j]
                    px=np.bincount(x,weights=abs(v)**2,minlength=20) if valid else None
                    if valid:
                        np.testing.assert_allclose(px.sum(),1,atol=1e-10,rtol=0)
                        group_profiles.append(px)
                    row=dict(alpha_1=task.alpha_1,sample_index=int(index),
                             finite_gap=valid,gap_value=float(z['gap_value'][j]),
                             endpoint_entropy_nats=float(s[j].sum()),
                             finite_mode_count=int(z['finite_mode_count'][j]),
                             gap_tie_count=int(z['gap_tie_count'][j]),
                             wall_weight=float(px[5]+px[15]) if valid else None,
                             near_wall_weight=float(px[[5,6,14,15]].sum()) if valid else None,
                             eigensolver_residual=float(z['eigensolver_residual'][j]),
                             hermiticity_residual=float(z['hermiticity_residual'][j]),
                             spectral_bound_excess=float(z['spectral_bound_excess'][j]))
                    samples.append(row);group.append(row)
                    for column in range(20):
                        profiles.append(dict(alpha_1=task.alpha_1,sample_index=int(index),x=column,
                                             mode_valid=valid,probability=float(px[column]) if valid else None))
                if start==0:
                    timings.append(dict(alpha_1=task.alpha_1,dynamics_seconds=float(z['dynamics_seconds']),
                                        endpoint_seconds=float(z['endpoint_seconds'])))
        entropy,entropy_sem=mean_sem([r['endpoint_entropy_nats'] for r in group])
        wall,wall_sem=mean_sem([r['wall_weight'] for r in group if r['finite_gap']])
        near,near_sem=mean_sem([r['near_wall_weight'] for r in group if r['finite_gap']])
        summary.append(dict(alpha_1=task.alpha_1,samples=10,valid_modes=len(group_profiles),
                            tied_valid_modes=sum(r['gap_tie_count']>1 for r in group),
                            mean_endpoint_entropy_nats=entropy,sem_endpoint_entropy_nats=entropy_sem,
                            mean_wall_weight=wall,sem_wall_weight=wall_sem,
                            mean_near_wall_weight=near,sem_near_wall_weight=near_sem))
    write_csv(destination/'endpoint_diagnostics_samples.csv',samples)
    write_csv(destination/'endpoint_diagnostics_summary.csv',summary)
    write_csv(destination/'gap_mode_x_profiles.csv',profiles)
    write_csv(destination/'batch_timings.csv',timings)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':8,'mathtext.fontset':'cm'})
    fig,(ax,frac)=plt.subplots(2,1,figsize=(3.375,4.1),sharex=True,layout='constrained')
    for category,marker,color,label in [('full_ensemble','o','C0','All 10 finite: mean $\\pm$ SEM'),
                                       ('finite_subset','s','C1','Finite subset: mean $\\pm$ SEM')]:
        rows=[r for r in base['summary'] if r['statistic']==category]
        if rows:
            ax.errorbar([r['alpha_1'] for r in rows],[r['mean'] for r in rows],
                        yerr=[r['sem'] or 0 for r in rows],fmt=marker,color=color,mfc='white',ms=4,capsize=2,label=label)
    ax.set(ylabel=r'$\Delta$ (finite samples)',ylim=(0,None))
    for row in base['summary']:
        if row['statistic']=='finite_subset':
            ax.annotate(f"n={row['finite_samples']}", (row['alpha_1'],row['mean']),
                        xytext=(6,-8),textcoords='offset points',fontsize=6)
    ax.legend(frameon=False,fontsize=6.5,loc='upper left')
    frac.plot([r['alpha_1'] for r in base['summary']],[r['infinite_fraction'] for r in base['summary']],
              '^:',color='C3',mfc='white',ms=4)
    frac.set(xlabel=r'$\alpha_1$',ylabel='All modes capped / samples',ylim=(-.04,1.04),xlim=(.95,3.05),yticks=[0,.5,1])
    for axis,label in zip((ax,frac),('a','b')):
        axis.tick_params(which='both',direction='in',top=True,right=True)
        axis.text(-.18,1.02,f'({label})',transform=axis.transAxes)
    for ext in ('pdf','png'):fig.savefig(destination/f'gap_and_saturation.{ext}',dpi=300)
    plt.close(fig)
    fig,(ent,loc)=plt.subplots(2,1,figsize=(3.375,4.1),sharex=True,layout='constrained')
    ent.errorbar([r['alpha_1'] for r in summary],
                 [r['mean_endpoint_entropy_nats'] for r in summary],
                 yerr=[r['sem_endpoint_entropy_nats'] for r in summary],
                 fmt='o',color='C0',mfc='white',ms=4,capsize=2)
    ent.set(yscale='log',ylabel=r'$\langle S(T)\rangle$ (nats)')
    ent.text(.04,.06,r'$N_x=20,\ N_y=30,\ T=60$',transform=ent.transAxes,fontsize=7)
    for key,label,marker in [('wall','Walls: $x=5,15$','o'),
                             ('near_wall','Walls + adjacent interior columns','s')]:
        rows=[r for r in summary if r['valid_modes']]
        loc.errorbar([r['alpha_1'] for r in rows],[r[f'mean_{key}_weight'] for r in rows],
                     yerr=[r[f'sem_{key}_weight'] or 0 for r in rows],fmt=marker,mfc='white',
                     ms=4,capsize=2,label=label)
    loc.set(xlabel=r'$\alpha_1$',ylabel='Gap-mode probability weight',ylim=(0,1.05),xlim=(.95,3.05))
    loc.legend(frameon=False,fontsize=6,loc='upper right')
    loc.text(.98,.04,'Valid modes only; mean $\\pm$ SEM',ha='right',transform=loc.transAxes,fontsize=6)
    for axis,label in zip((ent,loc),('a','b')):
        axis.tick_params(which='both',direction='in',top=True,right=True)
        axis.text(-.18,1.02,f'({label})',transform=axis.transAxes)
    for ext in ('pdf','png'):fig.savefig(destination/f'endpoint_entropy_and_localization.{ext}',dpi=300)
    plt.close(fig)
    report=dict(samples=len(samples),configurations=len(summary),verified_pairs=42,
                finite_gaps=sum(r['finite_gap'] for r in samples),
                infinite_gaps=sum(not r['finite_gap'] for r in samples),
                max_eigensolver_residual=max(r['eigensolver_residual'] for r in samples),
                max_hermiticity_residual=max(r['hermiticity_residual'] for r in samples),
                max_spectral_bound_excess=max(r['spectral_bound_excess'] for r in samples),
                cap_resolved_rate_limit=float(np.arctanh(1-1e-9)/60),
                total_dynamics_seconds=sum(r['dynamics_seconds'] for r in timings),
                gap_summary=base['summary'],endpoint_summary=summary,
                caveats=['100 is a display sentinel, never a measured finite rate.',
                         'Finite-subset means are conditional when some samples are all-capped.',
                         'One size/time and 10 samples per alpha do not establish thermodynamic gap closure.',
                         'Mode weights describe selected gap vectors only; ties can be basis dependent.',
                         'Endpoint entropy is reconstructed from raw occupations, removing only bound roundoff.',
                         'Batch timing is counted once per alpha, not once per five-sample shard.'])
    (destination/'diagnostic_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('gap_summary','endpoint_summary')},indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    diagnostics(a.output_root,a.destination)
