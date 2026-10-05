"""Compare fixed-cycle versus circumference-scaled convergence; no new dynamics."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import analyze as source

HERE=Path(__file__).resolve().parent
OUT=HERE/'size_comparison_v1'


def relative_curve(samples):
    """Ratio of ensemble means with ordinary paired covariance-propagated SEM."""
    a=np.asarray(samples,dtype=float)
    if a.ndim!=2 or len(a)<2 or not np.isfinite(a).all():
        raise ValueError('Need finite sample-by-time data')
    mean=a.mean(0);reference=mean[-1]
    if reference<=0:raise ValueError('Positive reference gap required')
    ratio=mean/reference
    influence=(a-mean)/reference - (a[:,-1,None]-reference)*mean[None,:]/reference**2
    sem=influence.std(0,ddof=1)/np.sqrt(len(a))
    np.testing.assert_allclose(ratio[-1],1.,atol=1e-15)
    np.testing.assert_allclose(sem[-1],0.,atol=1e-15)
    return ratio,sem


def first_stable_reference_time(cycles,ratio,tolerance=.1):
    """First SAVED time remaining in a reference band until the final time.

    This is a point-estimate finite-window diagnostic, not an asymptotic
    convergence time or an uncertainty-bearing fitted dynamic exponent.
    """
    good=abs(np.asarray(ratio)-1)<=tolerance
    stable=np.logical_and.accumulate(good[::-1])[::-1]
    indices=np.flatnonzero(stable)
    return None if not len(indices) else int(np.asarray(cycles)[indices[0]])


def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    groups,inputs,sources=source.load_slab()
    OUT.mkdir(exist_ok=True)
    table=[];curves=[]
    for ny,row in sorted(groups.items()):
        cycles=row['cycles'];ratio,sem=relative_curve(row['rate'])
        row['ratio'],row['ratio_sem']=ratio,sem
        t90=first_stable_reference_time(cycles,ratio)
        summary=dict(Ny=ny,samples=100,reference_cycle=4*ny,
                     t90_reference=t90,t90_over_Ny=t90/ny)
        for name,time in [('t40',40),('tNy',ny),('t2Ny',2*ny),('t3Ny',3*ny)]:
            i=int(np.flatnonzero(cycles==time)[0])
            summary[name+'_percent_of_reference']=float(100*ratio[i])
            summary[name+'_percent_sem']=float(100*sem[i])
        table.append(summary)
        for i,t in enumerate(cycles):
            curves.append(dict(Ny=ny,cycle=int(t),scaled_cycle=float(t/ny),
                               ratio=float(ratio[i]),ratio_sem=float(sem[i])))
    write_csv('comparison_by_size.csv',table)
    write_csv('relative_gap_curves.csv',curves)
    source.style()
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.9),layout='constrained',sharey=True)
    colors=['#d62728','#2ca02c','#1f77b4','#ff7f0e','#9467bd','#222222','#17becf']
    markers=['^','s','o','v','D','P','>']
    for k,(ny,row) in enumerate(sorted(groups.items())):
        for ax,x in zip(axes,(row['cycles'],row['cycles']/ny)):
            mean,sem=row['ratio'],row['ratio_sem']
            ax.fill_between(x,mean-sem,mean+sem,color=colors[k],alpha=.1,lw=0)
            ax.plot(x,mean,color=colors[k],marker=markers[k],mfc='white',ms=3,
                    lw=1,ls=[':','--','-'][k%3],markevery=max(1,len(x)//9),label=rf'$N_y={ny}$')
    for k,ax in enumerate(axes):
        ax.axhline(1,color='.25',ls='--',lw=.8,zorder=0)
        ax.axhline(.9,color='.6',ls=':',lw=.8,zorder=0)
        ax.set_ylim(.3,1.08)
        ax.text(-.12,1.025,'('+chr(97+k)+')',transform=ax.transAxes)
    axes[0].axvline(40,color='.65',ls='--',lw=.8,zorder=0)
    axes[0].set(xlabel=r'physical cycle $t$',
                ylabel=r'$\langle\Delta(t)\rangle/\langle\Delta(4N_y)\rangle$')
    axes[1].set(xlabel=r'scaled cycle $t/N_y$',xticks=[0,1,2,3,4])
    axes[1].legend(frameon=False,ncol=2,loc='lower right')
    for ext in ('pdf','png'):fig.savefig(OUT/f'gap_convergence_size_comparison.{ext}',dpi=300)
    plt.close(fig)

    # Separate size summary: reference-relative shortfall, with paired SEMs.
    fig,ax=plt.subplots(figsize=(3.375,2.65),layout='constrained')
    sizes=[r['Ny'] for r in table]
    for key,label,color,marker in [('t40',r'$t=40$','#d62728','^'),
                                   ('t2Ny',r'$t=2N_y$','#2ca02c','s'),
                                   ('t3Ny',r'$t=3N_y$','#1f77b4','o')]:
        ax.errorbar(sizes,[100-r[key+'_percent_of_reference'] for r in table],
                    yerr=[r[key+'_percent_sem'] for r in table],color=color,marker=marker,
                    ls='--',mfc='white',ms=3.5,capsize=2,lw=1,label=label)
    ax.axhline(0,color='.4',ls=':',lw=.8)
    ax.set(xlabel=r'circumference $N_y$',ylabel=r'shortfall from $4N_y$ value (%)',
           xticks=sizes,ylim=(-2,55))
    ax.legend(frameon=False,ncol=3,loc='upper left')
    for ext in ('pdf','png'):fig.savefig(OUT/f'gap_reference_shortfall_vs_size.{ext}',dpi=300)
    plt.close(fig)
    caption=r'''\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{gap_convergence_size_comparison.pdf}
\caption{Finite-time gap convergence across circumferences.
Hard-wall slab-only purification at $N_x=20$, with $S=100$ independent Born
trajectories per size, maximally mixed slab initialization and Born-conditioned
exterior preparation, through $4N_y$ cycles.
The minimum absolute modular energy is taken within each trajectory,
divided by $2t$, and then averaged.
The ratio of ensemble means
$\langle\Delta(t)\rangle/\langle\Delta(4N_y)\rangle$ is shown against
(a) physical cycle and (b) scaled cycle.
Bands are ordinary sampling SEMs propagated with the within-trajectory
covariance between numerator and reference; no bootstrap or fit is used.
The dashed and dotted horizontal lines mark unity and $0.9$;
the vertical line in (a) marks $t=40$.
The $4N_y$ value is a finite-time reference, not an assumed asymptotic gap;
the common endpoint at unity is imposed by normalization.
These data are distinct from the full-measurement protocol.}
\label{fig:gap-convergence-size-comparison}
\end{figure*}
'''
    (OUT/'caption.tex').write_text(caption)
    manifest=dict(schema='gap_size_convergence_comparison_v1',protocol='campaign13_slab_only',
                  full_measurement_data_included=False,Nx=20,Ny_values=sorted(groups),samples_per_size=100,
                  reference='mean gap at t=4Ny, NOT an asymptotic estimate',
                  normalization='ratio of ensemble means, not mean of trajectory-wise ratios',
                  uncertainty='paired covariance-propagated ordinary trajectory SEM; no bootstrap',
                  t90='first saved time whose later saved means remain within 10% of final mean; point estimate',
                  fitted_dynamic_exponent=None,summary=table,inputs=inputs,executed_sources=sources,
                  source_sha256={str(p):source.sha(p) for p in [Path(__file__),HERE/'analyze.py']},
                  outputs={p.name:source.sha(p) for p in OUT.iterdir()
                           if p.suffix in ('.csv','.pdf','.png','.tex')})
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps(table,indent=2))


if __name__=='__main__':main()

