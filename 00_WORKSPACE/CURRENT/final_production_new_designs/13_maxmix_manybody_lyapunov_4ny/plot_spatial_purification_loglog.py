#!/usr/bin/env python3
"""Standalone log-log rendition of purification panel (b), unchanged data."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'analysis_outputs/purification_contour_gap_3x1_v3_single_column/spatial_entropy_curves.csv'
OUT = ROOT/'analysis_outputs/spatial_purification_panel_b_loglog_v1'


def main(*, cutoff=4, figsize=(3.375,2.9), fontsize=8, output=OUT, normalized=True, xscale='log'):
    assert xscale in ('log','linear')
    output.mkdir(parents=True, exist_ok=True)
    with SOURCE.open() as f:
        rows = [{k: float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    plt.rcParams.update({'font.family':'CMU Sans Serif', 'font.size':fontsize,
                         'mathtext.fontset':'cm', 'axes.linewidth':.8})
    fig,ax = plt.subplots(figsize=figsize)
    colors={20:'#d62728',30:'#2ca02c',40:'#1f77b4'}
    marks={20:'^',30:'s',40:'o'}
    styles={5:'-',15:'--',10:':'}
    plotted=0
    plotted_rows=[]
    for ny in (20,30,40):
        for x in (5,15,10):
            selected=sorted((r for r in rows if r['Ny']==ny and r['x']==x and 0<r['cycle']<=cutoff*ny),key=lambda r:r['cycle'])
            assert len(selected)==cutoff*ny and all(r['samples']==100 for r in selected)
            u=np.array([r['normalized_cycle'] for r in selected])
            mean=np.array([r['mean_entropy_x_over_Ny'] for r in selected])
            sem=np.array([r['sem'] for r in selected])
            if not normalized:
                u=np.array([r['cycle'] for r in selected])
                mean=mean*ny
                sem=sem*ny
            for row,t,m,e in zip(selected,u,mean,sem):
                plotted_rows.append(dict(Ny=ny,x=x,cycle=int(row['cycle']),abscissa=float(t),
                                         mean=float(m),sample_sem=float(e),samples=100))
            assert np.all(u>0) and np.all(mean>0) and np.all(sem>=0)
            ax.fill_between(u,np.maximum(mean-sem,1e-12),mean+sem,color=colors[ny],alpha=.08,lw=0)
            # Sparse logarithmically spaced markers; all cycle data are connected.
            markers=np.unique(np.rint(np.geomspace(1,len(u),10)).astype(int)-1)
            if xscale=='linear':
                markers=np.unique(np.rint(np.linspace(0,len(u)-1,10)).astype(int))
            ax.plot(u,mean,color=colors[ny],ls=styles[x],lw=1.2,marker=marks[ny],
                    markevery=markers.tolist(),ms=3.2*fontsize/8,mfc='white',mew=.8,alpha=.7 if x==10 else 1)
            plotted+=len(u)
    ax.set(xscale=xscale,yscale='log',xlim=((1/40 if xscale=='log' else 0),cutoff) if normalized else ((1 if xscale=='log' else 0),40*cutoff),
           ylim=(1e-6,2) if normalized else (1e-5,80),
           xlabel=r'cycle $t/N_y$' if normalized else r'cycle $t$',
           ylabel=r'$\langle s_x(t)\rangle/N_y$' if normalized else r'$\langle s_x(t)\rangle$ (nats)')
    legend=ax.legend(handles=[Line2D([],[],color=colors[n],marker=marks[n],mfc='white',ms=4,
                      label=rf'$N_y={n}$') for n in (20,30,40)],loc='upper right',frameon=False,
                      handlelength=1.4,labelspacing=.15)
    ax.add_artist(legend)
    ax.legend(handles=[Line2D([],[],color='.2',ls=styles[x],label=rf'$x={x}$') for x in (5,15,10)],
              loc='lower left',frameon=False,handlelength=1.4,labelspacing=.15)
    ax.tick_params(which='both',direction='in',top=True,right=True)
    ticks=[v for v in (.03,.1,.3,1,2,4) if v<=cutoff] if normalized else [v for v in (1,2,5,10,20,40,60,80,160) if v<=40*cutoff]
    if xscale=='linear':
        ticks=np.linspace(0,cutoff if normalized else 40*cutoff,9)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.text(-.1 if figsize[0]>5 else -.19,1.025,'(b)',transform=ax.transAxes,fontsize=fontsize+1)
    fig.tight_layout(pad=.6)
    stem='hard_wall_spatial_purification_loglog' if xscale=='log' else 'hard_wall_spatial_purification_semilog'
    for ext in ('pdf','png'):
        fig.savefig(output/f'{stem}.{ext}',dpi=300)
    plt.close(fig)
    with (output/'plotted_curves.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(plotted_rows[0]))
        writer.writeheader();writer.writerows(plotted_rows)
    (output/'summary.json').write_text(json.dumps(dict(source=str(SOURCE),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        plotted_points=plotted,omitted='Cycle zero only; preserve the same data as the log-log version.',
        estimator=('Mean of y-summed entropy contour / Ny, sample SEM, S=100.' if normalized else
                   'Mean of y-summed entropy contour in nats, sample SEM, S=100; both mean and SEM multiplied by Ny from source table.'),
        normalized_axes=normalized,
        source_bundle=7,Nx=20,Ny=[20,30,40],x=[5,15,10],alpha_1=1,alpha_2=30,
        protocol='Hard wall, maxmix initialization, Born sampling, perfect correction; original trajectories T=4Ny',
        plotted_time_window=f'0<t/Ny<={cutoff}',
        figure_inches=list(figsize),axes={'x':xscale,'y':'log'},fits='none'),indent=2)+'\n')
    print(output)


if __name__=='__main__':
    main()
