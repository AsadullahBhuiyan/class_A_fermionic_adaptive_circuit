#!/usr/bin/env python3
"""Reference panels a,c,d: log-log entropy/gap and Ny=40 minimum-mode map."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, MaxNLocator
import numpy as np

from plot_purification_control_minmode import ROOT, OUT as PREVIOUS, minimum_mode_density
from plot_purification_gap_density_summary import SOURCE, read_rows, sha
from plot_endpoint_lyapunov_gap import configure_plotting, write_csv, weighted_power_law

OUT = ROOT/'analysis_outputs/purification_gap_minmode_Ny40_3x1_loglog_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    entropy_path = PREVIOUS/'total_entropy_alpha_comparison.csv'
    with entropy_path.open() as f:
        entropy = [{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
    gaps = read_rows('gap_summary.csv')
    fit = json.loads((PREVIOUS/'analysis_summary.json').read_text())['fit']
    recalculated = weighted_power_law(*[np.array([r[key] for r in gaps])
                                      for key in ('Ny','mean_gap','sample_sem')])
    for key in fit:
        np.testing.assert_allclose(fit[key], recalculated[key], rtol=1e-10)
    mean, densities, records, mode_inputs = minimum_mode_density(ny=40)
    assert densities.shape == (100,40,20)
    np.testing.assert_allclose(densities.sum(axis=(1,2)), 1, atol=1e-12)
    configure_plotting()
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':8,
                         'xtick.labelsize':8,'ytick.labelsize':8})
    fig, (a,b,c) = plt.subplots(3,1,figsize=(3.375,5.45),layout='constrained')
    fig.get_layout_engine().set(h_pad=.035,w_pad=.025,hspace=.025)
    for alpha,color,marker,style in [(1,'#d62728','^',':'),(3,'#1f77b4','o','-')]:
        rows=sorted([r for r in entropy if r['alpha_1']==alpha and r['cycle']>0],
                    key=lambda r:r['cycle'])
        assert len(rows)==160 and all(r['Ny']==40 and r['samples']==100 for r in rows)
        t,m,e = [np.array([r[k] for r in rows])
                 for k in ('normalized_cycle','mean_entropy_over_Ny','sem')]
        a.fill_between(t,np.maximum(m-e,1e-12),m+e,color=color,alpha=.1,lw=0)
        marks=np.unique(np.rint(np.geomspace(1,len(t),11)).astype(int)-1)
        a.plot(t,m,color=color,ls=style,marker=marker,markevery=marks,
               lw=1.2,ms=3.2,mfc='white',mew=.8,label=rf'$\alpha_1={alpha}$')
    a.set(xscale='log',yscale='log',xlim=(1/40,4),
          xlabel=r'cycle $t/N_y$',ylabel=r'$\langle S(t)\rangle/N_y$')
    a.xaxis.set_major_locator(FixedLocator([.03,.1,.3,1,4]))
    a.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
    a.legend(loc='lower left',frameon=False)
    a.text(.96,.58,r'$N_y=40$',ha='right',transform=a.transAxes)
    ny=np.array([r['Ny'] for r in gaps])
    m=np.array([r['mean_gap'] for r in gaps]); e=np.array([r['sample_sem'] for r in gaps])
    grid=np.geomspace(ny.min(),ny.max(),300)
    b.plot(grid,fit['amplitude']*grid**(-fit['exponent']),ls='--',color='.25',lw=1,
           label=rf"$A N_y^{{-z}},\ z={fit['exponent']:.2f}\pm{fit['exponent_sem']:.2f}$")
    b.errorbar(ny,m,yerr=e,color='#1f77b4',marker='o',ls='none',mfc='white',
               ms=3.7,capsize=2,lw=1,label=r'$T=4N_y$: mean $\pm$ SEM')
    b.set(xscale='log',yscale='log',xlim=(19,63),ylim=(.013,.1),
          xlabel=r'circumference $N_y$',ylabel=r'$\Delta$')
    b.xaxis.set_major_locator(FixedLocator(ny))
    b.yaxis.set_major_locator(FixedLocator([.015,.02,.03,.04,.06,.1]))
    for axis in (b.xaxis,b.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
    b.legend(loc='upper right',frameon=False,fontsize=7.5,handlelength=1.5,labelspacing=.25)
    for ax in (a,b):
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    im=c.imshow(mean,origin='lower',aspect='auto',interpolation='nearest',
                extent=(-.5,19.5,-.5,39.5),cmap='magma',vmin=0,vmax=mean.max())
    c.set(xlabel='$x$',ylabel='$y$',xticks=[0,5,10,15,19],yticks=[0,10,20,30,39])
    c.text(.5,.97,r'$N_y=40$',color='white',ha='center',va='top',transform=c.transAxes)
    cb=fig.colorbar(im,cax=c.inset_axes([1.025,0,.035,1]))
    cb.locator=MaxNLocator(nbins=3); cb.update_ticks()
    cb.set_label(r'$\overline{p_{\min}(x,y)}$',labelpad=2)
    cb.ax.tick_params(labelsize=8,pad=2)
    for ax,label in zip((a,b,c),'abc'):
        ax.text(-.18,1.035,f'({label})',transform=ax.transAxes,fontsize=9)
    fig.canvas.draw()
    assert all(ax.get_xscale()==ax.get_yscale()=='log' for ax in (a,b))
    for suffix in ('pdf','png'):
        fig.savefig(OUT/f'hard_wall_purification_gap_minmode_3x1.{suffix}',dpi=300)
    plt.close(fig)
    write_csv(OUT/'total_entropy_alpha_comparison.csv',entropy)
    write_csv(OUT/'gap_summary.csv',gaps)
    write_csv(OUT/'selected_minimum_modes.csv',records)
    np.savez_compressed(OUT/'Ny040_minimum_mode_density.npz',mean_density=mean,
                        sample_densities=densities,sample_indices=np.arange(100),cycles=160,
                        sem_density=densities.std(axis=0,ddof=1)/10)
    inputs=[dict(path=str(p),sha256=sha(p)) for p in
            (entropy_path,SOURCE/'gap_summary.csv',PREVIOUS/'analysis_summary.json')]
    summary=dict(figure_size_inches=[3.375,5.45],original_panels=['a','c','d'],
                 new_panels=['a','b','c'],endpoint_time='4Ny',mode_Ny=40,
                 mode_samples=100,mode_estimator='One unique argmin |lambda| mode per sample; sum orbital probabilities then average samples',
                 probability_sum=float(mean.sum()),uncertainty='sample SEM; no bootstrap',
                 fit=fit,axes=['log-log','log-log','linear x-y; linear color'],
                 omitted_cycle_zero=True,inputs=inputs+mode_inputs,
                 alpha3_entropy_floor_note='Entropy tail is the existing observer clipping floor, not residual mixedness.',
                 script_sha256=sha(Path(__file__)))
    (OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(OUT,flush=True)


if __name__=='__main__':
    main()
