#!/usr/bin/env python3
"""Matched Ny28 alpha1=1/3 endpoint x-resolved correlators; no new dynamics."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
import numpy as np

import analyze_finite_size_power_law as source
from large_ny_correlator_data import fit
from build_large_ny_correlator_review import write_csv, write_json

NY = 28
SITES = (5, 6, 14, 15)
COLORS = ('#0072B2', '#E69F00', '#009E73', '#D55E00')
MARKERS = ('o', 's', '^', 'v')
OUT = Path(__file__).resolve().parent/'outputs/alpha1_comparison_Ny028_nshell1'
STEM = 'hard_wall_xresolved_alpha1_1_vs_3'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source.configure_matplotlib()
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,
                         'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8})
    fig, axes = plt.subplots(1,2,figsize=(7.05,3.35),sharex=True,sharey=True)
    r = np.arange(1,NY//2+1)
    log_d = np.log(source.chord(NY,r))
    bounds = np.log(source.chord(NY,np.array([2,NY//4])))
    tables, summaries, provenance, compact = [], {}, [], {}
    recorded_sources = None
    for panel,(ax,alpha) in enumerate(zip(axes,(1,3))):
        avg,x,ids,records = source.load_new_size(NY,alpha_1=alpha)
        # Match recorded implementation identities, not just nominal parameters.
        for rec in records:
            if rec['role']=='new_completion':
                receipt=json.loads((source.REPO_ROOT/rec['path']).read_text())
                identity=(receipt['source_hashes'],receipt['observer_schema'],receipt['sampling_revision'])
                if recorded_sources is None:recorded_sources=identity
                if identity!=recorded_sources:raise ValueError('Implementation identities differ across comparison inputs')
        provenance.extend(records)
        compact[f'alpha1_{alpha}_xresolved']=x
        compact[f'alpha1_{alpha}_sample_ids']=ids
        compact[f'alpha1_{alpha}_xavg']=avg
        summaries[str(alpha)]={}
        ax.axvspan(*bounds,color='0.5',alpha=.15,lw=0,zorder=0)
        for site,color,marker in zip(SITES,COLORS,MARKERS):
            mean=x[:,site].mean(axis=0)
            if not np.all(np.isfinite(mean)) or np.any(mean[1:]<=0):
                raise ValueError('Nonpositive or nonfinite mean correlator')
            model=fit(mean,NY,2,NY//4)
            if not model['valid']:raise ValueError('Insufficient points in comparison fit')
            summaries[str(alpha)][str(site)]=model
            ax.plot(log_d,np.log(mean[1:]),ls='none',marker=marker,color=color,
                    mfc='white',mew=.8,ms=3.4,label=rf'$x={site}$',zorder=3)
            xx=np.linspace(log_d[0],log_d[-1],100)
            ax.plot(xx,model['log_amplitude']-model['beta']*xx,'k--',lw=.85,zorder=2)
            for ry,dd,c in zip(r,log_d,mean[1:]):
                tables.append(dict(alpha_1=alpha,Ny=NY,x=site,ry=int(ry),log_chord=float(dd),
                                   mean_correlator=float(c),log_mean_correlator=float(np.log(c)),
                                   in_fit_window=bool(2<=ry<=NY//4),
                                   fit_log_correlator=model['log_amplitude']-model['beta']*float(dd),
                                   mean_curve_slope=-model['beta'],fit_r_squared=model['r_squared']))
        ax.set_title(rf'$\alpha_1={alpha}$')
        ax.set_xlabel(r'$\log d_{N_y}(r_y)$')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_minor_locator(NullLocator());ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(direction='in',top=True,right=True)
        ax.text(-.12,1.04,f'({chr(97+panel)})',transform=ax.transAxes,fontsize=10)
    axes[0].set_ylabel(r'$\log\overline{C_G(x,r_y)}$')
    handles,labels=axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([],[],color='black',ls='--',lw=.85));labels.append('power-law fit')
    fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,
               columnspacing=1.3,handletextpad=.4,bbox_to_anchor=(.52,0))
    fig.suptitle(r'$20\times28$, $n_{\rm shell}=1$, $S=100$, $t=56$; hard walls',fontsize=9,y=.99)
    fig.subplots_adjust(left=.09,right=.985,bottom=.23,top=.83,wspace=.12)
    for ext in ('pdf','png'):fig.savefig(OUT/f'{STEM}.{ext}',dpi=300)
    plt.close(fig)
    write_csv(OUT/f'{STEM}_curves.csv',tables)
    np.savez_compressed(OUT/'endpoint_curves.npz',**compact)
    caption=(
        'Matched hard/support-truncated Nx=20, Ny=28, alpha_2=30, nshell=1 ensembles, '
        'alpha_1=1 (left) and 3 (right). Each has 100 independent pure-initialized, '
        'raster-y perfect-correction trajectories in complex128, evaluated at cycle 56=2Ny. '
        'Open markers show the natural log of the arithmetic trajectory mean of the '
        'x-resolved squared correlator, at x=5,6,14,15, with common axes. No SEM or error bands. '
        'Gray marks r=2..7. Dashed lines are unconstrained OLS fits of log(mean C_G) '
        'against log chord in that window, extended beyond it as extrapolations. '
        'They are comparison diagnostics, not an assertion of power-law behavior at alpha_1=3. '
        'All independent separations r=1..14 are shown, including very small positive values; '
        'no numerical-accuracy floor is certified by this plot. '
        'Ensembles are not paired by sample index. Raw data, previous figures and manuscript are unchanged.'
    )
    (OUT/'caption.md').write_text(caption+'\n')
    write_json(OUT/'summary.json',dict(Nx=20,Ny=NY,alpha_1=[1,3],alpha_2=30,nshell=1,
               cycles=56,samples_per_alpha=100,fit_window=[2,7],x_columns=SITES,
               fit_to_mean_curve=summaries,inputs=provenance,
               shared_implementation_identity=recorded_sources,
               source_script=dict(path=str(Path(__file__).resolve()),sha256=source.sha256(Path(__file__))),
               loader_sha256=source.sha256(Path(source.__file__)),caption=caption))
    print(json.dumps(summaries,indent=2))
    print(f'[done] {OUT/STEM}.pdf and .png')


if __name__=='__main__':main()
