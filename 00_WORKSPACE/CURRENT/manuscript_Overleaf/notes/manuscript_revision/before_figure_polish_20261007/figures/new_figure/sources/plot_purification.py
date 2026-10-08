#!/usr/bin/env python3
"""Render purification spectra and separate Lyapunov diagnostics from saved data."""
import csv, json, hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, MaxNLocator, FixedLocator, FuncFormatter
import numpy as np
from manuscript_palette import ALPHA_COLORS
from manuscript_typography import configure_style as manuscript_style, prepare_figure, record_typography
OUT = Path(__file__).resolve().parents[1]
DATA = OUT/'data/purification'

def configure_style():
    manuscript_style({'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'pdf.fonttype': 42, 'ps.fonttype': 42})

def plot(e1, e3, sx, density, gaps, fit, *, gap_time_label=r'$t=2N_y$',
         output_dir=None, stem='Figure_04_purification',
         raw_cycles=False, gap_loglog=False, heatmap_time_annotation=True, panels="combined"):
    configure_style()
    assert panels in ('entropy', 'lyapunov', 'combined')
    fig, axes = plt.subplots(3 if panels == 'combined' else 2, 1,
                             figsize=(3.375, 5.45 if panels == 'combined' else 3.65),
                             layout=None)
    # Reserve the same color-bar gutter for every row. Fixed shared anchors keep
    # PDF and PNG axes aligned across renderer-dependent text measurements.
    fig.subplots_adjust(left=.22,right=.81,bottom=.115,top=.935,hspace=.47)
    if panels == "combined":
        a,c,d = axes
    elif panels == "entropy":
        a,b = axes
    else:
        c,d = axes
    times = np.arange(61) if raw_cycles else np.arange(61)/30
    cycle_limits = (1, 60) if raw_cycles else (1/30, 2)
    cycle_label = r'cycle $t$' if raw_cycles else r'cycle $t/N_y$'
    positive = times > 0
    marker_indices = np.unique(np.rint(np.geomspace(1,60,10)).astype(int)-1).tolist()
    entropy_rows, spatial_rows = [], []
    if panels in ("entropy", "combined"):
        for alpha, samples, color, marker, style in [(1,e1,ALPHA_COLORS[1],'o','-'),(3,e3,ALPHA_COLORS[3],'^',':')]:
            m,e = samples
            a.fill_between(times[positive],np.maximum(m-e,1e-13)[positive],(m+e)[positive],color=color,alpha=.1,lw=0)
            a.plot(times[positive],m[positive],color=color,marker=marker,ls=style,lw=1.3,ms=3.5,mfc='white',mew=.8,
                   markevery=marker_indices,label=rf'$\alpha_1={alpha}$')
            entropy_rows.extend(dict(alpha_1=alpha,Ny=30,cycle=t,normalized_cycle=t/30,mean=float(m[t]),sem=float(e[t])) for t in range(61))
        a.set(xscale='log',yscale='log',xlim=cycle_limits,xlabel=cycle_label,ylabel=r'$\overline{S}(t)/N_y$')
        a.legend(loc='lower left',bbox_to_anchor=(.02,.07),frameon=False)
        a.text(.98,.51,r'$N_y=30$',transform=a.transAxes,ha='right')
        if panels == "entropy":
            region_labels = {5: 'left edge', 15: 'right edge', 2: 'left trivial bulk', 18: 'right trivial bulk'}
            for x,color,marker,style in [(5,'#1f77b4','o','-'),(15,'#1f77b4','s','--'),
                                        (2,'#d62728','^',':'),(18,'#2ca02c','v','-.')]:
                m,e = sx[x]
                b.fill_between(times[positive],np.maximum(m-e,1e-13)[positive],(m+e)[positive],color=color,alpha=.1,lw=0)
                b.plot(times[positive],m[positive],color=color,marker=marker,ls=style,lw=1.2,ms=3.1,mfc='white',mew=.8,
                       markevery=marker_indices if x!=2 else [i for i in range(60) if i+1 in (2,4,7,12,20,34,50)],label=region_labels[x])
                spatial_rows.extend(dict(Ny=30,x=x,cycle=t,normalized_cycle=t/30,mean=float(m[t]),sem=float(e[t])) for t in range(61))
            b.set(xscale='log',yscale='log',xlim=cycle_limits,xlabel=cycle_label,ylabel=r'$\overline{s_x}(t)/N_y$')
            b.legend(loc='center right',bbox_to_anchor=(1,.45),frameon=False,handlelength=1.8)
            b.text(.22,.73,r'$\alpha_1=1$',transform=b.transAxes,ha='left',va='top')
        for ax in ((a,b) if panels == "entropy" else (a,)):
            ax.xaxis.set_major_locator(FixedLocator([1,3,10,30,60] if raw_cycles else [.05,.1,.3,1,2]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_locator(LogLocator(base=10,numticks=4))
            ax.yaxis.set_minor_formatter(NullFormatter())
    if panels in ("lyapunov", "combined"):
        sizes=np.array([int(r['Ny']) for r in gaps])
        c.errorbar(sizes,[float(r['mean_gap']) for r in gaps],yerr=[float(r['sample_sem']) for r in gaps],
                   color='#1f77b4',marker='o',ls='none',ms=4,mfc='white',mew=1,capsize=2,
                   label=gap_time_label+r': mean $\pm$ SEM')
        dense=np.linspace(sizes.min(),sizes.max(),300)
        c.plot(dense,fit['amplitude']*dense**(-fit['exponent']),color='.25',ls='--',lw=1,
               label=rf"$A N_y^{{-z}},\ z={fit['exponent']:.2f}\pm{fit['exponent_sem']:.2f}$")
        c.set(xlabel=r'circumference $N_y$',ylabel=r'$\overline{\Delta}(t=2N_y)$',ylim=(0,.065),xticks=sizes)
        if gap_loglog:
            values = np.array([float(r['mean_gap']) for r in gaps])
            errors = np.array([float(r['sample_sem']) for r in gaps])
            assert np.all(values-errors > 0)
            c.set(ylim=(float((values-errors).min())*.75, float((values+errors).max())*2),
                  xscale='log', yscale='log', xlim=(sizes.min()*.95,sizes.max()*1.05))
            # Standard, sparse log-axis ticks, independent of sampled sizes.
            c.xaxis.set_major_locator(FixedLocator([20, 30, 40, 60]))
            c.xaxis.set_minor_locator(FixedLocator([25, 35, 45, 50, 55]))
            c.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
            c.xaxis.set_minor_formatter(NullFormatter())
            c.yaxis.set_major_locator(FixedLocator([.01, .02, .04, .08]))
            c.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
            c.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1,10), numticks=100))
            c.yaxis.set_minor_formatter(NullFormatter())
            c.tick_params(which='major', length=4)
            c.tick_params(which='minor', length=2.5)
        c.text(.09,.13,r'$\alpha_1=1$',transform=c.transAxes,ha='left',va='bottom')
        c.legend(loc='upper right',frameon=False)
        image=d.imshow(density.T,origin='lower',interpolation='nearest',aspect='auto',
                       extent=(-.5,19.5,-.5,29.5),cmap='magma',vmin=0,vmax=float(density.max()))
        d.set(xlabel='$x$',ylabel='$y$',xticks=[0,5,10,15,19],yticks=[0,10,20,29])
        d.text(.5,.97,r'$N_y=30,\ t=2N_y$' if heatmap_time_annotation else r'$N_y=30$',
               color='white',ha='center',va='top',transform=d.transAxes)
        cb=fig.colorbar(image,cax=d.inset_axes([1.025,0,.035,1]))
        cb.locator=MaxNLocator(nbins=3); cb.update_ticks()
        cb.set_label(r'$\overline{p_{\min}(x,y)}$',labelpad=2)
        cb.ax.tick_params(labelsize=8,pad=2)
    if panels == "combined":
        for line, samples in zip(a.lines, (e1, e3)):
            np.testing.assert_array_equal(line.get_xdata(), times[positive])
            np.testing.assert_array_equal(line.get_ydata(), samples[0][positive])
    if panels in ('combined','lyapunov'):
        np.testing.assert_array_equal(c.lines[0].get_ydata(), [float(r['mean_gap']) for r in gaps])
        np.testing.assert_array_equal(c.lines[-1].get_ydata(), fit['amplitude']*dense**(-fit['exponent']))
        np.testing.assert_array_equal(image.get_array(), density.T)
    panel_letters = []
    for ax,label in zip(axes,'abc'):
        panel_letters.append(fig.text(.075, ax.get_position().y1+.012,
                                      f'({label})',ha='left',va='bottom',fontsize=9))
    fig.align_ylabels(axes)
    np.testing.assert_allclose([ax.get_position().x0 for ax in axes], .22)
    np.testing.assert_allclose([ax.get_position().x1 for ax in axes], .81)
    np.testing.assert_allclose([text.get_position()[0] for text in panel_letters], .075)
    prepare_figure(fig, stem)
    destination = OUT if output_dir is None else Path(output_dir)
    destination.mkdir(parents=True,exist_ok=True)
    record_typography(fig, stem)
    for ext in ('pdf','png'):
        fig.savefig(destination/f'{stem}.{ext}',dpi=300)
    plt.close(fig)
    return entropy_rows,spatial_rows

def plot_entropy_spectra(e):
    """Entropy and rank-averaged occupations, with identical spectral axes."""
    configure_style()
    stem = 'Figure_04_purification'
    fig, axes = plt.subplots(3, 1, figsize=(3.375, 6.3))
    fig.subplots_adjust(left=.22, right=.95, bottom=.075, top=.955, hspace=.55)
    a = axes[0]
    times = np.arange(1,61)/30
    indices = np.unique(np.rint(np.geomspace(1,60,10)).astype(int)-1).tolist()
    for alpha, marker, style in ((1,'o','-'),(3,'^',':')):
        m,err = e[alpha]
        color = ALPHA_COLORS[alpha]
        a.fill_between(times,np.maximum(m-err,1e-13)[1:],(m+err)[1:],color=color,alpha=.1,lw=0)
        line, = a.plot(times,m[1:],color=color,marker=marker,ls=style,lw=1.3,ms=3.5,mfc='white',mew=.8,
                      markevery=indices,label=rf'$\alpha_1={alpha}$')
        np.testing.assert_array_equal(line.get_ydata(), m[1:])
    a.set(xscale='log',yscale='log',xlim=(1/30,2),xlabel=r'cycle $t/N_y$',ylabel=r'$\overline{S}(t)/N_y$')
    a.legend(loc='lower left',frameon=False)
    a.text(.97,.48,r'$N_y=30$',transform=a.transAxes,ha='right')
    a.xaxis.set_major_locator(FixedLocator([.05,.1,.3,1,2]))
    a.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
    a.xaxis.set_minor_formatter(NullFormatter())
    a.yaxis.set_major_locator(LogLocator(base=10,numticks=4))
    a.yaxis.set_minor_formatter(NullFormatter())
    with (DATA/'ranked_occupation_means.csv').open() as f:
        rows=list(csv.DictReader(f))
    styles = [(1,'#009E73','s'),(5,'#0072B2','o'),(60,'#333333','v')]
    ranks=np.arange(1,1201)
    shared_handles=[]
    for ax,alpha in zip(axes[1:],(1,3)):
        inset=ax.inset_axes([.08,.18,.36,.34],zorder=20)
        for k,(cycle,color,marker) in enumerate(styles):
            selected=sorted((r for r in rows if int(r['alpha_1'])==alpha and int(r['cycle'])==cycle),key=lambda r:int(r['rank']))
            np.testing.assert_array_equal([int(r['rank']) for r in selected],ranks)
            assert all(int(r['samples'])==100 for r in selected)
            mean=np.array([float(r['mean_occupation']) for r in selected])
            assert np.all(np.diff(mean)>=0)
            line,=ax.plot(ranks,mean,ls='none',marker=marker,color=color,ms=1.6,mfc='none',mew=.4,label=rf'${cycle}$',zorder=10-k)
            inset.plot(ranks,mean,ls='none',marker=marker,color=color,ms=2.4,mfc='none',mew=.55,zorder=10-k)
            np.testing.assert_array_equal(line.get_ydata(),mean)
            if alpha==1:shared_handles.append(line)
        ax.set(xlim=(1,1200),ylim=(-.035,1.035),xticks=[1,300,600,900,1200],yticks=[0,.5,1],
               ylabel=r'$\overline{\nu_{(j)}(t)}$',xlabel=r'ordered eigenvalue index $j$')
        ax.text(.96,.10,rf'$\alpha_1={alpha}$',transform=ax.transAxes,ha='right')
        inset.set(xlim=(580,620),ylim=(-.035,1.035),xticks=[580,600,620],yticks=[0,1])
        inset.tick_params(length=2,pad=1)
    # A single cycle legend governs both occupation panels.
    axes[1].legend(handles=shared_handles,title=r'cycle $t$',loc='upper left',ncol=3,
                   frameon=False,markerscale=2.5,handlelength=1,handletextpad=.25,columnspacing=.6,
                   borderpad=.15,labelspacing=.2)
    for ax,letter in zip(axes,'abc'):
        fig.text(.055,ax.get_position().y1+.008,f'({letter})',va='bottom')
    fig.align_ylabels(axes)
    prepare_figure(fig,stem)
    record_typography(fig,stem)
    for ext in ('pdf','png'):fig.savefig(OUT/f'{stem}.{ext}',dpi=300)
    plt.close(fig)

def main():
    def rows(name):
        with (DATA/name).open() as stream: return list(csv.DictReader(stream))
    ent = rows('total_entropy_curves.csv')
    spatial = rows('spatial_entropy_curves.csv')
    def arrays(selected):
        selected = sorted(selected, key=lambda row:int(row['cycle']))
        assert [int(row['cycle']) for row in selected] == list(range(61))
        return tuple(np.array([float(row[key]) for row in selected]) for key in ('mean','sem'))
    e = {alpha:arrays([r for r in ent if int(r['alpha_1'])==alpha]) for alpha in (1,3)}
    sx = {x:arrays([r for r in spatial if int(r['x'])==x]) for x in (5,15,2,18)}
    with np.load(DATA/'Ny030_slowest_mode_density.npz') as z:
        density=z['mean']
        np.testing.assert_allclose(density, z['sample_densities'].mean(0), atol=1e-16)
        np.testing.assert_allclose(density.sum(),1,atol=1e-12)
    gaps = rows('retained_gap_summary.csv')
    fit = json.loads((DATA/'analysis_manifest.json').read_text())['gap_fit']
    plot_entropy_spectra(e)
    plot(e[1],e[3],sx,density,gaps,fit,panels='lyapunov',gap_loglog=True,stem='Figure_04_lyapunov')
    (DATA/'notation_validation.json').write_text(json.dumps(dict(status='passed', data_changed=False,
      entropy_points=len(ent), spatial_points=len(spatial), gap_points=len(gaps), fit=fit,
      density_sum=float(density.sum()), labels='ensemble means overlined',
      layout=[3,1], displayed_panels=['total_entropy','occupation_alpha1_1','occupation_alpha1_3'],
      lyapunov_layout=[2,1], lyapunov_panels=['lyapunov_gap','slowest_mode_density'],
      occupation_cycles=[1,5,60], inset_rank_limits=[580,620], occupation_samples=100,
      spatial_panel_displayed=False, fit_unchanged=True, gap_axes_scale="log-log",
      gap_axis_label=r"$\overline{\Delta}(t=2N_y)$",
      horizontal_alignment=dict(purification_axes_left=.22,purification_axes_right=.95,lyapunov_axes_left=.22,lyapunov_axes_right=.81),
      gap_parameter_annotation="alpha_1=1, lower left at axes (0.09,0.13)"),indent=2)+'\n')
if __name__=='__main__': main()
