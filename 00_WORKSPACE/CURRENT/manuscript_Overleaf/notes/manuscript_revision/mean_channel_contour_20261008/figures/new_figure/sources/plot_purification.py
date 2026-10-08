#!/usr/bin/env python3
"""Compact purification stack and horizontal occupation/gap comparison from saved data."""
import csv, json, hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FixedLocator, FuncFormatter
import numpy as np
from manuscript_palette import ALPHA_COLORS
from log_ticks import add_log_minor_ticks
import manuscript_typography as typography
OUT = Path(__file__).resolve().parents[1]
DATA = OUT/'data/purification'


def configure_style():
    typography.configure_style({'axes.linewidth': .8, 'xtick.direction': 'in',
        'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})


def rows(name):
    with (DATA/name).open() as stream:
        return list(csv.DictReader(stream))


def save(fig, stem):
    add_log_minor_ticks(fig)
    typography.prepare_figure(fig, stem)
    typography.record_typography(fig, stem)
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'{stem}.{ext}',dpi=300)
    plt.close(fig)


def draw_entropy(ax, entropy):
    times = np.arange(1,61)/30
    indices = np.unique(np.rint(np.geomspace(1,60,10)).astype(int)-1).tolist()
    plotted = {}
    for alpha, marker, style in ((1,'o','-'),(3,'^',':')):
        old_mean, old_sem = entropy[alpha]
        mean, sem = 30*old_mean, 30*old_sem
        color = ALPHA_COLORS[alpha]
        ax.fill_between(times,np.maximum(mean-sem,30e-13)[1:],(mean+sem)[1:],color=color,alpha=.1,lw=0)
        line, = ax.plot(times,mean[1:],color=color,marker=marker,ls=style,lw=1.3,ms=3.5,
            mfc='white',mew=.8,markevery=indices,label=rf'$\alpha_1={alpha}$')
        np.testing.assert_array_equal(line.get_ydata(),30*old_mean[1:])
        np.testing.assert_array_equal(sem,30*old_sem)
        plotted[str(alpha)] = {'mean':mean.tolist(),'trajectory_SEM':sem.tolist()}
    ax.set(xscale='log',yscale='log',xlim=(1/30,2),xlabel=r'cycle $t/N_y$',ylabel=r'$\overline{S}(t)$')
    ax.legend(loc='lower left',frameon=False)
    ax.xaxis.set_major_locator(FixedLocator([.05,.1,.3,1,2]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10,numticks=4))
    ax.yaxis.set_minor_formatter(NullFormatter())
    return plotted


def draw_occupations(ax, alpha, spectra):
    styles = [(1,'s'),(5,'o'),(60,'v')]
    gradients = {1:['#6baed6','#2171b5','#08306b'],3:['#fdae6b','#e6550d','#a63603']}
    ranks = np.arange(1,1201)
    inset = ax.inset_axes([.08,.26,.34,.34],zorder=20)
    for k,(cycle,marker) in enumerate(styles):
        selected = sorted((r for r in spectra if int(r['alpha_1'])==alpha and int(r['cycle'])==cycle),key=lambda r:int(r['rank']))
        np.testing.assert_array_equal([int(r['rank']) for r in selected],ranks)
        assert all(int(r['samples'])==100 for r in selected)
        mean=np.array([float(r['mean_occupation']) for r in selected])
        assert np.all(np.diff(mean)>=0)
        line,=ax.plot(ranks,mean,ls='none',marker=marker,color=gradients[alpha][k],ms=1.6,
            mfc='none',mew=.4,label=rf'${cycle}$',zorder=10-k)
        inset.plot(ranks,mean,ls='none',marker=marker,color=gradients[alpha][k],ms=2.4,
            mfc='none',mew=.55,zorder=10-k)
        np.testing.assert_array_equal(line.get_ydata(),mean)
    ax.set(xlim=(1,1200),ylim=(-.035,1.035),xticks=[1,300,600,900,1200],yticks=[0,.5,1],
        ylabel=r'$\overline{\nu_j}(t)$',xlabel=r'ordered eigenvalue index $j$')
    ax.text(.96,.10,rf'$\alpha_1={alpha}$',transform=ax.transAxes,ha='right')
    inset.set(xlim=(580,620),ylim=(-.035,1.035),xticks=[580,620],yticks=[0,1])
    inset.tick_params(length=2,pad=1)
    ax.legend(title=r'cycle $t$',loc='upper left',ncol=3,frameon=False,markerscale=2.5,
        handlelength=.5,handletextpad=.15,columnspacing=.35,borderpad=.15,labelspacing=.2)


def draw_gap(ax, gaps, fit):
    sizes=np.array([int(r['Ny']) for r in gaps])
    values=np.array([float(r['mean_gap']) for r in gaps])
    errors=np.array([float(r['sample_sem']) for r in gaps])
    ax.errorbar(sizes,values,yerr=errors,color='#1f77b4',marker='o',ls='none',ms=4,
        mfc='white',mew=1,capsize=2,label='_nolegend_')
    dense=np.linspace(sizes.min(),sizes.max(),300)
    ax.plot(dense,fit['amplitude']*dense**(-fit['exponent']),color='.25',ls='--',lw=1,
        label=rf"$A N_y^{{-z}}$"+"\n"+rf"$z={fit['exponent']:.2f}\pm{fit['exponent_sem']:.2f}$")
    assert np.all(values-errors>0)
    ax.set(xlabel=r'circumference $N_y$',ylabel=r'$\overline{\Delta}(t=2N_y)$',
        ylim=(float((values-errors).min())*.75,float((values+errors).max())*2),
        xscale='log',yscale='log',xlim=(sizes.min()*.95,sizes.max()*1.05))
    ax.xaxis.set_major_locator(FixedLocator([20,30,40,60]))
    ax.xaxis.set_minor_locator(FixedLocator([25,35,45,50,55]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([.01,.02,.04,.08]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
    ax.yaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(1,10),numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which='major',length=4)
    ax.tick_params(which='minor',length=2.5)
    ax.text(.09,.13,r'$\alpha_1=1$',transform=ax.transAxes,ha='left',va='bottom')
    ax.legend(loc='upper right',frameon=False)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(),values)
    np.testing.assert_array_equal(ax.lines[-1].get_ydata(),fit['amplitude']*dense**(-fit['exponent']))


def plot_entropy_contour(entropy, mean):
    stem='Figure_04_purification'
    fig=plt.figure(figsize=(3.375,3.85))
    left,width=.60,2.00
    a=fig.add_axes([left/3.375,2.54/3.85,width/3.375,(width/1.822036796536796)/3.85])
    b=fig.add_axes([left/3.375,.55/3.85,width/3.375,1.50/3.85])
    plotted=draw_entropy(a,entropy)
    im=b.imshow(mean.T,origin='lower',interpolation='none',aspect='auto',
        extent=(-.5,19.5,-.5,29.5),cmap='Blues',vmin=0,vmax=.0075)
    b.set(xlabel='$x$',ylabel='$y$',xticks=[0,5,10,15,19],yticks=[0,10,20,29])
    b.text(.5,.5,r'$\alpha_1=1$',transform=b.transAxes,ha='center',va='center')
    cax=fig.add_axes([2.70/3.375,.55/3.85,.065/3.375,1.50/3.85])
    cb=fig.colorbar(im,cax=cax)
    cb.set_ticks([0,.0025,.005,.0075]);cb.set_label(r'$\overline{s}(x,y)$',labelpad=3)
    cb.ax.tick_params(pad=2,length=2)
    for ax,letter in zip((a,b),'ab'):
        fig.text(.03,ax.get_position().y1+.01,f'({letter})',va='bottom')
    np.testing.assert_allclose(a.get_position().bounds[0::2],b.get_position().bounds[0::2],atol=1e-12)
    np.testing.assert_array_equal(im.get_array(),mean.T)
    save(fig,stem)
    return plotted, {'left':left,'width':width,'heatmap_height':1.5,'canvas':[3.375,3.85],
        'entropy_axes_width_over_height':1.822036796536796,'heatmap_aspect':'display-compressed, unchanged 20 x 30 cells',
        'alpha_annotation_axes_coordinates':[.5,.5],'alpha_font_matches_entropy_legend':True}


def plot_occupation_gap(spectra,gaps,fit):
    stem='Figure_04_lyapunov'
    fig,axes=plt.subplots(1,3,figsize=(7.05,2.75))
    fig.subplots_adjust(left=.075,right=.985,bottom=.225,top=.92,wspace=.46)
    for ax,alpha in zip(axes[:2],(3,1)):draw_occupations(ax,alpha,spectra)
    draw_gap(axes[2],gaps,fit)
    for ax,letter in zip(axes,'abc'):
        fig.text(ax.get_position().x0-.052,ax.get_position().y1+.012,f'({letter})',va='bottom')
    fig.align_ylabels(axes)
    save(fig,stem)


def comparison_preview(raw,normalized):
    directory=OUT.parents[1]/'deliverables/horizontal_layout_20261008'
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'data/typography').mkdir(parents=True,exist_ok=True)
    old_root,old_width=typography.ROOT,typography.inclusion_width
    typography.ROOT=directory
    typography.inclusion_width=lambda stem: typography.TEXT_INCHES
    stem='Purification_contour_normalization_comparison'
    fig=plt.figure(figsize=(7.05,3.75))
    for i,(mean,label,vmax,ticks) in enumerate([
        (raw,r'raw: $\overline{s}(x,y)$',.0075,[0,.0025,.005,.0075]),
        (normalized,r'normalized: $\overline{\widetilde{s}}(x,y)$',.02,[0,.005,.01,.015,.02])]):
        ax=fig.add_axes([.09+i*.48,.15,.245,.70])
        im=ax.imshow(mean.T,origin='lower',interpolation='none',aspect='equal',extent=(-.5,19.5,-.5,29.5),cmap='Blues',vmin=0,vmax=vmax)
        ax.set(xlabel='$x$',ylabel='$y$',xticks=[0,5,10,15,19],yticks=[0,10,20,29])
        ax.set_title(label,pad=7)
        ax.text(-.20,1.045,'('+chr(97+i)+')',transform=ax.transAxes,va='bottom')
        cax=fig.add_axes([.355+i*.48,.15,.013,.70]);cb=fig.colorbar(im,cax=cax)
        cb.set_ticks(ticks);cb.ax.tick_params(pad=2,length=2)
    fig.text(.5,.97,r'$\alpha_1=1,\quad t=60,\quad N_x=20,\quad N_y=30,\quad S=100$',ha='center')
    typography.prepare_figure(fig,stem);typography.record_typography(fig,stem)
    for ext in ('pdf','png'):fig.savefig(directory/f'{stem}.{ext}',dpi=300)
    plt.close(fig)
    fonts=typography.verify_typography(stem)
    typography.ROOT,typography.inclusion_width=old_root,old_width
    (directory/'contour_comparison_validation.json').write_text(json.dumps(dict(
        raw_sum=float(raw.sum()),normalized_sum=float(normalized.sum()),
        estimator='Left: mean raw contour. Right: normalize each trajectory before averaging.',
        separate_color_scales=True,not_in_manuscript=True,fonts=fonts),indent=2)+'\n')


def main():
    configure_style()
    ent=rows('total_entropy_curves.csv')
    entropy={}
    for alpha in (1,3):
        selected=sorted((r for r in ent if int(r['alpha_1'])==alpha),key=lambda r:int(r['cycle']))
        assert [int(r['cycle']) for r in selected]==list(range(61))
        entropy[alpha]=tuple(np.array([float(r[k]) for r in selected]) for k in ('mean','sem'))
    spectra=rows('ranked_occupation_means.csv')
    gaps=rows('retained_gap_summary.csv')
    fit=json.loads((DATA/'analysis_manifest.json').read_text())['gap_fit']
    provenance=json.loads((DATA/'entropy_contour_provenance.json').read_text())
    path=DATA/'Ny030_entropy_contour_cycle60.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==provenance['output_sha256']
    with np.load(path) as z:
        mean=z['mean_raw_contour'];normalized=z['mean_normalized_contour']
        np.testing.assert_array_equal(mean,z['raw_contours'].mean(0))
        np.testing.assert_array_equal(normalized,z['normalized_contours'].mean(0))
        np.testing.assert_allclose(mean.sum(),30*entropy[1][0][-1],atol=1e-13)
    plotted,geometry=plot_entropy_contour(entropy,mean)
    plot_occupation_gap(spectra,gaps,fit)
    validation=dict(status='passed',data_changed=False,entropy_points=len(ent),gap_points=len(gaps),fit=fit,
        fit_unchanged=True,layout=[2,1],displayed_panels=['total_entropy','raw_entropy_contour_alpha1_1'],
        lyapunov_layout=[1,3],lyapunov_panels=['occupation_alpha1_3','occupation_alpha1_1','lyapunov_gap'],
        occupation_cycles=[1,5,60],occupation_legend_panels=['a','b'],occupation_gradients={1:'blue',3:'orange-red'},
        inset_position=[.08,.26,.34,.34],inset_rank_limits=[580,620],occupation_samples=100,
        entropy_scaling_factor=30,entropy_ylabel=r'$\overline{S}(t)$',entropy_mean_and_SEM_scaled_exactly=True,
        entropy_plotted_arrays=plotted,entropy_geometry_inches=geometry,
        spatial_panel_displayed=True,slowest_mode_panel_displayed=False,contour_mean_sum=float(mean.sum()),
        contour_source_sha256=provenance['output_sha256'],contour_normalized=False,
        gap_axes_scale='log-log',gap_axis_label=r'$\overline{\Delta}(t=2N_y)$')
    (DATA/'notation_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    comparison_preview(mean,normalized)
    print('Rendered Figure 3: total entropy and raw contour; Figure 4: alpha3, alpha1 spectra and unchanged gap.')


if __name__=='__main__':main()
