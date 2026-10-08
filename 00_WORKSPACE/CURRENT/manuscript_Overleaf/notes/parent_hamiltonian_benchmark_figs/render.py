"""Render the nine parent benchmark groups using shared manuscript typography."""
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

import benchmark as b
from analyze import chord

FIG=b.HERE/'figures'
FIG.mkdir(exist_ok=True)
(FIG/'data').mkdir(exist_ok=True)
STYLE=b.MANUSCRIPT/'figures/new_figure/sources/manuscript_typography.py'
spec=importlib.util.spec_from_file_location('parent_note_typography',STYLE)
style=importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)
# Reuse the exact shared implementation in an isolated module instance. No
# shared source or manuscript typography record is modified by this note.
style.ROOT=FIG
style.MANUSCRIPT=b.HERE/'parent_hamiltonian_benchmark_figs.tex'
WIDTH=.5*style.TEXT_INCHES
BLUE='#0072B2';ORANGE='#D55E00'
SIZE_STYLE={20:('#d62728','^',':'),24:('#2ca02c','s','--'),28:('#0072B2','o','-'),
            30:('#D92725','^',':'),35:('#F08050','<','--'),40:('#8FC1E3','v','-.'),
            45:('#2CA02C','s',':'),50:('#6B6B6B','D','--'),55:('#000000','P','-.'),
            60:('#1F77B4','o','-')}


def load(name):
    with np.load(b.HERE/'data'/name,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def rows(name):
    with (b.HERE/'data'/name).open() as f:return list(csv.DictReader(f))


def panel(ax,letter):
    ax.text(-.17,1.06,f'({letter})',transform=ax.transAxes,ha='left',va='bottom')


def save(fig,stem,*,manual=False):
    style.prepare_figure(fig,stem)
    if not manual:fig.tight_layout(pad=.75,h_pad=1.35)
    style.record_typography(fig,stem)
    fig.savefig(FIG/f'{stem}.pdf')
    fig.savefig(FIG/f'{stem}.png',dpi=300)
    plt.close(fig)
    print(stem,flush=True)


def scientific(value,digits=4):
    mantissa,exponent=f'{value:.{digits}e}'.split('e')
    return rf'{mantissa}\times10^{{{int(exponent)}}}'


def report_values():
    fits=json.loads((b.HERE/'data/fits.json').read_text())
    periodic=json.loads((b.HERE/'data/prior_periodic_comparison.json').read_text())
    regulator=json.loads((b.HERE/'data/prior_protocol_comparison.json').read_text())
    hist=rows('histogram_counts.csv')
    values={'EntropyCoefficient':f"{fits['entropy']['converted_coefficient']:.7f}",
            'ChargeCoefficient':f"{fits['variance']['converted_coefficient']:.7f}",
            'EntropyResidual':scientific(fits['entropy']['weighted_rms_residual'],2),
            'ChargeResidual':scientific(fits['variance']['weighted_rms_residual'],2),
            'LeftSlope':f"{fits['wall_left']['slope']:.7f}",
            'RightSlope':f"{fits['wall_right']['slope']:.7f}",
            'LeftWeight':f"{fits['wall_left']['converted_coefficient']:.7f}",
            'RightWeight':f"{fits['wall_right']['converted_coefficient']:.7f}",
            'CorrelationBeta':f"{fits['correlations']['beta']:.4f}",
            'CorrelationRSquared':f"{fits['correlations']['weighted_r_squared']:.4f}",
            'CotExponent':f"{periodic['wall_amplitude_cotangent_fit']['slope']:.7f}",
            'RetainedTopological':hist[0]['retained_count_per_cut'],
            'RetainedTrivial':hist[1]['retained_count_per_cut'],
            'RegulatedGap':scientific(regulator['regulated_gap'],5),
            'PeriodicGap':scientific(regulator['periodic_gap'],5)}
    (b.HERE/'data/report_values.tex').write_text('\n'.join(rf'\newcommand{{\{key}}}{{{value}}}' for key,value in values.items())+'\n')
    table=[r'\begin{tabular}{rrrrrr}',r'\toprule',
           r'$L$ & $(x_L,x_R)$ & $R$ & $\mathcal C_G$ & $\mathcal C_G-1$ & $|\mathcal C_G-1|$ \\',r'\midrule']
    for r in rows('chern_table.csv'):
        table.append(rf"{r['L']} & $({r['x_left']},{r['x_right']})$ & {float(r['radius']):g} & {float(r['chern']):.12f} & ${scientific(float(r['signed_deviation']),3)}$ & ${scientific(float(r['absolute_deviation']),3)}$ \\")
    table.extend([r'\bottomrule',r'\end{tabular}'])
    (b.HERE/'data/chern_table.tex').write_text('\n'.join(table)+'\n')


def topology():
    data=load('topology.npz')
    fig,ax=plt.subplots(figsize=(WIDTH,3.25))
    im=ax.pcolormesh(np.arange(31)-.5,np.arange(31)-.5,data['marker_display'],
                     cmap='RdBu_r',vmin=-1,vmax=1,shading='flat',edgecolors='face',linewidth=.2)
    ax.set_aspect('equal')
    for x in (8,22):ax.axvline(x,color='k',ls='--',lw=.65)
    ax.set(xlabel='$x$',ylabel='$y$',xticks=[0,8,15,22,29],yticks=[0,15,29])
    fig.colorbar(im,ax=ax,fraction=.047,pad=.035,label=r'$\tanh C(\boldsymbol r)$',ticks=[-1,0,1])
    save(fig,'Parent_02_topology')


def occupations():
    fig,axes=plt.subplots(2,1,figsize=(WIDTH,4.45))
    for ax,alpha,colors,letter in zip(axes,(1,3),
            (('#a1cce0','#4599c0','#005986'),('#ffc58c','#eb8448','#bc470b')),('a','b')):
        data,_=b.load_case(20,30,alpha)
        rank=np.arange(1,1201)
        for t,n,color,marker in zip(data['filter_times'],data['filter_occupations'],colors,('s','o','v')):
            ax.plot(rank,n,ls='',marker=marker,ms=1.4,mfc='none',mew=.35,color=color,label=f'${t:g}$')
        ax.set(xlim=(1,1200),ylim=(-.03,1.03),xlabel=r'ordered occupation rank $j$',ylabel=r'$n_j(t)$',
               xticks=[1,300,600,900,1200],yticks=[0,.5,1])
        ax.legend(loc='upper left',ncol=3,title='filter parameter $t$',handletextpad=.25,columnspacing=.5)
        ax.text(.96,.14,rf'$\alpha_1={alpha}$',transform=ax.transAxes,ha='right')
        inset=ax.inset_axes([.075,.22,.35,.30])
        for n,color,marker in zip(data['filter_occupations'],colors,('s','o','v')):
            inset.plot(rank,n,ls='',marker=marker,ms=1.7,mfc='none',mew=.4,color=color)
        inset.set(xlim=(580,620),ylim=(-.02,1.02),xticks=[580,600,620],yticks=[0,1])
        inset.tick_params(length=2,pad=1)
        panel(ax,letter)
    save(fig,'Parent_03_occupations')


def gaps():
    records=rows('gaps.csv')
    ny=np.array([int(r['Ny']) for r in records]);gap=np.array([float(r['minimum_absolute_energy']) for r in records])
    fig,axes=plt.subplots(2,1,figsize=(WIDTH,4.35))
    ax=axes[0];ax.plot(ny,gap*1e10,'o-',ms=4,mfc='white',color=BLUE,lw=1)
    ax.set(xlabel=r'circumference $N_y$',ylabel=r'$10^{10}\delta_{\rm par}$',ylim=(0,10),xticks=[20,30,40,50,60])
    ax.text(.05,.12,r'$N_x=20,\ \alpha_1=1$',transform=ax.transAxes)
    panel(ax,'a')
    data,_=b.load_case(20,30,1)
    ax=axes[1];im=ax.pcolormesh(np.arange(21)-.5,np.arange(31)-.5,
                               data['minimal_mode_density'],cmap='magma',vmin=0,shading='flat',
                               edgecolors='face',linewidth=.2)
    ax.set(xlabel='$x$',ylabel='$y$',xticks=[0,5,10,15,19],yticks=[0,15,29])
    fig.colorbar(im,ax=ax,pad=.025,fraction=.04,label=r'$p_{\min}(x,y)$')
    panel(ax,'b');save(fig,'Parent_04_gap')


def correlations(fits):
    fig,axes=plt.subplots(2,1,figsize=(WIDTH,4.7))
    for alpha,color,mark,ls in [(1,BLUE,'o','-'),(3,ORANGE,'^',':')]:
        data,_=b.load_case(20,60,alpha);r=np.arange(2,31);c=data['correlation'][r]
        keep=c>1e-20
        axes[0].loglog(r[keep],c[keep],color=color,marker=mark,mfc='white',ms=3,ls=ls,lw=1,label=rf'$\alpha_1={alpha}$')
    axes[0].set(xlabel=r'$r_y$',ylabel=r'$C_G(r_y)$',xlim=(1.8,32),ylim=(1e-20,1e-3),xticks=[2,5,10,30])
    axes[0].set_xticklabels(['2','5','10','30']);axes[0].legend(loc='lower left');panel(axes[0],'a')
    ax=axes[1];maxy=0
    styles=[(ORANGE,'^'),('#009E73','s'),(BLUE,'o'),('#CC79A7','v'),('#E69F00','D'),('#333333','>')]
    for ny,(color,mark) in zip(b.CORR_SIZES,styles):
        data,_=b.load_case(20,ny,1);r=np.arange(2,ny//2+1)
        if ny in fits['undefined_antipodal_sizes']:continue
        value=data['correlation'][r]/data['correlation'][-1];maxy=max(maxy,value.max())
        ax.loglog(np.sin(np.pi*r/ny),value,color=color,marker=mark,mfc='none',ms=3,ls='-',lw=.5,label=f'${ny}$')
    lower=np.sin(8*np.pi/60);x=np.linspace(lower,1,300)
    ax.axvspan(lower,1,color='.5',alpha=.13,lw=0)
    ax.plot(x,x**(-fits['correlations']['beta']),'k--',lw=1,label='joint fit')
    ax.set(xlabel=r'$\sin(\pi r_y/N_y)$',ylabel=r'$C_G(r_y)/C_G(N_y/2)$',xlim=(.09,1.06),ylim=(.7,maxy*2),xticks=[.1,.2,.5,1])
    ax.set_xticklabels(['0.1','0.2','0.5','1.0'])
    ax.legend(title='$N_y$',ncol=3,loc='lower left',bbox_to_anchor=(.02,.24),
              handlelength=.9,columnspacing=.5,handletextpad=.3)
    ax.text(.04,.08,rf"formal $\beta={fits['correlations']['beta']:.2f}$"+'\n'+rf"$R^2={fits['correlations']['weighted_r_squared']:.3f}$ (poor fit)",transform=ax.transAxes)
    panel(ax,'b');save(fig,'Parent_05_correlations')


def anchored(fits,wall=False):
    keys=('wall_left','wall_right') if wall else ('entropy','variance')
    sizes=b.WALL_SIZES if wall else b.ENTROPY_SIZES
    fig,axes=plt.subplots(2,1,figsize=(WIDTH,4.7),sharex=True)
    for ax,key,letter in zip(axes,keys,'ab'):
        xmin=0;fitmin=0
        for ny in sizes:
            data,_=b.load_case(20,ny,1);w=data['widths'];keep=w>=2
            x=np.log(chord(ny,w)/chord(ny,ny//2));y=data[key]-data[key][-1]
            color,marker,ls=SIZE_STYLE[ny]
            ax.plot(x[keep],y[keep],ls='',marker=marker,color=color,mfc='white',ms=3,mew=.7,label=f'${ny}$')
            xmin=min(xmin,x[keep].min());fitmin=min(fitmin,x[w>=8].min())
        ax.axvspan(fitmin,0,color='.5',alpha=.13,lw=0)
        fit=fits[key];line=np.linspace(xmin,0,200)
        ax.plot(line,fit['slope']*line,'k--',lw=1)
        if wall:
            suffix='L' if key=='wall_left' else 'R';columns='5,6' if suffix=='L' else '14,15'
            ax.set_ylabel(rf'$\Delta S_{{{suffix}}}^{{\rm wall}}$')
            ax.text(.04,.94,rf'$x={columns}$'+'\n'+rf'$3m_{suffix}={fit["converted_coefficient"]:.6f}$',transform=ax.transAxes,va='top')
            inset=ax.inset_axes([.065,.24,.17,.36]);inset.set(xlim=(0,20),ylim=(0,30));inset.axis('off')
            for x0 in (5,15):inset.axvline(x0,color='.6',ls='--',lw=.5)
            for x0 in ([5,6] if suffix=='L' else [14,15]):inset.axvspan(x0-.5,x0+.5,color=BLUE,alpha=.6)
            inset.add_patch(Rectangle((0,0),20,30,fill=False,edgecolor='.6',lw=.7))
        else:
            label='c' if key=='entropy' else 'k'
            ax.set_ylabel(r'$\Delta S$' if key=='entropy' else r'$\Delta F$')
            ax.text(.04,.94,rf'${label}={fit["converted_coefficient"]:.6f}$',transform=ax.transAxes,va='top')
        ax.set_xlim(xmin-.06,.04);panel(ax,letter)
    axes[0].legend(title='$N_y$',loc='lower right',ncol=3,handletextpad=.1,columnspacing=.3)
    axes[1].set_xlabel(r'$\log[D(A_y)/D(\lfloor N_y/2\rfloor)]$')
    save(fig,'Parent_08_wall_entropy' if wall else 'Parent_06_entropy_charge')


def spectrum(fits):
    data=load('histograms.npz')
    fig,axes=plt.subplots(3,1,figsize=(WIDTH,5.65))
    for a,color,ls in [(1,BLUE,'-'),(3,ORANGE,':')]:
        rho=data[f'occupation_density_a{a}'];rho=np.where(rho>0,rho,np.nan)
        axes[0].stairs(rho,data['occupation_edges'],color=color,ls=ls,lw=1,label=rf'$\alpha_1={a}$')
        axes[1].stairs(data[f'energy_density_a{a}'],data['energy_edges'],color=color,ls=ls,lw=1,label=rf'$\alpha_1={a}$')
    axes[0].set(yscale='log',xlabel=r'occupation $\nu$',ylabel=r'density $\rho(\nu)$',xlim=(-.01,1.01))
    axes[0].legend(loc='upper center',ncol=2,handlelength=1.5)
    axes[1].set(xlabel=r'entanglement energy $\varepsilon$',ylabel=r'conditional $p_W(\varepsilon)$',xlim=(-5.3,5.3))
    axes[1].text(.5,.94,r'$W:\ |2\nu-1|<0.99$',transform=axes[1].transAxes,ha='center',va='top')
    axes[1].set_ylim(0,axes[1].get_ylim()[1]*1.2)
    d,_=b.load_case(20,32,1);x=np.log(chord(32,d['widths']));fit=fits['mode_count']
    axes[2].plot(x,d['mode_count'],'o',mfc='white',ms=3,color=BLUE)
    axes[2].axvspan(x[4],x[-1],color='.5',alpha=.13,lw=0)
    axes[2].plot(x,fit['intercept']+fit['slope']*x,'k--',lw=1)
    axes[2].text(.06,.12,r'$N_{0.99}=30$ in fit window',transform=axes[2].transAxes)
    axes[2].set(xlabel=r'$\log D(A_y)$',ylabel=r'$N_{0.99}$',ylim=(21,31))
    for ax,letter in zip(axes,'abc'):panel(ax,letter)
    save(fig,'Parent_07_spectrum')


def modular():
    data=load('modular.npz');fig=plt.figure(figsize=(WIDTH,3.85))
    axes=[fig.add_axes([.145,.615,.345,.28]),fig.add_axes([.615,.615,.345,.28]),fig.add_axes([.145,.14,.815,.31])]
    colors=['#332288','#E69F00','#009E73'];xx,yy=np.meshgrid(np.arange(20),np.arange(16))
    for ax,a,letter in zip(axes[:2],(3,1),'ab'):
        densities=data[f'a{a}_eps10_density']
        for packet in range(2):
            for t,color in enumerate(colors):
                rho=densities[packet,t];mask=rho>1e-4
                ax.scatter(xx[mask],yy[mask],s=65*np.sqrt(rho[mask]/2),color=color,linewidths=.2,zorder=5-t)
        for wall in (5,15):ax.axvline(wall,color='.6',ls=':',lw=.6,zorder=0)
        ax.set(xlim=(-.5,19.5),ylim=(-.5,15.5),xlabel='$x$',xticks=[0,5,15,19],yticks=[0,8,15])
        ax.text(.5,.94,rf'$\alpha_1={a}$',transform=ax.transAxes,ha='center',va='top',bbox=dict(facecolor='white',edgecolor='none',pad=1))
        panel(ax,letter)
    axes[0].set_ylabel(r'$\delta y$');axes[1].tick_params(labelleft=False)
    handles=[Line2D([],[],ls='',marker='o',ms=3,color=c,label=f'${t:g}$') for c,t in zip(colors,(0,.1,.2))]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.56,.53),ncol=3,title=r'$t_{\rm mod}$',handletextpad=.3,columnspacing=1)
    ax=axes[2]
    for i,color,ls,label in [(0,ORANGE,'-','left'),(1,BLUE,'--','right')]:
        ax.plot(data['a1_eps10_times'],data['a1_eps10_displacement'][i],color=color,ls=ls,lw=1,label=label)
    ax.axhline(0,color='.65',lw=.5)
    ax.set(xlabel=r'modular time $t_{\rm mod}$',ylabel=r'$\Delta y$',xlim=(0,1),ylim=(-.75,.75),xticks=[0,.25,.5,.75,1])
    ax.legend(loc='center',bbox_to_anchor=(.5,.62),ncol=2);panel(ax,'c')
    save(fig,'Parent_09_modular',manual=True)


def contour_mi():
    fig=plt.figure(figsize=(WIDTH,4.7))
    maps=[b.load_case(20,32,a)[0]['half_contour'] for a in (1,3)]
    maximum=max(float(x.max()) for x in maps)
    norm=PowerNorm(gamma=.5,vmin=0,vmax=maximum)
    axes=[fig.add_axes([.145,.71,.345,.20]),fig.add_axes([.615,.71,.345,.20])]
    for ax,a,data in zip(axes,(1,3),maps):
        mesh=ax.pcolormesh(np.arange(21)-.5,np.arange(17)-.5,data,cmap='Blues',norm=norm,shading='flat',edgecolors=(.7,.7,.7,.3),lw=.12)
        ax.set(xlim=(-.5,19.5),ylim=(-.5,15.5),xticks=[0,5,15,19],yticks=[0,8,15],xlabel='$x$',title=rf'$\alpha_1={a}$')
    axes[0].set_ylabel(r'$\delta y$');axes[1].tick_params(labelleft=False);panel(axes[0],'a')
    cax=fig.add_axes([.255,.60,.61,.018])
    cb=fig.colorbar(mesh,cax=cax,orientation='horizontal',ticks=[0,.1,.3,maximum]);cb.set_label(r'$s(x,\delta y)$',labelpad=1)
    cb.ax.set_xticklabels(['0','0.1','0.3',f'{maximum:.2f}']);cb.ax.tick_params(pad=1,length=2)
    ax=fig.add_axes([.145,.105,.815,.385]);records=rows('mutual_information.csv')
    for ny in (20,24,28):
        subset=[r for r in records if int(r['Ny'])==ny];alpha=np.array([float(r['alpha_1']) for r in subset]);val=np.array([float(r['mutual_information']) for r in subset])
        color,marker,ls=SIZE_STYLE[ny]
        ax.plot(alpha,val,color=color,marker=marker,ms=3,ls=ls,lw=1,label=rf'$N_y={ny}$')
        ax.plot(alpha[alpha==2],val[alpha==2],'o',color='k',mfc='none',ms=5,mew=.6,zorder=5)
    ax.axhline(np.log(2)/3,color='#7F6F91',ls='--',lw=.8,label=r'$(\log2)/3$')
    ax.set(xlim=(.98,3.02),xlabel=r'$\alpha_1$',ylabel=r'$I_{a,b}$');ax.set_ylim(-.01,ax.get_ylim()[1]*1.05)
    ax.legend(loc='upper right',bbox_to_anchor=(.98,.80),handlelength=1.5,labelspacing=.2)
    inset=ax.inset_axes([.065,.06,.31,.24]);inset.set(xlim=(0,20),ylim=(0,28));inset.axis('off')
    inset.add_patch(Rectangle((0,0),20,28,fc='#ececf0',ec='.5',lw=.5))
    inset.add_patch(Rectangle((5,0),10,28,fc='#92bed2',ec='.5',lw=.5))
    inset.add_patch(Rectangle((0,1),20,6,fc='#d96662',alpha=.6,ec='.3',lw=.5))
    inset.add_patch(Rectangle((0,15),20,6,fc='#2376a6',alpha=.6,ec='.3',lw=.5))
    inset.text(1,2,'$a$');inset.text(1,16,'$b$')
    panel(ax,'b');save(fig,'Parent_A2_contour_mi',manual=True)


def main():
    style.configure_style({'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,
                           'axes.linewidth':.8,'legend.frameon':False,'lines.linewidth':1})
    report_values()
    fits=json.loads((b.HERE/'data/fits.json').read_text())
    topology();occupations();gaps();correlations(fits);anchored(fits);spectrum(fits)
    anchored(fits,wall=True);modular();contour_mi()


if __name__=='__main__':main()
