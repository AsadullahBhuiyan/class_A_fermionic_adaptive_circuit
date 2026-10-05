"""Two-panel hard-wall postselected entropy/control and column contours."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
from plot_postselected_alpha1_spatial_contours import DATA, sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'analysis_outputs/postselected_total_spatial_alpha1_alpha3_2x1_v1'


def load(alpha):
    manifest=json.loads((DATA/'DOWNLOAD_MANIFEST.json').read_text())
    prefix=f'alpha1_{alpha}/results/hard/'
    rows={r['path']:r for r in manifest['files']}
    for name in ['completion.json','postselected_trajectory.npz']:
        path=DATA/prefix/name; row=rows[prefix+name]
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    path=DATA/prefix/'postselected_trajectory.npz'
    receipt=json.loads(path.with_name('completion.json').read_text())
    run=next(r for r in manifest['runs'] if r['alpha_1']==alpha and r['construction']=='hard')
    assert receipt['source_hashes']==run['executed_source_hashes']
    assert sha(path)==receipt['result_sha256']
    with np.load(path,allow_pickle=False) as z:
        assert float(z['alpha_1'])==alpha and str(z['construction'])=='hard'
        assert str(z['configuration_hash'])==receipt['configuration_hash']==run['configuration_hash']
        assert (int(z['Nx']),int(z['Ny']))==(20,40)
        np.testing.assert_array_equal(z['cycles'],np.arange(161))
        s=z['total_entropy_nats']; sx=z['entropy_contour_x']
        np.testing.assert_allclose(sx,z['entropy_contour'].sum(axis=2),atol=1e-12)
        np.testing.assert_allclose(sx.sum(axis=1),s,atol=1e-10)
        assert np.isfinite(s).all() and np.isfinite(sx).all()
        assert (s>0).all() and (sx[:,[5,10,15]]>0).all()
        return s/40,sx/40,dict(path=str(path),sha256=sha(path),alpha_1=alpha)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['CMU Sans Serif'],
        'mathtext.fontset':'cm','font.size':8,'axes.labelsize':8,'legend.fontsize':8,
        'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,'pdf.fonttype':42})
    fig,(a,b)=plt.subplots(2,1,figsize=(3.375,4.6),sharex=True,layout='constrained')
    u=np.arange(161)/40
    rows,sources=[],[]
    xcolors={5:'#d62728',10:'#2ca02c',15:'#1f77b4'}
    xmarkers={5:'^',10:'s',15:'o'}
    for alpha,color,marker,style in [(1,'#d62728','^',':'),(3,'#1f77b4','o','-')]:
        s,sx,source=load(alpha); sources.append(source)
        a.plot(u,s,color=color,ls=style,marker=marker,markevery=16,ms=3,mfc='white',mew=.8,lw=1.2,
               label=rf'$\alpha_1={alpha}$')
        for j,x in enumerate([5,10,15]):
            b.plot(u,sx[:,x],color=xcolors[x],ls='-' if alpha==1 else '--',marker=xmarkers[x],
                   markevery=(j*5,16),ms=3,mfc='white',mew=.8,lw=1.1)
        for t in range(161):
            rows.append(dict(alpha_1=alpha,cycle=t,normalized_cycle=t/40,entropy_over_Ny=float(s[t]),
                **{f'entropy_x{x}_over_Ny':float(sx[t,x]) for x in [5,10,15]}))
    a.set(ylabel=r'$S(t)/N_y$',title=r'Hard wall, postselected; $N_y=40$')
    a.legend(frameon=False,loc='center right')
    b.set(ylabel=r'$s_x(t)/N_y$',xlabel=r'cycle $t/N_y$')
    first=b.legend(handles=[Line2D([],[],color=xcolors[x],marker=xmarkers[x],mfc='white',ms=3,
                                  label=rf'$x={x}$') for x in [5,10,15]],frameon=False,
                   loc='center right',bbox_to_anchor=(1,.69),labelspacing=.15)
    b.add_artist(first)
    b.legend(handles=[Line2D([],[],color='.2',ls=ls,label=rf'$\alpha_1={alpha}$')
                      for alpha,ls in [(1,'-'),(3,'--')]],frameon=False,
             loc='center right',bbox_to_anchor=(1,.44),labelspacing=.15)
    for ax,label in [(a,'(a)'),(b,'(b)')]:
        ax.set(yscale='log',xlim=(0,4),xticks=[0,1,2,3,4])
        ax.yaxis.set_major_locator(LogLocator(base=10,numticks=5))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.text(-.17,1.035,label,transform=ax.transAxes,fontsize=9)
    stem=OUT/'postselected_total_spatial_alpha1_alpha3_2x1'
    fig.savefig(stem.with_suffix('.pdf'));fig.savefig(stem.with_suffix('.png'),dpi=300);plt.close(fig)
    with (OUT/'curves.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'summary.json').write_text(json.dumps(dict(sources=sources,Nx=20,Ny=40,cycles=160,
        samples_per_alpha=1,postselected=True,uncertainty='None; deterministic single trajectory per alpha',
        normalization='Entropy and y-summed column contours divided by Ny; cycles divided by Ny',
        alpha3_tail='Near-zero residuals are roundoff-limited, not physical plateaus',
        figure_size_inches=[3.375,4.6]),indent=2)+'\n')
    print(OUT)


if __name__=='__main__': main()
