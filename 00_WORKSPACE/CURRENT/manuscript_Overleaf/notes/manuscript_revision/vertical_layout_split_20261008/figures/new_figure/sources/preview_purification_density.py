"""Standalone occupation-density previews from saved purification spectra."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FuncFormatter,LogLocator,NullFormatter
from manuscript_typography import configure_style,prepare_figure,record_typography
from log_ticks import add_log_minor_ticks
from manuscript_palette import ALPHA_COLORS

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/purification'
OUT=ROOT.parents[1]/'deliverables/spectral_updates_20261008/purification_previews'
CYCLES=[1,5,60]
GRADIENTS={1:['#6baed6','#2171b5','#08306b'],3:['#fdae6b','#e6550d','#a63603']}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((DATA/'occupation_source_provenance.json').read_text())
    values={a:{t:[] for t in CYCLES} for a in [1,3]};seen={1:[],3:[]};sources=[]
    for item in provenance['inputs']:
        p=Path(item['path']);assert sha(p)==item['sha256'];sources.append(item)
        alpha=1 if 'alpha1_1/' in str(p) else 3
        with np.load(p,allow_pickle=False) as z:
            seen[alpha].extend(z['sample_indices'].tolist())
            times=z['cycles'];v=z['occupation_spectrum']
            for t in CYCLES:values[alpha][t].append(v[:,np.flatnonzero(times==t)[0],:].copy())
    assert all(sorted(v)==list(range(100)) for v in seen.values())
    edges=np.linspace(0,1,101);counts={};notes=[]
    for alpha in [1,3]:
        for t in CYCLES:
            v=np.concatenate(values[alpha][t]);assert v.shape==(100,1200) and np.isfinite(v).all()
            assert v.min()>=-1e-8 and v.max()<=1+1e-8
            n=np.histogram(np.clip(v,0,1).ravel(),edges)[0];assert n.sum()==120000
            counts[alpha,t]=n;density=n/(120000*np.diff(edges))
            np.testing.assert_allclose(np.sum(density*np.diff(edges)),1,atol=1e-14)
            notes.append(dict(alpha_1=alpha,cycle=t,modes=120000,roundoff_outside_interval=int(((v<0)|(v>1)).sum()),
                              raw_extrema=[float(v.min()),float(v.max())],endpoint_bin_counts=[int(n[0]),int(n[-1])]))
    configure_style({'legend.frameon':False})
    with (DATA/'total_entropy_curves.csv').open() as f:ent=list(csv.DictReader(f))
    for candidate in [False,True]:
        stem='Purification_candidate' if candidate else 'Purification_spectral_density_preview'
        fig,axes=plt.subplots(3 if candidate else 2,1,figsize=(3.375,6.3 if candidate else 4.4))
        if candidate:
            a=axes[0];times=np.arange(1,61)/30
            indices=np.unique(np.rint(np.geomspace(1,60,10)).astype(int)-1).tolist()
            for alpha,marker,style in [(1,'o','-'),(3,'^',':')]:
                rows=sorted([r for r in ent if int(r['alpha_1'])==alpha],key=lambda r:int(r['cycle']))
                m=np.array([float(r['mean']) for r in rows]);e=np.array([float(r['sem']) for r in rows])
                a.fill_between(times,np.maximum(m-e,1e-13)[1:],(m+e)[1:],color=ALPHA_COLORS[alpha],alpha=.1,lw=0)
                a.plot(times,m[1:],color=ALPHA_COLORS[alpha],marker=marker,ls=style,lw=1.3,ms=3.5,mfc='white',mew=.8,markevery=indices,label=rf'$\alpha_1={alpha}$')
            a.set(xscale='log',yscale='log',xlim=(1/30,2),xlabel=r'cycle $t/N_y$',ylabel=r'$\overline{S}(t)/N_y$')
            a.legend(loc='lower left');a.xaxis.set_major_locator(FixedLocator([.05,.1,.3,1,2]));a.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
            a.yaxis.set_major_locator(LogLocator(base=10,numticks=4))
        for ax,alpha in zip(axes[1:] if candidate else axes,[1,3]):
            for k,(t,ls) in enumerate(zip(CYCLES,['-', '--', ':'])):
                density=counts[alpha,t]/(120000*np.diff(edges))
                ax.stairs(np.where(density>0,density,np.nan),edges,baseline=None,color=GRADIENTS[alpha][k],ls=ls,lw=1.2,label=rf'$t={t}$')
            ax.set(xlabel=r'Occupation $\nu$',ylabel=r'Spectral density $\rho_t(\nu)$',xlim=(-.01,1.01),yscale='log',ylim=(4e-4,200))
            ax.set_xticks([0,.25,.5,.75,1]);ax.legend(loc='upper center',ncol=3,handlelength=1.5,columnspacing=.7)
            ax.text(.50,.72,rf'$\alpha_1={alpha}$',ha='center',va='top',transform=ax.transAxes)
        for ax,letter in zip(axes,'abc'):
            ax.text(-.18,1.035,f'({letter})',transform=ax.transAxes);ax.tick_params(which='both',top=True,right=True,pad=2)
        add_log_minor_ticks(fig);prepare_figure(fig,stem)
        fig.subplots_adjust(left=.22,right=.97,bottom=.10 if not candidate else .075,top=.94,hspace=.53)
        record_typography(fig,stem)
        for ext in ['pdf','png']:fig.savefig(OUT/f'{stem}.{ext}',dpi=300)
        plt.close(fig)
    np.savez_compressed(OUT/'histogram_counts.npz',edges=edges,cycles=CYCLES,alpha_1=[1,3],counts=np.array([[counts[a,t] for t in CYCLES] for a in [1,3]]))
    caption='Occupation spectral densities for full-system maximally mixed initial states on a 20 x 30 lattice, alpha_1=1,3, at cycles 1,5,60. Each density pools 100 independent trajectories and all 1,200 occupations per trajectory, with 100 equal-width bins on [0,1] and unit-area normalization. No spectral-window cutoff is applied; endpoint bins are included and empty bins are omitted on the logarithmic vertical axis. Increasing cycle runs from light to dark blue or orange-red. Alpha_1=3 uses the saved numerically clipped continuation after cycle 30. The candidate entropy panel uses the unchanged means and trajectory SEMs. These are previews and have not replaced the manuscript figure.\n'
    (OUT/'caption.txt').write_text(caption)
    (OUT/'validation.json').write_text(json.dumps(dict(inputs=sources,cycles=CYCLES,samples=100,counts=notes,
        bins=100,range=[0,1],normalization='unit area; all occupations retained',near_purity_cutoff=False,
        manuscript_figure_replaced=False,entropy_data_unchanged=True,no_simulations=True,source_sha256=sha(__file__)),indent=2)+'\n')
    print(caption);print(json.dumps(notes,indent=2))

if __name__=='__main__':main()
