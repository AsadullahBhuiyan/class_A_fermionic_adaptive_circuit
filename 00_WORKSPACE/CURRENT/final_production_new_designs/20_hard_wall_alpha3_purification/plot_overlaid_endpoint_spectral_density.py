#!/usr/bin/env python3
"""Overlay the same normalized pooled endpoint spectra without changing bins."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from plot_pooled_endpoint_histograms import ROOT, SOURCE, sha

OUT = ROOT/'analysis_outputs/overlaid_endpoint_spectral_density_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=SOURCE/'sample_sorted_occupations.npz'
    previous=ROOT/'analysis_outputs/pooled_endpoint_spectral_density_alpha3_alpha1_v1/summary.json'
    summary=json.loads(previous.read_text())
    assert sha(path)==summary['source_sha256']
    bins=np.linspace(0,1,51)
    width=np.diff(bins)
    densities={}
    with np.load(path,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['sample_indices'],np.arange(100))
        for alpha in (3,1):
            values=z[f'alpha1_{alpha}']
            assert values.shape==(100,880) and np.isfinite(values).all()
            assert values.min()>-1e-9 and values.max()<1+1e-9
            counts,_=np.histogram(np.clip(values,0,1).ravel(),bins=bins)
            densities[alpha]=counts/(values.size*width)
            np.testing.assert_allclose(np.sum(densities[alpha]*width),1,atol=1e-15)
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':8,
                         'mathtext.fontset':'cm','axes.linewidth':.8})
    fig,ax=plt.subplots(figsize=(3.375,2.8))
    blue,red='#1879bb','#e41a1c'
    mask=densities[3]>0
    ax.bar(bins[:-1][mask],densities[3][mask],width=width[mask],align='edge',
           color=blue,edgecolor=blue,alpha=.35,linewidth=1)
    # Zero heights are truly zero and invisible on the logarithmic scale.
    ax.stairs(densities[1],bins,color=red,linewidth=1.1,baseline=None)
    mask=densities[1]>0
    ax.plot(((bins[:-1]+bins[1:])/2)[mask],densities[1][mask],ls='none',
            marker='^',ms=3,mfc='white',mew=.7,color=red)
    ax.set(yscale='log',ylim=(2e-4,70),xlim=(0,1),
           xlabel=r'Endpoint occupation $\nu$',ylabel=r'$\rho(\nu)$')
    ax.set_xticks(np.linspace(0,1,6))
    ax.tick_params(which='both',direction='in',top=True,right=True)
    ax.set_title(r'Hard wall: $N_y=40$, $T=160$, $S=100$',fontsize=8)
    ax.legend(handles=[Line2D([],[],color=red,marker='^',ms=3,mfc='white',
                              lw=1.1,label=r'$\alpha_1=1$'),
                       Patch(facecolor=blue,edgecolor=blue,alpha=.35,label=r'$\alpha_1=3$')],
              loc='upper center',frameon=False)
    fig.tight_layout(pad=.6)
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'overlaid_endpoint_spectral_density.{ext}',dpi=300)
    plt.close(fig)
    summary.update(display='Blue filled alpha=3 and red outlined alpha=1 with triangle bin-center markers; log density; no pseudocounts.',
                   parent_summary=str(previous),parent_summary_sha256=sha(previous),
                   script_sha256=sha(Path(__file__)),figure_size_inches=[3.375,2.8])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(OUT)


if __name__=='__main__':
    main()
