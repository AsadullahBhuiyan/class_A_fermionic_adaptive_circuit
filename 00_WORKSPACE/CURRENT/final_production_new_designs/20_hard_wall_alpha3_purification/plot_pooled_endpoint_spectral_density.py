#!/usr/bin/env python3
"""Unit-integral pooled occupation density, alpha=3 above alpha=1."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_pooled_endpoint_histograms import ROOT, SOURCE, sha

OUT = ROOT/'analysis_outputs/pooled_endpoint_spectral_density_alpha3_alpha1_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path=SOURCE/'sample_sorted_occupations.npz'
    previous=json.loads((ROOT/'analysis_outputs/pooled_endpoint_occupations_alpha1_alpha3_v1/summary.json').read_text())
    assert sha(path)==previous['source_sha256']
    assert sha(SOURCE/'summary.json')==previous['source_provenance_sha256']
    edges=np.linspace(0,1,51)
    widths=np.diff(edges)
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':8,
                         'mathtext.fontset':'cm','axes.linewidth':.8})
    fig,axes=plt.subplots(2,1,sharex=True,sharey=True,figsize=(3.375,4.25))
    rows=[]; cases={}
    with np.load(path,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['sample_indices'],np.arange(100))
        for ax,alpha,color,letter in zip(axes,(3,1),('#1879bb','#e41a1c'),'ab'):
            values=z[f'alpha1_{alpha}']
            assert values.shape==(100,880) and np.isfinite(values).all()
            assert values.min()>-1e-9 and values.max()<1+1e-9
            counts,_=np.histogram(np.clip(values,0,1).ravel(),bins=edges)
            assert counts.sum()==88000
            density=counts/(values.size*widths)
            np.testing.assert_allclose(np.sum(density*widths),1,atol=1e-15)
            nonzero=counts>0
            ax.bar(edges[:-1][nonzero],density[nonzero],width=widths[nonzero],
                   align='edge',color=color,edgecolor=color,alpha=.75,lw=.5)
            ax.set(yscale='log',ylim=(2e-4,50),xlim=(0,1),ylabel=r'$\rho(\nu)$')
            ax.tick_params(which='both',direction='in',top=True,right=True)
            ax.text(-.19,1.03,f'({letter})',transform=ax.transAxes,fontsize=9)
            ax.text(.5,.92,rf'$\alpha_1={alpha}$',ha='center',va='top',transform=ax.transAxes,fontsize=10)
            for lo,hi,n,d in zip(edges[:-1],edges[1:],counts,density):
                rows.append(dict(alpha_1=alpha,left_edge=float(lo),right_edge=float(hi),
                                 count=int(n),spectral_density=float(d)))
            cases[alpha]=dict(samples=100,modes_per_sample=880,pooled_modes=88000,
                              density_integral=float(np.sum(density*widths)),
                              interior_bin_count=int(counts[1:-1].sum()))
    axes[0].set_title(r'Hard wall: $N_y=40$, $T=160$, $S=100$',fontsize=8)
    axes[1].set_xlabel(r'Endpoint occupation $\nu$')
    axes[1].set_xticks(np.linspace(0,1,6))
    fig.tight_layout(pad=.6,h_pad=.8)
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'pooled_endpoint_spectral_density.{ext}',dpi=300)
    plt.close(fig)
    with (OUT/'histogram_bins.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(source=str(path),source_sha256=sha(path),cases=cases,
                 normalization='rho = count / (88000 * bin width); integral one for each alpha',
                 bin_width=.02,endpoint_cycles=160,Ny=40,Nx=20,
                 estimator='Pool individual occupation eigenvalues, not eigenvalues of the mean covariance.',
                 active_slab_only=True,excluded='Deterministic exterior product modes',
                 display='Logarithmic density; zero bins omitted without pseudocounts; occupation convention [0,1].',
                 numerical_handling='Clip only roundoff outside [0,1] within 1e-9; retain endpoint caps.',
                 script_sha256=sha(Path(__file__)))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(cases,indent=2));print(OUT)


if __name__=='__main__':
    main()
