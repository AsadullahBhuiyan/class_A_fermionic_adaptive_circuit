#!/usr/bin/env python3
"""Finite-time signed Lyapunov density with numerical cap mass kept separate."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_pooled_endpoint_histograms import ROOT, SOURCE, sha

OUT=ROOT/'analysis_outputs/pooled_lyapunov_spectral_density_v1'
T=160
CAP=1e-9


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=SOURCE/'sample_sorted_occupations.npz'
    prior=ROOT/'analysis_outputs/pooled_endpoint_spectral_density_alpha3_alpha1_v1/summary.json'
    assert sha(source)==json.loads(prior.read_text())['source_sha256']
    bound=np.log((1-CAP)/CAP)/(2*T)
    edges=np.linspace(-bound,bound,41)
    widths=np.diff(edges)
    cases={}; rows=[]; samples=[]; densities={}
    with np.load(source,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['sample_indices'],np.arange(100))
        for alpha in (1,3):
            nu=z[f'alpha1_{alpha}']
            assert nu.shape==(100,880) and np.isfinite(nu).all()
            assert nu.min()>-CAP and nu.max()<1+CAP
            lower=nu<=CAP; upper=nu>=1-CAP
            finite=~(lower|upper)
            lam=(np.log1p(-nu[finite])-np.log(nu[finite]))/(2*T)
            np.testing.assert_allclose(1/(1+np.exp(2*T*lam)),nu[finite],atol=1e-15)
            counts,_=np.histogram(lam,bins=edges)
            assert counts.sum()==finite.sum()
            rho=counts/(nu.size*widths)
            densities[alpha]=rho
            integral=float(np.sum(rho*widths))
            np.testing.assert_allclose(integral+(lower.sum()+upper.sum())/nu.size,1)
            cases[alpha]=dict(total_modes=nu.size,finite_modes=int(finite.sum()),
                              finite_mass=integral,positive_cap_modes=int(lower.sum()),
                              negative_cap_modes=int(upper.sum()),
                              positive_cap_mass=float(lower.mean()),negative_cap_mass=float(upper.mean()))
            for (sample,rank),value in zip(np.argwhere(finite),lam):
                samples.append(dict(alpha_1=alpha,sample_index=int(sample),rank=int(rank),signed_lambda=float(value)))
            for lo,hi,n,d in zip(edges[:-1],edges[1:],counts,rho):
                rows.append(dict(alpha_1=alpha,left_edge=float(lo),right_edge=float(hi),count=int(n),density=float(d)))
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':8,
                         'mathtext.fontset':'cm','axes.linewidth':.8})
    fig,ax=plt.subplots(figsize=(3.375,3.25))
    red,blue='#e41a1c','#1879bb'
    ax.stairs(densities[1],edges,baseline=None,color=red,lw=1.1)
    mask=densities[1]>0
    ax.plot(((edges[:-1]+edges[1:])/2)[mask],densities[1][mask],ls='none',
            marker='^',ms=3,mfc='white',mew=.7,color=red)
    assert not np.any(densities[3])  # Do not fabricate a blue finite-spectrum curve.
    ax.legend(handles=[Line2D([],[],color=red,marker='^',mfc='white',ms=3,label=r'$\alpha_1=1$'),
                       Line2D([],[],color=blue,ls='--',label=r'$\alpha_1=3$: no resolved finite modes')],
              frameon=False,loc='upper center',fontsize=7,handlelength=1.5)
    ax.set(xlim=(-bound*1.04,bound*1.04),yscale='log',ylim=(.002,1.5),
           xlabel=r'$\lambda=\log[(1-\nu)/\nu]/(2T)$',ylabel=r'$\rho_{\mathrm{finite}}(\lambda)$')
    ax.set_xticks([-.06,-.03,0,.03,.06])
    ax.tick_params(which='both',direction='in',top=True,right=True)
    ax.set_title(r'Hard wall: $N_y=40$, $T=160$, $S=100$',fontsize=8)
    notes=[]
    for alpha in (1,3):
        c=cases[alpha]
        notes.append(rf"$\alpha_1={alpha}$: finite {100*c['finite_mass']:.3f}\%; "
                     rf"caps $(-/+)$ {100*c['negative_cap_mass']:.2f}/{100*c['positive_cap_mass']:.2f}\%")
    fig.text(.5,.015,'\n'.join(notes).replace(r'\%','%'),ha='center',va='bottom',fontsize=7)
    fig.tight_layout(pad=.6,rect=(0,.13,1,1))
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'pooled_lyapunov_spectral_density.{ext}',dpi=300)
    plt.close(fig)
    for name,data in [('histogram_bins.csv',rows),('finite_sample_exponents.csv',samples)]:
        with (OUT/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    summary=dict(source=str(source),source_sha256=sha(source),Ny=40,T=T,samples=100,
                 modes_per_sample=880,cap_tolerance=CAP,finite_resolution_bound=bound,cases=cases,
                 convention='signed lambda = log((1-nu)/nu)/(2T); not the many-body spectrum',
                 normalization='count/(88000*bin_width); finite density integrates to finite-mode fraction, not one',
                 cap_interpretation='nu<=1e-9 is +infinity and nu>=1-1e-9 is -infinity under the numerical cap convention; these are unresolved tails, not evidence of physically exact infinities',
                 script_sha256=sha(Path(__file__)))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(cases,indent=2));print(OUT)


if __name__=='__main__':
    main()
