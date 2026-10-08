#!/usr/bin/env python3
"""Plot existing untwirled spectra only; no dynamics or eigensolver runs."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
MANUSCRIPT=OUT.parents[1]
sys.path.insert(0,str(MANUSCRIPT/'figures/new_figure/sources'))
import manuscript_typography as typography
STEM='untwirled_occupation_saved_cycles'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
protected={name:sha(MANUSCRIPT/name) for name in ['manuscript.tex','manuscript.pdf','figures/new_figure/manifest.json']}
summary=json.loads((MANUSCRIPT/'figures/new_figure/data/mean_channel/spectrum_summary.json').read_text())
arrays={};provenance={}
for alpha in (1,3):
    info=summary[f'alpha{alpha}_hard']; source=Path(info['source'])
    assert sha(source)==info['source_sha256']
    with np.load(source,allow_pickle=False) as saved:
        times=saved['spectral_cycles']; occ=saved['active_occupations']
        np.testing.assert_array_equal(times,[0,32,64,96,128])
        assert occ.shape==(5,2560)
        assert np.all(np.diff(occ,axis=1)>=-1e-14)
        np.testing.assert_array_equal(saved['active_indices'],np.arange(2560))
        assert saved['frozen_exterior_indices'].size==0
        np.testing.assert_allclose(occ[0],.5,atol=1e-14,rtol=0)
        arrays[f'alpha{alpha}']=occ.copy()
        provenance[str(alpha)]={'source':str(source),'source_sha256':info['source_sha256'],
          'keys':['spectral_cycles','active_occupations'],'cycles':times.tolist(),
          'max_spectrum_change_cycle32_to128':float(np.max(np.abs(occ[1:]-occ[-1]))),
          'intermediate_occupations_0p01_to_0p99': [int(np.sum((nu>.01)&(nu<.99))) for nu in occ]}
arrays['cycles']=times
np.savez_compressed(OUT/'data/saved_spectra.npz',**arrays)
typography.ROOT=OUT
typography.inclusion_width=lambda stem:6.5
def role(fig,artist,stem):
    for ax in fig.axes:
        if artist is ax.xaxis.label or artist is ax.yaxis.label:return 'axis',14
    return 'annotation_or_tick',12
typography.text_role=role
typography.configure_style({'xtick.direction':'in','ytick.direction':'in'})
fig,axes=plt.subplots(2,1,figsize=(6.5,7.8),sharex=True)
colors=['#777777','#D55E00','#009E73','#0072B2','#CC79A7']
markers=['s','^','o','D','v']
styles=['--',':','--','-.','-']
for ax,alpha in zip(axes,(1,3)):
    for index,(cycle,nu,color,marker,style) in enumerate(zip(times,arrays[f'alpha{alpha}'],colors,markers,styles)):
        # Stagger bulk markers for coincident curves, retaining every intermediate
        # eigenvalue marker so the near-half-filled spectrum is not undersampled.
        sparse=np.arange(index*16,len(nu),80)
        mid=np.flatnonzero((nu>.01)&(nu<.99)) if cycle else np.array([],dtype=int)
        marked=np.union1d(sparse,mid)
        ax.plot(np.arange(1,len(nu)+1),nu,color=color,marker=marker,markevery=marked,
                ms=2.2 if cycle else 3,mfc='none',mew=.7,lw=.9,ls=style,label=rf'${cycle}$')
    ax.set(ylabel=r'Occupation $\nu_j$',xlim=(1,2560),ylim=(-.04,1.04))
    ax.set_xticks([1,640,1280,1920,2560]); ax.set_yticks([0,.25,.5,.75,1])
    ax.tick_params(top=True,right=True)
    ax.legend(title=r'cycle $t$',loc='upper left',frameon=False,ncol=2,handlelength=1.7,
              columnspacing=1,handletextpad=.5)
    ax.set_title(rf'$\alpha_1={alpha}$',pad=9)
axes[-1].set_xlabel(r'Ordered mode index $j$')
fig.suptitle(r'Untwirled covariance spectra: $N_x=20$, $N_y=64$',y=.98)
typography.prepare_figure(fig,STEM)
fig.subplots_adjust(left=.14,right=.96,bottom=.09,top=.89,hspace=.27)
typography.record_typography(fig,STEM)
for ext in ('pdf','png'):fig.savefig(OUT/f'{STEM}.{ext}',dpi=300)
plt.close(fig)
review=typography.verify_typography(STEM)
assert all(sha(MANUSCRIPT/name)==value for name,value in protected.items())
(OUT/'validation.json').write_text(json.dumps({'provenance':provenance,'typography':review,
    'no_simulations_or_eigensolver_runs':True,'all_saved_eigenvalues_plotted':True,
    'marker_policy':'Staggered markers in the bulk; every eigenvalue with 0.01 < nu < 0.99 marked for nonzero cycles.',
    'manuscript_unchanged':protected},indent=2)+'\n')
print(json.dumps(provenance,indent=2))
print(OUT/f'{STEM}.pdf')
