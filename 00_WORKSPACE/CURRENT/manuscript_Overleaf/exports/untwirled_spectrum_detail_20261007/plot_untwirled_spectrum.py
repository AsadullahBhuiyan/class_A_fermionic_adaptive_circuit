#!/usr/bin/env python3
"""Standalone enlarged view of Figure 10's untwirled occupation inset."""
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
from manuscript_palette import ALPHA_COLORS
DATA=MANUSCRIPT/'figures/new_figure/data/mean_channel'
STEM='untwirled_occupation_spectra'
typography.ROOT=OUT
typography.inclusion_width=lambda stem: 6.5
def role(fig,artist,stem):
    for ax in fig.axes:
        if artist is ax.xaxis.label or artist is ax.yaxis.label:
            return 'axis',14
    return 'annotation_or_tick',12
typography.text_role=role
typography.configure_style({'xtick.direction':'in','ytick.direction':'in'})
fig,ax=plt.subplots(figsize=(6.5,4.2))
provenance=json.loads((DATA/'untwirled_spectra_provenance.json').read_text())
assert hashlib.sha256((DATA/'untwirled_spectra.npz').read_bytes()).hexdigest()==provenance['sha256']
with np.load(DATA/'untwirled_spectra.npz',allow_pickle=False) as saved:
    for alpha,marker in ((1,'o'),(3,'^')):
        nu=saved[f'alpha{alpha}']
        assert len(nu)==2560 and np.all(np.diff(nu)>=0)
        ax.plot(np.arange(1,len(nu)+1),nu,color=ALPHA_COLORS[alpha],
                marker=marker,markevery=80,ms=5,mfc='white',mew=.9,
                lw=1.2,ls='-' if alpha==1 else ':',label=rf'$\alpha_1={alpha}$')
ax.set(xlabel=r'Ordered mode index $j$',ylabel=r'Occupation $\nu_j$',
       xlim=(1,2560),ylim=(-.04,1.04))
ax.set_xticks([1,640,1280,1920,2560])
ax.set_yticks([0,.25,.5,.75,1])
ax.tick_params(top=True,right=True)
ax.legend(loc='upper left',frameon=False,handlelength=2)
ax.set_title(r'Untwirled covariance: $N_x=20$, $N_y=64$, cycle $128$',pad=12)
typography.prepare_figure(fig,STEM)
fig.subplots_adjust(left=.13,right=.965,bottom=.17,top=.87)
typography.record_typography(fig,STEM)
for ext in ('pdf','png'):fig.savefig(OUT/f'{STEM}.{ext}',dpi=300)
plt.close(fig)
review=typography.verify_typography(STEM)
(OUT/'validation.json').write_text(json.dumps({'source':provenance,'typography':review,'all_2560_eigenvalues_plotted':True,'marker_stride':80},indent=2)+'\n')
print(OUT/f'{STEM}.pdf')
