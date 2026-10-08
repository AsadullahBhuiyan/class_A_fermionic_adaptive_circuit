#!/usr/bin/env python3
"""Unnormalized cycle-60 entropy contours from the verified saved preview arrays."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
REPO=OUT.parents[3]
sys.path.insert(0,str(REPO/'00_WORKSPACE/CURRENT/manuscript_Overleaf/figures/new_figure/sources'))
import manuscript_typography as typography
typography.ROOT=OUT
typography.inclusion_width=lambda stem: typography.TEXT_INCHES
STEM='Raw_entropy_contours_both_cycle60'

def main():
 source=OUT/'entropy_contours_both_cycle60.npz'
 with np.load(source,allow_pickle=False) as z:
  raw={a:z[f'alpha{a}_raw_contours'] for a in (1,3)}
  totals={a:z[f'alpha{a}_entropy_per_trajectory'] for a in (1,3)}
 means={a:raw[a].mean(axis=0) for a in (1,3)}
 for a in (1,3):
  assert raw[a].shape==(100,20,30) and np.isfinite(raw[a]).all() and raw[a].min()>=0
  np.testing.assert_allclose(raw[a].sum(axis=(1,2)),totals[a],rtol=1e-6,atol=1e-14)
  assert means[a].max()<.0075
 typography.configure_style({'axes.linewidth':.7,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
 fig=plt.figure(figsize=(7.05,3.65))
 for index,alpha in enumerate((1,3)):
  ax=fig.add_axes([.09+index*.37,.15,.245,.71])
  im=ax.imshow(means[alpha].T,origin='lower',interpolation='none',aspect='equal',extent=(-.5,19.5,-.5,29.5),cmap='Blues',vmin=0,vmax=.0075)
  ax.set(xlabel='$x$',ylabel='$y$',xticks=[0,5,10,15,19],yticks=[0,10,20,29])
  ax.set_title(r'$\alpha_1='+str(alpha)+'$',pad=7)
  ax.text(-.20,1.035,'('+chr(97+index)+')',transform=ax.transAxes,ha='left',va='bottom')
 cax=fig.add_axes([.80,.15,.018,.71]);cb=fig.colorbar(im,cax=cax)
 cb.set_ticks([0,.0025,.005,.0075]);cb.set_label(r'$\overline{s}(x,y)$',labelpad=5)
 cb.ax.tick_params(pad=2,length=2)
 fig.text(.42,.965,r'$t=60,\quad N_x=20,\quad N_y=30,\quad S=100$',ha='center',va='center')
 typography.prepare_figure(fig,STEM);typography.record_typography(fig,STEM)
 for ext in ('.pdf','.png'):fig.savefig(OUT/(STEM+ext),dpi=300)
 plt.close(fig)
 validation={'Nx':20,'Ny':30,'cycle':60,'samples_per_alpha':100,'alpha_1':[1,3],
  'estimator':'Arithmetic mean of raw full-system entropy contour over independent trajectories, no normalization; natural logarithms (nats)',
  'initialization':'Full system maximally mixed','shared_color_scale':{'min':0,'max':.0075,'type':'linear','cmap':'Blues'},
  'mean_total_entropies':{str(a):float(means[a].sum()) for a in (1,3)},
  'map_ranges':{str(a):[float(means[a].min()),float(means[a].max())] for a in (1,3)},
  'source_data':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
  'source_validation':'entropy_contours_both_cycle60_validation.json',
  'alpha3_caveat':'Entropy is at the numerical floor; continuation uses covariance spectral clipping after cycle30.',
  'fonts':typography.verify_typography(STEM),'manuscript_modified':False,'new_simulations':False}
 (OUT/'raw_entropy_contours_cycle60_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
 (OUT/'raw_entropy_contours_cycle60_caption.txt').write_text('Raw full-system entropy contour at cycle60 for alpha1=1,3; Nx20,Ny30; 100 independent trajectories per phase initially fully maximally mixed. Average each trajectory contour without normalization. Sum of each mean map equals mean total entropy (natural-log units). Both orbitals summed, y dependence retained, no smoothing, shared linear color scale 0..0.0075. Alpha1=3 is at the numerical entropy floor and its continuation uses covariance clipping after cycle30. All source arrays and the manuscript remain unchanged.\n')
 print(json.dumps(validation['mean_total_entropies']))
if __name__=='__main__':main()
