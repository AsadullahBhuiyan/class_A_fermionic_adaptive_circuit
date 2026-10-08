#!/usr/bin/env python3
"""Standalone normalized full-system entropy contours; no manuscript edits."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
REPO=OUT.parents[3]
BUNDLE=REPO/'00_WORKSPACE/CURRENT/manuscript_Overleaf/figures/new_figure'
sys.path.insert(0,str(BUNDLE/'sources'))
import manuscript_typography as typography
typography.ROOT=OUT
typography.inclusion_width=lambda stem: typography.TEXT_INCHES
STEM='Normalized_entropy_contours_both_cycle60'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 provenance=json.loads((BUNDLE/'data/purification/occupation_source_provenance.json').read_text())
 arrays={}; checks={}; sources=[]
 for alpha in (1,3):
  contours=np.empty((100,20,30)); entropies=np.empty(100); seen=set(); spectral_error=0
  inputs=[s for s in provenance['inputs'] if f'/alpha1_{alpha}/' in s['path']]
  with threadpool_limits(limits=4):
   for s in tqdm(inputs,desc=f'alpha1={alpha}: saved contours',unit='shard'):
    assert sha(s['path'])==s['sha256'];sources.append(s)
    with np.load(s['path'],allow_pickle=False) as z:
     assert int(z['alpha_1'])==alpha and int(z['cycles'][-1])==60
     ids=z['sample_indices'];assert not seen.intersection(ids.tolist());seen.update(ids.tolist())
     entropies[ids]=z['total_entropy'][:,-1]
     if 'entropy_contour' in z.files:contours[ids]=z['entropy_contour'][:,-1]
     else:
      assert str(z['centered_covariance_convention'])=='G=2C-I'
      for sid,g,saved in zip(ids,z['G_final'],z['occupation_spectrum'][:,-1]):
       c=(g+g.conj().T)/4; c[np.diag_indices(1200)]+=.5
       nu,u=eigh(c,driver='evd',check_finite=False)
       spectral_error=max(spectral_error,float(abs(nu-saved).max()))
       v=np.clip(nu,1e-12,1-1e-12); h=-(v*np.log(v)+(1-v)*np.log(1-v))
       contours[sid]=((abs(u)**2)@h).reshape(30,20,2).sum(2).T
  assert seen==set(range(100)) and np.isfinite(contours).all() and contours.min()>=0 and entropies.min()>0
  closure=float(abs(contours.sum((1,2))-entropies).max());assert closure<1e-10 and spectral_error<1e-12
  # Use reconstructed/saved map sum as denominator for exact unit normalization.
  totals=contours.sum((1,2)); norm=contours/totals[:,None,None]; mean=norm.mean(0)
  np.testing.assert_allclose(norm.sum((1,2)),1,atol=1e-12)
  assert mean.max()<.02
  for name,a in [('raw_contours',contours),('entropy_per_trajectory',entropies),('normalized_contours',norm),('mean_normalized_contour',mean),('trajectory_SEM',norm.std(0,ddof=1)/10)]:arrays[f'alpha{alpha}_{name}']=a
  checks[str(alpha)]={'samples':100,'entropy_range':[float(entropies.min()),float(entropies.max())],
    'mean_entropy':float(entropies.mean()),'normalized_map_sum':float(mean.sum()),'map_range':[float(mean.min()),float(mean.max())],
    'entropy_closure_max_error':closure,'reconstructed_occupation_max_error':spectral_error,
    'wall_fraction':float(mean[[5,6,14,15]].sum()),
    'contour_source':'Saved observer contour' if alpha==1 else 'Reconstructed from saved cycle-60 covariance with observer entropy cutoff 1e-12'}
 np.savez_compressed(OUT/'entropy_contours_both_cycle60.npz',sample_ids=np.arange(100),**arrays)
 typography.configure_style({'axes.linewidth':.7,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
 fig=plt.figure(figsize=(7.05,3.65))
 for index,alpha in enumerate((1,3)):
  ax=fig.add_axes([.09+index*.37,.15,.245,.71])
  im=ax.imshow(arrays[f'alpha{alpha}_mean_normalized_contour'].T,origin='lower',interpolation='none',aspect='equal',extent=(-.5,19.5,-.5,29.5),cmap='Blues',vmin=0,vmax=.02)
  ax.set(xlabel='$x$',ylabel='$y$',xticks=[0,5,10,15,19],yticks=[0,10,20,29])
  ax.set_title(r'$\alpha_1='+str(alpha)+'$',pad=7)
  ax.text(-.20,1.035,'('+chr(97+index)+')',transform=ax.transAxes,ha='left',va='bottom')
 cax=fig.add_axes([.80,.15,.018,.71]);cb=fig.colorbar(im,cax=cax)
 cb.set_ticks([0,.005,.01,.015,.02]);cb.set_label(r'$\overline{\widetilde{s}}(x,y)$',labelpad=5)
 cb.ax.tick_params(pad=2,length=2)
 fig.text(.42,.965,r'$t=60,\quad N_x=20,\quad N_y=30,\quad S=100$',ha='center',va='center')
 typography.prepare_figure(fig,STEM);typography.record_typography(fig,STEM)
 for ext in ('.pdf','.png'):fig.savefig(OUT/(STEM+ext),dpi=300)
 plt.close(fig)
 validation={'Nx':20,'Ny':30,'cycle':60,'alpha_1':[1,3],'initialization':'full system maximally mixed',
  'normalization':'Normalize each full-system entropy contour to sum one, then average over 100 independent trajectories',
  'shared_color_scale':{'cmap':'Blues','minimum':0,'maximum':.02,'scaling':'linear'},
  'alpha3_limitation':'Covariance spectral clipping after cycle 30; cycle-60 entropy is at observer numerical entropy floor. Normalized map does not establish physical residual mixedness.',
  'checks':checks,'source_inputs':sources,'script_sha256':sha(__file__),'fonts':typography.verify_typography(STEM),'manuscript_modified':False,'new_simulations':False}
 (OUT/'entropy_contours_both_cycle60_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
 (OUT/'entropy_contours_both_cycle60_caption.txt').write_text('Cycle-60 full-system purification entropy contours for alpha1=1,3, Nx=20,Ny=30, S=100 independent maximally mixed initial trajectories per parameter. Each trajectory contour is normalized to unit sum before averaging. Both panels share one linear Blues color scale, 0 to 0.02 per unit cell, sum both orbitals, retain y dependence, and use no smoothing. Alpha1=1 uses saved contours; alpha1=3 reconstructs contours from saved final covariances using the observer entropy cutoff of 1e-12. The alpha1=3 continuation uses spectral clipping after cycle30 and its endpoint entropy is at the numerical floor; its nearly uniform normalized map is not evidence of physical residual entropy. Manuscript untouched; no new simulations.\n')
 print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
