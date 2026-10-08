#!/usr/bin/env python3
"""Normalized cycle-60 full-system entropy contour from the saved observer."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[3]
sys.path.insert(0,str(REPO/'00_WORKSPACE/CURRENT/manuscript_Overleaf/figures/new_figure/sources'))
import manuscript_typography as typography
typography.ROOT=OUT
typography.inclusion_width=lambda stem: typography.COLUMN_INCHES
STEM='Normalized_entropy_contour_cycle60'


def main():
    summary=json.loads((OUT/'analysis_summary.json').read_text())
    contours=np.empty((100,20,30))
    entropies=np.empty(100)
    seen=set()
    for source in summary['source_inputs']:
        with np.load(source['path'],allow_pickle=False) as z:
            assert int(z['alpha_1'])==1 and int(z['cycles'][-1])==60
            ids=z['sample_indices']
            assert not seen.intersection(ids.tolist())
            seen.update(ids.tolist())
            contours[ids]=z['entropy_contour'][:,-1]
            entropies[ids]=z['total_entropy'][:,-1]
    assert seen==set(range(100))
    assert np.isfinite(contours).all() and contours.min()>=0 and entropies.min()>0
    closure=np.max(abs(contours.sum(axis=(1,2))-entropies))
    assert closure<1e-10
    normalized=contours/entropies[:,None,None]
    np.testing.assert_allclose(normalized.sum(axis=(1,2)),1,atol=1e-12)
    mean=normalized.mean(axis=0)
    sem=normalized.std(axis=0,ddof=1)/10
    raw_mean=contours.mean(axis=0)
    mean_then_normalize=raw_mean/raw_mean.sum()
    np.testing.assert_allclose(mean.sum(),1,atol=1e-12)
    typography.configure_style({'axes.linewidth':.7,'xtick.direction':'in',
        'ytick.direction':'in','xtick.top':True,'ytick.right':True})
    fig=plt.figure(figsize=(3.375,3.9))
    ax=fig.add_axes([.17,.16,.54,.70])
    image=ax.imshow(mean.T,origin='lower',interpolation='none',aspect='equal',
        extent=(-.5,19.5,-.5,29.5),cmap='Blues',vmin=0,vmax=.02)
    assert mean.max()<.02
    ax.set(xlabel='$x$',ylabel='$y$',xticks=[0,5,10,15,19],yticks=[0,10,20,29])
    ax.text(.5,1.045,r'$\alpha_1=1,\quad t=60,\quad S=100$',
            transform=ax.transAxes,ha='center',va='bottom')
    cax=fig.add_axes([.755,.16,.035,.70])
    colorbar=fig.colorbar(image,cax=cax)
    colorbar.set_ticks([0,.005,.01,.015,.02])
    colorbar.set_label(r'$\overline{\widetilde{s}}(x,y)$',labelpad=4)
    colorbar.ax.tick_params(labelsize=8,pad=2,length=2)
    typography.prepare_figure(fig,STEM)
    typography.record_typography(fig,STEM)
    for ext in ('.pdf','.png'):fig.savefig(OUT/(STEM+ext),dpi=300)
    plt.close(fig)
    fonts=typography.verify_typography(STEM)
    np.savez_compressed(OUT/'entropy_contour_cycle60.npz',sample_ids=np.arange(100),
        raw_contours=contours,entropy_per_trajectory=entropies,
        normalized_contours=normalized,mean_normalized_contour=mean,
        trajectory_SEM=sem,mean_raw_contour=raw_mean,
        normalized_ensemble_mean_contour=mean_then_normalize)
    validation=dict(fonts=fonts,samples=100,Nx=20,Ny=30,cycle=60,alpha_1=1,
        initialization='Full system maximally mixed',protocol='full measurement; matching Figure 3(b)',
        source_observable='Saved full-system entropy contour; y not translation-averaged',
        normalization='s_m(x,y)/S_m summed to one within each trajectory, then mean over trajectories',
        entropy_contour_definition='s_m(x,y)=sum_j h(nu_j) sum_orbital |u_j(x,y,orbital)|^2',
        normalized_map_sum=float(mean.sum()),map_min=float(mean.min()),map_max=float(mean.max()),
        total_entropy_range=[float(entropies.min()),float(entropies.max())],
        entropy_closure_max_error=float(closure),
        wall_columns=[5,6,14,15],
        normalized_first_wall_fraction=float(mean[[5,6,14,15]].sum()),
        mean_first_normalized_wall_fraction=float(mean_then_normalize[[5,6,14,15]].sum()),
        normalization_order_L1_difference=float(abs(mean-mean_then_normalize).sum()),
        color_scale='linear, 0 to 0.02, same units as probability per cell',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manuscript_modified=False)
    (OUT/'entropy_contour_cycle60_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (OUT/'entropy_contour_cycle60_caption.txt').write_text(
        'Normalized full-system entropy contour at cycle 60 for alpha1=1, Nx=20, Ny=30, '
        'S=100 independent full-measurement trajectories starting from a fully maximally mixed state. '
        'Each saved contour s_m(x,y) sums to that trajectory\'s full-system entropy S_m. '
        'We divide each map by S_m and then average, so the displayed map sums to one and each trajectory has equal weight. '
        'Both orbitals are summed; y dependence is retained with no translation average, smoothing, or interpolation. '
        'This is the full-system purification entropy contour, not the half-strip entanglement contour. '
        'No new simulations or manuscript changes.\n')
    print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
