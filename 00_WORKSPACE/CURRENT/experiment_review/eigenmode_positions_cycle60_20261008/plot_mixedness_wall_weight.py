#!/usr/bin/env python3
"""Cycle-60 preview of residual mixedness and individual-mode wall localization."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
sys.path.insert(0, str(REPO/'00_WORKSPACE/CURRENT/manuscript_Overleaf/figures/new_figure/sources'))
import manuscript_typography as typography

typography.ROOT = OUT
typography.inclusion_width = lambda stem: typography.COLUMN_INCHES
STEM = 'Mixedness_and_wall_weight'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    summary = json.loads((OUT/'analysis_summary.json').read_text())
    assert sha256(OUT/'mode_positions.npz') == summary['outputs_sha256']['mode_positions.npz']
    with np.load(OUT/'mode_positions.npz') as z:
        nu, px, wall = z['occupations'], z['p_x'], z['wall_weight']
        samples = z['sample_ids']
    np.testing.assert_array_equal(samples, np.arange(100))
    assert nu.shape == (100,1200) and px.shape == (100,20,1200)
    assert nu.min() > -1e-10 and nu.max() < 1 + 1e-10
    # Remove only physical-bound roundoff, not finite impurity or nearly pure modes.
    bounded_nu = np.clip(nu,0,1)
    impurity = bounded_nu*(1-bounded_nu)
    profiles = (px*impurity[:,None,:]).sum(axis=2)
    np.testing.assert_allclose(profiles.sum(axis=1), impurity.sum(axis=1), atol=1e-12)
    np.testing.assert_allclose(wall, px[:,[5,6,14,15],:].sum(axis=1), atol=1e-13)
    # Independent check against the original observer's covariance variance contour.
    contour_error = []
    total_error = []
    for source in summary['source_inputs']:
        with np.load(source['path'],allow_pickle=False) as z:
            ids = z['sample_indices']
            reference = z['charge_variance_contour'][:,-1].sum(axis=2)
            totals = z['total_charge_variance'][:,-1]
        contour_error.extend(abs(profiles[ids]-reference).max(axis=1).tolist())
        total_error.extend(abs(profiles[ids].sum(axis=1)-totals).tolist())
    assert max(contour_error) < 1e-10 and max(total_error) < 1e-10
    mean = profiles.mean(axis=0)
    sem = profiles.std(axis=0,ddof=1)/np.sqrt(100)
    mixed = (nu>.005)&(nu<.995)
    typography.configure_style({'axes.linewidth':.7,'xtick.direction':'in',
        'ytick.direction':'in','xtick.top':True,'ytick.right':True})
    fig,axes = plt.subplots(2,1,figsize=(3.375,5.15))
    x = np.arange(20)
    ax=axes[0]
    ax.fill_between(x,np.maximum(mean-sem,0),mean+sem,color='#0072B2',alpha=.16,lw=0)
    ax.plot(x,mean,color='#0072B2',marker='o',mfc='white',mew=.7,ms=3.5,lw=.9)
    for position in (5,15):ax.axvline(position,color='.5',ls='--',lw=.65,zorder=0)
    ax.set(xlabel=r'Column $x$',ylabel=r'Residual mixedness $I(x)$',
           xlim=(-.5,19.5),ylim=(0,float((mean+sem).max()*1.14)),xticks=[0,5,10,15,19])
    ax.text(.5,1.02,r'$\alpha_1=1,\quad t=60,\quad S=100$',
            transform=ax.transAxes,ha='center',va='bottom')
    ax=axes[1]
    ax.scatter(wall.ravel(),nu.ravel(),s=2,facecolors='none',edgecolors='.6',
               linewidths=.3,alpha=.12,rasterized=True)
    ax.scatter(wall[mixed],nu[mixed],s=12,facecolors='none',edgecolors='#0072B2',
               linewidths=.65,alpha=.8,rasterized=True)
    ax.set(xlabel=r'Wall weight $W_j$',ylabel=r'Occupation $\nu_j$',
           xlim=(-.025,1.025),ylim=(-.025,1.025),xticks=[0,.25,.5,.75,1],yticks=[0,.25,.5,.75,1])
    for ax,letter in zip(axes,('a','b')):
        ax.text(-.20,1.04,f'({letter})',transform=ax.transAxes,ha='left',va='bottom')
    typography.prepare_figure(fig,STEM)
    fig.subplots_adjust(left=.235,right=.965,bottom=.105,top=.92,hspace=.54)
    typography.record_typography(fig,STEM)
    for extension in ('.pdf','.png'):fig.savefig(OUT/(STEM+extension),dpi=300)
    plt.close(fig)
    fonts=typography.verify_typography(STEM)
    np.savetxt(OUT/'mixedness_profile.csv',np.column_stack([x,mean,sem]),delimiter=',',
               header='x,mean_mixedness,trajectory_SEM',comments='')
    validation=dict(fonts=fonts,samples=100,cycle=60,alpha_1=1,Nx=20,Ny=30,
        modes_per_trajectory=1200,profile_uses_all_modes=True,profile_has_no_occupation_cutoff=True,
        physical_bound_roundoff_clip_max=float(np.max(abs(nu-bounded_nu))),
        mean_total_mixedness=float(mean.sum()),
        residual_mixedness_fraction_in_wall_columns=float(mean[[5,6,14,15]].sum()/mean.sum()),
        profile_sem='Independent trajectory SEM; nonlinear functional computed before ensemble averaging',
        wall_columns=[5,6,14,15],all_mode_scatter_points=120000,
        highlighted_modes=int(mixed.sum()),highlight_window='0.005 < nu < 0.995; display highlight only',
        original_observer_max_contour_difference=max(contour_error),
        original_observer_max_total_difference=max(total_error),
        input_sha256=sha256(OUT/'mode_positions.npz'),source_sha256=sha256(__file__),
        manuscript_modified=False)
    (OUT/'mixedness_wall_weight_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (OUT/'mixedness_wall_weight_caption.txt').write_text(
        'Cycle-60 residual mixedness and eigenmode localization, alpha1=1, Nx=20, Ny=30, S=100 independent trajectories. '
        'Full-system maximally mixed initial state and full-measurement protocol, matching Figure 3(b). '
        '(a) I(x)=mean sum_j nu_j(1-nu_j) p_j(x), with p_j(x) the eigenmode probability summed over y and both orbitals. '
        'All 1,200 modes per trajectory contribute; no occupation-window cutoff or normalization. Shading: one trajectory SEM. '
        'The expression equals the column sum of diagonal C(1-C), evaluated within each trajectory before averaging; '
        'it is a spatial contour of full-system charge variance, rather than a spatial variance of a column charge. '
        'Dashed lines: interfaces x=5,15. '
        '(b) All mode occupations versus W_j=sum_(x=5,6,14,15) p_j(x), without rank averaging. '
        'Modes with 0.005<nu<0.995 are highlighted with blue open markers; other modes are faint gray. '
        'The cutoff only controls display emphasis. Individual eigenmode weights can depend on basis choice within degenerate eigenspaces. '
        'No new simulations or manuscript changes.\n')
    print(json.dumps(validation,indent=2))


if __name__=='__main__':
    main()
