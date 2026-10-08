#!/usr/bin/env python3
"""Standalone preview: eigenmode centers versus position-resolved spectral weight."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
sys.path.insert(0,str(REPO/'00_WORKSPACE/CURRENT/manuscript_Overleaf/figures/new_figure/sources'))
import manuscript_typography as typography

# Isolate standalone typography receipts; the production manuscript is untouched.
typography.ROOT = OUT
typography.inclusion_width = lambda stem: typography.COLUMN_INCHES
STEM = 'Occupation_vs_mode_position'


def main():
    typography.configure_style({'axes.linewidth':.7,'xtick.direction':'in',
        'ytick.direction':'in','xtick.top':True,'ytick.right':True})
    (OUT/'data').mkdir(exist_ok=True)
    with np.load(OUT/'mode_positions.npz') as z:
        nu, mx = z['occupations'], z['mean_x']
        weight, edges = z['spatial_spectral_weight'], z['occupation_bin_edges']
    fig, axes = plt.subplots(2,1,figsize=(3.375,5.55))
    axes[0].scatter(mx.ravel(),nu.ravel(),s=2.5,facecolors='none',edgecolors='#0072B2',
                    linewidths=.3,alpha=.18,rasterized=True)
    mixed = (nu > .005) & (nu < .995)
    axes[0].scatter(mx[mixed],nu[mixed],s=9,facecolors='none',edgecolors='#0072B2',
                    linewidths=.55,alpha=.75,rasterized=True)
    axes[0].set(xlabel=r'Mode center $\langle x\rangle_j$',ylabel=r'Occupation $\nu_j$')
    axes[0].text(.5,1.02,r'$\alpha_1=1,\quad t=60,\quad S=100$',
                 transform=axes[0].transAxes,ha='center',va='bottom')
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad(cmap(0))
    cmap.set_under(cmap(0))
    image=axes[1].pcolormesh(np.arange(-.5,20.5),edges,weight.T,
        cmap=cmap,norm=LogNorm(vmin=1e-4,vmax=weight.max()),shading='flat',rasterized=True)
    axes[1].set(xlabel=r'Column $x$',ylabel=r'Occupation $\nu$')
    color_axis=fig.add_axes([.23,.105,.73,.017])
    colorbar=fig.colorbar(image,cax=color_axis,orientation='horizontal',extend='min')
    colorbar.set_ticks([1e-4,1e-2,1,10])
    colorbar.set_label('Mean mode weight per bin',labelpad=1)
    colorbar.ax.tick_params(labelsize=8,pad=2,length=2)
    for ax,letter in zip(axes,('a','b')):
        for wall in (5,15):ax.axvline(wall,color='.4',ls='--',lw=.6,zorder=1)
        ax.set(xlim=(-.5,19.5),ylim=(-.025,1.025),xticks=[0,5,10,15,19],yticks=[0,.25,.5,.75,1])
        ax.text(-.20,1.04,f'({letter})',transform=ax.transAxes,ha='left',va='bottom')
    typography.prepare_figure(fig,STEM)
    fig.subplots_adjust(left=.23,right=.965,bottom=.235,top=.92,hspace=.58)
    typography.record_typography(fig,STEM)
    for suffix in ('.pdf','.png'):fig.savefig(OUT/(STEM+suffix),dpi=300)
    plt.close(fig)
    fonts=typography.verify_typography(STEM)
    (OUT/'figure_validation.json').write_text(json.dumps(fonts,indent=2)+'\n')
    (OUT/'caption.txt').write_text(
        'Cycle-60 occupation and position for all 1,200 covariance modes in each of 100 alpha1=1 trajectories, Nx=20, Ny=30. '
        'Initial state: full-system maximally mixed; full-measurement protocol matching Figure 3(b). '
        '(a) Every mode plotted against its linear mean coordinate <x>, pooling trajectories without rank averaging; '
        'modes with 0.005<nu<0.995 are highlighted with larger, darker open markers. '
        '(b) Mean spatial spectral weight: each mode contributes its full probability profile p_j(x), rather than one point at its center. '
        'Occupation bins have width 0.025; color is mean mode weight per bin, on a logarithmic scale, without unit-area normalization. '
        'Columns and occupations integrate to 1,200 modes per trajectory; every column integrates to 60. '
        'Dashed lines mark x=5,15. Near-pure modes can be numerically degenerate, so their individual centers depend on the chosen eigenbasis; '
        'the spectral-weight map is invariant under rotations within eigenspaces contained in the same bin. '
        'No manuscript changes or new simulations.\n')
    print(json.dumps(fonts,indent=2))


if __name__ == '__main__':
    main()
