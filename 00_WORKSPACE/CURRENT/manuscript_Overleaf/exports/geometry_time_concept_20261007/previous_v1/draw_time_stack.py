#!/usr/bin/env python3
"""Standalone time-stack concept using the manuscript's unchanged geometry."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent
MANUSCRIPT = OUT.parents[1]
sys.path.insert(0, str(MANUSCRIPT / 'figures/new_figure/sources'))
import manuscript_typography as typography
import plot_schematic as geometry

STEM = 'domain_wall_time_stack'
# Register this standalone concept only in this process; do not edit the shared style.
original_width = typography.inclusion_width
original_role = typography.text_role
typography.ROOT = OUT
typography.inclusion_width = lambda stem: 7.05 if stem == STEM else original_width(stem)
def text_role(fig, artist, stem):
    if stem != STEM:
        return original_role(fig, artist, stem)
    value = artist.get_text()
    prominent = ('\\alpha' in value or '\\mathcal' in value or value == r'cycle $t$')
    return 'schematic', 11 if prominent else 10
typography.text_role = text_role


def ghost_plane(ax, dx, dy, opacity, zorder):
    """Same lattice and slab geometry, with no supports or parameter labels."""
    ax.add_patch(Rectangle((dx, 2+dy), 24, 26, facecolor='#E4E8EC',
                           edgecolor='none', alpha=opacity, zorder=zorder))
    ax.add_patch(Rectangle((6+dx, 2+dy), 12, 26, facecolor='#B4C8D1',
                           edgecolor='none', alpha=opacity, zorder=zorder+.1))
    x, y = np.meshgrid(np.arange(24)+.5+dx, np.arange(2,28)+.5+dy)
    ax.scatter(x.ravel(), y.ravel(), s=.8, color='#647887', alpha=opacity*.75,
               edgecolors='none', zorder=zorder+.2)
    for wall in (6,18):
        ax.plot([wall+dx]*2, [2+dy,28+dy], color='#7894A4',
                lw=1, ls='--', alpha=opacity, zorder=zorder+.3)
    ax.add_patch(Rectangle((dx,2+dy),24,26,facecolor='none',
                           edgecolor='#81919C',lw=.9,alpha=opacity,zorder=zorder+.4))


def main():
    (OUT / "data").mkdir(exist_ok=True)
    geometry._configure_style()
    fig, ax = plt.subplots(figsize=(7.05,7.5))
    fig.subplots_adjust(left=.025,right=.975,bottom=.035,top=.98)
    ghost_plane(ax,-7,6,.38,-30)
    ghost_plane(ax,-3.5,3,.60,-20)
    geometry._draw_geometry(ax)
    # The final plane retains the original supports, seam marks, and dimensions.
    geometry._arrow(ax,(-9,3.5),(-1,-3.35),lw=1.2,mutation_scale=12)
    ax.text(-5.5,-1.15,r'cycle $t$',rotation=-40.6,
            ha='center',va='center',rotation_mode='anchor')
    ax.set_xlim(-10.5,28.5)
    ax.set_ylim(-5.7,35.5)
    ax.set_aspect('equal')
    ax.axis('off')
    typography.prepare_figure(fig,STEM)
    typography.record_typography(fig,STEM)
    fig.savefig(OUT/f'{STEM}.pdf')
    fig.savefig(OUT/f'{STEM}.png',dpi=300)
    plt.close(fig)
    review=typography.verify_typography(STEM)
    result={'concept':'Three offset geometry slices with muted earlier layers and full annotations on the foreground/latest slice.',
            'offsets_oldest_to_latest':[[-7,6],[-3.5,3],[0,0]],
            'time_direction':'diagonally down and right toward the foreground',
            'interpretation':'Illustrative time slices, not simulated states or a depiction of changing topology.',
            'original_geometry_source':str(MANUSCRIPT/'figures/new_figure/sources/plot_schematic.py'),
            'typography':review,
            'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/f'{STEM}.pdf',OUT/f'{STEM}.png']}}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Created standalone PDF and PNG:',OUT)

if __name__=='__main__':
    main()
