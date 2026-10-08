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
    return 'schematic', 19 if prominent else 17
typography.text_role = text_role


def ghost_plane(ax, dx, dy, opacity, zorder):
    """Same lattice and slab geometry, with no supports or parameter labels."""
    ax.add_patch(Rectangle((dx, 2+dy), 24, 26, facecolor='#D5DDE3',
                           edgecolor='none', alpha=opacity, zorder=zorder))
    ax.add_patch(Rectangle((6+dx, 2+dy), 12, 26, facecolor='#93B7C8',
                           edgecolor='none', alpha=opacity, zorder=zorder+.1))
    x, y = np.meshgrid(np.arange(24)+.5+dx, np.arange(2,28)+.5+dy)
    ax.scatter(x.ravel(), y.ravel(), s=1.4, color='#566C7B', alpha=opacity*.9,
               edgecolors='none', zorder=zorder+.2)
    for wall in (6,18):
        ax.plot([wall+dx]*2, [2+dy,28+dy], color='#617F93',
                lw=1, ls='--', alpha=opacity, zorder=zorder+.3)
    ax.add_patch(Rectangle((dx,2+dy),24,26,facecolor='none',
                           edgecolor='#627783',lw=1.1,alpha=opacity,zorder=zorder+.4))
    # Matching periodic-boundary marks stay attached to each earlier plane.
    # Their layer-local z-order lets subsequent planes occlude hidden edges.
    for y in (2+dy, 28+dy):
        ax.plot([12+dx-.45, 12+dx+.45], [y-.35, y+.35],
                color='#627783', lw=2, alpha=opacity, solid_capstyle='butt',
                zorder=zorder+.5)
    for x in (dx, 24+dx):
        for offset in (-.4, .4):
            ax.plot([x-.35, x+.35], [15+dy+offset-.5, 15+dy+offset+.5],
                    color='#627783', lw=2, alpha=opacity, solid_capstyle='butt',
                    zorder=zorder+.5)


def main():
    (OUT / "data").mkdir(exist_ok=True)
    geometry._configure_style()
    fig, ax = plt.subplots(figsize=(7.05,7.5))
    fig.subplots_adjust(left=.025,right=.975,bottom=.035,top=.98)
    ghost_plane(ax,-7,6,.72,-30)
    ghost_plane(ax,-3.5,3,.86,-20)
    geometry._draw_geometry(ax)
    # Both dimension arrows use the same stroke weight as the time arrow.
    for artist in ax.texts:
        arrow = getattr(artist, 'arrow_patch', None)
        if arrow is not None:
            arrow.set_linewidth(1.2)
    # Put phase labels inside their regions on the latest plane.
    for label in ax.texts:
        if label.get_text() in (r'$\alpha_1$', r'$\alpha_2$'):
            label.set_y(26.5)
            label.set_verticalalignment('center')
            label.set_zorder(20)
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
            'seam_style':'Background seam marks match their layer outline color and opacity.',
            'arrow_linewidth_pt':1.2,
            'revision':'Periodic-boundary seam marks added to both background layers. Phase labels placed inside foreground regions without background boxes. Larger labels (19 pt prominent, 17 pt dimensions), stronger background layers (0.72 and 0.86 opacity), and darker lattice/boundaries retained.',
            'time_direction':'diagonally down and right toward the foreground',
            'interpretation':'Illustrative time slices, not simulated states or a depiction of changing topology.',
            'original_geometry_source':str(MANUSCRIPT/'figures/new_figure/sources/plot_schematic.py'),
            'typography':review,
            'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/f'{STEM}.pdf',OUT/f'{STEM}.png']}}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Created standalone PDF and PNG:',OUT)

if __name__=='__main__':
    main()
