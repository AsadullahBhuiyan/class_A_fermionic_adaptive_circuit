#!/usr/bin/env python3
"""Four successive OW-support placements; standalone schematic concept."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
OUT=Path(__file__).resolve().parent
MANUSCRIPT=OUT.parents[1]
sys.path.insert(0,str(MANUSCRIPT/'figures/new_figure/sources'))
import manuscript_typography as typography
import plot_schematic as geometry
STEM='domain_wall_four_step_sequence'
WIDTH=24; BOTTOM=2; TOP=28; LEFT=6; RIGHT=18
CENTER_Y=4.5
STAGES=[
    dict(dx=0,dy=0,opacity=.72,x=2.5,lo=0,hi=LEFT,region='trivial',kind='full trivial support'),
    dict(dx=2.8,dy=4.4,opacity=.80,x=5.5,lo=0,hi=LEFT,region='trivial',kind='trivial interface support'),
    dict(dx=5.6,dy=8.8,opacity=.88,x=6.5,lo=LEFT,hi=RIGHT,region='topological',kind='topological interface support'),
    dict(dx=8.4,dy=13.2,opacity=1,x=9.5,lo=LEFT,hi=RIGHT,region='topological',kind='full topological support'),
]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def draw_slice(ax,stage,index):
    dx,dy,alpha=stage['dx'],stage['dy'],stage['opacity']; z=20*index
    final=index==len(STAGES)-1
    bg=geometry.LIGHT_GRAY if final else '#D5DDE3'
    slab=geometry.TOPOLOGICAL if final else '#93B7C8'
    edge=geometry.MID_GRAY if final else '#627783'
    wall=geometry.INTERFACE if final else '#617F93'
    ax.add_patch(Rectangle((dx,BOTTOM+dy),WIDTH,TOP-BOTTOM,facecolor=bg,edgecolor='none',alpha=alpha,zorder=z))
    ax.add_patch(Rectangle((LEFT+dx,BOTTOM+dy),RIGHT-LEFT,TOP-BOTTOM,facecolor=slab,edgecolor='none',alpha=alpha,zorder=z+.1))
    xx,yy=np.meshgrid(np.arange(WIDTH)+.5+dx,np.arange(BOTTOM,TOP)+.5+dy)
    ax.scatter(xx.ravel(),yy.ravel(),s=1.8,color=geometry.INK if final else '#566C7B',
               alpha=.58 if final else alpha*.9,edgecolors='none',zorder=z+2)
    x=stage['x'];x0=max(x-1.5,stage['lo']);x1=min(x+1.5,stage['hi'])
    fill='#D97A58' if stage['region']=='trivial' else '#2386A8'
    dot=geometry.LEFT_WALL if stage['region']=='trivial' else geometry.RIGHT_WALL
    ax.add_patch(Rectangle((x0+dx,CENTER_Y-1.5+dy),x1-x0,3,facecolor=fill,
                           edgecolor='none',alpha=.52*alpha,zorder=z+1))
    ax.add_patch(Rectangle((x0+dx,CENTER_Y-1.5+dy),x1-x0,3,facecolor='none',
                           edgecolor='black',lw=1.5,alpha=alpha,zorder=z+4.5))
    ax.plot(x+dx,CENTER_Y+dy,'o',color=dot,mec='white',mew=.4,ms=4.5,alpha=alpha,zorder=z+5)
    for xwall in (LEFT,RIGHT):
        ax.plot([xwall+dx]*2,[BOTTOM+dy,TOP+dy],color=wall,lw=1.5,ls='--',alpha=alpha,zorder=z+4)
    ax.add_patch(Rectangle((dx,BOTTOM+dy),WIDTH,TOP-BOTTOM,facecolor='none',edgecolor=edge,lw=1.1,alpha=alpha,zorder=z+4))
    seam='black' if final else edge
    for y in (BOTTOM+dy,TOP+dy):
        ax.plot([WIDTH/2+dx-.45,WIDTH/2+dx+.45],[y-.35,y+.35],color=seam,lw=2,alpha=alpha,solid_capstyle='butt',zorder=z+6)
    for xs in (dx,WIDTH+dx):
        for offset in (-.4,.4):
            ax.plot([xs-.35,xs+.35],[15+dy+offset-.5,15+dy+offset+.5],color=seam,lw=2,alpha=alpha,solid_capstyle='butt',zorder=z+6)
    return {'support_center_local':[stage['x'],CENTER_Y], 'retained_support_local':[x0,CENTER_Y-1.5,x1-x0,3], 'retained_cells':int((x1-x0)*3)}

def main():
    protected={str(p.relative_to(MANUSCRIPT)):sha(p) for p in [MANUSCRIPT/'manuscript.tex',MANUSCRIPT/'manuscript.pdf',MANUSCRIPT/'figures/new_figure/manifest.json']}
    typography.ROOT=OUT
    original_width=typography.inclusion_width
    typography.inclusion_width=lambda stem:7.05 if stem==STEM else original_width(stem)
    def role(fig,artist,stem):
        prominent=('\\alpha' in artist.get_text() or '\\mathcal' in artist.get_text() or artist.get_text()==r'cycle $t$')
        return 'schematic',19 if prominent else 17
    typography.text_role=role
    geometry._configure_style()
    fig,ax=plt.subplots(figsize=(7.05,7.5))
    fig.subplots_adjust(left=.025,right=.975,bottom=.035,top=.98)
    records=[{**stage,**draw_slice(ax,stage,i)} for i,stage in enumerate(STAGES)]
    last=STAGES[-1];dx,dy=last['dx'],last['dy']
    for x,label in ((3,r'$\alpha_2$'),(12,r'$\alpha_1$'),(21,r'$\alpha_2$')):
        ax.text(x+dx,26.5+dy,label,ha='center',va='center',zorder=90)
    ax.text(last['x']+dx+.75,CENTER_Y+dy+2.35,
            r'$\hat{\mathcal{N}}_{\boldsymbol{r},\nu,\sigma}(\alpha_{\boldsymbol{r}})$',
            ha='center',va='bottom',zorder=90)
    geometry._arrow(ax,(WIDTH+dx+1.4,BOTTOM+dy),(WIDTH+dx+1.4,TOP+dy),arrowstyle='<->',lw=1.2)
    ax.text(WIDTH+dx+2.35,(BOTTOM+TOP)/2+dy,r'$N_y$ (circumference)',rotation=90,ha='left',va='center')
    geometry._arrow(ax,(0,.3),(WIDTH,.3),arrowstyle='<->',lw=1.2)
    ax.text(WIDTH/2,-.3,r'$N_x$',ha='center',va='top')
    start=np.array([-1.8,4.5]);end=np.array([-.4,18.5])
    geometry._arrow(ax,start,end,lw=1.2,mutation_scale=12)
    angle=np.degrees(np.arctan2(*(end-start)[::-1]))
    middle=(start+end)/2+np.array([-1.15,.1])
    ax.text(*middle,r'cycle $t$',rotation=angle,ha='center',va='center',rotation_mode='anchor')
    ax.set(xlim=(-4.2,38),ylim=(-3.2,43.8),aspect='equal');ax.axis('off')
    typography.prepare_figure(fig,STEM)
    typography.record_typography(fig,STEM)
    for ext in ('pdf','png'):fig.savefig(OUT/f'{STEM}.{ext}',dpi=300)
    plt.close(fig)
    review=typography.verify_typography(STEM)
    assert all(sha(MANUSCRIPT/p)==value for p,value in protected.items())
    (OUT/'validation.json').write_text(json.dumps({'stages_oldest_to_latest':records,'same_local_y_center':CENTER_Y,
      'plane_aspect_ratio':WIDTH/(TOP-BOTTOM),'figure_inches':[7.05,7.5],
      'time_direction':'up and right','support_rule':'3x3 cells before clipping; discard support across the wall',
      'typography':review,'manuscript_unchanged':protected,
      'outputs':{f'{STEM}.{ext}':sha(OUT/f'{STEM}.{ext}') for ext in ('pdf','png')}},indent=2)+'\n')
    print(OUT/f'{STEM}.pdf')
if __name__=='__main__':main()
