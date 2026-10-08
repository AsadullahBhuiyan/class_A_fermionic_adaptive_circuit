"""Exploratory vector torus schematic; no manuscript or scientific data edits.

x is the poloidal (short) cycle; y is the toroidal (long) cycle.
The blue x interval is 6 <= x <= 18 on a 24 x 64 periodic schematic lattice.
"""
from pathlib import Path
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, Polygon

HERE = Path(__file__).resolve().parent
MANUSCRIPT = HERE.parents[1]
sys.path.insert(0, str(MANUSCRIPT/'figures/new_figure/sources'))
from manuscript_typography import configure_style
configure_style()
R, r, NX, NY = 2.6, .88, 24, 64
az, el = np.deg2rad(-55), np.deg2rad(33)
u = np.array([-np.sin(az), np.cos(az), 0])
v = np.array([-np.sin(el)*np.cos(az), -np.sin(el)*np.sin(az), np.cos(el)])
w = np.cross(u, v)

def torus(x, y, lift=0):
    theta = -np.pi/3 + 2*np.pi*np.asarray(x)/NX
    phi = 2*np.pi*np.asarray(y)/NY
    rr = r+lift
    return np.stack([(R+rr*np.cos(theta))*np.cos(phi),
                     (R+rr*np.cos(theta))*np.sin(phi), rr*np.sin(theta)], axis=-1)

def project(xyz):
    return np.asarray(xyz) @ np.stack([u,v,w],axis=1)

# Coarse cell supports, including hard clipping at the domain walls.
supports = [(3.5,53.5,0,6,'#D97A58','#8B1E2D'),
            (6.5,43.5,6,18,'#2386A8','#164A7B'),
            (12.5,12.5,6,18,'#2386A8','#164A7B'),
            (5.5,61.5,0,6,'#D97A58','#8B1E2D')]
shapes=[]
def polygon(points,color,depth_offset=0):
    pr=project(points)
    shapes.append((float(pr[:,2].mean())+depth_offset,pr[:,:2],color))

def ribbon(x0,x1,y0,y1,color,lift=.003):
    polygon(torus(np.array([x0,x1,x1,x0]),np.array([y0,y0,y1,y1]),lift),color)

light=np.array([-.2,-.5,1]);light/=np.linalg.norm(light)
for x in np.arange(0,NX,.25):
    for y in np.arange(0,NY,.25):
        xc,yc=x+.125,y+.125
        base='#9BD0EA' if 6<=xc<18 else '#ECECF1'
        for sx,sy,lo,hi,color,dot in supports:
            if max(sx-1.5,lo)<=xc<min(sx+1.5,hi) and sy-1.5<=yc<sy+1.5:
                base=color
        th=-np.pi/3+2*np.pi*xc/NX; ph=2*np.pi*yc/NY
        normal=np.array([np.cos(th)*np.cos(ph),np.cos(th)*np.sin(ph),np.sin(th)])
        shade=.80+.20*max(0,float(normal@light))
        ribbon(x,x+.25,y,y+.25,np.array(to_rgb(base))*shade,lift=0)
# Dotted lattice; tangent-space circular patches participate in depth sorting.
angles=np.linspace(0,2*np.pi,32)
for x in np.arange(.5,NX):
    for y in np.arange(.5,NY):
        polygon(torus(x+.040*np.cos(angles),y+.065*np.sin(angles),.012),'#536774')
# Domain walls are two closed long-cycle loops.
for x in (6,18):
    for y in np.arange(0,NY,.20):
        if y%1.20<.76:
            ribbon(x-.052,x+.052,y,y+.20,'#164A7B',.018)
# Closed black support outlines are drawn on top of the dashed walls.
for sx,sy,lo,hi,color,dot in supports:
    x0,x1=max(sx-1.5,lo),min(sx+1.5,hi);y0,y1=sy-1.5,sy+1.5
    for yy in np.arange(y0,y1,.10):
        for xx in (x0,x1):ribbon(xx-.055,xx+.055,yy,min(yy+.10,y1),'#171B1E',.030)
    for xx in np.arange(x0,x1,.10):
        for yy in (y0,y1):ribbon(xx,min(xx+.10,x1),yy-.08,yy+.08,'#171B1E',.030)

fig,ax=plt.subplots(figsize=(7.05,4.75))
shapes.sort(key=lambda item:item[0])
ax.add_collection(PolyCollection([s[1] for s in shapes],facecolors=[s[2] for s in shapes],
                                 edgecolors='face',linewidths=.10,antialiaseds=False))
ax.set_aspect('equal');ax.set_xlim(-4.45,4.45);ax.set_ylim(-3.0,3.0);ax.axis('off')
fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99)

# Project each visible center and its tangent-plane ring together. Drawing the
# filled marker with one white stroke prevents depth-sorted faces from breaking
# the ring or shifting it relative to the Wannier center.
ring_angle = np.linspace(0, 2*np.pi, 128)
for sx,sy,lo,hi,color,dot in supports:
    center = project(torus(sx,sy))[:2]
    eps = 1e-4
    tangent_x = project((torus(sx+eps,sy)-torus(sx-eps,sy))/(2*eps))[:2]
    tangent_y = project((torus(sx,sy+eps)-torus(sx,sy-eps))/(2*eps))[:2]
    ring = (center + .25*np.cos(ring_angle)[:,None]*tangent_x
            + .3125*np.sin(ring_angle)[:,None]*tangent_y)
    ax.add_patch(Polygon(ring, closed=True, facecolor=dot, edgecolor='white',
                         linewidth=.75, antialiased=True, zorder=2))

# Exterior arrows illustrate a complete cycle without obscuring the surface.
phi=np.linspace(az-1.35,az+1.35,180)
cycle=np.stack([4.0*np.cos(phi),4.0*np.sin(phi),np.full_like(phi,-.7)],axis=-1)
xy=project(cycle)[:,:2]
ax.plot(xy[:,0],xy[:,1],color='#393E44',lw=1.2)
ax.add_patch(FancyArrowPatch(xy[-7],xy[-1],arrowstyle='-|>',mutation_scale=10,color='#393E44',lw=1.2))
ax.text(0,-2.90,r'$y$',ha='center',va='center',fontsize=11)
# A meridian follows the tube cross-section on the right. Its rear half is dashed.
th=np.linspace(0,2*np.pi,240)
ph=az+np.pi/2
short=project(np.stack([(R+1.03*np.cos(th))*np.cos(ph),
                        (R+1.03*np.cos(th))*np.sin(ph),1.03*np.sin(th)],axis=-1))[:,:2]
ax.plot(short[:120,0],short[:120,1],color='#393E44',lw=1.1)
ax.plot(short[119:,0],short[119:,1],color='#393E44',lw=1.1,ls=(0,(3,3)))
ax.add_patch(FancyArrowPatch(short[25],short[35],arrowstyle='-|>',mutation_scale=10,color='#393E44',lw=1.1))
ax.text(3.47,.66,r'$x$',ha='center',va='bottom',fontsize=11)
# Phase labels with small, unobtrusive leaders.
for label,point,target in [(r'$\alpha_1$',(-2.6,2.40),(12,18)),
                           (r'$\alpha_2$',(-3.70,-.92),(3,43))]:
    dest=project(torus(*target))[:2]
    ax.annotate(label,xy=dest,xytext=point,fontsize=11,ha='center',
                arrowprops=dict(arrowstyle='-',lw=.65,color='#444444'))
operator_center = project(torus(supports[2][0], supports[2][1]))[:2]
# End the leader inside the upper part of the support, clear of its center dot.
operator_target = project(torus(supports[2][0]-.75, supports[2][1]))[:2]
ax.annotate(r'$\hat{\mathcal N}_{\boldsymbol r,\nu,\sigma}(\alpha_{\boldsymbol r})$',
            xy=operator_target, xytext=(2.30,2.45),
            ha='center',va='center',fontsize=11,
            arrowprops=dict(arrowstyle='-',lw=.65,color='black',shrinkA=5,shrinkB=0))
# No manuscript figure is replaced by this optional visual experiment.
for ext in ('pdf','png'):fig.savefig(HERE/f'torus_geometry.{ext}',dpi=300)
plt.close(fig)
(HERE/'description.json').write_text(json.dumps({'status':'exploratory alternative, not included in manuscript',
    'cycles':{'x':'short/poloidal; wraps the tube','y':'long/toroidal; goes around the central hole'},
    'blue_region':'6 <= x <= 18; half of tube circumference',
    'gray_region':'periodically connected complement',
    'interfaces':'x=6 and x=18, both wind around the long y cycle',
    'supports':'four 3x3 windows, clipped at the interface when required',
    'lattice':[NX,NY],'major_radius':R,'minor_radius':r,'figure_inches':[7.05,4.75],
    'scientific_data_changed':False},indent=2)+'\n')
print(HERE/'torus_geometry.pdf')
