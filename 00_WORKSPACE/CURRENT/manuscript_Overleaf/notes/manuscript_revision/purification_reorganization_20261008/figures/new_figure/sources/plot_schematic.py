#!/usr/bin/env python3
"""Build a stacked single-column schematic and a standalone circuit copy."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path as DrawingPath
from manuscript_typography import configure_style as manuscript_style, prepare_figure, record_typography


HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE.parent

INK = "#20252B"
MID_GRAY = "#707780"
LIGHT_GRAY = "#ECECF1"
PALE_GRAY = "#F7F7F8"
TOPOLOGICAL = "#9BD0EA"
LEFT_WALL = "#8B1E2D"
RIGHT_WALL = "#164A7B"
INTERFACE = RIGHT_WALL
ANCILLA = "#E5B45B"
CONNECTOR_STROKE = 0.9


def _configure_style() -> None:
    manuscript_style({'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.linewidth': 0.7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': None})


def _arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], **kwargs: object) -> None:
    properties: dict[str, object] = {
        "arrowstyle": "-|>",
        "color": INK,
        "lw": CONNECTOR_STROKE,
        "capstyle": "round",
        "joinstyle": "miter",
        "mutation_scale": 8,
        "shrinkA": 0,
        "shrinkB": 0,
    }
    properties.update(kwargs)
    axis.annotate("", xy=end, xytext=start, arrowprops=properties, annotation_clip=False)


def _box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "white",
    edgecolor: str = INK,
    linewidth: float = 0.9,
    fontsize: float = 7.5,
    rounding: float = 0.025,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=3,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        zorder=4,
    )
    return patch


def _draw_geometry(axis: plt.Axes) -> None:
    """One periodic cell; shaded OW supports are clipped to their center's region."""
    width, height = 24, 28
    bottom = 2  # Remove two bottom rows without moving the supports.
    left, right = 6, 18
    axis.add_patch(Rectangle((0, bottom), width, height-bottom, facecolor=LIGHT_GRAY,
                             edgecolor="none", zorder=0))
    axis.add_patch(Rectangle((left, bottom), right-left, height-bottom,
                             facecolor=TOPOLOGICAL, edgecolor="none", zorder=0))
    for x in (i + 0.5 for i in range(width)):
        for y in (i + 0.5 for i in range(bottom, height)):
            axis.add_patch(Circle((x, y), 0.055, facecolor=INK,
                                  edgecolor="none", alpha=0.58, zorder=2))

    # The centers lie on unit cells, and the walls lie between columns.
    # Each nominal w=1 support contains exactly 3 x 3 cells before wall clipping.
    operations = [
        (2.5, 22.5, 0, left, "#D97A58", LEFT_WALL),
        (6.5, 19.5, left, right, "#2386A8", RIGHT_WALL),
        (12.5, 14.5, left, right, "#2386A8", RIGHT_WALL),
        (17.5, 9.5, left, right, "#2386A8", RIGHT_WALL),
        (18.5, 4.5, right, width, "#D97A58", LEFT_WALL),
    ]
    for x, y, lo, hi, fill, dot in operations:
        x0, x1 = max(x-1.5, lo), min(x+1.5, hi)
        axis.add_patch(Rectangle((x0, y-1.5), x1-x0, 3,
                                 facecolor=fill, alpha=0.52, edgecolor="none", zorder=1))
        axis.add_patch(Rectangle((x0, y-1.5), x1-x0, 3,
                                 facecolor="none", edgecolor="black", linewidth=1.5, zorder=4.5))
        axis.plot(x, y, "o", color=dot, markeredgecolor="white",
                  markeredgewidth=0.4, markersize=4.5, zorder=5)
    # One generic label identifies all illustrated occupation measurements.
    x, y = operations[2][:2]
    axis.text(x+.4, y+2.25,
              r"$\hat{\mathcal{N}}_{\boldsymbol{r},\nu,\sigma}(\alpha_{\boldsymbol{r}})$",
              ha="center", va="bottom", zorder=6)
    for x in (left, right):
        axis.plot([x, x], [height, bottom], color=INTERFACE, linewidth=1.5, linestyle="--", zorder=4)
    axis.add_patch(Rectangle((0, bottom), width, height-bottom, facecolor="none",
                             edgecolor=MID_GRAY, linewidth=1.1, zorder=4))
    # Matching oblique strokes identify opposite periodic boundaries.
    for y in (bottom, height):
        axis.plot([width/2-0.45, width/2+0.45], [y-0.35, y+0.35],
                  color="black", linewidth=2.0, solid_capstyle="butt", clip_on=False, zorder=7)
    for x in (0, width):
        for offset in (-0.40, 0.40):
            axis.plot([x-0.35, x+0.35],
                      [(bottom+height)/2+offset-0.50, (bottom+height)/2+offset+0.50],
                      color="black", linewidth=2.0, solid_capstyle="butt", clip_on=False, zorder=7)
    for x, text in ((3, r"$\alpha_2$"), (12, r"$\alpha_1$"), (21, r"$\alpha_2$")):
        axis.text(x, height+0.65, text, ha="center", va="bottom")
    # Dimension arrows sit outside the periodic cell, clear of its seam marks.
    _arrow(axis, (width+1.4, bottom), (width+1.4, height),
           arrowstyle='<->', lw=.7)
    axis.text(width+2.35, (bottom+height)/2, r'$N_y$ (circumference)',
              rotation=90, ha='left', va='center', clip_on=False)
    _arrow(axis, (0, bottom-1.7), (width, bottom-1.7),
           arrowstyle='<->', lw=.7)
    axis.text(width/2, bottom-2.3, r'$N_x$', ha='center', va='top', clip_on=False)
    axis.set_xlim(-0.8, width+0.8)
    axis.set_ylim(bottom-0.8, height+2.0)
    axis.set_aspect("equal")
    axis.axis("off")


# Sixteen columns (4 exterior + 8 slab + 4 exterior) and eighteen rows.
SLICE_WIDTH=16; SLICE_BOTTOM=2; SLICE_TOP=20; SLICE_LEFT=4; SLICE_RIGHT=12
SLICE_CENTER_Y=4.5
SLICE_STAGES=[
    dict(dx=0,dy=0,opacity=.72,x=SLICE_LEFT-1.5,lo=0,hi=SLICE_LEFT,region='trivial',kind='full trivial support'),
    dict(dx=2.8,dy=4.4,opacity=.80,x=SLICE_LEFT-.5,lo=0,hi=SLICE_LEFT,region='trivial',kind='trivial interface support'),
    dict(dx=5.6,dy=8.8,opacity=.88,x=SLICE_LEFT+.5,lo=SLICE_LEFT,hi=SLICE_RIGHT,region='topological',kind='topological interface support'),
    dict(dx=8.4,dy=13.2,opacity=1,x=SLICE_LEFT+1.5,lo=SLICE_LEFT,hi=SLICE_RIGHT,region='topological',kind='full topological support'),
]

SLICE_XLIM=(-3.1,29.1)
SLICE_YLIM=(-1.9,36.3)

def _draw_time_slice(ax,stage,index,scale=1):
    dx,dy,alpha=stage['dx'],stage['dy'],stage['opacity']; z=20*index
    final=index==len(SLICE_STAGES)-1
    bg=LIGHT_GRAY if final else '#D5DDE3'
    slab=TOPOLOGICAL if final else '#93B7C8'
    edge=MID_GRAY if final else '#627783'
    wall=INTERFACE if final else '#617F93'
    ax.add_patch(Rectangle((dx,SLICE_BOTTOM+dy),SLICE_WIDTH,SLICE_TOP-SLICE_BOTTOM,facecolor=bg,edgecolor='none',alpha=alpha,zorder=z))
    ax.add_patch(Rectangle((SLICE_LEFT+dx,SLICE_BOTTOM+dy),SLICE_RIGHT-SLICE_LEFT,SLICE_TOP-SLICE_BOTTOM,facecolor=slab,edgecolor='none',alpha=alpha,zorder=z+.1))
    xx,yy=np.meshgrid(np.arange(SLICE_WIDTH)+.5+dx,np.arange(SLICE_BOTTOM,SLICE_TOP)+.5+dy)
    ax.scatter(xx.ravel(),yy.ravel(),s=1.8*scale**2,color=INK if final else '#566C7B',
               alpha=.58 if final else alpha*.9,edgecolors='none',zorder=z+2)
    x=stage['x'];x0=max(x-1.5,stage['lo']);x1=min(x+1.5,stage['hi'])
    fill='#D97A58' if stage['region']=='trivial' else '#2386A8'
    dot=LEFT_WALL if stage['region']=='trivial' else RIGHT_WALL
    ax.add_patch(Rectangle((x0+dx,SLICE_CENTER_Y-1.5+dy),x1-x0,3,facecolor=fill,
                           edgecolor='none',alpha=.52*alpha,zorder=z+1))
    ax.add_patch(Rectangle((x0+dx,SLICE_CENTER_Y-1.5+dy),x1-x0,3,facecolor='none',
                           edgecolor='black',lw=1.5*scale,alpha=alpha,zorder=z+4.5))
    ax.plot(x+dx,SLICE_CENTER_Y+dy,'o',color=dot,mec='white',mew=.4*scale,ms=4.5*scale,alpha=alpha,zorder=z+5)
    for xwall in (SLICE_LEFT,SLICE_RIGHT):
        ax.plot([xwall+dx]*2,[SLICE_BOTTOM+dy,SLICE_TOP+dy],color=wall,lw=1.5*scale,ls='--',alpha=alpha,zorder=z+4)
    ax.add_patch(Rectangle((dx,SLICE_BOTTOM+dy),SLICE_WIDTH,SLICE_TOP-SLICE_BOTTOM,facecolor='none',edgecolor=edge,lw=1.1*scale,alpha=alpha,zorder=z+4))
    seam='black' if final else edge
    for y in (SLICE_BOTTOM+dy,SLICE_TOP+dy):
        ax.plot([SLICE_WIDTH/2+dx-.45,SLICE_WIDTH/2+dx+.45],[y-.35,y+.35],color=seam,lw=2*scale,alpha=alpha,solid_capstyle='butt',zorder=z+6)
    for xs in (dx,SLICE_WIDTH+dx):
        for offset in (-.4,.4):
            ax.plot([xs-.35,xs+.35],[(SLICE_BOTTOM+SLICE_TOP)/2+dy+offset-.5,(SLICE_BOTTOM+SLICE_TOP)/2+dy+offset+.5],color=seam,lw=2*scale,alpha=alpha,solid_capstyle='butt',zorder=z+6)
    return {'support_center_local':[stage['x'],SLICE_CENTER_Y], 'retained_support_local':[x0,SLICE_CENTER_Y-1.5,x1-x0,3], 'retained_cells':int((x1-x0)*3)}

def _draw_four_slice_geometry(ax):
    """Four illustrative OW updates on a 16-by-18 lattice with an eight-column slab."""
    ax.set(xlim=SLICE_XLIM,ylim=SLICE_YLIM,aspect='equal')
    ax.axis('off')
    ax.figure.canvas.draw()
    # The approved standalone figure used this physical spacing per lattice cell.
    original_spacing=7.5*(.98-.035)/52.2
    spacing=ax.get_window_extent().width/ax.figure.dpi/np.diff(SLICE_XLIM)[0]
    scale=spacing/original_spacing
    records=[{**stage,**_draw_time_slice(ax,stage,i,scale)} for i,stage in enumerate(SLICE_STAGES)]
    last=SLICE_STAGES[-1];dx,dy=last['dx'],last['dy']
    for x,label in ((SLICE_LEFT/2,r'$\alpha_2$'),((SLICE_LEFT+SLICE_RIGHT)/2,r'$\alpha_1$'),((SLICE_RIGHT+SLICE_WIDTH)/2,r'$\alpha_2$')):
        ax.text(x+dx,SLICE_TOP+.75+dy,label,ha='center',va='bottom',zorder=90)
    operator_label=ax.text(last['x']+dx+1.5+.2,SLICE_CENTER_Y+dy,
            r'$\hat{\mathcal{N}}_{\boldsymbol{r},\nu,\sigma}$',
            ha='left',va='center',zorder=90)
    _arrow(ax,(SLICE_WIDTH+dx+1.4,SLICE_BOTTOM+dy),(SLICE_WIDTH+dx+1.4,SLICE_TOP+dy),arrowstyle='<->',lw=1.2*scale,mutation_scale=8*scale)
    ax.text(SLICE_WIDTH+dx+2.35,(SLICE_BOTTOM+SLICE_TOP)/2+dy,r'$N_y$ (circumference)',rotation=90,ha='left',va='center')
    _arrow(ax,(0,.3),(SLICE_WIDTH,.3),arrowstyle='<->',lw=1.2*scale,mutation_scale=8*scale)
    ax.text(SLICE_WIDTH/2,-.3,r'$N_x$',ha='center',va='top')
    # Parallel to the displacement between corresponding slice corners.
    corner_step=np.array([SLICE_STAGES[1]['dx']-SLICE_STAGES[0]['dx'],SLICE_STAGES[1]['dy']-SLICE_STAGES[0]['dy']])
    # Endpoints are level with the first and fourth upper-left corners.
    # The common horizontal offset preserves the existing arrow line.
    corner_offset=np.array([-2.1,0.0])
    start=np.array([SLICE_STAGES[0]['dx'],SLICE_TOP+SLICE_STAGES[0]['dy']])+corner_offset
    end=np.array([last['dx'],SLICE_TOP+last['dy']])+corner_offset
    _arrow(ax,start,end,lw=1.2*scale,mutation_scale=12*scale)
    angle=np.degrees(np.arctan2(*(end-start)[::-1]))
    normal=np.array([-corner_step[1],corner_step[0]])/np.linalg.norm(corner_step)
    middle=(start+end)/2+1.2*normal
    ax.text(*middle,'time',rotation=angle,ha='center',va='center',rotation_mode='anchor')
    return operator_label, dict(stages_oldest_to_latest=records,
        lattice_columns_rows=[SLICE_WIDTH,SLICE_TOP-SLICE_BOTTOM],
        slab_columns=SLICE_RIGHT-SLICE_LEFT,
        exterior_columns=[SLICE_LEFT,SLICE_WIDTH-SLICE_RIGHT],
        local_support_y=SLICE_CENTER_Y,time_arrow_endpoints=[start.tolist(),end.tolist()],
        time_arrow_angle_degrees=float(angle),graphic_scale=float(scale))


def _center_operator_ink(fig, operator_label):
    """Align the visible TeX glyphs, including scripts, with the mode center."""
    probe=plt.figure(figsize=(3,1),dpi=300)
    probe.patch.set_alpha(0)
    probe.text(.5,.5,operator_label.get_text(),ha='center',va='center',
               fontproperties=operator_label.get_fontproperties(),usetex=True)
    probe.canvas.draw()
    pixels=np.asarray(probe.canvas.buffer_rgba())
    rows=np.nonzero(pixels[:,:,3]>127)[0]
    ink_offset=(pixels.shape[0]-1-(rows.min()+rows.max())/2)-pixels.shape[0]/2
    plt.close(probe)
    fig.canvas.draw()
    ax=operator_label.axes
    anchor=ax.transData.transform(operator_label.get_position())
    anchor[1]-=ink_offset*fig.dpi/300
    operator_label.set_position(ax.transData.inverted().transform(anchor))


def _draw_feedback_tile(axis: plt.Axes, *, spine: float = 1.30) -> None:
    """Shared centerline, equal measurement boxes, and a plain bypass junction."""
    # All positions derive from a small set of common alignment anchors.
    box_width, box_height = 1.42, 0.56
    gap, stroke, padding = 0.25, CONNECTOR_STROKE, 0.012
    decision_gap = gap + 0.10  # Equal extensions above and below the decision.
    final_bottom = 0.14
    final_top = final_bottom + box_height
    junction_y = final_top + gap
    swap_bottom = junction_y + gap
    swap_width, swap_height = 0.96, 0.46
    swap_top = swap_bottom + swap_height
    swap_y = (swap_bottom + swap_top)/2
    diamond_bottom = swap_top + decision_gap
    diamond_height, diamond_width = 0.74, 1.00
    diamond_top = diamond_bottom + diamond_height
    decision_y = (diamond_bottom + diamond_top)/2
    measure_bottom = diamond_top + decision_gap
    box_left = spine - box_width/2
    bypass_x = spine-1.10
    ancilla_x, ancilla_radius = 2.78, 0.13
    label_gap = 0.11

    _box(axis, (box_left, measure_bottom), box_width, box_height,
         "measure\n" + r"$\hat{\mathcal{N}}_{\boldsymbol{r},\nu,\sigma}(\alpha_{\boldsymbol{r}})$",
         facecolor=PALE_GRAY, linewidth=stroke)
    _arrow(axis, (spine, measure_bottom-padding), (spine, diamond_top))
    axis.add_patch(Polygon([(spine, diamond_top), (spine+diamond_width/2, decision_y),
                           (spine, diamond_bottom), (spine-diamond_width/2, decision_y)],
                          closed=True, facecolor="white", edgecolor=INK,
                          linewidth=stroke, zorder=3))
    axis.text(spine, decision_y, r"$\mathrm{m}=s_\sigma$?", ha="center", va="center", zorder=4)

    # Matching outcomes bypass the reset. No arrowhead or dot crowds the T-junction.
    # Use the same patch stroke as the arrowed connectors, with no arrowhead.
    bypass_path = DrawingPath([(spine-diamond_width/2, decision_y),
                               (bypass_x, decision_y), (bypass_x, junction_y),
                               (spine, junction_y)])
    axis.add_patch(FancyArrowPatch(path=bypass_path, arrowstyle='-',
                                  color=INK, linewidth=stroke,
                                  capstyle='round', joinstyle='miter'))
    axis.text((spine-diamond_width/2+bypass_x)/2, decision_y+label_gap,
              "yes", ha="center", va="bottom")
    _arrow(axis, (spine, diamond_bottom), (spine, swap_top+padding))
    axis.text(spine+label_gap, (diamond_bottom+swap_top)/2,
              "no", ha="left", va="center")
    _box(axis, (spine-swap_width/2, swap_bottom), swap_width, swap_height,
         "fSWAP", linewidth=stroke)

    # The ancilla, its labels, and the incoming wire form one aligned group.
    axis.add_patch(Circle((ancilla_x, swap_y), ancilla_radius, facecolor=ANCILLA,
                          edgecolor=INK, linewidth=stroke, zorder=3))
    axis.text(ancilla_x, swap_y, r"$s_\sigma$", ha="center", va="center", zorder=4)
    _arrow(axis, (ancilla_x-ancilla_radius, swap_y),
           (spine+swap_width/2+padding, swap_y))
    axis.text(ancilla_x, swap_y+ancilla_radius+label_gap, "fresh\nancilla",
              ha="center", va="bottom", linespacing=1.0)
    axis.text(ancilla_x+.13, swap_y-ancilla_radius-label_gap,
              "$s_-=1$ (fill)\n$s_+=0$ (empty)", ha="center", va="top",
              multialignment="left", linespacing=1.35)

    # A single arrow passes through the merge and enters the next measurement.
    _arrow(axis, (spine, swap_bottom-padding), (spine, final_top+padding))
    _box(axis, (box_left, final_bottom), box_width, box_height,
         "measure next\nOW mode", facecolor=PALE_GRAY, linewidth=stroke)
    final_y = final_bottom + box_height/2
    continuation_tip = 2.91
    _arrow(axis, (spine+box_width/2+padding, final_y), (continuation_tip, final_y))
    axis.text(continuation_tip+0.045, final_y, r"$\cdots$", ha="left", va="center")
    axis.set_xlim(0.05, 3.39)
    axis.set_ylim(0.01, 3.81)
    axis.set_aspect("equal")
    axis.axis("off")


def _save(figure, stem, *, operator_label=None):
    prepare_figure(figure, stem)
    if operator_label is not None:
        _center_operator_ink(figure, operator_label)
    record_typography(figure, stem)
    pdf = FIGURE_DIR / (stem + ".pdf")
    figure.savefig(pdf)
    plt.close(figure)
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile",
                    str(pdf), str(pdf.with_suffix(""))], check=True)


def build(only=None) -> None:
    _configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if only in (None, "combined", "geometry"):
        # Tight geometry bounds retain the approved four-slice composition.
        # The flowchart keeps exactly its previous physical size and position.
        geometry_width = 3.215
        geometry_height = geometry_width * np.diff(SLICE_YLIM)[0]/np.diff(SLICE_XLIM)[0]
        circuit_extra_height = 3.145 * (.20/3.60)
        geometry_bottom = 3.439 + circuit_extra_height + .18
        height = geometry_bottom + geometry_height + .16
        figure = plt.figure(figsize=(3.375, height))
        operator_label, _ = _draw_four_slice_geometry(figure.add_axes([
            .08/3.375, geometry_bottom/height, geometry_width/3.375, geometry_height/height]))
        _draw_feedback_tile(figure.add_axes([0.020, 0.111/height, 0.97, (3.145+circuit_extra_height)/height]),
                            spine=(0.05+3.39)/2)
        figure.text(0.025, 1-0.0888/height, "(a)", va="top")
        figure.text(0.025, (3.4262+circuit_extra_height)/height, "(b)", va="top")
        _save(figure, "Figure_01_schematic", operator_label=operator_label)
    if only in (None, "circuit"):
        figure = plt.figure(figsize=(3.375, 3.65 * (3.80/3.60)))
        _draw_feedback_tile(figure.add_axes([0.015, 0.015, 0.97, 0.97]))
        _save(figure, "Figure_02_adaptive_circuit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("combined", "geometry", "circuit"))
    build(parser.parse_args().only)
