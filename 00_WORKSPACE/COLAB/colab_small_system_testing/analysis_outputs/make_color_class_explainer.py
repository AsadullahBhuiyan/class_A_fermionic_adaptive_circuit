"""
Generate a 4-panel explainer figure for color-class parallel OW measurement sweeps.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import os

matplotlib.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

OUT = os.path.join(os.path.dirname(__file__))
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(
    2, 2, figure=fig,
    hspace=0.52, wspace=0.38,
    left=0.05, right=0.97, top=0.95, bottom=0.05,
)

# ── helpers ─────────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, txt, fc, ec="black", fs=8.5, lw=1.4, ha="center"):
    r = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.05",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(r)
    ax.text(x + w / 2, y + h / 2, txt,
            ha=ha, va="center", fontsize=fs, linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, color="black", lw=1.5):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=lw, mutation_scale=12),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Panel (a):  stride-3 lattice tiling with dw_truncation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax = fig.add_subplot(gs[0, 0])

Nx, Ny = 12, 9
nshell = 1
stride = 2 * nshell + 1          # = 3
DW_x = 5.5                       # domain-wall x position

# 9 class colors
tab = plt.cm.tab10(np.linspace(0, 0.9, 9))
cmap = {(r, c): tab[r * stride + c] for r in range(stride) for c in range(stride)}

HIGHLIGHT = (0, 0)               # class to show footprints for

# Draw footprints for the highlighted class
for Rx in range(Nx):
    for Ry in range(Ny):
        if (Rx % stride, Ry % stride) != HIGHLIGHT:
            continue
        # dw_truncation: footprint only on the side the site belongs to
        left_side = (Rx <= DW_x)
        fp_xmin = max(Rx - nshell, 0)
        fp_xmax = min(Rx + nshell, Nx - 1)
        if left_side:
            fp_xmax = min(fp_xmax, int(DW_x))   # truncate at DW
        else:
            fp_xmin = max(fp_xmin, int(DW_x) + 1)
        fp_ymin = Ry - nshell
        fp_ymax = Ry + nshell
        w = fp_xmax - fp_xmin + 1
        h = fp_ymax - fp_ymin + 1
        rect = Rectangle(
            (fp_xmin - 0.48, fp_ymin - 0.48), w - 0.04, h - 0.04,
            facecolor=(*cmap[HIGHLIGHT][:3], 0.18),
            edgecolor=(*cmap[HIGHLIGHT][:3], 0.7),
            linewidth=0.9, linestyle="--", zorder=1,
        )
        ax.add_patch(rect)

# Draw sites
for Rx in range(Nx):
    for Ry in range(Ny):
        cls = (Rx % stride, Ry % stride)
        c = cmap[cls]
        is_hi = cls == HIGHLIGHT
        ax.plot(
            Rx, Ry,
            "o" if is_hi else "s",
            color=c,
            markersize=9 if is_hi else 5,
            alpha=1.0 if is_hi else 0.45,
            zorder=3,
        )

# Domain wall
ax.axvline(DW_x, color="black", linewidth=2.2, zorder=2)
ax.text(DW_x + 0.15, Ny - 0.3, "DW", fontsize=9, va="top", fontweight="bold")
ax.text(DW_x - 0.15, Ny - 0.3, "DW", fontsize=9, va="top", ha="right", fontweight="bold")

# region labels
ax.text(DW_x / 2, -0.85, "topological", fontsize=8.5, ha="center", color="0.35")
ax.text((DW_x + Nx) / 2, -0.85, "trivial", fontsize=8.5, ha="center", color="0.35")

# stride annotation
ax.annotate(
    "", xy=(3, 0), xytext=(0, 0),
    arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1.2),
)
ax.text(1.5, 0.35, r"stride $= 3$", fontsize=8, ha="center", color="dimgray")

ax.set_xlim(-0.7, Nx - 0.3)
ax.set_ylim(-1.15, Ny - 0.2)
ax.set_aspect("equal")
ax.set_xlabel(r"$R_x$")
ax.set_ylabel(r"$R_y$")
ax.set_title(
    r"(a)  Stride-3 color-class tiling ($n_\mathrm{shell}=1$, "
    r"$\mathrm{stride}=2n_\mathrm{shell}+1=3$)",
    pad=6,
)

legend_handles = [
    mpatches.Patch(
        facecolor=(*cmap[HIGHLIGHT][:3], 0.3),
        edgecolor=cmap[HIGHLIGHT][:3],
        linestyle="--", linewidth=0.9,
        label=r"class $(0,0)$ footprints (dw\_trunc)",
    ),
] + [
    mpatches.Patch(color=cmap[(r, c)], alpha=0.6,
                   label=rf"class $({r},{c})$")
    for r in range(stride) for c in range(stride)
    if (r, c) != HIGHLIGHT
]
ax.legend(
    handles=legend_handles[:5], fontsize=7.5,
    loc="lower right", framealpha=0.85, ncol=1,
)

ax.text(
    0.02, 0.99,
    r"9 classes: $(R_x \mathrm{\ mod\ } 3,\, R_y \mathrm{\ mod\ } 3)$" + "\n"
    "Same-class footprints always disjoint\n"
    "dw_trunc keeps each footprint on one side",
    transform=ax.transAxes, fontsize=7.8, va="top",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85),
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Panel (b):  G matrix block structure — why two-pass is needed
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_aspect("equal")

k = 5      # schematic size of one support
nr = 10    # complement size
N = 2 * k + nr

# Build RGBA image
img = np.ones((N, N, 4))
img[:] = [0.93, 0.93, 0.93, 1.0]   # background

blue   = np.array([0.15, 0.50, 0.90, 0.85])
orange = np.array([0.95, 0.40, 0.10, 0.85])
lblue  = np.array([0.15, 0.50, 0.90, 0.35])
lorang = np.array([0.95, 0.40, 0.10, 0.35])
purple = np.array([0.55, 0.15, 0.80, 0.45])

# G_ss blocks
img[:k, :k] = blue
img[k:2*k, k:2*k] = orange

# G_sr blocks
img[:k, 2*k:] = lblue
img[2*k:, :k] = lblue
img[k:2*k, 2*k:] = lorang
img[2*k:, k:2*k] = lorang

# delta_rr region: C_A x C_A (includes S_B x S_B)
img[k:, k:] = purple
# redraw G_ss on top
img[:k, :k] = blue
img[k:2*k, k:2*k] = orange
# G_sr on top of purple in C region
img[:k, 2*k:] = lblue
img[2*k:, :k] = lblue
img[k:2*k, 2*k:] = lorang
img[2*k:, k:2*k] = lorang

ax2.imshow(img, origin="upper", aspect="equal", interpolation="nearest")

for pos in [k - 0.5, 2*k - 0.5]:
    ax2.axhline(pos, color="black", lw=1.6)
    ax2.axvline(pos, color="black", lw=1.6)

# Text labels inside blocks
def mtxt(ax, x, y, s, fs=9, **kw):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, **kw)

mtxt(ax2, k/2-0.5,       k/2-0.5,       r"$G_{ss}^{(A)}$",      fontweight="bold", color="white")
mtxt(ax2, 3*k/2-0.5,     3*k/2-0.5,     r"$G_{ss}^{(B)}$",      fontweight="bold", color="white")
mtxt(ax2, 2*k+nr/2-0.5,  k/2-0.5,       r"$G_{sr}^{(A)}$",      color="navy")
mtxt(ax2, k/2-0.5,       2*k+nr/2-0.5,  r"$G_{sr}^{(A)\dagger}$", color="navy")
mtxt(ax2, 2*k+nr/2-0.5,  3*k/2-0.5,     r"$G_{sr}^{(B)}$",      color="#7a2000")
mtxt(ax2, 3*k/2-0.5,     2*k+nr/2-0.5,  r"$G_{sr}^{(B)\dagger}$", color="#7a2000")
mtxt(ax2, 2*k+nr/2-0.5,  2*k+nr/2-0.5,
     r"$+\,\Delta_{rr}^{(A)}+\Delta_{rr}^{(B)}$",
     color="white", fs=8)

# Highlight: delta_rr(A) also writes into S_B x S_B
rect_sb = Rectangle((k - 0.5, k - 0.5), k, k,
                     facecolor="none", edgecolor="yellow",
                     linewidth=2.5, linestyle="-", zorder=5)
ax2.add_patch(rect_sb)
ax2.annotate(
    r"$\Delta_{rr}^{(A)}$ lands here too!",
    xy=(3*k/2 - 0.5, 3*k/2 - 0.5), xytext=(3*k + 2, 2*k - 2),
    fontsize=8, color="goldenrod", fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color="goldenrod", lw=1.4),
)

ax2.set_xticks([k/2-0.5, 3*k/2-0.5, 2*k+nr/2-0.5])
ax2.set_xticklabels([r"$S_A$", r"$S_B$", r"$C$"], fontsize=11)
ax2.set_yticks([k/2-0.5, 3*k/2-0.5, 2*k+nr/2-0.5])
ax2.set_yticklabels([r"$S_A$", r"$S_B$", r"$C$"], fontsize=11)
ax2.tick_params(length=0)

ax2.set_title(
    r"(b)  $G$ block structure — why two-pass write-back is needed",
    pad=6,
)
ax2.text(
    0.02, 0.02,
    r"Purple: $\Delta_{rr}^{(A)}$ range $= C_A \times C_A \supset S_B \times S_B$"
    "\nWrite $G_{ss}^{(B)}$ after $\Delta_{rr}$ to avoid clobbering",
    transform=ax2.transAxes, fontsize=7.8, va="bottom",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Panel (c):  CPU two-pass protocol
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 9)
ax3.axis("off")
ax3.set_title("(c)  CPU two-pass write-back protocol", pad=6)

# G snapshot
box(ax3, 1.0, 7.8, 3.8, 0.8, r"$G$ (frozen snapshot)", "#d4e6f1", fs=9.5)

# Per-site processing (parallel conceptually)
box(ax3, 0.3, 5.9, 1.7, 1.5,
    "site $A$\n4 channels\n"
    r"$\to(G_{ss}^{(A)},G_{sr}^{(A)},\Delta_{rr}^{(A)})$",
    "#d5f5e3", fs=7.5)
box(ax3, 2.3, 5.9, 1.7, 1.5,
    "site $B$\n4 channels\n"
    r"$\to(G_{ss}^{(B)},G_{sr}^{(B)},\Delta_{rr}^{(B)})$",
    "#d5f5e3", fs=7.5)
ax3.text(4.4, 6.65, r"$\cdots$", fontsize=13, va="center")

arrow(ax3, 1.7, 7.8, 1.15, 7.4)
arrow(ax3, 2.25, 7.8, 3.15, 7.4)

# Collect
box(ax3, 0.3, 4.5, 4.5, 1.0,
    r"collect all results: $\{(S_i,\,C_i,\,G_{ss}^{(i)},\,G_{sr}^{(i)},\,\Delta_{rr}^{(i)})\}_i$",
    "#fdebd0", fs=8)
arrow(ax3, 1.15, 5.9, 1.5, 5.5)
arrow(ax3, 3.15, 5.9, 3.2, 5.5)

# Pass 1
box(ax3, 0.3, 3.1, 4.5, 1.1,
    "Pass 1 — direct blocks (disjoint $S_i$: no conflict)\n"
    r"$G_{\rm new}[S_i, S_i]\leftarrow G_{ss}^{(i)}$,"
    r"$\quad G_{\rm new}[S_i, C_i]\leftarrow G_{sr}^{(i)}$",
    "#e8daef", fs=8)
arrow(ax3, 2.5, 4.5, 2.5, 4.2)

# Pass 2
box(ax3, 0.3, 1.8, 4.5, 1.0,
    r"Pass 2 — back-action: "
    r"$G_{\rm new}[C_i, C_i] += \Delta_{rr}^{(i)},\;\forall\,i$",
    "#fce4d6", fs=8)
arrow(ax3, 2.5, 3.1, 2.5, 2.8)

# Output
box(ax3, 0.3, 0.6, 4.5, 0.9,
    r"$G_{\rm new}\leftarrow\frac{1}{2}(G_{\rm new}+G_{\rm new}^\dagger)$",
    "#d6eaf8", fs=9)
arrow(ax3, 2.5, 1.8, 2.5, 1.5)

# Side note
ax3.text(
    5.5, 5.5,
    "Each site reads from the\nsame $G$ snapshot — no\nserial dependency within\nthe color class.",
    fontsize=8.5, ha="left", va="center", style="italic",
    bbox=dict(boxstyle="round", facecolor="#fffbe6", alpha=0.9),
)
ax3.annotate(
    "", xy=(4.75, 6.65), xytext=(5.4, 5.9),
    arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.2),
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Panel (d):  GPU deferred-G_ss protocol
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 9)
ax4.axis("off")
ax4.set_title("(d)  GPU write-back — deferred $G_{ss}$ to save memory", pad=6)

# G_new target
box(ax4, 1.0, 7.8, 3.8, 0.8, r"$G_{\rm new} = G.\mathrm{clone}()$", "#d4e6f1", fs=9.5)

# Per-site loop
box(ax4, 0.3, 5.5, 4.5, 2.0,
    "for each site $i$ in color class:\n"
    r"  read $G_{ss}, G_{sr}$ from snapshot $G$"
    "\n  run 4 channels → compute updates\n"
    r"  $\Delta_{rr}$: index_put_(..., accumulate=True)"
    r"$\to G_{\rm new}[C_i,C_i]$"
    "\n"
    r"  write $G_{sr}^{(i)}\to G_{\rm new}[S_i,C_i]$ (immediately)"
    "\n"
    r"  store $(S_i, G_{ss}^{(i)})$ in stored_ss",
    "#d5f5e3", fs=7.8)
arrow(ax4, 2.0, 7.8, 2.0, 7.5)

# Memory note
ax4.text(
    5.3, 6.5,
    r"Memory:"
    "\n"
    r"$G_{sr}$: $89\times b\times k\times N\approx10\,\mathrm{GB}$"
    "\n"
    r"$G_{ss}$: $89\times b\times k^2\approx120\,\mathrm{MB}$"
    "\n"
    r"$\Rightarrow$ buffer only $G_{ss}$!",
    fontsize=8.5, ha="left", va="center",
    bbox=dict(boxstyle="round", facecolor="#fff3cd", alpha=0.9),
)

# Deferred write
box(ax4, 0.3, 3.9, 4.5, 1.3,
    "after all sites:\n"
    r"for $(S_i, G_{ss}^{(i)})$ in stored_ss:"
    "\n"
    r"  $G_{\rm new}[S_i,S_i]\leftarrow G_{ss}^{(i)}$"
    "\n"
    r"  (overrides cross-$\Delta_{rr}$: error $O(G[S_A,S_B]^2)\approx 0$)",
    "#fce4d6", fs=7.8)
arrow(ax4, 2.5, 5.5, 2.5, 5.2)

# Symmetrize
box(ax4, 0.3, 2.7, 4.5, 0.9,
    r"$G_{\rm new}\leftarrow\frac{1}{2}(G_{\rm new}+G_{\rm new}^\dagger)$",
    "#d6eaf8", fs=9)
arrow(ax4, 2.5, 3.9, 2.5, 3.6)

# Why error is small
box(ax4, 0.3, 0.8, 9.2, 1.6,
    r"Key approximation:  $G[S_A, S_B]\approx 0$ for stride-separated sites  "
    r"$\;\Rightarrow\;$  cross-$\Delta_{rr}$ error $\sim O(G[S_A,S_B]^2)\approx 0$."
    "\n"
    "With dw_truncation=True: sites on opposite sides of DW have "
    r"$G[S_A,S_B]=0$ exactly.",
    "#f0e6fa", ec="#9b59b6", fs=8.5)
arrow(ax4, 5.5, 2.7, 5.5, 2.4)

# ── save ────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    path = os.path.join(OUT, f"color_class_explainer.{ext}")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close(fig)
