# --- project bootstrap ---
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)
# -------------------------

#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from threadpoolctl import threadpool_limits
from fgtn.classA_U1FGTN import classA_U1FGTN
# User knobs
CPU = 56               # set max BLAS threads; None uses all cores
CYCLES = 20               # nominal cycles; sweeps = round(CYCLES / p)
P_LIST = [1.0, 1e-1]
OUT_FIG = "figs/corr_y_profiles/exclude_dw_symmetric_seq_p_sweep.png"

# Model setup
Nx = Ny = 21
model = classA_U1FGTN(
    Nx=Nx,
    Ny=Ny,
    DW=True,
    nshell=None,
    alpha_1=30,
    alpha_2=1,
)

def corr_y_profile(G_top, Nx, Ny, x0):
    """Return (ry_vals, C(ry)) averaged over y and spins at fixed x0."""
    G_top = np.asarray(G_top, dtype=np.complex128)
    Nlayer = 2 * Nx * Ny
    if G_top.shape != (Nlayer, Nlayer):
        raise ValueError(f"Expected top-layer ({Nlayer},{Nlayer}); got {G_top.shape}")

    G2 = 0.5 * (G_top + np.eye(Nlayer, dtype=np.complex128))
    G6 = G2.reshape(2, Nx, Ny, 2, Nx, Ny, order="F")
    G6 = np.transpose(G6, (1, 2, 0, 4, 5, 3))  # (Nx,Ny,2, Nx,Ny,2)

    x0 = int(x0) % Nx
    ry_vals = np.arange(1, Ny // 2 + 1, dtype=int)  # start at 1 to avoid log(0)
    Ny_loc = Ny

    Gx = G6[x0, :, :, x0, :, :]  # (Ny,2,Ny,2)
    Y = np.arange(Ny_loc, dtype=np.intp)[:, None]
    profiles = []
    for ry in ry_vals:
        Yp = (Y + ry) % Ny_loc
        blocks = Gx[Y, :, Yp, :]  # (Ny,2,2)
        profiles.append(np.sum(np.abs(blocks) ** 2, axis=(0, 2, 3)) / (2.0 * Ny_loc))
    return ry_vals, np.array(profiles, dtype=float)

# Run and collect profiles
outdir = Path(OUT_FIG).parent
outdir.mkdir(parents=True, exist_ok=True)
xL, xR = model.DW_loc

def _pick_x_positions():
    xs = [
        (xL // 2) % Nx,
        (xL - 1) % Nx, xL % Nx, (xL + 1) % Nx,
        ((xL + xR) // 2) % Nx,
        (xR - 1) % Nx, xR % Nx, (xR + 1) % Nx,
        (xR + (Nx // 2)) % Nx,
    ]
    seen, uniq = set(), []
    for x in xs:
        if x not in seen:
            uniq.append(int(x)); seen.add(int(x))
    return [(x, f"{x}") for x in uniq]

profiles = {}
with threadpool_limits(limits=CPU or os.cpu_count() or 1):
    for p in P_LIST:
        res = model.run_markov_channel(
            G_history=False,
            progress=True,
            cycles=CYCLES,
            p=p,
            save=True,
            save_suffix=f"_DW1_seq_symm_dw_exclude",
            sequence="exclude_dw_symmetric",
        )
        G_ss = res["G_final"]
        entries = []
        for x, label in _pick_x_positions():
            ry_vals, vals = corr_y_profile(G_ss, Nx, Ny, x)
            entries.append((label, ry_vals, vals))
        profiles[p] = entries
"""
        sequence options (case-insensitive):
            - "snake_y": Rx outer loop, Ry inner loop increasing (previous default)
            - "reverse_snake_y": reverse ordering of snake_y
            - "snake_x": Ry outer loop, Rx inner loop increasing
            - "reverse_snake_x": reverse ordering of snake_x
            - "random": shuffle all (Rx,Ry) each cycle
            - "exclude_dw_random": random over sites excluding DW_loc columns
            - "dw_symmetric": Rx sweep mid->Nx-1 then mid-1->0 for each Ry
            - "dw_symmetric_2": Rx sweep 0->mid then Nx-1->mid+1 for each Ry
            - "dw_symmetric_y": Ry sweep mid->Ny-1 then mid-1->0 for each Rx; Rx order is mid->right then mid-1->left
            - "exclude_dw_symmetric": dw_symmetric but skip Rx in DW_loc
            - "exclude_dw_symmetric_y": dw_symmetric Rx (skipping DW_loc), Ry starts at Ny//2 then outward
"""
# Plot
fig, axes = plt.subplots(1, len(P_LIST), figsize=(5 * max(1, len(P_LIST)), 5), sharey=True)
if len(P_LIST) == 1:
    axes = [axes]

max_curves = max(len(v) for v in profiles.values())
gradient = np.linspace(0.0, 1.0, max_curves)

for ax, p in zip(axes, P_LIST):
    colors = plt.cm.viridis(gradient[: len(profiles[p])])
    for (label, ry, vals), color in zip(profiles[p], colors):
        ax.plot(ry, vals, marker="o", label=f"x={label}", color=color)
    ax.set_yscale("log")
    ax.set_xscale("linear")
    ax.set_title(f"p = {p:g}")
    ax.set_xlabel(r"$r_y$")
    ax.grid(True, which="both", alpha=0.3)
    ax.text(
        0.05,
        0.95,
        f"DWs: $x_0={xL},\\;x_1={xR}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.3"),
    )

axes[0].set_ylabel(r"$|G|^2$ avg")
axes[-1].legend(title="x position", bbox_to_anchor=(1.05, 1.0), loc="upper left")

plt.tight_layout()
fig.savefig(OUT_FIG, dpi=150)
print(f"Saved plot to {OUT_FIG}")
