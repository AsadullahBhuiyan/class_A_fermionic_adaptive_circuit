"""Generate figures for the Lyapunov spectrum writeup."""
from __future__ import annotations
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
CAMPAIGN    = (HERE.parent / "gpu_data" / "lyapunov_spectra" / "campaigns"
               / "N20_Ny30-50_nsh1_a1-1_S25_cyclesNy")
RUNS        = CAMPAIGN / "runs"
FIGDIR      = HERE
FIGDIR.mkdir(parents=True, exist_ok=True)

# ── rc ─────────────────────────────────────────────────────────────────────────
DPI   = 300
W1    = 3.375
W2    = 6.75
FS    = 8
FSS   = 7

available = {f.name for f in font_manager.fontManager.ttflist}
FONT = "CMU Sans Serif" if "CMU Sans Serif" in available else "DejaVu Sans"
mpl.rcParams.update({
    "font.family":     FONT,
    "font.size":       FS,
    "axes.titlesize":  FS,
    "axes.labelsize":  FS,
    "xtick.labelsize": FSS,
    "ytick.labelsize": FSS,
    "legend.fontsize": FSS,
    "figure.dpi":      DPI,
    "savefig.dpi":     DPI,
    "axes.linewidth":  0.6,
    "lines.linewidth": 1.1,
    "text.usetex":     False,
})

NY_VALUES = [30, 40, 50]
NX        = 20
_cmap     = mpl.colormaps["tab10"].colors
COLORS    = {ny: _cmap[i] for i, ny in enumerate(NY_VALUES)}
PROT_COL  = {"perfect_correction": "tab:blue", "postselect": "tab:orange"}
PROT_LABEL= {"perfect_correction": "perfect correction", "postselect": "post-selection"}

CONFIGS = {
    "PC_DW0": [f"N20x{ny}_DW0_dwtrunc0_a2-1_nsh1_perfect_correction"  for ny in NY_VALUES],
    "PC_DW1": [f"N20x{ny}_DW1_dwtrunc1_a2-30_nsh1_perfect_correction" for ny in NY_VALUES],
    "PS_DW0": [f"N20x{ny}_DW0_dwtrunc0_a2-1_nsh1_postselect"          for ny in NY_VALUES],
    "PS_DW1": [f"N20x{ny}_DW1_dwtrunc1_a2-30_nsh1_postselect"         for ny in NY_VALUES],
}


# ── load data ──────────────────────────────────────────────────────────────────
def load_run(config_id: str) -> dict:
    d = RUNS / config_id
    with np.load(d / "lyapunov_spectra.npz",  allow_pickle=False) as f:
        spectra = f["lyapunov_spectra"]
    with np.load(d / "real_space_chern.npz",  allow_pickle=False) as f:
        chern   = f["real_space_chern"]
    with np.load(d / "local_charge_cell.npz", allow_pickle=False) as f:
        charge  = f["local_charge_cell"]
    summary = json.loads((d / "run_summary.json").read_text())
    S, C, n_vec = spectra.shape
    Nx_, Ny_ = charge.shape[2], charge.shape[3]
    total_charge    = charge.sum(axis=(-1, -2))
    filling         = total_charge / float(Nx_ * Ny_)
    lyapunov_min_abs = np.min(np.abs(spectra), axis=-1)
    x_profile        = charge.mean(axis=-1)              # (S, C, Nx)
    return {
        "spectra":          spectra,
        "chern":            chern,
        "charge":           charge,
        "filling":          filling,
        "lyapunov_min_abs": lyapunov_min_abs,
        "x_profile":        x_profile,
        "summary":          summary,
        "S": S, "C": C, "n_vec": n_vec,
    }


runs: dict[str, dict] = {}
for grp, cfgs in CONFIGS.items():
    for cfg in cfgs:
        runs[cfg] = load_run(cfg)

print("Data loaded.")


def mean_err(a: np.ndarray, axis: int = 0):
    m = np.mean(a, axis=axis)
    n = a.shape[axis]
    e = np.std(a, axis=axis, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(m)
    return m, e


def cycle_axis(cfg: str) -> np.ndarray:
    return np.arange(1, runs[cfg]["C"] + 1)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1: Lyapunov gap proxy vs cycle — 2x2 grid (DW off/on) x (PC/PS)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(W2, 3.6), constrained_layout=True)
panels = [
    (axes[0, 0], CONFIGS["PC_DW0"], "perfect correction, DW off"),
    (axes[0, 1], CONFIGS["PC_DW1"], "perfect correction, DW on"),
    (axes[1, 0], CONFIGS["PS_DW0"], "post-selection, DW off"),
    (axes[1, 1], CONFIGS["PS_DW1"], "post-selection, DW on"),
]
for ax, cfgs, title in panels:
    for cfg in cfgs:
        ny    = runs[cfg]["summary"]["Ny"]
        color = COLORS[ny]
        cyc   = cycle_axis(cfg)
        m, e  = mean_err(runs[cfg]["lyapunov_min_abs"], axis=0)
        ax.plot(cyc, m, color=color, label=f"$N_y={ny}$")
        ax.fill_between(cyc, m - e, m + e, color=color, alpha=0.15, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel("cycle $t$")
    ax.set_ylabel(r"$\Delta_\lambda \equiv \min_i |\lambda_i(t)|$")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=FSS)
fig.savefig(FIGDIR / "fig_lyapunov_gap_vs_cycle.pdf", bbox_inches="tight")
fig.savefig(FIGDIR / "fig_lyapunov_gap_vs_cycle.png", bbox_inches="tight")
plt.close(fig)
print("Fig 1 done.")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2: Final-cycle Lyapunov gap vs Ny — DW off vs on, PC vs PS
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(W2, 2.2), constrained_layout=True)
grp_pairs = [
    (axes[0], [("PC_DW0", "perfect corr.", "o-", PROT_COL["perfect_correction"]),
               ("PS_DW0", "post-sel.",     "s--", PROT_COL["postselect"])],
     "DW off"),
    (axes[1], [("PC_DW1", "perfect corr.", "o-", PROT_COL["perfect_correction"]),
               ("PS_DW1", "post-sel.",     "s--", PROT_COL["postselect"])],
     "DW on"),
]
for ax, grps, title in grp_pairs:
    for grp_key, label, fmt, color in grps:
        nys, means, errs = [], [], []
        for cfg in CONFIGS[grp_key]:
            ny     = runs[cfg]["summary"]["Ny"]
            gap    = runs[cfg]["lyapunov_min_abs"][:, -1]
            m, e   = mean_err(gap, axis=0)
            nys.append(ny); means.append(float(m)); errs.append(float(e))
        nys = np.array(nys)
        ax.errorbar(nys, means, yerr=errs, fmt=fmt, color=color,
                    label=label, capsize=3, ms=4)
    ax.set_xticks(NY_VALUES)
    ax.set_xlabel(r"$N_y$")
    ax.set_ylabel(r"$\Delta_\lambda$ (final cycle)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=FSS)
fig.savefig(FIGDIR / "fig_gap_vs_Ny.pdf", bbox_inches="tight")
fig.savefig(FIGDIR / "fig_gap_vs_Ny.png", bbox_inches="tight")
plt.close(fig)
print("Fig 2 done.")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3: Mean Lyapunov spectrum (sorted) at final cycle — DW off vs on
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(W2, 2.2), constrained_layout=True)
for ax, grp_key, title in [
    (axes[0], "PC_DW0", "perfect correction, DW off"),
    (axes[1], "PC_DW1", "perfect correction, DW on"),
]:
    for cfg in CONFIGS[grp_key]:
        ny      = runs[cfg]["summary"]["Ny"]
        color   = COLORS[ny]
        n_vec   = runs[cfg]["n_vec"]
        spec    = runs[cfg]["spectra"][:, -1, :]   # (S, n_vec)
        mean_sp = np.sort(np.mean(spec, axis=0))
        frac    = np.arange(n_vec) / n_vec
        ax.plot(frac, mean_sp, color=color, label=f"$N_y={ny}$")
    ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel(r"mode index $i / n_{\rm vec}$")
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=FSS)
fig.savefig(FIGDIR / "fig_lyapunov_spectrum_shape.pdf", bbox_inches="tight")
fig.savefig(FIGDIR / "fig_lyapunov_spectrum_shape.png", bbox_inches="tight")
plt.close(fig)
print("Fig 3 done.")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4: Real-space Chern number vs cycle — 2x2 grid
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(W2, 3.6), constrained_layout=True)
panels = [
    (axes[0, 0], CONFIGS["PC_DW0"], "perfect correction, DW off"),
    (axes[0, 1], CONFIGS["PC_DW1"], "perfect correction, DW on"),
    (axes[1, 0], CONFIGS["PS_DW0"], "post-selection, DW off"),
    (axes[1, 1], CONFIGS["PS_DW1"], "post-selection, DW on"),
]
for ax, cfgs, title in panels:
    for cfg in cfgs:
        ny    = runs[cfg]["summary"]["Ny"]
        color = COLORS[ny]
        cyc   = cycle_axis(cfg)
        m, e  = mean_err(runs[cfg]["chern"], axis=0)
        ax.plot(cyc, m, color=color, label=f"$N_y={ny}$")
        ax.fill_between(cyc, m - e, m + e, color=color, alpha=0.15, linewidth=0)
    ax.axhline(1, color="k", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel("cycle $t$")
    ax.set_ylabel(r"$\langle \mathcal{C} \rangle$")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=FSS)
fig.savefig(FIGDIR / "fig_chern_vs_cycle.pdf", bbox_inches="tight")
fig.savefig(FIGDIR / "fig_chern_vs_cycle.png", bbox_inches="tight")
plt.close(fig)
print("Fig 4 done.")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5: Charge x-profile at final cycle — DW off vs on (PC)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(W2, 2.2), constrained_layout=True)
x_ax = np.arange(NX)
for ax, grp_key, title in [
    (axes[0], "PC_DW0", "DW off"),
    (axes[1], "PC_DW1", "DW on"),
]:
    for cfg in CONFIGS[grp_key]:
        ny    = runs[cfg]["summary"]["Ny"]
        color = COLORS[ny]
        xp    = runs[cfg]["x_profile"][:, -1, :]   # (S, Nx)
        m, e  = mean_err(xp, axis=0)
        ax.plot(x_ax, m, color=color, label=f"$N_y={ny}$")
        ax.fill_between(x_ax, m - e, m + e, color=color, alpha=0.15, linewidth=0)
    ax.axhline(1.0, color="k", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\langle Q_x \rangle / N_y$")
    ax.set_title(f"charge profile, {title}")
    ax.legend(frameon=False, fontsize=FSS)
    if grp_key == "PC_DW1":
        ax.axvspan(5, 15, color="tab:green", alpha=0.08, label="topo. region")
fig.savefig(FIGDIR / "fig_charge_profile.pdf", bbox_inches="tight")
fig.savefig(FIGDIR / "fig_charge_profile.png", bbox_inches="tight")
plt.close(fig)
print("Fig 5 done.")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6: Filling fraction vs cycle — 2x2 grid
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(W2, 3.6), constrained_layout=True)
panels = [
    (axes[0, 0], CONFIGS["PC_DW0"], "perfect correction, DW off"),
    (axes[0, 1], CONFIGS["PC_DW1"], "perfect correction, DW on"),
    (axes[1, 0], CONFIGS["PS_DW0"], "post-selection, DW off"),
    (axes[1, 1], CONFIGS["PS_DW1"], "post-selection, DW on"),
]
for ax, cfgs, title in panels:
    for cfg in cfgs:
        ny    = runs[cfg]["summary"]["Ny"]
        color = COLORS[ny]
        cyc   = cycle_axis(cfg)
        m, e  = mean_err(runs[cfg]["filling"], axis=0)
        ax.plot(cyc, m, color=color, label=f"$N_y={ny}$")
        ax.fill_between(cyc, m - e, m + e, color=color, alpha=0.15, linewidth=0)
    ax.axhline(1.0, color="k", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel("cycle $t$")
    ax.set_ylabel(r"$\nu = Q_{\rm tot} / (N_x N_y)$")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=FSS)
fig.savefig(FIGDIR / "fig_filling_vs_cycle.pdf", bbox_inches="tight")
fig.savefig(FIGDIR / "fig_filling_vs_cycle.png", bbox_inches="tight")
plt.close(fig)
print("Fig 6 done.")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7: DW gap proxy scatter — gap vs Chern at final cycle (PC, DW on)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(W1, 2.4), constrained_layout=True)
for cfg in CONFIGS["PC_DW1"]:
    ny    = runs[cfg]["summary"]["Ny"]
    color = COLORS[ny]
    gap   = runs[cfg]["lyapunov_min_abs"][:, -1]    # (S,)
    chern = runs[cfg]["chern"][:, -1]               # (S,)
    ax.scatter(chern, gap, color=color, s=14, alpha=0.7,
               label=f"$N_y={ny}$", linewidths=0)
ax.set_xlabel(r"$\mathcal{C}$ (final cycle)")
ax.set_ylabel(r"$\Delta_\lambda$ (final cycle)")
ax.set_yscale("log")
ax.set_title("gap vs Chern — PC, DW on")
ax.legend(frameon=False, fontsize=FSS)
fig.savefig(FIGDIR / "fig_gap_vs_chern.pdf", bbox_inches="tight")
fig.savefig(FIGDIR / "fig_gap_vs_chern.png", bbox_inches="tight")
plt.close(fig)
print("Fig 7 done.")


print("All figures saved to", FIGDIR)
