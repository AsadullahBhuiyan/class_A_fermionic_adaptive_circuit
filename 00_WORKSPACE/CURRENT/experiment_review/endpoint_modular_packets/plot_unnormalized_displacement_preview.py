"""Compare conditional COM with exact unnormalized moments at saved times.

No propagation is rerun. Importantly, this integrates the saved density, rather
than multiplying separately cut-averaged retained charges and displacements.
Original analyses and manuscript figures are not modified.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "outputs/hard_n20x32_a1-1-3_y8_chirality_figure_v2/averaged_observables.npz"
OUTPUT = PROJECT / "outputs/hard_n20x32_y8_unnormalized_displacement_preview_v1"


def main():
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    summary = json.loads(SOURCE.with_name("analysis_summary.json").read_text())
    assert summary["data_sha256"] == before
    with np.load(SOURCE, allow_pickle=False) as data:
        times, snapshots = data["times"], data["snapshot_times"]
        density = data["alpha1_density_samples"]
        com = data["alpha1_dy_window_samples"]
        full = data["alpha1_dy_full_samples"]
    assert density.shape == (100, 2, 5, 16, 20)
    np.testing.assert_allclose(snapshots, [0, .1, .2, .5, 1], atol=1e-12)
    np.testing.assert_allclose(density.sum(axis=(-2, -1)), 2, atol=1e-10, rtol=0)
    indices = [int(np.argmin(abs(times - t))) for t in snapshots]
    # Independent check: full-system first moment equals two times the COM.
    full_moment = np.einsum("sptyx,y->spt", density, np.arange(16)-8)
    np.testing.assert_allclose(full_moment, 2*full[:, :, indices], atol=1e-10, rtol=0)
    raw = np.empty((100, 2, 5))
    for packet, wall in enumerate((5, 15)):
        distance = abs(np.arange(20) - wall)
        window = np.minimum(distance, 20-distance) <= 2
        raw[:, packet] = np.einsum("styx,y->st", density[:, packet][:, :, :, window], np.arange(16)-8)
    np.testing.assert_allclose(raw[:, :, 0], 0, atol=1e-10)
    mean, sem = com.mean(0), com.std(0, ddof=1)/10
    raw_mean, raw_sem = raw.mean(0), raw.std(0, ddof=1)/10
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["CMU Sans Serif"],
        "font.size": 8, "axes.labelsize": 9, "mathtext.fontset": "cm",
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "legend.frameon": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.8))
    for packet, (wall, color, style, marker) in enumerate(zip(
            (5, 15), ("#D55E00", "#0072B2"), ("-", "--"), ("o", "s"))):
        label = rf"$({wall},8)$"
        axes[0].plot(times, mean[packet], color=color, ls=style, lw=1.1, label=label)
        axes[0].fill_between(times, mean[packet]-sem[packet], mean[packet]+sem[packet], color=color, alpha=.18, lw=0)
        axes[1].errorbar(snapshots, raw_mean[packet], yerr=raw_sem[packet], color=color,
                         marker=marker, ls="none", ms=4, capsize=2, lw=.8, label=label)
    for ax in axes:
        ax.axhline(0, color=".7", lw=.6, zorder=0)
        ax.set(xlim=(-.025, 1.025), xlabel=r"modular time $t_{\rm mod}$")
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.legend(loc="center right", bbox_to_anchor=(1, .6), ncol=2, handlelength=1.5)
    axes[0].set(title="Current: conditional center of mass", ylim=(-.95, .95),
                ylabel=r"$\overline{\Delta y_W}$ (lattice sites)")
    axes[1].set(title="Without retained-charge normalization", ylim=(-1.9, 1.9),
                ylabel=r"$\overline{\sum_{x\in W,y}(y-8)\rho(x,y,t)}$")
    axes[1].text(.5, .035, "Exact saved times; no interpolation", transform=axes[1].transAxes,
                 ha="center", va="bottom", fontsize=8)
    for ax, letter in zip(axes, "ab"):
        ax.text(-.17, 1.05, f"({letter})", transform=ax.transAxes, fontsize=9)
    fig.tight_layout(pad=1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT / f"normalization_comparison.{ext}", dpi=300)
    plt.close(fig)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == before
    metadata = dict(source=str(SOURCE), source_sha256=before,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        samples=100, translated_cuts_per_sample=32, wall_window_radius=2,
        injection_charge=2, injection_y=8, cutoff=1e-10,
        estimator="sum over the wall window of (y-8)*rho; then cut and trajectory averages",
        uncertainty="SEM across 100 trajectories; cuts are not independent samples",
        full_curve_recomputed=False, snapshot_times=snapshots.tolist(),
        raw_mean=raw_mean.tolist(), raw_sem=raw_sem.tolist(),
        conditional_mean_at_snapshot_times=mean[:, indices].tolist())
    (OUTPUT / "preview_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(OUTPUT / "normalization_comparison.png")


if __name__ == "__main__":
    main()
