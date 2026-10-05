#!/usr/bin/env python3
"""Wall/center contour curves and 1/e times from verified hard-wall data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
import numpy as np

from plot_hard_wall_purification_spatial import (
    configure_plotting, load_hard_wall_data, write_csv,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs" / "hard_wall_relaxation_scaling_v1"
SIZES = (20, 30, 40)
POSITIONS = (5, 15, 10)


def crossing_time(curve: np.ndarray, baseline: int) -> float:
    """First downward 1/e crossing after baseline, linearly interpolated.

    Return elapsed cycles, not absolute cycle. No smoothing or monotonicity
    constraint is imposed. A missing crossing is right-censored (NaN).
    """
    if not np.all(np.isfinite(curve)) or curve[baseline] <= 0:
        raise ValueError("Invalid contour curve")
    target = curve[baseline] / np.e
    hits = np.flatnonzero(curve[baseline + 1:] <= target)
    if not len(hits):
        return float("nan")
    j = baseline + 1 + int(hits[0])
    fraction = (curve[j - 1] - target) / (curve[j - 1] - curve[j])
    return float(j - 1 + fraction - baseline)


def summarize_threshold(samples: np.ndarray, baseline: int) -> dict:
    """Threshold of ensemble mean; SE from deleting whole trajectories.

    Keeping the baseline and crossing portions of each trajectory together
    retains their time correlation. No cycle or wall is an independent sample.
    """
    n = len(samples)
    mean = samples.mean(axis=0)
    tau = crossing_time(mean, baseline)
    deleted = (samples.sum(axis=0)[None, :] - samples) / (n - 1)
    jackknife = np.array([crossing_time(y, baseline) for y in deleted])
    if not np.isfinite(tau) or not np.all(np.isfinite(jackknife)):
        raise ValueError("Unresolved ensemble/jackknife threshold crossing")
    se = np.sqrt((n - 1) / n * np.sum((jackknife - jackknife.mean()) ** 2))
    return {
        "tau_cycles": tau, "tau_standard_error_cycles": float(se),
        "baseline_entropy_density": float(mean[baseline]),
        "threshold_entropy_density": float(mean[baseline] / np.e),
        "resolved_jackknife_replicates": n,
    }


def main() -> None:
    print("Verifying and reading 60 hard-wall shards / 300 trajectories", flush=True)
    arrays, provenance = load_hard_wall_data(verify_hashes=True)
    OUT.mkdir(parents=True, exist_ok=True)
    curve_rows, threshold_rows, sensitivity_rows = [], [], []
    summary = {}
    for ny in SIZES:
        samples = arrays[ny]["entropy_x"] / ny
        if not np.all(np.isfinite(samples)):
            raise ValueError(f"Ny={ny}: nonfinite entropy contour")
        for x in POSITIONS:
            values = samples[:, :, x]
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / np.sqrt(len(values))
            for t, (m, e) in enumerate(zip(mean, sem)):
                curve_rows.append(dict(Ny=ny, x=x, cycle=t, normalized_cycle=t / ny,
                                       mean_entropy_density=float(m), sem=float(e)))
            stats = summarize_threshold(values, baseline=ny)
            row = dict(Ny=ny, x=x, samples=len(values), baseline_cycle=ny,
                       tau_over_Ny=stats["tau_cycles"] / ny, **stats)
            threshold_rows.append(row)
            summary[ny, x] = row
            if x in (5, 15):
                for start in (0.5, 1.0, 1.5, 2.0):
                    stats = summarize_threshold(values, baseline=round(start * ny))
                    sensitivity_rows.append(dict(Ny=ny, x=x, baseline_t_over_Ny=start,
                                                 **stats))

    fits = {}
    for x in (5, 15):
        times = np.array([summary[ny, x]["tau_cycles"] for ny in SIZES])
        z, intercept = np.polyfit(np.log(SIZES), np.log(times), 1)
        fits[x] = dict(z=float(z), amplitude=float(np.exp(intercept)))
    write_csv(OUT / "contour_curves.csv", curve_rows)
    write_csv(OUT / "relaxation_times.csv", threshold_rows)
    write_csv(OUT / "baseline_sensitivity.csv", sensitivity_rows)

    configure_plotting()
    plt.rcParams.update({"legend.fontsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "axes.labelsize": 9})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.05, 3.05),
                              gridspec_kw={"width_ratios": [1.2, 1]},
                              layout="constrained")
    colors = {20: "#d62728", 30: "#2ca02c", 40: "#1f77b4"}
    markers = {20: "^", 30: "s", 40: "o"}
    styles = {5: "-", 15: "--", 10: ":"}
    for ny in SIZES:
        u = np.arange(4 * ny + 1) / ny
        for x in POSITIONS:
            data = arrays[ny]["entropy_x"][:, :, x] / ny
            m = data.mean(axis=0)
            e = data.std(axis=0, ddof=1) / np.sqrt(len(data))
            a.fill_between(u, np.maximum(m - e, 1e-12), m + e,
                           color=colors[ny], alpha=0.075, linewidth=0)
            a.plot(u, m, color=colors[ny], linestyle=styles[x], linewidth=1.05,
                   marker=markers[ny], markevery=max(1, ny // 2), markersize=3,
                   markerfacecolor="white", markeredgewidth=0.7,
                   alpha=0.7 if x == 10 else 1)
    a.set(yscale="log", xlim=(0, 4), ylim=(1e-6, 2), xlabel=r"cycle $t/N_y$",
          ylabel=r"$\langle s_x(t)\rangle/N_y$")
    size_legend = a.legend(handles=[Line2D([], [], color=colors[n], marker=markers[n],
                              markerfacecolor="white", lw=1, ms=3,
                              label=rf"$N_y={n}$") for n in SIZES],
                          loc="upper right", frameon=False)
    a.add_artist(size_legend)
    a.legend(handles=[Line2D([], [], color="0.2", linestyle=styles[x], lw=1,
                            label=rf"$x={x}$" + (" (center)" if x == 10 else " (wall)"))
                      for x in POSITIONS], loc="upper right", bbox_to_anchor=(1, .80),
             frameon=False, labelspacing=.22)
    a.text(.03, .03, r"Hard wall; $S=100$ per size", transform=a.transAxes, fontsize=8)

    ny_fit = np.geomspace(19, 42, 100)
    for x, color, marker, style in [(5, "#d62728", "^", ":"),
                                     (15, "#1f77b4", "o", "-")]:
        tau = np.array([summary[ny, x]["tau_cycles"] for ny in SIZES])
        se = np.array([summary[ny, x]["tau_standard_error_cycles"] for ny in SIZES])
        b.errorbar(SIZES, tau, yerr=se, fmt=marker, color=color, markerfacecolor="white",
                   markersize=4, capsize=2.5, elinewidth=.9,
                   label=rf"$x={x}:\ z_{{\rm fit}}={fits[x]['z']:.2f}$")
        b.plot(ny_fit, fits[x]["amplitude"] * ny_fit ** fits[x]["z"],
               color=color, linestyle=style, lw=1)
    anchor = np.mean([summary[30, x]["tau_cycles"] for x in (5, 15)])
    b.plot(ny_fit, anchor * ny_fit / 30, "--", color="0.4", lw=1,
           label=r"$z=1$ reference")
    b.set(xscale="log", yscale="log", xlim=(18, 43), ylim=(7, 60),
          xlabel=r"circumference $N_y$", ylabel=r"$1/e$ relaxation time $\tau_x$ (cycles)")
    b.xaxis.set_major_locator(FixedLocator(SIZES))
    b.yaxis.set_major_locator(FixedLocator([10, 15, 20, 30, 40, 50]))
    for axis in (b.xaxis, b.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(NullLocator())
    b.legend(loc="upper left", frameon=False, fontsize=8)
    b.text(.97, .03, r"$\langle s_x(N_y+\tau_x)\rangle=\langle s_x(N_y)\rangle/e$",
           ha="right", transform=b.transAxes, fontsize=8)
    for ax, label in [(a, "(a)"), (b, "(b)")]:
        ax.text(-.15, 1.035, label, transform=ax.transAxes, fontsize=10)
    stem = OUT / "hard_wall_contour_relaxation_scaling"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)

    result = {
        "provenance": provenance,
        "estimator": "first linearly interpolated 1/e crossing of ensemble mean after t=Ny",
        "curve_uncertainty": "sample SEM, S=100 independent complete trajectories",
        "threshold_uncertainty": "leave-one-trajectory-out jackknife standard error; no bootstrap",
        "fit": "unweighted log(tau) vs log(Ny) across Ny=20,30,40, descriptive only",
        "wall_fits": fits,
        "caveat": "Three sizes and size-dependent baseline t0=Ny do not establish an asymptotic dynamic exponent or exact curve collapse.",
        "output_pdf": str(stem.with_suffix(".pdf")),
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"thresholds": threshold_rows, "fits": fits}, indent=2), flush=True)


if __name__ == "__main__":
    main()
