#!/usr/bin/env python3
"""Plot a representative trajectory across every selected topological-slab column."""

from __future__ import annotations

import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from plot_typical_hard_wall_x_slices import (
    FIT_MAX,
    FIT_MIN,
    OUTPUT_DIR,
    configure_matplotlib,
    fit_decay_exponent,
    load_trajectories,
)


STEM = "typical_hard_wall_topological_slab_correlators"
X_SLICES = (5, 6, 7, 9, 11, 13, 14, 15)
X_LABELS = {
    5: r"$x_L=5$",
    6: r"$x_L+1=6$",
    7: r"$x=7$",
    9: r"$x=9$",
    11: r"$x=11$",
    13: r"$x=13$",
    14: r"$x_R-1=14$",
    15: r"$x_R=15$",
}
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")
LINESTYLES = ("-", "--", ":", "-.", "-.", ":", "--", "-")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, provenance = load_trajectories()
    betas = np.asarray([fit_decay_exponent(row) for row in rows], dtype=np.float64)
    median_beta = float(np.median(betas))
    position = int(np.argmin(np.abs(betas - median_beta)))
    typical = rows[position]
    typical_beta = float(betas[position])
    archived_ry = np.asarray(typical["ry"], dtype=np.int64)
    if not np.array_equal(archived_ry, np.arange(17, dtype=np.int64)):
        raise RuntimeError("Expected archived separations 0,...,Ny/2")
    ry = np.arange(1, 33, dtype=np.int64)
    x_resolved = np.asarray(typical["x_resolved"], dtype=np.float64)

    def full_periodic_curve(x: int) -> np.ndarray:
        half = x_resolved[x]
        # Translational y averaging and Hermiticity give G_x(r)=G_x(Ny-r).
        # The final entry r=Ny is the periodic recurrence of the r=0 contact term.
        return np.concatenate((half[1:], half[-2::-1]))

    configure_matplotlib()
    figure, axis = plt.subplots(figsize=(3.375, 3.0))
    colors = plt.get_cmap("viridis")(np.linspace(0.04, 0.96, len(X_SLICES)))
    for index, x in enumerate(X_SLICES):
        is_wall = x in (5, 15)
        axis.plot(
            ry,
            full_periodic_curve(x),
            color=colors[index],
            marker=MARKERS[index],
            linestyle=LINESTYLES[index],
            linewidth=1.35 if is_wall else 0.95,
            markersize=3.4 if is_wall else 2.7,
            markerfacecolor=colors[index] if is_wall else "white",
            markeredgecolor=colors[index],
            markeredgewidth=0.7,
            label=X_LABELS[x],
        )

    axis.set_yscale("log")
    axis.set_xlim(0.5, 32.5)
    axis.set_ylim(3e-12, 5e-1)
    axis.set_xticks([1, 8, 16, 24, 32])
    axis.set_xlabel(r"separation $r_y$")
    axis.set_ylabel(r"squared correlator $G_x(r_y)$")
    axis.set_title(
        rf"hard wall, $20\times32$, $t=2N_y$; typical trajectory {int(typical['sample_index'])}"
    )
    axis.axvspan(16, 32.5, color="#777777", alpha=0.055, linewidth=0, zorder=-10)
    axis.axvline(16, color="#777777", linestyle=(0, (2, 2)), linewidth=0.7, zorder=-9)
    axis.text(
        16.45,
        3.0e-3,
        r"antipode $N_y/2$",
        fontsize=6.2,
        color="#555555",
        rotation=90,
        va="top",
    )
    axis.legend(
        loc="upper center",
        frameon=False,
        ncol=4,
        columnspacing=0.55,
        handlelength=1.7,
        handletextpad=0.35,
        labelspacing=0.25,
        borderaxespad=0.3,
    )
    axis.text(21.5, 2.2e-3, r"periodic reflection", fontsize=6.2, color="#555555")
    figure.subplots_adjust(left=0.19, right=0.98, bottom=0.15, top=0.92)
    metadata = {
        "Title": "Typical hard-wall trajectory across the topological slab",
        "Author": "class_A_fermionic_adaptive_circuit analysis",
        "Subject": (
            f"Sample {int(typical['sample_index'])}; x-averaged beta={typical_beta:.8f}; "
            f"ensemble median beta={median_beta:.8f}"
        ),
    }
    figure.savefig(OUTPUT_DIR / f"{STEM}.pdf", metadata=metadata)
    figure.savefig(OUTPUT_DIR / f"{STEM}.png", dpi=300, metadata=metadata)
    plt.close(figure)

    with (OUTPUT_DIR / f"{STEM}_source_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["sample_index", "cycle", "x", "position", "r_y", "squared_correlator"],
        )
        writer.writeheader()
        for x in X_SLICES:
            curve = full_periodic_curve(x)
            for index, separation in enumerate(ry):
                writer.writerow(
                    {
                        "sample_index": int(typical["sample_index"]),
                        "cycle": int(typical["cycle"]),
                        "x": x,
                        "position": X_LABELS[x].replace("$", ""),
                        "r_y": int(separation),
                        "squared_correlator": f"{curve[index]:.17g}",
                    }
                )

    summary = {
        "schema": "typical_hard_wall_topological_slab_correlators_v1",
        "selection": {
            "rule": "minimum absolute distance from the ensemble-median trajectory-level x-averaged decay exponent",
            "fit_model": "log G_xavg = intercept - beta log(r_y)",
            "fit_window": [FIT_MIN, FIT_MAX],
            "ensemble_size": len(rows),
            "ensemble_median_beta": median_beta,
            "selected_sample_index": int(typical["sample_index"]),
            "selected_beta": typical_beta,
        },
        "contract": {
            "construction": "hard/support-terminated",
            "Nx": 20,
            "Ny": 32,
            "alpha_1": 1,
            "alpha_2": 30,
            "nshell": 1,
            "cycle": int(typical["cycle"]),
            "initialization": "pure/default",
            "sequence": "raster_y",
            "perfect_correction": True,
            "dtype": "complex128",
            "dw_location": [5, 15],
            "selected_global_charge": int(typical["global_charge"]),
        },
        "x_slices": [{"x": x, "label": X_LABELS[x].replace("$", "")} for x in X_SLICES],
        "separation_display": {
            "range": [1, 32],
            "archived_independent_range": [0, 16],
            "reflected_range": [17, 31],
            "reflection_identity": "G_x(r_y) = G_x(Ny-r_y)",
            "r_y_32": "periodic recurrence of the archived r_y=0 contact term",
        },
        "input_files": provenance,
        "outputs": [f"{STEM}.pdf", f"{STEM}.png", f"{STEM}_source_data.csv"],
    }
    (OUTPUT_DIR / f"{STEM}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["selection"], indent=2))


if __name__ == "__main__":
    main()
