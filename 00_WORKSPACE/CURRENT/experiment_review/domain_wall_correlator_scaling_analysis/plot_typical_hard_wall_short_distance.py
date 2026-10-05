#!/usr/bin/env python3
"""Plot only the numerically resolved short-distance dynamical correlators."""

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


STEM = "typical_hard_wall_topological_slab_short_distance"
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
RY_MAX = 7
VISUALIZATION_CUTOFF = 1.0e-8


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, provenance = load_trajectories()
    betas = np.asarray([fit_decay_exponent(row) for row in rows], dtype=np.float64)
    median_beta = float(np.median(betas))
    position = int(np.argmin(np.abs(betas - median_beta)))
    typical = rows[position]
    typical_beta = float(betas[position])
    x_resolved = np.asarray(typical["x_resolved"], dtype=np.float64)
    separations = np.arange(1, RY_MAX + 1, dtype=np.int64)

    configure_matplotlib()
    figure, axis = plt.subplots(figsize=(3.375, 3.05))
    colors = plt.get_cmap("viridis")(np.linspace(0.04, 0.96, len(X_SLICES)))
    for index, x in enumerate(X_SLICES):
        raw = x_resolved[x, 1 : RY_MAX + 1]
        displayed = np.where(raw > VISUALIZATION_CUTOFF, raw, np.nan)
        is_wall = x in (5, 15)
        axis.plot(
            separations,
            displayed,
            color=colors[index],
            marker=MARKERS[index],
            linestyle=LINESTYLES[index],
            linewidth=1.35 if is_wall else 0.95,
            markersize=3.7 if is_wall else 3.0,
            markerfacecolor=colors[index] if is_wall else "white",
            markeredgecolor=colors[index],
            markeredgewidth=0.75,
            label=X_LABELS[x],
        )

    axis.axhline(
        VISUALIZATION_CUTOFF,
        color="#555555",
        linestyle=(0, (2.5, 2.0)),
        linewidth=0.8,
        zorder=-5,
    )
    axis.text(
        6.92,
        1.25 * VISUALIZATION_CUTOFF,
        r"display cutoff $10^{-8}$",
        ha="right",
        va="bottom",
        fontsize=6.3,
        color="#555555",
    )
    axis.set_yscale("log")
    axis.set_xlim(0.8, 7.2)
    axis.set_ylim(7e-9, 1.1e-1)
    axis.set_xticks(separations)
    axis.set_xlabel(r"separation $r_y$")
    axis.set_ylabel(r"squared correlator $G_x(r_y)$")
    axis.set_title(
        rf"hard wall, $20\times32$, $t=2N_y$; typical trajectory {int(typical['sample_index'])}"
    )
    axis.legend(
        loc="upper right",
        frameon=False,
        ncol=2,
        columnspacing=0.7,
        handlelength=1.9,
        handletextpad=0.35,
        labelspacing=0.24,
        borderaxespad=0.25,
    )
    figure.subplots_adjust(left=0.19, right=0.98, bottom=0.15, top=0.92)
    metadata = {
        "Title": "Resolved short-distance hard-wall dynamical correlators",
        "Author": "class_A_fermionic_adaptive_circuit analysis",
        "Subject": (
            f"Sample {int(typical['sample_index'])}; r_y=1..{RY_MAX}; "
            f"visualization cutoff={VISUALIZATION_CUTOFF:g}"
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
            fieldnames=[
                "sample_index",
                "cycle",
                "x",
                "position",
                "r_y",
                "squared_correlator_raw",
                "above_visualization_cutoff",
            ],
        )
        writer.writeheader()
        for x in X_SLICES:
            for separation in separations:
                value = float(x_resolved[x, separation])
                writer.writerow(
                    {
                        "sample_index": int(typical["sample_index"]),
                        "cycle": int(typical["cycle"]),
                        "x": x,
                        "position": X_LABELS[x].replace("$", ""),
                        "r_y": int(separation),
                        "squared_correlator_raw": f"{value:.17g}",
                        "above_visualization_cutoff": value > VISUALIZATION_CUTOFF,
                    }
                )

    summary = {
        "schema": "typical_hard_wall_short_distance_correlator_v1",
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
        },
        "display": {
            "r_y_range": [1, RY_MAX],
            "visualization_cutoff": VISUALIZATION_CUTOFF,
            "cutoff_semantics": "values at or below the cutoff are retained in source data but masked in the figure",
        },
        "x_slices": [{"x": x, "label": X_LABELS[x].replace("$", "")} for x in X_SLICES],
        "input_files": provenance,
        "outputs": [f"{STEM}.pdf", f"{STEM}.png", f"{STEM}_source_data.csv"],
    }
    (OUTPUT_DIR / f"{STEM}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["display"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
