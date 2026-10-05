#!/usr/bin/env python3
"""Plot the hard-wall x-resolved correlators in log-chord coordinates."""

from __future__ import annotations

import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from analyze_finite_size_power_law import (
    OUTPUT_DIR,
    chord,
    configure_matplotlib,
    distribution_summary,
    fit_curve,
    load_new_size,
)


NX = 20
NY = 32
FIT_MIN = 2
FIT_MAX = 8
STEM = "hard_wall_x_resolved_log_chord"
SLICES = {
    5: {"label": r"$x_L$", "color": "#0072B2", "marker": "o", "linestyle": "-"},
    6: {"label": r"$x_L+1$", "color": "#56B4E9", "marker": "^", "linestyle": "--"},
    14: {"label": r"$x_R-1$", "color": "#E69F00", "marker": "v", "linestyle": "--"},
    15: {"label": r"$x_R$", "color": "#D55E00", "marker": "s", "linestyle": "-"},
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, x_resolved, sample_ids, provenance = load_new_size(NY)
    if x_resolved.shape != (100, NX, NY // 2 + 1):
        raise RuntimeError(f"Unexpected x-resolved shape: {x_resolved.shape}")

    separations = np.arange(1, NY // 2 + 1, dtype=np.int64)
    distances = chord(NY, separations)
    log_chord = np.log(distances)
    fit_mask = (separations >= FIT_MIN) & (separations <= FIT_MAX)
    summaries: dict[str, object] = {}

    configure_matplotlib()
    figure, axis = plt.subplots(figsize=(3.375, 2.75))
    for x, style in SLICES.items():
        values = x_resolved[:, x, :]
        mean_curve = values.mean(axis=0)[1:]
        trajectory_fits = np.asarray(
            [fit_curve(curve, NY, FIT_MIN, FIT_MAX) for curve in values],
            dtype=np.float64,
        )
        beta_summary = distribution_summary(trajectory_fits[:, 0])
        mean_curve_fit = fit_curve(values.mean(axis=0), NY, FIT_MIN, FIT_MAX)
        summaries[str(x)] = {
            "label": style["label"],
            "trajectory_first_beta": beta_summary,
            "fit_to_ensemble_mean_curve": {
                "beta": mean_curve_fit[0],
                "log_amplitude": mean_curve_fit[1],
                "r_squared": mean_curve_fit[2],
            },
        }
        axis.plot(
            log_chord,
            np.log(mean_curve),
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.05,
            markersize=3.2,
            markerfacecolor="white",
            markeredgewidth=0.8,
            label=style["label"],
        )

    left_mean = x_resolved[:, 5, :].mean(axis=0)[1:]
    guide_anchor_index = int(np.flatnonzero(separations == 3)[0])
    guide_x = np.linspace(log_chord[fit_mask][0], log_chord[fit_mask][-1], 100)
    guide_y = (
        np.log(left_mean[guide_anchor_index])
        - 2.0 * (guide_x - log_chord[guide_anchor_index])
        + 0.18
    )
    axis.plot(
        guide_x,
        guide_y,
        color="#555555",
        linestyle=(0, (3, 2)),
        linewidth=0.9,
    )
    axis.text(
        guide_x[-1] - 0.02,
        guide_y[-1] + 0.12,
        "slope $-2$",
        ha="right",
        va="bottom",
        color="#444444",
        fontsize=7.0,
    )
    axis.axvspan(
        log_chord[fit_mask][0],
        log_chord[fit_mask][-1],
        color="#777777",
        alpha=0.07,
        linewidth=0,
        zorder=0,
    )
    axis.set_xlabel(r"$\log d_{N_y}(r_y)$")
    axis.set_ylabel(r"$\log C_G(x,r_y)$")
    axis.set_xlim(log_chord[0] - 0.04, log_chord[-1] + 0.04)
    axis.legend(
        loc="lower left",
        frameon=False,
        ncol=2,
        handlelength=1.8,
        columnspacing=0.9,
    )
    axis.text(
        0.98,
        0.05,
        r"fit: $2\leq r_y\leq8$",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color="#444444",
        fontsize=7.0,
    )
    figure.subplots_adjust(left=0.20, right=0.975, bottom=0.18, top=0.97)
    metadata = {
        "Title": "Hard-wall x-resolved correlators in log-chord coordinates",
        "Author": "class_A_fermionic_adaptive_circuit analysis",
        "Subject": (
            "Nx=20; Ny=32; S=100; t=2Ny; x=5,6,14,15; "
            "trajectory-mean curves; critical slope -2"
        ),
    }
    figure.savefig(OUTPUT_DIR / f"{STEM}.pdf", metadata=metadata)
    figure.savefig(OUTPUT_DIR / f"{STEM}.png", dpi=300, metadata=metadata)
    plt.close(figure)

    curve_csv = OUTPUT_DIR / f"{STEM}_mean_curves.csv"
    with curve_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "x",
                "r_y",
                "chord_distance",
                "log_chord",
                "mean_correlator",
                "log_mean_correlator",
                "trajectory_q25",
                "trajectory_q75",
                "in_fit_window",
            ],
        )
        writer.writeheader()
        for x in SLICES:
            values = x_resolved[:, x, 1:]
            mean_curve = values.mean(axis=0)
            q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
            for index, separation in enumerate(separations):
                writer.writerow(
                    {
                        "x": x,
                        "r_y": int(separation),
                        "chord_distance": f"{distances[index]:.17g}",
                        "log_chord": f"{log_chord[index]:.17g}",
                        "mean_correlator": f"{mean_curve[index]:.17g}",
                        "log_mean_correlator": f"{np.log(mean_curve[index]):.17g}",
                        "trajectory_q25": f"{q25[index]:.17g}",
                        "trajectory_q75": f"{q75[index]:.17g}",
                        "in_fit_window": bool(fit_mask[index]),
                    }
                )

    fit_csv = OUTPUT_DIR / f"{STEM}_trajectory_fits.csv"
    with fit_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["sample_index", "x", "beta", "log_amplitude", "r_squared"],
        )
        writer.writeheader()
        for position, sample_index in enumerate(sample_ids):
            for x in SLICES:
                beta, log_amplitude, r_squared = fit_curve(
                    x_resolved[position, x], NY, FIT_MIN, FIT_MAX
                )
                writer.writerow(
                    {
                        "sample_index": int(sample_index),
                        "x": x,
                        "beta": f"{beta:.17g}",
                        "log_amplitude": f"{log_amplitude:.17g}",
                        "r_squared": f"{r_squared:.17g}",
                    }
                )

    payload = {
        "schema": "hard_wall_x_resolved_log_chord_v1",
        "contract": {
            "construction": "hard/support-terminated",
            "Nx": NX,
            "Ny": NY,
            "alpha_1": 1,
            "alpha_2": 30,
            "nshell": 1,
            "cycle": 2 * NY,
            "ensemble_size": int(sample_ids.size),
            "initialization": "pure/default",
            "sequence": "raster_y",
            "perfect_correction": True,
            "dtype": "complex128",
            "x_columns": list(SLICES),
        },
        "fit": {
            "coordinate": "log[(Ny/pi) sin(pi r_y/Ny)], with log denoting the natural logarithm",
            "window": [FIT_MIN, FIT_MAX],
            "critical_prediction": "slope=-2",
            "estimator_order": "fit each trajectory and x column first, then summarize",
        },
        "results": summaries,
        "input_files": provenance,
        "outputs": [
            f"{STEM}.pdf",
            f"{STEM}.png",
            curve_csv.name,
            fit_csv.name,
            f"{STEM}_summary.json",
        ],
    }
    (OUTPUT_DIR / f"{STEM}_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                x: summaries[str(x)]["trajectory_first_beta"]["mean"]
                for x in SLICES
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
