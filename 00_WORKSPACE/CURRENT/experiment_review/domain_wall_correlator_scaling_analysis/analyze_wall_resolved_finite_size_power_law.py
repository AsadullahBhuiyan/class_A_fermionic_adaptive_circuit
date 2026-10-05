#!/usr/bin/env python3
"""Fit the two hard-wall x-resolved correlators across available sizes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_finite_size_power_law import (
    ENSEMBLE_SIZE,
    NEW_SIZES,
    NX,
    OUTPUT_DIR,
    chord,
    configure_matplotlib,
    distribution_summary,
    fit_curve,
    load_new_size,
)


STEM = "hard_wall_xresolved_finite_size_power_law"
LEFT_WALL = 5
RIGHT_WALL = 15
WALLS = {
    "left_wall": LEFT_WALL,
    "right_wall": RIGHT_WALL,
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wall_curves: dict[int, dict[str, np.ndarray]] = {}
    provenance: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}

    for ny in sorted(NEW_SIZES):
        x_average, x_resolved, sample_ids, files = load_new_size(ny)
        if x_resolved.shape != (ENSEMBLE_SIZE, NX, ny // 2 + 1):
            raise RuntimeError(f"Ny={ny}: unexpected final x-resolved shape")
        if not np.allclose(
            x_average, x_resolved.mean(axis=1), rtol=2e-14, atol=1e-15
        ):
            raise RuntimeError(f"Ny={ny}: x-average identity failed after loading")

        curves = {
            name: x_resolved[:, wall_x, :].copy()
            for name, wall_x in WALLS.items()
        }
        curves["two_wall_average"] = 0.5 * (
            curves["left_wall"] + curves["right_wall"]
        )
        wall_curves[ny] = curves
        provenance.extend(files)

        fit_min, fit_max = 2, ny // 4
        summaries[str(ny)] = {
            "fit_window": [fit_min, fit_max],
            "walls": [LEFT_WALL, RIGHT_WALL],
        }
        fits_for_size: dict[str, np.ndarray] = {}
        for name, values in curves.items():
            records = np.asarray(
                [fit_curve(curve, ny, fit_min, fit_max) for curve in values],
                dtype=np.float64,
            )
            fits_for_size[name] = records
            mean_curve_fit = fit_curve(values.mean(axis=0), ny, fit_min, fit_max)
            summaries[str(ny)][name] = {
                "trajectory_first_beta": distribution_summary(records[:, 0]),
                "trajectory_fit_r_squared": distribution_summary(records[:, 2]),
                "fit_to_ensemble_mean_curve": {
                    "beta": mean_curve_fit[0],
                    "log_amplitude": mean_curve_fit[1],
                    "r_squared": mean_curve_fit[2],
                },
            }

        for position, sample_index in enumerate(sample_ids):
            row: dict[str, object] = {
                "Ny": ny,
                "sample_index": int(sample_index),
                "fit_min": fit_min,
                "fit_max": fit_max,
            }
            for name, records in fits_for_size.items():
                row[f"beta_{name}"] = float(records[position, 0])
                row[f"r_squared_{name}"] = float(records[position, 2])
            trajectory_rows.append(row)

    trajectory_csv = OUTPUT_DIR / f"{STEM}_trajectory_fits.csv"
    fieldnames = [
        "Ny",
        "sample_index",
        "fit_min",
        "fit_max",
        "beta_left_wall",
        "r_squared_left_wall",
        "beta_right_wall",
        "r_squared_right_wall",
        "beta_two_wall_average",
        "r_squared_two_wall_average",
    ]
    with trajectory_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in trajectory_rows:
            writer.writerow(
                {
                    key: f"{value:.17g}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )

    size_csv = OUTPUT_DIR / f"{STEM}_size_summary.csv"
    with size_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "Ny",
                "curve",
                "fit_min",
                "fit_max",
                "beta_mean",
                "beta_median",
                "beta_q25",
                "beta_q75",
                "ensemble_mean_curve_beta",
                "ensemble_mean_curve_r_squared",
            ],
        )
        writer.writeheader()
        for ny in sorted(NEW_SIZES):
            record = summaries[str(ny)]
            for name in ("left_wall", "right_wall", "two_wall_average"):
                summary = record[name]["trajectory_first_beta"]
                mean_fit = record[name]["fit_to_ensemble_mean_curve"]
                writer.writerow(
                    {
                        "Ny": ny,
                        "curve": name,
                        "fit_min": record["fit_window"][0],
                        "fit_max": record["fit_window"][1],
                        "beta_mean": f"{summary['mean']:.17g}",
                        "beta_median": f"{summary['median']:.17g}",
                        "beta_q25": f"{summary['interquartile_range'][0]:.17g}",
                        "beta_q75": f"{summary['interquartile_range'][1]:.17g}",
                        "ensemble_mean_curve_beta": f"{mean_fit['beta']:.17g}",
                        "ensemble_mean_curve_r_squared": f"{mean_fit['r_squared']:.17g}",
                    }
                )

    configure_matplotlib()
    figure, (curve_axis, exponent_axis) = plt.subplots(
        2, 1, figsize=(3.375, 5.25), gridspec_kw={"hspace": 0.34}
    )
    sizes = np.asarray(sorted(NEW_SIZES), dtype=np.int64)
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, sizes.size))

    for color, ny in zip(colors, sizes):
        separations = np.arange(1, ny // 2 + 1)
        distance = chord(int(ny), separations)
        mean_curve = wall_curves[int(ny)]["two_wall_average"].mean(axis=0)[1:]
        curve_axis.plot(
            distance,
            mean_curve,
            color=color,
            marker="o",
            markerfacecolor="white",
            markeredgewidth=0.8,
            markersize=3.0,
            linewidth=1.0,
            label=rf"${ny}$",
        )
    guide_x = np.geomspace(2.0, 8.0, 100)
    anchor = chord(32, np.asarray([3]))[0]
    anchor_value = wall_curves[32]["two_wall_average"].mean(axis=0)[3]
    curve_axis.plot(
        guide_x,
        1.25 * anchor_value * (guide_x / anchor) ** -2,
        color="#555555",
        linestyle="--",
        linewidth=0.9,
    )
    curve_axis.text(
        0.96,
        0.90,
        r"$d^{-2}$",
        transform=curve_axis.transAxes,
        ha="right",
        va="top",
        color="#444444",
    )
    curve_axis.set_xscale("log")
    curve_axis.set_yscale("log")
    curve_axis.set_xlabel(r"chord distance $d_{N_y}(r_y)$")
    curve_axis.set_ylabel(r"two-wall mean $C_G^{\rm wall}(r_y)$")
    curve_axis.legend(
        loc="lower left",
        frameon=False,
        ncol=3,
        title=r"$N_y$",
        handlelength=1.5,
        handletextpad=0.35,
        columnspacing=0.8,
    )

    curve_styles = {
        "left_wall": ("#0072B2", "o", r"$x_L$"),
        "right_wall": ("#D55E00", "s", r"$x_R$"),
        "two_wall_average": ("#009E73", "D", "walls"),
    }
    inverse_sizes = 1.0 / sizes.astype(np.float64)
    for name, (color, marker, label) in curve_styles.items():
        means = np.asarray(
            [
                summaries[str(int(ny))][name]["trajectory_first_beta"]["mean"]
                for ny in sizes
            ],
            dtype=np.float64,
        )
        q25 = np.asarray(
            [
                summaries[str(int(ny))][name]["trajectory_first_beta"][
                    "interquartile_range"
                ][0]
                for ny in sizes
            ],
            dtype=np.float64,
        )
        q75 = np.asarray(
            [
                summaries[str(int(ny))][name]["trajectory_first_beta"][
                    "interquartile_range"
                ][1]
                for ny in sizes
            ],
            dtype=np.float64,
        )
        order = np.argsort(inverse_sizes)
        exponent_axis.errorbar(
            inverse_sizes[order],
            means[order],
            yerr=np.vstack((means[order] - q25[order], q75[order] - means[order])),
            color=color,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=0.9,
            markersize=4.2,
            linestyle="none",
            linewidth=0.9,
            elinewidth=0.75,
            capsize=2.0,
            label=label,
        )
    exponent_axis.axhline(2.0, color="#555555", linestyle="--", linewidth=0.9)
    exponent_axis.text(
        0.03,
        0.055,
        r"$\beta=2$",
        transform=exponent_axis.transAxes,
        ha="left",
        va="bottom",
        color="#444444",
    )
    exponent_axis.set_xticks([1.0 / 32.0, 1.0 / 28.0, 1.0 / 24.0])
    exponent_axis.set_xticklabels([r"$1/32$", r"$1/28$", r"$1/24$"])
    exponent_axis.minorticks_off()
    exponent_axis.set_xlabel(r"$1/N_y$")
    exponent_axis.set_ylabel(r"$\langle\beta_\xi\rangle$")
    exponent_axis.legend(loc="upper left", frameon=False, ncol=3)

    for label, axis in zip(("(a)", "(b)"), (curve_axis, exponent_axis)):
        axis.text(-0.18, 1.035, label, transform=axis.transAxes, ha="left", va="bottom")
    figure.subplots_adjust(left=0.20, right=0.975, bottom=0.09, top=0.975)
    metadata = {
        "Title": "Finite-size hard-wall x-resolved wall correlator analysis",
        "Author": "class_A_fermionic_adaptive_circuit analysis",
        "Subject": (
            "Nx=20; Ny=24,28,32; S=100 each; xL=5 and xR=15; "
            "trajectory-first chord fits; IQR bars"
        ),
    }
    figure.savefig(OUTPUT_DIR / f"{STEM}.pdf", metadata=metadata)
    figure.savefig(OUTPUT_DIR / f"{STEM}.png", dpi=300, metadata=metadata)
    plt.close(figure)

    payload = {
        "schema": "hard_wall_xresolved_finite_size_power_law_v1",
        "contract": {
            "construction": "hard/support-terminated",
            "Nx": NX,
            "Ny": [int(value) for value in sizes],
            "alpha_1": 1,
            "alpha_2": 30,
            "nshell": 1,
            "cycle": "2Ny",
            "ensemble_size_per_size": ENSEMBLE_SIZE,
            "initialization": "pure/default",
            "sequence": "raster_y",
            "perfect_correction": True,
            "dtype": "complex128",
            "walls": [LEFT_WALL, RIGHT_WALL],
        },
        "fit": {
            "model": "log C_G(x_wall,r_y) = log A - beta log d_Ny(r_y)",
            "window": "2 <= r_y <= floor(Ny/4)",
            "estimator_order": "fit each trajectory and wall first, then summarize trajectories",
            "uncertainty_display": "trajectory IQR only; no SEM or confidence interval on the mean",
        },
        "results": summaries,
        "input_files": provenance,
        "outputs": [
            f"{STEM}.pdf",
            f"{STEM}.png",
            trajectory_csv.name,
            size_csv.name,
            f"{STEM}_summary.json",
        ],
        "limitation": (
            "Only Ny=24,28,32 retain x-resolved S100 arrays; the matched "
            "Ny=30,40,50 legacy files retain only the x average."
        ),
    }
    (OUTPUT_DIR / f"{STEM}_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    concise = {
        ny: {
            name: summaries[ny][name]["trajectory_first_beta"]["mean"]
            for name in ("left_wall", "right_wall", "two_wall_average")
        }
        for ny in summaries
    }
    print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
