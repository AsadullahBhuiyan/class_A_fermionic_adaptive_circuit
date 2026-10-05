#!/usr/bin/env python3
"""Extract trajectory-first hard-wall correlator exponents at final time."""

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
    load_trajectories,
)


STEM = "hard_wall_xavg_power_law_exponents"
NX = 20
NY = 32
LEFT_WALL = 5
RIGHT_WALL = 15


def fit_log_power_law(
    curve: np.ndarray, coordinate: np.ndarray, mask: np.ndarray
) -> tuple[float, float, float]:
    """Return beta, log-amplitude, and R^2 for G=A*coordinate**(-beta)."""
    selected = np.asarray(curve, dtype=np.float64)[mask]
    if np.any(~np.isfinite(selected)) or np.any(selected <= 0.0):
        raise RuntimeError("Power-law fit received a nonpositive or nonfinite value")
    x = np.log(np.asarray(coordinate, dtype=np.float64)[mask])
    y = np.log(selected)
    slope, intercept = np.polyfit(x, y, deg=1)
    residual = y - (slope * x + intercept)
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / denominator
    return float(-slope), float(intercept), float(r_squared)


def summarize(values: np.ndarray) -> dict[str, float | list[float]]:
    values = np.asarray(values, dtype=np.float64)
    central_68 = np.quantile(values, [0.16, 0.84])
    central_95 = np.quantile(values, [0.025, 0.975])
    quartiles = np.quantile(values, [0.25, 0.75])
    return {
        "mean": float(np.mean(values)),
        "sample_standard_deviation": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "interquartile_range": [float(quartiles[0]), float(quartiles[1])],
        "central_68_percent_trajectory_range": [float(central_68[0]), float(central_68[1])],
        "central_95_percent_trajectory_range": [float(central_95[0]), float(central_95[1])],
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, provenance = load_trajectories()
    sample_ids = np.asarray([row["sample_index"] for row in rows], dtype=np.int64)
    ry = np.asarray(rows[0]["ry"], dtype=np.int64)
    fit_mask = (ry >= FIT_MIN) & (ry <= FIT_MAX)
    raw_coordinate = ry.astype(np.float64)
    chord_coordinate = (NY / np.pi) * np.sin(np.pi * ry / NY)
    x_resolved = np.stack(
        [np.asarray(row["x_resolved"], dtype=np.float64) for row in rows]
    )
    x_average = np.stack([np.asarray(row["xavg"], dtype=np.float64) for row in rows])
    if not np.allclose(x_average, x_resolved.mean(axis=1), rtol=2e-14, atol=1e-15):
        raise RuntimeError("Archived x-average does not equal the mean of x-resolved data")

    curves = {
        "x_average": x_average,
        "two_wall_average": 0.5
        * (x_resolved[:, LEFT_WALL, :] + x_resolved[:, RIGHT_WALL, :]),
        "left_wall": x_resolved[:, LEFT_WALL, :],
        "right_wall": x_resolved[:, RIGHT_WALL, :],
    }
    fitted: dict[str, dict[str, np.ndarray]] = {}
    for curve_name, values in curves.items():
        fitted[curve_name] = {}
        for coordinate_name, coordinate in (
            ("raw_distance", raw_coordinate),
            ("cylinder_chord", chord_coordinate),
        ):
            records = np.asarray(
                [fit_log_power_law(curve, coordinate, fit_mask) for curve in values],
                dtype=np.float64,
            )
            fitted[curve_name][f"beta_{coordinate_name}"] = records[:, 0]
            fitted[curve_name][f"log_amplitude_{coordinate_name}"] = records[:, 1]
            fitted[curve_name][f"r_squared_{coordinate_name}"] = records[:, 2]

    summaries: dict[str, dict[str, object]] = {}
    for curve_name, values in curves.items():
        summaries[curve_name] = {}
        for coordinate_name, coordinate in (
            ("raw_distance", raw_coordinate),
            ("cylinder_chord", chord_coordinate),
        ):
            beta_key = f"beta_{coordinate_name}"
            r2_key = f"r_squared_{coordinate_name}"
            mean_curve_fit = fit_log_power_law(values.mean(axis=0), coordinate, fit_mask)
            summaries[curve_name][coordinate_name] = {
                "trajectory_first_beta": summarize(fitted[curve_name][beta_key]),
                "trajectory_fit_r_squared": summarize(fitted[curve_name][r2_key]),
                "fit_to_ensemble_mean_curve": {
                    "beta": mean_curve_fit[0],
                    "log_amplitude": mean_curve_fit[1],
                    "r_squared": mean_curve_fit[2],
                },
            }

    csv_path = OUTPUT_DIR / f"{STEM}_trajectory_fits.csv"
    fieldnames = ["sample_index"]
    for curve_name in curves:
        for coordinate_name in ("raw_distance", "cylinder_chord"):
            fieldnames.extend(
                [
                    f"{curve_name}_beta_{coordinate_name}",
                    f"{curve_name}_r_squared_{coordinate_name}",
                    f"{curve_name}_log_amplitude_{coordinate_name}",
                ]
            )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for position, sample_index in enumerate(sample_ids):
            row: dict[str, int | str] = {"sample_index": int(sample_index)}
            for curve_name in curves:
                for coordinate_name in ("raw_distance", "cylinder_chord"):
                    for metric in ("beta", "r_squared", "log_amplitude"):
                        key = f"{metric}_{coordinate_name}"
                        row[f"{curve_name}_{key}"] = (
                            f"{float(fitted[curve_name][key][position]):.17g}"
                        )
            writer.writerow(row)

    mean_curve = x_average.mean(axis=0)
    curve_q16, curve_q84 = np.quantile(x_average, [0.16, 0.84], axis=0)
    ensemble_fit = summaries["x_average"]["raw_distance"]["fit_to_ensemble_mean_curve"]
    fit_amplitude = float(np.exp(float(ensemble_fit["log_amplitude"])))
    fit_beta = float(ensemble_fit["beta"])
    betas = fitted["x_average"]["beta_raw_distance"]
    beta_summary = summaries["x_average"]["raw_distance"]["trajectory_first_beta"]

    configure_matplotlib()
    figure, (curve_axis, distribution_axis) = plt.subplots(
        1, 2, figsize=(7.0, 2.55), gridspec_kw={"wspace": 0.34}
    )
    shown = ry >= 1
    curve_axis.fill_between(
        ry[shown],
        curve_q16[shown],
        curve_q84[shown],
        color="#0072B2",
        alpha=0.18,
        linewidth=0,
        label="central 68% of trajectories",
    )
    curve_axis.plot(
        ry[shown],
        mean_curve[shown],
        color="#0072B2",
        marker="o",
        markerfacecolor="white",
        markeredgewidth=0.8,
        markersize=3.2,
        linewidth=1.0,
        label="trajectory mean",
    )
    fit_x = np.linspace(FIT_MIN, FIT_MAX, 200)
    curve_axis.plot(
        fit_x,
        fit_amplitude * fit_x ** (-fit_beta),
        color="#D55E00",
        linestyle="--",
        linewidth=1.2,
        label=rf"fit: $\beta={fit_beta:.3f}$",
    )
    curve_axis.axvspan(FIT_MIN, FIT_MAX, color="#777777", alpha=0.09, linewidth=0)
    curve_axis.set_xscale("log")
    curve_axis.set_yscale("log")
    curve_axis.set_xlim(0.9, 17.2)
    curve_axis.set_xlabel(r"separation $r_y$")
    curve_axis.set_ylabel(r"$x$-averaged squared correlator $\overline{G}(r_y)$")
    curve_axis.legend(loc="lower left", frameon=False, handlelength=2.2)
    curve_axis.text(-0.16, 1.04, "(a)", transform=curve_axis.transAxes)

    bins = np.linspace(float(np.min(betas)) - 0.015, float(np.max(betas)) + 0.015, 15)
    distribution_axis.hist(
        betas,
        bins=bins,
        color="#56B4E9",
        edgecolor="#1F1F1F",
        linewidth=0.55,
        alpha=0.85,
    )
    beta_mean = float(beta_summary["mean"])
    beta_sd = float(beta_summary["sample_standard_deviation"])
    distribution_axis.axvline(beta_mean, color="#D55E00", linewidth=1.25)
    distribution_axis.axvspan(
        beta_mean - beta_sd,
        beta_mean + beta_sd,
        color="#D55E00",
        alpha=0.16,
        linewidth=0,
    )
    distribution_axis.text(
        0.04,
        0.96,
        rf"mean $={beta_mean:.3f}$" + "\n" + rf"sample SD $={beta_sd:.3f}$",
        transform=distribution_axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.1,
    )
    distribution_axis.set_xlabel(r"trajectory exponent $\beta_\xi$")
    distribution_axis.set_ylabel("trajectory count")
    distribution_axis.text(-0.16, 1.04, "(b)", transform=distribution_axis.transAxes)
    figure.suptitle(
        r"hard wall, $20\times32$, $t=2N_y$, $S=100$; fit window $2\leq r_y\leq8$",
        y=0.995,
        fontsize=8.5,
    )
    figure.subplots_adjust(left=0.095, right=0.985, bottom=0.19, top=0.86)
    metadata = {
        "Title": "Trajectory-first hard-wall x-averaged correlator exponents",
        "Author": "class_A_fermionic_adaptive_circuit analysis",
        "Subject": (
            f"S=100; raw-distance fit r_y={FIT_MIN}..{FIT_MAX}; "
            f"mean beta={beta_mean:.10f}; sample SD={beta_sd:.10f}"
        ),
    }
    figure.savefig(OUTPUT_DIR / f"{STEM}.pdf", metadata=metadata)
    figure.savefig(OUTPUT_DIR / f"{STEM}.png", dpi=300, metadata=metadata)
    plt.close(figure)

    source_path = OUTPUT_DIR / f"{STEM}_mean_curve.csv"
    with source_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "r_y",
                "mean_xavg_correlator",
                "trajectory_q16_xavg_correlator",
                "trajectory_q84_xavg_correlator",
                "in_fit_window",
            ],
        )
        writer.writeheader()
        for index, separation in enumerate(ry):
            writer.writerow(
                {
                    "r_y": int(separation),
                    "mean_xavg_correlator": f"{float(mean_curve[index]):.17g}",
                    "trajectory_q16_xavg_correlator": f"{float(curve_q16[index]):.17g}",
                    "trajectory_q84_xavg_correlator": f"{float(curve_q84[index]):.17g}",
                    "in_fit_window": bool(fit_mask[index]),
                }
            )

    summary = {
        "schema": "hard_wall_xavg_power_law_exponents_v2",
        "contract": {
            "construction": "hard/support-terminated",
            "Nx": NX,
            "Ny": NY,
            "alpha_1": 1,
            "alpha_2": 30,
            "nshell": 1,
            "cycle": 64,
            "ensemble_size": 100,
            "initialization": "pure/default",
            "sequence": "raster_y",
            "perfect_correction": True,
            "dtype": "complex128",
            "walls": [LEFT_WALL, RIGHT_WALL],
        },
        "fit": {
            "window": [FIT_MIN, FIT_MAX],
            "primary_coordinate": "raw r_y, matching the legacy correlator fit convention",
            "primary_model": "log G_xavg = log A - beta log(r_y)",
            "estimator_order": "fit each trajectory first, then take the equal-weight trajectory mean",
            "uncertainty_display": "empirical trajectory spread only; no SEM and no confidence interval on the mean",
            "cylinder_chord_definition": "d_N(r)=(N_y/pi) sin(pi r/N_y)",
        },
        "results": summaries,
        "mixture_formalism": {
            "x_resolved": "G_x(r) ~= [A_L w_L(x)+A_R w_R(x)] d_N(r)^(-beta) + B_x P_N(r;xi_x) + floor_x",
            "periodic_exponential": "P_N(r;xi)=[exp(-r/xi)+exp(-(N_y-r)/xi)]/[1-exp(-N_y/xi)]",
            "x_averaged": "Gbar(r)=A_eff d_N(r)^(-beta)+(1/Nx) sum_x B_x P_N(r;xi_x)+floor",
            "interpretation": "the exponential controls sufficiently short separations, while any nonzero boundary amplitude dominates asymptotically over an exponential",
        },
        "input_files": provenance,
        "outputs": [
            f"{STEM}.pdf",
            f"{STEM}.png",
            csv_path.name,
            source_path.name,
            f"{STEM}_summary.json",
        ],
    }
    (OUTPUT_DIR / f"{STEM}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["results"]["x_average"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
