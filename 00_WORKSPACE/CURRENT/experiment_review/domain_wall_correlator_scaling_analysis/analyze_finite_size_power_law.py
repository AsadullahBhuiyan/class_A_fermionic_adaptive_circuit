#!/usr/bin/env python3
"""Finite-size power-law analysis of matched hard-wall S=100 ensembles."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
ANALYSIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ANALYSIS_DIR / "outputs"
STEM = "hard_wall_finite_size_power_law"
SIZES = (24, 28, 30, 32, 40, 50)
NEW_SIZES = frozenset((24, 28, 32))
NX = 20
ENSEMBLE_SIZE = 100
NEW_ROOT = (
    REPO_ROOT
    / "00_WORKSPACE/CURRENT/final_production_new_designs"
    / "08_domain_wall_correlator_scaling/gpu_data"
    / "domain_wall_correlator_nx20_ny24-32_a1-1-3_nsh1-2-dense_s100_2ny_raster_v1"
    / "results/hard"
)
LEGACY_ROOT = (
    REPO_ROOT
    / "00_WORKSPACE/COLAB/colab_charge_fluctuations/gpu_data"
    / "streaming_covariance_observables/campaigns"
    / "N20_selected_nsh1_dwtrunc1_init-default_S100_cycles-2Ny/runs"
)
EXPECTED_LEGACY_FORMULA = (
    "C=(G+I)/2; xavg_corr(ry)=(1/(2*Nx*Ny))*sum_x,y,mu,nu "
    "|C[x,y,mu;x,y+ry,nu]|^2"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def require_scalar(archive: Any, key: str, expected: Any) -> None:
    if key not in archive.files:
        raise RuntimeError(f"result is missing scalar {key!r}")
    actual = np.asarray(archive[key]).item()
    if isinstance(expected, float):
        matches = float(actual) == expected
    else:
        matches = actual == expected
    if not matches:
        raise RuntimeError(f"result scalar {key!r}: expected {expected!r}, got {actual!r}")


def load_new_size(
    ny: int,
    *, alpha_1: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    run_dir = NEW_ROOT / f"Ny{ny:03d}/alpha1_{alpha_1:g}/nshell_1"
    results = sorted(run_dir.glob("batch_*.npz"))
    completions = sorted(run_dir.glob("batch_*.complete.json"))
    if len(results) != 4 or len(completions) != 4:
        raise RuntimeError(f"Ny={ny}: expected four result/completion pairs")

    completion_by_result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for completion_path in completions:
        payload = json.loads(completion_path.read_text(encoding="utf-8"))
        filename = str(payload.get("result_filename", ""))
        if not filename or filename in completion_by_result:
            raise RuntimeError(f"Ny={ny}: duplicate or missing result filename")
        completion_by_result[filename] = (completion_path, payload)

    curves: list[np.ndarray] = []
    x_resolved_curves: list[np.ndarray] = []
    sample_ids: list[np.ndarray] = []
    provenance: list[dict[str, Any]] = []
    for result_path in results:
        if result_path.name not in completion_by_result:
            raise RuntimeError(f"Ny={ny}: {result_path.name} has no completion JSON")
        completion_path, completion = completion_by_result[result_path.name]
        actual_size = result_path.stat().st_size
        actual_hash = sha256(result_path)
        if completion.get("status") != "complete":
            raise RuntimeError(f"Ny={ny}: completion status is not complete")
        if int(completion.get("result_bytes", -1)) != actual_size:
            raise RuntimeError(f"Ny={ny}: result byte count mismatch")
        if completion.get("result_sha256") != actual_hash:
            raise RuntimeError(f"Ny={ny}: result checksum mismatch")
        completion_expectations = {
            "Nx": NX,
            "Ny": ny,
            "alpha_1": float(alpha_1),
            "alpha_2": 30.0,
            "nshell": 1,
            "construction": "hard",
            "cycles": 2 * ny,
            "canonical_entry_point": "classA_U1FGTN_gpu.run_markov_circuit",
        }
        for key, expected in completion_expectations.items():
            if completion.get(key) != expected:
                raise RuntimeError(
                    f"Ny={ny}: completion {key!r} differs from {expected!r}"
                )

        with np.load(result_path, allow_pickle=False) as archive:
            for key, expected in {
                "Nx": NX,
                "Ny": ny,
                "alpha_1": float(alpha_1),
                "alpha_2": 30.0,
                "nshell": 1,
                "construction": "hard",
                "dw_truncation": True,
                "meas_slab_only": True,
                "dtype": "complex128",
                "init_mode": "default",
                "sequence": "raster_y",
                "perfect_correction": True,
                "canonical_entry_point": "classA_U1FGTN_gpu.run_markov_circuit",
            }.items():
                require_scalar(archive, key, expected)
            cycles = np.asarray(archive["cycles"], dtype=np.int64)
            ry = np.asarray(archive["ry_values"], dtype=np.int64)
            ids = np.asarray(archive["global_sample_indices"], dtype=np.int64)
            x_resolved = np.asarray(
                archive["x_resolved_square_correlator"], dtype=np.float64
            )
            x_average = np.asarray(
                archive["xavg_square_correlator_vs_ry"], dtype=np.float64
            )
            if not np.array_equal(cycles, np.arange(2 * ny + 1)):
                raise RuntimeError(f"Ny={ny}: every-cycle labels are incomplete")
            if not np.array_equal(ry, np.arange(ny // 2 + 1)):
                raise RuntimeError(f"Ny={ny}: separation labels are incomplete")
            if x_resolved.shape != (25, 2 * ny + 1, NX, ny // 2 + 1):
                raise RuntimeError(f"Ny={ny}: unexpected x-resolved shape")
            if x_average.shape != (25, 2 * ny + 1, ny // 2 + 1):
                raise RuntimeError(f"Ny={ny}: unexpected x-average shape")
            if not np.allclose(
                x_average, x_resolved.mean(axis=2), rtol=2e-14, atol=1e-15
            ):
                raise RuntimeError(f"Ny={ny}: x-average identity failed")
            curves.append(x_average[:, -1, :].copy())
            x_resolved_curves.append(x_resolved[:, -1, :, :].copy())
            sample_ids.append(ids.copy())
        provenance.extend(
            [
                {
                    "path": relative(result_path),
                    "bytes": actual_size,
                    "sha256": actual_hash,
                    "role": "new_result",
                },
                {
                    "path": relative(completion_path),
                    "bytes": completion_path.stat().st_size,
                    "sha256": sha256(completion_path),
                    "role": "new_completion",
                },
            ]
        )

    all_curves = np.concatenate(curves, axis=0)
    all_x_resolved = np.concatenate(x_resolved_curves, axis=0)
    all_ids = np.concatenate(sample_ids)
    order = np.argsort(all_ids)
    all_curves = all_curves[order]
    all_x_resolved = all_x_resolved[order]
    all_ids = all_ids[order]
    if not np.array_equal(all_ids, np.arange(ENSEMBLE_SIZE)):
        raise RuntimeError(f"Ny={ny}: expected sample indices 0,...,99")
    return all_curves, all_x_resolved, all_ids, provenance


def load_legacy_size(ny: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    run_dir = LEGACY_ROOT / f"N20x{ny}_nsh1_init-default_perfect_correction"
    result_path = run_dir / "xavg_square_correlator_vs_ry.npz"
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key, expected in {
        "Nx": NX,
        "Ny": ny,
        "cycles": 2 * ny,
        "samples_requested": ENSEMBLE_SIZE,
        "dtype": "complex128",
        "sequence": "raster_y",
        "nshell": 1,
        "dw_truncation": True,
        "protocol": "perfect_correction",
        "perfect_correction": True,
        "init_mode": "default",
        "canonical_dynamics_entry_point": "classA_U1FGTN_gpu.run_markov_circuit",
    }.items():
        if summary.get(key) != expected:
            raise RuntimeError(f"Ny={ny}: legacy summary {key!r} mismatch")

    with np.load(result_path, allow_pickle=False) as archive:
        config = json.loads(str(np.asarray(archive["config_json"]).item()))
        for key, expected in {
            "Nx": NX,
            "Ny": ny,
            "alpha_1": 1.0,
            "alpha_2": 30.0,
            "cycles": 2 * ny,
            "samples_actual": ENSEMBLE_SIZE,
            "dtype": "complex128",
            "sequence": "raster_y",
            "nshell": 1,
            "dw_truncation": True,
            "perfect_correction": True,
            "postselect": False,
            "init_mode": "default",
            "canonical_dynamics_entry_point": "classA_U1FGTN_gpu.run_markov_circuit",
        }.items():
            if config.get(key) != expected:
                raise RuntimeError(f"Ny={ny}: legacy config {key!r} mismatch")
        formula = str(np.asarray(archive["formula"]).item())
        if formula != EXPECTED_LEGACY_FORMULA:
            raise RuntimeError(f"Ny={ny}: unexpected legacy correlator formula")
        cycles = np.asarray(archive["cycle_labels"], dtype=np.int64)
        ids = np.asarray(archive["sample_indices"], dtype=np.int64)
        ry = np.asarray(archive["ry_values"], dtype=np.int64)
        history = np.asarray(archive["xavg_square_correlator_vs_ry"], dtype=np.float64)
        if not np.array_equal(cycles, np.arange(1, 2 * ny + 1)):
            raise RuntimeError(f"Ny={ny}: legacy cycle labels are incomplete")
        if not np.array_equal(ids, np.arange(ENSEMBLE_SIZE)):
            raise RuntimeError(f"Ny={ny}: legacy sample indices are incomplete")
        if not np.array_equal(ry, np.arange(ny // 2 + 1)):
            raise RuntimeError(f"Ny={ny}: legacy separations are incomplete")
        if history.shape != (ENSEMBLE_SIZE, 2 * ny, ny // 2 + 1):
            raise RuntimeError(f"Ny={ny}: unexpected legacy correlator shape")
        curves = history[:, -1, :].copy()

    provenance = [
        {
            "path": relative(result_path),
            "bytes": result_path.stat().st_size,
            "sha256": sha256(result_path),
            "role": "legacy_result",
        },
        {
            "path": relative(summary_path),
            "bytes": summary_path.stat().st_size,
            "sha256": sha256(summary_path),
            "role": "legacy_summary",
        },
    ]
    return curves, ids, provenance


def load_all() -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    curves: dict[int, np.ndarray] = {}
    provenance: list[dict[str, Any]] = []
    for ny in SIZES:
        if ny in NEW_SIZES:
            values, _, _, files = load_new_size(ny)
        else:
            values, _, files = load_legacy_size(ny)
        if values.shape != (ENSEMBLE_SIZE, ny // 2 + 1):
            raise RuntimeError(f"Ny={ny}: final curve shape mismatch")
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise RuntimeError(f"Ny={ny}: correlator contains invalid values")
        curves[ny] = values
        provenance.extend(files)
    return curves, provenance


def chord(ny: int, separations: np.ndarray) -> np.ndarray:
    return (ny / np.pi) * np.sin(np.pi * separations / ny)


def fit_curve(
    curve: np.ndarray, ny: int, minimum: int, maximum: int
) -> tuple[float, float, float]:
    separations = np.arange(curve.size, dtype=np.float64)
    mask = (separations >= minimum) & (separations <= maximum)
    x = np.log(chord(ny, separations[mask]))
    y = np.log(curve[mask])
    slope, intercept = np.polyfit(x, y, deg=1)
    prediction = slope * x + intercept
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - float(np.sum((y - prediction) ** 2)) / denominator
    return float(-slope), float(intercept), float(r_squared)


def distribution_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    q025, q16, q25, q50, q75, q84, q975 = np.quantile(
        values, [0.025, 0.16, 0.25, 0.5, 0.75, 0.84, 0.975]
    )
    return {
        "mean": float(np.mean(values)),
        "sample_standard_deviation": float(np.std(values, ddof=1)),
        "median": float(q50),
        "interquartile_range": [float(q25), float(q75)],
        "central_68_percent_trajectory_range": [float(q16), float(q84)],
        "central_95_percent_trajectory_range": [float(q025), float(q975)],
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "mathtext.fontset": "cm",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.bbox": "tight",
        }
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curves, provenance = load_all()
    window_definitions = {
        "primary_scaled_quarter": lambda ny: (2, ny // 4),
        "fixed_short": lambda ny: (2, min(8, ny // 2)),
        "full_half_cylinder": lambda ny: (2, ny // 2),
        "tail_half_cylinder": lambda ny: (5, ny // 2),
    }

    fits: dict[str, dict[int, dict[str, Any]]] = {
        name: {} for name in window_definitions
    }
    trajectory_rows: list[dict[str, Any]] = []
    for ny in SIZES:
        for window_name, window_function in window_definitions.items():
            minimum, maximum = window_function(ny)
            records = np.asarray(
                [fit_curve(curve, ny, minimum, maximum) for curve in curves[ny]],
                dtype=np.float64,
            )
            mean_curve_fit = fit_curve(curves[ny].mean(axis=0), ny, minimum, maximum)
            fits[window_name][ny] = {
                "window": [minimum, maximum],
                "trajectory_beta": records[:, 0],
                "trajectory_r_squared": records[:, 2],
                "trajectory_beta_summary": distribution_summary(records[:, 0]),
                "trajectory_r_squared_summary": distribution_summary(records[:, 2]),
                "ensemble_mean_curve_fit": {
                    "beta": mean_curve_fit[0],
                    "log_amplitude": mean_curve_fit[1],
                    "r_squared": mean_curve_fit[2],
                },
            }
        for sample_index in range(ENSEMBLE_SIZE):
            row: dict[str, Any] = {
                "Ny": ny,
                "sample_index": sample_index,
                "source_cohort": "new_x_resolved" if ny in NEW_SIZES else "legacy_streaming",
            }
            for window_name in window_definitions:
                row[f"beta_{window_name}"] = float(
                    fits[window_name][ny]["trajectory_beta"][sample_index]
                )
                row[f"r_squared_{window_name}"] = float(
                    fits[window_name][ny]["trajectory_r_squared"][sample_index]
                )
            trajectory_rows.append(row)

    primary_means = np.asarray(
        [fits["primary_scaled_quarter"][ny]["trajectory_beta_summary"]["mean"] for ny in SIZES]
    )
    inverse_sizes = 1.0 / np.asarray(SIZES, dtype=np.float64)
    inverse_square_sizes = inverse_sizes**2
    regression_summaries: dict[str, Any] = {}
    for name, x in (("linear_1_over_Ny", inverse_sizes), ("linear_1_over_Ny_squared", inverse_square_sizes)):
        slope, intercept = np.polyfit(x, primary_means, deg=1)
        prediction = slope * x + intercept
        r_squared = 1.0 - float(np.sum((primary_means - prediction) ** 2)) / float(
            np.sum((primary_means - np.mean(primary_means)) ** 2)
        )
        regression_summaries[name] = {
            "beta_infinite_intercept": float(intercept),
            "slope": float(slope),
            "r_squared": r_squared,
            "status": "descriptive_only_no_uncertainty_claim",
        }

    trajectory_csv = OUTPUT_DIR / f"{STEM}_trajectory_fits.csv"
    with trajectory_csv.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["Ny", "sample_index", "source_cohort"]
        for name in window_definitions:
            fieldnames.extend([f"beta_{name}", f"r_squared_{name}"])
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
        fieldnames = [
            "Ny",
            "source_cohort",
            "fit_min",
            "fit_max",
            "beta_mean",
            "beta_sample_sd",
            "beta_median",
            "beta_q25",
            "beta_q75",
            "ensemble_mean_curve_beta",
            "ensemble_mean_curve_r_squared",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for ny in SIZES:
            record = fits["primary_scaled_quarter"][ny]
            summary = record["trajectory_beta_summary"]
            mean_fit = record["ensemble_mean_curve_fit"]
            writer.writerow(
                {
                    "Ny": ny,
                    "source_cohort": "new_x_resolved" if ny in NEW_SIZES else "legacy_streaming",
                    "fit_min": record["window"][0],
                    "fit_max": record["window"][1],
                    "beta_mean": f"{summary['mean']:.17g}",
                    "beta_sample_sd": f"{summary['sample_standard_deviation']:.17g}",
                    "beta_median": f"{summary['median']:.17g}",
                    "beta_q25": f"{summary['interquartile_range'][0]:.17g}",
                    "beta_q75": f"{summary['interquartile_range'][1]:.17g}",
                    "ensemble_mean_curve_beta": f"{mean_fit['beta']:.17g}",
                    "ensemble_mean_curve_r_squared": f"{mean_fit['r_squared']:.17g}",
                }
            )

    curve_csv = OUTPUT_DIR / f"{STEM}_mean_curves.csv"
    with curve_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "Ny",
                "r_y",
                "chord_distance",
                "mean_correlator",
                "trajectory_q16",
                "trajectory_q84",
                "in_primary_window",
            ],
        )
        writer.writeheader()
        for ny in SIZES:
            minimum, maximum = window_definitions["primary_scaled_quarter"](ny)
            mean = curves[ny].mean(axis=0)
            q16, q84 = np.quantile(curves[ny], [0.16, 0.84], axis=0)
            for separation in range(ny // 2 + 1):
                writer.writerow(
                    {
                        "Ny": ny,
                        "r_y": separation,
                        "chord_distance": f"{chord(ny, np.asarray([separation]))[0]:.17g}",
                        "mean_correlator": f"{mean[separation]:.17g}",
                        "trajectory_q16": f"{q16[separation]:.17g}",
                        "trajectory_q84": f"{q84[separation]:.17g}",
                        "in_primary_window": minimum <= separation <= maximum,
                    }
                )

    configure_matplotlib()
    figure, (raw_axis, scaling_axis) = plt.subplots(
        2, 1, figsize=(3.375, 5.35), gridspec_kw={"hspace": 0.34}
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.04, 0.94, len(SIZES)))
    markers = ("o", "s", "^", "D", "v", "P")
    linestyles = ("-", "--", ":", "-.", (0, (4, 1.5)), (0, (2, 1)))

    for index, ny in enumerate(SIZES):
        separations = np.arange(1, ny // 2 + 1)
        distances = chord(ny, separations)
        mean = curves[ny].mean(axis=0)[1:]
        raw_axis.plot(
            distances,
            mean,
            color=colors[index],
            marker=markers[index],
            linestyle=linestyles[index],
            linewidth=1.0,
            markersize=2.8,
            markerfacecolor="white",
            markeredgewidth=0.7,
            markevery=max(1, len(separations) // 6),
            label=rf"${ny}$",
        )
    guide_x = np.geomspace(2.2, 9.0, 100)
    guide_anchor = chord(40, np.asarray([3]))[0]
    guide_y = 1.35 * curves[40].mean(axis=0)[3] * (guide_x / guide_anchor) ** -2
    raw_axis.plot(
        guide_x,
        guide_y,
        color="#555555",
        linestyle="--",
        linewidth=0.9,
    )
    raw_axis.text(
        0.965,
        0.91,
        r"$d^{-2}$",
        transform=raw_axis.transAxes,
        ha="right",
        va="top",
        color="#444444",
    )
    raw_axis.set_xscale("log")
    raw_axis.set_yscale("log")
    raw_axis.set_xlabel(r"chord distance $d_{N_y}(r_y)$")
    raw_axis.set_ylabel(r"trajectory mean $\overline{G}(r_y)$")
    raw_axis.legend(
        loc="lower left",
        frameon=False,
        ncol=3,
        title=r"$N_y$",
        columnspacing=0.8,
        handlelength=1.5,
        handletextpad=0.35,
    )

    primary_summaries = [
        fits["primary_scaled_quarter"][ny]["trajectory_beta_summary"] for ny in SIZES
    ]
    primary_q25 = np.asarray(
        [record["interquartile_range"][0] for record in primary_summaries]
    )
    primary_q75 = np.asarray(
        [record["interquartile_range"][1] for record in primary_summaries]
    )
    order = np.argsort(inverse_sizes)
    scaling_axis.errorbar(
        inverse_sizes[order],
        primary_means[order],
        yerr=np.vstack(
            (
                primary_means[order] - primary_q25[order],
                primary_q75[order] - primary_means[order],
            )
        ),
        color="#0072B2",
        marker="o",
        linestyle="none",
        markerfacecolor="white",
        markeredgewidth=1.0,
        markersize=4.8,
        elinewidth=0.85,
        capsize=2.2,
        zorder=5,
    )
    linear = regression_summaries["linear_1_over_Ny"]
    quadratic = regression_summaries["linear_1_over_Ny_squared"]
    fit_x = np.linspace(0.0, inverse_sizes.max() * 1.03, 100)
    scaling_axis.plot(
        fit_x,
        linear["beta_infinite_intercept"] + linear["slope"] * fit_x,
        color="#D92725",
        linestyle="-",
        linewidth=1.05,
        label=r"$N_y^{-1}$",
    )
    scaling_axis.plot(
        fit_x,
        quadratic["beta_infinite_intercept"] + quadratic["slope"] * fit_x**2,
        color="#009E73",
        linestyle="--",
        linewidth=1.05,
        label=r"$N_y^{-2}$",
    )
    scaling_axis.plot(
        0.0,
        linear["beta_infinite_intercept"],
        marker="^",
        color="#D92725",
        markersize=4.5,
    )
    scaling_axis.plot(
        0.0,
        quadratic["beta_infinite_intercept"],
        marker="s",
        markerfacecolor="white",
        markeredgecolor="#009E73",
        markeredgewidth=0.9,
        markersize=4.5,
    )
    scaling_axis.axhline(2.0, color="#555555", linestyle="--", linewidth=0.9)
    scaling_axis.set_xlim(-0.0015, inverse_sizes.max() * 1.04)
    scaling_axis.set_ylim(1.985, 2.415)
    scaling_axis.set_xticks([0.0, 0.02, 0.04])
    scaling_axis.set_yticks([2.0, 2.2, 2.4])
    scaling_axis.minorticks_off()
    scaling_axis.set_xlabel(r"$1/N_y$")
    scaling_axis.set_ylabel(r"$\langle\beta_\xi\rangle$")
    scaling_axis.legend(
        loc="upper left",
        frameon=False,
        ncol=2,
        title="extrapolation",
        handlelength=1.8,
        columnspacing=0.9,
    )
    scaling_axis.text(
        0.02,
        0.035,
        r"$\beta=2$",
        transform=scaling_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#444444",
    )

    for label, axis in zip(("(a)", "(b)"), (raw_axis, scaling_axis)):
        axis.text(-0.14, 1.035, label, transform=axis.transAxes, ha="left", va="bottom")
    figure.subplots_adjust(left=0.19, right=0.975, bottom=0.09, top=0.975)
    metadata = {
        "Title": "Finite-size hard-wall power-law analysis",
        "Author": "class_A_fermionic_adaptive_circuit analysis",
        "Subject": (
            "Nx=20; Ny=24,28,30,32,40,50; S=100 each; trajectory-first chord fits; "
            "IQR bars and competing finite-size fits; no SEM or confidence intervals"
        ),
    }
    figure.savefig(OUTPUT_DIR / f"{STEM}.pdf", metadata=metadata)
    figure.savefig(OUTPUT_DIR / f"{STEM}.png", dpi=300, metadata=metadata)
    plt.close(figure)

    serialized_fits: dict[str, dict[str, Any]] = {}
    for window_name in window_definitions:
        serialized_fits[window_name] = {}
        for ny in SIZES:
            record = fits[window_name][ny]
            serialized_fits[window_name][str(ny)] = {
                "window": record["window"],
                "trajectory_beta_summary": record["trajectory_beta_summary"],
                "trajectory_r_squared_summary": record["trajectory_r_squared_summary"],
                "ensemble_mean_curve_fit": record["ensemble_mean_curve_fit"],
            }

    summary = {
        "schema": "hard_wall_finite_size_power_law_v3_vertical_two_panel",
        "contract": {
            "Nx": NX,
            "Ny": list(SIZES),
            "samples_per_size": ENSEMBLE_SIZE,
            "total_independent_trajectories": ENSEMBLE_SIZE * len(SIZES),
            "alpha_1": 1.0,
            "alpha_2": 30.0,
            "nshell": 1,
            "construction": "hard/support-terminated",
            "cycles": "2*Ny; final cycle analyzed",
            "initialization": "pure/default",
            "sequence": "raster_y",
            "perfect_correction": True,
            "dtype": "complex128",
            "correlator": EXPECTED_LEGACY_FORMULA,
        },
        "cohorts": {
            "new_x_resolved": [24, 28, 32],
            "legacy_streaming_x_average": [30, 40, 50],
            "compatibility": (
                "same locked scientific protocol and same x-averaged estimator; "
                "independent ensembles are compared by size and never merged within a size"
            ),
        },
        "estimator": {
            "model": "log G_xavg = log A - beta log(d_N)",
            "chord_distance": "d_N(r)=(Ny/pi) sin(pi*r/Ny)",
            "primary_window": "2 <= r_y <= floor(Ny/4)",
            "estimator_order": "fit each trajectory, then summarize the 100 exponents at each Ny",
            "spread": "sample SD and empirical percentiles; no SEM or confidence interval on the mean",
        },
        "fits": serialized_fits,
        "finite_size_regressions": regression_summaries,
        "interpretation_status": (
            "algebraic decay is strong, but the infinite-size intercept is ansatz-sensitive; "
            "do not quote a precision thermodynamic exponent from these sizes"
        ),
        "input_files": provenance,
        "outputs": [
            f"{STEM}.pdf",
            f"{STEM}.png",
            trajectory_csv.name,
            size_csv.name,
            curve_csv.name,
            f"{STEM}_summary.json",
        ],
    }
    summary_path = OUTPUT_DIR / f"{STEM}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    concise = {
        str(ny): fits["primary_scaled_quarter"][ny]["trajectory_beta_summary"]
        for ny in SIZES
    }
    print(json.dumps({"primary": concise, "finite_size": regression_summaries}, indent=2))


if __name__ == "__main__":
    main()
