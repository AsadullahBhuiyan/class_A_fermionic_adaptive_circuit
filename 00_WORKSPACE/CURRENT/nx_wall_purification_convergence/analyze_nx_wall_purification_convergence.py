#!/usr/bin/env python3
"""Analyze transverse-width dependence of max-mix wall purification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[2]
DEFAULT_CAMPAIGN_ID = "Nx20-24-28_Ny20_nsh1_dwtrunc1_init-maxmix_S25_C100"
DEFAULT_BOOTSTRAPS = 2000
DEFAULT_BOOTSTRAP_SEED = 20260821
DEFAULT_EPSILON = 1e-2
DEFAULT_EQUIVALENCE_MARGIN = 5.0
WALL_FIT_WINDOW = (5, 30)
BULK_FIT_WINDOW = (1, 30)
NX_VALUES = (20, 24, 28)

CHOI_CAMPAIGN = (
    REPO_ROOT
    / "00_WORKSPACE/LARGE_RESULTS/choi_covariance_cpu/cpu_data/complex_particle_choi_transfer/campaigns"
    / "Nx16-20-24_Ny20_DW1_dwtrunc1_openBC_transfer_a1-1-3x21_S10_C40_v6"
)
B0_CAMPAIGN = (
    REPO_ROOT
    / "00_WORKSPACE/CURRENT/experiment_review/b0_exact_domain_wall/results/20260816_191957"
)


@dataclass
class GeometryData:
    nx: int
    ny: int
    cycles: np.ndarray
    sample_indices: np.ndarray
    wall_entropy: np.ndarray
    bulk_entropy: np.ndarray
    total_entropy_per_total_mode: np.ndarray
    charge_variance_per_total_mode: np.ndarray
    active_purity_deficit_rms: np.ndarray
    active_covariance_frobenius_rms: np.ndarray
    full_covariance_frobenius_per_total_mode: np.ndarray
    entropy_profile_bits: np.ndarray
    run_directory: Path

    @property
    def sample_count(self) -> int:
        return int(self.wall_entropy.shape[0])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temporary_name).replace(path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([{key: jsonable(row.get(key, "")) for key in fieldnames} for row in rows])
        Path(temporary_name).replace(path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    Path(temporary_name).unlink(missing_ok=True)
    try:
        np.savez_compressed(temporary_name, **arrays)
        Path(temporary_name).replace(path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_campaign(campaign_root: Path) -> tuple[dict[str, Any], dict[int, GeometryData]]:
    manifest_path = campaign_root / "campaign_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Campaign manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    geometries: dict[int, GeometryData] = {}
    for row in manifest["results"]:
        run_dir = Path(row["run_directory"])
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        data_path = run_dir / "trajectory_observables.npz"
        if not data_path.exists():
            continue
        with np.load(data_path, allow_pickle=False) as payload:
            nx = int(row["Nx"])
            geometries[nx] = GeometryData(
                nx=nx,
                ny=int(row["Ny"]),
                cycles=np.asarray(payload["cycles"], dtype=np.int64),
                sample_indices=np.asarray(payload["sample_indices"], dtype=np.int64),
                wall_entropy=np.asarray(payload["wall_entropy_bits_per_cell"], dtype=np.float64),
                bulk_entropy=np.asarray(payload["bulk_entropy_bits_per_cell"], dtype=np.float64),
                total_entropy_per_total_mode=np.asarray(
                    payload["total_entropy_per_total_mode_bits"], dtype=np.float64
                ),
                charge_variance_per_total_mode=np.asarray(
                    payload["charge_variance_per_total_mode"], dtype=np.float64
                ),
                active_purity_deficit_rms=np.asarray(
                    payload["active_purity_deficit_rms"], dtype=np.float64
                ),
                active_covariance_frobenius_rms=np.asarray(
                    payload["active_covariance_frobenius_rms"], dtype=np.float64
                ),
                full_covariance_frobenius_per_total_mode=np.asarray(
                    payload["full_covariance_frobenius_per_total_mode"], dtype=np.float64
                ),
                entropy_profile_bits=np.asarray(payload["entropy_profile_bits"], dtype=np.float64),
                run_directory=run_dir,
            )
    if not geometries:
        raise RuntimeError(f"No completed trajectory aggregates found under {campaign_root}")
    return manifest, geometries


def rolling_mean(values: np.ndarray, window: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.copy()
    half = window // 2
    smoothed = np.full_like(values, np.nan)
    for index in range(values.size):
        start = max(0, index - half)
        stop = min(values.size, index + half + 1)
        smoothed[index] = np.nanmean(values[start:stop])
    return smoothed


def first_sustained_crossing(
    cycles: np.ndarray, curve: np.ndarray, epsilon: float
) -> float:
    cycles = np.asarray(cycles)
    curve = np.asarray(curve, dtype=np.float64)
    for position in range(curve.size):
        tail = curve[position:]
        if np.all(np.isfinite(tail)) and np.all(tail < epsilon):
            return float(cycles[position])
    return math.nan


def aicc_from_residuals(residuals: np.ndarray, parameter_count: int) -> float:
    residuals = np.asarray(residuals, dtype=np.float64)
    n_points = residuals.size
    rss = max(float(np.sum(residuals**2)), np.finfo(float).tiny)
    base = n_points * math.log(rss / n_points) + 2.0 * parameter_count
    denominator = n_points - parameter_count - 1
    return base + (2.0 * parameter_count * (parameter_count + 1) / denominator if denominator > 0 else math.inf)


def fit_wall_models(
    cycles: np.ndarray,
    values: np.ndarray,
    fit_window: tuple[int, int] = WALL_FIT_WINDOW,
) -> dict[str, float]:
    cycles = np.asarray(cycles, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    mask = (
        (cycles >= fit_window[0])
        & (cycles <= fit_window[1])
        & np.isfinite(values)
        & (values > 1e-14)
    )
    x = cycles[mask]
    y = np.log(values[mask])
    if x.size < 5:
        return {key: math.nan for key in ("power_amplitude", "power_alpha", "power_aicc", "exponential_amplitude", "exponential_tau", "exponential_aicc", "delta_aicc_exp_minus_power")}

    power_slope, power_intercept = np.polyfit(np.log(x), y, 1)
    power_prediction = power_intercept + power_slope * np.log(x)
    exponential_slope, exponential_intercept = np.polyfit(x, y, 1)
    exponential_prediction = exponential_intercept + exponential_slope * x
    power_aicc = aicc_from_residuals(y - power_prediction, 2)
    exponential_aicc = aicc_from_residuals(y - exponential_prediction, 2)
    return {
        "power_amplitude": float(np.exp(power_intercept)),
        "power_alpha": float(-power_slope),
        "power_aicc": power_aicc,
        "exponential_amplitude": float(np.exp(exponential_intercept)),
        "exponential_tau": float(-1.0 / exponential_slope) if exponential_slope < 0 else math.inf,
        "exponential_aicc": exponential_aicc,
        "delta_aicc_exp_minus_power": float(exponential_aicc - power_aicc),
    }


def fit_bulk_model(
    cycles: np.ndarray,
    values: np.ndarray,
    wall_alpha: float,
    fit_window: tuple[int, int] = BULK_FIT_WINDOW,
) -> dict[str, float]:
    cycles = np.asarray(cycles, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    mask = (
        (cycles >= fit_window[0])
        & (cycles <= fit_window[1])
        & np.isfinite(values)
        & (values > 1e-14)
    )
    x = cycles[mask]
    y = values[mask]
    if x.size < 5 or not np.isfinite(wall_alpha):
        return {"bulk_exponential_amplitude": math.nan, "bulk_tau": math.nan, "bulk_wall_tail_amplitude": math.nan}

    def model(t: np.ndarray, amplitude: float, tau: float, tail: float) -> np.ndarray:
        return amplitude * np.exp(-t / tau) + tail * np.power(t, -wall_alpha)

    try:
        parameters, _ = curve_fit(
            model,
            x,
            y,
            p0=(max(float(y[0]), 1e-8), 1.0, max(float(y[-1] * x[-1] ** wall_alpha), 1e-8)),
            bounds=((0.0, 0.05, 0.0), (10.0, 100.0, 10.0)),
            maxfev=20_000,
        )
        return {
            "bulk_exponential_amplitude": float(parameters[0]),
            "bulk_tau": float(parameters[1]),
            "bulk_wall_tail_amplitude": float(parameters[2]),
        }
    except (RuntimeError, ValueError, FloatingPointError):
        return {"bulk_exponential_amplitude": math.nan, "bulk_tau": math.nan, "bulk_wall_tail_amplitude": math.nan}


def bootstrap_geometry(
    data: GeometryData,
    *,
    bootstrap_count: int,
    rng: np.random.Generator,
    epsilon: float,
) -> dict[str, Any]:
    sample_count = data.sample_count
    indices = rng.integers(0, sample_count, size=(bootstrap_count, sample_count))
    wall_curves = np.mean(data.wall_entropy[indices], axis=1)
    bulk_curves = np.mean(data.bulk_entropy[indices], axis=1)
    wall_mean = np.mean(data.wall_entropy, axis=0)
    bulk_mean = np.mean(data.bulk_entropy, axis=0)
    wall_band = np.quantile(wall_curves, (0.025, 0.975), axis=0)
    bulk_band = np.quantile(bulk_curves, (0.025, 0.975), axis=0)

    wall_thresholds = np.asarray(
        [first_sustained_crossing(data.cycles, curve, epsilon) for curve in wall_curves]
    )
    bulk_thresholds = np.asarray(
        [first_sustained_crossing(data.cycles, curve, epsilon) for curve in bulk_curves]
    )
    alphas = np.full(bootstrap_count, np.nan, dtype=np.float64)
    delta_aicc = np.full(bootstrap_count, np.nan, dtype=np.float64)
    bulk_taus = np.full(bootstrap_count, np.nan, dtype=np.float64)
    for draw in range(bootstrap_count):
        wall_fit = fit_wall_models(data.cycles, wall_curves[draw])
        alphas[draw] = wall_fit["power_alpha"]
        delta_aicc[draw] = wall_fit["delta_aicc_exp_minus_power"]
        bulk_taus[draw] = fit_bulk_model(
            data.cycles, bulk_curves[draw], wall_fit["power_alpha"]
        )["bulk_tau"]

    mean_wall_fit = fit_wall_models(data.cycles, wall_mean)
    mean_bulk_fit = fit_bulk_model(data.cycles, bulk_mean, mean_wall_fit["power_alpha"])
    return {
        "wall_mean": wall_mean,
        "bulk_mean": bulk_mean,
        "wall_band": wall_band,
        "bulk_band": bulk_band,
        "wall_threshold_mean_curve": first_sustained_crossing(data.cycles, wall_mean, epsilon),
        "bulk_threshold_mean_curve": first_sustained_crossing(data.cycles, bulk_mean, epsilon),
        "wall_threshold_upper_band": first_sustained_crossing(data.cycles, wall_band[1], epsilon),
        "bulk_threshold_upper_band": first_sustained_crossing(data.cycles, bulk_band[1], epsilon),
        "wall_thresholds": wall_thresholds,
        "bulk_thresholds": bulk_thresholds,
        "wall_alphas": alphas,
        "wall_delta_aicc": delta_aicc,
        "bulk_taus": bulk_taus,
        "mean_wall_fit": mean_wall_fit,
        "mean_bulk_fit": mean_bulk_fit,
    }


def finite_quantiles(values: np.ndarray, probabilities: tuple[float, ...] = (0.025, 0.5, 0.975)) -> list[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [math.nan] * len(probabilities)
    return [float(value) for value in np.quantile(finite, probabilities)]


def compute_joint_decision(
    geometries: dict[int, GeometryData],
    bootstraps: dict[int, dict[str, Any]],
    *,
    epsilon: float,
    equivalence_margin: float,
) -> tuple[dict[str, Any], dict[int, np.ndarray], np.ndarray]:
    reference_nx = min(geometries)
    reference = bootstraps[reference_nx]["wall_thresholds"]
    differences: dict[int, np.ndarray] = {}
    pairwise_rows = []
    for nx in sorted(geometries):
        if nx == reference_nx:
            continue
        difference = bootstraps[nx]["wall_thresholds"] - reference
        differences[nx] = difference
        low, median, high = finite_quantiles(difference)
        pairwise_rows.append(
            {
                "reference_Nx": reference_nx,
                "comparison_Nx": nx,
                "difference_cycles_median": median,
                "difference_cycles_ci_low": low,
                "difference_cycles_ci_high": high,
                "within_equivalence_margin": bool(
                    np.isfinite(low) and np.isfinite(high) and low > -equivalence_margin and high < equivalence_margin
                ),
                "outside_equivalence_margin": bool(
                    np.isfinite(low) and np.isfinite(high) and (low > equivalence_margin or high < -equivalence_margin)
                ),
            }
        )

    bootstrap_count = next(iter(bootstraps.values()))["wall_thresholds"].size
    z_values = np.full(bootstrap_count, np.nan, dtype=np.float64)
    nx_values = np.asarray(sorted(geometries), dtype=np.float64)
    for draw in range(bootstrap_count):
        thresholds = np.asarray(
            [bootstraps[int(nx)]["wall_thresholds"][draw] for nx in nx_values], dtype=np.float64
        )
        if np.all(np.isfinite(thresholds)) and np.all(thresholds > 0):
            z_values[draw] = float(np.polyfit(np.log(nx_values / reference_nx), np.log(thresholds), 1)[0])
    z_low, z_median, z_high = finite_quantiles(z_values)

    extension_required = any(
        float(bootstraps[nx]["wall_band"][1, -1]) >= epsilon for nx in geometries
    )
    enough_samples = all(data.sample_count >= 25 for data in geometries.values())
    all_equivalent = bool(pairwise_rows) and all(row["within_equivalence_margin"] for row in pairwise_rows)
    z_excludes_zero = np.isfinite(z_low) and np.isfinite(z_high) and (z_low > 0 or z_high < 0)
    resolved_dependence = any(row["outside_equivalence_margin"] for row in pairwise_rows) and z_excludes_zero

    if extension_required:
        classification = "extend_cycle_horizon"
        recommended_action = "Resume every geometry to 150 cycles before interpreting Nx dependence."
    elif not enough_samples:
        classification = "insufficient_pilot_statistics"
        recommended_action = "Complete 25 trajectories per geometry."
    elif all_equivalent:
        classification = "no_resolved_material_Nx_dependence"
        recommended_action = "No trajectory top-up is required for the five-cycle equivalence criterion."
    elif resolved_dependence:
        classification = "resolved_Nx_dependence"
        recommended_action = "No automatic top-up is required; report the effect over Nx=20--28."
    else:
        classification = "inconclusive_top_up_to_50"
        recommended_action = "Resume to 50 trajectories per geometry and repeat the bootstrap."

    decision = {
        "classification": classification,
        "claim_scope": "Nx=20,24,28 at Ny=20 and epsilon=0.01 bits per cell only",
        "epsilon_bits_per_cell": epsilon,
        "equivalence_margin_cycles": equivalence_margin,
        "reference_Nx": reference_nx,
        "pairwise_wall_threshold_differences": pairwise_rows,
        "descriptive_z_x": {
            "median": z_median,
            "ci_low": z_low,
            "ci_high": z_high,
            "asymptotic_scaling_claim": False,
        },
        "extension_required": extension_required,
        "recommended_action": recommended_action,
    }
    return decision, differences, z_values


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 0.7,
        }
    )


def save_figure(fig: plt.Figure, output_directory: Path, stem: str) -> list[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    paths = [output_directory / f"{stem}.pdf", output_directory / f"{stem}.png"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_main_figure(
    geometries: dict[int, GeometryData],
    bootstraps: dict[int, dict[str, Any]],
    decision: dict[str, Any],
    output_directory: Path,
    *,
    epsilon: float,
    rolling_window: int,
) -> list[Path]:
    configure_plot_style()
    colors = {20: "#0072B2", 24: "#D55E00", 28: "#009E73"}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    for nx in sorted(geometries):
        data = geometries[nx]
        result = bootstraps[nx]
        positive = data.cycles >= 1
        label = rf"$N_x={nx}$"
        axes[0, 0].fill_between(
            data.cycles[positive], result["wall_band"][0, positive], result["wall_band"][1, positive],
            color=colors[nx], alpha=0.10, linewidth=0,
        )
        axes[0, 0].loglog(
            data.cycles[positive], result["wall_mean"][positive], color=colors[nx], lw=0.6, alpha=0.55
        )
        axes[0, 0].loglog(
            data.cycles[positive], rolling_mean(result["wall_mean"], rolling_window)[positive],
            color=colors[nx], lw=1.4, label=label,
        )
        axes[0, 1].fill_between(
            data.cycles[positive], result["bulk_band"][0, positive], result["bulk_band"][1, positive],
            color=colors[nx], alpha=0.10, linewidth=0,
        )
        axes[0, 1].semilogy(
            data.cycles[positive], result["bulk_mean"][positive], color=colors[nx], lw=0.6, alpha=0.55
        )
        axes[0, 1].semilogy(
            data.cycles[positive], rolling_mean(result["bulk_mean"], rolling_window)[positive],
            color=colors[nx], lw=1.4, label=label,
        )

    for axis in axes[0]:
        axis.axhline(epsilon, color="0.3", ls="--", lw=0.8, label=rf"$\epsilon={epsilon:g}$")
        axis.set_xlabel("cycle number")
        axis.set_ylabel("entropy density (bits/cell)")
    axes[0, 0].set_title("(a) Wall purification")
    axes[0, 1].set_title("(b) Active-bulk purification")
    axes[0, 0].legend(frameon=False, ncol=2)

    nx_values = np.asarray(sorted(geometries), dtype=np.float64)
    for offset, region, marker in ((-0.18, "wall", "o"), (0.18, "bulk", "s")):
        centers, lower, upper = [], [], []
        for nx in nx_values.astype(int):
            draws = bootstraps[nx][f"{region}_thresholds"]
            low, median, high = finite_quantiles(draws)
            centers.append(median)
            lower.append(median - low)
            upper.append(high - median)
        axes[1, 0].errorbar(
            nx_values + offset,
            centers,
            yerr=np.asarray([lower, upper]),
            marker=marker,
            ms=4,
            capsize=2,
            lw=0.9,
            label=region,
        )
    axes[1, 0].set(
        xlabel=r"$N_x$",
        ylabel=rf"$T_{{\mathrm{{conv}}}}({epsilon:g})$ (cycles)",
        title="(c) Sustained convergence threshold",
        xticks=nx_values,
    )
    axes[1, 0].legend(frameon=False)

    alpha_center, alpha_lower, alpha_upper = [], [], []
    for nx in nx_values.astype(int):
        low, median, high = finite_quantiles(bootstraps[nx]["wall_alphas"])
        alpha_center.append(median)
        alpha_lower.append(median - low)
        alpha_upper.append(high - median)
    axes[1, 1].errorbar(
        nx_values,
        alpha_center,
        yerr=np.asarray([alpha_lower, alpha_upper]),
        marker="o",
        capsize=2,
        lw=0.9,
    )
    z = decision["descriptive_z_x"]
    axes[1, 1].text(
        0.04,
        0.05,
        rf"descriptive $z_x={z['median']:.2f}$ [{z['ci_low']:.2f}, {z['ci_high']:.2f}]",
        transform=axes[1, 1].transAxes,
        va="bottom",
    )
    axes[1, 1].set(
        xlabel=r"$N_x$",
        ylabel=r"wall exponent $\alpha_{\mathrm{w}}$",
        title="(d) Algebraic wall tail",
        xticks=nx_values,
    )
    return save_figure(fig, output_directory, "nx_wall_purification_convergence_main")


def plot_secondary_figure(
    geometries: dict[int, GeometryData],
    bootstraps: dict[int, dict[str, Any]],
    output_directory: Path,
    *,
    rolling_window: int,
) -> list[Path]:
    configure_plot_style()
    colors = {20: "#0072B2", 24: "#D55E00", 28: "#009E73"}
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8))
    for nx in sorted(geometries):
        data = geometries[nx]
        cycles = data.cycles
        positive = cycles >= 1
        label = rf"$N_x={nx}$"
        purity = np.mean(data.active_purity_deficit_rms, axis=0)
        charge = np.mean(data.charge_variance_per_total_mode, axis=0)
        frobenius = np.mean(data.full_covariance_frobenius_per_total_mode, axis=0)
        axes[0, 0].semilogy(cycles[positive], np.maximum(purity[positive], 1e-14), color=colors[nx], label=label)
        axes[0, 1].semilogy(cycles[positive], np.maximum(frobenius[positive], 1e-14), color=colors[nx], label=label)
        axes[0, 2].semilogy(cycles[positive], np.maximum(charge[positive], 1e-14), color=colors[nx], label=label)
        axes[1, 0].semilogy(
            cycles[1:],
            np.maximum(rolling_mean(np.abs(np.diff(purity)), rolling_window), 1e-14),
            color=colors[nx],
            label=label,
        )
        axes[1, 1].semilogy(
            cycles[1:],
            np.maximum(rolling_mean(np.abs(np.diff(frobenius)), rolling_window), 1e-14),
            color=colors[nx],
            label=label,
        )
        axes[1, 2].semilogy(
            cycles[1:],
            np.maximum(rolling_mean(np.abs(np.diff(charge)), rolling_window), 1e-14),
            color=colors[nx],
            label=label,
        )
    labels = (
        (axes[0, 0], "(a) Active covariance purity deficit", r"$\Vert G^2-\mathbf{1}\Vert_F/\sqrt{N_{\mathrm{active}}}$"),
        (axes[0, 1], "(b) Raw covariance Frobenius norm", r"$\Vert G\Vert_F/(2N_xN_y)$"),
        (axes[0, 2], "(c) Total charge variance", r"$\operatorname{Var}(\hat N)/(2N_xN_y)$"),
        (axes[1, 0], "(d) Absolute purity-deficit difference", r"$|\Delta(\Vert G^2-\mathbf{1}\Vert_F/\sqrt{N_{\mathrm{active}}})|$"),
        (axes[1, 1], "(e) Absolute Frobenius difference", r"$|\Delta(\Vert G\Vert_F/(2N_xN_y))|$"),
        (axes[1, 2], "(f) Absolute charge-variance difference", r"$|\Delta\operatorname{Var}(\hat N)|/(2N_xN_y)$"),
    )
    for axis, title, ylabel in labels:
        axis.set_title(title)
        axis.set_xlabel("cycle number")
        axis.set_ylabel(ylabel)
    axes[0, 0].legend(frameon=False)
    return save_figure(fig, output_directory, "nx_wall_purification_global_and_derivatives")


def load_contextual_data() -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[str, list[tuple[int, float]]]]:
    choi: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for nx in (16, 20, 24):
        path = (
            CHOI_CAMPAIGN
            / "runs"
            / f"N{nx}x20_DW1_openBC_dwtrunc1_slab_a1-1_a2-30_nsh1_perfect_correction"
            / "particle_choi_transfer_gap_vs_cycle.npz"
        )
        if path.exists():
            with np.load(path, allow_pickle=False) as payload:
                choi[nx] = (
                    np.asarray(payload["cycles"], dtype=np.int64),
                    np.mean(np.asarray(payload["gap"], dtype=np.float64), axis=0),
                )
    b0: dict[str, list[tuple[int, float]]] = {"coupled": [], "hard_exterior": []}
    for construction in b0:
        for nx in (12, 16, 20, 24, 28, 32):
            path = B0_CAMPAIGN / "status" / f"{construction}__Nx{nx:03d}__Ny048.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                b0[construction].append((nx, float(payload["edge"]["width_ratio"])))
    return choi, b0


def plot_contextual_cross_checks(output_directory: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    configure_plot_style()
    choi, b0 = load_contextual_data()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))
    context_rows: list[dict[str, Any]] = []
    for nx, (cycles, gap) in sorted(choi.items()):
        axes[0].semilogy(cycles, np.maximum(gap, 1e-14), label=rf"$N_x={nx}$")
        for cycle, value in zip(cycles, gap):
            context_rows.append({"source": "Choi transfer", "series": f"Nx={nx}", "x": int(cycle), "value": float(value)})
    axes[0].set(
        xlabel="cycle number",
        ylabel="mean Choi-transfer gap",
        title="(a) Existing dynamical width scan",
    )
    axes[0].legend(frameon=False)

    for construction, rows in b0.items():
        if not rows:
            continue
        nx_values, ratios = zip(*rows)
        axes[1].semilogy(nx_values, ratios, "o-", label=construction.replace("_", " "))
        for nx, value in rows:
            context_rows.append({"source": "B0 static width gate", "series": construction, "x": nx, "value": value})
    axes[1].axhline(0.1, color="0.3", ls="--", lw=0.8, label="10% gate")
    axes[1].axvline(20, color="0.5", ls=":", lw=0.8)
    axes[1].set(
        xlabel=r"$N_x$",
        ylabel=r"$m/(2\pi|v|/N_y)$",
        title="(b) Existing static width calibration",
    )
    axes[1].legend(frameon=False)
    return save_figure(fig, output_directory, "contextual_existing_width_scans"), context_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--results-root", type=Path, default=PACKAGE_ROOT / "results")
    parser.add_argument("--output-root", type=Path, default=PACKAGE_ROOT / "analysis_outputs")
    parser.add_argument("--bootstrap-count", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--equivalence-margin", type=float, default=DEFAULT_EQUIVALENCE_MARGIN)
    parser.add_argument("--rolling-window", type=int, default=5)
    args = parser.parse_args(argv)
    if args.bootstrap_count < 1 or args.rolling_window < 1:
        parser.error("--bootstrap-count and --rolling-window must be positive")
    if not 0 < args.epsilon < 1:
        parser.error("--epsilon must lie in (0,1)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    campaign_root = args.results_root.resolve() / "campaigns" / args.campaign_id
    manifest, geometries = load_campaign(campaign_root)
    output_root = args.output_root.resolve() / args.campaign_id
    figure_root = output_root / "figures"
    table_root = output_root / "tables"
    rng = np.random.default_rng(args.bootstrap_seed)
    bootstraps = {
        nx: bootstrap_geometry(
            data,
            bootstrap_count=args.bootstrap_count,
            rng=rng,
            epsilon=args.epsilon,
        )
        for nx, data in sorted(geometries.items())
    }
    decision, differences, z_values = compute_joint_decision(
        geometries,
        bootstraps,
        epsilon=args.epsilon,
        equivalence_margin=args.equivalence_margin,
    )

    fit_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    bootstrap_arrays: dict[str, np.ndarray] = {"z_x": z_values}
    for nx, data in sorted(geometries.items()):
        result = bootstraps[nx]
        alpha_low, alpha_median, alpha_high = finite_quantiles(result["wall_alphas"])
        tau_low, tau_median, tau_high = finite_quantiles(result["bulk_taus"])
        fit_rows.append(
            {
                "Nx": nx,
                "Ny": data.ny,
                "samples": data.sample_count,
                **result["mean_wall_fit"],
                **result["mean_bulk_fit"],
                "wall_alpha_bootstrap_median": alpha_median,
                "wall_alpha_ci_low": alpha_low,
                "wall_alpha_ci_high": alpha_high,
                "bulk_tau_bootstrap_median": tau_median,
                "bulk_tau_ci_low": tau_low,
                "bulk_tau_ci_high": tau_high,
            }
        )
        for region in ("wall", "bulk"):
            draws = result[f"{region}_thresholds"]
            low, median, high = finite_quantiles(draws)
            threshold_rows.append(
                {
                    "Nx": nx,
                    "Ny": data.ny,
                    "samples": data.sample_count,
                    "region": region,
                    "epsilon_bits_per_cell": args.epsilon,
                    "mean_curve_threshold": result[f"{region}_threshold_mean_curve"],
                    "upper_band_threshold": result[f"{region}_threshold_upper_band"],
                    "bootstrap_median": median,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "bootstrap_censored_fraction": float(np.mean(~np.isfinite(draws))),
                }
            )
        bootstrap_arrays[f"Nx{nx}_wall_threshold"] = result["wall_thresholds"]
        bootstrap_arrays[f"Nx{nx}_bulk_threshold"] = result["bulk_thresholds"]
        bootstrap_arrays[f"Nx{nx}_wall_alpha"] = result["wall_alphas"]
        bootstrap_arrays[f"Nx{nx}_bulk_tau"] = result["bulk_taus"]
        bootstrap_arrays[f"Nx{nx}_wall_curve_ci"] = result["wall_band"]
        bootstrap_arrays[f"Nx{nx}_bulk_curve_ci"] = result["bulk_band"]
    for nx, values in differences.items():
        bootstrap_arrays[f"Nx{nx}_minus_Nx{min(geometries)}_wall_threshold"] = values

    write_csv_atomic(table_root / "fit_summary.csv", fit_rows)
    write_csv_atomic(table_root / "convergence_thresholds.csv", threshold_rows)
    save_npz_atomic(output_root / "bootstrap_draws.npz", **bootstrap_arrays)
    write_json_atomic(output_root / "conclusion.json", decision)

    figure_paths = []
    figure_paths += plot_main_figure(
        geometries,
        bootstraps,
        decision,
        figure_root,
        epsilon=args.epsilon,
        rolling_window=args.rolling_window,
    )
    figure_paths += plot_secondary_figure(
        geometries, bootstraps, figure_root, rolling_window=args.rolling_window
    )
    contextual_paths, context_rows = plot_contextual_cross_checks(figure_root)
    figure_paths += contextual_paths
    write_csv_atomic(table_root / "contextual_existing_width_scans.csv", context_rows)

    analysis_script = Path(__file__).resolve()
    analysis_manifest = {
        "schema_version": 1,
        "analysis_name": "nx_wall_purification_convergence",
        "campaign_id": args.campaign_id,
        "source_campaign_manifest": str(campaign_root / "campaign_manifest.json"),
        "source_campaign_status": manifest.get("status"),
        "bootstrap_count": args.bootstrap_count,
        "bootstrap_seed": args.bootstrap_seed,
        "wall_fit_window": list(WALL_FIT_WINDOW),
        "bulk_fit_window": list(BULK_FIT_WINDOW),
        "epsilon_bits_per_cell": args.epsilon,
        "equivalence_margin_cycles": args.equivalence_margin,
        "rolling_window_display_only": args.rolling_window,
        "raw_data_used_for_fits_and_thresholds": True,
        "contextual_sources_are_not_pooled": True,
        "geometry_sample_counts": {str(nx): data.sample_count for nx, data in geometries.items()},
        "decision": decision,
        "outputs": {
            "figures": [str(path) for path in figure_paths],
            "fit_table": str(table_root / "fit_summary.csv"),
            "threshold_table": str(table_root / "convergence_thresholds.csv"),
            "bootstrap_draws": str(output_root / "bootstrap_draws.npz"),
            "conclusion": str(output_root / "conclusion.json"),
        },
        "source_hashes": {str(analysis_script.relative_to(REPO_ROOT)): sha256_file(analysis_script)},
        "completed_at": utc_now(),
    }
    write_json_atomic(output_root / "analysis_manifest.json", analysis_manifest)
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"Analysis outputs: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
