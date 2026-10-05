#!/usr/bin/env python3
"""Make compact endpoint entropy and charge figures from the verified v2 campaign."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BUNDLE_ROOT = (
    ROOT
    / "00_WORKSPACE/CURRENT/final_production_new_designs"
    / "05_hard_wall_entropy_charge_batched_v2"
)
OUTPUT_ROOT = (
    BUNDLE_ROOT
    / "gpu_data"
    / "hard_wall_entropy_charge_nx20_ny30-60_s100_2ny_raster_v2_batched_endpoint"
)
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"
ENTROPY_STEM = FIGURE_DIR / "endpoint_entropy_cardy_calabrese_3x1"
CHARGE_STEM = FIGURE_DIR / "endpoint_charge_variance_cft_3x1"
CONTOUR_STEM = FIGURE_DIR / "endpoint_entropy_charge_contours_1x2"
COLLAPSE_STEM = FIGURE_DIR / "endpoint_entropy_typical_size_collapse_3x1"
CHARGE_COLLAPSE_STEM = FIGURE_DIR / "endpoint_charge_variance_size_collapse_1x1"
ENSEMBLE_SCALING_STEM = FIGURE_DIR / "endpoint_samplewise_cq_k_scaling_1x1"
MEAN_ENTROPY_COLLAPSE_STEM = FIGURE_DIR / "endpoint_entropy_mean_curve_size_collapse_3x1"
MEAN_CHARGE_COLLAPSE_STEM = FIGURE_DIR / "endpoint_charge_variance_mean_curve_size_collapse_1x1"
MEAN_SCALING_STEM = FIGURE_DIR / "endpoint_mean_curve_cq_k_scaling_1x1"
FIT_CSV = DATA_DIR / "typical_trajectory_endpoint_fits.csv"
SCALING_CSV = DATA_DIR / "samplewise_slope_scaling_sem.csv"
COLLAPSE_CSV = DATA_DIR / "typical_trajectory_size_collapse_fits.csv"
CHARGE_COLLAPSE_CSV = DATA_DIR / "typical_trajectory_charge_variance_collapse_fits.csv"
MEAN_CURVES_CSV = DATA_DIR / "ensemble_mean_endpoint_curves.csv"
MEAN_FITS_CSV = DATA_DIR / "ensemble_mean_curve_fits.csv"
MEAN_COLLAPSE_CSV = DATA_DIR / "ensemble_mean_anchored_collapse_fits.csv"
MEAN_RATIOS_CSV = DATA_DIR / "ensemble_mean_entropy_charge_ratios.csv"
MANIFEST_PATH = HERE / "contour_scaling_figure_manifest.json"

NX = 20
NY_VALUES = (30, 35, 40, 45, 50, 55, 60)
TYPICAL_NY = 60
FIT_MIN_AY = 8
RENYI_LABELS = ("c1", "c2", "c3")
CURVE_LABELS = {
    "c1": r"$S_1$",
    "c2": r"$S_2$",
    "c3": r"$S_3$",
    "k": r"$F_A$",
}
COLORS = {"c1": "#D92725", "c2": "#2CA02C", "c3": "#1F77B4", "k": "#6F4C9B"}
MARKERS = {"c1": "^", "c2": "s", "c3": "o", "k": "D"}
LINESTYLES = {"c1": ":", "c2": "--", "c3": "-", "k": "-."}
EXPECTED_SLOPES = {"c1": 1.0 / 3.0, "c2": 1.0 / 4.0, "c3": 2.0 / 9.0, "k": 1.0 / math.pi**2}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bundle_analysis() -> Any:
    name = "_contour_scaling_verified_endpoint_loader"
    spec = importlib.util.spec_from_file_location(name, BUNDLE_ROOT / "analyze_campaign.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the verified endpoint analyzer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(BUNDLE_ROOT))
    try:
        spec.loader.exec_module(module)
        module._load_runner()
    finally:
        sys.path.remove(str(BUNDLE_ROOT))
    return module


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times", "Nimbus Roman", "Times New Roman", "Liberation Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.7,
            "lines.linewidth": 0.9,
            "lines.markersize": 3.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    return float(slope), float(intercept), float(r_squared)


def endpoint_anchored_fit(
    groups: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float, dict[int, float]]:
    """Fit one common slope through the half-strip anchor.

    Each circumference has unit total weight, so the larger systems do not
    dominate merely because their fit windows contain more strip widths.  The
    supplied coordinates must already obey ``x(Ay=Ny/2)=y(Ay=Ny/2)=0``.
    Since the origin is a physically fixed anchor rather than a fitted
    intercept, the reported coefficient is the uncentered ``R_0^2``.
    """

    numerator = 0.0
    denominator = 0.0
    anchored_total = 0.0
    for ny, (x, y) in groups.items():
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 1 or y.shape != x.shape or x.size < 2:
            raise ValueError("each endpoint-anchored group must contain aligned 1D data")
        if not np.isclose(x[-1], 0.0, rtol=0.0, atol=1e-13):
            raise ValueError(f"Ny={ny} endpoint log-chord coordinate is not zero")
        if not np.isclose(y[-1], 0.0, rtol=0.0, atol=1e-13):
            raise ValueError(f"Ny={ny} endpoint entropy difference is not zero")
        weight = 1.0 / x.size
        numerator += weight * float(np.dot(x, y))
        denominator += weight * float(np.dot(x, x))
        anchored_total += weight * float(np.dot(y, y))
    if denominator <= 0.0 or anchored_total <= 0.0:
        raise RuntimeError("endpoint-anchored fit is singular")
    slope = numerator / denominator
    residual = 0.0
    group_r_squared: dict[int, float] = {}
    for ny, (x, y) in groups.items():
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        fitted = slope * x
        rss = float(np.dot(y - fitted, y - fitted))
        tss = float(np.dot(y, y))
        residual += rss / x.size
        group_r_squared[ny] = 1.0 - rss / tss if tss > 0.0 else 1.0
    anchored_r_squared = 1.0 - residual / anchored_total
    return float(slope), float(anchored_r_squared), group_r_squared


def samplewise_values(analyzer: Any, cases: dict[int, dict[str, np.ndarray]]) -> dict[int, dict[str, np.ndarray]]:
    values: dict[int, dict[str, np.ndarray]] = {}
    for ny in NY_VALUES:
        case = cases[ny]
        values[ny] = {}
        for label, key in analyzer.CURVE_KEYS.items():
            slopes, _ = analyzer.trajectory_slopes(case["ay_values"], case[key], ny)
            values[ny][f"m_{label}"] = slopes
            values[ny][label] = analyzer.PREFACTOR[label] * slopes
    return values


def choose_typical_sample(values: dict[int, dict[str, np.ndarray]]) -> tuple[int, np.ndarray]:
    return choose_typical_sample_at_ny(values, TYPICAL_NY)


def choose_typical_sample_at_ny(
    values: dict[int, dict[str, np.ndarray]], ny: int
) -> tuple[int, np.ndarray]:
    matrix = np.column_stack([values[ny][label] for label in ("c1", "c2", "c3", "k")])
    median = np.median(matrix, axis=0)
    scale = matrix.std(axis=0, ddof=1)
    if np.any(scale <= 0.0):
        raise RuntimeError("cannot define the typical-trajectory distance")
    distance = np.sqrt(np.sum(((matrix - median) / scale) ** 2, axis=1))
    index = int(np.argmin(distance))
    return index, distance


def choose_typical_samples_by_size(
    values: dict[int, dict[str, np.ndarray]],
) -> dict[int, tuple[int, np.ndarray]]:
    return {ny: choose_typical_sample_at_ny(values, ny) for ny in NY_VALUES}


def panel_letter(ax: plt.Axes, letter: str, *, x: float = -0.18) -> None:
    ax.text(x, 1.05, letter, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")


def contour_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    contour: np.ndarray,
    *,
    cax: plt.Axes | None = None,
    colorbar_orientation: str = "vertical",
    cmap: str,
    colorbar_label: str,
) -> dict[str, float]:
    contour = np.asarray(contour, dtype=np.float64)
    if not np.all(np.isfinite(contour)) or np.any(contour < 0.0):
        raise ValueError("contour must be finite and nonnegative")
    vmax = float(contour.max())
    if vmax <= 0.0:
        raise ValueError("contour must contain a positive value")
    vmin = 0.0
    image = ax.imshow(
        contour.T,
        origin="lower",
        extent=(-0.5, NX - 0.5, -0.5, TYPICAL_NY // 2 - 0.5),
        interpolation="nearest",
        aspect="equal",
        cmap=cmap,
        norm=mpl.colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax, clip=True),
    )
    ax.set_xticks(np.arange(0, NX, 5))
    ax.set_yticks(np.arange(0, TYPICAL_NY // 2, 5))
    ax.set_xticks(np.arange(-0.5, NX, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, TYPICAL_NY // 2, 1.0), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.22, alpha=0.28, zorder=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set(xlabel=r"$x$", ylabel=r"$y$")
    bar = (
        fig.colorbar(image, ax=ax, orientation=colorbar_orientation)
        if cax is None
        else fig.colorbar(image, cax=cax, orientation=colorbar_orientation)
    )
    bar.ax.tick_params(labelsize=5.5, pad=1.0)
    ticks = (vmin, 0.5 * (vmin + vmax), vmax)
    bar.set_ticks(ticks, labels=[f"{value:.2f}" for value in ticks])
    bar.set_label(colorbar_label, fontsize=6.0, labelpad=0.5)
    return {"vmin": vmin, "vmax": vmax, "gamma": 0.5}


def mean_sem(data: np.ndarray) -> tuple[float, float]:
    return float(data.mean()), float(data.std(ddof=1) / math.sqrt(data.size))


def mean_curve_fit(x: np.ndarray, curves: np.ndarray) -> dict[str, Any]:
    """Fit the ensemble-mean curve and propagate its full sample covariance."""

    x = np.asarray(x, dtype=np.float64)
    curves = np.asarray(curves, dtype=np.float64)
    if x.ndim != 1 or curves.ndim != 2 or curves.shape[1] != x.size:
        raise ValueError("mean-curve fit requires curves with shape (samples, points)")
    if curves.shape[0] < 2 or x.size < 2 or not np.all(np.isfinite(curves)):
        raise ValueError("mean-curve fit requires finite data and at least two samples/points")
    design = np.column_stack((x, np.ones_like(x)))
    projection = np.linalg.solve(design.T @ design, design.T)
    mean_curve = curves.mean(axis=0)
    slope, intercept = projection @ mean_curve
    fitted = slope * x + intercept
    residual = float(np.sum((mean_curve - fitted) ** 2))
    total = float(np.sum((mean_curve - mean_curve.mean()) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    covariance_of_mean = np.cov(curves, rowvar=False, ddof=1) / curves.shape[0]
    parameter_covariance = projection @ covariance_of_mean @ projection.T
    slope_sem = math.sqrt(max(0.0, float(parameter_covariance[0, 0])))
    intercept_sem = math.sqrt(max(0.0, float(parameter_covariance[1, 1])))
    pointwise_sem = curves.std(axis=0, ddof=1) / math.sqrt(curves.shape[0])

    # This identity is an audit of the linear estimator, not the production
    # evaluation order: the authoritative slope above is fitted to mean_curve.
    sample_slopes = curves @ projection[0]
    if not np.isclose(slope, sample_slopes.mean(), rtol=0.0, atol=2e-13):
        raise RuntimeError("fit(mean curve) does not match the OLS linearity audit")
    if not np.isclose(
        slope_sem,
        sample_slopes.std(ddof=1) / math.sqrt(sample_slopes.size),
        rtol=2e-12,
        atol=2e-14,
    ):
        raise RuntimeError("covariance-propagated SEM does not match its OLS audit")
    return {
        "x": x,
        "mean_curve": mean_curve,
        "pointwise_sem": pointwise_sem,
        "slope": float(slope),
        "slope_sem": slope_sem,
        "intercept": float(intercept),
        "intercept_sem": intercept_sem,
        "R_squared": float(r_squared),
        "slope_projection": projection[0],
    }


def mean_curve_results(
    analyzer: Any,
    cases: dict[int, dict[str, np.ndarray]],
) -> tuple[dict[int, dict[str, dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit every per-size ensemble mean and export its curve statistics."""

    fits: dict[int, dict[str, dict[str, Any]]] = {}
    curve_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for ny in NY_VALUES:
        case = cases[ny]
        ay = np.asarray(case["ay_values"], dtype=np.int64)
        fit_mask = ay >= FIT_MIN_AY
        x_fit = analyzer.log_chord(ay[fit_mask], ny)
        fits[ny] = {}
        for label in (*RENYI_LABELS, "k"):
            curves = np.asarray(case[analyzer.CURVE_KEYS[label]], dtype=np.float64)
            result = mean_curve_fit(x_fit, curves[:, fit_mask])
            factor = float(analyzer.PREFACTOR[label])
            result["converted"] = factor * result["slope"]
            result["converted_sem"] = factor * result["slope_sem"]
            fits[ny][label] = result
            full_mean = curves.mean(axis=0)
            full_sem = curves.std(axis=0, ddof=1) / math.sqrt(curves.shape[0])
            for index, width in enumerate(ay):
                curve_rows.append(
                    {
                        "Nx": NX,
                        "Ny": ny,
                        "samples": int(curves.shape[0]),
                        "observable": label,
                        "Ay": int(width),
                        "mean_curve_value": float(full_mean[index]),
                        "pointwise_sample_SEM": float(full_sem[index]),
                        "in_fit_window": bool(fit_mask[index]),
                    }
                )
            fit_rows.append(
                {
                    "Nx": NX,
                    "Ny": ny,
                    "samples": int(curves.shape[0]),
                    "observable": label,
                    "estimator_order": "average_curves_then_fit",
                    "Ay_fit_min": FIT_MIN_AY,
                    "Ay_fit_max": ny // 2,
                    "slope": result["slope"],
                    "slope_covariance_SEM": result["slope_sem"],
                    "intercept": result["intercept"],
                    "intercept_covariance_SEM": result["intercept_sem"],
                    "converted_coefficient": result["converted"],
                    "converted_covariance_SEM": result["converted_sem"],
                    "R_squared": result["R_squared"],
                    "CFT_slope_target": EXPECTED_SLOPES[label],
                }
            )
    return fits, curve_rows, fit_rows


def anchored_mean_curve_fit(
    analyzer: Any,
    cases: dict[int, dict[str, np.ndarray]],
    label: str,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Fit one equal-size-weighted slope to anchored ensemble-mean curves."""

    by_size: dict[int, dict[str, Any]] = {}
    groups: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    denominator = 0.0
    for ny in NY_VALUES:
        case = cases[ny]
        ay = np.asarray(case["ay_values"], dtype=np.int64)
        curves = np.asarray(case[analyzer.CURVE_KEYS[label]], dtype=np.float64)
        endpoint_indices = np.flatnonzero(ay == ny // 2)
        if endpoint_indices.size != 1:
            raise RuntimeError(f"Ny={ny} does not have one half-strip endpoint")
        endpoint_index = int(endpoint_indices[0])
        plotted = ay >= 1
        fit_mask = ay >= FIT_MIN_AY
        anchor_log_sin = float(np.log(np.sin(np.pi * (ny // 2) / ny)))
        x_all = np.log(np.sin(np.pi * ay[plotted] / ny)) - anchor_log_sin
        x_fit = np.log(np.sin(np.pi * ay[fit_mask] / ny)) - anchor_log_sin
        delta_all = curves[:, plotted] - curves[:, [endpoint_index]]
        delta_fit = curves[:, fit_mask] - curves[:, [endpoint_index]]
        mean_all = delta_all.mean(axis=0)
        sem_all = delta_all.std(axis=0, ddof=1) / math.sqrt(delta_all.shape[0])
        mean_fit = delta_fit.mean(axis=0)
        groups[ny] = (x_fit, mean_fit)
        size_weight = 1.0 / x_fit.size
        denominator += size_weight * float(np.dot(x_fit, x_fit))
        size_slope = float(np.dot(x_fit, mean_fit) / np.dot(x_fit, x_fit))
        size_projection = x_fit / np.dot(x_fit, x_fit)
        covariance_of_mean = np.cov(delta_fit, rowvar=False, ddof=1) / delta_fit.shape[0]
        size_sem = math.sqrt(max(0.0, float(size_projection @ covariance_of_mean @ size_projection)))
        size_residual = mean_fit - size_slope * x_fit
        size_r0 = 1.0 - float(np.dot(size_residual, size_residual)) / float(np.dot(mean_fit, mean_fit))
        by_size[ny] = {
            "x_all": x_all,
            "mean_all": mean_all,
            "sem_all": sem_all,
            "x_fit": x_fit,
            "delta_fit_samples": delta_fit,
            "endpoint_value_mean": float(curves[:, endpoint_index].mean()),
            "endpoint_value_SEM": float(curves[:, endpoint_index].std(ddof=1) / math.sqrt(curves.shape[0])),
            "anchor_Ay": ny // 2,
            "anchor_log_sin": anchor_log_sin,
            "size_slope": size_slope,
            "size_slope_SEM": size_sem,
            "size_R0_squared": size_r0,
        }

    slope, r0_squared, group_r0_squared = endpoint_anchored_fit(groups)
    variance = 0.0
    sample_contributions: list[np.ndarray] = []
    for ny in NY_VALUES:
        item = by_size[ny]
        x_fit = item["x_fit"]
        delta_fit = item["delta_fit_samples"]
        size_weight = 1.0 / x_fit.size
        projection = (size_weight / denominator) * x_fit
        covariance_of_mean = np.cov(delta_fit, rowvar=False, ddof=1) / delta_fit.shape[0]
        variance += float(projection @ covariance_of_mean @ projection)
        sample_contributions.append(delta_fit @ projection)
        item["group_R0_squared"] = group_r0_squared[ny]
    slope_sem = math.sqrt(max(0.0, variance))
    audit_sem = math.sqrt(
        sum(float(contribution.var(ddof=1) / contribution.size) for contribution in sample_contributions)
    )
    if not np.isclose(slope_sem, audit_sem, rtol=2e-12, atol=2e-14):
        raise RuntimeError("anchored covariance SEM does not match its linearity audit")
    factor = float(analyzer.PREFACTOR[label])
    summary = {
        "observable": label,
        "estimator_order": "average_curves_then_endpoint_anchor_then_fit",
        "slope": slope,
        "slope_covariance_SEM": slope_sem,
        "converted_coefficient": factor * slope,
        "converted_covariance_SEM": factor * slope_sem,
        "R0_squared": r0_squared,
    }
    return by_size, summary


def mean_curve_ratio_rows(
    analyzer: Any,
    cases: dict[int, dict[str, np.ndarray]],
    mean_fits: dict[int, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Form c_q/k from mean-curve slopes with their joint sample covariance."""

    rows: list[dict[str, Any]] = []
    for ny in NY_VALUES:
        case = cases[ny]
        ay = np.asarray(case["ay_values"], dtype=np.int64)
        fit_mask = ay >= FIT_MIN_AY
        k_fit = mean_fits[ny]["k"]
        f_curves = np.asarray(case[analyzer.CURVE_KEYS["k"]], dtype=np.float64)[:, fit_mask]
        for label in RENYI_LABELS:
            q_fit = mean_fits[ny][label]
            q_curves = np.asarray(case[analyzer.CURVE_KEYS[label]], dtype=np.float64)[:, fit_mask]
            q_factor = float(analyzer.PREFACTOR[label])
            k_factor = float(analyzer.PREFACTOR["k"])
            q_contrast = q_factor * (q_curves @ q_fit["slope_projection"])
            k_contrast = k_factor * (f_curves @ k_fit["slope_projection"])
            covariance_of_mean = np.cov(np.column_stack((q_contrast, k_contrast)), rowvar=False, ddof=1) / q_contrast.size
            c_value = float(q_fit["converted"])
            k_value = float(k_fit["converted"])
            ratio = c_value / k_value
            gradient = np.asarray([1.0 / k_value, -c_value / (k_value**2)])
            ratio_sem = math.sqrt(max(0.0, float(gradient @ covariance_of_mean @ gradient)))
            rows.append(
                {
                    "Nx": NX,
                    "Ny": ny,
                    "samples": int(q_contrast.size),
                    "renyi_order": int(label[-1]),
                    "entropy_coefficient": c_value,
                    "charge_level": k_value,
                    "mean_curve_coefficient_ratio": ratio,
                    "joint_covariance_SEM": ratio_sem,
                    "uncertainty_method": "delta_method_from_joint_trajectory_curve_covariance",
                }
            )
    return rows


def make_samplewise_cq_k_scaling_figure(
    values: dict[int, dict[str, np.ndarray]],
) -> None:
    """Plot means of the trajectory-wise fitted CFT coefficients."""

    configure_matplotlib()
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.linewidth": 0.8,
            "lines.linewidth": 0.9,
            "lines.markersize": 4.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "legend.fontsize": 6.2,
        }
    )
    fig, ax = plt.subplots(1, 1, figsize=(3.375, 2.65))
    labels = (*RENYI_LABELS, "k")
    for label in labels:
        summaries = [mean_sem(values[ny][label]) for ny in NY_VALUES]
        display = rf"$\langle c_{label[-1]}\rangle_\xi$" if label != "k" else r"$\langle k\rangle_\xi$"
        ax.errorbar(
            NY_VALUES,
            [item[0] for item in summaries],
            yerr=[item[1] for item in summaries],
            color=COLORS[label],
            marker=MARKERS[label],
            linestyle="--",
            linewidth=0.8,
            markerfacecolor="white",
            markeredgewidth=0.8,
            capsize=1.5,
            capthick=0.7,
            zorder=3,
            label=display,
        )
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, zorder=1)
    ax.set(
        xlabel=r"$N_y$",
        ylabel="mean fitted coefficient",
        title=r"trajectory-wise endpoint fits, $S=100$",
        xticks=NY_VALUES,
    )
    ax.legend(
        ncol=2,
        loc="upper right",
        columnspacing=0.8,
        handlelength=1.8,
        handletextpad=0.35,
        borderaxespad=0.35,
    )
    if not np.allclose(fig.get_size_inches(), (3.375, 2.65), rtol=0.0, atol=1e-12):
        raise RuntimeError("ensemble scaling figure does not have the locked 3.375 x 2.65 inch size")
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.19, top=0.91)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ENSEMBLE_SCALING_STEM.with_suffix(".pdf"))
    fig.savefig(ENSEMBLE_SCALING_STEM.with_suffix(".png"), dpi=300)
    plt.close(fig)


def make_entropy_figure(
    analyzer: Any,
    cases: dict[int, dict[str, np.ndarray]],
    values: dict[int, dict[str, np.ndarray]],
    typical_index: int,
    typical_fits: dict[str, tuple[float, float, float]],
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=(3.375, 7.05))
    case = cases[TYPICAL_NY]
    sample_id = int(case["sample_ids"][typical_index])

    ax = axes[0]
    ay = case["ay_values"]
    plotted = ay >= 1
    x_all = analyzer.log_chord(ay[plotted], TYPICAL_NY)
    fit_mask = ay >= FIT_MIN_AY
    x_fit = analyzer.log_chord(ay[fit_mask], TYPICAL_NY)
    x_line = np.linspace(float(x_all.min()), float(x_all.max()), 300)
    ax.axvspan(float(x_fit.min()), float(x_fit.max()), color="0.5", alpha=0.18, zorder=0)
    for label in RENYI_LABELS:
        curve = case[analyzer.CURVE_KEYS[label]][typical_index]
        slope, intercept, r_squared = typical_fits[label]
        converted = analyzer.PREFACTOR[label] * slope
        ax.plot(
            x_all,
            curve[plotted],
            color=COLORS[label],
            marker=MARKERS[label],
            linestyle="none",
            markerfacecolor="white",
            markeredgewidth=0.65,
            zorder=3,
            label=(
                rf"$q={label[-1]}$: $m={slope:.3f}$, "
                rf"$c_{label[-1]}={converted:.3f}$, $R^2={r_squared:.6f}$"
            ),
        )
        ax.plot(x_line, slope * x_line + intercept, color="black", linestyle="--", linewidth=0.9, zorder=2)
    ax.set(xlabel=r"$X=\log[(N_y/\pi)\sin(\pi A_y/N_y)]$", ylabel=r"$S_q(A_y)$")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(0.015, 0.57),
        handlelength=1.2,
        borderaxespad=0.0,
        labelspacing=0.25,
    )
    ax.set_title(rf"$N_y=60$, sample {sample_id}; fits $A_y=8,\ldots,30$")
    panel_letter(ax, "(a)")

    ax = axes[1]
    offsets = {"c1": -0.35, "c2": 0.0, "c3": 0.35}
    for label in RENYI_LABELS:
        summaries = [mean_sem(values[ny][label]) for ny in NY_VALUES]
        ax.errorbar(
            np.asarray(NY_VALUES) + offsets[label],
            [item[0] for item in summaries],
            yerr=[item[1] for item in summaries],
            color=COLORS[label],
            marker=MARKERS[label],
            linestyle="none",
            markerfacecolor="white",
            capsize=1.4,
            label=rf"$c_{label[-1]}$",
        )
    ax.axhline(1.0, color="black", ls="--", lw=0.75)
    ax.set(xlabel=r"$N_y$", ylabel=r"mean sample-wise $c_q$", title=r"Cardy--Calabrese conversion; error: SEM")
    ax.legend(ncol=3, loc="upper right", columnspacing=0.6, handletextpad=0.25)
    panel_letter(ax, "(b)")

    ax = axes[2]
    for label in RENYI_LABELS:
        summaries = [mean_sem(values[ny][f"m_{label}"]) for ny in NY_VALUES]
        ax.errorbar(
            NY_VALUES,
            [item[0] for item in summaries],
            yerr=[item[1] for item in summaries],
            color=COLORS[label],
            marker=MARKERS[label],
            linestyle="none",
            markerfacecolor="white",
            capsize=1.4,
            label=rf"$m_{label[-1]}$; CFT $={EXPECTED_SLOPES[label]:.3f}$",
        )
        ax.axhline(EXPECTED_SLOPES[label], color=COLORS[label], ls=LINESTYLES[label], lw=0.6, alpha=0.5)
    ax.set(xlabel=r"$N_y$", ylabel=r"mean sample-wise slope $m_q$", title="raw log-chord slopes")
    ax.legend(loc="center right", handlelength=1.3, labelspacing=0.25)
    panel_letter(ax, "(c)")

    fig.suptitle(
        rf"Hard-wall endpoint entropy, $N_x=20$, $S=100$, $t=2N_y$",
        fontsize=8.0,
        y=0.995,
    )
    if not np.allclose(fig.get_size_inches(), (3.375, 7.05), rtol=0.0, atol=1e-12):
        raise RuntimeError("entropy figure does not have the locked 3.375 x 7.05 inch size")
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.075, top=0.945, hspace=0.47)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ENTROPY_STEM.with_suffix(".pdf"))
    fig.savefig(ENTROPY_STEM.with_suffix(".png"), dpi=300)
    plt.close(fig)


def make_entropy_size_collapse_figure(
    analyzer: Any,
    cases: dict[int, dict[str, np.ndarray]],
    values: dict[int, dict[str, np.ndarray]],
    typical_by_size: dict[int, tuple[int, np.ndarray]],
) -> list[dict[str, Any]]:
    """Overlay one deterministic representative trajectory at every Ny."""

    configure_matplotlib()
    # Match the visual grammar used throughout the legacy evidence atlas:
    # Times/CM text, boxed inward-tick axes, open empirical markers, black
    # dashed fit models, and a pale-gray declared fit domain.  The three
    # canonical BPJ anchors (smallest/intermediate/largest) remain red
    # triangle, green square, and blue circle; the intervening sizes use only
    # the atlas's auxiliary orange, gray, and light-blue palette.
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "lines.markersize": 4.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "legend.fontsize": 6.0,
        }
    )
    size_styles = {
        30: {"color": "#D92725", "marker": "^"},
        35: {"color": "#F08050", "marker": "<"},
        40: {"color": "#8FC1E3", "marker": "v"},
        45: {"color": "#2CA02C", "marker": "s"},
        50: {"color": "#6B6B6B", "marker": "D"},
        55: {"color": "#000000", "marker": "P"},
        60: {"color": "#1F77B4", "marker": "o"},
    }
    panel_titles = {
        "c1": "von Neumann entropy",
        "c2": r"Rényi-$2$ entropy",
        "c3": r"Rényi-$3$ entropy",
    }
    fig, axes = plt.subplots(3, 1, figsize=(3.375, 7.05), sharex=False)
    fit_rows: list[dict[str, Any]] = []
    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    for panel, (ax, label) in enumerate(zip(axes, RENYI_LABELS)):
        plot_data: dict[int, dict[str, Any]] = {}
        fit_groups: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for ny in NY_VALUES:
            typical_index, distances = typical_by_size[ny]
            case = cases[ny]
            sample_id = int(case["sample_ids"][typical_index])
            ay = case["ay_values"]
            plotted = ay >= 1
            fit_mask = ay >= FIT_MIN_AY
            curve = case[analyzer.CURVE_KEYS[label]][typical_index]
            endpoint_mask = ay == ny // 2
            if np.count_nonzero(endpoint_mask) != 1:
                raise RuntimeError(f"Ny={ny} does not have one half-strip endpoint")
            endpoint_entropy = float(curve[endpoint_mask][0])
            endpoint_ay = ny // 2
            endpoint_log_sin = float(np.log(np.sin(np.pi * endpoint_ay / ny)))
            x_all = np.log(np.sin(np.pi * ay[plotted] / ny)) - endpoint_log_sin
            x_fit = np.log(np.sin(np.pi * ay[fit_mask] / ny)) - endpoint_log_sin
            y_all = curve[plotted] - endpoint_entropy
            y_fit = curve[fit_mask] - endpoint_entropy
            fit_groups[ny] = (x_fit, y_fit)
            plot_data[ny] = {
                "typical_index": typical_index,
                "selection_distance": float(distances[typical_index]),
                "sample_id": sample_id,
                "endpoint_ay": endpoint_ay,
                "endpoint_log_sin": endpoint_log_sin,
                "endpoint_entropy": endpoint_entropy,
                "x_all": x_all,
                "y_all": y_all,
                "x_fit": x_fit,
                "y_fit": y_fit,
            }

        anchored_slope, anchored_r_squared, group_r_squared = endpoint_anchored_fit(
            fit_groups
        )
        converted = float(analyzer.PREFACTOR[label] * anchored_slope)
        fit_domain_min = min(float(item[0].min()) for item in fit_groups.values())
        ax.axvspan(
            fit_domain_min,
            0.0,
            color="0.5",
            alpha=0.15,
            linewidth=0,
            zorder=0,
        )
        for ny in NY_VALUES:
            item = plot_data[ny]
            style = size_styles[ny]
            empirical = ax.plot(
                item["x_all"],
                item["y_all"],
                color=style["color"],
                marker=style["marker"],
                linestyle="none",
                markerfacecolor="white",
                markeredgewidth=0.8,
                markersize=3.4,
                alpha=1.0,
                zorder=3,
            )[0]
            if panel == 0:
                legend_handles.append(empirical)
                legend_labels.append(rf"$N_y={ny}$")
            anchored_size_slope = float(
                np.dot(item["x_fit"], item["y_fit"])
                / np.dot(item["x_fit"], item["x_fit"])
            )
            anchored_size_residual = item["y_fit"] - anchored_size_slope * item["x_fit"]
            anchored_size_r_squared = float(
                1.0
                - np.dot(anchored_size_residual, anchored_size_residual)
                / np.dot(item["y_fit"], item["y_fit"])
            )
            fit_rows.append(
                {
                    "Nx": NX,
                    "Ny": ny,
                    "sample_id": item["sample_id"],
                    "selection_distance": item["selection_distance"],
                    "observable": label,
                    "Ay_fit_min": FIT_MIN_AY,
                    "Ay_fit_max": ny // 2,
                    "anchor_Ay": item["endpoint_ay"],
                    "anchor_log_sin": item["endpoint_log_sin"],
                    "half_strip_entropy": item["endpoint_entropy"],
                    "anchored_shared_slope": anchored_slope,
                    "anchored_shared_converted_prefactor": converted,
                    "anchored_shared_R0_squared": anchored_r_squared,
                    "anchored_shared_group_R0_squared": group_r_squared[ny],
                    "anchored_size_slope": anchored_size_slope,
                    "anchored_size_R0_squared": anchored_size_r_squared,
                    "anchored_size_converted_prefactor": float(
                        analyzer.PREFACTOR[label] * anchored_size_slope
                    ),
                    "production_unconstrained_slope": float(
                        values[ny][f"m_{label}"][item["typical_index"]]
                    ),
                    "production_unconstrained_converted_prefactor": float(
                        values[ny][label][item["typical_index"]]
                    ),
                    "CFT_slope_target": EXPECTED_SLOPES[label],
                }
            )

        x_line = np.linspace(
            min(float(item["x_all"].min()) for item in plot_data.values()), 0.0, 300
        )
        ax.plot(
            x_line,
            anchored_slope * x_line,
            color="black",
            linestyle="--",
            linewidth=0.9,
            zorder=2,
        )
        ax.set_ylabel(rf"$\Delta S_{label[-1]}(A_y)$")
        ax.set_xlabel(r"$\log\!\left[\sin(\pi A_y/N_y)/\sin(\pi A_y^\star/N_y)\right]$")
        ax.set_xlim(-3.05, 0.05)
        ax.set_title(panel_titles[label], pad=4)
        ax.text(
            0.025,
            0.93,
            (
                rf"endpoint-anchored fit: $c_{label[-1]}={converted:.3f}$, "
                rf"$R_0^2={anchored_r_squared:.6f}$"
            ),
            transform=ax.transAxes,
            fontsize=6.0,
            va="top",
        )
        panel_letter(ax, f"({chr(ord('a') + panel)})")

    axes[0].legend(
        legend_handles,
        legend_labels,
        ncol=2,
        loc="lower right",
        columnspacing=0.7,
        handletextpad=0.3,
        borderaxespad=0.45,
    )
    if not np.allclose(fig.get_size_inches(), (3.375, 7.05), rtol=0.0, atol=1e-12):
        raise RuntimeError("collapse figure does not have the locked 3.375 x 7.05 inch size")
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.072, top=0.958, hspace=0.48)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(COLLAPSE_STEM.with_suffix(".pdf"))
    fig.savefig(COLLAPSE_STEM.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return fit_rows


def make_charge_size_collapse_figure(
    analyzer: Any,
    cases: dict[int, dict[str, np.ndarray]],
    values: dict[int, dict[str, np.ndarray]],
    typical_by_size: dict[int, tuple[int, np.ndarray]],
) -> list[dict[str, Any]]:
    """Make the charge-variance analogue of the entropy collapse."""

    configure_matplotlib()
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "lines.markersize": 4.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "legend.fontsize": 6.0,
        }
    )
    size_styles = {
        30: {"color": "#D92725", "marker": "^"},
        35: {"color": "#F08050", "marker": "<"},
        40: {"color": "#8FC1E3", "marker": "v"},
        45: {"color": "#2CA02C", "marker": "s"},
        50: {"color": "#6B6B6B", "marker": "D"},
        55: {"color": "#000000", "marker": "P"},
        60: {"color": "#1F77B4", "marker": "o"},
    }
    fig, ax = plt.subplots(1, 1, figsize=(3.375, 2.65))
    plot_data: dict[int, dict[str, Any]] = {}
    fit_groups: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for ny in NY_VALUES:
        typical_index, distances = typical_by_size[ny]
        case = cases[ny]
        sample_id = int(case["sample_ids"][typical_index])
        ay = case["ay_values"]
        plotted = ay >= 1
        fit_mask = ay >= FIT_MIN_AY
        curve = case[analyzer.CURVE_KEYS["k"]][typical_index]
        endpoint_mask = ay == ny // 2
        if np.count_nonzero(endpoint_mask) != 1:
            raise RuntimeError(f"Ny={ny} does not have one half-strip endpoint")
        endpoint_variance = float(curve[endpoint_mask][0])
        endpoint_ay = ny // 2
        endpoint_log_sin = float(np.log(np.sin(np.pi * endpoint_ay / ny)))
        x_all = np.log(np.sin(np.pi * ay[plotted] / ny)) - endpoint_log_sin
        x_fit = np.log(np.sin(np.pi * ay[fit_mask] / ny)) - endpoint_log_sin
        y_all = curve[plotted] - endpoint_variance
        y_fit = curve[fit_mask] - endpoint_variance
        fit_groups[ny] = (x_fit, y_fit)
        plot_data[ny] = {
            "typical_index": typical_index,
            "selection_distance": float(distances[typical_index]),
            "sample_id": sample_id,
            "endpoint_ay": endpoint_ay,
            "endpoint_log_sin": endpoint_log_sin,
            "endpoint_variance": endpoint_variance,
            "x_all": x_all,
            "y_all": y_all,
            "x_fit": x_fit,
            "y_fit": y_fit,
        }

    anchored_slope, anchored_r_squared, group_r_squared = endpoint_anchored_fit(
        fit_groups
    )
    level = float(math.pi**2 * anchored_slope)
    fit_domain_min = min(float(item[0].min()) for item in fit_groups.values())
    ax.axvspan(fit_domain_min, 0.0, color="0.5", alpha=0.15, linewidth=0, zorder=0)
    fit_rows: list[dict[str, Any]] = []
    for ny in NY_VALUES:
        item = plot_data[ny]
        style = size_styles[ny]
        ax.plot(
            item["x_all"],
            item["y_all"],
            color=style["color"],
            marker=style["marker"],
            linestyle="none",
            markerfacecolor="white",
            markeredgewidth=0.8,
            markersize=3.4,
            alpha=1.0,
            zorder=3,
            label=rf"$N_y={ny}$",
        )
        anchored_size_slope = float(
            np.dot(item["x_fit"], item["y_fit"])
            / np.dot(item["x_fit"], item["x_fit"])
        )
        anchored_size_residual = item["y_fit"] - anchored_size_slope * item["x_fit"]
        anchored_size_r_squared = float(
            1.0
            - np.dot(anchored_size_residual, anchored_size_residual)
            / np.dot(item["y_fit"], item["y_fit"])
        )
        fit_rows.append(
            {
                "Nx": NX,
                "Ny": ny,
                "sample_id": item["sample_id"],
                "selection_distance": item["selection_distance"],
                "observable": "k",
                "Ay_fit_min": FIT_MIN_AY,
                "Ay_fit_max": ny // 2,
                "anchor_Ay": item["endpoint_ay"],
                "anchor_log_sin": item["endpoint_log_sin"],
                "half_strip_charge_variance": item["endpoint_variance"],
                "anchored_shared_slope": anchored_slope,
                "anchored_shared_converted_level": level,
                "anchored_shared_R0_squared": anchored_r_squared,
                "anchored_shared_group_R0_squared": group_r_squared[ny],
                "anchored_size_slope": anchored_size_slope,
                "anchored_size_R0_squared": anchored_size_r_squared,
                "anchored_size_converted_level": float(math.pi**2 * anchored_size_slope),
                "production_unconstrained_slope": float(
                    values[ny]["m_k"][item["typical_index"]]
                ),
                "production_unconstrained_converted_level": float(
                    values[ny]["k"][item["typical_index"]]
                ),
                "CFT_slope_target": EXPECTED_SLOPES["k"],
            }
        )

    x_line = np.linspace(
        min(float(item["x_all"].min()) for item in plot_data.values()), 0.0, 300
    )
    ax.plot(
        x_line,
        anchored_slope * x_line,
        color="black",
        linestyle="--",
        linewidth=0.9,
        zorder=2,
    )
    ax.set(
        xlim=(-3.05, 0.05),
        xlabel=r"$\log\!\left[\sin(\pi A_y/N_y)/\sin(\pi A_y^\star/N_y)\right]$",
        ylabel=r"$\Delta F_A(A_y)$",
        title="intrinsic subsystem charge variance",
    )
    ax.text(
        0.025,
        0.93,
        rf"endpoint-anchored fit: $k={level:.3f}$, $R_0^2={anchored_r_squared:.6f}$",
        transform=ax.transAxes,
        fontsize=6.0,
        va="top",
    )
    ax.legend(
        ncol=2,
        loc="lower right",
        columnspacing=0.7,
        handletextpad=0.3,
        borderaxespad=0.45,
    )
    if not np.allclose(fig.get_size_inches(), (3.375, 2.65), rtol=0.0, atol=1e-12):
        raise RuntimeError("charge collapse figure does not have the locked 3.375 x 2.65 inch size")
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.19, top=0.91)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARGE_COLLAPSE_STEM.with_suffix(".pdf"))
    fig.savefig(CHARGE_COLLAPSE_STEM.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return fit_rows


def make_mean_entropy_collapse_figure(
    analyzer: Any,
    anchored: dict[str, tuple[dict[int, dict[str, Any]], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Plot ensemble-mean endpoint-anchored entropy curves."""

    configure_matplotlib()
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "lines.markersize": 4.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "legend.fontsize": 6.0,
        }
    )
    size_styles = {
        30: {"color": "#D92725", "marker": "^"},
        35: {"color": "#F08050", "marker": "<"},
        40: {"color": "#8FC1E3", "marker": "v"},
        45: {"color": "#2CA02C", "marker": "s"},
        50: {"color": "#6B6B6B", "marker": "D"},
        55: {"color": "#000000", "marker": "P"},
        60: {"color": "#1F77B4", "marker": "o"},
    }
    panel_titles = {
        "c1": "ensemble-mean von Neumann entropy",
        "c2": r"ensemble-mean Rényi-$2$ entropy",
        "c3": r"ensemble-mean Rényi-$3$ entropy",
    }
    fig, axes = plt.subplots(3, 1, figsize=(3.375, 7.05), sharex=False)
    rows: list[dict[str, Any]] = []
    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    for panel, (ax, label) in enumerate(zip(axes, RENYI_LABELS)):
        by_size, summary = anchored[label]
        fit_domain_min = min(float(item["x_fit"].min()) for item in by_size.values())
        ax.axvspan(fit_domain_min, 0.0, color="0.5", alpha=0.15, linewidth=0, zorder=0)
        for ny in NY_VALUES:
            item = by_size[ny]
            style = size_styles[ny]
            empirical = ax.errorbar(
                item["x_all"],
                item["mean_all"],
                yerr=item["sem_all"],
                color=style["color"],
                marker=style["marker"],
                linestyle="none",
                markerfacecolor="white",
                markeredgewidth=0.8,
                markersize=3.4,
                elinewidth=0.45,
                capsize=0.0,
                alpha=1.0,
                zorder=3,
            )
            if panel == 0:
                legend_handles.append(empirical[0])
                legend_labels.append(rf"$N_y={ny}$")
            rows.append(
                {
                    "Nx": NX,
                    "Ny": ny,
                    "samples": 100,
                    "observable": label,
                    "estimator_order": summary["estimator_order"],
                    "Ay_fit_min": FIT_MIN_AY,
                    "Ay_fit_max": ny // 2,
                    "anchor_Ay": item["anchor_Ay"],
                    "anchor_log_sin": item["anchor_log_sin"],
                    "endpoint_value_mean": item["endpoint_value_mean"],
                    "endpoint_value_sample_SEM": item["endpoint_value_SEM"],
                    "shared_slope": summary["slope"],
                    "shared_slope_covariance_SEM": summary["slope_covariance_SEM"],
                    "shared_converted_coefficient": summary["converted_coefficient"],
                    "shared_converted_covariance_SEM": summary["converted_covariance_SEM"],
                    "shared_R0_squared": summary["R0_squared"],
                    "group_R0_squared": item["group_R0_squared"],
                    "size_slope": item["size_slope"],
                    "size_slope_covariance_SEM": item["size_slope_SEM"],
                    "size_converted_coefficient": float(analyzer.PREFACTOR[label] * item["size_slope"]),
                    "size_converted_covariance_SEM": float(analyzer.PREFACTOR[label] * item["size_slope_SEM"]),
                    "size_R0_squared": item["size_R0_squared"],
                }
            )
        x_line = np.linspace(
            min(float(item["x_all"].min()) for item in by_size.values()), 0.0, 300
        )
        ax.plot(x_line, summary["slope"] * x_line, color="black", linestyle="--", linewidth=0.9, zorder=2)
        q = label[-1]
        ax.set(
            xlim=(-3.05, 0.05),
            xlabel=r"$\log\!\left[\sin(\pi A_y/N_y)/\sin(\pi A_y^\star/N_y)\right]$",
            ylabel=rf"$\Delta\langle \overline{{S}}_{q}\rangle_\xi$",
            title=panel_titles[label],
        )
        ax.text(
            0.025,
            0.93,
            (
                rf"mean-curve fit: $c_{q}={summary['converted_coefficient']:.4f}"
                rf"\pm{summary['converted_covariance_SEM']:.4f}$, "
                rf"$R_0^2={summary['R0_squared']:.6f}$"
            ),
            transform=ax.transAxes,
            fontsize=5.7,
            va="top",
        )
        panel_letter(ax, f"({chr(ord('a') + panel)})")

    axes[0].legend(
        legend_handles,
        legend_labels,
        ncol=2,
        loc="lower right",
        columnspacing=0.7,
        handletextpad=0.3,
        borderaxespad=0.45,
    )
    if not np.allclose(fig.get_size_inches(), (3.375, 7.05), rtol=0.0, atol=1e-12):
        raise RuntimeError("mean entropy collapse does not have the locked 3.375 x 7.05 inch size")
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.072, top=0.958, hspace=0.48)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEAN_ENTROPY_COLLAPSE_STEM.with_suffix(".pdf"))
    fig.savefig(MEAN_ENTROPY_COLLAPSE_STEM.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return rows


def make_mean_charge_collapse_figure(
    anchored_charge: tuple[dict[int, dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Plot the ensemble-mean endpoint-anchored charge variance."""

    configure_matplotlib()
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "lines.markersize": 4.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "legend.fontsize": 6.0,
        }
    )
    size_styles = {
        30: {"color": "#D92725", "marker": "^"},
        35: {"color": "#F08050", "marker": "<"},
        40: {"color": "#8FC1E3", "marker": "v"},
        45: {"color": "#2CA02C", "marker": "s"},
        50: {"color": "#6B6B6B", "marker": "D"},
        55: {"color": "#000000", "marker": "P"},
        60: {"color": "#1F77B4", "marker": "o"},
    }
    by_size, summary = anchored_charge
    fig, ax = plt.subplots(1, 1, figsize=(3.375, 2.65))
    fit_domain_min = min(float(item["x_fit"].min()) for item in by_size.values())
    ax.axvspan(fit_domain_min, 0.0, color="0.5", alpha=0.15, linewidth=0, zorder=0)
    rows: list[dict[str, Any]] = []
    for ny in NY_VALUES:
        item = by_size[ny]
        style = size_styles[ny]
        ax.errorbar(
            item["x_all"],
            item["mean_all"],
            yerr=item["sem_all"],
            color=style["color"],
            marker=style["marker"],
            linestyle="none",
            markerfacecolor="white",
            markeredgewidth=0.8,
            markersize=3.4,
            elinewidth=0.45,
            capsize=0.0,
            alpha=1.0,
            zorder=3,
            label=rf"$N_y={ny}$",
        )
        rows.append(
            {
                "Nx": NX,
                "Ny": ny,
                "samples": 100,
                "observable": "k",
                "estimator_order": summary["estimator_order"],
                "Ay_fit_min": FIT_MIN_AY,
                "Ay_fit_max": ny // 2,
                "anchor_Ay": item["anchor_Ay"],
                "anchor_log_sin": item["anchor_log_sin"],
                "endpoint_value_mean": item["endpoint_value_mean"],
                "endpoint_value_sample_SEM": item["endpoint_value_SEM"],
                "shared_slope": summary["slope"],
                "shared_slope_covariance_SEM": summary["slope_covariance_SEM"],
                "shared_converted_coefficient": summary["converted_coefficient"],
                "shared_converted_covariance_SEM": summary["converted_covariance_SEM"],
                "shared_R0_squared": summary["R0_squared"],
                "group_R0_squared": item["group_R0_squared"],
                "size_slope": item["size_slope"],
                "size_slope_covariance_SEM": item["size_slope_SEM"],
                "size_converted_coefficient": math.pi**2 * item["size_slope"],
                "size_converted_covariance_SEM": math.pi**2 * item["size_slope_SEM"],
                "size_R0_squared": item["size_R0_squared"],
            }
        )
    x_line = np.linspace(
        min(float(item["x_all"].min()) for item in by_size.values()), 0.0, 300
    )
    ax.plot(x_line, summary["slope"] * x_line, color="black", linestyle="--", linewidth=0.9, zorder=2)
    ax.set(
        xlim=(-3.05, 0.05),
        xlabel=r"$\log\!\left[\sin(\pi A_y/N_y)/\sin(\pi A_y^\star/N_y)\right]$",
        ylabel=r"$\Delta\langle \overline{F}_A\rangle_\xi$",
        title="ensemble-mean intrinsic charge variance",
    )
    ax.text(
        0.025,
        0.93,
        (
            rf"mean-curve fit: $k={summary['converted_coefficient']:.4f}"
            rf"\pm{summary['converted_covariance_SEM']:.4f}$, "
            rf"$R_0^2={summary['R0_squared']:.6f}$"
        ),
        transform=ax.transAxes,
        fontsize=5.7,
        va="top",
    )
    ax.legend(
        ncol=2,
        loc="lower right",
        columnspacing=0.7,
        handletextpad=0.3,
        borderaxespad=0.45,
    )
    if not np.allclose(fig.get_size_inches(), (3.375, 2.65), rtol=0.0, atol=1e-12):
        raise RuntimeError("mean charge collapse does not have the locked 3.375 x 2.65 inch size")
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.19, top=0.91)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEAN_CHARGE_COLLAPSE_STEM.with_suffix(".pdf"))
    fig.savefig(MEAN_CHARGE_COLLAPSE_STEM.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return rows


def make_mean_curve_scaling_figure(
    mean_fits: dict[int, dict[str, dict[str, Any]]],
) -> None:
    """Plot coefficients obtained from the per-size ensemble-mean curves."""

    configure_matplotlib()
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.linewidth": 0.8,
            "lines.linewidth": 0.9,
            "lines.markersize": 4.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "legend.fontsize": 6.2,
        }
    )
    fig, ax = plt.subplots(1, 1, figsize=(3.375, 2.65))
    for label in (*RENYI_LABELS, "k"):
        display = rf"$c_{label[-1]}$" if label != "k" else r"$k$"
        ax.errorbar(
            NY_VALUES,
            [mean_fits[ny][label]["converted"] for ny in NY_VALUES],
            yerr=[mean_fits[ny][label]["converted_sem"] for ny in NY_VALUES],
            color=COLORS[label],
            marker=MARKERS[label],
            linestyle="--",
            linewidth=0.8,
            markerfacecolor="white",
            markeredgewidth=0.8,
            capsize=1.5,
            capthick=0.7,
            zorder=3,
            label=display,
        )
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, zorder=1)
    ax.set(
        xlabel=r"$N_y$",
        ylabel="coefficient from mean curve",
        title=r"fits to ensemble-mean endpoint curves, $S=100$",
        xticks=NY_VALUES,
    )
    ax.legend(
        ncol=2,
        loc="upper right",
        columnspacing=0.8,
        handlelength=1.8,
        handletextpad=0.35,
        borderaxespad=0.35,
    )
    if not np.allclose(fig.get_size_inches(), (3.375, 2.65), rtol=0.0, atol=1e-12):
        raise RuntimeError("mean-curve scaling figure does not have the locked 3.375 x 2.65 inch size")
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.19, top=0.91)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEAN_SCALING_STEM.with_suffix(".pdf"))
    fig.savefig(MEAN_SCALING_STEM.with_suffix(".png"), dpi=300)
    plt.close(fig)


def make_charge_figure(
    analyzer: Any,
    cases: dict[int, dict[str, np.ndarray]],
    values: dict[int, dict[str, np.ndarray]],
    typical_index: int,
    typical_fits: dict[str, tuple[float, float, float]],
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=(3.375, 7.05))
    case = cases[TYPICAL_NY]
    sample_id = int(case["sample_ids"][typical_index])

    ax = axes[0]
    ay = case["ay_values"]
    plotted = ay >= 1
    fit_mask = ay >= FIT_MIN_AY
    x_all = analyzer.log_chord(ay[plotted], TYPICAL_NY)
    x_fit = analyzer.log_chord(ay[fit_mask], TYPICAL_NY)
    x_line = np.linspace(float(x_all.min()), float(x_all.max()), 300)
    ax.axvspan(float(x_fit.min()), float(x_fit.max()), color="0.5", alpha=0.18, zorder=0)
    curve = case[analyzer.CURVE_KEYS["k"]][typical_index]
    slope, intercept, r_squared = typical_fits["k"]
    level = math.pi**2 * slope
    ax.plot(
        x_all,
        curve[plotted],
        color=COLORS["k"],
        marker=MARKERS["k"],
        linestyle="none",
        markerfacecolor="white",
        markeredgewidth=0.65,
        zorder=3,
        label=rf"$m_F={slope:.4f}$, $k={level:.3f}$, $R^2={r_squared:.6f}$",
    )
    ax.plot(x_line, slope * x_line + intercept, color="black", linestyle="--", linewidth=0.9, zorder=2)
    ax.set(
        xlabel=r"$X=\log[(N_y/\pi)\sin(\pi A_y/N_y)]$",
        ylabel=r"$F_A(A_y)$",
        title=rf"$N_y=60$, sample {sample_id}; fit $A_y=8,\ldots,30$",
    )
    ax.legend(loc="upper left", handlelength=1.2, borderaxespad=0.2)
    panel_letter(ax, "(a)")

    ax = axes[1]
    summaries = [mean_sem(values[ny]["k"]) for ny in NY_VALUES]
    ax.errorbar(
        NY_VALUES,
        [item[0] for item in summaries],
        yerr=[item[1] for item in summaries],
        color=COLORS["k"],
        marker=MARKERS["k"],
        linestyle="none",
        markerfacecolor="white",
        capsize=1.4,
        label=r"$k=\pi^2m_F$",
    )
    ax.axhline(1.0, color="black", ls="--", lw=0.75, label="CFT target")
    ax.set(xlabel=r"$N_y$", ylabel=r"mean sample-wise $k$", title=r"charge level; error: SEM")
    ax.legend(loc="upper right", handlelength=1.3)
    panel_letter(ax, "(b)")

    ax = axes[2]
    for label in RENYI_LABELS:
        ratios_by_ny = []
        sem_by_ny = []
        for ny in NY_VALUES:
            ratio = values[ny][label] / values[ny]["k"]
            mean, sem = mean_sem(ratio)
            ratios_by_ny.append(mean)
            sem_by_ny.append(sem)
        ax.errorbar(
            NY_VALUES,
            ratios_by_ny,
            yerr=sem_by_ny,
            color=COLORS[label],
            marker=MARKERS[label],
            linestyle="none",
            markerfacecolor="white",
            capsize=1.4,
            label=rf"$\mathcal{{R}}_{label[-1]}$",
        )
    ax.axhline(1.0, color="black", ls="--", lw=0.75)
    ax.set(
        xlabel=r"$N_y$",
        ylabel=r"mean sample-wise $\mathcal{R}_q$",
        title="normalized entropy--charge slope ratio",
    )
    ax.legend(ncol=3, loc="upper right", columnspacing=0.55, handletextpad=0.25)
    panel_letter(ax, "(c)")

    fig.suptitle(
        rf"Hard-wall endpoint charge variance, $N_x=20$, $S=100$, $t=2N_y$",
        fontsize=8.0,
        y=0.995,
    )
    if not np.allclose(fig.get_size_inches(), (3.375, 7.05), rtol=0.0, atol=1e-12):
        raise RuntimeError("charge figure does not have the locked 3.375 x 7.05 inch size")
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.075, top=0.945, hspace=0.47)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARGE_STEM.with_suffix(".pdf"))
    fig.savefig(CHARGE_STEM.with_suffix(".png"), dpi=300)
    plt.close(fig)


def make_contour_figure(
    analyzer: Any,
    cases: dict[int, dict[str, np.ndarray]],
) -> dict[str, dict[str, float]]:
    configure_matplotlib()
    fig = plt.figure(figsize=(3.375, 2.25))
    # Explicit axes keep both 20 x 30 data regions at equal physical scale.
    # Horizontal colorbars preserve readable plotting area when the complete
    # 1-by-2 figure is reduced to a single journal column.
    axes = (
        fig.add_axes((0.095, 0.31, 0.245, 0.60)),
        fig.add_axes((0.585, 0.31, 0.245, 0.60)),
    )
    colorbar_axes = (
        fig.add_axes((0.095, 0.105, 0.245, 0.035)),
        fig.add_axes((0.585, 0.105, 0.245, 0.035)),
    )
    case = cases[TYPICAL_NY]
    entropy_scale = contour_panel(
        fig,
        axes[0],
        case[analyzer.CONTOUR_KEYS["c1"]].mean(axis=0),
        cax=colorbar_axes[0],
        colorbar_orientation="horizontal",
        cmap="Blues",
        colorbar_label=r"$s_1(x,y)$",
    )
    panel_letter(axes[0], "(a)", x=-0.30)
    charge_scale = contour_panel(
        fig,
        axes[1],
        case[analyzer.CONTOUR_KEYS["k"]].mean(axis=0),
        cax=colorbar_axes[1],
        colorbar_orientation="horizontal",
        cmap="Blues",
        colorbar_label=r"$f_A(x,y)$",
    )
    panel_letter(axes[1], "(b)", x=-0.30)
    if not np.allclose(fig.get_size_inches(), (3.375, 2.25), rtol=0.0, atol=1e-12):
        raise RuntimeError("contour figure does not have the locked 3.375 x 2.25 inch size")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONTOUR_STEM.with_suffix(".pdf"))
    fig.savefig(CONTOUR_STEM.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return {"entropy": entropy_scale, "charge_variance": charge_scale}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    analyzer = load_bundle_analysis()
    cases = analyzer.load_cases(analyzer.discover(OUTPUT_ROOT))
    # The analyzer verifies exactly 140 result/completion pairs and the complete
    # sample-ID inventory before returning these seven cases.
    values = samplewise_values(analyzer, cases)
    wall_validation = analyzer.wall_rows(cases)
    closure_max_abs = max(float(row["closure_max_abs"]) for row in wall_validation)

    mean_fits, mean_curve_rows, mean_fit_rows = mean_curve_results(analyzer, cases)
    linearity_max_abs = 0.0
    sem_audit_max_abs = 0.0
    for ny in NY_VALUES:
        for label in (*RENYI_LABELS, "k"):
            expected_center = float(values[ny][label].mean())
            expected_sem = float(values[ny][label].std(ddof=1) / math.sqrt(values[ny][label].size))
            linearity_max_abs = max(
                linearity_max_abs,
                abs(mean_fits[ny][label]["converted"] - expected_center),
            )
            sem_audit_max_abs = max(
                sem_audit_max_abs,
                abs(mean_fits[ny][label]["converted_sem"] - expected_sem),
            )
    if linearity_max_abs > 2e-12 or sem_audit_max_abs > 2e-12:
        raise RuntimeError("mean-first fit failed the OLS central-value/SEM audit")

    anchored = {
        label: anchored_mean_curve_fit(analyzer, cases, label)
        for label in (*RENYI_LABELS, "k")
    }
    collapse_rows = make_mean_entropy_collapse_figure(analyzer, anchored)
    collapse_rows.extend(make_mean_charge_collapse_figure(anchored["k"]))
    ratio_rows = mean_curve_ratio_rows(analyzer, cases, mean_fits)

    write_csv(MEAN_CURVES_CSV, mean_curve_rows)
    write_csv(MEAN_FITS_CSV, mean_fit_rows)
    write_csv(MEAN_COLLAPSE_CSV, collapse_rows)
    write_csv(MEAN_RATIOS_CSV, ratio_rows)
    make_mean_curve_scaling_figure(mean_fits)

    # Figure 11 is the cellwise mean of 100 independently sampled contours.
    contour_scales = make_contour_figure(analyzer, cases)

    output_paths = (
        MEAN_ENTROPY_COLLAPSE_STEM.with_suffix(".pdf"),
        MEAN_ENTROPY_COLLAPSE_STEM.with_suffix(".png"),
        MEAN_CHARGE_COLLAPSE_STEM.with_suffix(".pdf"),
        MEAN_CHARGE_COLLAPSE_STEM.with_suffix(".png"),
        MEAN_SCALING_STEM.with_suffix(".pdf"),
        MEAN_SCALING_STEM.with_suffix(".png"),
        CONTOUR_STEM.with_suffix(".pdf"),
        CONTOUR_STEM.with_suffix(".png"),
        MEAN_CURVES_CSV,
        MEAN_FITS_CSV,
        MEAN_COLLAPSE_CSV,
        MEAN_RATIOS_CSV,
    )
    manifest = {
        "schema": "endpoint_contour_scaling_figures_v7_ensemble_mean_sqrt_blues_contours",
        "sampling_revision": analyzer.SAMPLING_REVISION,
        "verified_input_shards": 140,
        "independent_trajectories_per_size": 100,
        "Ny_values": list(NY_VALUES),
        "endpoint": "t=2Ny",
        "fit_window": "Ay=8..Ny/2 inclusive",
        "figure_layouts": {
            "entropy_mean_curve_size_collapse": {"panels": "3x1", "size_inches": [3.375, 7.05]},
            "charge_variance_mean_curve_size_collapse": {"panels": "1x1", "size_inches": [3.375, 2.65]},
            "mean_curve_cq_k_scaling": {"panels": "1x1", "size_inches": [3.375, 2.65]},
            "contours": {"panels": "1x2", "size_inches": [3.375, 2.25]},
        },
        "fit_curve_format": (
            "legacy atlas Result 4: open unconnected empirical markers, gray declared-fit-window "
            "band, and black dashed fit extrapolated across the displayed domain"
        ),
        "estimator_order": "average_100_trajectory_curves_at_fixed_Ny_then_fit",
        "mean_curve_size_collapse_fit": (
            "ensemble means are anchored at Ay*=floor(Ny/2), then one equal-size-weighted "
            "through-origin slope is fitted across Ny=30..60 over Ay=8..Ay*"
        ),
        "uncertainty": (
            "ordinary trajectory sampling SEM propagated through the full within-trajectory "
            "Ay covariance matrix; no bootstrap and no fit-residual uncertainty"
        ),
        "linearity_audit": {
            "fit_mean_minus_mean_fit_max_abs": linearity_max_abs,
            "covariance_SEM_minus_sample_slope_SEM_max_abs": sem_audit_max_abs,
        },
        "contour_estimator": {
            "Ny": TYPICAL_NY,
            "independent_trajectories": 100,
            "region": "fixed y0=0, Ay=Ny//2",
            "origin_average": "none",
            "ensemble_operation": "cellwise arithmetic mean after per-trajectory contour construction",
        },
        "contour_color_normalization": {
            "method": "independent_panel_PowerNorm",
            "gamma": 0.5,
            "colormap": "Blues",
            "range": "zero_to_panel_max",
            "wall_overlays": "none",
            "panel_scales": contour_scales,
        },
        "charge_variance_notation": (
            "F_A denotes the intrinsic subsystem-A charge variance and f_A its spatial "
            "contour; q is reserved exclusively for Renyi order"
        ),
        "contour_to_scalar_closure_max_abs": closure_max_abs,
        "input_download_manifest": {
            "path": str((OUTPUT_ROOT / "DOWNLOAD_MANIFEST.json").relative_to(ROOT)),
            "bytes": (OUTPUT_ROOT / "DOWNLOAD_MANIFEST.json").stat().st_size,
            "sha256": sha256(OUTPUT_ROOT / "DOWNLOAD_MANIFEST.json"),
        },
        "entropy_charge_relation": (
            "c_q/k is formed from coefficients fitted to the ensemble-mean entropy and "
            "charge curves; its SEM uses their joint trajectory covariance and the delta method"
        ),
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in output_paths
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[verified input] 140 shards; 100 trajectories at each Ny={list(NY_VALUES)}")
    for label in (*RENYI_LABELS, "k"):
        summary = anchored[label][1]
        print(
            f"[mean collapse] {label}: slope={summary['slope']:.8f} "
            f"+/- {summary['slope_covariance_SEM']:.8f}, "
            f"converted={summary['converted_coefficient']:.8f} "
            f"+/- {summary['converted_covariance_SEM']:.8f}, "
            f"R0^2={summary['R0_squared']:.8f}"
        )
    print(f"[contour ensemble mean] Ny={TYPICAL_NY}, trajectories=100")
    for path in output_paths:
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
