"""Source fitting and drawing helpers, extracted without changes."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import math, csv
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from manuscript_typography import configure_style as manuscript_style, prepare_figure, record_typography
NY_VALUES=(30,35,40,45,50,55,60)
FIT_MIN_AY=8
def configure_matplotlib() -> None:
    manuscript_style({'figure.dpi': 140, 'savefig.dpi': 300, 'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.linewidth': 0.7, 'lines.linewidth': 0.9, 'lines.markersize': 3.5, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.width': 0.65, 'ytick.major.width': 0.65, 'xtick.major.size': 2.8, 'ytick.major.size': 2.8, 'axes.spines.top': True, 'axes.spines.right': True, 'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})

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

def panel_letter(ax: plt.Axes, letter: str, *, x: float = -0.18) -> None:
    ax.text(x, 1.05, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=9)

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
