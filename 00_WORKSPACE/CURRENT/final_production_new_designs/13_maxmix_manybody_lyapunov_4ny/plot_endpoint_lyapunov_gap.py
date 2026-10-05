#!/usr/bin/env python3
"""Plot the sample-averaged endpoint Lyapunov half-gap for bundle 13."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import analyze_campaign


BUNDLE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = analyze_campaign.DATA_ROOT
OUTPUT_ROOT = BUNDLE_ROOT / "analysis_outputs" / "endpoint_lyapunov_gap_v1"
NY_VALUES = analyze_campaign.NY_VALUES
CAP_TOLERANCE = analyze_campaign.CAP_TOLERANCE


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def weighted_power_law(ny: np.ndarray, mean: np.ndarray, sem: np.ndarray) -> dict[str, float]:
    """Fit mean=A*Ny**(-z) by weighted least squares in log space."""

    x = np.log(np.asarray(ny, dtype=np.float64))
    y = np.log(np.asarray(mean, dtype=np.float64))
    sigma = np.asarray(sem, dtype=np.float64) / np.asarray(mean, dtype=np.float64)
    design = np.column_stack((np.ones(x.size), -x))
    weights = 1.0 / sigma**2
    normal = design.T @ (weights[:, None] * design)
    covariance = np.linalg.inv(normal)
    coefficients = covariance @ (design.T @ (weights * y))
    predicted = design @ coefficients
    chi_squared = float(np.sum(((y - predicted) / sigma) ** 2))
    return {
        "amplitude": float(np.exp(coefficients[0])),
        "exponent": float(coefficients[1]),
        "amplitude_sem": float(np.exp(coefficients[0]) * math.sqrt(covariance[0, 0])),
        "exponent_sem": float(math.sqrt(covariance[1, 1])),
        "chi_squared": chi_squared,
        "degrees_of_freedom": int(x.size - 2),
    }


def load_endpoint_gaps() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    # Reuse the campaign's full manifest, completion, checksum, numerical, and
    # many-body-spectrum verification before reading the endpoint observables.
    _, _, provenance = analyze_campaign.load_and_verify(verify_hashes=True)

    sample_rows: list[dict[str, Any]] = []
    by_size: dict[int, list[float]] = {ny: [] for ny in NY_VALUES}
    maximum_soft_mode_difference = 0.0

    for result_path in sorted((DATA_ROOT / "results").rglob("*.npz")):
        with np.load(result_path, allow_pickle=False) as data:
            ny = int(np.asarray(data["Ny"]).item())
            total_time = 4 * ny
            spectrum_cycles = np.asarray(data["spectrum_cycles"], dtype=np.int64)
            if int(spectrum_cycles[-1]) != total_time:
                raise RuntimeError(f"{result_path.name}: endpoint is not 4Ny")
            occupations = np.asarray(data["occupations"][:, -1, :], dtype=np.float64)
            cap_mask = np.asarray(data["cap_mask"][:, -1, :], dtype=bool)
            sample_indices = np.asarray(data["sample_indices"], dtype=np.int64)
            soft_costs = np.asarray(data["soft_mode_flip_costs"][:, -1, :], dtype=np.float64)

            for local_index, sample_index in enumerate(sample_indices):
                interior = ~cap_mask[local_index]
                if not np.any(interior):
                    raise RuntimeError(f"Ny={ny} sample={sample_index}: no finite endpoint mode")
                nu = occupations[local_index, interior]
                if np.any(nu <= CAP_TOLERANCE) or np.any(nu >= 1.0 - CAP_TOLERANCE):
                    raise RuntimeError(f"Ny={ny} sample={sample_index}: cap-mask inconsistency")
                exponents = np.abs(np.log(nu) - np.log1p(-nu)) / (2.0 * total_time)
                gap = float(np.min(exponents))
                stored_gap = float(np.min(soft_costs[local_index]) / (2.0 * total_time))
                difference = abs(gap - stored_gap)
                maximum_soft_mode_difference = max(maximum_soft_mode_difference, difference)
                if difference > 5.0e-14:
                    raise RuntimeError(f"Ny={ny} sample={sample_index}: soft-mode gap mismatch")
                by_size[ny].append(gap)
                sample_rows.append(
                    {
                        "Ny": ny,
                        "sample_index": int(sample_index),
                        "endpoint_cycle": total_time,
                        "finite_mode_count": int(np.count_nonzero(interior)),
                        "endpoint_lyapunov_half_gap": gap,
                    }
                )

    summary_rows: list[dict[str, Any]] = []
    for ny in NY_VALUES:
        values = np.asarray(by_size[ny], dtype=np.float64)
        if values.size != 100 or not np.all(np.isfinite(values)):
            raise RuntimeError(f"Ny={ny}: expected 100 finite sample gaps")
        sample_ids = sorted(row["sample_index"] for row in sample_rows if row["Ny"] == ny)
        if sample_ids != list(range(100)):
            raise RuntimeError(f"Ny={ny}: sample coverage mismatch")
        summary_rows.append(
            {
                "Ny": ny,
                "samples": int(values.size),
                "endpoint_cycle": 4 * ny,
                "mean_endpoint_lyapunov_half_gap": float(np.mean(values)),
                "sample_standard_deviation": float(np.std(values, ddof=1)),
                "sample_sem": float(np.std(values, ddof=1) / math.sqrt(values.size)),
                "minimum_sample_gap": float(np.min(values)),
                "maximum_sample_gap": float(np.max(values)),
                "mean_Ny_times_gap": float(ny * np.mean(values)),
                "sem_Ny_times_gap": float(ny * np.std(values, ddof=1) / math.sqrt(values.size)),
            }
        )

    diagnostics = {
        "definition": "min_j abs(lambda_j(T)); lambda_j=[log(1-nu_j)-log(nu_j)]/(2T)",
        "endpoint": "T=4Ny",
        "uncertainty": "sample standard error of the mean; std(ddof=1)/sqrt(S)",
        "cap_handling": "stored exact-cap modes are retained at infinite magnitude and excluded from the minimum",
        "maximum_direct_vs_stored_soft_mode_gap_difference": maximum_soft_mode_difference,
        "provenance": provenance,
    }
    return sample_rows, summary_rows, diagnostics


def configure_plotting() -> None:
    available = {font.name for font in mpl.font_manager.fontManager.ttflist}
    sans = "CMU Sans Serif" if "CMU Sans Serif" in available else "DejaVu Sans"
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [sans],
            "mathtext.fontset": "cm",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_figure(summary_rows: list[dict[str, Any]], fit: dict[str, float]) -> None:
    configure_plotting()
    ny = np.asarray([row["Ny"] for row in summary_rows], dtype=np.float64)
    mean = np.asarray([row["mean_endpoint_lyapunov_half_gap"] for row in summary_rows])
    error = np.asarray([row["sample_sem"] for row in summary_rows])

    figure, axis = plt.subplots(figsize=(3.375, 2.72), constrained_layout=True)
    axis.errorbar(
        ny,
        mean,
        yerr=error,
        color="#2b6cb0",
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.0,
        linewidth=1.15,
        capsize=2.2,
        label=r"mean $\pm$ SEM ($S=100$)",
        zorder=3,
    )
    dense_ny = np.linspace(float(ny.min()), float(ny.max()), 300)
    fit_curve = fit["amplitude"] * dense_ny ** (-fit["exponent"])
    axis.plot(
        dense_ny,
        fit_curve,
        color="0.25",
        linestyle="--",
        linewidth=1.0,
        label=rf"$A N_y^{{-z}}$, $z={fit['exponent']:.2f}\pm{fit['exponent_sem']:.2f}$",
        zorder=2,
    )
    axis.set_xlabel(r"circumference $N_y$")
    axis.set_ylabel(r"$\Delta_\lambda$")
    axis.set_xticks(ny.astype(int))
    axis.set_xlim(17.5, 62.5)
    axis.set_ylim(bottom=0.0)
    axis.legend(frameon=False, loc="upper right", handlelength=2.4)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_ROOT / "endpoint_lyapunov_gap_vs_Ny.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT_ROOT / "endpoint_lyapunov_gap_vs_Ny.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    sample_rows, summary_rows, diagnostics = load_endpoint_gaps()
    ny = np.asarray([row["Ny"] for row in summary_rows], dtype=np.float64)
    mean = np.asarray([row["mean_endpoint_lyapunov_half_gap"] for row in summary_rows])
    error = np.asarray([row["sample_sem"] for row in summary_rows])
    fit = weighted_power_law(ny, mean, error)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_ROOT / "endpoint_lyapunov_gap_samples.csv", sample_rows)
    write_csv(OUTPUT_ROOT / "endpoint_lyapunov_gap_summary.csv", summary_rows)
    summary = {
        "schema": "bundle13_endpoint_lyapunov_gap_analysis_v1",
        "campaign": analyze_campaign.REVISION,
        "sizes": list(NY_VALUES),
        "samples_per_size": 100,
        "trajectory_count": 700,
        "summary_rows": summary_rows,
        "weighted_log_space_power_law_fit": fit,
        **diagnostics,
    }
    (OUTPUT_ROOT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_figure(summary_rows, fit)
    print(json.dumps({"output_root": str(OUTPUT_ROOT), "fit": fit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
