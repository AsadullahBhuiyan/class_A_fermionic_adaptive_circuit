#!/usr/bin/env python3
"""Build the ensemble-stationarity figures and tables used by the working note.

The script is analysis-only: it reads archived trajectory observables and writes
compact figures/tables beside the note.  It does not rerun circuit dynamics or
modify any campaign output.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
COLAB = REPO / "00_WORKSPACE/COLAB/colab_charge_fluctuations"

FIT_ROOT = (
    COLAB
    / "analysis_outputs/streaming_covariance_scaling_fits"
    / "N20_selected_nsh1_dwtrunc1_init-default_S100_cycles-2Ny/runs"
)
PURE_ROOT = (
    COLAB
    / "gpu_data/streaming_covariance_observables/campaigns"
    / "N20_selected_nsh1_dwtrunc1_init-default_S100_cycles-2Ny/runs"
)
MAXMIX_ROOT = (
    COLAB
    / "gpu_data/purification_dynamics_maxmix/campaigns"
    / "N20_multi_geometry_nsh1_dwtrunc1_init-maxmix_S100_cycles-2Ny/runs"
)

FIG_DIR = HERE / "figures"
TABLE_DIR = HERE / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SIZES = (30, 40, 50)
NX = 20
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 314159
ROLLING_WINDOW = 5

OBSERVABLE_ORDER = (
    "pure_charge_density_offset",
    "entropy_slope",
    "correlator_exponent",
    "real_space_chern",
    "covariance_step_rms",
    "maxmix_charge_variance_density",
)

OBSERVABLE_LABELS = {
    "pure_charge_density_offset": r"charge density offset",
    "entropy_slope": r"entropy slope $c_{\rm fit}$",
    "correlator_exponent": r"correlator exponent $\alpha_C$",
    "real_space_chern": r"trajectory Chern marker",
    "covariance_step_rms": r"covariance step $\|G_t-G_{t-1}\|_F/\sqrt{V}$",
    "maxmix_charge_variance_density": r"maxmix $\operatorname{Var}(\hat N_F)/V$",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "CMU Sans Serif",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def pivot_scalar(df: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    pivot = (
        df.pivot(index="sample_index", columns="cycle_label", values=column)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    return pivot.columns.to_numpy(dtype=int), pivot.to_numpy(dtype=float)


def load_observables(ny: int) -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    pure_run = f"N20x{ny}_nsh1_init-default_perfect_correction"
    maxmix_run = f"N20x{ny}_nsh1_init-maxmix_perfect_correction"
    volume = 2 * NX * ny

    fits = np.load(FIT_ROOT / pure_run / "streaming_covariance_scaling_fits.npz")
    fit_cycles = fits["cycle_labels"].astype(int)

    scalar = pd.read_csv(PURE_ROOT / pure_run / "scalar_metrics.csv")
    scalar_cycles, chern = pivot_scalar(scalar, "real_space_chern")
    step_cycles, covariance_step = pivot_scalar(scalar, "frob_successive_delta")
    if not np.array_equal(scalar_cycles, step_cycles):
        raise RuntimeError(f"cycle mismatch in scalar table for Ny={ny}")

    local = np.load(PURE_ROOT / pure_run / "local_charge_cell.npz")
    charge_cycles = local["cycle_labels"].astype(int)
    trace = local["local_charge_cell"].sum(axis=(2, 3))
    charge_density_offset = (trace - NX * ny) / volume

    maxmix = np.load(MAXMIX_ROOT / maxmix_run / "total_charge_variance.npz")
    maxmix_cycles = maxmix["cycle_labels"].astype(int)
    charge_variance_density = maxmix["total_charge_variance"] / volume

    arrays = {
        "pure_charge_density_offset": (
            charge_cycles,
            charge_density_offset,
            "init-default pure-state streaming",
        ),
        "entropy_slope": (
            fit_cycles,
            fits["entropy_slope"].astype(float),
            "init-default pure-state streaming",
        ),
        "correlator_exponent": (
            fit_cycles,
            fits["correlator_loglog_slope"].astype(float),
            "init-default pure-state streaming",
        ),
        "real_space_chern": (
            scalar_cycles,
            chern,
            "init-default pure-state streaming",
        ),
        "covariance_step_rms": (
            step_cycles,
            covariance_step / np.sqrt(volume),
            "init-default pure-state streaming",
        ),
        "maxmix_charge_variance_density": (
            maxmix_cycles,
            charge_variance_density,
            "init-maxmix purification",
        ),
    }

    for name, (cycles, values, _) in arrays.items():
        if values.shape != (100, 2 * ny):
            raise RuntimeError(f"unexpected {name} shape {values.shape} for Ny={ny}")
        if not np.isfinite(values).all():
            raise RuntimeError(f"nonfinite {name} values for Ny={ny}")
        if not np.array_equal(cycles, np.arange(1, 2 * ny + 1)):
            raise RuntimeError(f"unexpected cycle grid for {name}, Ny={ny}")
    return arrays


def pooled_acf(values: np.ndarray) -> np.ndarray:
    """Correlation of trajectory residuals after removing each cycle mean."""
    residual = values - values.mean(axis=0, keepdims=True)
    n_cycles = residual.shape[1]
    rho = [1.0]
    for lag in range(1, n_cycles // 2 + 1):
        left = residual[:, :-lag].ravel()
        right = residual[:, lag:].ravel()
        denom = np.sqrt(np.dot(left, left) * np.dot(right, right))
        rho.append(float(np.dot(left, right) / denom) if denom > 0 else np.nan)
    return np.asarray(rho)


def integrated_autocorrelation_time(values: np.ndarray) -> tuple[float, float]:
    rho = pooled_acf(values)
    positive = []
    for value in rho[1:]:
        if not np.isfinite(value) or value <= 0:
            break
        positive.append(float(value))
    tau = 1.0 + 2.0 * float(np.sum(positive))
    return float(rho[1]), tau


def block_statistics(
    cycles: np.ndarray,
    values: np.ndarray,
    ny: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    mask = (cycles >= ny) & (cycles <= 2 * ny)
    window = values[:, mask]
    window_cycles = cycles[mask]
    first = window[:, window_cycles < 1.5 * ny].mean(axis=1)
    second = window[:, window_cycles >= 1.5 * ny].mean(axis=1)
    paired_difference = second - first
    pooled_sd = float(window.std(ddof=1))
    raw_difference = float(paired_difference.mean())
    standardized_difference = raw_difference / pooled_sd
    rho1, tau = integrated_autocorrelation_time(window)

    boot_raw = np.empty(BOOTSTRAP_REPETITIONS)
    boot_standardized = np.empty(BOOTSTRAP_REPETITIONS)
    boot_tau = np.empty(BOOTSTRAP_REPETITIONS)
    sample_count = window.shape[0]
    for rep in range(BOOTSTRAP_REPETITIONS):
        indices = rng.integers(0, sample_count, size=sample_count)
        selected = window[indices]
        selected_first = selected[:, window_cycles < 1.5 * ny].mean(axis=1)
        selected_second = selected[:, window_cycles >= 1.5 * ny].mean(axis=1)
        boot_raw[rep] = float((selected_second - selected_first).mean())
        selected_sd = float(selected.std(ddof=1))
        boot_standardized[rep] = boot_raw[rep] / selected_sd
        boot_tau[rep] = integrated_autocorrelation_time(selected)[1]

    raw_low, raw_high = np.quantile(boot_raw, [0.025, 0.975])
    standardized_low, standardized_high = np.quantile(
        boot_standardized, [0.025, 0.975]
    )
    tau_low, tau_high = np.quantile(boot_tau, [0.025, 0.975])
    effective_samples = sample_count * window.shape[1] / tau
    effective_per_trajectory = window.shape[1] / tau

    return {
        "trajectories": sample_count,
        "window_cycle_min": int(window_cycles.min()),
        "window_cycle_max": int(window_cycles.max()),
        "window_cycles_inclusive": int(window.shape[1]),
        "first_block_cycle_min": int(window_cycles[window_cycles < 1.5 * ny].min()),
        "first_block_cycle_max": int(window_cycles[window_cycles < 1.5 * ny].max()),
        "second_block_cycle_min": int(window_cycles[window_cycles >= 1.5 * ny].min()),
        "second_block_cycle_max": int(window_cycles[window_cycles >= 1.5 * ny].max()),
        "first_block_mean": float(first.mean()),
        "second_block_mean": float(second.mean()),
        "second_minus_first": raw_difference,
        "second_minus_first_ci95_low": float(raw_low),
        "second_minus_first_ci95_high": float(raw_high),
        "standardized_block_drift": standardized_difference,
        "standardized_block_drift_ci95_low": float(standardized_low),
        "standardized_block_drift_ci95_high": float(standardized_high),
        "drift_detected_95pct": bool(raw_low > 0 or raw_high < 0),
        "lag1_autocorrelation": rho1,
        "integrated_autocorrelation_cycles": tau,
        "integrated_autocorrelation_ci95_low": float(tau_low),
        "integrated_autocorrelation_ci95_high": float(tau_high),
        "effective_independent_samples_per_trajectory": effective_per_trajectory,
        "effective_independent_samples_total": effective_samples,
    }


def save_time_series_figure(all_data: dict[int, dict]) -> pd.DataFrame:
    configure_matplotlib()
    colors = {30: "#0072B2", 40: "#D55E00", 50: "#009E73"}
    panels = (
        ("pure_charge_density_offset", r"$(\operatorname{tr}C-V/2)/V$", "(a) Charge-density offset"),
        ("entropy_slope", r"entropy slope $c_{\rm fit}$", "(b) Entanglement scaling"),
        ("correlator_exponent", r"correlator exponent $\alpha_C$", "(c) Long-range correlator"),
        ("real_space_chern", r"trajectory Chern marker", "(d) Topological diagnostic"),
    )

    rows: list[dict[str, float | int | str]] = []
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.1), sharex=True)
    for ax, (name, ylabel, title) in zip(axes.flat, panels):
        visible_low: list[float] = []
        visible_high: list[float] = []
        for ny in SIZES:
            cycles, values, ensemble = all_data[ny][name]
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
            rolling = pd.Series(mean).rolling(
                ROLLING_WINDOW, center=True, min_periods=1
            ).mean().to_numpy()
            scaled_cycle = cycles / ny
            visible = scaled_cycle >= 0.75
            visible_low.extend((mean[visible] - sem[visible]).tolist())
            visible_high.extend((mean[visible] + sem[visible]).tolist())
            ax.plot(scaled_cycle, mean, color=colors[ny], alpha=0.24, linewidth=0.8)
            ax.fill_between(
                scaled_cycle,
                mean - sem,
                mean + sem,
                color=colors[ny],
                alpha=0.12,
                linewidth=0,
            )
            ax.plot(
                scaled_cycle,
                rolling,
                color=colors[ny],
                label=rf"$N_y={ny}$",
            )
            for cycle, avg, error, smooth in zip(cycles, mean, sem, rolling):
                rows.append(
                    {
                        "Nx": NX,
                        "Ny": ny,
                        "observable": name,
                        "ensemble": ensemble,
                        "cycle": int(cycle),
                        "cycle_over_Ny": float(cycle / ny),
                        "trajectory_mean": float(avg),
                        "trajectory_sem": float(error),
                        "trajectory_mean_rolling_w5": float(smooth),
                    }
                )

        ax.axvspan(1.0, 2.0, color="0.5", alpha=0.07, zorder=-3)
        ax.axvline(1.0, color="0.35", linestyle="--", linewidth=0.9)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.set_xlim(0.75, 2.02)
        lower = min(visible_low)
        upper = max(visible_high)
        padding = 0.08 * (upper - lower)
        ax.set_ylim(lower - padding, upper + padding)

    for ax in axes[-1]:
        ax.set_xlabel(r"cycle/$N_y$")
    axes[0, 0].legend(frameon=False, loc="best")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"post_burnin_stationarity_time_series.{suffix}")
    plt.close(fig)
    return pd.DataFrame(rows)


def save_stationarity_summary_figure(summary: pd.DataFrame) -> None:
    configure_matplotlib()
    colors = {30: "#0072B2", 40: "#D55E00", 50: "#009E73"}
    markers = {30: "o", 40: "s", 50: "^"}
    labels = [OBSERVABLE_LABELS[name] for name in OBSERVABLE_ORDER]
    y_base = np.arange(len(labels), dtype=float)
    offsets = {30: -0.20, 40: 0.0, 50: 0.20}

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.65))
    ax = axes[0]
    for ny in SIZES:
        rows = summary[summary["Ny"] == ny].set_index("observable").loc[list(OBSERVABLE_ORDER)]
        x = rows["standardized_block_drift"].to_numpy()
        low = rows["standardized_block_drift_ci95_low"].to_numpy()
        high = rows["standardized_block_drift_ci95_high"].to_numpy()
        ax.errorbar(
            x,
            y_base + offsets[ny],
            xerr=np.vstack((x - low, high - x)),
            fmt=markers[ny],
            color=colors[ny],
            markersize=4.3,
            capsize=2,
            linewidth=1,
            label=rf"$N_y={ny}$",
        )
    ax.axvline(0, color="0.25", linestyle="--", linewidth=0.9)
    ax.set_yticks(y_base, labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"late-minus-early block drift / pooled $\sigma$")
    ax.set_title("(a) Stationarity within $[N_y,2N_y]$")
    ax.grid(axis="x", alpha=0.18, linewidth=0.5)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    line_styles = {
        "pure_charge_density_offset": ("o", "-"),
        "entropy_slope": ("s", "-"),
        "correlator_exponent": ("^", "-"),
        "real_space_chern": ("D", "--"),
        "covariance_step_rms": ("v", "--"),
        "maxmix_charge_variance_density": ("P", ":"),
    }
    palette = ("#000000", "#0072B2", "#D55E00", "#CC79A7", "#009E73", "#E69F00")
    for color, name in zip(palette, OBSERVABLE_ORDER):
        rows = summary[summary["observable"] == name].sort_values("Ny")
        marker, linestyle = line_styles[name]
        y = rows["integrated_autocorrelation_cycles"].to_numpy()
        low = rows["integrated_autocorrelation_ci95_low"].to_numpy()
        high = rows["integrated_autocorrelation_ci95_high"].to_numpy()
        ax.errorbar(
            rows["Ny"],
            y,
            yerr=np.vstack((y - low, high - y)),
            marker=marker,
            linestyle=linestyle,
            color=color,
            markersize=4,
            capsize=2,
            linewidth=1.2,
            label=OBSERVABLE_LABELS[name],
        )
    ny_grid = np.linspace(min(SIZES), max(SIZES), 100)
    ax.plot(ny_grid, ny_grid, color="0.35", linestyle="--", linewidth=0.9, label=r"window length $N_y$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N_y$")
    ax.set_ylabel(r"integrated autocorrelation time $\tau_{\rm int}$ [cycles]")
    ax.set_title("(b) Samples are correlated")
    ax.grid(alpha=0.18, linewidth=0.5, which="both")
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        legend_labels,
        frameon=False,
        fontsize=6.4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.29)
    for suffix in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"post_burnin_block_stationarity_and_autocorrelation.{suffix}")
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    all_data = {ny: load_observables(ny) for ny in SIZES}
    time_series = save_time_series_figure(all_data)
    time_series.to_csv(TABLE_DIR / "post_burnin_stationarity_time_series.csv", index=False)

    rows: list[dict[str, float | int | str | bool]] = []
    for ny in SIZES:
        for observable in OBSERVABLE_ORDER:
            cycles, values, ensemble = all_data[ny][observable]
            stats = block_statistics(cycles, values, ny, rng)
            rows.append(
                {
                    "Nx": NX,
                    "Ny": ny,
                    "observable": observable,
                    "observable_label": OBSERVABLE_LABELS[observable],
                    "ensemble": ensemble,
                    **stats,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLE_DIR / "post_burnin_stationarity_autocorrelation_summary.csv", index=False)
    save_stationarity_summary_figure(summary)

    manifest = {
        "analysis": "post_burnin_stationarity_and_autocorrelation",
        "created_by": Path(__file__).name,
        "dynamics_rerun": False,
        "Nx": NX,
        "Ny": list(SIZES),
        "trajectories": 100,
        "measurement_window": "inclusive cycles Ny through 2Ny",
        "block_split": "first block t < 1.5Ny; second block t >= 1.5Ny",
        "stationarity_test": (
            "paired trajectory block means; percentile 95% confidence interval "
            "from trajectory bootstrap"
        ),
        "autocorrelation": (
            "pooled trajectory-residual autocorrelation after subtracting the "
            "ensemble mean separately at each cycle; integrated through the "
            "first nonpositive lag"
        ),
        "effective_sample_count": "S*(Ny+1)/tau_int",
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "rolling_average_for_display_only": ROLLING_WINDOW,
        "source_roots": [str(FIT_ROOT), str(PURE_ROOT), str(MAXMIX_ROOT)],
        "outputs": {
            "figures": [
                "figures/post_burnin_stationarity_time_series.pdf",
                "figures/post_burnin_block_stationarity_and_autocorrelation.pdf",
            ],
            "tables": [
                "tables/post_burnin_stationarity_time_series.csv",
                "tables/post_burnin_stationarity_autocorrelation_summary.csv",
            ],
        },
    }
    (HERE / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
