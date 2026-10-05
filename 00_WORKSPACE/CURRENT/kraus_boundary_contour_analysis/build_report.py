#!/usr/bin/env python3
"""Combine the four boundary-contour checks into one reproducible report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "analysis_outputs/kraus_boundary_contours_v1"
SEED = 2026091705


def mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 2000):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    replicas = np.mean(
        values[rng.integers(0, values.size, size=(draws, values.size))], axis=1
    )
    return float(values.mean()), *np.quantile(replicas, [0.025, 0.975]).tolist()


def endpoint_index(cycles: np.ndarray) -> int:
    valid = np.flatnonzero(np.asarray(cycles) >= 0)
    if valid.size == 0:
        raise RuntimeError("missing spectrum cycle grid")
    return int(valid[-1])


def fit_gap_coefficients(csv_path: Path, rng: np.random.Generator):
    rows: list[dict[str, float]] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "Ny": float(raw["Ny"]),
                    "gap": float(raw["gap_index"]),
                    "slope": float(raw["slope_W3_3Ny_to_4Ny"]),
                }
            )
    ny_values = sorted({int(row["Ny"]) for row in rows})
    coefficients = []
    bootstrap = np.empty((2000, 4), dtype=np.float64)
    for gap in range(1, 5):
        groups = [
            np.asarray(
                [row["slope"] for row in rows if row["Ny"] == ny and row["gap"] == gap],
                dtype=np.float64,
            )
            for ny in ny_values
        ]
        groups = [values[np.isfinite(values)] for values in groups]
        means = np.asarray([values.mean() for values in groups])
        x = 1.0 / np.asarray(ny_values, dtype=np.float64)
        coefficient = float(np.dot(x, means) / np.dot(x, x))
        coefficients.append(coefficient)
        for draw in range(bootstrap.shape[0]):
            draw_means = np.asarray(
                [
                    np.mean(values[rng.integers(0, values.size, size=values.size)])
                    for values in groups
                ]
            )
            bootstrap[draw, gap - 1] = np.dot(x, draw_means) / np.dot(x, x)
    coefficients = np.asarray(coefficients)
    ratios = coefficients / coefficients[0]
    ratio_bootstrap = bootstrap / bootstrap[:, :1]
    return {
        "ny_values": ny_values,
        "coefficients": coefficients,
        "coefficient_ci": np.quantile(bootstrap, [0.025, 0.975], axis=0).T,
        "ratios": ratios,
        "ratio_ci": np.quantile(ratio_bootstrap, [0.025, 0.975], axis=0).T,
    }


def main() -> int:
    required = [
        OUTPUT / "legacy_time_contours.npz",
        OUTPUT / "purification_endpoints.npz",
        OUTPUT / "bundle13_gap_contours.npz",
        OUTPUT / "event_replay/event_replay_contours.npz",
        OUTPUT / "event_replay/summary.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing analysis products: " + ", ".join(missing))

    rng = np.random.default_rng(SEED)
    with np.load(required[0], allow_pickle=False) as data:
        legacy_cycles = np.asarray(data["cycles"], dtype=np.float64)
        legacy_wall = np.asarray(data["soft_wall_weights"], dtype=np.float64)
        legacy_spectral_x = np.asarray(data["spectral_x"], dtype=np.float64)
    legacy_aggregate = legacy_wall[:, :, :4].sum(axis=-1).mean(axis=-1)

    with np.load(required[1], allow_pickle=False) as data:
        constructions = np.asarray(data["constructions"]).astype(str)
        endpoint_ny = np.asarray(data["ny_values"], dtype=np.int64)
        endpoint_wall = np.asarray(data["soft_wall_weights"], dtype=np.float64)
    endpoint_aggregate = endpoint_wall[:, :, :, :4].sum(axis=-1).mean(axis=-1)

    with np.load(required[2], allow_pickle=False) as data:
        gap_ny = np.asarray(data["ny_values"], dtype=np.int64)
        gap_cycles = np.asarray(data["spectrum_cycles"], dtype=np.int64)
        gap_wall = np.asarray(data["gap_wall_fraction"], dtype=np.float64)
        bundle_soft_wall = np.asarray(data["first_four_soft_wall_weight"], dtype=np.float64)

    with np.load(required[3], allow_pickle=False) as data:
        event_cycles = np.asarray(data["cycles"], dtype=np.int64)
        event_omega = np.asarray(data["cumulative_omega_support"], dtype=np.float64)
        event_spectral = np.asarray(data["spectral_x"], dtype=np.float64)
        event_ell0 = np.asarray(data["ell0_support_no_constant"], dtype=np.float64)
    event_summary = json.loads(required[4].read_text(encoding="utf-8"))

    gap_fit = fit_gap_coefficients(
        OUTPUT / "bundle13_gap_contours.trajectory_slopes.csv", rng
    )

    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.15), constrained_layout=True)

    ax = axes[0, 0]
    legacy_mean = legacy_aggregate.mean(axis=0)
    legacy_sem = legacy_aggregate.std(axis=0, ddof=1) / np.sqrt(legacy_aggregate.shape[0])
    ax.plot(legacy_cycles / 20.0, legacy_mean, color="#1976d2", lw=1.4)
    ax.fill_between(
        legacy_cycles / 20.0,
        legacy_mean - legacy_sem,
        legacy_mean + legacy_sem,
        color="#1976d2",
        alpha=0.22,
        lw=0,
    )
    ax.axhline(1.0, color="0.3", ls="--", lw=0.8)
    ax.set(xlabel=r"cycle $t/N_y$", ylabel="wall weight", ylim=(0.6, 1.01))
    ax.set_title("(a) Legacy history: first four soft modes")

    ax = axes[0, 1]
    colors = {"hard": "#d84315", "soft": "#00897b"}
    endpoint_stats: dict[str, dict[str, list[float]]] = {}
    for ci, construction in enumerate(constructions):
        means, lows, highs = [], [], []
        for ni in range(endpoint_ny.size):
            mean, low, high = mean_ci(endpoint_aggregate[ci, ni], rng)
            means.append(mean)
            lows.append(low)
            highs.append(high)
        endpoint_stats[construction] = {"mean": means, "low": lows, "high": highs}
        ax.errorbar(
            endpoint_ny,
            means,
            yerr=[np.asarray(means) - lows, np.asarray(highs) - means],
            marker="o",
            ms=4,
            capsize=2,
            lw=1.1,
            label=f"{construction}, S=100",
            color=colors[construction],
        )
    ax.axhline(1.0, color="0.3", ls="--", lw=0.8)
    ax.set(xlabel=r"$N_y$", ylabel="mean wall weight", ylim=(0.86, 1.005))
    ax.set_title("(b) Independent endpoint ensembles")
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1, 0]
    first_gap_medians = []
    soft_medians = []
    for ni in range(gap_ny.size):
        ti = endpoint_index(gap_cycles[ni])
        first_gap_medians.append(np.nanmedian(gap_wall[ni, :, ti, 0]))
        soft_medians.append(np.nanmedian(bundle_soft_wall[ni, :, ti]))
    ax.plot(gap_ny, first_gap_medians, "o-", color="#6a1b9a", label="first MB gap")
    ax.plot(gap_ny, soft_medians, "s--", color="#ef6c00", label="first four modes")
    ax.set(xlabel=r"$N_y$", ylabel="median wall fraction", ylim=(0.90, 1.002))
    ax.set_title("(c) 700-trajectory gap-contour check")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 1]
    x = np.arange(event_omega.shape[-1])
    ax.plot(x, event_omega[-1], marker="o", ms=2.5, lw=1.0, label=r"record $\omega_x$")
    ax.plot(x, event_spectral[-1], marker="s", ms=2.5, lw=1.0, label="spectral")
    ax.plot(x, event_ell0[-1], marker="^", ms=2.5, lw=1.0, label=r"sum $\ell_{0,x}$")
    ax.axvline(5, color="0.5", ls=":", lw=0.8)
    ax.axvline(15, color="0.5", ls=":", lw=0.8)
    ax.set(xlabel=r"transverse coordinate $x$", ylabel="log-weight contour")
    ax.set_title("(d) Exact event-resolved replay at cycle 4")
    ax.legend(frameon=False, ncol=2, loc="best")
    for axis in axes.flat:
        axis.tick_params(direction="in", top=True, right=True)

    png_path = OUTPUT / "kraus_boundary_contour_main.png"
    pdf_path = OUTPUT / "kraus_boundary_contour_report.pdf"
    fig.savefig(png_path, dpi=300)

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
        page = plt.figure(figsize=(7.05, 4.7), constrained_layout=True)
        grid = page.add_gridspec(1, 2, width_ratios=(1.25, 1.0))
        ax = page.add_subplot(grid[0, 0])
        image = ax.imshow(
            legacy_spectral_x.mean(axis=0).T,
            origin="lower",
            aspect="auto",
            extent=(0, 2, -0.5, 19.5),
            cmap="magma",
        )
        ax.axhline(5, color="w", ls=":", lw=0.8)
        ax.axhline(15, color="w", ls=":", lw=0.8)
        ax.set(xlabel=r"cycle $t/N_y$", ylabel=r"$x$", title="Mean legacy spectral contour")
        page.colorbar(image, ax=ax, label=r"$\sum_a |u_a(x)|^2\log p_a^\star$")
        ax = page.add_subplot(grid[0, 1])
        gap_indices = np.arange(1, 5)
        ax.errorbar(
            gap_indices,
            gap_fit["ratios"],
            yerr=[
                gap_fit["ratios"] - gap_fit["ratio_ci"][:, 0],
                gap_fit["ratio_ci"][:, 1] - gap_fit["ratios"],
            ],
            fmt="o",
            capsize=3,
            color="#6a1b9a",
        )
        ax.axhline(1.0, color="0.5", ls="--", lw=0.8)
        ax.set(
            xlabel="ordered many-body gap",
            ylabel=r"$A_i/A_1$",
            xticks=gap_indices,
            title=r"Late-window gap ratios ($\Delta_i=A_i/N_y$)",
        )
        ax.tick_params(direction="in", top=True, right=True)
        pdf.savefig(page)
        plt.close(page)
    plt.close(fig)

    summary = {
        "schema": "kraus_boundary_contour_combined_report_v1",
        "bootstrap_seed": SEED,
        "inputs": {
            "legacy_full_history": {"samples": 10, "cycles": 40, "snapshots": 410},
            "purification_endpoints": {"samples": 600, "sizes": endpoint_ny.tolist()},
            "bundle13_gap_contours": {"samples": 700, "sizes": gap_ny.tolist()},
            "event_replay": event_summary["scientific_contract"],
        },
        "legacy_endpoint_mean_first_four_soft_wall_weight": float(legacy_mean[-1]),
        "endpoint_first_four_soft_wall_weight": endpoint_stats,
        "bundle13_endpoint_median_first_gap_wall_fraction": {
            str(ny): float(value) for ny, value in zip(gap_ny, first_gap_medians)
        },
        "bundle13_endpoint_median_first_four_soft_wall_weight": {
            str(ny): float(value) for ny, value in zip(gap_ny, soft_medians)
        },
        "late_window_gap_coefficients_Ai": gap_fit["coefficients"].tolist(),
        "late_window_gap_coefficient_ci": gap_fit["coefficient_ci"].tolist(),
        "late_window_gap_ratios_Ai_over_A1": gap_fit["ratios"].tolist(),
        "late_window_gap_ratio_ci": gap_fit["ratio_ci"].tolist(),
        "event_replay_checks": {
            "maximum_covariance_replay_error": event_summary[
                "maximum_covariance_replay_error"
            ],
            "center_contour_sum_rule_error": event_summary[
                "center_contour_sum_rule_error"
            ],
            "support_contour_sum_rule_error": event_summary[
                "support_contour_sum_rule_error"
            ],
        },
        "interpretation": {
            "boundary_assignment": "supported",
            "gap_contours": "supported for ordered low-lying gaps",
            "central_charge_from_existing_data": "not identified: production data lack an event-resolved record-weight contour and a matched bulk subtraction",
            "operator_scaling": "relative ordered-gap coefficients are accessible; absolute CFT labels require sector-resolved boundary data",
        },
    }
    (OUTPUT / "combined_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    ratios = summary["late_window_gap_ratios_Ai_over_A1"]
    markdown = f"""# Kraus boundary-contour results

All four checks completed.  The inputs remain separate rather than being pooled.

- The complete legacy covariance history contains 10 trajectories, 41 time slices each.  The endpoint mean wall weight of the first four soft modes is {legacy_mean[-1]:.4f}.
- The independent hard/soft endpoint ensembles contain 600 trajectories.  Across their six construction/size cells, the first-four-mode mean wall weight lies between {min(min(value['mean']) for value in endpoint_stats.values()):.4f} and {max(max(value['mean']) for value in endpoint_stats.values()):.4f}.
- The bundle-13 reconstruction covers 700 trajectories and 27,900 spectrum checkpoints.  The endpoint first-gap wall fraction lies between {min(first_gap_medians):.4f} and {max(first_gap_medians):.4f}.
- The deterministic event replay reproduces the covariance to {event_summary['maximum_covariance_replay_error']:.3e}; the support-contour sum rule closes to {event_summary['support_contour_sum_rule_error']:.3e}.
- Fitting late-window ordered gaps to $\\Delta_i=A_i/N_y$ gives $A_i/A_1={ratios[0]:.3f}, {ratios[1]:.3f}, {ratios[2]:.3f}, {ratios[3]:.3f}$.

## Interpretation

The low-lying spectrum is genuinely boundary-localized; this is not an artifact of one system size, wall construction, or ten-sample legacy file.  Additive gap contours are therefore meaningful.  Relative ordered-gap coefficients can be extracted without the extensive leading level.

The present production files still do **not** determine a boundary central charge.  Their saved covariances determine the spectral term, but not the event-resolved spatial contour of the record log probability.  The new replay proves that the missing contour can be recorded exactly.  A production central-charge analysis would need that event contour plus a matched bulk/reference subtraction.  Sector labels are also needed before assigning the ordered gaps to named boundary operators.
"""
    (OUTPUT / "RESULTS.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[done] {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
