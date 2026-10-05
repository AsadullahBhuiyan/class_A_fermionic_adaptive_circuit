#!/usr/bin/env python3
"""Build cycle-resolved and late-time S100 bulk-Chern summary figures."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import build_uniform_bulk_chern_L12_L32 as base


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = base.OUTPUT_DIR

CYCLE_STEM = "fig03_uniform_bulk_chern_cycle_convergence_L16_L24_L32_s100"
FINAL_STEM = "fig04_uniform_bulk_chern_late_time_L12_L32_s100"
CYCLE_PDF = OUTPUT_DIR / f"{CYCLE_STEM}.pdf"
CYCLE_PNG = OUTPUT_DIR / f"{CYCLE_STEM}.png"
FINAL_PDF = OUTPUT_DIR / f"{FINAL_STEM}.pdf"
FINAL_PNG = OUTPUT_DIR / f"{FINAL_STEM}.png"
SOURCE_DATA = OUTPUT_DIR / "fig03_04_uniform_bulk_chern_source_data.csv"
FINAL_SOURCE_DATA = OUTPUT_DIR / "fig04_uniform_bulk_chern_late_time_source_data.csv"
SUMMARY_JSON = OUTPUT_DIR / "fig03_04_uniform_bulk_chern_summary.json"
LATE_TIME_START = 21
LATE_TIME_STOP = 40
CYCLE_FIGURE_SIZES = (16, 24, 32)
CYCLE_SIZE_COLORS = {
    16: "#D55E4A",
    24: "#3A9D5D",
    32: "#2878B5",
}
CYCLE_SHELL_STYLES = {
    "1": {"marker": "^", "linestyle": (0, (1.0, 1.6))},
    "2": {"marker": "s", "linestyle": (0, (5.0, 2.0))},
    "dense": {"marker": "o", "linestyle": "-"},
}

SHELL_STYLES = {
    "1": {"color": "#D55E00", "marker": "^", "linestyle": ":"},
    "2": {"color": "#009E73", "marker": "s", "linestyle": "--"},
    "dense": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
}


def _sem_statistics(values: np.ndarray) -> dict[str, np.ndarray]:
    """Return the sample mean and its trajectory-to-trajectory standard error."""

    sample_count = values.shape[0]
    if sample_count < 2:
        raise ValueError("at least two trajectories are required for a standard error")
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / np.sqrt(float(sample_count))
    chern_low = mean - sem
    chern_high = mean + sem
    deviation = np.abs(mean - 1.0)
    deviation_from_low = np.abs(chern_low - 1.0)
    deviation_from_high = np.abs(chern_high - 1.0)
    crosses_unity = (chern_low <= 1.0) & (chern_high >= 1.0)
    deviation_low = np.where(
        crosses_unity, 0.0, np.minimum(deviation_from_low, deviation_from_high)
    )
    deviation_high = np.maximum(deviation_from_low, deviation_from_high)
    return {
        "mean_real_space_chern": mean,
        "chern_standard_error": sem,
        "chern_sem_low": chern_low,
        "chern_sem_high": chern_high,
        "abs_mean_chern_minus_one": deviation,
        "abs_deviation_sem_low": deviation_low,
        "abs_deviation_sem_high": deviation_high,
    }


def _summarize(cases: dict[tuple[int, str], np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for shell in base.SHELLS:
        for size in base.SIZES:
            statistics = _sem_statistics(cases[(size, shell)])
            for cycle_index, cycle in enumerate(base.SAVED_CYCLES):
                row: dict[str, object] = {
                    "L": size,
                    "n_shell": shell,
                    "cycle": int(cycle),
                    "trajectories": base.SAMPLE_COUNT,
                }
                for key, values in statistics.items():
                    row[key] = float(values[cycle_index])
                rows.append(row)
    return pd.DataFrame(rows)


def _summarize_late_time(
    cases: dict[tuple[int, str], np.ndarray],
) -> pd.DataFrame:
    cycle_mask = (base.SAVED_CYCLES >= LATE_TIME_START) & (
        base.SAVED_CYCLES <= LATE_TIME_STOP
    )
    if int(cycle_mask.sum()) != LATE_TIME_STOP - LATE_TIME_START + 1:
        raise ValueError("late-time window does not contain every integer cycle")
    rows: list[dict[str, object]] = []
    for shell in base.SHELLS:
        for size in base.SIZES:
            sample_cycle_values = cases[(size, shell)][:, cycle_mask].reshape(-1, 1)
            statistics = _sem_statistics(sample_cycle_values)
            row: dict[str, object] = {
                "L": size,
                "n_shell": shell,
                "cycle_start": LATE_TIME_START,
                "cycle_stop": LATE_TIME_STOP,
                "saved_cycles": int(cycle_mask.sum()),
                "trajectories": base.SAMPLE_COUNT,
                "ensemble_values": int(sample_cycle_values.shape[0]),
            }
            for key, values in statistics.items():
                row[key] = float(values[0])
            rows.append(row)
    return pd.DataFrame(rows)


def _save_pdf_and_png(fig: plt.Figure, pdf_path: Path, png_path: Path) -> None:
    fig.savefig(pdf_path)
    plt.close(fig)
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-r",
            "300",
            "-singlefile",
            str(pdf_path),
            str(png_path.with_suffix("")),
        ],
        check=True,
    )


def _size_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=CYCLE_SIZE_COLORS[size],
            linestyle="-",
            linewidth=1.3,
            label=rf"$L={size}$",
        )
        for size in CYCLE_FIGURE_SIZES
    ]


def _cycle_shell_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=CYCLE_SHELL_STYLES[shell]["linestyle"],
            marker=CYCLE_SHELL_STYLES[shell]["marker"],
            markerfacecolor="white",
            markeredgewidth=0.8,
            linewidth=1.35,
            markersize=4.0,
            label=base.SHELL_LABELS[shell],
        )
        for shell in base.SHELLS
    ]


def _draw_cycle_figure(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(3.375, 4.65),
        sharex=True,
        constrained_layout=False,
    )
    marker_cycles = np.arange(base.SAVED_CYCLES.size)

    selected = summary.loc[
        summary["L"].isin(CYCLE_FIGURE_SIZES)
        & summary["n_shell"].isin(base.SHELLS)
    ]
    chern_low = float(selected["chern_sem_low"].min())
    chern_high = float(selected["chern_sem_high"].max())
    chern_padding = 0.035 * (chern_high - chern_low)
    positive_deviation_bounds = selected.loc[
        selected["abs_deviation_sem_low"] > 0.0,
        ["abs_deviation_sem_low", "abs_deviation_sem_high"],
    ].to_numpy()
    if positive_deviation_bounds.size == 0:
        raise ValueError("log-deviation panels have no positive standard-error bounds")
    deviation_floor = 10.0 ** np.floor(np.log10(positive_deviation_bounds.min()))
    deviation_ceiling = 10.0 ** np.ceil(np.log10(positive_deviation_bounds.max()))

    left, right = axes
    for shell in base.SHELLS:
        shell_style = CYCLE_SHELL_STYLES[shell]
        for size in CYCLE_FIGURE_SIZES:
            group = summary.loc[
                (summary["n_shell"] == shell) & (summary["L"] == size)
            ].sort_values("cycle")
            cycles = group["cycle"].to_numpy(dtype=np.float64)
            color = CYCLE_SIZE_COLORS[size]

            mean = group["mean_real_space_chern"].to_numpy(dtype=np.float64)
            mean_low = group["chern_sem_low"].to_numpy(dtype=np.float64)
            mean_high = group["chern_sem_high"].to_numpy(dtype=np.float64)
            left.fill_between(
                cycles, mean_low, mean_high, color=color, alpha=0.035, linewidth=0
            )
            left.plot(
                cycles,
                mean,
                color=color,
                linestyle=shell_style["linestyle"],
                linewidth=0.8,
                marker=shell_style["marker"],
                markevery=marker_cycles,
                markersize=1.9,
                markerfacecolor="white",
                markeredgewidth=0.5,
                zorder=3,
            )

            deviation = group["abs_mean_chern_minus_one"].to_numpy(dtype=np.float64)
            deviation_low = group["abs_deviation_sem_low"].to_numpy(
                dtype=np.float64
            )
            deviation_high = group["abs_deviation_sem_high"].to_numpy(
                dtype=np.float64
            )
            right.fill_between(
                cycles,
                np.maximum(deviation_low, deviation_floor),
                np.maximum(deviation_high, deviation_floor),
                color=color,
                alpha=0.035,
                linewidth=0,
            )
            right.plot(
                cycles,
                np.maximum(deviation, deviation_floor),
                color=color,
                linestyle=shell_style["linestyle"],
                linewidth=0.8,
                marker=shell_style["marker"],
                markevery=marker_cycles,
                markersize=1.9,
                markerfacecolor="white",
                markeredgewidth=0.5,
                zorder=3,
            )

    for axis, letter in zip(axes, "ab"):
        axis.set_xlim(-0.8, 40.8)
        axis.set_xticks((0, 10, 20, 30, 40))
        axis.text(
            -0.14,
            1.025,
            f"({letter})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
        )
    left.axhline(1.0, color="black", linestyle="--", linewidth=0.75, zorder=1)
    left.set_ylim(chern_low - chern_padding, chern_high + chern_padding)
    left.set_ylabel(r"sample mean $\overline{\mathcal{C}_G}$")
    right.set_yscale("log")
    right.set_ylim(deviation_floor, deviation_ceiling)
    right.set_ylabel(r"$\lvert\overline{\mathcal{C}_G}-1\rvert$")
    right.set_xlabel("cycle")
    size_handles = _size_legend_handles()
    shell_handles = _cycle_shell_legend_handles()
    legend_handles = [
        handle
        for size_handle, shell_handle in zip(size_handles, shell_handles)
        for handle in (size_handle, shell_handle)
    ]
    left.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.055),
        ncol=3,
        frameon=False,
        handlelength=2.25,
        handletextpad=0.35,
        columnspacing=0.45,
        labelspacing=0.18,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(
        left=0.19,
        right=0.985,
        bottom=0.095,
        top=0.94,
        hspace=0.10,
    )
    _save_pdf_and_png(fig, CYCLE_PDF, CYCLE_PNG)


def _shell_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=SHELL_STYLES[shell]["color"],
            linestyle=SHELL_STYLES[shell]["linestyle"],
            marker=SHELL_STYLES[shell]["marker"],
            markerfacecolor="white",
            markeredgewidth=0.8,
            linewidth=1.15,
            markersize=4.2,
            label=base.SHELL_LABELS[shell],
        )
        for shell in base.SHELLS
    ]


def _draw_final_figure(late_time: pd.DataFrame) -> None:
    positive_lower_bounds = late_time.loc[
        late_time["abs_deviation_sem_low"] > 0.0, "abs_deviation_sem_low"
    ].to_numpy(dtype=np.float64)
    if positive_lower_bounds.size == 0:
        raise ValueError("final log-deviation panel has no positive lower bounds")
    deviation_floor = 10.0 ** np.floor(np.log10(positive_lower_bounds.min()))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(3.375, 4.8),
        sharex=True,
        constrained_layout=False,
    )

    for shell in base.SHELLS:
        group = late_time.loc[late_time["n_shell"] == shell].sort_values("L")
        sizes = group["L"].to_numpy(dtype=np.float64)
        style = SHELL_STYLES[shell]

        mean = group["mean_real_space_chern"].to_numpy(dtype=np.float64)
        mean_yerr = np.vstack(
            (
                group["chern_standard_error"].to_numpy(dtype=np.float64),
                group["chern_standard_error"].to_numpy(dtype=np.float64),
            )
        )
        # Dense cases can be trajectory-identical, leaving only floating-point
        # roundoff between the point estimate and a nominally equal quantile.
        mean_yerr = np.maximum(mean_yerr, 0.0)
        axes[0].errorbar(
            sizes,
            mean,
            yerr=mean_yerr,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor="white",
            markeredgewidth=0.8,
            linewidth=1.15,
            markersize=4.2,
            capsize=2.0,
            elinewidth=0.8,
        )

        deviation = group["abs_mean_chern_minus_one"].to_numpy(dtype=np.float64)
        deviation_low = group["abs_deviation_sem_low"].to_numpy(dtype=np.float64)
        deviation_low_for_plot = np.where(
            deviation_low == 0.0, deviation_floor, deviation_low
        )
        deviation_yerr = np.vstack(
            (
                deviation - deviation_low_for_plot,
                group["abs_deviation_sem_high"].to_numpy(dtype=np.float64)
                - deviation,
            )
        )
        # The absolute-value transform can make the propagated interval
        # asymmetric. Numerical roundoff at a boundary is clipped.
        deviation_yerr = np.maximum(deviation_yerr, 0.0)
        axes[1].errorbar(
            sizes,
            deviation,
            yerr=deviation_yerr,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor="white",
            markeredgewidth=0.8,
            linewidth=1.15,
            markersize=4.2,
            capsize=2.0,
            elinewidth=0.8,
        )

    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.75, zorder=1)
    axes[0].set_ylabel(r"late-time mean $\overline{\mathcal{C}_G}$")
    axes[1].set_ylabel(r"$\lvert\overline{\mathcal{C}_G}-1\rvert$, cycles 21--40")
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=deviation_floor)
    for axis, letter in zip(axes, "ab"):
        axis.set_xticks(base.SIZES)
        axis.set_xlim(10.7, 33.3)
        axis.text(
            -0.13,
            1.035,
            f"({letter})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
        )
    axes[1].set_xlabel(r"system size $L$")
    fig.legend(
        handles=_shell_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=0.75,
    )
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.105, top=0.90, hspace=0.14)
    _save_pdf_and_png(fig, FINAL_PDF, FINAL_PNG)


def main() -> int:
    cases, verification = base._load_cases()
    summary = _summarize(cases)
    late_time = _summarize_late_time(cases)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base._configure_style()
    summary.to_csv(SOURCE_DATA, index=False, float_format="%.17g")
    late_time.to_csv(FINAL_SOURCE_DATA, index=False, float_format="%.17g")
    _draw_cycle_figure(summary)
    _draw_final_figure(late_time)

    caption_cycle = (
        r"Uniform perfect-correction bulk topology without domain walls. The upper panel shows the "
        r"sample-averaged real-space Chern estimator; the lower panel shows "
        r"$|\overline{C_G}-1|$ on a logarithmic scale. Red, green, and blue "
        r"distinguish $L=16,24,32$; markers and line styles distinguish "
        r"$n_{\rm shell}=1,2,\infty$. Each curve contains $S=100$ "
        r"independent random-pure half-filled Born trajectories evolved for 40 "
        r"random-serial perfect-correction cycles in complex128. $C_G$ is evaluated "
        r"trajectory by trajectory before averaging. Bands are the sample-to-sample "
        r"standard error over the 100 independent trajectories."
    )
    caption_final = (
        r"Late-time finite-size summary of the same uniform $S=100$ campaign. "
        r"The mean and standard error use the 2,000 sample-cycle values from "
        r"100 trajectories and cycles 21--40. The right panel shows "
        r"the absolute deviation of this late-time mean from unity on a "
        r"logarithmic scale for $n_{\rm shell}=1,2,\infty$."
    )
    payload = {
        "schema": "uniform_bulk_chern_convergence_figures_v2",
        "campaigns": list(base.CAMPAIGNS),
        "sizes": list(base.SIZES),
        "cycle_figure_sizes": list(CYCLE_FIGURE_SIZES),
        "shells": list(base.SHELLS),
        "cycles": list(map(int, base.SAVED_CYCLES)),
        "trajectories_per_case": base.SAMPLE_COUNT,
        "total_trajectories": len(base.SIZES) * len(base.SHELLS) * base.SAMPLE_COUNT,
        "estimator_order": "trajectory-resolved C_G, sample mean, then absolute deviation from unity",
        "late_time_estimator_order": "mean and SEM across the 100 by 20 sample-cycle ensemble from cycles 21--40, then absolute deviation from unity",
        "late_time_window": {
            "cycle_start": LATE_TIME_START,
            "cycle_stop": LATE_TIME_STOP,
            "inclusive": True,
            "saved_cycles": LATE_TIME_STOP - LATE_TIME_START + 1,
        },
        "uncertainty": {
            "unit": "independent trajectory",
            "statistic": "sample standard deviation divided by sqrt(S)",
            "sample_standard_deviation_ddof": 1,
            "sample_count": base.SAMPLE_COUNT,
            "absolute_deviation_method": "propagate mean plus/minus one standard error through abs(mean minus one)",
        },
        "late_time_uncertainty": {
            "ensemble": "100 trajectories by 20 cycles",
            "ensemble_values": base.SAMPLE_COUNT
            * (LATE_TIME_STOP - LATE_TIME_START + 1),
            "statistic": "sample standard deviation across sample-cycle values divided by sqrt(2000)",
            "sample_standard_deviation_ddof": 1,
        },
        "verification": {
            "result_completion_pairs": len(verification),
            "all_result_sizes_and_sha256_verified": True,
            "all_selected_cases_are_S100": True,
            "cycles_are_exactly_0_through_40": True,
            "charge_rank_identity_verified": True,
            "inputs": verification,
        },
        "captions": {"cycle_figure": caption_cycle, "final_figure": caption_final},
        "outputs": {
            str(CYCLE_PDF.relative_to(PROJECT_DIR)): base._fingerprint(CYCLE_PDF),
            str(CYCLE_PNG.relative_to(PROJECT_DIR)): base._fingerprint(CYCLE_PNG),
            str(FINAL_PDF.relative_to(PROJECT_DIR)): base._fingerprint(FINAL_PDF),
            str(FINAL_PNG.relative_to(PROJECT_DIR)): base._fingerprint(FINAL_PNG),
            str(SOURCE_DATA.relative_to(PROJECT_DIR)): base._fingerprint(SOURCE_DATA),
            str(FINAL_SOURCE_DATA.relative_to(PROJECT_DIR)): base._fingerprint(FINAL_SOURCE_DATA),
        },
        "late_time_average": late_time.to_dict(orient="records"),
    }
    SUMMARY_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "cycle_figure": str(CYCLE_PNG),
                "final_figure": str(FINAL_PNG),
                "source_data": str(SOURCE_DATA),
                "late_time_source_data": str(FINAL_SOURCE_DATA),
                "summary": str(SUMMARY_JSON),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
