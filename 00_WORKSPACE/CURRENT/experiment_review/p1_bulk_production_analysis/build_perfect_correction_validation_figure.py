#!/usr/bin/env python3
"""Build the manuscript-scale uniform perfect-correction validation figure.

The frozen P1 trajectory table is read-only input.  The nonlinear real-space
Chern estimator is always reduced trajectory by trajectory before averaging.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import linregress


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "production_25sample_v1"
FIGURE_DIR = OUTPUT_DIR / "figures"
TRAJECTORY_TABLE = OUTPUT_DIR / "trajectory_observations.csv"
CHERN_SUMMARY_TABLE = OUTPUT_DIR / "cycle_resolved_chern_summary.csv"
ANALYSIS_MANIFEST = OUTPUT_DIR / "analysis_manifest.json"

FIGURE_STEM = "fig10_perfect_correction_bulk_validation"
PDF_PATH = FIGURE_DIR / f"{FIGURE_STEM}.pdf"
PNG_PATH = FIGURE_DIR / f"{FIGURE_STEM}.png"
SOURCE_DATA_PATH = OUTPUT_DIR / f"{FIGURE_STEM}_source_data.csv"
SUMMARY_PATH = OUTPUT_DIR / f"{FIGURE_STEM}_summary.json"

BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2026082101
FIGURE_WIDTH = 7.05
FIGURE_HEIGHT = 5.20
FIGURE_DPI = 300
SELECTED_SIZES = (12, 20, 32)
ALL_SIZES = (12, 16, 20, 24, 28, 32)
SHELL_ORDER = ("1", "2", "dense")

BPJ_RED = "#D92725"
BPJ_GREEN = "#2CA02C"
BPJ_BLUE = "#1F77B4"
BPJ_BLACK = "#000000"
SHELL_COLORS = {"1": BPJ_RED, "2": BPJ_GREEN, "dense": BPJ_BLUE}
SHELL_MARKERS = {"1": "^", "2": "s", "dense": "o"}
SHELL_LABELS = {
    "1": r"$n_{\rm shell}=1$",
    "2": r"$n_{\rm shell}=2$",
    "dense": "dense",
}
SIZE_LINESTYLES = {12: ":", 20: "-", 32: (0, (4, 1.4, 1.1, 1.4))}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _keyed_rng(label: str) -> np.random.Generator:
    raw = f"{BOOTSTRAP_SEED}:{label}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")
    return np.random.default_rng(seed)


def _bootstrap_means(values: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (25,) or not np.isfinite(values).all():
        raise ValueError(f"{label}: expected 25 finite trajectory values, got {values.shape}")
    rng = _keyed_rng(label)
    indices = rng.integers(
        0, values.size, size=(BOOTSTRAP_REPLICATES, values.size)
    )
    return values[indices].mean(axis=1)


def _mean_ci(values: np.ndarray, label: str) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    replicates = _bootstrap_means(values, label)
    low, high = np.quantile(replicates, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def _errorbar(
    axis: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    shell: str,
    size: int,
) -> None:
    yerr = np.vstack((mean - low, high - mean))
    if np.any(yerr < -1e-15):
        raise ValueError("confidence interval does not contain its point estimate")
    axis.errorbar(
        x,
        mean,
        yerr=np.maximum(yerr, 0.0),
        color=SHELL_COLORS[shell],
        marker=SHELL_MARKERS[shell],
        markerfacecolor="white",
        markeredgewidth=0.75,
        markersize=3.2,
        linestyle=SIZE_LINESTYLES[size],
        linewidth=0.85,
        elinewidth=0.5,
        capsize=1.35,
        capthick=0.5,
        alpha=0.92,
        zorder=3,
    )


def _panel_label(axis: plt.Axes, letter: str) -> None:
    axis.text(
        -0.14,
        1.035,
        f"({letter})",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )


def _configure_style() -> None:
    latex_support = PROJECT_DIR / "latex_support"
    os.environ["TEXINPUTS"] = (
        str(latex_support) + os.pathsep + os.environ.get("TEXINPUTS", "")
    )
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": FIGURE_DPI,
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "text.usetex": True,
            "text.latex.preamble": (
                r"\usepackage{amsmath}\renewcommand{\familydefault}{\sfdefault}"
            ),
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "lines.markersize": 4.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in (TRAJECTORY_TABLE, CHERN_SUMMARY_TABLE, ANALYSIS_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)

    trajectories = pd.read_csv(TRAJECTORY_TABLE)
    chern_summary = pd.read_csv(CHERN_SUMMARY_TABLE)
    if len(trajectories) != 7200:
        raise ValueError(f"expected 7200 frozen P1 observations, found {len(trajectories)}")
    trajectories["shell"] = trajectories["shell"].astype(str)
    chern_summary["shell"] = chern_summary["shell"].astype(str)
    return trajectories, chern_summary


def _build_panel_a(
    trajectories: pd.DataFrame, chern_summary: pd.DataFrame
) -> pd.DataFrame:
    selected = chern_summary.loc[
        (chern_summary["initialization"] == "pure")
        & (chern_summary["phase"] == "topological")
        & chern_summary["L"].isin(SELECTED_SIZES)
        & chern_summary["shell"].isin(SHELL_ORDER)
    ].copy()
    if len(selected) != len(SELECTED_SIZES) * len(SHELL_ORDER) * 6:
        raise ValueError(f"panel (a): expected 54 summary rows, found {len(selected)}")

    direct = trajectories.loc[
        (trajectories["initialization"] == "pure")
        & (trajectories["phase"] == "topological")
        & trajectories["L"].isin(SELECTED_SIZES)
        & trajectories["shell"].isin(SHELL_ORDER)
    ]
    counts = direct.groupby(["shell", "L", "cycle"]).size()
    if not (counts == 25).all() or len(counts) != 54:
        raise ValueError("panel (a): every shell/size/checkpoint must have S=25")
    direct_means = direct.groupby(["shell", "L", "cycle"])[
        "real_space_chern"
    ].mean()
    for row in selected.itertuples(index=False):
        observed = direct_means.loc[(row.shell, row.L, row.cycle)]
        if not np.isclose(observed, row.chern_mean, atol=1e-14, rtol=0):
            raise ValueError("panel (a): saved summary does not match trajectory table")
    return selected.sort_values(["shell", "L", "cycle"]).reset_index(drop=True)


def _build_panel_b(trajectories: pd.DataFrame) -> pd.DataFrame:
    selected = trajectories.loc[
        (trajectories["initialization"] == "pure")
        & (trajectories["phase"] == "topological")
        & trajectories["L"].isin(SELECTED_SIZES)
        & trajectories["shell"].isin(SHELL_ORDER)
    ].copy()
    selected["absolute_chern_error"] = np.abs(selected["real_space_chern"] - 1.0)
    rows: list[dict[str, float | int | str]] = []
    keys = ["shell", "L", "cycle", "normalized_cycle"]
    for (shell, size, cycle, normalized_cycle), group in selected.groupby(
        keys, sort=True
    ):
        values = group["absolute_chern_error"].to_numpy(dtype=np.float64)
        mean, low, high = _mean_ci(
            values,
            f"cycle-chern-error:pure:{shell}:L{int(size)}:cycle{int(cycle)}",
        )
        rows.append(
            {
                "shell": shell,
                "L": int(size),
                "cycle": int(cycle),
                "normalized_cycle": float(normalized_cycle),
                "trajectories": len(group),
                "mean_absolute_chern_error": mean,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 54 or (result["trajectories"] != 25).any():
        raise ValueError("panel (b): incomplete checkpoint groups")
    if not (
        result[["mean_absolute_chern_error", "ci95_low", "ci95_high"]]
        .to_numpy()
        > 0
    ).all():
        raise ValueError("panel (b): logarithmic values must be positive")
    return result.sort_values(["shell", "L", "cycle"]).reset_index(drop=True)


def _build_charge_panels(
    trajectories: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | list[float]]]:
    final = trajectories.loc[
        (trajectories["initialization"] == "pure")
        & (trajectories["phase"] == "topological")
        & (trajectories["shell"] == "1")
        & np.isclose(trajectories["normalized_cycle"], 2.0)
    ].copy()
    if len(final) != len(ALL_SIZES) * 25:
        raise ValueError(f"charge panels: expected 150 final trajectories, found {len(final)}")

    final["delta_Q_float"] = (final["density_mean"] - 1.0) * final["L"] ** 2
    final["delta_Q"] = np.rint(final["delta_Q_float"]).astype(np.int64)
    residual = np.max(np.abs(final["delta_Q_float"] - final["delta_Q"]))
    if residual > 1e-8:
        raise ValueError(f"charge is not integer within tolerance: residual={residual}")
    final["absolute_half_filling_deviation_percent"] = (
        100.0 * np.abs(final["delta_Q"]) / final["L"] ** 2
    )

    histogram_values = final.loc[final["L"] == 32, "delta_Q"]
    if len(histogram_values) != 25:
        raise ValueError("panel (c): expected S=25 at L=32")
    histogram = (
        histogram_values.value_counts()
        .sort_index()
        .rename_axis("delta_Q")
        .reset_index(name="trajectory_count")
    )

    scaling_rows: list[dict[str, float | int]] = []
    bootstrap_by_size: dict[int, np.ndarray] = {}
    for size in ALL_SIZES:
        values = final.loc[
            final["L"] == size, "absolute_half_filling_deviation_percent"
        ].to_numpy(dtype=np.float64)
        mean, low, high = _mean_ci(values, f"half-filling-deviation:nsh1:L{size}")
        replicates = _bootstrap_means(values, f"half-filling-deviation:nsh1:L{size}")
        bootstrap_by_size[size] = replicates
        scaling_rows.append(
            {
                "L": size,
                "trajectories": values.size,
                "mean_absolute_half_filling_deviation_percent": mean,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    scaling = pd.DataFrame(scaling_rows)
    sizes = scaling["L"].to_numpy(dtype=np.float64)
    means = scaling["mean_absolute_half_filling_deviation_percent"].to_numpy()
    fit = linregress(np.log(sizes), np.log(means))
    bootstrap_matrix = np.column_stack(
        [bootstrap_by_size[size] for size in ALL_SIZES]
    )
    valid_fit = np.all(bootstrap_matrix > 0, axis=1)
    if int(valid_fit.sum()) < int(0.99 * BOOTSTRAP_REPLICATES):
        raise ValueError("too many undefined logarithmic charge-fit replicates")
    exponents = np.asarray(
        [
            -linregress(np.log(sizes), np.log(sample_means)).slope
            for sample_means in bootstrap_matrix[valid_fit]
        ],
        dtype=np.float64,
    )
    exponent_low, exponent_high = np.quantile(exponents, [0.025, 0.975])
    fit_summary: dict[str, float | int | list[float]] = {
        "model": "A*L**(-p)",
        "sizes": list(ALL_SIZES),
        "A": float(np.exp(fit.intercept)),
        "p": float(-fit.slope),
        "p_ci95": [float(exponent_low), float(exponent_high)],
        "r_squared_log_means": float(fit.rvalue**2),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "valid_log_fit_replicates": int(valid_fit.sum()),
    }
    return histogram, scaling, fit_summary


def _source_data(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    histogram: pd.DataFrame,
    scaling: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    a = panel_a.rename(
        columns={
            "chern_mean": "value",
            "chern_ci95_low": "ci95_low",
            "chern_ci95_high": "ci95_high",
        }
    ).copy()
    a["panel"] = "a"
    a["metric"] = "trajectory_mean_real_space_chern"
    frames.append(a)

    b = panel_b.rename(columns={"mean_absolute_chern_error": "value"}).copy()
    b["panel"] = "b"
    b["metric"] = "trajectory_mean_absolute_real_space_chern_error"
    b["initialization"] = "pure"
    b["phase"] = "topological"
    frames.append(b)

    c = histogram.copy()
    c["panel"] = "c"
    c["metric"] = "delta_Q_histogram_count"
    c["value"] = c["trajectory_count"]
    c["L"] = 32
    c["shell"] = "1"
    c["cycle"] = 64
    c["normalized_cycle"] = 2.0
    c["trajectories"] = 25
    c["initialization"] = "pure"
    c["phase"] = "topological"
    frames.append(c)

    d = scaling.rename(
        columns={"mean_absolute_half_filling_deviation_percent": "value"}
    ).copy()
    d["panel"] = "d"
    d["metric"] = "mean_absolute_half_filling_deviation_percent"
    d["shell"] = "1"
    d["cycle"] = 2 * d["L"]
    d["normalized_cycle"] = 2.0
    d["initialization"] = "pure"
    d["phase"] = "topological"
    frames.append(d)

    columns = [
        "panel",
        "metric",
        "initialization",
        "phase",
        "shell",
        "L",
        "cycle",
        "normalized_cycle",
        "trajectories",
        "delta_Q",
        "value",
        "ci95_low",
        "ci95_high",
    ]
    result = pd.concat(frames, ignore_index=True, sort=False)
    return result.reindex(columns=columns)


def _draw_figure(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    histogram: pd.DataFrame,
    scaling: pd.DataFrame,
    charge_fit: dict[str, float | int | list[float]],
) -> None:
    _configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    for shell in SHELL_ORDER:
        for size in SELECTED_SIZES:
            group = panel_a.loc[
                (panel_a["shell"] == shell) & (panel_a["L"] == size)
            ].sort_values("normalized_cycle")
            _errorbar(
                ax_a,
                group["normalized_cycle"].to_numpy(),
                group["chern_mean"].to_numpy(),
                group["chern_ci95_low"].to_numpy(),
                group["chern_ci95_high"].to_numpy(),
                shell=shell,
                size=size,
            )
            error = panel_b.loc[
                (panel_b["shell"] == shell) & (panel_b["L"] == size)
            ].sort_values("normalized_cycle")
            _errorbar(
                ax_b,
                error["normalized_cycle"].to_numpy(),
                error["mean_absolute_chern_error"].to_numpy(),
                error["ci95_low"].to_numpy(),
                error["ci95_high"].to_numpy(),
                shell=shell,
                size=size,
            )

    ax_a.axhline(1.0, color=BPJ_BLACK, linestyle="--", linewidth=0.75, zorder=1)
    ax_a.set_xlabel(r"normalized circuit depth $t/L$")
    ax_a.set_ylabel(r"$\langle C_G\rangle$")
    ax_a.set_xticks([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    ax_a.set_xlim(0.19, 2.06)
    a_low = float(panel_a["chern_ci95_low"].min())
    a_high = float(panel_a["chern_ci95_high"].max())
    a_pad = max(0.00035, 0.07 * (a_high - a_low))
    ax_a.set_ylim(a_low - a_pad, a_high + a_pad)

    ax_b.set_yscale("log")
    ax_b.set_xlabel(r"normalized circuit depth $t/L$")
    ax_b.set_ylabel(r"$\langle|C_G-1|\rangle$")
    ax_b.set_xticks([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    ax_b.set_xlim(0.19, 2.06)

    ax_c.bar(
        histogram["delta_Q"],
        histogram["trajectory_count"],
        width=0.82,
        color=BPJ_RED,
        edgecolor=BPJ_BLACK,
        linewidth=0.65,
        alpha=0.82,
    )
    ax_c.axvline(0, color=BPJ_BLACK, linestyle="--", linewidth=0.75)
    ax_c.set_xticks(
        np.arange(int(histogram["delta_Q"].min()), int(histogram["delta_Q"].max()) + 1)
    )
    ax_c.set_xlabel(r"global charge offset $\Delta Q=Q-L^2$")
    ax_c.set_ylabel("trajectories")
    ax_c.set_title(r"$L=32$, $n_{\rm shell}=1$, $t=2L$", pad=3)
    ax_c.set_ylim(0, 1.16 * float(histogram["trajectory_count"].max()))

    sizes = scaling["L"].to_numpy(dtype=np.float64)
    means = scaling["mean_absolute_half_filling_deviation_percent"].to_numpy()
    low = scaling["ci95_low"].to_numpy()
    high = scaling["ci95_high"].to_numpy()
    ax_d.errorbar(
        sizes,
        means,
        yerr=np.vstack((means - low, high - means)),
        color=BPJ_RED,
        marker="^",
        markerfacecolor="white",
        markeredgewidth=0.8,
        linestyle="none",
        markersize=4.0,
        elinewidth=0.65,
        capsize=2.0,
        zorder=3,
        label="trajectory mean",
    )
    fit_x = np.linspace(float(min(ALL_SIZES)), float(max(ALL_SIZES)), 300)
    fit_y = float(charge_fit["A"]) * fit_x ** (-float(charge_fit["p"]))
    ax_d.plot(
        fit_x,
        fit_y,
        color=BPJ_BLACK,
        linestyle="--",
        linewidth=0.85,
        label=r"$A L^{-p}$ fit",
    )
    exponent_ci = charge_fit["p_ci95"]
    assert isinstance(exponent_ci, list)
    ax_d.text(
        0.96,
        0.94,
        (
            rf"$p={float(charge_fit['p']):.2f}$ "
            rf"$[{float(exponent_ci[0]):.2f},{float(exponent_ci[1]):.2f}]$"
            "\n"
            rf"$R^2={float(charge_fit['r_squared_log_means']):.2f}$"
        ),
        transform=ax_d.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
    )
    ax_d.set_xlabel(r"linear size $L$")
    ax_d.set_ylabel(r"$100\,\langle|\Delta Q|\rangle/L^2$ (\%)")
    ax_d.set_xticks(ALL_SIZES)
    ax_d.set_ylim(0, max(0.54, 1.08 * float(high.max())))
    ax_d.legend(loc="upper center", bbox_to_anchor=(0.51, 0.76), fontsize=7)

    for axis, letter in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        _panel_label(axis, letter)
        axis.tick_params(direction="in")
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)

    shell_handles = [
        Line2D(
            [0],
            [0],
            color=SHELL_COLORS[shell],
            marker=SHELL_MARKERS[shell],
            markerfacecolor="white",
            linestyle="-",
            linewidth=0.85,
            markersize=3.5,
            label=SHELL_LABELS[shell],
        )
        for shell in SHELL_ORDER
    ]
    size_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle=SIZE_LINESTYLES[size],
            linewidth=0.95,
            label=rf"$L={size}$",
        )
        for size in SELECTED_SIZES
    ]
    fig.legend(
        handles=shell_handles + size_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=6,
        handlelength=1.7,
        columnspacing=1.05,
        fontsize=7.2,
    )
    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.10,
        top=0.915,
        wspace=0.28,
        hspace=0.38,
    )
    fig.savefig(PDF_PATH)
    plt.close(fig)
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-r",
            str(FIGURE_DPI),
            "-singlefile",
            str(PDF_PATH),
            str(PNG_PATH.with_suffix("")),
        ],
        check=True,
    )


def main() -> int:
    trajectories, chern_summary = _load_inputs()
    panel_a = _build_panel_a(trajectories, chern_summary)
    panel_b = _build_panel_b(trajectories)
    histogram, scaling, charge_fit = _build_charge_panels(trajectories)
    source_data = _source_data(panel_a, panel_b, histogram, scaling)
    source_data.to_csv(SOURCE_DATA_PATH, index=False, float_format="%.17g")

    _draw_figure(panel_a, panel_b, histogram, scaling, charge_fit)
    for path in (PDF_PATH, PNG_PATH, SOURCE_DATA_PATH):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing generated product: {path}")

    caption = (
        r"Uniform perfect-correction dynamics prepare a Chern insulator without "
        r"domain walls. (a) Trajectory-averaged real-space Chern estimate for "
        r"random pure initial states at $L=12,20,32$ and three OW shell choices. "
        r"Points are the six saved depths $t/L=0.25,0.5,0.75,1,1.5,2$; the "
        r"dashed line is $C_G=1$. (b) The estimator-order-preserving error "
        r"$\langle|C_{G,\xi}-1|\rangle_\xi$ for the same trajectories. "
        r"(c) Final global charge offset from half filling for $L=32$ and "
        r"$n_{\rm shell}=1$. (d) Final mean absolute half-filling deviation for "
        r"the same local controller and its power-law fit. Each configuration "
        r"contains $S=25$ independent Born trajectories, evolved for $2L$ "
        r"random-serial perfect-correction cycles in complex128. Error bars are "
        r"deterministic 95\% whole-trajectory bootstrap intervals (20,000 "
        r"resamples); nonlinear Chern observables are evaluated before averaging."
    )
    summary = {
        "schema_version": 1,
        "figure": FIGURE_STEM,
        "campaign": "P1 frozen production_25sample_v1",
        "protocol": {
            "geometry": "uniform; DW=False",
            "phase": "topological; alpha_1=alpha_2=1",
            "initialization": "random pure",
            "perfect_correction": True,
            "sequence": "random serial",
            "dtype": "complex128",
            "trajectory_count_per_configuration": 25,
            "total_cycles": "2L",
            "saved_normalized_cycles": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        },
        "panels": {
            "a": {
                "metric": "trajectory mean real-space Chern",
                "sizes": list(SELECTED_SIZES),
                "shells": list(SHELL_ORDER),
            },
            "b": {
                "metric": "trajectory mean of abs(real-space Chern - 1)",
                "sizes": list(SELECTED_SIZES),
                "shells": list(SHELL_ORDER),
                "scale": "logarithmic y",
            },
            "c": {
                "metric": "signed integer global charge offset Delta Q=Q-L^2",
                "L": 32,
                "shell": "1",
                "cycle": 64,
                "histogram": histogram.to_dict(orient="records"),
            },
            "d": {
                "metric": "100*mean(abs(Delta Q))/L^2 percent",
                "sizes": list(ALL_SIZES),
                "shell": "1",
                "fit": charge_fit,
            },
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "root_seed": BOOTSTRAP_SEED,
            "unit": "whole trajectory",
            "interval": "equal-tailed 95 percent",
        },
        "inputs": {
            str(path.relative_to(PROJECT_DIR)): _fingerprint(path)
            for path in (TRAJECTORY_TABLE, CHERN_SUMMARY_TABLE, ANALYSIS_MANIFEST)
        },
        "outputs": {
            str(path.relative_to(PROJECT_DIR)): _fingerprint(path)
            for path in (PDF_PATH, PNG_PATH, SOURCE_DATA_PATH)
        },
        "manuscript_caption_tex": caption,
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "pdf": str(PDF_PATH),
                "png": str(PNG_PATH),
                "source_data": str(SOURCE_DATA_PATH),
                "summary": str(SUMMARY_PATH),
                "charge_fit": charge_fit,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
