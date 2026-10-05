#!/usr/bin/env python3
"""Build the S100 every-cycle uniform perfect-correction validation figure."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_DIR = Path(__file__).resolve().parent


def _repository_root() -> Path:
    for candidate in (PROJECT_DIR, *PROJECT_DIR.parents):
        if (candidate / "PROJECT_ADMIN" / "REPO_POLICY.md").is_file():
            return candidate
    raise RuntimeError("could not locate repository root")


REPOSITORY_ROOT = _repository_root()
REVISION = "uniform_perfect_correction_40cycle_s100_maxl24_batched_v1"
INPUT_ROOT = (
    REPOSITORY_ROOT
    / "00_WORKSPACE"
    / "LARGE_RESULTS"
    / "classA_final_production_outputs"
    / REVISION
)
RESULT_ROOT = INPUT_ROOT / "results"
OUTPUT_DIR = PROJECT_DIR / "outputs" / REVISION
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_STEM = "fig01_uniform_perfect_correction_bulk_validation_s100"
PDF_PATH = FIGURE_DIR / f"{FIGURE_STEM}.pdf"
PNG_PATH = FIGURE_DIR / f"{FIGURE_STEM}.png"
SOURCE_DATA_PATH = OUTPUT_DIR / f"{FIGURE_STEM}_source_data.csv"
SUMMARY_PATH = OUTPUT_DIR / f"{FIGURE_STEM}_summary.json"

EXPECTED_CONFIG_SHA256 = (
    "5f8ced37fc3bfa5ada24c7f3ff744ae14773cf347d98260551730130c6713b11"
)
SIZES = (12, 16, 20, 24)
SHELLS = ("1", "2", "dense")
SAMPLE_COUNT = 100
SAVED_CYCLES = np.arange(41, dtype=np.int64)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2026090301

FIGURE_WIDTH = 7.05
FIGURE_HEIGHT = 5.25
FIGURE_DPI = 300
BPJ_RED = "#D92725"
BPJ_GREEN = "#2CA02C"
BPJ_BLUE = "#1F77B4"
SHELL_COLORS = {"1": BPJ_RED, "2": BPJ_GREEN, "dense": BPJ_BLUE}
SHELL_MARKERS = {"1": "^", "2": "s", "dense": "o"}
SHELL_LABELS = {
    "1": r"$n_{\rm shell}=1$",
    "2": r"$n_{\rm shell}=2$",
    "dense": "dense",
}
SIZE_LINESTYLES: dict[int, Any] = {
    12: ":",
    16: "--",
    20: "-",
    24: (0, (4, 1.3, 1.0, 1.3)),
}


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
    return np.random.default_rng(int.from_bytes(hashlib.sha256(raw).digest()[:8], "little"))


def _bootstrap_trajectory_means(values: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != SAMPLE_COUNT:
        raise ValueError(f"{label}: expected shape (100, checkpoints), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{label}: nonfinite input")
    weights = _keyed_rng(label).multinomial(
        SAMPLE_COUNT,
        np.full(SAMPLE_COUNT, 1.0 / SAMPLE_COUNT),
        size=BOOTSTRAP_REPLICATES,
    )
    return (weights @ values) / SAMPLE_COUNT


def _mean_and_ci(values: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    replicates = _bootstrap_trajectory_means(values, label)
    low, high = np.quantile(replicates, [0.025, 0.975], axis=0)
    return values.mean(axis=0), low, high, replicates


def _shell_key(nshell: object) -> str:
    return "dense" if nshell is None else str(int(nshell))


def _load_cases() -> tuple[dict[tuple[int, str], dict[str, np.ndarray]], list[dict[str, object]]]:
    if not RESULT_ROOT.is_dir():
        raise FileNotFoundError(RESULT_ROOT)
    receipts = sorted(RESULT_ROOT.rglob("*.complete.json"))
    if len(receipts) != 24:
        raise ValueError(f"expected 24 completion records, found {len(receipts)}")

    batches: dict[tuple[int, str], list[dict[str, np.ndarray]]] = {}
    verification: list[dict[str, object]] = []
    source_identity: str | None = None
    for receipt_path in receipts:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "complete":
            raise ValueError(f"incomplete receipt: {receipt_path}")
        if receipt.get("sampling_revision") != REVISION:
            raise ValueError(f"sampling revision mismatch: {receipt_path}")
        if receipt.get("config_sha256") != EXPECTED_CONFIG_SHA256:
            raise ValueError(f"configuration mismatch: {receipt_path}")
        if receipt.get("cycles_saved") != SAVED_CYCLES.size:
            raise ValueError(f"checkpoint count mismatch: {receipt_path}")

        encoded_sources = json.dumps(receipt.get("source_hashes"), sort_keys=True)
        if source_identity is None:
            source_identity = encoded_sources
        elif encoded_sources != source_identity:
            raise ValueError("batch source hashes are inconsistent")

        result_path = receipt_path.with_name(str(receipt["result_filename"]))
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        fingerprint = _fingerprint(result_path)
        if fingerprint != {
            "bytes": int(receipt["result_bytes"]),
            "sha256": str(receipt["result_sha256"]),
        }:
            raise ValueError(f"result checksum mismatch: {result_path}")

        with np.load(result_path, allow_pickle=False) as archive:
            required = {
                "cycles",
                "normalized_cycles",
                "real_space_chern",
                "global_charge",
                "particle_number",
                "half_filling_offset",
                "global_sample_indices",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"{result_path}: missing {sorted(missing)}")
            cycles = np.array(archive["cycles"], copy=True)
            chern = np.array(archive["real_space_chern"], dtype=np.float64, copy=True)
            charge = np.array(archive["global_charge"], dtype=np.float64, copy=True)
            particle_number = np.array(archive["particle_number"], dtype=np.int64, copy=True)
            offset = np.array(archive["half_filling_offset"], dtype=np.int64, copy=True)
            indices = np.array(archive["global_sample_indices"], dtype=np.int64, copy=True)

        count = int(receipt["samples_saved"])
        size = int(receipt["size"])
        shell = _shell_key(receipt.get("nshell"))
        if not np.array_equal(cycles, SAVED_CYCLES):
            raise ValueError(f"{result_path}: cycles are not exactly 0..40")
        expected_shape = (count, SAVED_CYCLES.size)
        for name, array in (("Chern", chern), ("charge", charge), ("offset", offset)):
            if array.shape != expected_shape or not np.isfinite(array).all():
                raise ValueError(f"{result_path}: invalid {name} array {array.shape}")
        if not np.allclose(charge, particle_number, atol=1e-8, rtol=0):
            raise ValueError(f"{result_path}: global charge does not equal occupied rank")
        if not np.array_equal(offset, particle_number - size * size):
            raise ValueError(f"{result_path}: half-filling offset mismatch")

        batches.setdefault((size, shell), []).append(
            {
                "batch_index": np.asarray(int(receipt["batch_index"])),
                "indices": indices,
                "chern": chern,
                "charge": charge,
                "offset": offset,
            }
        )
        verification.append(
            {
                "completion": str(receipt_path.relative_to(INPUT_ROOT)),
                "result": str(result_path.relative_to(INPUT_ROOT)),
                "samples": count,
                **fingerprint,
            }
        )

    expected_keys = {(size, shell) for size in SIZES for shell in SHELLS}
    if set(batches) != expected_keys:
        raise ValueError(f"case grid mismatch: found {sorted(batches)}")

    cases: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for key, parts in batches.items():
        parts.sort(key=lambda row: int(row["batch_index"]))
        indices = np.concatenate([row["indices"] for row in parts])
        if not np.array_equal(indices, np.arange(SAMPLE_COUNT)):
            raise ValueError(f"{key}: samples are not exactly 0..99")
        cases[key] = {
            "indices": indices,
            "chern": np.concatenate([row["chern"] for row in parts], axis=0),
            "charge": np.concatenate([row["charge"] for row in parts], axis=0),
            "offset": np.concatenate([row["offset"] for row in parts], axis=0),
        }
    return cases, verification


def _summaries(cases: dict[tuple[int, str], dict[str, np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    chern_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    for size in SIZES:
        for shell in SHELLS:
            chern = cases[(size, shell)]["chern"]
            mean, low, high, _ = _mean_and_ci(chern, f"chern:L{size}:shell{shell}")
            error = np.abs(chern - 1.0)
            error_mean, error_low, error_high, _ = _mean_and_ci(
                error, f"absolute-chern-error:L{size}:shell{shell}"
            )
            for index, cycle in enumerate(SAVED_CYCLES):
                chern_rows.append(
                    {
                        "panel": "a",
                        "metric": "trajectory_mean_real_space_chern",
                        "L": size,
                        "shell": shell,
                        "cycle": int(cycle),
                        "trajectories": SAMPLE_COUNT,
                        "value": float(mean[index]),
                        "ci95_low": float(low[index]),
                        "ci95_high": float(high[index]),
                    }
                )
                error_rows.append(
                    {
                        "panel": "b",
                        "metric": "trajectory_mean_absolute_real_space_chern_error",
                        "L": size,
                        "shell": shell,
                        "cycle": int(cycle),
                        "trajectories": SAMPLE_COUNT,
                        "value": float(error_mean[index]),
                        "ci95_low": float(error_low[index]),
                        "ci95_high": float(error_high[index]),
                    }
                )

    final_offsets = cases[(24, "1")]["offset"][:, -1]
    values, counts = np.unique(final_offsets, return_counts=True)
    histogram = pd.DataFrame(
        {
            "panel": "c",
            "metric": "delta_Q_histogram_count",
            "L": 24,
            "shell": "1",
            "cycle": 40,
            "trajectories": SAMPLE_COUNT,
            "delta_Q": values,
            "value": counts,
        }
    )

    scaling_rows: list[dict[str, object]] = []
    bootstrap_means: list[np.ndarray] = []
    for size in SIZES:
        percentages = 100.0 * np.abs(cases[(size, "1")]["offset"][:, -1]) / size**2
        mean, low, high, replicates = _mean_and_ci(
            percentages, f"half-filling-deviation:L{size}:shell1"
        )
        bootstrap_means.append(replicates[:, 0])
        scaling_rows.append(
            {
                "panel": "d",
                "metric": "mean_absolute_half_filling_deviation_percent",
                "L": size,
                "shell": "1",
                "cycle": 40,
                "trajectories": SAMPLE_COUNT,
                "value": float(mean[0]),
                "ci95_low": float(low[0]),
                "ci95_high": float(high[0]),
            }
        )
    scaling = pd.DataFrame(scaling_rows)
    x = np.log(np.asarray(SIZES, dtype=np.float64))
    y = np.log(scaling["value"].to_numpy(dtype=np.float64))
    x_centered = x - x.mean()
    denominator = float(x_centered @ x_centered)
    slope = float(x_centered @ y / denominator)
    intercept = float(y.mean() - slope * x.mean())
    fitted = intercept + slope * x
    r_squared = float(1.0 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2))

    bootstrap_matrix = np.column_stack(bootstrap_means)
    if np.any(bootstrap_matrix <= 0):
        raise ValueError("charge bootstrap contains a zero mean; logarithmic fit undefined")
    bootstrap_log = np.log(bootstrap_matrix)
    bootstrap_slopes = (bootstrap_log @ x_centered) / denominator
    exponent_low, exponent_high = np.quantile(-bootstrap_slopes, [0.025, 0.975])
    fit: dict[str, object] = {
        "model": "A*L**(-p)",
        "sizes": list(SIZES),
        "A": float(np.exp(intercept)),
        "p": float(-slope),
        "p_ci95": [float(exponent_low), float(exponent_high)],
        "r_squared_log_means": r_squared,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    return pd.DataFrame(chern_rows), pd.DataFrame(error_rows), pd.concat(
        [histogram, scaling], ignore_index=True, sort=False
    ), fit


def _configure_style() -> None:
    os.environ["TEXINPUTS"] = str(PROJECT_DIR / "latex_support") + os.pathsep + os.environ.get("TEXINPUTS", "")
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": FIGURE_DPI,
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\renewcommand{\familydefault}{\sfdefault}",
            "axes.linewidth": 0.8,
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


def _panel_label(axis: plt.Axes, letter: str) -> None:
    axis.text(-0.14, 1.035, f"({letter})", transform=axis.transAxes, ha="left", va="bottom", fontsize=9)


def _draw(
    chern: pd.DataFrame,
    error: pd.DataFrame,
    charge: pd.DataFrame,
    fit: dict[str, object],
) -> None:
    _configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    errorbar_cycles = np.arange(0, 41, 5)

    for shell in SHELLS:
        for size in SIZES:
            for axis, table in ((ax_a, chern), (ax_b, error)):
                group = table.loc[(table["shell"] == shell) & (table["L"] == size)].sort_values("cycle")
                x = group["cycle"].to_numpy(dtype=np.float64)
                mean = group["value"].to_numpy(dtype=np.float64)
                low = group["ci95_low"].to_numpy(dtype=np.float64)
                high = group["ci95_high"].to_numpy(dtype=np.float64)
                axis.plot(
                    x,
                    mean,
                    color=SHELL_COLORS[shell],
                    linestyle=SIZE_LINESTYLES[size],
                    linewidth=0.85,
                    alpha=0.9,
                    zorder=2,
                )
                pick = np.isin(x.astype(np.int64), errorbar_cycles)
                axis.errorbar(
                    x[pick],
                    mean[pick],
                    yerr=np.maximum(
                        np.vstack((mean[pick] - low[pick], high[pick] - mean[pick])),
                        0.0,
                    ),
                    color=SHELL_COLORS[shell],
                    marker=SHELL_MARKERS[shell],
                    markerfacecolor="white",
                    markeredgewidth=0.6,
                    markersize=2.5,
                    linestyle="none",
                    elinewidth=0.4,
                    capsize=1.0,
                    alpha=0.78,
                    zorder=3,
                )

    ax_a.axhline(1.0, color="black", linestyle="--", linewidth=0.75, zorder=1)
    ax_a.set_xlabel("cycle")
    ax_a.set_ylabel(r"$\langle C_G\rangle$")
    ax_a.set_xlim(-0.8, 40.8)
    a_low = float(chern["ci95_low"].min())
    a_high = float(chern["ci95_high"].max())
    ax_a.set_ylim(a_low - 0.06 * (a_high - a_low), a_high + 0.06 * (a_high - a_low))

    ax_b.set_yscale("log")
    ax_b.set_xlabel("cycle")
    ax_b.set_ylabel(r"$\langle|C_G-1|\rangle$")
    ax_b.set_xlim(-0.8, 40.8)

    histogram = charge.loc[charge["panel"] == "c"].copy()
    ax_c.bar(
        histogram["delta_Q"],
        histogram["value"],
        width=0.82,
        color=BPJ_RED,
        edgecolor="black",
        linewidth=0.65,
        alpha=0.82,
    )
    ax_c.axvline(0, color="black", linestyle="--", linewidth=0.75)
    ax_c.set_xticks(np.arange(int(histogram["delta_Q"].min()), int(histogram["delta_Q"].max()) + 1))
    ax_c.set_xlabel(r"global charge offset $\Delta Q=Q-L^2$")
    ax_c.set_ylabel("trajectories")
    ax_c.set_title(r"$L=24$, $n_{\rm shell}=1$, $t=40$", pad=3)
    ax_c.set_ylim(0, 1.15 * float(histogram["value"].max()))

    scaling = charge.loc[charge["panel"] == "d"].sort_values("L")
    sizes = scaling["L"].to_numpy(dtype=np.float64)
    means = scaling["value"].to_numpy(dtype=np.float64)
    low = scaling["ci95_low"].to_numpy(dtype=np.float64)
    high = scaling["ci95_high"].to_numpy(dtype=np.float64)
    ax_d.errorbar(
        sizes,
        means,
        yerr=np.maximum(np.vstack((means - low, high - means)), 0.0),
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
    fit_x = np.linspace(min(SIZES), max(SIZES), 300)
    fit_y = float(fit["A"]) * fit_x ** (-float(fit["p"]))
    ax_d.plot(fit_x, fit_y, color="black", linestyle="--", linewidth=0.85, label=r"$A L^{-p}$ fit")
    exponent_ci = fit["p_ci95"]
    if not isinstance(exponent_ci, list):
        raise TypeError("fit confidence interval must be a list")
    ax_d.text(
        0.96,
        0.94,
        rf"$p={float(fit['p']):.2f}$ $[{float(exponent_ci[0]):.2f},{float(exponent_ci[1]):.2f}]$" "\n" rf"$R^2={float(fit['r_squared_log_means']):.2f}$",
        transform=ax_d.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
    )
    ax_d.set_xlabel(r"linear size $L$")
    ax_d.set_ylabel(r"$100\,\langle|\Delta Q|\rangle/L^2$ (\%)")
    ax_d.set_xticks(SIZES)
    ax_d.set_ylim(0, 1.1 * float(high.max()))
    ax_d.legend(loc="upper center", bbox_to_anchor=(0.51, 0.77), fontsize=6.8)

    for axis, letter in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        _panel_label(axis, letter)
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)

    shell_handles = [
        Line2D(
            [0],
            [0],
            color=SHELL_COLORS[shell],
            marker=SHELL_MARKERS[shell],
            markerfacecolor="white",
            linewidth=0.85,
            markersize=3.4,
            label=SHELL_LABELS[shell],
        )
        for shell in SHELLS
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
        for size in SIZES
    ]
    fig.legend(
        handles=shell_handles + size_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=7,
        handlelength=1.55,
        columnspacing=0.85,
        fontsize=6.8,
    )
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.10, top=0.915, wspace=0.28, hspace=0.38)
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
    cases, verification = _load_cases()
    chern, error, charge, fit = _summaries(cases)
    source_data = pd.concat([chern, error, charge], ignore_index=True, sort=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_data.to_csv(SOURCE_DATA_PATH, index=False, float_format="%.17g")
    _draw(chern, error, charge, fit)

    histogram = charge.loc[charge["panel"] == "c", ["delta_Q", "value"]]
    scaling = charge.loc[charge["panel"] == "d", ["L", "value", "ci95_low", "ci95_high"]]
    final_chern = chern.loc[chern["cycle"] == 40, ["L", "shell", "value", "ci95_low", "ci95_high"]]
    caption = (
        r"Uniform perfect-correction dynamics prepare a Chern insulator without domain walls. "
        r"(a) Trajectory-averaged real-space Chern estimate at every cycle for "
        r"$L=12,16,20,24$ and three overcomplete-Wannier shell choices; error bars are "
        r"shown every five cycles and the dashed line is $C_G=1$. (b) The estimator-order-preserving "
        r"error $\langle|C_{G,\xi}-1|\rangle_\xi$ on a logarithmic scale. (c) The final global "
        r"charge offset from half filling for $L=24$ and $n_{\rm shell}=1$. (d) The final mean "
        r"absolute half-filling deviation and a power-law guide. Every configuration contains "
        r"$S=100$ independent Born trajectories with random pure initialization, evolved for 40 "
        r"random-serial perfect-correction cycles in complex128. Nonlinear Chern observables are "
        r"evaluated trajectory by trajectory before averaging. Error bars are deterministic 95\% "
        r"whole-trajectory bootstrap intervals from 20,000 resamples."
    )
    summary = {
        "schema": "uniform_bulk_validation_figure_summary_v1",
        "campaign": REVISION,
        "figure": FIGURE_STEM,
        "protocol": {
            "geometry": "uniform; DW=False",
            "phase": "topological; alpha_1=alpha_2=1",
            "initialization": "random pure at half filling",
            "measurement_order": "random serial",
            "perfect_correction": True,
            "dtype": "complex128",
            "sizes": list(SIZES),
            "shells": list(SHELLS),
            "cycles": list(map(int, SAVED_CYCLES)),
            "trajectories_per_configuration": SAMPLE_COUNT,
        },
        "verification": {
            "result_completion_pairs": len(verification),
            "total_trajectories": len(cases) * SAMPLE_COUNT,
            "configuration_sha256": EXPECTED_CONFIG_SHA256,
            "all_result_sizes_and_sha256_verified": True,
            "all_arrays_finite": True,
            "charge_rank_identity_verified": True,
            "input_pairs": verification,
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "root_seed": BOOTSTRAP_SEED,
            "unit": "whole trajectory",
            "interval": "equal-tailed 95 percent",
        },
        "panels": {
            "a": {"metric": "trajectory mean real-space Chern", "final_cycle": final_chern.to_dict(orient="records")},
            "b": {"metric": "trajectory mean abs(real-space Chern - 1)", "scale": "logarithmic y"},
            "c": {"metric": "signed integer global charge offset", "histogram": histogram.to_dict(orient="records")},
            "d": {"metric": "100*mean(abs(Delta Q))/L^2 percent", "scaling": scaling.to_dict(orient="records"), "fit": fit},
        },
        "inputs": {
            str((INPUT_ROOT / "IMPORT_PROVENANCE.json").relative_to(REPOSITORY_ROOT)): _fingerprint(INPUT_ROOT / "IMPORT_PROVENANCE.json"),
            str((INPUT_ROOT / "SHA256SUMS").relative_to(REPOSITORY_ROOT)): _fingerprint(INPUT_ROOT / "SHA256SUMS"),
        },
        "outputs": {
            str(PDF_PATH.relative_to(PROJECT_DIR)): _fingerprint(PDF_PATH),
            str(PNG_PATH.relative_to(PROJECT_DIR)): _fingerprint(PNG_PATH),
            str(SOURCE_DATA_PATH.relative_to(PROJECT_DIR)): _fingerprint(SOURCE_DATA_PATH),
        },
        "manuscript_caption_tex": caption,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pdf": str(PDF_PATH), "png": str(PNG_PATH), "source_data": str(SOURCE_DATA_PATH), "summary": str(SUMMARY_PATH), "charge_fit": fit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
