#!/usr/bin/env python3
"""Rebuild the cycle-resolved effective-central-charge panel for three Ny.

The source arrays are the completed S=100 streaming-observable campaign.  At
each physical cycle, the estimator first averages strip entropy over the 100
independent trajectories and then fits the resulting mean curve over
Ay=8,...,Ny/2 against log[sin(pi Ay/Ny)].  This is the same estimator order and
OLS curve-shape uncertainty used by panel (d) of the legacy evidence atlas.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE_ROOT = (
    ROOT
    / "00_WORKSPACE/COLAB/colab_charge_fluctuations/gpu_data"
    / "streaming_covariance_observables/campaigns"
    / "N20_selected_nsh1_dwtrunc1_init-default_S100_cycles-2Ny/runs"
)
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"
CSV_PATH = DATA_DIR / "ceff_vs_cycle_Nx20_Ny30_40_50_S100.csv"
PNG_PATH = FIGURE_DIR / "ceff_vs_cycle_Nx20_Ny30_40_50_S100.png"
PDF_PATH = FIGURE_DIR / "ceff_vs_cycle_Nx20_Ny30_40_50_S100.pdf"
MANIFEST_PATH = HERE / "figure_manifest.json"

NX = 20
NY_VALUES = (30, 40, 50)
NSHELL = 1
SAMPLES = 100
FIT_AY_MIN = 8
PLOT_CYCLE_MIN = 10
PLOT_CYCLE_STRIDE = 5

STYLES = {
    30: {"color": "#D92725", "marker": "^"},
    40: {"color": "#2CA02C", "marker": "s"},
    50: {"color": "#1F77B4", "marker": "o"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_mean_curve(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    coefficients, covariance = np.polyfit(x, y, 1, cov=True)
    slope = float(coefficients[0])
    intercept = float(coefficients[1])
    slope_error = float(np.sqrt(covariance[0, 0]))
    prediction = slope * x + intercept
    ss_res = float(np.sum((y - prediction) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "slope": slope,
        "slope_error": slope_error,
        "intercept": intercept,
        "r2": float(1.0 - ss_res / ss_tot),
    }


def load_and_fit(ny: int) -> tuple[list[dict[str, float | int | str]], dict[str, str | int]]:
    source = (
        SOURCE_ROOT
        / f"N20x{ny}_nsh1_init-default_perfect_correction"
        / "entropy_y0avg_vs_ay.npz"
    )
    if not source.is_file():
        raise FileNotFoundError(source)

    with np.load(source, allow_pickle=False) as payload:
        required = {
            "cycle_labels",
            "sample_indices",
            "config_json",
            "entropy_y0avg_vs_ay",
            "ay_values",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"{source}: missing keys {sorted(missing)}")
        cycles = np.asarray(payload["cycle_labels"], dtype=np.int64)
        sample_indices = np.asarray(payload["sample_indices"], dtype=np.int64)
        entropies = np.asarray(payload["entropy_y0avg_vs_ay"], dtype=np.float64)
        ay_values = np.asarray(payload["ay_values"], dtype=np.int64)
        config = json.loads(str(payload["config_json"].item()))

    expected_contract = {
        "Nx": NX,
        "Ny": ny,
        "nshell": NSHELL,
        "samples_actual": SAMPLES,
        "cycles": 2 * ny,
        "perfect_correction": True,
        "postselect": False,
        "sequence": "raster_y",
        "dtype": "complex128",
        "alpha_1": 1.0,
        "alpha_2": 30.0,
        "dw_truncation": True,
        "init_mode": "default",
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in expected_contract.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"{source}: scientific-contract mismatch: {mismatches}")
    if entropies.shape != (SAMPLES, 2 * ny, ny // 2 + 1):
        raise ValueError(f"{source}: unexpected entropy shape {entropies.shape}")
    if not np.array_equal(sample_indices, np.arange(SAMPLES)):
        raise ValueError(f"{source}: sample indices are not 0,...,{SAMPLES - 1}")
    if not np.array_equal(cycles, np.arange(1, 2 * ny + 1)):
        raise ValueError(f"{source}: cycle labels are not 1,...,{2 * ny}")
    if not np.array_equal(ay_values, np.arange(ny // 2 + 1)):
        raise ValueError(f"{source}: unexpected Ay grid")
    if not np.all(np.isfinite(entropies)):
        raise FloatingPointError(f"{source}: entropy array contains nonfinite values")

    fit_mask = (ay_values >= FIT_AY_MIN) & (ay_values <= ny // 2)
    x_fit = np.log(np.sin(np.pi * ay_values[fit_mask] / float(ny)))
    rows: list[dict[str, float | int | str]] = []
    for cycle_index, cycle in enumerate(cycles):
        mean_curve = np.mean(entropies[:, cycle_index, :], axis=0)
        fit = fit_mean_curve(x_fit, mean_curve[fit_mask])
        rows.append(
            {
                "Nx": NX,
                "Ny": ny,
                "nshell": NSHELL,
                "samples": SAMPLES,
                "cycle": int(cycle),
                "normalized_cycle": float(cycle / ny),
                "Ay_fit_min": FIT_AY_MIN,
                "Ay_fit_max": ny // 2,
                "n_fit_points": int(np.count_nonzero(fit_mask)),
                "slope": fit["slope"],
                "slope_error": fit["slope_error"],
                "c_eff": 3.0 * fit["slope"],
                "c_eff_error": 3.0 * fit["slope_error"],
                "intercept": fit["intercept"],
                "r2": fit["r2"],
                "uncertainty": "OLS standard error of mean-curve fit slope",
                "source": str(source.relative_to(ROOT)),
            }
        )
    return rows, {
        "path": str(source.relative_to(ROOT)),
        "bytes": source.stat().st_size,
        "sha256": sha256(source),
    }


def write_csv(rows: list[dict[str, float | int | str]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict[str, float | int | str]]) -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "mathtext.fontset": "cm",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.8,
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

    fig, ax = plt.subplots(figsize=(3.375, 2.75))
    for ny in NY_VALUES:
        selected = [
            row
            for row in rows
            if row["Ny"] == ny
            and row["cycle"] >= PLOT_CYCLE_MIN
            and row["cycle"] % PLOT_CYCLE_STRIDE == 0
        ]
        style = STYLES[ny]
        ax.errorbar(
            [row["cycle"] for row in selected],
            [row["c_eff"] for row in selected],
            yerr=[row["c_eff_error"] for row in selected],
            linestyle="none",
            color=style["color"],
            marker=style["marker"],
            markerfacecolor="white",
            markeredgewidth=0.9,
            capsize=1.2,
            label=rf"$N_y={ny}$",
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlim(5, 105)
    ax.set_ylim(0.90, 2.45)
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.set_xlabel("cycle")
    ax.set_ylabel(r"$c_{\rm eff}(C)=3m_1(C)$")
    ax.set_title(r"$N_x=20$, $S=100$")
    ax.legend(loc="upper right", handletextpad=0.25, borderaxespad=0.35)
    ax.text(-0.18, 1.035, "(d)", transform=ax.transAxes, ha="left", va="bottom")
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.18, top=0.88)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_PATH)
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    rows: list[dict[str, float | int | str]] = []
    sources = []
    for ny in NY_VALUES:
        ny_rows, source = load_and_fit(ny)
        rows.extend(ny_rows)
        sources.append(source)

    write_csv(rows)
    make_figure(rows)
    manifest = {
        "figure": "cycle-resolved effective central charge for multiple circumferences",
        "outputs": {
            "pdf": str(PDF_PATH.relative_to(ROOT)),
            "png": str(PNG_PATH.relative_to(ROOT)),
            "csv": str(CSV_PATH.relative_to(ROOT)),
        },
        "scientific_contract": {
            "Nx": NX,
            "Ny": list(NY_VALUES),
            "nshell": NSHELL,
            "samples_per_size": SAMPLES,
            "cycles": "1..2Ny",
            "perfect_correction": True,
            "postselect": False,
            "sequence": "raster_y",
            "dtype": "complex128",
            "alpha_1": 1.0,
            "alpha_2": 30.0,
            "dw_truncation": True,
            "initial_state": "default random pure state",
        },
        "estimator": {
            "order": "mean entropy over trajectories, then OLS log-chord fit",
            "fit_window": "Ay=8..Ny/2 inclusive",
            "c_eff": "3 * fitted slope",
            "uncertainty": "3 * OLS standard error of the mean-curve fit slope; not a trajectory confidence interval",
            "plotted_cycles": "multiples of 5 from cycle 10 through 2Ny",
            "all_cycles_saved_to_csv": True,
        },
        "sources": sources,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[saved] {PDF_PATH}")
    print(f"[saved] {PNG_PATH}")
    print(f"[saved] {CSV_PATH}")
    for ny in NY_VALUES:
        final = next(row for row in rows if row["Ny"] == ny and row["cycle"] == 2 * ny)
        print(
            f"Ny={ny}: final cycle={2 * ny}, "
            f"c_eff={final['c_eff']:.6f} +/- {final['c_eff_error']:.6f}, "
            f"R^2={final['r2']:.8f}"
        )


if __name__ == "__main__":
    main()
