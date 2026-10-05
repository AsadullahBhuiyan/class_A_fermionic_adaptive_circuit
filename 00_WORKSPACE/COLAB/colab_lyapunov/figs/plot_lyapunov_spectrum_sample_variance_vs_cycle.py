"""Plot max sample-to-sample variance of saved Lyapunov spectra vs cycle."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
CAMPAIGN = (
    HERE.parent
    / "gpu_data"
    / "lyapunov_spectra"
    / "campaigns"
    / "N20_Ny30-50_nsh1_a1-1_S25_cyclesNy"
)
RUNS = CAMPAIGN / "runs"
FIGDIR = HERE

DPI = 300
W2 = 6.75
FS = 8
FSS = 7

available = {f.name for f in font_manager.fontManager.ttflist}
FONT = "CMU Sans Serif" if "CMU Sans Serif" in available else "DejaVu Sans"
mpl.rcParams.update(
    {
        "font.family": FONT,
        "font.size": FS,
        "axes.titlesize": FS,
        "axes.labelsize": FS,
        "xtick.labelsize": FSS,
        "ytick.labelsize": FSS,
        "legend.fontsize": FSS,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.1,
        "text.usetex": False,
    }
)

NY_VALUES = [30, 40, 50]
COLORS = {ny: mpl.colormaps["tab10"].colors[i] for i, ny in enumerate(NY_VALUES)}
DW_CONFIGS = {
    "DW0": {
        "label": "DW off",
        "config_template": "N20x{ny}_DW0_dwtrunc0_a2-1_nsh1_perfect_correction",
    },
    "DW1": {
        "label": "DW on",
        "config_template": "N20x{ny}_DW1_dwtrunc1_a2-30_nsh1_perfect_correction",
    },
}
STAT_STYLES = {
    "max_var": {"label": r"$\max_i$", "linestyle": "-"},
    "q95_var": {"label": r"$q_{95,i}$", "linestyle": "--"},
    "median_var": {"label": r"${\rm median}_i$", "linestyle": ":"},
}

FIG_BASENAME = "lyapunov_spectrum_sample_variance_vs_cycle"
CSV_PATH = FIGDIR / f"{FIG_BASENAME}_summary.csv"


def load_spectra(config_id: str) -> tuple[np.ndarray, dict]:
    run_dir = RUNS / config_id
    spectra_path = run_dir / "lyapunov_spectra.npz"
    summary_path = run_dir / "run_summary.json"
    if not spectra_path.exists():
        raise FileNotFoundError(f"Missing spectra file: {spectra_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {summary_path}")
    with np.load(spectra_path, allow_pickle=False) as data:
        spectra = np.asarray(data["lyapunov_spectra"], dtype=np.float64)
    summary = json.loads(summary_path.read_text())
    return spectra, summary


def validate_spectra(config_id: str, spectra: np.ndarray, summary: dict) -> None:
    if spectra.ndim != 3:
        raise ValueError(f"{config_id}: expected spectra shape (samples, cycles, modes), got {spectra.shape}")
    samples, cycles, n_vec = spectra.shape
    if samples != 25:
        raise ValueError(f"{config_id}: expected 25 perfect-correction samples, got {samples}")
    if cycles != int(summary["cycles"]):
        raise ValueError(f"{config_id}: spectra cycles={cycles} but summary cycles={summary['cycles']}")
    if n_vec != int(summary["lyapunov_nvec"]):
        raise ValueError(f"{config_id}: spectra n_vec={n_vec} but summary lyapunov_nvec={summary['lyapunov_nvec']}")
    if not np.all(np.isfinite(spectra)):
        raise ValueError(f"{config_id}: spectra contain non-finite entries")
    if not bool(np.all(np.diff(spectra, axis=-1) >= -1e-12)):
        raise ValueError(f"{config_id}: spectra are not sorted along the mode axis")


def variance_stats_by_cycle(spectra: np.ndarray) -> dict[str, np.ndarray]:
    var_by_mode = np.var(spectra, axis=0, ddof=1)
    if not np.all(np.isfinite(var_by_mode)):
        raise ValueError("mode-wise sample variance contains non-finite entries")
    if np.any(var_by_mode < -1e-12):
        raise ValueError("mode-wise sample variance contains negative entries below tolerance")
    var_by_mode = np.maximum(var_by_mode, 0.0)
    return {
        "max_var": np.max(var_by_mode, axis=-1),
        "q95_var": np.quantile(var_by_mode, 0.95, axis=-1),
        "median_var": np.median(var_by_mode, axis=-1),
        "mean_var": np.mean(var_by_mode, axis=-1),
        "min_var": np.min(var_by_mode, axis=-1),
    }


def main() -> None:
    loaded: dict[tuple[str, int], dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    positive_values: list[np.ndarray] = []

    for dw_key, dw_info in DW_CONFIGS.items():
        for ny in NY_VALUES:
            config_id = dw_info["config_template"].format(ny=ny)
            spectra, summary = load_spectra(config_id)
            validate_spectra(config_id, spectra, summary)
            stats = variance_stats_by_cycle(spectra)
            cycles = np.arange(1, spectra.shape[1] + 1, dtype=np.int64)
            tau = cycles / float(ny)
            loaded[(dw_key, ny)] = {
                "config_id": config_id,
                "summary": summary,
                "stats": stats,
                "cycles": cycles,
                "tau": tau,
                "samples": spectra.shape[0],
                "n_vec": spectra.shape[2],
            }
            for values in stats.values():
                positive = np.asarray(values)[np.asarray(values) > 0.0]
                if positive.size:
                    positive_values.append(positive)
            for idx, cycle in enumerate(cycles):
                rows.append(
                    {
                        "config_id": config_id,
                        "dw": dw_key,
                        "dw_label": dw_info["label"],
                        "Ny": ny,
                        "cycle": int(cycle),
                        "normalized_cycle": float(tau[idx]),
                        "samples": spectra.shape[0],
                        "n_vec": spectra.shape[2],
                        "max_var": float(stats["max_var"][idx]),
                        "q95_var": float(stats["q95_var"][idx]),
                        "median_var": float(stats["median_var"][idx]),
                        "mean_var": float(stats["mean_var"][idx]),
                        "min_var": float(stats["min_var"][idx]),
                    }
                )

    if positive_values:
        y_floor = max(1e-12, 0.5 * min(float(np.min(v)) for v in positive_values))
    else:
        y_floor = 1e-12
    y_ceil = 2.0 * max(float(np.max(np.asarray(run["stats"]["max_var"]))) for run in loaded.values())

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.75), sharey=True, constrained_layout=True)
    for ax, (dw_key, dw_info) in zip(axes, DW_CONFIGS.items()):
        for ny in NY_VALUES:
            run = loaded[(dw_key, ny)]
            tau = np.asarray(run["tau"], dtype=np.float64)
            stats = run["stats"]
            color = COLORS[ny]
            for stat_key, style in STAT_STYLES.items():
                values = np.asarray(stats[stat_key], dtype=np.float64)
                plot_values = np.where(values > 0.0, values, np.nan)
                ax.plot(
                    tau,
                    plot_values,
                    color=color,
                    linestyle=style["linestyle"],
                    alpha=0.95,
                )
        ax.set_title(dw_info["label"])
        ax.set_xlabel(r"normalized cycle $t/N_y$")
        ax.set_yscale("log")
        ax.set_ylim(y_floor, y_ceil)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"${\rm Var}_s[\lambda_i(t)]$ over sorted spectrum")
    ny_handles = [
        Line2D([0], [0], color=COLORS[ny], lw=1.4, label=fr"$N_y={ny}$")
        for ny in NY_VALUES
    ]
    stat_handles = [
        Line2D([0], [0], color="0.25", lw=1.4, linestyle=style["linestyle"], label=style["label"])
        for style in STAT_STYLES.values()
    ]
    axes[0].legend(handles=ny_handles, frameon=False, loc="upper right")
    axes[1].legend(handles=stat_handles, frameon=False, loc="upper right")
    fig.suptitle("Sample-to-sample Lyapunov spectrum variance")
    fig.savefig(FIGDIR / f"{FIG_BASENAME}.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{FIG_BASENAME}.png", bbox_inches="tight")
    plt.close(fig)

    with CSV_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {FIGDIR / (FIG_BASENAME + '.pdf')}")
    print(f"Saved {FIGDIR / (FIG_BASENAME + '.png')}")
    print(f"Saved {CSV_PATH}")


if __name__ == "__main__":
    main()
