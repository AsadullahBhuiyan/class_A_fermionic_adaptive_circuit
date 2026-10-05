"""Open-BC transfer-notebook analog for saved Lyapunov spectra."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import warnings

import imageio.v2 as imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import font_manager


REPO_ROOT = Path("/home/abhuiyan/class_A_fermionic_adaptive_circuit")
COLAB_ROOT = REPO_ROOT / "00_WORKSPACE" / "COLAB"
CAMPAIGN_ROOT = (
    COLAB_ROOT
    / "colab_lyapunov"
    / "gpu_data"
    / "lyapunov_spectra"
    / "campaigns"
    / "N20_Ny30-50_nsh1_a1-1_S25_cyclesNy"
)
RUNS = CAMPAIGN_ROOT / "runs"
FIG_DIR = COLAB_ROOT / "colab_lyapunov" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NY_VALUES = [30, 40, 50]
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
QGRID = np.linspace(0.0, 1.0, 101)
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

COLORS = {ny: mpl.colormaps["tab10"].colors[i] for i, ny in enumerate(NY_VALUES)}
LYAP_BINS = np.linspace(-45.0, 1.0, 93)
MOVIE_FRAME_COUNT = max(NY_VALUES)
MOVIE_FRAME_DURATION_S = 0.28
MOVIE_GIF_PATH = FIG_DIR / "lyapunov_spectrum_histogram_vs_cycle_pc_dw.gif"
MOVIE_MP4_PATH = FIG_DIR / "lyapunov_spectrum_histogram_vs_cycle_pc_dw.mp4"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows generated for {path}")
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.sort(arr[np.isfinite(arr)])


def interp_quantiles(values: np.ndarray, grid: np.ndarray = QGRID) -> np.ndarray | None:
    vals = finite_values(values)
    if vals.size == 0:
        return None
    if vals.size == 1:
        return np.full_like(grid, vals[0], dtype=np.float64)
    src = np.linspace(0.0, 1.0, vals.size)
    return np.interp(grid, src, vals)


def nan_quantiles(arr: np.ndarray, qs=(0.25, 0.5, 0.75), axis=0) -> tuple[np.ndarray, ...]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return tuple(np.nanquantile(arr, q, axis=axis) for q in qs)


def load_run(config_id: str) -> dict[str, object]:
    run_dir = RUNS / config_id
    spectra_path = run_dir / "lyapunov_spectra.npz"
    summary_path = run_dir / "run_summary.json"
    if not spectra_path.exists():
        raise FileNotFoundError(f"Missing spectra file: {spectra_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {summary_path}")
    with np.load(spectra_path, allow_pickle=False) as spec:
        spectra = np.asarray(spec["lyapunov_spectra"], dtype=np.float64)
    summary = json.loads(summary_path.read_text())
    validate_run(config_id, spectra, summary)
    samples, cycles, n_vec = spectra.shape
    cycle_axis = np.arange(1, cycles + 1, dtype=np.int64)
    return {
        "config_id": config_id,
        "summary": summary,
        "spectra": spectra,
        "cycles": cycle_axis,
        "tau": cycle_axis / float(summary["Ny"]),
        "gap": np.min(np.abs(spectra), axis=-1),
        "finite_counts": np.isfinite(spectra).sum(axis=-1),
        "samples": samples,
        "n_vec": n_vec,
    }


def validate_run(config_id: str, spectra: np.ndarray, summary: dict) -> None:
    if spectra.ndim != 3:
        raise ValueError(f"{config_id}: expected (samples, cycles, modes), got {spectra.shape}")
    samples, cycles, n_vec = spectra.shape
    if samples != 25:
        raise ValueError(f"{config_id}: expected 25 perfect-correction samples, got {samples}")
    if cycles != int(summary["cycles"]):
        raise ValueError(f"{config_id}: spectra cycles={cycles}, summary cycles={summary['cycles']}")
    if n_vec != int(summary["lyapunov_nvec"]):
        raise ValueError(f"{config_id}: spectra n_vec={n_vec}, summary lyapunov_nvec={summary['lyapunov_nvec']}")
    if not np.all(np.isfinite(spectra)):
        raise ValueError(f"{config_id}: spectra contain non-finite entries")
    if not bool(np.all(np.diff(spectra, axis=-1) >= -1e-12)):
        raise ValueError(f"{config_id}: spectra are not sorted along the mode axis")


def cycle_change_metrics(spectra: np.ndarray) -> dict[str, np.ndarray]:
    samples, ncycles, _nvec = spectra.shape
    max_change = np.full((samples, ncycles - 1), np.nan, dtype=np.float64)
    rms_change = np.full_like(max_change, np.nan)
    quantile_distance = np.full_like(max_change, np.nan)
    comparable_count = np.zeros((samples, ncycles - 1), dtype=np.int64)

    for sample_idx in range(samples):
        for cycle_idx in range(1, ncycles):
            prev = finite_values(spectra[sample_idx, cycle_idx - 1])
            curr = finite_values(spectra[sample_idx, cycle_idx])
            n = min(prev.size, curr.size)
            comparable_count[sample_idx, cycle_idx - 1] = n
            if n > 0:
                delta = curr[:n] - prev[:n]
                max_change[sample_idx, cycle_idx - 1] = float(np.max(np.abs(delta)))
                rms_change[sample_idx, cycle_idx - 1] = float(np.sqrt(np.mean(delta**2)))
            prev_q = interp_quantiles(prev)
            curr_q = interp_quantiles(curr)
            if prev_q is not None and curr_q is not None:
                quantile_distance[sample_idx, cycle_idx - 1] = float(np.sqrt(np.mean((curr_q - prev_q) ** 2)))
    return {
        "max_change": max_change,
        "rms_change": rms_change,
        "quantile_distance": quantile_distance,
        "comparable_count": comparable_count,
    }


def load_all_runs() -> dict[tuple[str, int], dict[str, object]]:
    run_data: dict[tuple[str, int], dict[str, object]] = {}
    for dw_key, dw_info in DW_CONFIGS.items():
        for ny in NY_VALUES:
            config_id = dw_info["config_template"].format(ny=ny)
            data = load_run(config_id)
            data.update(cycle_change_metrics(np.asarray(data["spectra"])))
            run_data[(dw_key, ny)] = data
    return run_data


def plot_final_gap(run_data: dict[tuple[str, int], dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.25), constrained_layout=True)
    colors = {"DW0": "tab:blue", "DW1": "tab:orange"}
    for dw_key, dw_info in DW_CONFIGS.items():
        nys: list[int] = []
        final_rms: list[float] = []
        final_median: list[float] = []
        finite_samples: list[int] = []
        for ny in NY_VALUES:
            data = run_data[(dw_key, ny)]
            gap_final = np.asarray(data["gap"])[:, -1]
            finite = np.isfinite(gap_final)
            nys.append(ny)
            final_rms.append(float(np.sqrt(np.mean(gap_final[finite] ** 2))))
            final_median.append(float(np.median(gap_final[finite])))
            finite_samples.append(int(np.count_nonzero(finite)))
            rows.append(
                {
                    "dw": dw_key,
                    "dw_label": dw_info["label"],
                    "Ny": ny,
                    "final_rms_gap": final_rms[-1],
                    "final_median_gap": final_median[-1],
                    "finite_samples": finite_samples[-1],
                    "samples": int(data["samples"]),
                    "n_vec": int(data["n_vec"]),
                }
            )
        axes[0].plot(nys, final_rms, marker="o", ms=3, color=colors[dw_key], label=dw_info["label"])
        axes[1].plot(nys, final_median, marker="o", ms=3, color=colors[dw_key], label=dw_info["label"])
        axes[2].plot(nys, finite_samples, marker="o", ms=3, color=colors[dw_key], label=dw_info["label"])

    axes[0].set_title("RMS")
    axes[0].set_ylabel(r"final $\Delta_\lambda$")
    axes[0].set_yscale("log")
    axes[1].set_title("median")
    axes[1].set_yscale("log")
    axes[2].set_title("finite samples")
    axes[2].set_ylabel("count / 25")
    axes[2].set_ylim(-0.5, 25.5)
    for ax in axes:
        ax.set_xlabel(r"$N_y$")
        ax.set_xticks(NY_VALUES)
        ax.grid(True, alpha=0.25)
    axes[2].legend(frameon=False, loc="best")
    fig.suptitle("Final-cycle Lyapunov gap versus system length")
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"final_gap_vs_Ny_lyapunov_pc.{ext}", bbox_inches="tight")
    plt.close(fig)
    write_csv(FIG_DIR / "final_gap_vs_Ny_lyapunov_pc_summary.csv", rows)
    return rows


def plot_cycle_metric(
    run_data: dict[tuple[str, int], dict[str, object]],
    metric_key: str,
    ylabel: str,
    title: str,
    basename: str,
    *,
    logy: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.35), sharey=True, constrained_layout=True)
    for ax, (dw_key, dw_info) in zip(axes, DW_CONFIGS.items()):
        for ny in NY_VALUES:
            data = run_data[(dw_key, ny)]
            values = np.asarray(data[metric_key])
            x = np.asarray(data["tau"])[1:] if values.shape[1] == len(data["tau"]) - 1 else np.asarray(data["tau"])
            lo, med, hi = nan_quantiles(values)
            ax.plot(x, med, color=COLORS[ny], label=fr"$N_y={ny}$")
            ax.fill_between(x, lo, hi, color=COLORS[ny], alpha=0.18, linewidth=0)
            rows.append(
                {
                    "dw": dw_key,
                    "dw_label": dw_info["label"],
                    "Ny": ny,
                    f"final_median_{metric_key}": float(np.nanmedian(values[:, -1])),
                    f"final_q25_{metric_key}": float(np.nanquantile(values[:, -1], 0.25)),
                    f"final_q75_{metric_key}": float(np.nanquantile(values[:, -1], 0.75)),
                }
            )
        ax.set_title(dw_info["label"])
        ax.set_xlabel(r"normalized cycle $t/N_y$")
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="best")
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{basename}.{ext}", bbox_inches="tight")
    plt.close(fig)
    write_csv(FIG_DIR / f"{basename}_summary.csv", rows)
    return rows


def make_spectrum_movie(run_data: dict[tuple[str, int], dict[str, object]]) -> None:
    frames: list[np.ndarray] = []
    tau_values = np.linspace(1.0 / MOVIE_FRAME_COUNT, 1.0, MOVIE_FRAME_COUNT)
    for tau in tau_values:
        fig, axes = plt.subplots(
            len(NY_VALUES),
            len(DW_CONFIGS),
            figsize=(6.8, 6.0),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        for row_idx, ny in enumerate(NY_VALUES):
            for col_idx, (dw_key, dw_info) in enumerate(DW_CONFIGS.items()):
                ax = axes[row_idx, col_idx]
                data = run_data[(dw_key, ny)]
                cycle = int(np.clip(np.ceil(float(tau) * ny), 1, ny))
                spectrum = np.asarray(data["spectra"])[:, cycle - 1, :].reshape(-1)
                weights = np.full(spectrum.shape, 1.0 / max(1, spectrum.size), dtype=np.float64)
                ax.hist(spectrum, bins=LYAP_BINS, weights=weights, color="#356d9a", alpha=0.82, edgecolor="none")
                ax.axvline(0.0, color="black", lw=0.7, alpha=0.55)
                ax.set_yscale("log")
                ax.set_ylim(1e-5, 1.0)
                ax.grid(True, alpha=0.18)
                ax.text(0.04, 0.88, fr"$t={cycle}$", transform=ax.transAxes, ha="left", va="top", fontsize=FSS)
                if row_idx == 0:
                    ax.set_title(dw_info["label"])
                if col_idx == 0:
                    ax.set_ylabel(fr"$N_y={ny}$" + "\nfraction")
                if row_idx == len(NY_VALUES) - 1:
                    ax.set_xlabel(r"Lyapunov exponent $\lambda_i(t)$")
        fig.suptitle(fr"Lyapunov spectrum distribution, normalized time $t/N_y={tau:.2f}$", fontsize=10)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        frames.append(rgba[:, :, :3].copy())
        plt.close(fig)

    imageio.mimsave(MOVIE_GIF_PATH, frames, duration=MOVIE_FRAME_DURATION_S, loop=0)
    imageio.mimsave(
        MOVIE_MP4_PATH,
        frames,
        fps=1.0 / MOVIE_FRAME_DURATION_S,
        codec="libx264",
        quality=8,
        macro_block_size=2,
    )


def write_overall_summary(run_data: dict[tuple[str, int], dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for dw_key, dw_info in DW_CONFIGS.items():
        for ny in NY_VALUES:
            data = run_data[(dw_key, ny)]
            gap = np.asarray(data["gap"])
            finite_counts = np.asarray(data["finite_counts"])
            rows.append(
                {
                    "dw": dw_key,
                    "dw_label": dw_info["label"],
                    "Ny": ny,
                    "samples": int(data["samples"]),
                    "cycles": int(len(data["cycles"])),
                    "n_vec": int(data["n_vec"]),
                    "final_rms_gap": float(np.sqrt(np.mean(gap[:, -1] ** 2))),
                    "all_cycle_rms_gap": float(np.sqrt(np.mean(gap**2))),
                    "initial_median_finite_count": float(np.median(finite_counts[:, 0])),
                    "final_median_finite_count": float(np.median(finite_counts[:, -1])),
                    "final_median_max_change": float(np.nanmedian(np.asarray(data["max_change"])[:, -1])),
                    "final_median_rms_change": float(np.nanmedian(np.asarray(data["rms_change"])[:, -1])),
                    "final_median_quantile_distance": float(
                        np.nanmedian(np.asarray(data["quantile_distance"])[:, -1])
                    ),
                }
            )
    write_csv(FIG_DIR / "lyapunov_spectrum_time_convergence_summary.csv", rows)


def main() -> None:
    manifest = json.loads((CAMPAIGN_ROOT / "campaign_manifest.json").read_text())
    if manifest["campaign_id"] != "N20_Ny30-50_nsh1_a1-1_S25_cyclesNy":
        raise ValueError(f"Unexpected campaign: {manifest['campaign_id']}")
    run_data = load_all_runs()
    plot_final_gap(run_data)
    plot_cycle_metric(
        run_data,
        "max_change",
        r"median max $|\Delta\lambda_i|$",
        "Max adjacent-cycle change of Lyapunov spectrum",
        "lyapunov_spectrum_cycle_to_cycle_max_change",
    )
    plot_cycle_metric(
        run_data,
        "rms_change",
        r"median RMS $\Delta\lambda$",
        "RMS adjacent-cycle change of Lyapunov spectrum",
        "lyapunov_spectrum_cycle_to_cycle_rms_change",
    )
    plot_cycle_metric(
        run_data,
        "quantile_distance",
        r"median quantile RMS distance",
        "Empirical Lyapunov-spectrum distance between adjacent cycles",
        "lyapunov_spectrum_cycle_to_cycle_quantile_distance",
    )
    plot_cycle_metric(
        run_data,
        "finite_counts",
        "finite exponent count",
        "Finite Lyapunov spectral support",
        "lyapunov_finite_spectral_support_vs_cycle",
        logy=False,
    )
    make_spectrum_movie(run_data)
    write_overall_summary(run_data)
    print("Saved Lyapunov spectrum time-convergence figures to", FIG_DIR)
    print("Saved", MOVIE_GIF_PATH)
    print("Saved", MOVIE_MP4_PATH)


if __name__ == "__main__":
    main()
