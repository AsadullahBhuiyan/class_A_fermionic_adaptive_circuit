from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
import numpy as np
import pandas as pd

try:
    from purification_data_loader import load_all_purification_runs
except ImportError:  # pragma: no cover - package import path during direct module execution
    from colab_charge_fluctuations.src.purification_data_loader import load_all_purification_runs


ANALYSIS_NAME = "purification_total_charge_histogram_video_cpu"
TOTAL_CHARGE_VARIANCE_ANALYSIS_NAME = "purification_total_charge_variance_histogram_video_cpu"
TARGET_PROTOCOL = "perfect_correction"
POSTSELECT_PROTOCOL = "postselect"
TARGET_NY = (30, 40, 50)
ROLLING_WINDOW = 7

PERCENT_XLIM = (-4.0, 4.0)
PERCENT_BIN_WIDTH = 0.25
PERCENT_BINS = np.arange(PERCENT_XLIM[0], PERCENT_XLIM[1] + PERCENT_BIN_WIDTH, PERCENT_BIN_WIDTH)
COUNT_YLIM = (0.0, 50.0)

FRAME_REPEAT = 3
FPS = 6
TOTAL_CYCLES = 100
VIDEO_FIGSIZE = (10.125, 3.6)
HEATMAP_VIDEO_FIGSIZE = (10.125, 4.0)
PLOT_FIGSIZE = (3.375, 2.4)
PNG_DPI = 300
VIDEO_DPI = 200

VIDEO_FILENAME = "purification_centered_total_charge_percent_histogram_comparison.mp4"
FINAL_FRAME_FILENAME = "purification_centered_total_charge_percent_histogram_final_frame.png"
RAW_VARIANCE_FILENAME = "purification_centered_total_charge_raw_variance_vs_cycle.png"
PERCENT_VARIANCE_FILENAME = "purification_centered_total_charge_percent_variance_vs_cycle.png"
RAW_MEAN_FILENAME = "purification_centered_total_charge_raw_mean_vs_cycle.png"
RAW_MEAN_STD_FILENAME = "purification_centered_total_charge_raw_mean_plusminus_stdev_vs_cycle.png"
PERCENT_MEAN_FILENAME = "purification_centered_total_charge_percent_mean_vs_cycle.png"
PERCENT_MEAN_STD_FILENAME = "purification_centered_total_charge_percent_mean_plusminus_stdev_vs_cycle.png"
ENTROPY_LOGLOG_FILENAME = "purification_total_entropy_sample_mean_vs_cycle_loglog.png"
ENTROPY_LOGLOG_STD_FILENAME = "purification_total_entropy_sample_mean_plusminus_stdev_vs_cycle_loglog.png"
CHERN_LOGLOG_FILENAME = "purification_real_space_chern_sample_mean_vs_cycle_loglog.png"
EXPECTED_CHARGE_VIDEO_FILENAME = "purification_sample_averaged_expected_charge_heatmap_comparison.mp4"
EXPECTED_CHARGE_FINAL_FRAME_FILENAME = "purification_sample_averaged_expected_charge_heatmap_final_frame.png"
STATS_TABLE_FILENAME = "purification_centered_total_charge_sample_stats_vs_cycle.csv"
MANIFEST_FILENAME = "analysis_manifest.json"
RAW_FORMULA = "2 * (sum(local_charge_cell_mean) - Nx * Ny)"
PERCENT_FORMULA = "100 * raw_centered_total_charge / (2 * Nx * Ny)"

TOTAL_CHARGE_VARIANCE_VIDEO_FILENAME = "purification_total_charge_variance_percent_histogram_comparison.mp4"
TOTAL_CHARGE_VARIANCE_FINAL_FRAME_FILENAME = "purification_total_charge_variance_percent_histogram_final_frame.png"
TOTAL_CHARGE_VARIANCE_RAW_VARIANCE_FILENAME = "purification_total_charge_variance_raw_variance_vs_cycle.png"
TOTAL_CHARGE_VARIANCE_PERCENT_VARIANCE_FILENAME = "purification_total_charge_variance_percent_variance_vs_cycle.png"
TOTAL_CHARGE_VARIANCE_RAW_MEAN_FILENAME = "purification_total_charge_variance_raw_mean_vs_cycle.png"
TOTAL_CHARGE_VARIANCE_RAW_MEAN_STD_FILENAME = "purification_total_charge_variance_raw_mean_plusminus_stdev_vs_cycle.png"
TOTAL_CHARGE_VARIANCE_PERCENT_MEAN_FILENAME = "purification_total_charge_variance_percent_mean_vs_cycle.png"
TOTAL_CHARGE_VARIANCE_PERCENT_MEAN_STD_FILENAME = "purification_total_charge_variance_percent_mean_plusminus_stdev_vs_cycle.png"
TOTAL_CHARGE_VARIANCE_HEATMAP_VIDEO_FILENAME = "purification_sample_averaged_spatial_charge_variance_heatmap_comparison.mp4"
TOTAL_CHARGE_VARIANCE_HEATMAP_FINAL_FRAME_FILENAME = "purification_sample_averaged_spatial_charge_variance_heatmap_final_frame.png"
TOTAL_CHARGE_VARIANCE_STATS_TABLE_FILENAME = "purification_total_charge_variance_sample_stats_vs_cycle.csv"
TOTAL_CHARGE_VARIANCE_RAW_FORMULA = "total_charge_variance"
TOTAL_CHARGE_VARIANCE_PERCENT_FORMULA = "100 * total_charge_variance / (Nx * Ny / 2)"


def _ensure_cmu_sans_serif() -> bool:
    target_name = "CMU Sans Serif"
    known_names = {entry.name for entry in font_manager.fontManager.ttflist}
    if target_name in known_names:
        return True

    font_paths: list[str] = []
    for extension in ("ttf", "otf"):
        font_paths.extend(font_manager.findSystemFonts(fontext=extension))

    for font_path in font_paths:
        lower_name = Path(font_path).name.lower()
        if "cmunss" not in lower_name and "cmu" not in lower_name:
            continue
        try:
            font_manager.fontManager.addfont(font_path)
        except RuntimeError:
            continue

    refreshed_names = {entry.name for entry in font_manager.fontManager.ttflist}
    return target_name in refreshed_names


def _preferred_font_family() -> list[str]:
    if _ensure_cmu_sans_serif():
        return ["CMU Sans Serif", "DejaVu Sans"]
    return ["DejaVu Sans"]


def _style_context() -> dict[str, Any]:
    return {
        "font.family": _preferred_font_family(),
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": PNG_DPI,
        "savefig.dpi": PNG_DPI,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _selected_run_items(
    payload: dict[str, Any],
    protocol: str = TARGET_PROTOCOL,
) -> list[tuple[str, dict[str, Any]]]:
    selected: list[tuple[str, dict[str, Any]]] = []
    for config_id, run_payload in payload["runs"].items():
        if str(run_payload["protocol"]) != protocol:
            continue
        if int(run_payload["Ny"]) not in TARGET_NY:
            continue
        selected.append((config_id, run_payload))

    selected.sort(key=lambda item: (int(item[1]["Nx"]), int(item[1]["Ny"])))
    selected_ids = [config_id for config_id, _ in selected]
    expected_ids = [f"N20x{ny}_{protocol}" for ny in TARGET_NY]
    if selected_ids != expected_ids:
        raise ValueError(f"Expected selected config_ids {expected_ids}, got {selected_ids}.")
    return selected


def prepare_centered_total_charge_histogram_records(
    *,
    payload: dict[str, Any] | None = None,
    start: Path | str | None = None,
    protocol: str = TARGET_PROTOCOL,
) -> list[dict[str, Any]]:
    if payload is None:
        payload = load_all_purification_runs(start)

    records: list[dict[str, Any]] = []
    for config_id, run_payload in _selected_run_items(payload, protocol):
        local_charge = run_payload["arrays"]["local_charge_cell_mean"]["local_charge_cell_mean"]
        samples, cycles, nx, ny = local_charge.shape
        q_raw = 2.0 * (local_charge.sum(axis=(2, 3)) - float(nx * ny))
        q_pct = 100.0 * q_raw / (2.0 * float(nx * ny))
        local_charge_mean_avg = np.mean(local_charge, axis=0).astype(np.float64, copy=False)
        raw_total_charge_mean = 2.0 * local_charge_mean_avg.sum(axis=(1, 2))
        records.append(
            {
                "config_id": config_id,
                "protocol": str(run_payload["protocol"]),
                "Nx": int(nx),
                "Ny": int(ny),
                "samples_actual": int(samples),
                "cycles": int(cycles),
                "q_raw": q_raw.astype(np.float64, copy=False),
                "q_pct": q_pct.astype(np.float64, copy=False),
                "local_charge_cell_mean_avg": local_charge_mean_avg,
                "raw_total_charge_mean": raw_total_charge_mean.astype(np.float64, copy=False),
                "summary": run_payload["summary"],
            }
        )
    return records


def prepare_total_entropy_records(
    *,
    payload: dict[str, Any] | None = None,
    start: Path | str | None = None,
    protocol: str = TARGET_PROTOCOL,
) -> list[dict[str, Any]]:
    if payload is None:
        payload = load_all_purification_runs(start)

    records: list[dict[str, Any]] = []
    for config_id, run_payload in _selected_run_items(payload, protocol):
        total_entropy = run_payload["arrays"]["total_entropy"]["total_entropy"]
        samples, cycles = total_entropy.shape
        records.append(
            {
                "config_id": config_id,
                "protocol": str(run_payload["protocol"]),
                "Nx": int(run_payload["Nx"]),
                "Ny": int(run_payload["Ny"]),
                "samples_actual": int(samples),
                "cycles": int(cycles),
                "total_entropy": total_entropy.astype(np.float64, copy=False),
                "summary": run_payload["summary"],
            }
        )
    return records


def prepare_real_space_chern_records(
    *,
    payload: dict[str, Any] | None = None,
    start: Path | str | None = None,
    protocol: str = TARGET_PROTOCOL,
) -> list[dict[str, Any]]:
    if payload is None:
        payload = load_all_purification_runs(start)

    records: list[dict[str, Any]] = []
    for config_id, run_payload in _selected_run_items(payload, protocol):
        metrics_df = run_payload["metrics_df"]
        chern_mean = (
            metrics_df.groupby("cycle_label", sort=True)["real_space_chern"].mean().to_numpy(dtype=np.float64)
        )
        records.append(
            {
                "config_id": config_id,
                "protocol": str(run_payload["protocol"]),
                "Nx": int(run_payload["Nx"]),
                "Ny": int(run_payload["Ny"]),
                "cycles": int(len(chern_mean)),
                "chern_mean": chern_mean,
            }
        )
    return records


def prepare_total_charge_variance_histogram_records(
    *,
    payload: dict[str, Any] | None = None,
    start: Path | str | None = None,
    protocol: str = TARGET_PROTOCOL,
) -> list[dict[str, Any]]:
    if payload is None:
        payload = load_all_purification_runs(start)

    records: list[dict[str, Any]] = []
    for config_id, run_payload in _selected_run_items(payload, protocol):
        variance_raw = run_payload["arrays"]["total_charge_variance"]["total_charge_variance"]
        local_charge_variance = run_payload["arrays"]["local_charge_cell_variance"]["local_charge_cell_variance"]
        samples, cycles = variance_raw.shape
        nx = int(run_payload["Nx"])
        ny = int(run_payload["Ny"])
        maxmix_baseline = float(nx * ny) / 2.0
        variance_pct = 100.0 * variance_raw / maxmix_baseline
        records.append(
            {
                "config_id": config_id,
                "protocol": str(run_payload["protocol"]),
                "Nx": nx,
                "Ny": ny,
                "samples_actual": int(samples),
                "cycles": int(cycles),
                "variance_raw": variance_raw.astype(np.float64, copy=False),
                "variance_pct": variance_pct.astype(np.float64, copy=False),
                "local_charge_cell_variance_avg": np.mean(local_charge_variance, axis=0).astype(np.float64, copy=False),
                "maxmix_baseline": maxmix_baseline,
                "summary": run_payload["summary"],
            }
        )
    return records


def compute_centered_total_charge_sample_stats(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        q_raw = record["q_raw"]
        q_pct = record["q_pct"]
        cycles = int(record["cycles"])
        sample_count = int(record["samples_actual"])
        raw_mean = q_raw.mean(axis=0)
        raw_var = q_raw.var(axis=0, ddof=0)
        raw_std = np.sqrt(raw_var)
        pct_mean = q_pct.mean(axis=0)
        pct_var = q_pct.var(axis=0, ddof=0)
        pct_std = np.sqrt(pct_var)
        record["raw_sample_mean"] = raw_mean
        record["raw_sample_variance"] = raw_var
        record["raw_sample_stdev"] = raw_std
        record["percent_sample_mean"] = pct_mean
        record["percent_sample_variance"] = pct_var
        record["percent_sample_stdev"] = pct_std
        for cycle_idx in range(cycles):
            rows.append(
                {
                    "config_id": record["config_id"],
                    "Nx": int(record["Nx"]),
                    "Ny": int(record["Ny"]),
                    "protocol": str(record["protocol"]),
                    "cycle_label": cycle_idx + 1,
                    "sample_count": sample_count,
                    "centered_total_charge_raw_sample_mean": float(raw_mean[cycle_idx]),
                    "centered_total_charge_raw_sample_variance": float(raw_var[cycle_idx]),
                    "centered_total_charge_raw_sample_stdev": float(raw_std[cycle_idx]),
                    "centered_total_charge_percent_sample_mean": float(pct_mean[cycle_idx]),
                    "centered_total_charge_percent_sample_variance": float(pct_var[cycle_idx]),
                    "centered_total_charge_percent_sample_stdev": float(pct_std[cycle_idx]),
                }
            )
    return pd.DataFrame(rows).sort_values(["Ny", "cycle_label"]).reset_index(drop=True)


def compute_total_charge_variance_sample_stats(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        variance_raw = record["variance_raw"]
        variance_pct = record["variance_pct"]
        cycles = int(record["cycles"])
        sample_count = int(record["samples_actual"])
        raw_mean = variance_raw.mean(axis=0)
        raw_var = variance_raw.var(axis=0, ddof=0)
        raw_std = np.sqrt(raw_var)
        pct_mean = variance_pct.mean(axis=0)
        pct_var = variance_pct.var(axis=0, ddof=0)
        pct_std = np.sqrt(pct_var)
        record["variance_raw_sample_mean"] = raw_mean
        record["variance_raw_sample_variance"] = raw_var
        record["variance_raw_sample_stdev"] = raw_std
        record["variance_percent_sample_mean"] = pct_mean
        record["variance_percent_sample_variance"] = pct_var
        record["variance_percent_sample_stdev"] = pct_std
        for cycle_idx in range(cycles):
            rows.append(
                {
                    "config_id": record["config_id"],
                    "Nx": int(record["Nx"]),
                    "Ny": int(record["Ny"]),
                    "protocol": str(record["protocol"]),
                    "cycle_label": cycle_idx + 1,
                    "sample_count": sample_count,
                    "total_charge_variance_raw_sample_mean": float(raw_mean[cycle_idx]),
                    "total_charge_variance_raw_sample_variance": float(raw_var[cycle_idx]),
                    "total_charge_variance_raw_sample_stdev": float(raw_std[cycle_idx]),
                    "total_charge_variance_percent_sample_mean": float(pct_mean[cycle_idx]),
                    "total_charge_variance_percent_sample_variance": float(pct_var[cycle_idx]),
                    "total_charge_variance_percent_sample_stdev": float(pct_std[cycle_idx]),
                }
            )
    return pd.DataFrame(rows).sort_values(["Ny", "cycle_label"]).reset_index(drop=True)


def _analysis_paths(bundle_root: Path, campaign_id: str) -> dict[str, Path]:
    analysis_root = bundle_root / "analysis_outputs" / ANALYSIS_NAME / campaign_id
    figure_root = analysis_root / "figures"
    table_root = analysis_root / "tables"
    figure_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)
    return {
        "analysis_root": analysis_root,
        "figure_root": figure_root,
        "table_root": table_root,
        "video_path": figure_root / VIDEO_FILENAME,
        "final_frame_path": figure_root / FINAL_FRAME_FILENAME,
        "expected_charge_video_path": figure_root / EXPECTED_CHARGE_VIDEO_FILENAME,
        "expected_charge_final_frame_path": figure_root / EXPECTED_CHARGE_FINAL_FRAME_FILENAME,
        "raw_variance_path": figure_root / RAW_VARIANCE_FILENAME,
        "percent_variance_path": figure_root / PERCENT_VARIANCE_FILENAME,
        "raw_mean_path": figure_root / RAW_MEAN_FILENAME,
        "raw_mean_std_path": figure_root / RAW_MEAN_STD_FILENAME,
        "percent_mean_path": figure_root / PERCENT_MEAN_FILENAME,
        "percent_mean_std_path": figure_root / PERCENT_MEAN_STD_FILENAME,
        "entropy_loglog_path": figure_root / ENTROPY_LOGLOG_FILENAME,
        "entropy_loglog_std_path": figure_root / ENTROPY_LOGLOG_STD_FILENAME,
        "chern_loglog_path": figure_root / CHERN_LOGLOG_FILENAME,
        "stats_table_path": table_root / STATS_TABLE_FILENAME,
        "manifest_path": analysis_root / MANIFEST_FILENAME,
    }


def _total_charge_variance_analysis_paths(bundle_root: Path, campaign_id: str) -> dict[str, Path]:
    analysis_root = bundle_root / "analysis_outputs" / TOTAL_CHARGE_VARIANCE_ANALYSIS_NAME / campaign_id
    figure_root = analysis_root / "figures"
    table_root = analysis_root / "tables"
    figure_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)
    return {
        "analysis_root": analysis_root,
        "figure_root": figure_root,
        "table_root": table_root,
        "video_path": figure_root / TOTAL_CHARGE_VARIANCE_VIDEO_FILENAME,
        "final_frame_path": figure_root / TOTAL_CHARGE_VARIANCE_FINAL_FRAME_FILENAME,
        "spatial_variance_video_path": figure_root / TOTAL_CHARGE_VARIANCE_HEATMAP_VIDEO_FILENAME,
        "spatial_variance_final_frame_path": figure_root / TOTAL_CHARGE_VARIANCE_HEATMAP_FINAL_FRAME_FILENAME,
        "raw_variance_path": figure_root / TOTAL_CHARGE_VARIANCE_RAW_VARIANCE_FILENAME,
        "percent_variance_path": figure_root / TOTAL_CHARGE_VARIANCE_PERCENT_VARIANCE_FILENAME,
        "raw_mean_path": figure_root / TOTAL_CHARGE_VARIANCE_RAW_MEAN_FILENAME,
        "raw_mean_std_path": figure_root / TOTAL_CHARGE_VARIANCE_RAW_MEAN_STD_FILENAME,
        "percent_mean_path": figure_root / TOTAL_CHARGE_VARIANCE_PERCENT_MEAN_FILENAME,
        "percent_mean_std_path": figure_root / TOTAL_CHARGE_VARIANCE_PERCENT_MEAN_STD_FILENAME,
        "stats_table_path": table_root / TOTAL_CHARGE_VARIANCE_STATS_TABLE_FILENAME,
        "manifest_path": analysis_root / MANIFEST_FILENAME,
    }


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for MP4 export but was not found on PATH.")


def _frame_cycle(frame_idx: int) -> int:
    return frame_idx // FRAME_REPEAT + 1


def _held_cycle_index(frame_idx: int, cycles: int) -> int:
    return min(_frame_cycle(frame_idx) - 1, cycles - 1)


def _even_canvas_figsize(figsize: tuple[float, float], dpi: int) -> tuple[float, float]:
    width_px = max(2, int(round(figsize[0] * dpi)))
    height_px = max(2, int(round(figsize[1] * dpi)))
    if width_px % 2:
        width_px += 1
    if height_px % 2:
        height_px += 1
    return (width_px / dpi, height_px / dpi)


def _pooled_histogram_spec(
    values: list[np.ndarray],
    *,
    bins: int = 32,
    min_width: float = 1.0,
    pad_fraction: float = 0.05,
) -> tuple[np.ndarray, tuple[float, float]]:
    pooled = np.concatenate([np.asarray(value, dtype=np.float64).ravel() for value in values])
    finite = pooled[np.isfinite(pooled)]
    if finite.size == 0:
        raise ValueError("No finite values available for histogram specification.")
    left = float(finite.min())
    right = float(finite.max())
    span = right - left
    if span < min_width:
        center = 0.5 * (left + right)
        half_width = 0.5 * min_width
        xlim = (center - half_width, center + half_width)
    else:
        pad = pad_fraction * span
        xlim = (left - pad, right + pad)
    edges = np.linspace(xlim[0], xlim[1], bins + 1, dtype=np.float64)
    return edges, xlim


def _rolling_mean(values: np.ndarray, *, window: int = ROLLING_WINDOW) -> np.ndarray:
    return (
        pd.Series(np.asarray(values, dtype=np.float64))
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64)
    )


def _loglog_band(mean: np.ndarray, stdev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.maximum(mean - stdev, np.finfo(np.float64).tiny)
    upper = np.maximum(mean + stdev, np.finfo(np.float64).tiny)
    return lower, upper


def _plot_variance_with_overlay(
    ax: plt.Axes,
    cycles: np.ndarray,
    values: np.ndarray,
    *,
    label: str,
    color: Any,
) -> None:
    smooth = _rolling_mean(values)
    ax.plot(cycles, values, linewidth=1.0, alpha=0.35, color=color)
    ax.plot(cycles, smooth, linewidth=1.6, color=color, label=label)


def _make_three_panel_heatmap_video(
    records: list[dict[str, Any]],
    *,
    data_key: str,
    output_root: Path | str,
    video_filename: str,
    final_frame_filename: str,
    suptitle: str,
    colorbar_label: str,
    cmap: str,
    norm: colors.Normalize,
    title_fn,
) -> dict[str, Any]:
    _require_ffmpeg()
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    video_path = output_root / video_filename
    final_frame_path = output_root / final_frame_filename
    video_figsize = _even_canvas_figsize(HEATMAP_VIDEO_FIGSIZE, VIDEO_DPI)

    with plt.rc_context(_style_context()):
        fig, axes = plt.subplots(1, len(records), figsize=video_figsize, constrained_layout=True)
        axes = np.atleast_1d(axes)
        images = []
        for ax, record in zip(axes, records):
            data0 = np.asarray(record[data_key][0], dtype=np.float64)
            image = ax.imshow(
                data0,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
            )
            ax.set_xlabel("y")
            ax.set_ylabel("x")
            images.append(image)
        fig.suptitle(suptitle)
        colorbar = fig.colorbar(images[-1], ax=list(axes), pad=0.02, shrink=0.9)
        colorbar.set_label(colorbar_label)

        def _update(frame_idx: int):
            artists = []
            for ax, record, image in zip(axes, records, images):
                cycle_idx = _held_cycle_index(frame_idx, int(record["cycles"]))
                image.set_data(np.asarray(record[data_key][cycle_idx], dtype=np.float64))
                artists.append(image)
                ax.set_title(title_fn(record, cycle_idx), pad=8)
                artists.append(ax.title)
            return artists

        _update(0)
        writer = FFMpegWriter(
            fps=FPS,
            codec="libx264",
            bitrate=2200,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        animation = FuncAnimation(fig, _update, frames=TOTAL_CYCLES * FRAME_REPEAT, interval=1000 / FPS, blit=False)
        animation.save(video_path, writer=writer, dpi=VIDEO_DPI)
        _update(TOTAL_CYCLES * FRAME_REPEAT - 1)
        fig.savefig(final_frame_path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig)

    return {
        "video_path": video_path,
        "final_frame_path": final_frame_path,
        "fps": FPS,
        "frame_repeat": FRAME_REPEAT,
        "total_cycles": TOTAL_CYCLES,
        "total_frames": TOTAL_CYCLES * FRAME_REPEAT,
        "video_figsize_inches": list(video_figsize),
        "codec": "libx264",
        "pix_fmt": "yuv420p",
        "movflags": "+faststart",
    }


def make_centered_total_charge_histogram_video(
    records: list[dict[str, Any]],
    *,
    output_root: Path | str,
) -> dict[str, Any]:
    _require_ffmpeg()
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    video_path = output_root / VIDEO_FILENAME
    final_frame_path = output_root / FINAL_FRAME_FILENAME

    bin_edges = PERCENT_BINS
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = float(bin_edges[1] - bin_edges[0])
    histograms = [
        np.stack([np.histogram(record["q_pct"][:, idx], bins=bin_edges)[0] for idx in range(record["cycles"])])
        for record in records
    ]
    video_figsize = _even_canvas_figsize(VIDEO_FIGSIZE, VIDEO_DPI)

    with plt.rc_context(_style_context()):
        fig, axes = plt.subplots(1, 3, figsize=video_figsize, sharex=True, sharey=True, constrained_layout=True)
        bar_sets = []
        mean_lines = []
        for ax, record, histogram in zip(axes, records, histograms):
            bars = ax.bar(
                bin_centers,
                histogram[0],
                width=0.92 * bin_width,
                color="#2b6cb0",
                edgecolor="#1a365d",
                linewidth=0.35,
                align="center",
            )
            mean_line = ax.axvline(
                float(record["percent_sample_mean"][0]),
                color="#c53030",
                linestyle="--",
                linewidth=1.2,
            )
            ax.set_xlim(*PERCENT_XLIM)
            ax.set_ylim(*COUNT_YLIM)
            ax.set_xlabel(r"$100\,(Q-N_xN_y)/(N_xN_y)$ (%)")
            ax.set_ylabel("sample count")
            ax.grid(axis="y", alpha=0.25, linewidth=0.4)
            ax.set_box_aspect(0.9)
            bar_sets.append(bars)
            mean_lines.append(mean_line)
        fig.suptitle("Sample-to-sample histogram of size-normalized centered total charge")

        def _panel_title(record: dict[str, Any], cycle_idx: int) -> str:
            return (
                f"N{record['Nx']}x{record['Ny']} cycle {cycle_idx + 1}\n"
                f"Mean_raw={record['raw_sample_mean'][cycle_idx]:.4f}; "
                f"Mean_%={record['percent_sample_mean'][cycle_idx]:.4f}\n"
                f"Var_raw={record['raw_sample_variance'][cycle_idx]:.4f}; "
                f"Var_%={record['percent_sample_variance'][cycle_idx]:.5f}"
            )

        def _update(frame_idx: int):
            artists = []
            for ax, record, histogram, bars, mean_line in zip(axes, records, histograms, bar_sets, mean_lines):
                cycle_idx = _held_cycle_index(frame_idx, int(record["cycles"]))
                counts = histogram[cycle_idx]
                for bar, height in zip(bars, counts):
                    bar.set_height(float(height))
                    artists.append(bar)
                mean_value = float(record["percent_sample_mean"][cycle_idx])
                mean_line.set_xdata([mean_value, mean_value])
                artists.append(mean_line)
                ax.set_title(_panel_title(record, cycle_idx), pad=8)
                artists.append(ax.title)
            return artists

        _update(0)
        writer = FFMpegWriter(
            fps=FPS,
            codec="libx264",
            bitrate=2200,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        animation = FuncAnimation(fig, _update, frames=TOTAL_CYCLES * FRAME_REPEAT, interval=1000 / FPS, blit=False)
        animation.save(video_path, writer=writer, dpi=VIDEO_DPI)
        _update(TOTAL_CYCLES * FRAME_REPEAT - 1)
        fig.savefig(final_frame_path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig)

    return {
        "video_path": video_path,
        "final_frame_path": final_frame_path,
        "fps": FPS,
        "frame_repeat": FRAME_REPEAT,
        "total_cycles": TOTAL_CYCLES,
        "total_frames": TOTAL_CYCLES * FRAME_REPEAT,
        "video_figsize_inches": list(video_figsize),
        "bin_edges": bin_edges.tolist(),
        "xlim": list(PERCENT_XLIM),
        "ylim": list(COUNT_YLIM),
        "codec": "libx264",
        "pix_fmt": "yuv420p",
        "movflags": "+faststart",
    }


def make_centered_total_charge_variance_figures(
    records: list[dict[str, Any]],
    *,
    output_root: Path | str,
) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_path = output_root / RAW_VARIANCE_FILENAME
    percent_path = output_root / PERCENT_VARIANCE_FILENAME
    raw_mean_path = output_root / RAW_MEAN_FILENAME
    raw_mean_std_path = output_root / RAW_MEAN_STD_FILENAME
    percent_mean_path = output_root / PERCENT_MEAN_FILENAME
    percent_mean_std_path = output_root / PERCENT_MEAN_STD_FILENAME

    with plt.rc_context(_style_context()):
        fig_raw_mean, ax_raw_mean = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
        for record in records:
            cycles = np.arange(1, int(record["cycles"]) + 1, dtype=np.float64)
            ax_raw_mean.plot(cycles, record["raw_sample_mean"], linewidth=1.2, label=f"N{record['Nx']}x{record['Ny']}")
        ax_raw_mean.set_xlabel("cycle")
        ax_raw_mean.set_ylabel(r"$\mathbb{E}_{\mathrm{samples}}[2(Q-N_xN_y)]$")
        ax_raw_mean.legend(loc="best")
        ax_raw_mean.grid(alpha=0.25, linewidth=0.4)
        fig_raw_mean.savefig(raw_mean_path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig_raw_mean)

        fig_raw_mean_std, ax_raw_mean_std = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
        for record in records:
            cycles = np.arange(1, int(record["cycles"]) + 1, dtype=np.float64)
            mean = record["raw_sample_mean"]
            std = record["raw_sample_stdev"]
            ax_raw_mean_std.plot(cycles, mean, linewidth=1.2, label=f"N{record['Nx']}x{record['Ny']}")
            ax_raw_mean_std.fill_between(cycles, mean - std, mean + std, alpha=0.18)
        ax_raw_mean_std.set_xlabel("cycle")
        ax_raw_mean_std.set_ylabel(r"$\mathbb{E}_{\mathrm{samples}}[2(Q-N_xN_y)] \pm \sigma_{\mathrm{samples}}$")
        ax_raw_mean_std.legend(loc="best")
        ax_raw_mean_std.grid(alpha=0.25, linewidth=0.4)
        fig_raw_mean_std.savefig(raw_mean_std_path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig_raw_mean_std)

        fig_pct_mean, ax_pct_mean = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
        for record in records:
            cycles = np.arange(1, int(record["cycles"]) + 1, dtype=np.float64)
            ax_pct_mean.plot(cycles, record["percent_sample_mean"], linewidth=1.2, label=f"N{record['Nx']}x{record['Ny']}")
        ax_pct_mean.set_xlabel("cycle")
        ax_pct_mean.set_ylabel(r"$\mathbb{E}_{\mathrm{samples}}[100(Q-N_xN_y)/(N_xN_y)]$")
        ax_pct_mean.legend(loc="best")
        ax_pct_mean.grid(alpha=0.25, linewidth=0.4)
        fig_pct_mean.savefig(percent_mean_path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig_pct_mean)

        fig_pct_mean_std, ax_pct_mean_std = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
        for record in records:
            cycles = np.arange(1, int(record["cycles"]) + 1, dtype=np.float64)
            mean = record["percent_sample_mean"]
            std = record["percent_sample_stdev"]
            ax_pct_mean_std.plot(cycles, mean, linewidth=1.2, label=f"N{record['Nx']}x{record['Ny']}")
            ax_pct_mean_std.fill_between(cycles, mean - std, mean + std, alpha=0.18)
        ax_pct_mean_std.set_xlabel("cycle")
        ax_pct_mean_std.set_ylabel(r"$\mathbb{E}_{\mathrm{samples}}[100(Q-N_xN_y)/(N_xN_y)] \pm \sigma_{\mathrm{samples}}$")
        ax_pct_mean_std.legend(loc="best")
        ax_pct_mean_std.grid(alpha=0.25, linewidth=0.4)
        fig_pct_mean_std.savefig(percent_mean_std_path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig_pct_mean_std)

        fig_raw, ax_raw = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for idx, record in enumerate(records):
            cycles = np.arange(1, int(record["cycles"]) + 1, dtype=np.float64)
            _plot_variance_with_overlay(
                ax_raw,
                cycles,
                record["raw_sample_variance"],
                label=f"N{record['Nx']}x{record['Ny']}",
                color=color_cycle[idx % len(color_cycle)],
            )
        ax_raw.set_xlabel("cycle")
        ax_raw.set_ylabel(r"$\mathrm{Var}_{\mathrm{samples}}[2(Q-N_xN_y)]$")
        ax_raw.legend(loc="best")
        ax_raw.grid(alpha=0.25, linewidth=0.4)
        fig_raw.savefig(raw_path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig_raw)

        fig_pct, ax_pct = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
        for idx, record in enumerate(records):
            cycles = np.arange(1, int(record["cycles"]) + 1, dtype=np.float64)
            _plot_variance_with_overlay(
                ax_pct,
                cycles,
                record["percent_sample_variance"],
                label=f"N{record['Nx']}x{record['Ny']}",
                color=color_cycle[idx % len(color_cycle)],
            )
        ax_pct.set_xlabel("cycle")
        ax_pct.set_ylabel(r"$\mathrm{Var}_{\mathrm{samples}}[100(Q-N_xN_y)/(N_xN_y)]$")
        ax_pct.legend(loc="best")
        ax_pct.grid(alpha=0.25, linewidth=0.4)
        fig_pct.savefig(percent_path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig_pct)

    return {
        "raw_mean_path": raw_mean_path,
        "raw_mean_std_path": raw_mean_std_path,
        "percent_mean_path": percent_mean_path,
        "percent_mean_std_path": percent_mean_std_path,
        "raw_variance_path": raw_path,
        "percent_variance_path": percent_path,
    }


def make_total_entropy_loglog_figure(
    records: list[dict[str, Any]],
    *,
    output_root: Path | str,
) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    entropy_loglog_path = output_root / ENTROPY_LOGLOG_FILENAME
    entropy_loglog_std_path = output_root / ENTROPY_LOGLOG_STD_FILENAME

    with plt.rc_context(_style_context()):
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
        for record in records:
            cycles = np.arange(1, int(record["cycles"]) + 1, dtype=np.float64)
            entropy_mean = np.mean(record["total_entropy"], axis=0)
            ax.plot(cycles, entropy_mean, linewidth=1.2, label=f"N{record['Nx']}x{record['Ny']}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("cycle")
        ax.set_ylabel(r"$\mathbb{E}_{\mathrm{samples}}[S_{\mathrm{tot}}]$")
        ax.legend(loc="best")
        ax.grid(alpha=0.25, linewidth=0.4, which="both")
        fig.savefig(entropy_loglog_path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig)

        fig_std, ax_std = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
        for record in records:
            cycles = np.arange(1, int(record["cycles"]) + 1, dtype=np.float64)
            entropy_mean = np.mean(record["total_entropy"], axis=0)
            entropy_std = np.std(record["total_entropy"], axis=0, ddof=0)
            lower, upper = _loglog_band(entropy_mean, entropy_std)
            ax_std.plot(cycles, entropy_mean, linewidth=1.2, label=f"N{record['Nx']}x{record['Ny']}")
            ax_std.fill_between(cycles, lower, upper, alpha=0.18)
        ax_std.set_xscale("log")
        ax_std.set_yscale("log")
        ax_std.set_xlabel("cycle")
        ax_std.set_ylabel(r"$\mathbb{E}_{\mathrm{samples}}[S_{\mathrm{tot}}] \pm \sigma_{\mathrm{samples}}$")
        ax_std.legend(loc="best")
        ax_std.grid(alpha=0.25, linewidth=0.4, which="both")
        fig_std.savefig(entropy_loglog_std_path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig_std)

    return {
        "entropy_loglog_path": entropy_loglog_path,
        "entropy_loglog_std_path": entropy_loglog_std_path,
    }


def make_real_space_chern_loglog_figure(
    records: list[dict[str, Any]],
    *,
    output_root: Path | str,
) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    chern_loglog_path = output_root / CHERN_LOGLOG_FILENAME

    with plt.rc_context(_style_context()):
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
        for record in records:
            cycles = np.arange(1, int(record["cycles"]) + 1, dtype=np.float64)
            ax.plot(cycles, record["chern_mean"], linewidth=1.2, label=f"N{record['Nx']}x{record['Ny']}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("cycle")
        ax.set_ylabel(r"$\mathbb{E}_{\mathrm{samples}}[\mathcal{C}_{\mathrm{RS}}]$")
        ax.legend(loc="best")
        ax.grid(alpha=0.25, linewidth=0.4, which="both")
        fig.savefig(chern_loglog_path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig)

    return {"chern_loglog_path": chern_loglog_path}


def make_sample_averaged_expected_charge_video(
    records: list[dict[str, Any]],
    *,
    output_root: Path | str,
) -> dict[str, Any]:
    max_dev = max(
        float(np.max(np.abs(np.asarray(record["local_charge_cell_mean_avg"], dtype=np.float64) - 1.0))) for record in records
    )
    norm = colors.TwoSlopeNorm(vmin=1.0 - max_dev, vcenter=1.0, vmax=1.0 + max_dev)

    def _title(record: dict[str, Any], cycle_idx: int) -> str:
        return (
            f"N{record['Nx']}x{record['Ny']} cycle {cycle_idx + 1}\n"
            f"Qbar={record['raw_total_charge_mean'][cycle_idx]:.2f}; "
            f"Var_samp[2(Q-NxNy)]={record['raw_sample_variance'][cycle_idx]:.4f}"
        )

    return _make_three_panel_heatmap_video(
        records,
        data_key="local_charge_cell_mean_avg",
        output_root=output_root,
        video_filename=EXPECTED_CHARGE_VIDEO_FILENAME,
        final_frame_filename=EXPECTED_CHARGE_FINAL_FRAME_FILENAME,
        suptitle="Sample-averaged spatial expected charge",
        colorbar_label="sample-averaged expected charge",
        cmap="RdBu_r",
        norm=norm,
        title_fn=_title,
    )


def make_total_charge_variance_histogram_video(
    records: list[dict[str, Any]],
    *,
    output_root: Path | str,
) -> dict[str, Any]:
    _require_ffmpeg()
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    video_path = output_root / TOTAL_CHARGE_VARIANCE_VIDEO_FILENAME
    final_frame_path = output_root / TOTAL_CHARGE_VARIANCE_FINAL_FRAME_FILENAME

    bin_edges, xlim = _pooled_histogram_spec([record["variance_pct"] for record in records], bins=32, min_width=4.0)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = float(bin_edges[1] - bin_edges[0])
    histograms = [
        np.stack([np.histogram(record["variance_pct"][:, idx], bins=bin_edges)[0] for idx in range(record["cycles"])])
        for record in records
    ]
    max_count = max(float(np.max(histogram)) for histogram in histograms)
    ylim = (0.0, max(10.0, np.ceil(1.05 * max_count / 5.0) * 5.0))
    video_figsize = _even_canvas_figsize(VIDEO_FIGSIZE, VIDEO_DPI)

    with plt.rc_context(_style_context()):
        fig, axes = plt.subplots(1, 3, figsize=video_figsize, sharex=True, sharey=True, constrained_layout=True)
        bar_sets = []
        mean_lines = []
        for ax, record, histogram in zip(axes, records, histograms):
            bars = ax.bar(
                bin_centers,
                histogram[0],
                width=0.92 * bin_width,
                color="#2b6cb0",
                edgecolor="#1a365d",
                linewidth=0.35,
                align="center",
            )
            mean_line = ax.axvline(
                float(record["variance_percent_sample_mean"][0]),
                color="#c53030",
                linestyle="--",
                linewidth=1.2,
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel(r"$100\,\mathrm{Var}(Q)/(N_xN_y/2)$ (%)")
            ax.set_ylabel("sample count")
            ax.grid(axis="y", alpha=0.25, linewidth=0.4)
            ax.set_box_aspect(0.9)
            bar_sets.append(bars)
            mean_lines.append(mean_line)
        fig.suptitle("Sample-to-sample histogram of normalized total-charge variance")

        def _panel_title(record: dict[str, Any], cycle_idx: int) -> str:
            return (
                f"N{record['Nx']}x{record['Ny']} cycle {cycle_idx + 1}\n"
                f"Mean_raw={record['variance_raw_sample_mean'][cycle_idx]:.4f}; "
                f"Mean_%={record['variance_percent_sample_mean'][cycle_idx]:.4f}\n"
                f"Var_raw={record['variance_raw_sample_variance'][cycle_idx]:.4f}; "
                f"Var_%={record['variance_percent_sample_variance'][cycle_idx]:.5f}"
            )

        def _update(frame_idx: int):
            artists = []
            for ax, record, histogram, bars, mean_line in zip(axes, records, histograms, bar_sets, mean_lines):
                cycle_idx = _held_cycle_index(frame_idx, int(record["cycles"]))
                counts = histogram[cycle_idx]
                for bar, height in zip(bars, counts):
                    bar.set_height(float(height))
                    artists.append(bar)
                mean_value = float(record["variance_percent_sample_mean"][cycle_idx])
                mean_line.set_xdata([mean_value, mean_value])
                artists.append(mean_line)
                ax.set_title(_panel_title(record, cycle_idx), pad=8)
                artists.append(ax.title)
            return artists

        _update(0)
        writer = FFMpegWriter(
            fps=FPS,
            codec="libx264",
            bitrate=2200,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        animation = FuncAnimation(fig, _update, frames=TOTAL_CYCLES * FRAME_REPEAT, interval=1000 / FPS, blit=False)
        animation.save(video_path, writer=writer, dpi=VIDEO_DPI)
        _update(TOTAL_CYCLES * FRAME_REPEAT - 1)
        fig.savefig(final_frame_path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig)

    return {
        "video_path": video_path,
        "final_frame_path": final_frame_path,
        "fps": FPS,
        "frame_repeat": FRAME_REPEAT,
        "total_cycles": TOTAL_CYCLES,
        "total_frames": TOTAL_CYCLES * FRAME_REPEAT,
        "video_figsize_inches": list(video_figsize),
        "bin_edges": bin_edges.tolist(),
        "xlim": list(xlim),
        "ylim": list(ylim),
        "codec": "libx264",
        "pix_fmt": "yuv420p",
        "movflags": "+faststart",
    }


def make_total_charge_variance_summary_figures(
    records: list[dict[str, Any]],
    *,
    output_root: Path | str,
) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_path = output_root / TOTAL_CHARGE_VARIANCE_RAW_VARIANCE_FILENAME
    percent_path = output_root / TOTAL_CHARGE_VARIANCE_PERCENT_VARIANCE_FILENAME
    raw_mean_path = output_root / TOTAL_CHARGE_VARIANCE_RAW_MEAN_FILENAME
    raw_mean_std_path = output_root / TOTAL_CHARGE_VARIANCE_RAW_MEAN_STD_FILENAME
    percent_mean_path = output_root / TOTAL_CHARGE_VARIANCE_PERCENT_MEAN_FILENAME
    percent_mean_std_path = output_root / TOTAL_CHARGE_VARIANCE_PERCENT_MEAN_STD_FILENAME

    with plt.rc_context(_style_context()):
        fig_raw_mean, ax_raw_mean = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
        for record in records:
            cycles = np.arange(1, int(record["cycles"]) + 1, dtype=np.float64)
            ax_raw_mean.plot(cycles, record["variance_raw_sample_mean"], linewidth=1.2, label=f"N{record['Nx']}x{record['Ny']}")
        ax_raw_mean.set_xscale("log")
        ax_raw_mean.set_yscale("log")
        ax_raw_mean.set_xlabel("cycle")
        ax_raw_mean.set_ylabel(r"$\mathbb{E}_{\mathrm{samples}}[\mathrm{Var}(Q)]$")
        ax_raw_mean.legend(loc="best")
        ax_raw_mean.grid(alpha=0.25, linewidth=0.4, which="both")
        fig_raw_mean.savefig(raw_mean_path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig_raw_mean)

        fig_raw_mean_std, ax_raw_mean_std = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
        for record in records:
            cycles = np.arange(1, int(record["cycles"]) + 1, dtype=np.float64)
            mean = record["variance_raw_sample_mean"]
            std = record["variance_raw_sample_stdev"]
            lower, upper = _loglog_band(mean, std)
            ax_raw_mean_std.plot(cycles, mean, linewidth=1.2, label=f"N{record['Nx']}x{record['Ny']}")
            ax_raw_mean_std.fill_between(cycles, lower, upper, alpha=0.18)
        ax_raw_mean_std.set_xscale("log")
        ax_raw_mean_std.set_yscale("log")
        ax_raw_mean_std.set_xlabel("cycle")
        ax_raw_mean_std.set_ylabel(r"$\mathbb{E}_{\mathrm{samples}}[\mathrm{Var}(Q)] \pm \sigma_{\mathrm{samples}}$")
        ax_raw_mean_std.legend(loc="best")
        ax_raw_mean_std.grid(alpha=0.25, linewidth=0.4, which="both")
        fig_raw_mean_std.savefig(raw_mean_std_path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig_raw_mean_std)

        fig_pct_mean, ax_pct_mean = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
        for record in records:
            cycles = np.arange(1, int(record["cycles"]) + 1, dtype=np.float64)
            ax_pct_mean.plot(cycles, record["variance_percent_sample_mean"], linewidth=1.2, label=f"N{record['Nx']}x{record['Ny']}")
        ax_pct_mean.set_xscale("log")
        ax_pct_mean.set_yscale("log")
        ax_pct_mean.set_xlabel("cycle")
        ax_pct_mean.set_ylabel(r"$\mathbb{E}_{\mathrm{samples}}[100\,\mathrm{Var}(Q)/(N_xN_y/2)]$")
        ax_pct_mean.legend(loc="best")
        ax_pct_mean.grid(alpha=0.25, linewidth=0.4, which="both")
        fig_pct_mean.savefig(percent_mean_path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig_pct_mean)

        fig_pct_mean_std, ax_pct_mean_std = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
        for record in records:
            cycles = np.arange(1, int(record["cycles"]) + 1, dtype=np.float64)
            mean = record["variance_percent_sample_mean"]
            std = record["variance_percent_sample_stdev"]
            lower, upper = _loglog_band(mean, std)
            ax_pct_mean_std.plot(cycles, mean, linewidth=1.2, label=f"N{record['Nx']}x{record['Ny']}")
            ax_pct_mean_std.fill_between(cycles, lower, upper, alpha=0.18)
        ax_pct_mean_std.set_xscale("log")
        ax_pct_mean_std.set_yscale("log")
        ax_pct_mean_std.set_xlabel("cycle")
        ax_pct_mean_std.set_ylabel(r"$\mathbb{E}_{\mathrm{samples}}[100\,\mathrm{Var}(Q)/(N_xN_y/2)] \pm \sigma_{\mathrm{samples}}$")
        ax_pct_mean_std.legend(loc="best")
        ax_pct_mean_std.grid(alpha=0.25, linewidth=0.4, which="both")
        fig_pct_mean_std.savefig(percent_mean_std_path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig_pct_mean_std)

        fig_raw, ax_raw = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for idx, record in enumerate(records):
            cycles = np.arange(1, int(record["cycles"]) + 1, dtype=np.float64)
            _plot_variance_with_overlay(
                ax_raw,
                cycles,
                record["variance_raw_sample_variance"],
                label=f"N{record['Nx']}x{record['Ny']}",
                color=color_cycle[idx % len(color_cycle)],
            )
        ax_raw.set_xlabel("cycle")
        ax_raw.set_ylabel(r"$\mathrm{Var}_{\mathrm{samples}}[\mathrm{Var}(Q)]$")
        ax_raw.legend(loc="best")
        ax_raw.grid(alpha=0.25, linewidth=0.4)
        fig_raw.savefig(raw_path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig_raw)

        fig_pct, ax_pct = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
        for idx, record in enumerate(records):
            cycles = np.arange(1, int(record["cycles"]) + 1, dtype=np.float64)
            _plot_variance_with_overlay(
                ax_pct,
                cycles,
                record["variance_percent_sample_variance"],
                label=f"N{record['Nx']}x{record['Ny']}",
                color=color_cycle[idx % len(color_cycle)],
            )
        ax_pct.set_xlabel("cycle")
        ax_pct.set_ylabel(r"$\mathrm{Var}_{\mathrm{samples}}[100\,\mathrm{Var}(Q)/(N_xN_y/2)]$")
        ax_pct.legend(loc="best")
        ax_pct.grid(alpha=0.25, linewidth=0.4)
        fig_pct.savefig(percent_path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig_pct)

    return {
        "raw_mean_path": raw_mean_path,
        "raw_mean_std_path": raw_mean_std_path,
        "percent_mean_path": percent_mean_path,
        "percent_mean_std_path": percent_mean_std_path,
        "raw_variance_path": raw_path,
        "percent_variance_path": percent_path,
    }


def make_sample_averaged_spatial_charge_variance_video(
    records: list[dict[str, Any]],
    *,
    output_root: Path | str,
) -> dict[str, Any]:
    pooled_min = min(float(np.min(record["local_charge_cell_variance_avg"])) for record in records)
    pooled_max = max(float(np.max(record["local_charge_cell_variance_avg"])) for record in records)
    norm = colors.Normalize(vmin=pooled_min, vmax=pooled_max)

    def _title(record: dict[str, Any], cycle_idx: int) -> str:
        return (
            f"N{record['Nx']}x{record['Ny']} cycle {cycle_idx + 1}\n"
            f"E_samp[Var(Q)]={record['variance_raw_sample_mean'][cycle_idx]:.4f}; "
            f"Var_samp[Var(Q)]={record['variance_raw_sample_variance'][cycle_idx]:.4f}"
        )

    return _make_three_panel_heatmap_video(
        records,
        data_key="local_charge_cell_variance_avg",
        output_root=output_root,
        video_filename=TOTAL_CHARGE_VARIANCE_HEATMAP_VIDEO_FILENAME,
        final_frame_filename=TOTAL_CHARGE_VARIANCE_HEATMAP_FINAL_FRAME_FILENAME,
        suptitle="Sample-averaged spatial charge variance",
        colorbar_label="sample-averaged local charge variance",
        cmap="magma",
        norm=norm,
        title_fn=_title,
    )


def write_centered_total_charge_analysis_manifest(
    *,
    bundle_root: Path | str,
    campaign_id: str,
    records: list[dict[str, Any]],
    stats_df: pd.DataFrame,
    video_artifacts: dict[str, Any],
    variance_artifacts: dict[str, Any],
    entropy_artifacts: dict[str, Any] | None = None,
    chern_artifacts: dict[str, Any] | None = None,
    expected_charge_video_artifacts: dict[str, Any] | None = None,
) -> dict[str, Path]:
    bundle_root = Path(bundle_root)
    paths = _analysis_paths(bundle_root, campaign_id)
    stats_df.to_csv(paths["stats_table_path"], index=False)
    manifest = {
        "analysis_name": ANALYSIS_NAME,
        "campaign_id": campaign_id,
        "selected_config_ids": [record["config_id"] for record in records],
        "raw_centered_total_charge_formula": RAW_FORMULA,
        "percent_normalized_centered_total_charge_formula": PERCENT_FORMULA,
        "variance_ddof": 0,
        "histogram_observable": "percent_normalized_centered_total_charge",
        "frame_repeat": video_artifacts["frame_repeat"],
        "fps": video_artifacts["fps"],
        "total_cycles": video_artifacts["total_cycles"],
        "total_frames": video_artifacts["total_frames"],
        "percent_bin_edges": video_artifacts["bin_edges"],
        "percent_xlim": video_artifacts["xlim"],
        "count_ylim": video_artifacts["ylim"],
        "codec": video_artifacts["codec"],
        "pix_fmt": video_artifacts["pix_fmt"],
        "movflags": video_artifacts["movflags"],
        "video_path": str(video_artifacts["video_path"]),
        "final_frame_path": str(video_artifacts["final_frame_path"]),
        "raw_mean_figure_path": str(variance_artifacts["raw_mean_path"]),
        "raw_mean_std_figure_path": str(variance_artifacts["raw_mean_std_path"]),
        "percent_mean_figure_path": str(variance_artifacts["percent_mean_path"]),
        "percent_mean_std_figure_path": str(variance_artifacts["percent_mean_std_path"]),
        "raw_variance_figure_path": str(variance_artifacts["raw_variance_path"]),
        "percent_variance_figure_path": str(variance_artifacts["percent_variance_path"]),
        "entropy_loglog_figure_path": None if entropy_artifacts is None else str(entropy_artifacts["entropy_loglog_path"]),
        "entropy_loglog_std_figure_path": None if entropy_artifacts is None else str(entropy_artifacts["entropy_loglog_std_path"]),
        "chern_loglog_figure_path": None if chern_artifacts is None else str(chern_artifacts["chern_loglog_path"]),
        "expected_charge_video_path": None
        if expected_charge_video_artifacts is None
        else str(expected_charge_video_artifacts["video_path"]),
        "expected_charge_final_frame_path": None
        if expected_charge_video_artifacts is None
        else str(expected_charge_video_artifacts["final_frame_path"]),
        "stats_table_path": str(paths["stats_table_path"]),
        "sample_stats_rows": int(len(stats_df)),
    }
    _write_json(paths["manifest_path"], manifest)
    return paths


def write_total_charge_variance_analysis_manifest(
    *,
    bundle_root: Path | str,
    campaign_id: str,
    records: list[dict[str, Any]],
    stats_df: pd.DataFrame,
    video_artifacts: dict[str, Any],
    summary_artifacts: dict[str, Any],
    spatial_variance_video_artifacts: dict[str, Any] | None = None,
) -> dict[str, Path]:
    bundle_root = Path(bundle_root)
    paths = _total_charge_variance_analysis_paths(bundle_root, campaign_id)
    stats_df.to_csv(paths["stats_table_path"], index=False)
    manifest = {
        "analysis_name": TOTAL_CHARGE_VARIANCE_ANALYSIS_NAME,
        "campaign_id": campaign_id,
        "selected_config_ids": [record["config_id"] for record in records],
        "raw_total_charge_variance_formula": TOTAL_CHARGE_VARIANCE_RAW_FORMULA,
        "percent_normalized_total_charge_variance_formula": TOTAL_CHARGE_VARIANCE_PERCENT_FORMULA,
        "variance_ddof": 0,
        "histogram_observable": "percent_normalized_total_charge_variance",
        "frame_repeat": video_artifacts["frame_repeat"],
        "fps": video_artifacts["fps"],
        "total_cycles": video_artifacts["total_cycles"],
        "total_frames": video_artifacts["total_frames"],
        "percent_bin_edges": video_artifacts["bin_edges"],
        "percent_xlim": video_artifacts["xlim"],
        "count_ylim": video_artifacts["ylim"],
        "codec": video_artifacts["codec"],
        "pix_fmt": video_artifacts["pix_fmt"],
        "movflags": video_artifacts["movflags"],
        "video_path": str(video_artifacts["video_path"]),
        "final_frame_path": str(video_artifacts["final_frame_path"]),
        "spatial_variance_video_path": None
        if spatial_variance_video_artifacts is None
        else str(spatial_variance_video_artifacts["video_path"]),
        "spatial_variance_final_frame_path": None
        if spatial_variance_video_artifacts is None
        else str(spatial_variance_video_artifacts["final_frame_path"]),
        "raw_mean_figure_path": str(summary_artifacts["raw_mean_path"]),
        "raw_mean_std_figure_path": str(summary_artifacts["raw_mean_std_path"]),
        "percent_mean_figure_path": str(summary_artifacts["percent_mean_path"]),
        "percent_mean_std_figure_path": str(summary_artifacts["percent_mean_std_path"]),
        "raw_variance_figure_path": str(summary_artifacts["raw_variance_path"]),
        "percent_variance_figure_path": str(summary_artifacts["percent_variance_path"]),
        "stats_table_path": str(paths["stats_table_path"]),
        "sample_stats_rows": int(len(stats_df)),
    }
    _write_json(paths["manifest_path"], manifest)
    return paths


_PROTOCOL_COLORS = {
    TARGET_PROTOCOL: None,
    POSTSELECT_PROTOCOL: None,
}
_PROTOCOL_LINESTYLES = {
    TARGET_PROTOCOL: "-",
    POSTSELECT_PROTOCOL: "--",
}
_PROTOCOL_LABELS = {
    TARGET_PROTOCOL: "perf. corr.",
    POSTSELECT_PROTOCOL: "post-sel.",
}

_NY_PALETTE = ["#2b6cb0", "#276749", "#9b2c2c"]


def make_protocol_comparison_figures(
    perf_entropy_records: list[dict[str, Any]],
    post_entropy_records: list[dict[str, Any]],
    perf_chern_records: list[dict[str, Any]],
    post_chern_records: list[dict[str, Any]],
    perf_variance_records: list[dict[str, Any]],
    post_variance_records: list[dict[str, Any]],
    *,
    output_root: Path | str,
) -> dict[str, Path]:
    """Side-by-side loglog comparison of perfect_correction vs postselect for entropy, Chern, and charge variance."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    entropy_path = output_root / "purification_protocol_comparison_entropy_loglog.png"
    chern_path = output_root / "purification_protocol_comparison_chern_loglog.png"
    variance_path = output_root / "purification_protocol_comparison_charge_variance_loglog.png"

    def _ny_color(record: dict[str, Any]) -> str:
        try:
            idx = list(TARGET_NY).index(int(record["Ny"]))
        except ValueError:
            idx = 0
        return _NY_PALETTE[idx % len(_NY_PALETTE)]

    with plt.rc_context(_style_context()):
        # --- Entropy ---
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
        for protocol_records, protocol in [
            (perf_entropy_records, TARGET_PROTOCOL),
            (post_entropy_records, POSTSELECT_PROTOCOL),
        ]:
            ls = _PROTOCOL_LINESTYLES[protocol]
            proto_label = _PROTOCOL_LABELS[protocol]
            for record in protocol_records:
                cycles = np.arange(1, int(record["cycles"]) + 1, dtype=np.float64)
                entropy_mean = np.mean(record["total_entropy"], axis=0)
                color = _ny_color(record)
                ax.plot(
                    cycles,
                    entropy_mean,
                    linewidth=1.2,
                    linestyle=ls,
                    color=color,
                    label=f"Ny={record['Ny']} {proto_label}",
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("cycle")
        ax.set_ylabel(r"$S_{\mathrm{tot}}$")
        ax.set_title("Total entropy: perf. corr. vs post-sel.")
        ax.legend(loc="best", fontsize=6)
        ax.grid(alpha=0.25, linewidth=0.4, which="both")
        fig.savefig(entropy_path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig)

        # --- Chern ---
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
        for protocol_records, protocol in [
            (perf_chern_records, TARGET_PROTOCOL),
            (post_chern_records, POSTSELECT_PROTOCOL),
        ]:
            ls = _PROTOCOL_LINESTYLES[protocol]
            proto_label = _PROTOCOL_LABELS[protocol]
            for record in protocol_records:
                cycles = np.arange(1, int(record["cycles"]) + 1, dtype=np.float64)
                color = _ny_color(record)
                ax.plot(
                    cycles,
                    record["chern_mean"],
                    linewidth=1.2,
                    linestyle=ls,
                    color=color,
                    label=f"Ny={record['Ny']} {proto_label}",
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("cycle")
        ax.set_ylabel(r"$\mathcal{C}_{\mathrm{RS}}$")
        ax.set_title("Chern number: perf. corr. vs post-sel.")
        ax.legend(loc="best", fontsize=6)
        ax.grid(alpha=0.25, linewidth=0.4, which="both")
        fig.savefig(chern_path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig)

        # --- Total charge variance ---
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
        for protocol_records, protocol in [
            (perf_variance_records, TARGET_PROTOCOL),
            (post_variance_records, POSTSELECT_PROTOCOL),
        ]:
            ls = _PROTOCOL_LINESTYLES[protocol]
            proto_label = _PROTOCOL_LABELS[protocol]
            for record in protocol_records:
                cycles = np.arange(1, int(record["cycles"]) + 1, dtype=np.float64)
                variance_mean = np.mean(record["variance_pct"], axis=0)
                color = _ny_color(record)
                ax.plot(
                    cycles,
                    variance_mean,
                    linewidth=1.2,
                    linestyle=ls,
                    color=color,
                    label=f"Ny={record['Ny']} {proto_label}",
                )
        ax.set_xscale("log")
        ax.set_xlabel("cycle")
        ax.set_ylabel(r"$\mathrm{Var}(Q)$ / baseline [%]")
        ax.set_title("Total charge variance: perf. corr. vs post-sel.")
        ax.legend(loc="best", fontsize=6)
        ax.grid(alpha=0.25, linewidth=0.4, which="both")
        fig.savefig(variance_path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig)

    return {
        "entropy_comparison_path": entropy_path,
        "chern_comparison_path": chern_path,
        "variance_comparison_path": variance_path,
    }


__all__ = [
    "ANALYSIS_NAME",
    "POSTSELECT_PROTOCOL",
    "TARGET_PROTOCOL",
    "compute_centered_total_charge_sample_stats",
    "compute_total_charge_variance_sample_stats",
    "make_centered_total_charge_histogram_video",
    "make_centered_total_charge_variance_figures",
    "make_protocol_comparison_figures",
    "make_real_space_chern_loglog_figure",
    "make_sample_averaged_expected_charge_video",
    "make_sample_averaged_spatial_charge_variance_video",
    "make_total_charge_variance_histogram_video",
    "make_total_charge_variance_summary_figures",
    "make_total_entropy_loglog_figure",
    "prepare_centered_total_charge_histogram_records",
    "prepare_real_space_chern_records",
    "prepare_total_charge_variance_histogram_records",
    "prepare_total_entropy_records",
    "write_centered_total_charge_analysis_manifest",
    "write_total_charge_variance_analysis_manifest",
]
