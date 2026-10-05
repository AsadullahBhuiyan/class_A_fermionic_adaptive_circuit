#!/usr/bin/env python3
"""Make the Lane-B left/right half-system entropy-collapse figure."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BUNDLE_ROOT = (
    ROOT
    / "00_WORKSPACE/CURRENT/final_production_new_designs"
    / "16_hard_wall_entropy_contour_all_ay"
)
OUTPUT_ROOT = (
    BUNDLE_ROOT
    / "gpu_data"
    / "hard_wall_entropy_contour_all_ay_nx20_ny30-60_s100_2ny_raster_v3"
)
FIGURE_DIR = HERE / "figures"
DATA_DIR = HERE / "data"
HALF_FIGURE_STEM = FIGURE_DIR / "lane_B_half_system_entropy_collapse_2x1"
HALF_CURVE_CSV = DATA_DIR / "lane_B_half_system_anchored_curves.csv"
HALF_FIT_CSV = DATA_DIR / "lane_B_half_system_joint_fits.csv"
WALL_FIGURE_STEM = FIGURE_DIR / "lane_B_wall_only_entropy_collapse_2x1"
WALL_CURVE_CSV = DATA_DIR / "lane_B_wall_only_anchored_curves.csv"
WALL_FIT_CSV = DATA_DIR / "lane_B_wall_only_joint_fits.csv"
MANIFEST_PATH = HERE / "analysis_manifest.json"

NX = 20
NY_VALUES = (30, 35, 40, 45, 55)
SAMPLES = 100
FIT_MIN_AY = 8
SLOPE_TARGET = 1.0 / 6.0
LEFT_X = np.arange(0, NX // 2, dtype=np.int64)
RIGHT_X = np.arange(NX // 2, NX, dtype=np.int64)
LEFT_WALL_X = np.asarray((4, 5, 6), dtype=np.int64)
RIGHT_WALL_X = np.asarray((14, 15, 16), dtype=np.int64)
CONTOUR_KEY = "endpoint__contour_von_neumann_y0avg"
ENTROPY_KEY = "endpoint__entropy_von_neumann"
CAMPAIGN = "hard_wall_entropy_contour_all_ay_nx20_ny30-60_s100_2ny_raster_v3"
CONFIG_SHA256 = "3323c08f8b7d6cf90c514c91faec16335cadcb483632e588d7111e77c3a380f2"
RESULT_SCHEMA = "hard_wall_entropy_contour_all_ay_result_shard_v3"
COMPLETION_SCHEMA = "hard_wall_entropy_contour_all_ay_completion_v3"

SIZE_STYLES = {
    30: {"color": "#D92725", "marker": "^"},
    35: {"color": "#F08050", "marker": "<"},
    40: {"color": "#8FC1E3", "marker": "v"},
    45: {"color": "#2CA02C", "marker": "s"},
    55: {"color": "#000000", "marker": "P"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(payload: dict[str, np.ndarray], key: str) -> Any:
    if key not in payload or payload[key].ndim != 0:
        raise RuntimeError(f"missing scalar field {key!r}")
    return payload[key].item()


def load_lane_b() -> tuple[dict[int, dict[str, np.ndarray]], float, list[Path]]:
    """Validate all Lane-B result/receipt pairs and load trajectory arrays."""

    receipt_paths = sorted(OUTPUT_ROOT.glob("results/lane_B/Ny*/*.complete.json"))
    expected_receipts = len(NY_VALUES) * (SAMPLES // 5)
    if len(receipt_paths) != expected_receipts:
        raise RuntimeError(
            f"expected {expected_receipts} Lane-B receipts, found {len(receipt_paths)}"
        )

    shards: dict[int, list[dict[str, np.ndarray]]] = {ny: [] for ny in NY_VALUES}
    inputs: list[Path] = []
    closure_max = 0.0
    for receipt_path in receipt_paths:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        ny = int(receipt.get("Ny", -1))
        if (
            receipt.get("schema") != COMPLETION_SCHEMA
            or receipt.get("sampling_revision") != CAMPAIGN
            or receipt.get("bundle") != "16_hard_wall_entropy_contour_all_ay"
            or receipt.get("lane") != "B"
            or receipt.get("canonical_entry_point")
            != "classA_U1FGTN_gpu.run_markov_circuit"
            or receipt.get("config_sha256") != CONFIG_SHA256
            or ny not in NY_VALUES
            or int(receipt.get("sample_count", -1)) != 5
            or int(receipt.get("cycles", -1)) != 2 * ny
            or receipt.get("endpoint_only") is not True
        ):
            raise RuntimeError(f"completion identity mismatch: {receipt_path}")

        start = int(receipt["sample_start"])
        stop = int(receipt["sample_stop"])
        shard_index = int(receipt["shard_index"])
        expected_ids = list(range(start, stop))
        if stop != start + 5 or shard_index != start // 5:
            raise RuntimeError(f"invalid shard bounds: {receipt_path}")
        if receipt.get("global_sample_indices") != expected_ids:
            raise RuntimeError(f"sample IDs mismatch: {receipt_path}")

        result_path = receipt_path.with_name(str(receipt["result_filename"]))
        if not result_path.is_file():
            raise RuntimeError(f"missing result for {receipt_path}")
        if result_path.stat().st_size != int(receipt["result_bytes"]):
            raise RuntimeError(f"result byte-count mismatch: {result_path}")
        if sha256_file(result_path) != receipt["result_sha256"]:
            raise RuntimeError(f"result checksum mismatch: {result_path}")

        with np.load(result_path, allow_pickle=False) as archive:
            payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
        if (
            _scalar(payload, "schema") != RESULT_SCHEMA
            or _scalar(payload, "sampling_revision") != CAMPAIGN
            or _scalar(payload, "lane") != "B"
            or int(_scalar(payload, "Nx")) != NX
            or int(_scalar(payload, "Ny")) != ny
            or int(_scalar(payload, "cycles_total")) != 2 * ny
            or _scalar(payload, "endpoint_only") is not True
            or _scalar(payload, "config_sha256") != CONFIG_SHA256
        ):
            raise RuntimeError(f"result identity mismatch: {result_path}")
        if not np.array_equal(payload["sample_ids"], np.asarray(expected_ids)):
            raise RuntimeError(f"result sample IDs mismatch: {result_path}")
        if int(_scalar(payload, "origin_average_count")) != ny:
            raise RuntimeError(f"origin-average count mismatch: {result_path}")
        if _scalar(payload, "contour_coordinate") != "relative_dy=(y-y0)_mod_Ny":
            raise RuntimeError(f"contour-coordinate mismatch: {result_path}")

        half = ny // 2
        ay_values = np.asarray(payload["ay_values"], dtype=np.int64)
        if not np.array_equal(ay_values, np.arange(half + 1)):
            raise RuntimeError(f"Ay axis mismatch: {result_path}")
        contour = np.asarray(payload[CONTOUR_KEY], dtype=np.float64)
        entropy = np.asarray(payload[ENTROPY_KEY], dtype=np.float64)
        if contour.shape != (5, half + 1, NX, half):
            raise RuntimeError(f"contour shape mismatch: {result_path}")
        if entropy.shape != (5, half + 1):
            raise RuntimeError(f"entropy shape mismatch: {result_path}")
        if not np.all(np.isfinite(contour)) or np.min(contour) < -1.0e-9:
            raise RuntimeError(f"invalid contour values: {result_path}")
        for width_index, ay in enumerate(ay_values):
            active = contour[:, width_index, :, :ay]
            padding = contour[:, width_index, :, ay:]
            if padding.size and not np.all(padding == 0.0):
                raise RuntimeError(f"nonzero contour padding: {result_path}, Ay={ay}")
            closure = active.sum(axis=(1, 2)) - entropy[:, width_index]
            closure_max = max(closure_max, float(np.max(np.abs(closure), initial=0.0)))
        shards[ny].append(payload)
        inputs.extend((receipt_path, result_path))

    cases: dict[int, dict[str, np.ndarray]] = {}
    for ny in NY_VALUES:
        rows = shards[ny]
        if len(rows) != SAMPLES // 5:
            raise RuntimeError(f"Ny={ny}: expected 20 shards, found {len(rows)}")
        ids = np.concatenate([row["sample_ids"] for row in rows])
        order = np.argsort(ids)
        if not np.array_equal(ids[order], np.arange(SAMPLES)):
            raise RuntimeError(f"Ny={ny}: sample IDs are not exactly 0..99")
        cases[ny] = {
            "sample_ids": ids[order],
            "ay_values": rows[0]["ay_values"],
            CONTOUR_KEY: np.concatenate([row[CONTOUR_KEY] for row in rows], axis=0)[order],
            ENTROPY_KEY: np.concatenate([row[ENTROPY_KEY] for row in rows], axis=0)[order],
        }
    if closure_max > 2.0e-8:
        raise RuntimeError(f"contour closure failed: maximum error {closure_max:.3e}")
    return cases, closure_max, inputs


def integrate_halves(case: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Integrate every origin-averaged contour over the two x half systems."""

    ay_values = np.asarray(case["ay_values"], dtype=np.int64)
    contour = np.asarray(case[CONTOUR_KEY], dtype=np.float64)
    left = np.zeros((SAMPLES, ay_values.size), dtype=np.float64)
    right = np.zeros_like(left)
    for index, ay in enumerate(ay_values):
        active = contour[:, index, :, :ay]
        left[:, index] = active[:, LEFT_X, :].sum(axis=(1, 2))
        right[:, index] = active[:, RIGHT_X, :].sum(axis=(1, 2))
    full = np.asarray(case[ENTROPY_KEY], dtype=np.float64)
    if not np.allclose(left + right, full, rtol=0.0, atol=2.0e-8):
        raise RuntimeError("left/right half-system integrals do not close to S1")
    return {"left": left, "right": right}


def integrate_walls(case: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Integrate only the three-cell windows centered on the two walls."""

    ay_values = np.asarray(case["ay_values"], dtype=np.int64)
    contour = np.asarray(case[CONTOUR_KEY], dtype=np.float64)
    left = np.zeros((SAMPLES, ay_values.size), dtype=np.float64)
    right = np.zeros_like(left)
    for index, ay in enumerate(ay_values):
        active = contour[:, index, :, :ay]
        left[:, index] = active[:, LEFT_WALL_X, :].sum(axis=(1, 2))
        right[:, index] = active[:, RIGHT_WALL_X, :].sum(axis=(1, 2))
    full = np.asarray(case[ENTROPY_KEY], dtype=np.float64)
    if np.any(left + right > full + 2.0e-8):
        raise RuntimeError("wall-window entropy exceeds the full-strip entropy")
    return {"left": left, "right": right}


def anchored_joint_fit(
    cases: dict[int, dict[str, np.ndarray]],
    component: str,
    integrator: Any,
) -> tuple[dict[int, dict[str, Any]], dict[str, float]]:
    """Fit one equal-size-weighted slope after per-trajectory endpoint anchoring."""

    by_size: dict[int, dict[str, Any]] = {}
    denominator = 0.0
    numerator = 0.0
    anchored_total = 0.0
    for ny in NY_VALUES:
        ay = np.asarray(cases[ny]["ay_values"], dtype=np.int64)
        curves = integrator(cases[ny])[component]
        endpoint = int(np.flatnonzero(ay == ny // 2)[0])
        plotted = ay >= 1
        selected = ay >= FIT_MIN_AY
        anchor = math.log(math.sin(math.pi * (ny // 2) / ny))
        x_all = np.log(np.sin(math.pi * ay[plotted] / ny)) - anchor
        x_fit = np.log(np.sin(math.pi * ay[selected] / ny)) - anchor
        delta_all = curves[:, plotted] - curves[:, [endpoint]]
        delta_fit = curves[:, selected] - curves[:, [endpoint]]
        mean_all = delta_all.mean(axis=0)
        mean_fit = delta_fit.mean(axis=0)
        sem_all = delta_all.std(axis=0, ddof=1) / math.sqrt(SAMPLES)
        if not np.isclose(x_fit[-1], 0.0, atol=1.0e-14, rtol=0.0):
            raise RuntimeError(f"Ny={ny}: endpoint x is not zero")
        if not np.isclose(mean_fit[-1], 0.0, atol=1.0e-14, rtol=0.0):
            raise RuntimeError(f"Ny={ny}: endpoint entropy is not zero")
        weight = 1.0 / x_fit.size
        numerator += weight * float(x_fit @ mean_fit)
        denominator += weight * float(x_fit @ x_fit)
        anchored_total += weight * float(mean_fit @ mean_fit)
        by_size[ny] = {
            "ay_all": ay[plotted],
            "ay_fit": ay[selected],
            "x_all": x_all,
            "x_fit": x_fit,
            "mean_all": mean_all,
            "mean_fit": mean_fit,
            "sem_all": sem_all,
            "delta_fit_samples": delta_fit,
            "anchor_Ay": ny // 2,
        }

    slope = numerator / denominator
    rss = 0.0
    variance = 0.0
    for ny in NY_VALUES:
        item = by_size[ny]
        x_fit = item["x_fit"]
        mean_fit = item["mean_fit"]
        delta_fit = item["delta_fit_samples"]
        weight = 1.0 / x_fit.size
        residual = mean_fit - slope * x_fit
        rss += weight * float(residual @ residual)
        tss = float(mean_fit @ mean_fit)
        item["group_R0_squared"] = 1.0 - float(residual @ residual) / tss
        projection = (weight / denominator) * x_fit
        covariance_of_mean = np.cov(delta_fit, rowvar=False, ddof=1) / SAMPLES
        variance += float(projection @ covariance_of_mean @ projection)
    summary = {
        "slope": float(slope),
        "slope_covariance_sem": math.sqrt(max(variance, 0.0)),
        "R0_squared": 1.0 - rss / anchored_total,
    }
    return by_size, summary


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times", "Nimbus Roman", "Times New Roman", "Liberation Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "lines.linewidth": 0.9,
            "lines.markersize": 3.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    fits: dict[str, tuple[dict[int, dict[str, Any]], dict[str, float]]],
    figure_stem: Path,
    panel_names: dict[str, tuple[str, str, str, str]],
    wall_window_cells: dict[str, np.ndarray] | None = None,
    inset_right_wall_at_cell_edge: bool = False,
    plot_min_ay: int = 1,
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(3.375, 4.85), sharex=True)
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    for panel, (axis, component) in enumerate(zip(axes, ("left", "right"))):
        by_size, summary = fits[component]
        fit_min = min(float(item["x_fit"].min()) for item in by_size.values())
        axis.axvspan(fit_min, 0.0, color="0.5", alpha=0.15, linewidth=0, zorder=0)
        for ny in NY_VALUES:
            item = by_size[ny]
            style = SIZE_STYLES[ny]
            shown = item['ay_all'] >= plot_min_ay
            empirical = axis.errorbar(
                item["x_all"][shown],
                item["mean_all"][shown],
                yerr=item["sem_all"][shown],
                color=style["color"],
                marker=style["marker"],
                linestyle="none",
                markerfacecolor="white",
                markeredgewidth=0.8,
                markersize=3.4,
                elinewidth=0.45,
                capsize=0.0,
                zorder=3,
            )
            if panel == 0:
                legend_handles.append(empirical[0])
                legend_labels.append(rf"$N_y={ny}$")
        x_line = np.linspace(
            min(float(item["x_all"][item['ay_all'] >= plot_min_ay].min())
                for item in by_size.values()), 0.0, 300
        )
        axis.plot(
            x_line,
            summary["slope"] * x_line,
            color="black",
            linestyle="--",
            linewidth=0.9,
            zorder=2,
        )
        wall_name, x_range, slope_symbol, entropy_symbol = panel_names[component]
        axis.text(
            0.025,
            0.93,
            (
                rf"{wall_name}  ({x_range})" "\n"
                rf"${slope_symbol}={summary['slope']:.5f}\pm"
                rf"{summary['slope_covariance_sem']:.5f}$, "
                rf"$R_0^2={summary['R0_squared']:.6f}$"
            ),
            transform=axis.transAxes,
            fontsize=6.2,
            va="top",
        )
        if wall_window_cells is not None:
            selected = np.asarray(wall_window_cells[component], dtype=np.int64)
            if selected.ndim != 1 or selected.size == 0:
                raise ValueError(f"invalid wall-window cells for {component}")
            inset_axis = axis.inset_axes([0.075, 0.33, 0.17, 0.43])
            inset_axis.add_patch(
                Rectangle(
                    (float(selected.min()), 0.0),
                    float(selected.max() - selected.min() + 1),
                    30.0,
                    facecolor="#4C78A8",
                    edgecolor="none",
                    alpha=0.40,
                    zorder=0,
                )
            )
            for coordinate in range(NX + 1):
                inset_axis.axvline(
                    coordinate,
                    color="0.45",
                    alpha=0.34,
                    linewidth=0.16,
                    zorder=1,
                )
            for coordinate in range(31):
                inset_axis.axhline(
                    coordinate,
                    color="0.45",
                    alpha=0.34,
                    linewidth=0.16,
                    zorder=1,
                )
            for wall_x, wall_label in ((5, r"$x_{\rm L}$"), (15, r"$x_{\rm R}$")):
                if inset_right_wall_at_cell_edge and component == "right" and wall_x == 15:
                    wall_x = 16  # Right edge of cell x=15 in the inset grid.
                inset_axis.axvline(
                    wall_x,
                    color="#A51C30",
                    linestyle="--",
                    linewidth=0.75,
                    zorder=3,
                )
                inset_axis.text(
                    wall_x,
                    30.7,
                    wall_label,
                    color="#A51C30",
                    fontsize=4.8,
                    ha="center",
                    va="bottom",
                    clip_on=False,
                    zorder=4,
                )
            inset_axis.set_xlim(0, NX)
            inset_axis.set_ylim(0, 30)
            inset_axis.set_aspect("equal")
            inset_axis.set_xticks(())
            inset_axis.set_yticks(())
            inset_axis.set_axis_off()
        axis.text(
            -0.18,
            1.04,
            f"({chr(ord('a') + panel)})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
        )
        axis.set_xlim(float(x_line.min()) - .06 if plot_min_ay > 1 else -3.05, 0.05)
        axis.set_ylabel(rf"$\Delta\langle {entropy_symbol}\rangle_\xi$")
    axes[0].legend(
        legend_handles,
        legend_labels,
        ncol=2,
        loc="lower right",
        columnspacing=0.7,
        handletextpad=0.3,
        borderaxespad=0.4,
    )
    axes[1].set_xlabel(
        r"$\log[\sin(\pi A_y/N_y)/\sin(\pi A_y^\star/N_y)]$"
    )
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.105,
                        top=0.95 if plot_min_ay > 1 else 0.965, hspace=0.16)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_stem.with_suffix(".pdf"))
    fig.savefig(figure_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> int:
    cases, closure_max, inputs = load_lane_b()
    fits = {
        component: anchored_joint_fit(cases, component, integrate_halves)
        for component in ("left", "right")
    }

    curve_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for component, (by_size, summary) in fits.items():
        for ny in NY_VALUES:
            item = by_size[ny]
            for ay, x, mean, sem in zip(
                item["ay_all"], item["x_all"], item["mean_all"], item["sem_all"]
            ):
                curve_rows.append(
                    {
                        "half": component,
                        "Nx": NX,
                        "Ny": ny,
                        "samples": SAMPLES,
                        "Ay": int(ay),
                        "anchor_Ay": ny // 2,
                        "delta_log_sine_chord": float(x),
                        "anchored_entropy_mean": float(mean),
                        "anchored_entropy_trajectory_SEM": float(sem),
                        "in_fit_window": bool(ay >= FIT_MIN_AY),
                    }
                )
        fit_rows.append(
            {
                "half": component,
                "x_cells": "0..9" if component == "left" else "10..19",
                "Ny_values": ",".join(map(str, NY_VALUES)),
                "samples_per_Ny": SAMPLES,
                "Ay_fit_min": FIT_MIN_AY,
                "Ay_fit_max": "Ny//2",
                "estimator_order": "anchor_each_trajectory_then_average_then_joint_fit",
                "size_weighting": "equal_total_weight_per_Ny",
                "slope": summary["slope"],
                "slope_covariance_SEM": summary["slope_covariance_sem"],
                "slope_target": SLOPE_TARGET,
                "R0_squared": summary["R0_squared"],
                "uncertainty": "ordinary_trajectory_SEM_full_Ay_covariance",
            }
        )

    write_csv(HALF_CURVE_CSV, curve_rows)
    write_csv(HALF_FIT_CSV, fit_rows)
    make_figure(
        fits,
        HALF_FIGURE_STEM,
        {
            "left": ("left wall", r"$x=0,\ldots,9$", r"m_{\rm L}", r"\overline{S}_{\rm L}"),
            "right": ("right wall", r"$x=10,\ldots,19$", r"m_{\rm R}", r"\overline{S}_{\rm R}"),
        },
    )

    outputs = [
        HALF_FIGURE_STEM.with_suffix(".pdf"),
        HALF_FIGURE_STEM.with_suffix(".png"),
        HALF_CURVE_CSV,
        HALF_FIT_CSV,
    ]
    manifest = {
        "schema": "hard_wall_lane_B_half_system_entropy_collapse_v1",
        "campaign": CAMPAIGN,
        "input_scope": "completed_lane_B_only",
        "Nx": NX,
        "Ny_values": list(NY_VALUES),
        "samples_per_Ny": SAMPLES,
        "trajectory_count": SAMPLES * len(NY_VALUES),
        "endpoint": "t=2Ny",
        "contour_semantics": "exact periodic-y0 average within each trajectory in relative-dy coordinates",
        "partitions": {"left": LEFT_X.tolist(), "right": RIGHT_X.tolist()},
        "integration": "sum over selected x cells and all valid relative-dy cells",
        "fit": {
            "Ay": "8..Ny//2",
            "anchor": "each trajectory at Ay*=Ny//2",
            "cross_size_weighting": "equal total weight per Ny",
            "intercept": 0.0,
            "target_slope": SLOPE_TARGET,
            "uncertainty": "ordinary trajectory SEM propagated with full within-trajectory Ay covariance",
            "bootstrap": False,
        },
        "contour_closure_max_abs": closure_max,
        "verified_result_receipt_pairs": len(inputs) // 2,
        "fits": {component: summary for component, (_, summary) in fits.items()},
        "outputs": {
            path.relative_to(HERE).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in outputs
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
