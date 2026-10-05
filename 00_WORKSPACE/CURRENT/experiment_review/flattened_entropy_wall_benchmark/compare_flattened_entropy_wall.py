#!/usr/bin/env python3
"""Compare hard-wall endpoint entanglement with the flattened-parent ground state.

The dynamical inputs are immutable verified campaigns.  The deterministic
reference is the half-filled ground state of

    H_flat = sum_R(P_A+ + P_B+ - P_A- - P_B-)

with exactly the production hard-wall OW contract (Nx=20, nshell=1,
alpha_1=1, alpha_2=30).  The same Ay window, half-strip anchoring, equal-size
weighting, and x={4,5,6}/{14,15,16} wall integrations are used on both sides.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.linalg import eigh
from scipy.special import xlogy
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"
CACHE_DIR = HERE / "flattened_cache"
MANIFEST_PATH = HERE / "analysis_manifest.json"

ENDPOINT_REVIEW = (
    ROOT
    / "00_WORKSPACE/CURRENT/experiment_review"
    / "entropy_charge_endpoint_sample_resolved"
    / "make_contour_scaling_figures.py"
)
WALL_REVIEW = (
    ROOT
    / "00_WORKSPACE/CURRENT/experiment_review"
    / "hard_wall_all_ay_half_entropy"
    / "make_half_entropy_collapse.py"
)
FLAT_RUNNER = (
    ROOT
    / "00_WORKSPACE/CURRENT/final_production_new_designs"
    / "06_domain_wall_flattened_ground_state_reference"
    / "run_flattened_ground_state_large_ny.py"
)

NX = 20
NY_VALUES = (30, 35, 40, 45, 50, 55, 60)
WALL_DYNAMIC_NY = (30, 35, 40, 45, 55)
FIT_MIN_AY = 8
LEFT_X = np.asarray((4, 5, 6), dtype=np.int64)
RIGHT_X = np.asarray((14, 15, 16), dtype=np.int64)
ENTROPY_TARGET = 1.0 / 3.0
WALL_TARGET = 1.0 / 6.0
TOLERANCE = 1.0e-8

SIZE_STYLES = {
    30: {"color": "#D92725", "marker": "^"},
    35: {"color": "#F08050", "marker": "<"},
    40: {"color": "#8FC1E3", "marker": "v"},
    45: {"color": "#2CA02C", "marker": "s"},
    50: {"color": "#6B6B6B", "marker": "D"},
    55: {"color": "#000000", "marker": "P"},
    60: {"color": "#1F77B4", "marker": "o"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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
            "legend.fontsize": 5.8,
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


def cache_identity(ny: int) -> dict[str, Any]:
    return {
        "schema": "hard_wall_flattened_entropy_wall_benchmark_v1",
        "source_sha256": sha256_file(Path(__file__)),
        "flat_runner_sha256": sha256_file(FLAT_RUNNER),
        "Nx": NX,
        "Ny": int(ny),
        "nshell": 1,
        "alpha_1": 1.0,
        "alpha_2": 30.0,
        "wall": "hard_support_truncated",
        "trial_orbital": "X",
        "filling": 0.5,
    }


def cache_paths(ny: int) -> tuple[Path, Path]:
    return CACHE_DIR / f"Ny{ny:03d}.npz", CACHE_DIR / f"Ny{ny:03d}.complete.json"


def verified_cache(ny: int) -> bool:
    result_path, completion_path = cache_paths(ny)
    if not result_path.is_file() or not completion_path.is_file():
        return False
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    for key, value in cache_identity(ny).items():
        if completion.get(key) != value:
            return False
    return (
        completion.get("result_file") == result_path.name
        and int(completion.get("result_bytes", -1)) == result_path.stat().st_size
        and completion.get("result_sha256") == sha256_file(result_path)
    )


def atomic_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def compute_flattened_size(flat: Any, ny: int) -> dict[str, np.ndarray]:
    started = time.monotonic()
    case = flat.Case(
        wall_index=0,
        nshell_index=0,
        ny_index=NY_VALUES.index(ny),
        alpha_index=20,
        wall="hard",
        nshell_label="1",
        nshell=1,
        ny=int(ny),
        alpha_1=1.0,
    )
    with threadpool_limits(limits=8):
        model = flat._build_model(case)
        if tuple(int(value) for value in model.DW_loc) != (5, 15):
            raise RuntimeError(f"Ny={ny}: unexpected walls {model.DW_loc}")
        projector_delta, diagnostics = flat.flattened_momentum_projector(model)
        del model

        half = ny // 2
        ay_values = np.arange(half + 1, dtype=np.int64)
        entropy = np.zeros(half + 1, dtype=np.float64)
        left = np.zeros_like(entropy)
        right = np.zeros_like(entropy)
        contours = np.zeros((half + 1, NX, half), dtype=np.float64)
        occupation_min = 0.0
        occupation_max = 0.0
        closure_max = 0.0
        for ay in tqdm(
            range(1, half + 1),
            desc=f"flattened Ny={ny}",
            unit="width",
            leave=False,
        ):
            restricted = flat.restricted_projector(projector_delta, range(ay))
            restricted = 0.5 * (restricted + restricted.conj().T)
            values, vectors = eigh(
                restricted,
                check_finite=False,
                overwrite_a=True,
                driver="evd",
            )
            occupation_min = min(occupation_min, float(values.min()))
            occupation_max = max(occupation_max, float(values.max()))
            if values.min() < -TOLERANCE or values.max() > 1.0 + TOLERANCE:
                raise FloatingPointError(
                    f"Ny={ny}, Ay={ay}: occupations outside tolerance "
                    f"[{values.min():.3e},{values.max():.3e}]"
                )
            occupation = np.clip(values, 0.0, 1.0)
            weights = -xlogy(occupation, occupation) - xlogy(
                1.0 - occupation, 1.0 - occupation
            )
            entropy[ay] = float(weights.sum())
            contour = (np.abs(vectors) ** 2 @ weights).reshape(ay, NX, 2)
            contour = contour.sum(axis=2).T
            contours[ay, :, :ay] = contour
            left[ay] = float(contour[LEFT_X].sum())
            right[ay] = float(contour[RIGHT_X].sum())
            closure_max = max(closure_max, abs(float(contour.sum()) - entropy[ay]))

    if closure_max > 2.0e-10:
        raise RuntimeError(f"Ny={ny}: contour closure failed: {closure_max:.3e}")
    payload: dict[str, Any] = {
        "schema": np.asarray("hard_wall_flattened_entropy_wall_benchmark_result_v1"),
        "Nx": np.asarray(NX),
        "Ny": np.asarray(ny),
        "Ay_values": ay_values,
        "entropy_von_neumann": entropy,
        "contour_von_neumann": contours,
        "left_wall_entropy": left,
        "right_wall_entropy": right,
        "wall_windows": np.asarray((LEFT_X, RIGHT_X), dtype=np.int64),
        "contour_closure_max_abs": np.asarray(closure_max),
        "restricted_occupation_min": np.asarray(occupation_min),
        "restricted_occupation_max": np.asarray(occupation_max),
        "elapsed_seconds": np.asarray(time.monotonic() - started),
    }
    for key, value in diagnostics.items():
        payload[f"diagnostic__{key}"] = np.asarray(value)
    return {key: np.asarray(value) for key, value in payload.items()}


def load_or_compute_flattened(flat: Any) -> dict[int, dict[str, np.ndarray]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[int, dict[str, np.ndarray]] = {}
    for ny in NY_VALUES:
        result_path, completion_path = cache_paths(ny)
        if not verified_cache(ny):
            payload = compute_flattened_size(flat, ny)
            atomic_npz(result_path, payload)
            completion = {
                **cache_identity(ny),
                "result_file": result_path.name,
                "result_bytes": result_path.stat().st_size,
                "result_sha256": sha256_file(result_path),
            }
            atomic_json(completion_path, completion)
            if not verified_cache(ny):
                raise OSError(f"Ny={ny}: cache publication failed verification")
        with np.load(result_path, allow_pickle=False) as archive:
            results[ny] = {key: np.asarray(archive[key]).copy() for key in archive.files}
    return results


def deterministic_anchored_fit(
    curves: dict[int, np.ndarray], sizes: tuple[int, ...]
) -> tuple[dict[int, dict[str, Any]], dict[str, float]]:
    by_size: dict[int, dict[str, Any]] = {}
    numerator = 0.0
    denominator = 0.0
    total = 0.0
    for ny in sizes:
        curve = np.asarray(curves[ny], dtype=np.float64)
        ay = np.arange(curve.size, dtype=np.int64)
        endpoint = ny // 2
        plotted = ay >= 1
        selected = ay >= FIT_MIN_AY
        x_all = np.log(np.sin(np.pi * ay[plotted] / ny))
        x_fit = np.log(np.sin(np.pi * ay[selected] / ny))
        y_all = curve[plotted] - curve[endpoint]
        y_fit = curve[selected] - curve[endpoint]
        weight = 1.0 / x_fit.size
        numerator += weight * float(x_fit @ y_fit)
        denominator += weight * float(x_fit @ x_fit)
        total += weight * float(y_fit @ y_fit)
        size_slope = float(x_fit @ y_fit / (x_fit @ x_fit))
        size_residual = y_fit - size_slope * x_fit
        by_size[ny] = {
            "x_all": x_all,
            "y_all": y_all,
            "x_fit": x_fit,
            "y_fit": y_fit,
            "size_slope": size_slope,
            "size_R0_squared": 1.0 - float(size_residual @ size_residual) / float(y_fit @ y_fit),
        }
    slope = numerator / denominator
    rss = 0.0
    for ny in sizes:
        item = by_size[ny]
        residual = item["y_fit"] - slope * item["x_fit"]
        rss += float(residual @ residual) / item["x_fit"].size
    return by_size, {"slope": slope, "R0_squared": 1.0 - rss / total}


def dynamic_wall_size_fit(item: dict[str, Any]) -> tuple[float, float, float]:
    x = np.asarray(item["x_fit"], dtype=np.float64)
    samples = np.asarray(item["delta_fit_samples"], dtype=np.float64)
    mean = samples.mean(axis=0)
    projection = x / float(x @ x)
    slope = float(projection @ mean)
    covariance = np.cov(samples, rowvar=False, ddof=1) / samples.shape[0]
    sem = math.sqrt(max(0.0, float(projection @ covariance @ projection)))
    residual = mean - slope * x
    r0 = 1.0 - float(residual @ residual) / float(mean @ mean)
    return slope, sem, r0


def load_dynamics(endpoint_fig: Any, wall_base: Any) -> dict[str, Any]:
    analyzer = endpoint_fig.load_bundle_analysis()
    endpoint_cases = analyzer.load_cases(analyzer.discover(endpoint_fig.OUTPUT_ROOT))
    full_by_size, full_joint = endpoint_fig.anchored_mean_curve_fit(
        analyzer, endpoint_cases, "c1"
    )
    wall_cases, contour_closure, wall_inputs = wall_base.load_lane_b()
    left_by_size, left_joint = wall_base.anchored_joint_fit(
        wall_cases, "left", wall_base.integrate_walls
    )
    right_by_size, right_joint = wall_base.anchored_joint_fit(
        wall_cases, "right", wall_base.integrate_walls
    )
    return {
        "analyzer": analyzer,
        "endpoint_cases": endpoint_cases,
        "full_by_size": full_by_size,
        "full_joint": full_joint,
        "wall_cases": wall_cases,
        "left_by_size": left_by_size,
        "left_joint": left_joint,
        "right_by_size": right_by_size,
        "right_joint": right_joint,
        "contour_closure": contour_closure,
        "wall_input_pairs": len(wall_inputs) // 2,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def coefficient_rows(
    dynamics: dict[str, Any],
    flat_fits: dict[str, tuple[dict[int, dict[str, Any]], dict[str, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ny in NY_VALUES:
        item = dynamics["full_by_size"][ny]
        rows.append(
            {
                "observable": "full_von_neumann",
                "method": "trajectory_ensemble_mean",
                "Ny": ny,
                "slope": item["size_slope"],
                "slope_SEM": item["size_slope_SEM"],
                "converted_coefficient": 3.0 * item["size_slope"],
                "converted_SEM": 3.0 * item["size_slope_SEM"],
                "target": 1.0,
                "R0_squared": item["size_R0_squared"],
            }
        )
        flat_item = flat_fits["full"][0][ny]
        rows.append(
            {
                "observable": "full_von_neumann",
                "method": "flattened_parent_ground_state",
                "Ny": ny,
                "slope": flat_item["size_slope"],
                "slope_SEM": "",
                "converted_coefficient": 3.0 * flat_item["size_slope"],
                "converted_SEM": "",
                "target": 1.0,
                "R0_squared": flat_item["size_R0_squared"],
            }
        )
    for component in ("left", "right"):
        dynamic_by_size = dynamics[f"{component}_by_size"]
        for ny in WALL_DYNAMIC_NY:
            slope, sem, r0 = dynamic_wall_size_fit(dynamic_by_size[ny])
            rows.append(
                {
                    "observable": f"{component}_three_cell_wall",
                    "method": "trajectory_ensemble_mean",
                    "Ny": ny,
                    "slope": slope,
                    "slope_SEM": sem,
                    "converted_coefficient": 3.0 * slope,
                    "converted_SEM": 3.0 * sem,
                    "target": 0.5,
                    "R0_squared": r0,
                }
            )
        for ny in NY_VALUES:
            item = flat_fits[component][0][ny]
            rows.append(
                {
                    "observable": f"{component}_three_cell_wall",
                    "method": "flattened_parent_ground_state",
                    "Ny": ny,
                    "slope": item["size_slope"],
                    "slope_SEM": "",
                    "converted_coefficient": 3.0 * item["size_slope"],
                    "converted_SEM": "",
                    "target": 0.5,
                    "R0_squared": item["size_R0_squared"],
                }
            )
    return rows


def curve_rows(flattened: dict[int, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ny in NY_VALUES:
        ay = flattened[ny]["Ay_values"].astype(np.int64)
        for index, width in enumerate(ay):
            rows.append(
                {
                    "Nx": NX,
                    "Ny": ny,
                    "Ay": int(width),
                    "entropy_von_neumann": float(flattened[ny]["entropy_von_neumann"][index]),
                    "left_wall_entropy_x456": float(flattened[ny]["left_wall_entropy"][index]),
                    "right_wall_entropy_x141516": float(flattened[ny]["right_wall_entropy"][index]),
                    "in_fit_window": bool(width >= FIT_MIN_AY),
                }
            )
    return rows


def make_full_collapse(
    dynamics: dict[str, Any],
    flat_fit: tuple[dict[int, dict[str, Any]], dict[str, float]],
) -> None:
    configure_matplotlib()
    flat_by_size, flat_summary = flat_fit
    dyn_by_size = dynamics["full_by_size"]
    dyn_summary = dynamics["full_joint"]
    fig, ax = plt.subplots(figsize=(3.375, 2.75))
    fit_min = min(float(item["x_fit"].min()) for item in dyn_by_size.values())
    ax.axvspan(fit_min, 0.0, color="0.5", alpha=0.15, linewidth=0, zorder=0)
    size_handles: list[Any] = []
    for ny in NY_VALUES:
        style = SIZE_STYLES[ny]
        dyn = dyn_by_size[ny]
        empirical = ax.errorbar(
            dyn["x_all"],
            dyn["mean_all"],
            yerr=dyn["sem_all"],
            color=style["color"],
            marker=style["marker"],
            linestyle="none",
            markerfacecolor="white",
            markeredgewidth=0.75,
            markersize=3.2,
            elinewidth=0.4,
            capsize=0.0,
            zorder=3,
        )
        size_handles.append(empirical[0])
        flat = flat_by_size[ny]
        ax.plot(
            flat["x_all"],
            flat["y_all"],
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=0.45,
            markersize=2.0,
            alpha=0.75,
            zorder=2,
        )
    xline = np.linspace(-3.05, 0.0, 300)
    ax.plot(xline, dyn_summary["slope"] * xline, "k--", linewidth=0.9)
    ax.plot(xline, flat_summary["slope"] * xline, color="#6F4C9B", linestyle=":", linewidth=1.0)
    ax.text(
        0.025,
        0.95,
        (
            rf"dynamics: $c_1={dyn_summary['converted_coefficient']:.5f}"
            rf"\pm{dyn_summary['converted_covariance_SEM']:.5f}$" "\n"
            rf"flattened: $c_1={3.0 * flat_summary['slope']:.5f}$"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=6.0,
    )
    size_legend = ax.legend(
        size_handles,
        [rf"$N_y={ny}$" for ny in NY_VALUES],
        ncol=2,
        loc="lower right",
        columnspacing=0.6,
        handletextpad=0.25,
    )
    ax.add_artist(size_legend)
    method_handles = [
        Line2D([], [], color="black", marker="o", markerfacecolor="white", linestyle="none", label="dynamics"),
        Line2D([], [], color="#6F4C9B", marker="o", linestyle="-", markersize=2.5, label="flattened"),
    ]
    ax.legend(handles=method_handles, loc="upper right", borderaxespad=0.5)
    ax.set(
        xlim=(-3.05, 0.05),
        xlabel=r"$\log[\sin(\pi A_y/N_y)/\sin(\pi A_y^\star/N_y)]$",
        ylabel=r"$\Delta S_1$",
    )
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.17, top=0.975)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "dynamics_vs_flattened_full_entropy_collapse.pdf")
    fig.savefig(FIGURE_DIR / "dynamics_vs_flattened_full_entropy_collapse.png", dpi=300)
    plt.close(fig)


def make_wall_collapse(
    dynamics: dict[str, Any],
    flat_fits: dict[str, tuple[dict[int, dict[str, Any]], dict[str, float]]],
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(3.375, 4.85), sharex=True)
    for panel, (axis, component) in enumerate(zip(axes, ("left", "right"))):
        dyn_by = dynamics[f"{component}_by_size"]
        dyn_summary = dynamics[f"{component}_joint"]
        flat_by, flat_summary = flat_fits[component]
        fit_min = min(float(item["x_fit"].min()) for item in dyn_by.values())
        axis.axvspan(fit_min, 0.0, color="0.5", alpha=0.15, linewidth=0, zorder=0)
        handles: list[Any] = []
        for ny in WALL_DYNAMIC_NY:
            style = SIZE_STYLES[ny]
            dyn = dyn_by[ny]
            empirical = axis.errorbar(
                dyn["x_all"], dyn["mean_all"], yerr=dyn["sem_all"],
                color=style["color"], marker=style["marker"], linestyle="none",
                markerfacecolor="white", markeredgewidth=0.75, markersize=3.2,
                elinewidth=0.4, capsize=0.0, zorder=3,
            )
            if panel == 0:
                handles.append(empirical[0])
            flat = flat_by[ny]
            axis.plot(
                flat["x_all"], flat["y_all"], color=style["color"],
                marker=style["marker"], linestyle="-", linewidth=0.45,
                markersize=2.0, alpha=0.75, zorder=2,
            )
        xline = np.linspace(-3.05, 0.0, 300)
        axis.plot(xline, dyn_summary["slope"] * xline, "k--", linewidth=0.9)
        axis.plot(xline, flat_summary["slope"] * xline, color="#6F4C9B", linestyle=":", linewidth=1.0)
        side = "L" if component == "left" else "R"
        axis.text(
            0.025, 0.94,
            (
                rf"{component} wall" "\n"
                rf"dynamics: $3m_{{\rm {side}}}={3.0 * dyn_summary['slope']:.5f}"
                rf"\pm{3.0 * dyn_summary['slope_covariance_sem']:.5f}$" "\n"
                rf"flattened: $3m_{{\rm {side}}}={3.0 * flat_summary['slope']:.5f}$"
            ),
            transform=axis.transAxes, va="top", fontsize=5.8,
        )
        axis.text(-0.18, 1.04, f"({chr(ord('a') + panel)})", transform=axis.transAxes, fontweight="bold")
        axis.set_xlim(-3.05, 0.05)
        axis.set_ylabel(rf"$\Delta S_{{\rm {side}}}^{{\rm wall}}$")
    axes[0].legend(
        handles, [rf"$N_y={ny}$" for ny in WALL_DYNAMIC_NY],
        ncol=2, loc="lower right", columnspacing=0.7, handletextpad=0.3,
    )
    axes[1].legend(
        handles=[
            Line2D([], [], color="black", marker="o", markerfacecolor="white", linestyle="none", label="dynamics"),
            Line2D([], [], color="#6F4C9B", marker="o", linestyle="-", markersize=2.5, label="flattened"),
        ],
        loc="lower right",
    )
    axes[1].set_xlabel(r"$\log[\sin(\pi A_y/N_y)/\sin(\pi A_y^\star/N_y)]$")
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.105, top=0.965, hspace=0.16)
    fig.savefig(FIGURE_DIR / "dynamics_vs_flattened_wall_entropy_collapse_2x1.pdf")
    fig.savefig(FIGURE_DIR / "dynamics_vs_flattened_wall_entropy_collapse_2x1.png", dpi=300)
    plt.close(fig)


def make_size_coefficients(
    rows: list[dict[str, Any]],
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(3.375, 4.35), sharex=True)
    def selected(observable: str, method: str) -> list[dict[str, Any]]:
        return [row for row in rows if row["observable"] == observable and row["method"] == method]

    dynamic = selected("full_von_neumann", "trajectory_ensemble_mean")
    flat = selected("full_von_neumann", "flattened_parent_ground_state")
    axes[0].axhline(1.0, color="0.3", linestyle=":", linewidth=0.8)
    axes[0].errorbar(
        [row["Ny"] for row in dynamic], [row["converted_coefficient"] for row in dynamic],
        yerr=[row["converted_SEM"] for row in dynamic], color="#D92725", marker="o",
        markerfacecolor="white", linestyle="--", capsize=2.0, label="dynamics",
    )
    axes[0].plot(
        [row["Ny"] for row in flat], [row["converted_coefficient"] for row in flat],
        color="#1F77B4", marker="s", linestyle="--", label="flattened",
    )
    axes[0].set_ylabel(r"$c_1=3m$")
    axes[0].legend(loc="best")
    axes[0].text(-0.16, 1.04, "(a)", transform=axes[0].transAxes, fontweight="bold")

    axes[1].axhline(0.5, color="0.3", linestyle=":", linewidth=0.8)
    styles = {
        ("left", "trajectory_ensemble_mean"): ("#D92725", "o", "left dynamics", True),
        ("right", "trajectory_ensemble_mean"): ("#F08050", "^", "right dynamics", True),
        ("left", "flattened_parent_ground_state"): ("#1F77B4", "s", "left flattened", False),
        ("right", "flattened_parent_ground_state"): ("#2CA02C", "D", "right flattened", False),
    }
    for (component, method), (color, marker, label, with_error) in styles.items():
        subset = selected(f"{component}_three_cell_wall", method)
        x = [row["Ny"] for row in subset]
        y = [row["converted_coefficient"] for row in subset]
        if with_error:
            axes[1].errorbar(
                x, y, yerr=[row["converted_SEM"] for row in subset], color=color,
                marker=marker, markerfacecolor="white", linestyle="--", capsize=2.0,
                label=label,
            )
        else:
            axes[1].plot(x, y, color=color, marker=marker, linestyle="--", label=label)
    axes[1].set(xlabel=r"$N_y$", ylabel=r"wall entropy weight $3m_w$", xticks=NY_VALUES)
    axes[1].legend(ncol=2, loc="best", columnspacing=0.6, handletextpad=0.3)
    axes[1].text(-0.16, 1.04, "(b)", transform=axes[1].transAxes, fontweight="bold")
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.10, top=0.97, hspace=0.14)
    fig.savefig(FIGURE_DIR / "dynamics_vs_flattened_entropy_coefficients_2x1.pdf")
    fig.savefig(FIGURE_DIR / "dynamics_vs_flattened_entropy_coefficients_2x1.png", dpi=300)
    plt.close(fig)


def joint_rows(
    dynamics: dict[str, Any],
    flat_fits: dict[str, tuple[dict[int, dict[str, Any]], dict[str, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specifications = (
        ("full_von_neumann", "full", 3.0, dynamics["full_joint"], NY_VALUES),
        ("left_three_cell_wall", "left", 3.0, dynamics["left_joint"], WALL_DYNAMIC_NY),
        ("right_three_cell_wall", "right", 3.0, dynamics["right_joint"], WALL_DYNAMIC_NY),
    )
    for observable, key, factor, dynamic, sizes in specifications:
        rows.append(
            {
                "observable": observable,
                "method": "trajectory_ensemble_mean",
                "Ny_values": ",".join(map(str, sizes)),
                "slope": dynamic["slope"],
                "slope_SEM": dynamic.get("slope_covariance_SEM", dynamic.get("slope_covariance_sem")),
                "converted_coefficient": factor * dynamic["slope"],
                "converted_SEM": factor * dynamic.get("slope_covariance_SEM", dynamic.get("slope_covariance_sem")),
                "R0_squared": dynamic["R0_squared"],
            }
        )
        flat = flat_fits[key][1]
        rows.append(
            {
                "observable": observable,
                "method": "flattened_parent_ground_state",
                "Ny_values": ",".join(map(str, sizes)),
                "slope": flat["slope"],
                "slope_SEM": "",
                "converted_coefficient": factor * flat["slope"],
                "converted_SEM": "",
                "R0_squared": flat["R0_squared"],
            }
        )
    return rows


def main() -> int:
    endpoint_fig = load_module("_flat_benchmark_endpoint", ENDPOINT_REVIEW)
    wall_base = load_module("_flat_benchmark_wall", WALL_REVIEW)
    flat = load_module("_flat_benchmark_runner", FLAT_RUNNER)

    print("[stage] verifying and loading dynamical campaigns", flush=True)
    dynamics = load_dynamics(endpoint_fig, wall_base)
    print("[stage] computing/loading flattened ground states", flush=True)
    flattened = load_or_compute_flattened(flat)

    flat_curves = {
        "full": {ny: flattened[ny]["entropy_von_neumann"] for ny in NY_VALUES},
        "left": {ny: flattened[ny]["left_wall_entropy"] for ny in NY_VALUES},
        "right": {ny: flattened[ny]["right_wall_entropy"] for ny in NY_VALUES},
    }
    flat_fits = {
        "full": deterministic_anchored_fit(flat_curves["full"], NY_VALUES),
        "left": deterministic_anchored_fit(flat_curves["left"], WALL_DYNAMIC_NY),
        "right": deterministic_anchored_fit(flat_curves["right"], WALL_DYNAMIC_NY),
    }
    flat_size_fits = {
        "full": flat_fits["full"],
        "left": deterministic_anchored_fit(flat_curves["left"], NY_VALUES),
        "right": deterministic_anchored_fit(flat_curves["right"], NY_VALUES),
    }

    coefficients = coefficient_rows(dynamics, flat_size_fits)
    joints = joint_rows(dynamics, flat_fits)
    write_csv(DATA_DIR / "flattened_entropy_wall_curves.csv", curve_rows(flattened))
    write_csv(DATA_DIR / "dynamics_vs_flattened_size_coefficients.csv", coefficients)
    write_csv(DATA_DIR / "dynamics_vs_flattened_joint_fits.csv", joints)

    make_full_collapse(dynamics, flat_fits["full"])
    make_wall_collapse(dynamics, flat_fits)
    make_size_coefficients(coefficients)

    outputs = sorted(DATA_DIR.glob("*.csv")) + sorted(FIGURE_DIR.glob("*.pdf")) + sorted(FIGURE_DIR.glob("*.png"))
    cache_products = sorted(CACHE_DIR.glob("*"))
    manifest = {
        "schema": "hard_wall_dynamics_vs_flattened_entropy_benchmark_v1",
        "scientific_contract": {
            "Nx": NX,
            "Ny_values": list(NY_VALUES),
            "wall_contour_dynamic_Ny_values": list(WALL_DYNAMIC_NY),
            "nshell": 1,
            "alpha_1": 1.0,
            "alpha_2": 30.0,
            "wall": "hard_support_truncated",
            "filling": 0.5,
            "flattened_parent": "sum_R(P_A+ + P_B+ - P_A- - P_B-)",
            "fit_Ay": "8..Ny//2",
            "anchor_Ay": "Ny//2",
            "cross_size_weighting": "equal total weight per Ny",
            "wall_windows": {"left": LEFT_X.tolist(), "right": RIGHT_X.tolist()},
        },
        "verified_dynamic_inputs": {
            "endpoint_v2_shards": 140,
            "endpoint_v2_trajectories_per_Ny": 100,
            "all_Ay_v3_shards": dynamics["wall_input_pairs"],
            "all_Ay_v3_trajectories_per_Ny": 100,
            "all_Ay_v3_contour_closure_max_abs": dynamics["contour_closure"],
        },
        "flattened_reference": {
            "deterministic": True,
            "sampling_SEM": None,
            "cache_products": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in cache_products
            },
        },
        "joint_fits": joints,
        "outputs": {
            path.relative_to(HERE).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in outputs
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"joint_fits": joints}, indent=2), flush=True)
    for path in outputs:
        print(f"[saved] {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
