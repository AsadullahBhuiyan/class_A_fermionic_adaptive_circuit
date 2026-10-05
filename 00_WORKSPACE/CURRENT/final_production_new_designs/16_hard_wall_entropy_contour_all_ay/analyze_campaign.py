#!/usr/bin/env python3
"""Analyze verified all-Ay hard-wall entropy-contour results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parent
EXPECTED_NY = (30, 35, 40, 45, 50, 55, 60)
EXPECTED_SAMPLES = 100
EXPECTED_SHARDS = 140
SAMPLING_REVISION = "hard_wall_entropy_contour_all_ay_nx20_ny30-60_s100_2ny_raster_v3"
BUNDLE_NAME = "16_hard_wall_entropy_contour_all_ay"
RESULT_SCHEMA = "hard_wall_entropy_contour_all_ay_result_shard_v3"
COMPLETION_SCHEMA = "hard_wall_entropy_contour_all_ay_completion_v3"
OBSERVER_SCHEMA = "hard_wall_entropy_contour_all_ay_observer_v3"
CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
SOURCE_FILES = (
    "run_campaign.py",
    "entropy_contour_observer.py",
    "src/classA_U1FGTN_gpu.py",
    "src/occupied_frame_gpu.py",
)
EXECUTION_BATCH_SIZE = {30: 80, 35: 60, 40: 40, 45: 30, 50: 25, 55: 20, 60: 20}
LANE_NY = {"A": (50, 60), "B": (30, 35, 40, 45, 55)}
ROOT_SEED = 2026091416
WALL_WINDOWS = {"left": (4, 5, 6), "right": (14, 15, 16)}
CURVE_KEYS = {
    "S1": "endpoint__entropy_von_neumann",
    "S2": "endpoint__entropy_renyi2",
    "S3": "endpoint__entropy_renyi3",
    "charge_mean": "endpoint__charge_mean",
    "charge_variance": "endpoint__charge_variance",
}
CONTOUR_KEY = "endpoint__contour_von_neumann_y0avg"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_runner() -> Any:
    name = "_hard_wall_all_ay_contour_analysis_runner"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PACKAGE_ROOT / "run_campaign.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load sibling campaign runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _expected_execution(ny: int, sample_start: int) -> tuple[str, str, int]:
    lane = next(label for label, values in LANE_NY.items() if ny in values)
    size = EXECUTION_BATCH_SIZE[ny]
    index = sample_start // size
    start = index * size
    stop = min(start + size, EXPECTED_SAMPLES)
    task = f"lane-{lane}_Ny{ny:03d}_execution-{index:03d}_samples-{start:03d}-{stop - 1:03d}"
    label = (
        f"{ROOT_SEED}|lane={lane}|Nx=20|Ny={ny}|execution={index}|samples={start}:{stop}"
    )
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    return lane, task, seed & ((1 << 63) - 1)


def discover(output_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    found: list[tuple[Path, dict[str, Any]]] = []
    for receipt_path in output_root.glob("results/lane_*/Ny*/*.complete.json"):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema") != COMPLETION_SCHEMA:
            continue
        ny = int(receipt.get("Ny", -1))
        start = int(receipt.get("sample_start", -1))
        stop = int(receipt.get("sample_stop", -1))
        shard = int(receipt.get("shard_index", -1))
        if ny not in EXPECTED_NY or stop != start + 5 or shard != start // 5:
            raise RuntimeError(f"invalid result identity in {receipt_path}")
        lane, execution, seed = _expected_execution(ny, start)
        expected_ids = list(range(start, stop))
        expected_task = f"Ny{ny:03d}_shard-{shard:03d}_samples-{start:03d}-{stop - 1:03d}"
        identity = {
            "status": "complete",
            "bundle": BUNDLE_NAME,
            "sampling_revision": SAMPLING_REVISION,
            "lane": lane,
            "task_id": expected_task,
            "execution_batch_id": execution,
            "Nx": 20,
            "Ny": ny,
            "cycles": 2 * ny,
            "endpoint_only": True,
            "sample_count": 5,
            "global_sample_indices": expected_ids,
            "batch_seed": seed,
            "canonical_entry_point": CANONICAL_ENTRY_POINT,
            "observer_schema": OBSERVER_SCHEMA,
        }
        for key, value in identity.items():
            if receipt.get(key) != value:
                raise RuntimeError(f"completion identity mismatch {key}: {receipt_path}")
        result = receipt_path.with_name(str(receipt.get("result_filename")))
        if result != receipt_path.with_suffix("").with_suffix(".npz"):
            raise RuntimeError(f"non-sibling result receipt: {receipt_path}")
        if not result.is_file():
            raise RuntimeError(f"missing completed result: {result}")
        if result.stat().st_size != int(receipt.get("result_bytes", -1)):
            raise RuntimeError(f"result byte-count mismatch: {result}")
        if sha256_file(result) != receipt.get("result_sha256"):
            raise RuntimeError(f"result checksum mismatch: {result}")
        found.append((result, receipt))
    if len(found) != EXPECTED_SHARDS:
        raise RuntimeError(
            f"analysis requires exactly {EXPECTED_SHARDS} verified shards; found {len(found)}"
        )
    return sorted(found, key=lambda item: str(item[0]))


def _validate_contours(payload: Mapping[str, np.ndarray], *, ny: int, path: Path) -> None:
    half = ny // 2
    contour = np.asarray(payload[CONTOUR_KEY], dtype=np.float64)
    if contour.shape != (5, half + 1, 20, half):
        raise RuntimeError(f"all-Ay contour shape mismatch: {path}")
    if not np.isfinite(contour).all() or np.min(contour) < -1.0e-9:
        raise FloatingPointError(f"invalid contour values: {path}")
    valid = np.asarray(payload["valid_dy_count"], dtype=np.int64)
    if not np.array_equal(valid, np.arange(half + 1)):
        raise RuntimeError(f"valid-dy axis mismatch: {path}")
    entropy = np.asarray(payload[CURVE_KEYS["S1"]], dtype=np.float64)
    for width_index, ay in enumerate(valid):
        active = contour[:, width_index, :, :ay]
        padding = contour[:, width_index, :, ay:]
        if padding.size and not np.all(padding == 0.0):
            raise RuntimeError(f"nonzero Ay={ay} contour padding: {path}")
        closure = active.sum(axis=(1, 2)) - entropy[:, width_index]
        if np.max(np.abs(closure), initial=0.0) > 2.0e-8:
            raise RuntimeError(f"Ay={ay} contour closure mismatch: {path}")


def load_cases(results: list[tuple[Path, dict[str, Any]]]) -> dict[int, dict[str, np.ndarray]]:
    runner = _load_runner()
    locked_config_hash = runner.config_hash(runner.expected_config())
    locked_sources = {relative: sha256_file(PACKAGE_ROOT / relative) for relative in SOURCE_FILES}
    rows: dict[int, list[dict[str, np.ndarray]]] = {ny: [] for ny in EXPECTED_NY}
    for path, receipt in results:
        if receipt.get("config_sha256") != locked_config_hash:
            raise RuntimeError(f"result uses another configuration: {path}")
        if receipt.get("source_hashes") != locked_sources:
            raise RuntimeError(f"result uses another source identity: {path}")
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
        ny = int(receipt["Ny"])
        scalar_identity = {
            "schema": RESULT_SCHEMA,
            "observer_schema": OBSERVER_SCHEMA,
            "bundle": BUNDLE_NAME,
            "sampling_revision": SAMPLING_REVISION,
            "canonical_entry_point": CANONICAL_ENTRY_POINT,
            "lane": receipt["lane"],
            "task_id": receipt["task_id"],
            "execution_batch_id": receipt["execution_batch_id"],
            "config_sha256": locked_config_hash,
            "Nx": 20,
            "Ny": ny,
            "cycles_total": 2 * ny,
            "endpoint_only": True,
            "shard_index": int(receipt["shard_index"]),
            "sample_start": int(receipt["sample_start"]),
            "sample_stop": int(receipt["sample_stop"]),
            "execution_batch_seed": int(receipt["batch_seed"]),
        }
        for key, value in scalar_identity.items():
            if key not in payload or payload[key].item() != value:
                raise RuntimeError(f"result identity mismatch {key}: {path}")
        if not np.array_equal(payload["sample_ids"], receipt["global_sample_indices"]):
            raise RuntimeError(f"sample identity mismatch: {path}")
        if not np.array_equal(payload["ay_values"], np.arange(ny // 2 + 1)):
            raise RuntimeError(f"Ay axis mismatch: {path}")
        if not np.array_equal(payload["cycles"], np.arange(2 * ny + 1)):
            raise RuntimeError(f"cycle axis mismatch: {path}")
        if int(payload["origin_average_count"]) != ny:
            raise RuntimeError(f"origin-average count mismatch: {path}")
        if payload["contour_coordinate"].item() != "relative_dy=(y-y0)_mod_Ny":
            raise RuntimeError(f"contour coordinate mismatch: {path}")
        _validate_contours(payload, ny=ny, path=path)
        rows[ny].append(payload)

    cases: dict[int, dict[str, np.ndarray]] = {}
    trajectory_keys = (
        "global_charge",
        "half_filling_offset",
        *CURVE_KEYS.values(),
        CONTOUR_KEY,
    )
    for ny in EXPECTED_NY:
        shards = rows[ny]
        ids = np.concatenate([row["sample_ids"] for row in shards])
        order = np.argsort(ids)
        if not np.array_equal(ids[order], np.arange(EXPECTED_SAMPLES)):
            raise RuntimeError(f"Ny={ny} does not contain sample IDs 0..99 exactly once")
        case: dict[str, np.ndarray] = {
            "sample_ids": ids[order],
            "cycles": shards[0]["cycles"],
            "ay_values": shards[0]["ay_values"],
            "valid_dy_count": shards[0]["valid_dy_count"],
        }
        for key in trajectory_keys:
            values = np.concatenate([row[key] for row in shards], axis=0)[order]
            if not np.isfinite(values).all():
                raise FloatingPointError(f"nonfinite {key} at Ny={ny}")
            case[key] = values
        for key in CURVE_KEYS.values():
            if case[key].shape != (100, ny // 2 + 1) or np.any(case[key] < -1.0e-9):
                raise RuntimeError(f"invalid endpoint curve {key} at Ny={ny}")
            if not np.all(case[key][:, 0] == 0.0):
                raise RuntimeError(f"Ay=0 is not exact zero for {key} at Ny={ny}")
        cases[ny] = case
    return cases


def log_chord(ay: np.ndarray, ny: int) -> np.ndarray:
    return np.log((ny / math.pi) * np.sin(math.pi * ay / ny))


def mean_curve_fit(
    ay: np.ndarray, curves: np.ndarray, ny: int
) -> dict[str, float | np.ndarray]:
    selected = np.flatnonzero((ay >= 8) & (ay <= ny // 2))
    if selected.size < 3:
        raise RuntimeError(f"Ny={ny} has too few locked fit points")
    x = log_chord(ay[selected], ny)
    design = np.column_stack((x, np.ones_like(x)))
    operator = np.linalg.inv(design.T @ design) @ design.T
    sample_curves = curves[:, selected]
    mean = sample_curves.mean(axis=0)
    slope, intercept = operator @ mean
    sample_slopes = sample_curves @ operator[0]
    if not np.isclose(slope, sample_slopes.mean(), atol=2.0e-13, rtol=2.0e-13):
        raise RuntimeError("fit(mean curve) does not equal mean(sample slopes)")
    covariance = np.cov(sample_curves, rowvar=False, ddof=1)
    slope_variance = float(operator[0] @ (covariance / curves.shape[0]) @ operator[0])
    fitted = slope * x + intercept
    residual = mean - fitted
    denominator = float(np.sum((mean - mean.mean()) ** 2))
    r2 = 1.0 - float(residual @ residual) / denominator if denominator > 0 else 1.0
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_sem": math.sqrt(max(slope_variance, 0.0)),
        "r2": float(r2),
        "fit_ay": ay[selected],
        "fit_x": x,
        "fit_mean": mean,
        "fit_sem": sample_curves.std(axis=0, ddof=1) / math.sqrt(curves.shape[0]),
    }


def integrated_curves(case: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    contour = np.asarray(case[CONTOUR_KEY], dtype=np.float64)
    ay_values = np.asarray(case["ay_values"], dtype=np.int64)
    samples = contour.shape[0]
    output = {
        key: np.zeros((samples, len(ay_values)), dtype=np.float64)
        for key in ("left", "right", "wall_average", "left_minus_right", "full", "leakage")
    }
    for width_index, ay in enumerate(ay_values):
        active = contour[:, width_index, :, :ay]
        left = active[:, WALL_WINDOWS["left"], :].sum(axis=(1, 2))
        right = active[:, WALL_WINDOWS["right"], :].sum(axis=(1, 2))
        full = active.sum(axis=(1, 2))
        output["left"][:, width_index] = left
        output["right"][:, width_index] = right
        output["wall_average"][:, width_index] = 0.5 * (left + right)
        output["left_minus_right"][:, width_index] = left - right
        output["full"][:, width_index] = full
        output["leakage"][:, width_index] = full - left - right
    closure = output["full"] - np.asarray(case[CURVE_KEYS["S1"]])
    if np.max(np.abs(closure), initial=0.0) > 2.0e-8:
        raise RuntimeError("integrated full contour does not close to S1")
    return output


def summarize(
    cases: Mapping[int, Mapping[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    curve_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    resolved: dict[int, dict[str, np.ndarray]] = {}
    for ny in EXPECTED_NY:
        case = cases[ny]
        ay_values = np.asarray(case["ay_values"], dtype=np.int64)
        curves = integrated_curves(case)
        resolved[ny] = curves
        for component, values in curves.items():
            means = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1)
            for index, ay in enumerate(ay_values):
                curve_rows.append(
                    {
                        "Ny": ny,
                        "Ay": int(ay),
                        "component": component,
                        "mean": float(means[index]),
                        "sample_sd": float(sd[index]),
                        "sem": float(sd[index] / math.sqrt(EXPECTED_SAMPLES)),
                        "independent_samples": EXPECTED_SAMPLES,
                        "origins_averaged_within_sample": ny,
                    }
                )
        for component in ("left", "right", "wall_average", "left_minus_right", "full"):
            fit = mean_curve_fit(ay_values, curves[component], ny)
            prefactor = 3.0 if component == "full" else 6.0
            target = 1.0 / 3.0 if component == "full" else (0.0 if component == "left_minus_right" else 1.0 / 6.0)
            fit_rows.append(
                {
                    "Ny": ny,
                    "component": component,
                    "Ay_min": 8,
                    "Ay_max": ny // 2,
                    "points": len(fit["fit_ay"]),
                    "slope": fit["slope"],
                    "slope_sem": fit["slope_sem"],
                    "slope_target": target,
                    "converted_prefactor": prefactor,
                    "converted_coefficient": prefactor * float(fit["slope"]),
                    "converted_coefficient_sem": prefactor * float(fit["slope_sem"]),
                    "intercept": fit["intercept"],
                    "r2": fit["r2"],
                    "uncertainty": "full_Ay_covariance_propagated_trajectory_SEM",
                }
            )
    return curve_rows, fit_rows, resolved


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _style_axis(axis: plt.Axes) -> None:
    axis.tick_params(direction="in", top=True, right=True)
    for spine in axis.spines.values():
        spine.set_visible(True)


def save_figure(fig: plt.Figure, root: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(root / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(root / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def make_figures(
    cases: Mapping[int, Mapping[str, np.ndarray]],
    fits: list[dict[str, Any]],
    resolved: Mapping[int, Mapping[str, np.ndarray]],
    analysis_root: Path,
) -> None:
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(EXPECTED_NY)))
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.0), sharey=True)
    for color, ny in zip(colors, EXPECTED_NY):
        ay = np.asarray(cases[ny]["ay_values"])
        selected = ay >= 1
        x = log_chord(ay[selected], ny)
        for axis, component in zip(axes, ("left", "right")):
            values = resolved[ny][component][:, selected]
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / math.sqrt(EXPECTED_SAMPLES)
            axis.plot(x, mean, color=color, lw=1.1, label=fr"$N_y={ny}$")
            axis.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.16, linewidth=0)
            axis.set_xlabel("log chord length")
            _style_axis(axis)
    axes[0].set_ylabel(r"integrated contour $S_{\rm wall}$")
    axes[0].set_title("left wall")
    axes[1].set_title("right wall")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    save_figure(fig, analysis_root, "wall_integrated_entropy_log_chord")

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.0))
    markers = {"left": "^", "right": "s", "wall_average": "o"}
    styles = {"left": ":", "right": "--", "wall_average": "-"}
    plot_colors = {"left": "#d62728", "right": "#2ca02c", "wall_average": "#1f77b4"}
    for component in ("left", "right", "wall_average"):
        rows = [row for row in fits if row["component"] == component]
        axes[0].errorbar(
            [row["Ny"] for row in rows],
            [row["slope"] for row in rows],
            yerr=[row["slope_sem"] for row in rows],
            color=plot_colors[component], marker=markers[component], ls=styles[component],
            capsize=2, lw=1.0, label=component.replace("_", " "),
        )
    axes[0].axhline(1.0 / 6.0, color="0.25", ls="--", lw=0.9, label=r"$1/6$")
    axes[0].set(xlabel=r"$N_y$", ylabel=r"single-wall slope $m$")
    axes[0].legend(frameon=False, fontsize=7)
    for ny, color in zip(EXPECTED_NY, colors):
        difference = resolved[ny]["left_minus_right"]
        leakage = resolved[ny]["leakage"]
        ay = np.asarray(cases[ny]["ay_values"])
        index = int(np.where(ay == ny // 2)[0][0])
        full = resolved[ny]["full"][:, index]
        diff_fraction = np.divide(
            difference[:, index], full, out=np.zeros_like(full), where=full != 0
        )
        leakage_fraction = np.divide(
            leakage[:, index], full, out=np.zeros_like(full), where=full != 0
        )
        axes[1].errorbar(
            ny, diff_fraction.mean(), yerr=diff_fraction.std(ddof=1) / 10,
            color=color, marker="o", capsize=2,
        )
        axes[1].errorbar(
            ny, leakage_fraction.mean(), yerr=leakage_fraction.std(ddof=1) / 10,
            color=color, marker="x", capsize=2,
        )
    axes[1].axhline(0, color="0.25", ls="--", lw=0.9)
    axes[1].set(xlabel=r"$N_y$", ylabel="half-strip entropy fraction")
    axes[1].plot([], [], "ko", label="left-right")
    axes[1].plot([], [], "kx", label="outside windows")
    axes[1].legend(frameon=False, fontsize=7)
    for axis in axes:
        _style_axis(axis)
    save_figure(fig, analysis_root, "single_wall_slopes_and_localization")

    fig, axes = plt.subplots(2, 4, figsize=(7.05, 4.7))
    axes = axes.ravel()
    for axis, ny in zip(axes, EXPECTED_NY):
        half = ny // 2
        contour = cases[ny][CONTOUR_KEY][:, half].mean(axis=0)
        image = axis.imshow(contour, cmap="Blues", origin="lower", aspect="auto")
        axis.set_title(fr"$N_y={ny}$", fontsize=8)
        axis.set_xlabel(r"relative $y$")
        axis.set_ylabel(r"$x$")
        axis.set_xticks(np.arange(-0.5, half, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 20, 1), minor=True)
        axis.grid(which="minor", color="0.5", alpha=0.25, linewidth=0.25)
        axis.tick_params(which="minor", bottom=False, left=False)
        fig.colorbar(image, ax=axis, fraction=0.047, pad=0.02)
        _style_axis(axis)
    axes[-1].axis("off")
    save_figure(fig, analysis_root, "ensemble_mean_half_strip_contours")


def write_contour_statistics(
    path: Path, cases: Mapping[int, Mapping[str, np.ndarray]]
) -> None:
    payload: dict[str, np.ndarray] = {}
    for ny in EXPECTED_NY:
        values = np.asarray(cases[ny][CONTOUR_KEY], dtype=np.float64)
        payload[f"Ny{ny:03d}__mean"] = values.mean(axis=0)
        payload[f"Ny{ny:03d}__sem"] = values.std(axis=0, ddof=1) / math.sqrt(EXPECTED_SAMPLES)
        payload[f"Ny{ny:03d}__valid_dy_count"] = np.asarray(cases[ny]["valid_dy_count"])
    np.savez_compressed(path, **payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_cases(discover(args.output_root))
    curve_rows, fit_rows, resolved = summarize(cases)
    args.analysis_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.analysis_root / "integrated_wall_entropy_curves.csv", curve_rows)
    write_csv(args.analysis_root / "integrated_wall_entropy_fits.csv", fit_rows)
    write_contour_statistics(args.analysis_root / "ensemble_contour_mean_sem.npz", cases)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
    })
    make_figures(cases, fit_rows, resolved, args.analysis_root)
    generated = sorted(path for path in args.analysis_root.iterdir() if path.is_file())
    summary = {
        "schema": "hard_wall_entropy_contour_all_ay_analysis_v3",
        "sampling_revision": SAMPLING_REVISION,
        "verified_shards": EXPECTED_SHARDS,
        "samples_per_Ny": EXPECTED_SAMPLES,
        "Ny_values": list(EXPECTED_NY),
        "subsystem": "[0,Nx)x[y0,y0+Ay), periodic y",
        "origin_average": "all Ny origins within each trajectory in relative-dy coordinates",
        "independent_unit": "one complete trajectory",
        "uncertainty": "ordinary trajectory SEM after the within-trajectory y0 average",
        "fit_Ay": "8..Ny//2",
        "wall_windows": {key: list(value) for key, value in WALL_WINDOWS.items()},
        "wall_slope_target": 1.0 / 6.0,
        "full_strip_slope_target": 1.0 / 3.0,
        "bootstrap": False,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in generated
            if path.name != "analysis_manifest.json"
        },
    }
    (args.analysis_root / "analysis_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("[analysis complete] " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
