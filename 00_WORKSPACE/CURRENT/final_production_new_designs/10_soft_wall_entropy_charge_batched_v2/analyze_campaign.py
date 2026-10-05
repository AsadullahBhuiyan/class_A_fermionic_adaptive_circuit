#!/usr/bin/env python3
"""Analyze the verified endpoint-only soft-wall entropy/charge campaign v2."""

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
SAMPLING_REVISION = (
    "soft_wall_entropy_charge_nx20_ny30-60_s100_2ny_raster_v2_batched_endpoint"
)
BUNDLE_NAME = "10_soft_wall_entropy_charge_batched_v2"
RESULT_SCHEMA = "soft_wall_entropy_charge_result_shard_v2"
COMPLETION_SCHEMA = "soft_wall_entropy_charge_completion_v2"
OBSERVER_SCHEMA = "soft_wall_entropy_charge_endpoint_observer_v2"
CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
SOURCE_FILES = (
    "run_campaign.py",
    "entropy_charge_observer.py",
    "src/classA_U1FGTN_gpu.py",
    "src/occupied_frame_gpu.py",
)
EXECUTION_BATCH_SIZE = {30: 80, 35: 60, 40: 40, 45: 30, 50: 25, 55: 20, 60: 20}
ROOT_SEED = 2026090305
WALL_WINDOWS = {"left": (4, 5, 6), "right": (14, 15, 16)}
CURVE_KEYS = {
    "c1": "endpoint__entropy_von_neumann",
    "c2": "endpoint__entropy_renyi2",
    "c3": "endpoint__entropy_renyi3",
    "k": "endpoint__charge_variance",
}
CONTOUR_KEYS = {
    "c1": "fixed__contour_von_neumann",
    "c2": "fixed__contour_renyi2",
    "c3": "fixed__contour_renyi3",
    "k": "fixed__contour_charge_variance",
}
FIXED_SCALAR_KEYS = {
    "c1": "fixed_scalar__entropy_von_neumann",
    "c2": "fixed_scalar__entropy_renyi2",
    "c3": "fixed_scalar__entropy_renyi3",
    "k": "fixed_scalar__charge_variance",
}
PREFACTOR = {"c1": 3.0, "c2": 4.0, "c3": 4.5, "k": math.pi**2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_runner() -> Any:
    name = "_soft_wall_entropy_charge_v2_analysis_runner"
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
    lane = "A" if ny in (40, 60) else "B"
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
        raise RuntimeError(f"analysis requires exactly 140 verified shards; found {len(found)}")
    return sorted(found, key=lambda item: str(item[0]))


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
            "Ny": int(receipt["Ny"]),
            "cycles_total": 2 * int(receipt["Ny"]),
            "endpoint_only": True,
            "shard_index": int(receipt["shard_index"]),
            "sample_start": int(receipt["sample_start"]),
            "sample_stop": int(receipt["sample_stop"]),
            "execution_batch_seed": int(receipt["batch_seed"]),
        }
        for key, value in scalar_identity.items():
            if key not in payload or payload[key].item() != value:
                raise RuntimeError(f"result identity mismatch {key}: {path}")
        ny = int(receipt["Ny"])
        if not np.array_equal(payload["sample_ids"], receipt["global_sample_indices"]):
            raise RuntimeError(f"sample identity mismatch: {path}")
        if not np.array_equal(payload["ay_values"], np.arange(ny // 2 + 1)):
            raise RuntimeError(f"Ay axis mismatch: {path}")
        if not np.array_equal(payload["cycles"], np.arange(2 * ny + 1)):
            raise RuntimeError(f"cycle axis mismatch: {path}")
        rows[ny].append(payload)
    cases: dict[int, dict[str, np.ndarray]] = {}
    trajectory_keys = (
        "global_charge",
        "half_filling_offset",
        *CURVE_KEYS.values(),
        "endpoint__charge_mean",
        *CONTOUR_KEYS.values(),
        *FIXED_SCALAR_KEYS.values(),
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
        }
        for key in trajectory_keys:
            values = np.concatenate([row[key] for row in shards], axis=0)[order]
            if not np.isfinite(values).all():
                raise FloatingPointError(f"nonfinite {key} at Ny={ny}")
            case[key] = values
        for key in CURVE_KEYS.values():
            if case[key].shape != (100, ny // 2 + 1) or np.any(case[key] < -1e-9):
                raise RuntimeError(f"invalid endpoint curve {key} at Ny={ny}")
            if not np.all(case[key][:, 0] == 0.0):
                raise RuntimeError(f"Ay=0 is not exact zero for {key} at Ny={ny}")
        for key in CONTOUR_KEYS.values():
            if case[key].shape != (100, 20, ny // 2) or np.any(case[key] < -1e-9):
                raise RuntimeError(f"invalid fixed contour {key} at Ny={ny}")
        cases[ny] = case
    return cases


def log_chord(ay: np.ndarray, ny: int) -> np.ndarray:
    return np.log((ny / math.pi) * np.sin(math.pi * ay / ny))


def trajectory_slopes(ay: np.ndarray, curves: np.ndarray, ny: int) -> tuple[np.ndarray, np.ndarray]:
    selected = np.flatnonzero((ay >= 8) & (ay <= ny // 2))
    if selected.size < 2:
        raise RuntimeError(f"Ny={ny} has too few locked fit points")
    outputs = []
    for indices in (selected, selected[1:-1] if selected.size >= 4 else selected):
        x = log_chord(ay[indices], ny)
        centered = x - x.mean()
        outputs.append((curves[:, indices] @ centered) / float(centered @ centered))
    return outputs[0], outputs[1]


def sem_summary(cases: Mapping[int, Mapping[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ny in EXPECTED_NY:
        case = cases[ny]
        slopes: dict[str, np.ndarray] = {}
        trimmed: dict[str, np.ndarray] = {}
        for label, key in CURVE_KEYS.items():
            slopes[label], trimmed[label] = trajectory_slopes(case["ay_values"], case[key], ny)
        row: dict[str, Any] = {"Ny": ny, "samples": EXPECTED_SAMPLES}
        for label in ("c1", "c2", "c3", "k"):
            resolved = PREFACTOR[label] * slopes[label]
            resolved_trimmed = PREFACTOR[label] * trimmed[label]
            row.update(
                {
                    label: float(resolved.mean()),
                    f"{label}_sample_sd": float(resolved.std(ddof=1)),
                    f"{label}_sem": float(resolved.std(ddof=1) / np.sqrt(EXPECTED_SAMPLES)),
                    f"{label}_trimmed": float(resolved_trimmed.mean()),
                    f"{label}_trimmed_sample_sd": float(resolved_trimmed.std(ddof=1)),
                    f"{label}_trimmed_sem": float(
                        resolved_trimmed.std(ddof=1) / np.sqrt(EXPECTED_SAMPLES)
                    ),
                }
            )
        for label in ("c1", "c2", "c3"):
            delta = PREFACTOR[label] * slopes[label] - PREFACTOR["k"] * slopes["k"]
            row[f"{label}_minus_k"] = float(delta.mean())
            row[f"{label}_minus_k_sample_sd"] = float(delta.std(ddof=1))
            row[f"{label}_minus_k_sem"] = float(
                delta.std(ddof=1) / np.sqrt(EXPECTED_SAMPLES)
            )
        rows.append(row)
    return rows


def wall_rows(cases: Mapping[int, Mapping[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ny in EXPECTED_NY:
        case = cases[ny]
        for label, key in CONTOUR_KEYS.items():
            contour = case[key]
            total = contour.sum(axis=(1, 2))
            left = contour[:, WALL_WINDOWS["left"], :].sum(axis=(1, 2))
            right = contour[:, WALL_WINDOWS["right"], :].sum(axis=(1, 2))
            scalar = case[FIXED_SCALAR_KEYS[label]]
            if np.max(np.abs(total - scalar)) > 2e-8:
                raise RuntimeError(f"fixed contour closure failed for {label}, Ny={ny}")
            left_fraction = np.divide(left, total, out=np.zeros_like(left), where=total > 0)
            right_fraction = np.divide(right, total, out=np.zeros_like(right), where=total > 0)
            rows.append(
                {
                    "Ny": ny,
                    "observable": label,
                    "left_fraction_mean": float(left_fraction.mean()),
                    "right_fraction_mean": float(right_fraction.mean()),
                    "left_minus_right_mean": float((left_fraction - right_fraction).mean()),
                    "leakage_fraction_mean": float((1.0 - left_fraction - right_fraction).mean()),
                    "closure_max_abs": float(np.max(np.abs(total - scalar))),
                }
            )
    return rows


def charge_rows(cases: Mapping[int, Mapping[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows = []
    for ny in EXPECTED_NY:
        offsets = cases[ny]["half_filling_offset"]
        for cycle in range(2 * ny + 1):
            rows.append(
                {
                    "Ny": ny,
                    "cycle": cycle,
                    "mean_offset": float(offsets[:, cycle].mean()),
                    "std_offset": float(offsets[:, cycle].std(ddof=1)),
                    "mean_absolute_offset": float(np.abs(offsets[:, cycle]).mean()),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, root: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(root / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(root / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def make_figures(
    cases: Mapping[int, Mapping[str, np.ndarray]],
    scaling: list[dict[str, Any]],
    walls: list[dict[str, Any]],
    analysis_root: Path,
) -> None:
    ny = np.asarray(EXPECTED_NY)
    fig, axis = plt.subplots(figsize=(7.0, 3.8))
    for label, marker in zip(("c1", "c2", "c3", "k"), ("o", "s", "^", "D")):
        mean = np.asarray([row[label] for row in scaling])
        sem = np.asarray([row[f"{label}_sem"] for row in scaling])
        axis.errorbar(ny, mean, yerr=sem, marker=marker, label=label)
    axis.axhline(1.0, color="k", ls="--", lw=1)
    axis.set(xlabel=r"$N_y$", ylabel="endpoint prefactor")
    axis.legend(ncol=4)
    save_figure(fig, analysis_root, "endpoint_c1_c2_c3_k")

    fig, axis = plt.subplots(figsize=(7.0, 3.8))
    for label, marker in zip(("c1", "c2", "c3"), ("o", "s", "^")):
        key = f"{label}_minus_k"
        mean = np.asarray([row[key] for row in scaling])
        sem = np.asarray([row[f"{key}_sem"] for row in scaling])
        axis.errorbar(ny, mean, yerr=sem, marker=marker, label=key)
    axis.axhline(0.0, color="k", ls="--", lw=1)
    axis.set(xlabel=r"$N_y$", ylabel=r"$c_q-k$")
    axis.legend()
    save_figure(fig, analysis_root, "paired_cq_minus_k")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for sample_ny in (30, 60):
        case = cases[sample_ny]
        x = log_chord(case["ay_values"][1:], sample_ny)
        axes[0].plot(x, case[CURVE_KEYS["c1"]][:, 1:].mean(0), label=f"Ny={sample_ny}")
        axes[1].plot(x, case[CURVE_KEYS["k"]][:, 1:].mean(0), label=f"Ny={sample_ny}")
    axes[0].set(xlabel="log chord", ylabel=r"$S_1$")
    axes[1].set(xlabel="log chord", ylabel=r"$F^q$")
    axes[0].legend(); axes[1].legend()
    save_figure(fig, analysis_root, "representative_endpoint_scaling")

    fig, axes = plt.subplots(len(EXPECTED_NY), 4, figsize=(7.1, 12.0), sharex=True)
    for column, label in enumerate(("c1", "c2", "c3", "k")):
        for row_index, sample_ny in enumerate(EXPECTED_NY):
            image = cases[sample_ny][CONTOUR_KEYS[label]].mean(0)
            axes[row_index, column].imshow(image, aspect="auto", origin="lower")
            axes[row_index, column].set_title(f"{label}, Ny={sample_ny}", fontsize=8)
    save_figure(fig, analysis_root, "fixed_half_strip_contours")

    fig, axis = plt.subplots(figsize=(7.0, 3.8))
    for sample_ny in EXPECTED_NY:
        offsets = cases[sample_ny]["half_filling_offset"]
        axis.plot(cases[sample_ny]["cycles"] / sample_ny, np.abs(offsets).mean(0), label=str(sample_ny))
    axis.set(xlabel=r"cycle/$N_y$", ylabel=r"mean $|Q-N_xN_y|$")
    axis.legend(title=r"$N_y$", ncol=4)
    save_figure(fig, analysis_root, "global_charge_wandering")

    # Compact wall-fraction comparison.
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.6), sharey=True)
    for axis, label in zip(axes, ("c1", "c2", "c3", "k")):
        selected = [row for row in walls if row["observable"] == label]
        axis.plot(ny, [row["left_fraction_mean"] for row in selected], "o-", label="left")
        axis.plot(ny, [row["right_fraction_mean"] for row in selected], "s-", label="right")
        axis.set_title(label)
        axis.set_xlabel(r"$N_y$")
    axes[0].set_ylabel("fixed-half-strip fraction")
    axes[0].legend(fontsize=7)
    save_figure(fig, analysis_root, "fixed_half_strip_wall_fractions")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_cases(discover(args.output_root))
    scaling = sem_summary(cases)
    walls = wall_rows(cases)
    charges = charge_rows(cases)
    args.analysis_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.analysis_root / "endpoint_prefactors.csv", scaling)
    write_csv(
        args.analysis_root / "paired_cq_minus_k.csv",
        [
            {
                "Ny": row["Ny"],
                **{
                    key: value
                    for key, value in row.items()
                    if "minus_k" in key
                },
            }
            for row in scaling
        ],
    )
    write_csv(args.analysis_root / "fixed_half_strip_wall_fractions.csv", walls)
    write_csv(args.analysis_root / "global_charge_wandering.csv", charges)
    make_figures(cases, scaling, walls, args.analysis_root)
    summary = {
        "schema": "soft_wall_entropy_charge_analysis_v2",
        "sampling_revision": SAMPLING_REVISION,
        "verified_shards": EXPECTED_SHARDS,
        "samples_per_Ny": EXPECTED_SAMPLES,
        "Ny_values": list(EXPECTED_NY),
        "uncertainty": "sample-wise standard error SD/sqrt(100)",
        "independent_unit": "one complete trajectory",
        "fit_Ay": "8..Ny//2",
        "sensitivity": "exclude both fit-window endpoints",
        "wall_windows": {key: list(value) for key, value in WALL_WINDOWS.items()},
        "time_dependent_entropy_products": False,
    }
    (args.analysis_root / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("[analysis complete] " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
