from __future__ import annotations

import argparse
import csv
import io
import json
import math
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from campaign_cases import expand_cases
from production_runtime import (
    PRODUCTION_SAMPLES,
    SHARD_SIZE,
    sha256_file,
    verify_archive_receipt,
    write_json_atomic,
)


ANALYSIS_SCHEMA = "s2_two_construction_entanglement_transition_v1"
CONSTRUCTIONS = ("explicit_interface", "support_terminated")


def chord_coordinate(ny: int, ay: np.ndarray) -> np.ndarray:
    ay = np.asarray(ay, dtype=np.float64)
    return np.log(float(ny) / math.pi * np.sin(math.pi * ay / float(ny)))


def _aicc(residuals: np.ndarray, parameters: int) -> float:
    residuals = np.asarray(residuals, dtype=np.float64)
    n = int(residuals.size)
    k = int(parameters)
    if n <= k + 1:
        return float("inf")
    rss = max(float(np.dot(residuals, residuals)), np.finfo(float).tiny)
    aic = n * math.log(rss / n) + 2 * k
    return aic + 2 * k * (k + 1) / (n - k - 1)


def fit_entropy_curve(curve: np.ndarray, ny: int) -> dict[str, float]:
    """Fit one trajectory to constant and log-chord models on the locked A_y window."""
    curve = np.asarray(curve, dtype=np.float64)
    ay = np.arange(2, int(ny) // 2, dtype=np.int64)
    if curve.ndim != 1 or curve.size <= int(ay[-1]):
        raise ValueError(f"entropy curve does not cover the locked Ny={ny} fit window")
    y = curve[ay]
    if not np.isfinite(y).all():
        raise FloatingPointError("entropy fit input contains non-finite values")
    x = chord_coordinate(ny, ay)
    design = np.column_stack((np.ones_like(x), x))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    log_residual = y - design @ np.asarray([intercept, slope])
    area_intercept = float(np.mean(y))
    area_residual = y - area_intercept
    aicc_log = _aicc(log_residual, 2)
    aicc_area = _aicc(area_residual, 1)
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "c_eff": float(3.0 * slope),
        "area_intercept": area_intercept,
        "aicc_log": aicc_log,
        "aicc_area": aicc_area,
        "delta_aicc": float(aicc_area - aicc_log),
        "fit_points": int(y.size),
        "rss_log": float(np.dot(log_residual, log_residual)),
        "rss_area": float(np.dot(area_residual, area_residual)),
    }


def bootstrap_interval(
    values: np.ndarray, *, resamples: int = 2000, seed: int = 2026081924
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be one finite nonempty vector")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.size, size=(int(resamples), values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def analyze_case_arrays(
    entropy_by_cycle: dict[int, np.ndarray],
    *,
    ny: int,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 2026081924,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required = (int(ny), 3 * int(ny) // 2, 2 * int(ny))
    missing = set(required) - set(entropy_by_cycle)
    if missing:
        raise ValueError(f"missing entropy checkpoints {sorted(missing)}")
    samples = {np.asarray(entropy_by_cycle[c]).shape[0] for c in required}
    if len(samples) != 1:
        raise ValueError("entropy checkpoints have inconsistent trajectory counts")
    per_trajectory: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for cycle in required:
        curves = np.asarray(entropy_by_cycle[cycle], dtype=np.float64)
        fits = [fit_entropy_curve(curve, ny) for curve in curves]
        for sample_index, fit in enumerate(fits):
            per_trajectory.append(
                {"checkpoint_cycle": cycle, "sample_index": sample_index, **fit}
            )
        row: dict[str, Any] = {
            "checkpoint_cycle": cycle,
            "actual_samples": int(curves.shape[0]),
        }
        for offset, field in enumerate(("c_eff", "delta_aicc")):
            mean, low, high = bootstrap_interval(
                np.asarray([fit[field] for fit in fits]),
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + cycle * 17 + offset,
            )
            row.update({field: mean, f"{field}_ci_low": low, f"{field}_ci_high": high})
        summaries.append(row)
    return per_trajectory, summaries


@dataclass(frozen=True)
class ArchiveInput:
    path: Path
    sha256: str
    priority: int
    manifest: dict[str, Any]


def _manifest_from_archive(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "r:gz") as archive:
        members = [m for m in archive.getmembers() if m.name.lstrip("./") == "manifest.json"]
        if len(members) != 1:
            raise RuntimeError(f"{path}: expected exactly one root manifest")
        handle = archive.extractfile(members[0])
        if handle is None:
            raise RuntimeError(f"{path}: root manifest is unreadable")
        return json.loads(handle.read().decode("utf-8"))


def _npz_from_archive(path: Path) -> dict[str, np.ndarray]:
    with tarfile.open(path, "r:gz") as archive:
        members = [
            m for m in archive.getmembers()
            if m.isfile() and m.name.endswith("/selected_observables.npz")
        ]
        if len(members) != 1:
            raise RuntimeError(f"{path}: expected exactly one selected_observables.npz")
        handle = archive.extractfile(members[0])
        if handle is None:
            raise RuntimeError(f"{path}: selected observables are unreadable")
        with np.load(io.BytesIO(handle.read()), allow_pickle=False) as loaded:
            return {key: loaded[key] for key in loaded.files}


def discover_inputs(roots: Iterable[Path]) -> list[ArchiveInput]:
    rows: list[ArchiveInput] = []
    for priority, root in enumerate(roots):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.tar.gz")):
            receipt = verify_archive_receipt(path)
            rows.append(
                ArchiveInput(
                    path=path,
                    sha256=str(receipt["archive_sha256"]),
                    priority=priority,
                    manifest=_manifest_from_archive(path),
                )
            )
    return rows


def _case_without_samples(case: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(case, sort_keys=True))
    value.get("run", {}).pop("samples", None)
    return value


def select_case_archives(
    inputs: Iterable[ArchiveInput], expected_case: dict[str, Any]
) -> list[ArchiveInput]:
    by_shard: dict[int, list[ArchiveInput]] = {}
    expected = _case_without_samples(expected_case)
    for row in inputs:
        manifest = row.manifest
        case = manifest.get("run_config", {}).get("case", {})
        if manifest.get("status") != "complete_local":
            continue
        if _case_without_samples(case) != expected:
            continue
        shard = int(manifest.get("shard_index", -1))
        if shard < 0:
            continue
        by_shard.setdefault(shard, []).append(row)
    if not by_shard or 0 not in by_shard:
        raise RuntimeError(f"{expected_case['case_id']}: missing shard zero")
    selected: list[ArchiveInput] = []
    for shard in range(max(by_shard) + 1):
        candidates = by_shard.get(shard)
        if not candidates:
            raise RuntimeError(f"{expected_case['case_id']}: non-consecutive shard set")
        best_priority = min(row.priority for row in candidates)
        best = [row for row in candidates if row.priority == best_priority]
        unique = {row.sha256: row for row in best}
        if len(unique) != 1:
            raise RuntimeError(
                f"{expected_case['case_id']} shard {shard}: conflicting archives at one priority"
            )
        selected.append(next(iter(unique.values())))
    minimum_shards = math.ceil(PRODUCTION_SAMPLES / SHARD_SIZE)
    if len(selected) < minimum_shards:
        raise RuntimeError(
            f"{expected_case['case_id']}: fewer than {minimum_shards} verified shards"
        )
    actual_samples = sum(
        len(row.manifest.get("global_sample_indices", [])) for row in selected
    )
    if actual_samples < PRODUCTION_SAMPLES or actual_samples % SHARD_SIZE:
        raise RuntimeError(
            f"{expected_case['case_id']}: incompatible sample superset {actual_samples}"
        )
    return selected


def _construction(case_id: str) -> str:
    return "support_terminated" if "support_terminated_alpha_wall" in case_id else "explicit_interface"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(summary: list[dict[str, Any]], curves: dict[str, np.ndarray], output: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), constrained_layout=True)
    colors = {"explicit_interface": "#0072B2", "support_terminated": "#D55E00"}
    for key, values in curves.items():
        construction, alpha, ny = key.split("|")
        ay = np.arange(values.shape[1])
        label = f"{construction.replace('_', ' ')}, $\\alpha={alpha}$"
        axes[0].plot(ay[1:], values.mean(axis=0)[1:], color=colors[construction],
                     linestyle="-" if float(alpha) == 1 else "--", label=label)
    axes[0].set(xlabel="$A_y$", ylabel="$S(A_y)$", title="Representative entropy")
    axes[0].legend(fontsize=6, frameon=False)
    finals = [row for row in summary if row["checkpoint_cycle"] == 2 * row["Ny"]]
    maximum_ny = max(row["Ny"] for row in finals)
    full_scan_ny = max(
        {row["Ny"] for row in finals},
        key=lambda ny: sum(row["Ny"] == ny for row in finals),
    )
    for construction in CONSTRUCTIONS:
        for ny in sorted({row["Ny"] for row in finals}):
            rows = sorted(
                (row for row in finals if row["construction"] == construction and row["Ny"] == ny),
                key=lambda row: row["alpha_in"],
            )
            axes[1].plot([r["alpha_in"] for r in rows], [r["c_eff"] for r in rows],
                         marker="o", ms=2.5, lw=0.8, color=colors[construction],
                         alpha=0.35 + 0.65 * ny / maximum_ny, label=f"{construction[:3]}, {ny}")
    axes[1].axhline(1, color="0.6", lw=0.6)
    axes[1].axhline(0, color="0.6", lw=0.6)
    axes[1].set(xlabel="$\\alpha_{\\rm in}$", ylabel="$c_{\\rm eff}$", title="Log-chord coefficient")
    axes[1].legend(fontsize=5.5, frameon=False, ncol=2)
    for construction in CONSTRUCTIONS:
        rows = sorted(
            (row for row in finals if row["construction"] == construction and row["Ny"] == full_scan_ny),
            key=lambda row: row["alpha_in"],
        )
        axes[2].plot([r["alpha_in"] for r in rows], [r["delta_aicc"] for r in rows],
                     marker="o", ms=3, color=colors[construction], label=construction.replace("_", " "))
    axes[2].axhline(0, color="black", lw=0.6)
    axes[2].set(xlabel="$\\alpha_{\\rm in}$", ylabel="$\\Delta$AICc", title="Scaling preference")
    axes[2].legend(fontsize=6, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def run_analysis(
    *, bundle_root: Path, archive_roots: list[Path], output_root: Path
) -> dict[str, Any]:
    config = json.loads((bundle_root / "production_config.json").read_text(encoding="utf-8"))
    expected = [
        case for case in expand_cases(config, accepted_width=20, m3_wall_sigma=[])
        if case["campaign"] == "S2" and case["model"]["init_mode"] == "default"
    ]
    if len(expected) != 48:
        raise RuntimeError(f"expected 48 pure two-construction S2 cases, found {len(expected)}")
    inputs = discover_inputs(archive_roots)
    trajectory_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    representative_curves: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    for case in expected:
        archives = select_case_archives(inputs, case)
        entropy: dict[int, list[np.ndarray]] = {c: [] for c in case["strip_entropy_cycles"]}
        markers: list[np.ndarray] = []
        for archive in archives:
            payload = _npz_from_archive(archive.path)
            for cycle in entropy:
                entropy[cycle].append(payload[f"strip_entropy_cycle_{cycle:04d}"])
            markers.append(payload[f"local_chern_marker_cycle_{2 * case['model']['Ny']:04d}"])
        merged = {cycle: np.concatenate(parts, axis=0) for cycle, parts in entropy.items()}
        marker = np.concatenate(markers, axis=0)
        per, summaries = analyze_case_arrays(
            merged,
            ny=int(case["model"]["Ny"]),
            bootstrap_resamples=int(config["S2_entanglement_analysis"]["bootstrap_resamples"]),
            bootstrap_seed=int(config["S2_entanglement_analysis"]["bootstrap_seed"]),
        )
        construction = _construction(case["case_id"])
        alpha = float(case["model"]["alpha_1"])
        ny = int(case["model"]["Ny"])
        common = {
            "case_id": case["case_id"], "construction": construction,
            "alpha_in": alpha, "Ny": ny, "actual_samples": int(marker.shape[0]),
        }
        trajectory_rows.extend({**common, **row} for row in per)
        x0, x1 = int(case["model"]["Nx"]) // 4 + 2, 3 * int(case["model"]["Nx"]) // 4 - 1
        bulk_marker = marker[:, x0:x1, :].mean(axis=(1, 2))
        marker_mean, marker_low, marker_high = bootstrap_interval(
            bulk_marker,
            resamples=int(config["S2_entanglement_analysis"]["bootstrap_resamples"]),
            seed=int(config["S2_entanglement_analysis"]["bootstrap_seed"]) + ny,
        )
        for row in summaries:
            summary_rows.append({**common, **row, "bulk_chern_marker": marker_mean,
                                 "bulk_chern_marker_ci_low": marker_low,
                                 "bulk_chern_marker_ci_high": marker_high})
        if ny == 40 and alpha in (1.0, 3.0):
            representative_curves[f"{construction}|{alpha:g}|{ny}"] = merged[2 * ny]
        provenance[case["case_id"]] = {
            "input_archives": [str(row.path) for row in archives],
            "input_sha256": [row.sha256 for row in archives],
            "shard_indices": [int(row.manifest["shard_index"]) for row in archives],
            "actual_samples": int(marker.shape[0]),
        }
    _write_csv(output_root / "s2_entanglement_per_trajectory.csv", trajectory_rows)
    _write_csv(output_root / "s2_entanglement_summary.csv", summary_rows)
    _plot(summary_rows, representative_curves, output_root / "s2_two_construction_transition")
    finals = [row for row in summary_rows if row["checkpoint_cycle"] == 2 * row["Ny"] and row["Ny"] == 40]
    checks: dict[str, Any] = {}
    for construction in CONSTRUCTIONS:
        rows = {row["alpha_in"]: row for row in finals if row["construction"] == construction}
        low, high = rows[1.0], rows[3.0]
        checks[construction] = {
            "critical_endpoint": low["c_eff_ci_low"] <= 1.0 <= low["c_eff_ci_high"]
            and low["c_eff_ci_low"] > 0 and low["delta_aicc_ci_low"] > 0,
            "area_endpoint": abs(high["c_eff"]) < 0.1
            and high["c_eff_ci_low"] <= 0 <= high["c_eff_ci_high"]
            and high["delta_aicc"] <= 0,
            "crossover_bracket": rows[1.875]["c_eff"] >= 0.5 >= rows[2.125]["c_eff"],
        }
    valid = all(all(group.values()) for group in checks.values())
    report = {
        "schema": ANALYSIS_SCHEMA,
        "status": "validated_common_transition" if valid else "distinct_or_inconclusive",
        "checks": checks,
        "bootstrap_resamples": int(config["S2_entanglement_analysis"]["bootstrap_resamples"]),
        "provenance": provenance,
        "outputs": {
            "per_trajectory": "s2_entanglement_per_trajectory.csv",
            "summary": "s2_entanglement_summary.csv",
            "figure_pdf": "s2_two_construction_transition.pdf",
            "figure_png": "s2_two_construction_transition.png",
        },
    }
    write_json_atomic(output_root / "s2_entanglement_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze the two-construction S2 entropy scan")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, action="append", required=True,
                        help="repeat in priority order: v2, v1, then unversioned bundle roots")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_analysis(
        bundle_root=args.bundle_root.resolve(),
        archive_roots=[path.resolve() for path in args.archive_root],
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
