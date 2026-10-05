from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm


BUNDLE_NAME = "colab_charge_fluctuations"
DATASET_NAME = "streaming_covariance_observables"
ANALYSIS_NAME = "streaming_covariance_scaling_fits"
ENTROPY_FILENAME = "entropy_y0avg_vs_ay.npz"
CORRELATOR_FILENAME = "xavg_square_correlator_vs_ry.npz"
RUN_SUMMARY_FILENAME = "run_summary.json"
SCALAR_METRICS_FILENAME = "scalar_metrics.csv"
MANIFEST_FILENAME = "analysis_manifest.json"
CSV_FILENAME = "streaming_covariance_scaling_fit_summary.csv"
DEFAULT_CPU_LIST = "40-50"


@dataclass(frozen=True)
class RunInfo:
    campaign_id: str
    run_dir: Path
    config_id: str
    geometry_key: str
    protocol: str
    nx: int
    ny: int
    cycles: int
    samples: int
    entropy_path: Path
    correlator_path: Path
    summary_path: Path | None
    metrics_path: Path | None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_cpu_list(cpu_list: str) -> set[int]:
    cpus: set[int] = set()
    for chunk in cpu_list.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_text, stop_text = token.split("-", 1)
            start = int(start_text)
            stop = int(stop_text)
            if stop < start:
                raise ValueError(f"Invalid CPU range {token!r}: stop is less than start.")
            cpus.update(range(start, stop + 1))
        else:
            cpus.add(int(token))
    if not cpus:
        raise ValueError("CPU list cannot be empty.")
    return cpus


def set_process_affinity(cpu_list: str | None) -> dict[str, Any]:
    if not cpu_list:
        return {"enabled": False, "reason": "disabled"}

    requested = parse_cpu_list(cpu_list)
    if not hasattr(os, "sched_setaffinity"):
        print("[cpu affinity] os.sched_setaffinity is unavailable on this platform; continuing without pinning.")
        return {"enabled": False, "requested_cpus": sorted(requested), "reason": "sched_setaffinity unavailable"}

    available = set(os.sched_getaffinity(0))
    missing = sorted(requested - available)
    if missing:
        raise ValueError(
            f"Requested CPUs {missing} are not available to this process. "
            f"Available CPUs are {sorted(available)}."
        )

    os.sched_setaffinity(0, requested)
    active = sorted(os.sched_getaffinity(0))
    print(f"[cpu affinity] pinned process to CPUs {active}")
    return {"enabled": True, "requested_cpus": sorted(requested), "active_cpus": active}


def resolve_bundle_root(start: Path | str | None = None) -> Path:
    start_path = Path.cwd() if start is None else Path(start)
    candidates: list[Path] = []
    for base in [start_path, *start_path.parents]:
        candidates.extend([base, base / BUNDLE_NAME])

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            resolved = candidate.absolute()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "gpu_data" / DATASET_NAME / "campaigns").exists() and (resolved / "src").exists():
            return resolved

    raise FileNotFoundError(f"Could not locate {BUNDLE_NAME} with src/ and gpu_data/{DATASET_NAME}.")


def default_campaign_id(bundle_root: Path) -> str:
    campaigns_root = bundle_root / "gpu_data" / DATASET_NAME / "campaigns"
    campaigns = sorted(path.name for path in campaigns_root.iterdir() if path.is_dir())
    if len(campaigns) != 1:
        raise ValueError(
            f"Expected exactly one {DATASET_NAME} campaign under {campaigns_root}, found {campaigns}."
        )
    return campaigns[0]


def parse_config_json(npz_payload: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "config_json" not in npz_payload.files:
        return {}
    raw = npz_payload["config_json"]
    if raw.shape != ():
        return {}
    try:
        return json.loads(str(raw.item()))
    except json.JSONDecodeError:
        return {}


def infer_run_info(campaign_id: str, run_dir: Path) -> RunInfo | None:
    entropy_path = run_dir / ENTROPY_FILENAME
    correlator_path = run_dir / CORRELATOR_FILENAME
    if not entropy_path.exists() or not correlator_path.exists():
        return None

    summary_path = run_dir / RUN_SUMMARY_FILENAME
    metrics_path = run_dir / SCALAR_METRICS_FILENAME
    summary = load_json(summary_path) if summary_path.exists() else {}

    with np.load(entropy_path, allow_pickle=False) as entropy_npz:
        entropy_data = entropy_npz["entropy_y0avg_vs_ay"]
        sample_indices = entropy_npz["sample_indices"]
        cycle_labels = entropy_npz["cycle_labels"]
        config = parse_config_json(entropy_npz)

    if entropy_data.ndim != 3:
        raise ValueError(f"{entropy_path}: expected entropy array shape (samples, cycles, ay), got {entropy_data.shape}")

    samples, cycles, _ = entropy_data.shape
    nx = int(summary.get("Nx", config.get("Nx", 0)))
    ny = int(summary.get("Ny", config.get("Ny", 0)))
    if nx <= 0 or ny <= 0:
        raise ValueError(f"{run_dir}: could not infer positive Nx/Ny from run summary or config_json.")
    if int(sample_indices.size) != samples:
        raise ValueError(f"{entropy_path}: sample_indices length does not match entropy samples.")
    if int(cycle_labels.size) != cycles:
        raise ValueError(f"{entropy_path}: cycle_labels length does not match entropy cycles.")

    protocol = str(summary.get("protocol", config.get("protocol", "unknown")))
    geometry_key = str(summary.get("geometry_key", f"N{nx}x{ny}"))
    config_id = str(summary.get("config_id", f"{geometry_key}_{protocol}"))

    return RunInfo(
        campaign_id=campaign_id,
        run_dir=run_dir,
        config_id=config_id,
        geometry_key=geometry_key,
        protocol=protocol,
        nx=nx,
        ny=ny,
        cycles=cycles,
        samples=samples,
        entropy_path=entropy_path,
        correlator_path=correlator_path,
        summary_path=summary_path if summary_path.exists() else None,
        metrics_path=metrics_path if metrics_path.exists() else None,
    )


def discover_runs(bundle_root: Path, campaign_id: str, protocol: str) -> tuple[list[RunInfo], list[dict[str, Any]]]:
    runs_root = bundle_root / "gpu_data" / DATASET_NAME / "campaigns" / campaign_id / "runs"
    if not runs_root.exists():
        raise FileNotFoundError(runs_root)

    runs: list[RunInfo] = []
    skipped: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        run_info = infer_run_info(campaign_id, run_dir)
        if run_info is None:
            skipped.append(
                {
                    "run_dir": str(run_dir),
                    "reason": f"missing {ENTROPY_FILENAME} or {CORRELATOR_FILENAME}",
                }
            )
            continue
        if protocol != "all" and run_info.protocol != protocol:
            skipped.append(
                {
                    "run_dir": str(run_dir),
                    "config_id": run_info.config_id,
                    "protocol": run_info.protocol,
                    "reason": f"filtered by --protocol {protocol}",
                }
            )
            continue
        runs.append(run_info)

    runs.sort(key=lambda item: (item.nx, item.ny, item.protocol))
    return runs, skipped


def log_chord(ay_values: np.ndarray, ny: int) -> np.ndarray:
    ay = np.asarray(ay_values, dtype=np.float64)
    return np.log((float(ny) / np.pi) * np.sin(np.pi * ay / float(ny)))


def linear_fit_1d(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[mask]
    y_valid = y[mask]
    n_points = int(x_valid.size)
    if n_points < 2:
        return np.nan, np.nan, np.nan, np.nan, n_points

    x_mean = float(np.mean(x_valid))
    y_mean = float(np.mean(y_valid))
    x_centered = x_valid - x_mean
    y_centered = y_valid - y_mean
    sxx = float(np.sum(x_centered * x_centered))
    if sxx <= 0.0:
        return np.nan, np.nan, np.nan, np.nan, n_points

    slope = float(np.sum(x_centered * y_centered) / sxx)
    intercept = float(y_mean - slope * x_mean)
    residual = y_valid - (slope * x_valid + intercept)
    sse = float(np.sum(residual * residual))
    sst = float(np.sum(y_centered * y_centered))
    r2 = float(1.0 - sse / sst) if sst > 0.0 else np.nan
    slope_stderr = float(np.sqrt((sse / float(n_points - 2)) / sxx)) if n_points > 2 else np.nan
    return slope, intercept, slope_stderr, r2, n_points


def fit_curves(x: np.ndarray, y_values: np.ndarray, *, desc: str) -> dict[str, np.ndarray]:
    if y_values.ndim != 3:
        raise ValueError(f"Expected y_values shape (samples, cycles, points), got {y_values.shape}")

    samples, cycles, _ = y_values.shape
    slope = np.full((samples, cycles), np.nan, dtype=np.float64)
    intercept = np.full((samples, cycles), np.nan, dtype=np.float64)
    slope_stderr = np.full((samples, cycles), np.nan, dtype=np.float64)
    r2 = np.full((samples, cycles), np.nan, dtype=np.float64)
    n_points = np.zeros((samples, cycles), dtype=np.int64)

    flat_y = y_values.reshape(samples * cycles, y_values.shape[-1])
    iterator = tqdm(flat_y, desc=desc, unit="fit", leave=False)
    for flat_idx, y in enumerate(iterator):
        fit = linear_fit_1d(x, y)
        sample_idx = flat_idx // cycles
        cycle_idx = flat_idx % cycles
        slope[sample_idx, cycle_idx] = fit[0]
        intercept[sample_idx, cycle_idx] = fit[1]
        slope_stderr[sample_idx, cycle_idx] = fit[2]
        r2[sample_idx, cycle_idx] = fit[3]
        n_points[sample_idx, cycle_idx] = fit[4]

    return {
        "slope": slope,
        "intercept": intercept,
        "slope_stderr": slope_stderr,
        "r2": r2,
        "n_points": n_points,
    }


def relative_to(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def output_path_for_run(output_root: Path, run_info: RunInfo) -> Path:
    return output_root / run_info.campaign_id / "runs" / run_info.run_dir.name / "streaming_covariance_scaling_fits.npz"


def fit_run(
    run_info: RunInfo,
    *,
    output_root: Path,
    bundle_root: Path,
    entropy_min_ay: int,
    corr_min_ry: int,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    with np.load(run_info.entropy_path, allow_pickle=False) as entropy_npz:
        entropy = entropy_npz["entropy_y0avg_vs_ay"].astype(np.float64, copy=False)
        ay_values = entropy_npz["ay_values"].astype(np.int64, copy=False)
        entropy_sample_indices = entropy_npz["sample_indices"].astype(np.int64, copy=False)
        entropy_cycle_labels = entropy_npz["cycle_labels"].astype(np.int64, copy=False)
        entropy_helper_version = str(entropy_npz["helper_version"].item()) if "helper_version" in entropy_npz else ""

    with np.load(run_info.correlator_path, allow_pickle=False) as corr_npz:
        corr = corr_npz["xavg_square_correlator_vs_ry"].astype(np.float64, copy=False)
        ry_values = corr_npz["ry_values"].astype(np.int64, copy=False)
        corr_sample_indices = corr_npz["sample_indices"].astype(np.int64, copy=False)
        corr_cycle_labels = corr_npz["cycle_labels"].astype(np.int64, copy=False)
        corr_formula = str(corr_npz["formula"].item()) if "formula" in corr_npz else ""
        corr_helper_version = str(corr_npz["helper_version"].item()) if "helper_version" in corr_npz else ""

    if entropy.shape[:2] != corr.shape[:2]:
        raise ValueError(f"{run_info.run_dir}: entropy and correlator sample/cycle shapes differ.")
    if not np.array_equal(entropy_sample_indices, corr_sample_indices):
        raise ValueError(f"{run_info.run_dir}: entropy and correlator sample_indices differ.")
    if not np.array_equal(entropy_cycle_labels, corr_cycle_labels):
        raise ValueError(f"{run_info.run_dir}: entropy and correlator cycle_labels differ.")

    entropy_mask = (ay_values >= int(entropy_min_ay)) & (ay_values <= run_info.ny // 2)
    corr_mask = (ry_values >= int(corr_min_ry)) & (ry_values <= run_info.ny // 2)
    if int(np.count_nonzero(entropy_mask)) < 2:
        raise ValueError(f"{run_info.run_dir}: entropy fit window has fewer than two Ay values.")
    if int(np.count_nonzero(corr_mask)) < 2:
        raise ValueError(f"{run_info.run_dir}: correlator fit window has fewer than two ry values.")

    entropy_x = log_chord(ay_values[entropy_mask], run_info.ny)
    entropy_y = entropy[:, :, entropy_mask]
    entropy_fit = fit_curves(entropy_x, entropy_y, desc=f"{run_info.run_dir.name} entropy")

    corr_selected = corr[:, :, corr_mask]
    corr_positive = np.where((corr_selected > 0.0) & np.isfinite(corr_selected), corr_selected, np.nan)
    corr_x = np.log(ry_values[corr_mask].astype(np.float64))
    corr_y = np.log(corr_positive)
    corr_fit = fit_curves(corr_x, corr_y, desc=f"{run_info.run_dir.name} correlator")

    out_path = output_path_for_run(output_root, run_info)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        sample_indices=entropy_sample_indices,
        cycle_labels=entropy_cycle_labels,
        entropy_slope=entropy_fit["slope"],
        entropy_intercept=entropy_fit["intercept"],
        entropy_slope_stderr=entropy_fit["slope_stderr"],
        entropy_r2=entropy_fit["r2"],
        entropy_n_points=entropy_fit["n_points"],
        entropy_fit_ay_values=ay_values[entropy_mask],
        entropy_fit_log_chord=entropy_x,
        entropy_fit_formula=np.asarray("S_y0avg(Ay) = slope * log((Ny/pi) * sin(pi*Ay/Ny)) + intercept"),
        correlator_loglog_slope=corr_fit["slope"],
        correlator_loglog_intercept=corr_fit["intercept"],
        correlator_loglog_slope_stderr=corr_fit["slope_stderr"],
        correlator_loglog_r2=corr_fit["r2"],
        correlator_loglog_n_points=corr_fit["n_points"],
        correlator_fit_ry_values=ry_values[corr_mask],
        correlator_fit_log_ry=corr_x,
        correlator_fit_formula=np.asarray("log(xavg_square_correlator_vs_ry) = slope * log(ry) + intercept"),
        source_entropy_path=np.asarray(relative_to(run_info.entropy_path, bundle_root)),
        source_correlator_path=np.asarray(relative_to(run_info.correlator_path, bundle_root)),
        source_correlator_formula=np.asarray(corr_formula),
        entropy_helper_version=np.asarray(entropy_helper_version),
        correlator_helper_version=np.asarray(corr_helper_version),
        config_id=np.asarray(run_info.config_id),
        geometry_key=np.asarray(run_info.geometry_key),
        protocol=np.asarray(run_info.protocol),
        Nx=np.asarray(run_info.nx, dtype=np.int64),
        Ny=np.asarray(run_info.ny, dtype=np.int64),
    )

    rows: list[dict[str, Any]] = []
    for sample_pos, sample_index in enumerate(entropy_sample_indices):
        for cycle_pos, cycle_label in enumerate(entropy_cycle_labels):
            rows.append(
                {
                    "config_id": run_info.config_id,
                    "geometry_key": run_info.geometry_key,
                    "protocol": run_info.protocol,
                    "Nx": run_info.nx,
                    "Ny": run_info.ny,
                    "sample_index": int(sample_index),
                    "cycle_label": int(cycle_label),
                    "entropy_slope": entropy_fit["slope"][sample_pos, cycle_pos],
                    "entropy_intercept": entropy_fit["intercept"][sample_pos, cycle_pos],
                    "entropy_slope_stderr": entropy_fit["slope_stderr"][sample_pos, cycle_pos],
                    "entropy_r2": entropy_fit["r2"][sample_pos, cycle_pos],
                    "entropy_n_points": int(entropy_fit["n_points"][sample_pos, cycle_pos]),
                    "correlator_loglog_slope": corr_fit["slope"][sample_pos, cycle_pos],
                    "correlator_loglog_intercept": corr_fit["intercept"][sample_pos, cycle_pos],
                    "correlator_loglog_slope_stderr": corr_fit["slope_stderr"][sample_pos, cycle_pos],
                    "correlator_loglog_r2": corr_fit["r2"][sample_pos, cycle_pos],
                    "correlator_loglog_n_points": int(corr_fit["n_points"][sample_pos, cycle_pos]),
                }
            )

    manifest_entry = {
        "config_id": run_info.config_id,
        "geometry_key": run_info.geometry_key,
        "protocol": run_info.protocol,
        "Nx": run_info.nx,
        "Ny": run_info.ny,
        "samples": run_info.samples,
        "cycles": run_info.cycles,
        "source_entropy_path": relative_to(run_info.entropy_path, bundle_root),
        "source_correlator_path": relative_to(run_info.correlator_path, bundle_root),
        "source_run_summary_path": relative_to(run_info.summary_path, bundle_root) if run_info.summary_path else None,
        "source_scalar_metrics_path": relative_to(run_info.metrics_path, bundle_root) if run_info.metrics_path else None,
        "output_npz_path": relative_to(out_path, bundle_root),
        "entropy_fit_ay_values": ay_values[entropy_mask].astype(int).tolist(),
        "correlator_fit_ry_values": ry_values[corr_mask].astype(int).tolist(),
    }
    return out_path, rows, manifest_entry


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config_id",
        "geometry_key",
        "protocol",
        "Nx",
        "Ny",
        "sample_index",
        "cycle_label",
        "entropy_slope",
        "entropy_intercept",
        "entropy_slope_stderr",
        "entropy_r2",
        "entropy_n_points",
        "correlator_loglog_slope",
        "correlator_loglog_intercept",
        "correlator_loglog_slope_stderr",
        "correlator_loglog_r2",
        "correlator_loglog_n_points",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit streaming covariance entropy and square-correlator scaling exponents from saved NPZ data."
    )
    parser.add_argument("--bundle-root", type=Path, default=None, help="Path to colab_charge_fluctuations.")
    parser.add_argument("--campaign-id", default=None, help="Campaign id under gpu_data/streaming_covariance_observables.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root. Defaults to <bundle-root>/analysis_outputs/streaming_covariance_scaling_fits.",
    )
    parser.add_argument(
        "--protocol",
        choices=("all", "perfect_correction", "postselect"),
        default="all",
        help="Protocol filter.",
    )
    parser.add_argument("--entropy-min-ay", type=int, default=8, help="Inclusive lower Ay for entropy fits.")
    parser.add_argument("--corr-min-ry", type=int, default=5, help="Inclusive lower ry for correlator fits.")
    parser.add_argument(
        "--cpu-list",
        default=DEFAULT_CPU_LIST,
        help="Comma/range CPU affinity list, e.g. 40-50 or 40,42. Defaults to 40-50.",
    )
    parser.add_argument("--no-affinity", action="store_true", help="Do not pin the process to a CPU affinity list.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without writing outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    affinity_info = set_process_affinity(None if args.no_affinity else args.cpu_list)
    bundle_root = resolve_bundle_root(args.bundle_root)
    campaign_id = args.campaign_id or default_campaign_id(bundle_root)
    output_root = args.output_root or (bundle_root / "analysis_outputs" / ANALYSIS_NAME)
    output_dir = output_root / campaign_id

    runs, skipped = discover_runs(bundle_root, campaign_id, args.protocol)
    print(f"[bundle root] {bundle_root}")
    print(f"[campaign id] {campaign_id}")
    print(f"[output dir] {output_dir}")
    print(f"[runs discovered] {len(runs)}")
    print(f"[runs skipped] {len(skipped)}")
    for run_info in runs:
        print(
            "[run] "
            f"{run_info.run_dir.name} protocol={run_info.protocol} "
            f"Nx={run_info.nx} Ny={run_info.ny} samples={run_info.samples} cycles={run_info.cycles}"
        )
    for entry in skipped:
        print(f"[skip] {entry['run_dir']} reason={entry['reason']}")

    if args.dry_run:
        return 0

    all_rows: list[dict[str, Any]] = []
    manifest_runs: list[dict[str, Any]] = []
    for run_info in tqdm(runs, desc="processing runs", unit="run"):
        out_path, rows, manifest_entry = fit_run(
            run_info,
            output_root=output_root,
            bundle_root=bundle_root,
            entropy_min_ay=args.entropy_min_ay,
            corr_min_ry=args.corr_min_ry,
        )
        all_rows.extend(rows)
        manifest_runs.append(manifest_entry)
        print(f"[wrote run npz] {out_path}")

    csv_path = output_dir / "tables" / CSV_FILENAME
    write_csv(csv_path, all_rows)
    print(f"[wrote csv] {csv_path} rows={len(all_rows)}")

    manifest = {
        "analysis_name": ANALYSIS_NAME,
        "bundle_root": str(bundle_root),
        "campaign_id": campaign_id,
        "dataset_name": DATASET_NAME,
        "cpu_affinity": affinity_info,
        "protocol_filter": args.protocol,
        "fit_definitions": {
            "entropy": {
                "source_array": "entropy_y0avg_vs_ay",
                "data_interpretation": "Already averaged over all y0; shape is (samples, cycles, Ay).",
                "x_formula": "log((Ny/pi) * sin(pi*Ay/Ny))",
                "y_formula": "entropy_y0avg_vs_ay",
                "window": {"min_ay": int(args.entropy_min_ay), "max_ay": "Ny//2"},
            },
            "correlator": {
                "source_array": "xavg_square_correlator_vs_ry",
                "data_interpretation": "Averaged over x, y, and orbital indices at fixed ry.",
                "x_formula": "log(ry)",
                "y_formula": "log(xavg_square_correlator_vs_ry)",
                "window": {"min_ry": int(args.corr_min_ry), "max_ry": "Ny//2"},
                "validity_filter": "finite source values greater than zero",
            },
        },
        "outputs": {
            "csv_path": relative_to(csv_path, bundle_root),
            "run_npz_root": relative_to(output_dir / "runs", bundle_root),
            "manifest_path": relative_to(output_dir / MANIFEST_FILENAME, bundle_root),
        },
        "runs": manifest_runs,
        "skipped_runs": skipped,
        "row_count": len(all_rows),
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    write_json(manifest_path, manifest)
    print(f"[wrote manifest] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
