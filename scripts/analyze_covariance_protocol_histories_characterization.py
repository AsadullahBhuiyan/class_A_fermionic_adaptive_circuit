from __future__ import annotations

import json
import math
import os
import sys
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMALL_SYSTEM_ROOT = ROOT / "00_WORKSPACE" / "COLAB" / "colab_small_system_testing"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tmp" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".tmp" / "cache"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.auto import tqdm

from fgtn.classA_U1FGTN import classA_U1FGTN

CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
DEFAULT_CAMPAIGN_POINTER = (
    SMALL_SYSTEM_ROOT
    / "gpu_data"
    / "covariance_protocol_histories"
    / "latest_campaign.json"
)
DEFAULT_ANALYSIS_NAME = "covariance_protocol_histories_characterization_cpu"
EARLY_CYCLES = tuple(range(1, 6))
LATE_CYCLES = tuple(range(6, 11))
SNAPSHOT_CYCLES = (5, 6, 7, 8, 9, 10)
PROTOCOL_ORDER = ("perfect_correction", "postselect")
GEOMETRY_ORDER = ((16, 20), (16, 25), (16, 30))
HERM_TOL = 1e-9
TRACE_IMAG_TOL = 1e-8

# ---- Local run config ----
# Edit these directly before running the script locally.
CAMPAIGN_POINTER = DEFAULT_CAMPAIGN_POINTER
OUTPUT_ROOT_OVERRIDE = None
CPU_START = 0
CPU_COUNT = 30
N_JOBS = 30
OVERWRITE = True
WRITE_PDF = True
# --------------------------


class _TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
    def __init__(self, tqdm_object, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tqdm_object = tqdm_object

    def __call__(self, *args, **kwargs):
        self.tqdm_object.update(n=self.batch_size)
        return super().__call__(*args, **kwargs)


@contextmanager
def tqdm_joblib(tqdm_object):
    original_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = (
        lambda *args, **kwargs: _TqdmBatchCompletionCallback(tqdm_object, *args, **kwargs)
    )
    try:
        with tqdm_object as pbar:
            yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = original_callback


def joblib_progress(total: int, desc: str):
    if total <= 0:
        return nullcontext()
    return tqdm_joblib(tqdm(total=total, desc=desc, unit="task"))


def load_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    tmp_path.replace(path)


def write_csv_atomic(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def save_npz_atomic(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    np.savez_compressed(tmp_path, **payload)
    generated = tmp_path.with_suffix(tmp_path.suffix + ".npz")
    generated.replace(path)


def configure_cpu(cpu_start: int, cpu_count: int) -> dict:
    os.environ["MY_CPU_COUNT"] = str(cpu_count)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_MAX_THREADS"] = "1"

    affinity_error = None
    requested = list(range(cpu_start, cpu_start + cpu_count))
    try:
        os.sched_setaffinity(0, set(requested))
    except Exception as exc:  # pragma: no cover - platform-specific
        affinity_error = str(exc)
        print(f"[warn] CPU affinity not set: {exc}")

    actual_affinity = None
    try:
        actual_affinity = sorted(os.sched_getaffinity(0))
        print(f"[info] CPU affinity: {actual_affinity}")
    except Exception as exc:  # pragma: no cover - platform-specific
        affinity_error = str(exc) if affinity_error is None else affinity_error
        print(f"[warn] Could not query CPU affinity: {exc}")

    return {
        "cpu_start": int(cpu_start),
        "cpu_count": int(cpu_count),
        "requested_affinity": requested,
        "actual_affinity": actual_affinity,
        "affinity_error": affinity_error,
    }


def infer_default_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:  # pragma: no cover - platform-specific
        return max(1, int(os.cpu_count() or 1))


def rel_to(path: Path, root: Path) -> str | None:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except Exception:
        return None


def cycle_group_label(cycle_label: int) -> str:
    if cycle_label in EARLY_CYCLES:
        return "cycles_1_5"
    if cycle_label in LATE_CYCLES:
        return "cycles_6_10"
    raise ValueError(f"Unexpected cycle label {cycle_label}")


def build_wrapped_subregion_indices(nx: int, ny: int, ay: int, y0: int) -> np.ndarray:
    ay = int(ay)
    if ay < 0 or ay > ny:
        raise ValueError(f"ay must satisfy 0 <= ay <= {ny}; got {ay}")
    if ay == 0:
        return np.empty(0, dtype=np.int64)
    ys = (np.arange(ay, dtype=np.int64) + int(y0)) % ny
    sub_indices = np.empty(2 * nx * ay, dtype=np.int64)
    cursor = 0
    for y in ys:
        base = 2 * nx * int(y)
        for x in range(nx):
            sub_indices[cursor] = base + 2 * x
            sub_indices[cursor + 1] = base + 2 * x + 1
            cursor += 2
    return sub_indices


def precompute_subregion_indices(nx: int, ny: int) -> dict[tuple[int, int], np.ndarray]:
    mapping: dict[tuple[int, int], np.ndarray] = {}
    for ay in range(0, ny // 2 + 1):
        for y0 in range(ny):
            mapping[(ay, y0)] = build_wrapped_subregion_indices(nx, ny, ay, y0)
    return mapping


def gaussian_entropy_from_covariance(G_sub: np.ndarray) -> float:
    if G_sub.size == 0:
        return 0.0
    G_sub = np.asarray(G_sub, dtype=np.complex128)
    I = np.eye(G_sub.shape[0], dtype=np.complex128)
    G2 = 0.5 * (I + G_sub)
    evals = np.linalg.eigvalsh(G2)
    evals = np.clip(np.real_if_close(evals), 1e-12, 1.0 - 1e-12)
    entropy = -(evals * np.log(evals) + (1.0 - evals) * np.log(1.0 - evals))
    return float(np.sum(entropy))


def fit_entropy_log_chord(ay_values: np.ndarray, entropy_values: np.ndarray, ny: int) -> dict[str, float]:
    ay_values = np.asarray(ay_values, dtype=np.int64)
    entropy_values = np.asarray(entropy_values, dtype=float)
    fit_mask = (
        (ay_values >= 8)
        & (ay_values <= ny // 2)
        & np.isfinite(entropy_values)
        & (np.sin(np.pi * ay_values / ny) > 0)
    )
    x = np.log(np.sin(np.pi * ay_values[fit_mask] / ny))
    y = entropy_values[fit_mask]
    if x.size < 3:
        return {
            "fit_point_count": int(x.size),
            "slope": np.nan,
            "intercept": np.nan,
            "slope_err": np.nan,
            "r2": np.nan,
        }
    try:
        coeffs, cov = np.polyfit(x, y, 1, cov=True)
        slope, intercept = float(coeffs[0]), float(coeffs[1])
        slope_err = float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else np.nan
    except Exception:
        slope, intercept, slope_err = np.nan, np.nan, np.nan
    if not np.isfinite(slope):
        return {
            "fit_point_count": int(x.size),
            "slope": np.nan,
            "intercept": np.nan,
            "slope_err": np.nan,
            "r2": np.nan,
        }
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot == 0 else float(1.0 - ss_res / ss_tot)
    return {
        "fit_point_count": int(x.size),
        "slope": slope,
        "intercept": intercept,
        "slope_err": slope_err,
        "r2": r2,
    }


def expected_saved_samples(protocol: str) -> int:
    if protocol == "perfect_correction":
        return 10
    if protocol == "postselect":
        return 1
    raise ValueError(f"Unexpected protocol {protocol}")


def load_campaign(bundle_pointer: Path) -> tuple[dict, Path, Path]:
    latest_pointer = load_json(bundle_pointer)
    gpu_data_root = bundle_pointer.resolve().parents[1]
    manifest_rel = latest_pointer.get("campaign_manifest_path_relative") or latest_pointer.get("manifest")
    if manifest_rel is None:
        raise KeyError("latest_campaign.json missing campaign manifest pointer")
    campaign_manifest_path = gpu_data_root / manifest_rel
    campaign_manifest = load_json(campaign_manifest_path)
    return campaign_manifest, campaign_manifest_path, gpu_data_root


def default_analysis_output_root(campaign_id: str) -> Path:
    return (
        SMALL_SYSTEM_ROOT
        / "analysis_outputs"
        / DEFAULT_ANALYSIS_NAME
        / str(campaign_id)
    )


def load_analysis_outputs(output_root: Path) -> dict:
    output_root = Path(output_root)
    manifest_path = output_root / "analysis_manifest.json"
    metrics_path = output_root / "per_sample_cycle_metrics.csv"
    frob_path = output_root / "frob_successive_deltas.csv"
    summary_path = output_root / "ensemble_summaries.csv"
    curves_path = output_root / "entropy_curves.npz"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    payload = {
        "output_root": output_root,
        "manifest": load_json(manifest_path),
        "metrics_df": pd.read_csv(metrics_path),
        "frob_df": pd.read_csv(frob_path),
        "ensemble_df": pd.read_csv(summary_path),
        "entropy_curves": np.load(curves_path, allow_pickle=False),
    }
    return payload


def validate_and_prepare_run(entry: dict, gpu_data_root: Path) -> dict:
    protocol = str(entry["protocol"])
    nx = int(entry["Nx"])
    ny = int(entry["Ny"])
    nshell = int(entry["nshell"])
    saved_samples = expected_saved_samples(protocol)
    run_dir = gpu_data_root / entry["run_dir_relative"]
    manifest_path = gpu_data_root / entry["manifest_path_relative"]
    summary_rel = entry.get("summary_path_relative") or entry.get("run_summary_path_relative")
    summary_path = gpu_data_root / summary_rel

    manifest = load_json(manifest_path)
    summary = load_json(summary_path)
    cfg = manifest.get("config", {})
    shards = manifest.get("shards", {})

    if summary.get("canonical_dynamics_entry_point") != CANONICAL_ENTRY_POINT:
        raise ValueError(f"{protocol}: unexpected canonical entry point in run summary")
    if manifest.get("store_mode") != "history":
        raise ValueError(f"{protocol}: expected manifest store_mode='history'")
    if cfg.get("store_mode") != "history":
        raise ValueError(f"{protocol}: expected config store_mode='history'")
    if tuple(sorted(int(k) for k in shards)) != (0,):
        raise ValueError(f"{protocol}: expected exactly one shard, got {sorted(shards)}")
    if int(summary.get("history_length_saved", -1)) != 10:
        raise ValueError(f"{protocol}: expected history length 10")
    if int(summary.get("samples", -1)) != saved_samples:
        raise ValueError(f"{protocol}: expected saved sample count {saved_samples}")
    if int(summary.get("samples_requested", -1)) != 10:
        raise ValueError(f"{protocol}: expected requested sample count 10")
    if summary.get("snapshot_cycles") is not None:
        raise ValueError(f"{protocol}: expected snapshot_cycles=None")

    shard_meta = shards["0"]
    shard_path = run_dir / shard_meta["filename"]
    if not shard_path.exists():
        raise FileNotFoundError(shard_path)
    shard = np.load(shard_path, mmap_mode="r", allow_pickle=False)
    nlayer = 2 * nx * ny
    expected_shape = (saved_samples, 10, nlayer, nlayer)
    if tuple(int(x) for x in shard.shape) != expected_shape:
        raise ValueError(f"{protocol}: expected shard shape {expected_shape}, got {tuple(shard.shape)}")
    if np.dtype(shard.dtype) != np.dtype(np.complex128):
        raise ValueError(f"{protocol}: expected complex128 shard, got {shard.dtype}")

    return {
        "config_id": f"N{nx}x{ny}_{protocol}",
        "geometry_key": f"N{nx}x{ny}",
        "protocol": protocol,
        "Nx": nx,
        "Ny": ny,
        "nshell": nshell,
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "manifest": manifest,
        "summary": summary,
        "shard_path": shard_path,
        "shard": shard,
        "saved_samples": saved_samples,
        "history_length_saved": 10,
        "dw_loc": [int(x) for x in summary.get("dw_loc", [])],
        "alpha_1": float(summary.get("alpha_1", 1.0)),
        "alpha_2": float(summary.get("alpha_2", 30.0)),
        "trial_orbitals": str(summary.get("trial_orbitals", "X")),
        "dw_truncation": bool(summary.get("dw_truncation", False)),
    }


def build_chern_model(run_record: dict) -> classA_U1FGTN:
    return classA_U1FGTN(
        run_record["Nx"],
        run_record["Ny"],
        DW=True,
        nshell=run_record["nshell"],
        alpha_1=run_record["alpha_1"],
        alpha_2=run_record["alpha_2"],
        trial_orbitals=run_record["trial_orbitals"],
        dw_truncation=run_record["dw_truncation"],
    )


def process_snapshot(
    G_snapshot: np.ndarray,
    run_record: dict,
    model: classA_U1FGTN,
    subregion_indices: dict[tuple[int, int], np.ndarray],
) -> tuple[dict, np.ndarray]:
    nx = run_record["Nx"]
    ny = run_record["Ny"]
    nlayer = 2 * nx * ny
    cycle_label = int(run_record["_cycle_label"])
    sample_index = int(run_record["_sample_index"])
    G_snapshot = np.asarray(G_snapshot, dtype=np.complex128)
    if G_snapshot.shape != (nlayer, nlayer):
        raise ValueError(f"Expected snapshot shape {(nlayer, nlayer)}, got {G_snapshot.shape}")
    herm_err = float(np.max(np.abs(G_snapshot - G_snapshot.conj().T)))
    if herm_err > HERM_TOL:
        raise ValueError(
            f"{run_record['config_id']} sample={sample_index} cycle={cycle_label}: "
            f"Hermitian error {herm_err} exceeds tolerance {HERM_TOL}"
        )

    ay_values = np.arange(0, ny // 2 + 1, dtype=np.int64)
    entropy_curve = np.zeros_like(ay_values, dtype=float)
    for ay in ay_values:
        if ay == 0:
            entropy_curve[ay] = 0.0
            continue
        entropies = []
        for y0 in range(ny):
            sub_idx = subregion_indices[(int(ay), y0)]
            G_sub = G_snapshot[np.ix_(sub_idx, sub_idx)]
            entropies.append(gaussian_entropy_from_covariance(G_sub))
        entropy_curve[ay] = float(np.mean(entropies))

    fit = fit_entropy_log_chord(ay_values, entropy_curve, ny)
    trace_val = np.trace(G_snapshot)
    normalized_trace = float(np.real(trace_val) / nlayer)
    trace_imag_abs = float(abs(np.imag(trace_val)))
    if trace_imag_abs > TRACE_IMAG_TOL:
        raise ValueError(
            f"{run_record['config_id']} sample={sample_index} cycle={cycle_label}: "
            f"|Im tr(G)|={trace_imag_abs} exceeds tolerance {TRACE_IMAG_TOL}"
        )

    dw_loc = run_record["dw_loc"]
    if len(dw_loc) == 2:
        xref = int(math.floor((dw_loc[0] + dw_loc[1]) / 2))
    else:
        xref = nx // 2
    yref = ny // 2
    radius = 0.4 * min(nx, ny)
    chern = model.real_space_chern_number(G_snapshot, xref=xref, yref=yref, radius=radius)

    row = {
        "config_id": run_record["config_id"],
        "geometry_key": run_record["geometry_key"],
        "protocol": run_record["protocol"],
        "Nx": nx,
        "Ny": ny,
        "nshell": run_record["nshell"],
        "sample_index": sample_index,
        "cycle_label": cycle_label,
        "cycle_group": cycle_group_label(cycle_label),
        "fit_point_count": fit["fit_point_count"],
        "entropy_slope": fit["slope"],
        "entropy_intercept": fit["intercept"],
        "entropy_slope_err": fit["slope_err"],
        "entropy_r2": fit["r2"],
        "real_space_chern": float(np.real_if_close(chern)),
        "normalized_trace": normalized_trace,
        "trace_imag_abs": trace_imag_abs,
        "hermitian_max_err": herm_err,
    }
    return row, entropy_curve


def analyze_run_record(run_record: dict, n_jobs: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[np.ndarray]]:
    shard = run_record["shard"]
    nx = run_record["Nx"]
    ny = run_record["Ny"]
    model = build_chern_model(run_record)
    subregion_indices = precompute_subregion_indices(nx, ny)

    snapshot_tasks = []
    for sample_index in range(run_record["saved_samples"]):
        for cycle_idx in range(run_record["history_length_saved"]):
            snapshot_tasks.append((sample_index, cycle_idx))

    def _snapshot_task(sample_index: int, cycle_idx: int):
        local_record = dict(run_record)
        local_record["_sample_index"] = int(sample_index)
        local_record["_cycle_label"] = int(cycle_idx + 1)
        row, curve = process_snapshot(
            shard[sample_index, cycle_idx],
            local_record,
            model,
            subregion_indices,
        )
        key = (
            f"{run_record['config_id']}|sample={sample_index:03d}|cycle={cycle_idx + 1:02d}"
        )
        return row, key, curve

    progress_desc = f"snapshots {run_record['config_id']}"
    if n_jobs == 1:
        snapshot_results = [
            _snapshot_task(sample_index, cycle_idx)
            for sample_index, cycle_idx in tqdm(snapshot_tasks, desc=progress_desc, unit="snapshot")
        ]
    else:
        with joblib_progress(len(snapshot_tasks), progress_desc):
            snapshot_results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_snapshot_task)(sample_index, cycle_idx)
                for sample_index, cycle_idx in snapshot_tasks
            )

    metric_rows = []
    curve_keys = []
    curve_values = []
    for row, key, curve in snapshot_results:
        metric_rows.append(row)
        curve_keys.append(key)
        curve_values.append(curve)

    frob_rows = []
    frob_tasks = [
        (sample_index, cycle_idx)
        for sample_index in range(run_record["saved_samples"])
        for cycle_idx in range(1, run_record["history_length_saved"])
    ]
    for sample_index, cycle_idx in tqdm(
        frob_tasks,
        desc=f"frob {run_record['config_id']}",
        unit="delta",
        leave=False,
    ):
            delta = np.asarray(shard[sample_index, cycle_idx] - shard[sample_index, cycle_idx - 1])
            frob_rows.append(
                {
                    "config_id": run_record["config_id"],
                    "geometry_key": run_record["geometry_key"],
                    "protocol": run_record["protocol"],
                    "Nx": nx,
                    "Ny": ny,
                    "nshell": run_record["nshell"],
                    "sample_index": int(sample_index),
                    "cycle_label": int(cycle_idx + 1),
                    "cycle_prev_label": int(cycle_idx),
                    "cycle_group": cycle_group_label(cycle_idx + 1),
                    "frob_successive_delta": float(np.linalg.norm(delta, ord="fro")),
                }
            )

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["Nx", "Ny", "protocol", "sample_index", "cycle_label"]
    ).reset_index(drop=True)
    frob_df = pd.DataFrame(frob_rows).sort_values(
        ["Nx", "Ny", "protocol", "sample_index", "cycle_label"]
    ).reset_index(drop=True)
    return metrics_df, frob_df, curve_keys, curve_values


def build_entropy_curve_payload(
    metrics_df: pd.DataFrame,
    curve_keys: list[str],
    curve_values: list[np.ndarray],
) -> dict[str, np.ndarray]:
    offsets = []
    counts = []
    ay_flat = []
    entropy_flat = []
    offset = 0
    for curve in curve_values:
        ay_values = np.arange(curve.shape[0], dtype=np.int64)
        offsets.append(offset)
        counts.append(int(curve.shape[0]))
        ay_flat.append(ay_values)
        entropy_flat.append(np.asarray(curve, dtype=np.float64))
        offset += int(curve.shape[0])

    return {
        "curve_keys": np.asarray(curve_keys, dtype=f"<U{max(len(k) for k in curve_keys)}"),
        "curve_offsets": np.asarray(offsets, dtype=np.int64),
        "curve_counts": np.asarray(counts, dtype=np.int64),
        "ay_values_flat": np.concatenate(ay_flat).astype(np.int64, copy=False),
        "entropy_values_flat": np.concatenate(entropy_flat).astype(np.float64, copy=False),
        "curve_table_columns": np.asarray(metrics_df.columns.tolist(), dtype="<U64"),
    }


def summarize_distribution(
    df: pd.DataFrame,
    value_col: str,
    metric_name: str,
    group_kind: str,
    *,
    cycle_label: int | None = None,
) -> pd.DataFrame:
    work = df[np.isfinite(df[value_col])].copy()
    if cycle_label is not None:
        work = work[work["cycle_label"] == int(cycle_label)].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "group_kind",
                "config_id",
                "geometry_key",
                "protocol",
                "Nx",
                "Ny",
                "cycle_label",
                "count",
                "mean",
                "std",
                "min",
                "max",
            ]
        )
    grouped = (
        work.groupby(["config_id", "geometry_key", "protocol", "Nx", "Ny"], dropna=False)[value_col]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    grouped.insert(0, "metric", metric_name)
    grouped.insert(1, "group_kind", group_kind)
    grouped["cycle_label"] = np.nan if cycle_label is None else int(cycle_label)
    return grouped[
        [
            "metric",
            "group_kind",
            "config_id",
            "geometry_key",
            "protocol",
            "Nx",
            "Ny",
            "cycle_label",
            "count",
            "mean",
            "std",
            "min",
            "max",
        ]
    ]


def build_ensemble_summaries(metrics_df: pd.DataFrame, frob_df: pd.DataFrame) -> pd.DataFrame:
    parts = []

    frob_grouped = (
        frob_df.groupby(
            ["config_id", "geometry_key", "protocol", "Nx", "Ny", "cycle_label"], dropna=False
        )["frob_successive_delta"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    frob_grouped.insert(0, "metric", "frob_successive_delta")
    frob_grouped.insert(1, "group_kind", "cycle_snapshot")
    parts.append(
        frob_grouped[
            [
                "metric",
                "group_kind",
                "config_id",
                "geometry_key",
                "protocol",
                "Nx",
                "Ny",
                "cycle_label",
                "count",
                "mean",
                "std",
                "min",
                "max",
            ]
        ]
    )

    metrics = [
        ("entropy_slope", "entropy_slope"),
        ("real_space_chern", "real_space_chern"),
        ("normalized_trace", "normalized_trace"),
    ]
    early = metrics_df[metrics_df["cycle_label"].isin(EARLY_CYCLES)].copy()
    late = metrics_df[metrics_df["cycle_label"].isin(LATE_CYCLES)].copy()
    for value_col, metric_name in metrics:
        parts.append(summarize_distribution(early, value_col, metric_name, "cycles_1_5"))
        parts.append(summarize_distribution(late, value_col, metric_name, "cycles_6_10"))
        for cycle_label in SNAPSHOT_CYCLES:
            parts.append(
                summarize_distribution(
                    metrics_df,
                    value_col,
                    metric_name,
                    "cycle_snapshot",
                    cycle_label=cycle_label,
                )
            )

    summary_df = pd.concat(parts, ignore_index=True)
    return summary_df.sort_values(
        ["metric", "group_kind", "Nx", "Ny", "protocol", "cycle_label"],
        na_position="last",
    ).reset_index(drop=True)


def finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def compute_hist_bins(value_sets: list[np.ndarray]) -> np.ndarray:
    arrays = [finite_values(v) for v in value_sets if finite_values(v).size > 0]
    if not arrays:
        return np.linspace(-0.5, 0.5, 2)
    merged = np.concatenate(arrays)
    vmin = float(np.min(merged))
    vmax = float(np.max(merged))
    if np.isclose(vmin, vmax):
        delta = max(1e-6, abs(vmin) * 0.05, 0.05)
        return np.array([vmin - delta, vmax + delta], dtype=float)
    return np.linspace(vmin, vmax, 16)


def histogram_panel(ax, values: np.ndarray, bins: np.ndarray, title: str, xlabel: str) -> None:
    values = finite_values(values)
    if values.size == 0:
        ax.text(0.5, 0.5, "no finite data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        return
    ax.hist(values, bins=bins, color="#4C72B0", alpha=0.8, edgecolor="black")
    mean_val = float(np.mean(values))
    ax.axvline(mean_val, color="#DD8452", lw=2, linestyle="-")
    ax.set_title(f"{title}\nmean={mean_val:.6g}, n={values.size}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)


def create_overview_figure(geometry_key: str, metrics_df: pd.DataFrame, frob_df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    late_df = metrics_df[
        (metrics_df["geometry_key"] == geometry_key) & (metrics_df["cycle_label"].isin(LATE_CYCLES))
    ].copy()
    frob_sub = frob_df[frob_df["geometry_key"] == geometry_key].copy()

    slope_bins = compute_hist_bins(
        [late_df[late_df["protocol"] == protocol]["entropy_slope"].to_numpy() for protocol in PROTOCOL_ORDER]
    )
    chern_bins = compute_hist_bins(
        [late_df[late_df["protocol"] == protocol]["real_space_chern"].to_numpy() for protocol in PROTOCOL_ORDER]
    )
    trace_bins = compute_hist_bins(
        [late_df[late_df["protocol"] == protocol]["normalized_trace"].to_numpy() for protocol in PROTOCOL_ORDER]
    )

    for col, protocol in enumerate(PROTOCOL_ORDER):
        ax = axes[0, col]
        line_df = frob_sub[frob_sub["protocol"] == protocol].copy()
        stats = (
            line_df.groupby("cycle_label", dropna=False)["frob_successive_delta"]
            .agg(["mean", "std"])
            .reindex(range(2, 11))
        )
        cycles = np.asarray(stats.index, dtype=int)
        mean = stats["mean"].to_numpy(dtype=float)
        std = np.nan_to_num(stats["std"].to_numpy(dtype=float), nan=0.0)
        ax.plot(cycles, mean, marker="o", color="#4C72B0")
        ax.fill_between(cycles, mean - std, mean + std, color="#4C72B0", alpha=0.2)
        ax.set_title(f"{geometry_key} {protocol}\nFrobenius successive delta")
        ax.set_xlabel("cycle label c for ||G_c - G_(c-1)||_F")
        ax.set_ylabel("mean ± std")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)

        proto_df = late_df[late_df["protocol"] == protocol].copy()
        histogram_panel(
            axes[1, col],
            proto_df["entropy_slope"].to_numpy(),
            slope_bins,
            f"{geometry_key} {protocol}\nlate slope histogram (cycles 6-10)",
            "entropy slope",
        )
        histogram_panel(
            axes[2, col],
            proto_df["real_space_chern"].to_numpy(),
            chern_bins,
            f"{geometry_key} {protocol}\nlate Chern histogram (cycles 6-10)",
            "real-space Chern number",
        )
        histogram_panel(
            axes[3, col],
            proto_df["normalized_trace"].to_numpy(),
            trace_bins,
            f"{geometry_key} {protocol}\nlate normalized trace histogram (cycles 6-10)",
            "tr(G) / Nlayer",
        )

    fig.suptitle(f"Covariance characterization overview: {geometry_key}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def overview_page(pdf: PdfPages, geometry_key: str, metrics_df: pd.DataFrame, frob_df: pd.DataFrame) -> None:
    fig = create_overview_figure(geometry_key, metrics_df, frob_df)
    pdf.savefig(fig)
    plt.close(fig)


def create_snapshot_histogram_figure(
    metrics_df: pd.DataFrame,
    *,
    geometry_key: str,
    protocol: str,
    metric_col: str,
    metric_label: str,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    page_df = metrics_df[
        (metrics_df["geometry_key"] == geometry_key)
        & (metrics_df["protocol"] == protocol)
        & (metrics_df["cycle_label"].isin(SNAPSHOT_CYCLES))
    ].copy()
    bins = compute_hist_bins([page_df[page_df["cycle_label"] == c][metric_col].to_numpy() for c in SNAPSHOT_CYCLES])

    for ax, cycle_label in zip(axes, SNAPSHOT_CYCLES):
        cycle_df = page_df[page_df["cycle_label"] == cycle_label]
        histogram_panel(
            ax,
            cycle_df[metric_col].to_numpy(),
            bins,
            f"{geometry_key} {protocol}\ncycle {cycle_label}",
            metric_label,
        )

    fig.suptitle(f"{geometry_key} {protocol}: {metric_label} snapshot distributions", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def snapshot_histogram_page(
    pdf: PdfPages,
    metrics_df: pd.DataFrame,
    *,
    geometry_key: str,
    protocol: str,
    metric_col: str,
    metric_label: str,
) -> None:
    fig = create_snapshot_histogram_figure(
        metrics_df,
        geometry_key=geometry_key,
        protocol=protocol,
        metric_col=metric_col,
        metric_label=metric_label,
    )
    pdf.savefig(fig)
    plt.close(fig)


def create_late_ensemble_compare_figure(
    metrics_df: pd.DataFrame,
    *,
    protocol: str,
    metric_col: str,
    metric_label: str,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    page_df = metrics_df[
        (metrics_df["protocol"] == protocol)
        & (metrics_df["cycle_label"].isin(LATE_CYCLES))
        & (metrics_df["geometry_key"].isin([f"N{nx}x{ny}" for nx, ny in GEOMETRY_ORDER]))
    ].copy()
    bins = compute_hist_bins(
        [page_df[page_df["geometry_key"] == f"N{nx}x{ny}"][metric_col].to_numpy() for nx, ny in GEOMETRY_ORDER]
    )

    for ax, (nx, ny) in zip(axes, GEOMETRY_ORDER):
        geometry_key = f"N{nx}x{ny}"
        sub = page_df[page_df["geometry_key"] == geometry_key]
        histogram_panel(
            ax,
            sub[metric_col].to_numpy(),
            bins,
            f"{geometry_key} {protocol}\nlate ensemble (cycles 6-10)",
            metric_label,
        )

    fig.suptitle(f"{protocol}: late ensemble {metric_label} comparison", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def late_ensemble_compare_page(
    pdf: PdfPages,
    metrics_df: pd.DataFrame,
    *,
    protocol: str,
    metric_col: str,
    metric_label: str,
) -> None:
    fig = create_late_ensemble_compare_figure(
        metrics_df,
        protocol=protocol,
        metric_col=metric_col,
        metric_label=metric_label,
    )
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf_report(pdf_path: Path, metrics_df: pd.DataFrame, frob_df: pd.DataFrame) -> None:
    page_jobs = []
    for nx, ny in GEOMETRY_ORDER:
        page_jobs.append(("overview", {"geometry_key": f"N{nx}x{ny}"}))
    page_jobs.extend(
        [
            (
                "snapshot",
                {
                    "geometry_key": "N16x20",
                    "protocol": "perfect_correction",
                    "metric_col": "entropy_slope",
                    "metric_label": "entropy slope",
                },
            ),
            (
                "snapshot",
                {
                    "geometry_key": "N16x20",
                    "protocol": "postselect",
                    "metric_col": "entropy_slope",
                    "metric_label": "entropy slope",
                },
            ),
            (
                "snapshot",
                {
                    "geometry_key": "N16x20",
                    "protocol": "perfect_correction",
                    "metric_col": "real_space_chern",
                    "metric_label": "real-space Chern number",
                },
            ),
            (
                "snapshot",
                {
                    "geometry_key": "N16x20",
                    "protocol": "postselect",
                    "metric_col": "real_space_chern",
                    "metric_label": "real-space Chern number",
                },
            ),
            (
                "snapshot",
                {
                    "geometry_key": "N16x20",
                    "protocol": "perfect_correction",
                    "metric_col": "normalized_trace",
                    "metric_label": "tr(G) / Nlayer",
                },
            ),
            (
                "snapshot",
                {
                    "geometry_key": "N16x20",
                    "protocol": "postselect",
                    "metric_col": "normalized_trace",
                    "metric_label": "tr(G) / Nlayer",
                },
            ),
            (
                "late",
                {
                    "protocol": "perfect_correction",
                    "metric_col": "entropy_slope",
                    "metric_label": "entropy slope",
                },
            ),
            (
                "late",
                {
                    "protocol": "postselect",
                    "metric_col": "entropy_slope",
                    "metric_label": "entropy slope",
                },
            ),
            (
                "late",
                {
                    "protocol": "perfect_correction",
                    "metric_col": "real_space_chern",
                    "metric_label": "real-space Chern number",
                },
            ),
            (
                "late",
                {
                    "protocol": "postselect",
                    "metric_col": "real_space_chern",
                    "metric_label": "real-space Chern number",
                },
            ),
            (
                "late",
                {
                    "protocol": "perfect_correction",
                    "metric_col": "normalized_trace",
                    "metric_label": "tr(G) / Nlayer",
                },
            ),
            (
                "late",
                {
                    "protocol": "postselect",
                    "metric_col": "normalized_trace",
                    "metric_label": "tr(G) / Nlayer",
                },
            ),
        ]
    )

    with PdfPages(pdf_path) as pdf:
        for kind, kwargs in tqdm(page_jobs, desc="pdf pages", unit="page"):
            if kind == "overview":
                overview_page(pdf, kwargs["geometry_key"], metrics_df, frob_df)
            elif kind == "snapshot":
                snapshot_histogram_page(pdf, metrics_df, **kwargs)
            elif kind == "late":
                late_ensemble_compare_page(pdf, metrics_df, **kwargs)
            else:  # pragma: no cover
                raise ValueError(f"Unknown PDF page kind {kind}")


def run_internal_checks() -> None:
    idx_a = build_wrapped_subregion_indices(4, 7, 3, 2)
    idx_b = build_wrapped_subregion_indices(4, 7, 3, 9)
    if not np.array_equal(idx_a, idx_b):
        raise AssertionError("Wrapped subregion indexing failed modulo equivalence check")

    if gaussian_entropy_from_covariance(np.empty((0, 0), dtype=np.complex128)) != 0.0:
        raise AssertionError("Ay=0 entropy must be exactly zero")

    fit = fit_entropy_log_chord(
        np.asarray([8, 9, 10], dtype=np.int64),
        np.asarray([1.0, 1.2, 1.4], dtype=float),
        20,
    )
    if not np.isfinite(fit["slope"]):
        raise AssertionError("Entropy fit should return a finite slope with >=3 fit points")

    model = classA_U1FGTN(4, 6, DW=True, nshell=1, alpha_1=1.0, alpha_2=30.0, dw_truncation=True)
    nlayer = 2 * model.Nx * model.Ny
    full = np.zeros((model.Ntot, model.Ntot), dtype=np.complex128)
    top = np.zeros((nlayer, nlayer), dtype=np.complex128)
    default_val = model.real_space_chern_number(full)
    explicit_val = model.real_space_chern_number(
        full,
        xref=model.Nx // 2,
        yref=model.Ny // 2,
        radius=0.4 * min(model.Nx, model.Ny),
    )
    top_val = model.real_space_chern_number(top)
    if not np.allclose(default_val, explicit_val):
        raise AssertionError("Explicit real_space_chern_number defaults changed behavior")
    if not np.allclose(default_val, top_val):
        raise AssertionError("Top-layer and full-covariance Chern evaluation should match")


def ensure_output_paths(output_root: Path, no_pdf: bool, overwrite: bool) -> dict[str, Path]:
    paths = {
        "manifest": output_root / "analysis_manifest.json",
        "metrics_csv": output_root / "per_sample_cycle_metrics.csv",
        "frob_csv": output_root / "frob_successive_deltas.csv",
        "curves_npz": output_root / "entropy_curves.npz",
        "summary_csv": output_root / "ensemble_summaries.csv",
    }
    if not no_pdf:
        paths["pdf"] = output_root / "covariance_protocol_histories_characterization.pdf"
    if not overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "Output files already exist. Set OVERWRITE = True to recompute: "
                + ", ".join(str(path) for path in existing)
            )
    output_root.mkdir(parents=True, exist_ok=True)
    return paths


def main() -> None:
    t0 = time.time()
    run_internal_checks()

    campaign_pointer = Path(CAMPAIGN_POINTER)
    campaign_manifest, campaign_manifest_path, gpu_data_root = load_campaign(campaign_pointer)
    campaign_id = str(campaign_manifest["campaign_id"])
    if OUTPUT_ROOT_OVERRIDE is None:
        output_root = (
            SMALL_SYSTEM_ROOT
            / "analysis_outputs"
            / DEFAULT_ANALYSIS_NAME
            / campaign_id
        )
    else:
        output_root = Path(OUTPUT_ROOT_OVERRIDE)

    cpu_count = int(CPU_COUNT if CPU_COUNT is not None else infer_default_cpu_count())
    n_jobs = int(N_JOBS if N_JOBS is not None else cpu_count)
    if cpu_count <= 0 or n_jobs <= 0:
        raise ValueError("cpu-count and n-jobs must be positive")
    cpu_info = configure_cpu(int(CPU_START), cpu_count)

    paths = ensure_output_paths(output_root, not bool(WRITE_PDF), bool(OVERWRITE))
    run_records = [validate_and_prepare_run(entry, gpu_data_root) for entry in campaign_manifest["results"]]

    all_metrics = []
    all_frob = []
    all_curve_keys: list[str] = []
    all_curve_values: list[np.ndarray] = []
    for run_record in tqdm(run_records, desc="configs", unit="config"):
        metrics_df, frob_df, curve_keys, curve_values = analyze_run_record(run_record, n_jobs=n_jobs)
        all_metrics.append(metrics_df)
        all_frob.append(frob_df)
        all_curve_keys.extend(curve_keys)
        all_curve_values.extend(curve_values)

    metrics_df = pd.concat(all_metrics, ignore_index=True).sort_values(
        ["Nx", "Ny", "protocol", "sample_index", "cycle_label"]
    ).reset_index(drop=True)
    frob_df = pd.concat(all_frob, ignore_index=True).sort_values(
        ["Nx", "Ny", "protocol", "sample_index", "cycle_label"]
    ).reset_index(drop=True)
    ensemble_df = build_ensemble_summaries(metrics_df, frob_df)

    curve_payload = build_entropy_curve_payload(metrics_df, all_curve_keys, all_curve_values)

    write_csv_atomic(paths["metrics_csv"], metrics_df)
    write_csv_atomic(paths["frob_csv"], frob_df)
    write_csv_atomic(paths["summary_csv"], ensemble_df)
    save_npz_atomic(paths["curves_npz"], **curve_payload)

    if WRITE_PDF:
        write_pdf_report(paths["pdf"], metrics_df, frob_df)

    manifest_payload = {
        "analysis_name": DEFAULT_ANALYSIS_NAME,
        "campaign_id": campaign_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_campaign_pointer": str(campaign_pointer.resolve()),
        "source_campaign_pointer_relative_to_gpu_data": rel_to(campaign_pointer, gpu_data_root),
        "source_campaign_manifest": str(campaign_manifest_path.resolve()),
        "source_campaign_manifest_relative_to_gpu_data": rel_to(campaign_manifest_path, gpu_data_root),
        "gpu_data_root": str(gpu_data_root.resolve()),
        "output_root": str(output_root.resolve()),
        "canonical_entry_point_expected": CANONICAL_ENTRY_POINT,
        "cpu_info": cpu_info,
        "run_config": {
            "campaign_pointer": str(campaign_pointer),
            "output_root_override": None if OUTPUT_ROOT_OVERRIDE is None else str(Path(OUTPUT_ROOT_OVERRIDE)),
            "cpu_start": int(CPU_START),
            "cpu_count": cpu_count,
            "n_jobs": n_jobs,
            "overwrite": bool(OVERWRITE),
            "write_pdf": bool(WRITE_PDF),
        },
        "cycle_labeling": {
            "saved_init": False,
            "history_length_saved": 10,
            "cycle_labels": list(range(1, 11)),
            "early_cycles": list(EARLY_CYCLES),
            "late_cycles": list(LATE_CYCLES),
            "snapshot_cycles": list(SNAPSHOT_CYCLES),
        },
        "files": {name: str(path.resolve()) for name, path in paths.items()},
        "run_configs": [
            {
                "config_id": record["config_id"],
                "geometry_key": record["geometry_key"],
                "protocol": record["protocol"],
                "Nx": record["Nx"],
                "Ny": record["Ny"],
                "nshell": record["nshell"],
                "saved_samples": record["saved_samples"],
                "history_length_saved": record["history_length_saved"],
                "summary_path": str(record["summary_path"].resolve()),
                "manifest_path": str(record["manifest_path"].resolve()),
                "shard_path": str(record["shard_path"].resolve()),
            }
            for record in run_records
        ],
        "row_counts": {
            "per_sample_cycle_metrics": int(len(metrics_df)),
            "frob_successive_deltas": int(len(frob_df)),
            "ensemble_summaries": int(len(ensemble_df)),
            "entropy_curves": int(len(all_curve_keys)),
        },
        "runtime_seconds": float(time.time() - t0),
    }
    write_json_atomic(paths["manifest"], manifest_payload)
    print(f"[done] wrote analysis outputs to {output_root}")


if __name__ == "__main__":
    main()
