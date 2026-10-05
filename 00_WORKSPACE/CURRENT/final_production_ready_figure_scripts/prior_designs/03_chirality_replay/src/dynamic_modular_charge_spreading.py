from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


HELPER_VERSION = "dynamic_modular_hamiltonian_precompute_cpu_v1"

SOURCE_CAMPAIGN = "pure_state_covariance_snapshots"
OUTPUT_CAMPAIGN = "dynamic_modular_hamiltonians"
CANONICAL_SOURCE_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
EXPECTED_SOURCE_KEYS = {(16, 30, 1), (16, 30, 2), (16, 40, 1), (16, 40, 2)}
EXPECTED_SNAPSHOT_CYCLES = [5, 10, 20, 50]
DEFAULT_EPS = 1e-10


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str, *, enabled: bool = True) -> None:
    if enabled:
        print(f"[{timestamp()}] {message}", flush=True)


def progress_iter(iterable, *, enabled: bool, **kwargs):
    if not enabled:
        return iterable
    if _TimestampTqdm is None:
        return iterable
    kwargs.setdefault(
        "bar_format",
        "[{timestamp}] {desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit} "
        "[elapsed {elapsed}, eta {remaining}, {rate_fmt}]",
    )
    return _TimestampTqdm(iterable, **kwargs)


try:
    from tqdm.auto import tqdm as _tqdm_base

    class _TimestampTqdm(_tqdm_base):
        @property
        def format_dict(self):
            data = super().format_dict
            data["timestamp"] = timestamp()
            return data

except Exception:
    _TimestampTqdm = None


def load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    tmp_path.replace(path)


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    tmp_path.replace(path)


def rel_to_root(path: Path, root: Path) -> str | None:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except Exception:
        return None


def gpu_data_root(bundle_root: Path) -> Path:
    return Path(bundle_root) / "gpu_data"


def source_root(bundle_root: Path) -> Path:
    return gpu_data_root(bundle_root) / SOURCE_CAMPAIGN


def default_output_root(bundle_root: Path) -> Path:
    return gpu_data_root(bundle_root) / OUTPUT_CAMPAIGN


def expected_shard_shape(nx: int, ny: int) -> tuple[int, int, int, int]:
    nlayer = 2 * int(nx) * int(ny)
    return (10, len(EXPECTED_SNAPSHOT_CYCLES), nlayer, nlayer)


def validate_expected_source_entry(
    *,
    entry: dict[str, Any],
    manifest: dict[str, Any],
    summary: dict[str, Any],
    shard_path: Path,
) -> tuple[int, int, int, tuple[int, int, int, int]]:
    nx = int(entry["Nx"])
    ny = int(entry["Ny"])
    nshell = int(entry["nshell"])
    key = (nx, ny, nshell)
    if key not in EXPECTED_SOURCE_KEYS:
        raise ValueError(f"Unexpected source config {key}.")
    if int(entry.get("samples", -1)) != 10 or int(entry.get("cycles", -1)) != 50:
        raise ValueError(f"Unexpected samples/cycles for {key}.")
    if [int(c) for c in entry.get("snapshot_cycles", [])] != EXPECTED_SNAPSHOT_CYCLES:
        raise ValueError(f"Unexpected snapshot cycles for {key}: {entry.get('snapshot_cycles')}.")
    if entry.get("protocol") != "perfect_correction" or entry.get("init_mode") != "default":
        raise ValueError(f"Unexpected protocol/init for {key}.")
    if entry.get("dtype_resolved") != "complex128":
        raise ValueError(f"Unexpected entry dtype for {key}: {entry.get('dtype_resolved')}.")
    if summary.get("canonical_dynamics_entry_point") != CANONICAL_SOURCE_ENTRY_POINT:
        raise ValueError(f"Unexpected source entry point for {key}.")

    cfg = manifest.get("config", {})
    if manifest.get("store_mode") != "snapshots" or cfg.get("store_mode") != "snapshots":
        raise ValueError(f"Expected snapshot store mode for {key}.")
    if manifest.get("num_batches") != 1:
        raise ValueError(f"Expected one shard for {key}.")
    if cfg.get("dtype") != "c128":
        raise ValueError(f"Expected c128 manifest dtype for {key}.")
    if cfg.get("sequence") != "raster_y":
        raise ValueError(f"Expected raster_y sequence for {key}.")
    if cfg.get("postselect") is not False or cfg.get("perfect_correction") is not True:
        raise ValueError(f"Expected non-postselected perfect-correction source for {key}.")
    if [int(c) for c in cfg.get("snapshot_cycles", [])] != EXPECTED_SNAPSHOT_CYCLES:
        raise ValueError(f"Unexpected manifest snapshot cycles for {key}.")

    shards = manifest.get("shards", {})
    if sorted(int(k) for k in shards) != [0]:
        raise ValueError(f"Expected exactly shard 0 for {key}, got {sorted(shards)}.")
    expected_shape = expected_shard_shape(nx, ny)
    if [int(v) for v in shards["0"].get("shape", [])] != list(expected_shape):
        raise ValueError(f"Manifest shard shape mismatch for {key}.")
    if not Path(shard_path).exists():
        raise FileNotFoundError(f"Missing source shard for {key}: {shard_path}")
    shard = np.load(shard_path, mmap_mode="r")
    if shard.shape != expected_shape or shard.dtype != np.complex128:
        raise ValueError(f"{shard_path}: expected {expected_shape} complex128, got {shard.shape} {shard.dtype}.")
    return nx, ny, nshell, expected_shape


def source_run_records(bundle_root: Path) -> list[dict[str, Any]]:
    bundle_root = Path(bundle_root)
    data_root = gpu_data_root(bundle_root)
    campaign_path = source_root(bundle_root) / "campaign_manifest.json"
    campaign = load_json(campaign_path)
    if campaign.get("canonical_dynamics_entry_point") != CANONICAL_SOURCE_ENTRY_POINT:
        raise ValueError(f"Unexpected campaign entry point in {campaign_path}.")
    if campaign.get("protocol") != "perfect_correction" or campaign.get("dtype") != "complex128":
        raise ValueError(f"Unexpected source campaign protocol/dtype in {campaign_path}.")
    if [int(c) for c in campaign.get("snapshot_cycles", [])] != EXPECTED_SNAPSHOT_CYCLES:
        raise ValueError(f"Unexpected campaign snapshot cycles: {campaign.get('snapshot_cycles')}.")

    results = campaign.get("results", [])
    if len(results) != len(EXPECTED_SOURCE_KEYS):
        raise ValueError(f"Expected {len(EXPECTED_SOURCE_KEYS)} source configs, found {len(results)}.")

    records: list[dict[str, Any]] = []
    seen = set()
    for entry in results:
        run_dir = data_root / entry["run_dir_relative"]
        manifest_path = data_root / entry["manifest_path_relative"]
        summary_rel = entry.get("summary_path_relative") or entry.get("run_summary_path_relative")
        summary_path = data_root / summary_rel
        manifest = load_json(manifest_path)
        summary = load_json(summary_path)
        shard_path = run_dir / manifest["shards"]["0"]["filename"]
        nx, ny, nshell, shape = validate_expected_source_entry(
            entry=entry,
            manifest=manifest,
            summary=summary,
            shard_path=shard_path,
        )
        key = (nx, ny, nshell)
        if key in seen:
            raise ValueError(f"Duplicate source config {key}.")
        seen.add(key)
        records.append(
            {
                **entry,
                "Nx": nx,
                "Ny": ny,
                "nshell": nshell,
                "source_key": f"N{nx}x{ny}_nsh{nshell}_{entry['protocol']}",
                "run_dir": run_dir,
                "manifest_path": manifest_path,
                "summary_path": summary_path,
                "shard_path": shard_path,
                "source_manifest": manifest,
                "source_summary": summary,
                "expected_shard_shape": shape,
            }
        )

    missing = EXPECTED_SOURCE_KEYS - seen
    if missing:
        raise ValueError(f"Missing source configs: {sorted(missing)}.")
    return sorted(records, key=lambda rec: (int(rec["Ny"]), int(rec["nshell"])))


def reduced_site_index(nx: int, x: int, y_rel: int, orbital: int) -> int:
    return int(orbital) + 2 * int(x) + 2 * int(nx) * int(y_rel)


def initial_occupancy_vector(nx: int, ny: int, *, dw_x=(5, 11)) -> np.ndarray:
    nx = int(nx)
    ny = int(ny)
    ny_sub = ny // 2
    q0 = np.zeros(nx * ny_sub * 2, dtype=np.float64)
    for x in dw_x:
        for y_rel in (0, ny_sub - 1):
            for orbital in (0, 1):
                q0[reduced_site_index(nx, int(x) % nx, y_rel, orbital)] = 1.0
    if not np.isclose(float(q0.sum()), 8.0):
        raise ValueError(f"Initial occupancy should contain charge 8, got {q0.sum()}.")
    if np.max(np.abs(q0 * q0 - q0)) > 0.0:
        raise ValueError("Initial occupancy must be idempotent.")
    return q0


def half_window_indices(nx: int, ny: int, y0: int) -> np.ndarray:
    nx = int(nx)
    ny = int(ny)
    y0 = int(y0)
    ny_sub = ny // 2
    return np.asarray(
        [
            int(orbital) + 2 * int(x) + 2 * nx * ((y0 + int(y_rel)) % ny)
            for y_rel in range(ny_sub)
            for x in range(nx)
            for orbital in range(2)
        ],
        dtype=np.int64,
    )


def restrict_covariance(G_full: np.ndarray, *, nx: int, ny: int, y0: int) -> np.ndarray:
    idx = half_window_indices(nx, ny, y0)
    return np.asarray(G_full[np.ix_(idx, idx)], dtype=np.complex128)


def hermitize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.conj().T)


def max_hermiticity_error(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(matrix - matrix.conj().T)))


def modular_hamiltonian_from_restricted_covariance(
    G_sub: np.ndarray,
    *,
    eps: float = DEFAULT_EPS,
) -> dict[str, Any]:
    G_sub = np.asarray(G_sub, dtype=np.complex128)
    covariance_hermiticity_error = max_hermiticity_error(G_sub)
    G_sub = hermitize(G_sub)
    g_vals, g_vecs = np.linalg.eigh(G_sub)
    g_vals = np.asarray(g_vals.real, dtype=np.float64)
    eps = float(eps)
    clip_low = g_vals < (-1.0 + eps)
    clip_high = g_vals > (1.0 - eps)
    g_clipped = np.clip(g_vals, -1.0 + eps, 1.0 - eps)
    h_from_g = -2.0 * np.arctanh(g_clipped)
    if not np.all(np.isfinite(h_from_g)):
        raise FloatingPointError("Non-finite modular Hamiltonian spectrum from covariance.")
    h_mod = (g_vecs * h_from_g[None, :]) @ g_vecs.conj().T
    h_mod = hermitize(h_mod)
    h_mod_hermiticity_error = max_hermiticity_error(h_mod)
    return {
        "h_mod": h_mod,
        "g_vals": g_vals,
        "h_from_g": h_from_g,
        "covariance_hermiticity_error": covariance_hermiticity_error,
        "h_mod_hermiticity_error": h_mod_hermiticity_error,
        "clip_low_count": int(np.count_nonzero(clip_low)),
        "clip_high_count": int(np.count_nonzero(clip_high)),
    }


def eigensystem_from_h_mod(h_mod: np.ndarray) -> dict[str, Any]:
    h_mod = hermitize(np.asarray(h_mod, dtype=np.complex128))
    h_vals, h_vecs = np.linalg.eigh(h_mod)
    h_vals = np.asarray(h_vals.real, dtype=np.float64)
    reconstruction = (h_vecs * h_vals[None, :]) @ h_vecs.conj().T
    reconstruction_error = float(np.max(np.abs(reconstruction - h_mod)))
    return {
        "h_vals": h_vals,
        "h_vecs": np.asarray(h_vecs, dtype=np.complex128),
        "reconstruction_error": reconstruction_error,
        "h_mod_hermiticity_error": max_hermiticity_error(h_mod),
    }


def selected_indices(total: int, max_count: int | None) -> list[int]:
    total = int(total)
    if max_count is None:
        return list(range(total))
    return list(range(min(total, max(0, int(max_count)))))


def product_dir(output_root: Path, record: dict[str, Any]) -> Path:
    return Path(output_root) / "runs" / str(record["source_key"])


def product_path(output_root: Path, record: dict[str, Any]) -> Path:
    return product_dir(output_root, record) / "modular_hamiltonians.npz"


def compute_modular_hamiltonian_product(
    *,
    record: dict[str, Any],
    output_root: Path,
    eps: float = DEFAULT_EPS,
    single_sample_idx: int = 0,
    single_y0: int = 0,
    max_cycles: int | None = None,
    max_samples_for_avg: int | None = None,
    max_y0_for_avg: int | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    output_root = Path(output_root)
    out_path = product_path(output_root, record)
    if out_path.exists() and not overwrite:
        log(f"Reusing existing product after validation: {out_path}", enabled=progress)
        validate_modular_hamiltonian_product(out_path)
        return out_path

    nx = int(record["Nx"])
    ny = int(record["Ny"])
    dim = nx * (ny // 2) * 2
    shard = np.load(record["shard_path"], mmap_mode="r")
    sample_count, time_count, nlayer, nlayer2 = shard.shape
    if nlayer != nlayer2 or nlayer != 2 * nx * ny:
        raise ValueError(f"Unexpected shard shape for {record['source_key']}: {shard.shape}.")
    if not (0 <= int(single_sample_idx) < sample_count):
        raise ValueError(f"single_sample_idx={single_sample_idx} is outside sample range 0..{sample_count - 1}.")
    if not (0 <= int(single_y0) < ny):
        raise ValueError(f"single_y0={single_y0} is outside y0 range 0..{ny - 1}.")

    cycle_indices = selected_indices(time_count, max_cycles)
    sample_indices = selected_indices(sample_count, max_samples_for_avg)
    y0_values = selected_indices(ny, max_y0_for_avg)
    if not cycle_indices:
        raise ValueError("At least one cycle must be selected.")
    if not sample_indices:
        raise ValueError("At least one sample must be selected for averaging.")
    if not y0_values:
        raise ValueError("At least one y0 must be selected for averaging.")

    snapshot_cycles_all = [int(c) for c in record["snapshot_cycles"]]
    snapshot_cycles = np.asarray([snapshot_cycles_all[i] for i in cycle_indices], dtype=np.int64)
    T = len(cycle_indices)
    log(
        (
            f"Starting product {record['source_key']}: Nx={nx}, Ny={ny}, nshell={record['nshell']}, "
            f"dim={dim}, selected_cycles={snapshot_cycles.tolist()}, "
            f"single=(sample {int(single_sample_idx)}, y0 {int(single_y0)}), "
            f"avg_cuts_per_cycle={len(sample_indices) * len(y0_values)} "
            f"({len(sample_indices)} samples x {len(y0_values)} y0), output={out_path}"
        ),
        enabled=progress,
    )

    single_h_mod = np.empty((T, dim, dim), dtype=np.complex128)
    single_h_vals = np.empty((T, dim), dtype=np.float64)
    single_h_vecs = np.empty((T, dim, dim), dtype=np.complex128)
    single_g_vals = np.empty((T, dim), dtype=np.float64)
    single_cov_herm = np.empty((T,), dtype=np.float64)
    single_h_herm = np.empty((T,), dtype=np.float64)
    single_reconstruction_error = np.empty((T,), dtype=np.float64)
    single_clip_low = np.empty((T,), dtype=np.int64)
    single_clip_high = np.empty((T,), dtype=np.int64)

    avg_h_mod = np.empty((T, dim, dim), dtype=np.complex128)
    avg_h_vals = np.empty((T, dim), dtype=np.float64)
    avg_h_vecs = np.empty((T, dim, dim), dtype=np.complex128)
    avg_observation_count = np.empty((T,), dtype=np.int64)
    avg_cut_cov_herm_mean = np.empty((T,), dtype=np.float64)
    avg_cut_cov_herm_max = np.empty((T,), dtype=np.float64)
    avg_cut_h_herm_mean = np.empty((T,), dtype=np.float64)
    avg_cut_h_herm_max = np.empty((T,), dtype=np.float64)
    avg_h_herm = np.empty((T,), dtype=np.float64)
    avg_reconstruction_error = np.empty((T,), dtype=np.float64)
    avg_clip_low_total = np.empty((T,), dtype=np.int64)
    avg_clip_high_total = np.empty((T,), dtype=np.int64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cycle_iter = list(enumerate(cycle_indices))
    for out_t, cycle_idx in progress_iter(
        cycle_iter,
        enabled=progress,
        desc=f"{record['source_key']} cycles",
        unit="cycle",
        leave=True,
    ):
        cycle = snapshot_cycles_all[cycle_idx]
        log(f"{record['source_key']}: cycle {cycle} single-y0 modular Hamiltonian", enabled=progress)

        G_single = np.asarray(shard[int(single_sample_idx), int(cycle_idx)], dtype=np.complex128)
        single_sub = restrict_covariance(G_single, nx=nx, ny=ny, y0=int(single_y0))
        single_mod = modular_hamiltonian_from_restricted_covariance(single_sub, eps=eps)
        single_eig = eigensystem_from_h_mod(single_mod["h_mod"])
        single_h_mod[out_t] = single_mod["h_mod"]
        single_h_vals[out_t] = single_eig["h_vals"]
        single_h_vecs[out_t] = single_eig["h_vecs"]
        single_g_vals[out_t] = single_mod["g_vals"]
        single_cov_herm[out_t] = single_mod["covariance_hermiticity_error"]
        single_h_herm[out_t] = single_eig["h_mod_hermiticity_error"]
        single_reconstruction_error[out_t] = single_eig["reconstruction_error"]
        single_clip_low[out_t] = single_mod["clip_low_count"]
        single_clip_high[out_t] = single_mod["clip_high_count"]

        avg_accum = np.zeros((dim, dim), dtype=np.complex128)
        cov_herm_values = []
        h_herm_values = []
        clip_low_total = 0
        clip_high_total = 0
        obs_count = 0
        sample_cache: dict[int, np.ndarray] = {}
        avg_tasks = [(int(sample_idx), int(y0)) for sample_idx in sample_indices for y0 in y0_values]
        for sample_idx, y0 in progress_iter(
            avg_tasks,
            enabled=progress,
            desc=f"{record['source_key']} cycle {cycle} avg H",
            unit="cut",
            leave=False,
        ):
            if sample_idx not in sample_cache:
                sample_cache[sample_idx] = np.asarray(shard[int(sample_idx), int(cycle_idx)], dtype=np.complex128)
            G_sample = sample_cache[sample_idx]
            sub = restrict_covariance(G_sample, nx=nx, ny=ny, y0=int(y0))
            mod = modular_hamiltonian_from_restricted_covariance(sub, eps=eps)
            avg_accum += mod["h_mod"]
            cov_herm_values.append(float(mod["covariance_hermiticity_error"]))
            h_herm_values.append(float(mod["h_mod_hermiticity_error"]))
            clip_low_total += int(mod["clip_low_count"])
            clip_high_total += int(mod["clip_high_count"])
            obs_count += 1
        sample_cache.clear()
        h_avg = hermitize(avg_accum / float(obs_count))
        avg_eig = eigensystem_from_h_mod(h_avg)
        avg_h_mod[out_t] = h_avg
        avg_h_vals[out_t] = avg_eig["h_vals"]
        avg_h_vecs[out_t] = avg_eig["h_vecs"]
        avg_observation_count[out_t] = int(obs_count)
        avg_cut_cov_herm_mean[out_t] = float(np.mean(cov_herm_values))
        avg_cut_cov_herm_max[out_t] = float(np.max(cov_herm_values))
        avg_cut_h_herm_mean[out_t] = float(np.mean(h_herm_values))
        avg_cut_h_herm_max[out_t] = float(np.max(h_herm_values))
        avg_h_herm[out_t] = avg_eig["h_mod_hermiticity_error"]
        avg_reconstruction_error[out_t] = avg_eig["reconstruction_error"]
        avg_clip_low_total[out_t] = int(clip_low_total)
        avg_clip_high_total[out_t] = int(clip_high_total)
        log(
            (
                f"{record['source_key']}: cycle {cycle} complete; "
                f"single_recon={single_reconstruction_error[out_t]:.3e}, "
                f"avg_recon={avg_reconstruction_error[out_t]:.3e}, "
                f"avg_obs={avg_observation_count[out_t]}"
            ),
            enabled=progress,
        )

    metadata = {
        "helper_version": HELPER_VERSION,
        "source_campaign": SOURCE_CAMPAIGN,
        "source_key": record["source_key"],
        "source_shard_path": str(record["shard_path"]),
        "source_manifest_path": str(record["manifest_path"]),
        "source_summary_path": str(record["summary_path"]),
        "Nx": nx,
        "Ny": ny,
        "nshell": int(record["nshell"]),
        "protocol": record["protocol"],
        "modular_eps": float(eps),
        "dim": int(dim),
        "basis_order": "y_rel, x, orbital",
        "subsystem": "[0,Nx) x [y0,y0+Ny//2)",
        "single_mode": "single_y0_sample",
        "single_sample_idx": int(single_sample_idx),
        "single_y0": int(single_y0),
        "avg_mode": "sample_y0_avg_hmod",
        "avg_sample_indices": [int(i) for i in sample_indices],
        "avg_y0_values": [int(y0) for y0 in y0_values],
        "avg_observation_count_per_cycle": int(len(sample_indices) * len(y0_values)),
        "cycle_indices": [int(i) for i in cycle_indices],
        "snapshot_cycles": [int(c) for c in snapshot_cycles],
        "is_truncated_smoke": bool(
            max_cycles is not None or max_samples_for_avg is not None or max_y0_for_avg is not None
        ),
    }

    save_npz_atomic(
        out_path,
        snapshot_cycles=snapshot_cycles,
        cycle_indices=np.asarray(cycle_indices, dtype=np.int64),
        Nx=np.asarray(nx, dtype=np.int64),
        Ny=np.asarray(ny, dtype=np.int64),
        nshell=np.asarray(int(record["nshell"]), dtype=np.int64),
        dim=np.asarray(dim, dtype=np.int64),
        single_sample_idx=np.asarray(int(single_sample_idx), dtype=np.int64),
        single_y0=np.asarray(int(single_y0), dtype=np.int64),
        single_h_mod=single_h_mod,
        single_h_vals=single_h_vals,
        single_h_vecs=single_h_vecs,
        single_g_vals=single_g_vals,
        single_covariance_hermiticity_error=single_cov_herm,
        single_h_mod_hermiticity_error=single_h_herm,
        single_reconstruction_error=single_reconstruction_error,
        single_clip_low_count=single_clip_low,
        single_clip_high_count=single_clip_high,
        avg_h_mod=avg_h_mod,
        avg_h_vals=avg_h_vals,
        avg_h_vecs=avg_h_vecs,
        avg_observation_count=avg_observation_count,
        avg_cut_covariance_hermiticity_error_mean=avg_cut_cov_herm_mean,
        avg_cut_covariance_hermiticity_error_max=avg_cut_cov_herm_max,
        avg_cut_h_mod_hermiticity_error_mean=avg_cut_h_herm_mean,
        avg_cut_h_mod_hermiticity_error_max=avg_cut_h_herm_max,
        avg_h_mod_hermiticity_error=avg_h_herm,
        avg_reconstruction_error=avg_reconstruction_error,
        avg_clip_low_count_total=avg_clip_low_total,
        avg_clip_high_count_total=avg_clip_high_total,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    validate_modular_hamiltonian_product(out_path)
    log(f"Saved and validated product: {out_path}", enabled=progress)
    return out_path


def validate_modular_hamiltonian_product(
    path: Path,
    *,
    hermiticity_tol: float = 1e-9,
    reconstruction_tol: float = 1e-8,
) -> None:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        T = len(data["snapshot_cycles"])
        dim = int(data["dim"])
        for prefix in ("single", "avg"):
            h_mod = np.asarray(data[f"{prefix}_h_mod"])
            h_vals = np.asarray(data[f"{prefix}_h_vals"])
            h_vecs = np.asarray(data[f"{prefix}_h_vecs"])
            if h_mod.shape != (T, dim, dim):
                raise ValueError(f"{path}: {prefix}_h_mod shape {h_mod.shape} != {(T, dim, dim)}.")
            if h_vals.shape != (T, dim):
                raise ValueError(f"{path}: {prefix}_h_vals shape {h_vals.shape} != {(T, dim)}.")
            if h_vecs.shape != (T, dim, dim):
                raise ValueError(f"{path}: {prefix}_h_vecs shape {h_vecs.shape} != {(T, dim, dim)}.")
            if not np.all(np.isfinite(h_mod)) or not np.all(np.isfinite(h_vals)) or not np.all(np.isfinite(h_vecs)):
                raise FloatingPointError(f"{path}: non-finite {prefix} modular Hamiltonian data.")
            herm_errors = np.max(np.abs(h_mod - h_mod.conj().transpose(0, 2, 1)), axis=(1, 2))
            if float(np.max(herm_errors)) > hermiticity_tol:
                raise ValueError(f"{path}: {prefix} Hermiticity error exceeds {hermiticity_tol}.")
            for t_idx in range(T):
                reconstructed = (h_vecs[t_idx] * h_vals[t_idx][None, :]) @ h_vecs[t_idx].conj().T
                err = float(np.max(np.abs(reconstructed - h_mod[t_idx])))
                if err > reconstruction_tol:
                    raise ValueError(f"{path}: {prefix} reconstruction error {err:g} exceeds {reconstruction_tol:g}.")
        expected_obs = int(metadata["avg_observation_count_per_cycle"])
        if not np.all(np.asarray(data["avg_observation_count"], dtype=np.int64) == expected_obs):
            raise ValueError(f"{path}: avg observation count mismatch.")


def summary_rows_for_product(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        cycles = np.asarray(data["snapshot_cycles"], dtype=np.int64)
        for prefix, label in (("single", "single_y0_sample"), ("avg", "sample_y0_avg_hmod")):
            h_vals = np.asarray(data[f"{prefix}_h_vals"], dtype=np.float64)
            for t_idx, cycle in enumerate(cycles):
                row = {
                    "source_key": metadata["source_key"],
                    "Nx": int(metadata["Nx"]),
                    "Ny": int(metadata["Ny"]),
                    "nshell": int(metadata["nshell"]),
                    "cycle": int(cycle),
                    "cycle_idx": int(data["cycle_indices"][t_idx]),
                    "construction": label,
                    "dim": int(metadata["dim"]),
                    "h_min": float(np.min(h_vals[t_idx])),
                    "h_q01": float(np.quantile(h_vals[t_idx], 0.01)),
                    "h_q05": float(np.quantile(h_vals[t_idx], 0.05)),
                    "h_median": float(np.quantile(h_vals[t_idx], 0.50)),
                    "h_q95": float(np.quantile(h_vals[t_idx], 0.95)),
                    "h_q99": float(np.quantile(h_vals[t_idx], 0.99)),
                    "h_max": float(np.max(h_vals[t_idx])),
                    "h_abs_max": float(np.max(np.abs(h_vals[t_idx]))),
                    "product_path": str(path),
                }
                if prefix == "single":
                    row.update(
                        {
                            "sample_idx": int(data["single_sample_idx"]),
                            "y0": int(data["single_y0"]),
                            "observation_count": 1,
                            "covariance_hermiticity_error": float(data["single_covariance_hermiticity_error"][t_idx]),
                            "h_mod_hermiticity_error": float(data["single_h_mod_hermiticity_error"][t_idx]),
                            "reconstruction_error": float(data["single_reconstruction_error"][t_idx]),
                            "clip_low_count": int(data["single_clip_low_count"][t_idx]),
                            "clip_high_count": int(data["single_clip_high_count"][t_idx]),
                        }
                    )
                else:
                    row.update(
                        {
                            "sample_idx": -1,
                            "y0": -1,
                            "observation_count": int(data["avg_observation_count"][t_idx]),
                            "covariance_hermiticity_error": float(data["avg_cut_covariance_hermiticity_error_max"][t_idx]),
                            "h_mod_hermiticity_error": float(data["avg_h_mod_hermiticity_error"][t_idx]),
                            "reconstruction_error": float(data["avg_reconstruction_error"][t_idx]),
                            "clip_low_count": int(data["avg_clip_low_count_total"][t_idx]),
                            "clip_high_count": int(data["avg_clip_high_count_total"][t_idx]),
                        }
                    )
                rows.append(row)
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write empty summary.")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0].keys())
    with tmp_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def maybe_write_summary_parquet(path: Path, rows: list[dict[str, Any]]) -> str:
    try:
        import pandas as pd
    except Exception as exc:
        return f"skipped: {type(exc).__name__}: {exc}"
    try:
        pd.DataFrame(rows).to_parquet(path, index=False)
    except Exception as exc:
        return f"skipped: {type(exc).__name__}: {exc}"
    return "saved"


def update_gpu_data_index(bundle_root: Path, output_root: Path) -> None:
    root = gpu_data_root(bundle_root)
    index_path = root / "index.json"
    if index_path.exists():
        index = load_json(index_path)
    else:
        index = {"bundle_root": str(bundle_root), "gpu_data_root": str(root), "campaigns": []}
    campaigns = index.setdefault("campaigns", [])
    entry = {
        "name": OUTPUT_CAMPAIGN,
        "kind": "derived_modular_hamiltonian_observable",
        "manifest": f"{rel_to_root(output_root, root)}/campaign_manifest.json",
    }
    campaigns[:] = [campaign for campaign in campaigns if campaign.get("name") != OUTPUT_CAMPAIGN]
    campaigns.append(entry)
    index["bundle_root"] = str(Path(bundle_root))
    index["gpu_data_root"] = str(root)
    write_json_atomic(index_path, index)


def run_precompute_campaign(
    *,
    bundle_root: Path,
    output_root: Path | None = None,
    eps: float = DEFAULT_EPS,
    single_sample_idx: int = 0,
    single_y0: int = 0,
    max_configs: int | None = None,
    max_cycles: int | None = None,
    max_samples_for_avg: int | None = None,
    max_y0_for_avg: int | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    bundle_root = Path(bundle_root).resolve()
    output_root = default_output_root(bundle_root) if output_root is None else Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records = source_run_records(bundle_root)
    records = records[: max_configs if max_configs is not None else len(records)]
    log(
        (
            f"Precompute campaign start: bundle_root={bundle_root}, output_root={output_root}, "
            f"configs={len(records)}, eps={eps:g}, overwrite={overwrite}, "
            f"truncated={max_configs is not None or max_cycles is not None or max_samples_for_avg is not None or max_y0_for_avg is not None}"
        ),
        enabled=progress,
    )
    for record in records:
        log(
            (
                f"Source validated: {record['source_key']} shape={record['expected_shard_shape']} "
                f"cycles={record['snapshot_cycles']} shard={record['shard_path']}"
            ),
            enabled=progress,
        )
    product_results = []
    summary_rows: list[dict[str, Any]] = []
    for record in progress_iter(records, enabled=progress, desc="configs", unit="config", leave=True):
        log(f"Config start: {record['source_key']}", enabled=progress)
        path = compute_modular_hamiltonian_product(
            record=record,
            output_root=output_root,
            eps=eps,
            single_sample_idx=single_sample_idx,
            single_y0=single_y0,
            max_cycles=max_cycles,
            max_samples_for_avg=max_samples_for_avg,
            max_y0_for_avg=max_y0_for_avg,
            overwrite=overwrite,
            progress=progress,
        )
        rows = summary_rows_for_product(path)
        summary_rows.extend(rows)
        product_results.append(
            {
                "source_key": record["source_key"],
                "Nx": int(record["Nx"]),
                "Ny": int(record["Ny"]),
                "nshell": int(record["nshell"]),
                "product_path": str(path),
                "product_path_relative": rel_to_root(path, gpu_data_root(bundle_root)),
                "summary_rows": int(len(rows)),
            }
        )
        log(f"Config complete: {record['source_key']} summary_rows={len(rows)}", enabled=progress)

    summary_csv = output_root / "modular_hamiltonian_summary.csv"
    summary_parquet = output_root / "modular_hamiltonian_summary.parquet"
    write_summary_csv(summary_csv, summary_rows)
    parquet_status = maybe_write_summary_parquet(summary_parquet, summary_rows)
    log(
        f"Summary tables written: csv={summary_csv}, parquet_status={parquet_status}, rows={len(summary_rows)}",
        enabled=progress,
    )

    output_root_relative = rel_to_root(output_root, gpu_data_root(bundle_root))
    manifest = {
        "helper_version": HELPER_VERSION,
        "derived_observable": "dynamic modular Hamiltonians",
        "source_campaign": SOURCE_CAMPAIGN,
        "source_campaign_manifest_relative": f"{SOURCE_CAMPAIGN}/campaign_manifest.json",
        "bundle_root": str(bundle_root),
        "gpu_data_root": str(gpu_data_root(bundle_root)),
        "output_root": str(output_root),
        "output_root_relative": output_root_relative,
        "canonical_source_entry_point": CANONICAL_SOURCE_ENTRY_POINT,
        "modular_eps": float(eps),
        "single_mode": "single_y0_sample",
        "single_sample_idx": int(single_sample_idx),
        "single_y0": int(single_y0),
        "avg_mode": "sample_y0_avg_hmod",
        "is_truncated_smoke": bool(
            max_configs is not None
            or max_cycles is not None
            or max_samples_for_avg is not None
            or max_y0_for_avg is not None
        ),
        "max_configs": None if max_configs is None else int(max_configs),
        "max_cycles": None if max_cycles is None else int(max_cycles),
        "max_samples_for_avg": None if max_samples_for_avg is None else int(max_samples_for_avg),
        "max_y0_for_avg": None if max_y0_for_avg is None else int(max_y0_for_avg),
        "results": product_results,
        "summary_tables": {
            "modular_hamiltonian_summary_csv": str(summary_csv),
            "modular_hamiltonian_summary_parquet": str(summary_parquet),
            "parquet_status": parquet_status,
            "row_count": int(len(summary_rows)),
        },
        "index_updated": output_root_relative is not None,
    }
    manifest_path = output_root / "campaign_manifest.json"
    write_json_atomic(manifest_path, manifest)
    if output_root_relative is not None:
        update_gpu_data_index(bundle_root, output_root)
    manifest["manifest_path"] = str(manifest_path)
    log(
        f"Campaign manifest written: {manifest_path}; gpu_data_index_updated={manifest['index_updated']}",
        enabled=progress,
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute dense modular Hamiltonians from saved covariance snapshots.")
    parser.add_argument("--bundle-root", type=Path, default=Path("colab_small_system_testing"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--single-sample-idx", type=int, default=0)
    parser.add_argument("--single-y0", type=int, default=0)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--max-samples-for-avg", type=int, default=None)
    parser.add_argument("--max-y0-for-avg", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = run_precompute_campaign(
        bundle_root=args.bundle_root,
        output_root=args.output_root,
        eps=args.eps,
        single_sample_idx=args.single_sample_idx,
        single_y0=args.single_y0,
        max_configs=args.max_configs,
        max_cycles=args.max_cycles,
        max_samples_for_avg=args.max_samples_for_avg,
        max_y0_for_avg=args.max_y0_for_avg,
        overwrite=args.overwrite,
        progress=not args.quiet,
    )
    print(json.dumps({"manifest_path": manifest["manifest_path"], "results": len(manifest["results"])}, indent=2))


if __name__ == "__main__":
    main()
