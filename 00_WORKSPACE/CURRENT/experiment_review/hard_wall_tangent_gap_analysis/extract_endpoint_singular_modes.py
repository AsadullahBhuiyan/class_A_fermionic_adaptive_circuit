#!/usr/bin/env python3
"""Extract Ny=40 endpoint matrix SVDs; do not infer missing occupied/empty labels."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
from scipy.linalg import svd
from threadpoolctl import threadpool_limits
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
INPUT_ROOT = REPO / "00_WORKSPACE/LARGE_RESULTS/classA_final_production_outputs"
REVISION = "hard_wall_pure_tangent_alpha21_ny24-40_s100_c2ny_"
OUTPUT = HERE / "endpoint_singular_modes_ny40_v1"
SCHEMA = "slot17_ny40_full_matrix_svd_v1"
TOLERANCE = 1e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def x_profiles(vectors: np.ndarray, indices: np.ndarray, nx: int = 20) -> np.ndarray:
    """Return (mode,x) probability profiles; flat orbital index is 2*(x+Nx*y)+a."""
    if vectors.ndim != 2 or len(indices) != vectors.shape[0]:
        raise ValueError("vector/coordinate shape mismatch")
    x = (indices // 2) % nx
    return np.stack([np.sum(np.abs(vectors[x == i]) ** 2, axis=0)
                     for i in range(nx)], axis=1)


def decompose(matrix: np.ndarray, log_scale: float, cycles: int,
              active_indices: np.ndarray, nx: int = 20) -> dict[str, np.ndarray]:
    """Khat=U diag(s) Vh, with U on initial active rows and V on endpoint rows.

    For the saved covariance action K^dagger H0 K, U are INITIAL directions,
    whereas V are ENDPOINT directions. These are unclassified full-matrix
    modes, not the occupied-empty blocks used for the published tangent gap.
    """
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or not np.isfinite(log_scale):
        raise ValueError("nonfinite matrix or logarithmic scale")
    if cycles <= 0 or len(active_indices) != matrix.shape[0]:
        raise ValueError("invalid window or active indices")
    start = time.monotonic()
    u, s, vh = svd(matrix, full_matrices=False, check_finite=False, lapack_driver="gesdd")
    # Paired phase rotation preserves U diag(s) Vh exactly.
    pivots = np.argmax(np.abs(u), axis=0)
    phase = u[pivots, np.arange(len(s))]
    phase /= np.abs(phase)
    u *= phase.conj()[None, :]
    vh *= phase[:, None]
    norm = np.linalg.norm(matrix)
    if norm == 0:
        raise ValueError("zero matrix has no resolved singular modes")
    residual = float(np.linalg.norm((u * s) @ vh - matrix) / norm)
    gram_u = float(np.max(np.abs(u.conj().T @ u - np.eye(len(s)))))
    gram_v = float(np.max(np.abs(vh @ vh.conj().T - np.eye(len(s)))))
    if max(residual, gram_u, gram_v) > TOLERANCE:
        raise FloatingPointError(f"SVD validation failed: {residual=}, {gram_u=}, {gram_v=}")
    # This diagnoses the numerical rank of the SAVED normalized matrix, not
    # the null threshold used before normalization for the original gap.
    threshold = float(np.finfo(np.float64).eps * max(matrix.shape) * s[0])
    resolved = s > threshold
    with np.errstate(divide="ignore"):
        raw_logs = np.log(s) + log_scale
    logs = np.where(resolved, raw_logs, np.nan)
    rates = logs / cycles
    eligible = np.flatnonzero(resolved)
    nearest = eligible[np.argsort(np.abs(rates[eligible]), kind="stable")[:16]]
    initial_profile = x_profiles(u, active_indices, nx)
    endpoint_profile = x_profiles(vh.conj().T, np.arange(matrix.shape[1]), nx)
    np.testing.assert_allclose(initial_profile.sum(axis=1), 1, atol=TOLERANCE)
    np.testing.assert_allclose(endpoint_profile.sum(axis=1), 1, atol=TOLERANCE)
    return {
        "left_vectors_initial_active": u,
        "right_vectors_endpoint_dagger": vh,
        "singular_values_normalized": s,
        "chronological_cocycle_log_scale": np.asarray(log_scale),
        "raw_log_singular_values": raw_logs,
        "resolved_log_singular_values": logs,
        "resolved_one_leg_rates_per_cycle": rates,
        "resolved_mask": resolved,
        "numerical_rank_threshold_normalized": np.asarray(threshold),
        "numerical_rank": np.asarray(np.count_nonzero(resolved)),
        "numerical_null_count": np.asarray(np.count_nonzero(~resolved)),
        "omitted_right_nullspace_dimensions": np.asarray(max(0, matrix.shape[1] - len(s))),
        "one_leg_nearest_zero_mode_indices": nearest,
        "initial_x_profiles": initial_profile,
        "endpoint_x_profiles": endpoint_profile,
        "left_vector_phase_pivot_rows": pivots,
        "reconstruction_relative_frobenius_error": np.asarray(residual),
        "left_orthogonality_max_abs_error": np.asarray(gram_u),
        "right_orthogonality_max_abs_error": np.asarray(gram_v),
        "decomposition_seconds": np.asarray(time.monotonic() - start),
    }


def tasks() -> list[dict]:
    rows = []
    seen = set()
    paths = sorted(p for rev in ("v1", "v2")
                   for p in (INPUT_ROOT / (REVISION + rev) / "results/Ny040").rglob("*.complete.json"))
    for receipt in tqdm(paths, desc="Verify source cocycles", unit="batch"):
        d = json.loads(receipt.read_text())
        result = receipt.parent / d["result_filename"]
        if Path(d["result_filename"]).name != d["result_filename"]:
            raise ValueError("unsafe source filename")
        if result.stat().st_size != d["result_bytes"] or sha256(result) != d["result_sha256"]:
            raise ValueError(f"source checksum mismatch: {result}")
        if (d["Nx"], d["Ny"], d["cycles"], d["alpha_2"], d["nshell"]) != (20, 40, 80, 30, 1):
            raise ValueError("unexpected source geometry or dynamics")
        if not d["full_cocycle_saved"] or d["alpha_1"] not in (1, 3):
            raise ValueError("missing product or unexpected alpha")
        with np.load(result, allow_pickle=False) as z:
            for name in ("case_sample_indices", "global_sample_indices"):
                np.testing.assert_array_equal(z[name], d[name])
            if str(z["task_id"]) != d["task_id"]:
                raise ValueError("source task identity mismatch")
            if str(z["sampling_revision"]) != d["sampling_revision"]:
                raise ValueError("source revision mismatch")
            if str(z["dtype"]) != "complex128":
                raise ValueError("source is not complex128")
        for row, sample in enumerate(d["case_sample_indices"]):
            key = (int(d["alpha_1"]), int(sample))
            if key in seen:
                raise ValueError("duplicate source sample")
            seen.add(key)
            rows.append({
                "alpha_1": key[0], "sample_index": key[1], "row": row,
                "global_sample_index": d["global_sample_indices"][row],
                "source_result": str(result), "source_sha256": d["result_sha256"],
                "source_completion_sha256": sha256(receipt),
                "source_task_id": d["task_id"], "source_revision": d["sampling_revision"],
                "batch_seed": d["batch_seed"], "source_schema": d["schema"],
            })
    if seen != {(alpha, sample) for alpha in (1, 3) for sample in range(100)}:
        raise ValueError("source campaign does not contain all 200 distinct samples")
    return sorted(rows, key=lambda row: (row["alpha_1"], row["sample_index"]))


def result_paths(root: Path, task: dict) -> tuple[Path, Path]:
    result = root / f"alpha1_{task['alpha_1']}" / f"sample_{task['sample_index']:03d}.npz"
    return result, result.with_suffix(".complete.json")


def valid_completion(root: Path, task: dict, script_hash: str) -> dict | None:
    result, receipt = result_paths(root, task)
    if not result.is_file() or not receipt.is_file():
        return None
    try:
        d = json.loads(receipt.read_text())
        if (d["schema"] != SCHEMA or d["task"] != task or d["script_sha256"] != script_hash
                or d["result_filename"] != result.name or d["result_bytes"] != result.stat().st_size
                or d["result_sha256"] != sha256(result)):
            return None
        return d
    except (ValueError, KeyError, OSError):
        return None


@lru_cache(maxsize=1)
def load_source(path: str) -> dict:
    keys = ["chronological_cocycle_hat", "chronological_cocycle_log_scale", "active_input_indices",
            "slow_pair_rates_per_cycle", "slow_pair_indices", "occupied_empty_block_sizes",
            "one_leg_singular_null_counts", "source_hashes_json"]
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in keys}


def process(task: dict, root: str, script_hash: str, threads: int) -> dict:
    source = load_source(task["source_result"])
    row = task["row"]
    matrix = source["chronological_cocycle_hat"][row]
    if matrix.shape != (880, 1600):
        raise ValueError("unexpected Ny=40 cocycle shape")
    with threadpool_limits(limits=threads):
        arrays = decompose(matrix, float(source["chronological_cocycle_log_scale"][row]),
                           80, source["active_input_indices"])
    arrays.update({
        "schema": np.asarray(SCHEMA), "Nx": np.asarray(20), "Ny": np.asarray(40),
        "alpha_1": np.asarray(task["alpha_1"]), "alpha_2": np.asarray(30.),
        "cycles": np.asarray(80), "sample_index": np.asarray(task["sample_index"]),
        "global_sample_index": np.asarray(task["global_sample_index"]),
        "active_input_indices": source["active_input_indices"],
        "source_result_sha256": np.asarray(task["source_sha256"]),
        "source_hashes_json": source["source_hashes_json"],
        "source_task_json": np.asarray(json.dumps(task, sort_keys=True)),
        "source_signed_slow_pair_rates": source["slow_pair_rates_per_cycle"][row],
        "source_slow_pair_indices": source["slow_pair_indices"][row],
        "source_occupied_empty_block_sizes": source["occupied_empty_block_sizes"][row],
        "source_one_leg_singular_null_counts": source["one_leg_singular_null_counts"][row],
        "occupied_empty_labels_recovered": np.asarray(False),
        "original_slow_tangent_modes_recovered": np.asarray(False),
        "matrix_convention": np.asarray("Khat=U*diag(s)*Vh; K_full[active,:]=exp(log_scale)*Khat"),
        "covariance_action": np.asarray("H_T=K_full^dagger H_0 K_full"),
        "coordinate_convention": np.asarray("index=orbital+2*x+2*Nx*y"),
    })
    result, receipt = result_paths(Path(root), task)
    result.parent.mkdir(parents=True, exist_ok=True)
    temp = result.with_suffix(".partial.npz")
    np.savez(temp, **arrays)
    os.replace(temp, result)
    d = {"schema": SCHEMA, "task": task, "script_sha256": script_hash,
         "result_filename": result.name, "result_bytes": result.stat().st_size,
         "result_sha256": sha256(result), "numerical_rank": int(arrays["numerical_rank"]),
         "reconstruction_error": float(arrays["reconstruction_relative_frobenius_error"]),
         "left_orthogonality_error": float(arrays["left_orthogonality_max_abs_error"]),
         "right_orthogonality_error": float(arrays["right_orthogonality_max_abs_error"]),
         "decomposition_seconds": float(arrays["decomposition_seconds"])}
    temp_json = receipt.with_suffix(".partial.json")
    temp_json.write_text(json.dumps(d, indent=2) + "\n")
    os.replace(temp_json, receipt)
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--max-new-samples", type=int)
    args = parser.parse_args()
    if args.workers < 1 or args.threads_per_worker < 1:
        parser.error("workers and threads must be positive")
    if args.max_new_samples is not None and args.max_new_samples < 0:
        parser.error("max-new-samples must be nonnegative")
    task_list = tasks()
    script_hash = sha256(Path(__file__))
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    completed, pending = [], []
    for task in tqdm(task_list, desc="Check extracted modes", unit="sample"):
        d = valid_completion(root, task, script_hash)
        if d is None:
            pending.append(task)
        else:
            completed.append(d)
    selected = pending if args.max_new_samples is None else pending[:args.max_new_samples]
    print(json.dumps({"total": len(task_list), "verified_complete": len(completed),
                      "pending": len(pending), "run_now": len(selected), "workers": args.workers,
                      "blas_threads_per_worker": args.threads_per_worker,
                      "output_root": str(root), "circuit_replay": False,
                      "occupied_empty_labels_recovered": False}, indent=2), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(process, task, str(root), script_hash, args.threads_per_worker)
                   for task in selected]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Endpoint SVD extraction", unit="sample"):
            completed.append(future.result())
    completed.sort(key=lambda d: (d["task"]["alpha_1"], d["task"]["sample_index"]))
    manifest = {
        "schema": SCHEMA, "complete": len(completed) == 200, "samples_completed": len(completed),
        "samples_expected": 200, "Ny": 40, "alpha_1_values": [1, 3], "script_sha256": script_hash,
        "source_files_unchanged": True, "circuit_replay": False,
        "saved_basis": "full thin SVD, 880 singular triplets for each 880x1600 saved matrix",
        "limitations": ["No initial occupied frame in Ny40 products; no occupied/empty labels inferred.",
                        "Numerically null singular vectors are arbitrary and must not be interpreted physically.",
                        "Whole-matrix one-leg modes are not the source occupied-empty pair modes or their gap.",
                        "720 additional endpoint right-nullspace vectors omitted by thin SVD."],
        "results": completed,
    }
    temp = root / "manifest.partial.json"
    temp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temp, root / "manifest.json")
    print(f"[extraction complete] {len(completed)}/200 saved sample decompositions", flush=True)
    for alpha in (1, 3):
        group = [d for d in completed if d["task"]["alpha_1"] == alpha]
        if group:
            print(f"alpha1={alpha}: samples={len(group)}, numerical-rank range="
                  f"{min(d['numerical_rank'] for d in group)}..{max(d['numerical_rank'] for d in group)}, "
                  f"max reconstruction error={max(d['reconstruction_error'] for d in group):.3e}", flush=True)


if __name__ == "__main__":
    main()
