#!/usr/bin/env python3
"""Analyze all 600 hard-v2/soft-v3 endpoint covariances."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from common import NX, SOFT_MODE_COUNT, spectral_snapshot, verify_completion_pair


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = (
    ROOT
    / "00_WORKSPACE/CURRENT/final_production_new_designs/07_maxmix_hard_soft_purification/gpu_data"
)
HARD_ROOT = DATA_ROOT / "maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v2/hard"
SOFT_ROOT = DATA_ROOT / "maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v3/soft"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "analysis_outputs/kraus_boundary_contours_v1/purification_endpoints.npz"
)
NY_VALUES = (20, 30, 40)
CONSTRUCTIONS = ("hard", "soft")
SAMPLES = 100
SCHEMA = "kraus_boundary_purification_endpoints_v1"


def result_files() -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for construction, root in (("hard", HARD_ROOT), ("soft", SOFT_ROOT)):
        for ny in NY_VALUES:
            files = sorted((root / f"Ny{ny:03d}").glob("*.npz"))
            if len(files) != 20:
                raise RuntimeError(
                    f"expected 20 {construction} Ny={ny} result files, found {len(files)}"
                )
            rows.extend((construction, path) for path in files)
    return rows


def analyze_shard(payload: tuple[str, str, int]) -> dict[str, Any]:
    construction, path_text, blas_threads = payload
    path = Path(path_text)
    with threadpool_limits(limits=max(1, int(blas_threads))):
        with np.load(path, allow_pickle=False) as data:
            nx = int(data["Nx"].item())
            ny = int(data["Ny"].item())
            if nx != NX or ny not in NY_VALUES:
                raise RuntimeError(f"unexpected geometry in {path}")
            if str(data["construction"].item()) != construction:
                raise RuntimeError(f"construction mismatch in {path}")
            samples = np.asarray(data["sample_indices"], dtype=np.int64)
            covariances = np.asarray(data["G_final"], dtype=np.complex128)
            endpoint_occupations = np.asarray(
                data["occupation_spectrum"][:, -1], dtype=np.float64
            )
            entropy_x = np.asarray(
                data["entropy_contour"][:, -1].sum(axis=-1), dtype=np.float64
            )
            cumulative_log_probability = np.asarray(
                data["cumulative_log_probability"][:, -1], dtype=np.float64
            )
            transfer_mode_count = int(data["transfer_mode_count"].item())
            revision = str(data["sampling_revision"].item())

        count = samples.size
        spectral_x = np.empty((count, NX), dtype=np.float64)
        soft_costs = np.empty((count, SOFT_MODE_COUNT), dtype=np.float64)
        soft_profiles = np.empty((count, SOFT_MODE_COUNT, NX), dtype=np.float64)
        soft_wall_weights = np.empty((count, SOFT_MODE_COUNT, 2), dtype=np.float64)
        occupation_error = np.empty(count, dtype=np.float64)
        spectral_total = np.empty(count, dtype=np.float64)
        cap_count = np.empty(count, dtype=np.int64)
        hermiticity = np.empty(count, dtype=np.float64)
        bound_residual = np.empty(count, dtype=np.float64)
        for offset in range(count):
            snapshot = spectral_snapshot(covariances[offset], nx=nx, ny=ny)
            occupation_error[offset] = float(
                np.max(
                    np.abs(
                        np.asarray(snapshot["occupations"])
                        - endpoint_occupations[offset]
                    )
                )
            )
            spectral_x[offset] = snapshot["spectral_x"]
            spectral_total[offset] = snapshot["spectral_total"]
            soft_costs[offset] = snapshot["soft_costs"]
            soft_profiles[offset] = snapshot["soft_profiles"]
            soft_wall_weights[offset] = snapshot["soft_wall_weights"]
            cap_count[offset] = snapshot["cap_count"]
            hermiticity[offset] = snapshot["hermiticity_residual"]
            bound_residual[offset] = snapshot["occupation_bound_residual"]
        if float(np.max(occupation_error)) > 2.0e-10:
            raise FloatingPointError(f"stored occupation mismatch in {path}")
        log_z = transfer_mode_count * np.log(2.0) + cumulative_log_probability
        return {
            "construction": construction,
            "ny": ny,
            "revision": revision,
            "sample_indices": samples,
            "spectral_x": spectral_x,
            "spectral_total": spectral_total,
            "soft_costs": soft_costs,
            "soft_profiles": soft_profiles,
            "soft_wall_weights": soft_wall_weights,
            "entropy_x": entropy_x,
            "log_z": log_z,
            "ell0": log_z + spectral_total,
            "occupation_error": occupation_error,
            "cap_count": cap_count,
            "hermiticity": hermiticity,
            "bound_residual": bound_residual,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--blas-threads", type=int, default=2)
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args()
    files = result_files()
    if not args.skip_checksums:
        for _, path in tqdm(files, desc="verify purification files", unit="file"):
            verify_completion_pair(path)

    shape = (len(CONSTRUCTIONS), len(NY_VALUES), SAMPLES)
    arrays = {
        "spectral_x": np.full(shape + (NX,), np.nan),
        "spectral_total": np.full(shape, np.nan),
        "soft_costs": np.full(shape + (SOFT_MODE_COUNT,), np.nan),
        "soft_profiles": np.full(shape + (SOFT_MODE_COUNT, NX), np.nan),
        "soft_wall_weights": np.full(shape + (SOFT_MODE_COUNT, 2), np.nan),
        "entropy_x": np.full(shape + (NX,), np.nan),
        "log_z": np.full(shape, np.nan),
        "ell0": np.full(shape, np.nan),
        "occupation_error": np.full(shape, np.nan),
        "cap_count": np.full(shape, -1, dtype=np.int64),
        "hermiticity": np.full(shape, np.nan),
        "bound_residual": np.full(shape, np.nan),
    }
    revisions: dict[str, str] = {}
    payloads = [(c, str(p), args.blas_threads) for c, p in files]
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [executor.submit(analyze_shard, payload) for payload in payloads]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="endpoint eigensystems", unit="shard"
        ):
            row = future.result()
            ci = CONSTRUCTIONS.index(row["construction"])
            ni = NY_VALUES.index(row["ny"])
            sample_indices = row["sample_indices"]
            revisions[row["construction"]] = row["revision"]
            for key in arrays:
                arrays[key][ci, ni, sample_indices] = row[key]

    for key, value in arrays.items():
        if np.issubdtype(value.dtype, np.floating) and not np.all(np.isfinite(value)):
            raise RuntimeError(f"nonfinite or missing output in {key}")
        if np.issubdtype(value.dtype, np.integer) and np.any(value < 0):
            raise RuntimeError(f"missing integer output in {key}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema=np.asarray(SCHEMA),
        constructions=np.asarray(CONSTRUCTIONS),
        ny_values=np.asarray(NY_VALUES, dtype=np.int64),
        revisions_json=np.asarray(json.dumps(revisions, sort_keys=True)),
        **arrays,
    )
    summary = {
        "schema": SCHEMA,
        "result_files": len(files),
        "trajectories": int(np.prod(shape)),
        "revisions": revisions,
        "maximum_occupation_reconstruction_error": float(
            np.max(arrays["occupation_error"])
        ),
        "maximum_hermiticity_residual": float(np.max(arrays["hermiticity"])),
        "maximum_occupation_bound_residual": float(
            np.max(arrays["bound_residual"])
        ),
        "median_sample_mean_first_four_soft_wall_weight": {
            construction: {
                str(ny): float(
                    np.median(
                        arrays["soft_wall_weights"][ci, ni, :, :4]
                        .sum(axis=-1)
                        .mean(axis=-1)
                    )
                )
                for ni, ny in enumerate(NY_VALUES)
            }
            for ci, construction in enumerate(CONSTRUCTIONS)
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[done] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
