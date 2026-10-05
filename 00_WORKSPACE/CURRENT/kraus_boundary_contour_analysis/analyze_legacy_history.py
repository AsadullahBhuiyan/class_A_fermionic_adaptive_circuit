#!/usr/bin/env python3
"""Construct time-resolved spectral and gap contours from the legacy history."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from common import NX, SOFT_MODE_COUNT, sha256_file, spectral_snapshot


ROOT = Path(__file__).resolve().parents[3]
HISTORY_PATH = ROOT / (
    "cache/G_history_samples/N20x20/"
    "N20x20_C40_S10_nsh1_DW1_alpha_top1.0_alpha_triv30.0_trial-X_"
    "dwtrunc1_init-maxmix_n_a0.5_seq-raster_y_exclNone_ps0_psp0_pc1_"
    "mslab1_markov_circuit_n20x20_dwtrunc_convergence_cpu.npz"
)
EXPECTED_SHA256 = "1cee7090b85c8e79b6ca389bb55a63cc9a6412fa8e3992e5453810a75ce333ea"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "analysis_outputs/kraus_boundary_contours_v1/legacy_time_contours.npz"
)
SCHEMA = "kraus_boundary_legacy_time_contours_v1"
SAMPLES = 10
NY = 20
CYCLES = 40


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--blas-threads", type=int, default=4)
    parser.add_argument("--skip-checksum", action="store_true")
    args = parser.parse_args()
    if not args.skip_checksum:
        actual = sha256_file(HISTORY_PATH)
        if actual != EXPECTED_SHA256:
            raise RuntimeError(f"legacy history SHA-256 mismatch: {actual}")

    with np.load(HISTORY_PATH, allow_pickle=False) as data:
        run_config = str(data["run_config"].item())
        history = np.asarray(data["G_hist"], dtype=np.complex128)
    expected_shape = (SAMPLES, CYCLES + 1, 2 * NX * NY, 2 * NX * NY)
    if history.shape != expected_shape:
        raise RuntimeError(f"legacy history shape {history.shape} != {expected_shape}")

    base_shape = (SAMPLES, CYCLES + 1)
    arrays = {
        "spectral_x": np.full(base_shape + (NX,), np.nan),
        "spectral_total": np.full(base_shape, np.nan),
        "soft_costs": np.full(base_shape + (SOFT_MODE_COUNT,), np.nan),
        "soft_profiles": np.full(base_shape + (SOFT_MODE_COUNT, NX), np.nan),
        "soft_wall_weights": np.full(base_shape + (SOFT_MODE_COUNT, 2), np.nan),
        "cap_count": np.full(base_shape, -1, dtype=np.int64),
        "hermiticity": np.full(base_shape, np.nan),
        "bound_residual": np.full(base_shape, np.nan),
    }

    def analyze_one(sample: int, cycle: int):
        return sample, cycle, spectral_snapshot(
            history[sample, cycle], nx=NX, ny=NY
        )

    # threadpoolctl changes process-global BLAS state.  Applying it independently
    # inside concurrent workers can deadlock some OpenBLAS builds, so establish
    # one limit around the entire executor instead.
    with threadpool_limits(limits=max(1, int(args.blas_threads))):
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = [
                executor.submit(analyze_one, sample, cycle)
                for sample in range(SAMPLES)
                for cycle in range(CYCLES + 1)
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="legacy eigensystems", unit="snapshot"
            ):
                sample, cycle, snapshot = future.result()
                arrays["spectral_x"][sample, cycle] = snapshot["spectral_x"]
                arrays["spectral_total"][sample, cycle] = snapshot["spectral_total"]
                arrays["soft_costs"][sample, cycle] = snapshot["soft_costs"]
                arrays["soft_profiles"][sample, cycle] = snapshot["soft_profiles"]
                arrays["soft_wall_weights"][sample, cycle] = snapshot[
                    "soft_wall_weights"
                ]
                arrays["cap_count"][sample, cycle] = snapshot["cap_count"]
                arrays["hermiticity"][sample, cycle] = snapshot[
                    "hermiticity_residual"
                ]
                arrays["bound_residual"][sample, cycle] = snapshot[
                    "occupation_bound_residual"
                ]
    del history

    for key, value in arrays.items():
        if np.issubdtype(value.dtype, np.floating):
            if np.any(np.isnan(value)):
                raise RuntimeError(f"NaN output in {key}")
            if key != "soft_costs" and not np.all(np.isfinite(value)):
                raise RuntimeError(f"nonfinite output in {key}")
        if np.issubdtype(value.dtype, np.integer) and np.any(value < 0):
            raise RuntimeError(f"missing integer output in {key}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema=np.asarray(SCHEMA),
        source_sha256=np.asarray(EXPECTED_SHA256),
        run_config=np.asarray(run_config),
        cycles=np.arange(CYCLES + 1, dtype=np.int64),
        **arrays,
    )
    aggregate_wall_weight = arrays["soft_wall_weights"][..., :4, :].sum(axis=(-1, -2)) / 4.0
    summary = {
        "schema": SCHEMA,
        "samples": SAMPLES,
        "cycles": CYCLES,
        "snapshots": SAMPLES * (CYCLES + 1),
        "source_sha256": EXPECTED_SHA256,
        "maximum_hermiticity_residual": float(np.max(arrays["hermiticity"])),
        "maximum_occupation_bound_residual": float(
            np.max(arrays["bound_residual"])
        ),
        "mean_first_four_soft_wall_weight_by_cycle": np.mean(
            aggregate_wall_weight, axis=0
        ).tolist(),
        "endpoint_mean_first_four_soft_wall_weight": float(
            np.mean(aggregate_wall_weight[:, -1])
        ),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[done] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
