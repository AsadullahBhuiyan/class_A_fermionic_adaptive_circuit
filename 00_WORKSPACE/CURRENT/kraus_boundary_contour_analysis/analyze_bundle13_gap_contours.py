#!/usr/bin/env python3
"""Reconstruct wall-resolved many-body gap contours for all bundle-13 data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from common import (
    LEADING_LEVEL_COUNT,
    NX,
    WALL_CELLS,
    fit_slope,
    leading_subset_sums,
    mask_gap_contour,
    verify_completion_pair,
)


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / (
    "00_WORKSPACE/CURRENT/final_production_new_designs/13_maxmix_manybody_lyapunov_4ny/"
    "gpu_data/maxmix_manybody_lyapunov_nx20_ny20-60_hard-soft_s100_4ny_gpu_v4_38gib_memory_scaled/results"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "analysis_outputs/kraus_boundary_contours_v1/bundle13_gap_contours.npz"
)
NY_VALUES = (20, 24, 30, 36, 44, 56, 60)
SAMPLES = 100
GAPS = 8
MAX_SPECTRA = max(NY_VALUES) + 1
SCHEMA = "kraus_boundary_bundle13_gap_contours_v1"


def expected_spectrum_cycles(ny: int) -> np.ndarray:
    return np.asarray(
        sorted(set(range(0, 4 * int(ny) + 1, 4)) | {int(ny), 3 * int(ny)}),
        dtype=np.int64,
    )


def files() -> list[Path]:
    paths: list[Path] = []
    for ny in NY_VALUES:
        found = sorted((DATA_ROOT / f"Ny{ny:03d}").glob("*.npz"))
        if len(found) != 20:
            raise RuntimeError(f"expected 20 Ny={ny} shards, found {len(found)}")
        paths.extend(found)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args()
    result_paths = files()
    if not args.skip_checksums:
        for path in tqdm(result_paths, desc="verify bundle-13 files", unit="file"):
            verify_completion_pair(path)

    shape = (len(NY_VALUES), SAMPLES, MAX_SPECTRA)
    spectrum_seen = np.zeros(shape, dtype=bool)
    spectrum_cycles = np.full((len(NY_VALUES), MAX_SPECTRA), -1, dtype=np.int64)
    gap_values = np.full(shape + (GAPS,), np.nan)
    gap_contours = np.full(shape + (GAPS, NX), np.nan)
    gap_wall_fraction = np.full(shape + (GAPS,), np.nan)
    reconstruction_error = np.full(shape, np.nan)
    first_four_soft_wall_weight = np.full(shape, np.nan)
    trajectory_rows: list[dict[str, object]] = []

    for path in tqdm(result_paths, desc="bundle-13 contours", unit="shard"):
        with np.load(path, allow_pickle=False) as data:
            ny = int(data["Ny"].item())
            ni = NY_VALUES.index(ny)
            sample_indices = np.asarray(data["sample_indices"], dtype=np.int64)
            cycles = np.asarray(data["spectrum_cycles"], dtype=np.int64)
            seen = np.asarray(data["spectrum_seen"], dtype=bool)
            costs_all = np.asarray(data["soft_mode_flip_costs"], dtype=np.float64)
            profiles_all = np.asarray(data["soft_mode_x_profiles"], dtype=np.float64)
            weights_all = np.asarray(data["soft_mode_wall_weights"], dtype=np.float64)
            levels_all = np.asarray(data["leading_log_sigma2"], dtype=np.float64)
        count = cycles.size
        if not np.array_equal(cycles, expected_spectrum_cycles(ny)):
            raise RuntimeError(f"unexpected spectrum cycle grid in {path}")
        spectrum_cycles[ni, :count] = cycles
        for local, sample in enumerate(sample_indices):
            if not np.all(seen[local]):
                raise RuntimeError(f"missing spectrum checkpoint in {path}, sample {sample}")
            for ti in range(count):
                costs = costs_all[local, ti]
                profiles = profiles_all[local, ti]
                finite_modes = np.flatnonzero(np.isfinite(costs))
                profile_sums = profiles.sum(axis=1)
                if finite_modes.size and float(
                    np.max(np.abs(profile_sums[finite_modes] - 1.0))
                ) > 2.0e-8:
                    raise FloatingPointError(f"soft profile closure failure in {path}")
                profiles = profiles / profile_sums[:, None]
                generated_finite, local_masks = leading_subset_sums(
                    costs[finite_modes], LEADING_LEVEL_COUNT
                )
                masks = np.zeros(LEADING_LEVEL_COUNT, dtype=np.uint64)
                generated = np.full(LEADING_LEVEL_COUNT, np.inf, dtype=np.float64)
                available = min(LEADING_LEVEL_COUNT, generated_finite.size)
                generated[:available] = generated_finite[:available]
                for level_index in range(available):
                    local_mask = int(local_masks[level_index])
                    global_mask = 0
                    for local_mode, global_mode in enumerate(finite_modes):
                        if local_mask & (1 << local_mode):
                            global_mask |= 1 << int(global_mode)
                    masks[level_index] = np.uint64(global_mask)
                stored = levels_all[local, ti]
                stored_gaps = stored[0] - stored
                if not np.array_equal(np.isfinite(stored_gaps), np.isfinite(generated)):
                    raise FloatingPointError(
                        f"finite-level mask mismatch in {path}, checkpoint {ti}"
                    )
                finite_levels = np.isfinite(generated)
                error = float(
                    np.max(np.abs(stored_gaps[finite_levels] - generated[finite_levels]))
                )
                reconstruction_error[ni, sample, ti] = error
                if error > 2.0e-8:
                    raise FloatingPointError(
                        f"many-body level reconstruction error {error:.3e} in {path}"
                    )
                spectrum_seen[ni, sample, ti] = True
                first_four_soft_wall_weight[ni, sample, ti] = float(
                    np.mean(weights_all[local, ti, :4].sum(axis=-1))
                )
                for gap in range(GAPS):
                    level_index = gap + 1
                    if not np.isfinite(generated[level_index]):
                        continue
                    contour = mask_gap_contour(
                        int(masks[level_index]), costs, profiles
                    )
                    value = float(generated[level_index])
                    gap_values[ni, sample, ti, gap] = value
                    gap_contours[ni, sample, ti, gap] = contour
                    gap_wall_fraction[ni, sample, ti, gap] = (
                        float(contour[WALL_CELLS].sum() / value)
                        if value > 0.0
                        else np.nan
                    )

            for gap in range(4):
                values = gap_values[ni, sample, :count, gap]
                row = {
                    "Ny": ny,
                    "sample_index": int(sample),
                    "gap_index": gap + 1,
                    "slope_W1_Ny_to_2Ny": fit_slope(cycles, values, ny, 2 * ny),
                    "slope_W2_2Ny_to_3Ny": fit_slope(cycles, values, 2 * ny, 3 * ny),
                    "slope_W3_3Ny_to_4Ny": fit_slope(cycles, values, 3 * ny, 4 * ny),
                    "endpoint_gap": float(values[-1]),
                    "endpoint_wall_fraction": float(
                        gap_wall_fraction[ni, sample, count - 1, gap]
                    ),
                }
                trajectory_rows.append(row)

    valid = spectrum_cycles >= 0
    for ni, ny in enumerate(NY_VALUES):
        count = expected_spectrum_cycles(ny).size
        if not np.all(spectrum_seen[ni, :, :count]):
            raise RuntimeError(f"incomplete reconstructed inventory for Ny={ny}")
        if np.any(spectrum_seen[ni, :, count:]):
            raise RuntimeError(f"unexpected padded checkpoints for Ny={ny}")
    max_error = float(np.nanmax(reconstruction_error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema=np.asarray(SCHEMA),
        ny_values=np.asarray(NY_VALUES, dtype=np.int64),
        spectrum_cycles=spectrum_cycles,
        spectrum_seen=spectrum_seen,
        gap_values=gap_values,
        gap_contours=gap_contours,
        gap_wall_fraction=gap_wall_fraction,
        reconstruction_error=reconstruction_error,
        first_four_soft_wall_weight=first_four_soft_wall_weight,
    )
    csv_path = args.output.with_suffix(".trajectory_slopes.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0]))
        writer.writeheader()
        writer.writerows(trajectory_rows)
    summary = {
        "schema": SCHEMA,
        "result_files": len(result_paths),
        "trajectories": len(NY_VALUES) * SAMPLES,
        "spectrum_checkpoints": int(np.count_nonzero(spectrum_seen)),
        "maximum_level_reconstruction_error": max_error,
        "endpoint_median_first_gap_wall_fraction": {
            str(ny): float(
                np.nanmedian(
                    gap_wall_fraction[
                        ni, :, expected_spectrum_cycles(ny).size - 1, 0
                    ]
                )
            )
            for ni, ny in enumerate(NY_VALUES)
        },
        "endpoint_median_first_four_soft_wall_weight": {
            str(ny): float(
                np.nanmedian(
                    first_four_soft_wall_weight[
                        ni, :, expected_spectrum_cycles(ny).size - 1
                    ]
                )
            )
            for ni, ny in enumerate(NY_VALUES)
        },
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[done] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
