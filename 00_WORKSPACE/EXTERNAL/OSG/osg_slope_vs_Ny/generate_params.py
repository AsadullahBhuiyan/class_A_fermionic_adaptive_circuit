"""
Run locally before submitting. Generates params.json.

Based on real A100 timings (10 samples, cycles=Ny/2):
  Ny=40: 20min, Ny=60: 60min, Ny=80: 150min, Ny=100: 360min, Ny=120: 750min

Usage:
    python generate_params.py --mode full
    python generate_params.py --mode pilot
"""
from __future__ import annotations

import argparse
import json
import math

Ny_measured = [40, 60, 80, 100, 120]
t_measured = [value * 2 for value in [20, 60, 150, 360, 750]]  # minutes, 10 samples, cycles=Ny (2x the Ny/2 timings)

NY_VALUES = [40, 60, 80, 100, 120, 140, 160, 180, 200]
TOTAL_SAMPLES = 100
TARGET_H = 15
DEFAULT_PILOT_NY = [40, 120, 200]


def fit_log_scaling(x_values: list[int], y_values: list[int]) -> tuple[float, float]:
    x_log = [math.log(float(x)) for x in x_values]
    y_log = [math.log(float(y)) for y in y_values]
    x_mean = sum(x_log) / len(x_log)
    y_mean = sum(y_log) / len(y_log)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_log, y_log))
    denominator = sum((x - x_mean) ** 2 for x in x_log)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


alpha, c = fit_log_scaling(Ny_measured, t_measured)


def build_full_params() -> list[dict[str, int]]:
    params: list[dict[str, int]] = []
    for Ny in NY_VALUES:
        t_per_sample = math.exp(c) * Ny**alpha / 10
        max_per_job = max(1, min(int(TARGET_H * 60 / t_per_sample), TOTAL_SAMPLES))
        start = 0
        while start < TOTAL_SAMPLES:
            stop = min(TOTAL_SAMPLES, start + max_per_job)
            params.append({
                "Ny": Ny,
                "block_start": start,
                "block_stop": stop,
                "cycles": Ny,
                "fit_cycle": Ny,
            })
            start = stop
    return params


def build_pilot_params(full_params: list[dict[str, int]], pilot_ny: list[int]) -> list[dict[str, int]]:
    selected: list[dict[str, int]] = []
    for Ny in pilot_ny:
        match = next((p for p in full_params if int(p["Ny"]) == int(Ny)), None)
        if match is None:
            raise ValueError(f"No generated block found for Ny={Ny}")
        selected.append(match)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "pilot"], default="full")
    parser.add_argument("--pilot-ny", nargs="+", type=int, default=DEFAULT_PILOT_NY)
    parser.add_argument("--pilot-pick", choices=["first"], default="first")
    parser.add_argument("--output", default="params.json")
    return parser.parse_args()


def print_full_summary(params: list[dict[str, int]], output_path: str) -> None:
    print(f"Wrote {len(params)} jobs to {output_path}")
    for Ny in NY_VALUES:
        jobs = [p for p in params if p["Ny"] == Ny]
        t_per_sample = math.exp(c) * Ny**alpha / 10
        max_s = max(j["block_stop"] - j["block_start"] for j in jobs)
        print(
            f"  Ny={Ny:3d}: {len(jobs):2d} job(s), "
            f"up to {max_s} samples/job, "
            f"~{max_s * t_per_sample / 60:.1f}h/job"
        )


def print_pilot_summary(params: list[dict[str, int]], output_path: str) -> None:
    print(f"Wrote {len(params)} pilot jobs to {output_path}")
    for i, p in enumerate(params):
        print(
            f"  [{i}] Ny={p['Ny']} block={p['block_start']}:{p['block_stop']} "
            f"cycles={p['cycles']} fit_cycle={p['fit_cycle']}"
        )


def main() -> None:
    args = parse_args()
    full_params = build_full_params()
    if args.mode == "pilot":
        params = build_pilot_params(full_params, args.pilot_ny)
    else:
        params = full_params

    with open(args.output, "w") as f:
        json.dump(params, f, indent=2)

    if args.mode == "pilot":
        print_pilot_summary(params, args.output)
    else:
        print_full_summary(params, args.output)
    print(f"JOB_COUNT={len(params)}")


if __name__ == "__main__":
    main()
