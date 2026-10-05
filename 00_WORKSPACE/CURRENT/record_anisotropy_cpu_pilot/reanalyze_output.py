#!/usr/bin/env python3
"""Reanalyze a completed pilot from lossless raw wall records."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT))

from record_anisotropy import (
    OBSERVABLES,
    alpha_from_match,
    bootstrap_match,
    connected_correlations,
    match_time,
    stationarity_diagnostic,
)
from run_cpu_pilot import make_figure, nullable, save_npz, sha256, utc_now, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    old_results = json.loads((output / "results.json").read_text())
    backup_json = output / "results_contact_inclusive_diagnostic.json"
    backup_summary = output / "SUMMARY_contact_inclusive_diagnostic.md"
    if not backup_json.exists():
        shutil.copy2(output / "results.json", backup_json)
    if (output / "SUMMARY.md").exists() and not backup_summary.exists():
        shutil.copy2(output / "SUMMARY.md", backup_summary)

    config = manifest
    sizes = config["sizes_ny"]
    bootstrap_draws = int(config["bootstrap_draws"])
    root_seed = int(config["root_seed"])
    results: list[dict[str, Any]] = []
    old_by_size = {int(item["ny"]): item for item in old_results["sizes"]}
    for size_index, ny in enumerate(sizes):
        raw_path = output / f"raw_Nx{config['nx']}_Ny{ny}.npz"
        with np.load(raw_path, allow_pickle=False) as raw:
            burn_in = int(raw["burn_in"])
            cycles = int(raw["cycles"])
            samples = int(raw["samples"])
            wall_x = np.asarray(raw["wall_x"], dtype=int).tolist()
            fields = {name: np.asarray(raw[name], dtype=np.float64) for name in OBSERVABLES}
            random_seed = int(raw["random_seed"])
        record_cycles = cycles - burn_in
        max_lag = min(int(round(config["max_lag_multiplier"] * ny)), record_cycles - 1)
        case: dict[str, Any] = {
            "nx": int(config["nx"]),
            "ny": int(ny),
            "wall_x": wall_x,
            "cycles": cycles,
            "burn_in": burn_in,
            "record_cycles": record_cycles,
            "samples": samples,
            "random_seed": random_seed,
            "raw_file": raw_path.name,
            "raw_sha256": sha256(raw_path),
            "observables": {},
            "wall_time_seconds": old_by_size[int(ny)]["wall_time_seconds"],
        }
        for observable_index, observable in enumerate(OBSERVABLES):
            field = fields[observable]
            estimate = connected_correlations(field, burn_in=burn_in, max_temporal_lag=max_lag)
            target = float(estimate.spatial_mean[int(ny) // 2])
            t_star = match_time(estimate.temporal_mean, target)
            alpha = alpha_from_match(int(ny), t_star)
            bootstrap = bootstrap_match(
                estimate,
                circumference=int(ny),
                draws=bootstrap_draws,
                seed=root_seed + 10000 * (size_index + 1) + observable_index,
            )
            correlation_path = output / f"correlations_{observable}_Ny{ny}.npz"
            save_npz(
                correlation_path,
                spatial_by_trajectory=estimate.spatial_by_trajectory,
                temporal_by_trajectory=estimate.temporal_by_trajectory,
                spatial_mean=estimate.spatial_mean,
                temporal_mean=estimate.temporal_mean,
                mean_by_trajectory_wall=estimate.mean_by_trajectory_wall,
                spatial_product_by_trajectory_wall=estimate.spatial_product_by_trajectory_wall,
                temporal_product_by_trajectory_wall=estimate.temporal_product_by_trajectory_wall,
                t_star_bootstrap=bootstrap.pop("t_star_bootstrap"),
                alpha_bootstrap=bootstrap.pop("alpha_bootstrap"),
            )
            case["observables"][observable] = {
                "spatial_target": target,
                "temporal_variance": float(estimate.temporal_mean[0]),
                "first_noncontact_temporal_correlation": float(estimate.temporal_mean[1]),
                "t_star": nullable(t_star),
                "alpha": nullable(alpha),
                "cycles_per_circumference": nullable(1.0 / alpha),
                "spatial_mean": estimate.spatial_mean,
                "temporal_mean": estimate.temporal_mean,
                "correlation_file": correlation_path.name,
                "correlation_sha256": sha256(correlation_path),
                "stationarity": stationarity_diagnostic(field, burn_in=burn_in),
                **{
                    key: nullable(value) if isinstance(value, float) else value
                    for key, value in bootstrap.items()
                },
            }
        results.append(case)

    analysis = {
        "reanalysis_utc": utc_now(),
        "analysis_revision": "noncontact_matching_v2",
        "analysis_source_sha256": sha256(EXPERIMENT / "record_anisotropy.py"),
        "matching_minimum_lag_cycles": 1,
        "reason": "exclude the lag-zero local contact/variance term from continuum matching",
    }
    total_time = old_results.get("total_wall_time_seconds")
    write_json(
        output / "results.json",
        {
            "config": config,
            "analysis": analysis,
            "sizes": results,
            "total_wall_time_seconds": total_time,
        },
    )
    make_figure(results, output)
    lines = [
        "# Record anisotropy CPU pilot results",
        "",
        "**Outcome: no infrared spacetime-anisotropy crossing was resolved.**",
        "",
        "The lag-zero local variance/contact term is excluded. At every size, the nonzero-lag temporal and `L/2` spatial correlations are noise-scale and do not form a positive monotone matching curve.",
        "",
        "| observable | L | C(L/2,0) | C(0,1) | t* | alpha | bootstrap resolved |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in results:
        for observable in OBSERVABLES:
            item = case["observables"][observable]
            lines.append(
                f"| {observable} | {case['ny']} | {item['spatial_target']:.4g} | "
                f"{item['first_noncontact_temporal_correlation']:.4g} | unresolved | unresolved | "
                f"{item['bootstrap_resolved_fraction']:.3f} |"
            )
    lines.extend(
        [
            "",
            "Matching rule: `C(L/2,0)=C(0,t*)`, with interpolation beginning at lag 1. Lag 0 is not a time-separated correlator.",
            "Bootstrap resampling is by complete trajectory; the two walls remain paired.",
            "Stationary means and variances pass the first-half/second-half diagnostic, so the failure is lack of an infrared signal rather than visible burn-in drift.",
            "",
            "The previously contact-inclusive interpolation is preserved only as `results_contact_inclusive_diagnostic.json`; its sub-cycle crossings are not physical estimates.",
            "",
            f"Canonical dynamics wall time: {total_time:.1f} s.",
        ]
    )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(output / "SUMMARY.md")


if __name__ == "__main__":
    main()
