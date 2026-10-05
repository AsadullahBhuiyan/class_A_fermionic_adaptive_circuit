#!/usr/bin/env python3
"""Reanalyze a completed reference pilot from retained per-cycle raw observables."""

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

from run_cpu_pilot import analyze_size, make_figure, save_npz, sha256, utc_now, write_json


def load_payload(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as raw:
        return {key: np.asarray(raw[key]) for key in raw.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    raw_index = json.loads((output / "raw_index.json").read_text())
    previous = json.loads((output / "results.json").read_text())
    backup_results = output / "results_permissive_crossing_diagnostic.json"
    backup_summary = output / "SUMMARY_permissive_crossing_diagnostic.md"
    if not backup_results.exists():
        shutil.copy2(output / "results.json", backup_results)
    if not backup_summary.exists():
        shutil.copy2(output / "SUMMARY.md", backup_summary)

    size_payloads: dict[int, dict[int, dict[str, Any]]] = {
        int(ny): {} for ny in manifest["sizes_ny"]
    }
    for row in raw_index:
        size_payloads[int(row["ny"])][int(row["separation"])] = load_payload(
            output / row["raw_file"]
        )

    analyses = []
    for size_index, ny in enumerate(manifest["sizes_ny"]):
        analysis = analyze_size(
            ny=int(ny),
            payloads=size_payloads[int(ny)],
            bootstrap_draws=int(manifest["bootstrap_draws"]),
            bootstrap_seed=int(manifest["root_seed"]) + 1000 * (size_index + 1),
        )
        bootstrap_path = output / f"bootstrap_Ny{ny}.npz"
        save_npz(
            bootstrap_path,
            time_star=analysis.pop("bootstrap_time_star"),
            alpha=analysis.pop("bootstrap_alpha"),
        )
        analysis["bootstrap_file"] = bootstrap_path.name
        analysis["bootstrap_sha256"] = sha256(bootstrap_path)
        analyses.append(analysis)

    make_figure(output=output, size_payloads=size_payloads, analyses=analyses)
    reanalysis = {
        "utc": utc_now(),
        "revision": "ordered_downward_crossing_v2",
        "reason": "reject a late crossing created only by an upward sampling fluctuation",
        "analysis_source_sha256": sha256(EXPERIMENT / "reference_probe.py"),
    }
    total_time = previous.get("total_wall_time_seconds")
    write_json(
        output / "results.json",
        {
            "config": manifest,
            "reanalysis": reanalysis,
            "sizes": analyses,
            "raw_index": raw_index,
            "total_wall_time_seconds": total_time,
        },
    )

    lines = [
        "# Gaussian reference-ancilla anisotropy CPU pilot",
        "",
        "**Outcome: the exact Gaussian reference probe is feasible, but four trajectories do not calibrate alpha.**",
        "",
        "| L | I_space | t* | alpha | bootstrap resolved | estimated S for 20% spatial SEM |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for analysis in analyses:
        fmt = lambda value: "unresolved" if value is None else f"{value:.4g}"
        estimate = analysis["sampling_diagnostics"]["spatial_samples_estimated_for_target"]
        lines.append(
            f"| {analysis['ny']} | {analysis['spatial_mean']:.4g} | {fmt(analysis['time_star'])} | "
            f"{fmt(analysis['alpha'])} | {analysis['bootstrap_resolved_fraction']:.3f} | {estimate} |"
        )
    lines.extend(
        [
            "",
            "The `L=6` crossing is a candidate only: its 95% bootstrap alpha interval is broad and the final plateau window still shifts by about one standard error.",
            "The `L=8` mean temporal curve starts below the spatial target at `delta_tau=1`, rises at `delta_tau=2` because of one rare trajectory, and then falls. The ordered CFT crossing rule therefore marks it unresolved.",
            "",
            "Plateau values average the final `L` cycles of a `2L` post-insertion follow. Complete trajectories are the bootstrap units.",
            "The raw reference entropies and mutual information are retained for every sample and every follow cycle.",
            "The earlier permissive interpolation is preserved only as a diagnostic artifact and is not an accepted alpha estimate.",
            "",
            f"Canonical dynamics wall time: {total_time:.1f} s.",
        ]
    )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(output / "SUMMARY.md")


if __name__ == "__main__":
    main()
