#!/usr/bin/env python3
"""Verify the hard-wall campaign and render its MI panel for the report.

The plotting and validation routines are imported from the owning production
bundle so the technical report uses exactly the same plotting engine and never
rewrites the checksum-bound trajectory data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent
PLOT_ENGINE_PATH = (
    REPOSITORY_ROOT
    / "00_WORKSPACE"
    / "CURRENT"
    / "final_production_new_designs"
    / "02_domain_wall_bipartite_mutual_information"
    / "make_hard_wall_figures.py"
)
FIGURE_DIR = HERE / "figures"


def _load_plot_engine() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "domain_wall_bipartite_mutual_information_figures", PLOT_ENGINE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load plotting engine: {PLOT_ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        help="Override the production bundle's default checksum-bound input tree.",
    )
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = _load_plot_engine()
    input_root = args.input_root or engine.DEFAULT_INPUT_ROOT
    manifest = engine.verify_download_manifest(input_root)
    data, validation = engine.load_verified_hard_wall(input_root)
    summary = engine.summarize(data)
    report = {
        "status": "verified",
        "plot_engine": str(PLOT_ENGINE_PATH),
        "input_root": str(Path(input_root).resolve()),
        "manifest": manifest,
        "validation": validation,
        "sample_mean_shape": list(summary["mean"].shape),
    }
    if not args.check_only:
        report["figures"] = engine.make_report_inset_figure(summary, FIGURE_DIR)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
