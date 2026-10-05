#!/usr/bin/env python3
"""Make Lane-B wall-window entropy-collapse figures."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "make_half_entropy_collapse.py"
MANIFEST_PATH = HERE / "wall_only_analysis_manifest.json"


def load_base() -> Any:
    name = "_hard_wall_all_ay_half_entropy_base"
    spec = importlib.util.spec_from_file_location(name, BASE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    base = load_base()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--window",
        choices=("three-cell", "two-cell", "center"),
        default="three-cell",
        help="x cells included around each domain wall",
    )
    args = parser.parse_args()
    if args.window == "center":
        left_x = np.asarray((5,), dtype=np.int64)
        right_x = np.asarray((15,), dtype=np.int64)
        x_labels = {"left": "5", "right": "15"}
        x_math_labels = {"left": r"$x=5$", "right": r"$x=15$"}
        figure_stem = base.FIGURE_DIR / "lane_B_wall_center_only_entropy_collapse_2x1"
        curve_csv = base.DATA_DIR / "lane_B_wall_center_only_anchored_curves.csv"
        fit_csv = base.DATA_DIR / "lane_B_wall_center_only_joint_fits.csv"
        manifest_path = HERE / "wall_center_only_analysis_manifest.json"
        manifest_schema = "hard_wall_lane_B_wall_center_only_entropy_collapse_v1"
    elif args.window == "two-cell":
        left_x = np.asarray((5, 6), dtype=np.int64)
        right_x = np.asarray((14, 15), dtype=np.int64)
        x_labels = {"left": "5,6", "right": "14,15"}
        x_math_labels = {"left": r"$x=5,6$", "right": r"$x=14,15$"}
        figure_stem = base.FIGURE_DIR / "lane_B_wall_two_cell_entropy_collapse_2x1"
        curve_csv = base.DATA_DIR / "lane_B_wall_two_cell_anchored_curves.csv"
        fit_csv = base.DATA_DIR / "lane_B_wall_two_cell_joint_fits.csv"
        manifest_path = HERE / "wall_two_cell_analysis_manifest.json"
        manifest_schema = "hard_wall_lane_B_wall_two_cell_entropy_collapse_v1"
    else:
        left_x = base.LEFT_WALL_X
        right_x = base.RIGHT_WALL_X
        x_labels = {"left": "4,5,6", "right": "14,15,16"}
        x_math_labels = {"left": r"$x=4,5,6$", "right": r"$x=14,15,16$"}
        figure_stem = base.WALL_FIGURE_STEM
        curve_csv = base.WALL_CURVE_CSV
        fit_csv = base.WALL_FIT_CSV
        manifest_path = MANIFEST_PATH
        manifest_schema = "hard_wall_lane_B_wall_only_entropy_collapse_v1"

    def integrate_selected(case: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        ay_values = np.asarray(case["ay_values"], dtype=np.int64)
        contour = np.asarray(case[base.CONTOUR_KEY], dtype=np.float64)
        left = np.zeros((base.SAMPLES, ay_values.size), dtype=np.float64)
        right = np.zeros_like(left)
        for index, ay in enumerate(ay_values):
            active = contour[:, index, :, :ay]
            left[:, index] = active[:, left_x, :].sum(axis=(1, 2))
            right[:, index] = active[:, right_x, :].sum(axis=(1, 2))
        full = np.asarray(case[base.ENTROPY_KEY], dtype=np.float64)
        if np.any(left + right > full + 2.0e-8):
            raise RuntimeError("selected wall entropy exceeds the full-strip entropy")
        return {"left": left, "right": right}

    cases, closure_max, inputs = base.load_lane_b()
    fits = {
        component: base.anchored_joint_fit(cases, component, integrate_selected)
        for component in ("left", "right")
    }
    denominator = sum(
        (1.0 / fits["left"][0][ny]["x_fit"].size)
        * float(fits["left"][0][ny]["x_fit"] @ fits["left"][0][ny]["x_fit"])
        for ny in base.NY_VALUES
    )
    left_right_covariance = 0.0
    for ny in base.NY_VALUES:
        left_item = fits["left"][0][ny]
        right_item = fits["right"][0][ny]
        x_fit = left_item["x_fit"]
        projection = (1.0 / x_fit.size) * x_fit / denominator
        left_contribution = left_item["delta_fit_samples"] @ projection
        right_contribution = right_item["delta_fit_samples"] @ projection
        left_right_covariance += float(
            np.cov(left_contribution, right_contribution, ddof=1)[0, 1]
            / base.SAMPLES
        )
    left_slope = fits["left"][1]["slope"]
    right_slope = fits["right"][1]["slope"]
    left_variance = fits["left"][1]["slope_covariance_sem"] ** 2
    right_variance = fits["right"][1]["slope_covariance_sem"] ** 2
    paired_slope = 0.5 * (left_slope + right_slope)
    paired_sem = float(
        np.sqrt(
            max(
                0.0,
                0.25
                * (
                    left_variance
                    + right_variance
                    + 2.0 * left_right_covariance
                ),
            )
        )
    )
    paired_summary = {
        "slope": paired_slope,
        "slope_covariance_SEM": paired_sem,
        "converted_abs_chiral_central_charge": 6.0 * paired_slope,
        "converted_abs_chiral_central_charge_SEM": 6.0 * paired_sem,
        "left_right_covariance": left_right_covariance,
        "relative_error_from_one_sixth_percent": (
            abs(paired_slope - base.SLOPE_TARGET) / base.SLOPE_TARGET * 100.0
        ),
    }


    curve_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for component, (by_size, summary) in fits.items():
        for ny in base.NY_VALUES:
            item = by_size[ny]
            for ay, x, mean, sem in zip(
                item["ay_all"], item["x_all"], item["mean_all"], item["sem_all"]
            ):
                curve_rows.append(
                    {
                        "wall": component,
                        "x_cells": x_labels[component],
                        "Nx": base.NX,
                        "Ny": ny,
                        "samples": base.SAMPLES,
                        "Ay": int(ay),
                        "anchor_Ay": ny // 2,
                        "delta_log_sine_chord": float(x),
                        "anchored_entropy_mean": float(mean),
                        "anchored_entropy_trajectory_SEM": float(sem),
                        "in_fit_window": bool(ay >= base.FIT_MIN_AY),
                    }
                )
        fit_rows.append(
            {
                "wall": component,
                "x_cells": x_labels[component],
                "Ny_values": ",".join(map(str, base.NY_VALUES)),
                "samples_per_Ny": base.SAMPLES,
                "Ay_fit_min": base.FIT_MIN_AY,
                "Ay_fit_max": "Ny//2",
                "estimator_order": "anchor_each_trajectory_then_average_then_joint_fit",
                "size_weighting": "equal_total_weight_per_Ny",
                "slope": summary["slope"],
                "slope_covariance_SEM": summary["slope_covariance_sem"],
                "slope_target": base.SLOPE_TARGET,
                "R0_squared": summary["R0_squared"],
                "uncertainty": "ordinary_trajectory_SEM_full_Ay_covariance",
            }
        )

    base.write_csv(curve_csv, curve_rows)
    base.write_csv(fit_csv, fit_rows)
    base.make_figure(
        fits,
        figure_stem,
        {
            "left": (
                "left wall",
                x_math_labels["left"],
                r"m_{\rm L}",
                r"\overline{S}^{\rm wall}_{\rm L}",
            ),
            "right": (
                "right wall",
                x_math_labels["right"],
                r"m_{\rm R}",
                r"\overline{S}^{\rm wall}_{\rm R}",
            ),
        },
        wall_window_cells={"left": left_x, "right": right_x},
        inset_right_wall_at_cell_edge=(args.window == "two-cell"),
        plot_min_ay=2 if args.window == "two-cell" else 1,
    )

    outputs = [
        figure_stem.with_suffix(".pdf"),
        figure_stem.with_suffix(".png"),
        curve_csv,
        fit_csv,
    ]
    manifest = {
        "schema": manifest_schema.replace("_v1", "_v2"),
        "campaign": base.CAMPAIGN,
        "input_scope": "completed_lane_B_only",
        "Nx": base.NX,
        "Ny_values": list(base.NY_VALUES),
        "samples_per_Ny": base.SAMPLES,
        "trajectory_count": base.SAMPLES * len(base.NY_VALUES),
        "endpoint": "t=2Ny",
        "display_min_Ay": 2 if args.window == "two-cell" else 1,
        "contour_semantics": (
            "exact periodic-y0 average within each trajectory in relative-dy coordinates"
        ),
        "wall_windows": {
            "left": left_x.tolist(),
            "right": right_x.tolist(),
        },
        "figure_insets": {
            "geometry": "20x30 unit-cell grid",
            "domain_walls_x": [5, 15],
            "highlight": "the x-cell integration window used in each panel",
            "right_panel_xR_marker": (
                "right edge of cell x=15 (grid coordinate 16); display only"
                if args.window == "two-cell" else "grid coordinate 15"
            ),
        },
        "integration": "sum over wall-window x cells and all valid relative-dy cells",
        "fit": {
            "Ay": "8..Ny//2",
            "anchor": "each trajectory at Ay*=Ny//2",
            "cross_size_weighting": "equal total weight per Ny",
            "intercept": 0.0,
            "target_slope": base.SLOPE_TARGET,
            "uncertainty": (
                "ordinary trajectory SEM propagated with full within-trajectory Ay covariance"
            ),
            "bootstrap": False,
        },
        "contour_closure_max_abs": closure_max,
        "verified_result_receipt_pairs": len(inputs) // 2,
        "fits": {
            component: summary for component, (_, summary) in fits.items()
        },
        "paired_wall_average": paired_summary,
        "outputs": {
            path.relative_to(HERE).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": base.sha256_file(path),
            }
            for path in outputs
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
