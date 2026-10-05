#!/usr/bin/env python3
"""Reconstruct compact deterministic data behind hybrid Atlas Result 24."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CPU_CAMPAIGN = ROOT / "00_WORKSPACE/CURRENT/mean_channel_lindblad_cpu_campaign"
if str(CPU_CAMPAIGN) not in sys.path:
    sys.path.insert(0, str(CPU_CAMPAIGN))

from mean_channel_lindblad_cpu import run_gain_loss_case, validate_selected_observable_schema
from run_campaign import expand_cases


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(case: dict[str, Any]) -> tuple[int, float, int | None, bool]:
    model = case["model"]
    return (
        int(model["Ny"]), float(model["alpha_top"]), model["nshell"],
        bool(model["dw_truncation"]),
    )


def main() -> int:
    data_dir = HERE / "data"
    data_dir.mkdir(exist_ok=True)
    config_path = CPU_CAMPAIGN / "campaign_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    all_cases = expand_cases(config, nx=int(config["default_Nx"]), smoke=False)
    selected_cases = [case for case in all_cases if case["campaign"] == "L1_MAIN"]
    selected_cases += [
        case for case in all_cases
        if case["campaign"] == "L1_CONTROL" and "uniform-" in case["case_id"]
    ]
    completed: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
    by_key: dict[tuple[int, float, int | None, bool], tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
    uniform: dict[tuple[str, int | None], tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
    for case in selected_cases:
        arrays, metadata = run_gain_loss_case(case)
        validate_selected_observable_schema(case["model"]["nshell"], arrays)
        completed[case["case_id"]] = arrays, metadata
        if case["campaign"] == "L1_MAIN":
            by_key[key(case)] = arrays, metadata
        else:
            phase = "topological" if "uniform-topological" in case["case_id"] else "trivial"
            uniform[(phase, case["model"]["nshell"])] = arrays, metadata

    shells: list[int | None] = [1, 2, None]
    shell_codes = np.asarray([1, 2, -1], dtype=np.int64)
    alpha = np.asarray(config["alpha_in"], dtype=np.float64)
    sizes = np.asarray(config["response_Ny"], dtype=np.int64)
    truncations = [True, False]
    static_ny = int(config["static_scan_Ny"])

    entropy_density = np.empty((len(shells), alpha.size))
    half_gap = np.empty_like(entropy_density)
    bulk_gap = np.empty_like(entropy_density)
    static_wall_profile = np.empty((len(shells), alpha.size, int(config["default_Nx"])))
    branch_slope_alpha = np.full((alpha.size, 2), np.nan)
    residuals = []
    for ai, alpha_value in enumerate(alpha):
        for si, shell in enumerate(shells):
            arrays, metadata = by_key[(static_ny, float(alpha_value), shell, True)]
            diagnostics = metadata["diagnostics"]
            entropy_density[si, ai] = diagnostics["stationary_entropy_per_circumference"]
            half_gap[si, ai] = diagnostics["stationary_half_occupation_gap"]
            bulk_gap[si, ai] = (
                np.nan if diagnostics["bulk_half_occupation_gap"] is None
                else diagnostics["bulk_half_occupation_gap"]
            )
            static_wall_profile[si, ai] = arrays["wall_midgap_x_profile"]
            residuals.append(metadata["solve"]["stationary_relative_residual_max"])
            if shell is None:
                branch_slope_alpha[ai] = [
                    row["slope_dnu_dky"] for row in diagnostics["branch_slopes"]
                ]

    shape = (len(truncations), len(shells), sizes.size, 2)
    velocity = np.empty(shape)
    velocity_r2 = np.empty(shape)
    directionality = np.empty(shape)
    absolute_asymmetry = np.empty(shape)
    response_branch_slopes = np.full((len(truncations), sizes.size, 2), np.nan)
    for di, dw_truncation in enumerate(truncations):
        for si, shell in enumerate(shells):
            for ni, ny in enumerate(sizes):
                arrays, metadata = by_key[(int(ny), 1.0, shell, dw_truncation)]
                velocity[di, si, ni] = arrays["response_velocity"]
                velocity_r2[di, si, ni] = arrays["response_velocity_r2"]
                directionality[di, si, ni] = arrays["response_mean_directionality"]
                absolute_asymmetry[di, si, ni] = arrays["response_mean_asymmetry"]
                residuals.append(metadata["solve"]["stationary_relative_residual_max"])
                if shell is None:
                    response_branch_slopes[di, ni] = [
                        row["slope_dnu_dky"] for row in metadata["diagnostics"]["branch_slopes"]
                    ]

    largest_arrays, largest_metadata = by_key[(64, 1.0, None, True)]
    off_arrays, off_metadata = by_key[(64, 1.0, None, False)]
    uniform_velocity = np.empty((2, len(shells), 2))
    uniform_directionality = np.empty_like(uniform_velocity)
    for pi, phase in enumerate(("topological", "trivial")):
        for si, shell in enumerate(shells):
            arrays, _ = uniform[(phase, shell)]
            uniform_velocity[pi, si] = arrays["response_velocity"]
            uniform_directionality[pi, si] = arrays["response_mean_directionality"]

    canonical_velocity = velocity[0, 2, -1]
    canonical_directionality = directionality[0, 2, -1]
    canonical_r2 = velocity_r2[0, 2, -1]
    opposite_sign = bool(
        np.prod(canonical_velocity) < 0.0
        and np.prod(canonical_directionality) < 0.0
        and np.min(canonical_r2) >= 0.9
    )
    uniform_null = bool(np.nanmax(np.abs(uniform_directionality)) < 0.1)
    interpretation_status = (
        "wall_resolved_directional_mean_response_supported"
        if opposite_sign and uniform_null
        else "appendix_only_directional_response_not_established"
    )

    output = data_dir / "result_24_continuous_lindblad_reconstruction.npz"
    np.savez_compressed(
        output,
        alpha_in=alpha,
        static_scan_Ny=np.asarray(static_ny, dtype=np.int64),
        response_Ny=sizes,
        nshell=shell_codes,
        dw_truncation=np.asarray([1, 0], dtype=np.int64),
        static_entropy_density=entropy_density,
        static_half_occupation_gap=half_gap,
        static_bulk_half_occupation_gap=bulk_gap,
        static_wall_midgap_x_profile=static_wall_profile,
        static_branch_slope_alpha=branch_slope_alpha,
        response_velocity=velocity,
        response_velocity_r2=velocity_r2,
        response_mean_directionality=directionality,
        response_absolute_asymmetry=absolute_asymmetry,
        response_branch_slopes=response_branch_slopes,
        uniform_response_velocity=uniform_velocity,
        uniform_response_directionality=uniform_directionality,
        response_times=largest_arrays["response_times"],
        response_density_ty=largest_arrays["response_density_ty"],
        response_positive_center_time=largest_arrays["response_positive_center_time"],
        response_negative_center_time=largest_arrays["response_negative_center_time"],
        response_finite_channel_p=largest_arrays["response_finite_channel_p"],
        response_finite_channel_relative_error=largest_arrays["response_finite_channel_relative_error"],
        response_epsilon_relative_error=largest_arrays["response_epsilon_relative_error"],
        ky=largest_arrays["ky"],
        occupation_spectrum_ky=largest_arrays["occupation_spectrum_ky"],
        wall_branch_occupations_ky=largest_arrays["wall_branch_occupations_ky"],
        wall_branch_weights_ky=largest_arrays["wall_branch_weights_ky"],
        off_wall_branch_occupations_ky=off_arrays["wall_branch_occupations_ky"],
    )

    solver_path = CPU_CAMPAIGN / "mean_channel_lindblad_cpu.py"
    metadata_path = data_dir / "result_24_continuous_lindblad_reconstruction.json"
    metadata = {
        "schema": "atlas_mean_channel_lindblad_cpu_reconstruction_v3_hybrid_response",
        "source_module": str(solver_path.relative_to(ROOT)),
        "source_module_sha256": sha256(solver_path),
        "config_sha256": sha256(config_path),
        "data_sha256": sha256(output),
        "saved_covariance_history": False,
        "permanent_covariance_bytes": 0,
        "response_covariance_materialized": False,
        "trajectory_samples": 0,
        "schedule_samples": 0,
        "primary_wall_case_count": 39,
        "reconstruction_case_count": len(selected_cases),
        "static_scan": {"Nx": 20, "Ny": 64, "alpha_in": alpha.tolist()},
        "response_sizes": sizes.tolist(),
        "shells": [1, 2, None],
        "momentum_resolved_shell": None,
        "canonical_dw_truncation": True,
        "endpoint_dw_truncation": [True, False],
        "finite_shell_product_rule": "real-space and momentum-integrated only",
        "response_probe": largest_metadata["response"],
        "interpretation_status": interpretation_status,
        "opposite_wall_directionality_gate": opposite_sign,
        "uniform_directionality_null_gate": uniform_null,
        "largest_stationary_residual": float(np.max(residuals)),
        "largest_case_diagnostics": largest_metadata["diagnostics"],
        "largest_untruncated_off_diagnostics": off_metadata["diagnostics"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print(metadata_path)
    print(json.dumps({
        "interpretation_status": interpretation_status,
        "canonical_velocity": canonical_velocity.tolist(),
        "canonical_directionality": canonical_directionality.tolist(),
        "uniform_directionality_max": float(np.nanmax(np.abs(uniform_directionality))),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
