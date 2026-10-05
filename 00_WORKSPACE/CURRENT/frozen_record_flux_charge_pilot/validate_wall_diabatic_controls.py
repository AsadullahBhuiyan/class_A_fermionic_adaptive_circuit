#!/usr/bin/env python3
"""Validate exact benchmarks and controls for the wall-diabatic pump.

Scientific quantization of monitored endpoint states is reported rather than
used as a completion gate.  The gates here are deliberately restricted to an
exact equilibrium benchmark, null/sign/gauge controls, and numerical
identities that must hold independently of the measured physics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_wall_diabatic_width_sweep import load_path_rows


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "N20_24_28_32x24_wall_diabatic_spectral_pump_s100_v1"
)
DEFAULT_EXACT_ENDPOINTS = (
    REPO_ROOT
    / "notebooks/flattened_hamiltonian_analysis/outputs/exact_wall_charge_pump_refined_v1"
    / "run_20260901_002219/pump_endpoint_convergence.csv"
)
DEFAULT_EXACT_REVERSIBILITY = (
    REPO_ROOT
    / "notebooks/flattened_hamiltonian_analysis/outputs/exact_wall_charge_pump_reversibility_v1"
    / "run_20260901_162444/reversibility_summary.csv"
)


def _csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_exact_reference(endpoint_csv: Path, reversibility_csv: Path) -> dict[str, Any]:
    endpoint_rows = _csv(endpoint_csv)
    continued = [
        row for row in endpoint_rows
        if row["state_family"] == "continued" and row["grid"] in {"global_129", "global_257"}
    ]
    if len(continued) != 8:
        raise RuntimeError(f"expected eight exact continued-grid rows, found {len(continued)}")
    minimum_abs_qx = min(abs(float(row["q_pump"])) for row in continued)
    maximum_charge_residual = max(abs(float(row["charge_residual"])) for row in continued)
    keyed = {(row["dw_truncation"], row["direction"], row["grid"]): float(row["q_pump"]) for row in continued}
    grid_differences = [
        abs(keyed[(wall, direction, "global_257")] - keyed[(wall, direction, "global_129")])
        for wall in ("False", "True") for direction in ("1", "-1")
    ]
    reversal = _csv(reversibility_csv)
    if not reversal:
        raise RuntimeError("exact reversibility table is empty")
    metrics = {
        "minimum_abs_q_x": minimum_abs_qx,
        "maximum_charge_residual": maximum_charge_residual,
        "maximum_129_to_257_endpoint_change": max(grid_differences),
        "minimum_abs_forward_q_x": min(abs(float(row["q_forward"])) for row in reversal),
        "minimum_abs_opposite_q_x": min(abs(float(row["q_opposite_from_same_initial"])) for row in reversal),
        "maximum_true_undo_error": max(abs(float(row["true_undo_error"])) for row in reversal),
        "maximum_roundtrip_projector_residual": max(abs(float(row["roundtrip_projector_frobenius_per_dim"])) for row in reversal),
        "maximum_large_gauge_parent_error": max(abs(float(row["large_gauge_h_error"])) for row in reversal),
    }
    gates = {
        "quantized_continued_endpoint": minimum_abs_qx >= 0.97,
        "grid_stability": metrics["maximum_129_to_257_endpoint_change"] <= 1e-4,
        "charge_conservation": maximum_charge_residual <= 1e-10,
        "orientation_reversal": metrics["minimum_abs_opposite_q_x"] >= 0.97,
        "true_undo": metrics["maximum_true_undo_error"] <= 1e-10,
        "roundtrip_projector": metrics["maximum_roundtrip_projector_residual"] <= 1e-10,
        "large_gauge_equivalence": metrics["maximum_large_gauge_parent_error"] <= 1e-10,
    }
    return {
        "endpoint_csv": str(Path(endpoint_csv).resolve()),
        "reversibility_csv": str(Path(reversibility_csv).resolve()),
        "metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _control_kind(row: dict[str, Any]) -> str:
    value = row.get("control_kind")
    if bool(row.get("is_primary")) and value in (
        None, "", "none", "wall_diabatic_with_ordinary_and_instantaneous"
    ):
        return "primary"
    if value not in (None, "", "none"):
        return str(value).lower()
    protocol = str(row.get("protocol", "primary")).lower()
    return protocol if protocol in {"trivial", "conjugated", "gauge_uniform", "gauge_seam"} else "primary"


def _pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("control_pair_id", row.get("sample_id")),
        row.get("Nx"), row.get("Ny"), row.get("wall"), row.get("direction"),
        row.get("grid_intervals"), row.get("edge_block_rank"), row.get("wall_window"),
    )


def validate_new_controls(rows: list[dict[str, Any]], *, require_controls: bool) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_kind.setdefault(_control_kind(row), []).append(row)
    primary = {_pair_key(row): row for row in by_kind.get("primary", [])}
    trivial = by_kind.get("trivial", [])
    conjugated = by_kind.get("conjugated", [])
    seam = {_pair_key(row): row for row in by_kind.get("gauge_seam", [])}

    numerical = by_kind.get("primary", []) + trivial + conjugated + list(seam.values())
    maximum_charge_residual = max(
        (row["maximum_total_charge_residual"] for row in numerical if np.isfinite(row["maximum_total_charge_residual"])),
        default=float("nan"),
    )
    maximum_projector_residual = max(
        (row["maximum_projector_residual"] for row in numerical if np.isfinite(row["maximum_projector_residual"])),
        default=float("nan"),
    )
    trivial_maximum = max((abs(row["endpoint_q_x"]) for row in trivial), default=float("nan"))
    conjugated_errors = []
    for row in conjugated:
        base = primary.get(_pair_key(row))
        if base is not None:
            conjugated_errors.append(abs(row["endpoint_q_x"] + base["endpoint_q_x"]))
    gauge_errors = []
    for key, row in seam.items():
        other = primary.get(key)
        if other is not None:
            gauge_errors.append(abs(row["endpoint_q_x"] - other["endpoint_q_x"]))
    exact_mesh: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("protocol", "")) != "exact_topological":
            continue
        interval = int(row.get("grid_intervals", 0))
        if interval not in {128, 256, 512}:
            continue
        key = (row.get("wall"), row.get("direction"))
        exact_mesh.setdefault(key, {})[interval] = row
    mesh_errors = [
        max(
            abs(group[128]["endpoint_q_x"] - group[256]["endpoint_q_x"]),
            abs(group[512]["endpoint_q_x"] - group[256]["endpoint_q_x"]),
        )
        for group in exact_mesh.values()
        if {128, 256, 512}.issubset(group)
    ]
    exact_primary_signed = [
        ({"ccw": 1.0, "cw": -1.0}[str(row["direction"])])
        * float(row["endpoint_q_x"])
        for row in primary.values()
        if str(row.get("protocol", "")) == "exact_topological"
        and str(row.get("direction", "")) in {"ccw", "cw"}
    ]

    availability = {
        "trivial_paths": len(trivial),
        "conjugated_paths": len(conjugated),
        "conjugated_pairs": len(conjugated_errors),
        "gauge_pairs": len(gauge_errors),
        "exact_mesh_direction_pairs": len(mesh_errors),
        "exact_primary_paths": len(exact_primary_signed),
    }
    gates: dict[str, bool] = {
        "charge_conservation": bool(np.isfinite(maximum_charge_residual) and maximum_charge_residual <= 1e-10),
        "projector_numerics": bool(np.isfinite(maximum_projector_residual) and maximum_projector_residual <= 1e-10),
    }
    if trivial:
        gates["trivial_null"] = trivial_maximum <= 0.05
    if conjugated_errors:
        gates["chern_conjugation_sign"] = max(conjugated_errors) <= 0.05
    if gauge_errors:
        gates["twist_gauge_equivalence"] = max(gauge_errors) <= 1e-8
    if mesh_errors:
        gates["exact_M128_M256_M512_stability"] = max(mesh_errors) <= 1e-4
    if exact_primary_signed:
        gates["exact_wall_diabatic_quantization"] = min(exact_primary_signed) >= 0.97
    if require_controls:
        gates["control_coverage"] = bool(
            trivial and conjugated_errors and gauge_errors
            and len(mesh_errors) == 4 and len(exact_primary_signed) == 4
        )
    return {
        "availability": availability,
        "metrics": {
            "maximum_charge_residual": maximum_charge_residual,
            "maximum_projector_residual": maximum_projector_residual,
            "trivial_maximum_abs_q_x": trivial_maximum,
            "conjugated_maximum_sign_error": max(conjugated_errors, default=float("nan")),
            "gauge_maximum_endpoint_difference": max(gauge_errors, default=float("nan")),
            "exact_M128_M256_M512_maximum_endpoint_change": max(mesh_errors, default=float("nan")),
            "exact_primary_minimum_correctly_signed_q_x": min(
                exact_primary_signed, default=float("nan")
            ),
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def validate(
    output_root: Path | None,
    *,
    endpoint_csv: Path = DEFAULT_EXACT_ENDPOINTS,
    reversibility_csv: Path = DEFAULT_EXACT_REVERSIBILITY,
    require_controls: bool = True,
) -> dict[str, Any]:
    exact = validate_exact_reference(endpoint_csv, reversibility_csv)
    new_controls: dict[str, Any] | None = None
    if output_root is not None:
        rows, invalid = load_path_rows(output_root, allow_invalid=False)
        new_controls = validate_new_controls(rows, require_controls=require_controls)
        new_controls["invalid_pair_count"] = len(invalid)
    passed = exact["passed"] and (new_controls is None or new_controls["passed"])
    summary = {
        "schema": "wall_diabatic_control_validation_v1",
        "exact_reference": exact,
        "new_campaign_controls": new_controls,
        "passed": passed,
        "quantization_gate_scope": "exact_equilibrium_control_only",
        "monitored_endpoint_quantization_is_acceptance_gate": False,
    }
    if output_root is not None:
        target = Path(output_root) / "analysis" / "control_validation.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary["written_to"] = str(target)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("wall-diabatic control validation failed")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--exact-endpoints", type=Path, default=DEFAULT_EXACT_ENDPOINTS)
    parser.add_argument("--exact-reversibility", type=Path, default=DEFAULT_EXACT_REVERSIBILITY)
    parser.add_argument("--allow-pending-controls", action="store_true")
    args = parser.parse_args()
    validate(
        args.output_root,
        endpoint_csv=args.exact_endpoints,
        reversibility_csv=args.exact_reversibility,
        require_controls=not args.allow_pending_controls,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
