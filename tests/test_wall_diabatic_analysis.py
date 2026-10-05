from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT = (
    Path(__file__).resolve().parents[1]
    / "00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot"
)
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

analysis = importlib.import_module("analyze_wall_diabatic_width_sweep")
io = importlib.import_module("wall_diabatized_io")
controls = importlib.import_module("validate_wall_diabatic_controls")


def _publish_pair(root: Path, *, task_id: str = "wall_diabatic_test_soft_sample_000") -> Path:
    metadata = {
        "task_id": task_id,
        "stage": "wall_diabatic_pump",
        "cell": "nsh1_N20x24",
        "protocol": "nsh1",
        "size": "N20x24",
        "Nx": 20,
        "Ny": 24,
        "wall": "soft",
        "sample_id": 0,
        "grid_intervals": 4,
        "edge_block_rank": 2,
        "wall_window": 2,
        "is_primary": True,
        "control_kind": "none",
        "config_hash": "config",
        "source_hashes": {"runner": "source"},
    }
    phi = np.stack((np.linspace(-1e-7, 2 * np.pi - 1e-7, 5), np.linspace(1e-7, -2 * np.pi + 1e-7, 5)))
    q_x = np.stack((np.linspace(0, 1, 5), np.linspace(0, -1, 5)))
    delta_left, delta_right = -q_x, q_x
    result = root / "paths" / "soft" / "sample_000.npz"
    result.parent.mkdir(parents=True)
    np.savez_compressed(
        result,
        schema=np.asarray("wall_diabatic_spectral_pump_result_v1"),
        directions=np.asarray(["ccw", "cw"]),
        sigma=np.asarray([1, -1]),
        phi=phi,
        delta_N_left=delta_left,
        delta_N_right=delta_right,
        delta_N_total=delta_left + delta_right,
        q_x=q_x,
        density_x=np.zeros((2, 5, 20)),
        edge_internal_gap=np.full((2, 5), 0.01),
        edge_external_gap=np.full((2, 5), 0.2),
        edge_link_min_singular=np.full((2, 5), 0.99),
        edge_minimum_gap_index=np.asarray([2, 2]),
        edge_B_eigenvalues=np.tile(np.asarray([-0.95, 0.96]), (2, 5, 1)),
        edge_combined_wall_weight=np.full((2, 5, 2), 0.98),
        total_charge_residual=np.zeros((2, 5)),
        projector_residual=np.full((2, 5), 1e-13),
        endpoint_defect_eigenvalues=np.asarray([[-1, 1], [-1, 1]]),
        endpoint_particle_density_x=np.tile(np.eye(1, 20, 17), (2, 1)),
        endpoint_hole_density_x=np.tile(np.eye(1, 20, 2), (2, 1)),
        multicut_q_x=np.stack((np.linspace(0.99, 1.01, 5), np.linspace(-1.01, -0.99, 5))),
        multicut_positions=np.arange(5),
        center_of_charge_displacement=np.asarray([15.0, -15.0]),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    )
    digest = hashlib.sha256(result.read_bytes()).hexdigest()
    completion = {
        "schema": "wall_diabatic_spectral_pump_completion_v1",
        **metadata,
        "result": {"name": result.name, "bytes": result.stat().st_size, "sha256": digest},
    }
    completion_path = result.with_suffix(".completion.json")
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    return completion_path


def test_combined_direction_pair_verifies_and_explodes(tmp_path: Path) -> None:
    completion = _publish_pair(tmp_path)
    verified = io.verify_completion(completion)
    assert verified.arrays["q_x"].shape == (2, 5)
    rows, invalid = analysis.load_path_rows(tmp_path)
    assert not invalid
    assert [row["direction"] for row in rows] == ["ccw", "cw"]
    assert [row["endpoint_q_x"] for row in rows] == pytest.approx([1.0, -1.0])
    assert [row["correctly_signed_endpoint_q_x"] for row in rows] == pytest.approx([1.0, 1.0])
    assert [row["crossing_wall_character_margin"] for row in rows] == pytest.approx([0.95, 0.95])
    assert [row["crossing_minimum_wall_weight"] for row in rows] == pytest.approx([0.98, 0.98])


def test_analysis_writes_raw_samplewise_products(tmp_path: Path) -> None:
    _publish_pair(tmp_path)
    summary = analysis.analyze(tmp_path, expected_pairs=1)
    assert summary["verified_pair_count"] == 1
    assert summary["primary_raw_directional_path_count"] == 2
    text = Path(summary["samplewise_csv"]).read_text(encoding="utf-8")
    assert "endpoint_q_x" in text
    assert "direction" in text
    assert "q_x_odd" not in text
    for product in summary["figures"].values():
        assert Path(product["pdf"]).is_file()
        assert Path(product["png"]).is_file()


def test_checksum_corruption_is_rejected(tmp_path: Path) -> None:
    completion = _publish_pair(tmp_path)
    result = completion.with_name("sample_000.npz")
    result.write_bytes(result.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="byte count"):
        io.verify_completion(completion)


def _row(
    kind: str,
    q_x: float,
    *,
    direction: str = "ccw",
    wall: str = "soft",
    intervals: int = 256,
    protocol: str | None = None,
) -> dict[str, object]:
    return {
        "control_kind": kind,
        "protocol": protocol or ("primary" if kind == "primary" else kind),
        "control_pair_id": f"exact_{wall}_M{intervals}",
        "sample_id": 7,
        "Nx": 20,
        "Ny": 40,
        "wall": wall,
        "direction": direction,
        "grid_intervals": intervals,
        "edge_block_rank": 2,
        "wall_window": 2,
        "endpoint_q_x": q_x,
        "maximum_total_charge_residual": 1e-12,
        "maximum_projector_residual": 1e-13,
    }


def test_control_gates_are_scoped_and_paired() -> None:
    rows = []
    for wall in ("soft", "hard"):
        for direction, q_x in (("ccw", 0.99), ("cw", -0.99)):
            rows.extend(
                [
                    _row("mesh_sensitivity", q_x, direction=direction, wall=wall, intervals=128, protocol="exact_topological"),
                    _row("primary", q_x, direction=direction, wall=wall, intervals=256, protocol="exact_topological"),
                    _row("mesh_sensitivity", q_x, direction=direction, wall=wall, intervals=512, protocol="exact_topological"),
                    _row("trivial", 0.01, direction=direction, wall=wall, protocol="exact_trivial"),
                    _row("conjugated", -q_x, direction=direction, wall=wall, protocol="exact_conjugated"),
                    _row("gauge_seam", q_x + 1e-10, direction=direction, wall=wall, protocol="exact_topological"),
                ]
            )
    summary = controls.validate_new_controls(rows, require_controls=True)
    assert summary["passed"]
    assert summary["availability"]["conjugated_pairs"] == 4
    assert summary["availability"]["gauge_pairs"] == 4
    assert summary["availability"]["exact_mesh_direction_pairs"] == 4
    assert summary["availability"]["exact_primary_paths"] == 4
    assert summary["gates"]["exact_wall_diabatic_quantization"]


def test_exact_wall_diabatic_control_must_reproduce_quantized_flow() -> None:
    rows = []
    for wall in ("soft", "hard"):
        for direction, q_x in (("ccw", 0.99), ("cw", -0.99)):
            rows.extend(
                [
                    _row("mesh_sensitivity", q_x, direction=direction, wall=wall, intervals=128, protocol="exact_topological"),
                    _row("primary", 0.5 if wall == "hard" else q_x, direction=direction, wall=wall, intervals=256, protocol="exact_topological"),
                    _row("mesh_sensitivity", q_x, direction=direction, wall=wall, intervals=512, protocol="exact_topological"),
                    _row("trivial", 0.01, direction=direction, wall=wall, protocol="exact_trivial"),
                    _row("conjugated", -(0.5 if wall == "hard" else q_x), direction=direction, wall=wall, protocol="exact_conjugated"),
                    _row("gauge_seam", 0.5 if wall == "hard" else q_x, direction=direction, wall=wall, protocol="exact_topological"),
                ]
            )
    summary = controls.validate_new_controls(rows, require_controls=True)
    assert not summary["passed"]
    assert not summary["gates"]["exact_wall_diabatic_quantization"]


def test_bridge_prefix_mapping_and_backend_factor() -> None:
    primary = []
    bridge = []
    for sample_id in range(25):
        primary.append(
            {
                "cell": "nsh1_N20x24", "wall": "soft", "direction": "ccw",
                "sample_id": sample_id, "endpoint_q_x": 0.9 + sample_id / 1000,
                "is_primary": True,
            }
        )
        bridge.append(
            {
                "cell": "bridge_nsh1_N20x24", "wall": "soft", "direction": "ccw",
                "sample_id": sample_id, "endpoint_q_x": 1.0 + sample_id / 1000,
                "is_primary": False, "source_backend": "gpu", "result_collection": "bridge_pump",
            }
        )
    summary = analysis._backend_bridge(primary, primary + bridge)
    assert summary["status"] == "complete"
    assert summary["comparisons"][0]["cpu_count"] == 25
    assert summary["comparisons"][0]["primary_cell"] == "nsh1_N20x24"
    assert summary["backend_factor_sensitivity"]["gpu_coefficient"] == pytest.approx(0.1)


def test_edge_fit_uses_wall_separation_and_crossing_gap() -> None:
    rows = []
    for nx in (20, 24, 28, 32):
        for sample_id in range(5):
            rows.append(
                {
                    "protocol": "nsh1", "wall": "soft", "direction": "ccw", "Nx": nx,
                    "crossing_edge_internal_gap": float(np.exp(-(nx / 2) / 2)),
                    "minimum_edge_internal_gap": 1e-99,
                    "sample_id": sample_id,
                }
            )
    fit = analysis._edge_gap_fits(rows, bootstrap_draws=32)[0]
    assert fit["wall_separation_sites"] == [10.0, 12.0, 14.0, 16.0]
    assert fit["decay_length_sites"] == pytest.approx(2.0)


def test_sensitivity_variant_names_and_full_count_recognized() -> None:
    primary = []
    sensitivities = []
    variants = ("M128", "M512", "seam_M256", "radius3_M256", "rank4_M256")
    for pair in range(410):
        cell = f"cell_{pair}"
        for direction in ("ccw", "cw"):
            primary.append(
                {
                    "cell": cell, "wall": "soft", "sample_id": pair, "direction": direction,
                    "task_id": f"primary_{pair}", "endpoint_q_x": 1.0, "is_primary": True,
                }
            )
            for variant in variants:
                sensitivities.append(
                    {
                        "cell": cell, "wall": "soft", "sample_id": pair, "direction": direction,
                        "task_id": f"{variant}_{pair}", "endpoint_q_x": 1.0,
                        "variant": variant, "is_primary": False,
                    }
                )
    summary = analysis._sensitivity_status(primary, primary + sensitivities)
    assert summary["status"] == "complete"
    assert {row["variant"] for row in summary["variants"]} == set(variants)
    assert all(row["verified_pairs"] == 410 for row in summary["variants"])


def test_exact_saved_benchmark_passes_current_gates() -> None:
    summary = controls.validate_exact_reference(
        controls.DEFAULT_EXACT_ENDPOINTS, controls.DEFAULT_EXACT_REVERSIBILITY
    )
    assert summary["passed"]
    assert summary["metrics"]["minimum_abs_q_x"] > 0.97
