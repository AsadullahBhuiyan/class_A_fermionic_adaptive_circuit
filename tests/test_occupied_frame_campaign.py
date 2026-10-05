from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from src.fgtn.occupied_frame import OccupiedFrameState, UpdateTimingCollector


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "00_WORKSPACE/CURRENT/occupied_frame_validation"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BUNDLE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = _load("occupied_frame_campaign_runner", "run_campaign.py")
analyzer = _load("occupied_frame_campaign_analyzer", "analyze_results.py")


def test_locked_configuration_and_seed_expansion_are_deterministic():
    config = runner.load_config()
    assert config["geometry"] == {
        "Nx": 16,
        "Ny": 16,
        "cycles": 32,
        "samples": 10,
        "nshell": 1,
        "filling_frac": 0.5,
        "DW": False,
        "alpha_1": 1.0,
        "alpha_2": 1.0,
        "trial_orbitals": "X",
        "dw_truncation": False,
        "meas_slab_only": False,
    }
    first = runner.derived_sample_seeds(config)
    second = runner.derived_sample_seeds(config)
    assert first == second
    assert set(first) == {"random_pure", "maxmix"}
    assert all(len(values) == 10 and len(set(values)) == 10 for values in first.values())
    assert not set(first["random_pure"]) & set(first["maxmix"])


def test_square_size_override_scales_cycles_regions_and_checkpoints():
    original = runner.CAMPAIGN_SIZE
    try:
        runner.CAMPAIGN_SIZE = 18
        config = runner.load_config()
    finally:
        runner.CAMPAIGN_SIZE = original
    assert config["geometry"]["Nx"] == config["geometry"]["Ny"] == 18
    assert config["geometry"]["cycles"] == 36
    assert config["half_region"]["y_stop_exclusive"] == 9
    assert config["checkpoint_cycles"] == [0, 1, 9, 18, 27, 36]
    assert config["physics_gate"]["terminal_window_start_cycle"] == 27


def test_event_sketch_agrees_between_covariance_and_frame():
    rng = np.random.default_rng(71)
    raw = rng.standard_normal((18, 7)) + 1j * rng.standard_normal((18, 7))
    frame, _ = np.linalg.qr(raw, mode="reduced")
    state = OccupiedFrameState(
        frame,
        representation="physical_frame",
        physical_dimension=18,
    )
    covariance = state.centered_covariance()
    frame_capture = runner.EventSketchCapture("frame", 18)
    covariance_capture = runner.EventSketchCapture("covariance", 18)
    payload = {"cycle": 3, "site_id": 8, "channel": "Bm"}
    frame_capture(state=state, **payload)
    covariance_capture(state=covariance, **payload)
    assert frame_capture.identities == covariance_capture.identities
    np.testing.assert_allclose(frame_capture.values, covariance_capture.values, atol=3e-14)


def test_frame_native_chern_matches_canonical_covariance_estimator():
    model, _ = runner.build_model()
    geometry = runner.load_config()["geometry"]
    target = model.G_CI_domain_wall(
        periodic=True,
        alpha=np.ones((geometry["Nx"], geometry["Ny"]), dtype=np.float64),
    )
    state = OccupiedFrameState.from_centered_covariance(
        target, representation="physical_frame"
    )
    covariance_value = float(np.real(model.real_space_chern_number(target)))
    frame_value = runner.real_space_chern_from_frame(
        state, runner.chern_partition_indices()
    )
    np.testing.assert_allclose(frame_value, covariance_value, atol=2e-12)
    np.testing.assert_allclose(covariance_value, 0.9994649680567276, atol=2e-12)


def test_branch_cycle_comparison_localizes_probability_and_identity_errors():
    reference = [
        {
            "cycle": 1,
            "site_id": 2,
            "cumulative_log_weight": -0.2,
            "branch_events": [
                {
                    "kind": "measurement",
                    "channel": "Ap",
                    "probability": 0.25,
                    "outcome_occupied": False,
                }
            ],
        }
    ]
    observed = [
        {
            **reference[0],
            "cumulative_log_weight": -0.21,
            "branch_events": [{**reference[0]["branch_events"][0], "probability": 0.251}],
        }
    ]
    result = runner._branch_cycle_metrics(reference, observed, 1)
    assert result["branch_disagreement_count"] == 0
    np.testing.assert_allclose(result["max_probability_error"], [0.0, 0.001])
    np.testing.assert_allclose(result["cumulative_log_weight"], [0.0, -0.21])


def test_timing_call_counts_and_speed_classification():
    timing = UpdateTimingCollector("detailed")
    with timing.measure("gain_overlap", detailed=True):
        np.linalg.norm(np.ones(8))
    snapshot = timing.snapshot()
    assert snapshot["counts"]["gain_overlap_calls"] == 1
    assert snapshot["total_ns"]["gain_overlap"] >= 0
    assert analyzer.classification((1.06, 1.2), 0.05) == "faster"
    assert analyzer.classification((0.7, 0.94), 0.05) == "slower"
    assert analyzer.classification((0.98, 1.08), 0.05) == "comparable"


def test_analyzer_and_report_accept_complete_synthetic_shards(tmp_path, monkeypatch):
    config = runner.load_config()
    config["geometry"]["samples"] = 1
    config["geometry"]["cycles"] = 1
    (tmp_path / "campaign_config.v1.json").write_text(json.dumps(config))
    timing = {
        "total_ns": {"trajectory_total": 100, "cycle_total": 90, "gain_overlap": 10},
        "counts": {"trajectory_total_calls": 1, "cycle_total_calls": 1, "gain_overlap_calls": 2},
        "per_cycle_total_ns": {"1": {"cycle_total": 90}},
    }
    for family_index, spec in enumerate(config["initializations"]):
        label = spec["label"]
        directory = tmp_path / "raw/correctness" / label / "sample_00"
        directory.mkdir(parents=True)
        arrays = {
            name: np.zeros(2, dtype=np.float64) for name in analyzer.CORRECTNESS_ARRAYS
        }
        arrays["frame_rank"] = np.asarray([4, 4], dtype=np.int64)
        if label == "random_pure":
            arrays["choi_relative_error"][:] = np.nan
        np.savez_compressed(directory / "cycle_data.npz", cycles=np.arange(2), **arrays)
        backend = {
            "wall_ns": 120,
            "cpu_ns": 110,
            "native_state_bytes": 1024,
            "observer_timing_ns": {},
            "event_sketch_observer_ns": 0,
            "timing": timing,
        }
        summary = {
            "label": label,
            "sample": 0,
            "seed": 10 + family_index,
            "cpu": family_index,
            "gate_passed": True,
            "branch_disagreement_count": 0,
            "regularized_dense_fallback_count": 0,
            "peak_rss_kib_process": 4096,
            "checks": {
                "covariance": 0.0,
                "branch_probability": 0.0,
                "log_weight": 0.0,
                "entropy": 0.0,
                "gram": 0.0,
                "choi": 0.0,
            },
            "backends": {"covariance": backend, "frame": backend},
        }
        (directory / "summary.json").write_text(json.dumps(summary))

        benchmark_dir = tmp_path / "raw/benchmark" / label / "sample_00"
        benchmark_dir.mkdir(parents=True)
        benchmark_summary = {
            "label": label,
            "sample": 0,
            "seed": 10 + family_index,
            "cpu": family_index,
            "backend_order": ["covariance", "frame"],
            "peak_rss_kib_process": 4096,
            "backends": {
                "covariance": {
                    "wall_ns": 120,
                    "cpu_ns": 110,
                    "update_ns": 100,
                    "native_state_bytes": 1024,
                    "timing": timing,
                },
                "frame": {
                    "wall_ns": 100,
                    "cpu_ns": 90,
                    "update_ns": 80,
                    "native_state_bytes": 512,
                    "timing": timing,
                },
            },
        }
        (benchmark_dir / "summary.json").write_text(json.dumps(benchmark_summary))
    (tmp_path / "status").mkdir()
    (tmp_path / "status/benchmark.json").write_text(
        json.dumps(
            {
                "parallel_timing_ns": {
                    spec["label"]: {"stage_total": 1_000_000_000}
                    for spec in config["initializations"]
                }
            }
        )
    )
    summary = analyzer.analyze(tmp_path)
    assert summary["correctness_gate_passed"]
    assert set(summary["benchmark"]) == {"random_pure", "maxmix"}
    assert (tmp_path / "figures/paired_speedup.pdf").exists()
    monkeypatch.setattr(analyzer.shutil, "which", lambda _: None)
    report = analyzer.render_report(tmp_path)
    assert report.exists()
    assert "no covariance construction" in report.read_text().lower()
