from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "00_WORKSPACE/CURRENT/trajectory_independence_validation"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BUNDLE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = _load("trajectory_independence_runner", "run_campaign.py")
analyzer = _load("trajectory_independence_analyzer", "analyze_results.py")


def test_trajectory_independence_locked_config_and_seeds():
    config = runner.load_config()
    assert config["geometry"]["Nx"] == config["geometry"]["Ny"] == 16
    assert config["geometry"]["cycles"] == 32
    assert config["geometry"]["samples"] == 10
    assert config["geometry"]["DW"] is False
    assert config["protocol"]["state_representation"] == "physical_frame"
    first = runner.derived_seeds(config)
    second = runner.derived_seeds(config)
    assert first == second
    all_seeds = [first["initial_state"], first["schedule"], *first["outcomes"]]
    assert len(first["outcomes"]) == 10
    assert len(set(all_seeds)) == 12


def test_trajectory_independence_projector_distance_and_unequal_rank_defects():
    left = np.eye(6, dtype=np.complex128)[:, :3]
    phase = np.diag(np.exp(1j * np.asarray([0.2, 0.4, 0.6])))
    same = left @ phase
    smaller = left[:, :2]
    assert analyzer.projector_distance(left, same) < 1e-14
    assert analyzer.projector_distance(left, smaller) > 0.0
    distance, hole, excess = analyzer.target_defects(left, smaller)
    np.testing.assert_allclose(hole, 1.0, atol=1e-14)
    np.testing.assert_allclose(excess, 0.0, atol=1e-14)
    np.testing.assert_allclose(distance, np.sqrt(1.0 / 5.0), atol=1e-14)


def test_trajectory_independence_analyzer_classifies_identical_synthetic_states(tmp_path: Path):
    config = runner.load_config()
    config["geometry"]["Nx"] = 1
    config["geometry"]["Ny"] = 1
    config["geometry"]["cycles"] = 1
    config["geometry"]["samples"] = 2
    config["terminal_window"] = {"start_cycle": 1, "stop_cycle_inclusive": 1}
    (tmp_path / "campaign_config.v1.json").write_text(json.dumps(config))
    frame = np.eye(2, dtype=np.complex128)[:, :1]
    target_dir = tmp_path / "prepared"
    target_dir.mkdir()
    np.savez_compressed(
        target_dir / "target_state.npz",
        frame=frame,
        real_space_chern=np.asarray(1.0),
    )
    for sample, outcome in enumerate((False, True)):
        directory = tmp_path / f"raw/trajectories/sample_{sample:02d}"
        directory.mkdir(parents=True)
        np.savez_compressed(
            directory / "cycle_data.npz",
            cycles=np.arange(2),
            real_space_chern=np.asarray([0.0, 1.0]),
            charge=np.ones(2),
            rank=np.ones(2, dtype=np.int64),
            half_entropy=np.zeros(2),
            gram_residual=np.zeros(2),
            local_occupations=np.asarray([[1.0, 0.0], [1.0, 0.0]]),
            cumulative_log_weight=np.zeros(2),
            frame_cycle_000=frame,
            frame_cycle_001=frame,
        )
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "outcome_seed": sample + 10,
                    "outcome_digest": str(outcome),
                }
            )
        )
        record = {
            "entries": [
                {
                    "cycle": 1,
                    "site_id": 0,
                    "branch_events": [
                        {
                            "kind": "measurement",
                            "channel": "Ap",
                            "outcome_occupied": outcome,
                        }
                    ],
                }
            ]
        }
        with gzip.open(directory / "record.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(record, handle)
    summary = analyzer.analyze(tmp_path)
    assert summary["classification"] == "exact_and_topological"
    assert summary["exact_state_gate_passed"]
    assert summary["topology_gate_passed"]
    assert (tmp_path / "processed/cycle_resolved_analysis.npz").exists()
    assert (tmp_path / "figures/pairwise_heatmaps.pdf").exists()
    audit_dir = tmp_path / "raw/audit"
    audit_dir.mkdir(parents=True)
    (audit_dir / "summary.json").write_text(
        json.dumps(
            {
                "maximum_relative_error": 0.0,
                "maximum_element_error": 0.0,
            }
        )
    )
    report = analyzer.render_report(tmp_path)
    assert report.exists()
    assert "Outcome-Trajectory Independence" in report.read_text()
