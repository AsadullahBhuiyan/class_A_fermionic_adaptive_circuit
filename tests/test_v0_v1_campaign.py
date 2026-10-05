from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import validation_campaigns.v0_v1.run as campaign
from validation_campaigns.v0_v1.run import (
    build_v1_tasks,
    CompactEventRecorder,
    SCHEDULES,
    exhaustive_local_probability,
    file_sha256,
    load_checkpoint,
    load_v0_prerequisite,
    load_npz_dict,
    merge_v1_shards,
    model_for,
    save_checkpoint,
    save_npz_atomic,
    trajectory_directory,
    write_json_atomic,
)


def test_compact_event_record_preserves_order_channels_and_transfers():
    recorder = CompactEventRecorder(cycles=1, site_ids=(3, 7), replay=True)
    recorder(
        cycle=1,
        site_id=7,
        branch_log_weight=-0.3,
        branch_events=(
            {"kind": "measurement", "channel": "Ap", "probability": 0.25, "outcome_occupied": True},
            {
                "kind": "correction",
                "channel": "Ap",
                "expected_occupied": False,
                "target_occupied": False,
            },
            {"kind": "measurement", "channel": "Am", "probability": 0.75, "outcome_occupied": True},
            {"kind": "measurement", "channel": "Bp", "probability": 0.50, "outcome_occupied": False},
            {"kind": "measurement", "channel": "Bm", "probability": 0.50, "outcome_occupied": True},
        ),
    )
    recorder(
        cycle=1,
        site_id=3,
        branch_log_weight=-0.4,
        branch_events=tuple(
            {
                "kind": "measurement",
                "channel": channel,
                "probability": 0.5,
                "outcome_occupied": expected,
            }
            for channel, expected in zip(("Ap", "Am", "Bp", "Bm"), (False, True, False, True))
        ),
    )
    recorder.validate_complete()
    np.testing.assert_array_equal(recorder.ordered_site_ids(), [[7, 3]])
    assert recorder.transfer.sum() == -1
    assert len(recorder.entries) == 2


def test_checkpoint_disk_round_trip_is_hash_checked(tmp_path: Path):
    state = {
        "version": 1,
        "signature": {"test": True},
        "completed_cycles": 2,
        "G": np.eye(3, dtype=np.complex128),
        "cumulative_log_weight": -1.0,
        "random_seed": 1,
        "sample_seed": 2,
        "last_ordered_site_ids": np.asarray((2, 0, 1)),
        "rng_states": {
            name: np.random.default_rng(index).bit_generator.state
            for index, name in enumerate(("initialization", "exterior", "schedule", "dynamics"))
        },
    }
    save_checkpoint(tmp_path, state)
    actual = load_checkpoint(tmp_path)
    np.testing.assert_array_equal(actual["G"], state["G"])
    assert actual["completed_cycles"] == 2
    assert actual["last_ordered_site_ids"] == [2, 0, 1]


def test_local_four_channel_tree_normalizes():
    result = exhaustive_local_probability(model_for(4, 6))
    assert result["branch_count"] == 16
    assert result["residual"] < 1.0e-12


def test_shard_merge_is_deterministic_and_schedule_major(tmp_path: Path):
    required = (
        "events.npz",
        "cycle_observables.npz",
        "stationary_snapshots.npz",
        "snapshot_metrics.npz",
    )
    for schedule_index, sequence in enumerate(SCHEDULES):
        run_dir = trajectory_directory(tmp_path, sequence, 0)
        run_dir.mkdir(parents=True)
        for name in required[:-1]:
            save_npz_atomic(run_dir / name, placeholder=np.asarray([schedule_index]))
        save_npz_atomic(
            run_dir / required[-1],
            bulk_chern=np.asarray([schedule_index, schedule_index + 0.5]),
            wall_localized_weight=np.asarray([0.75, 0.8]),
            entropy_coefficient=np.asarray([0.1, 0.2]),
            interval_charge_coefficient=np.asarray([0.3, 0.4]),
            conditional_velocity=np.asarray([-1.0, 1.0]),
            max_hermiticity=np.asarray(1.0e-14),
            max_spectral_violation=np.asarray(0.0),
        )
        write_json_atomic(
            run_dir / "summary.json",
            {
                "status": "complete",
                "config_hash": f"hash-{schedule_index}",
                "config": {"sequence": sequence, "sample": 0},
                "files": {
                    name: file_sha256(run_dir / name)
                    for name in required
                },
            },
        )

    first = merge_v1_shards(tmp_path, samples=1)
    first_arrays = load_npz_dict(tmp_path / "merged_shards.npz")
    second = merge_v1_shards(tmp_path, samples=1)
    second_arrays = load_npz_dict(tmp_path / "merged_shards.npz")
    assert [item["sequence"] for item in first["shards"]] == list(SCHEDULES)
    assert [item["sequence"] for item in second["shards"]] == list(SCHEDULES)
    for name in first_arrays:
        np.testing.assert_array_equal(first_arrays[name], second_arrays[name])
    np.testing.assert_array_equal(first_arrays["bulk_chern"][:, 0, 0], [0.0, 1.0, 2.0])


def test_reduced_v1_builds_exactly_ten_samples_for_each_schedule(tmp_path: Path):
    tasks = build_v1_tasks(
        tmp_path,
        nx=16,
        ny=16,
        samples=10,
        target_cycles=64,
        seed=20260816,
        resume=True,
    )
    assert len(tasks) == 30
    for sequence in SCHEDULES:
        selected = [task for task in tasks if task["sequence"] == sequence]
        assert [task["sample"] for task in selected] == list(range(10))
        assert len({task["seed"] for task in selected}) == 10
    assert all(task["nx"] == task["ny"] == 16 for task in tasks)
    assert all(task["target_cycles"] == 64 and task["resume"] for task in tasks)
    assert len({task["seed"] for task in tasks}) == 30


def test_v0_prerequisite_requires_cpu_pass_and_records_hash(tmp_path: Path):
    summary = tmp_path / "v0.json"
    write_json_atomic(
        summary,
        {
            "status": "CPU_PASS / GPU_NOT_RUN",
            "gpu_audit_status": "AUDIT_ONLY / NOT_VALIDATED",
        },
    )
    actual = load_v0_prerequisite(summary)
    assert actual["status"] == "CPU_PASS / GPU_NOT_RUN"
    assert actual["sha256"] == file_sha256(summary)

    write_json_atomic(summary, {"status": "CPU_FAIL / GPU_NOT_RUN"})
    with pytest.raises(ValueError, match="not a passing CPU-only V0"):
        load_v0_prerequisite(summary)


def test_analyze_uses_saved_reduced_configuration(tmp_path: Path, monkeypatch):
    (tmp_path / "v1").mkdir()
    write_json_atomic(
        tmp_path / "campaign_manifest.json",
        {"v1_config": {"Nx": 16, "Ny": 16, "samples_per_schedule": 10}},
    )
    observed = {}

    def fake_analyze(root, *, nx, ny, samples, bootstrap_draws):
        observed.update(
            root=root,
            nx=nx,
            ny=ny,
            samples=samples,
            bootstrap_draws=bootstrap_draws,
        )
        return {"status": "test"}

    monkeypatch.setattr(campaign, "analyze_v1", fake_analyze)
    args = SimpleNamespace(
        smoke=False,
        v1_nx=20,
        v1_ny=48,
        v1_samples=100,
        bootstrap_samples=2000,
    )
    assert campaign.analyze_existing(tmp_path, args) == {"status": "test"}
    assert observed == {
        "root": tmp_path / "v1",
        "nx": 16,
        "ny": 16,
        "samples": 10,
        "bootstrap_draws": 2000,
    }


def test_reduced_v1_disables_automatic_escalation(tmp_path: Path, monkeypatch):
    calls = []
    monkeypatch.setattr(campaign, "run_schedule_preflight", lambda *_args, **_kwargs: {})

    def fake_system(args, output, *, nx, ny, samples, seed_offset=0):
        calls.append((output, nx, ny, samples, seed_offset))
        return {"status": "SCHEDULE_DEPENDENT", "reasons": []}

    monkeypatch.setattr(campaign, "run_v1_system", fake_system)
    args = SimpleNamespace(
        smoke=False,
        v1_nx=16,
        v1_ny=16,
        v1_samples=10,
        no_v1_auto_escalation=True,
    )
    result = campaign.run_v1(args, tmp_path)
    assert calls == [(tmp_path / "v1", 16, 16, 10, 0)]
    assert result["campaign_qualification"] == "S10_REDUCED"
    assert not result["auto_escalation_enabled"]
    assert not list((tmp_path / "v1").glob("escalation_Ny*"))
    assert "disabled" in result["reasons"][-1]


def test_main_records_reduced_manifest_and_v0_lineage(tmp_path: Path, monkeypatch):
    v0_summary = tmp_path / "prior_v0.json"
    write_json_atomic(
        v0_summary,
        {
            "status": "CPU_PASS / GPU_NOT_RUN",
            "gpu_audit_status": "AUDIT_ONLY / NOT_VALIDATED",
        },
    )
    output = tmp_path / "new_run"
    monkeypatch.setattr(campaign, "run_v1", lambda *_args, **_kwargs: {"status": "test"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "v1",
            "--output-dir",
            str(output),
            "--v1-nx",
            "16",
            "--v1-ny",
            "16",
            "--v1-samples",
            "10",
            "--no-v1-auto-escalation",
            "--v0-summary",
            str(v0_summary),
        ],
    )
    assert campaign.main() == 0
    manifest = json.loads((output / "campaign_manifest.json").read_text())
    assert manifest["v1_config"]["total_trajectories"] == 30
    assert manifest["v1_config"]["initial_target_cycles"] == 64
    assert manifest["v1_config"]["max_target_cycles"] == 288
    assert manifest["v1_config"]["checkpoint_stride"] == 8
    assert manifest["v1_config"]["qualification"] == "S10_REDUCED"
    assert not manifest["v1_config"]["auto_escalation_enabled"]
    assert manifest["v0_prerequisite"]["sha256"] == file_sha256(v0_summary)


def test_benchmark_reuses_matching_completed_measurement(tmp_path: Path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        campaign,
        "run_v1_trajectory",
        lambda task: calls.append(task) or {"status": "complete"},
    )
    ticks = iter((10.0, 25.0))
    monkeypatch.setattr(campaign.time, "monotonic", lambda: next(ticks))
    args = SimpleNamespace(v1_nx=16, v1_ny=16, v1_samples=10, seed=9, workers=28)
    first = campaign.run_v1_benchmark(args, tmp_path)
    second = campaign.run_v1_benchmark(args, tmp_path)
    assert first == second
    assert first["elapsed_seconds"] == 15.0
    assert first["projected_initial_stage_seconds"] == 30.0
    assert len(calls) == 1
