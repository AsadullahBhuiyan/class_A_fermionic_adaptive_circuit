from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


PROJECT = (
    Path(__file__).resolve().parents[1]
    / "00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot"
)
MODULE_PATH = PROJECT / "run_wall_pump_endpoint_cpu_fallback.py"
SPEC = importlib.util.spec_from_file_location("wall_pump_endpoint_cpu_fallback_under_test", MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_locked_task_matrix_and_two_lane_shard_assignment() -> None:
    config = runner.load_config()
    runner.validate_config(config)
    rows = runner.tasks(config)
    assert len(rows) == 1000
    assert len({row.seed for row in rows}) == 1000
    assert len({row.task_id for row in rows}) == 1000
    assert {row.lane for row in rows} == {0, 1}
    assert abs(sum(row.lane == 0 for row in rows) - sum(row.lane == 1 for row in rows)) <= 5
    shard_lanes: dict[tuple[object, ...], set[int]] = {}
    for row in rows:
        key = row.collection, row.protocol, row.nx, row.wall, row.shard_index
        shard_lanes.setdefault(key, set()).add(row.lane)
    assert len(shard_lanes) == 200
    assert all(len(lanes) == 1 for lanes in shard_lanes.values())


def test_nested_native_checkpoint_round_trip(tmp_path: Path) -> None:
    config = runner.load_config()
    hashes = runner.source_hashes()
    task = runner.tasks(config)[0]
    charge = np.full(49, -1, dtype=np.int64)
    charge[:6] = np.arange(5, 11)
    state = {
        "version": 2,
        "completed_cycles": 5,
        "signature": {"shape": [1, 2], "flag": True},
        "rng_states": {"dynamics": {"state": 12345678901234567890}},
        "native_state": {
            "frame": np.eye(3, 2, dtype=np.complex128),
            "rank": 2,
            "labels": ("physical_frame", None),
        },
    }
    runner.save_checkpoint(tmp_path, task, config, hashes, state, charge, 1.5)
    loaded, loaded_charge, elapsed, reason = runner.load_checkpoint(
        tmp_path, task, config, hashes
    )
    assert reason == "verified"
    assert loaded is not None and loaded_charge is not None
    assert loaded["completed_cycles"] == 5
    assert loaded["native_state"]["labels"] == ("physical_frame", None)
    np.testing.assert_array_equal(loaded["native_state"]["frame"], state["native_state"]["frame"])
    np.testing.assert_array_equal(loaded_charge, charge)
    assert elapsed == 1.5


def test_five_complete_checkpoints_publish_compatible_shard(tmp_path: Path) -> None:
    config = runner.load_config()
    hashes = runner.source_hashes()
    all_tasks = runner.tasks(config)
    first = all_tasks[0]
    members = runner._shard_members(all_tasks, first)
    dimension, rank = 2 * first.nx * 24, 4
    frame = np.zeros((dimension, rank), dtype=np.complex128)
    frame[:rank] = np.eye(rank, dtype=np.complex128)
    for member in members:
        charge = np.full(49, first.nx * 24, dtype=np.int64)
        state = {
            "version": 2,
            "completed_cycles": 48,
            "signature": {"test": True},
            "rng_states": {},
            "native_state": {
                "representation": "physical_frame",
                "frame": frame,
                "physical_dimension": dimension,
                "rank": rank,
                "log_weight": 0.0,
                "min_rank": rank,
                "max_rank": rank,
                "gram_residual": 0.0,
            },
        }
        runner.save_checkpoint(tmp_path, member, config, hashes, state, charge, 2.0)
    published, reason = runner.publish_shard_if_ready(
        tmp_path, all_tasks, first, config, hashes
    )
    assert published and reason == "published and verified"
    assert runner.verify_shard(tmp_path, first, config, hashes) == (True, "verified")
    result, completion = runner.shard_paths(tmp_path, first)
    assert result.is_file() and completion.is_file()
    with np.load(result, allow_pickle=False) as saved:
        assert str(saved["schema"].item()) == "wall_pump_width_endpoint_shard_v1"
        assert saved["frames"].shape == (5, dimension, rank)
        assert saved["execution_backend"].item() == "canonical_cpu"
    for member in members:
        data, receipt = runner.checkpoint_paths(tmp_path, member)
        assert not data.exists() and not receipt.exists()


def test_corrupt_checkpoint_is_not_resumed(tmp_path: Path) -> None:
    config = runner.load_config()
    hashes = runner.source_hashes()
    task = runner.tasks(config)[0]
    charge = np.full(49, -1, dtype=np.int64)
    charge[0] = 1
    runner.save_checkpoint(
        tmp_path,
        task,
        config,
        hashes,
        {"completed_cycles": 0, "native_state": {"frame": np.eye(2)}},
        charge,
        0.1,
    )
    data, _ = runner.checkpoint_paths(tmp_path, task)
    with data.open("ab") as handle:
        handle.write(b"corrupt")
    state, loaded_charge, _, reason = runner.load_checkpoint(tmp_path, task, config, hashes)
    assert state is None and loaded_charge is None
    assert "mismatch" in reason
