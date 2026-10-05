from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = REPO_ROOT / "00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot"
RUNNER_PATH = PILOT_ROOT / "run_parent_schrodinger_rk4_s50.py"
CONFIG_PATH = PILOT_ROOT / "campaign_config.parent_schrodinger_rk4_n20x24_s50_tau1e4_v1.json"
REFINEMENT_RUNNER_PATH = PILOT_ROOT / "run_parent_schrodinger_rk4_refinement.py"
REFINEMENT_CONFIG_PATH = PILOT_ROOT / "campaign_config.parent_schrodinger_rk4_n20x24_s2_tau1e4_dt_half_v1.json"
MONITOR_PATH = PILOT_ROOT / "monitor_parent_schrodinger_rk4_s50.py"
REFINEMENT_ENTRYPOINT = PILOT_ROOT / "parent_schrodinger_rk4_refinement_tmux_entrypoint.sh"


def _module():
    spec = importlib.util.spec_from_file_location("parent_schrodinger_rk4_test_module", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    old = list(sys.path)
    try:
        sys.path.insert(0, str(PILOT_ROOT))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old
    return module


def _refinement_module():
    spec = importlib.util.spec_from_file_location(
        "parent_schrodinger_rk4_refinement_test_module", REFINEMENT_RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    old = list(sys.path)
    try:
        sys.path.insert(0, str(PILOT_ROOT))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old
    return module


def _monitor_module():
    spec = importlib.util.spec_from_file_location(
        "parent_schrodinger_rk4_monitor_test_module", MONITOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    old = list(sys.path)
    try:
        sys.path.insert(0, str(PILOT_ROOT))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old
    return module


def test_locked_task_table_and_discretization() -> None:
    runner = _module()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    runner.validate_config(config)
    tasks = runner.tasks(config)
    assert len(tasks) == 100
    assert len({row["task_id"] for row in tasks}) == 100
    assert set(config["source_campaign"]["sample_ids"]) == set(range(0, 100, 4))
    assert {row["wall"] for row in tasks} == {"soft", "hard"}
    assert {row["direction"] for row in tasks} == {"ccw", "cw"}
    assert config["evolution"]["ramp_time"] == 10000.0
    assert config["evolution"]["flux_intervals"] * config["evolution"]["steps_per_interval"] == 40960
    assert np.isclose(config["evolution"]["ramp_time"] / 40960, 0.244140625)


def test_zero_flux_parent_leaves_occupied_projector_unchanged() -> None:
    runner = _module()
    rng = np.random.default_rng(123)
    raw = rng.normal(size=(16, 7)) + 1j * rng.normal(size=(16, 7))
    frame, _ = np.linalg.qr(raw)
    frame = np.asarray(frame, dtype=np.complex128, order="F")
    projector = frame @ frame.conj().T
    h0 = np.asarray(np.eye(16) - 2.0 * projector, dtype=np.complex128, order="F")
    ny = 4
    y = np.repeat(np.arange(ny), 4)
    dy = y[:, None] - y[None, :]
    dy = ((dy + ny // 2) % ny) - ny // 2
    dy_index = np.asarray(dy + ny // 2, dtype=np.int8)
    evolved = frame.copy()
    for step in range(20):
        evolved = runner.rk4_step(
            evolved, step=step, dt=0.05, ramp_time=1e30, sigma=1,
            h0=h0, dy_index=dy_index,
            dy_values=np.arange(-ny // 2, ny // 2, dtype=float), ny=ny,
        )
    evolved, _, post = runner._fix_qr_gauge(evolved)
    assert post < 1e-12
    assert np.max(np.abs(evolved @ evolved.conj().T - projector)) < 1e-11


def test_completion_pair_detects_corruption(tmp_path: Path) -> None:
    runner = _module()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    task = runner.tasks(config)[0]
    source_row = {"path": "/source", "name": "sample_000.npz", "bytes": 7, "sha256": "a", "source_config_hash": "b"}
    hashes = {"source_campaign_runner": "c", "campaign_runner": "d"}
    config_hash = runner.scientific_config_hash(config)
    count = config["evolution"]["flux_intervals"] + 1
    arrays = {
        "phi": np.linspace(0, 2 * np.pi, count),
        "time": np.linspace(0, config["evolution"]["ramp_time"], count),
        "N_left": np.ones(count), "N_right": np.ones(count),
        "delta_N_left": np.zeros(count), "delta_N_right": np.zeros(count),
        "delta_N_total": np.zeros(count), "q_x": np.zeros(count),
        "density_x": np.zeros((count, 20)), "current_left": np.zeros(count),
        "energy": np.zeros(count), "maximum_post_qr_gram_residual": np.asarray(0.0),
    }
    runner.publish_result(
        tmp_path, task, arrays, config_hash=config_hash, hashes=hashes,
        source_row=source_row, elapsed_seconds=1.0,
    )
    assert runner.verify_result(
        tmp_path, task, config_hash=config_hash, hashes=hashes,
        source_row=source_row, config=config,
    )[0]
    result, _ = runner.result_paths(tmp_path, task)
    with result.open("ab") as handle:
        handle.write(b"corrupt")
    assert not runner.verify_result(
        tmp_path, task, config_hash=config_hash, hashes=hashes,
        source_row=source_row, config=config,
    )[0]


def test_checkpoint_pair_roundtrip_and_corruption(tmp_path: Path) -> None:
    runner = _module()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    task = runner.tasks(config)[0]
    metadata = {"task": task["task_id"], "identity": "test"}
    count = config["evolution"]["flux_intervals"] + 1
    arrays = runner._initial_arrays(count, 20)
    for key in arrays:
        arrays[key][:5] = 0.0
    raw = np.zeros((960, 2), dtype=np.complex128)
    raw[0, 0] = raw[1, 1] = 1.0
    runner._write_checkpoint(
        tmp_path, task, frame=raw, completed_step=4 * 320,
        completed_interval=4, arrays=arrays, maximum_pre_qr_gram=1e-7,
        maximum_post_qr_gram=1e-14, metadata=metadata,
    )
    loaded = runner._load_checkpoint(
        tmp_path, task, metadata=metadata, count=count, nx=20
    )
    assert loaded is not None
    assert loaded["completed_step"] == 1280
    assert loaded["completed_interval"] == 4
    checkpoint, _ = runner.checkpoint_paths(tmp_path, task)
    with checkpoint.open("ab") as handle:
        handle.write(b"corrupt")
    assert runner._load_checkpoint(
        tmp_path, task, metadata=metadata, count=count, nx=20
    ) is None


def test_half_step_refinement_is_isolated_and_exactly_halves_dt() -> None:
    refinement = _refinement_module()
    spec = refinement.load_refinement(REFINEMENT_CONFIG_PATH)
    config = refinement.refined_config(spec)
    tasks = refinement.selected_tasks(config, spec)
    assert len(tasks) == 4
    assert {row["wall"] for row in tasks} == {"soft", "hard"}
    assert {row["direction"] for row in tasks} == {"ccw", "cw"}
    assert {row["sample_id"] for row in tasks} == {0}
    assert config["evolution"]["steps_per_interval"] == 640
    assert np.isclose(
        config["evolution"]["ramp_time"]
        / (config["evolution"]["flux_intervals"] * config["evolution"]["steps_per_interval"]),
        0.1220703125,
    )
    assert refinement.DEFAULT_OUTPUT.name != refinement.primary.DEFAULT_OUTPUT.name
    entrypoint = REFINEMENT_ENTRYPOINT.read_text(encoding="utf-8")
    assert "taskset -c 28-31" in entrypoint
    assert "taskset -c 0-3" not in entrypoint


def test_read_only_monitor_rejects_corruption_without_deleting_it(tmp_path: Path) -> None:
    runner = _module()
    monitor = _monitor_module()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    task = runner.tasks(config)[0]
    metadata = {"task": task["task_id"], "identity": "monitor-test"}
    count = config["evolution"]["flux_intervals"] + 1
    arrays = runner._initial_arrays(count, 20)
    for key in arrays:
        arrays[key][:5] = 0.0
    frame = np.zeros((960, 2), dtype=np.complex128)
    frame[0, 0] = frame[1, 1] = 1.0
    runner._write_checkpoint(
        tmp_path,
        task,
        frame=frame,
        completed_step=4 * 320,
        completed_interval=4,
        arrays=arrays,
        maximum_pre_qr_gram=1e-7,
        maximum_post_qr_gram=1e-14,
        metadata=metadata,
    )

    valid = monitor.verify_checkpoint(
        tmp_path, task, metadata=metadata, config=config
    )
    assert valid["status"] == "checkpoint"
    assert valid["completed_interval"] == 4

    checkpoint, receipt = runner.checkpoint_paths(tmp_path, task)
    with checkpoint.open("ab") as handle:
        handle.write(b"corrupt")
    corrupt_size = checkpoint.stat().st_size
    invalid = monitor.verify_checkpoint(
        tmp_path, task, metadata=metadata, config=config
    )
    assert invalid["status"] == "invalid"
    assert "mismatch" in invalid["reason"]
    assert checkpoint.is_file() and checkpoint.stat().st_size == corrupt_size
    assert receipt.is_file()
