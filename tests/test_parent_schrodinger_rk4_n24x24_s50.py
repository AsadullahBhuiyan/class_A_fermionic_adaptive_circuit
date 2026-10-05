from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = REPO_ROOT / "00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot"
RUNNER_PATH = PILOT_ROOT / "run_parent_schrodinger_rk4_n24x24_s50.py"
CONFIG_PATH = PILOT_ROOT / "campaign_config.parent_schrodinger_rk4_n24x24_s50_tau1e4_v1.json"
BASELINE_PATH = PILOT_ROOT / "results/N24x24_parent_schrodinger_rk4_s50_tau1e4_v1/preregistered_comparison_baseline.json"
ENTRYPOINT = PILOT_ROOT / "parent_schrodinger_rk4_n24x24_tmux_entrypoint.sh"


def _module():
    spec = importlib.util.spec_from_file_location("parent_rk4_n24_test_module", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    old = list(sys.path)
    try:
        sys.path.insert(0, str(PILOT_ROOT))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old
    return module


def test_locked_geometry_task_table_and_kernel_reuse() -> None:
    runner = _module()
    config = runner.load_config(CONFIG_PATH)
    runner.validate_config(config)
    rows = runner.tasks(config)
    assert len(rows) == 100
    assert len({row["task_id"] for row in rows}) == 100
    assert config["source_campaign"]["sample_ids"] == list(range(0, 100, 4))
    assert config["geometry"] == {
        "Nx": 24,
        "Ny": 24,
        "wall_x": [6, 18],
        "wall_separation": 12,
        "left_x_stop_exclusive": 12,
        "periodic_direction": "y",
    }
    assert config["evolution"]["flux_intervals"] * config["evolution"]["steps_per_interval"] == 40960
    assert np.isclose(config["evolution"]["ramp_time"] / 40960, 0.244140625)
    assert runner.kernel.rk4_step.__code__.co_filename == str(
        PILOT_ROOT / "run_parent_schrodinger_rk4_s50.py"
    )


def test_all_selected_endpoint_pairs_are_verified_and_complex128() -> None:
    runner = _module()
    config = runner.load_config(CONFIG_PATH)
    context = runner.source_context(config)
    assert len(context["rows"]) == 50
    for row in context["rows"].values():
        frame = runner._load_frame(row, config)
        assert frame.dtype == np.complex128
        assert frame.shape[0] == 2 * 24 * 24


def test_n24_checkpoint_roundtrip_and_corruption(tmp_path: Path) -> None:
    runner = _module()
    config = runner.load_config(CONFIG_PATH)
    task = runner.tasks(config)[0]
    count = config["evolution"]["flux_intervals"] + 1
    arrays = runner.kernel._initial_arrays(count, 24)
    for value in arrays.values():
        value[:5] = 0.0
    frame = np.zeros((1152, 2), dtype=np.complex128)
    frame[0, 0] = frame[1, 1] = 1.0
    metadata = {"task_id": task["task_id"], "test": True}
    runner.kernel._write_checkpoint(
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
    loaded = runner.kernel._load_checkpoint(
        tmp_path, task, metadata=metadata, count=count, nx=24
    )
    assert loaded is not None
    assert loaded["frame"].shape == (1152, 2)
    checkpoint, _ = runner.kernel.checkpoint_paths(tmp_path, task)
    with checkpoint.open("ab") as handle:
        handle.write(b"corrupt")
    assert runner.kernel._load_checkpoint(
        tmp_path, task, metadata=metadata, count=count, nx=24
    ) is None


def test_result_pair_detects_partial_and_checksum_corruption(tmp_path: Path) -> None:
    runner = _module()
    config = runner.load_config(CONFIG_PATH)
    task = runner.tasks(config)[0]
    source_row = {
        "path": "/source", "name": "sample_000.npz", "bytes": 1,
        "sha256": "source", "source_config_hash": "source-config",
    }
    hashes = {key: "hash" for key in runner.SOURCE_PATHS}
    config_hash = runner.scientific_config_hash(config)
    count = config["evolution"]["flux_intervals"] + 1
    arrays = {
        "phi": np.linspace(0, 2 * np.pi, count),
        "time": np.linspace(0, 10000, count),
        "N_left": np.ones(count), "N_right": np.ones(count),
        "delta_N_left": np.zeros(count), "delta_N_right": np.zeros(count),
        "delta_N_total": np.zeros(count), "q_x": np.zeros(count),
        "density_x": np.zeros((count, 24)), "current_left": np.zeros(count),
        "energy": np.zeros(count), "maximum_post_qr_gram_residual": np.asarray(0.0),
    }
    runner.kernel.publish_result(
        tmp_path, task, arrays, config_hash=config_hash, hashes=hashes,
        source_row=source_row, elapsed_seconds=1.0,
    )
    assert runner.kernel.verify_result(
        tmp_path, task, config_hash=config_hash, hashes=hashes,
        source_row=source_row, config=config,
    )[0]
    result, completion = runner.kernel.result_paths(tmp_path, task)
    completion.unlink()
    assert not runner.kernel.verify_result(
        tmp_path, task, config_hash=config_hash, hashes=hashes,
        source_row=source_row, config=config,
    )[0]
    runner.kernel.publish_result(
        tmp_path, task, arrays, config_hash=config_hash, hashes=hashes,
        source_row=source_row, elapsed_seconds=1.0,
    )
    with result.open("ab") as handle:
        handle.write(b"corrupt")
    assert not runner.kernel.verify_result(
        tmp_path, task, config_hash=config_hash, hashes=hashes,
        source_row=source_row, config=config,
    )[0]


def test_dependency_gate_accepts_only_complete_primary_and_converged_refinement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _module()
    config = runner.load_config(CONFIG_PATH)
    monkeypatch.setattr(runner.kernel, "inventory", lambda *_: {"paths": {str(i): (True, "verified", {}) for i in range(100)}})
    refinement_root = tmp_path / "refinement"
    refinement_root.mkdir()
    config["dependency_gate"]["refinement_output_root"] = str(refinement_root)
    config["dependency_gate"]["primary_output_root"] = str(tmp_path / "primary")
    # Absolute paths remain absolute when joined to PROJECT_ROOT.
    summary = {
        "schema": "parent_schrodinger_rk4_step_halving_summary_v1",
        "maximum_endpoint_absolute_difference": 0.009,
        "maximum_path_absolute_difference": 0.019,
        "rows": [],
    }
    (refinement_root / "step_halving_summary.json").write_text(json.dumps(summary))
    assert runner.verify_dependencies(config)["primary_complete"] == 100
    summary["maximum_endpoint_absolute_difference"] = 0.011
    (refinement_root / "step_halving_summary.json").write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="endpoint error"):
        runner.verify_dependencies(config)


def test_preregistered_baseline_and_tmux_queue_contract() -> None:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["recorded_before_any_completed_rk4_path"] is True
    assert baseline["selection"]["rule"] == "sample IDs 0,4,...,96 for each wall"
    assert baseline["static_reference"]["soft"]["events"] == 23
    assert baseline["static_reference"]["hard"]["events"] == 11
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "parent_schrodinger_rk4_N20x24_s50_tau1e4_v1" in text
    assert "parent_schrodinger_rk4_dt_half_followup" in text
    assert "taskset -c 28-55" in text
    assert "--workers 28" in text
    assert "--resume" in text and "--analyze" in text
