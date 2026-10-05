from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
PARENT = REPO / "00_WORKSPACE/CURRENT/final_production_new_designs"
BUNDLE = PARENT / "09_pure_tangent_replay_acquisition"
GPU = BUNDLE / "gpu_tangent_replay"
NOTEBOOK = BUNDLE / "run_pure_tangent_gpu_replay.ipynb"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for candidate in (BUNDLE, BUNDLE / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
RUNNER = _load(GPU / "run_gpu_tangent_replay.py", "tested_pure_tangent_gpu_replay")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def test_locked_gpu_grid_is_48_batches_of_25_and_1200_rows() -> None:
    config = RUNNER.load_config()
    assert config == RUNNER.expected_config()
    assert config["samples_per_gpu_task"] == 25
    assert config["tangent_basis_mode"] == "batched_canonical_from_occupied_empty_basis"
    assert config["dtype"] == "complex128"
    assert config["perfect_correction"] is True
    assert config["maximum_task_runtime_seconds"] == 3600.0
    assert config["maximum_peak_cuda_reserved_gib"] == 36.0
    assert config["saved_products"] == {
        "final_scale_separated_one_leg_cocycle": True,
        "per_cycle_qr_log_increments": True,
        "full_window_one_leg_singular_logs": True,
        "late_window_one_leg_singular_logs": True,
        "slowest_particle_hole_rates_and_x_profiles": True,
        "per_sample_occupied_empty_block_sizes": True,
        "per_cycle_dense_jacobians": False,
        "choi_covariance": False,
        "covariance_history": False,
        "intermediate_occupied_frames": False,
    }
    tasks = RUNNER.expand_tasks()
    assert len(tasks) == 48
    assert {task.sample_count for task in tasks} == {25}
    assert sum(task.sample_count for task in tasks) == 1200
    assert len({task.task_id for task in tasks}) == 48
    assert len({(task.construction, task.ny, task.alpha_1) for task in tasks}) == 12
    assert sorted(index for task in tasks for index in task.global_sample_indices) == list(
        range(1200)
    )


def test_batched_occupied_empty_basis_spans_each_sample_projector() -> None:
    rng = np.random.default_rng(1701)
    frames = []
    for _ in range(3):
        raw = rng.normal(size=(8, 4)) + 1j * rng.normal(size=(8, 4))
        frame, _ = np.linalg.qr(raw, mode="reduced")
        frames.append(frame)
    frames_np = np.asarray(frames, dtype=np.complex128)

    class Model:
        device = torch.device("cpu")

        @staticmethod
        def active_top_layer_indices(*, meas_slab_only: bool) -> torch.Tensor:
            assert meas_slab_only is False
            return torch.arange(8)

    basis, active, blocks, occupations, defects = RUNNER.occupied_empty_basis(
        Model(),
        frames_np,
        np.full(3, 4, dtype=np.int64),
        hard=False,
        purity_tolerance=2e-12,
    )
    assert tuple(basis.shape) == (3, 8, 8)
    assert torch.equal(blocks, torch.full((3, 2), 4, dtype=torch.int64))
    assert torch.equal(active, torch.arange(8))
    assert float(defects.max()) < 2e-15
    for row in range(3):
        expected = torch.from_numpy(frames_np[row]) @ torch.from_numpy(frames_np[row]).mH
        occupied = basis[row, :, :4]
        torch.testing.assert_close(occupied @ occupied.mH, expected, atol=2e-14, rtol=2e-14)
        torch.testing.assert_close(basis[row].mH @ basis[row], torch.eye(8, dtype=torch.complex128))
    assert tuple(occupations.shape) == (3, 8)


def test_batched_basis_keeps_per_sample_charge_blocks() -> None:
    frame = np.zeros((2, 8, 4), dtype=np.complex128)
    frame[0, :3, :3] = np.eye(3)
    frame[1, :4, :4] = np.eye(4)

    class Model:
        device = torch.device("cpu")

        @staticmethod
        def active_top_layer_indices(*, meas_slab_only: bool) -> torch.Tensor:
            return torch.arange(8)

    basis, _active, blocks, _occupations, defects = RUNNER.occupied_empty_basis(
        Model(),
        frame,
        np.asarray([3, 4], dtype=np.int64),
        hard=False,
        purity_tolerance=1e-14,
    )
    assert tuple(basis.shape) == (2, 8, 8)
    assert torch.equal(blocks, torch.tensor([[3, 5], [4, 4]]))
    assert float(defects.max()) == 0.0


def test_batched_canonical_capture_reconstructs_each_one_leg_product() -> None:
    torch.manual_seed(741)
    batch = 2
    dimension = 4
    basis_raw = torch.randn(batch, dimension, dimension, dtype=torch.complex128)
    basis, _ = torch.linalg.qr(basis_raw)
    frame_raw = torch.randn(batch, dimension, dimension, dtype=torch.complex128)
    frame, _ = torch.linalg.qr(frame_raw)
    core = torch.randn(batch, dimension, dimension, dtype=torch.complex128)
    core_norm = torch.linalg.matrix_norm(core, ord="fro", dim=(-2, -1))
    core = core / core_norm[:, None, None]
    scale = torch.tensor([0.7, -0.2], dtype=torch.float64)

    @dataclass
    class State:
        frame: torch.Tensor
        ranks: torch.Tensor

    capture = RUNNER.CanonicalTangentCapture(physical_cycles=1, start_cycle=1)
    capture(
        cycle=1,
        spectra=torch.zeros(batch, dimension),
        G=State(frame=frame.clone(), ranks=torch.full((batch,), 2)),
        lyapunov_frame=frame,
        lyapunov_log_diag=torch.zeros(batch, dimension),
        lyapunov_cycle_null_mask=torch.zeros(batch, dimension, dtype=torch.bool),
        lyapunov_null_counts=torch.zeros(batch, dtype=torch.int64),
        lyapunov_min_branch_probability=torch.ones(batch),
        lyapunov_min_abs_born_denominator=torch.ones(batch),
        lyapunov_invalid_branch_count=torch.zeros(batch, dtype=torch.int64),
        lyapunov_core_hat=core,
        lyapunov_core_log_scale=scale,
        lyapunov_core_null_count=torch.zeros(batch, dtype=torch.int64),
    )
    arrays = capture.finalize(
        prefix="full",
        initial_basis=basis,
        active_indices=torch.arange(dimension),
        block_sizes=(2, 2),
        nx=1,
        ny=2,
        slow_mode_count=4,
        singular_tolerance=1e-14,
        materialize_cocycle=True,
    )
    restored = arrays["final_cocycle_hat"] * np.exp(
        arrays["final_cocycle_log_scale"][:, None, None]
    )
    expected = (
        frame @ core @ basis.mH * torch.exp(scale)[:, None, None]
    ).numpy()
    np.testing.assert_allclose(restored, expected, atol=2e-13, rtol=2e-13)
    assert arrays["full_qr_log_increments"].shape == (batch, 1, dimension)
    assert arrays["full_slow_x_profiles"].shape == (batch, 4, 1)
    assert arrays["full_block_sizes"].shape == (batch, 2)


def test_drive_temp_readback_failure_preserves_old_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "local.bin"
    final = tmp_path / "drive" / "result.bin"
    local.write_bytes(b"new")
    final.parent.mkdir()
    final.write_bytes(b"old")
    original = RUNNER.sha256_file

    def fail_temp(path: Path) -> str:
        if path.name.startswith(".result.bin"):
            return "0" * 64
        return original(path)

    monkeypatch.setattr(RUNNER, "sha256_file", fail_temp)
    with pytest.raises(OSError, match="temporary checksum"):
        RUNNER.publish_file(local, final)
    assert final.read_bytes() == b"old"


def test_notebook_manifest_sources_and_visible_progress_contract() -> None:
    assert _sha256(BUNDLE / "src/classA_U1FGTN_gpu.py") == _sha256(
        REPO / "src/fgtn/classA_U1FGTN_gpu.py"
    )
    assert _sha256(BUNDLE / "src/occupied_frame_gpu.py") == _sha256(
        REPO / "src/fgtn/occupied_frame_gpu.py"
    )
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    all_source = "\n".join(_source(cell) for cell in notebook["cells"])
    assert "REPORT_ONLY = False" in all_source
    assert "MAX_NEW_TASKS = None" in all_source
    assert "VERIFY_ALL_INPUT_CHECKSUMS = False" in all_source
    assert "A100" in all_source and "complex128" in all_source
    assert "gpu_tangent_replay/run_gpu_tangent_replay.py" in all_source
    assert "subprocess.run(command, check=True, env=environment)" in all_source
    assert "PYTHONUNBUFFERED" in all_source
    assert "TQDM_MININTERVAL" in all_source
    assert "shutil.copytree" not in all_source
    final = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"][-1]
    assert _source(final).strip().splitlines() == [
        "from google.colab import runtime",
        "runtime.unassign()",
        "print('done')",
    ]

    manifest = json.loads(
        (GPU / "deployment_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == "pure_tangent_gpu_replay_deployment_manifest_v1"
    for relative, record in manifest["files"].items():
        path = BUNDLE / relative
        assert path.stat().st_size == record["bytes"]
        assert _sha256(path) == record["sha256"]
