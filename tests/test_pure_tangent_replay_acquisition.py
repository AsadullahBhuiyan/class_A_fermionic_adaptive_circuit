from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
PARENT = REPO / "00_WORKSPACE/CURRENT/final_production_new_designs"
BUNDLE = PARENT / "09_pure_tangent_replay_acquisition"
NOTEBOOK = BUNDLE / "run_pure_tangent_replay_acquisition.ipynb"


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
OBSERVER = _load(BUNDLE / "replay_record_observer.py", "tested_replay_record_observer")
RUNNER = _load(BUNDLE / "run_campaign.py", "tested_replay_acquisition_runner")


def _cell_source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _notebook_config() -> tuple[dict, str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        _cell_source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and "CONFIG = {" in _cell_source(cell)
    )
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "CONFIG" for target in node.targets)
    )
    return ast.literal_eval(assignment.value), source


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_locked_grid_expands_to_12_cases_32_batches_1200_samples() -> None:
    config, _ = _notebook_config()
    assert RUNNER.validate_config(config) == RUNNER.expected_config()
    assert config["batch_size_by_Ny"] == {"24": 50, "28": 50, "32": 25}
    assert config["Nx"] == 20
    assert config["Ny_values"] == [24, 28, 32]
    assert config["alpha_1_values"] == [1.0, 3.0]
    assert config["alpha_2"] == 30.0
    assert config["nshell"] == 1
    assert config["samples_per_case"] == 100
    assert config["cycles_multiplier"] == 2
    assert config["sequence"] == "raster_y"
    assert config["perfect_correction"] is True
    assert config["state_representation"] == "physical_frame"
    assert config["dtype"] == "complex128"
    assert config["constructions"] == {
        "hard": {"DW": True, "dw_truncation": True, "meas_slab_only": True},
        "soft": {"DW": True, "dw_truncation": False, "meas_slab_only": False},
    }
    assert config["saved_products"]["tangent_frame"] is False
    assert config["saved_products"]["choi_covariance"] is False
    assert config["saved_products"]["covariance_history"] is False

    tasks = RUNNER.expand_tasks(config)
    assert len(tasks) == 32
    assert sum(task.sample_count for task in tasks) == 1200
    assert len({(task.construction, task.ny, task.alpha_1) for task in tasks}) == 12
    assert len({task.task_id for task in tasks}) == 32
    assert len({task.seed for task in tasks}) == 32
    assert sorted(index for task in tasks for index in task.global_sample_indices) == list(range(1200))
    for task in tasks:
        assert task.sample_count == {24: 50, 28: 50, 32: 25}[task.ny]
        assert task.cycles == 2 * task.ny


def test_boolean_record_pack_round_trip_and_padding_validation() -> None:
    rng = np.random.default_rng(99)
    values = rng.integers(0, 2, size=(3, 5, 7, 4), dtype=np.uint8).astype(bool)
    packed, shape = OBSERVER.pack_boolean_record(values)
    restored = OBSERVER.unpack_boolean_record(packed, shape)
    assert restored.dtype == np.bool_
    np.testing.assert_array_equal(restored, values)
    with pytest.raises(ValueError, match="byte shape"):
        OBSERVER.unpack_boolean_record(packed[:, :-1], shape)


def _run_and_replay(*, hard: bool) -> tuple[dict, dict, float]:
    nx = ny = 4
    samples = 2
    cycles = 2
    torch.manual_seed(1409)
    model = RUNNER.classA_U1FGTN_gpu(
        Nx=nx,
        Ny=ny,
        DW=True,
        nshell=1,
        filling_frac=0.5,
        alpha_1=1.0,
        alpha_2=30.0,
        trial_orbitals="X",
        dw_truncation=hard,
        triv_region_local_mode=False,
        device="cpu",
        dtype="complex128",
        backend="local",
    )
    walls = tuple(int(value) for value in model.DW_loc)
    updates = ((walls[1] - walls[0] + 1) if hard else nx) * ny
    observer = OBSERVER.ReplayRecordObserver(
        samples=samples, cycles=cycles, updates_per_cycle=updates
    )
    result = model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=cycles,
        postselect=False,
        postselect_probability=0.0,
        perfect_correction=True,
        samples=samples,
        init_mode="default",
        save=False,
        n_a=0.5,
        sequence="raster_y",
        meas_slab_only=hard,
        batch_size=samples,
        return_data=True,
        state_representation="physical_frame",
        native_cycle_observer=observer.capture_native_cycle,
        record_observer=observer.record_event,
        track_choi=False,
        return_native_state=True,
        require_no_covariance_materialization=True,
        frame_reorthonormalize_interval=1,
    )
    payload = observer.result_arrays()
    final_frame, final_ranks = OBSERVER.native_frame_arrays(result["native_final"])
    outcomes = OBSERVER.unpack_boolean_record(
        payload["record_outcomes_packed"], payload["record_outcomes_shape"]
    )

    replay_model = RUNNER.classA_U1FGTN_gpu(
        Nx=nx,
        Ny=ny,
        DW=True,
        nshell=1,
        filling_frac=0.5,
        alpha_1=1.0,
        alpha_2=30.0,
        trial_orbitals="X",
        dw_truncation=hard,
        triv_region_local_mode=False,
        device="cpu",
        dtype="complex128",
        backend="local",
    )
    replay = replay_model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=cycles,
        postselect=False,
        postselect_probability=0.0,
        perfect_correction=True,
        samples=1,
        init_mode="default",
        frame_init=payload["initial_frame"][0:1],
        frame_ranks=payload["initial_ranks"][0:1],
        frame_init_prepared=hard,
        save=False,
        n_a=0.5,
        sequence="raster_y",
        meas_slab_only=hard,
        batch_size=1,
        return_data=True,
        state_representation="physical_frame",
        frozen_schedule=payload["record_schedule"][0:1],
        frozen_outcomes=outcomes[0:1],
        track_choi=False,
        return_native_state=True,
        require_no_covariance_materialization=True,
        frame_reorthonormalize_interval=1,
    )
    replay_frame, replay_ranks = OBSERVER.native_frame_arrays(replay["native_final"])
    assert replay_ranks[0] == final_ranks[0]
    rank = int(final_ranks[0])
    direct = torch.from_numpy(final_frame[0, :, :rank])
    restored = torch.from_numpy(replay_frame[0, :, :rank])
    projector_error = float(torch.max(torch.abs(direct @ direct.mH - restored @ restored.mH)))
    return result, replay, projector_error


@pytest.mark.parametrize("hard", [False, True])
def test_saved_initial_frame_and_record_exactly_replay_endpoint(hard: bool) -> None:
    result, replay, projector_error = _run_and_replay(hard=hard)
    assert bool(result["exterior_preparation_performed"]) is hard
    assert replay["exterior_preparation_performed"] is False
    assert result["state_representation_resolved"] == "physical_frame"
    assert result["covariance_materialization_count"] == 0
    assert result["choi_tracked"] is False
    assert result["lyapunov_tracked"] is False
    assert projector_error < 2.0e-14


def test_completion_resume_rejects_partial_and_corrupted_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = RUNNER.expected_config()
    task = RUNNER.expand_tasks(config)[0]
    hashes = RUNNER.source_hashes()
    config_hash = RUNNER.config_sha256(config)
    result_path, completion_path = RUNNER.result_paths(tmp_path, task)
    result_path.parent.mkdir(parents=True)
    result_path.write_bytes(b"stable replay payload")
    complete, reason = RUNNER.verified_complete(
        output_root=tmp_path,
        task=task,
        configuration_sha256=config_hash,
        hashes=hashes,
    )
    assert complete is False and "incomplete" in reason

    monkeypatch.setattr(RUNNER, "_validate_result_npz", lambda *args, **kwargs: None)
    completion = RUNNER._completion_identity(
        task=task, configuration_sha256=config_hash, hashes=hashes
    )
    completion.update(
        {
            "result_filename": result_path.name,
            "result_bytes": result_path.stat().st_size,
            "result_sha256": RUNNER.sha256_file(result_path),
        }
    )
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    assert RUNNER.verified_complete(
        output_root=tmp_path,
        task=task,
        configuration_sha256=config_hash,
        hashes=hashes,
    ) == (True, "verified")
    result_path.write_bytes(result_path.read_bytes() + b"!")
    complete, reason = RUNNER.verified_complete(
        output_root=tmp_path,
        task=task,
        configuration_sha256=config_hash,
        hashes=hashes,
    )
    assert complete is False and "byte-count" in reason


def test_drive_publication_reads_back_before_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "local.bin"
    final = tmp_path / "drive" / "final.bin"
    local.write_bytes(b"new payload")
    final.parent.mkdir()
    final.write_bytes(b"old payload")
    original_hash = RUNNER.sha256_file

    def fail_temporary(path: Path) -> str:
        if path.name.startswith(".final.bin"):
            return "0" * 64
        return original_hash(path)

    monkeypatch.setattr(RUNNER, "sha256_file", fail_temporary)
    with pytest.raises(OSError, match="temporary checksum"):
        RUNNER.publish_file(local, final)
    assert final.read_bytes() == b"old payload"


def test_bundle_sources_registry_and_notebook_contract() -> None:
    assert _sha256(BUNDLE / "src/classA_U1FGTN_gpu.py") == _sha256(
        REPO / "src/fgtn/classA_U1FGTN_gpu.py"
    )
    assert _sha256(BUNDLE / "src/occupied_frame_gpu.py") == _sha256(
        REPO / "src/fgtn/occupied_frame_gpu.py"
    )
    layout = _load(PARENT / "bundle_layout.py", "tested_replay_bundle_layout")
    assert "08_domain_wall_correlator_scaling" in layout.NEW_DESIGN_BUNDLES
    assert "09_pure_tangent_replay_acquisition" in layout.NEW_DESIGN_BUNDLES
    validated = layout.validate_bundle_layout(PARENT)
    slot_08 = validated.index("08_domain_wall_correlator_scaling")
    assert validated[slot_08 : slot_08 + 2] == (
        "08_domain_wall_correlator_scaling",
        "09_pure_tangent_replay_acquisition",
    )

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    all_source = "\n".join(_cell_source(cell) for cell in notebook["cells"])
    assert "REPORT_ONLY = False" in all_source
    assert "MAX_NEW_TASKS = None" in all_source
    assert "shutil.copytree(BUNDLE_DRIVE_DIR, LOCAL_BUNDLE_DIR)" in all_source
    assert "sys.executable, '-u'" in all_source
    assert "subprocess.Popen(" in all_source
    assert "stdout=subprocess.PIPE" in all_source
    assert "stderr=subprocess.STDOUT" in all_source
    assert "os.read(process.stdout.fileno(), 4096)" in all_source
    assert "subprocess.run(command, check=True)" not in all_source
    assert "A100" in all_source and "complex128" in all_source
    assert "runtime.unassign()" in all_source
    assert "print('done')" in all_source
    final_code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"][-1]
    assert _cell_source(final_code).strip().splitlines() == [
        "from google.colab import runtime",
        "runtime.unassign()",
        "print('done')",
    ]
