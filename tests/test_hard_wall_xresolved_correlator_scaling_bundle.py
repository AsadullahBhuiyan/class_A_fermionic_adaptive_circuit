from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
PARENT = REPO / "00_WORKSPACE/CURRENT/final_production_new_designs"
BUNDLE = PARENT / "14_hard_wall_xresolved_correlator_scaling"
NOTEBOOK = BUNDLE / "run_hard_wall_xresolved_correlator.ipynb"
LEGACY_OBSERVER = (
    REPO
    / "00_WORKSPACE/COLAB/colab_charge_fluctuations/src/streaming_covariance_observables_gpu.py"
)
REVISION = (
    "hard_wall_xresolved_nx20_ny40-50-60_a1-1_nsh1_s100_2ny_raster_"
    "endpoint_frame_halfcov_occupations_v2_30gib_batched"
)


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
OBSERVER = _load(
    BUNDLE / "endpoint_correlator.py", "tested_large_ny_xresolved_observer"
)
RUNNER = _load(BUNDLE / "run_campaign.py", "tested_large_ny_xresolved_runner")
LEGACY = _load(LEGACY_OBSERVER, "tested_large_ny_legacy_correlator")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _cpu_model(*, nx: int, ny: int):
    return RUNNER.classA_U1FGTN_gpu(
        Nx=nx,
        Ny=ny,
        DW=True,
        nshell=1,
        filling_frac=0.5,
        alpha_1=1.0,
        alpha_2=30.0,
        trial_orbitals="X",
        dw_truncation=True,
        triv_region_local_mode=False,
        device="cpu",
        dtype="complex128",
        backend="local",
    )


def _cpu_config(nx: int) -> dict:
    config = RUNNER.expected_config()
    config["Nx"] = nx
    config["device"] = "cpu"
    return config


def _rng_equal(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> None:
    assert set(left) == set(right)
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)


def test_locked_contract_expands_7_batches_60_shards_and_300_slots() -> None:
    config = RUNNER.expected_config()
    assert RUNNER.validate_config(config) == config
    assert config["sampling_revision"] == REVISION
    assert config["root_seed"] == 2026091001
    assert config["Nx"] == 20
    assert config["Ny_values"] == [60, 50, 40]
    assert config["samples_per_Ny"] == 100
    assert config["execution_batch_size_by_Ny"] == {"40": 80, "50": 50, "60": 40}
    assert config["result_shard_size"] == 5
    assert config["cycles_rule"] == "2*Ny"
    assert config["segment_cycles"] == 5
    assert config["dtype"] == "complex128"
    assert config["gpu_memory_hard_limit_gib"] == 30.0
    assert config["protocol"] == {
        "DW": True,
        "domain_wall_interval": [5, 15],
        "dw_truncation": True,
        "meas_slab_only": True,
        "nshell": 1,
        "alpha_1": 1.0,
        "alpha_2": 30.0,
        "filling_frac": 0.5,
        "trial_orbitals": "X",
        "init_mode": "default",
        "sequence": "raster_y",
        "perfect_correction": True,
        "postselect": False,
        "postselect_probability": 0.0,
        "n_a": 0.5,
        "state_representation": "physical_frame",
        "triv_region_local_mode": False,
        "frame_reorthonormalize_interval": 1,
    }
    assert config["observables"]["occupied_frame"] is True
    assert config["observables"]["occupied_ranks"] is True
    assert config["observables"]["half_system_covariance"] is True
    assert config["observables"]["half_system_occupation_spectrum"] is True
    assert config["observables"]["half_system_region"] == "[0,Nx)x[0,Ny//2)"
    assert config["observables"]["half_system_Ay_rule"] == "Ny//2"
    assert config["observables"]["half_system_covariance_convention"] == (
        "C_A=F_A@F_A_dagger"
    )
    assert config["observables"]["frame_covariance_convention"] == (
        "C=F@F_dagger;G=2C-I"
    )

    batches = RUNNER.expand_execution_batches(config)
    shards = RUNNER.all_result_shards(batches)
    assert [(task.ny, task.sample_count) for task in batches] == [
        (60, 40),
        (60, 40),
        (60, 20),
        (50, 50),
        (50, 50),
        (40, 80),
        (40, 20),
    ]
    assert len(shards) == 60
    assert len({task.seed for task in batches}) == 7
    assert (
        len(
            {
                (shard.ny, index)
                for shard in shards
                for index in shard.global_sample_indices
            }
        )
        == 300
    )
    assert all(shard.sample_count == 5 for shard in shards)


def test_frame_native_correlator_matches_legacy_dense_estimator() -> None:
    torch.manual_seed(1401)
    samples, nx, ny, capacity = 3, 4, 6, 24
    dimension = 2 * nx * ny
    frame = torch.linalg.qr(
        torch.randn(samples, dimension, capacity, dtype=torch.complex128)
    ).Q
    ranks = torch.tensor([24, 19, 13], dtype=torch.long)
    for row, rank in enumerate(ranks.tolist()):
        frame[row, :, rank:] = 0

    x_resolved = OBSERVER.x_resolved_square_correlator_from_frame(
        frame, ranks, nx=nx, ny=ny
    )
    projector = frame @ frame.mH
    covariance = 2.0 * projector - torch.eye(
        dimension, dtype=torch.complex128
    ).unsqueeze(0)
    pairs = LEGACY.build_square_correlator_pair_indices(nx=nx, ny=ny, device="cpu")
    legacy = LEGACY.xavg_square_correlator_batch_torch(covariance, pairs, nx=nx, ny=ny)
    torch.testing.assert_close(x_resolved.mean(dim=1), legacy, rtol=0, atol=2e-16)

    half_covariance, half_occupations, diagnostics = (
        OBSERVER.half_system_covariance_and_occupations_from_frame(
            frame, ranks, nx=nx, ny=ny
        )
    )
    region_rows = frame.reshape(samples, ny, nx, 2, capacity)[:, : ny // 2]
    region_rows = region_rows.reshape(samples, 2 * nx * (ny // 2), capacity)
    expected_half_covariance = region_rows @ region_rows.mH
    torch.testing.assert_close(half_covariance, expected_half_covariance, rtol=0, atol=0)
    torch.testing.assert_close(
        half_occupations,
        torch.linalg.eigvalsh(expected_half_covariance).clamp(0.0, 1.0),
        rtol=0,
        atol=0,
    )
    assert diagnostics["maximum_hermiticity_residual"] <= 1.0e-10


def test_endpoint_payload_has_length_one_time_axis_and_does_not_change_rng() -> None:
    torch.manual_seed(91)
    samples, nx, ny, rank = 5, 4, 6, 24
    frame = torch.linalg.qr(
        torch.randn(samples, 2 * nx * ny, rank, dtype=torch.complex128)
    ).Q
    ranks = torch.full((samples,), rank, dtype=torch.long)
    frame_before = frame.clone()
    ranks_before = ranks.clone()
    rng_before = torch.get_rng_state().clone()
    payload = OBSERVER.endpoint_result_payload(
        frame=frame,
        ranks=ranks,
        nx=nx,
        ny=ny,
        global_sample_indices=np.arange(samples),
    )
    assert torch.equal(frame, frame_before)
    assert torch.equal(ranks, ranks_before)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert payload["cycles"].tolist() == [12]
    assert payload["normalized_cycles"].tolist() == [2.0]
    assert payload["x_resolved_square_correlator"].shape == (5, 1, 4, 4)
    assert payload["xavg_square_correlator_vs_ry"].shape == (5, 1, 4)
    np.testing.assert_array_equal(payload["global_charge"], rank)
    np.testing.assert_array_equal(payload["occupied_frame"], frame.numpy())
    np.testing.assert_array_equal(payload["occupied_ranks"], ranks.numpy())
    assert payload["half_system_Ay"].item() == 3
    np.testing.assert_array_equal(
        payload["half_system_region_bounds"], np.asarray([0, 4, 0, 3])
    )
    assert payload["half_system_covariance"].shape == (5, 1, 24, 24)
    assert payload["half_system_covariance"].dtype == np.complex128
    assert payload["half_system_occupation_spectrum"].shape == (5, 1, 24)
    assert payload["half_system_occupation_spectrum"].dtype == np.float64
    region_rows = frame.reshape(samples, ny, nx, 2, rank)[:, : ny // 2]
    region_rows = region_rows.reshape(samples, 2 * nx * (ny // 2), rank)
    expected_half_covariance = region_rows @ region_rows.mH
    expected_half_occupations = torch.linalg.eigvalsh(expected_half_covariance)
    torch.testing.assert_close(
        torch.from_numpy(payload["half_system_covariance"][:, 0]),
        expected_half_covariance,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch.from_numpy(payload["half_system_occupation_spectrum"][:, 0]),
        expected_half_occupations.clamp(0.0, 1.0),
        rtol=0,
        atol=0,
    )
    assert payload["half_system_covariance_convention"].item() == (
        "C_A=F_A@F_A_dagger;A=[0,Nx)x[0,Ny//2)"
    )
    assert payload["half_system_occupation_ordering"].item() == "ascending"
    assert payload["occupied_frame"].dtype == np.complex128
    assert payload["frame_covariance_convention"].item() == ("C=F@F_dagger;G=2C-I")
    reconstructed = torch.from_numpy(payload["occupied_frame"])
    torch.testing.assert_close(reconstructed @ reconstructed.mH, frame @ frame.mH)
    OBSERVER.validate_endpoint_payload(
        payload,
        nx=nx,
        ny=ny,
        global_sample_indices=np.arange(samples),
    )


def test_cpu_checkpoint_resume_is_bitwise_equivalent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(RUNNER, "EXPECTED_NX", 4)
    task = RUNNER.ExecutionBatch(
        ny=4,
        batch_index=0,
        sample_start=0,
        sample_stop=2,
        seed=445566,
    )
    config = _cpu_config(4)
    hashes = {"unit": "checkpoint-resume"}
    configuration_sha256 = "cpu-config"

    reference = RUNNER.run_dynamics(
        model=_cpu_model(nx=4, ny=4),
        config=config,
        output_root=tmp_path / "reference",
        scratch_root=tmp_path / "reference_scratch",
        task=task,
        configuration_sha256=configuration_sha256,
        hashes=hashes,
        show_progress=False,
    )

    class InjectedCrash(RuntimeError):
        pass

    def crash_after_cycle_five(cycle: int) -> None:
        if cycle == 5:
            raise InjectedCrash("simulated disconnect")

    interrupted_root = tmp_path / "interrupted"
    with pytest.raises(InjectedCrash):
        RUNNER.run_dynamics(
            model=_cpu_model(nx=4, ny=4),
            config=config,
            output_root=interrupted_root,
            scratch_root=tmp_path / "interrupted_scratch",
            task=task,
            configuration_sha256=configuration_sha256,
            hashes=hashes,
            show_progress=False,
            after_checkpoint=crash_after_cycle_five,
        )
    partial, reason = RUNNER.load_checkpoint(
        output_root=interrupted_root,
        task=task,
        configuration_sha256=configuration_sha256,
        hashes=hashes,
    )
    assert reason == "verified"
    assert partial is not None and partial.completed_cycle == 5

    np.random.seed(7)
    torch.manual_seed(8)
    resumed = RUNNER.run_dynamics(
        model=_cpu_model(nx=4, ny=4),
        config=config,
        output_root=interrupted_root,
        scratch_root=tmp_path / "resumed_scratch",
        task=task,
        configuration_sha256=configuration_sha256,
        hashes=hashes,
        show_progress=False,
    )
    np.testing.assert_array_equal(resumed.frame, reference.frame)
    np.testing.assert_array_equal(resumed.ranks, reference.ranks)
    _rng_equal(resumed.rng_payload, reference.rng_payload)


def test_final_checkpoint_finishes_endpoint_without_engine_and_is_then_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(RUNNER, "EXPECTED_NX", 4)
    task = RUNNER.ExecutionBatch(
        ny=4,
        batch_index=0,
        sample_start=0,
        sample_stop=5,
        seed=778899,
    )
    config = _cpu_config(4)
    hashes = {"unit": "final-frame-recovery"}
    configuration_sha256 = "cpu-final-config"
    output_root = tmp_path / "outputs"
    scratch_root = tmp_path / "scratch"
    RUNNER.run_dynamics(
        model=_cpu_model(nx=4, ny=4),
        config=config,
        output_root=output_root,
        scratch_root=scratch_root,
        task=task,
        configuration_sha256=configuration_sha256,
        hashes=hashes,
        show_progress=False,
    )

    class ForbiddenModel:
        device = torch.device("cpu")

        def run_markov_circuit(self, **_: object) -> None:
            raise AssertionError("final-frame recovery must not rerun dynamics")

    elapsed = RUNNER.execute_batch(
        model=ForbiddenModel(),
        config=config,
        output_root=output_root,
        scratch_root=scratch_root,
        task=task,
        configuration_sha256=configuration_sha256,
        hashes=hashes,
        gpu={"name": "CPU test"},
    )
    assert elapsed >= 0
    shard = RUNNER.result_shards(task)[0]
    valid, reason = RUNNER.verified_complete(
        output_root=output_root,
        shard=shard,
        configuration_sha256=configuration_sha256,
        hashes=hashes,
    )
    assert valid, reason
    assert not any(path.exists() for path in RUNNER.checkpoint_paths(output_root, task))
    result_path, _ = RUNNER.result_paths(output_root, shard)
    with zipfile.ZipFile(result_path) as archive:
        assert archive.infolist()
        assert all(
            member.compress_type == zipfile.ZIP_STORED for member in archive.infolist()
        )
    with result_path.open("ab") as handle:
        handle.write(b"corrupt")
    valid, reason = RUNNER.verified_complete(
        output_root=output_root,
        shard=shard,
        configuration_sha256=configuration_sha256,
        hashes=hashes,
    )
    assert not valid and "byte-count mismatch" in reason


def test_endpoint_failure_preserves_final_checkpoint_and_completed_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(RUNNER, "EXPECTED_NX", 4)
    task = RUNNER.ExecutionBatch(4, 0, 0, 10, 31337)
    dimension, rank = 32, 16
    frame = torch.linalg.qr(torch.randn(10, dimension, rank, dtype=torch.complex128)).Q
    native = {"frame": frame, "ranks": torch.full((10,), rank, dtype=torch.long)}
    output_root = tmp_path / "outputs"
    scratch_root = tmp_path / "scratch"
    configuration_sha256 = "endpoint-failure-config"
    hashes = {"unit": "endpoint-failure"}
    RUNNER.save_checkpoint(
        output_root=output_root,
        scratch_root=scratch_root,
        task=task,
        completed_cycle=task.cycles,
        elapsed_seconds=2.0,
        native=native,
        configuration_sha256=configuration_sha256,
        hashes=hashes,
    )

    original_publish = RUNNER.publish_result_shard
    calls = 0

    def fail_second(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated endpoint publication failure")
        return original_publish(**kwargs)

    monkeypatch.setattr(RUNNER, "publish_result_shard", fail_second)
    with pytest.raises(OSError, match="simulated endpoint"):
        RUNNER.execute_batch(
            model=SimpleNamespace(device=torch.device("cpu")),
            config=_cpu_config(4),
            output_root=output_root,
            scratch_root=scratch_root,
            task=task,
            configuration_sha256=configuration_sha256,
            hashes=hashes,
            gpu={"name": "CPU test"},
        )
    assert all(path.exists() for path in RUNNER.checkpoint_paths(output_root, task))
    first, second = RUNNER.result_shards(task)
    assert RUNNER.verified_complete(
        output_root=output_root,
        shard=first,
        configuration_sha256=configuration_sha256,
        hashes=hashes,
    )[0]
    assert not RUNNER.verified_complete(
        output_root=output_root,
        shard=second,
        configuration_sha256=configuration_sha256,
        hashes=hashes,
    )[0]

    monkeypatch.setattr(RUNNER, "publish_result_shard", original_publish)
    RUNNER.execute_batch(
        model=SimpleNamespace(device=torch.device("cpu")),
        config=_cpu_config(4),
        output_root=output_root,
        scratch_root=scratch_root,
        task=task,
        configuration_sha256=configuration_sha256,
        hashes=hashes,
        gpu={"name": "CPU test"},
    )
    assert all(
        RUNNER.verified_complete(
            output_root=output_root,
            shard=shard,
            configuration_sha256=configuration_sha256,
            hashes=hashes,
        )[0]
        for shard in (first, second)
    )
    assert not any(path.exists() for path in RUNNER.checkpoint_paths(output_root, task))


def test_partial_pair_and_checksum_corruption_are_not_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(RUNNER, "EXPECTED_NX", 4)
    task = RUNNER.ExecutionBatch(4, 0, 0, 5, 1122)
    shard = RUNNER.result_shards(task)[0]
    result_path, completion_path = RUNNER.result_paths(tmp_path, shard)
    result_path.parent.mkdir(parents=True)
    result_path.write_bytes(b"partial")
    valid, reason = RUNNER.verified_complete(
        output_root=tmp_path,
        shard=shard,
        configuration_sha256="config",
        hashes={"unit": "hash"},
    )
    assert not valid and reason == "incomplete result/completion pair"
    completion_path.write_text("{}\n", encoding="utf-8")
    valid, reason = RUNNER.verified_complete(
        output_root=tmp_path,
        shard=shard,
        configuration_sha256="config",
        hashes={"unit": "hash"},
    )
    assert not valid and "identity mismatch" in reason


def test_publish_file_fails_before_final_on_bad_temporary_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "local.bin"
    final = tmp_path / "drive" / "final.bin"
    local.write_bytes(b"correct bytes")

    def corrupt_copy(source: Path, destination: Path) -> None:
        del source
        Path(destination).write_bytes(b"wrong")

    monkeypatch.setattr(shutil, "copyfile", corrupt_copy)
    with pytest.raises(OSError, match="byte-count mismatch"):
        RUNNER.publish_file(local, final)
    assert not final.exists()


def test_checkpoint_identity_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(RUNNER, "EXPECTED_NX", 2)
    task = RUNNER.ExecutionBatch(2, 0, 0, 1, 44)
    dimension, rank = 8, 4
    frame = torch.linalg.qr(torch.randn(1, dimension, rank, dtype=torch.complex128)).Q
    native = {"frame": frame, "ranks": torch.tensor([rank])}
    RUNNER.save_checkpoint(
        output_root=tmp_path,
        scratch_root=tmp_path / "scratch",
        task=task,
        completed_cycle=task.cycles,
        elapsed_seconds=1.0,
        native=native,
        configuration_sha256="config-a",
        hashes={"unit": "a"},
    )
    loaded, reason = RUNNER.load_checkpoint(
        output_root=tmp_path,
        task=task,
        configuration_sha256="config-b",
        hashes={"unit": "a"},
    )
    assert loaded is None
    assert reason == "checkpoint identity mismatch: configuration_sha256"


def test_notebook_stages_locally_surfaces_progress_and_disconnects() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = [_cell_source(cell) for cell in notebook["cells"]]
    joined = "\n".join(sources)
    assert "drive.mount('/content/drive')" in joined
    assert "BUNDLE = '14_hard_wall_xresolved_correlator_scaling'" in joined
    assert "'Ny_values': [60, 50, 40]" in joined
    assert "'execution_batch_size_by_Ny': {'40': 80, '50': 50, '60': 40}" in joined
    assert "'gpu_memory_hard_limit_gib': 30.0" in joined
    assert "'occupied_frame': True" in joined
    assert "'occupied_ranks': True" in joined
    assert "'half_system_covariance': True" in joined
    assert "'half_system_occupation_spectrum': True" in joined
    assert "'half_system_region': '[0,Nx)x[0,Ny//2)'" in joined
    assert "'execution_batches': 7" in joined
    assert "'estimated_final_occupied_frames_gib': 9.18" in joined
    assert "'estimated_final_half_system_covariances_gib': 4.59" in joined
    assert "'estimated_final_large_payloads_gib': 13.77" in joined
    assert "'required_fresh_drive_free_gib': 20.84" in joined
    assert (
        "'estimated_dynamics_a100_hours_by_Ny': {'40': 1.5, '50': 5.7, '60': 11.5}"
        in joined
    )
    assert "'estimated_total_a100_hours': '21-24'" in joined
    assert "'sequence': 'raster_y'" in joined
    assert "'perfect_correction': True" in joined
    assert "'dtype': 'complex128'" in joined
    assert "A100" in joined and "38 * 1024**3" in joined
    assert "shutil.copytree(BUNDLE_DRIVE_DIR, LOCAL_BUNDLE_DIR)" in joined
    assert "importlib.util.spec_from_file_location" in joined
    assert "runner.main(runner_args)" in joined
    assert "subprocess.run" not in joined
    assert "stdout.buffer" not in joined
    assert "stdout=subprocess.PIPE" not in joined
    assert "--report-only" in joined
    assert "--max-new-execution-batches" in joined
    assert sources[-1] == (
        "from google.colab import runtime\n" "runtime.unassign()\n" "print('done')\n"
    )


def test_bundle_sources_are_current_canonical_copies_and_registered() -> None:
    assert _sha256(BUNDLE / "src/classA_U1FGTN_gpu.py") == _sha256(
        REPO / "src/fgtn/classA_U1FGTN_gpu.py"
    )
    assert _sha256(BUNDLE / "src/occupied_frame_gpu.py") == _sha256(
        REPO / "src/fgtn/occupied_frame_gpu.py"
    )
    index = json.loads((PARENT / "bundle_index.json").read_text(encoding="utf-8"))
    assert BUNDLE.name in index["bundles"]
    assert index["standalone_contracts"][BUNDLE.name] == REVISION
    assert "occupied frames/ranks" in index["descriptions"][BUNDLE.name]


def test_large_payload_storage_budget_tracks_missing_uncompressed_results() -> None:
    shards = RUNNER.all_result_shards(
        RUNNER.expand_execution_batches(RUNNER.expected_config())
    )
    frames = sum(RUNNER.estimated_frame_result_bytes(shard) for shard in shards)
    covariances = sum(
        RUNNER.estimated_half_covariance_result_bytes(shard) for shard in shards
    )
    total = sum(RUNNER.estimated_result_payload_bytes(shard) for shard in shards)
    assert 9.1 < frames / 1024**3 < 9.3
    assert 4.5 < covariances / 1024**3 < 4.7
    assert total == frames + covariances
    assert 13.7 < total / 1024**3 < 13.9
    required = RUNNER.required_drive_free_bytes(shards, set())
    assert 20.7 < required / 1024**3 < 21.0
    first = shards[0]
    after_one = RUNNER.required_drive_free_bytes(shards, {first.task_id})
    assert after_one == (
        RUNNER.MINIMUM_DRIVE_HEADROOM_BYTES
        + int(np.ceil(1.15 * (total - RUNNER.estimated_result_payload_bytes(first))))
    )
