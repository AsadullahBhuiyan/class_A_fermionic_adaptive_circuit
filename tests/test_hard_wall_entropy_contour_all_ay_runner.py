from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
BUNDLE = (
    REPO
    / "00_WORKSPACE/CURRENT/final_production_new_designs/16_hard_wall_entropy_contour_all_ay"
)
RUNNER_PATH = BUNDLE / "run_campaign.py"


def _load_runner():
    if str(BUNDLE) not in sys.path:
        sys.path.insert(0, str(BUNDLE))
    spec = importlib.util.spec_from_file_location(
        "tested_hard_wall_entropy_contour_all_ay_runner", RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def test_locked_config_expands_disjoint_execution_batches_and_five_sample_shards() -> None:
    config = RUNNER.expected_config()
    assert RUNNER.validate_config(config) == config
    lane_a = RUNNER.expand_execution_batches(config, lane="A")
    lane_b = RUNNER.expand_execution_batches(config, lane="B")
    assert len(lane_a) == 9
    assert len(lane_b) == 16
    assert {task.ny for task in lane_a} == {50, 60}
    assert {task.ny for task in lane_b} == {30, 35, 40, 45, 55}
    assert sum(task.sample_count for task in lane_a) == 200
    assert sum(task.sample_count for task in lane_b) == 500
    all_tasks = lane_a + lane_b
    all_shards = [shard for task in all_tasks for shard in RUNNER.result_shards(task)]
    assert len(all_shards) == 140
    assert all(shard.sample_count == 5 for shard in all_shards)
    assert RUNNER.execution_seed(
        config["root_seed"],
        lane="A",
        ny=40,
        batch_index=0,
        sample_start=0,
        sample_stop=40,
    ) != RUNNER.execution_seed(
        config["root_seed"],
        lane="B",
        ny=40,
        batch_index=0,
        sample_start=0,
        sample_stop=40,
    )
    for ny in RUNNER.EXPECTED_NY_VALUES:
        indices = sorted(
            index
            for shard in all_shards
            if shard.ny == ny
            for index in shard.global_sample_indices
        )
        assert indices == list(range(100))
    example_result, _ = RUNNER.result_paths(Path("/drive"), all_shards[0])
    example_checkpoint, _ = RUNNER.checkpoint_paths(Path("/drive"), lane_a[0])
    assert f"lane_{all_shards[0].lane}" in example_result.parts
    assert f"lane_{lane_a[0].lane}" in example_checkpoint.parts


def test_new_execution_seed_ensemble_is_disjoint_from_hard_wall_v2() -> None:
    new_tasks = [
        *RUNNER.expand_execution_batches(RUNNER.expected_config(), lane="A"),
        *RUNNER.expand_execution_batches(RUNNER.expected_config(), lane="B"),
    ]
    old_lanes = {"A": (40, 60), "B": (30, 35, 45, 50, 55)}
    batch_sizes = RUNNER.EXPECTED_EXECUTION_BATCH_SIZES
    old_seeds: set[int] = set()
    for lane, sizes in old_lanes.items():
        for ny in sizes:
            for batch_index, start in enumerate(range(0, 100, batch_sizes[ny])):
                stop = min(start + batch_sizes[ny], 100)
                label = (
                    f"2026090305|lane={lane}|Nx=20|Ny={ny}|execution={batch_index}|"
                    f"samples={start}:{stop}"
                )
                old_seeds.add(
                    int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
                    & ((1 << 63) - 1)
                )
    assert RUNNER.EXPECTED_ROOT_SEED != 2026090305
    assert not ({task.seed for task in new_tasks} & old_seeds)


def test_notebook_configs_are_exactly_the_runner_config() -> None:
    for filename in (
        "run_lane_A_Ny50_Ny60.ipynb",
        "run_lane_B_Ny30_35_40_45_55.ipynb",
    ):
        notebook = json.loads((BUNDLE / filename).read_text(encoding="utf-8"))
        source = next(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if "CONFIG = {" in "".join(cell.get("source", []))
        )
        namespace: dict = {}
        exec(compile(source, filename, "exec"), namespace)
        assert namespace["CONFIG"] == RUNNER.expected_config()


def test_rng_state_round_trip_restores_numpy_and_torch_streams() -> None:
    np.random.seed(1847)
    torch.manual_seed(913)
    state = RUNNER._capture_rng_state()
    expected_numpy = np.random.random(8)
    expected_torch = torch.rand(8)
    np.random.seed(11)
    torch.manual_seed(12)
    RUNNER._restore_rng_state(state)
    assert np.array_equal(np.random.random(8), expected_numpy)
    assert torch.equal(torch.rand(8), expected_torch)


class _CheckpointObserver:
    def __init__(self, sample_ids: np.ndarray) -> None:
        self.sample_ids = sample_ids
        self.values = np.arange(len(sample_ids), dtype=np.float64)

    def validate(self, *, require_dynamics: bool, require_endpoint: bool) -> None:
        assert isinstance(require_dynamics, bool)
        assert require_endpoint is False

    def checkpoint_payload(self) -> dict[str, np.ndarray]:
        return {
            "sample_ids": self.sample_ids.copy(),
            "values": self.values.copy(),
        }


def test_checkpoint_round_trip_and_checksum_rejection(tmp_path: Path) -> None:
    task = next(
        task
        for task in RUNNER.expand_execution_batches(RUNNER.expected_config(), lane="B")
        if task.ny == 30 and task.batch_index == 0
    )
    frame = np.zeros(
        (task.sample_count, 2 * RUNNER.EXPECTED_NX * task.ny, 1),
        dtype=np.complex128,
    )
    ranks = np.zeros(task.sample_count, dtype=np.int64)
    observer = _CheckpointObserver(
        np.asarray(task.global_sample_indices, dtype=np.int64)
    )
    hashes = {"runner": "abc"}
    RUNNER.save_checkpoint(
        output_root=tmp_path / "drive",
        scratch_root=tmp_path / "scratch",
        task=task,
        completed_cycle=5,
        elapsed_seconds=4.25,
        native_state={"frame": frame, "ranks": ranks},
        observer=observer,
        config_sha256="config-hash",
        hashes=hashes,
    )
    restored, reason = RUNNER.load_checkpoint(
        output_root=tmp_path / "drive",
        task=task,
        config_sha256="config-hash",
        hashes=hashes,
    )
    assert reason == "verified"
    assert restored is not None
    assert restored.completed_cycle == 5
    assert restored.elapsed_seconds == pytest.approx(4.25)
    assert np.array_equal(restored.frame, frame)
    assert np.array_equal(restored.ranks, ranks)
    assert np.array_equal(restored.observer_payload["values"], observer.values)
    assert "torch_cpu" in restored.rng_payload

    checkpoint_npz, _ = RUNNER.checkpoint_paths(tmp_path / "drive", task)
    with checkpoint_npz.open("ab") as handle:
        handle.write(b"tamper")
    rejected, reason = RUNNER.load_checkpoint(
        output_root=tmp_path / "drive",
        task=task,
        config_sha256="config-hash",
        hashes=hashes,
    )
    assert rejected is None
    assert reason == "checkpoint byte-count mismatch"


class _NativeObserver:
    def __init__(self) -> None:
        self.cycles: list[int] = []

    def __call__(self, *, cycle: int, **_: object) -> None:
        self.cycles.append(int(cycle))


class _FakeModel:
    def __init__(self, task) -> None:
        self.device = "cuda:0"
        self.task = task
        self.calls: list[dict] = []

    def run_markov_circuit(self, **kwargs):
        self.calls.append(kwargs)
        observer = kwargs["native_cycle_observer"]
        state = object()
        for cycle in range(int(kwargs["cycles"]) + 1):
            observer(
                cycle=cycle,
                state=state,
                batch_index=0,
                batch_start=0,
                batch_count=self.task.sample_count,
            )
        frame = np.zeros(
            (
                self.task.sample_count,
                2 * RUNNER.EXPECTED_NX * self.task.ny,
                1,
            ),
            dtype=np.complex128,
        )
        return {
            "samples": self.task.sample_count,
            "state_representation_resolved": "physical_frame",
            "covariance_materialization_count": 0,
            "choi_tracked": False,
            "frame_init_prepared": bool(kwargs["frame_init_prepared"]),
            "exterior_preparation_performed": not bool(
                kwargs["frame_init_prepared"]
            ),
            "exterior_preparation": (
                "skipped_prepared_frame"
                if kwargs["frame_init_prepared"]
                else "born_conditioned_onsite_before_cycle_0"
            ),
            "exterior_preparation_mode": (
                "already_prepared"
                if kwargs["frame_init_prepared"]
                else "born_conditioned"
            ),
            "native_final": {
                "frame": frame,
                "ranks": np.zeros(self.task.sample_count, dtype=np.int64),
            },
        }


def test_segments_translate_cycles_and_skip_repreparation(monkeypatch) -> None:
    task = RUNNER.expand_execution_batches(RUNNER.expected_config(), lane="A")[-1]
    model = _FakeModel(task)
    observer = _NativeObserver()
    completed_cycles: list[int] = []
    monkeypatch.setattr(RUNNER.torch.cuda, "synchronize", lambda *_: None)

    native, _ = RUNNER._run_segment(
        model=model,
        config=RUNNER.expected_config(),
        task=task,
        observer=observer,
        segment_start=0,
        frame=None,
        ranks=None,
        on_cycle_complete=completed_cycles.append,
    )
    assert observer.cycles == [0, 1, 2, 3, 4, 5]
    assert model.calls[-1]["frame_init_prepared"] is False

    RUNNER._run_segment(
        model=model,
        config=RUNNER.expected_config(),
        task=task,
        observer=observer,
        segment_start=5,
        frame=native["frame"],
        ranks=native["ranks"],
        on_cycle_complete=completed_cycles.append,
    )
    assert observer.cycles == list(range(11))
    assert model.calls[-1]["frame_init_prepared"] is True
    assert model.calls[-1]["cycles"] == 5
    assert model.calls[-1]["require_no_covariance_materialization"] is True
    assert completed_cycles == list(range(1, 11))


class _ResumeState:
    def __init__(self, frame: torch.Tensor, ranks: torch.Tensor) -> None:
        self.frame = frame
        self.ranks = ranks


class _ResumeObserver:
    def __init__(self, task) -> None:
        self.sample_ids = np.asarray(task.global_sample_indices, dtype=np.int64)
        self.cycles: list[int] = []
        self.frame_sums: list[float] = []

    def __call__(self, *, cycle: int, state: _ResumeState, **_: object) -> None:
        self.cycles.append(int(cycle))
        self.frame_sums.append(float(state.frame.real.sum().item()))

    def validate(self, *, require_dynamics: bool, require_endpoint: bool) -> None:
        if require_dynamics:
            assert self.cycles[-1] == max(self.cycles)

    def checkpoint_payload(self) -> dict[str, np.ndarray]:
        return {
            "sample_ids": self.sample_ids.copy(),
            "cycles": np.asarray(self.cycles, dtype=np.int64),
            "frame_sums": np.asarray(self.frame_sums, dtype=np.float64),
        }

    def restore_checkpoint(self, payload) -> None:
        assert np.array_equal(payload["sample_ids"], self.sample_ids)
        self.cycles = np.asarray(payload["cycles"], dtype=np.int64).tolist()
        self.frame_sums = np.asarray(payload["frame_sums"], dtype=np.float64).tolist()

    def result_payload(self, sample_slice=None) -> dict[str, np.ndarray]:
        selected = self.sample_ids[sample_slice]
        return {
            "sample_ids": selected.copy(),
            "cycles": np.asarray(self.cycles, dtype=np.int64),
            "resume_test_value": selected.astype(np.float64),
        }


class _RngDrivenCpuModel:
    """Small CPU state evolution with both NumPy and Torch RNG streams."""

    def __init__(self, task) -> None:
        self.task = task
        self.device = "cpu"
        self.calls = 0

    def run_markov_circuit(self, **kwargs):
        self.calls += 1
        sample_count = self.task.sample_count
        dimension = 2 * RUNNER.EXPECTED_NX * self.task.ny
        if kwargs["frame_init"] is None:
            real = torch.randn((sample_count, dimension, 2), dtype=torch.float64)
            imag = torch.randn((sample_count, dimension, 2), dtype=torch.float64)
            frame = torch.complex(real, imag)
            ranks = torch.ones(sample_count, dtype=torch.int64)
        else:
            frame = torch.as_tensor(kwargs["frame_init"], dtype=torch.complex128).clone()
            ranks = torch.as_tensor(kwargs["frame_ranks"], dtype=torch.int64).clone()
        state = _ResumeState(frame, ranks)
        observer = kwargs["native_cycle_observer"]
        observer(
            cycle=0,
            state=state,
            batch_index=0,
            batch_start=0,
            batch_count=sample_count,
        )
        for cycle in range(1, int(kwargs["cycles"]) + 1):
            torch_noise = torch.rand(frame.shape, dtype=torch.float64)
            numpy_noise = torch.from_numpy(np.random.random(frame.shape))
            frame = frame + torch.complex(torch_noise, numpy_noise) * 1.0e-4
            state = _ResumeState(frame, ranks)
            observer(
                cycle=cycle,
                state=state,
                batch_index=0,
                batch_start=0,
                batch_count=sample_count,
            )
        return {
            "samples": sample_count,
            "state_representation_resolved": "physical_frame",
            "covariance_materialization_count": 0,
            "choi_tracked": False,
            "frame_init_prepared": bool(kwargs["frame_init_prepared"]),
            "exterior_preparation_performed": not bool(
                kwargs["frame_init_prepared"]
            ),
            "exterior_preparation": (
                "skipped_prepared_frame"
                if kwargs["frame_init_prepared"]
                else "born_conditioned_onsite_before_cycle_0"
            ),
            "exterior_preparation_mode": (
                "already_prepared"
                if kwargs["frame_init_prepared"]
                else "born_conditioned"
            ),
            "native_final": {
                "frame": frame.numpy(),
                "ranks": ranks.numpy(),
            },
        }


def _small_resume_task():
    return RUNNER.ExecutionBatch(
        lane="B",
        ny=5,
        batch_index=0,
        sample_start=0,
        sample_stop=5,
        task_id="lane-B_Ny005_execution-000_samples-000-004",
        seed=91731,
    )


def _assert_rng_payload_equal(left, right) -> None:
    assert set(left) == set(right)
    for key in left:
        assert np.array_equal(left[key], right[key]), key


def test_cpu_interrupted_resume_exactly_matches_continuous_segments(
    tmp_path: Path, monkeypatch
) -> None:
    task = _small_resume_task()
    config = RUNNER.expected_config()
    monkeypatch.setattr(RUNNER.torch.cuda, "synchronize", lambda *_: None)

    RUNNER._seed_rng(task.seed)
    continuous_model = _RngDrivenCpuModel(task)
    continuous_observer = _ResumeObserver(task)
    first_native, first_elapsed = RUNNER._run_segment(
        model=continuous_model,
        config=config,
        task=task,
        observer=continuous_observer,
        segment_start=0,
        frame=None,
        ranks=None,
    )
    continuous_rng = RUNNER._capture_rng_state()
    RUNNER._restore_rng_state(continuous_rng)
    continuous_native, _ = RUNNER._run_segment(
        model=continuous_model,
        config=config,
        task=task,
        observer=continuous_observer,
        segment_start=5,
        frame=first_native["frame"],
        ranks=first_native["ranks"],
    )
    continuous_final_rng = RUNNER._capture_rng_state()

    RUNNER._seed_rng(task.seed)
    interrupted_model = _RngDrivenCpuModel(task)
    interrupted_observer = _ResumeObserver(task)
    interrupted_first, _ = RUNNER._run_segment(
        model=interrupted_model,
        config=config,
        task=task,
        observer=interrupted_observer,
        segment_start=0,
        frame=None,
        ranks=None,
    )
    RUNNER.save_checkpoint(
        output_root=tmp_path / "drive",
        scratch_root=tmp_path / "scratch",
        task=task,
        completed_cycle=5,
        elapsed_seconds=first_elapsed,
        native_state=interrupted_first,
        observer=interrupted_observer,
        config_sha256="config",
        hashes={"source": "hash"},
    )
    np.random.seed(1)
    torch.manual_seed(2)
    restored, reason = RUNNER.load_checkpoint(
        output_root=tmp_path / "drive",
        task=task,
        config_sha256="config",
        hashes={"source": "hash"},
    )
    assert reason == "verified" and restored is not None
    resumed_observer = _ResumeObserver(task)
    resumed_observer.restore_checkpoint(restored.observer_payload)
    RUNNER._restore_rng_state(restored.rng_payload)
    resumed_native, _ = RUNNER._run_segment(
        model=interrupted_model,
        config=config,
        task=task,
        observer=resumed_observer,
        segment_start=restored.completed_cycle,
        frame=restored.frame,
        ranks=restored.ranks,
    )
    resumed_final_rng = RUNNER._capture_rng_state()

    assert np.array_equal(resumed_native["frame"], continuous_native["frame"])
    assert np.array_equal(resumed_native["ranks"], continuous_native["ranks"])
    assert resumed_observer.cycles == continuous_observer.cycles == list(range(11))
    assert resumed_observer.frame_sums == continuous_observer.frame_sums
    _assert_rng_payload_equal(resumed_final_rng, continuous_final_rng)


def test_partial_checkpoint_pair_is_never_resumed(tmp_path: Path) -> None:
    task = _small_resume_task()
    checkpoint_npz, checkpoint_json = RUNNER.checkpoint_paths(tmp_path, task)
    checkpoint_npz.parent.mkdir(parents=True)
    checkpoint_npz.write_bytes(b"orphan")
    restored, reason = RUNNER.load_checkpoint(
        output_root=tmp_path,
        task=task,
        config_sha256="config",
        hashes={"source": "hash"},
    )
    assert restored is None
    assert reason == "incomplete checkpoint pair"
    checkpoint_npz.unlink()
    checkpoint_json.write_text("{}", encoding="utf-8")
    restored, reason = RUNNER.load_checkpoint(
        output_root=tmp_path,
        task=task,
        config_sha256="config",
        hashes={"source": "hash"},
    )
    assert restored is None
    assert reason == "incomplete checkpoint pair"


def test_final_checkpoint_and_endpoint_progress_are_independent_and_cleanup_last(
    tmp_path: Path,
) -> None:
    task = _small_resume_task()
    output_root = tmp_path / "drive"
    scratch_root = tmp_path / "scratch"
    hashes = {"source": "hash"}
    observer = RUNNER.HardWallAllAyContourObserver(
        nx=RUNNER.EXPECTED_NX,
        ny=task.ny,
        physical_cycles=task.cycles,
        sample_ids=task.global_sample_indices,
    )
    state = _ResumeState(
        torch.zeros((task.sample_count, 2 * RUNNER.EXPECTED_NX * task.ny, 1), dtype=torch.complex128),
        torch.zeros(task.sample_count, dtype=torch.int64),
    )
    for cycle in range(task.cycles + 1):
        observer(cycle=cycle, state=state, batch_start=0, batch_count=task.sample_count)
    RUNNER.save_checkpoint(
        output_root=output_root,
        scratch_root=scratch_root,
        task=task,
        completed_cycle=task.cycles,
        elapsed_seconds=2.0,
        native_state={"frame": state.frame.numpy(), "ranks": state.ranks.numpy()},
        observer=observer,
        config_sha256="config",
        hashes=hashes,
    )
    checkpoint_npz, checkpoint_json = RUNNER.checkpoint_paths(output_root, task)
    assert checkpoint_npz.is_file() and checkpoint_json.is_file()
    digest = RUNNER.final_checkpoint_sha256(output_root, task)
    observer.endpoint["entropy_von_neumann"][:, 0] = 0
    observer.endpoint["entropy_renyi2"][:, 0] = 0
    observer.endpoint["entropy_renyi3"][:, 0] = 0
    observer.endpoint["charge_mean"][:, 0] = 0
    observer.endpoint["charge_variance"][:, 0] = 0
    observer.endpoint_width_seen[0] = True
    RUNNER.save_endpoint_progress(
        output_root=output_root,
        scratch_root=scratch_root,
        task=task,
        observer=observer,
        config_sha256="config",
        hashes=hashes,
        final_checkpoint_sha256_value=digest,
    )
    restored = RUNNER.HardWallAllAyContourObserver(
        nx=RUNNER.EXPECTED_NX,
        ny=task.ny,
        physical_cycles=task.cycles,
        sample_ids=task.global_sample_indices,
    )
    valid, reason = RUNNER.load_endpoint_progress(
        output_root=output_root,
        task=task,
        observer=restored,
        config_sha256="config",
        hashes=hashes,
        final_checkpoint_sha256_value=digest,
    )
    assert valid and reason == "verified"
    assert restored.endpoint_width_seen[0]
    assert checkpoint_npz.exists() and checkpoint_json.exists()
    RUNNER.remove_checkpoint(output_root, task)
    assert not checkpoint_npz.exists() and not checkpoint_json.exists()


def test_result_completion_requires_matching_bytes_and_checksum(tmp_path: Path) -> None:
    task = RUNNER.expand_execution_batches(RUNNER.expected_config(), lane="A")[0]
    shard = RUNNER.result_shards(task)[0]
    result_path, completion_path = RUNNER.result_paths(tmp_path, shard)
    result_path.parent.mkdir(parents=True)
    result_path.write_bytes(b"scientific result")
    hashes = {"runner": "hash"}
    completion = RUNNER._completion_identity(
        shard=shard, config_sha256="config", hashes=hashes
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
        shard=shard,
        config_sha256="config",
        hashes=hashes,
    ) == (True, "verified")
    result_path.write_bytes(b"corrupt")
    valid, reason = RUNNER.verified_complete(
        output_root=tmp_path,
        shard=shard,
        config_sha256="config",
        hashes=hashes,
    )
    assert not valid
    assert reason == "result byte-count mismatch"


def test_verified_completion_skips_execution_and_gpu_preflight(
    tmp_path: Path, monkeypatch
) -> None:
    task = _small_resume_task()
    shard = RUNNER.result_shards(task)[0]
    hashes = {"source": "hash"}
    config = RUNNER.expected_config()
    config_sha256 = RUNNER.config_hash(config)
    result_path, completion_path = RUNNER.result_paths(tmp_path / "output", shard)
    result_path.parent.mkdir(parents=True)
    result_path.write_bytes(b"verified")
    completion = RUNNER._completion_identity(
        shard=shard, config_sha256=config_sha256, hashes=hashes
    )
    completion.update(
        {
            "result_filename": result_path.name,
            "result_bytes": result_path.stat().st_size,
            "result_sha256": RUNNER.sha256_file(result_path),
        }
    )
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    monkeypatch.setattr(RUNNER, "expand_execution_batches", lambda *_args, **_kwargs: [task])
    monkeypatch.setattr(RUNNER, "source_hashes", lambda: hashes)
    monkeypatch.setattr(
        RUNNER,
        "validate_a100",
        lambda: (_ for _ in ()).throw(AssertionError("GPU preflight should be skipped")),
    )
    summary = RUNNER.run_campaign(
        config=config,
        lane="B",
        output_root=tmp_path / "output",
        scratch_root=tmp_path / "scratch",
    )
    assert summary["status"] == "complete"
    assert summary["completed_execution_batches"] == 1
    assert summary["new_execution_batches"] == 0


def test_publish_file_rejects_failed_drive_readback(tmp_path: Path, monkeypatch) -> None:
    local = tmp_path / "local.bin"
    final = tmp_path / "drive" / "final.bin"
    local.write_bytes(b"correct")
    original_copy = RUNNER.shutil.copyfile

    def corrupt_copy(source, destination):
        result = original_copy(source, destination)
        Path(destination).write_bytes(b"wrong")
        return result

    monkeypatch.setattr(RUNNER.shutil, "copyfile", corrupt_copy)
    with pytest.raises(OSError, match="byte-count mismatch"):
        RUNNER.publish_file(local, final)
    assert not final.exists()


def test_cli_rejects_negative_execution_batch_limit() -> None:
    with pytest.raises(SystemExit):
        RUNNER.parse_args(
            [
                "--config",
                "config.json",
                "--output-root",
                "output",
                "--scratch-root",
                "scratch",
                "--lane",
                "A",
                "--max-new-execution-batches",
                "-1",
            ]
        )


def test_adaptive_matrix_batch_skips_oom_and_headroom_failures() -> None:
    gib = 1024**3
    rows = [
        {"matrix_batch_size": 128, "representative_seconds": 1.0, "projected_headroom_bytes": 0, "accepted": False, "error": "OOM"},
        {"matrix_batch_size": 80, "representative_seconds": 1.1, "projected_headroom_bytes": 7 * gib, "accepted": False, "error": None},
        {"matrix_batch_size": 64, "representative_seconds": 1.3, "projected_headroom_bytes": 9 * gib, "accepted": True, "error": None},
        {"matrix_batch_size": 32, "representative_seconds": 1.8, "projected_headroom_bytes": 20 * gib, "accepted": True, "error": None},
    ]
    assert RUNNER.select_fastest_accepted_matrix_batch(rows) == 64
    with pytest.raises(RuntimeError, match="no candidate"):
        RUNNER.select_fastest_accepted_matrix_batch(rows[:2])


def test_v2_checkpoint_identity_is_rejected_by_v3(tmp_path: Path) -> None:
    task = _small_resume_task()
    npz, metadata = RUNNER.checkpoint_paths(tmp_path, task)
    npz.parent.mkdir(parents=True)
    npz.write_bytes(b"old")
    metadata.write_text(
        json.dumps({"schema": "hard_wall_entropy_charge_checkpoint_v2"}),
        encoding="utf-8",
    )
    restored, reason = RUNNER.load_checkpoint(
        output_root=tmp_path,
        task=task,
        config_sha256="v3",
        hashes={"source": "v3"},
    )
    assert restored is None
    assert reason == "checkpoint identity mismatch: schema"


def test_endpoint_progress_partial_pair_is_not_resumed(tmp_path: Path) -> None:
    task = _small_resume_task()
    npz, _ = RUNNER.endpoint_progress_paths(tmp_path, task)
    npz.parent.mkdir(parents=True)
    npz.write_bytes(b"partial")
    observer = RUNNER.HardWallAllAyContourObserver(
        nx=RUNNER.EXPECTED_NX,
        ny=task.ny,
        physical_cycles=task.cycles,
        sample_ids=task.global_sample_indices,
    )
    valid, reason = RUNNER.load_endpoint_progress(
        output_root=tmp_path,
        task=task,
        observer=observer,
        config_sha256="v3",
        hashes={"source": "v3"},
        final_checkpoint_sha256_value="a" * 64,
    )
    assert not valid
    assert reason == "incomplete endpoint-progress pair"


def test_benchmark_receipt_above_one_hour_never_unlocks_production(tmp_path: Path) -> None:
    hashes = {"source": "hash"}
    payload = RUNNER._benchmark_identity(
        lane="A", config_sha256="config", hashes=hashes, gpu_name="NVIDIA A100"
    )
    payload.update(
        {
            "matrix_batch_size_by_Ay": {str(ay): 64 for ay in range(31)},
            "projected_20_trajectory_seconds": 3600 + 1,
        }
    )
    path = RUNNER.benchmark_path(tmp_path, "A")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    accepted, reason = RUNNER._load_benchmark(
        output_root=tmp_path,
        lane="A",
        config_sha256="config",
        hashes=hashes,
        gpu_name="NVIDIA A100",
    )
    assert accepted is None
    assert "exceeds one hour" in reason
