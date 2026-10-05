from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

from src.fgtn.classA_U1FGTN import classA_U1FGTN


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    REPO_ROOT
    / "00_WORKSPACE/CURRENT/final_production_new_designs"
    / "09_pure_tangent_replay_acquisition/cpu_tangent_replay"
    / "run_cpu_tangent_replay.py"
)
SPEC = importlib.util.spec_from_file_location("pure_tangent_cpu_replay_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class _Record:
    def __init__(self) -> None:
        self.entries: list[dict] = []

    def __call__(self, *, cycle, site_id, branch_events, **_):
        self.entries.append(
            {
                "cycle": int(cycle),
                "site_id": int(site_id),
                "branch_events": [dict(event) for event in branch_events],
            }
        )


class _NoCovarianceCapture:
    requires_physical_covariance = False

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def __call__(self, **row):
        assert row["G"] is None
        assert row["native_state"] is not None
        self.rows.append(row)


def _small_model() -> classA_U1FGTN:
    model = classA_U1FGTN(Nx=1, Ny=2, DW=False, nshell=0)
    model.construct_OW_projectors(nshell=0, DW=False)
    return model


def test_real_acquisition_expands_to_1200_replay_samples_without_loading_batches():
    config = RUNNER.load_config()
    assert config["schema"] == "pure_tangent_cpu_replay_config_v3"
    assert config["sampling_revision"].endswith("_v3")
    assert config["endpoint_projector_relative_frobenius_tolerance"] == 1e-6
    assert (
        config["endpoint_gate_calibration_revision"]
        == "cpu_gpu_projector_replay_pilot_20260910_v1"
    )
    assert config["cross_replay_projector_relative_frobenius_tolerance"] == 2e-6
    assert (
        config["cross_replay_gate_calibration_revision"]
        == "full_late_cpu_projector_sample018_20260910_v2"
    )
    assert config["saved_products"]["choi_covariance"] is False
    assert config["saved_products"]["per_cycle_dense_jacobians"] is False
    sources = RUNNER.discover_batches(
        RUNNER.DEFAULT_ACQUISITION_ROOT, verify_checksums=False
    )
    tasks = RUNNER.expand_sample_tasks(sources)
    assert len(sources) == 32
    assert len(tasks) == 1200
    assert len({(task.construction, task.ny, task.alpha_1) for task in tasks}) == 12
    assert [task.global_sample_index for task in tasks] == list(range(1200))


def test_real_compact_record_decodes_one_sample_and_preserves_branch_chronology():
    sources = RUNNER.discover_batches(
        RUNNER.DEFAULT_ACQUISITION_ROOT, verify_checksums=False
    )
    task = RUNNER.expand_sample_tasks(sources)[0]
    sample = RUNNER.load_sample(task)
    assert sample["initial_frame"].dtype == np.complex128
    assert sample["final_frame"].dtype == np.complex128
    assert sample["initial_frame"].shape[0] == 40 * task.ny
    assert len(sample["record"]) == task.cycles * sample["updates_per_cycle"]
    assert sample["record"][0]["cycle"] == 1
    assert sample["record"][-1]["cycle"] == task.cycles
    for site in sample["record"][:25]:
        seen_measurements: set[str] = set()
        for event in site["branch_events"]:
            if event["kind"] == "measurement":
                seen_measurements.add(event["channel"])
                assert "probability" not in event
            else:
                assert event["kind"] == "correction"
                assert event["channel"] in seen_measurements
                assert event["target_occupied"] == event["expected_occupied"]
        assert seen_measurements == set(RUNNER.CHANNELS)


def test_cpu_engine_replays_compact_record_with_tangent_observer_without_covariance():
    rng = np.random.default_rng(9182)
    raw = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    initial_frame, _ = np.linalg.qr(raw, mode="reduced")
    recorder = _Record()
    reference = _small_model().run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=2,
        samples=1,
        parallelize_samples=False,
        frame_init=initial_frame,
        save=False,
        sequence="raster_y",
        meas_slab_only=False,
        random_seed=812,
        perfect_correction=True,
        state_representation="physical_frame",
        return_native_state=True,
        trajectory_weight_observer=recorder,
    )
    compact = []
    for entry in recorder.entries:
        compact.append(
            {
                "cycle": entry["cycle"],
                "site_id": entry["site_id"],
                "branch_events": [
                    {
                        key: value
                        for key, value in event.items()
                        if key
                        in {
                            "kind",
                            "channel",
                            "outcome_occupied",
                            "expected_occupied",
                            "target_occupied",
                        }
                    }
                    for event in entry["branch_events"]
                ],
            }
        )
    capture = _NoCovarianceCapture()
    replay = _small_model().run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=2,
        samples=1,
        parallelize_samples=False,
        frame_init=initial_frame,
        save=False,
        sequence="raster_y",
        meas_slab_only=False,
        random_seed=812,
        perfect_correction=True,
        state_representation="physical_frame",
        return_native_state=True,
        trajectory_replay=compact,
        trajectory_replay_probability_tol=0.0,
        lyapunov_frame_observer=capture,
        lyapunov_basis_mode="pure_occupied_empty",
        lyapunov_start_cycle=1,
        lyapunov_full_space=True,
        lyapunov_track_restricted_core=True,
        require_no_covariance_materialization=True,
    )
    assert len(capture.rows) == 2
    assert replay["covariance_materialization_count"] == 0
    np.testing.assert_allclose(
        replay["native_final"]["frame"] @ replay["native_final"]["frame"].conj().T,
        reference["native_final"]["frame"]
        @ reference["native_final"]["frame"].conj().T,
        rtol=1e-12,
        atol=1e-12,
    )


def test_tangent_capture_reconstructs_scale_separated_block_product():
    capture = RUNNER.TangentCapture(physical_cycles=1, start_cycle=1)
    occupied = np.eye(4, dtype=np.complex128)[:, :2]
    empty = np.eye(4, dtype=np.complex128)[:, 2:]
    frame = np.concatenate((occupied, empty), axis=1)[None, ...]
    occupied_core = np.diag([2.0, 1.0]).astype(np.complex128)[None, ...]
    empty_core = np.diag([0.5, 0.25]).astype(np.complex128)[None, ...]
    capture(
        cycle=1,
        lyapunov_cycle=1,
        G=None,
        native_state=object(),
        spectra=np.zeros((1, 4)),
        lyapunov_frame=frame,
        lyapunov_log_diag=np.zeros((1, 4)),
        lyapunov_cycle_null_mask=np.zeros((1, 4), dtype=bool),
        lyapunov_null_counts=np.zeros(1, dtype=np.int64),
        lyapunov_min_branch_probability=np.ones(1),
        lyapunov_min_abs_born_denominator=np.ones(1),
        lyapunov_invalid_branch_count=np.zeros(1, dtype=np.int64),
        lyapunov_block_sizes=(2, 2),
        lyapunov_block_core_hat=(occupied_core, empty_core),
        lyapunov_block_core_log_scale=(np.zeros(1), np.zeros(1)),
        lyapunov_block_core_null_count=(
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
        ),
        lyapunov_initial_block_basis=(occupied[None, ...], empty[None, ...]),
        lyapunov_initial_active_occupations=np.asarray([1.0, 1.0, 0.0, 0.0]),
        lyapunov_initial_active_purity_defect=0.0,
    )
    arrays = capture.finalize(
        prefix="full",
        nx=1,
        ny=2,
        slow_mode_count=2,
        singular_tolerance=1e-14,
        materialize_cocycle=True,
    )
    expected = np.diag([2.0, 1.0, 0.5, 0.25]).astype(np.complex128)
    reconstructed = arrays["final_cocycle_hat"] * np.exp(
        arrays["final_cocycle_log_scale"]
    )
    np.testing.assert_allclose(reconstructed, expected, rtol=1e-13, atol=1e-13)
    assert arrays["full_qr_log_increments"].shape == (1, 4)
    assert arrays["full_slow_x_profiles"].shape == (2, 1)


def test_one_cycle_materializer_reconstructs_qr_map(tmp_path):
    source = RUNNER.BatchSource(
        task_id="batch",
        construction="soft",
        ny=24,
        alpha_1=1.0,
        cycles=48,
        batch_index=0,
        batch_seed=1,
        case_sample_indices=(0,),
        global_sample_indices=(0,),
        result_path="input.npz",
        completion_path="input.complete.json",
        result_bytes=1,
        result_sha256="a" * 64,
        completion_sha256="b" * 64,
        acquisition_configuration_sha256="c" * 64,
        acquisition_source_hashes={},
    )
    task = RUNNER.SampleTask(
        task_id="soft_Ny024_a1-1_sample-000_global-0000",
        construction="soft",
        ny=24,
        alpha_1=1.0,
        cycles=48,
        case_sample_index=0,
        global_sample_index=0,
        batch_row=0,
        source=source,
    )
    observer = RUNNER.OneCycleMaterializer(
        task=task,
        requested_cycles={1},
        root=tmp_path,
        config_sha256="d" * 64,
        hashes={},
    )
    q0 = np.eye(4, dtype=np.complex128)
    permutation = q0[:, [1, 0, 3, 2]]
    r = np.diag([2.0, 1.0, 0.5, 0.25]).astype(np.complex128)
    observer(
        cycle=1,
        G=None,
        lyapunov_frame=permutation[None, ...],
        lyapunov_qr_r=r[None, ...],
        lyapunov_initial_block_basis=(q0[:, :2][None, ...], q0[:, 2:][None, ...]),
    )
    assert len(observer.written) == 1
    with np.load(observer.written[0], allow_pickle=False) as saved:
        actual = saved["one_cycle_cocycle_hat"] * np.exp(
            saved["one_cycle_cocycle_log_scale"]
        )
    np.testing.assert_allclose(actual, permutation @ r, rtol=1e-13, atol=1e-13)


def test_sample_completion_is_checksum_resumable_and_partial_pairs_fail_closed(tmp_path):
    source = RUNNER.BatchSource(
        task_id="batch",
        construction="hard",
        ny=24,
        alpha_1=1.0,
        cycles=48,
        batch_index=0,
        batch_seed=1,
        case_sample_indices=(0,),
        global_sample_indices=(0,),
        result_path="input.npz",
        completion_path="input.complete.json",
        result_bytes=123,
        result_sha256="a" * 64,
        completion_sha256="b" * 64,
        acquisition_configuration_sha256="c" * 64,
        acquisition_source_hashes={"gpu": "d" * 64},
    )
    task = RUNNER.SampleTask(
        task_id="hard_Ny024_a1-1_sample-000_global-0000",
        construction="hard",
        ny=24,
        alpha_1=1.0,
        cycles=48,
        case_sample_index=0,
        global_sample_index=0,
        batch_row=0,
        source=source,
    )
    hashes = {"runner": "e" * 64}
    config_sha = "f" * 64
    result_path, completion_path = RUNNER.result_paths(tmp_path, task)
    RUNNER.atomic_npz(
        result_path,
        {
            "schema": np.asarray(RUNNER.RESULT_SCHEMA),
            "task_id": np.asarray(task.task_id),
            "configuration_sha256": np.asarray(config_sha),
            "acquisition_result_sha256": np.asarray(source.result_sha256),
            "endpoint_replay_verified": np.asarray(True),
            "endpoint_projector_relative_frobenius_error_full": np.asarray(0.0),
            "endpoint_projector_relative_frobenius_error_late": np.asarray(0.0),
            "endpoint_projector_relative_frobenius_tolerance": np.asarray(1e-6),
            "cross_replay_projector_relative_frobenius_error": np.asarray(0.0),
            "cross_replay_projector_relative_frobenius_tolerance": np.asarray(2e-6),
            "endpoint_gate_calibration_revision": np.asarray(
                "cpu_gpu_projector_replay_pilot_20260910_v1"
            ),
            "cross_replay_gate_calibration_revision": np.asarray(
                "full_late_cpu_projector_sample018_20260910_v2"
            ),
            "active_dimension": np.asarray(2, dtype=np.int64),
            "final_cocycle_hat": np.zeros((960, 960), dtype=np.complex128),
            "full_qr_log_increments": np.zeros((48, 2), dtype=np.float64),
        },
    )
    complete, reason = RUNNER.verified_complete(
        tmp_path, task, config_sha256=config_sha, hashes=hashes
    )
    assert not complete and reason == "incomplete pair"

    completion = RUNNER.task_identity(
        task, config_sha256=config_sha, hashes=hashes
    )
    completion.update(
        {
            "endpoint_projector_relative_frobenius_tolerance": 1e-6,
            "endpoint_gate_calibration_revision": "cpu_gpu_projector_replay_pilot_20260910_v1",
            "cross_replay_projector_relative_frobenius_tolerance": 2e-6,
            "cross_replay_gate_calibration_revision": "full_late_cpu_projector_sample018_20260910_v2",
            "result_filename": result_path.name,
            "result_bytes": result_path.stat().st_size,
            "result_sha256": RUNNER.sha256_file(result_path),
        }
    )
    RUNNER.atomic_json(completion_path, completion)
    assert RUNNER.verified_complete(
        tmp_path, task, config_sha256=config_sha, hashes=hashes
    ) == (True, "verified")

    with result_path.open("ab") as handle:
        handle.write(b"corruption")
    complete, reason = RUNNER.verified_complete(
        tmp_path, task, config_sha256=config_sha, hashes=hashes
    )
    assert not complete and reason == "result byte-count mismatch"
