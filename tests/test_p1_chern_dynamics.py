from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "00_WORKSPACE/CURRENT/final_production_new_designs/01_p1_chern_dynamics"
SHARED = ROOT / "00_WORKSPACE/CURRENT/final_production_new_designs/_shared_src"
sys.path.insert(0, str(SHARED))

_observer_spec = importlib.util.spec_from_file_location(
    "new_design_p1_observables_under_test", SHARED / "p1_chern_observables.py"
)
assert _observer_spec is not None and _observer_spec.loader is not None
NEW_P1_OBSERVABLES = importlib.util.module_from_spec(_observer_spec)
sys.modules[_observer_spec.name] = NEW_P1_OBSERVABLES
_observer_spec.loader.exec_module(NEW_P1_OBSERVABLES)

_prior_observer = sys.modules.get("p1_chern_observables")
sys.modules["p1_chern_observables"] = NEW_P1_OBSERVABLES
_runner_spec = importlib.util.spec_from_file_location(
    "new_design_p1_runner_under_test", BUNDLE / "src/p1_chern_runner.py"
)
assert _runner_spec is not None and _runner_spec.loader is not None
NEW_P1_RUNNER = importlib.util.module_from_spec(_runner_spec)
sys.modules[_runner_spec.name] = NEW_P1_RUNNER
_runner_spec.loader.exec_module(NEW_P1_RUNNER)
_prior_runner = sys.modules.get("p1_chern_runner")
sys.modules["p1_chern_runner"] = NEW_P1_RUNNER
_analysis_spec = importlib.util.spec_from_file_location(
    "new_design_p1_analysis_under_test", SHARED / "p1_chern_analysis.py"
)
assert _analysis_spec is not None and _analysis_spec.loader is not None
NEW_P1_ANALYSIS = importlib.util.module_from_spec(_analysis_spec)
sys.modules[_analysis_spec.name] = NEW_P1_ANALYSIS
_analysis_spec.loader.exec_module(NEW_P1_ANALYSIS)
if _prior_runner is None:
    sys.modules.pop("p1_chern_runner", None)
else:
    sys.modules["p1_chern_runner"] = _prior_runner
if _prior_observer is None:
    sys.modules.pop("p1_chern_observables", None)
else:
    sys.modules["p1_chern_observables"] = _prior_observer

analyze = NEW_P1_ANALYSIS.analyze
merge_archives = NEW_P1_ANALYSIS.merge_archives
P1ChernObserver = NEW_P1_OBSERVABLES.P1ChernObserver
build_periodic_chern_partition_indices = (
    NEW_P1_OBSERVABLES.build_periodic_chern_partition_indices
)
real_space_chern_from_centered_covariance = (
    NEW_P1_OBSERVABLES.real_space_chern_from_centered_covariance
)
real_space_chern_from_frame = NEW_P1_OBSERVABLES.real_space_chern_from_frame
sample_trijunction_centers = NEW_P1_OBSERVABLES.sample_trijunction_centers
A100_PREFLIGHT_SCHEMA = NEW_P1_RUNNER.A100_PREFLIGHT_SCHEMA
CONTRACT_AUDIT_SHA256 = NEW_P1_RUNNER.CONTRACT_AUDIT_SHA256
SAMPLING_REVISION = NEW_P1_RUNNER.SAMPLING_REVISION
_archive_paths = NEW_P1_RUNNER._archive_paths
_preflight_receipt_path = NEW_P1_RUNNER._preflight_receipt_path
_require_safe_preflight = NEW_P1_RUNNER._require_safe_preflight
expand_cases = NEW_P1_RUNNER.expand_cases
global_sample_indices = NEW_P1_RUNNER.global_sample_indices
load_config = NEW_P1_RUNNER.load_config
samples_per_shard = NEW_P1_RUNNER.samples_per_shard
shard_count = NEW_P1_RUNNER.shard_count
sha256_file = NEW_P1_RUNNER.sha256_file
from src.fgtn.occupied_frame_gpu import BatchedOccupiedFrameState  # noqa: E402
from src.fgtn.classA_U1FGTN_gpu import classA_U1FGTN_gpu  # noqa: E402


def test_case_matrix_is_exactly_the_approved_twelve_cases() -> None:
    config = load_config(BUNDLE)
    cases = expand_cases(config)
    assert len(cases) == 12
    assert {(case["model"]["Nx"], case["model"]["nshell"]) for case in cases} == {
        (size, shell) for size in (16, 24, 32, 64) for shell in (1, 2, None)
    }
    for case in cases:
        size = case["model"]["Nx"]
        assert case["model"]["Ny"] == size
        assert case["model"]["DW"] is False
        assert case["model"]["alpha_1"] == case["model"]["alpha_2"] == 1.0
        assert case["model"]["init_mode"] == "default"
        assert case["run"] == {
            "cycles": size,
            "samples": 25,
            "sequence": "random",
            "perfect_correction": True,
            "postselect": False,
            "postselect_probability": 0.0,
            "n_a": 0.5,
        }
        expected_width = 1 if size == 64 else 5
        assert case["execution"] == {"samples_per_shard": expected_width}
        assert samples_per_shard(case) == expected_width
        assert shard_count(case) == (25 if size == 64 else 5)
        assert case["observer"]["cycles"] == list(range(size + 1))


def test_remote_checkpoint_correction_has_a_new_locked_v4_identity() -> None:
    config = load_config(BUNDLE)
    assert SAMPLING_REVISION == "production_25sample_p1_chern_v4"
    assert config["sampling_revision"] == SAMPLING_REVISION
    assert config["production_output_collection"].endswith(SAMPLING_REVISION)
    assert config["pilot_output_collection"].endswith(SAMPLING_REVISION)
    checkpoint_unit = str(config["locked_contract"]["checkpoint_unit"])
    assert "cycle" in checkpoint_unit
    assert "trajectory_archive" not in checkpoint_unit
    assert A100_PREFLIGHT_SCHEMA.endswith("_v4")
    assert config["drive_commit"]["authoritative_backend"] == "google_drive_api_v3"
    assert config["drive_commit"]["drivefs_role"] == "non_authoritative_cache"

def test_shards_have_unique_archives_and_production_requires_safe_preflight(
    tmp_path: Path,
) -> None:
    config = load_config(BUNDLE)
    case = expand_cases(config)[0]
    first = _archive_paths(
        bundle_root=BUNDLE, config=config, case=case, shard_index=0,
        drive_root=tmp_path, mode="production",
    )
    second = _archive_paths(
        bundle_root=BUNDLE, config=config, case=case, shard_index=1,
        drive_root=tmp_path, mode="production",
    )
    assert first[1] != second[1]
    assert first[2] != second[2]
    with pytest.raises(RuntimeError, match="locked until --a100-preflight"):
        _require_safe_preflight(
            drive_root=tmp_path, config=config, bundle_root=BUNDLE
        )


def _write_bootstrap_evidence(tmp_path: Path) -> tuple[dict, Path]:
    config = load_config(BUNDLE)
    case = next(
        case
        for case in expand_cases(config)
        if case["case_id"] == "P1_CHERN_L64_nsh-1"
    )
    _, archive, run_id, run_config = _archive_paths(
        bundle_root=BUNDLE,
        config=config,
        case=case,
        shard_index=0,
        drive_root=tmp_path,
        mode="production",
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    engine_hash = run_config["canonical_engine_sha256"]
    run_config_hash = NEW_P1_RUNNER.sha256_json(run_config)
    manifest = {
        "status": "complete_local",
        "bundle": "01_p1_chern_dynamics",
        "sampling_revision": SAMPLING_REVISION,
        "audit_sha256": CONTRACT_AUDIT_SHA256,
        "canonical_entry_point": NEW_P1_RUNNER.ENTRY_POINT,
        "case_id": case["case_id"],
        "shard_index": 0,
        "global_sample_indices": [0],
        "canonical_engine_sha256": engine_hash,
        "run_config": run_config,
        "run_config_hash": run_config_hash,
        "source_hashes": run_config["bundle_source_hashes"],
        "elapsed_seconds": 123.0,
        "gpu_preflight": {"device": "NVIDIA A100-SXM4-40GB", "total_bytes": 40_000},
        "gpu_peak_allocated_bytes": 10_000,
        "gpu_peak_reserved_bytes": 20_000,
    }
    with tarfile.open(archive, "w:gz") as handle:
        _tar_member(handle, "manifest.json", json.dumps(manifest).encode())
    archive_hash = sha256_file(archive)
    archive.with_suffix(archive.suffix + ".receipt.json").write_text(
        json.dumps({
            "schema_version": 1,
            "run_id": run_id,
            "archive": archive.name,
            "archive_sha256": archive_hash,
            "archive_bytes": archive.stat().st_size,
        }),
        encoding="utf-8",
    )
    receipt = _preflight_receipt_path(drive_root=tmp_path, config=config)
    receipt.write_text(
        json.dumps({
            "schema": A100_PREFLIGHT_SCHEMA,
            "bundle": "01_p1_chern_dynamics",
            "sampling_revision": SAMPLING_REVISION,
            "audit_sha256": CONTRACT_AUDIT_SHA256,
            "scientific_contract_changed": False,
            "execution_contract_changed": True,
            "canonical_engine_sha256": engine_hash,
            "bundle_source_hashes_sha256": run_config[
                "bundle_source_hashes_sha256"
            ],
            "bootstrap_run_config_hash": run_config_hash,
            "checkpoint_interval_cycles": NEW_P1_RUNNER.CHECKPOINT_INTERVAL_CYCLES,
            "measured_seconds_per_cycle": 123.0 / 64.0,
            "bootstrap_shard_index": 0,
            "bootstrap_global_sample_indices": [0],
            "bootstrap_archive": archive.name,
            "bootstrap_archive_sha256": archive_hash,
            "safe": True,
        }),
        encoding="utf-8",
    )
    return config, archive


def test_safe_preflight_is_bound_to_the_bootstrap_archive(tmp_path: Path) -> None:
    config, archive = _write_bootstrap_evidence(tmp_path)
    assert _require_safe_preflight(
        drive_root=tmp_path, config=config, bundle_root=BUNDLE
    )["safe"] is True
    archive.write_bytes(archive.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="archive checksum mismatch"):
        _require_safe_preflight(
            drive_root=tmp_path, config=config, bundle_root=BUNDLE
        )


def test_a100_qualification_reuses_real_sample_zero_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, archive = _write_bootstrap_evidence(tmp_path)
    _preflight_receipt_path(drive_root=tmp_path, config=config).unlink()

    def existing_archive_run(**kwargs):
        assert kwargs["archive_result"] is True
        assert kwargs["case"]["case_id"] == "P1_CHERN_L64_nsh-1"
        assert kwargs["case"]["execution"]["samples_per_shard"] == 1
        assert kwargs["shard_index"] == 0
        return {"status": "already_archived"}

    monkeypatch.setattr(NEW_P1_RUNNER, "_run_case", existing_archive_run)
    result = NEW_P1_RUNNER._a100_preflight(
        bundle_root=BUNDLE, config=config, drive_root=tmp_path
    )
    assert result["safe"] is True
    assert result["measured_trajectories"] == 1
    assert result["bootstrap_archive"] == archive.name
    assert result["bootstrap_global_sample_indices"] == [0]


def test_lightweight_production_preflight_does_not_require_a_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        NEW_P1_RUNNER, "_require_a100", lambda *, smoke: {"device": "A100-test"}
    )

    def forbidden_receipt_check(**_kwargs):
        raise AssertionError("lightweight preflight consulted the qualification receipt")

    monkeypatch.setattr(NEW_P1_RUNNER, "_require_safe_preflight", forbidden_receipt_check)
    assert NEW_P1_RUNNER.main([
        "--bundle-root", str(BUNDLE),
        "--drive-root", str(tmp_path),
        "--mode", "production",
        "--case-id", "P1_CHERN_L16_nsh-1",
        "--shard-index", "0",
        "--preflight-only",
    ]) == 0


def test_a100_qualification_reuses_a_current_safe_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _write_bootstrap_evidence(tmp_path)

    def forbidden_expensive_run(**_kwargs):
        raise AssertionError("current safe qualification was rerun")

    monkeypatch.setattr(NEW_P1_RUNNER, "_a100_preflight", forbidden_expensive_run)
    assert NEW_P1_RUNNER.main([
        "--bundle-root", str(BUNDLE),
        "--drive-root", str(tmp_path),
        "--mode", "production",
        "--a100-preflight",
    ]) == 0


def test_centers_are_deterministic_distinct_fresh_and_shell_independent() -> None:
    first = sample_trijunction_centers(
        root_seed=2026081701, size=16, sample_id=3, cycle=7
    )
    repeat = sample_trijunction_centers(
        root_seed=2026081701, size=16, sample_id=3, cycle=7
    )
    next_cycle = sample_trijunction_centers(
        root_seed=2026081701, size=16, sample_id=3, cycle=8
    )
    np.testing.assert_array_equal(first, repeat)
    assert len(np.unique(first[:, 0] + 16 * first[:, 1])) == 10
    assert not np.array_equal(first, next_cycle)


def test_canonical_gpu_engine_cycle_segments_resume_bitwise_on_cpu(
    tmp_path: Path,
) -> None:
    """The public native-frame API is an exact cycle-boundary restart primitive."""

    model_kwargs = {
        "Nx": 2,
        "Ny": 2,
        "DW": False,
        "nshell": 1,
        "alpha_1": 1.0,
        "alpha_2": 1.0,
        "trial_orbitals": "X",
        "dw_truncation": False,
        "device": "cpu",
        "dtype": "complex128",
        "backend": "local",
    }
    run_kwargs = {
        "samples": 2,
        "batch_size": 2,
        "init_mode": "default",
        "sequence": "random",
        "perfect_correction": True,
        "postselect": False,
        "postselect_probability": 0.0,
        "n_a": 0.5,
        "G_history": False,
        "save": False,
        "progress": False,
        "return_data": True,
        "state_representation": "auto",
        "return_native_state": True,
        "require_no_covariance_materialization": True,
    }
    seed = 917_231

    torch.manual_seed(seed)
    uninterrupted = classA_U1FGTN_gpu(**model_kwargs).run_markov_circuit(
        cycles=3, **run_kwargs
    )
    uninterrupted_rng = torch.get_rng_state().clone()

    torch.manual_seed(seed)
    frame = None
    ranks = None
    for _ in range(2):
        initial = {} if frame is None else {"frame_init": frame, "frame_ranks": ranks}
        segment = classA_U1FGTN_gpu(**model_kwargs).run_markov_circuit(
            cycles=1, **initial, **run_kwargs
        )
        frame = np.array(segment["native_final"]["frame"], copy=True)
        ranks = np.array(segment["native_final"]["ranks"], copy=True)

    checkpoint = tmp_path / "cycle_checkpoint.npz"
    with checkpoint.open("wb") as handle:
        np.savez(
            handle,
            completed_cycle=np.asarray(2, dtype=np.int64),
            frame=frame,
            ranks=ranks,
            torch_cpu_rng_state=torch.get_rng_state().cpu().numpy(),
        )

    # Emulate a fresh process whose ambient RNG state is unrelated to the run.
    torch.manual_seed(1)
    with np.load(checkpoint, allow_pickle=False) as saved:
        assert int(saved["completed_cycle"]) == 2
        frame = np.array(saved["frame"], copy=True)
        ranks = np.array(saved["ranks"], copy=True)
        torch.set_rng_state(
            torch.as_tensor(
                np.array(saved["torch_cpu_rng_state"], copy=True), dtype=torch.uint8
            )
        )

    resumed = classA_U1FGTN_gpu(**model_kwargs).run_markov_circuit(
        cycles=1,
        frame_init=frame,
        frame_ranks=ranks,
        **run_kwargs,
    )
    np.testing.assert_array_equal(
        resumed["native_final"]["frame"], uninterrupted["native_final"]["frame"]
    )
    np.testing.assert_array_equal(
        resumed["native_final"]["ranks"], uninterrupted["native_final"]["ranks"]
    )
    assert torch.equal(torch.get_rng_state(), uninterrupted_rng)
    assert resumed["covariance_materialization_count"] == 0

def _translate_frame(frame: torch.Tensor, size: int, dx: int, dy: int) -> torch.Tensor:
    translated = torch.empty_like(frame)
    for y in range(size):
        for x in range(size):
            for orbital in range(2):
                old = orbital + 2 * x + 2 * size * y
                new = orbital + 2 * ((x + dx) % size) + 2 * size * ((y + dy) % size)
                translated[:, new] = frame[:, old]
    return translated


def test_periodic_chern_is_translation_covariant_and_frame_matches_dense() -> None:
    size = 8
    dimension = 2 * size * size
    state = BatchedOccupiedFrameState.random_pure(
        2,
        dimension,
        dimension // 2,
        device="cpu",
        dtype=torch.complex128,
        generator=torch.Generator(device="cpu").manual_seed(90210),
    )
    partition = build_periodic_chern_partition_indices(
        nx=size, ny=size, xref=0, yref=1, radius=0.4 * size
    )
    frame_value = real_space_chern_from_frame(state.frame, partition)
    centered = 2.0 * (state.frame @ state.frame.mH) - torch.eye(
        dimension, dtype=torch.complex128
    )
    dense_value = real_space_chern_from_centered_covariance(centered, partition)
    np.testing.assert_allclose(frame_value.numpy(), dense_value.numpy(), atol=2e-10)

    dx, dy = 3, 2
    shifted = _translate_frame(state.frame, size, dx, dy)
    shifted_partition = build_periodic_chern_partition_indices(
        nx=size,
        ny=size,
        xref=(0 + dx) % size,
        yref=(1 + dy) % size,
        radius=0.4 * size,
    )
    shifted_value = real_space_chern_from_frame(shifted, shifted_partition)
    np.testing.assert_allclose(frame_value.numpy(), shifted_value.numpy(), atol=2e-10)


def test_observer_saves_only_complete_minimal_chern_schema(tmp_path: Path) -> None:
    size = 4
    dimension = 2 * size * size
    state = BatchedOccupiedFrameState.random_pure(
        2,
        dimension,
        dimension // 2,
        device="cpu",
        dtype=torch.complex128,
        generator=torch.Generator(device="cpu").manual_seed(11),
    )
    observer = P1ChernObserver(
        size=size,
        physical_cycles=size,
        global_sample_ids=[5, 6],
        root_seed=123,
    )
    for cycle in range(size + 1):
        observer(cycle=cycle, state=state, batch_start=0, batch_count=2)
    path = tmp_path / "p1_chern.npz"
    observer.save(path, config={"test": True})
    with np.load(path, allow_pickle=False) as data:
        assert data["chern_by_center"].shape == (2, size + 1, 10)
        assert data["chern_center_mean"].shape == (2, size + 1)
        assert data["cycles"].tolist() == list(range(size + 1))
        assert not {
            "covariance", "bott_index", "density", "entropy_contour", "tangent",
            "ordered_record", "purity_gap", "convergence",
        }.intersection(data.files)


def test_observer_checkpoint_round_trip_and_rejects_invalid_prefix() -> None:
    size = 4
    dimension = 2 * size * size
    state = BatchedOccupiedFrameState.random_pure(
        1,
        dimension,
        dimension // 2,
        device="cpu",
        dtype=torch.complex128,
        generator=torch.Generator(device="cpu").manual_seed(1234),
    )
    observer = P1ChernObserver(
        size=size,
        physical_cycles=size,
        global_sample_ids=[7],
        root_seed=2026081701,
    )
    for cycle in range(3):
        observer(cycle=cycle, state=state, batch_start=0, batch_count=1)
    checkpoint = observer.checkpoint_state()

    restored = P1ChernObserver(
        size=size,
        physical_cycles=size,
        global_sample_ids=[7],
        root_seed=2026081701,
    )
    result = restored.restore_checkpoint_state(checkpoint, completed_cycle=2)
    assert result["completed_cycle"] == 2
    np.testing.assert_array_equal(restored.center_x, observer.center_x)
    np.testing.assert_array_equal(restored.center_y, observer.center_y)
    np.testing.assert_array_equal(restored.chern_by_center, observer.chern_by_center)
    np.testing.assert_array_equal(restored.seen, observer.seen)

    noncontiguous = {key: np.array(value, copy=True) for key, value in checkpoint.items()}
    noncontiguous["seen"][0, 3] = True
    with pytest.raises(ValueError, match="contiguous cycle prefix"):
        restored.restore_checkpoint_state(noncontiguous, completed_cycle=2)

    wrong_centers = {key: np.array(value, copy=True) for key, value in checkpoint.items()}
    wrong_centers["center_x"][0, 1, 0] = (
        wrong_centers["center_x"][0, 1, 0] + 1
    ) % size
    with pytest.raises(ValueError, match="deterministic validation"):
        restored.restore_checkpoint_state(wrong_centers, completed_cycle=2)

def _tar_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def _write_synthetic_archive(
    root: Path, case: dict, shard: int, root_seed: int = 2026081701
) -> None:
    size = int(case["model"]["Nx"])
    shell = case["model"]["nshell"]
    width = int(case["execution"]["samples_per_shard"])
    sample_ids = np.arange(shard * width, shard * width + width, dtype=np.int64)
    cycles = np.arange(size + 1, dtype=np.int64)
    center_x = np.empty((width, size + 1, 10), dtype=np.int64)
    center_y = np.empty_like(center_x)
    for local, sample_id in enumerate(sample_ids):
        for cycle in cycles:
            centers = sample_trijunction_centers(
                root_seed=root_seed,
                size=size,
                sample_id=int(sample_id),
                cycle=int(cycle),
            )
            center_x[local, cycle], center_y[local, cycle] = centers.T
    shell_offset = {1: 0.0, 2: 0.01, None: 0.02}[shell]
    center_axis = np.arange(10, dtype=np.float64)[None, None, :] * 1e-4
    raw = (
        0.5
        + shell_offset
        + sample_ids[:, None, None] * 1e-3
        + cycles[None, :, None] * 1e-2
        + center_axis
    )
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        cycles=cycles,
        global_sample_ids=sample_ids,
        center_x=center_x,
        center_y=center_y,
        chern_by_center=raw,
        chern_center_mean=raw.mean(axis=-1),
        observer_seconds=np.zeros((width, size + 1)),
    )
    manifest = {
        "bundle": "01_p1_chern_dynamics",
        "audit_sha256": CONTRACT_AUDIT_SHA256,
        "shard_index": shard,
        "global_sample_indices": sample_ids.tolist(),
        "run_config": {"sampling_revision": SAMPLING_REVISION, "case": case},
    }
    run_id = f"{case['case_id']}_shard_{shard}"
    path = root / f"{run_id}.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        _tar_member(archive, "manifest.json", json.dumps(manifest).encode())
        _tar_member(archive, f"shards/shard_{shard:03d}/p1_chern.npz", buffer.getvalue())
    receipt = {
        "archive": path.name,
        "archive_sha256": sha256_file(path),
    }
    path.with_suffix(path.suffix + ".receipt.json").write_text(json.dumps(receipt))


def test_merge_uses_exact_25_samples_and_center_then_sample_average(tmp_path: Path) -> None:
    config = load_config(BUNDLE)
    for case in expand_cases(config):
        for shard in range(shard_count(case)):
            _write_synthetic_archive(tmp_path, case, shard)
    merged = merge_archives(tmp_path)
    assert len(merged) == 12
    item = merged["P1_CHERN_L16_nsh-1"]
    assert item["chern_by_center"].shape == (25, 17, 10)
    expected = item["chern_by_center"].mean(axis=-1).mean(axis=0)
    np.testing.assert_allclose(
        item["chern_center_mean"].mean(axis=0), expected, rtol=0.0, atol=1e-14
    )
    output = tmp_path / "analysis"
    summary = analyze(tmp_path, output)
    assert summary["case_count"] == 12
    assert summary["samples_per_case"] == 25
    assert (output / "p1_chern_dynamics_reference_style.pdf").is_file()
    assert (output / "p1_chern_dynamics_reference_style.png").is_file()
