from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
BUNDLE = (
    REPO
    / "00_WORKSPACE/CURRENT/final_production_new_designs/07_maxmix_hard_soft_purification"
)


def _load(relative: str, name: str):
    path = BUNDLE / relative
    for candidate in (BUNDLE, BUNDLE / "src"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OBSERVER = _load("purification_observer.py", "tested_purification_observer")
RUNNER = _load("run_campaign.py", "tested_purification_runner")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _notebook_config(path: Path) -> tuple[dict, str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "CONFIG = {" in "".join(cell.get("source", []))
    )
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "CONFIG" for target in node.targets)
    )
    return ast.literal_eval(assignment.value), source


def test_locked_contract_expands_six_cases_and_120_five_sample_shards() -> None:
    config = RUNNER.expected_config()
    assert RUNNER.validate_config(config) is None
    assert config["Nx"] == 20
    assert config["Ny_values"] == [20, 30, 40]
    assert config["samples_per_case"] == 100
    assert config["cycles_multiplier"] == 4
    assert config["nshell"] == 1
    assert config["init_mode"] == "maxmix"
    assert config["perfect_correction"] is True
    assert config["sequence"] == "raster_y"
    assert config["dtype"] == "complex128"
    assert config["constructions"] == {
        "hard": {"DW": True, "dw_truncation": True, "meas_slab_only": True},
        "soft": {"DW": True, "dw_truncation": False, "meas_slab_only": False},
    }
    assert config["execution_batch_size_by_Ny"] == {"20": 100, "30": 100, "40": 40}
    assert config["resume_initial_purity_tolerance"] == 0.50000001
    assert config["observables"]["measurement_log_probability"] is True
    assert config["observables"]["cumulative_log_probability"] is True
    assert config["observables"]["log_probability_dtype"] == "float64"
    assert RUNNER.SAMPLING_REVISION.endswith("_v3")
    all_tasks = []
    all_shards = []
    for construction in ("hard", "soft"):
        tasks = RUNNER.expand_execution_batches(config, construction)
        shards = RUNNER.all_result_shards(config, construction)
        assert len(tasks) == 5
        assert len(shards) == 60
        assert sum(task.samples for task in tasks) == 300
        for ny in (20, 30, 40):
            ids = sorted(
                int(index)
                for shard in shards
                if shard.execution.ny == ny
                for index in shard.sample_indices
            )
            assert ids == list(range(100))
        all_tasks.extend(tasks)
        all_shards.extend(shards)
    assert len(all_tasks) == 10
    assert len(all_shards) == 120
    assert len({task.seed for task in all_tasks}) == 10
    assert all(shard.sample_indices.size == 5 for shard in all_shards)
    assert all(task.cycles == 4 * task.ny for task in all_tasks)


def test_covariance_observables_match_dense_formulas_and_close() -> None:
    nx, ny, samples = 2, 2, 3
    n = 2 * nx * ny
    generator = np.random.default_rng(4103)
    covariances = []
    expected_nu = []
    expected_entropy = []
    expected_variance = []
    for _ in range(samples):
        unitary, _ = np.linalg.qr(
            generator.normal(size=(n, n)) + 1j * generator.normal(size=(n, n))
        )
        nu = generator.uniform(0.05, 0.95, size=n)
        C = (unitary * nu[None, :]) @ unitary.conj().T
        covariances.append(2.0 * C - np.eye(n))
        expected_nu.append(np.linalg.eigvalsh(C))
        expected_entropy.append(np.sum(-nu * np.log(nu) - (1.0 - nu) * np.log(1.0 - nu)))
        expected_variance.append(np.sum(nu * (1.0 - nu)))
    result = OBSERVER.covariance_observables(
        torch.as_tensor(np.asarray(covariances), dtype=torch.complex128),
        nx=nx,
        ny=ny,
        sample_chunk=2,
    )
    np.testing.assert_allclose(result.occupation_spectrum, expected_nu, atol=2e-12, rtol=0)
    np.testing.assert_allclose(result.total_entropy, expected_entropy, atol=2e-12, rtol=0)
    np.testing.assert_allclose(result.total_charge_variance, expected_variance, atol=2e-12, rtol=0)
    np.testing.assert_allclose(
        result.entropy_contour.sum(axis=(1, 2)), result.total_entropy, atol=2e-12, rtol=0
    )
    np.testing.assert_allclose(
        result.charge_variance_contour.sum(axis=(1, 2)),
        result.total_charge_variance,
        atol=2e-12,
        rtol=0,
    )
    assert np.max(result.hermiticity_residual) < 2e-15


def _gpu_model():
    return RUNNER.classA_U1FGTN_gpu(
        Nx=4,
        Ny=4,
        DW=True,
        nshell=1,
        alpha_1=1,
        alpha_2=30,
        dw_truncation=True,
        device="cpu",
        dtype="complex128",
        backend="local",
    )


def _engine_call(
    model,
    *,
    cycles: int,
    G_init=None,
    prepared: bool = False,
    callback=None,
    record_callback=None,
):
    return model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=cycles,
        postselect=False,
        perfect_correction=True,
        samples=2,
        init_mode="maxmix",
        G_init=G_init,
        G_init_prepared=prepared,
        save=False,
        n_a=0.5,
        sequence="raster_y",
        meas_slab_only=True,
        batch_size=2,
        return_data=True,
        state_representation="covariance",
        cycle_observer=callback,
        record_observer=record_callback,
    )


def test_prepared_covariance_resume_is_bitwise_exact_on_cpu_backend() -> None:
    torch.manual_seed(7731)
    np.random.seed(7731)
    continuous_cycles = {}
    continuous = _engine_call(
        _gpu_model(),
        cycles=4,
        callback=lambda cycle, G, **_: continuous_cycles.__setitem__(
            int(cycle), G.detach().cpu().numpy().copy()
        ),
    )
    continuous_torch_rng = torch.get_rng_state().clone()
    continuous_numpy_next = np.random.random(8)

    torch.manual_seed(7731)
    np.random.seed(7731)
    resumed_cycles = {}
    first = _engine_call(
        _gpu_model(),
        cycles=2,
        callback=lambda cycle, G, **_: resumed_cycles.__setitem__(
            int(cycle), G.detach().cpu().numpy().copy()
        ),
    )
    saved_torch = torch.get_rng_state().clone()
    saved_numpy = np.random.get_state()
    continuation_model = _gpu_model()
    torch.set_rng_state(saved_torch)
    np.random.set_state(saved_numpy)

    def resumed_callback(*, cycle, G, **_):
        if int(cycle):
            resumed_cycles[2 + int(cycle)] = G.detach().cpu().numpy().copy()

    second = _engine_call(
        continuation_model,
        cycles=2,
        G_init=first["G_final"],
        prepared=True,
        callback=resumed_callback,
    )
    assert first["exterior_preparation_performed"] is True
    assert second["exterior_preparation_performed"] is False
    assert second["exterior_preparation"] == "skipped_prepared_covariance"
    assert second["G_init_prepared"] is True
    assert np.array_equal(continuous["G_final"], second["G_final"])
    assert torch.equal(continuous_torch_rng, torch.get_rng_state())
    assert np.array_equal(continuous_numpy_next, np.random.random(8))
    assert sorted(continuous_cycles) == sorted(resumed_cycles) == [0, 1, 2, 3, 4]
    assert all(np.array_equal(continuous_cycles[k], resumed_cycles[k]) for k in continuous_cycles)


def test_prepared_covariance_flag_is_backward_compatible_and_validated() -> None:
    model = _gpu_model()
    with pytest.raises(ValueError, match="requires G_init"):
        _engine_call(model, cycles=1, prepared=True)


def test_locked_resume_tolerance_accepts_a_valid_heterogeneous_covariance_batch() -> None:
    model = _gpu_model()
    n = model.Nlayer
    pure = np.diag(
        np.concatenate((np.ones(n // 2), -np.ones(n - n // 2)))
    ).astype(np.complex128)
    mixed = np.zeros((n, n), dtype=np.complex128)
    heterogeneous = np.stack((pure, mixed))
    result = model.run_markov_circuit(
        cycles=0,
        samples=2,
        init_mode="maxmix",
        G_init=heterogeneous,
        G_init_prepared=True,
        save=False,
        progress=False,
        return_data=True,
        state_representation="covariance",
        initial_purity_tolerance=RUNNER.RESUME_INITIAL_PURITY_TOLERANCE,
    )
    assert result["state_representation_resolved"] == "covariance"
    assert result["initial_purity_tolerance"] == RUNNER.RESUME_INITIAL_PURITY_TOLERANCE
    with pytest.raises(ValueError, match="heterogeneous pure/mixed"):
        model.run_markov_circuit(
            cycles=0,
            samples=2,
            init_mode="maxmix",
            G_init=heterogeneous,
            G_init_prepared=True,
            save=False,
            progress=False,
            return_data=True,
            state_representation="covariance",
        )


def test_runner_passes_the_locked_resume_tolerance_to_the_canonical_engine() -> None:
    captured = {}

    class Model:
        def run_markov_circuit(self, **kwargs):
            captured.update(kwargs)
            return {
                "state_representation_resolved": "covariance",
                "G_init_prepared": True,
                "exterior_preparation_performed": False,
                "G_final": kwargs["G_init"],
            }

    class Progress:
        def update(self, _count):
            pass

    task = _TinyTask()
    G = np.zeros((task.samples, 8, 8), dtype=np.complex128)
    returned = RUNNER.run_segment(
        Model(),
        RUNNER.expected_config(),
        task,
        object(),
        completed_cycle=10,
        segment_cycles=0,
        G_init=G,
        continuing=True,
        progress_bar=Progress(),
    )
    assert np.array_equal(returned, G)
    assert captured["state_representation"] == "covariance"
    assert captured["initial_purity_tolerance"] == 0.50000001


@dataclass(frozen=True)
class _TinyTask:
    construction: str = "hard"
    ny: int = 2
    execution_index: int = 0
    sample_start: int = 0
    sample_stop: int = 2
    seed: int = 17

    @property
    def nx(self):
        return 2

    @property
    def cycles(self):
        return 10

    @property
    def samples(self):
        return 2

    @property
    def sample_indices(self):
        return np.arange(2, dtype=np.int64)

    @property
    def task_id(self):
        return "hard_Ny002_exec000_samples000-001"


def _filled_tiny_observer() -> object:
    observer = OBSERVER.PurificationObserver(
        nx=2,
        ny=2,
        cycles=10,
        sample_indices=np.arange(2),
        sample_chunk=2,
        construction="hard",
    )
    G = torch.zeros((2, 8, 8), dtype=torch.complex128)
    for cycle in range(11):
        if cycle:
            for _ in range(observer.expected_sites_per_cycle):
                observer.record_event(
                    cycle=cycle,
                    sample_offsets=torch.arange(2),
                    conditional_log_probability=torch.full(
                        (2, 4), -0.125 * cycle, dtype=torch.float64
                    ),
                )
        observer.observe(cycle=cycle, G=G)
    return observer


@pytest.mark.parametrize(
    ("construction", "expected_sites", "expected_modes", "origin"),
    (
        ("hard", 12, 24, "after_born_conditioned_exterior_preparation"),
        ("soft", 16, 32, "global_maxmix_cycle_zero"),
    ),
)
def test_record_probability_is_cycle_resolved_float64_and_complete(
    construction: str,
    expected_sites: int,
    expected_modes: int,
    origin: str,
) -> None:
    observer = OBSERVER.PurificationObserver(
        nx=4,
        ny=4,
        cycles=2,
        sample_indices=np.arange(2),
        sample_chunk=2,
        construction=construction,
    )
    G = torch.zeros((2, 32, 32), dtype=torch.complex128)
    observer.observe(cycle=0, G=G)
    for cycle, value in ((1, -0.25), (2, -0.5)):
        for _ in range(expected_sites):
            observer.record_event(
                cycle=cycle,
                sample_offsets=torch.arange(2),
                conditional_log_probability=torch.full(
                    (2, 4), value, dtype=torch.float64
                ),
            )
        observer.observe(cycle=cycle, G=G)
    observer.validate(completed_cycle=2, final=True)
    expected_increment = np.asarray(
        [0.0, expected_sites * 4 * -0.25, expected_sites * 4 * -0.5]
    )
    np.testing.assert_array_equal(
        observer.measurement_log_probability,
        np.broadcast_to(expected_increment, (2, 3)),
    )
    np.testing.assert_array_equal(
        observer.cumulative_log_probability,
        np.broadcast_to(np.cumsum(expected_increment), (2, 3)),
    )
    assert observer.measurement_log_probability.dtype == np.float64
    assert observer.cumulative_log_probability.dtype == np.float64
    assert np.all(observer.site_event_count[:, 1:] == expected_sites)
    assert np.all(observer.channel_event_count[:, 1:] == 4 * expected_sites)
    payload = observer.result_payload(slice(None))
    assert int(payload["transfer_mode_count"]) == expected_modes
    assert str(payload["log_probability_origin"]) == origin
    assert str(payload["log_probability_dtype"]) == "float64"


def test_segmented_record_probability_matches_uninterrupted_engine() -> None:
    samples, cycles = 2, 4

    def collect(target: np.ndarray, offset: int = 0):
        def callback(*, cycle, sample_offsets, conditional_log_probability, **_):
            rows = sample_offsets.detach().cpu().numpy()
            target[rows, offset + int(cycle)] += (
                conditional_log_probability.sum(dim=1).detach().cpu().numpy()
            )

        return callback

    torch.manual_seed(9341)
    np.random.seed(9341)
    baseline = _engine_call(_gpu_model(), cycles=cycles)
    baseline_torch_rng = torch.get_rng_state().clone()

    torch.manual_seed(9341)
    np.random.seed(9341)
    continuous_logp = np.zeros((samples, cycles + 1), dtype=np.float64)
    continuous = _engine_call(
        _gpu_model(),
        cycles=cycles,
        record_callback=collect(continuous_logp),
    )
    assert np.array_equal(baseline["G_final"], continuous["G_final"])
    assert torch.equal(baseline_torch_rng, torch.get_rng_state())

    torch.manual_seed(9341)
    np.random.seed(9341)
    segmented_logp = np.zeros_like(continuous_logp)
    first = _engine_call(
        _gpu_model(),
        cycles=2,
        record_callback=collect(segmented_logp),
    )
    saved_torch = torch.get_rng_state().clone()
    saved_numpy = np.random.get_state()
    continuation_model = _gpu_model()
    torch.set_rng_state(saved_torch)
    np.random.set_state(saved_numpy)
    second = _engine_call(
        continuation_model,
        cycles=2,
        G_init=first["G_final"],
        prepared=True,
        record_callback=collect(segmented_logp, offset=2),
    )
    assert np.array_equal(continuous["G_final"], second["G_final"])
    np.testing.assert_array_equal(continuous_logp, segmented_logp)
    np.testing.assert_array_equal(
        np.cumsum(continuous_logp, axis=1),
        np.cumsum(segmented_logp, axis=1),
    )


def test_checkpoint_and_result_publication_resume_and_checksum_rejection(tmp_path: Path) -> None:
    task = _TinyTask()
    observer = _filled_tiny_observer()
    G = np.zeros((2, 8, 8), dtype=np.complex128)
    output = tmp_path / "drive"
    scratch = tmp_path / "scratch"
    hashes = {"source": "abc"}
    RUNNER.save_checkpoint(
        output,
        scratch,
        task,
        completed_cycle=10,
        elapsed_seconds=2.5,
        G=G,
        observer=observer,
        cfg_hash="cfg",
        hashes=hashes,
    )
    restored, reason = RUNNER.load_checkpoint(
        output, task, cfg_hash="cfg", hashes=hashes
    )
    assert reason == "verified"
    assert restored is not None and restored.completed_cycle == 10
    assert np.array_equal(restored.G, G)
    restored_observer = OBSERVER.PurificationObserver(
        nx=2,
        ny=2,
        cycles=10,
        sample_indices=np.arange(2),
        sample_chunk=2,
        construction="hard",
    )
    restored_observer.restore_checkpoint(restored.observer_payload, completed_cycle=10)
    np.testing.assert_array_equal(
        restored_observer.cumulative_log_probability,
        observer.cumulative_log_probability,
    )

    shard = RUNNER.ResultShard(task, shard_index=0, sample_start=0, sample_stop=2)
    RUNNER.publish_result(
        output,
        scratch,
        shard,
        G_final=G,
        observer=observer,
        elapsed_seconds=2.5,
        cfg_hash="cfg",
        hashes=hashes,
    )
    assert RUNNER.verified_complete(output, shard, cfg_hash="cfg", hashes=hashes) == (
        True,
        "verified",
    )
    result_path, _ = RUNNER.result_paths(output, shard)
    with np.load(result_path, allow_pickle=False) as result:
        assert str(result["result_schema"]) == "maxmix_purification_result_v3"
        assert str(result["observer_schema"]) == "maxmix_purification_observer_v2"
        assert result["measurement_log_probability"].shape == (2, 11)
        assert result["cumulative_log_probability"].shape == (2, 11)
        assert str(result["log_probability_origin"]) == (
            "after_born_conditioned_exterior_preparation"
        )
    with result_path.open("ab") as handle:
        handle.write(b"corrupt")
    valid, reason = RUNNER.verified_complete(output, shard, cfg_hash="cfg", hashes=hashes)
    assert valid is False and "checksum" in reason


def test_exact_v2_checkpoint_is_imported_without_mutating_its_files(tmp_path: Path) -> None:
    task = _TinyTask()
    observer = _filled_tiny_observer()
    G = np.zeros((2, 8, 8), dtype=np.complex128)
    legacy = tmp_path / "v2"
    output = tmp_path / "v3"
    scratch = tmp_path / "scratch"
    old_npz, old_json = RUNNER.checkpoint_paths(legacy, task)
    rng = RUNNER.capture_rng()
    RUNNER.save_npz(
        old_npz,
        {
            "checkpoint_schema": np.asarray(RUNNER.LEGACY_V2_CHECKPOINT_SCHEMA),
            "completed_cycle": np.asarray(10, dtype=np.int64),
            "elapsed_seconds": np.asarray(7.5, dtype=np.float64),
            "sample_indices": task.sample_indices,
            "G": G,
            **rng,
            **observer.checkpoint_payload(),
        },
    )
    old_sha = RUNNER.sha256_file(old_npz)
    RUNNER.write_json(
        old_json,
        {
            **RUNNER._legacy_v2_checkpoint_identity(task),
            "completed_cycle": 10,
            "checkpoint_filename": old_npz.name,
            "checkpoint_bytes": old_npz.stat().st_size,
            "checkpoint_sha256": old_sha,
        },
    )
    old_json_bytes = old_json.read_bytes()
    restored_observer = OBSERVER.PurificationObserver(
        nx=2,
        ny=2,
        cycles=10,
        sample_indices=np.arange(2),
        sample_chunk=2,
        construction="hard",
    )
    imported, reason = RUNNER.import_legacy_v2_checkpoint(
        legacy,
        output,
        scratch,
        task,
        restored_observer,
        cfg_hash="v3-config",
        hashes={"current": "source"},
    )
    assert imported is not None and imported.completed_cycle == 10
    assert reason.startswith("imported exact v2 cycle 10")
    assert RUNNER.sha256_file(old_npz) == old_sha
    assert old_json.read_bytes() == old_json_bytes
    new_npz, new_json = RUNNER.checkpoint_paths(output, task)
    assert new_npz.is_file() and new_json.is_file()
    metadata = json.loads(new_json.read_text(encoding="utf-8"))
    assert metadata["schema"] == RUNNER.CHECKPOINT_SCHEMA
    assert metadata["migrated_from"]["checkpoint_sha256"] == old_sha
    assert metadata["migrated_from"]["configuration_hash"] == (
        RUNNER.LEGACY_V2_CONFIG_HASH
    )


def test_v2_checkpoint_import_fails_closed_on_identity_drift(tmp_path: Path) -> None:
    task = _TinyTask()
    legacy = tmp_path / "v2"
    old_npz, old_json = RUNNER.checkpoint_paths(legacy, task)
    old_npz.parent.mkdir(parents=True, exist_ok=True)
    old_npz.write_bytes(b"not-used-after-identity-rejection")
    RUNNER.write_json(
        old_json,
        {
            **RUNNER._legacy_v2_checkpoint_identity(task),
            "configuration_hash": "wrong",
            "completed_cycle": 10,
            "checkpoint_filename": old_npz.name,
            "checkpoint_bytes": old_npz.stat().st_size,
            "checkpoint_sha256": RUNNER.sha256_file(old_npz),
        },
    )
    with pytest.raises(RuntimeError, match="identity mismatch.*configuration_hash"):
        RUNNER.load_legacy_v2_checkpoint(legacy, task)


def test_notebooks_expose_exact_config_local_staging_progress_and_disconnect() -> None:
    for filename, construction in (
        ("run_hard_wall_purification.ipynb", "hard"),
        ("run_soft_wall_purification.ipynb", "soft"),
    ):
        path = BUNDLE / filename
        config, config_source = _notebook_config(path)
        assert config == RUNNER.expected_config()
        assert f"CONSTRUCTION = '{construction}'" in config_source
        text = path.read_text(encoding="utf-8")
        assert "drive.mount('/content/drive')" in text
        assert "shutil.copytree(BUNDLE_DRIVE_DIR, LOCAL_BUNDLE_DIR)" in text
        assert "subprocess.Popen" in text and "os.read" in text
        assert "codecs.getincrementaldecoder('utf-8')" in text
        assert "sys.stdout.write(decoder.decode(chunk))" in text
        assert "sys.stdout.buffer" not in text
        assert "REPORT_ONLY = False" in text
        assert "MAX_NEW_EXECUTION_BATCHES = None" in text
        assert "IMPORT_VERIFIED_V2_CHECKPOINTS = True" in text
        assert "--import-v2-checkpoints-from" in text
        assert "maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v3" in text
        assert "A100" in text and "complex128" in text
        assert "runtime.unassign()" in text and "print('done')" in text


def test_bundle_registration_and_canonical_gpu_sources() -> None:
    layout_path = BUNDLE.parent / "bundle_layout.py"
    spec = importlib.util.spec_from_file_location("tested_layout_with_purification", layout_path)
    assert spec is not None and spec.loader is not None
    layout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layout)
    assert "07_maxmix_hard_soft_purification" in layout.NEW_DESIGN_BUNDLES
    assert layout.validate_bundle_layout(BUNDLE.parent) == layout.NEW_DESIGN_BUNDLES
    assert _sha256(BUNDLE / "src/classA_U1FGTN_gpu.py") == _sha256(
        REPO / "src/fgtn/classA_U1FGTN_gpu.py"
    )
    assert _sha256(BUNDLE / "src/occupied_frame_gpu.py") == _sha256(
        REPO / "src/fgtn/occupied_frame_gpu.py"
    )
    manifest = json.loads((BUNDLE / "deployment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sampling_revision"] == RUNNER.SAMPLING_REVISION
    assert manifest["configuration_sha256"] == RUNNER.config_hash(RUNNER.expected_config())
    for relative, expected in manifest["files"].items():
        path = BUNDLE / relative
        assert path.stat().st_size == expected["bytes"]
        assert _sha256(path) == expected["sha256"]
