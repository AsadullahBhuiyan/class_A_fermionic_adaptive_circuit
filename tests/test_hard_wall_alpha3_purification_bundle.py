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
    / "00_WORKSPACE/CURRENT/final_production_new_designs/20_hard_wall_alpha3_purification"
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


OBSERVER = _load("purification_observer.py", "tested_alpha3_purification_observer")
sys.modules["purification_observer"] = OBSERVER
RUNNER = _load("run_campaign.py", "tested_alpha3_purification_runner")


def test_contract_and_disjoint_batch_shard_expansion():
    config = RUNNER.expected_config()
    RUNNER.validate_config(config)
    assert config['alpha_1'] == 3 and config['alpha_2'] == 30
    assert config['Ny_values'] == [40] and config['Nx'] == 20
    assert config['perfect_correction'] and not config['postselect']
    assert config['init_mode'] == 'maxmix' and config['sequence'] == 'raster_y'
    assert list(config['constructions']) == ['hard']
    assert config['constructions']['hard']['meas_slab_only']
    tasks = RUNNER.expand_execution_batches(config, 'hard')
    assert [t.samples for t in tasks] == [40, 40, 20]
    assert all(t.cycles == 160 for t in tasks)
    assert len({t.seed for t in tasks}) == 3
    assert tasks == RUNNER.expand_execution_batches(config, 'hard')
    shards = RUNNER.all_result_shards(config, 'hard')
    assert len(shards) == 20 and all(len(s.sample_indices) == 5 for s in shards)
    assert np.concatenate([s.sample_indices for s in shards]).tolist() == list(range(100))
    with pytest.raises(ValueError):
        RUNNER.expand_execution_batches(config, 'soft')
    with pytest.raises(ValueError):
        RUNNER.validate_config({**config, 'alpha_1': 1})


def test_model_receives_alpha3_and_hardwall(monkeypatch):
    monkeypatch.setattr(RUNNER, 'classA_U1FGTN_gpu', lambda **kw: kw)
    model = RUNNER.build_model(RUNNER.expected_config(), 'hard', 40)
    assert model['alpha_1'] == 3 and model['dw_truncation'] is True
    assert model['dtype'] == 'complex128'


def test_notebook_and_manifest_and_sources():
    nb = json.loads((BUNDLE / 'run_hard_wall_alpha3_purification.ipynb').read_text())
    source = '\n'.join(''.join(c['source']) for c in nb['cells'])
    config, _ = _notebook_config(BUNDLE / 'run_hard_wall_alpha3_purification.ipynb')
    assert config == RUNNER.expected_config()
    for text in ('/content/', 'REPORT_ONLY', 'MAX_NEW_EXECUTION_BATCHES',
                 'runner.main(args)', 'A100', 'complex128'):
        assert text in source
    assert 'subprocess.run' not in source
    assert ''.join(nb['cells'][-1]['source']) == "from google.colab import runtime\nruntime.unassign()\nprint('done')\n"
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            ast.parse(''.join(c['source']))
    manifest = json.loads((BUNDLE / 'deployment_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        assert (BUNDLE / name).stat().st_size == expected['bytes']
        assert _sha256(BUNDLE / name) == expected['sha256']
    for name in ['classA_U1FGTN_gpu.py', 'occupied_frame_gpu.py']:
        assert _sha256(BUNDLE / 'src' / name) == _sha256(REPO / 'src/fgtn' / name)
    assert _sha256(BUNDLE / 'purification_observer.py') == _sha256(
        BUNDLE.parent / '07_maxmix_hard_soft_purification/purification_observer.py')
    layout = _load('../bundle_layout.py', 'alpha3_test_bundle_layout')
    assert BUNDLE.name in layout.validate_bundle_layout(BUNDLE.parent)


def test_memory_limit_and_monitor(monkeypatch):
    from types import SimpleNamespace
    config = RUNNER.expected_config()
    cuda = RUNNER.torch.cuda
    monkeypatch.setattr(cuda, 'is_available', lambda: True)
    monkeypatch.setattr(cuda, 'get_device_properties', lambda _: SimpleNamespace(total_memory=40_000_000_000))
    monkeypatch.setattr(cuda, 'get_device_name', lambda _: 'NVIDIA A100')
    monkeypatch.setattr(cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(cuda, 'memory_reserved', lambda _: 0)
    monkeypatch.setattr(cuda, 'mem_get_info', lambda _: (38_000_000_000, 40_000_000_000))
    limits = []
    monkeypatch.setattr(cuda, 'set_per_process_memory_fraction', lambda fraction, device: limits.append(fraction))
    RUNNER.require_runtime(config)
    assert limits == [35 / 40]
    monkeypatch.setattr(cuda, 'max_memory_reserved', lambda _: 34_000_000_000)
    assert RUNNER.check_gpu_memory(config)['device_used_GB'] == 2
    monkeypatch.setattr(cuda, 'mem_get_info', lambda _: (1_000_000_000, 40_000_000_000))
    with pytest.raises(RuntimeError, match='safety budget'):
        RUNNER.check_gpu_memory(config)


def test_report_only_does_not_construct_model(tmp_path, monkeypatch):
    monkeypatch.setattr(RUNNER, 'build_model', lambda *a: pytest.fail('report launched GPU'))
    result = RUNNER.run_campaign(RUNNER.expected_config(), construction='hard',
        output_root=tmp_path / 'output', scratch_root=tmp_path / 'scratch',
        report_only=True, max_new_execution_batches=None)
    assert result['pending_shards'] == 20 and result['execution_batches'] == 3


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
        alpha_1=3,
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
        initial_purity_tolerance=RUNNER.RESUME_INITIAL_PURITY_TOLERANCE,
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
        assert str(result["result_schema"]) == "alpha3_purification_result_v1"
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


def test_failed_drive_readback_preserves_stable_file(tmp_path, monkeypatch):
    local, final = tmp_path / 'local', tmp_path / 'drive' / 'result'
    local.write_bytes(b'new data')
    final.parent.mkdir()
    final.write_bytes(b'previous verified data')
    real_copy = RUNNER.shutil.copyfile
    def bad_copy(src, dst):
        real_copy(src, dst)
        Path(dst).write_bytes(b'broken')
    monkeypatch.setattr(RUNNER.shutil, 'copyfile', bad_copy)
    with pytest.raises(OSError, match='readback mismatch'):
        RUNNER.publish_file(local, final)
    assert final.read_bytes() == b'previous verified data'


def test_final_checkpoint_survives_publication_failure_and_resumes(tmp_path, monkeypatch):
    task = _TinyTask()
    observer = _filled_tiny_observer()
    output, scratch = tmp_path / 'drive', tmp_path / 'scratch'
    config = RUNNER.expected_config()
    config['device'] = 'cpu'
    config['observer_sample_chunk_by_Ny']['2'] = 2
    G = np.zeros((2, 8, 8), dtype=np.complex128)
    RUNNER.save_checkpoint(output, scratch, task, completed_cycle=10, elapsed_seconds=1,
        G=G, observer=observer, cfg_hash='cfg', hashes={})
    checkpoint_files = RUNNER.checkpoint_paths(output, task)
    checkpoint_hashes = [_sha256(p) for p in checkpoint_files]
    monkeypatch.setattr(RUNNER, 'build_model', lambda *args: None)
    monkeypatch.setattr(RUNNER, 'run_segment', lambda *a, **kw: pytest.fail('final checkpoint reran dynamics'))
    publish = RUNNER.publish_result
    def fail(*a, **kw):
        raise OSError('simulated archive/result write failure')
    monkeypatch.setattr(RUNNER, 'publish_result', fail)
    class Bar:
        def update(self, n): pass
    with pytest.raises(OSError, match='simulated'):
        RUNNER.execute_batch(config, output, scratch, task, cfg_hash='cfg', hashes={}, shard_bar=Bar())
    assert [_sha256(p) for p in checkpoint_files] == checkpoint_hashes
    monkeypatch.setattr(RUNNER, 'publish_result', publish)
    RUNNER.execute_batch(config, output, scratch, task, cfg_hash='cfg', hashes={}, shard_bar=Bar())
    assert all(not p.exists() for p in checkpoint_files)
    shard = RUNNER.result_shards(task)[0]
    assert RUNNER.verified_complete(output, shard, cfg_hash='cfg', hashes={})[0]
    result_path, receipt = RUNNER.result_paths(output, shard)
    receipt.unlink()
    assert not RUNNER.verified_complete(output, shard, cfg_hash='cfg', hashes={})[0]
    assert result_path.exists()


@pytest.mark.parametrize('boundary', [1, 2, 3, 4])
def test_alpha3_observer_state_and_rng_exact_through_serialized_checkpoint(tmp_path, monkeypatch, boundary):
    from types import SimpleNamespace
    monkeypatch.setattr(RUNNER, 'SEGMENT_CYCLES', 1)  # miniature validation analogue
    task = SimpleNamespace(nx=4, ny=4, samples=2, cycles=4, construction='hard',
        seed=432, task_id='tiny_alpha3', sample_indices=np.arange(2),
        sample_start=0, sample_stop=2)
    config = RUNNER.expected_config()
    config['device'] = 'cpu'
    def new_observer():
        return OBSERVER.PurificationObserver(nx=4, ny=4, cycles=4,
            sample_indices=np.arange(2), sample_chunk=2, construction='hard')
    class Bar:
        def update(self, n): pass
    def seed():
        np.random.seed(task.seed)
        torch.manual_seed(task.seed)
    seed()
    baseline = _engine_call(_gpu_model(), cycles=4)
    baseline_rng = RUNNER.capture_rng()
    seed()
    whole = new_observer()
    full_G = RUNNER.run_segment(_gpu_model(), config, task, whole,
        completed_cycle=0, segment_cycles=4, G_init=None, continuing=False, progress_bar=Bar())
    np.testing.assert_array_equal(full_G, baseline['G_final'])
    for k, value in baseline_rng.items():
        np.testing.assert_array_equal(RUNNER.capture_rng()[k], value)
    seed()
    prefix = new_observer()
    first_G = RUNNER.run_segment(_gpu_model(), config, task, prefix,
        completed_cycle=0, segment_cycles=boundary, G_init=None, continuing=False, progress_bar=Bar())
    RUNNER.save_checkpoint(tmp_path/'out', tmp_path/'scratch', task, completed_cycle=boundary,
        elapsed_seconds=1, G=first_G, observer=prefix, cfg_hash='cfg', hashes={})
    restored, reason = RUNNER.load_checkpoint(tmp_path/'out', task, cfg_hash='cfg', hashes={})
    assert reason == 'verified'
    resumed = new_observer()
    resumed.restore_checkpoint(restored.observer_payload, completed_cycle=boundary)
    model = _gpu_model()
    RUNNER.restore_rng(restored.rng_payload)
    final_G = restored.G if boundary == 4 else RUNNER.run_segment(model, config, task, resumed,
        completed_cycle=boundary, segment_cycles=4-boundary, G_init=restored.G,
        continuing=True, progress_bar=Bar())
    np.testing.assert_array_equal(final_G, full_G)
    for k in whole.ARRAY_NAMES:
        np.testing.assert_array_equal(getattr(resumed, k), getattr(whole, k))
    for k, value in baseline_rng.items():
        np.testing.assert_array_equal(RUNNER.capture_rng()[k], value)
