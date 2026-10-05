from __future__ import annotations

import hashlib
import ast
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
BUNDLE = PARENT / "15_hard_wall_alpha3_xresolved_correlator"
NOTEBOOK = BUNDLE / "run_hard_wall_alpha3_correlator.ipynb"
LEGACY_OBSERVER = (
    REPO
    / "00_WORKSPACE/COLAB/colab_charge_fluctuations/src/streaming_covariance_observables_gpu.py"
)
REVISION = "hard_wall_xresolved_nx20_ny60_a1-3_nsh1_s100_2ny_raster_endpoint_v1"


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
    BUNDLE / "alpha3_correlator.py", "tested_alpha3_ny60_xresolved_observer"
)
RUNNER = _load(BUNDLE / "run_campaign.py", "tested_alpha3_ny60_xresolved_runner")
LEGACY = _load(LEGACY_OBSERVER, "tested_alpha3_ny60_legacy_correlator")


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
        alpha_1=3.0,
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


def test_locked_hard_only_contract_and_exact_workload():
    config = RUNNER.expected_config()
    tasks = RUNNER.expand_execution_batches(config)
    shards = RUNNER.all_result_shards(tasks)
    assert config['sampling_revision'] == REVISION
    assert config['root_seed'] == 2026091401
    assert config['Nx'] == 20 and config['Ny_values'] == [60]
    assert [(t.ny,t.sample_count,t.cycles) for t in tasks] == [(60,40,120),(60,40,120),(60,20,120)]
    assert len({t.seed for t in tasks}) == 3
    assert len(shards) == 20 and all(s.sample_count == 5 for s in shards)
    assert sorted(i for s in shards for i in s.global_sample_indices) == list(range(100))
    p = config['protocol']
    assert p['alpha_1'] == 3 and p['alpha_2'] == 30 and p['nshell'] == 1
    assert p['DW'] and p['dw_truncation'] and p['meas_slab_only']
    assert p['perfect_correction'] and not p['postselect']
    assert p['sequence'] == 'raster_y' and p['init_mode'] == 'default'
    assert config['dtype'] == 'complex128' and config['gpu_memory_hard_limit_gib'] == 30
    assert config['segment_cycles'] == 5
    assert not any(config['observables'].get(k) for k in ('occupied_frame','half_system_covariance','half_system_occupation_spectrum'))
    for key,value in [('alpha_1',1),('dw_truncation',False),('meas_slab_only',False),('perfect_correction',False)]:
        bad=RUNNER.expected_config();bad['protocol'][key]=value
        with pytest.raises(ValueError): RUNNER.validate_config(bad)


def test_compact_payload_and_observer_does_not_change_state_or_rng():
    torch.manual_seed(1591)
    frame=torch.linalg.qr(torch.randn(5,48,24,dtype=torch.complex128)).Q
    ranks=torch.full((5,),24,dtype=torch.long)
    original=frame.clone();rng=RUNNER._capture_rng_state()
    p=OBSERVER.endpoint_result_payload(frame=frame,ranks=ranks,nx=4,ny=6,global_sample_indices=np.arange(5))
    np.testing.assert_array_equal(frame,original)
    _rng_equal(rng,RUNNER._capture_rng_state())
    assert p['x_resolved_square_correlator'].shape==(5,1,4,4)
    np.testing.assert_array_equal(p['cycles'],[12])
    np.testing.assert_array_equal(p['normalized_cycles'],[2.])
    assert np.all(p['global_charge']==24) and np.all(p['half_filling_offset']==0)
    assert not {'occupied_frame','half_system_covariance','half_system_occupation_spectrum'} & p.keys()
    p['xavg_square_correlator_vs_ry']+=.01
    with pytest.raises(ValueError,match='x average'):OBSERVER.validate_endpoint_payload(p,nx=4,ny=6,global_sample_indices=np.arange(5))


def test_report_only_does_not_start_gpu_and_counts_pending(tmp_path,monkeypatch):
    def forbidden():raise AssertionError('report-only must not require GPU')
    monkeypatch.setattr(RUNNER,'validate_a100',forbidden)
    result=RUNNER.run_campaign(config=RUNNER.expected_config(),output_root=tmp_path/'out',
                              scratch_root=tmp_path/'scratch',report_only=True)
    assert result['status']=='report_only'
    assert result['total_trajectories']==100 and result['total_result_shards']==20
    assert result['pending_execution_batches']==3 and result['verified_result_shards']==0
    assert result['estimated_final_correlator_gib']<.01


def test_notebook_native_progress_visible_contract_and_source_sync():
    notebook=json.loads(NOTEBOOK.read_text())
    codes=[_cell_source(c) for c in notebook['cells'] if c['cell_type']=='code']
    for code in codes:compile(code,str(NOTEBOOK),'exec')
    tree=ast.parse(codes[0])
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFIG' for t in n.targets))
    assert ast.literal_eval(assignment.value)==RUNNER.expected_config()
    combined='\n'.join(codes)
    assert combined.count('drive.mount(')==1
    assert 'A100' in codes[0] and '38 * 1024**3' in codes[0]
    assert 'REPORT_ONLY' in codes[0] and 'MAX_NEW_EXECUTION_BATCHES' in codes[0]
    assert 'runner.main(runner_args)' in codes[1]
    assert 'sys.stdout.buffer' not in combined and 'subprocess.run' not in combined
    assert 'shutil.copyfile(source, target)' in codes[1] and "Path('/content')" in codes[0]
    assert codes[-1].strip()=="from google.colab import runtime\nruntime.unassign()\nprint('done')"
    for filename in ('classA_U1FGTN_gpu.py','occupied_frame_gpu.py'):
        assert (BUNDLE/'src'/filename).read_bytes()==(REPO/'src/fgtn'/filename).read_bytes()
    layout=_load(PARENT/'bundle_layout.py','alpha3_test_bundle_layout')
    assert BUNDLE.name in layout.validate_bundle_layout(PARENT)
    assert sum(RUNNER.estimated_result_payload_bytes(s) for s in RUNNER.all_result_shards(RUNNER.expand_execution_batches(RUNNER.expected_config())))<2*1024**2


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
            member.compress_type == zipfile.ZIP_DEFLATED for member in archive.infolist()
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
