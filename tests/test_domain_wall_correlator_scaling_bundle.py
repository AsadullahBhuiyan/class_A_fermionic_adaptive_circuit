from __future__ import annotations

import ast
import codecs
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tqdm.auto import tqdm


REPO = Path(__file__).resolve().parents[1]
PARENT = REPO / "00_WORKSPACE/CURRENT/final_production_new_designs"
BUNDLE = PARENT / "08_domain_wall_correlator_scaling"
NOTEBOOKS = {
    construction: BUNDLE / f"run_{construction}_wall_correlator.ipynb"
    for construction in ("hard", "soft")
}
LEGACY_OBSERVER = (
    REPO
    / "00_WORKSPACE/COLAB/colab_charge_fluctuations/src/streaming_covariance_observables_gpu.py"
)
REVISION = (
    "domain_wall_correlator_nx20_ny24-32_a1-1-3_"
    "nsh1-2-dense_s100_2ny_raster_v1"
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
OBSERVER = _load(BUNDLE / "correlator_observer.py", "tested_dw_correlator_observer")
RUNNER = _load(BUNDLE / "run_campaign.py", "tested_dw_correlator_runner")
LEGACY = _load(LEGACY_OBSERVER, "tested_legacy_streaming_correlator")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _notebook_config(path: Path) -> tuple[dict, str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
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
        and any(
            isinstance(target, ast.Name) and target.id == "CONFIG"
            for target in node.targets
        )
    )
    return ast.literal_eval(assignment.value), source


def test_locked_grid_has_36_configurations_144_tasks_and_3600_slots() -> None:
    hard_config, _ = _notebook_config(NOTEBOOKS["hard"])
    soft_config, _ = _notebook_config(NOTEBOOKS["soft"])
    assert hard_config == soft_config
    assert RUNNER.validate_config(hard_config) == hard_config
    assert hard_config["sampling_revision"] == REVISION
    assert hard_config["root_seed"] == 2026090408
    assert hard_config["Nx"] == 20
    assert hard_config["Ny_values"] == [24, 28, 32]
    assert hard_config["alpha_1_values"] == [1.0, 3.0]
    assert hard_config["alpha_2"] == 30.0
    assert hard_config["nshell_values"] == [1, 2, None]
    assert hard_config["samples_per_case"] == 100
    assert hard_config["batch_size"] == 25
    assert hard_config["cycles_multiplier"] == 2
    assert hard_config["segment_cycles"] == 16
    assert hard_config["sequence"] == "raster_y"
    assert hard_config["perfect_correction"] is True
    assert hard_config["init_mode"] == "default"
    assert hard_config["dtype"] == "complex128"
    assert hard_config["backend_by_nshell"] == {
        "1": "local",
        "2": "local",
        "dense": "dense",
    }
    assert hard_config["state_representation"] == "physical_frame"
    assert hard_config["triv_region_local_mode"] is False
    assert hard_config["frame_reorthonormalize_interval"] == 1
    assert hard_config["constructions"] == {
        "hard": {"DW": True, "dw_truncation": True, "meas_slab_only": True},
        "soft": {"DW": True, "dw_truncation": False, "meas_slab_only": False},
    }

    all_tasks = []
    cases = set()
    slots = set()
    for construction in ("hard", "soft"):
        lane = RUNNER.expand_tasks(hard_config, construction)
        assert len(lane) == 72
        all_tasks.extend(lane)
        for task in lane:
            cases.add((construction, task.ny, task.alpha_1, task.nshell))
            assert task.cycles == 2 * task.ny
            assert task.sample_count == 25
            slots.update(
                (construction, task.ny, task.alpha_1, task.nshell, index)
                for index in task.global_sample_indices
            )
    assert len(cases) == 36
    assert len(all_tasks) == 144
    assert len(slots) == 3600
    assert len({task.task_id for task in all_tasks}) == 144
    assert len({task.seed for task in all_tasks}) == 144


def test_frame_native_x_average_exactly_matches_legacy_dense_estimator() -> None:
    torch.manual_seed(4008)
    batch, nx, ny, capacity = 3, 4, 6, 24
    dimension = 2 * nx * ny
    random = torch.randn(batch, dimension, capacity, dtype=torch.complex128)
    frame = torch.linalg.qr(random).Q
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
    pairs = LEGACY.build_square_correlator_pair_indices(
        nx=nx, ny=ny, device="cpu"
    )
    legacy = LEGACY.xavg_square_correlator_batch_torch(
        covariance, pairs, nx=nx, ny=ny
    )
    torch.testing.assert_close(x_resolved.mean(dim=1), legacy, rtol=0, atol=2e-16)


def test_observer_records_charge_without_changing_state_or_rng() -> None:
    torch.manual_seed(81)
    nx, ny, samples, rank = 3, 4, 2, 12
    frame = torch.linalg.qr(
        torch.randn(samples, 2 * nx * ny, rank, dtype=torch.complex128)
    ).Q
    ranks = torch.full((samples,), rank, dtype=torch.long)
    state = SimpleNamespace(frame=frame, ranks=ranks)
    observer = OBSERVER.DomainWallCorrelatorObserver(
        nx=nx,
        ny=ny,
        physical_cycles=2,
        sample_ids=np.arange(samples),
    )
    frame_before = frame.clone()
    ranks_before = ranks.clone()
    rng_before = torch.get_rng_state().clone()
    observer(
        cycle=0,
        state=state,
        batch_index=0,
        batch_start=0,
        batch_count=samples,
    )
    assert torch.equal(frame, frame_before)
    assert torch.equal(ranks, ranks_before)
    assert torch.equal(torch.get_rng_state(), rng_before)
    np.testing.assert_array_equal(observer.global_charge[:, 0], rank)
    assert observer.seen.tolist() == [True, False, False]


def _cpu_model(*, hard: bool):
    return RUNNER.classA_U1FGTN_gpu(
        Nx=4,
        Ny=4,
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


def test_hard_wall_disk_checkpoint_resume_is_bitwise_equivalent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _notebook_config(NOTEBOOKS["hard"])
    monkeypatch.setattr(RUNNER, "EXPECTED_NX", 4)
    task = RUNNER.Task(
        construction="hard",
        ny=4,
        alpha_1=1.0,
        nshell=1,
        batch_index=0,
        sample_start=0,
        sample_stop=2,
        seed=7351,
    )

    monkeypatch.setattr(RUNNER, "EXPECTED_SEGMENT_CYCLES", task.cycles)
    RUNNER._seed_rng(task.seed)
    continuous_observer = OBSERVER.DomainWallCorrelatorObserver(
        nx=4,
        ny=4,
        physical_cycles=task.cycles,
        sample_ids=np.asarray(task.global_sample_indices),
    )
    with tqdm(total=task.cycles, disable=True) as bar:
        continuous_native, _ = RUNNER._run_segment(
            config=config,
            model=_cpu_model(hard=True),
            task=task,
            observer=continuous_observer,
            segment_start=0,
            frame=None,
            ranks=None,
            cycle_bar=bar,
        )
    continuous_rng = RUNNER._capture_rng_state()
    continuous_frame, continuous_ranks = RUNNER._native_arrays(continuous_native)

    monkeypatch.setattr(RUNNER, "EXPECTED_SEGMENT_CYCLES", 2)
    RUNNER._seed_rng(task.seed)
    resumed_observer = OBSERVER.DomainWallCorrelatorObserver(
        nx=4,
        ny=4,
        physical_cycles=task.cycles,
        sample_ids=np.asarray(task.global_sample_indices),
    )
    native = None
    frame = ranks = None
    elapsed = 0.0
    with tqdm(total=task.cycles, disable=True) as bar:
        for completed in (0, 2):
            native, segment_elapsed = RUNNER._run_segment(
                config=config,
                model=_cpu_model(hard=True),
                task=task,
                observer=resumed_observer,
                segment_start=completed,
                frame=frame,
                ranks=ranks,
                cycle_bar=bar,
            )
            elapsed += segment_elapsed
            frame, ranks = RUNNER._native_arrays(native)
            rng_payload = RUNNER.save_checkpoint(
                output_root=tmp_path / "drive",
                scratch_root=tmp_path / "scratch",
                task=task,
                completed_cycle=completed + 2,
                elapsed_seconds=elapsed,
                native=native,
                observer=resumed_observer,
                configuration_sha256="config",
                hashes={"source": "hash"},
            )
            if completed == 0:
                RUNNER._restore_rng_state(rng_payload)

    checkpoint, reason = RUNNER.load_checkpoint(
        output_root=tmp_path / "drive",
        task=task,
        configuration_sha256="config",
        hashes={"source": "hash"},
    )
    assert reason == "verified" and checkpoint is not None
    restored_observer = OBSERVER.DomainWallCorrelatorObserver(
        nx=4,
        ny=4,
        physical_cycles=task.cycles,
        sample_ids=np.asarray(task.global_sample_indices),
    )
    restored_observer.restore_checkpoint(
        checkpoint.observer_payload, completed_cycle=checkpoint.completed_cycle
    )
    RUNNER._restore_rng_state(checkpoint.rng_payload)
    frame, ranks = checkpoint.frame, checkpoint.ranks
    with tqdm(total=task.cycles, initial=4, disable=True) as bar:
        for completed in (4, 6):
            native, _ = RUNNER._run_segment(
                config=config,
                model=_cpu_model(hard=True),
                task=task,
                observer=restored_observer,
                segment_start=completed,
                frame=frame,
                ranks=ranks,
                cycle_bar=bar,
            )
            frame, ranks = RUNNER._native_arrays(native)
            if completed == 4:
                rng = RUNNER._capture_rng_state()
                RUNNER._restore_rng_state(rng)

    resumed_rng = RUNNER._capture_rng_state()
    assert np.array_equal(frame, continuous_frame)
    assert np.array_equal(ranks, continuous_ranks)
    assert np.array_equal(
        restored_observer.x_resolved, continuous_observer.x_resolved
    )
    assert np.array_equal(
        restored_observer.global_charge, continuous_observer.global_charge
    )
    for key in continuous_rng:
        assert np.array_equal(resumed_rng[key], continuous_rng[key])


def test_checkpoint_identity_mismatch_and_atomic_publish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "local.bin"
    source.write_bytes(b"new durable bytes")
    final = tmp_path / "drive" / "result.bin"
    final.parent.mkdir()
    final.write_bytes(b"old verified bytes")

    real_copy = RUNNER.shutil.copyfile

    def corrupt_copy(_source: Path, destination: Path) -> None:
        Path(destination).write_bytes(b"corrupt")

    monkeypatch.setattr(RUNNER.shutil, "copyfile", corrupt_copy)
    with pytest.raises(OSError, match="mismatch"):
        RUNNER.publish_file(source, final)
    assert final.read_bytes() == b"old verified bytes"
    monkeypatch.setattr(RUNNER.shutil, "copyfile", real_copy)

    monkeypatch.setattr(RUNNER, "EXPECTED_NX", 2)
    monkeypatch.setattr(RUNNER, "EXPECTED_SEGMENT_CYCLES", 2)
    task = RUNNER.Task("soft", 2, 1.0, 1, 0, 0, 1, 9)
    observer = OBSERVER.DomainWallCorrelatorObserver(
        nx=2, ny=2, physical_cycles=4, sample_ids=np.asarray([0])
    )
    observer.seen[:3] = True
    observer.x_resolved[:, :3] = 0.0
    observer.global_charge[:, :3] = 4
    frame = np.eye(8, 4, dtype=np.complex128)[None]
    native = {"frame": frame, "ranks": np.asarray([4], dtype=np.int64)}
    RUNNER._seed_rng(9)
    RUNNER.save_checkpoint(
        output_root=tmp_path / "checkpoint_drive",
        scratch_root=tmp_path / "checkpoint_scratch",
        task=task,
        completed_cycle=2,
        elapsed_seconds=1.0,
        native=native,
        observer=observer,
        configuration_sha256="correct",
        hashes={"source": "hash"},
    )
    checkpoint, reason = RUNNER.load_checkpoint(
        output_root=tmp_path / "checkpoint_drive",
        task=task,
        configuration_sha256="wrong",
        hashes={"source": "hash"},
    )
    assert checkpoint is None
    assert "identity mismatch" in reason


def test_completion_pairs_are_verified_and_checkpoint_cleanup_is_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(RUNNER, "EXPECTED_NX", 2)
    task = RUNNER.Task("soft", 2, 1.0, 1, 0, 0, 1, 19)
    observer = OBSERVER.DomainWallCorrelatorObserver(
        nx=2, ny=2, physical_cycles=4, sample_ids=np.asarray([0])
    )
    observer.seen[:] = True
    observer.x_resolved[:] = 0.125
    observer.global_charge[:] = 4
    output_root = tmp_path / "drive"
    scratch_root = tmp_path / "scratch"
    identity = {"configuration_sha256": "config", "hashes": {"source": "hash"}}
    RUNNER.save_result(
        output_root=output_root,
        scratch_root=scratch_root,
        task=task,
        observer=observer,
        elapsed_seconds=2.0,
        **identity,
    )
    assert RUNNER.verified_complete(
        output_root=output_root, task=task, **identity
    ) == (True, "verified")

    result_path, completion_path = RUNNER.result_paths(output_root, task)
    completion_bytes = completion_path.read_bytes()
    completion_path.unlink()
    valid, reason = RUNNER.verified_complete(
        output_root=output_root, task=task, **identity
    )
    assert not valid and "incomplete" in reason
    completion_path.write_bytes(completion_bytes)
    corrupted = bytearray(result_path.read_bytes())
    corrupted[-1] ^= 1
    result_path.write_bytes(corrupted)
    valid, reason = RUNNER.verified_complete(
        output_root=output_root, task=task, **identity
    )
    assert not valid and "checksum mismatch" in reason

    RUNNER.save_result(
        output_root=output_root,
        scratch_root=scratch_root,
        task=task,
        observer=observer,
        elapsed_seconds=2.0,
        **identity,
    )
    checkpoint_npz, checkpoint_json = RUNNER.checkpoint_paths(output_root, task)
    checkpoint_npz.parent.mkdir(parents=True)
    checkpoint_npz.write_bytes(b"checkpoint")
    checkpoint_json.write_text("{}\n", encoding="utf-8")
    assert RUNNER.verified_complete(
        output_root=output_root, task=task, **identity
    ) == (True, "verified")
    RUNNER.remove_checkpoint(output_root, task)
    assert not checkpoint_npz.exists() and not checkpoint_json.exists()


def test_notebooks_are_canonical_visible_and_stream_nested_progress() -> None:
    for construction, path in NOTEBOOKS.items():
        notebook = json.loads(path.read_text(encoding="utf-8"))
        joined = "\n".join(_cell_source(cell) for cell in notebook["cells"])
        assert "drive.mount('/content/drive')" in joined
        assert "shutil.copytree(BUNDLE_DRIVE_DIR, LOCAL_BUNDLE_DIR)" in joined
        assert "stdout=subprocess.PIPE" in joined
        assert "os.read(process.stdout.fileno(), 8192)" in joined
        assert "codecs.getincrementaldecoder('utf-8')" in joined
        assert "sys.stdout.write(text)" in joined
        assert "sys.stdout.buffer" not in joined
        assert "REPORT_ONLY = False" in joined
        assert "MAX_NEW_TASKS = None" in joined
        assert "A100" in joined and "complex128" in joined
        assert "'sequence': 'raster_y'" in joined
        assert f"CONSTRUCTION = '{construction}'" in joined
        code_cells = [
            _cell_source(cell)
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        assert code_cells[-1].strip() == (
            "from google.colab import runtime\n"
            "runtime.unassign()\n"
            "print('done')"
        )


def test_notebook_streamer_supports_colab_outstream_without_buffer() -> None:
    launch_source = next(
        _cell_source(cell)
        for cell in json.loads(NOTEBOOKS["hard"].read_text())["cells"]
        if "def _stream_process_output" in _cell_source(cell)
    )
    tree = ast.parse(launch_source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_stream_process_output"
    )

    class OutStream:
        def __init__(self) -> None:
            self.text = ""
            self.flushes = 0

        def write(self, value: str) -> None:
            self.text += value

        def flush(self) -> None:
            self.flushes += 1

    chunks = iter([b"task 1 ", b"\xe2\x9c", b"\x93\r", b"cycle 2\n", b""])
    output = OutStream()
    namespace = {
        "codecs": codecs,
        "os": SimpleNamespace(read=lambda _fd, _size: next(chunks)),
        "sys": SimpleNamespace(stdout=output),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "streamer", "exec"), namespace)
    process = SimpleNamespace(
        stdout=SimpleNamespace(fileno=lambda: 3), wait=lambda: 0
    )
    assert namespace["_stream_process_output"](process) == 0
    assert output.text == "task 1 ✓\rcycle 2\n"
    assert output.flushes >= 1
def test_bundle_registration_and_canonical_sources() -> None:
    layout = _load(PARENT / "bundle_layout.py", "tested_dw_bundle_layout")
    assert BUNDLE.name in layout.NEW_DESIGN_BUNDLES
    assert layout.validate_bundle_layout(PARENT) == layout.NEW_DESIGN_BUNDLES
    index = json.loads((PARENT / "bundle_index.json").read_text(encoding="utf-8"))
    assert index["standalone_contracts"][BUNDLE.name] == REVISION
    assert _sha256(BUNDLE / "src/classA_U1FGTN_gpu.py") == _sha256(
        REPO / "src/fgtn/classA_U1FGTN_gpu.py"
    )
    assert _sha256(BUNDLE / "src/occupied_frame_gpu.py") == _sha256(
        REPO / "src/fgtn/occupied_frame_gpu.py"
    )
