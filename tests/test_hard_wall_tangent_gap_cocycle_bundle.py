from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
PARENT = REPO / "00_WORKSPACE/CURRENT/final_production_new_designs"
BUNDLE = PARENT / "17_hard_wall_tangent_gap_cocycle"


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
RUNNER = _load(BUNDLE / "run_campaign.py", "tested_hard_wall_tangent_gap")
ANALYSIS = _load(BUNDLE / "analyze_campaign.py", "tested_hard_wall_tangent_analysis")


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_locked_grid_lane_partition_and_persistence() -> None:
    config = RUNNER.load_config()
    assert config == RUNNER.expected_config()
    assert config["alpha_1_sweep_values"] == list(RUNNER.ALPHA_SWEEP)
    assert config["batch_size_by_Ny"] == {
        "24": 25,
        "28": 20,
        "32": 15,
        "36": 10,
        "40": 10,
    }
    assert config["sampling_revision"].endswith("_v2")
    assert config["verified_v1_import"]["result_sha256"] == RUNNER.V1_RESULT_SHA256
    assert config["wall_construction"] == "hard_support_truncated"
    assert config["nshell"] == 1
    assert config["perfect_correction"] is True
    assert config["postselect"] is False
    assert config["dtype"] == "complex128"
    assert config["cycles_multiplier"] == 2
    assert config["maximum_peak_cuda_reserved_gib"] == 38.0

    tasks = RUNNER.expand_tasks(config)
    assert len(tasks) == 375
    assert sum(task.sample_count for task in tasks) == 6700
    assert len({(task.ny, task.alpha_1) for task in tasks}) == 67
    assert len({task.task_id for task in tasks}) == 375
    assert len({task.seed for task in tasks}) == 375
    assert {lane: len(RUNNER.tasks_for_lane(lane)) for lane in RUNNER.LANES} == {
        "A": 190,
        "B": 185,
    }
    assert set(RUNNER.tasks_for_lane("A")).isdisjoint(RUNNER.tasks_for_lane("B"))
    assert sum(task.sample_count for task in tasks if task.reuse_record) == 600
    imports = [task for task in tasks if task.import_v1]
    assert len(imports) == 1
    assert imports[0].case_sample_indices == tuple(range(25))
    assert imports[0].global_sample_indices == tuple(range(6600, 6625))
    assert max(task.sample_count for task in tasks if not task.import_v1) == 25
    assert all(task.sample_count <= RUNNER.BATCH_SIZE_BY_NY[task.ny] for task in tasks if not task.import_v1)
    assert {(task.ny, task.alpha_1) for task in tasks if task.save_cocycle} == {
        (40, 3.0),
        (40, 1.0),
    }
    assert all(RUNNER.tasks_for_lane(lane)[0].ny == 40 for lane in RUNNER.LANES)


def test_lane_partition_is_balanced_by_declared_parity() -> None:
    tasks = RUNNER.expand_tasks()
    cases = {(task.ny, task.alpha_1, task.alpha_index, task.lane) for task in tasks}
    for ny, _alpha, index, lane in cases:
        assert lane == RUNNER._lane_for(ny, index)
    assert RUNNER._lane_for(24, 0) == "A"
    assert RUNNER._lane_for(28, 0) == "B"
    assert RUNNER._lane_for(32, 0) == "A"
    assert RUNNER._lane_for(36, 0) == "B"
    assert RUNNER._lane_for(40, 1) == "A"


def test_scale_separated_endpoint_cocycle_and_gaps_are_exact() -> None:
    torch.manual_seed(1701)
    batch, full_dimension, active_dimension = 2, 8, 6
    basis_raw = torch.randn(batch, active_dimension, active_dimension, dtype=torch.complex128)
    basis, _ = torch.linalg.qr(basis_raw)
    frame_raw = torch.randn(batch, full_dimension, active_dimension, dtype=torch.complex128)
    frame, _ = torch.linalg.qr(frame_raw)
    core = torch.randn(batch, active_dimension, active_dimension, dtype=torch.complex128)
    core = core / torch.linalg.matrix_norm(core, ord="fro", dim=(-2, -1))[:, None, None]
    scale = torch.tensor([0.5, -0.25], dtype=torch.float64)
    capture = RUNNER.EndpointTangentCapture(final_cycle=80)
    capture.frame = frame
    capture.core = core
    capture.scale = scale
    task = RUNNER.Task(
        ny=40,
        alpha_1=1.0,
        alpha_index=1,
        case_index=66,
        batch_index=0,
        sample_start=0,
        sample_stop=2,
        seed=1,
        lane="A",
    )
    arrays = RUNNER.finalize_tangent(
        task,
        capture,
        basis,
        torch.arange(active_dimension),
        torch.tensor([[3, 3], [3, 3]]),
        singular_tolerance=1e-14,
    )
    restored = arrays["chronological_cocycle_hat"] * np.exp(
        arrays["chronological_cocycle_log_scale"][:, None, None]
    )
    expected = ((frame @ core @ basis.mH).mH * torch.exp(scale)[:, None, None]).numpy()
    np.testing.assert_allclose(restored, expected, atol=3e-13, rtol=3e-13)
    assert arrays["slow_effective_gaps_per_cycle"].shape == (2, 5)
    np.testing.assert_array_equal(
        arrays["slow_effective_gaps_per_cycle"],
        -2.0 * arrays["slow_pair_rates_per_cycle"],
    )


def test_noncommuting_factors_use_chronological_first_factor_leftmost() -> None:
    first = torch.eye(6, dtype=torch.complex128)
    second = torch.eye(6, dtype=torch.complex128)
    first[0, 1] = 0.4 + 0.2j
    second[1, 0] = -0.3j
    direct = first @ second
    assert not torch.allclose(direct, second @ first)
    propagated_columns = direct.mH
    frame, core_raw = torch.linalg.qr(propagated_columns)
    norm = torch.linalg.matrix_norm(core_raw, ord="fro")
    capture = RUNNER.EndpointTangentCapture(final_cycle=80)
    capture.frame = frame.unsqueeze(0)
    capture.core = (core_raw / norm).unsqueeze(0)
    capture.scale = torch.log(norm).reshape(1)
    task = RUNNER.Task(40, 1.0, 1, 66, 0, 0, 1, 1, "A")
    arrays = RUNNER.finalize_tangent(
        task,
        capture,
        torch.eye(6, dtype=torch.complex128).unsqueeze(0),
        torch.arange(6),
        torch.tensor([[3, 3]]),
        singular_tolerance=1e-14,
    )
    restored = arrays["chronological_cocycle_hat"][0] * np.exp(
        arrays["chronological_cocycle_log_scale"][0]
    )
    np.testing.assert_allclose(restored, direct.numpy(), atol=3e-13, rtol=3e-13)


def test_gap_only_task_does_not_materialize_saved_cocycle() -> None:
    torch.manual_seed(1702)
    capture = RUNNER.EndpointTangentCapture(final_cycle=48)
    capture.frame = torch.eye(8, 6, dtype=torch.complex128).unsqueeze(0)
    capture.core = torch.eye(6, dtype=torch.complex128).unsqueeze(0)
    capture.scale = torch.zeros(1, dtype=torch.float64)
    task = RUNNER.Task(24, 2.0, 10, 10, 0, 0, 1, 1, "A")
    arrays = RUNNER.finalize_tangent(
        task,
        capture,
        torch.eye(6, dtype=torch.complex128).unsqueeze(0),
        torch.arange(6),
        torch.tensor([[3, 3]]),
        singular_tolerance=1e-14,
    )
    assert "chronological_cocycle_hat" not in arrays
    assert "chronological_cocycle_log_scale" not in arrays


def test_direct_cocycle_average_can_cancel_while_quenched_gaps_do_not() -> None:
    product = np.diag([0.8, 0.4]).astype(np.complex128)
    samples = np.stack((product, -product))
    np.testing.assert_allclose(samples.mean(axis=0), 0.0)
    singular_logs = np.log(np.linalg.svd(samples, compute_uv=False))
    np.testing.assert_allclose(singular_logs[0], singular_logs[1])
    assert np.isfinite(singular_logs.mean(axis=0)).all()


def test_transient_record_digest_binds_every_replay_array() -> None:
    values = {
        "initial_frame": np.eye(2, dtype=np.complex128)[None],
        "initial_ranks": np.asarray([1], dtype=np.int64),
        "final_frame": np.eye(2, dtype=np.complex128)[None],
        "final_ranks": np.asarray([1], dtype=np.int64),
        "schedule": np.asarray([[[0, 1]]], dtype=np.int32),
        "outcomes": np.asarray([[[[True], [False]]]], dtype=np.bool_),
        "measurement_log_probability": np.asarray([[0.0, -0.5]], dtype=np.float64),
    }
    reference = RUNNER.record_payload_sha256(values)
    for name in values:
        changed = {key: np.array(value, copy=True) for key, value in values.items()}
        if changed[name].dtype == np.bool_:
            changed[name].flat[0] = not bool(changed[name].flat[0])
        else:
            changed[name].flat[0] = changed[name].flat[0] + 1
        assert RUNNER.record_payload_sha256(changed) != reference


def test_runner_uses_only_supported_canonical_engine_keywords() -> None:
    tree = ast.parse((BUNDLE / "run_campaign.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_markov_circuit"
    ]
    assert len(calls) == 2
    supported = set(inspect.signature(RUNNER.classA_U1FGTN_gpu.run_markov_circuit).parameters)
    for call in calls:
        assert {keyword.arg for keyword in call.keywords if keyword.arg is not None} <= supported


def test_bootstrap_is_fixed_seed_and_trajectory_first() -> None:
    values = np.arange(500, dtype=np.float64).reshape(100, 5)
    first = ANALYSIS.bootstrap_mean_interval(values, draws=200, seed=99)
    second = ANALYSIS.bootstrap_mean_interval(values, draws=200, seed=99)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert np.all(first[0] < values.mean(axis=0))
    assert np.all(first[1] > values.mean(axis=0))


def test_reused_slot09_metadata_covers_all_600_rows() -> None:
    root = (
        BUNDLE.parent
        / "09_pure_tangent_replay_acquisition/gpu_data"
        / RUNNER.REUSED_REVISION
    )
    reused = [task for task in RUNNER.expand_tasks() if task.reuse_record]
    assert sum(task.sample_count for task in reused) == 600
    sources = {source for task in reused for source in RUNNER.source_slot09_tasks(task)}
    assert len(sources) == 16
    for task in reused:
        slices = RUNNER.source_slot09_tasks(task)
        assert min(source.sample_start for source in slices) <= task.sample_start
        assert max(source.sample_stop for source in slices) >= task.sample_stop
    for task in sources:
        path, completion = RUNNER.verify_reused_pair(root, task, checksum=False)
        assert path.is_file()
        assert completion["result_sha256"]


def test_v1_import_is_exact_read_only_and_checksum_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = next(task for task in RUNNER.expand_tasks() if task.import_v1)
    result_path, completion_path = RUNNER.result_paths(tmp_path / "saved_v1", task)
    result_path.parent.mkdir(parents=True)
    RUNNER._atomic_npz(
        result_path,
        {
            "schema": np.asarray(RUNNER.RESULT_SCHEMA),
            "sampling_revision": np.asarray(RUNNER.V1_REVISION),
            "task_id": np.asarray(task.task_id),
            "Nx": np.asarray(20),
            "Ny": np.asarray(40),
            "alpha_1": np.asarray(1.0),
            "cycles": np.asarray(80),
            "sample_count": np.asarray(25),
            "dtype": np.asarray("complex128"),
            "full_cocycle_saved": np.asarray(True),
            "source_hashes_json": np.asarray(RUNNER.canonical_json(RUNNER.V1_SOURCE_HASHES)),
            "case_sample_indices": np.arange(25),
            "slow_effective_gaps_per_cycle": np.ones((25, 5)),
        },
    )
    monkeypatch.setattr(RUNNER, "V1_RESULT_BYTES", result_path.stat().st_size)
    monkeypatch.setattr(RUNNER, "V1_RESULT_SHA256", RUNNER.sha256_file(result_path))
    completion = {
        "schema": RUNNER.COMPLETION_SCHEMA,
        "status": "complete",
        "bundle": RUNNER.BUNDLE,
        "sampling_revision": RUNNER.V1_REVISION,
        "task_id": task.task_id,
        "lane": "A",
        "Nx": 20,
        "Ny": 40,
        "alpha_1": 1.0,
        "alpha_2": 30.0,
        "nshell": 1,
        "cycles": 80,
        "case_sample_indices": list(range(25)),
        "global_sample_indices": list(range(6600, 6625)),
        "sample_count": 25,
        "batch_seed": RUNNER.V1_BATCH_SEED,
        "configuration_sha256": RUNNER.V1_CONFIGURATION_SHA256,
        "source_hashes": RUNNER.V1_SOURCE_HASHES,
        "full_cocycle_saved": True,
        "result_filename": result_path.name,
        "result_bytes": RUNNER.V1_RESULT_BYTES,
        "result_sha256": RUNNER.V1_RESULT_SHA256,
        "performance_gate_passed": False,
    }
    RUNNER._atomic_json(completion_path, completion)
    complete = RUNNER.verified_complete(
        tmp_path / "new_v2",
        task,
        config_sha256="irrelevant-to-pinned-v1",
        hashes={},
        v1_root=tmp_path / "saved_v1",
    )
    assert complete == (True, "verified pinned v1 import")
    assert not RUNNER.result_paths(tmp_path / "new_v2", task)[0].exists()
    with pytest.raises(RuntimeError, match="imported read-only"):
        RUNNER.run_task(
            task,
            config=RUNNER.expected_config(),
            reused_root=tmp_path,
            scratch_root=tmp_path,
            config_sha256="irrelevant",
            hashes={},
        )
    with result_path.open("ab") as handle:
        handle.write(b"corruption")
    assert RUNNER.verified_complete(
        tmp_path / "new_v2",
        task,
        config_sha256="irrelevant",
        hashes={},
        v1_root=tmp_path / "saved_v1",
    )[0] is False


def test_slot09_slices_preserve_sample_order_and_original_batch_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = next(
        task
        for task in RUNNER.expand_tasks()
        if task.ny == 28 and task.alpha_1 == 1.0 and task.sample_start == 40
    )
    assert task.sample_stop == 60
    sources = RUNNER.source_slot09_tasks(task)
    assert [(source.sample_start, source.sample_stop) for source in sources] == [(0, 50), (50, 100)]

    def stage(_root: Path, source: RUNNER.Task, _scratch: Path):
        local = tmp_path / f"source_{source.batch_index}.npz"
        local.write_bytes(b"transient")
        return local, {
            "task_id": source.task_id,
            "result_sha256": str(source.batch_index) * 64,
            "seed": 111 + source.batch_index * 111,
        }

    def loaded(_path: Path, source: RUNNER.Task):
        rows = np.arange(source.sample_start, source.sample_stop, dtype=np.int64)
        width = 2 if source.batch_index == 0 else 3
        return {
            "initial_frame": np.ones((rows.size, 4, 2), dtype=np.complex128),
            "initial_ranks": np.full(rows.size, 2, dtype=np.int64),
            "final_frame": np.ones((rows.size, 4, width), dtype=np.complex128),
            "final_ranks": np.full(rows.size, width, dtype=np.int64),
            "schedule": rows[:, None].astype(np.int32),
            "outcomes": (rows % 2 == 0)[:, None],
            "measurement_log_probability": rows[:, None].astype(np.float64),
        }

    monkeypatch.setattr(RUNNER, "stage_reused_record", stage)
    monkeypatch.setattr(RUNNER, "_loaded_from_npz", loaded)
    merged, provenance = RUNNER.load_reused_record(tmp_path, task, tmp_path)
    np.testing.assert_array_equal(merged["schedule"][:, 0], np.arange(40, 60))
    assert merged["final_frame"].shape == (20, 4, 3)
    np.testing.assert_array_equal(merged["final_ranks"], [2] * 10 + [3] * 10)
    np.testing.assert_array_equal(merged["final_frame"][:10, :, 2], 0.0)
    assert provenance["record_original_batch_seeds"] == [111] * 10 + [222] * 10
    assert len(provenance["record_source_batches"]) == 2


def test_record_piece_concatenation_handles_observed_672_688_padding() -> None:
    first = {
        "final_frame": np.ones((1, 700, 672), dtype=np.complex128),
        "final_ranks": np.asarray([672], dtype=np.int64),
    }
    second = {
        "final_frame": np.full((1, 700, 688), 2.0, dtype=np.complex128),
        "final_ranks": np.asarray([688], dtype=np.int64),
    }
    combined = RUNNER.concatenate_record_pieces([first, second])
    assert combined["final_frame"].shape == (2, 700, 688)
    np.testing.assert_array_equal(combined["final_frame"][0, :, :672], 1.0)
    np.testing.assert_array_equal(combined["final_frame"][0, :, 672:], 0.0)
    np.testing.assert_array_equal(combined["final_frame"][1], 2.0)


def test_pre_padding_runner_outputs_remain_compatible() -> None:
    current = RUNNER.source_hashes()
    old = dict(current)
    old["runner"] = RUNNER.PRE_PADDING_HOTFIX_RUNNER_SHA256
    pre_nvrtc = dict(current)
    pre_nvrtc["runner"] = RUNNER.PRE_NVRTC_HOTFIX_RUNNER_SHA256
    pre_nvrtc_layout = dict(current)
    pre_nvrtc_layout["runner"] = RUNNER.PRE_NVRTC_LAYOUT_HOTFIX_RUNNER_SHA256
    assert RUNNER.compatible_v2_source_hashes(current, current)
    assert RUNNER.compatible_v2_source_hashes(old, current)
    assert RUNNER.compatible_v2_source_hashes(pre_nvrtc, current)
    assert RUNNER.compatible_v2_source_hashes(pre_nvrtc_layout, current)
    tampered = dict(old)
    tampered["gpu_engine"] = "0" * 64
    assert not RUNNER.compatible_v2_source_hashes(tampered, current)
    extra = dict(old)
    extra["unexpected"] = "0" * 64
    assert not RUNNER.compatible_v2_source_hashes(extra, current)


def test_nvrtc_runtime_discovery_is_versioned_and_exact(tmp_path: Path) -> None:
    assert RUNNER.nvrtc_builtins_soname(None) is None
    assert RUNNER.nvrtc_builtins_soname("13.0") == "libnvrtc-builtins.so.13.0"
    assert RUNNER.nvrtc_builtins_soname("12.8") == "libnvrtc-builtins.so.12.8"
    assert RUNNER.nvrtc_builtins_soname("nightly") is None

    package_root = tmp_path / "site-packages"
    library_dir = package_root / "nvidia" / "cuda_nvrtc" / "lib"
    library_dir.mkdir(parents=True)
    soname = "libnvrtc-builtins.so.13.0"
    (library_dir / soname).write_bytes(b"test fixture")
    assert RUNNER.find_nvrtc_library_dir(soname, (package_root,)) == library_dir
    assert RUNNER.find_nvrtc_library_dir("missing.so", (package_root,)) is None

    wheel_library_dir = package_root / "nvidia" / "cu13" / "lib"
    wheel_library_dir.mkdir(parents=True)
    (wheel_library_dir / soname).write_bytes(b"real wheel layout fixture")
    (library_dir / soname).unlink()
    assert RUNNER.find_nvrtc_library_dir(soname, (package_root,)) == wheel_library_dir


def test_drive_readback_failure_keeps_old_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "local.bin"
    final = tmp_path / "drive" / "result.bin"
    local.write_bytes(b"new")
    final.parent.mkdir()
    final.write_bytes(b"old")
    original = RUNNER.sha256_file

    def fail_temporary(path: Path) -> str:
        return "0" * 64 if path.name.startswith(".result.bin") else original(path)

    monkeypatch.setattr(RUNNER, "sha256_file", fail_temporary)
    with pytest.raises(OSError, match="temporary readback"):
        RUNNER.publish_file(local, final)
    assert final.read_bytes() == b"old"


def test_notebooks_sources_manifest_and_progress_contract() -> None:
    assert _sha(BUNDLE / "src/classA_U1FGTN_gpu.py") == _sha(
        REPO / "src/fgtn/classA_U1FGTN_gpu.py"
    )
    assert _sha(BUNDLE / "src/occupied_frame_gpu.py") == _sha(
        REPO / "src/fgtn/occupied_frame_gpu.py"
    )
    for lane in ("a", "b"):
        path = BUNDLE / f"run_hard_wall_tangent_lane_{lane}.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(_source(cell) for cell in notebook["cells"])
        assert f"LANE = '{lane.upper()}'" in source
        assert "REPORT_ONLY = False" in source
        assert "MAX_NEW_TASKS = None" in source
        assert "RUN_ANALYSIS = False" in source
        assert "SAVED_V1_ROOT" in source
        assert "c2ny_v2" in source
        assert "allocator cutoff=38 GiB" in source
        assert "PYTHONUNBUFFERED" in source and "TQDM_MININTERVAL" in source
        assert "environment['TQDM_MININTERVAL'] = '300'" in source
        assert "stderr=subprocess.STDOUT" in source
        assert "stdout=subprocess.PIPE" in source
        assert "os.read(process.stdout.fileno(), 4096)" in source
        assert "sys.stdout.write(decoder.decode(block))" in source
        assert "stream_child(command, environment)" in source
        assert "stream_child(analysis_command, environment)" in source
        final = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"][-1]
        assert _source(final).strip().splitlines() == [
            "from google.colab import runtime",
            "runtime.unassign()",
            "print('done')",
        ]
    manifest = json.loads((BUNDLE / "deployment_manifest.json").read_text())
    for relative, record in manifest["files"].items():
        path = BUNDLE / relative
        assert path.stat().st_size == record["bytes"]
        assert _sha(path) == record["sha256"]


def test_notebook_stream_child_relays_stdout_stderr_and_failure(capsys: pytest.CaptureFixture[str]) -> None:
    notebook = json.loads((BUNDLE / "run_hard_wall_tangent_lane_a.ipynb").read_text())
    run_cell = next(_source(cell) for cell in notebook["cells"] if "def stream_child" in _source(cell))
    tree = ast.parse(run_cell)
    declarations = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.FunctionDef))
    ]
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=declarations, type_ignores=[]), "<notebook-stream>", "exec"), namespace)
    stream_child = namespace["stream_child"]
    assert callable(stream_child)

    command = [
        sys.executable,
        "-u",
        "-c",
        "import sys; print('[configuration] ready', flush=True); "
        "sys.stderr.write('\\rprobe: 1/2'); sys.stderr.flush(); "
        "print('[summary] done', flush=True)",
    ]
    stream_child(command, namespace["os"].environ.copy())
    output = capsys.readouterr().out
    assert "[configuration] ready" in output
    assert "\rprobe: 1/2" in output
    assert "[summary] done" in output

    failing = [sys.executable, "-u", "-c", "import sys; print('failed', flush=True); sys.exit(7)"]
    with pytest.raises(namespace["subprocess"].CalledProcessError) as exc:
        stream_child(failing, namespace["os"].environ.copy())
    assert exc.value.returncode == 7
    assert "failed" in capsys.readouterr().out
