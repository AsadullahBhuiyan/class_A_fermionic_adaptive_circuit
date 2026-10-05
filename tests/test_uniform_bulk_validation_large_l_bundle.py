from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PARENT = REPO / "00_WORKSPACE/CURRENT/final_production_new_designs"
BUNDLE = PARENT / "03_uniform_bulk_validation_large_l"
RUNNER_PATH = BUNDLE / "run_campaign.py"
NOTEBOOK_PATH = BUNDLE / "run_uniform_bulk_validation_large_l.ipynb"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _notebook_config() -> tuple[dict, dict]:
    for cell in _notebook()["cells"]:
        source = _cell_source(cell)
        if cell.get("cell_type") == "code" and "CONFIG = {" in source:
            namespace: dict = {}
            exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)
            return namespace["CONFIG"], namespace
    raise AssertionError("notebook configuration cell was not found")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if str(BUNDLE) not in sys.path:
    sys.path.insert(0, str(BUNDLE))
RUNNER = _load(RUNNER_PATH, "tested_uniform_bulk_large_l_runner")


def test_large_l_notebook_exposes_locked_add_on_contract() -> None:
    config, namespace = _notebook_config()
    assert RUNNER.validate_config(config) == config
    assert config["sampling_revision"] == (
        "uniform_perfect_correction_40cycle_s100_l28_l40_batched_v2"
    )
    assert config["root_seed"] == 2026090301
    assert config["sizes"] == [28, 32, 36, 40]
    assert config["nshell_values"] == [1, 2, None]
    assert config["samples_per_case"] == 100
    assert config["batch_size_by_L"] == {
        "28": 100,
        "32": 50,
        "36": 30,
        "40": 20,
    }
    assert config["cycles"] == 40
    assert config["device"] == "cuda:0"
    assert config["dtype"] == "complex128"
    assert config["protocol"]["DW"] is False
    assert config["protocol"]["init_mode"] == "default"
    assert config["protocol"]["alpha_1"] == 1.0
    assert config["protocol"]["alpha_2"] == 1.0
    assert config["protocol"]["perfect_correction"] is True
    assert namespace["REPORT_ONLY"] is False
    assert namespace["MAX_NEW_TASKS"] is None


def test_large_l_task_table_covers_12_cases_and_1200_trajectories() -> None:
    config, _ = _notebook_config()
    tasks = RUNNER.expand_tasks(config)
    assert len(tasks) == 36
    assert len({task.task_id for task in tasks}) == 36
    assert len({task.seed for task in tasks}) == 36
    cases = {(task.size, task.nshell) for task in tasks}
    assert cases == {
        (size, nshell)
        for size in (28, 32, 36, 40)
        for nshell in (1, 2, None)
    }
    batches_per_case = {28: 1, 32: 2, 36: 4, 40: 5}
    for size, nshell in cases:
        case_tasks = [
            task for task in tasks if task.size == size and task.nshell == nshell
        ]
        assert len(case_tasks) == batches_per_case[size]
        assert [
            index for task in case_tasks for index in task.global_sample_indices
        ] == list(range(100))
    assert sum(task.sample_count for task in tasks) == 1200


def test_large_l_bundle_sources_are_canonical_and_observer_is_shared() -> None:
    assert _sha256(BUNDLE / "src/classA_U1FGTN_gpu.py") == _sha256(
        REPO / "src/fgtn/classA_U1FGTN_gpu.py"
    )
    assert _sha256(BUNDLE / "src/occupied_frame_gpu.py") == _sha256(
        REPO / "src/fgtn/occupied_frame_gpu.py"
    )
    assert _sha256(BUNDLE / "compact_observer.py") == _sha256(
        PARENT / "01_uniform_bulk_validation/compact_observer.py"
    )


def test_large_l_notebook_compiles_and_streams_progress() -> None:
    code_sources = [
        _cell_source(cell)
        for cell in _notebook()["cells"]
        if cell.get("cell_type") == "code"
    ]
    for index, source in enumerate(code_sources):
        compile(source, f"{NOTEBOOK_PATH}#code-{index}", "exec")
    joined = "\n".join(code_sources)
    assert joined.count("drive.mount('/content/drive')") == 1
    assert "03_uniform_bulk_validation_large_l" in joined
    assert "shutil.copytree(BUNDLE_DRIVE_DIR, LOCAL_BUNDLE_DIR)" in joined
    assert "subprocess.Popen(" in joined
    assert "stderr=subprocess.STDOUT" in joined
    assert "os.read(process.stdout.fileno(), 4096)" in joined
    assert "sys.stdout.write(decoder.decode(raw))" in joined
    assert "runtime.unassign()" in joined


def test_large_l_runner_keeps_simple_canonical_execution_contract() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert ".run_markov_circuit(" in source
    assert "native_cycle_observer=observer" in source
    assert "samples=task.sample_count" in source
    assert "batch_size=task.sample_count" in source
    assert "progress=True" in source
    assert "G_history=False" in source
    assert "save=False" in source
    assert "require_no_covariance_materialization=True" in source
    assert "DriveRemoteCommit" not in source
    assert "checkpoint" not in source.lower()
