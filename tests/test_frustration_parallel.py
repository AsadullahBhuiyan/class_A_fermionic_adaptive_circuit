from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import time

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT = REPO_ROOT / "00_WORKSPACE" / "CURRENT"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(CURRENT))

from fgtn.diagnostics.cpu_parallel import ParallelPolicy, resolve_parallel_decision, run_parallel_tasks
from topological_frustration_diagnostics.launch_campaign import run_stage


CLI = CURRENT / "topological_frustration_diagnostics" / "run_cpu.py"


def _ordered_worker(value: int) -> int:
    time.sleep(0.01 * (3 - int(value)))
    return int(value) ** 2


def _failing_worker(value: int) -> int:
    if int(value) == 1:
        raise ValueError("intentional worker failure")
    return int(value)


def _run_cli(output: Path, *arguments: str, parallel: bool) -> None:
    parallel_arguments = (
        ["--workers", "2", "--threads-per-worker", "1", "--cpu-budget", "2"]
        if parallel
        else ["--no-parallel", "--cpu-budget", "2"]
    )
    subprocess.run(
        [
            sys.executable,
            str(CLI),
            *arguments,
            "--output-dir",
            str(output),
            *parallel_arguments,
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=240,
    )


def _assert_npz_trees_equivalent(serial: Path, parallel: Path) -> None:
    serial_files = sorted(path.relative_to(serial) for path in serial.rglob("*.npz"))
    parallel_files = sorted(path.relative_to(parallel) for path in parallel.rglob("*.npz"))
    assert serial_files == parallel_files
    for relative in serial_files:
        with np.load(serial / relative) as left, np.load(parallel / relative) as right:
            assert left.files == right.files
            for key in left.files:
                lhs, rhs = np.asarray(left[key]), np.asarray(right[key])
                if np.issubdtype(lhs.dtype, np.number):
                    if np.issubdtype(lhs.dtype, np.integer) or np.issubdtype(lhs.dtype, np.bool_):
                        np.testing.assert_array_equal(lhs, rhs)
                    elif key == "lyapunov_spectrum":
                        np.testing.assert_allclose(lhs, rhs, rtol=0.1, atol=1.0, equal_nan=True)
                    elif key in ("lyapunov_gap", "lyapunov_final_value"):
                        np.testing.assert_allclose(lhs, rhs, rtol=0.02, atol=0.01, equal_nan=True)
                    elif key == "lyapunov_final_vector":
                        assert np.array_equal(np.isfinite(lhs), np.isfinite(rhs))
                    elif key.startswith("lyapunov_mode_"):
                        np.testing.assert_allclose(lhs, rhs, rtol=0.02, atol=0.02, equal_nan=True)
                    elif key == "choi_spectrum":
                        np.testing.assert_allclose(lhs, rhs, rtol=2e-7, atol=1e-6, equal_nan=True)
                    else:
                        np.testing.assert_allclose(lhs, rhs, rtol=1e-9, atol=1e-10, equal_nan=True)
                else:
                    np.testing.assert_array_equal(lhs, rhs)


def _assert_seed_metadata_equal(serial: Path, parallel: Path) -> None:
    for serial_path in sorted(serial.rglob("run_summary.json")):
        relative = serial_path.relative_to(serial)
        left = json.loads(serial_path.read_text())
        right = json.loads((parallel / relative).read_text())
        assert left.get("sample_seeds") == right.get("sample_seeds")
        assert left.get("suite_worker_seeds") == right.get("suite_worker_seeds")
        assert left.get("choi_sample_seeds") == right.get("choi_sample_seeds")
        assert left.get("seed_records") == right.get("seed_records")


def test_parallel_policy_caps_workers_and_assigns_one_thread_to_sample_tasks():
    decision = resolve_parallel_decision(
        5,
        policy=ParallelPolicy(cpu_budget=2, workers=4),
        single_thread_tasks=True,
    )
    assert decision.workers == 2
    assert decision.threads_per_worker == 1
    assert decision.effective_cpu_budget == 2


def test_parallel_results_are_returned_in_input_order_and_fail_fast():
    values, metadata = run_parallel_tasks(
        _ordered_worker,
        [0, 1, 2],
        policy=ParallelPolicy(cpu_budget=2, workers=2),
    )
    assert values == [0, 1, 4]
    assert metadata["completed_tasks"] == 3
    assert metadata["backend"] == "loky"
    with pytest.raises(RuntimeError, match="Parallel task 1 failed"):
        run_parallel_tasks(
            _failing_worker,
            [0, 1, 2],
            policy=ParallelPolicy(cpu_budget=2, workers=2),
        )


@pytest.mark.parametrize(
    "arguments",
    [
        (
            "activity",
            "--nx",
            "4",
            "--ny",
            "6",
            "--geometry",
            "both",
            "--samples",
            "3",
            "--cycles",
            "2",
            "--burn-in",
            "1",
            "--bootstrap-samples",
            "4",
            "--seed",
            "301",
        ),
        (
            "spectral",
            "--nx",
            "4",
            "--ny",
            "6",
            "--geometry",
            "both",
            "--samples",
            "2",
            "--cycles",
            "2",
            "--protocol",
            "perfect_correction",
            "--seed",
            "302",
        ),
        (
            "response",
            "--nx",
            "4",
            "--ny",
            "6",
            "--geometry",
            "both",
            "--samples",
            "2",
            "--equilibration-cycles",
            "1",
            "--response-cycles",
            "2",
            "--sequence",
            "random",
            "--seed",
            "303",
        ),
    ],
)
def test_cli_serial_and_process_parallel_outputs_are_numerically_equivalent(
    tmp_path: Path,
    arguments: tuple[str, ...],
):
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    _run_cli(serial, *arguments, parallel=False)
    _run_cli(parallel, *arguments, parallel=True)
    _assert_npz_trees_equivalent(serial, parallel)
    _assert_seed_metadata_equal(serial, parallel)


def test_failed_launcher_stage_cannot_publish_success(tmp_path: Path):
    args = SimpleNamespace(dry_run=False)
    with pytest.raises(RuntimeError, match="failed"):
        run_stage(
            args,
            tmp_path,
            "failed_stage",
            lambda _: [sys.executable, "-c", "raise SystemExit(3)"],
            None,
        )
    stage = tmp_path / "failed_stage"
    assert (stage / "_FAILED").is_file()
    assert not (stage / "_SUCCESS").exists()
    status = json.loads((stage / "stage_status.json").read_text())
    assert status["status"] == "failed"
    assert status["execution"]["return_code"] == 3
