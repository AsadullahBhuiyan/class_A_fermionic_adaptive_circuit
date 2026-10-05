from __future__ import annotations

import copy
import importlib.util
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = (
    ROOT
    / "00_WORKSPACE/CURRENT/final_production_new_designs/01_p1_chern_dynamics"
)


def _load_runner():
    observer_path = BUNDLE / "src/p1_chern_observables.py"
    observer_spec = importlib.util.spec_from_file_location(
        "p1_checkpoint_resume_observables_under_test", observer_path
    )
    assert observer_spec is not None and observer_spec.loader is not None
    observer_module = importlib.util.module_from_spec(observer_spec)
    sys.modules[observer_spec.name] = observer_module
    observer_spec.loader.exec_module(observer_module)

    prior_observer = sys.modules.get("p1_chern_observables")
    sys.modules["p1_chern_observables"] = observer_module
    try:
        runner_path = BUNDLE / "src/p1_chern_runner.py"
        runner_spec = importlib.util.spec_from_file_location(
            "p1_checkpoint_resume_runner_under_test", runner_path
        )
        assert runner_spec is not None and runner_spec.loader is not None
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
    finally:
        if prior_observer is None:
            sys.modules.pop("p1_chern_observables", None)
        else:
            sys.modules["p1_chern_observables"] = prior_observer
    return runner_module


RUNNER = _load_runner()


def _tiny_case(config: dict) -> dict:
    case = copy.deepcopy(RUNNER.expand_cases(config)[0])
    case["case_id"] = "P1_CHECKPOINT_E2E_L4_nsh-1"
    case["model"].update({"Nx": 4, "Ny": 4, "backend": "local"})
    case["run"].update({"cycles": 4, "samples": 1})
    case["execution"] = {"samples_per_shard": 1}
    case["observer"].update(
        {"center_count": 10, "radius_fraction": 0.4, "cycles": list(range(5))}
    )
    return case


def _archive_npz(archive: Path, suffix: str) -> dict[str, np.ndarray]:
    with tarfile.open(archive, "r:gz") as handle:
        members = [
            member
            for member in handle.getmembers()
            if member.isfile() and member.name.lstrip("./").endswith(suffix)
        ]
        assert len(members) == 1
        extracted = handle.extractfile(members[0])
        assert extracted is not None
        payload = extracted.read()
    with np.load(io.BytesIO(payload), allow_pickle=False) as stored:
        return {key: np.array(stored[key], copy=True) for key in stored.files}


def _assert_arrays_exact(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    *,
    keys: set[str] | None = None,
) -> None:
    selected = set(expected) if keys is None else keys
    assert selected <= set(actual)
    assert selected <= set(expected)
    for key in sorted(selected):
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


def _run_case(*, config: dict, case: dict, drive_root: Path) -> dict:
    return RUNNER._run_case(
        bundle_root=BUNDLE,
        config=config,
        case=case,
        shard_index=0,
        drive_root=drive_root,
        mode="smoke",
        archive_result=True,
    )


def test_cycle_checkpoint_crash_resume_archive_retry_is_bitwise_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = RUNNER.load_config(BUNDLE)
    case = _tiny_case(config)
    reference_root = tmp_path / "reference"
    resumed_root = tmp_path / "resumed"

    reference = _run_case(
        config=config, case=case, drive_root=reference_root
    )
    assert reference["status"] == "archived_to_drive"
    _, reference_archive, _, _ = RUNNER._archive_paths(
        bundle_root=BUNDLE,
        config=config,
        case=case,
        shard_index=0,
        drive_root=reference_root,
        mode="smoke",
    )
    reference_science = _archive_npz(reference_archive, "p1_chern.npz")
    reference_rng = _archive_npz(reference_archive, "rng_after.npz")

    original_write_checkpoint = RUNNER._write_cycle_checkpoint

    class SimulatedCrash(RuntimeError):
        pass

    def crash_after_cycle_two(**kwargs):
        receipt = original_write_checkpoint(**kwargs)
        if int(kwargs["completed_cycle"]) == 2:
            raise SimulatedCrash("crash after durable cycle 2")
        return receipt

    monkeypatch.setattr(RUNNER, "_write_cycle_checkpoint", crash_after_cycle_two)
    with pytest.raises(SimulatedCrash, match="durable cycle 2"):
        _run_case(config=config, case=case, drive_root=resumed_root)
    monkeypatch.setattr(RUNNER, "_write_cycle_checkpoint", original_write_checkpoint)

    _, resumed_archive, run_id, _ = RUNNER._archive_paths(
        bundle_root=BUNDLE,
        config=config,
        case=case,
        shard_index=0,
        drive_root=resumed_root,
        mode="smoke",
    )
    checkpoint_dir, pointer = RUNNER._checkpoint_paths(
        archive=resumed_archive, run_id=run_id
    )
    cycle_two_receipt = json.loads(pointer.read_text(encoding="utf-8"))
    assert cycle_two_receipt["completed_cycle"] == 2
    cycle_two_payload = checkpoint_dir / cycle_two_receipt["checkpoint"]
    assert cycle_two_payload.is_file()

    # A bit flip must fail closed before the engine advances the copied shard.
    corrupt_root = tmp_path / "corrupt"
    _, corrupt_archive, corrupt_run_id, _ = RUNNER._archive_paths(
        bundle_root=BUNDLE,
        config=config,
        case=case,
        shard_index=0,
        drive_root=corrupt_root,
        mode="smoke",
    )
    corrupt_dir, _ = RUNNER._checkpoint_paths(
        archive=corrupt_archive, run_id=corrupt_run_id
    )
    corrupt_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(checkpoint_dir, corrupt_dir)
    corrupt_payload = corrupt_dir / cycle_two_receipt["checkpoint"]
    with corrupt_payload.open("r+b") as handle:
        handle.seek(corrupt_payload.stat().st_size // 2)
        original = handle.read(1)
        assert original
        handle.seek(-1, 1)
        handle.write(bytes([original[0] ^ 0x01]))

    class EngineMustNotRun:
        def __init__(self, **_kwargs):
            pass

        def run_markov_circuit(self, **_kwargs):
            raise AssertionError("engine advanced a corrupt checkpoint")

    original_engine = RUNNER.classA_U1FGTN_gpu
    monkeypatch.setattr(RUNNER, "classA_U1FGTN_gpu", EngineMustNotRun)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _run_case(config=config, case=case, drive_root=corrupt_root)
    monkeypatch.setattr(RUNNER, "classA_U1FGTN_gpu", original_engine)

    # Valid resume also removes interrupted-write and superseded-generation debris.
    stale_temporary = checkpoint_dir / f".{run_id}.cycle_000001.stale.npz.tmp"
    stale_generation = checkpoint_dir / f"{run_id}.cycle_000001.stale.npz"
    stale_temporary.write_bytes(b"partial")
    stale_generation.write_bytes(b"superseded")

    original_archive = RUNNER._archive

    def fail_archive(*_args, **_kwargs):
        raise OSError("simulated archive commit failure")

    monkeypatch.setattr(RUNNER, "_archive", fail_archive)
    with pytest.raises(OSError, match="archive commit failure"):
        _run_case(config=config, case=case, drive_root=resumed_root)
    monkeypatch.setattr(RUNNER, "_archive", original_archive)

    final_checkpoint_receipt = json.loads(pointer.read_text(encoding="utf-8"))
    assert final_checkpoint_receipt["completed_cycle"] == 4
    assert (checkpoint_dir / final_checkpoint_receipt["checkpoint"]).is_file()
    assert not stale_temporary.exists()
    assert not stale_generation.exists()
    assert not resumed_archive.exists()

    # A completed-cycle checkpoint must finalize without another dynamics call.
    monkeypatch.setattr(RUNNER, "classA_U1FGTN_gpu", EngineMustNotRun)
    relaunched = _run_case(config=config, case=case, drive_root=resumed_root)
    assert relaunched["status"] == "archived_to_drive"
    assert relaunched["manifest"]["resumed_from_cycle"] == 4
    assert relaunched["manifest"]["canonical_engine_calls_this_session"] == 0
    assert resumed_archive.is_file()
    assert not pointer.exists()
    assert not checkpoint_dir.exists()

    resumed_science = _archive_npz(resumed_archive, "p1_chern.npz")
    science_keys = set(reference_science) - {"observer_seconds"}
    _assert_arrays_exact(resumed_science, reference_science, keys=science_keys)
    resumed_rng = _archive_npz(resumed_archive, "rng_after.npz")
    _assert_arrays_exact(resumed_rng, reference_rng)


def test_shard_lease_rejects_a_nested_writer(tmp_path: Path) -> None:
    archive = tmp_path / "outputs" / "synthetic.tar.gz"
    run_id = "p1_nested_lease_test"
    with RUNNER._ShardLease(archive=archive, run_id=run_id):
        with pytest.raises(RuntimeError, match="another P1 writer owns"):
            with RUNNER._ShardLease(archive=archive, run_id=run_id):
                raise AssertionError("nested writer acquired the live lease")

