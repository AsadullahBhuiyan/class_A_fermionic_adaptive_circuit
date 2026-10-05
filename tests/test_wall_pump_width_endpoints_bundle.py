from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
BUNDLE = (
    REPO
    / "00_WORKSPACE/CURRENT/final_production_new_designs/11_wall_pump_width_endpoints"
)


def _load_runner():
    path = BUNDLE / "run_campaign.py"
    spec = importlib.util.spec_from_file_location("_wall_pump_width_endpoint_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


def test_locked_grid_and_durable_shards(runner):
    config = runner.expected_config()
    cases = runner.expand_cases(config)
    primary = [case for case in cases if case.collection == "endpoints"]
    bridge = [case for case in cases if case.collection == "bridge"]
    assert {(case.protocol, case.nx) for case in primary} == {
        ("nsh1", 28),
        ("nsh1", 32),
        ("dense", 24),
        ("dense", 28),
        ("dense", 32),
    }
    assert {(case.protocol, case.nx) for case in bridge} == {
        ("nsh1", 20),
        ("nsh1", 24),
        ("dense", 20),
    }
    assert sum(case.samples for case in primary) == 1000
    assert sum(case.samples for case in bridge) == 150
    assert len(runner.all_result_shards(config, 40)) == 230
    assert all(len(shard.sample_ids) == 5 for shard in runner.all_result_shards(config, 40))


def test_scientific_contract_and_geometry(runner):
    config = runner.expected_config()
    assert config["Ny"] == 24 and config["cycles"] == 48
    assert config["dtype"] == "complex128"
    assert config["execution_backend"] == "gpu"
    assert config["protocol"]["sequence"] == "raster_y"
    assert config["protocol"]["perfect_correction"] is True
    assert config["protocol"]["postselect"] is False
    assert config["shells"] == {"nsh1": 1, "dense": None}
    assert config["walls"] == runner.WALL_FLAGS
    for case in runner.expand_cases(config):
        assert case.wall_locations == (case.nx // 4, 3 * case.nx // 4)


def test_every_locked_case_uses_the_engine_backend_required_by_nshell(
    runner, monkeypatch
):
    constructed = []

    class FakeModel:
        def __init__(self, **kwargs):
            constructed.append(dict(kwargs))
            self.DW_loc = (kwargs["Nx"] // 4, 3 * kwargs["Nx"] // 4)
            self.dtype = runner.torch.complex128

    monkeypatch.setattr(runner, "classA_U1FGTN_gpu", FakeModel)
    config = runner.expected_config()
    cases = runner.expand_cases(config)
    for case in cases:
        runner.build_model(config, case)

    assert len(constructed) == len(cases) == 16
    for case, kwargs in zip(cases, constructed, strict=True):
        expected_nshell = runner.SHELLS[case.protocol]
        assert kwargs["nshell"] == expected_nshell
        assert kwargs["backend"] == (
            "dense" if expected_nshell is None else "local"
        )


def test_execution_batches_cover_every_sample_once(runner):
    config = runner.expected_config()
    tasks = runner.expand_execution_batches(config, 40)
    assert len(tasks) == 36
    by_case = {}
    for task in tasks:
        by_case.setdefault(task.case.key, []).extend(task.sample_ids)
        assert task.sample_start % 5 == 0 and task.sample_stop % 5 == 0
    for case in runner.expand_cases(config):
        assert by_case[case.key] == list(range(case.samples))
    assert len({task.seed for task in tasks}) == len(tasks)
    assert runner.resolved_execution_identity(config, 40)["execution_backend"] == "gpu"


def test_authoritative_result_layout(runner, tmp_path):
    config = runner.expected_config()
    shards = runner.all_result_shards(config, 40)
    primary = shards[0]
    result, completion = runner.result_paths(tmp_path, primary)
    assert result.relative_to(tmp_path).as_posix() == "endpoints/nsh1/N28x24/soft/shard_00.npz"
    assert completion.name == "shard_00.completion.json"
    bridge = next(shard for shard in shards if shard.case.collection == "bridge")
    result, _ = runner.result_paths(tmp_path, bridge)
    assert result.relative_to(tmp_path).as_posix().startswith("bridge/")


def test_fastest_safe_benchmark_selection(runner):
    rows = [
        {
            "candidate": candidate,
            "projected_headroom_bytes": headroom,
            "trajectory_cycles_per_second": speed,
            "error": error,
        }
        for candidate, headroom, speed, error in (
            (10, 20 * 1024**3, 1.0, None),
            (20, 15 * 1024**3, 3.0, None),
            (40, 12 * 1024**3, 5.0, None),
            (60, 7 * 1024**3, 8.0, None),
            (80, 10 * 1024**3, 4.0, None),
            (100, 20 * 1024**3, 0.0, "oom"),
        )
    ]
    assert runner.select_fastest_safe_candidate(rows) == 40
    with pytest.raises(RuntimeError, match="8 GiB"):
        runner.select_fastest_safe_candidate(rows[3:4])


def test_saved_benchmark_cannot_bypass_40_hour_gate(runner, tmp_path):
    config = runner.expected_config()
    hashes = runner.source_hashes()
    selected = 40
    payload = {
        **runner._benchmark_identity(config=config, hashes=hashes, gpu_name="NVIDIA A100"),
        "status": "accepted",
        "selected_execution_batch_size": selected,
        "resolved_execution_identity": runner.resolved_execution_identity(config, selected),
        "resolved_execution_sha256": runner.resolved_execution_sha256(config, selected),
        "full_48_cycle_timing_rows": [{} for _ in range(10)],
        "projected_missing_1000_seconds": 40 * 3600 + 1,
    }
    runner._write_json(runner.benchmark_path(tmp_path), payload)
    benchmark, reason = runner.load_benchmark(
        output_root=tmp_path,
        config=config,
        hashes=hashes,
        gpu_name="NVIDIA A100",
    )
    assert benchmark is None
    assert "40-hour" in reason


def test_completion_requires_bound_identity_size_and_checksum(runner, tmp_path):
    config = runner.expected_config()
    selected = 40
    shard = runner.all_result_shards(config, selected)[0]
    hashes = runner.source_hashes()
    result_path, completion_path = runner.result_paths(tmp_path, shard)
    runner._write_npz(
        result_path,
        {
            "schema": np.asarray(runner.RESULT_SCHEMA),
            "sample_ids": np.asarray(shard.sample_ids, dtype=np.int64),
            "frames": np.zeros(
                (5, 2 * shard.case.nx * runner.NY, 0), dtype=np.complex128
            ),
            "ranks": np.zeros(5, dtype=np.int64),
        },
    )
    completion = {
        **runner._shard_identity(
            shard=shard,
            config=config,
            hashes=hashes,
            selected_batch_size=selected,
        ),
        "result_filename": result_path.name,
        "result_bytes": result_path.stat().st_size,
        "result_sha256": runner.sha256_file(result_path),
        "result": {
            "name": result_path.name,
            "bytes": result_path.stat().st_size,
            "sha256": runner.sha256_file(result_path),
        },
        "config_hash": runner.resolved_execution_sha256(config, selected),
    }
    runner._write_json(completion_path, completion)
    assert runner.verified_complete(
        output_root=tmp_path,
        shard=shard,
        config=config,
        hashes=hashes,
        selected_batch_size=selected,
    ) == (True, "verified")
    completion["result_sha256"] = "0" * 64
    runner._write_json(completion_path, completion)
    valid, reason = runner.verified_complete(
        output_root=tmp_path,
        shard=shard,
        config=config,
        hashes=hashes,
        selected_batch_size=selected,
    )
    assert not valid and "checksum" in reason


def test_spectral_runner_ingests_exact_published_shard_interface(runner, tmp_path):
    spectral_path = (
        REPO
        / "00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot/run_wall_diabatic_spectral_pump_s100.py"
    )
    spec = importlib.util.spec_from_file_location("_wall_diabatic_spectral_runner", spectral_path)
    assert spec is not None and spec.loader is not None
    spectral = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = spectral
    spec.loader.exec_module(spectral)

    config = runner.expected_config()
    selected = 40
    shard = runner.all_result_shards(config, selected)[0]
    hashes = runner.source_hashes()
    result_path, completion_path = runner.result_paths(tmp_path, shard)
    dimension = 2 * shard.case.nx * runner.NY
    frames = np.zeros((5, dimension, 1), dtype=np.complex128)
    frames[:, 0, 0] = 1.0
    observer = runner.ChargeObserver(samples=5)
    observer.seen_cycles[:] = True
    observer.global_charge[:] = 1
    payload = runner._result_payload(
        shard=shard,
        frame=frames,
        ranks=np.ones(5, dtype=np.int64),
        gram_residual=np.zeros(5, dtype=np.float64),
        observer=observer,
        elapsed_seconds=1.0,
        config=config,
        hashes=hashes,
        selected_batch_size=selected,
        benchmark={"projected_missing_1000_seconds": 1.0},
    )
    runner._write_npz(result_path, payload)
    identity = runner._shard_identity(
        shard=shard,
        config=config,
        hashes=hashes,
        selected_batch_size=selected,
    )
    digest = runner.sha256_file(result_path)
    completion = {
        **identity,
        "result_filename": result_path.name,
        "result_bytes": result_path.stat().st_size,
        "result_sha256": digest,
        "result": {
            "name": result_path.name,
            "bytes": result_path.stat().st_size,
            "sha256": digest,
        },
    }
    runner._write_json(completion_path, completion)
    spectral_config = spectral.load_config(spectral.DEFAULT_CONFIG)
    source = next(
        row for row in spectral_config["sources"] if row["cell"] == shard.case.cell
    )
    reference = spectral.endpoint_ref(source, "soft", 0, tmp_path)
    loaded = spectral.load_endpoint(reference, shard.case.nx, runner.NY)
    assert loaded.dtype == np.complex128
    assert loaded.shape == (dimension, 1)
    assert np.array_equal(loaded, frames[0])


def test_drivefs_publication_reads_back_and_replaces(runner, tmp_path):
    local = tmp_path / "local.bin"
    final = tmp_path / "drive" / "stable.bin"
    local.write_bytes(b"endpoint-data")
    receipt = runner.publish_file(local, final)
    assert final.read_bytes() == b"endpoint-data"
    assert receipt == {
        "filename": "stable.bin",
        "bytes": len(b"endpoint-data"),
        "sha256": runner.sha256_file(final),
    }
    assert not list(final.parent.glob(".*.tmp"))


def test_five_cycle_checkpoint_roundtrip_binds_rng_and_observer(runner, tmp_path):
    config = runner.expected_config()
    hashes = runner.source_hashes()
    case = runner.Case("bridge", "nsh1", 4, "soft", 5)
    batch = runner.ExecutionBatch(case, 0, 0, 5, 12345)
    observer = runner.ChargeObserver(5)
    observer.seen_cycles[:6] = True
    observer.global_charge[:, :6] = 0
    native = {
        "frame": np.zeros((5, 2 * 4 * runner.NY, 0), dtype=np.complex128),
        "ranks": np.zeros(5, dtype=np.int64),
        "gram_residual": np.zeros(5, dtype=np.float64),
    }
    runner._seed_rng(9876)
    runner.save_checkpoint(
        output_root=tmp_path / "out",
        scratch_root=tmp_path / "scratch",
        batch=batch,
        completed_cycle=5,
        elapsed_seconds=1.25,
        native=native,
        observer=observer,
        config=config,
        hashes=hashes,
        selected_batch_size=40,
    )
    restored, reason = runner.load_checkpoint(
        output_root=tmp_path / "out",
        batch=batch,
        config=config,
        hashes=hashes,
        selected_batch_size=40,
    )
    assert reason == "verified" and restored is not None
    assert restored.completed_cycle == 5
    assert np.array_equal(restored.seen_cycles, observer.seen_cycles)
    assert np.array_equal(restored.global_charge, observer.global_charge)
    assert {"numpy_keys", "torch_cpu", "torch_cuda_count"} <= set(restored.rng)


def test_production_drive_free_space_gate_is_25_gib(runner, monkeypatch, tmp_path):
    class Usage:
        free = 24 * 1024**3

    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: Usage())
    with pytest.raises(RuntimeError, match="Drive production"):
        runner._check_space(
            tmp_path,
            required_bytes=runner.MINIMUM_PRODUCTION_DRIVE_FREE_BYTES,
            label="Drive production",
        )
    assert runner.MINIMUM_PRODUCTION_DRIVE_FREE_BYTES == 25 * 1024**3


def test_canonical_sources_are_byte_identical():
    assert (BUNDLE / "src/classA_U1FGTN_gpu.py").read_bytes() == (
        REPO / "src/fgtn/classA_U1FGTN_gpu.py"
    ).read_bytes()


@pytest.mark.parametrize("hard_wall", [False, True])
def test_cpu_gpu_kernels_match_for_prepared_frame_and_frozen_record(hard_wall):
    """The backend bridge changes hardware, not the conditioned dynamics."""
    from src.fgtn.classA_U1FGTN import classA_U1FGTN
    from src.fgtn.classA_U1FGTN_gpu import classA_U1FGTN_gpu

    class Record:
        def __init__(self):
            self.entries = []

        def __call__(self, **payload):
            self.entries.append(
                {
                    "cycle": int(payload["cycle"]),
                    "site_id": int(payload["site_id"]),
                    "branch_events": tuple(
                        dict(event) for event in payload["branch_events"]
                    ),
                }
            )

    nx, ny = 4, 2
    dimension, rank = 2 * nx * ny, nx * ny
    rng = np.random.default_rng(7301)
    trial = rng.normal(size=(dimension, rank)) + 1j * rng.normal(
        size=(dimension, rank)
    )
    frame = np.linalg.qr(trial, mode="reduced")[0].astype(np.complex128)
    common = {
        "Nx": nx,
        "Ny": ny,
        "DW": True,
        "nshell": 1,
        "alpha_1": 1.0,
        "alpha_2": 30.0,
        "dw_truncation": hard_wall,
    }
    record = Record()
    cpu = classA_U1FGTN(**common)
    cpu_result = cpu.run_markov_circuit(
        cycles=1,
        samples=1,
        frame_init=frame,
        frame_init_prepared=True,
        sequence="raster_y",
        perfect_correction=True,
        meas_slab_only=hard_wall,
        random_seed=91,
        G_history=False,
        save=False,
        progress=False,
        state_representation="physical_frame",
        return_native_state=True,
        trajectory_weight_observer=record,
    )
    schedule = np.asarray(
        [[[entry["site_id"] for entry in record.entries]]], dtype=np.int64
    )
    outcomes = np.asarray(
        [[[
            [
                event["outcome_occupied"]
                for event in entry["branch_events"]
                if event["kind"] == "measurement"
            ]
            for entry in record.entries
        ]]],
        dtype=np.bool_,
    )
    assert outcomes.shape[-1] == 4

    gpu = classA_U1FGTN_gpu(
        **common, device="cpu", dtype="complex128", backend="local"
    )
    gpu_result = gpu.run_markov_circuit(
        cycles=1,
        samples=1,
        frame_init=frame[None, ...],
        frame_ranks=np.asarray([rank], dtype=np.int64),
        frame_init_prepared=True,
        sequence="raster_y",
        perfect_correction=True,
        meas_slab_only=hard_wall,
        G_history=False,
        save=False,
        progress=False,
        state_representation="physical_frame",
        return_native_state=True,
        require_no_covariance_materialization=True,
        batch_size=1,
        return_data=True,
        frozen_schedule=schedule,
        frozen_outcomes=outcomes,
    )
    cpu_frame = np.asarray(cpu_result["native_final"]["frame"])
    gpu_rank = int(gpu_result["native_final"]["ranks"][0])
    gpu_frame = np.asarray(gpu_result["native_final"]["frame"])[0, :, :gpu_rank]
    assert int(cpu_result["native_final"]["rank"]) == gpu_rank
    np.testing.assert_allclose(
        cpu_frame @ cpu_frame.conj().T,
        gpu_frame @ gpu_frame.conj().T,
        atol=2e-11,
        rtol=2e-11,
    )
    assert (BUNDLE / "src/occupied_frame_gpu.py").read_bytes() == (
        REPO / "src/fgtn/occupied_frame_gpu.py"
    ).read_bytes()


def test_notebook_is_reproducible_visible_and_disconnects():
    path = BUNDLE / "run_wall_pump_width_endpoints.ipynb"
    before = path.read_bytes()
    subprocess.run([sys.executable, str(BUNDLE / "build_notebook.py")], check=True)
    assert path.read_bytes() == before
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "REPORT_ONLY = False" in source
    assert "BENCHMARK_ONLY = False" in source
    assert "MAX_NEW_EXECUTION_BATCHES = None" in source
    assert "A100" in source and "complex128" in source
    assert "run_streaming_child" in source and "-u" in source
    assert "OUTPUT_ROOT" in source and "SCRATCH_ROOT" in source
    assert "tqdm" in (BUNDLE / "run_campaign.py").read_text(encoding="utf-8")
    last = "".join(notebook["cells"][-1]["source"])
    assert last == "from google.colab import runtime\nruntime.unassign()\nprint('done')\n"
    runner_text = (BUNDLE / "run_campaign.py").read_text(encoding="utf-8")
    assert "googleapiclient" not in runner_text
    assert "lease" not in runner_text.lower()


def test_bundle_registry_contains_slot_11():
    spec = importlib.util.spec_from_file_location(
        "_new_design_bundle_layout",
        REPO / "00_WORKSPACE/CURRENT/final_production_new_designs/bundle_layout.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "11_wall_pump_width_endpoints" in module.validate_bundle_layout(
        REPO / "00_WORKSPACE/CURRENT/final_production_new_designs"
    )


def test_locked_config_rejects_mutation(runner):
    config = runner.expected_config()
    config["cycles"] = 47
    with pytest.raises(ValueError, match="locked campaign"):
        runner.validate_config(config)


def test_endpoint_bundle_readme_documents_operator_and_recovery_contract():
    readme = (BUNDLE / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split()).lower()
    for required in (
        "before benchmark dynamics began",
        "no benchmark receipt, endpoint shard, or scientific data",
        "nshell=1` to the canonical local backend",
        "nshell=none` to the canonical dense backend",
        "BENCHMARK_ONLY=True",
        "REPORT_ONLY=True",
        "MAX_NEW_EXECUTION_BATCHES=1",
        "230 durable endpoint shards",
        "25 GiB",
        "at most five completed cycles are lost",
        "completion JSON is published last",
        "classA_U1FGTN_gpu.py",
        "imported_endpoints/wall_pump_width_endpoints_s100_v1",
    ):
        assert required.lower() in normalized
