from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
PARENT = REPO / "00_WORKSPACE/CURRENT/final_production_new_designs"
BUNDLE = PARENT / "01_uniform_bulk_validation"
RUNNER_PATH = BUNDLE / "run_campaign.py"
OBSERVER_PATH = BUNDLE / "compact_observer.py"
NOTEBOOK_PATH = BUNDLE / "run_uniform_bulk_validation.ipynb"
LEGACY_ESTIMATOR_PATH = (
    REPO
    / "00_WORKSPACE/CURRENT/final_production_ready_figure_scripts/prior_designs"
    / "01_bulk_width_gate/src/streaming_covariance_observables_gpu.py"
)

if str(BUNDLE) not in sys.path:
    sys.path.insert(0, str(BUNDLE))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OBSERVER = _load(OBSERVER_PATH, "tested_uniform_bulk_observer")
RUNNER = _load(RUNNER_PATH, "tested_uniform_bulk_runner")


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


def test_registered_flat_bundle_layout_is_complete() -> None:
    layout = _load(PARENT / "bundle_layout.py", "tested_simple_bundle_layout")
    assert layout.NEW_DESIGN_BUNDLES == (
        "01_uniform_bulk_validation",
        "02_domain_wall_bipartite_mutual_information",
        "03_uniform_bulk_validation_large_l",
        "04_maxmix_manybody_lyapunov_pilot",
        "05_hard_wall_entropy_charge_batched_v2",
        "06_domain_wall_flattened_ground_state_reference",
        "07_maxmix_hard_soft_purification",
        "08_domain_wall_correlator_scaling",
        "09_pure_tangent_replay_acquisition",
        "10_soft_wall_entropy_charge_batched_v2",
        "11_wall_pump_width_endpoints",
        "12_wall_diabatic_spectral_pump_gpu",
        "13_maxmix_manybody_lyapunov_4ny",
        "14_hard_wall_xresolved_correlator_scaling",
        "15_hard_wall_alpha3_xresolved_correlator",
        "16_hard_wall_entropy_contour_all_ay",
        "17_hard_wall_tangent_gap_cocycle",
        "18_hard_wall_purification_alpha_endpoint",
    )
    assert layout.validate_bundle_layout(PARENT) == layout.NEW_DESIGN_BUNDLES
    index = json.loads((PARENT / "bundle_index.json").read_text(encoding="utf-8"))
    assert index["campaign_parent"] == "final_production_new_designs"
    assert index["standalone_contracts"] == {
        "01_uniform_bulk_validation": RUNNER.EXPECTED_REVISION,
        "02_domain_wall_bipartite_mutual_information": (
            "domain_wall_bmi_nx20_ny20-28_alpha21_desc_c2ny_s100_v2_batched_50-25-25"
        ),
        "03_uniform_bulk_validation_large_l": (
            "uniform_perfect_correction_40cycle_s100_l28_l40_batched_v2"
        ),
        "04_maxmix_manybody_lyapunov_pilot": (
            "maxmix_manybody_lyapunov_nx20_ny20to40_s100_gpu_v2"
        ),
        "05_hard_wall_entropy_charge_batched_v2": (
            "hard_wall_entropy_charge_nx20_ny30-60_s100_2ny_raster_v2_batched_endpoint"
        ),
        "06_domain_wall_flattened_ground_state_reference": (
            "domain_wall_flattened_ground_state_nx20_ny20-60_alpha21_shell1-2-inf_cpu_v2"
        ),
        "07_maxmix_hard_soft_purification": (
            "maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v3"
        ),
        "08_domain_wall_correlator_scaling": (
            "domain_wall_correlator_nx20_ny24-32_a1-1-3_"
            "nsh1-2-dense_s100_2ny_raster_v1"
        ),
        "09_pure_tangent_replay_acquisition": (
            "pure_tangent_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1"
        ),
        "10_soft_wall_entropy_charge_batched_v2": (
            "soft_wall_entropy_charge_nx20_ny30-60_s100_2ny_raster_"
            "v2_batched_endpoint"
        ),
        "11_wall_pump_width_endpoints": "wall_pump_width_endpoints_s100_v1",
        "12_wall_diabatic_spectral_pump_gpu": (
            "wall_diabatic_spectral_pump_gpu_primary_v1"
        ),
            "13_maxmix_manybody_lyapunov_4ny": (
                "maxmix_manybody_lyapunov_nx20_ny20-60_hard-soft_s100_4ny_"
                "gpu_v4_38gib_memory_scaled"
            ),
            "14_hard_wall_xresolved_correlator_scaling": (
                "hard_wall_xresolved_nx20_ny40-50-60_a1-1_nsh1_s100_2ny_"
                "raster_endpoint_frame_halfcov_occupations_v2_30gib_batched"
            ),
            "15_hard_wall_alpha3_xresolved_correlator": (
                "hard_wall_xresolved_nx20_ny60_a1-3_nsh1_s100_2ny_raster_"
                "endpoint_v1"
            ),
            "16_hard_wall_entropy_contour_all_ay": (
                "hard_wall_entropy_contour_all_ay_nx20_ny30-60_s100_2ny_"
                "raster_v3"
            ),
            "17_hard_wall_tangent_gap_cocycle": (
                "hard_wall_pure_tangent_alpha21_ny24-40_s100_c2ny_v2"
            ),
            "18_hard_wall_purification_alpha_endpoint": (
                "hard_wall_maxmix_purification_alpha21_ny24-50_s100_2ny_"
                "endpoint_lyapunov_v1"
            ),
    }


def test_notebook_exposes_the_exact_locked_production_contract() -> None:
    config, namespace = _notebook_config()
    assert RUNNER.validate_config(config) == config
    assert config["sizes"] == [12, 16, 20, 24]
    assert config["nshell_values"] == [1, 2, None]
    assert config["samples_per_case"] == 100
    assert config["batch_size_by_L"] == {
        "12": 100,
        "16": 100,
        "20": 50,
        "24": 25,
    }
    assert config["cycles"] == 40
    assert config["dtype"] == "complex128"
    assert config["protocol"]["DW"] is False
    assert config["protocol"]["alpha_1"] == 1.0
    assert config["protocol"]["alpha_2"] == 1.0
    assert config["protocol"]["perfect_correction"] is True
    assert namespace["REPORT_ONLY"] is False
    assert namespace["MAX_NEW_TASKS"] is None


def test_task_table_has_12_cases_and_24_unique_batches() -> None:
    config, _ = _notebook_config()
    tasks = RUNNER.expand_tasks(config)
    assert len(tasks) == 24
    assert len({task.task_id for task in tasks}) == 24
    assert len({task.seed for task in tasks}) == 24
    cases = {(task.size, task.nshell) for task in tasks}
    assert cases == {
        (size, nshell) for size in (12, 16, 20, 24) for nshell in (1, 2, None)
    }
    batches_per_case = {12: 1, 16: 1, 20: 2, 24: 4}
    for size, nshell in cases:
        case_tasks = [
            task for task in tasks if task.size == size and task.nshell == nshell
        ]
        assert len(case_tasks) == batches_per_case[size]
        assert [
            index for task in case_tasks for index in task.global_sample_indices
        ] == list(range(100))


def test_task_seed_is_stable_and_identity_sensitive() -> None:
    kwargs = {
        "size": 24,
        "nshell": 1,
        "batch_index": 0,
        "sample_start": 0,
        "sample_stop": 5,
    }
    first = RUNNER.task_seed(2026090201, **kwargs)
    assert first == RUNNER.task_seed(2026090201, **kwargs)
    assert first != RUNNER.task_seed(2026090201, **{**kwargs, "nshell": 2})
    assert first != RUNNER.task_seed(
        2026090201,
        size=24,
        nshell=1,
        batch_index=1,
        sample_start=5,
        sample_stop=10,
    )


def test_bundle_engine_copies_match_canonical_sources() -> None:
    assert _sha256(BUNDLE / "src/classA_U1FGTN_gpu.py") == _sha256(
        REPO / "src/fgtn/classA_U1FGTN_gpu.py"
    )
    assert _sha256(BUNDLE / "src/occupied_frame_gpu.py") == _sha256(
        REPO / "src/fgtn/occupied_frame_gpu.py"
    )


def test_compact_chern_estimator_matches_legacy_covariance_estimator() -> None:
    legacy = _load(LEGACY_ESTIMATOR_PATH, "tested_legacy_chern_estimator")
    torch.manual_seed(17)
    size = 6
    nlayer = 2 * size * size
    rank = size * size
    raw = torch.randn(nlayer, rank, dtype=torch.complex128)
    frame = torch.linalg.qr(raw, mode="reduced").Q.unsqueeze(0)
    partitions = OBSERVER.build_chern_partition_indices(nx=size, ny=size)
    new_value = OBSERVER.frame_real_space_chern(frame, partitions)

    occupied = frame @ frame.mH
    centered_covariance = 2.0 * occupied - torch.eye(
        nlayer, dtype=torch.complex128
    ).unsqueeze(0)
    legacy_partitions = legacy.build_chern_partition_indices(nx=size, ny=size)
    legacy_value = legacy.real_space_chern_batch_torch(
        centered_covariance, legacy_partitions
    )
    torch.testing.assert_close(new_value, legacy_value, rtol=0.0, atol=2.0e-12)


def _engine_arguments(*, cycles: int, observer=None, samples: int = 1) -> dict:
    return {
        "G_history": False,
        "progress": False,
        "cycles": cycles,
        "postselect": False,
        "postselect_probability": 0.0,
        "perfect_correction": True,
        "samples": samples,
        "init_mode": "default",
        "save": False,
        "n_a": 0.5,
        "sequence": "random",
        "meas_slab_only": False,
        "batch_size": samples,
        "return_data": True,
        "state_representation": "auto",
        "native_cycle_observer": observer,
        "track_choi": False,
        "return_native_state": True,
        "require_no_covariance_materialization": True,
    }


def _cpu_model(size: int):
    return RUNNER.classA_U1FGTN_gpu(
        Nx=size,
        Ny=size,
        DW=False,
        nshell=1,
        filling_frac=0.5,
        alpha_1=1.0,
        alpha_2=1.0,
        trial_orbitals="X",
        dw_truncation=False,
        device="cpu",
        dtype="complex128",
        backend="local",
    )


def test_every_cycle_observer_includes_initial_state_and_is_noninvasive() -> None:
    size = 4
    cycles = 3
    seed = 4102
    samples = 2

    np.random.seed(seed)
    torch.manual_seed(seed)
    observer = OBSERVER.CompactChernChargeObserver(
        size=size, physical_cycles=cycles, samples=samples
    )
    with_observer = _cpu_model(size).run_markov_circuit(
        **_engine_arguments(cycles=cycles, observer=observer, samples=samples)
    )
    rng_with_observer = torch.get_rng_state().clone()
    payload = observer.payload()

    np.random.seed(seed)
    torch.manual_seed(seed)
    without_observer = _cpu_model(size).run_markov_circuit(
        **_engine_arguments(cycles=cycles, observer=None, samples=samples)
    )
    rng_without_observer = torch.get_rng_state().clone()

    assert payload["cycles"].tolist() == [0, 1, 2, 3]
    assert payload["normalized_cycles"].tolist() == [0.0, 0.25, 0.5, 0.75]
    assert payload["real_space_chern"].shape == (samples, cycles + 1)
    assert payload["global_charge"].shape == (samples, cycles + 1)
    np.testing.assert_allclose(
        payload["global_charge"], payload["particle_number"], atol=1.0e-10, rtol=0
    )
    np.testing.assert_array_equal(
        payload["half_filling_offset"], payload["particle_number"] - size * size
    )
    for key in ("frame", "ranks", "min_ranks", "max_ranks", "log_weight"):
        np.testing.assert_array_equal(
            with_observer["native_final"][key], without_observer["native_final"][key]
        )
    torch.testing.assert_close(rng_with_observer, rng_without_observer, rtol=0, atol=0)


def _fake_payload(task) -> dict[str, np.ndarray]:
    count = 41
    shape = (task.sample_count, count)
    particle_number = np.full(shape, task.size * task.size, dtype=np.int64)
    return {
        "cycles": np.arange(count, dtype=np.int64),
        "normalized_cycles": np.arange(count, dtype=np.float64) / task.size,
        "real_space_chern": np.broadcast_to(np.linspace(0.0, 1.0, count), shape).copy(),
        "global_charge": particle_number.astype(np.float64),
        "particle_number": particle_number,
        "half_filling_offset": np.zeros(shape, dtype=np.int64),
    }


def test_completion_pair_is_verified_and_corruption_is_pending(tmp_path: Path) -> None:
    config, _ = _notebook_config()
    task = RUNNER.expand_tasks(config)[0]
    hashes = RUNNER.source_hashes()
    config_sha256 = RUNNER.config_hash(config)
    RUNNER._save_task(
        output_root=tmp_path / "drive",
        scratch_root=tmp_path / "scratch",
        task=task,
        payload=_fake_payload(task),
        elapsed_seconds=1.0,
        config_sha256=config_sha256,
        hashes=hashes,
    )
    complete, reason = RUNNER.verified_complete(
        output_root=tmp_path / "drive",
        task=task,
        config_sha256=config_sha256,
        hashes=hashes,
    )
    assert complete and reason == "verified"

    result_path, completion_path = RUNNER.task_paths(tmp_path / "drive", task)
    completion_raw = completion_path.read_bytes()
    completion_path.unlink()
    complete, reason = RUNNER.verified_complete(
        output_root=tmp_path / "drive",
        task=task,
        config_sha256=config_sha256,
        hashes=hashes,
    )
    assert not complete
    assert reason == "incomplete result/completion pair"
    completion_path.write_bytes(completion_raw)

    result_path.write_bytes(result_path.read_bytes() + b"corrupt")
    complete, reason = RUNNER.verified_complete(
        output_root=tmp_path / "drive",
        task=task,
        config_sha256=config_sha256,
        hashes=hashes,
    )
    assert not complete
    assert "byte count mismatch" in reason


def test_completion_json_is_not_written_when_result_publication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _notebook_config()
    task = RUNNER.expand_tasks(config)[0]

    def fail_publish(local_path: Path, final_path: Path):
        del local_path, final_path
        raise OSError("simulated Drive readback failure")

    monkeypatch.setattr(RUNNER, "publish_file", fail_publish)
    with pytest.raises(OSError, match="simulated Drive"):
        RUNNER._save_task(
            output_root=tmp_path / "drive",
            scratch_root=tmp_path / "scratch",
            task=task,
            payload=_fake_payload(task),
            elapsed_seconds=1.0,
            config_sha256=RUNNER.config_hash(config),
            hashes=RUNNER.source_hashes(),
        )
    result_path, completion_path = RUNNER.task_paths(tmp_path / "drive", task)
    assert not result_path.exists()
    assert not completion_path.exists()


def test_notebook_code_compiles_and_uses_simple_colab_contract() -> None:
    notebook = _notebook()
    code_sources = [
        _cell_source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    for index, source in enumerate(code_sources):
        compile(source, f"{NOTEBOOK_PATH}#code-{index}", "exec")
    joined = "\n".join(code_sources)
    assert joined.count("drive.mount('/content/drive')") == 1
    assert "shutil.copytree(BUNDLE_DRIVE_DIR, LOCAL_BUNDLE_DIR)" in joined
    assert "subprocess.Popen(" in joined
    assert "stderr=subprocess.STDOUT" in joined
    assert "os.read(process.stdout.fileno(), 4096)" in joined
    assert "sys.stdout.write(decoder.decode(raw))" in joined
    assert (
        "from google.colab import runtime\nruntime.unassign()\nprint('done')" in joined
    )
    assert "googleapiclient" not in joined
    assert "remote-status" not in joined
    assert "lease" not in joined.lower()


def test_runner_uses_canonical_engine_and_no_covariance_history() -> None:
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
