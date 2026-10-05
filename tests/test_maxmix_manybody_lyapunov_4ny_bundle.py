from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
BUNDLE = (
    REPO
    / "00_WORKSPACE"
    / "CURRENT"
    / "final_production_new_designs"
    / "13_maxmix_manybody_lyapunov_4ny"
)


def _load_modules():
    saved = {name: sys.modules.get(name) for name in ("lyapunov_observer", "classA_U1FGTN_gpu")}
    observer_spec = importlib.util.spec_from_file_location(
        "lyapunov_observer", BUNDLE / "lyapunov_observer.py"
    )
    observer = importlib.util.module_from_spec(observer_spec)
    sys.modules["lyapunov_observer"] = observer
    assert observer_spec.loader is not None
    observer_spec.loader.exec_module(observer)
    sys.path.insert(0, str(BUNDLE / "src"))
    try:
        runner_spec = importlib.util.spec_from_file_location(
            "bundle13_run_campaign", BUNDLE / "run_campaign.py"
        )
        runner = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner
        assert runner_spec.loader is not None
        runner_spec.loader.exec_module(runner)
    finally:
        sys.path.remove(str(BUNDLE / "src"))
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return runner, observer


RUNNER, OBSERVER = _load_modules()


def _config() -> dict:
    return json.loads((BUNDLE / "campaign_config.json").read_text(encoding="utf-8"))


def test_locked_grid_batching_and_seed_inventory() -> None:
    config = _config()
    assert config == RUNNER.expected_config()
    assert config["Ny_values"] == [20, 24, 30, 36, 44, 56, 60]
    assert config["cycles_multiplier"] == 4
    assert config["samples_per_case"] == 100
    assert config["init_mode"] == "maxmix"
    assert config["sequence"] == "raster_y"
    assert config["perfect_correction"] is True
    assert config["postselect"] is False
    assert config["dtype"] == "complex128"
    assert config["gpu_memory_hard_limit_gib"] == 38.0
    assert config["execution_batch_size_by_Ny"] == {
        "20": 100,
        "24": 100,
        "30": 100,
        "36": 90,
        "44": 60,
        "56": 40,
        "60": 35,
    }
    all_seeds: list[int] = []
    for construction in ("hard", "soft"):
        tasks = RUNNER.expand_execution_batches(config, construction)
        shards = RUNNER.all_result_shards(config, construction)
        assert len(tasks) == 13
        assert len(shards) == 140
        assert sum(task.samples for task in tasks) == 700
        assert all(task.cycles == 4 * task.ny for task in tasks)
        assert all(shard.sample_indices.size == 5 for shard in shards)
        assert {task.ny for task in tasks} == {20, 24, 30, 36, 44, 56, 60}
        assert max(task.samples * task.ny**2 for task in tasks) <= 126_000
        all_seeds.extend(task.seed for task in tasks)
    assert len(all_seeds) == len(set(all_seeds))


def test_observer_identity_spectrum_caps_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    for construction, active_count in (("hard", 44), ("soft", 80)):
        active = torch.arange(active_count, dtype=torch.long)
        observer = OBSERVER.BatchedActiveSpectrumObserver(
            nx=20,
            ny=2,
            cycles=8,
            samples=2,
            active_indices=active,
            full_mode_count=80,
            wall_locations=(5, 15),
            construction=construction,
            sample_indices=np.asarray([0, 1], dtype=np.int64),
            sample_chunk=1,
        )
        covariance = torch.zeros((2, 80, 80), dtype=torch.complex128)
        observer.observe(cycle=0, G=covariance)
        sites = observer.expected_sites_per_cycle
        rows = torch.arange(2).repeat_interleave(sites)
        probabilities = torch.zeros((rows.numel(), 4), dtype=torch.float64)
        for cycle in range(1, 9):
            observer.record_event(
                cycle=cycle,
                sample_offsets=rows,
                conditional_log_probability=probabilities,
            )
            observer.observe(cycle=cycle, G=covariance)
        observer.validate()
        assert np.allclose(observer.occupations[:, 0], 0.5)
        assert np.allclose(observer.leading_log_sigma2[:, 0], 0.0, atol=2e-14)
        assert not np.any(observer.cap_mask[:, 0])

        task = RUNNER.ExecutionBatch(construction, 2, 0, 0, 2, 123)
        output = tmp_path / construction / "out"
        scratch = tmp_path / construction / "scratch"
        hashes = {"unit": "hash"}
        RUNNER.save_checkpoint(
            output,
            scratch,
            task,
            completed_cycle=8,
            elapsed_seconds=1.0,
            G=np.zeros((2, 80, 80), dtype=np.complex128),
            observer=observer,
            cfg_hash="config",
            hashes=hashes,
        )
        checkpoint, reason = RUNNER.load_checkpoint(
            output, task, cfg_hash="config", hashes=hashes
        )
        assert reason == "verified"
        assert checkpoint is not None and checkpoint.completed_cycle == 8
        npz_path, json_path = RUNNER.checkpoint_paths(output, task)
        assert npz_path.name == "checkpoint.npz"
        assert json_path.name == "checkpoint.json"


def test_heap_levels_match_exhaustive_products() -> None:
    occupations = np.asarray([0.2, 0.35, 0.7, 0.9])
    log_z = 1.234
    actual = OBSERVER.leading_log_sigma2_levels(occupations, log_z, count=16)
    exhaustive = []
    for bits in range(16):
        value = log_z
        for index, nu in enumerate(occupations):
            value += np.log(nu if bits & (1 << index) else 1.0 - nu)
        exhaustive.append(value)
    assert np.allclose(actual, np.sort(exhaustive)[::-1], atol=2e-14)
    capped = np.asarray([0.0, 0.25, 1.0])
    _, costs, caps = OBSERVER.natural_spectrum_factors(capped)
    assert np.array_equal(caps, [True, False, True])
    assert np.isinf(costs[[0, 2]]).all()


def test_segmented_engine_execution_preserves_state_rng_and_observer() -> None:
    class Bar:
        def update(self, _: int) -> None:
            pass

    config = RUNNER.expected_config()

    def model(construction: str):
        return RUNNER.classA_U1FGTN_gpu(
            Nx=20,
            Ny=2,
            DW=True,
            nshell=1,
            filling_frac=0.5,
            alpha_1=1.0,
            alpha_2=30.0,
            trial_orbitals="X",
            dw_truncation=construction == "hard",
            triv_region_local_mode=False,
            device="cpu",
            dtype="complex128",
            backend="local",
        )

    def observer(instance, construction: str):
        active = instance.active_top_layer_indices(
            meas_slab_only=config["constructions"][construction]["meas_slab_only"]
        )
        return OBSERVER.BatchedActiveSpectrumObserver(
            nx=20,
            ny=2,
            cycles=8,
            samples=2,
            active_indices=active,
            full_mode_count=instance.Nlayer,
            wall_locations=tuple(instance.DW_loc),
            construction=construction,
            sample_indices=np.asarray([0, 1]),
            sample_chunk=1,
        )

    for construction in ("hard", "soft"):
        task = RUNNER.ExecutionBatch(construction, 2, 0, 0, 2, 777)
        continuous_model = model(construction)
        continuous_observer = observer(continuous_model, construction)
        np.random.seed(task.seed)
        torch.manual_seed(task.seed)
        continuous = RUNNER.run_segment(
            continuous_model,
            config,
            task,
            continuous_observer,
            completed_cycle=0,
            segment_cycles=4,
            G_init=None,
            continuing=False,
            progress_bar=Bar(),
        )
        continuous_rng = RUNNER.capture_rng()

        segmented_model = model(construction)
        segmented_observer = observer(segmented_model, construction)
        np.random.seed(task.seed)
        torch.manual_seed(task.seed)
        segmented = RUNNER.run_segment(
            segmented_model,
            config,
            task,
            segmented_observer,
            completed_cycle=0,
            segment_cycles=2,
            G_init=None,
            continuing=False,
            progress_bar=Bar(),
        )
        saved_rng = RUNNER.capture_rng()
        RUNNER.restore_rng(saved_rng)
        segmented = RUNNER.run_segment(
            segmented_model,
            config,
            task,
            segmented_observer,
            completed_cycle=2,
            segment_cycles=2,
            G_init=segmented,
            continuing=True,
            progress_bar=Bar(),
        )
        segmented_rng = RUNNER.capture_rng()

        assert np.array_equal(continuous, segmented)
        assert np.array_equal(
            continuous_rng["rng_torch_cpu"], segmented_rng["rng_torch_cpu"]
        )
        for name in continuous_observer.ARRAY_NAMES:
            left = getattr(continuous_observer, name)
            right = getattr(segmented_observer, name)
            if left.dtype.kind == "f":
                assert np.array_equal(left, right, equal_nan=True), name
            else:
                assert np.array_equal(left, right), name


def test_result_pair_verification_and_corruption(tmp_path: Path) -> None:
    config = _config()
    task = RUNNER.expand_execution_batches(config, "hard")[-1]
    shard = RUNNER.result_shards(task)[0]
    cfg_hash = RUNNER.config_hash(config)
    hashes = RUNNER.source_hashes(BUNDLE)
    result_path, completion_path = RUNNER.result_paths(tmp_path, shard)
    result_path.parent.mkdir(parents=True)
    ncheck = OBSERVER.spectrum_checkpoint_cycles(task.ny).size
    neff = 22 * task.ny
    with result_path.open("wb") as handle:
        np.savez(
            handle,
            result_schema=np.asarray(RUNNER.RESULT_SCHEMA),
            sample_indices=shard.sample_indices,
            cumulative_log_probability=np.zeros((5, task.cycles + 1)),
            occupations=np.full((5, ncheck, neff), 0.5),
            leading_log_sigma2=np.zeros((5, ncheck, 64)),
        )
    completion = RUNNER._shard_identity(shard, cfg_hash=cfg_hash, hashes=hashes)
    completion.update(
        {
            "result_filename": result_path.name,
            "result_bytes": result_path.stat().st_size,
            "result_sha256": RUNNER.sha256_file(result_path),
        }
    )
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    assert RUNNER.verified_complete(
        tmp_path, shard, cfg_hash=cfg_hash, hashes=hashes
    ) == (True, "verified")
    result_path.write_bytes(result_path.read_bytes() + b"x")
    valid, reason = RUNNER.verified_complete(
        tmp_path, shard, cfg_hash=cfg_hash, hashes=hashes
    )
    assert not valid and "checksum" in reason


def test_notebooks_are_canonical_visible_and_stream_progress() -> None:
    builder_spec = importlib.util.spec_from_file_location(
        "bundle13_builder", BUNDLE / "build_notebooks.py"
    )
    builder = importlib.util.module_from_spec(builder_spec)
    assert builder_spec.loader is not None
    builder_spec.loader.exec_module(builder)
    for construction in ("hard", "soft"):
        path = BUNDLE / f"run_{construction}_wall_manybody_lyapunov_4ny.ipynb"
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved == builder.notebook(construction)
        joined = "\n".join("".join(cell["source"]) for cell in saved["cells"])
        assert f"CONSTRUCTION = '{construction}'" in joined
        assert "Ny_values': [20, 24, 30, 36, 44, 56, 60]" in joined
        assert "Expected an A100 40-GB-class GPU" in joined
        assert "'gpu_memory_hard_limit_gib': 38.0" in joined
        assert "'60': 35" in joined
        assert "shutil.copytree(BUNDLE_DRIVE_DIR, LOCAL_BUNDLE_DIR)" in joined
        assert "importlib.util.spec_from_file_location" in joined
        assert "runner.main(runner_args)" in joined
        assert "subprocess.Popen" not in joined
        assert "os.read" not in joined
        final = "".join(saved["cells"][-1]["source"])
        assert final == (
            "from google.colab import runtime\n"
            "runtime.unassign()\n"
            "print('done')\n"
        )
    source = (BUNDLE / "run_campaign.py").read_text(encoding="utf-8")
    assert "tqdm(" in source
    assert "durable shards" in source and "cycles" in source
    assert "Drive API" not in source


def test_hard_wall_lane_notebooks_are_disjoint_and_checkpoint_compatible() -> None:
    builder_spec = importlib.util.spec_from_file_location(
        "bundle13_lane_builder", BUNDLE / "build_notebooks.py"
    )
    builder = importlib.util.module_from_spec(builder_spec)
    assert builder_spec.loader is not None
    builder_spec.loader.exec_module(builder)

    assert builder.HARD_LANES == {
        "a": (60, 44, 20),
        "b": (56, 36, 30, 24),
    }
    assert set(builder.HARD_LANES["a"]).isdisjoint(builder.HARD_LANES["b"])
    assert set(builder.HARD_LANES["a"]) | set(builder.HARD_LANES["b"]) == {
        20, 24, 30, 36, 44, 56, 60
    }

    for lane_name, ny_lane in builder.HARD_LANES.items():
        path = BUNDLE / f"run_hard_wall_manybody_lyapunov_4ny_lane_{lane_name}.ipynb"
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved == builder.notebook(
            "hard", lane_name=lane_name, ny_lane=ny_lane
        )
        joined = "\n".join("".join(cell["source"]) for cell in saved["cells"])
        assert "CONSTRUCTION = 'hard'" in joined
        assert f"LANE_NAME = '{lane_name}'" in joined
        assert f"NY_LANE = {list(ny_lane)!r}" in joined
        assert "runner.expand_execution_batches = _lane_expand" in joined
        assert "runner.main(runner_args)" in joined
        assert "gpu_v4_38gib_memory_scaled" in joined
        assert "'gpu_memory_hard_limit_gib': 38.0" in joined
        final = "".join(saved["cells"][-1]["source"])
        assert final == (
            "from google.colab import runtime\n"
            "runtime.unassign()\n"
            "print('done')\n"
        )


def test_canonical_gpu_sources_are_byte_identical() -> None:
    pairs = (
        (REPO / "src/fgtn/classA_U1FGTN_gpu.py", BUNDLE / "src/classA_U1FGTN_gpu.py"),
        (REPO / "src/fgtn/occupied_frame_gpu.py", BUNDLE / "src/occupied_frame_gpu.py"),
    )
    for canonical, bundled in pairs:
        assert canonical.read_bytes() == bundled.read_bytes()
        assert hashlib.sha256(canonical.read_bytes()).hexdigest() == hashlib.sha256(
            bundled.read_bytes()
        ).hexdigest()
