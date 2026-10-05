from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
PARENT = REPO / "00_WORKSPACE/CURRENT/final_production_new_designs"
BUNDLE = PARENT / "02_domain_wall_bipartite_mutual_information"
RUNNER_PATH = BUNDLE / "run_campaign.py"
OBSERVER_PATH = BUNDLE / "mutual_information_observer.py"
NOTEBOOK_PATH = BUNDLE / "run_domain_wall_bipartite_mutual_information.ipynb"
FIGURE_SCRIPT_PATH = BUNDLE / "make_hard_wall_figures.py"

if str(BUNDLE) not in sys.path:
    sys.path.insert(0, str(BUNDLE))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OBSERVER = _load(OBSERVER_PATH, "tested_domain_wall_bmi_observer")
RUNNER = _load(RUNNER_PATH, "tested_domain_wall_bmi_runner")
FIGURES = _load(FIGURE_SCRIPT_PATH, "tested_domain_wall_bmi_figures")


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


def test_notebook_exposes_exact_locked_campaign() -> None:
    config, namespace = _notebook_config()
    assert RUNNER.validate_config(config) == config
    assert config["Nx"] == 20
    assert config["Ny_values"] == [20, 24, 28]
    assert config["width_rule"] == "Ny//4"
    assert config["alpha_1_values"] == list(RUNNER.EXPECTED_ALPHA_VALUES)
    assert config["alpha_1_values"][0] == 3.0
    assert config["alpha_1_values"][-1] == 1.0
    assert np.all(np.diff(config["alpha_1_values"]) < 0.0)
    assert config["alpha_2"] == 30.0
    assert config["wall_constructions"] == ["hard", "soft"]
    assert config["nshell"] == 1
    assert config["samples_per_case"] == 100
    assert config["batch_size_by_Ny"] == {"20": 50, "24": 25, "28": 25}
    assert config["cycles_rule"] == "2*Ny"
    assert config["protocol"] == {
        "DW": True,
        "filling_frac": 0.5,
        "trial_orbitals": "X",
        "init_mode": "default",
        "sequence": "raster_y",
        "perfect_correction": True,
        "postselect": False,
        "postselect_probability": 0.0,
        "meas_slab_only": True,
        "n_a": 0.5,
        "triv_region_local_mode": False,
    }
    assert namespace["REPORT_ONLY"] is False
    assert namespace["MAX_NEW_TASKS"] is None
    assert str(namespace["LEGACY_OUTPUT_ROOT"]).endswith(
        "domain_wall_bmi_nx20_ny20-28_alpha21_desc_c2ny_s100_v1"
    )
    assert str(namespace["OUTPUT_ROOT"]).endswith(
        "domain_wall_bmi_nx20_ny20-28_alpha21_desc_c2ny_s100_v2_batched_50-25-25"
    )
    index = json.loads((PARENT / "bundle_index.json").read_text(encoding="utf-8"))
    assert index["standalone_contracts"][BUNDLE.name] == RUNNER.EXPECTED_REVISION


def test_task_table_has_exact_cases_samples_and_independent_seeds() -> None:
    config, _ = _notebook_config()
    tasks = RUNNER.expand_tasks(config)
    assert len(tasks) == 420
    assert len({task.task_id for task in tasks}) == 420
    assert len({task.seed for task in tasks}) == 420
    cases = {(task.ny, task.wall, task.alpha_1) for task in tasks}
    assert len(cases) == 126
    assert sum(task.sample_count for task in tasks) == 12600
    first_case_alphas = [
        task.alpha_1
        for task in tasks
        if task.ny == 20 and task.wall == "hard" and task.batch_index == 0
    ]
    assert first_case_alphas == list(RUNNER.EXPECTED_ALPHA_VALUES)
    for ny, wall, alpha_1 in cases:
        case_tasks = [
            task
            for task in tasks
            if (task.ny, task.wall, task.alpha_1) == (ny, wall, alpha_1)
        ]
        expected_count = 2 if ny == 20 else 4
        assert len(case_tasks) == expected_count
        assert {task.sample_count for task in case_tasks} == {
            RUNNER.EXPECTED_BATCH_SIZES[ny]
        }
        assert [
            sample for task in case_tasks for sample in task.global_sample_indices
        ] == list(range(100))
        assert {task.width for task in case_tasks} == {ny // 4}
        assert {task.cycles for task in case_tasks} == {2 * ny}
        for task in case_tasks:
            legacy = RUNNER.legacy_tasks_for_macro(task)
            assert len(legacy) == task.sample_count // RUNNER.LEGACY_SHARD_SIZE
            assert [
                sample for shard in legacy for sample in shard.global_sample_indices
            ] == list(task.global_sample_indices)


def test_quarter_width_opposite_strip_geometry() -> None:
    nx = 20
    for ny, width in ((20, 5), (24, 6), (28, 7)):
        for y0 in range(ny // 2):
            a, b, union = OBSERVER.opposite_strip_indices(
                nx=nx, ny=ny, width=width, y0=y0
            )
            assert a.numel() == b.numel() == 2 * nx * width
            assert union.numel() == 4 * nx * width
            assert torch.unique(union).numel() == union.numel()
            a_rows = set((a.numpy() // (2 * nx)).tolist())
            b_rows = set((b.numpy() // (2 * nx)).tolist())
            assert a_rows == {(y0 + offset) % ny for offset in range(width)}
            assert b_rows == {(y0 + ny // 2 + offset) % ny for offset in range(width)}
            assert a_rows.isdisjoint(b_rows)


def _random_pure_centered_covariance(
    *, samples: int, nx: int, ny: int, seed: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    modes = 2 * nx * ny
    rank = nx * ny
    covariances = []
    for _ in range(samples):
        raw = torch.randn(modes, rank, dtype=torch.complex128, generator=generator)
        frame = torch.linalg.qr(raw, mode="reduced").Q
        occupation = frame @ frame.mH
        covariances.append(2.0 * occupation - torch.eye(modes, dtype=torch.complex128))
    return torch.stack(covariances)


def _numpy_entropy(centered: np.ndarray, indices: np.ndarray) -> float:
    restricted = centered[np.ix_(indices, indices)]
    occupation = 0.5 * (restricted + np.eye(indices.size))
    occupation = 0.5 * (occupation + occupation.conj().T)
    eigenvalues = np.linalg.eigvalsh(occupation).clip(1.0e-12, 1.0 - 1.0e-12)
    return float(
        -np.sum(
            eigenvalues * np.log(eigenvalues)
            + (1.0 - eigenvalues) * np.log(1.0 - eigenvalues)
        )
    )


def test_observer_matches_numpy_and_half_translation_average() -> None:
    nx, ny, width, samples = 2, 8, 2, 2
    centered = _random_pure_centered_covariance(
        samples=samples, nx=nx, ny=ny, seed=1907
    )
    observer = OBSERVER.FixedWidthMutualInformationObserver(
        nx=nx,
        ny=ny,
        width=width,
        samples=samples,
        expected_cycle=16,
    )
    observer(
        cycle=16,
        G=centered,
        batch_index=0,
        batch_start=0,
        batch_count=samples,
    )
    payload = observer.payload()
    direct = np.empty((samples, ny), dtype=np.float64)
    components = np.empty((samples, ny, 3), dtype=np.float64)
    for sample in range(samples):
        matrix = centered[sample].numpy()
        for y0 in range(ny):
            a, b, union = OBSERVER.opposite_strip_indices(
                nx=nx, ny=ny, width=width, y0=y0
            )
            sa = _numpy_entropy(matrix, a.numpy())
            sb = _numpy_entropy(matrix, b.numpy())
            su = _numpy_entropy(matrix, union.numpy())
            components[sample, y0] = (sa, sb, su)
            direct[sample, y0] = sa + sb - su
    np.testing.assert_allclose(
        payload["mutual_information_y0avg"], direct.mean(axis=1), atol=2.0e-11
    )
    np.testing.assert_allclose(
        payload["entropy_a_y0avg"], components[:, :, 0].mean(axis=1), atol=2.0e-11
    )
    np.testing.assert_allclose(
        payload["entropy_b_y0avg"], components[:, :, 1].mean(axis=1), atol=2.0e-11
    )
    np.testing.assert_allclose(
        payload["entropy_union_y0avg"],
        components[:, :, 2].mean(axis=1),
        atol=2.0e-11,
    )
    np.testing.assert_allclose(direct[:, : ny // 2], direct[:, ny // 2 :], atol=2.0e-11)
    assert payload["nominal_y0_count"].item() == ny
    assert payload["unique_y0_count"].item() == ny // 2
    assert payload["entropy_identity_max_abs_residual"].item() < 1.0e-12


def test_canonical_engine_endpoint_callback_runs_for_both_wall_constructions() -> None:
    for wall, truncation in (("hard", True), ("soft", False)):
        seed = 3200 + int(truncation)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = RUNNER.classA_U1FGTN_gpu(
            Nx=4,
            Ny=4,
            DW=True,
            nshell=1,
            filling_frac=0.5,
            alpha_1=1.5,
            alpha_2=30.0,
            trial_orbitals="X",
            dw_truncation=truncation,
            triv_region_local_mode=False,
            device="cpu",
            dtype="complex128",
            backend="local",
        )
        observer = OBSERVER.FixedWidthMutualInformationObserver(
            nx=4, ny=4, width=1, samples=1, expected_cycle=1
        )
        result = model.run_markov_circuit(
            G_history=False,
            progress=False,
            cycles=1,
            postselect=False,
            postselect_probability=0.0,
            perfect_correction=True,
            samples=1,
            init_mode="default",
            save=False,
            n_a=0.5,
            sequence="raster_y",
            meas_slab_only=True,
            batch_size=1,
            return_data=False,
            state_representation="auto",
            cycle_observer=observer,
            cycle_observer_cycles=[1],
            track_choi=False,
            return_native_state=False,
        )
        payload = observer.payload()
        assert result["samples"] == 1
        assert result["meas_slab_only_effective"] is truncation
        assert np.isfinite(payload["mutual_information_y0avg"]).all(), wall


def _fake_observer_payload(
    sample_count: int, *, offset: float = 0.0
) -> dict[str, np.ndarray]:
    values = np.linspace(0.1, 0.5, sample_count) + offset
    return {
        "mutual_information_y0avg": values,
        "entropy_a_y0avg": values + 1.0,
        "entropy_b_y0avg": values + 1.1,
        "entropy_union_y0avg": values + 2.1,
        "full_covariance_max_hermiticity_error": np.asarray(1.0e-14),
        "restricted_max_hermiticity_error": np.asarray(2.0e-14),
        "restricted_occupation_eigenvalue_min": np.asarray(-1.0e-14),
        "restricted_occupation_eigenvalue_max": np.asarray(1.0 + 1.0e-14),
        "restricted_eigensolve_count": np.asarray(3, dtype=np.int64),
        "materially_negative_mi_count": np.asarray(0, dtype=np.int64),
        "mutual_information_min": np.asarray(float(values.min())),
        "entropy_identity_max_abs_residual": np.asarray(0.0),
    }


def _write_fake_legacy_pair(
    output_root: Path, task, *, offset: float = 0.0
) -> dict[str, np.ndarray]:
    payload = _fake_observer_payload(task.sample_count, offset=offset)
    payload.update(
        {
            "schema": np.asarray(RUNNER.LEGACY_RESULT_SCHEMA),
            "observer_schema": np.asarray(RUNNER.OBSERVER_SCHEMA),
            "bundle": np.asarray(RUNNER.BUNDLE),
            "sampling_revision": np.asarray(RUNNER.LEGACY_REVISION),
            "canonical_entry_point": np.asarray(RUNNER.CANONICAL_ENTRY_POINT),
            "task_id": np.asarray(task.task_id),
            "Nx": np.asarray(RUNNER.EXPECTED_NX, dtype=np.int64),
            "Ny": np.asarray(task.ny, dtype=np.int64),
            "width": np.asarray(task.width, dtype=np.int64),
            "alpha_1": np.asarray(task.alpha_1, dtype=np.float64),
            "alpha_2": np.asarray(RUNNER.EXPECTED_ALPHA_2, dtype=np.float64),
            "wall": np.asarray(task.wall),
            "dw_truncation": np.asarray(task.dw_truncation, dtype=np.bool_),
            "nshell": np.asarray(RUNNER.EXPECTED_NSHELL, dtype=np.int64),
            "batch_index": np.asarray(task.batch_index, dtype=np.int64),
            "sample_start": np.asarray(task.sample_start, dtype=np.int64),
            "sample_stop": np.asarray(task.sample_stop, dtype=np.int64),
            "endpoint_cycle": np.asarray(task.cycles, dtype=np.int64),
            "batch_seed": np.asarray(task.seed, dtype=np.int64),
            "global_sample_indices": np.asarray(
                task.global_sample_indices, dtype=np.int64
            ),
        }
    )
    result_path, completion_path = RUNNER.legacy_task_paths(output_root, task)
    RUNNER._write_npz(result_path, payload)
    completion = RUNNER._legacy_completion_identity(task)
    completion.update(
        {
            "source_hashes": {name: "a" * 64 for name in RUNNER.SOURCE_FILES},
            "result_filename": result_path.name,
            "result_bytes": result_path.stat().st_size,
            "result_sha256": RUNNER.sha256_file(result_path),
            "elapsed_seconds": 1.0,
        }
    )
    RUNNER._write_json(completion_path, completion)
    return payload


def test_completion_pair_verification_detects_missing_and_corrupt_data(
    tmp_path: Path,
) -> None:
    config, _ = _notebook_config()
    task = RUNNER.expand_tasks(config)[0]
    hashes = RUNNER.source_hashes()
    config_sha256 = RUNNER.config_hash(config)
    computed_indices = task.global_sample_indices
    computed_rng_seed = RUNNER.computed_seed(task, computed_indices)
    payload = RUNNER.merge_task_payload(
        task=task,
        legacy_inputs=[],
        computed_indices=computed_indices,
        computed_payload=_fake_observer_payload(task.sample_count),
        computed_elapsed_seconds=1.0,
        computed_rng_seed=computed_rng_seed,
    )
    RUNNER._save_task(
        output_root=tmp_path / "drive",
        scratch_root=tmp_path / "scratch",
        task=task,
        payload=payload,
        elapsed_seconds=1.0,
        legacy_inputs=[],
        computed_indices=computed_indices,
        computed_rng_seed=computed_rng_seed,
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
    assert not complete and reason == "incomplete result/completion pair"
    completion_path.write_bytes(completion_raw)
    result_path.write_bytes(result_path.read_bytes() + b"corrupt")
    complete, reason = RUNNER.verified_complete(
        output_root=tmp_path / "drive",
        task=task,
        config_sha256=config_sha256,
        hashes=hashes,
    )
    assert not complete and "byte count mismatch" in reason


def test_legacy_pair_is_verified_and_checksum_corruption_is_rejected(
    tmp_path: Path,
) -> None:
    config, _ = _notebook_config()
    macro = RUNNER.expand_tasks(config)[0]
    legacy_task = RUNNER.legacy_tasks_for_macro(macro)[0]
    expected = _write_fake_legacy_pair(tmp_path / "legacy", legacy_task, offset=3.0)
    record, reason = RUNNER.verified_legacy_record(
        output_root=tmp_path / "legacy", task=legacy_task
    )
    assert record is not None and reason == "verified"
    loaded = RUNNER.load_legacy_payload(record)
    for key in RUNNER.TRAJECTORY_KEYS:
        np.testing.assert_array_equal(loaded[key], expected[key])

    record.result_path.write_bytes(record.result_path.read_bytes() + b"corrupt")
    rejected, reason = RUNNER.verified_legacy_record(
        output_root=tmp_path / "legacy", task=legacy_task
    )
    assert rejected is None
    assert "byte count mismatch" in reason


def test_partial_legacy_macro_computes_only_missing_samples(
    tmp_path: Path, monkeypatch
) -> None:
    config, _ = _notebook_config()
    macro = RUNNER.expand_tasks(config)[0]
    legacy_tasks = RUNNER.legacy_tasks_for_macro(macro)
    written: list[tuple[object, dict[str, np.ndarray]]] = []
    for legacy_task in (legacy_tasks[0], legacy_tasks[2]):
        payload = _write_fake_legacy_pair(
            tmp_path / "legacy",
            legacy_task,
            offset=10.0 + legacy_task.batch_index,
        )
        written.append((legacy_task, payload))

    calls: list[tuple[int, ...]] = []
    model_token = object()
    monkeypatch.setattr(RUNNER, "validate_a100", lambda: {"name": "fake A100"})
    monkeypatch.setattr(RUNNER, "_check_space", lambda *args, **kwargs: 10**12)
    monkeypatch.setattr(RUNNER, "build_model", lambda config, task: model_token)

    def fake_run_missing_samples(*, model, config, task, sample_indices):
        assert model is model_token
        calls.append(sample_indices)
        seed = RUNNER.computed_seed(task, sample_indices)
        return _fake_observer_payload(len(sample_indices), offset=100.0), 2.0, seed

    monkeypatch.setattr(RUNNER, "run_missing_samples", fake_run_missing_samples)
    summary = RUNNER.run_campaign(
        config=config,
        output_root=tmp_path / "v2",
        legacy_output_root=tmp_path / "legacy",
        scratch_root=tmp_path / "scratch",
        max_new_tasks=1,
    )
    expected_imported = {
        index
        for legacy_task, _ in written
        for index in legacy_task.global_sample_indices
    }
    expected_missing = tuple(
        index for index in macro.global_sample_indices if index not in expected_imported
    )
    assert summary["status"] == "partial_limit_reached"
    assert summary["legacy_samples_imported"] == 10
    assert calls == [expected_missing]
    assert len(calls[0]) == 40

    result_path, _ = RUNNER.task_paths(tmp_path / "v2", macro)
    with np.load(result_path, allow_pickle=False) as archive:
        origins = np.asarray(archive["trajectory_origin"])
        assert np.count_nonzero(origins == "legacy_v1") == 10
        assert np.count_nonzero(origins == "computed_v2") == 40
        np.testing.assert_array_equal(
            archive["computed_sample_indices"], expected_missing
        )
        for legacy_task, legacy_payload in written:
            positions = (
                np.asarray(legacy_task.global_sample_indices) - macro.sample_start
            )
            for key in RUNNER.TRAJECTORY_KEYS:
                np.testing.assert_array_equal(
                    archive[key][positions], legacy_payload[key]
                )


def test_fully_legacy_macro_publishes_without_constructing_gpu_model(
    tmp_path: Path, monkeypatch
) -> None:
    config, _ = _notebook_config()
    macro = RUNNER.expand_tasks(config)[0]
    for legacy_task in RUNNER.legacy_tasks_for_macro(macro):
        _write_fake_legacy_pair(
            tmp_path / "legacy",
            legacy_task,
            offset=float(legacy_task.batch_index),
        )

    monkeypatch.setattr(RUNNER, "validate_a100", lambda: {"name": "fake A100"})
    monkeypatch.setattr(RUNNER, "_check_space", lambda *args, **kwargs: 10**12)

    def unexpected_model(*args, **kwargs):
        raise AssertionError("a fully imported macro must not construct a model")

    monkeypatch.setattr(RUNNER, "build_model", unexpected_model)
    summary = RUNNER.run_campaign(
        config=config,
        output_root=tmp_path / "v2",
        legacy_output_root=tmp_path / "legacy",
        scratch_root=tmp_path / "scratch",
        max_new_tasks=1,
    )
    assert summary["status"] == "partial_limit_reached"
    assert summary["legacy_samples_imported"] == macro.sample_count
    result_path, completion_path = RUNNER.task_paths(tmp_path / "v2", macro)
    assert result_path.is_file() and completion_path.is_file()
    with np.load(result_path, allow_pickle=False) as archive:
        assert np.all(archive["trajectory_origin"] == "legacy_v1")
        assert archive["computed_sample_indices"].size == 0
        assert int(archive["computed_rng_seed"].item()) == -1


def test_report_only_inventory_is_read_only_and_reports_gpu_work(
    tmp_path: Path,
) -> None:
    config, _ = _notebook_config()
    output_root = tmp_path / "v2"
    legacy_root = tmp_path / "legacy"
    summary = RUNNER.run_campaign(
        config=config,
        output_root=output_root,
        legacy_output_root=legacy_root,
        scratch_root=tmp_path / "scratch",
        report_only=True,
    )
    assert summary["status"] == "report_only"
    assert summary["tasks"] == 420
    assert summary["pending"] == 420
    assert summary["gpu_trajectories_remaining"] == 12600
    assert not output_root.exists()
    assert not legacy_root.exists()


def test_missing_samples_are_sent_to_one_full_gpu_batch(monkeypatch) -> None:
    config, _ = _notebook_config()
    tasks = RUNNER.expand_tasks(config)
    selected = [
        next(task for task in tasks if task.ny == ny)
        for ny in RUNNER.EXPECTED_NY_VALUES
    ]
    captured: list[dict] = []

    class FakeObserver:
        def __init__(self, *, samples, **kwargs):
            self.samples = samples

        def payload(self):
            return _fake_observer_payload(self.samples)

    class FakeModel:
        device = "cpu"

        def run_markov_circuit(self, **kwargs):
            captured.append(kwargs)
            return {"samples": kwargs["samples"], "choi_tracked": False}

    monkeypatch.setattr(RUNNER, "FixedWidthMutualInformationObserver", FakeObserver)
    for task in selected:
        payload, elapsed, seed = RUNNER.run_missing_samples(
            model=FakeModel(),
            config=config,
            task=task,
            sample_indices=task.global_sample_indices,
        )
        assert payload["mutual_information_y0avg"].shape == (task.sample_count,)
        assert elapsed >= 0.0
        assert seed == RUNNER.computed_seed(task, task.global_sample_indices)

    assert [call["samples"] for call in captured] == [50, 25, 25]
    assert [call["batch_size"] for call in captured] == [50, 25, 25]
    assert all(call["G_history"] is False for call in captured)
    assert all(call["save"] is False for call in captured)
    assert [call["cycle_observer_cycles"] for call in captured] == [
        [40],
        [48],
        [56],
    ]


def test_bundle_sources_match_canonical_and_notebook_contract_compiles() -> None:
    assert _sha256(BUNDLE / "src/classA_U1FGTN_gpu.py") == _sha256(
        REPO / "src/fgtn/classA_U1FGTN_gpu.py"
    )
    assert _sha256(BUNDLE / "src/occupied_frame_gpu.py") == _sha256(
        REPO / "src/fgtn/occupied_frame_gpu.py"
    )
    code_sources = [
        _cell_source(cell)
        for cell in _notebook()["cells"]
        if cell.get("cell_type") == "code"
    ]
    for index, source in enumerate(code_sources):
        compile(source, f"{NOTEBOOK_PATH}#code-{index}", "exec")
    joined = "\n".join(code_sources)
    assert joined.count("drive.mount('/content/drive')") == 1
    assert "shutil.copytree(BUNDLE_DRIVE_DIR, LOCAL_BUNDLE_DIR)" in joined
    assert "The Drive bundle is stale" in joined
    assert "expected_revision not in runner_source" in joined
    assert "subprocess.Popen(" in joined
    assert "stderr=subprocess.STDOUT" in joined
    assert "--legacy-output-root" in joined
    assert (
        "from google.colab import runtime\nruntime.unassign()\nprint('done')" in joined
    )
    assert "googleapiclient" not in joined
    assert "lease" not in joined.lower()


def test_runner_uses_endpoint_canonical_covariance_observer() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert ".run_markov_circuit(" in source
    assert "G_history=False" in source
    assert "save=False" in source
    assert "cycle_observer=observer" in source
    assert "cycle_observer_cycles=[task.cycles]" in source
    assert 'sequence=config["protocol"]["sequence"]' in source
    assert 'meas_slab_only=config["protocol"]["meas_slab_only"]' in source
    assert "native_cycle_observer=observer" not in source
    assert "DriveRemoteCommit" not in source
    assert "checkpoint" not in source.lower()


def test_hard_wall_figure_layer_verifies_and_averages_completed_lane(
    tmp_path: Path,
) -> None:
    manifest = FIGURES.verify_download_manifest(FIGURES.DEFAULT_INPUT_ROOT)
    assert manifest["files"] == 420
    assert manifest["pairs"] == 210
    assert manifest["trajectories"] == 6300
    data, validation = FIGURES.load_verified_hard_wall(FIGURES.DEFAULT_INPUT_ROOT)
    assert data["mutual_information_y0avg"].shape == (3, 21, 100)
    assert validation["materially_negative_mi_count"] == 0
    assert validation["trajectory_origins"] == {
        "computed_v2": 4465,
        "legacy_v1": 1835,
    }
    summary = FIGURES.summarize(data)
    assert summary["mean"].shape == (3, 21)
    maximum_index = np.unravel_index(np.argmax(summary["mean"]), summary["mean"].shape)
    assert FIGURES.NY_VALUES[maximum_index[0]] == 20
    assert FIGURES.ALPHA_VALUES[maximum_index[1]] == 1.7
    np.testing.assert_allclose(
        summary["mean"][maximum_index], 0.624591546505746, rtol=0.0, atol=1.0e-14
    )

    outputs = FIGURES.make_sample_average_figure(summary, tmp_path)
    outputs.extend(FIGURES.make_geometry_figure(tmp_path))
    csv_path = FIGURES.write_summary_csv(summary, tmp_path)
    for path in (*outputs, csv_path):
        assert Path(path).is_file() and Path(path).stat().st_size > 100
    assert len(Path(csv_path).read_text(encoding="utf-8").splitlines()) == 64


def test_hard_wall_figure_layer_has_no_flattened_benchmark_overlay() -> None:
    source = FIGURE_SCRIPT_PATH.read_text(encoding="utf-8").lower()
    assert "flattened_ground_state" not in source
    assert "cft_mutual_information" not in source
    assert "error bars: $\\pm1$ s.e.m." in source
