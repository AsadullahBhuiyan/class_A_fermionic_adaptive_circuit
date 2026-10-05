from __future__ import annotations

import ast
import codecs
import hashlib
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
PARENT = REPO / "00_WORKSPACE/CURRENT/final_production_new_designs"
BUNDLE = PARENT / "04_maxmix_manybody_lyapunov_pilot"
RUNNER_PATH = BUNDLE / "run_campaign.py"
OBSERVER_PATH = BUNDLE / "lyapunov_observer.py"
ANALYSIS_PATH = BUNDLE / "analyze_campaign.py"
NOTEBOOK_PATH = BUNDLE / "run_maxmix_manybody_lyapunov_pilot.ipynb"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


old_path = list(sys.path)
old_observer = sys.modules.get("lyapunov_observer")
old_runner = sys.modules.get("run_campaign")
try:
    sys.path.insert(0, str(BUNDLE))
    OBSERVER = _load(OBSERVER_PATH, "lyapunov_observer")
    RUNNER = _load(RUNNER_PATH, "tested_maxmix_lyapunov_gpu_runner")
    ANALYSIS = _load(ANALYSIS_PATH, "tested_maxmix_lyapunov_gpu_analysis")
finally:
    sys.path[:] = old_path
    if old_observer is None:
        sys.modules.pop("lyapunov_observer", None)
    else:
        sys.modules["lyapunov_observer"] = old_observer
    if old_runner is None:
        sys.modules.pop("run_campaign", None)
    else:
        sys.modules["run_campaign"] = old_runner


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _notebook_config() -> tuple[dict, dict]:
    fake_drive = types.SimpleNamespace(mount=lambda *_args, **_kwargs: None)
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.drive = fake_drive
    old_google = sys.modules.get("google")
    old_colab = sys.modules.get("google.colab")
    sys.modules["google"] = google
    sys.modules["google.colab"] = colab
    try:
        for cell in _notebook()["cells"]:
            source = _cell_source(cell)
            if cell.get("cell_type") == "code" and "CONFIG = {" in source:
                namespace: dict = {}
                exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)
                return namespace["CONFIG"], namespace
    finally:
        if old_google is None:
            sys.modules.pop("google", None)
        else:
            sys.modules["google"] = old_google
        if old_colab is None:
            sys.modules.pop("google.colab", None)
        else:
            sys.modules["google.colab"] = old_colab
    raise AssertionError("notebook configuration cell was not found")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_notebook_exposes_exact_locked_campaign() -> None:
    config, namespace = _notebook_config()
    assert RUNNER.validate_config(config) == RUNNER.expected_config()
    assert config["Nx"] == 20
    assert config["Ny_values"] == [20, 22, 24, 26, 28, 30, 36, 40]
    assert config["samples_per_Ny"] == 100
    assert config["shard_size"] == 5
    assert config["cycles_rule"] == "2*Ny"
    assert config["device"] == "cuda:0"
    assert config["dtype"] == "complex128"
    assert config["protocol"] == RUNNER.expected_config()["protocol"]
    assert namespace["REPORT_ONLY"] is False
    assert namespace["MAX_NEW_TASKS"] is None
    assert namespace["RUN_ANALYSIS"] is False
    assert namespace["NOTEBOOK_BUILD"] == "lyapunov-v2-a100-cap-roundoff-fix"


def test_task_table_has_160_batches_and_800_trajectories() -> None:
    tasks = RUNNER.expand_tasks(RUNNER.expected_config())
    assert len(tasks) == 160
    assert len({task.task_id for task in tasks}) == 160
    assert len({task.seed for task in tasks}) == 160
    assert sum(task.sample_count for task in tasks) == 800
    assert tasks[0].ny == 40
    for ny in RUNNER.EXPECTED_NY_VALUES:
        selected = [task for task in tasks if task.ny == ny]
        assert len(selected) == 20
        assert [
            sample for task in selected for sample in task.global_sample_indices
        ] == list(range(100))
        assert {task.cycles for task in selected} == {2 * ny}


def test_bundle_engine_copies_match_canonical_sources() -> None:
    assert _sha256(BUNDLE / "src/classA_U1FGTN_gpu.py") == _sha256(
        REPO / "src/fgtn/classA_U1FGTN_gpu.py"
    )
    assert _sha256(BUNDLE / "src/occupied_frame_gpu.py") == _sha256(
        REPO / "src/fgtn/occupied_frame_gpu.py"
    )


def test_leading_levels_match_exhaustive_fock_products_and_normalization() -> None:
    nu = np.asarray([0.2, 0.7, 0.0, 1.0], dtype=np.float64)
    log_z = 1.7
    exhaustive = []
    weights = []
    for bits in itertools.product((0, 1), repeat=nu.size):
        weight = 1.0
        for bit, occupation in zip(bits, nu):
            weight *= occupation if bit else 1.0 - occupation
        weights.append(weight)
        exhaustive.append(-np.inf if weight == 0 else log_z + np.log(weight))
    exhaustive = np.sort(np.asarray(exhaustive))[::-1]
    generated = OBSERVER.leading_log_sigma2_levels(nu, log_z, count=16)
    np.testing.assert_allclose(generated, exhaustive, rtol=0.0, atol=1.0e-14)
    assert np.isclose(np.sum(np.exp(exhaustive)), np.exp(log_z))
    _, costs, caps = OBSERVER.natural_spectrum_factors(nu)
    assert caps.tolist() == [False, False, True, True]
    assert np.isinf(costs[caps]).all()


def test_analysis_accepts_roundoff_but_rejects_changed_saved_levels() -> None:
    stored = np.asarray([0.0, -5204.893919400649, -np.inf])
    roundoff = stored.copy()
    roundoff[1] = np.nextafter(roundoff[1], np.inf)
    assert ANALYSIS.reconstructed_levels_match(roundoff, stored)

    changed = stored.copy()
    changed[1] += 1.0e-8
    assert not ANALYSIS.reconstructed_levels_match(changed, stored)


def test_analysis_fits_only_preregistered_finite_leading_levels() -> None:
    cycles = np.arange(0, 41, 4, dtype=np.float64)
    finite = np.stack([-(index + 1.0) * cycles for index in range(5)], axis=1)
    padded = np.full((cycles.size, 64), -np.inf)
    padded[:, :5] = finite
    middle, late = ANALYSIS.trajectory_window_slopes(cycles, padded, ny=20)
    np.testing.assert_allclose(middle, -np.arange(1.0, 6.0))
    np.testing.assert_allclose(late, -np.arange(1.0, 6.0))


def test_a100_roundoff_scale_occupation_caps_are_snapped_and_recorded() -> None:
    nx, ny = 20, 2
    active_indices = _active_indices(nx, ny)
    observer = OBSERVER.BatchedActiveSpectrumObserver(
        nx=nx,
        ny=ny,
        cycles=2 * ny,
        samples=1,
        active_indices=active_indices,
        full_mode_count=2 * nx * ny,
        wall_locations=(5, 15),
        cap_tolerance=1.0e-9,
    )
    # Model the production A100 failure: Hermitian ``eigh`` returned an exact
    # cap a few 1e-10 outside [0, 1].  This is roundoff, while a >1e-9
    # violation remains a hard failure.
    correlation = torch.full(
        (2 * nx * ny,), 0.5, dtype=torch.float64
    )
    correlation[int(active_indices[0])] = -5.0e-10
    correlation[int(active_indices[1])] = 1.0 + 5.0e-10
    centered = torch.diag(2.0 * correlation - 1.0).to(torch.complex128)[None]
    observer.cumulative_log_probability[0, 4] = 0.0
    position = observer._spectrum_position[4]
    observer._observe_spectrum(position, 4, centered)
    assert observer.occupations[0, position, 0] == 0.0
    assert observer.occupations[0, position, -1] == 1.0
    assert observer.cap_mask[0, position, [0, -1]].tolist() == [True, True]
    assert observer.occupation_bound_residual[0, position] == pytest.approx(5.0e-10)
    assert np.isfinite(observer.leading_log_sigma2[0, position]).all()

    correlation[int(active_indices[0])] = -2.0e-9
    invalid = torch.diag(2.0 * correlation - 1.0).to(torch.complex128)[None]
    with pytest.raises(FloatingPointError, match="roundoff allowance"):
        observer._observe_spectrum(position, 4, invalid)


def _active_indices(nx: int, ny: int) -> torch.Tensor:
    return torch.as_tensor(
        [
            mu + 2 * x + 2 * nx * y
            for y in range(ny)
            for x in range(5, 16)
            for mu in (0, 1)
        ],
        dtype=torch.long,
    )


def test_observer_covers_every_cycle_and_identity_initial_spectrum() -> None:
    nx, ny, samples = 20, 2, 2
    cycles = 2 * ny
    observer = OBSERVER.BatchedActiveSpectrumObserver(
        nx=nx,
        ny=ny,
        cycles=cycles,
        samples=samples,
        active_indices=_active_indices(nx, ny),
        full_mode_count=2 * nx * ny,
        wall_locations=(5, 15),
    )
    centered = torch.zeros(
        samples, 2 * nx * ny, 2 * nx * ny, dtype=torch.complex128
    )
    observer.observe(cycle=0, G=centered, batch_start=0, batch_count=samples)
    for cycle in range(1, cycles + 1):
        for site in range(11 * ny):
            observer.record_event(
                cycle=cycle,
                sample_offsets=torch.arange(samples),
                conditional_log_probability=torch.zeros(
                    samples, 4, dtype=torch.float64
                ),
                update_index=site,
            )
        observer.observe(
            cycle=cycle, G=centered, batch_start=0, batch_count=samples
        )
    payload = observer.result_arrays()
    assert payload["cycles"].tolist() == list(range(cycles + 1))
    assert payload["cumulative_log_probability"].shape == (samples, cycles + 1)
    assert payload["occupations"].shape[-1] == 22 * ny
    np.testing.assert_allclose(payload["occupations"][:, 0], 0.5, atol=0.0)
    np.testing.assert_allclose(payload["leading_log_sigma2"][:, 0], 0.0, atol=1e-13)
    assert np.all(payload["site_event_count"][:, 1:] == 11 * ny)
    assert np.all(payload["channel_event_count"][:, 1:] == 44 * ny)


def test_real_engine_observer_leaves_state_and_rng_evolution_unchanged() -> None:
    def run(with_observer: bool):
        seed = 90304
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = RUNNER.classA_U1FGTN_gpu(
            Nx=20,
            Ny=2,
            DW=True,
            nshell=1,
            filling_frac=0.5,
            alpha_1=1.0,
            alpha_2=30.0,
            trial_orbitals="X",
            dw_truncation=True,
            triv_region_local_mode=False,
            device="cpu",
            dtype="complex128",
            backend="local",
        )
        observer = OBSERVER.BatchedActiveSpectrumObserver(
            nx=20,
            ny=2,
            cycles=4,
            samples=1,
            active_indices=model.active_top_layer_indices(True),
            full_mode_count=model.Nlayer,
            wall_locations=tuple(model.DW_loc),
        )
        result = model.run_markov_circuit(
            G_history=True,
            progress=False,
            cycles=4,
            postselect=False,
            postselect_probability=0.0,
            perfect_correction=True,
            samples=1,
            init_mode="maxmix",
            save=False,
            n_a=0.5,
            sequence="raster_y",
            meas_slab_only=True,
            batch_size=1,
            return_data=True,
            state_representation="covariance",
            cycle_observer=observer.observe if with_observer else None,
            record_observer=observer.record_event if with_observer else None,
            track_choi=False,
            return_native_state=False,
        )
        return np.asarray(result["G_hist"]), torch.get_rng_state().clone()

    observed_history, observed_rng = run(True)
    reference_history, reference_rng = run(False)
    np.testing.assert_array_equal(observed_history, reference_history)
    torch.testing.assert_close(observed_rng, reference_rng, rtol=0.0, atol=0.0)


def _fake_payload(task) -> dict[str, np.ndarray]:
    return {
        "schema": np.asarray(RUNNER.RESULT_SCHEMA),
        "task_id": np.asarray(task.task_id),
        "values": np.arange(task.sample_count, dtype=np.float64),
    }


def test_completion_resume_and_checksum_failure(tmp_path: Path) -> None:
    task = RUNNER.expand_tasks(RUNNER.expected_config())[0]
    output = tmp_path / "output"
    scratch = tmp_path / "scratch"
    hashes = RUNNER.source_hashes()
    config_sha = RUNNER.config_hash(RUNNER.expected_config())
    RUNNER.save_task(
        output_root=output,
        scratch_root=scratch,
        task=task,
        payload=_fake_payload(task),
        elapsed_seconds=1.0,
        config_sha256=config_sha,
        hashes=hashes,
    )
    assert RUNNER.verified_complete(
        output_root=output,
        task=task,
        config_sha256=config_sha,
        hashes=hashes,
    ) == (True, "verified")
    result_path, completion_path = RUNNER.task_paths(output, task)
    result_path.write_bytes(result_path.read_bytes() + b"corrupt")
    valid, reason = RUNNER.verified_complete(
        output_root=output,
        task=task,
        config_sha256=config_sha,
        hashes=hashes,
    )
    assert not valid and "byte-count mismatch" in reason
    result_path.unlink()
    valid, reason = RUNNER.verified_complete(
        output_root=output,
        task=task,
        config_sha256=config_sha,
        hashes=hashes,
    )
    assert not valid and "incomplete" in reason
    assert completion_path.is_file()


def test_empty_resume_scan_is_visible_and_avoids_taskwise_drive_probes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tasks = RUNNER.expand_tasks(RUNNER.expected_config())

    def unexpected_probe(**_kwargs):
        raise AssertionError("empty output must not probe 160 absent file pairs")

    monkeypatch.setattr(RUNNER, "verified_complete", unexpected_probe)
    inventory = RUNNER.scan_resume_inventory(
        output_root=tmp_path / "never_created",
        tasks=tasks,
        config_sha256=RUNNER.config_hash(RUNNER.expected_config()),
        hashes=RUNNER.source_hashes(),
    )
    assert len(inventory) == 160
    assert not any(valid for _, valid, _ in inventory)
    output = capsys.readouterr().out
    assert "[resume scan] checking 160 deterministic task slots" in output
    assert "Resume scan" in output


def test_startup_report_precedes_resume_scan_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(RUNNER.expected_config()), encoding="utf-8")

    def fail_scan(**_kwargs):
        raise OSError("synthetic Drive listing failure")

    monkeypatch.setattr(RUNNER, "scan_resume_inventory", fail_scan)
    with pytest.raises(OSError, match="synthetic Drive listing failure"):
        RUNNER.main(
            [
                "--config",
                str(config_path),
                "--output-root",
                str(tmp_path / "output"),
                "--scratch-root",
                str(tmp_path / "scratch"),
                "--report-only",
            ]
        )
    output = capsys.readouterr().out
    assert output.startswith("[startup] Lyapunov campaign runner entered")
    assert "[resolved configuration]" in output
    assert "[workload] sizes=8, tasks=160, trajectories=800" in output


def test_publish_failure_never_creates_stable_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "local.bin"
    final = tmp_path / "drive" / "final.bin"
    local.write_bytes(b"correct payload")
    original = RUNNER.shutil.copyfile

    def corrupt_copy(source, destination):
        result = original(source, destination)
        Path(destination).write_bytes(b"wrong payload")
        return result

    monkeypatch.setattr(RUNNER.shutil, "copyfile", corrupt_copy)
    with pytest.raises(OSError, match="byte-count mismatch|checksum mismatch"):
        RUNNER.publish_file(local, final)
    assert not final.exists()


def test_analysis_window_slopes_and_finite_size_contract() -> None:
    cycles = OBSERVER.spectrum_checkpoint_cycles(20)
    slopes = np.asarray([-2.0, -2.1, -2.2, -2.3, -2.4])
    levels = 0.7 + cycles[:, None] * slopes[None, :]
    middle, late = ANALYSIS.trajectory_window_slopes(cycles, levels, ny=20)
    np.testing.assert_allclose(middle, slopes, atol=1e-14)
    np.testing.assert_allclose(late, slopes, atol=1e-14)
    ny = np.asarray([20, 22, 24, 26, 28, 30, 36, 40])
    lambda_values = np.stack(
        [-(0.4 * value + 3.0 / value) - 0.2 * np.arange(5) / value for value in ny]
    )
    fit = ANALYSIS.fit_finite_size(ny, lambda_values)
    assert fit["Ny"] == ny.tolist()
    assert "alpha_c_eff" in fit["primary"]
    assert len(fit["gaps"]) == 4


def test_analysis_loads_batched_completion_products(tmp_path: Path) -> None:
    config = RUNNER.expected_config()
    hashes = RUNNER.source_hashes()
    config_sha = RUNNER.config_hash(config)
    results = tmp_path / "results"
    scratch = tmp_path / "scratch"
    tasks = RUNNER.expand_tasks(config)
    for ny in RUNNER.EXPECTED_NY_VALUES:
        task = next(task for task in tasks if task.ny == ny)
        spectrum_cycles = OBSERVER.spectrum_checkpoint_cycles(ny)
        n_eff = 22 * ny
        occupations = np.full(
            (task.sample_count, spectrum_cycles.size, n_eff), 0.4
        )
        log_z = np.broadcast_to(
            n_eff * np.log(2.0) - 0.2 * spectrum_cycles[None, :],
            (task.sample_count, spectrum_cycles.size),
        ).copy()
        levels = np.empty((task.sample_count, spectrum_cycles.size, 64))
        for sample in range(task.sample_count):
            for position in range(spectrum_cycles.size):
                levels[sample, position] = OBSERVER.leading_log_sigma2_levels(
                    occupations[sample, position], log_z[sample, position]
                )
        omega = np.broadcast_to(
            -0.2 * np.arange(2 * ny + 1)[None, :],
            (task.sample_count, 2 * ny + 1),
        ).copy()
        payload = {
            "spectrum_cycles": spectrum_cycles,
            "occupations": occupations,
            "log_z": log_z,
            "leading_log_sigma2": levels,
            "cumulative_log_probability": omega,
            "soft_mode_x_profiles": np.full(
                (task.sample_count, spectrum_cycles.size, 16, 20), 0.05
            ),
        }
        RUNNER.save_task(
            output_root=results,
            scratch_root=scratch,
            task=task,
            payload=payload,
            elapsed_seconds=1.0,
            config_sha256=config_sha,
            hashes=hashes,
        )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    analysis_output = tmp_path / "analysis"
    assert ANALYSIS.main(
        [
            "--config",
            str(config_path),
            "--results-root",
            str(results),
            "--output-root",
            str(analysis_output),
            "--bootstrap-count",
            "10",
            "--allow-incomplete",
        ]
    ) == 0
    summary = json.loads(
        (analysis_output / "analysis_summary.json").read_text(encoding="utf-8")
    )
    assert summary["verified_tasks"] == 8
    assert summary["verified_trajectories"] == 40
    assert summary["requested_trajectories"] == 800
    assert len(summary["finite_size"]["Ny"]) == 8


def test_notebook_stages_locally_surfaces_progress_and_disconnects() -> None:
    notebook = _notebook()
    joined = "\n".join(_cell_source(cell) for cell in notebook["cells"])
    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "drive.mount('/content/drive')" in joined
    assert "shutil.copytree(BUNDLE_DRIVE_DIR, LOCAL_BUNDLE_DIR)" in joined
    assert "sys.executable, '-u'" in joined
    assert "REPORT_ONLY = False" in joined
    assert "MAX_NEW_TASKS = None" in joined
    assert "tqdm(" in runner_source
    assert "progress=True" in runner_source
    assert "[task start]" in runner_source
    assert "subprocess.Popen(" in joined
    assert "stderr=subprocess.STDOUT" in joined
    assert "os.read(process.stdout.fileno(), 4096)" in joined
    assert "[runner failed] child exit code=" in joined
    assert "run_streaming_child(command)" in joined
    assert "run_streaming_child(analysis_command)" in joined
    assert "subprocess.run(command, check=True)" not in joined
    final = _cell_source(notebook["cells"][-1]).strip()
    assert final == (
        "from google.colab import runtime\n"
        "runtime.unassign()\n"
        "print('done')"
    )


def test_notebook_streamer_surfaces_carriage_returns_and_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    notebook = _notebook()
    launch_source = next(
        _cell_source(cell)
        for cell in notebook["cells"]
        if "def run_streaming_child(command):" in _cell_source(cell)
    )
    tree = ast.parse(launch_source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_streaming_child"
    )
    namespace = {
        "codecs": codecs,
        "os": os,
        "subprocess": subprocess,
        "sys": sys,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<streamer>", "exec"), namespace)
    namespace["run_streaming_child"](
        [
            sys.executable,
            "-u",
            "-c",
            (
                "import sys; "
                "sys.stdout.write('outer 1/2\\r'); sys.stdout.flush(); "
                "sys.stderr.write('child diagnostic\\n'); sys.stderr.flush()"
            ),
        ]
    )
    output = capsys.readouterr().out
    assert "outer 1/2\r" in output
    assert "child diagnostic\n" in output
