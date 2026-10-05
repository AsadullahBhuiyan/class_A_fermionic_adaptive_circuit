from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = REPO_ROOT / "00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot"
RUNNER_PATH = PILOT_ROOT / "run_kato_overlap_continuation_s25.py"
CONFIG_PATH = PILOT_ROOT / "campaign_config.kato_overlap_continuation_n20x24_n24x24_s25_v1.json"


def _runner():
    name = "kato_overlap_continuation_s25_under_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(PILOT_ROOT))
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _toy_config() -> dict:
    config = _config()
    config["continuation"]["grid_intervals"] = 4
    config["continuation"]["adaptive_tolerance"] = 1e-7
    config["continuation"]["minimum_step_fraction_of_interval"] = 1.0 / 4096.0
    config["execution"]["checkpoint_every_intervals"] = 2
    config["acceptance"]["transported_selected_projector_mismatch_tolerance"] = 1e-5
    return config


def _toy_frame() -> np.ndarray:
    rng = np.random.default_rng(8721)
    raw = rng.normal(size=(8, 3)) + 1j * rng.normal(size=(8, 3))
    frame, _ = np.linalg.qr(raw)
    return np.asarray(frame, dtype=np.complex128, order="F")


def _toy_task() -> dict:
    return {
        "task_id": "kato_toy_soft_ccw_sample_000", "size": "toy", "Nx": 2, "Ny": 2,
        "wall": "soft", "direction": "ccw", "sigma": 1, "sample_id": 0,
        "source_task_id": "burnin_soft_sample_000",
    }


def test_locked_task_table_and_flux_convention() -> None:
    runner = _runner()
    config = _config()
    runner.validate_config(config)
    tasks = runner.tasks(config)
    assert len(tasks) == len({row["task_id"] for row in tasks}) == 200
    assert {row["size"] for row in tasks} == {"N20x24", "N24x24"}
    assert {row["wall"] for row in tasks} == {"soft", "hard"}
    assert {row["direction"] for row in tasks} == {"ccw", "cw"}
    assert {row["sample_id"] for row in tasks} == set(range(0, 100, 4))
    ccw, cw = runner.flux_grid(config, 1), runner.flux_grid(config, -1)
    assert ccw.shape == cw.shape == (129,)
    assert np.array_equal(cw, -ccw)
    assert ccw[0] == -1e-7
    assert np.isclose(ccw[-1] - ccw[0], 2 * np.pi)
    assert config["execution"] == {
        "cpu_list": "28-55", "workers": 28, "blas_threads": 1,
        "checkpoint_every_intervals": 8,
    }


def test_projector_overlap_selector_equals_legacy_frame_rule() -> None:
    runner = _runner()
    rng = np.random.default_rng(182)
    previous, _ = np.linalg.qr(rng.normal(size=(12, 5)) + 1j * rng.normal(size=(12, 5)))
    eigenvectors, _ = np.linalg.qr(rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12)))
    eigenvalues = np.linspace(-2.0, 2.0, 12)
    selected, indices, _, diagnostics = runner.select_by_previous_projector(
        previous, eigenvalues, eigenvectors, 5
    )
    legacy_weights = np.real(np.sum(np.abs(previous.conj().T @ eigenvectors) ** 2, axis=0))
    legacy_indices = np.lexsort((eigenvalues, -legacy_weights))[:5]
    assert np.array_equal(indices, legacy_indices)
    assert np.max(np.abs(selected @ selected.conj().T - eigenvectors[:, legacy_indices] @ eigenvectors[:, legacy_indices].conj().T)) < 1e-12
    assert diagnostics["selected_weight_margin"] == pytest.approx(
        np.min(legacy_weights[legacy_indices]) - np.max(np.delete(legacy_weights, legacy_indices))
    )


def _rotating_spectrum(theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vector = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.complex128)
    derivative = np.asarray([-np.sin(theta), np.cos(theta)], dtype=np.complex128)
    projector = np.outer(vector, vector.conj())
    projector_derivative = np.outer(derivative, vector.conj()) + np.outer(vector, derivative.conj())
    hamiltonian = np.eye(2) - 2.0 * projector
    hamiltonian_derivative = -2.0 * projector_derivative
    return vector[:, None], projector, hamiltonian, hamiltonian_derivative


def _rotating_generator(runner, theta: float, previous: np.ndarray) -> np.ndarray:
    _, _, hamiltonian, derivative = _rotating_spectrum(theta)
    values, vectors = np.linalg.eigh(hamiltonian)
    _, selected, excluded, _ = runner.select_by_previous_projector(previous, values, vectors, 1)
    _, generator, _ = runner.spectral_projector_generator(
        values, vectors, selected, excluded, derivative
    )
    return generator


def _integrate_rotating(runner, steps: int) -> np.ndarray:
    frame = _rotating_spectrum(0.0)[0]
    step = 1.0 / steps
    theta = 0.0
    for _ in range(steps):
        k1 = _rotating_generator(runner, theta, frame) @ frame
        k2_frame = frame + 0.5 * step * k1
        k2 = _rotating_generator(runner, theta + 0.5 * step, k2_frame) @ k2_frame
        k3_frame = frame + 0.5 * step * k2
        k3 = _rotating_generator(runner, theta + 0.5 * step, k3_frame) @ k3_frame
        k4_frame = frame + step * k3
        k4 = _rotating_generator(runner, theta + step, k4_frame) @ k4_frame
        frame, _ = runner._symmetric_polar(
            frame + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        )
        theta += step
    return frame


def test_analytic_spectral_derivative_and_fourth_order_kato_convergence() -> None:
    runner = _runner()
    theta = 0.37
    exact_frame, exact_projector, hamiltonian, derivative = _rotating_spectrum(theta)
    values, vectors = np.linalg.eigh(hamiltonian)
    selected_frame, selected, excluded, _ = runner.select_by_previous_projector(
        exact_frame, values, vectors, 1
    )
    projector_derivative, generator, gap = runner.spectral_projector_generator(
        values, vectors, selected, excluded, derivative
    )
    tangent = np.asarray([-np.sin(theta), np.cos(theta)], dtype=np.complex128)
    exact_derivative = np.outer(tangent, exact_frame[:, 0].conj()) + np.outer(exact_frame[:, 0], tangent.conj())
    assert gap == pytest.approx(2.0)
    assert np.max(np.abs(projector_derivative - exact_derivative)) < 1e-12
    assert np.max(np.abs(generator + generator.conj().T)) < 1e-12
    assert np.max(np.abs(generator @ exact_projector - exact_projector @ generator - exact_derivative)) < 1e-12
    assert np.max(np.abs(selected_frame @ selected_frame.conj().T - exact_projector)) < 1e-12

    coarse = _integrate_rotating(runner, 8)
    fine = _integrate_rotating(runner, 16)
    exact = _rotating_spectrum(1.0)[1]
    coarse_error = np.linalg.norm(coarse @ coarse.conj().T - exact)
    fine_error = np.linalg.norm(fine @ fine.conj().T - exact)
    assert fine_error < coarse_error / 12.0
    assert np.max(np.abs(fine.conj().T @ fine - np.eye(1))) < 1e-12


def test_small_path_checkpoint_resume_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    config = _toy_config()
    task, frame = _toy_task(), _toy_frame()
    metadata = {"identity": "toy", "task": task}
    reference = runner.compute_path(
        frame, task, config, tmp_path / "reference", metadata,
        adaptive_tolerance=1e-7,
    )

    original_write = runner._write_checkpoint
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        original_write(*args, **kwargs)
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated interruption after durable checkpoint")

    monkeypatch.setattr(runner, "_write_checkpoint", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.compute_path(
            frame, task, config, tmp_path / "resumed", metadata,
            adaptive_tolerance=1e-7,
        )
    monkeypatch.setattr(runner, "_write_checkpoint", original_write)
    resumed = runner.compute_path(
        frame, task, config, tmp_path / "resumed", metadata,
        adaptive_tolerance=1e-7,
    )
    for key in runner.HISTORY_FIELDS + ("density_x",):
        assert np.array_equal(reference[key], resumed[key])
    assert int(reference["accepted_steps_total"]) == int(resumed["accepted_steps_total"])
    assert int(reference["rejected_steps_total"]) == int(resumed["rejected_steps_total"])


def test_atomic_completion_pair_and_corrupt_checkpoint_recovery(tmp_path: Path) -> None:
    runner = _runner()
    config = _toy_config()
    task, frame = _toy_task(), _toy_frame()
    source_row = {
        "path": "/test/source.npz", "name": "source.npz", "bytes": 11,
        "sha256": "source-hash", "source_campaign_id": "toy",
        "source_config_hash": "config", "source_hashes": {"runner": "hash"}, "rank": 3,
    }
    hashes, config_hash, tolerance = {"runner": "hash"}, "config-hash", 1e-7
    metadata = runner._metadata(task, config_hash, hashes, source_row, tolerance)
    arrays = runner.compute_path(
        frame, task, config, tmp_path, metadata, adaptive_tolerance=tolerance
    )
    runner.publish_result(
        tmp_path, task, arrays, config_hash=config_hash, hashes=hashes,
        source_row=source_row, adaptive_tolerance=tolerance, elapsed_seconds=0.1,
    )
    assert runner.verify_result(
        tmp_path, task, config_hash=config_hash, hashes=hashes,
        source_row=source_row, config=config, adaptive_tolerance=tolerance,
    )[0]
    result, _ = runner.result_paths(tmp_path, task)
    with result.open("ab") as handle:
        handle.write(b"corrupt")
    assert not runner.verify_result(
        tmp_path, task, config_hash=config_hash, hashes=hashes,
        source_row=source_row, config=config, adaptive_tolerance=tolerance,
    )[0]

    checkpoint_arrays = runner._initial_arrays(5, 2)
    for key in checkpoint_arrays:
        checkpoint_arrays[key][:3] = 0
    runner._write_checkpoint(
        tmp_path, task, frame=frame, completed_interval=2, s=np.pi,
        next_step=0.1, arrays=checkpoint_arrays, metadata=metadata,
        maximum_orthonormality_residual=1e-14,
    )
    checkpoint, receipt = runner.checkpoint_paths(tmp_path, task)
    with checkpoint.open("ab") as handle:
        handle.write(b"corrupt")
    assert runner._load_checkpoint(
        tmp_path, task, metadata=metadata, count=5, nx=2, ambient=8, rank=3
    ) is None
    assert not checkpoint.exists() and not receipt.exists()


def test_real_source_inventory_pins_100_endpoint_states() -> None:
    runner = _runner()
    config = _config()
    context = runner.source_context(config)
    assert set(context) == {"N20x24", "N24x24"}
    assert all(len(context[size]["rows"]) == 50 for size in context)
    assert all(row["rank"] > 0 for context in context.values() for row in context["rows"].values())


def test_documentation_launch_and_cancellation_contracts() -> None:
    note = (PILOT_ROOT / "docs/kato_overlap_continuation_methods.tex").read_text(encoding="utf-8")
    working = (PILOT_ROOT / "docs/charge_pump_results_working.tex").read_text(encoding="utf-8")
    launcher = (PILOT_ROOT / "launch_kato_overlap_continuation_tmux.sh").read_text(encoding="utf-8")
    entrypoint = (PILOT_ROOT / "kato_overlap_continuation_tmux_entrypoint.sh").read_text(encoding="utf-8")
    cancellation = json.loads(
        (PILOT_ROOT / "results/N24x24_parent_schrodinger_rk4_s50_tau1e4_v1/cancellation_inventory.json").read_text(encoding="utf-8")
    )
    for needle in (
        r"P_{\mathrm{dyn}}", r"P_{\mathrm{inst}}", r"P_{\mathrm{cont}}",
        r"[\partial_\phi P_{\mathrm{spec}},P_{\mathrm{spec}}]",
        r"P^2=P", r"\delta_{\mathrm{sel}}", "step doubling", "cannot force",
    ):
        assert needle in note
    assert "physical-time" in working and "adaptive Kato" in working
    assert cancellation["status_before_cancel"]["verified_completions"] == 0
    assert cancellation["status_before_cancel"]["verified_checkpoint_pairs"] == 28
    assert cancellation["status_before_cancel"]["checkpoint_interval_histogram"] == {"104": 28}
    assert 'CPU_LIST="${CPU_LIST:-28-55}"' in launcher
    assert 'WORKERS="${WORKERS:-28}"' in launcher
    assert "sample-0 smoke" in entrypoint
    assert "--adaptive-tolerance 2.5e-9" in entrypoint
    assert "analyze_kato_overlap_continuation_s25.py" in entrypoint
