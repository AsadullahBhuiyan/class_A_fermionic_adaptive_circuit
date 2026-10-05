from __future__ import annotations

import importlib.util
import itertools
import json
import math
import copy
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = (
    REPO_ROOT / "00_WORKSPACE" / "CURRENT" / "many_body_lyapunov_maxmix_cpu_pilot"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from lyapunov_observer import (
    ActiveSpectrumObserver,
    leading_log_sigma2_levels,
    lowest_subset_sums,
    natural_spectrum_factors,
)
from run_campaign import (
    RESULT_SCHEMA,
    TaskSpec,
    build_tasks,
    load_checkpoint,
    make_model,
    run_task,
    save_checkpoint,
    stable_seed,
    verify_complete,
)
from analyze_campaign import fit_finite_size, paired_convergence


def load_config() -> dict:
    return json.loads((PACKAGE_ROOT / "campaign_config.json").read_text(encoding="utf-8"))


def test_locked_task_table_has_1200_unique_cpu_trajectories(tmp_path: Path) -> None:
    tasks = build_tasks(load_config(), tmp_path)
    assert len(tasks) == 1200
    assert len({task.task_id for task in tasks}) == 1200
    assert len({task.seed for task in tasks}) == 1200
    assert {task.nx for task in tasks} == {16}
    assert {task.construction for task in tasks} == {"hard", "soft"}
    assert {task.ny for task in tasks} == {20, 22, 24, 26, 28, 30}
    for construction in ("hard", "soft"):
        for ny in (20, 22, 24, 26, 28, 30):
            selected = [
                task
                for task in tasks
                if task.construction == construction and task.ny == ny
            ]
            assert len(selected) == 100
            assert len({task.seed for task in selected}) == 100
            assert {task.cycles for task in selected} == {4 * ny}
    assert [(task.ny, task.construction) for task in tasks[:4]] == [
        (30, "hard"), (30, "soft"), (28, "hard"), (28, "soft")
    ]


def test_heap_levels_match_exhaustive_products_and_normalization() -> None:
    occupations = np.asarray([0.2, 0.35, 0.7, 1.0])
    log_z = 1.7
    levels = leading_log_sigma2_levels(occupations, log_z, count=8)
    exhaustive = []
    weights = []
    for bits in itertools.product((0, 1), repeat=occupations.size):
        factors = np.where(bits, occupations, 1.0 - occupations)
        weight = float(np.prod(factors))
        weights.append(math.exp(log_z) * weight)
        exhaustive.append(-math.inf if weight == 0 else log_z + math.log(weight))
    expected = np.sort(np.asarray(exhaustive))[::-1][:8]
    assert np.allclose(levels, expected, atol=1e-13)
    assert np.isclose(sum(weights), math.exp(log_z), atol=1e-13)
    _, costs, caps = natural_spectrum_factors(occupations)
    assert caps[-1]
    assert math.isinf(costs[-1])
    assert np.allclose(
        lowest_subset_sums(costs, 8),
        np.sort(np.asarray([levels[0] - value for value in expected])),
        atol=1e-13,
    )


def test_cycle_zero_is_identity_operator_spectrum() -> None:
    nx, ny = 20, 2
    active = np.asarray(
        [mu + 2 * x + 2 * nx * y for y in range(ny) for x in range(5, 16) for mu in (0, 1)],
        dtype=np.int64,
    )
    observer = ActiveSpectrumObserver(
        nx=nx,
        ny=ny,
        cycles=4,
        active_indices=active,
        wall_locations=(5, 15),
    )
    observer.observe(cycle=0, G=np.zeros((2 * nx * ny, 2 * nx * ny), dtype=np.complex128))
    observer.validate(completed_cycle=0)
    assert np.all(observer.occupations[0] == 0.5)
    assert np.allclose(observer.leading_log_sigma2[0], 0.0, atol=1e-13)
    assert np.isclose(observer.log_z[0], 22 * ny * math.log(2.0))


def test_observer_rejects_lower_precision_covariance() -> None:
    active = np.asarray(
        [mu + 2 * x + 40 * y for y in range(2) for x in range(5, 16) for mu in (0, 1)],
        dtype=np.int64,
    )
    observer = ActiveSpectrumObserver(
        nx=20, ny=2, cycles=4, active_indices=active, wall_locations=(5, 15)
    )
    with pytest.raises(TypeError, match="complex128"):
        observer.observe(cycle=0, G=np.zeros((80, 80), dtype=np.complex64))


def test_observer_does_not_change_physical_state_or_rng(tmp_path: Path) -> None:
    spec = tiny_spec(tmp_path, label="neutrality")

    def evolve(with_observer: bool):
        model = make_model(spec)
        final = {}
        observer = None
        if with_observer:
            observer = ActiveSpectrumObserver(
                nx=20,
                ny=2,
                cycles=4,
                active_indices=model.active_top_layer_indices(True),
                wall_locations=(5, 15),
                soft_mode_count=4,
                leading_level_count=8,
            )

        def checkpoint(*, cycle, state):
            if int(cycle) == spec.cycles:
                final.update(copy.deepcopy(state))

        model.run_markov_circuit(
            G_history=False,
            progress=False,
            cycles=spec.cycles,
            postselect=False,
            perfect_correction=True,
            samples=1,
            parallelize_samples=False,
            init_mode="maxmix",
            save=False,
            n_a=0.5,
            sequence="raster_y",
            meas_slab_only=True,
            random_seed=spec.seed,
            state_representation="covariance",
            cycle_observer=None if observer is None else observer.observe,
            trajectory_weight_observer=None if observer is None else observer.record_site,
            checkpoint_observer=checkpoint,
        )
        return final

    plain = evolve(False)
    observed = evolve(True)
    assert np.allclose(plain["G"], observed["G"], atol=2e-13, rtol=2e-13)
    assert plain["rng_states"] == observed["rng_states"]
    assert np.array_equal(plain["last_ordered_site_ids"], observed["last_ordered_site_ids"])


def tiny_spec(tmp_path: Path, *, label: str) -> TaskSpec:
    result = tmp_path / label / "trajectory.npz"
    return TaskSpec(
        task_id=f"tiny_{label}",
        revision="test_revision",
        nx=20,
        ny=2,
        cycles=4,
        sample_index=0,
        seed=stable_seed(7, "test", 20, 2, 0),
        config_hash="abc",
        source_hashes={"test": "hash"},
        result_path=str(result),
        completion_path=str(result.with_suffix(".complete.json")),
        checkpoint_path=str(tmp_path / label / "checkpoint.npz"),
        checkpoint_stride=1,
        soft_mode_count=4,
        leading_level_count=8,
        cap_tolerance=1e-12,
    )


def test_actual_cpu_observer_is_rng_neutral_and_checkpoint_resume_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tiny_spec(tmp_path, label="reference")
    resumed = tiny_spec(tmp_path, label="resumed")
    resumed = replace(resumed, task_id=reference.task_id)
    run_task(asdict(reference))
    import run_campaign as runner

    original_save = runner.save_checkpoint
    crashed = False
    captured_checkpoint = {}

    def save_then_crash(*args, **kwargs):
        nonlocal crashed
        original_save(*args, **kwargs)
        state = kwargs["engine_state"]
        if int(state["completed_cycles"]) == 2 and not crashed:
            captured_checkpoint.update(copy.deepcopy(state))
            crashed = True
            raise RuntimeError("simulated process death after durable cycle 2")

    monkeypatch.setattr(runner, "save_checkpoint", save_then_crash)
    with pytest.raises(RuntimeError, match="simulated process death"):
        runner.run_task(asdict(resumed))
    assert Path(resumed.checkpoint_path).exists()
    active = np.asarray(
        [mu + 2 * x + 40 * y for y in range(2) for x in range(5, 16) for mu in (0, 1)],
        dtype=np.int64,
    )
    restored_observer = ActiveSpectrumObserver(
        nx=20,
        ny=2,
        cycles=4,
        active_indices=active,
        wall_locations=(5, 15),
        soft_mode_count=4,
        leading_level_count=8,
    )
    restored_state, _ = load_checkpoint(resumed, restored_observer)
    assert restored_state is not None
    assert np.array_equal(restored_state["G"], captured_checkpoint["G"])
    assert np.array_equal(
        restored_state["last_ordered_site_ids"], captured_checkpoint["last_ordered_site_ids"]
    )
    assert restored_state["rng_states"] == captured_checkpoint["rng_states"]
    monkeypatch.setattr(runner, "save_checkpoint", original_save)
    runner.run_task(asdict(resumed))
    assert verify_complete(resumed)[0]
    with np.load(reference.result_path, allow_pickle=False) as left, np.load(
        resumed.result_path, allow_pickle=False
    ) as right:
        excluded = {"elapsed_seconds", "final_engine_state_sha256"}
        for key in sorted(set(left.files) - excluded):
            assert key in right.files
            if left[key].dtype.kind in "fci":
                assert np.allclose(left[key], right[key], atol=2e-13, rtol=2e-13, equal_nan=True), key
            else:
                assert np.array_equal(left[key], right[key]), key


def test_completion_pair_fails_closed_on_partial_or_corrupt_result(tmp_path: Path) -> None:
    spec = tiny_spec(tmp_path, label="completion")
    result_path = Path(spec.result_path)
    result_path.parent.mkdir(parents=True)
    np.savez_compressed(
        result_path,
        schema=np.asarray(RESULT_SCHEMA),
        task_id=np.asarray(spec.task_id),
        cycles=np.arange(spec.cycles + 1),
        cycle_seen=np.ones(spec.cycles + 1, dtype=bool),
    )
    assert not verify_complete(spec)[0]
    Path(spec.completion_path).write_text("{}\n", encoding="utf-8")
    assert not verify_complete(spec)[0]


def test_corrupt_checkpoint_fails_closed(tmp_path: Path) -> None:
    spec = tiny_spec(tmp_path, label="bad_checkpoint")
    path = Path(spec.checkpoint_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not an npz")
    active = np.asarray(
        [mu + 2 * x + 40 * y for y in range(2) for x in range(5, 16) for mu in (0, 1)],
        dtype=np.int64,
    )
    observer = ActiveSpectrumObserver(
        nx=20, ny=2, cycles=4, active_indices=active, wall_locations=(5, 15), soft_mode_count=4
    )
    with pytest.raises(RuntimeError, match="invalid checkpoint"):
        load_checkpoint(spec, observer)


def test_analysis_recovers_declared_finite_size_coefficients() -> None:
    ny = np.asarray([20, 22, 24, 26, 28])
    x = 1.0 / ny**2
    a0 = -0.7
    f0 = 1.3 + a0 * x
    lambda0 = -ny * f0
    slopes = np.empty((ny.size, 5))
    slopes[:, 0] = lambda0
    expected_ratios = []
    for level, ai in enumerate((0.12, 0.2, 0.31, 0.45), start=1):
        slopes[:, level] = lambda0 - ny * (0.05 * level + ai * x)
        expected_ratios.append(-ai / (12 * a0))
    fit = fit_finite_size(ny, slopes)
    assert np.isclose(fit["primary"]["A0"], a0, atol=1e-10)
    assert np.allclose(
        [row["x_typ_over_c_eff"] for row in fit["gaps"]], expected_ratios, atol=1e-10
    )


def test_paired_convergence_uses_whole_trajectory_shifts() -> None:
    rng = np.random.default_rng(4)
    middle = np.tile(np.asarray([-2.0, -2.2, -2.4, -2.6, -2.8]), (100, 1))
    late = middle.copy()
    rows = paired_convergence(
        middle, late, bootstrap_count=200, rng=rng, relative_threshold=0.1
    )
    assert len(rows) == 5
    assert all(row["passes"] for row in rows)


def test_launcher_is_tmux_cpu_only_and_progress_visible() -> None:
    launcher = (PACKAGE_ROOT / "launch_tmux.sh").read_text(encoding="utf-8")
    runner = (PACKAGE_ROOT / "run_campaign.py").read_text(encoding="utf-8")
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    assert "tmux new-session" in launcher
    assert "OMP_NUM_THREADS=1" in launcher
    assert "desc=\"Nx16 hard/soft Lyapunov CPU\"" in runner
    assert "local CPU campaign" in readme
    assert "google.colab" not in launcher + runner
    assert "drive.google" not in launcher + runner
