from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("pure_tangent_flux_runner_test", PILOT / "run_pure_tangent_flux.py")
sys.modules.setdefault("run_pure_tangent_flux", runner)
analysis = _load("pure_tangent_flux_analysis_test", PILOT / "analyze_pure_tangent_flux.py")


def test_locked_raster_y_campaign_expands_68_tasks():
    config_path = PILOT / "campaign_config.pure_tangent_v1.json"
    config = json.loads(config_path.read_text())
    runner.validate_config(config)
    sources = runner.source_hashes(config_path)
    config_hash = runner.canonical_hash(
        {"schema": runner.SCHEMA, "config": config, "source_hashes": sources}
    )
    tasks = runner.expand_tasks(config, config_hash, sources)
    assert len(tasks) == 68
    assert len({task["task_id"] for task in tasks}) == 68
    assert config["dynamics"]["sequence"] == "raster_y"
    assert config["tangent"]["candidate_mode_count"] == 64
    assert config["tangent"]["minimum_candidate_mode_count"] == 32
    assert config["geometry"] == {
        "Nx": 16,
        "Ny": 20,
        "cycles": 40,
        "nshell": 1,
        "filling_frac": 0.5,
        "alpha_1": 1.0,
        "alpha_2": 30.0,
        "trial_orbitals": "X",
        "dw_interval": [4, 12],
    }
    for direction in config["twist"]["directions"]:
        sigma = direction["sigma"]
        rows = [task for task in tasks if task["direction"] == direction["name"] and task["arm"] == "soft"]
        np.testing.assert_allclose(rows[0]["phi"], -sigma * 1e-7)
        np.testing.assert_allclose(rows[-1]["phi"], -sigma * 1e-7 + sigma * 2 * np.pi)


def test_tangent_observer_saves_wall_weights_and_particle_hole_charge():
    config = json.loads((PILOT / "campaign_config.pure_tangent_v1.json").read_text())
    config["geometry"]["Nx"] = 2
    config["geometry"]["Ny"] = 2
    config["geometry"]["cycles"] = 1
    config["regions"] = {
        "left_x_start": 0,
        "left_x_stop_exclusive": 1,
        "right_x_start": 1,
        "right_x_stop_exclusive": 2,
    }
    config["tangent"]["candidate_mode_count"] = 4
    config["tangent"]["minimum_candidate_mode_count"] = 4
    observer = runner.FinalPureTangentObserver(cycles=1, phi=0.0, config=config)
    frame = np.eye(8, dtype=np.complex128)[None, ...]
    core = np.diag([0.5, 0.6, 0.7, 0.8]).astype(np.complex128)[None, ...]
    observer(
        lyapunov_cycle=1,
        lyapunov_frame=frame,
        lyapunov_block_sizes=(4, 4),
        lyapunov_block_core_hat=(core, core),
        lyapunov_block_core_log_scale=(np.zeros(1), np.zeros(1)),
        lyapunov_block_core_null_count=(np.zeros(1, dtype=int), np.zeros(1, dtype=int)),
        lyapunov_initial_block_basis=(frame[:, :, :4], frame[:, :, 4:]),
        lyapunov_initial_active_purity_defect=0.0,
        lyapunov_min_branch_probability=np.ones(1),
        lyapunov_min_abs_born_denominator=np.ones(1),
        lyapunov_invalid_branch_count=np.zeros(1, dtype=int),
    )
    assert observer.arrays is not None
    saved = observer.arrays
    assert saved["candidate_pair_indices"].shape == (4, 2)
    for side in ("left", "right"):
        assert saved[f"candidate_occupied_{side}_weight"].shape == (4,)
        assert saved[f"candidate_empty_{side}_weight"].shape == (4,)
    np.testing.assert_allclose(
        saved["candidate_occupied_left_weight"] + saved["candidate_occupied_right_weight"],
        1.0,
    )
    np.testing.assert_allclose(
        saved["candidate_empty_left_weight"] + saved["candidate_empty_right_weight"],
        1.0,
    )
    singular_logs = np.log(np.asarray([0.8, 0.7, 0.6, 0.5]))
    pair_indices = saved["candidate_pair_indices"]
    expected_rates = (
        singular_logs[pair_indices[:, 0]] + singular_logs[pair_indices[:, 1]]
    )
    np.testing.assert_allclose(saved["candidate_pair_rates"], expected_rates)
    np.testing.assert_allclose(
        saved["candidate_effective_gaps_per_cycle"], -2.0 * expected_rates
    )
    assert np.max(np.abs(saved["candidate_excitation_charge"])) <= 1.0


def test_tangent_tracking_does_not_change_physical_trajectory():
    from src.fgtn.classA_U1FGTN import classA_U1FGTN

    common = dict(
        G_history=False,
        progress=False,
        cycles=2,
        samples=1,
        parallelize_samples=False,
        init_mode="default",
        save=False,
        sequence="raster_y",
        perfect_correction=True,
        random_seed=314,
        state_representation="physical_frame",
        physical_covariance_update="rank1",
    )
    plain = classA_U1FGTN(2, 2, DW=False, nshell=0).run_markov_circuit(**common)
    rows = []
    tangent = classA_U1FGTN(2, 2, DW=False, nshell=0).run_markov_circuit(
        **common,
        lyapunov_frame_observer=lambda **row: rows.append(row),
        lyapunov_basis_mode="pure_occupied_empty",
        lyapunov_start_cycle=1,
        lyapunov_full_space=True,
        lyapunov_track_restricted_core=True,
    )
    np.testing.assert_array_equal(plain["G_final"], tangent["G_final"])
    assert len(rows) == 2


def test_completion_pair_is_verified_and_corruption_reruns(tmp_path: Path):
    config_path = PILOT / "campaign_config.pure_tangent_v1.json"
    config = json.loads(config_path.read_text())
    sources = runner.source_hashes(config_path)
    config_hash = runner.canonical_hash(
        {"schema": runner.SCHEMA, "config": config, "source_hashes": sources}
    )
    task = runner.expand_tasks(config, config_hash, sources)[0]
    result, completion = runner.task_paths(tmp_path, task)
    candidate_count = 36
    runner.atomic_npz(
        result,
        schema=np.asarray(runner.TASK_SCHEMA),
        task_id=np.asarray(task["task_id"]),
        task_hash=np.asarray(task["task_hash"]),
        candidate_pair_indices=np.zeros((candidate_count, 2), dtype=np.int32),
        candidate_pair_rates=np.zeros(candidate_count),
        candidate_mode_count=np.asarray(candidate_count, dtype=np.int32),
        candidate_output_occupied=np.zeros((640, candidate_count), dtype=np.complex128),
        candidate_output_empty=np.zeros((640, candidate_count), dtype=np.complex128),
        candidate_output_occupied_physical_occupation=np.ones(candidate_count),
        candidate_output_empty_physical_occupation=np.zeros(candidate_count),
    )
    runner.atomic_json(
        completion,
        {
            "schema": runner.COMPLETION_SCHEMA,
            "task_id": task["task_id"],
            "task_hash": task["task_hash"],
            "result": runner.file_record(result),
        },
    )
    assert runner.verify_task(tmp_path, task)
    with result.open("ab") as handle:
        handle.write(b"corrupt")
    assert not runner.verify_task(tmp_path, task)


def test_overlap_continuation_recovers_permuted_candidates():
    dimension, candidates, tracked, points = 12, 6, 3, 4
    basis = np.eye(dimension, dtype=np.complex128)
    rows = []
    permutations = [np.arange(candidates), np.array([2, 0, 1, 3, 4, 5]), np.array([1, 2, 0, 3, 4, 5]), np.arange(candidates)]
    for point in range(points):
        permutation = permutations[point]
        rows.append(
            {
                "candidate_output_occupied": basis[:, :candidates][:, permutation],
                "candidate_output_empty": basis[:, 6:12][:, permutation],
                "candidate_effective_gaps_per_cycle": np.arange(candidates, dtype=float)[permutation],
                "candidate_pair_indices": np.stack((permutation, permutation), axis=1),
                "candidate_pair_rates": -0.5 * np.arange(candidates, dtype=float)[permutation],
                "candidate_excitation_charge": np.zeros(candidates),
                "candidate_occupied_left_weight": np.ones(candidates),
                "candidate_occupied_right_weight": np.zeros(candidates),
                "candidate_empty_left_weight": np.zeros(candidates),
                "candidate_empty_right_weight": np.ones(candidates),
                "candidate_svd_residual": np.zeros(candidates),
                "candidate_output_occupied_physical_occupation": np.ones(candidates),
                "candidate_output_empty_physical_occupation": np.zeros(candidates),
                "candidate_output_occupied_projector_residual": np.zeros(candidates),
                "candidate_output_empty_projector_residual": np.zeros(candidates),
                "candidate_x_density": np.zeros((candidates, 2)),
            }
        )
    continued = analysis._continue_direction(rows, tracked)
    for point in range(points):
        np.testing.assert_array_equal(
            permutations[point][continued["candidate_index"][point]], np.arange(tracked)
        )
    np.testing.assert_allclose(continued["step_overlap"], 1.0)
