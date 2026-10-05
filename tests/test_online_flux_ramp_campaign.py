from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np

from src.fgtn.occupied_frame import OccupiedFrameState


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "00_WORKSPACE" / "CURRENT" / "frozen_record_flux_charge_pilot"
sys.path.insert(0, str(PILOT))
import run_online_flux_ramp as ramp  # noqa: E402


def _config() -> dict:
    config = ramp.load_config(PILOT / "campaign_config.online_ramp_s10_v1.json")
    ramp.validate_config(config)
    return config


def test_production_task_table_and_branch_contract():
    config = _config()
    burnins = ramp.expand_burnin_tasks(config)
    ramps = ramp.expand_ramp_tasks(config)
    assert len(burnins) == 20
    assert len(ramps) == 40
    assert len({task["seed"] for task in burnins + ramps}) == 60
    for parent in burnins:
        children = [task for task in ramps if task["burnin_task_id"] == parent["task_id"]]
        assert {task["direction"] for task in children} == {"ccw", "cw"}
        assert len({task["seed"] for task in children}) == 2
        assert {task["wall"] for task in children} == {parent["wall"]}


def test_online_schedule_has_exact_zero_and_signed_full_flux():
    config = _config()
    tasks = ramp.expand_ramp_tasks(config)
    for task in tasks:
        phi = ramp.schedule_for(task, config)
        assert phi.shape == (17,)
        assert phi[0] == 0.0
        assert phi[-1] == task["sigma"] * 2.0 * np.pi
        assert np.allclose(np.diff(phi), task["sigma"] * 2.0 * np.pi / 16)


def test_atomic_pair_verification_and_checksum_rejection(tmp_path, monkeypatch):
    config = _config()
    task = ramp.expand_burnin_tasks(config)[0]
    hashes = {"cpu_engine": "a", "occupied_frame": "b", "campaign_runner": "c"}
    monkeypatch.setattr(ramp, "source_hashes", lambda: hashes)
    config_hash = ramp.scientific_config_hash(config)
    arrays = {
        "schema": np.asarray(ramp.BURNIN_SCHEMA),
        "frame": np.eye(4, 2, dtype=np.complex128),
    }
    ramp.publish_pair(tmp_path, task, arrays, config_hash, hashes, elapsed_seconds=1.0)
    assert ramp.verify_pair(tmp_path, task, config_hash, hashes)[0]
    result, completion = ramp.result_paths(tmp_path, task)
    result.write_bytes(result.read_bytes() + b"x")
    assert not ramp.verify_pair(tmp_path, task, config_hash, hashes)[0]
    result.unlink()
    assert completion.is_file()
    assert not ramp.verify_pair(tmp_path, task, config_hash, hashes)[0]


def test_ramp_pair_is_bound_to_exact_burnin_checksum(tmp_path, monkeypatch):
    config = _config()
    task = ramp.expand_ramp_tasks(config)[0]
    hashes = {"cpu_engine": "a", "occupied_frame": "b", "campaign_runner": "c"}
    monkeypatch.setattr(ramp, "source_hashes", lambda: hashes)
    config_hash = ramp.scientific_config_hash(config)
    arrays = {"schema": np.asarray(ramp.RAMP_SCHEMA), "burnin_sha256": np.asarray("parent-a")}
    for key in (
        "phi", "N_left", "N_right", "N_total", "delta_N_left", "delta_N_right",
        "source_A_left", "source_A_right", "q_x_raw", "q_x_corrected", "rank",
    ):
        arrays[key] = np.zeros(17)
    ramp.publish_pair(
        tmp_path, task, arrays, config_hash, hashes, burnin_sha256="parent-a", elapsed_seconds=1.0
    )
    assert ramp.verify_pair(
        tmp_path, task, config_hash, hashes, burnin_sha256="parent-a"
    )[0]
    assert not ramp.verify_pair(
        tmp_path, task, config_hash, hashes, burnin_sha256="parent-b"
    )[0]


def test_locked_contract_rejects_random_serial_or_offset():
    config = _config()
    changed = copy.deepcopy(config)
    changed["dynamics"]["sequence"] = "random"
    try:
        ramp.validate_config(changed)
    except ValueError:
        pass
    else:
        raise AssertionError("random serial was accepted")
    changed = copy.deepcopy(config)
    changed["twist"]["spectral_origin_offset"] = 1e-7
    try:
        ramp.validate_config(changed)
    except ValueError:
        pass
    else:
        raise AssertionError("spectral offset was accepted for online evolution")


def test_campaign_sources_and_launch_contract_are_explicit():
    config = _config()
    assert all(path.is_file() for path in ramp.SOURCE_PATHS.values())
    launcher = (PILOT / "launch_online_flux_ramp_tmux.sh").read_text(encoding="utf-8")
    entrypoint = (PILOT / "online_flux_ramp_tmux_entrypoint.sh").read_text(encoding="utf-8")
    assert "online_flux_ramp_N16x20_s10_v1" in launcher
    assert "28-55" in launcher
    assert "taskset" in entrypoint and "--resume" in entrypoint and "--analyze" in entrypoint
    assert config["dynamics"]["canonical_entry_point"] == "classA_U1FGTN.run_markov_circuit"


def test_small_online_branches_share_origin_and_close_source_balance():
    config = _config()
    config = copy.deepcopy(config)
    config["geometry"].update({"Nx": 4, "Ny": 4, "dw_interval": [1, 3]})
    config["regions"] = {
        "left_x_start": 0,
        "left_x_stop_exclusive": 2,
        "right_x_start": 2,
        "right_x_stop_exclusive": 4,
    }
    config["dynamics"]["ramp_cycles"] = 2
    dimension = 2 * 4 * 4
    frame = OccupiedFrameState.random_pure(
        dimension, dimension // 2, rng=np.random.default_rng(91)
    ).snapshot()
    origins = []
    final_arrays = []
    for direction, sigma, seed in (("ccw", 1, 101), ("cw", -1, 202)):
        task = {"sigma": sigma}
        phi = ramp.schedule_for(task, config)
        model = ramp._model(config, "soft")
        observer = ramp.RampObserver(model, config, phi, None)
        model.run_markov_circuit(
            frame_init=frame,
            frame_init_prepared=True,
            controller_twist_schedule=phi,
            controller_twist_gauge="uniform",
            native_cycle_observer=observer.cycle,
            native_event_observer=observer.event,
            **ramp._engine_kwargs(config, "soft", cycles=2, seed=seed),
        )
        arrays = observer.arrays(config)
        origins.append((arrays["N_left"][0], arrays["N_right"][0], arrays["density_x"][0]))
        final_arrays.append(arrays)
    assert origins[0][0] == origins[1][0]
    assert origins[0][1] == origins[1][1]
    assert np.array_equal(origins[0][2], origins[1][2])
    for arrays in final_arrays:
        assert arrays["phi"].shape == (3,)
        assert np.max(np.abs(arrays["charge_continuity_residual"])) < 1e-9
        assert np.max(np.abs(arrays["corrected_balance_residual"])) < 1e-9
        assert float(arrays["maximum_source_partition_residual"]) < 1e-12
        assert arrays["rank"].shape == (3,)


def test_serial_and_parallel_campaign_outputs_are_scientifically_identical(tmp_path):
    config = copy.deepcopy(_config())
    config["campaign_id"] = "online_flux_ramp_parallel_parity_test"
    config["geometry"].update({"Nx": 4, "Ny": 4, "dw_interval": [1, 3]})
    config["regions"] = {
        "left_x_start": 0,
        "left_x_stop_exclusive": 2,
        "right_x_start": 2,
        "right_x_stop_exclusive": 4,
    }
    config["dynamics"]["burn_in_cycles"] = 1
    config["dynamics"]["ramp_cycles"] = 1
    config["ensemble"]["samples_per_wall"] = 1
    serial_root = tmp_path / "serial"
    parallel_root = tmp_path / "parallel"

    for stage in ("burnin", "ramp"):
        ramp.run_stage(stage, config, serial_root, workers=1, resume=True)
        ramp.run_stage(stage, config, parallel_root, workers=4, resume=True)

    tasks = ramp.expand_burnin_tasks(config) + ramp.expand_ramp_tasks(config)
    for task in tasks:
        serial_path, _ = ramp.result_paths(serial_root, task)
        parallel_path, _ = ramp.result_paths(parallel_root, task)
        with np.load(serial_path, allow_pickle=False) as serial, np.load(
            parallel_path, allow_pickle=False
        ) as parallel:
            assert serial.files == parallel.files
            for key in serial.files:
                if key == "burnin_sha256":
                    # Each ramp correctly binds its own parent's file checksum;
                    # compressed-container bytes need not match across roots.
                    continue
                if key == "frame":
                    serial_frame = np.asarray(serial[key])
                    parallel_frame = np.asarray(parallel[key])
                    assert np.allclose(
                        serial_frame @ serial_frame.conj().T,
                        parallel_frame @ parallel_frame.conj().T,
                        rtol=0.0,
                        atol=1e-12,
                    ), (task["task_id"], key)
                elif np.issubdtype(serial[key].dtype, np.inexact):
                    assert np.allclose(
                        serial[key], parallel[key], rtol=0.0, atol=1e-12
                    ), (task["task_id"], key)
                else:
                    assert np.array_equal(serial[key], parallel[key]), (
                        task["task_id"],
                        key,
                    )
