from __future__ import annotations

import copy
import importlib.util
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


BUNDLE = Path(__file__).resolve().parents[1]
REPO = BUNDLE.parents[2]
sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(REPO))

import modular_packet_analysis as packet  # noqa: E402
import run_campaign as campaign  # noqa: E402


def tiny_config() -> dict:
    config = copy.deepcopy(campaign.load_config())
    config["sampling_revision"] += "_test"
    config["root_seed"] += 7000
    config["geometry"].update({"Nx": 4, "Ny": 6, "cycles": 1, "samples_per_construction": 1})
    config["modular_analysis"].update({
        "subsystem_width": 3, "translated_cuts": 6, "wall_x": [1, 3], "packet_y_rel": [0, 2],
        "time_stop": 0.1, "time_step": 0.05, "spectral_cutoffs": [1e-10],
        "primary_spectral_cutoff": 1e-10, "wall_window_radii": [1], "primary_wall_window_radius": 1,
        "fit_windows": [[0.0, 0.1]], "primary_fit_window": [0.0, 0.1], "bootstrap_draws": 100,
        "time_chunk": 8,
    })
    return config


class LocalQueue:
    def put(self, _value):
        return None


def test_production_task_table_and_contract():
    config = campaign.load_config()
    campaign.validate_production_config(config)
    tasks = campaign.expand_tasks(config)
    assert len(tasks) == 200
    assert len({row["task_id"] for row in tasks}) == 200
    assert len({row["seed"] for row in tasks}) == 200
    assert sum(row["construction"] == "hard" for row in tasks) == 100
    assert sum(row["construction"] == "soft" for row in tasks) == 100


def test_frame_reconstruction_matches_full_covariance():
    rng = np.random.default_rng(5)
    nx, ny, rank = 4, 6, 24
    raw = rng.normal(size=(2 * nx * ny, rank)) + 1j * rng.normal(size=(2 * nx * ny, rank))
    frame = np.linalg.qr(raw)[0]
    full = 2.0 * frame @ frame.conj().T - np.eye(2 * nx * ny)
    for y0 in range(ny):
        actual = packet.restricted_centered_covariance_from_frame(frame, nx=nx, ny=ny, y0=y0)
        expected = packet.restrict_legacy_full_covariance(full, nx=nx, ny=ny, y0=y0)
        np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-15)


def test_packet_estimator_matches_legacy_implementation():
    legacy_path = REPO / "00_WORKSPACE/COLAB/colab_small_system_testing/src/run_sample_y0_averaged_dy_com.py"
    legacy_helper_path = legacy_path.with_name("dynamic_modular_charge_spreading.py")
    helper_spec = importlib.util.spec_from_file_location("dynamic_modular_charge_spreading", legacy_helper_path)
    helper = importlib.util.module_from_spec(helper_spec); helper_spec.loader.exec_module(helper)
    old_helper = sys.modules.get("dynamic_modular_charge_spreading")
    sys.modules["dynamic_modular_charge_spreading"] = helper
    try:
        spec = importlib.util.spec_from_file_location("legacy_packet_runner", legacy_path)
        legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    finally:
        if old_helper is None:
            sys.modules.pop("dynamic_modular_charge_spreading", None)
        else:
            sys.modules["dynamic_modular_charge_spreading"] = old_helper
    rng = np.random.default_rng(9)
    nx, ny, rank = 16, 4, 64
    raw = rng.normal(size=(2 * nx * ny, rank)) + 1j * rng.normal(size=(2 * nx * ny, rank))
    frame = np.linalg.qr(raw)[0]
    restricted = packet.restricted_centered_covariance_from_frame(frame, nx=nx, ny=ny, y0=1)
    mod = helper.modular_hamiltonian_from_restricted_covariance(restricted, eps=1e-10)
    eig = helper.eigensystem_from_h_mod(mod["h_mod"])
    times = np.array([0.0, 0.05, 0.10])
    legacy_specs = legacy.packet_specs(ny)
    expected = legacy.evolve_packet_dy_com(
        h_vals=eig["h_vals"], h_vecs=eig["h_vecs"], nx=nx, ny=ny,
        packets=legacy_specs, times=times, x_window_radius=2, time_chunk=3,
    )
    values, vectors = np.linalg.eigh(restricted)
    actual = packet._evolve_packets(
        eigenvalues=values, eigenvectors=vectors, eps=1e-10, nx=nx, ny=ny,
        specs=packet.packet_specs(nx, ny, [5, 11]), times=times, radii=[2], time_chunk=3,
    )
    np.testing.assert_allclose(actual["displacement"][0], expected["dy_com"], atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(actual["retention"][0] * 2.0, expected["Q_window"], atol=2e-12, rtol=2e-12)


def test_serial_and_process_execution_are_identical(tmp_path):
    config = tiny_config()
    task = campaign.expand_tasks(config)[0]
    hashes = campaign.source_hashes("simulation")
    config_hash = campaign.scientific_config_hash(config, "simulation")
    serial_root, process_root = tmp_path / "serial", tmp_path / "process"
    serial = campaign._run_simulation_worker((task, config, str(serial_root), config_hash, hashes, LocalQueue()))
    assert serial["ok"]
    manager = mp.Manager()
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("fork")) as pool:
            parallel = pool.submit(campaign._run_simulation_worker, (task, config, str(process_root), config_hash, hashes, manager.Queue())).result()
    finally:
        manager.shutdown()
    assert parallel["ok"]
    with np.load(campaign.result_paths(serial_root, task, "simulation")[0]) as a, np.load(campaign.result_paths(process_root, task, "simulation")[0]) as b:
        # LAPACK phase/roundoff can differ below 1e-15 while other production
        # workers are active; require physical-frame equivalence at complex128
        # precision rather than a byte-identical QR gauge.
        np.testing.assert_allclose(a["frame"], b["frame"], atol=2e-14, rtol=2e-14)
        np.testing.assert_allclose(
            a["frame"] @ a["frame"].conj().T,
            b["frame"] @ b["frame"].conj().T,
            atol=2e-14,
            rtol=2e-14,
        )
        assert int(a["rank"]) == int(b["rank"])


def test_completion_resume_corruption_and_analysis_only_restart(tmp_path):
    config = tiny_config()
    task = campaign.expand_tasks(config)[0]
    hashes = campaign.source_hashes("simulation")
    config_hash = campaign.scientific_config_hash(config, "simulation")
    assert campaign._run_simulation_worker((task, config, str(tmp_path), config_hash, hashes, LocalQueue()))["ok"]
    assert campaign.verify_pair(tmp_path, task, "simulation", config_hash, hashes)[0]
    frame_path, frame_completion = campaign.result_paths(tmp_path, task, "simulation")
    original_frame_sha = campaign.sha256_path(frame_path)
    analysis_hashes = campaign.source_hashes("analysis")
    analysis_config_hash = campaign.scientific_config_hash(config, "analysis")
    assert campaign._run_analysis_worker((task, config, str(tmp_path), analysis_config_hash, analysis_hashes, LocalQueue()))["ok"]
    assert campaign.verify_pair(tmp_path, task, "analysis", analysis_config_hash, analysis_hashes)[0]
    analysis_path, analysis_completion = campaign.result_paths(tmp_path, task, "analysis")
    analysis_completion.unlink()
    assert not campaign.verify_pair(tmp_path, task, "analysis", analysis_config_hash, analysis_hashes)[0]
    assert campaign._run_analysis_worker((task, config, str(tmp_path), analysis_config_hash, analysis_hashes, LocalQueue()))["ok"]
    assert campaign.sha256_path(frame_path) == original_frame_sha
    raw = bytearray(analysis_path.read_bytes()); raw[-1] ^= 1; analysis_path.write_bytes(raw)
    valid, reason = campaign.verify_pair(tmp_path, task, "analysis", analysis_config_hash, analysis_hashes)
    assert not valid and "checksum" in reason
    frame_completion.unlink()
    assert not campaign.verify_pair(tmp_path, task, "simulation", config_hash, hashes)[0]


def test_analysis_revision_does_not_invalidate_saved_dynamics():
    original = tiny_config()
    revised = copy.deepcopy(original)
    revised["modular_analysis"]["time_stop"] = 0.2
    assert campaign.scientific_config_hash(original, "simulation") == campaign.scientific_config_hash(revised, "simulation")
    assert campaign.source_hashes("simulation") == campaign.source_hashes("simulation")
    assert campaign.scientific_config_hash(original, "analysis") != campaign.scientific_config_hash(revised, "analysis")


def test_worker_failure_is_recorded_without_completion(tmp_path):
    config = tiny_config()
    task = campaign.expand_tasks(config)[0] | {"construction": "invalid", "task_id": "invalid_sample_000"}
    hashes = campaign.source_hashes("simulation")
    config_hash = campaign.scientific_config_hash(config, "simulation")
    result = campaign._run_simulation_worker((task, config, str(tmp_path), config_hash, hashes, LocalQueue()))
    assert not result["ok"]
    assert campaign.marker_path(tmp_path, task, "simulation", "failed").is_file()
    assert not campaign.result_paths(tmp_path, task, "simulation")[1].exists()
