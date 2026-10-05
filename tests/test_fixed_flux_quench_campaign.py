from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np

from src.fgtn.occupied_frame import OccupiedFrameState


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "00_WORKSPACE" / "CURRENT" / "frozen_record_flux_charge_pilot"
if str(PILOT) not in sys.path:
    sys.path.insert(0, str(PILOT))
import run_fixed_flux_quench as quench  # noqa: E402


def _config() -> dict:
    config = quench.load_config(PILOT / "campaign_config.fixed_flux_quench_s10_v1.json")
    quench.validate_config(config)
    return config


def _small_source_config() -> dict:
    config = quench.frozen.load_config(
        PILOT / "campaign_config.frozen_continuous_ramp_s10_v1.json"
    )
    config = copy.deepcopy(config)
    config["geometry"].update({"Nx": 4, "Ny": 4, "dw_interval": [1, 3]})
    config["regions"] = {
        "left_x_start": 0,
        "left_x_stop_exclusive": 2,
        "right_x_start": 2,
        "right_x_stop_exclusive": 4,
    }
    return config


def test_task_expansion_fixed_twists_and_paired_seeds():
    config = _config()
    rows = quench.tasks(config)
    assert len(rows) == 160
    assert len({row["task_id"] for row in rows}) == 160
    assert {row["twist_name"] for row in rows} == {
        "pi_over_2", "pi", "three_pi_over_2", "near_2pi"
    }
    near = [row for row in rows if row["twist_name"] == "near_2pi"]
    assert all(np.isclose(abs(row["fixed_phi"]), 2 * np.pi - 1e-7) for row in near)
    for wall in quench.WALLS:
        for sample_id in range(10):
            for twist_index in range(4):
                pair = [
                    row for row in rows
                    if row["wall"] == wall and row["sample_id"] == sample_id
                    and row["twist_index"] == twist_index
                ]
                assert {row["direction"] for row in pair} == {"ccw", "cw"}
                assert len({row["seed"] for row in pair}) == 1
                assert np.array_equal(
                    quench.fixed_schedule(pair[0]), -quench.fixed_schedule(pair[1])
                )


def test_all_source_burnins_and_records_are_verified():
    source = quench.source_context(_config())
    assert len(source["burnins"]) == 20
    assert len(source["references"]) == 20
    assert all(len(row["sha256"]) == 64 for row in source["references"].values())


def _synthetic_arrays(task: dict) -> dict:
    length = int(task["cycles"]) + 1
    zeros = np.zeros(length)
    density = np.zeros((length, 16))
    return {
        "schema": np.asarray(quench.RESULT_SCHEMA),
        "cycles": np.arange(length),
        "phi": quench.fixed_schedule(task),
        "N_left": zeros,
        "N_right": zeros,
        "N_total": zeros,
        "raw_delta_N_left": zeros,
        "raw_delta_N_right": zeros,
        "raw_delta_N_total": zeros,
        "raw_q_x": zeros,
        "reference_N_left": zeros,
        "reference_N_right": zeros,
        "reference_N_total": zeros,
        "response_delta_N_left": zeros,
        "response_delta_N_right": zeros,
        "response_delta_N_total": zeros,
        "response_q_x": zeros,
        "net_injected_charge": zeros,
        "injection_count": zeros,
        "charge_continuity_residual": zeros,
        "rank": zeros,
        "density_x": density,
        "reference_density_x": density,
        "response_density_x": density,
        "minimum_selected_probability_by_cycle": np.ones(length),
        "branch_log_probability_by_cycle": zeros,
        "burnin_sha256": np.asarray("b" * 64),
        "reference_sha256": np.asarray("r" * 64),
        "record_sha256": np.asarray("q" * 64),
        "record_entries": np.asarray(64),
        "final_frame_gram_residual": np.asarray(0.0),
    }


def test_atomic_completion_resume_and_corruption(tmp_path):
    config = _config()
    task = quench.tasks(config)[0]
    hashes = {name: name for name in quench.SOURCE_PATHS}
    config_hash = quench.scientific_config_hash(config)
    dependencies = {"burnin_sha256": "b" * 64, "reference_sha256": "r" * 64}
    quench.publish_pair(
        tmp_path, task, _synthetic_arrays(task), config_hash, hashes,
        dependencies, elapsed_seconds=1.0,
    )
    assert quench.verify_pair(tmp_path, task, config_hash, hashes, dependencies)[0]
    result, completion = quench.result_paths(tmp_path, task)
    result.write_bytes(result.read_bytes() + b"corrupt")
    assert not quench.verify_pair(tmp_path, task, config_hash, hashes, dependencies)[0]
    result.unlink()
    assert completion.is_file()
    assert not quench.verify_pair(tmp_path, task, config_hash, hashes, dependencies)[0]


def test_fixed_flux_replay_uses_same_record_and_zero_flux_reproduces_reference():
    config = _small_source_config()
    cycles = 2
    dimension = 2 * 4 * 4
    frame = OccupiedFrameState.random_pure(
        dimension, dimension // 2, rng=np.random.default_rng(194)
    ).snapshot()

    model = quench.frozen._model(config, "soft")
    reference_audit = quench.frozen.RecordAudit(cycles, capture=True)
    reference_observer = quench.frozen.ChargeObserver(
        model, config, np.zeros(cycles + 1), None, "reference"
    )
    reference_result = model.run_markov_circuit(
        frame_init=frame,
        frame_init_prepared=True,
        native_cycle_observer=reference_observer.cycle,
        native_event_observer=reference_observer.event,
        trajectory_weight_observer=reference_audit,
        **quench.frozen._engine_kwargs(config, "soft", cycles, seed=301),
    )
    record = quench.frozen.record_prefix(reference_audit.entries, cycles)
    reference_arrays = reference_observer.arrays(config, reference_audit)

    outputs = {}
    for name, fixed_phi in (("zero", 0.0), ("ccw", np.pi / 2), ("cw", -np.pi / 2)):
        phi = np.full(cycles + 1, fixed_phi)
        model = quench.frozen._model(config, "soft")
        audit = quench.frozen.RecordAudit(cycles)
        observer = quench.frozen.ChargeObserver(model, config, phi, None, "fixed")
        result = model.run_markov_circuit(
            frame_init=frame,
            frame_init_prepared=True,
            controller_twist_schedule=phi,
            controller_twist_gauge="uniform",
            trajectory_replay=record,
            trajectory_replay_probability_tol=1e-14,
            native_cycle_observer=observer.cycle,
            native_event_observer=observer.event,
            trajectory_weight_observer=audit,
            **quench.frozen._engine_kwargs(config, "soft", cycles, seed=302),
        )
        outputs[name] = (observer.arrays(config, audit), result["native_final"])

    reference_frame = np.asarray(reference_result["native_final"]["frame"])
    zero_frame = np.asarray(outputs["zero"][1]["frame"])
    assert np.allclose(
        reference_frame @ reference_frame.conj().T,
        zero_frame @ zero_frame.conj().T,
        rtol=0.0,
        atol=1e-12,
    )
    for field in ("N_left", "N_right", "N_total"):
        assert np.allclose(outputs["zero"][0][field], reference_arrays[field], rtol=0.0, atol=1e-12)
    for field in ("rank", "net_injected_charge", "injection_count"):
        assert np.array_equal(outputs["ccw"][0][field], outputs["cw"][0][field])
    assert np.max(np.abs(outputs["ccw"][0]["charge_continuity_residual"])) < 1e-9
    assert np.max(np.abs(outputs["cw"][0]["charge_continuity_residual"])) < 1e-9


def test_tmux_launch_contract():
    launch = (PILOT / "launch_fixed_flux_quench_tmux.sh").read_text(encoding="utf-8")
    entrypoint = (PILOT / "fixed_flux_quench_tmux_entrypoint.sh").read_text(encoding="utf-8")
    assert "frozen_record_fixed_flux_quench_N16x20_s10_v1" in launch
    assert 'CPU_LIST="${CPU_LIST:-28-55}"' in launch
    assert 'WORKERS="${WORKERS:-28}"' in launch
    assert "--sample-ids 0" in entrypoint
    assert "--resume" in entrypoint
    assert "--analyze" in entrypoint
