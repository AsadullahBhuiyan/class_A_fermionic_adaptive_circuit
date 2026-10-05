from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np

from src.fgtn.occupied_frame import OccupiedFrameState


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "00_WORKSPACE" / "CURRENT" / "frozen_record_flux_charge_pilot"
if str(PILOT) not in sys.path:
    sys.path.insert(0, str(PILOT))
import run_frozen_continuous_ramp as frozen  # noqa: E402


def _config() -> dict:
    config = frozen.load_config(PILOT / "campaign_config.frozen_continuous_ramp_s10_v1.json")
    frozen.validate_config(config)
    return config


def _small_config() -> dict:
    config = copy.deepcopy(_config())
    config["geometry"].update({"Nx": 4, "Ny": 4, "dw_interval": [1, 3]})
    config["regions"] = {
        "left_x_start": 0,
        "left_x_stop_exclusive": 2,
        "right_x_start": 2,
        "right_x_stop_exclusive": 4,
    }
    return config


def test_task_expansion_and_paired_seeds():
    config = _config()
    references = frozen.reference_tasks(config)
    replays = frozen.replay_tasks(config)
    assert len(references) == 20
    assert len(replays) == 120
    assert len({task["task_id"] for task in references + replays}) == 140
    for wall in frozen.WALLS:
        for sample in range(10):
            for cycles in (16, 32, 64):
                pair = [
                    task for task in replays
                    if task["wall"] == wall and task["sample_id"] == sample
                    and task["cycles"] == cycles
                ]
                assert {task["direction"] for task in pair} == {"ccw", "cw"}
                assert len({task["seed"] for task in pair}) == 1


def test_schedule_and_record_prefix_contract():
    config = _config()
    for task in frozen.replay_tasks(config):
        phi = frozen.schedule(task)
        assert phi.shape == (task["cycles"] + 1,)
        assert phi[0] == 0.0
        assert phi[-1] == task["sigma"] * 2.0 * np.pi
    record = [
        {"cycle": cycle, "site_id": 0, "branch_events": []}
        for cycle in range(1, 65)
    ]
    for cycles in (16, 32, 64):
        prefix = frozen.record_prefix(record, cycles)
        assert len(prefix) == cycles
        assert prefix[-1]["cycle"] == cycles


def test_existing_burnins_are_all_verified_and_pinned():
    rows = frozen.verify_burnins(_config())
    assert len(rows) == 20
    assert all(len(row["sha256"]) == 64 for row in rows.values())
    assert all(Path(row["result_path"]).is_file() for row in rows.values())


def _synthetic_arrays(task: dict, reference: bool) -> dict:
    length = int(task["cycles"]) + 1
    arrays = {
        "schema": np.asarray(frozen.REFERENCE_SCHEMA if reference else frozen.RAMP_SCHEMA),
        "cycles": np.arange(length),
        "phi": np.zeros(length),
        "N_left": np.zeros(length),
        "N_right": np.zeros(length),
        "N_total": np.zeros(length),
        "delta_N_left": np.zeros(length),
        "delta_N_right": np.zeros(length),
        "delta_N_total": np.zeros(length),
        "q_x": np.zeros(length),
        "net_injected_charge": np.zeros(length),
        "injection_count": np.zeros(length),
        "charge_continuity_residual": np.zeros(length),
        "rank": np.zeros(length),
        "density_x": np.zeros((length, 16)),
        "minimum_selected_probability_by_cycle": np.ones(length),
        "branch_log_probability_by_cycle": np.zeros(length),
        "final_frame_gram_residual": np.asarray(0.0),
    }
    if reference:
        raw = frozen.record_bytes(
            [
                {"cycle": cycle, "site_id": 0, "branch_events": []}
                for cycle in range(1, int(task["cycles"]) + 1)
            ]
        )
        arrays.update(
            {
                "record_json_utf8": np.frombuffer(raw, dtype=np.uint8),
                "record_sha256": np.asarray(frozen.sha256_bytes(raw)),
                "final_frame": np.eye(4, 2, dtype=np.complex128),
            }
        )
    else:
        arrays["reference_sha256"] = np.asarray("r" * 64)
    return arrays


def test_atomic_pair_resume_and_corruption(tmp_path):
    config = _config()
    task = frozen.reference_tasks(config)[0]
    hashes = {name: name for name in frozen.SOURCE_PATHS}
    config_hash = frozen.scientific_config_hash(config)
    dependencies = {"burnin_sha256": "b" * 64}
    frozen.publish_pair(
        tmp_path, task, _synthetic_arrays(task, True), config_hash, hashes,
        dependencies, elapsed_seconds=1.0,
    )
    assert frozen.verify_pair(tmp_path, task, config_hash, hashes, dependencies)[0]
    result, completion = frozen.result_paths(tmp_path, task)
    result.write_bytes(result.read_bytes() + b"corrupt")
    assert not frozen.verify_pair(tmp_path, task, config_hash, hashes, dependencies)[0]
    result.unlink()
    assert completion.is_file()
    assert not frozen.verify_pair(tmp_path, task, config_hash, hashes, dependencies)[0]


def test_campaign_identity_pins_reused_burnins_and_sources(tmp_path):
    config = _config()
    burnins = frozen.verify_burnins(config)
    frozen.write_identity(config, tmp_path, burnins)
    frozen.write_identity(config, tmp_path, burnins)
    changed = copy.deepcopy(burnins)
    changed[next(iter(changed))]["sha256"] = "f" * 64
    try:
        frozen.write_identity(config, tmp_path, changed)
    except RuntimeError as exc:
        assert "identity changed" in str(exc)
    else:
        raise AssertionError("changed burn-in pin was accepted")


def test_exact_record_replay_with_twist_schedule_and_zero_twist_reproduction():
    config = _small_config()
    cycles = 2
    dimension = 2 * 4 * 4
    frame = OccupiedFrameState.random_pure(
        dimension, dimension // 2, rng=np.random.default_rng(91)
    ).snapshot()

    model = frozen._model(config, "soft")
    reference_audit = frozen.RecordAudit(cycles, capture=True)
    reference_observer = frozen.ChargeObserver(
        model, config, np.zeros(cycles + 1), None, "reference"
    )
    reference_result = model.run_markov_circuit(
        frame_init=frame,
        frame_init_prepared=True,
        native_cycle_observer=reference_observer.cycle,
        native_event_observer=reference_observer.event,
        trajectory_weight_observer=reference_audit,
        **frozen._engine_kwargs(config, "soft", cycles, seed=101),
    )
    record = frozen.record_prefix(reference_audit.entries, cycles)

    outputs = {}
    for direction, sigma in (("zero", 0), ("ccw", 1), ("cw", -1)):
        phi = sigma * np.linspace(0.0, 2.0 * np.pi, cycles + 1)
        model = frozen._model(config, "soft")
        audit = frozen.RecordAudit(cycles)
        observer = frozen.ChargeObserver(model, config, phi, None, "replay")
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
            **frozen._engine_kwargs(config, "soft", cycles, seed=202),
        )
        outputs[direction] = (observer.arrays(config, audit), result["native_final"])

    reference_frame = np.asarray(reference_result["native_final"]["frame"])
    zero_frame = np.asarray(outputs["zero"][1]["frame"])
    assert np.allclose(
        reference_frame @ reference_frame.conj().T,
        zero_frame @ zero_frame.conj().T,
        rtol=0.0,
        atol=1e-12,
    )
    for field in ("rank", "net_injected_charge", "injection_count"):
        assert np.array_equal(outputs["ccw"][0][field], outputs["cw"][0][field])
    assert np.array_equal(outputs["ccw"][0]["phi"], -outputs["cw"][0]["phi"])
    for direction in outputs:
        assert np.max(np.abs(outputs[direction][0]["charge_continuity_residual"])) < 1e-9


def test_launch_is_resumable_and_core_isolated():
    launcher = (PILOT / "launch_frozen_continuous_ramp_tmux.sh").read_text(encoding="utf-8")
    entrypoint = (PILOT / "frozen_continuous_ramp_tmux_entrypoint.sh").read_text(encoding="utf-8")
    assert "frozen_record_continuous_ramp_N16x20_s10_v1" in launcher
    assert "28-55" in launcher and "28-55" in entrypoint
    assert "--sample-ids 0" in entrypoint
    assert "--resume" in entrypoint and "--analyze" in entrypoint
    assert "taskset" in entrypoint
