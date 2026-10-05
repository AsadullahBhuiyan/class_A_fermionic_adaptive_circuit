from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot"


def _load():
    name = "state_projector_pump_s100_under_test"
    spec = importlib.util.spec_from_file_location(name, PROJECT / "run_state_projector_pump_s100.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_locked_s100_task_table_and_unique_seeds() -> None:
    campaign = _load()
    config = campaign.load_config(campaign.DEFAULT_CONFIG)
    campaign.validate_config(config)
    burnins, pumps = campaign.burnin_tasks(config), campaign.pump_tasks(config)
    assert len(burnins) == len({row["task_id"] for row in burnins}) == 200
    assert len({row["seed"] for row in burnins}) == 200
    assert len(pumps) == len({row["task_id"] for row in pumps}) == 400
    assert {row["wall"] for row in burnins + pumps} == {"soft", "hard"}
    assert {row["direction"] for row in pumps} == {"ccw", "cw"}
    assert config["dynamics"]["burn_in_cycles"] == 2 * config["geometry"]["Ny"] == 48


def test_signed_flux_grid_and_region_contract() -> None:
    campaign = _load()
    config = campaign.load_config(campaign.DEFAULT_CONFIG)
    ccw, cw = campaign.flux_grid(config, 1), campaign.flux_grid(config, -1)
    assert ccw.shape == cw.shape == (65,)
    assert np.array_equal(cw, -ccw)
    assert ccw[0] == -1e-7
    assert np.isclose(ccw[-1] - ccw[0], 2 * np.pi)
    assert config["regions"]["left_x_stop_exclusive"] == 10


def test_static_projector_math_on_coordinate_subspace() -> None:
    campaign = _load()
    config = campaign.load_config(campaign.DEFAULT_CONFIG)
    toy = json.loads(json.dumps(config))
    toy["projector_pump"]["grid_intervals"] = 2
    frame = np.eye(960, 8, dtype=np.complex128)
    task = campaign.pump_tasks(config)[0]
    arrays = campaign.compute_pump_path(frame, task, toy)
    assert arrays["continued_q_x"].shape == (3,)
    assert np.max(np.abs(arrays["continued_delta_N_total"])) < 1e-12
    assert abs(float(arrays["instantaneous_q_x"][-1])) < 1e-12
    assert float(arrays["input_projector_residual"]) == 0.0


def test_atomic_pair_rejects_corruption(tmp_path: Path) -> None:
    campaign = _load()
    config = campaign.load_config(campaign.DEFAULT_CONFIG)
    hashes, config_hash = campaign.source_hashes(), campaign.scientific_config_hash(config)
    task = campaign.burnin_tasks(config)[0]
    frame = np.eye(960, 8, dtype=np.complex128)
    arrays = {"schema": np.asarray(campaign.BURNIN_SCHEMA), "frame": frame, "rank": np.asarray(8)}
    campaign.publish_pair(tmp_path, task, arrays, config_hash, hashes, 0.1)
    ok, reason, _ = campaign.verify_pair(tmp_path, task, config_hash, hashes, config)
    assert ok, reason
    result, _ = campaign.result_paths(tmp_path, task)
    with result.open("ab") as handle:
        handle.write(b"bad")
    ok, reason, _ = campaign.verify_pair(tmp_path, task, config_hash, hashes, config)
    assert not ok
    assert "byte count" in reason or "checksum" in reason


def test_documentation_and_tmux_contract() -> None:
    note = (PROJECT / "docs/state_projector_pump_s100_methods.tex").read_text(encoding="utf-8")
    launch = (PROJECT / "launch_state_projector_pump_s100_tmux.sh").read_text(encoding="utf-8")
    entry = (PROJECT / "state_projector_pump_s100_tmux_entrypoint.sh").read_text(encoding="utf-8")
    assert "F_{\\xi}F_{\\xi}^{\\dagger}" in note
    assert "G_{\\xi}=2C_{\\xi}-\\mathbf{1}" in note
    assert "q_x" in note and "Delta N_R" in note
    assert 'CPU_LIST="${CPU_LIST:-28-55}"' in launch
    assert 'WORKERS="${WORKERS:-28}"' in launch
    assert "--sample-ids 0" in entry
    assert "--analyze" in entry
    assert "wait" in entry.lower()
