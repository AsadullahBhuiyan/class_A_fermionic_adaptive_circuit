from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot"
NY_VALUES = (28, 30, 32, 34, 36)


def _load():
    name = "state_projector_pump_size_series_under_test"
    spec = importlib.util.spec_from_file_location(
        name, PROJECT / "run_state_projector_pump_size_series.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _config(campaign, ny: int):
    return campaign.load_config(
        PROJECT / f"campaign_config.state_projector_pump_n20x{ny}_s100_v3.json"
    )


def test_all_size_contracts_and_tasks_are_locked() -> None:
    campaign = _load()
    all_seeds = []
    config_hashes = []
    for ny in NY_VALUES:
        config = _config(campaign, ny)
        campaign.validate_config(config)
        assert config["campaign_id"] == f"N20x{ny}_state_projector_pump_s100_v3"
        assert config["geometry"]["Nx"] == 20
        assert config["dynamics"]["burn_in_cycles"] == 2 * ny
        assert config["dynamics"]["dtype"] == "complex128"
        assert "endpoint_central_charge" not in config
        burnins, pumps = campaign.burnin_tasks(config), campaign.pump_tasks(config)
        assert len(burnins) == 200
        assert len(pumps) == 400
        assert len({task["task_id"] for task in burnins + pumps}) == 600
        all_seeds.extend(task["seed"] for task in burnins)
        config_hashes.append(campaign.scientific_config_hash(config))
    assert len(all_seeds) == len(set(all_seeds)) == 1000
    assert len(config_hashes) == len(set(config_hashes)) == len(NY_VALUES)


def test_dynamic_frame_dimension_and_corruption_rejection(tmp_path: Path) -> None:
    campaign = _load()
    config = _config(campaign, 28)
    hashes = campaign.source_hashes()
    config_hash = campaign.scientific_config_hash(config)
    task = campaign.burnin_tasks(config)[0]
    frame = np.eye(2 * 20 * 28, 8, dtype=np.complex128)
    arrays = {
        "schema": np.asarray(campaign.BURNIN_SCHEMA),
        "frame": frame,
        "rank": np.asarray(8),
        "density_x": np.zeros(20),
    }
    campaign.publish_pair(tmp_path, task, arrays, config_hash, hashes, 0.1)
    ok, reason, _ = campaign.verify_pair(tmp_path, task, config_hash, hashes, config)
    assert ok, reason
    result, _ = campaign.result_paths(tmp_path, task)
    with result.open("ab") as handle:
        handle.write(b"corrupt")
    ok, reason, _ = campaign.verify_pair(tmp_path, task, config_hash, hashes, config)
    assert not ok
    assert "byte count" in reason or "checksum" in reason


def test_completed_ny24_runner_is_unchanged() -> None:
    path = PROJECT / "run_state_projector_pump_s100.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == "8179ee9089b045bec26726d5720c2d87bef0247086a559802ffcd3f3f5866886"


def test_size_series_has_no_central_charge_dependency_or_output() -> None:
    campaign = _load()
    assert "endpoint_cft" not in campaign.SOURCE_PATHS
    source = (PROJECT / "run_state_projector_pump_size_series.py").read_text()
    assert "endpoint_central_charge" not in source
    assert "endpoint_c_eff" not in source
    assert "endpoint_entropy" not in source


def test_size_series_orchestration_and_analysis_contract() -> None:
    launch = (PROJECT / "launch_state_projector_pump_size_series_tmux.sh").read_text()
    entry = (PROJECT / "state_projector_pump_size_series_tmux_entrypoint.sh").read_text()
    analysis = (PROJECT / "analyze_state_projector_pump_across_sizes.py").read_text()
    note = (PROJECT / "docs/state_projector_pump_s100_methods.tex").read_text()
    assert 'SESSION="${SESSION:-state_projector_pump_N20_Ny28_36_s100_v3}"' in launch
    assert "NY_VALUES=(28 30 32 34 36)" in entry
    assert "--sample-ids 0" in entry
    assert "--resume --analyze" in entry
    assert "analyze_state_projector_pump_across_sizes.py" in entry
    assert "NY_VALUES = (24, 28, 30, 32, 34, 36)" in analysis
    assert "$N_y=28,30,32,34,36$" in note
    assert "central_charge" not in analysis
