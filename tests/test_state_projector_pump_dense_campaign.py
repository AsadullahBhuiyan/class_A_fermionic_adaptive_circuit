from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT = REPO_ROOT / "00_WORKSPACE" / "CURRENT" / "frozen_record_flux_charge_pilot"
RUNNER = PROJECT / "run_state_projector_pump_dense_s100.py"
CONFIG = PROJECT / "campaign_config.state_projector_pump_n20x24_s100_dense_v1.json"


def _load_runner():
    spec = importlib.util.spec_from_file_location("state_projector_pump_dense_campaign_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dense_campaign_contract_and_tasks() -> None:
    runner = _load_runner()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    runner.validate_config(config)
    assert config["geometry"]["nshell"] is None
    assert config["geometry"]["Ny"] == 24
    assert config["dynamics"]["burn_in_cycles"] == 48
    assert config["dynamics"]["sequence"] == "raster_y"
    assert config["dynamics"]["perfect_correction"] is True
    assert config["dynamics"]["dtype"] == "complex128"
    burnins = runner.burnin_tasks(config)
    pumps = runner.pump_tasks(config)
    assert len(burnins) == 200
    assert len(pumps) == 400
    assert len({row["seed"] for row in burnins}) == 200
    assert len({row["task_id"] for row in burnins + pumps}) == 600


def test_dense_model_uses_canonical_none_setting() -> None:
    runner = _load_runner()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for wall in ("soft", "hard"):
        model = runner._model(config, wall)
        assert model.nshell is None
        assert model.Nx == 20
        assert model.Ny == 24
        assert model.DW is True
