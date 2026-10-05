from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT = REPO_ROOT / "00_WORKSPACE" / "CURRENT" / "frozen_record_flux_charge_pilot"
RUNNER = PROJECT / "run_state_projector_pump_variants.py"
CONFIGS = {
    "size": PROJECT / "campaign_config.state_projector_pump_n24x24_s100_v1.json",
    "grid": PROJECT / "campaign_config.state_projector_pump_n20x24_grid128_s100_v1.json",
}


def _load_runner():
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(PROJECT))
        spec = importlib.util.spec_from_file_location("state_projector_pump_variants_test", RUNNER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path


def test_locked_variant_contracts_and_task_counts() -> None:
    runner = _load_runner()
    for path in CONFIGS.values():
        config = json.loads(path.read_text(encoding="utf-8"))
        runner.validate_config(config)
        assert len(runner.burnin_tasks(config)) == 200
        assert len(runner.pump_tasks(config)) == 400
        assert len({row["seed"] for row in runner.burnin_tasks(config)}) == 200
        assert config["dynamics"]["burn_in_cycles"] == 48
        assert config["dynamics"]["sequence"] == "raster_y"
        assert config["dynamics"]["dtype"] == "complex128"


def test_grid128_reuses_exact_source_and_has_129_points() -> None:
    runner = _load_runner()
    config = json.loads(CONFIGS["grid"].read_text(encoding="utf-8"))
    source = config["endpoint_source"]
    assert source["campaign_id"] == "N20x24_state_projector_pump_s100_v1"
    assert source["config_sha256"] == "aafda9bcd52995b486df745762e249ec5b75af12561ca050601ec6bc3dc6245f"
    assert runner.base.scientific_config_hash(
        runner.base.load_config(PROJECT / source["config"])
    ) == source["config_sha256"]
    assert runner.base.source_hashes() == source["source_hashes"]
    ccw = runner.flux_grid(config, 1)
    cw = runner.flux_grid(config, -1)
    assert ccw.shape == cw.shape == (129,)
    np.testing.assert_allclose(ccw, -cw)
    assert abs(ccw[64] - (np.pi - 1e-7)) < 1e-14


def test_n24_geometry_uses_larger_wall_separation() -> None:
    runner = _load_runner()
    config = json.loads(CONFIGS["size"].read_text(encoding="utf-8"))
    assert config["geometry"]["dw_interval"] == [6, 18]
    assert config["regions"]["left_x_stop_exclusive"] == 12
    model = runner._model(config, "soft")
    assert (model.Nx, model.Ny, model.nshell) == (24, 24, 1)
    assert model.dw_interval == (6, 18)


def test_launchers_queue_behind_current_socket_owners() -> None:
    grid = (PROJECT / "state_projector_pump_grid128_tmux_entrypoint.sh").read_text()
    size = (PROJECT / "state_projector_pump_n24x24_tmux_entrypoint.sh").read_text()
    assert "state_projector_pump_N20x24_dense_s100_v1" in grid
    assert 'CPU_LIST="${CPU_LIST:-0-27}"' in grid
    assert "state_projector_pump_N20_Ny28_36_s100_v3" in size
    assert 'CPU_LIST="${CPU_LIST:-28-55}"' in size
    assert "while tmux has-session" in grid
    assert "while tmux has-session" in size
