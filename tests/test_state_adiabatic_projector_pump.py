from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import nbformat
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot"


def _load():
    name = "state_adiabatic_projector_pump_under_test"
    sys.path.insert(0, str(PROJECT))
    try:
        spec = importlib.util.spec_from_file_location(name, PROJECT / "run_state_adiabatic_projector_pump.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(PROJECT))


def test_locked_campaign_and_source_inventory() -> None:
    campaign = _load()
    config = campaign.load_config(campaign.DEFAULT_CONFIG)
    campaign.validate_config(config)
    rows = campaign.tasks(config)
    assert len(rows) == len({row["task_id"] for row in rows}) == 40
    assert {row["wall"] for row in rows} == {"soft", "hard"}
    assert {row["direction"] for row in rows} == {"ccw", "cw"}
    assert {row["grid_points"] for row in rows} == {65}
    source = campaign.source_context(config)
    assert len(source["burnins"]) == 20
    assert all(len(row["sha256"]) == 64 for row in source["burnins"].values())


def test_flux_grid_has_regulator_and_reverses() -> None:
    campaign = _load()
    config = campaign.load_config(campaign.DEFAULT_CONFIG)
    ccw = campaign.flux_grid(config, +1)
    cw = campaign.flux_grid(config, -1)
    assert len(ccw) == len(cw) == 65
    assert np.array_equal(cw, -ccw)
    assert ccw[0] == -1e-7
    assert np.isclose(ccw[-1] - ccw[0], 2 * np.pi)


def test_toy_projector_path_is_finite_and_charge_conserving() -> None:
    campaign = _load()
    config = campaign.load_config(campaign.DEFAULT_CONFIG)
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(640, 8)) + 1j * rng.normal(size=(640, 8))
    frame, _ = np.linalg.qr(raw)
    task = campaign.tasks(config)[0] | {"grid_intervals": 2, "grid_points": 3}
    toy_config = json.loads(json.dumps(config))
    toy_config["continuation"]["grid_intervals"] = 2
    arrays = campaign.compute_path(np.asarray(frame, dtype=np.complex128), task, toy_config)
    assert arrays["continued_q_x"].shape == (3,)
    assert np.all(np.isfinite(arrays["continued_q_x"]))
    assert np.max(np.abs(arrays["continued_delta_N_total"])) < 1e-10
    assert abs(float(arrays["continued_q_x"][0])) < 1e-5
    assert abs(float(arrays["instantaneous_q_x"][-1])) < 1e-5


def test_completion_pair_detects_corruption(tmp_path: Path) -> None:
    campaign = _load()
    config = campaign.load_config(campaign.DEFAULT_CONFIG)
    task = campaign.tasks(config)[0] | {"grid_intervals": 2, "grid_points": 3}
    toy_config = json.loads(json.dumps(config))
    toy_config["continuation"]["grid_intervals"] = 2
    hashes = campaign.source_hashes()
    config_hash = campaign.scientific_config_hash(config)
    burnin = {"sha256": "a" * 64, "bytes": 123}
    rng = np.random.default_rng(9)
    frame, _ = np.linalg.qr(rng.normal(size=(640, 8)) + 1j * rng.normal(size=(640, 8)))
    arrays = campaign.compute_path(np.asarray(frame, dtype=np.complex128), task, toy_config)
    campaign.publish_pair(tmp_path, task, arrays, config_hash, hashes, burnin, 0.1)
    ok, reason, _ = campaign.verify_pair(tmp_path, task, config_hash, hashes, burnin, toy_config)
    assert ok, reason
    result, _ = campaign.result_paths(tmp_path, task)
    with result.open("ab") as handle:
        handle.write(b"corrupt")
    ok, reason, _ = campaign.verify_pair(tmp_path, task, config_hash, hashes, burnin, toy_config)
    assert not ok
    assert "byte-count" in reason or "checksum" in reason


def test_notebook_exposes_cpu_allocation_theory_progress_and_figures() -> None:
    notebook = nbformat.read(PROJECT / "state_adiabatic_projector_pump.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "CPU_LIST = '56-63'" in source
    assert "WORKERS = 8" in source
    assert "P_\\xi=F_\\xi F_\\xi^\\dagger" in source
    assert "run_state_adiabatic_projector_pump.py" in source
    assert "subprocess.run(run_command" in source
    assert "tmux" not in source.lower()
    assert "state_adiabatic_projector_qx.png" in source
    assert "analysis_summary.json" in source
