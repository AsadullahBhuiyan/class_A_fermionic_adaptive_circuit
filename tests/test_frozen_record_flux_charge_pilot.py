from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = REPO_ROOT / "00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot"
RUNNER_PATH = PILOT_ROOT / "run_campaign.py"
SOURCE_ANALYSIS_PATH = PILOT_ROOT / "analyze_source_subtracted.py"
CONFIG_PATH = PILOT_ROOT / "campaign_config.v1.json"


def load_runner():
    name = f"frozen_record_flux_charge_runner_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_source_analysis():
    name = f"frozen_record_source_analysis_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, SOURCE_ANALYSIS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_task_expansion_and_opposite_offset_convention() -> None:
    runner = load_runner()
    config = load_config()
    tasks = runner.expand_tasks(config, "configuration-hash")
    assert len(tasks) == 68
    assert len({task["task_id"] for task in tasks}) == 68
    offset = 1e-7
    for arm in ("soft", "hard"):
        ccw = [task for task in tasks if task["arm"] == arm and task["direction"] == "ccw"]
        cw = [task for task in tasks if task["arm"] == arm and task["direction"] == "cw"]
        assert len(ccw) == len(cw) == 17
        assert np.isclose(ccw[0]["phi"], -offset, rtol=0.0, atol=0.0)
        assert np.isclose(ccw[-1]["phi"], 2 * np.pi - offset, rtol=0.0, atol=1e-15)
        assert np.isclose(cw[0]["phi"], offset, rtol=0.0, atol=0.0)
        assert np.isclose(cw[-1]["phi"], -2 * np.pi + offset, rtol=0.0, atol=1e-15)
        assert [task["sweep_fraction"] for task in ccw] == [
            task["sigma"] * task["twist_index"] / 16 for task in ccw
        ]


def test_injected_charge_is_derived_once_from_the_fixed_record() -> None:
    runner = load_runner()
    record = [
        {
            "cycle": 1,
            "site_id": 0,
            "branch_events": [
                {"channel": "Ap", "kind": "measurement", "outcome_occupied": True},
                {"channel": "Ap", "kind": "correction", "target_occupied": False},
                {"channel": "Am", "kind": "measurement", "outcome_occupied": False},
                {"channel": "Am", "kind": "correction", "target_occupied": True},
                {"channel": "Bp", "kind": "measurement", "outcome_occupied": False},
            ],
        }
    ]
    assert runner.injected_charge(record) == 0


def test_endpoint_regions_partition_centered_covariance() -> None:
    runner = load_runner()
    config = load_config()
    nx, ny = config["geometry"]["Nx"], config["geometry"]["Ny"]
    occupations = np.zeros(2 * nx * ny)
    x = runner.mode_x(nx, ny)
    occupations[x < nx // 2] = 0.25
    occupations[x >= nx // 2] = 0.75
    centered = np.diag(2.0 * occupations - 1.0).astype(np.complex128)
    charge = runner.endpoint_charges(centered, config)
    assert np.isclose(charge["N_left"], occupations[x < nx // 2].sum())
    assert np.isclose(charge["N_right"], occupations[x >= nx // 2].sum())
    assert np.isclose(charge["N_total"], occupations.sum())
    assert np.allclose(charge["charge_by_x"].sum(), occupations.sum())


def test_source_analysis_reduces_record_to_signed_correction_counts() -> None:
    analysis = load_source_analysis()
    record = [
        {
            "site_id": 2,
            "branch_events": [
                {"channel": "Ap", "kind": "measurement", "outcome_occupied": True},
                {"channel": "Ap", "kind": "correction", "target_occupied": False},
                {"channel": "Am", "kind": "measurement", "outcome_occupied": False},
                {"channel": "Am", "kind": "correction", "target_occupied": True},
            ],
        },
        {
            "site_id": 2,
            "branch_events": [
                {"channel": "Ap", "kind": "measurement", "outcome_occupied": False},
                {"channel": "Ap", "kind": "correction", "target_occupied": True},
            ],
        },
    ]
    assert analysis.correction_multiplicities(record) == {(2, "Am"): 1}


def test_source_profile_exactly_resolves_normalized_orbital_weight() -> None:
    analysis = load_source_analysis()
    nx, ny = 2, 1
    orbital = np.sqrt(np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)).astype(
        np.complex128
    )

    class Model:
        WF_Ap = orbital[:, None, None]
        WF_Am = orbital[:, None, None]
        WF_Bp = orbital[:, None, None]
        WF_Bm = orbital[:, None, None]

    profile = analysis.source_profile_from_model(
        Model(), {(0, "Ap"): 2}, nx=nx, ny=ny
    )
    assert np.allclose(profile, [0.6, 1.4], rtol=0.0, atol=1e-15)
    assert np.isclose(profile.sum(), 2.0, rtol=0.0, atol=1e-15)


def test_result_completion_verification_fails_closed(tmp_path: Path) -> None:
    runner = load_runner()
    result = tmp_path / "point.npz"
    completion = tmp_path / "point.completion.json"
    runner.atomic_npz(
        result,
        schema=np.asarray(runner.TASK_SCHEMA),
        task_hash=np.asarray("task-hash"),
        record_sha256=np.asarray("record-hash"),
    )
    runner.atomic_json(
        completion,
        {
            "schema": runner.COMPLETION_SCHEMA,
            "kind": "twist_replay",
            "task_hash": "task-hash",
            "record_sha256": "record-hash",
            "result": runner.file_record(result),
        },
    )
    assert runner.verify_task_result(result, completion, "task-hash", "record-hash")
    with result.open("ab") as handle:
        handle.write(b"corruption")
    assert not runner.verify_task_result(result, completion, "task-hash", "record-hash")


def test_readme_documents_offset_quantization_boundary_and_tmux() -> None:
    readme = (PILOT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "phi_j = -sigma * 1e-7" in readme
    assert "exponentially small avoided crossing" in readme
    assert "0.9999999976" in readme
    assert "0.983971" in readme
    assert "0.98080969" in readme
    assert "no quantization acceptance gate" in readme
    assert "launch_tmux.sh --dry-run" in readme
    assert "--resume" in readme


def test_tmux_launcher_dry_run_reports_68_tasks() -> None:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [0]
    cpu = str(affinity[0])
    environment = {
        **os.environ,
        "CPU_LIST": cpu,
        "WORKERS": "1",
        "BLAS_THREADS": "1",
        "CAMPAIGN_ID": "dry_run_only",
    }
    result = subprocess.run(
        [str(PILOT_ROOT / "launch_tmux.sh"), "--dry-run"],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert '"reference_task_count": 2' in result.stdout
    assert '"replay_task_count": 68' in result.stdout
    assert '"origin_rule": "phi_initial = -sigma * phi_offset"' in result.stdout


def test_small_serial_parallel_campaign_and_resume(tmp_path: Path) -> None:
    config = load_config()
    config["root_seed"] = 9917
    config["geometry"].update(
        {"Nx": 4, "Ny": 4, "cycles": 2, "dw_interval": [1, 2]}
    )
    config["regions"] = {
        "left_x_start": 0,
        "left_x_stop_exclusive": 2,
        "right_x_start": 2,
        "right_x_stop_exclusive": 4,
    }
    config["twist"]["points_per_direction"] = 2
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [0]
    cpus = affinity[: min(2, len(affinity))]
    config["parallel"].update(
        {"cpu_list": ",".join(map(str, cpus)), "workers": len(cpus), "blas_threads": 1}
    )
    config_path = tmp_path / "smoke_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_root = tmp_path / "outputs"
    command = [
        sys.executable,
        "-u",
        str(RUNNER_PATH),
        "all",
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
        "--campaign-id",
        "smoke",
        "--cpu-list",
        ",".join(map(str, cpus)),
        "--workers",
        str(len(cpus)),
        "--blas-threads",
        "1",
    ]
    first = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    campaign = output_root / "smoke"
    assert (campaign / "manifest.json").is_file()
    manifest = json.loads((campaign / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["task_count"] == 8
    assert manifest["completed_task_count"] == 8
    assert (campaign / "figures/frozen_record_flux_charge.pdf").is_file()
    assert "[complete] verified=8/8" in first.stdout

    second = subprocess.run(
        [*command, "--resume"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "complete=8 pending=0" in second.stdout
    assert "launched_this_run=0" in second.stdout

    changed_config = json.loads(config_path.read_text(encoding="utf-8"))
    changed_config["root_seed"] += 1
    changed_path = tmp_path / "changed_config.json"
    changed_path.write_text(json.dumps(changed_config), encoding="utf-8")
    mismatch_command = command[:]
    mismatch_command[mismatch_command.index(str(config_path))] = str(changed_path)
    mismatch = subprocess.run(
        [*mismatch_command, "--resume"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert mismatch.returncode != 0
    assert "choose a new campaign ID" in mismatch.stderr

    serial_command = [
        *command[:],
    ]
    serial_command[serial_command.index("smoke")] = "serial"
    serial_command[serial_command.index(str(len(cpus)))] = "1"
    subprocess.run(
        serial_command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    with np.load(campaign / "charge_vs_phi.npz", allow_pickle=False) as parallel:
        with np.load(output_root / "serial/charge_vs_phi.npz", allow_pickle=False) as serial:
            for field in (
                "phi",
                "N_left",
                "N_right",
                "N_total",
                "delta_N_left",
                "delta_N_right",
                "q_wall",
                "charge_continuity_residual",
                "regional_balance_residual",
                "minimum_selected_probability",
                "branch_log_probability",
                "charge_by_x",
            ):
                if field == "phi":
                    assert np.array_equal(parallel[field], serial[field]), field
                else:
                    assert np.allclose(
                        parallel[field], serial[field], rtol=0.0, atol=1e-12
                    ), field
