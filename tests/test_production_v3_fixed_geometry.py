from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "00_WORKSPACE/CURRENT/final_production_ready_figure_scripts"
PRIOR_DESIGNS = PACKAGE / "prior_designs"
SHARED = PACKAGE / "_shared_src"
sys.path.insert(0, str(SHARED))

from b1_controller_frame import b1_cases  # noqa: E402
from campaign_cases import expand_cases  # noqa: E402


ACTIVE = (
    "00_validation",
    "01_bulk_width_gate",
    "02_pure_wall_master",
    "03_chirality_replay",
    "04_maxmix_master",
    "05_scans_and_controls",
    "06_b1_controller_frame",
)
NY_GRID = {20, 30, 40, 50, 60}


def config(bundle: str) -> dict:
    return json.loads((PRIOR_DESIGNS / bundle / "production_config.json").read_text())


def test_v3_contract_and_geometry_are_locked() -> None:
    for bundle in ACTIVE:
        payload = config(bundle)
        assert payload["locked_contract"]["samples"] == 10
        assert payload["locked_contract"]["sample_shard_size"] == 5
        assert payload["audit_sha256"] == (
            "d0173317608b2da2a45f6185d15a85237e11265917a0ceed6ff79dc761261de9"
        )
        assert "accepted_width.json" not in payload.get("requires", [])

    for bundle in ("01_bulk_width_gate", "02_pure_wall_master", "04_maxmix_master"):
        cases = expand_cases(config(bundle))
        assert all(case["model"]["Nx"] == 20 for case in cases)
        assert {case["model"]["Ny"] for case in cases} == NY_GRID
        assert all(case["run"]["samples"] == 10 for case in cases)

    scans = expand_cases(config("05_scans_and_controls"))
    assert all(case["model"]["Nx"] == 20 for case in scans)
    assert {case["model"]["Ny"] for case in scans} == NY_GRID
    assert all(case["run"]["samples"] == 10 for case in scans)

    b1 = b1_cases(config("06_b1_controller_frame"))
    assert {(case["model"]["Nx"], case["model"]["Ny"]) for case in b1} == {(20, 40)}
    assert sum(case["run"]["samples"] // 5 for case in b1) == 4


def test_v3_exact_queue_counts_and_m3_geometry() -> None:
    assert len(expand_cases(config("01_bulk_width_gate"))) == 20
    assert len(expand_cases(config("02_pure_wall_master"))) == 30
    assert len(expand_cases(config("03_chirality_replay"))) == 2
    assert len(expand_cases(config("04_maxmix_master"))) == 10
    base = expand_cases(config("05_scans_and_controls"))
    assert len(base) == 137
    expanded = expand_cases(
        config("05_scans_and_controls"), m3_wall_sigma=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    assert len(expanded) == 187
    bulk = [case for case in base if case["campaign"] == "M3_BULK"]
    assert len(bulk) == 45
    assert {(case["model"]["Nx"], case["model"]["Ny"]) for case in bulk} == {
        (20, 20),
        (20, 30),
        (20, 40),
        (20, 50),
        (20, 60),
    }


def test_frozen_p1_tree_is_byte_identical() -> None:
    frozen = PRIOR_DESIGNS / "01_p1_existing_completion"
    rows = []
    for path in sorted(path for path in frozen.rglob("*") if path.is_file()):
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(frozen)}\n")
    inventory = hashlib.sha256("".join(rows).encode()).hexdigest()
    assert inventory == "197687b37553ba7f12ef2601869bbef93d09c7193de7a4147a502acae3b8010a"


def test_handoff_manifest_has_exact_base_and_final_counts() -> None:
    handoff = json.loads((PACKAGE / "collaborator_handoff_manifest.json").read_text())
    assert handoff["base_job_count_including_two_h3_descendants"] == 400
    assert handoff["base_ordinary_stochastic_shards"] == 398
    assert handoff["final_ordinary_stochastic_shards"] == {
        "minimum": 458,
        "maximum": 498,
    }
    assert len(handoff["jobs"]) == 400
    assert len({job["ordinal"] for job in handoff["jobs"]}) == 400
    assert all("--case-id" in job["command"] and "--shard-index" in job["command"] for job in handoff["jobs"])


def test_validation_report_only_enumerates_the_deterministic_case(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PACKAGE / "colab_bundle_runner.py"),
            "--drive-root",
            str(tmp_path),
            "--bundle",
            "00_validation",
            "--profile",
            "production",
            "--resume-report-only",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "00_validation: 1 total, 0 current, 0 legacy, 1 pending" in result.stdout
    assert "V0_engine_record_replay_validation shard=0" in result.stdout


def test_b1_report_only_uses_the_chargefix_output_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PACKAGE / "colab_bundle_runner.py"),
            "--drive-root",
            str(tmp_path),
            "--bundle",
            "06_b1_controller_frame",
            "--profile",
            "production",
            "--resume-report-only",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "/06_b1_controller_frame_frame_native_v2" in result.stdout
    sessions = sorted(
        (
            tmp_path
            / "classA_final_production_outputs"
            / "production_10sample_v4_occupied_frame_cycle_resolved"
            / "_bundle_sessions"
        ).glob("*.json")
    )
    assert sessions
    payload = json.loads(sessions[-1].read_text())
    assert payload["output_bundles"] == {
        "06_b1_controller_frame": "06_b1_controller_frame_frame_native_v2"
    }
