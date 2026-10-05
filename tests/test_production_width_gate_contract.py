from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SHARED = (
    ROOT
    / "00_WORKSPACE"
    / "CURRENT"
    / "final_production_ready_figure_scripts"
    / "_shared_src"
)
sys.path.insert(0, str(SHARED))

from run_core_shard import _accepted_width, _validate_launch_gates  # noqa: E402
from run_h3_shard import _accepted_width as _h3_accepted_width  # noqa: E402


def _joint_gate() -> dict:
    return {
        "schema_version": 1,
        "status": "accepted",
        "accepted_Nx": 20,
        "W1_candidate_Nx": 20,
        "exact_B0_transverse_gate": {"status": "accepted", "accepted_Nx": 20},
        "exact_B0_reference_sha256": "a" * 64,
    }


def test_joint_width_gate_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "accepted_width.json"
    path.write_text(json.dumps(_joint_gate()), encoding="utf-8")
    assert _accepted_width(str(path)) == 20
    assert _h3_accepted_width(str(path)) == 20


def test_b0_only_gate_cannot_launch_descendants(tmp_path: Path) -> None:
    path = tmp_path / "b0.json"
    path.write_text(
        json.dumps({"status": "accepted", "accepted_Nx": 20}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="joint W1/B0"):
        _accepted_width(str(path))
    with pytest.raises(ValueError, match="joint W1/B0"):
        _h3_accepted_width(str(path))


def test_launch_decisions_are_case_specific_and_hashed(tmp_path: Path) -> None:
    path = tmp_path / "core_gate_decisions.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "accepted",
                "decisions": {"P1_viable": True, "S1_viable": False},
            }
        ),
        encoding="utf-8",
    )
    config = {"launch_gate_requirements": {"M3_BULK": ["P1_viable"]}}
    receipt = _validate_launch_gates(config, {"campaign": "M3_BULK", "case_id": "case"}, str(path))
    assert receipt is not None and receipt["file_name"] == path.name
    assert len(receipt["sha256"]) == 64


def test_unaccepted_launch_decision_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "decisions.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "accepted",
                "decisions": {"S1_viable": False},
            }
        ),
        encoding="utf-8",
    )
    config = {"launch_gate_requirements": {"M2": ["S1_viable"]}}
    with pytest.raises(RuntimeError, match="not viable"):
        _validate_launch_gates(config, {"campaign": "M2", "case_id": "case"}, str(path))
