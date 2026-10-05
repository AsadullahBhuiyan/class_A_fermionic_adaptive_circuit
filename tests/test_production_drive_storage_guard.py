from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NEW_CAMPAIGN = ROOT / "00_WORKSPACE/CURRENT/final_production_new_designs"
PRIOR_CAMPAIGN = ROOT / "00_WORKSPACE/CURRENT/final_production_ready_figure_scripts"
SHARED = NEW_CAMPAIGN / "_shared_src"
sys.path.insert(0, str(SHARED))

from drive_storage_guard import storage_status, tree_bytes  # noqa: E402


QUEUE_BUNDLES = (
    "01_p1_chern_dynamics",
    "02_wall_cft_windows",
    "03_h1_modular_response",
    "08_h1_endpoint_packet",
)


def test_tree_bytes_counts_nested_regular_files(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.bin").write_bytes(b"1234")
    (tmp_path / "two.bin").write_bytes(b"56")
    assert tree_bytes(tmp_path) == 6


def test_storage_guard_reserves_headroom_before_launch(tmp_path: Path) -> None:
    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle" / "archive.tar.gz").write_bytes(b"123456")
    allowed = storage_status(
        tmp_path,
        working_limit_bytes=10,
        absolute_edge_bytes=20,
        required_headroom_bytes=4,
    )
    blocked = storage_status(
        tmp_path,
        working_limit_bytes=9,
        absolute_edge_bytes=20,
        required_headroom_bytes=4,
    )
    assert allowed["clear_to_run"] is True
    assert blocked["clear_to_run"] is False
    assert allowed["bundle_bytes"] == {"bundle": 6}


def test_storage_guard_blocks_at_absolute_edge(tmp_path: Path) -> None:
    (tmp_path / "archive.tar.gz").write_bytes(b"123456")
    status = storage_status(
        tmp_path,
        working_limit_bytes=100,
        absolute_edge_bytes=6,
        required_headroom_bytes=0,
    )
    assert status["clear_to_run"] is False


def test_redesigned_queue_notebooks_use_only_the_new_parent() -> None:
    runner_source = (NEW_CAMPAIGN / "colab_bundle_runner.py").read_text()
    assert "drive_storage_guard.py" in runner_source
    assert "run_storage_guard" in runner_source
    for bundle in QUEUE_BUNDLES:
        path = NEW_CAMPAIGN / bundle / "run_production_bundle.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "".join(
            line for cell in notebook["cells"] for line in cell.get("source", [])
        )
        assert "final_production_new_designs" in source
        assert "final_production_ready_figure_scripts" not in source
        assert "CASE_IDS = ()" in source
        assert "CASE_PREFIXES = ()" in source
        assert "RUN_QUEUE = True" in source
        assert "RESUME_REPORT_ONLY = True" in source
        assert "runtime.unassign()" in source
        assert (NEW_CAMPAIGN / bundle / "src/drive_storage_guard.py").is_file()


def test_prior_case_listing_still_reports_mixed_shard_counts(tmp_path: Path) -> None:
    bundle = PRIOR_CAMPAIGN / "prior_designs/05_scans_and_controls"
    result = subprocess.run(
        [
            sys.executable,
            str(bundle / "run_bundle.py"),
            "--drive-root",
            str(tmp_path),
            "--mode",
            "smoke",
            "--list-cases-json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = json.loads(result.stdout)
    counts = {row["case_id"]: row["shard_count"] for row in rows}
    assert not [case for case in counts if "postselect" in case]
    assert set(counts.values()) == {2}
