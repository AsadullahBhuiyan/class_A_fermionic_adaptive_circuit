from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "00_WORKSPACE/CURRENT/final_production_new_designs"
OFFLINE_QUEUE_BUNDLES = (
    "02_wall_cft_windows",
    "03_h1_modular_response",
)


@pytest.mark.parametrize("bundle", OFFLINE_QUEUE_BUNDLES)
def test_non_server_queue_runs_report_only_without_siblings_or_prior_designs(
    tmp_path: Path, bundle: str
) -> None:
    package = tmp_path / "final_production_new_designs"
    (package / "_shared_src").mkdir(parents=True)
    shutil.copy2(SOURCE / "colab_bundle_runner.py", package)
    shutil.copy2(SOURCE / "drive_remote_commit.py", package)
    shutil.copy2(SOURCE / "bundle_layout.py", package)
    shutil.copy2(SOURCE / "pilot_plan.json", package)
    shutil.copy2(
        SOURCE / "_shared_src/production_runtime.py",
        package / "_shared_src/production_runtime.py",
    )
    shutil.copytree(SOURCE / bundle, package / bundle)
    (tmp_path / "drive").mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(package / "colab_bundle_runner.py"),
            "--drive-root",
            str(tmp_path / "drive"),
            "--bundle",
            bundle,
            "--profile",
            "production",
            "--resume-report-only",
            "--heartbeat-seconds",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "[SUMMARY]" in result.stdout
    assert bundle in result.stdout
    assert not (package / "prior_designs").exists()
    assert sorted(
        path.name
        for path in package.iterdir()
        if (path / "production_config.json").is_file()
    ) == [bundle]
