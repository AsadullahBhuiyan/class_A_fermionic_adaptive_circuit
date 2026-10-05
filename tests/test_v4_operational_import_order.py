from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = (
    ROOT / "00_WORKSPACE" / "CURRENT" / "final_production_new_designs"
)


def test_actual_parent_imports_the_operational_drive_helper() -> None:
    runner = CAMPAIGN_ROOT / "colab_bundle_runner.py"
    program = f"""
import importlib.util
import json
import pathlib
import sys

runner = pathlib.Path({str(runner)!r})
spec = importlib.util.spec_from_file_location("_actual_v4_parent", runner)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
import drive_remote_commit
print(json.dumps({{
    "helper": str(pathlib.Path(drive_remote_commit.__file__).absolute()),
    "exact_path_api": hasattr(drive_remote_commit.DriveRemoteCommitter, "verify_record_for_path"),
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=CAMPAIGN_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert Path(payload["helper"]) == CAMPAIGN_ROOT / "drive_remote_commit.py"
    assert payload["exact_path_api"] is True


def test_parent_ignores_a_cached_frozen_drive_helper() -> None:
    runner = CAMPAIGN_ROOT / "colab_bundle_runner.py"
    frozen = (
        CAMPAIGN_ROOT
        / "08_h1_endpoint_packet"
        / "src"
        / "drive_remote_commit.py"
    )
    program = f"""
import importlib.util
import json
import pathlib
import sys

frozen = pathlib.Path({str(frozen)!r})
frozen_spec = importlib.util.spec_from_file_location("drive_remote_commit", frozen)
frozen_module = importlib.util.module_from_spec(frozen_spec)
sys.modules["drive_remote_commit"] = frozen_module
frozen_spec.loader.exec_module(frozen_module)

runner = pathlib.Path({str(runner)!r})
spec = importlib.util.spec_from_file_location("_parent_after_frozen", runner)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print(json.dumps({{
    "selected": str(module._DRIVE_HELPER_PATH),
    "module": module.DriveRemoteCommitter.__module__,
    "exact_path_api": hasattr(module.DriveRemoteCommitter, "verify_record_for_path"),
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=CAMPAIGN_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert Path(payload["selected"]) == CAMPAIGN_ROOT / "drive_remote_commit.py"
    assert payload["module"] == "classA_parent_operational_drive_remote_commit"
    assert payload["exact_path_api"] is True
