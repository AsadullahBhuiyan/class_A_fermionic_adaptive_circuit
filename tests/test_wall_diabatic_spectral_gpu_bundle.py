from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = (
    ROOT
    / "00_WORKSPACE/CURRENT/final_production_new_designs"
    / "12_wall_diabatic_spectral_pump_gpu"
)


def test_bundle_files_and_primary_only_contract() -> None:
    required = {
        "run_campaign.py",
        "gpu_backend.py",
        "spectral_cpu_reference.py",
        "campaign_config.json",
        "build_notebook.py",
        "run_wall_diabatic_spectral_pump_gpu.ipynb",
        "README.md",
    }
    assert all((BUNDLE / name).is_file() for name in required)
    runner = (BUNDLE / "run_campaign.py").read_text(encoding="utf-8")
    assert "sensitivity_tasks" not in runner
    assert 'REVISION = "wall_diabatic_spectral_pump_gpu_primary_v1"' in runner
    assert "publish_from_scratch" in runner
    assert "A100 complex128 parity gate failed" in runner


def test_notebook_has_safe_defaults_and_disconnect_cell() -> None:
    notebook = json.loads(
        (BUNDLE / "run_wall_diabatic_spectral_pump_gpu.ipynb").read_text(
            encoding="utf-8"
        )
    )
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    assert any("REPORT_ONLY = True" in source for source in sources)
    assert any("MAX_NEW_BATCHES = None" in source for source in sources)
    assert any("LANES = None" in source for source in sources)
    assert any("LEGACY_ENDPOINT_ROOT" in source for source in sources)
    assert any("NEW_ENDPOINT_ROOT" in source for source in sources)
    assert sources[-1] == (
        "from google.colab import runtime\n"
        "runtime.unassign()\n"
        "print('done')\n"
    )


def test_campaign_expands_to_locked_primary_and_optional_bridge_counts() -> None:
    spec = importlib.util.spec_from_file_location(
        "spectral_gpu_bundle_reference", BUNDLE / "spectral_cpu_reference.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = json.loads((BUNDLE / "campaign_config.json").read_text(encoding="utf-8"))
    module.validate_config(config)
    assert len(module.tasks(config, include_bridge=False)) == 1600
    assert len(module.tasks(config, include_bridge=True)) == 1750
