"""Generate the two fixed Colab lanes and deployment manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "campaign_config.json").read_text(encoding="utf-8"))
REVISION = CONFIG["sampling_revision"]
BUNDLE_DRIVE_DIR = (
    "/content/drive/MyDrive/final_production_new_designs/"
    "18_hard_wall_purification_alpha_endpoint"
)
NOTEBOOKS = {
    "A": "run_hard_wall_purification_alpha_endpoint_lane_a.ipynb",
    "B": "run_hard_wall_purification_alpha_endpoint_lane_b.ipynb",
}


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def notebook(lane: str) -> dict[str, object]:
    lane = lane.upper()
    if lane not in NOTEBOOKS:
        raise ValueError(f"unknown lane {lane!r}")
    config_literal = repr(CONFIG)
    return {
        "cells": [
            markdown(
                f"# Hard-wall purification alpha sweep — lane {lane}\n\n"
                "Endpoint-only occupation-derived finite-time single-particle Lyapunov "
                "diagnostics. This is **not** a tangent-cocycle or Choi spectrum. The "
                "two notebooks own disjoint complete `(Ny, alpha_1)` configurations and "
                "may run concurrently. Results are immutable five-trajectory shards; "
                "state and RNG are checkpointed every ten cycles.\n"
            ),
            code(
                "from google.colab import drive\n"
                "drive.mount('/content/drive')\n"
            ),
            code(
                "from pathlib import Path\n"
                "import json\n"
                "import torch\n\n"
                f"LANE = {lane!r}\n"
                "REPORT_ONLY = False\n"
                "# Qualification default: run exactly one worst-case execution batch.\n"
                "# After it succeeds below 38 GiB, set this to None for the full lane.\n"
                "MAX_NEW_EXECUTION_BATCHES = 1\n"
                f"BUNDLE_DRIVE_DIR = Path({BUNDLE_DRIVE_DIR!r})\n"
                f"OUTPUT_ROOT = Path('/content/drive/MyDrive/classA_final_production_outputs/{REVISION}')\n"
                "LOCAL_BUNDLE_DIR = Path(f'/content/18_hard_wall_purification_alpha_endpoint_lane_{LANE.lower()}')\n"
                "SCRATCH_ROOT = Path(f'/content/hard_wall_purification_alpha_endpoint_scratch_lane_{LANE.lower()}')\n"
                f"CONFIG = {config_literal}\n\n"
                "if not torch.cuda.is_available():\n"
                "    raise RuntimeError('This production notebook requires CUDA.')\n"
                "gpu_name = torch.cuda.get_device_name(0)\n"
                "gpu_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3\n"
                "if 'A100' not in gpu_name or gpu_gib < 38.0:\n"
                "    raise RuntimeError(f'Expected an A100 40-GB-class GPU, got {gpu_name} ({gpu_gib:.2f} GiB)')\n"
                "if torch.complex128 != getattr(torch, CONFIG['dtype']):\n"
                "    raise RuntimeError('Locked dtype is not complex128.')\n"
                "print(json.dumps({\n"
                "    'lane': LANE, 'config': CONFIG,\n"
                "    'bundle': str(BUNDLE_DRIVE_DIR), 'output': str(OUTPUT_ROOT),\n"
                "    'scratch': str(SCRATCH_ROOT), 'gpu': gpu_name, 'gpu_GiB': gpu_gib,\n"
                "    'gpu_memory_hard_limit_GiB': CONFIG['gpu_memory_hard_limit_gib'],\n"
                "    'report_only': REPORT_ONLY,\n"
                "    'max_new_execution_batches': MAX_NEW_EXECUTION_BATCHES,\n"
                "}, indent=2, sort_keys=True), flush=True)\n"
            ),
            code(
                "import importlib.util\n"
                "import shutil\n"
                "import sys\n\n"
                "if not BUNDLE_DRIVE_DIR.is_dir():\n"
                "    raise FileNotFoundError(f'Missing Drive bundle: {BUNDLE_DRIVE_DIR}')\n"
                "if LOCAL_BUNDLE_DIR.exists():\n"
                "    shutil.rmtree(LOCAL_BUNDLE_DIR)\n"
                "shutil.copytree(BUNDLE_DRIVE_DIR, LOCAL_BUNDLE_DIR)\n"
                "config_path = Path(f'/content/hard_wall_purification_alpha_endpoint_lane_{LANE.lower()}.json')\n"
                "config_path.write_text(json.dumps(CONFIG, indent=2, sort_keys=True) + '\\n', encoding='utf-8')\n"
                "runner_args = [\n"
                "    '--config', str(config_path), '--lane', LANE,\n"
                "    '--output-root', str(OUTPUT_ROOT), '--scratch-root', str(SCRATCH_ROOT),\n"
                "]\n"
                "if REPORT_ONLY:\n"
                "    runner_args.append('--report-only')\n"
                "if MAX_NEW_EXECUTION_BATCHES is not None:\n"
                "    runner_args.extend(['--max-new-execution-batches', str(int(MAX_NEW_EXECUTION_BATCHES))])\n"
                "runner_path = LOCAL_BUNDLE_DIR / 'run_campaign.py'\n"
                "print('[launch in notebook kernel] ' + str(runner_path) + ' ' + ' '.join(runner_args), flush=True)\n"
                "module_name = f'_hard_wall_purification_alpha_endpoint_lane_{LANE.lower()}'\n"
                "sys.modules.pop(module_name, None)\n"
                "sys.path.insert(0, str(LOCAL_BUNDLE_DIR))\n"
                "spec = importlib.util.spec_from_file_location(module_name, runner_path)\n"
                "if spec is None or spec.loader is None:\n"
                "    raise RuntimeError(f'Could not load staged runner: {runner_path}')\n"
                "runner = importlib.util.module_from_spec(spec)\n"
                "sys.modules[module_name] = runner\n"
                "spec.loader.exec_module(runner)\n"
                "returncode = runner.main(runner_args)\n"
                "if returncode:\n"
                "    raise RuntimeError(f'Runner exited with status {returncode}')\n"
                "print(f'[notebook] lane {LANE} runner exited successfully', flush=True)\n"
            ),
            code(
                "from google.colab import runtime\n"
                "runtime.unassign()\n"
                "print('done')\n"
            ),
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for lane, filename in NOTEBOOKS.items():
        path = ROOT / filename
        path.write_text(json.dumps(notebook(lane), indent=1) + "\n", encoding="utf-8")
        print(path)
    relative_files = (
        "README.md",
        "build_notebooks.py",
        "campaign_config.json",
        "endpoint_spectrum_observer.py",
        "run_campaign.py",
        *NOTEBOOKS.values(),
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    )
    files: dict[str, dict[str, object]] = {}
    for relative in relative_files:
        raw = (ROOT / relative).read_bytes()
        files[relative] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    manifest = {
        "schema": "simple_colab_bundle_manifest_v1",
        "bundle": ROOT.name,
        "sampling_revision": REVISION,
        "lane_contract": {
            "A": {"configurations": 74, "trajectories": 7400, "execution_batches": 116},
            "B": {"configurations": 73, "trajectories": 7300, "execution_batches": 115},
        },
        "files": files,
    }
    manifest_path = ROOT / "deployment_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
