"""Generate the canonical full-campaign and hard-wall lane notebooks."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "campaign_config.json").read_text(encoding="utf-8"))
REVISION = CONFIG["sampling_revision"]
OUTPUT_COLLECTION = REVISION
BUNDLE_DRIVE_DIR = (
    "/content/drive/MyDrive/final_production_new_designs/"
    "13_maxmix_manybody_lyapunov_4ny"
)
HARD_LANES = {
    "a": (60, 44, 20),
    "b": (56, 36, 30, 24),
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


def notebook(
    construction: str,
    *,
    lane_name: str | None = None,
    ny_lane: tuple[int, ...] | None = None,
) -> dict[str, object]:
    if (lane_name is None) != (ny_lane is None):
        raise ValueError("lane_name and ny_lane must be supplied together")
    if lane_name is not None and construction != "hard":
        raise ValueError("parallel lanes are hard-wall only")
    label = "Hard/support-truncated" if construction == "hard" else "Soft/untruncated"
    lane_label = "" if lane_name is None else f" — lane {lane_name.upper()}"
    lane_sentence = (
        ""
        if ny_lane is None
        else f" This lane runs only Ny={','.join(map(str, ny_lane))}; the other hard-wall lane uses disjoint sizes."
    )
    config_literal = repr(CONFIG)
    return {
        "cells": [
            markdown(
                f"# {label} many-body Lyapunov campaign{lane_label}\n\n"
                "Nx=20; Ny=20,24,30,36,44,56,60; S=100 per size; T=4Ny. "
                "The runner uses 38-GiB-capped resident A100 batches, five-trajectory durable "
                "result shards, a rolling ten-cycle state checkpoint, and live outer/inner "
                "`tqdm` progress. Existing compatible datasets are not overwritten or pooled."
                f"{lane_sentence}\n"
            ),
            code(
                "from google.colab import drive\n"
                "drive.mount('/content/drive')\n"
            ),
            code(
                "from pathlib import Path\n"
                "import json\n"
                "import torch\n\n"
                f"CONSTRUCTION = {construction!r}\n"
                f"LANE_NAME = {lane_name!r}\n"
                f"NY_LANE = {None if ny_lane is None else list(ny_lane)!r}\n"
                "REPORT_ONLY = False\n"
                "MAX_NEW_EXECUTION_BATCHES = None  # Set to 1 for a worst-size timing test.\n"
                f"BUNDLE_DRIVE_DIR = Path({BUNDLE_DRIVE_DIR!r})\n"
                f"OUTPUT_ROOT = Path('/content/drive/MyDrive/classA_final_production_outputs/{OUTPUT_COLLECTION}')\n"
                f"LOCAL_BUNDLE_DIR = Path('/content/13_maxmix_manybody_lyapunov_4ny_{construction}{'' if lane_name is None else '_lane_' + lane_name}')\n"
                f"SCRATCH_ROOT = Path('/content/maxmix_manybody_lyapunov_4ny_scratch_{construction}{'' if lane_name is None else '_lane_' + lane_name}')\n"
                f"CONFIG = {config_literal}\n"
                "if not torch.cuda.is_available():\n"
                "    raise RuntimeError('This production notebook requires CUDA.')\n"
                "gpu_name = torch.cuda.get_device_name(0)\n"
                "gpu_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3\n"
                "if 'A100' not in gpu_name or gpu_gib < 35:\n"
                "    raise RuntimeError(f'Expected an A100 40-GB-class GPU, got {gpu_name} ({gpu_gib:.2f} GiB)')\n"
                "if torch.complex128 != getattr(torch, CONFIG['dtype']):\n"
                "    raise RuntimeError('Locked dtype is not complex128.')\n"
                "print(json.dumps({\n"
                "    'construction': CONSTRUCTION, 'lane': LANE_NAME, 'Ny_lane': NY_LANE,\n"
                "    'config': CONFIG,\n"
                "    'bundle': str(BUNDLE_DRIVE_DIR), 'output': str(OUTPUT_ROOT),\n"
                "    'scratch': str(SCRATCH_ROOT), 'gpu': gpu_name, 'gpu_GiB': gpu_gib,\n"
                "    'gpu_memory_hard_limit_GiB': CONFIG['gpu_memory_hard_limit_gib'],\n"
                "    'execution_batch_size_by_Ny': CONFIG['execution_batch_size_by_Ny'],\n"
                "    'report_only': REPORT_ONLY,\n"
                "    'max_new_execution_batches': MAX_NEW_EXECUTION_BATCHES,\n"
                "}, indent=2, sort_keys=True))\n"
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
                "config_path = Path(f'/content/maxmix_manybody_lyapunov_4ny_{CONSTRUCTION}.json')\n"
                "config_path.write_text(json.dumps(CONFIG, indent=2, sort_keys=True) + '\\n', encoding='utf-8')\n"
                "runner_args = [\n"
                "    '--config', str(config_path), '--construction', CONSTRUCTION,\n"
                "    '--output-root', str(OUTPUT_ROOT), '--scratch-root', str(SCRATCH_ROOT),\n"
                "]\n"
                "if REPORT_ONLY:\n"
                "    runner_args.append('--report-only')\n"
                "if MAX_NEW_EXECUTION_BATCHES is not None:\n"
                "    runner_args.extend(['--max-new-execution-batches', str(int(MAX_NEW_EXECUTION_BATCHES))])\n"
                "runner_path = LOCAL_BUNDLE_DIR / 'run_campaign.py'\n"
                "print('[launch in notebook kernel] ' + str(runner_path) + ' ' + ' '.join(runner_args), flush=True)\n"
                "module_name = f'_maxmix_manybody_lyapunov_runner_{CONSTRUCTION}'\n"
                "sys.modules.pop(module_name, None)\n"
                "sys.path.insert(0, str(LOCAL_BUNDLE_DIR))\n"
                "spec = importlib.util.spec_from_file_location(module_name, runner_path)\n"
                "if spec is None or spec.loader is None:\n"
                "    raise RuntimeError(f'Could not load staged runner: {runner_path}')\n"
                "runner = importlib.util.module_from_spec(spec)\n"
                "sys.modules[module_name] = runner\n"
                "spec.loader.exec_module(runner)\n"
                "if NY_LANE is not None:\n"
                "    _unfiltered_expand = runner.expand_execution_batches\n"
                "    _lane_set = frozenset(int(value) for value in NY_LANE)\n"
                "    def _lane_expand(config, construction):\n"
                "        expanded = _unfiltered_expand(config, construction)\n"
                "        selected = [task for task in expanded if task.ny in _lane_set]\n"
                "        if {task.ny for task in selected} != _lane_set:\n"
                "            raise RuntimeError(f'Lane {LANE_NAME} did not resolve every requested Ny: {NY_LANE}')\n"
                "        return selected\n"
                "    runner.expand_execution_batches = _lane_expand\n"
                "returncode = runner.main(runner_args)\n"
                "if returncode:\n"
                "    raise RuntimeError(f'Runner exited with status {returncode}')\n"
                "print(f'[notebook] {CONSTRUCTION}-wall runner exited successfully', flush=True)\n"
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
    for construction in ("hard", "soft"):
        path = ROOT / f"run_{construction}_wall_manybody_lyapunov_4ny.ipynb"
        path.write_text(
            json.dumps(notebook(construction), indent=1) + "\n", encoding="utf-8"
        )
        print(path)
    for lane_name, ny_lane in HARD_LANES.items():
        path = ROOT / f"run_hard_wall_manybody_lyapunov_4ny_lane_{lane_name}.ipynb"
        path.write_text(
            json.dumps(
                notebook("hard", lane_name=lane_name, ny_lane=ny_lane), indent=1
            )
            + "\n",
            encoding="utf-8",
        )
        print(path)
    relative_files = (
        "README.md",
        "build_notebooks.py",
        "campaign_config.json",
        "lyapunov_observer.py",
        "run_campaign.py",
        "run_hard_wall_manybody_lyapunov_4ny.ipynb",
        "run_hard_wall_manybody_lyapunov_4ny_lane_a.ipynb",
        "run_hard_wall_manybody_lyapunov_4ny_lane_b.ipynb",
        "run_soft_wall_manybody_lyapunov_4ny.ipynb",
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
        "files": files,
    }
    manifest_path = ROOT / "deployment_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
