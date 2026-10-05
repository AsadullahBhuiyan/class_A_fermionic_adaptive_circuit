"""Build the fixed sequential hard-then-soft A100 notebook and manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "campaign_config.json").read_text(encoding="utf-8"))
DRIVE_BUNDLE = "/content/drive/MyDrive/final_production_new_designs/19_postselected_hard_soft_n20x40"
OUTPUT = (
    "/content/drive/MyDrive/classA_final_production_outputs/"
    + CONFIG["sampling_revision"]
)


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def build() -> dict[str, object]:
    return {
        "cells": [
            markdown(
                "# Postselected hard/soft purification at $20\\times40$\n\n"
                "Runs hard then soft walls at alpha_1=1, then hard then soft at alpha_1=3. "
                "Both are deterministic full-postselection trajectories from the maximally mixed "
                "state through $4N_y=160$ cycles. Every cycle saves spatial entropy contours, total entropy and the "
                "occupation-derived finite-time Lyapunov gap; endpoint spectra and eigenvectors "
                "are retained.\n"
            ),
            code("from google.colab import drive\ndrive.mount('/content/drive')\n"),
            code(
                "from pathlib import Path\n"
                "import json\n"
                "import torch\n\n"
                "REPORT_ONLY = False\n"
                "MAX_NEW_CONSTRUCTIONS = None  # all four alpha/wall runs; 1 limits to one pending run\n"
                f"BUNDLE_DRIVE_DIR = Path({DRIVE_BUNDLE!r})\n"
                "LOCAL_BUNDLE_DIR = Path('/content/19_postselected_hard_soft_n20x40')\n"
                "SCRATCH_ROOT = Path('/content/postselected_hard_soft_n20x40_scratch')\n"
                f"CONFIG = {CONFIG!r}\n\n"
                "OUTPUT_ROOT = Path('/content/drive/MyDrive/classA_final_production_outputs') / CONFIG['sampling_revision']\n\n"
                "if not torch.cuda.is_available():\n"
                "    raise RuntimeError('This notebook requires CUDA.')\n"
                "gpu_name = torch.cuda.get_device_name(0)\n"
                "gpu_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3\n"
                "if 'A100' not in gpu_name or gpu_gib < 38.0:\n"
                "    raise RuntimeError(f'Expected A100 40-GB-class GPU, got {gpu_name} ({gpu_gib:.2f} GiB)')\n"
                "print(json.dumps({'config': CONFIG, 'bundle': str(BUNDLE_DRIVE_DIR), "
                "'output': str(OUTPUT_ROOT), 'gpu': gpu_name, 'gpu_GiB': gpu_gib, "
                "'report_only': REPORT_ONLY, 'max_new_constructions': MAX_NEW_CONSTRUCTIONS}, "
                "indent=2, sort_keys=True), flush=True)\n"
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
                "config_path = Path('/content/postselected_hard_soft_n20x40_config.json')\n"
                "config_path.write_text(json.dumps(CONFIG, indent=2, sort_keys=True) + '\\n', encoding='utf-8')\n"
                "runner_args = ['--config', str(config_path), '--output-root', str(OUTPUT_ROOT), "
                "'--scratch-root', str(SCRATCH_ROOT)]\n"
                "if REPORT_ONLY:\n"
                "    runner_args.append('--report-only')\n"
                "if MAX_NEW_CONSTRUCTIONS is not None:\n"
                "    runner_args.extend(['--max-new-constructions', str(int(MAX_NEW_CONSTRUCTIONS))])\n"
                "runner_path = LOCAL_BUNDLE_DIR / 'run_campaign.py'\n"
                "print('[launch in notebook kernel] ' + str(runner_path) + ' ' + ' '.join(runner_args), flush=True)\n"
                "sys.path.insert(0, str(LOCAL_BUNDLE_DIR))\n"
                "spec = importlib.util.spec_from_file_location('_postselected_hard_soft_runner', runner_path)\n"
                "if spec is None or spec.loader is None:\n"
                "    raise RuntimeError(f'Could not load staged runner: {runner_path}')\n"
                "runner = importlib.util.module_from_spec(spec)\n"
                "sys.modules[spec.name] = runner\n"
                "spec.loader.exec_module(runner)\n"
                "returncode = runner.main(runner_args)\n"
                "if returncode:\n"
                "    raise RuntimeError(f'Runner exited with status {returncode}')\n"
                "print('[notebook] hard/soft runner exited successfully', flush=True)\n"
            ),
            code("from google.colab import runtime\nruntime.unassign()\nprint('done')\n"),
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
    notebook_path = ROOT / "run_postselected_hard_soft_n20x40.ipynb"
    notebook_path.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    relative_files = (
        "README.md",
        "build_notebook.py",
        "campaign_config.json",
        "postselected_observer.py",
        "run_campaign.py",
        "run_postselected_hard_soft_n20x40.ipynb",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    )
    files = {}
    for relative in relative_files:
        raw = (ROOT / relative).read_bytes()
        files[relative] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    manifest = {
        "schema": "simple_colab_bundle_manifest_v1",
        "bundle": ROOT.name,
        "sampling_revision": CONFIG["sampling_revision"],
        "execution_order": CONFIG["construction_order"],
        "files": files,
    }
    (ROOT / "deployment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(notebook_path)
    print(ROOT / "deployment_manifest.json")


if __name__ == "__main__":
    main()
