#!/usr/bin/env python3
"""Build the drop-in A100 notebook for the batched tangent replay."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
BUNDLE_ROOT = HERE.parent
OUTPUT = BUNDLE_ROOT / "run_pure_tangent_gpu_replay.ipynb"
MANIFEST = HERE / "deployment_manifest.json"


def markdown(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


cells = [
    markdown(
        r"""
# A100 batched tangent-cocycle replay

This notebook consumes the completed slot-09 physical records and computes the
full- and late-window tangent cocycles on an A100. Each durable task contains
**25 trajectories evolved together on the GPU**. No Choi covariance, covariance
history, intermediate physical-frame history, or dense superoperator is built.

Run from top to bottom. Restarting verifies and skips completed NPZ/JSON pairs;
only the active 25-trajectory batch is repeated after an interruption.
"""
    ),
    code(
        """
from google.colab import drive
drive.mount('/content/drive')
"""
    ),
    markdown(
        """
## 1. Editable configuration

All user-editable paths and run controls are in this cell. The scientific and
batching contract inside `campaign_config.json` is locked.
"""
    ),
    code(
        """
from pathlib import Path

BUNDLE_DRIVE_DIR = Path('/content/drive/MyDrive/final_production_new_designs/09_pure_tangent_replay_acquisition')
ACQUISITION_ROOT = Path('/content/drive/MyDrive/classA_final_production_outputs/pure_tangent_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1')
OUTPUT_ROOT = Path('/content/drive/MyDrive/classA_final_production_outputs/pure_tangent_gpu_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1')
LOCAL_BUNDLE_DIR = Path('/content/09_pure_tangent_replay_acquisition')
SCRATCH_ROOT = Path('/content/pure_tangent_gpu_replay_scratch')

REPORT_ONLY = False
MAX_NEW_TASKS = None              # e.g. 1 for one 25-trajectory calibration batch
VERIFY_ALL_INPUT_CHECKSUMS = False  # each input is always checksummed when staged
"""
    ),
    markdown(
        """
## 2. Verify the A100 and stage executable files locally

Only the small executable set is copied. Input records remain in their existing
Drive output collection and are staged one acquisition batch at a time.
"""
    ),
    code(
        """
import shutil
import torch

if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable; select an A100 GPU runtime')
props = torch.cuda.get_device_properties(0)
if 'A100' not in props.name.upper() or props.total_memory < 38 * 1024**3:
    raise RuntimeError(f'Need an A100 with 40-GB-class memory; found {props.name}, {props.total_memory / 1024**3:.2f} GiB')
print(f'[device] {props.name}; total GPU RAM={props.total_memory / 1024**3:.2f} GiB; dtype=complex128', flush=True)

required = [
    'run_campaign.py',
    'replay_record_observer.py',
    'gpu_tangent_replay/run_gpu_tangent_replay.py',
    'gpu_tangent_replay/campaign_config.json',
    'src/classA_U1FGTN_gpu.py',
    'src/occupied_frame_gpu.py',
]
if not BUNDLE_DRIVE_DIR.is_dir():
    raise FileNotFoundError(f'Missing Drive bundle: {BUNDLE_DRIVE_DIR}')
if LOCAL_BUNDLE_DIR.exists():
    shutil.rmtree(LOCAL_BUNDLE_DIR)
for relative in required:
    source = BUNDLE_DRIVE_DIR / relative
    destination = LOCAL_BUNDLE_DIR / relative
    if not source.is_file():
        raise FileNotFoundError(f'Missing bundle file: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
print(f'[stage] copied {len(required)} executable files to {LOCAL_BUNDLE_DIR}', flush=True)
print(f'[input] {ACQUISITION_ROOT}', flush=True)
print(f'[output] {OUTPUT_ROOT}', flush=True)
print(f'[scratch] {SCRATCH_ROOT}', flush=True)
"""
    ),
    markdown(
        """
## 3. Run or resume

The outer bar tracks 48 durable GPU batches. A task-level bar tracks the full
and late replay windows, and the canonical engine prints a cycle bar for each
window. The first new task is the memory/runtime calibration; the runner stops
before another task if it exceeds one hour or 36 GiB reserved GPU memory.
"""
    ),
    code(
        """
import os
import subprocess
import sys

command = [
    sys.executable,
    '-u',
    str(LOCAL_BUNDLE_DIR / 'gpu_tangent_replay' / 'run_gpu_tangent_replay.py'),
    '--config',
    str(LOCAL_BUNDLE_DIR / 'gpu_tangent_replay' / 'campaign_config.json'),
    '--acquisition-root',
    str(ACQUISITION_ROOT),
    '--output-root',
    str(OUTPUT_ROOT),
    '--scratch-root',
    str(SCRATCH_ROOT),
]
if REPORT_ONLY:
    command.append('--report-only')
if MAX_NEW_TASKS is not None:
    command.extend(['--max-new-tasks', str(int(MAX_NEW_TASKS))])
if not VERIFY_ALL_INPUT_CHECKSUMS:
    command.append('--skip-input-checksums')

print('[launch] ' + ' '.join(command), flush=True)
environment = dict(os.environ)
environment['PYTHONUNBUFFERED'] = '1'
environment['TQDM_MININTERVAL'] = '1'
subprocess.run(command, check=True, env=environment)
print('[notebook] GPU tangent replay runner exited successfully', flush=True)
"""
    ),
    markdown(
        """
## 4. Release the runtime

Run only after the campaign cell exits (normally, by limit, or report-only).
"""
    ),
    code(
        """
from google.colab import runtime
runtime.unassign()
print('done')
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


deployment_files = (
    "run_pure_tangent_gpu_replay.ipynb",
    "run_campaign.py",
    "replay_record_observer.py",
    "gpu_tangent_replay/run_gpu_tangent_replay.py",
    "gpu_tangent_replay/campaign_config.json",
    "gpu_tangent_replay/build_notebook.py",
    "gpu_tangent_replay/README.md",
    "src/classA_U1FGTN_gpu.py",
    "src/occupied_frame_gpu.py",
)
manifest = {
    "schema": "pure_tangent_gpu_replay_deployment_manifest_v1",
    "sampling_revision": "pure_tangent_gpu_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1",
    "files": {
        relative: {
            "bytes": int((BUNDLE_ROOT / relative).stat().st_size),
            "sha256": sha256_file(BUNDLE_ROOT / relative),
        }
        for relative in deployment_files
    },
}
MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(MANIFEST)
