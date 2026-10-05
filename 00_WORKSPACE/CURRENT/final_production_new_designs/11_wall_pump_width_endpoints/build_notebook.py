#!/usr/bin/env python3
"""Regenerate the canonical A100 notebook for this endpoint bundle."""

from __future__ import annotations

import json
import pprint
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "campaign_config.json").read_text(encoding="utf-8"))


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


config_source = pprint.pformat(CONFIG, sort_dicts=True, width=100)
cells = [
    markdown(
        "# Wall-pump width endpoints S100 v1\n\n"
        "Generate the missing width-controlled endpoint frames and the small CPU/GPU "
        "bridge ensemble. Results are five-trajectory shards; execution batches are "
        "chosen by an A100 throughput/memory benchmark and checkpoint every five cycles.\n"
    ),
    code("from google.colab import drive\ndrive.mount('/content/drive')\n"),
    markdown("## 1. Locked scientific configuration and run controls\n"),
    code(
        "from pathlib import Path\n\n"
        "BUNDLE_DRIVE_DIR = Path('/content/drive/MyDrive/final_production_new_designs/11_wall_pump_width_endpoints')\n"
        "OUTPUT_ROOT = Path('/content/drive/MyDrive/classA_final_production_outputs/wall_pump_width_endpoints_s100_v1')\n"
        "LOCAL_BUNDLE_DIR = Path('/content/11_wall_pump_width_endpoints')\n"
        "SCRATCH_ROOT = Path('/content/wall_pump_width_endpoints_s100_v1_scratch')\n"
        "REPORT_ONLY = False\n"
        "BENCHMARK_ONLY = False\n"
        "MAX_NEW_EXECUTION_BATCHES = None\n\n"
        f"CONFIG = {config_source}\n\n"
        "print(f'Revision: {CONFIG[\"sampling_revision\"]}')\n"
        "print('Workload: 1,000 primary + 150 bridge endpoints; 230 five-sample shards')\n"
        "print('Scientific contract: Ny=24, 48 cycles, raster_y, pure half filling, perfect correction')\n"
        "print(f'Bundle: {BUNDLE_DRIVE_DIR}')\n"
        "print(f'Output: {OUTPUT_ROOT}')\n"
        "print(f'Scratch: {SCRATCH_ROOT}')\n"
    ),
    markdown("## 2. A100 and complex128 check\n"),
    code(
        "import torch\n"
        "if not torch.cuda.is_available():\n    raise RuntimeError('CUDA is required')\n"
        "name = torch.cuda.get_device_name(0)\n"
        "gib = torch.cuda.get_device_properties(0).total_memory / 1024**3\n"
        "if 'A100' not in name.upper() or gib < 38:\n"
        "    raise RuntimeError(f'A 40-GB-class A100 is required; found {name}, {gib:.1f} GiB')\n"
        "probe = torch.zeros(1, dtype=torch.complex128, device='cuda:0')\n"
        "del probe\n"
        "print(f'Device: {name}; memory={gib:.1f} GiB; dtype=complex128')\n"
    ),
    markdown(
        "## 3. Stage locally and run/resume\n\n"
        "The child output is streamed so the outer shard bar and inner cycle bar remain "
        "visible. A restart verifies every result/completion pair and restores the exact "
        "last five-cycle state/RNG checkpoint for the current execution batch.\n"
    ),
    code(
        "import codecs, json, os, shutil, subprocess, sys\n\n"
        "def run_streaming_child(command):\n"
        "    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)\n"
        "    assert process.stdout is not None\n"
        "    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')\n"
        "    try:\n"
        "        while True:\n"
        "            raw = os.read(process.stdout.fileno(), 4096)\n"
        "            if not raw:\n                break\n"
        "            sys.stdout.write(decoder.decode(raw)); sys.stdout.flush()\n"
        "        tail = decoder.decode(b'', final=True)\n"
        "        if tail:\n            sys.stdout.write(tail); sys.stdout.flush()\n"
        "    except BaseException:\n"
        "        process.terminate()\n"
        "        try:\n            process.wait(timeout=5)\n"
        "        except subprocess.TimeoutExpired:\n            process.kill(); process.wait()\n"
        "        raise\n"
        "    finally:\n        process.stdout.close()\n"
        "    returncode = process.wait()\n"
        "    if returncode:\n        raise subprocess.CalledProcessError(returncode, command)\n\n"
        "if not BUNDLE_DRIVE_DIR.is_dir():\n    raise FileNotFoundError(BUNDLE_DRIVE_DIR)\n"
        "if LOCAL_BUNDLE_DIR.exists():\n    shutil.rmtree(LOCAL_BUNDLE_DIR)\n"
        "shutil.copytree(BUNDLE_DRIVE_DIR, LOCAL_BUNDLE_DIR)\n"
        "config_path = Path('/content/wall_pump_width_endpoints_s100_v1_config.json')\n"
        "config_path.write_text(json.dumps(CONFIG, indent=2, sort_keys=True) + '\\n', encoding='utf-8')\n"
        "command = [sys.executable, '-u', str(LOCAL_BUNDLE_DIR/'run_campaign.py'), "
        "'--config', str(config_path), '--output-root', str(OUTPUT_ROOT), "
        "'--scratch-root', str(SCRATCH_ROOT)]\n"
        "if REPORT_ONLY:\n    command.append('--report-only')\n"
        "if BENCHMARK_ONLY:\n    command.append('--benchmark-only')\n"
        "if MAX_NEW_EXECUTION_BATCHES is not None:\n"
        "    command += ['--max-new-execution-batches', str(int(MAX_NEW_EXECUTION_BATCHES))]\n"
        "print('[launch]', ' '.join(command), flush=True)\n"
        "run_streaming_child(command)\n"
        "print('[notebook] runner exited successfully', flush=True)\n"
    ),
    markdown("## 4. Release the runtime\n"),
    code("from google.colab import runtime\nruntime.unassign()\nprint('done')\n"),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

(ROOT / "run_wall_pump_width_endpoints.ipynb").write_text(
    json.dumps(notebook, indent=1) + "\n", encoding="utf-8"
)
