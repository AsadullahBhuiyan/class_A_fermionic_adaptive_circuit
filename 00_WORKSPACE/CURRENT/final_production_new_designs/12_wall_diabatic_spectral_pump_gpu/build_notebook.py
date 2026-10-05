#!/usr/bin/env python3
"""Build the A100 primary spectral-pump Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        "# A100 wall-diabatized spectral pump — primary campaign only\n\n"
        "This notebook consumes existing frozen endpoint states and computes the "
        "M=256 CW/CCW primary spectral-flow products on an A100 in complex128. "
        "It does **not** run the 2,050-task mesh/gauge/window/rank sensitivity suite. "
        "A benchmark selects 1, 2, 4, or 8 concurrent endpoint lanes so CUDA "
        "eigensolvers overlap with continuation bookkeeping. Every endpoint is "
        "independently resumable through a verified NPZ/completion pair.\n"
    ),
    markdown("## 1. Mount Drive\n"),
    code("from google.colab import drive\ndrive.mount('/content/drive')\n"),
    markdown(
        "## 2. Paths and run controls\n\n"
        "`LEGACY_ENDPOINT_ROOT` must contain the three reused CPU burn-in trees under "
        "`results/`. `NEW_ENDPOINT_ROOT` is the output of bundle 11 and normally already "
        "contains the 1,000 new endpoints plus 150 bridge endpoints. Start with report mode.\n"
    ),
    code(
        "from pathlib import Path\n\n"
        "BUNDLE_DRIVE_DIR = Path('/content/drive/MyDrive/final_production_new_designs/12_wall_diabatic_spectral_pump_gpu')\n"
        "LEGACY_ENDPOINT_ROOT = Path('/content/drive/MyDrive/frozen_record_flux_charge_pilot')\n"
        "NEW_ENDPOINT_ROOT = Path('/content/drive/MyDrive/classA_final_production_outputs/wall_pump_width_endpoints_s100_v1')\n"
        "OUTPUT_ROOT = Path('/content/drive/MyDrive/classA_final_production_outputs/wall_diabatic_spectral_pump_gpu_primary_v1')\n"
        "LOCAL_BUNDLE_DIR = Path('/content/12_wall_diabatic_spectral_pump_gpu')\n"
        "SCRATCH_ROOT = Path('/content/wall_diabatic_spectral_pump_gpu_scratch')\n\n"
        "REPORT_ONLY = True\n"
        "BENCHMARK_ONLY = False\n"
        "RERUN_BENCHMARK = False\n"
        "INCLUDE_BRIDGE = True\n"
        "MAX_NEW_BATCHES = None  # use 1 for the first timed concurrent batch\n"
        "LANES = None  # None uses the benchmark-selected value; diagnostic override only\n\n"
        "print(f'Bundle: {BUNDLE_DRIVE_DIR}')\n"
        "print(f'Legacy endpoints: {LEGACY_ENDPOINT_ROOT}')\n"
        "print(f'New endpoints: {NEW_ENDPOINT_ROOT}')\n"
        "print(f'Output: {OUTPUT_ROOT}')\n"
        "print(f'Workload: {1750 if INCLUDE_BRIDGE else 1600} endpoint tasks; sensitivity disabled')\n"
    ),
    markdown("## 3. A100, dtype, and free-space checks\n"),
    code(
        "import shutil, torch\n"
        "if not torch.cuda.is_available():\n    raise RuntimeError('CUDA is required')\n"
        "name = torch.cuda.get_device_name(0)\n"
        "total = torch.cuda.get_device_properties(0).total_memory / 1024**3\n"
        "if 'A100' not in name.upper() or total < 38:\n"
        "    raise RuntimeError(f'A 40-GB-class A100 is required; found {name}, {total:.1f} GiB')\n"
        "probe = torch.zeros(1, dtype=torch.complex128, device='cuda:0'); del probe\n"
        "local_free = shutil.disk_usage('/content').free / 1024**3\n"
        "drive_free = shutil.disk_usage(OUTPUT_ROOT.parent).free / 1024**3\n"
        "if local_free < 10 or drive_free < 10:\n"
        "    raise RuntimeError(f'At least 10 GiB free is required locally and on Drive; local={local_free:.1f}, Drive={drive_free:.1f}')\n"
        "print(f'Device={name}; memory={total:.1f} GiB; dtype=complex128')\n"
        "print(f'Free space: local={local_free:.1f} GiB, Drive={drive_free:.1f} GiB')\n"
    ),
    markdown(
        "## 4. Stage the small executable bundle and report/run\n\n"
        "The runner first verifies all endpoint result/completion pairs. Production is "
        "blocked unless the A100 eigensolver agrees with a CPU complex128 reference at "
        "`1e-10`. Results are computed under `/content`, copied to a temporary Drive path, "
        "read back and checksummed, renamed atomically, and completed by writing JSON last.\n"
    ),
    code(
        "import codecs, os, subprocess, sys\n\n"
        "def stream(command):\n"
        "    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)\n"
        "    assert process.stdout is not None\n"
        "    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')\n"
        "    try:\n"
        "        while True:\n"
        "            raw = os.read(process.stdout.fileno(), 4096)\n"
        "            if not raw:\n                break\n"
        "            sys.stdout.write(decoder.decode(raw)); sys.stdout.flush()\n"
        "        sys.stdout.write(decoder.decode(b'', final=True)); sys.stdout.flush()\n"
        "    except BaseException:\n"
        "        process.terminate()\n"
        "        try:\n            process.wait(timeout=10)\n"
        "        except subprocess.TimeoutExpired:\n            process.kill(); process.wait()\n"
        "        raise\n"
        "    finally:\n        process.stdout.close()\n"
        "    if process.wait():\n        raise subprocess.CalledProcessError(process.returncode, command)\n\n"
        "if not BUNDLE_DRIVE_DIR.is_dir():\n    raise FileNotFoundError(BUNDLE_DRIVE_DIR)\n"
        "if LOCAL_BUNDLE_DIR.exists():\n    shutil.rmtree(LOCAL_BUNDLE_DIR)\n"
        "shutil.copytree(BUNDLE_DRIVE_DIR, LOCAL_BUNDLE_DIR)\n"
        "command = [sys.executable, '-u', str(LOCAL_BUNDLE_DIR/'run_campaign.py'),\n"
        "           '--config', str(LOCAL_BUNDLE_DIR/'campaign_config.json'),\n"
        "           '--legacy-endpoint-root', str(LEGACY_ENDPOINT_ROOT),\n"
        "           '--new-endpoint-root', str(NEW_ENDPOINT_ROOT),\n"
        "           '--output-root', str(OUTPUT_ROOT),\n"
        "           '--scratch-root', str(SCRATCH_ROOT)]\n"
        "command.append('--include-bridge' if INCLUDE_BRIDGE else '--no-include-bridge')\n"
        "if REPORT_ONLY:\n    command.append('--report-only')\n"
        "if BENCHMARK_ONLY:\n    command.append('--benchmark-only')\n"
        "if RERUN_BENCHMARK:\n    command.append('--rerun-benchmark')\n"
        "if MAX_NEW_BATCHES is not None:\n    command += ['--max-new-batches', str(int(MAX_NEW_BATCHES))]\n"
        "if LANES is not None:\n    command += ['--lanes', str(int(LANES))]\n"
        "print('[launch]', ' '.join(command), flush=True)\n"
        "stream(command)\n"
        "print('[notebook] runner exited successfully', flush=True)\n"
    ),
    markdown(
        "## 5. Operating sequence\n\n"
        "1. Run with `REPORT_ONLY=True` to verify Drive inventory.\n"
        "2. Set `REPORT_ONLY=False`, `BENCHMARK_ONLY=True` to write the parity/timing receipt.\n"
        "3. Set `BENCHMARK_ONLY=False`, `MAX_NEW_BATCHES=1` for one fully timed "
        "benchmark-selected concurrent batch at the largest pending Nx.\n"
        "4. Inspect its elapsed times and projected 1,600-task runtime, then set "
        "`MAX_NEW_BATCHES=None` to resume production. Unlimited mode is blocked until "
        "this receipt exists and every worst-case task took less than one hour.\n"
    ),
    markdown("## 6. Release the runtime after the selected run completes\n"),
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

(ROOT / "run_wall_diabatic_spectral_pump_gpu.ipynb").write_text(
    json.dumps(notebook, indent=1) + "\n", encoding="utf-8"
)
